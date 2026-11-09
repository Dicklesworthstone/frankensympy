//! Original-source acceptance, adversarial transport, and cumulative resources.

use frankensympy::certification::{
    CertificationError, MAX_CERTIFICATION_SOURCE_BYTES, MAX_ROOT_CERTIFICATE_BYTES,
    MAX_ROOT_CERTIFICATE_JSON_DEPTH, ROOT_CERTIFICATE_SCHEMA_VERSION, ResidualProblem,
    export_root_certificate, replay_root_certificate,
};
use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError,
    Unbounded,
};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_RESIDUALS,
    MAX_ROOT_BOX_VARIABLES, RealBall, RootBoxError, RootBoxReceipt, RootBoxWitness, Symbol,
    SymbolIdentity,
};
use serde::Serialize;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn problem() -> ResidualProblem {
    ResidualProblem::parse_sources(&["x^2-2".into()], &["x".into()], &mut Unbounded).unwrap()
}

fn witness() -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(q(7, 5), q(1, 10)).unwrap()],
        center: vec![q(7, 5)],
        preconditioner: vec![vec![q(5, 14)]],
        precision_digits: 24,
    }
}

fn receipt() -> RootBoxReceipt {
    problem().verify_root(&witness(), &mut Unbounded).unwrap()
}

#[derive(Serialize)]
struct ProposedArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    witness: &'a RootBoxWitness,
}

fn proposed_bytes(problem: &ResidualProblem, witness: &RootBoxWitness) -> Vec<u8> {
    serde_json::to_vec(&ProposedArtifact {
        schema_version: ROOT_CERTIFICATE_SCHEMA_VERSION,
        context_digest: [7; 32],
        residuals: problem.residuals(),
        variables: problem.variables(),
        witness,
    })
    .unwrap()
}

#[test]
fn exact_source_enclosure_and_unique_root_have_independent_rational_expectations() {
    let problem = problem();
    let domain = [RealBall::new(q(3, 2), q(1, 2)).unwrap()];
    let enclosed = problem.enclose(&domain, 24, &mut Unbounded).unwrap();
    assert_eq!(enclosed.residuals(), problem.residuals());
    assert_eq!(enclosed.variables(), problem.variables());
    assert_eq!(enclosed.domain(), &domain);
    assert_eq!(enclosed.residual_values()[0].lower(), q(-1, 1));
    assert_eq!(enclosed.residual_values()[0].upper(), q(2, 1));
    assert_eq!(enclosed.jacobian()[0][0].lower(), q(2, 1));
    assert_eq!(enclosed.jacobian()[0][0].upper(), q(4, 1));

    let accepted = problem.verify_root(&witness(), &mut Unbounded).unwrap();
    assert_eq!(accepted.root_enclosure()[0].lower(), q(197, 140));
    assert_eq!(accepted.root_enclosure()[0].upper(), q(199, 140));
    assert_eq!(accepted.contraction_bound(), &q(1, 14));
    assert!(accepted.root_enclosure()[0].lower().pow(2).unwrap() < q(2, 1));
    assert!(accepted.root_enclosure()[0].upper().pow(2).unwrap() > q(2, 1));
    assert_eq!(accepted.residuals(), problem.residuals());
    assert_eq!(accepted.witness(), &witness());
}

#[test]
fn rectangular_problem_encloses_without_claiming_a_square_system_root() {
    let problem = ResidualProblem::parse_sources(
        &["x+y".into(), "x-y".into(), "x*y".into()],
        &["y".into(), "x".into()],
        &mut Unbounded,
    )
    .unwrap();
    let enclosed = problem
        .enclose(
            &[RealBall::from_i64(3), RealBall::from_i64(2)],
            16,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(
        enclosed.residual_values(),
        &[
            RealBall::from_i64(5),
            RealBall::from_i64(-1),
            RealBall::from_i64(6)
        ]
    );
    assert_eq!(
        enclosed.jacobian(),
        &[
            vec![RealBall::from_i64(1), RealBall::from_i64(1)],
            vec![RealBall::from_i64(-1), RealBall::from_i64(1)],
            vec![RealBall::from_i64(2), RealBall::from_i64(3)],
        ]
    );
    assert_eq!(
        problem.verify_root(&witness(), &mut Unbounded).unwrap_err(),
        CertificationError::Root(RootBoxError::DimensionMismatch)
    );
}

#[test]
fn textual_construction_never_cancels_original_domain_obligations() {
    for (source, point) in [
        ("0*(1/x)", 0),
        ("0/(x-x)", 1),
        ("log(x)-log(x)", -1),
        ("(1/x)^0", 0),
        ("0*sqrt(x)", -1),
    ] {
        let problem =
            ResidualProblem::parse_sources(&[source.into()], &["x".into()], &mut Unbounded)
                .unwrap();
        assert!(
            matches!(
                problem.enclose(&[RealBall::from_i64(point)], 16, &mut Unbounded),
                Err(CertificationError::Root(_))
            ),
            "{source}"
        );
    }
    let exact =
        ResidualProblem::parse_sources(&["0.125*x-1/2".into()], &["x".into()], &mut Unbounded)
            .unwrap()
            .enclose(&[RealBall::from_i64(4)], 16, &mut Unbounded)
            .unwrap();
    assert_eq!(exact.residual_values(), &[RealBall::from_i64(0)]);
    assert_eq!(exact.jacobian(), &[vec![RealBall::exact(q(1, 8))]]);
}

#[test]
fn typed_symbol_identity_is_not_replaced_by_its_printed_name() {
    let symbol = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let source = Expr::Sym(symbol.clone());
    let bound =
        ResidualProblem::new(vec![source.clone()], vec![symbol.clone()], &mut Unbounded).unwrap();
    assert_eq!(
        bound
            .enclose(&[RealBall::from_i64(2)], 16, &mut Unbounded)
            .unwrap()
            .residual_values(),
        &[RealBall::from_i64(2)]
    );
    let wrong = ResidualProblem::new(vec![source], vec![Symbol::new("x")], &mut Unbounded).unwrap();
    assert_eq!(
        wrong
            .enclose(&[RealBall::from_i64(2)], 16, &mut Unbounded)
            .unwrap_err(),
        CertificationError::Root(RootBoxError::Unsupported("unbound symbol identity"))
    );
    let duplicated = ResidualProblem::new(
        vec![Expr::Sym(symbol.clone())],
        vec![symbol.clone(), symbol],
        &mut Unbounded,
    );
    assert_eq!(
        duplicated.unwrap_err(),
        CertificationError::Root(RootBoxError::DuplicateVariable)
    );
}

#[test]
fn source_bounds_precede_recursive_numeric_work() {
    let excessive = "1".repeat(MAX_CERTIFICATION_SOURCE_BYTES + 1);
    assert!(matches!(
        ResidualProblem::parse_sources(&[excessive], &["x".into()], &mut Unbounded),
        Err(CertificationError::SourceLimit(_))
    ));
    for variable in ["", "x+y", "x x", "1x", "pi", "E", "nan"] {
        assert_eq!(
            ResidualProblem::parse_sources(&["1".into()], &[variable.into()], &mut Unbounded)
                .unwrap_err(),
            CertificationError::InvalidVariable
        );
    }
    assert_eq!(
        ResidualProblem::parse_sources(&["x +".into()], &["x".into()], &mut Unbounded).unwrap_err(),
        CertificationError::Parse { index: 0 }
    );
    assert_eq!(
        ResidualProblem::new(
            vec![Expr::from_i64(0); MAX_ROOT_BOX_RESIDUALS + 1],
            vec![Symbol::new("x")],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::Root(RootBoxError::DimensionMismatch)
    );
    assert_eq!(
        ResidualProblem::new(
            vec![Expr::from_i64(0)],
            vec![Symbol::new("x"); MAX_ROOT_BOX_VARIABLES + 1],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::Root(RootBoxError::DimensionMismatch)
    );
    let mut deep = Expr::symbol("x");
    for _ in 0..=MAX_ROOT_BOX_DEPTH {
        deep = Expr::Add(vec![deep]);
    }
    assert!(matches!(
        ResidualProblem::new(vec![deep], vec![Symbol::new("x")], &mut Unbounded),
        Err(CertificationError::Root(RootBoxError::ResourceLimit(_)))
    ));
    let huge = BigInt::from(1) << u32::try_from(MAX_ROOT_BOX_LIMBS * 64).unwrap();
    assert!(matches!(
        ResidualProblem::new(
            vec![Expr::Integer(huge)],
            vec![Symbol::new("x")],
            &mut Unbounded
        ),
        Err(CertificationError::Root(RootBoxError::ResourceLimit(
            "integer height"
        )))
    ));
    let explosive =
        ResidualProblem::parse_sources(&["2^1000000000+x".into()], &["x".into()], &mut Unbounded)
            .unwrap();
    assert!(matches!(
        explosive.enclose(&[RealBall::from_i64(0)], 16, &mut Unbounded),
        Err(CertificationError::Root(RootBoxError::ResourceLimit(
            "integer exponent"
        )))
    ));
}

#[test]
fn portable_witness_replays_in_a_fresh_problem_and_contains_no_accepted_claim() {
    let accepted = receipt();
    let bytes = export_root_certificate(&accepted, [7; 32], &mut Unbounded).unwrap();
    let fresh = problem();
    let replayed = replay_root_certificate(&bytes, &fresh, [7; 32], &mut Unbounded).unwrap();
    assert_eq!(replayed, accepted);
    assert_eq!(
        export_root_certificate(&replayed, [7; 32], &mut Unbounded).unwrap(),
        bytes
    );
    let object: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let object = object.as_object().unwrap();
    assert_eq!(object.len(), 5);
    for name in [
        "accepted",
        "verified",
        "receipt",
        "root_enclosure",
        "contraction_bound",
    ] {
        assert!(!object.contains_key(name));
    }
}

#[test]
fn replay_binds_source_variable_order_identity_and_expected_context() {
    let bytes = export_root_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let other =
        ResidualProblem::parse_sources(&["x^2-3".into()], &["x".into()], &mut Unbounded).unwrap();
    assert_eq!(
        replay_root_certificate(&bytes, &other, [7; 32], &mut Unbounded).unwrap_err(),
        CertificationError::SourceMismatch
    );
    assert_eq!(
        replay_root_certificate(&bytes, &problem(), [8; 32], &mut Unbounded).unwrap_err(),
        CertificationError::UniverseMismatch {
            found: [7; 32],
            expected: [8; 32]
        }
    );
    let identified = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let typed = ResidualProblem::new(
        problem().residuals().to_vec(),
        vec![identified],
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        replay_root_certificate(&bytes, &typed, [7; 32], &mut Unbounded).unwrap_err(),
        CertificationError::SourceMismatch
    );

    let two = ResidualProblem::parse_sources(
        &["x-1".into(), "y-2".into()],
        &["x".into(), "y".into()],
        &mut Unbounded,
    )
    .unwrap();
    let proposed = RootBoxWitness {
        domain: vec![
            RealBall::new(q(1, 1), q(1, 10)).unwrap(),
            RealBall::new(q(2, 1), q(1, 10)).unwrap(),
        ],
        center: vec![q(1, 1), q(2, 1)],
        preconditioner: vec![vec![q(1, 1), q(0, 1)], vec![q(0, 1), q(1, 1)]],
        precision_digits: 16,
    };
    let swapped = ResidualProblem::new(
        two.residuals().to_vec(),
        vec![Symbol::new("y"), Symbol::new("x")],
        &mut Unbounded,
    )
    .unwrap();
    let accepted = two.verify_root(&proposed, &mut Unbounded).unwrap();
    let bytes = export_root_certificate(&accepted, [7; 32], &mut Unbounded).unwrap();
    assert_eq!(
        replay_root_certificate(&bytes, &swapped, [7; 32], &mut Unbounded).unwrap_err(),
        CertificationError::SourceMismatch
    );
}

#[test]
fn a_well_formed_canonical_forged_witness_never_gets_an_accepted_receipt() {
    let problem = problem();
    let mut forged = witness();
    forged.preconditioner[0][0] = q(0, 1);
    assert_eq!(
        replay_root_certificate(
            &proposed_bytes(&problem, &forged),
            &problem,
            [7; 32],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::Root(RootBoxError::NotContractive)
    );
    forged = witness();
    forged.center[0] = q(2, 1);
    assert_eq!(
        replay_root_certificate(
            &proposed_bytes(&problem, &forged),
            &problem,
            [7; 32],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::Root(RootBoxError::CenterOutsideBox { coordinate: 0 })
    );
    forged = witness();
    forged.precision_digits = 129;
    assert_eq!(
        replay_root_certificate(
            &proposed_bytes(&problem, &forged),
            &problem,
            [7; 32],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::Root(RootBoxError::InvalidPrecision)
    );
}

#[test]
fn transport_rejects_unknown_noncanonical_duplicate_and_unbounded_payloads() {
    let bytes = export_root_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let problem = problem();
    let mut mutants = vec![
        format!(" {}", text),
        text.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        ),
        text.replacen(
            "\"precision_digits\":24",
            "\"precision_digits\":24,\"accepted\":true",
            1,
        ),
        text.replacen("\"name\":\"x\"", "\"name\":\"x\",\"ignored\":true", 1),
        format!("{},\"verified\":true}}", &text[..text.len() - 1]),
        format!("{} null", text),
    ];
    let mut malformed = witness();
    malformed.preconditioner = vec![vec![q(1, 1); MAX_ROOT_BOX_VARIABLES + 1]];
    mutants.push(String::from_utf8(proposed_bytes(&problem, &malformed)).unwrap());
    for mutant in mutants {
        assert!(
            replay_root_certificate(mutant.as_bytes(), &problem, [7; 32], &mut Unbounded).is_err()
        );
    }
    let future = text.replacen("\"schema_version\":1", "\"schema_version\":2", 1);
    assert_eq!(
        replay_root_certificate(future.as_bytes(), &problem, [7; 32], &mut Unbounded).unwrap_err(),
        CertificationError::UnsupportedVersion {
            found: 2,
            supported: 1
        }
    );
    let oversize = vec![b' '; MAX_ROOT_CERTIFICATE_BYTES + 1];
    assert_eq!(
        replay_root_certificate(&oversize, &problem, [7; 32], &mut Unbounded).unwrap_err(),
        CertificationError::ByteLimit {
            actual: MAX_ROOT_CERTIFICATE_BYTES + 1,
            max: MAX_ROOT_CERTIFICATE_BYTES
        }
    );
    let depth = MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1;
    let deep = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
    assert_eq!(
        replay_root_certificate(deep.as_bytes(), &problem, [7; 32], &mut Unbounded).unwrap_err(),
        CertificationError::DepthLimit {
            depth,
            max: MAX_ROOT_CERTIFICATE_JSON_DEPTH
        }
    );
}

#[test]
fn export_stops_at_the_byte_limit_without_returning_a_partial_artifact() {
    // A small linear mathematical problem can still have a large exact typed
    // payload: repeated, long symbol names must not bypass output accounting.
    let symbol = Symbol::new("x".repeat(3000));
    let mut addends = vec![Expr::Sym(symbol.clone()); 400];
    addends.push(Expr::from_i64(-400));
    let problem =
        ResidualProblem::new(vec![Expr::Add(addends)], vec![symbol], &mut Unbounded).unwrap();
    let proposed = RootBoxWitness {
        domain: vec![RealBall::new(q(1, 1), q(1, 10)).unwrap()],
        center: vec![q(1, 1)],
        preconditioner: vec![vec![q(1, 400)]],
        precision_digits: 16,
    };
    let accepted = problem.verify_root(&proposed, &mut Unbounded).unwrap();
    assert_eq!(accepted.root_enclosure(), &[RealBall::from_i64(1)]);
    assert!(matches!(
        export_root_certificate(&accepted, [7; 32], &mut Unbounded),
        Err(CertificationError::ByteLimit {
            max: MAX_ROOT_CERTIFICATE_BYTES,
            ..
        })
    ));
}

#[derive(Default)]
struct Probe {
    checkpoints: usize,
    interrupt: Option<usize>,
    stopped: bool,
    charges: [u64; DIMENSION_COUNT],
}

impl BudgetMeter for Probe {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "work must not resume after cancellation");
        let mut next = self.charges;
        for &(dimension, amount) in charges {
            assert_ne!(amount, 0);
            next[dimension.index()] += amount;
        }
        self.charges = next;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "work must not resume after cancellation");
        self.checkpoints += 1;
        if self.interrupt == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn cancel_at_every_checkpoint(
    mut operation: impl FnMut(&mut Probe) -> Result<(), CertificationError>,
) {
    let mut measured = Probe::default();
    operation(&mut measured).unwrap();
    assert!(measured.checkpoints > 1);
    for checkpoint in 1..=measured.checkpoints {
        let mut interrupted = Probe {
            interrupt: Some(checkpoint),
            ..Probe::default()
        };
        assert_eq!(
            operation(&mut interrupted).unwrap_err(),
            CertificationError::Meter(MeterError::Cancelled)
        );
        assert!(interrupted.stopped);
    }
}

#[test]
fn construction_verification_and_transport_refuse_at_every_cancellation_safe_point() {
    let problem = problem();
    let witness = witness();
    let receipt = problem.verify_root(&witness, &mut Unbounded).unwrap();
    let bytes = export_root_certificate(&receipt, [7; 32], &mut Unbounded).unwrap();
    cancel_at_every_checkpoint(|meter| {
        ResidualProblem::parse_sources(&["x^2-2".into()], &["x".into()], meter).map(|_| ())
    });
    cancel_at_every_checkpoint(|meter| problem.enclose(&witness.domain, 24, meter).map(|_| ()));
    cancel_at_every_checkpoint(|meter| problem.verify_root(&witness, meter).map(|_| ()));
    cancel_at_every_checkpoint(|meter| {
        export_root_certificate(&receipt, [7; 32], meter).map(|_| ())
    });
    cancel_at_every_checkpoint(|meter| {
        replay_root_certificate(&bytes, &problem, [7; 32], meter).map(|_| ())
    });
}

#[test]
fn repeated_replay_and_failed_verification_keep_one_cumulative_ledger() {
    let problem = problem();
    let bytes = export_root_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let mut measured = Probe::default();
    replay_root_certificate(&bytes, &problem, [7; 32], &mut measured).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(measured.charges[dimension.index()] > 0);
    }
    let mut shared = Budget::new(BudgetLimits {
        dimensions: measured.charges.map(|amount| amount * 2),
        verifier_pool: 193,
    });
    for _ in 0..2 {
        replay_root_certificate(&bytes, &problem, [7; 32], &mut shared).unwrap();
    }
    assert_eq!(shared.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(shared.remaining(Dimension::MemoryBytes), 0);
    assert_eq!(shared.remaining(Dimension::AllocationCount), 0);
    assert!(matches!(
        replay_root_certificate(&bytes, &problem, [7; 32], &mut shared),
        Err(CertificationError::Meter(MeterError::Budget(
            BudgetError::Exhausted { .. }
        )))
    ));
    assert_eq!(shared.verifier_remaining(), 193);

    let mut forged = witness();
    forged.preconditioner[0][0] = q(0, 1);
    let bytes = proposed_bytes(&problem, &forged);
    let mut budget = Budget::new(BudgetLimits::uniform(1 << 40, 29));
    let before = budget.remaining(Dimension::ComputeSteps);
    assert!(matches!(
        replay_root_certificate(&bytes, &problem, [7; 32], &mut budget),
        Err(CertificationError::Root(RootBoxError::NotContractive))
    ));
    let after = budget.remaining(Dimension::ComputeSteps);
    assert!(after < before);
    assert!(replay_root_certificate(&bytes, &problem, [7; 32], &mut budget).is_err());
    assert!(budget.remaining(Dimension::ComputeSteps) < after);
    assert_eq!(budget.verifier_remaining(), 29);
}

#[test]
fn protected_receiving_budget_can_authorize_the_independent_checker() {
    let problem = problem();
    let bytes = export_root_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let mut budget = Budget::new(BudgetLimits::uniform(1 << 40, 1 << 30));
    let lease = budget.verifier_lease().unwrap();
    let before = budget.verifier_remaining();
    let mut meter = budget.verifier_meter(&lease).unwrap();
    let accepted = replay_root_certificate(&bytes, &problem, [7; 32], &mut meter).unwrap();
    assert_eq!(accepted.root_enclosure()[0].lower(), q(197, 140));
    assert!(budget.verifier_remaining() < before);
}
