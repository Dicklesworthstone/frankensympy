//! Stationary transport must bind the exact requested statement and recheck it.

use frankensympy::objective::ObjectiveProblem;
use frankensympy::objective_transport::{
    MAX_STATIONARY_CERTIFICATE_BYTES, STATIONARY_CERTIFICATE_SCHEMA_VERSION,
    export_stationary_certificate, replay_stationary_certificate,
};
use frankensympy::{CertificationError, Expr, RealBall, RootBoxWitness, Symbol};
use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{
    BallError, BigInt, BigRational, MAX_ROOT_BOX_VARIABLES, RootBoxError, StationaryPointKind,
    StationaryPointReceipt, SymbolIdentity,
};
use serde::Serialize;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn problem(source: &str) -> ObjectiveProblem {
    ObjectiveProblem::parse_source(source, &["x".into()], &mut Unbounded).unwrap()
}

fn witness() -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(q(1, 1), q(1, 2)).unwrap()],
        center: vec![q(1, 1)],
        preconditioner: vec![vec![q(1, 2)]],
        precision_digits: 24,
    }
}

fn receipt() -> StationaryPointReceipt {
    problem("(x-1)^2")
        .verify_stationary(&witness(), &mut Unbounded)
        .unwrap()
}

#[derive(Serialize)]
struct Proposal<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    objective: &'a Expr,
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    witness: &'a RootBoxWitness,
}

fn proposal(problem: &ObjectiveProblem, witness: &RootBoxWitness) -> Vec<u8> {
    serde_json::to_vec(&Proposal {
        schema_version: STATIONARY_CERTIFICATE_SCHEMA_VERSION,
        context_digest: [7; 32],
        objective: problem.objective(),
        variables: problem.variables(),
        domain: &witness.domain,
        witness,
    })
    .unwrap()
}

#[test]
fn fresh_receiver_recomputes_the_point_and_its_minimum_classification() {
    let accepted = receipt();
    let bytes = export_stationary_certificate(&accepted, [7; 32], &mut Unbounded).unwrap();
    let replayed = replay_stationary_certificate(
        &bytes,
        &problem("(x-1)^2"),
        &witness().domain,
        [7; 32],
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(replayed, accepted);
    assert_eq!(replayed.kind(), StationaryPointKind::StrictMinimum);
    assert_eq!(replayed.stationary_enclosure(), &[RealBall::from_i64(1)]);
    assert_eq!(replayed.hessian_enclosure(), &[vec![RealBall::from_i64(2)]]);
    assert_eq!(replayed.contraction_bound(), &q(0, 1));
    assert_eq!(
        export_stationary_certificate(&replayed, [7; 32], &mut Unbounded).unwrap(),
        bytes
    );
    let decoded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded.as_object().unwrap().len(), 6);
    for key in [
        "accepted",
        "verified",
        "kind",
        "hessian",
        "stationary_enclosure",
    ] {
        assert!(decoded.get(key).is_none());
    }
}

#[test]
fn receiver_reclassifies_maximum_saddle_and_unclassified_stationary_points() {
    let cases = [
        (
            "-x^2-y^2",
            vec![vec![q(-1, 2), q(0, 1)], vec![q(0, 1), q(-1, 2)]],
            StationaryPointKind::StrictMaximum,
        ),
        (
            "x^2-y^2",
            vec![vec![q(1, 2), q(0, 1)], vec![q(0, 1), q(-1, 2)]],
            StationaryPointKind::Saddle,
        ),
        (
            "x*y",
            vec![vec![q(0, 1), q(1, 1)], vec![q(1, 1), q(0, 1)]],
            StationaryPointKind::Unclassified,
        ),
    ];
    for (source, preconditioner, kind) in cases {
        let problem =
            ObjectiveProblem::parse_source(source, &["x".into(), "y".into()], &mut Unbounded)
                .unwrap();
        let witness = RootBoxWitness {
            domain: vec![RealBall::new(q(0, 1), q(1, 2)).unwrap(); 2],
            center: vec![q(0, 1); 2],
            preconditioner,
            precision_digits: 16,
        };
        let replayed = replay_stationary_certificate(
            &proposal(&problem, &witness),
            &problem,
            &witness.domain,
            [7; 32],
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(replayed.kind(), kind, "{source}");
        assert_eq!(
            replayed.stationary_enclosure(),
            &[RealBall::from_i64(0), RealBall::from_i64(0)]
        );
    }
}

#[test]
fn changed_source_context_box_and_symbol_identity_cannot_reuse_a_certificate() {
    let bytes = export_stationary_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let expected = problem("(x-1)^2");
    let domain = witness().domain;
    // A constant shift leaves every stationary point and derivative unchanged,
    // yet changes the objective claim and must not reuse this receipt.
    assert_eq!(
        replay_stationary_certificate(
            &bytes,
            &problem("(x-1)^2+1"),
            &domain,
            [7; 32],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::SourceMismatch,
    );
    assert_eq!(
        replay_stationary_certificate(&bytes, &expected, &domain, [8; 32], &mut Unbounded)
            .unwrap_err(),
        CertificationError::UniverseMismatch {
            found: [7; 32],
            expected: [8; 32]
        },
    );
    let other_box = [RealBall::new(q(1, 1), q(1, 4)).unwrap()];
    assert_eq!(
        replay_stationary_certificate(&bytes, &expected, &other_box, [7; 32], &mut Unbounded)
            .unwrap_err(),
        CertificationError::DomainMismatch,
    );
    let other_identity = ObjectiveProblem::new(
        expected.objective().clone(),
        vec![Symbol::with_identity(
            "x",
            SymbolIdentity::from_assumptions_digest([9; 32]),
        )],
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        replay_stationary_certificate(&bytes, &other_identity, &domain, [7; 32], &mut Unbounded)
            .unwrap_err(),
        CertificationError::SourceMismatch,
    );
}

#[test]
fn coordinate_order_is_part_of_the_receiver_statement() {
    let original =
        ObjectiveProblem::parse_source("x^2+y^2", &["x".into(), "y".into()], &mut Unbounded)
            .unwrap();
    let reordered =
        ObjectiveProblem::parse_source("x^2+y^2", &["y".into(), "x".into()], &mut Unbounded)
            .unwrap();
    let witness = RootBoxWitness {
        domain: vec![RealBall::new(q(0, 1), q(1, 2)).unwrap(); 2],
        center: vec![q(0, 1); 2],
        preconditioner: vec![vec![q(1, 2), q(0, 1)], vec![q(0, 1), q(1, 2)]],
        precision_digits: 24,
    };
    let bytes = proposal(&original, &witness);
    assert_eq!(
        replay_stationary_certificate(&bytes, &reordered, &witness.domain, [7; 32], &mut Unbounded)
            .unwrap_err(),
        CertificationError::SourceMismatch,
    );
}

#[test]
fn original_hidden_poles_are_rechecked_during_replay() {
    let original = problem("(x-1)^2+0/(x-1)");
    let witness = witness();
    assert!(matches!(
        replay_stationary_certificate(
            &proposal(&original, &witness),
            &original,
            &witness.domain,
            [7; 32],
            &mut Unbounded
        ),
        Err(CertificationError::Root(RootBoxError::Ball(
            BallError::DivisionByZero(_)
        )))
    ));
}

#[test]
fn canonical_forged_preconditioner_and_wrong_center_are_rejected() {
    let original = problem("(x-1)^2");
    let mut forged = witness();
    forged.preconditioner[0][0] = q(0, 1);
    assert_eq!(
        replay_stationary_certificate(
            &proposal(&original, &forged),
            &original,
            &forged.domain,
            [7; 32],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::Root(RootBoxError::NotContractive),
    );
    forged = witness();
    forged.center[0] = q(2, 1);
    assert_eq!(
        replay_stationary_certificate(
            &proposal(&original, &forged),
            &original,
            &forged.domain,
            [7; 32],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::Root(RootBoxError::CenterOutsideBox { coordinate: 0 }),
    );
}

#[test]
fn inconsistent_embedded_box_is_rejected_even_with_a_valid_alternative_witness() {
    let original = problem("(x-1)^2");
    let witness = witness();
    let other = [RealBall::new(q(1, 1), q(1, 4)).unwrap()];
    let bytes = serde_json::to_vec(&Proposal {
        schema_version: STATIONARY_CERTIFICATE_SCHEMA_VERSION,
        context_digest: [7; 32],
        objective: original.objective(),
        variables: original.variables(),
        domain: &other,
        witness: &witness,
    })
    .unwrap();
    assert_eq!(
        replay_stationary_certificate(&bytes, &original, &other, [7; 32], &mut Unbounded)
            .unwrap_err(),
        CertificationError::DomainMismatch,
    );
}

#[test]
fn transport_rejects_stored_conclusions_duplicate_keys_and_noncanonical_bytes() {
    let bytes = export_stationary_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    let original = problem("(x-1)^2");
    let mutants = [
        format!(" {text}"),
        text.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        ),
        text.replacen(
            "\"precision_digits\":24",
            "\"precision_digits\":24,\"verified\":true",
            1,
        ),
        text.replacen(
            "\"objective\":",
            "\"classification\":\"strict_minimum\",\"objective\":",
            1,
        ),
        text.replacen("\"name\":\"x\"", "\"name\":\"x\",\"ignored\":true", 1),
        format!("{text} null"),
    ];
    for mutant in mutants {
        assert!(
            replay_stationary_certificate(
                mutant.as_bytes(),
                &original,
                &witness().domain,
                [7; 32],
                &mut Unbounded
            )
            .is_err()
        );
    }
    let future = text.replacen("\"schema_version\":1", "\"schema_version\":2", 1);
    assert_eq!(
        replay_stationary_certificate(
            future.as_bytes(),
            &original,
            &witness().domain,
            [7; 32],
            &mut Unbounded
        )
        .unwrap_err(),
        CertificationError::UnsupportedVersion {
            found: 2,
            supported: 1
        },
    );
}

#[test]
fn payload_bounds_fail_before_unbounded_witness_allocation() {
    let original = problem("(x-1)^2");
    let mut malformed = witness();
    malformed.preconditioner = vec![vec![q(1, 1); MAX_ROOT_BOX_VARIABLES + 1]];
    assert!(matches!(
        replay_stationary_certificate(
            &proposal(&original, &malformed),
            &original,
            &malformed.domain,
            [7; 32],
            &mut Unbounded
        ),
        Err(CertificationError::Encoding(_))
    ));
    let oversized = vec![b' '; MAX_STATIONARY_CERTIFICATE_BYTES + 1];
    assert!(matches!(
        replay_stationary_certificate(
            &oversized,
            &original,
            &witness().domain,
            [7; 32],
            &mut Unbounded
        ),
        Err(CertificationError::ByteLimit { .. })
    ));
    let depth = frankensympy::certification::MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1;
    let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
    assert!(matches!(
        replay_stationary_certificate(
            nested.as_bytes(),
            &original,
            &witness().domain,
            [7; 32],
            &mut Unbounded
        ),
        Err(CertificationError::DepthLimit { .. })
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
        assert!(!self.stopped);
        for &(dimension, amount) in charges {
            assert_ne!(amount, 0);
            self.charges[dimension.index()] += amount;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.checkpoints += 1;
        if self.interrupt == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_export_and_replay_checkpoint_never_returns_an_artifact_or_receipt() {
    let accepted = receipt();
    let bytes = export_stationary_certificate(&accepted, [7; 32], &mut Unbounded).unwrap();
    let original = problem("(x-1)^2");
    for exporting in [true, false] {
        let operation = |meter: &mut Probe| {
            if exporting {
                export_stationary_certificate(&accepted, [7; 32], meter).map(|_| ())
            } else {
                replay_stationary_certificate(&bytes, &original, &witness().domain, [7; 32], meter)
                    .map(|_| ())
            }
        };
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
}

#[test]
fn retries_consume_one_ledger_and_plain_meters_cannot_spend_verifier_reserves() {
    let bytes = export_stationary_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let original = problem("(x-1)^2");
    let domain = witness().domain;
    let mut measured = Probe::default();
    replay_stationary_certificate(&bytes, &original, &domain, [7; 32], &mut measured).unwrap();
    let mut shared = Budget::new(BudgetLimits {
        dimensions: measured.charges.map(|x| x * 2),
        verifier_pool: 29,
    });
    for _ in 0..2 {
        replay_stationary_certificate(&bytes, &original, &domain, [7; 32], &mut shared).unwrap();
    }
    assert_eq!(shared.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(shared.remaining(Dimension::MemoryBytes), 0);
    assert!(matches!(
        replay_stationary_certificate(&bytes, &original, &domain, [7; 32], &mut shared),
        Err(CertificationError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(shared.verifier_remaining(), 29);
}

#[test]
fn a_receiving_verifier_lease_authorizes_the_independent_recheck() {
    let bytes = export_stationary_certificate(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let original = problem("(x-1)^2");
    let mut budget = Budget::new(BudgetLimits::uniform(1 << 40, 1 << 30));
    let lease = budget.verifier_lease().unwrap();
    let before = budget.verifier_remaining();
    let accepted = replay_stationary_certificate(
        &bytes,
        &original,
        &witness().domain,
        [7; 32],
        &mut budget.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
    assert!(budget.verifier_remaining() < before);
}
