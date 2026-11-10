//! Native integral computation, independently bound transport and resource refusal.

use frankensympy::certification::{CertificationError, ResidualProblem};
use frankensympy::integration::{
    INTEGRAL_CERTIFICATE_SCHEMA_VERSION, IntegrationError, MAX_INTEGRAL_CERTIFICATE_BYTES,
    export_integral_certificate, replay_integral_certificate,
};
use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_calculus::quadrature::QuadratureOptions;
use fsym_core::{
    BigInt, BigRational, Expr, IntegralEnclosureReceipt, MAX_QUADRATURE_INPUT_BITS,
    MAX_QUADRATURE_INTERVALS, QuadratureError, QuadratureWitness, RootBoxError, Symbol,
    SymbolIdentity,
};
use serde::Serialize;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn problem(source: &str) -> ResidualProblem {
    ResidualProblem::parse_sources(&[source.into()], &["x".into()], &mut Unbounded).unwrap()
}

fn options() -> QuadratureOptions {
    QuadratureOptions {
        precision_digits: 16,
        max_intervals: 64,
    }
}

fn checked(
    problem: &ResidualProblem,
    lower: &BigRational,
    upper: &BigRational,
    tolerance: &BigRational,
) -> IntegralEnclosureReceipt {
    let candidate = problem
        .propose_integral(lower, upper, tolerance, options(), &mut Unbounded)
        .unwrap();
    problem
        .verify_integral(lower, upper, tolerance, &candidate, &mut Unbounded)
        .unwrap()
}

#[derive(Serialize)]
struct ProposedArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    lower: &'a BigRational,
    upper: &'a BigRational,
    absolute_tolerance: &'a BigRational,
    witness: &'a QuadratureWitness,
}

fn proposed_bytes(
    problem: &ResidualProblem,
    lower: &BigRational,
    upper: &BigRational,
    tolerance: &BigRational,
    witness: &QuadratureWitness,
) -> Vec<u8> {
    serde_json::to_vec(&ProposedArtifact {
        schema_version: INTEGRAL_CERTIFICATE_SCHEMA_VERSION,
        context_digest: [7; 32],
        residuals: problem.residuals(),
        variables: problem.variables(),
        lower,
        upper,
        absolute_tolerance: tolerance,
        witness,
    })
    .unwrap()
}

fn replay(
    bytes: &[u8],
    problem: &ResidualProblem,
) -> Result<IntegralEnclosureReceipt, IntegrationError> {
    replay_integral_certificate(
        bytes,
        problem,
        &q(0, 1),
        &q(1, 1),
        &q(1, 1000),
        [7; 32],
        &mut Unbounded,
    )
}

#[test]
fn exact_quadratic_integral_has_independent_value_and_original_source() {
    let problem = problem("2*x^2+5*x-7");
    let tolerance = BigRational::new(1.into(), BigInt::from(10).pow(100));
    let accepted = checked(&problem, &q(-2, 1), &q(3, 1), &tolerance);
    // [2*x^3/3 + 5*x^2/2 - 7*x] evaluated at 3 and -2 is 5/6.
    assert_eq!(accepted.enclosure().lower(), q(5, 6));
    assert_eq!(accepted.enclosure().upper(), q(5, 6));
    assert_eq!(accepted.interval_count(), 1);
    assert_eq!(accepted.absolute_tolerance(), &tolerance);
    assert_eq!(accepted.integrand(), &problem.residuals()[0]);
}

fn gaussian_partial_sum(last: i64) -> BigRational {
    let mut factorial = BigInt::from(1);
    let mut total = q(0, 1);
    for n in 0..=last {
        if n > 0 {
            factorial *= BigInt::from(n);
        }
        total += BigRational::new(
            BigInt::from(if n % 2 == 0 { 1 } else { -1 }),
            &factorial * BigInt::from(2 * n + 1),
        );
    }
    total
}

#[test]
fn non_elementary_gaussian_integral_agrees_with_independent_series_bounds() {
    let problem = problem("exp(-x^2)");
    let accepted = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    // Integrating alternating Taylor terms gives lower/upper bounds without
    // calling the native interval evaluator or symbolic integration engine.
    assert!(accepted.enclosure().contains(&gaussian_partial_sum(11)));
    assert!(accepted.enclosure().contains(&gaussian_partial_sum(12)));
    assert!(accepted.enclosure().width() <= q(1, 1000));
    assert!(accepted.interval_count() > 1);
}

#[test]
fn reversed_and_zero_length_intervals_keep_their_proper_real_contract() {
    let problem = problem("x^2");
    let forward = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    let reverse = checked(&problem, &q(1, 1), &q(0, 1), &q(1, 1000));
    assert_eq!(reverse.enclosure(), &forward.enclosure().neg());
    assert_eq!(reverse.witness(), forward.witness());
    let point = checked(&problem, &q(2, 1), &q(2, 1), &q(1, 1000));
    assert_eq!(point.enclosure().lower(), q(0, 1));
    assert_eq!(point.witness().partition, vec![q(2, 1)]);
    let pole = self::problem("0*(1/x)");
    assert!(
        pole.propose_integral(&q(0, 1), &q(0, 1), &q(1, 1000), options(), &mut Unbounded)
            .is_err()
    );
}

#[test]
fn original_source_poles_unknown_functions_and_wrong_dimensions_refuse() {
    for source in ["0*(1/x)", "(1/x)^0", "log(x)-log(x)", "gamma(x)", "y"] {
        let problem = problem(source);
        assert!(
            problem
                .propose_integral(&q(0, 1), &q(1, 1), &q(1, 1000), options(), &mut Unbounded)
                .is_err(),
            "{source}"
        );
    }
    for (sources, variables) in [
        (vec!["x".into(), "x^2".into()], vec!["x".into()]),
        (vec!["x+y".into()], vec!["x".into(), "y".into()]),
    ] {
        let problem = ResidualProblem::parse_sources(&sources, &variables, &mut Unbounded).unwrap();
        assert!(matches!(
            problem.propose_integral(&q(0, 1), &q(1, 1), &q(1, 1000), options(), &mut Unbounded),
            Err(IntegrationError::ProblemDimension)
        ));
        assert!(matches!(
            problem.verify_integral(
                &q(0, 1),
                &q(1, 1),
                &q(1, 1000),
                &QuadratureWitness {
                    partition: vec![q(0, 1), q(1, 1)],
                    precision_digits: 16
                },
                &mut Unbounded
            ),
            Err(IntegrationError::ProblemDimension)
        ));
    }
}

#[test]
fn canonical_roundtrip_recomputes_and_exports_no_cached_integral() {
    let problem = problem("x^4");
    let accepted = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_integral_certificate(&problem, &accepted, [7; 32], &mut Unbounded).unwrap();
    let fresh = self::problem("x^4");
    let replayed = replay(&bytes, &fresh).unwrap();
    assert_eq!(replayed, accepted);
    assert!(replayed.enclosure().contains(&q(1, 5)));
    assert_eq!(
        export_integral_certificate(&fresh, &replayed, [7; 32], &mut Unbounded).unwrap(),
        bytes
    );
    let object: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let fields = object.as_object().unwrap();
    assert_eq!(fields.len(), 8);
    for field in ["accepted", "verified", "enclosure", "integral", "receipt"] {
        assert!(!fields.contains_key(field));
    }
}

#[test]
fn transport_binds_raw_source_and_symbol_identity_without_using_printed_names() {
    let symbol = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let problem = ResidualProblem::new(
        vec![Expr::Sym(symbol.clone())],
        vec![symbol],
        &mut Unbounded,
    )
    .unwrap();
    let accepted = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_integral_certificate(&problem, &accepted, [7; 32], &mut Unbounded).unwrap();
    assert!(matches!(
        replay(&bytes, &self::problem("x")),
        Err(IntegrationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    assert!(matches!(
        export_integral_certificate(&self::problem("x"), &accepted, [7; 32], &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    let plain = self::problem("x");
    let plain_receipt = checked(&plain, &q(0, 1), &q(1, 1), &q(1, 1000));
    let plain_bytes =
        export_integral_certificate(&plain, &plain_receipt, [7; 32], &mut Unbounded).unwrap();
    assert!(matches!(
        replay(&plain_bytes, &self::problem("x+0")),
        Err(IntegrationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
}

#[test]
fn foreign_context_endpoints_and_tolerance_refuse_before_witness_construction() {
    let problem = problem("x^2");
    let accepted = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_integral_certificate(&problem, &accepted, [7; 32], &mut Unbounded).unwrap();
    let mut malformed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    malformed["witness"] = serde_json::json!({"partition":"not a partition"});
    let malformed = serde_json::to_vec(&malformed).unwrap();
    for (lower, upper, tolerance) in [
        (q(1, 1), q(0, 1), q(1, 1000)),
        (q(0, 1), q(2, 1), q(1, 1000)),
        (q(0, 1), q(1, 1), q(1, 100)),
    ] {
        assert!(matches!(
            replay_integral_certificate(
                &malformed,
                &problem,
                &lower,
                &upper,
                &tolerance,
                [7; 32],
                &mut Unbounded
            ),
            Err(IntegrationError::StatementMismatch)
        ));
    }
    assert!(matches!(
        replay_integral_certificate(
            &malformed,
            &problem,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            [8; 32],
            &mut Unbounded
        ),
        Err(IntegrationError::Certification(
            CertificationError::UniverseMismatch { .. }
        ))
    ));
    assert!(matches!(
        replay(&malformed, &self::problem("x^3")),
        Err(IntegrationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
}

#[test]
fn a_canonical_partition_is_not_authority_for_accuracy_or_source_domain() {
    let witness = QuadratureWitness {
        partition: vec![q(0, 1), q(1, 1)],
        precision_digits: 16,
    };
    let quartic = problem("x^4");
    let bytes = proposed_bytes(&quartic, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
    assert!(matches!(
        replay(&bytes, &quartic),
        Err(IntegrationError::Quadrature(
            QuadratureError::AccuracyNotReached
        ))
    ));
    let concealed_pole = problem("0*(1/x)");
    let bytes = proposed_bytes(&concealed_pole, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
    assert!(matches!(
        replay(&bytes, &concealed_pole),
        Err(IntegrationError::Quadrature(QuadratureError::Enclosure(_)))
    ));
}

#[test]
fn incomplete_overlapping_and_duplicate_partitions_refuse() {
    let problem = problem("x^2");
    for partition in [
        vec![],
        vec![q(0, 1), q(1, 2)],
        vec![q(1, 2), q(1, 1)],
        vec![q(0, 1), q(1, 2), q(1, 2), q(1, 1)],
        vec![q(0, 1), q(3, 4), q(1, 4), q(1, 1)],
    ] {
        let witness = QuadratureWitness {
            partition,
            precision_digits: 16,
        };
        let bytes = proposed_bytes(&problem, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
        assert!(matches!(
            replay(&bytes, &problem),
            Err(IntegrationError::Quadrature(
                QuadratureError::InvalidPartition(_)
            ))
        ));
    }
}

#[test]
fn unknown_schema_fields_duplicate_keys_and_noncanonical_bytes_fail_closed() {
    let problem = problem("x^2");
    let accepted = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_integral_certificate(&problem, &accepted, [7; 32], &mut Unbounded).unwrap();
    let mut object: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    object["schema_version"] = serde_json::json!(2);
    assert!(matches!(
        replay(&serde_json::to_vec(&object).unwrap(), &problem),
        Err(IntegrationError::Certification(
            CertificationError::UnsupportedVersion { .. }
        ))
    ));
    object["schema_version"] = serde_json::json!(1);
    object["verified"] = serde_json::json!(true);
    assert!(matches!(
        replay(&serde_json::to_vec(&object).unwrap(), &problem),
        Err(IntegrationError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    object.as_object_mut().unwrap().remove("verified");
    object["witness"]["integral"] = serde_json::json!(0);
    assert!(matches!(
        replay(&serde_json::to_vec(&object).unwrap(), &problem),
        Err(IntegrationError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    let duplicate = format!(
        "{{\"schema_version\":1,{}",
        std::str::from_utf8(&bytes[1..]).unwrap()
    );
    assert!(matches!(
        replay(duplicate.as_bytes(), &problem),
        Err(IntegrationError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    let mut padded = bytes.clone();
    padded.push(b'\n');
    assert!(matches!(
        replay(&padded, &problem),
        Err(IntegrationError::Certification(
            CertificationError::NonCanonical
        ))
    ));
}

#[test]
fn length_depth_partition_and_rational_height_preflight_refuse() {
    let problem = problem("x^2");
    assert!(matches!(
        replay(&vec![b' '; MAX_INTEGRAL_CERTIFICATE_BYTES + 1], &problem),
        Err(IntegrationError::Certification(
            CertificationError::ByteLimit { .. }
        ))
    ));
    let nested = format!("{}0{}", "[".repeat(100), "]".repeat(100));
    assert!(matches!(
        replay(nested.as_bytes(), &problem),
        Err(IntegrationError::Certification(
            CertificationError::DepthLimit { .. }
        ))
    ));
    for witness in [
        QuadratureWitness {
            partition: vec![q(0, 1); MAX_QUADRATURE_INTERVALS + 2],
            precision_digits: 16,
        },
        QuadratureWitness {
            partition: vec![q(0, 1), q(1, 1)],
            precision_digits: 7,
        },
        QuadratureWitness {
            partition: vec![
                q(0, 1),
                BigRational::from_integer(BigInt::from(1) << MAX_QUADRATURE_INPUT_BITS as u32),
                q(1, 1),
            ],
            precision_digits: 16,
        },
    ] {
        let bytes = proposed_bytes(&problem, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
        assert!(matches!(
            replay(&bytes, &problem),
            Err(IntegrationError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
}

#[test]
fn proposal_and_protected_verification_share_one_cumulative_ledger_sequentially() {
    let problem = problem("x^2");
    let mut budget = Budget::new(BudgetLimits::uniform(1 << 40, 1 << 30));
    let lease = budget.verifier_lease().unwrap();
    let verifier_before = budget.verifier_remaining();
    let candidate = problem
        .propose_integral(&q(0, 1), &q(1, 1), &q(1, 1000), options(), &mut budget)
        .unwrap();
    assert_eq!(budget.verifier_remaining(), verifier_before);
    assert!(budget.remaining(Dimension::ComputeSteps) < 1 << 40);
    let ordinary_remaining = budget.remaining(Dimension::ComputeSteps);
    budget
        .try_charge(Dimension::ComputeSteps, ordinary_remaining)
        .unwrap();
    assert!(
        problem
            .propose_integral(&q(0, 1), &q(1, 1), &q(1, 1000), options(), &mut budget,)
            .is_err()
    );
    let accepted = problem
        .verify_integral(
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            &candidate,
            &mut budget.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
    assert_eq!(accepted.enclosure().lower(), q(1, 3));
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert!(budget.verifier_remaining() < verifier_before);
}

#[derive(Default)]
struct Counted {
    charges: [u64; DIMENSION_COUNT],
}

impl BudgetMeter for Counted {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charges[dimension.index()] += amount;
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        Ok(())
    }
}

#[test]
fn repeated_replay_cannot_reset_compute_memory_or_allocation_charges() {
    let problem = problem("x^2");
    let accepted = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_integral_certificate(&problem, &accepted, [7; 32], &mut Unbounded).unwrap();
    let mut counted = Counted::default();
    replay_integral_certificate(
        &bytes,
        &problem,
        &q(0, 1),
        &q(1, 1),
        &q(1, 1000),
        [7; 32],
        &mut counted,
    )
    .unwrap();
    let mut budget = Budget::new(BudgetLimits {
        dimensions: counted.charges.map(|value| value * 2),
        verifier_pool: 193,
    });
    for _ in 0..2 {
        replay_integral_certificate(
            &bytes,
            &problem,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            [7; 32],
            &mut budget,
        )
        .unwrap();
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(counted.charges[dimension.index()] > 0);
        assert_eq!(budget.remaining(dimension), 0);
    }
    assert!(
        replay_integral_certificate(
            &bytes,
            &problem,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            [7; 32],
            &mut budget,
        )
        .is_err()
    );
    assert_eq!(budget.verifier_remaining(), 193);
}

struct CancelAfter(usize);

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        self.checkpoint()
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.0 == 0 {
            return Err(MeterError::Cancelled);
        }
        self.0 -= 1;
        Ok(())
    }
}

fn cancelled(error: IntegrationError) {
    assert!(matches!(
        error,
        IntegrationError::Certification(CertificationError::Meter(MeterError::Cancelled))
            | IntegrationError::Quadrature(QuadratureError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
    ));
}

#[test]
fn cancellation_in_proposal_encoding_and_replay_never_returns_partial_results() {
    let problem = problem("x^2");
    let accepted = checked(&problem, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_integral_certificate(&problem, &accepted, [7; 32], &mut Unbounded).unwrap();
    for checkpoint in [0, 1, 20] {
        cancelled(
            problem
                .propose_integral(
                    &q(0, 1),
                    &q(1, 1),
                    &q(1, 1000),
                    options(),
                    &mut CancelAfter(checkpoint),
                )
                .unwrap_err(),
        );
        cancelled(
            export_integral_certificate(&problem, &accepted, [7; 32], &mut CancelAfter(checkpoint))
                .unwrap_err(),
        );
        cancelled(
            replay_integral_certificate(
                &bytes,
                &problem,
                &q(0, 1),
                &q(1, 1),
                &q(1, 1000),
                [7; 32],
                &mut CancelAfter(checkpoint),
            )
            .unwrap_err(),
        );
    }
}
