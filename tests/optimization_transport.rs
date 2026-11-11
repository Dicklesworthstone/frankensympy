//! Portable global bounds bind the receiver's full mathematical statement.

use frankensympy::CertificationError;
use frankensympy::objective::ObjectiveProblem;
use frankensympy::optimization::{
    BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION, MAX_BOX_OPTIMIZATION_ARTIFACT_BYTES,
    OptimizationError, export_box_optimization, replay_box_optimization,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::BoxOptimizationOptions;
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationReceipt,
    BoxOptimizationWitness, BoxPartitionStep, Expr, RealBall, RootBoxError, Symbol, SymbolIdentity,
};
use serde::Serialize;
use serde_json::json;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn problem(source: &str) -> ObjectiveProblem {
    ObjectiveProblem::parse_source(source, &["x".into()], &mut Unbounded).unwrap()
}

fn witness() -> BoxOptimizationWitness {
    BoxOptimizationWitness {
        partition: vec![BoxPartitionStep::Leaf],
        point: vec![q(0, 1)],
        precision_digits: 16,
    }
}

fn receipt() -> BoxOptimizationReceipt {
    problem("x^2")
        .verify_box_optimization(
            &[interval(-1, 1)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
            &witness(),
            &mut Unbounded,
        )
        .unwrap()
}

#[derive(Serialize)]
struct Proposal<'a, T: Serialize + ?Sized> {
    schema_version: u32,
    context_digest: [u8; 32],
    objective: &'a Expr,
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    kind: BoxExtremumKind,
    absolute_tolerance: &'a BigRational,
    witness: &'a T,
}

fn proposal<T: Serialize + ?Sized>(
    source: &ObjectiveProblem,
    domain: &[RealBall],
    kind: BoxExtremumKind,
    tolerance: &BigRational,
    witness: &T,
) -> Vec<u8> {
    serde_json::to_vec(&Proposal {
        schema_version: BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION,
        context_digest: [7; 32],
        objective: source.objective(),
        variables: source.variables(),
        domain,
        kind,
        absolute_tolerance: tolerance,
        witness,
    })
    .unwrap()
}

fn replay(bytes: &[u8]) -> Result<BoxOptimizationReceipt, OptimizationError> {
    replay_box_optimization(
        bytes,
        &problem("x^2"),
        &[interval(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
        [7; 32],
        &mut Unbounded,
    )
}

#[test]
fn a_fresh_receiver_recomputes_global_bounds_without_serialized_conclusions() {
    let accepted = receipt();
    let bytes = export_box_optimization(&accepted, [7; 32], &mut Unbounded).unwrap();
    let restored = replay(&bytes).unwrap();
    assert_eq!(restored, accepted);
    assert_eq!(restored.value(), &RealBall::from_i64(0));
    assert_eq!(restored.gap(), &q(0, 1));
    assert_eq!(
        export_box_optimization(&restored, [7; 32], &mut Unbounded).unwrap(),
        bytes
    );
    let artifact: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(artifact.as_object().unwrap().len(), 8);
    for name in [
        "accepted",
        "verified",
        "value",
        "point_value",
        "gap",
        "leaf_count",
    ] {
        assert!(artifact.get(name).is_none());
    }
    assert_eq!(artifact["kind"], "minimum");
}

#[test]
fn maximum_and_fixed_coordinate_certificates_round_trip_exactly() {
    for (source, domain, kind, value) in [
        ("x", interval(-1, 2), BoxExtremumKind::Maximum, 2),
        ("x^2", RealBall::from_i64(3), BoxExtremumKind::Minimum, 9),
    ] {
        let objective = problem(source);
        let tolerance = q(1, 1000);
        let witness = objective
            .propose_box_optimization(
                std::slice::from_ref(&domain),
                kind,
                &tolerance,
                BoxOptimizationOptions {
                    precision_digits: 16,
                    max_leaves: 8,
                },
                &mut Unbounded,
            )
            .unwrap();
        let checked = objective
            .verify_box_optimization(
                std::slice::from_ref(&domain),
                kind,
                &tolerance,
                &witness,
                &mut Unbounded,
            )
            .unwrap();
        let bytes = export_box_optimization(&checked, [7; 32], &mut Unbounded).unwrap();
        let restored = replay_box_optimization(
            &bytes,
            &objective,
            &[domain],
            kind,
            &tolerance,
            [7; 32],
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(restored, checked);
        assert_eq!(restored.value(), &RealBall::from_i64(value));
    }
}

#[test]
fn every_receiver_statement_field_is_bound_before_witness_decoding() {
    let original = problem("x^2");
    let domain = [interval(-1, 1)];
    let tolerance = q(1, 1000);
    // Malformed witness proves mismatches are detected before its decoder runs.
    let bytes = proposal(
        &original,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        &json!(null),
    );
    assert!(matches!(
        replay_box_optimization(
            &bytes,
            &problem("x^2+1"),
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            [7; 32],
            &mut Unbounded
        ),
        Err(OptimizationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    let foreign_axis = ObjectiveProblem::new(
        original.objective().clone(),
        vec![Symbol::with_identity(
            "x",
            SymbolIdentity::from_assumptions_digest([8; 32]),
        )],
        &mut Unbounded,
    )
    .unwrap();
    assert!(matches!(
        replay_box_optimization(
            &bytes,
            &foreign_axis,
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            [7; 32],
            &mut Unbounded
        ),
        Err(OptimizationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    assert!(matches!(
        replay_box_optimization(
            &bytes,
            &original,
            &[interval(-2, 2)],
            BoxExtremumKind::Minimum,
            &tolerance,
            [7; 32],
            &mut Unbounded
        ),
        Err(OptimizationError::Certification(
            CertificationError::DomainMismatch
        ))
    ));
    assert!(matches!(
        replay_box_optimization(
            &bytes,
            &original,
            &domain,
            BoxExtremumKind::Maximum,
            &tolerance,
            [7; 32],
            &mut Unbounded
        ),
        Err(OptimizationError::StatementMismatch)
    ));
    // Even a weaker requested tolerance is a distinct immutable statement.
    assert!(matches!(
        replay_box_optimization(
            &bytes,
            &original,
            &domain,
            BoxExtremumKind::Minimum,
            &q(1, 10),
            [7; 32],
            &mut Unbounded
        ),
        Err(OptimizationError::StatementMismatch)
    ));
    assert!(matches!(
        replay_box_optimization(
            &bytes,
            &original,
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            [8; 32],
            &mut Unbounded
        ),
        Err(OptimizationError::Certification(
            CertificationError::UniverseMismatch { .. }
        ))
    ));
}

#[test]
fn complete_coverage_point_feasibility_and_requested_accuracy_are_rechecked() {
    let source = problem("x^2");
    let domain = [interval(-1, 1)];
    let tolerance = q(1, 1000);
    let mut proposed = witness();
    proposed.partition = vec![BoxPartitionStep::Split { axis: 0 }, BoxPartitionStep::Leaf];
    assert!(matches!(
        replay(&proposal(
            &source,
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            &proposed
        )),
        Err(OptimizationError::Optimization(
            BoxOptimizationError::InvalidPartition(_)
        ))
    ));
    proposed = witness();
    proposed.point[0] = q(2, 1);
    assert!(matches!(
        replay(&proposal(
            &source,
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            &proposed
        )),
        Err(OptimizationError::Optimization(
            BoxOptimizationError::Verification(RootBoxError::CenterOutsideBox { .. })
        ))
    ));
    proposed.point[0] = q(1, 1);
    assert!(matches!(
        replay(&proposal(
            &source,
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            &proposed
        )),
        Err(OptimizationError::Optimization(
            BoxOptimizationError::AccuracyNotReached
        ))
    ));
}

#[test]
fn original_source_domains_survive_portable_partition_replay() {
    for text in ["0*(1/x)", "log(x)-log(x)", "(1/x)^0", "sqrt(x)"] {
        let source = problem(text);
        let domain = [interval(-1, 1)];
        let tolerance = q(1, 1000);
        let bytes = proposal(
            &source,
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            &witness(),
        );
        assert!(
            replay_box_optimization(
                &bytes,
                &source,
                &domain,
                BoxExtremumKind::Minimum,
                &tolerance,
                [7; 32],
                &mut Unbounded
            )
            .is_err(),
            "{text}"
        );
    }
}

#[test]
fn canonical_encoding_rejects_whitespace_unknown_fields_and_alternate_scalars() {
    let bytes = export_box_optimization(&receipt(), [7; 32], &mut Unbounded).unwrap();
    let mut spaced = b" ".to_vec();
    spaced.extend_from_slice(&bytes);
    assert!(matches!(
        replay(&spaced),
        Err(OptimizationError::Certification(
            CertificationError::NonCanonical
        ))
    ));
    let mut unknown = bytes[..bytes.len() - 1].to_vec();
    unknown.extend_from_slice(b",\"accepted\":true}");
    assert!(matches!(
        replay(&unknown),
        Err(OptimizationError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    let mut duplicate = bytes[..bytes.len() - 1].to_vec();
    duplicate.extend_from_slice(b",\"kind\":\"minimum\"}");
    assert!(matches!(
        replay(&duplicate),
        Err(OptimizationError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    for scalar in [
        json!([[1, [2]], [1, [2]]]), // unreduced 2/2
        json!([[0, []], [0, []]]),   // zero denominator
        json!([[0, [0]], [1, [1]]]), // noncanonical zero magnitude
        json!([[0, []], [-1, [1]]]), // negative denominator
    ] {
        let mut proposed = serde_json::to_value(witness()).unwrap();
        proposed["point"][0] = scalar;
        assert!(matches!(
            replay(&proposal(
                &problem("x^2"),
                &[interval(-1, 1)],
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                &proposed
            )),
            Err(OptimizationError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
}

#[test]
fn hostile_byte_nesting_and_witness_lengths_are_bounded_before_acceptance() {
    let oversized = vec![b' '; MAX_BOX_OPTIMIZATION_ARTIFACT_BYTES + 1];
    assert!(matches!(
        replay(&oversized),
        Err(OptimizationError::Certification(
            CertificationError::ByteLimit { .. }
        ))
    ));
    let nesting = "[".repeat(100) + &"]".repeat(100);
    assert!(matches!(
        replay(nesting.as_bytes()),
        Err(OptimizationError::Certification(
            CertificationError::DepthLimit { .. }
        ))
    ));
    for field in ["point", "partition"] {
        let mut proposed = serde_json::to_value(witness()).unwrap();
        proposed[field] = if field == "point" {
            json!(vec![q(0, 1); 17])
        } else {
            json!(vec!["leaf"; 2048])
        };
        assert!(matches!(
            replay(&proposal(
                &problem("x^2"),
                &[interval(-1, 1)],
                BoxExtremumKind::Minimum,
                &q(1, 1000),
                &proposed
            )),
            Err(OptimizationError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
}

#[derive(Default)]
struct CancelAt {
    checkpoints: usize,
    at: usize,
    stopped: bool,
    charged: u64,
}

impl BudgetMeter for CancelAt {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.charged += amount;
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.checkpoints += 1;
        if self.checkpoints == self.at {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn export_and_replay_preserve_cumulative_work_and_terminal_cancellation() {
    let receipt = receipt();
    let source = problem("x^2");
    let domain = [interval(-1, 1)];
    let tolerance = q(1, 1000);
    let mut meter = CancelAt::default();
    let bytes = export_box_optimization(&receipt, [7; 32], &mut meter).unwrap();
    let export_work = meter.charged;
    let export_checkpoints = meter.checkpoints;
    replay_box_optimization(
        &bytes,
        &source,
        &domain,
        BoxExtremumKind::Minimum,
        &tolerance,
        [7; 32],
        &mut meter,
    )
    .unwrap();
    assert!(meter.charged > export_work);
    assert!(meter.checkpoints > export_checkpoints);
    let replay_checkpoints = meter.checkpoints - export_checkpoints;
    for at in [1, 2, replay_checkpoints / 2, replay_checkpoints] {
        let mut cancelled = CancelAt {
            at,
            ..CancelAt::default()
        };
        assert!(matches!(
            replay_box_optimization(
                &bytes,
                &source,
                &domain,
                BoxExtremumKind::Minimum,
                &tolerance,
                [7; 32],
                &mut cancelled
            ),
            Err(OptimizationError::Certification(CertificationError::Meter(
                MeterError::Cancelled
            ))) | Err(OptimizationError::Optimization(
                BoxOptimizationError::Verification(RootBoxError::Meter(MeterError::Cancelled))
            ))
        ));
        assert!(cancelled.stopped);
    }
    for at in [1, export_checkpoints] {
        let mut cancelled = CancelAt {
            at,
            ..CancelAt::default()
        };
        assert!(matches!(
            export_box_optimization(&receipt, [7; 32], &mut cancelled),
            Err(OptimizationError::Certification(CertificationError::Meter(
                MeterError::Cancelled
            )))
        ));
        assert!(cancelled.stopped);
    }
}
