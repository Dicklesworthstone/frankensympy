//! Portable parameter families are candidates until the receiver rechecks them.

use frankensympy::parametric::{ParametricError, ParametricProblem};
use frankensympy::parametric_transport::{
    MAX_PARAMETRIC_FAMILY_ARTIFACT_BYTES, PARAMETRIC_FAMILY_ARTIFACT_SCHEMA_VERSION,
    export_parametric_family, replay_parametric_family,
};
use frankensympy::{CertificationError, ParametricFamilyReceipt};
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::parametric_cover::ParametricRootCoverOptions;
use fsym_core::{
    BigRational, Expr, MAX_ROOT_BOX_LIMBS, ParametricRootBoxWitness, ParametricRootCoverStep,
    ParametricRootCoverWitness, RealBall, RootBoxError, Symbol, SymbolIdentity,
};
use serde::Serialize;
use serde_json::{Value, json};

const CONTEXT: [u8; 32] = [7; 32];

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn problem(source: &str) -> ParametricProblem {
    ParametricProblem::parse_sources(&[source], &["x"], &["p"], &mut Unbounded).unwrap()
}

fn root_witness() -> ParametricRootBoxWitness {
    ParametricRootBoxWitness {
        unknown_domain: vec![interval(-2, 2)],
        parameter_domain: vec![interval(-1, 1)],
        center: vec![q(0, 1)],
        preconditioner: vec![vec![q(1, 1)]],
        precision_digits: 16,
    }
}

fn accepted_affine() -> ParametricFamilyReceipt {
    ParametricFamilyReceipt::RootBox(
        problem("x-p")
            .verify_root_box(&root_witness(), &mut Unbounded)
            .unwrap(),
    )
}

fn adaptive() -> (ParametricProblem, ParametricFamilyReceipt) {
    let source = problem("(1+p^2)*x-1");
    let unknown_domain = [interval(0, 2)];
    let parameter_domain = [interval(0, 2)];
    let witness = source
        .propose_cover(
            &unknown_domain,
            &parameter_domain,
            ParametricRootCoverOptions {
                precision_digits: 16,
                max_leaves: 128,
            },
            &mut Unbounded,
        )
        .unwrap();
    assert!(witness.leaves.len() > 1);
    let receipt = source
        .verify_cover(&unknown_domain, &parameter_domain, &witness, &mut Unbounded)
        .unwrap();
    (source, ParametricFamilyReceipt::Cover(receipt))
}

// Deliberately build arbitrary proposals without constructing private accepted
// core receipts. The field order is the public canonical transport order.
#[derive(Clone, Serialize)]
struct RootWire {
    unknown_domain: Vec<RealBall>,
    parameter_domain: Vec<RealBall>,
    center: Vec<BigRational>,
    preconditioner: Vec<Vec<BigRational>>,
    precision_digits: u32,
}

impl From<&ParametricRootBoxWitness> for RootWire {
    fn from(value: &ParametricRootBoxWitness) -> Self {
        Self {
            unknown_domain: value.unknown_domain.clone(),
            parameter_domain: value.parameter_domain.clone(),
            center: value.center.clone(),
            preconditioner: value.preconditioner.clone(),
            precision_digits: value.precision_digits,
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum StepWire {
    Split { axis: usize },
    Leaf { index: usize },
}

#[derive(Clone, Serialize)]
struct CoverWire {
    leaves: Vec<RootWire>,
    partition: Vec<StepWire>,
    precision_digits: u32,
}

impl From<&ParametricRootCoverWitness> for CoverWire {
    fn from(value: &ParametricRootCoverWitness) -> Self {
        Self {
            leaves: value.leaves.iter().map(RootWire::from).collect(),
            partition: value
                .partition
                .iter()
                .map(|step| match *step {
                    ParametricRootCoverStep::Split { axis } => StepWire::Split { axis },
                    ParametricRootCoverStep::Leaf { index } => StepWire::Leaf { index },
                })
                .collect(),
            precision_digits: value.precision_digits,
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
enum WitnessWire {
    RootBox(RootWire),
    Cover(CoverWire),
}

#[derive(Serialize)]
struct Proposal<'a, T: Serialize + ?Sized> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    unknowns: &'a [Symbol],
    parameters: &'a [Symbol],
    unknown_domain: &'a [RealBall],
    parameter_domain: &'a [RealBall],
    witness: &'a T,
}

fn proposal<T: Serialize + ?Sized>(
    source: &ParametricProblem,
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
    witness: &T,
) -> Vec<u8> {
    serde_json::to_vec(&Proposal {
        schema_version: PARAMETRIC_FAMILY_ARTIFACT_SCHEMA_VERSION,
        context_digest: CONTEXT,
        residuals: source.residuals(),
        unknowns: source.unknowns(),
        parameters: source.parameters(),
        unknown_domain,
        parameter_domain,
        witness,
    })
    .unwrap()
}

fn replay(bytes: &[u8]) -> Result<ParametricFamilyReceipt, ParametricError> {
    replay_parametric_family(
        bytes,
        &problem("x-p"),
        &[interval(-2, 2)],
        &[interval(-1, 1)],
        CONTEXT,
        &mut Unbounded,
    )
}

#[test]
fn uniform_family_round_trip_recomputes_bounds_and_allows_fresh_sensitivity() {
    let accepted = accepted_affine();
    let bytes = export_parametric_family(&accepted, CONTEXT, &mut Unbounded).unwrap();
    let restored = replay(&bytes).unwrap();
    assert_eq!(restored, accepted);
    assert_eq!(restored.root_enclosure(), [interval(-1, 1)]);
    assert_eq!(
        restored
            .enclose_sensitivity(&mut Unbounded)
            .unwrap()
            .solution_jacobian(),
        [vec![RealBall::from_i64(1)]]
    );
    assert_eq!(
        export_parametric_family(&restored, CONTEXT, &mut Unbounded).unwrap(),
        bytes
    );
    let wire: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire.as_object().unwrap().len(), 8);
    assert_eq!(wire["witness"]["kind"], "root_box");
    for absent in [
        "accepted",
        "verified",
        "root_enclosure",
        "contraction_bound",
        "solution_jacobian",
        "leaf_count",
    ] {
        assert!(wire.get(absent).is_none());
        assert!(wire["witness"]["data"].get(absent).is_none());
    }
    // An independently assembled proposal has the same encoding as export.
    assert_eq!(
        proposal(
            &problem("x-p"),
            &[interval(-2, 2)],
            &[interval(-1, 1)],
            &WitnessWire::RootBox(RootWire::from(&root_witness())),
        ),
        bytes
    );
}

#[test]
fn adaptive_cover_round_trip_checks_all_parameter_cells_and_rebuilds_sensitivity() {
    let (source, accepted) = adaptive();
    let bytes = export_parametric_family(&accepted, CONTEXT, &mut Unbounded).unwrap();
    let restored = replay_parametric_family(
        &bytes,
        &source,
        &[interval(0, 2)],
        &[interval(0, 2)],
        CONTEXT,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(restored, accepted);
    assert!(restored.leaf_count() > 1);
    assert!(restored.root_enclosure()[0].contains(&q(1, 5)));
    assert!(restored.root_enclosure()[0].contains(&q(1, 1)));
    let sensitivity = restored.enclose_sensitivity(&mut Unbounded).unwrap();
    // Independently known dx/dp = -2p/(1+p^2)^2 at three distinct parameters.
    for (p, derivative) in [
        (q(0, 1), q(0, 1)),
        (q(1, 1), q(-1, 2)),
        (q(2, 1), q(-4, 25)),
    ] {
        assert!(restored.parameter_domain()[0].contains(&p));
        assert!(sensitivity.solution_jacobian()[0][0].contains(&derivative));
    }
    assert_eq!(sensitivity.leaf_count(), restored.leaf_count());
    assert_eq!(
        export_parametric_family(&restored, CONTEXT, &mut Unbounded).unwrap(),
        bytes
    );
}

#[test]
fn a_large_complete_cover_uses_per_scalar_work_instead_of_squaring_all_artifact_bytes() {
    fn partition(depth: usize, next_leaf: &mut usize, output: &mut Vec<StepWire>) {
        if depth == 0 {
            output.push(StepWire::Leaf { index: *next_leaf });
            *next_leaf += 1;
        } else {
            output.push(StepWire::Split { axis: 0 });
            partition(depth - 1, next_leaf, output);
            partition(depth - 1, next_leaf, output);
        }
    }

    // This complete certificate is assembled directly, without consulting
    // the adaptive generator. Each of 128 parameter cells has exact x=p.
    let leaves: Vec<_> = (0..128)
        .map(|index| {
            let mut leaf = RootWire::from(&root_witness());
            leaf.parameter_domain =
                vec![RealBall::new(q(2 * index - 127, 128), q(1, 128)).unwrap()];
            leaf
        })
        .collect();
    let mut steps = Vec::new();
    partition(7, &mut 0, &mut steps);
    let source = problem("x-p");
    let bytes = proposal(
        &source,
        &[interval(-2, 2)],
        &[interval(-1, 1)],
        &WitnessWire::Cover(CoverWire {
            leaves,
            partition: steps,
            precision_digits: 16,
        }),
    );
    assert!(bytes.len() > 24 * 1024);
    let mut limits = BudgetLimits::uniform(10_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 512 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    let mut meter = Budget::new(limits);
    let restored = replay_parametric_family(
        &bytes,
        &source,
        &[interval(-2, 2)],
        &[interval(-1, 1)],
        CONTEXT,
        &mut meter,
    )
    .unwrap();
    assert_eq!(restored.leaf_count(), 128);
    assert_eq!(restored.root_enclosure(), [interval(-1, 1)]);
    assert!(meter.remaining(Dimension::ComputeSteps) > 0);
    assert_eq!(meter.verifier_remaining(), 100_000_000);
}

#[test]
fn typed_same_name_coordinates_and_fixed_parameter_intervals_round_trip() {
    let unknown = Symbol::with_identity("z", SymbolIdentity::from_assumptions_digest([1; 32]));
    let parameter = Symbol::with_identity("z", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = ParametricProblem::from_exprs(
        vec![Expr::Sym(unknown.clone()) - Expr::Sym(parameter.clone())],
        vec![unknown],
        vec![parameter],
        &mut Unbounded,
    )
    .unwrap();
    let mut witness = root_witness();
    witness.parameter_domain = vec![RealBall::exact(q(1, 2))];
    let accepted =
        ParametricFamilyReceipt::RootBox(source.verify_root_box(&witness, &mut Unbounded).unwrap());
    let bytes = export_parametric_family(&accepted, CONTEXT, &mut Unbounded).unwrap();
    let restored = replay_parametric_family(
        &bytes,
        &source,
        &witness.unknown_domain,
        &witness.parameter_domain,
        CONTEXT,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(restored, accepted);
    assert_eq!(restored.root_enclosure(), [RealBall::exact(q(1, 2))]);
    assert_ne!(restored.unknowns(), restored.parameters());
}

#[test]
fn receiver_source_roles_domains_and_context_are_bound_before_witness_decoding() {
    let source = problem("x-p");
    let unknown_domain = [interval(-2, 2)];
    let parameter_domain = [interval(-1, 1)];
    let bytes = proposal(&source, &unknown_domain, &parameter_domain, &json!(null));
    let swapped = ParametricProblem::from_exprs(
        source.residuals().to_vec(),
        source.parameters().to_vec(),
        source.unknowns().to_vec(),
        &mut Unbounded,
    )
    .unwrap();
    for expected in [problem("x-p+1"), swapped] {
        assert_eq!(
            replay_parametric_family(
                &bytes,
                &expected,
                &unknown_domain,
                &parameter_domain,
                CONTEXT,
                &mut Unbounded,
            )
            .unwrap_err(),
            ParametricError::Certification(CertificationError::SourceMismatch)
        );
    }
    for (unknowns, parameters) in [
        (vec![interval(-3, 3)], parameter_domain.to_vec()),
        (unknown_domain.to_vec(), vec![interval(0, 1)]),
    ] {
        assert_eq!(
            replay_parametric_family(
                &bytes,
                &source,
                &unknowns,
                &parameters,
                CONTEXT,
                &mut Unbounded,
            )
            .unwrap_err(),
            ParametricError::Certification(CertificationError::DomainMismatch)
        );
    }
    assert!(matches!(
        replay_parametric_family(
            &bytes,
            &source,
            &unknown_domain,
            &parameter_domain,
            [8; 32],
            &mut Unbounded,
        ),
        Err(ParametricError::Certification(
            CertificationError::UniverseMismatch { .. }
        ))
    ));

    let ordered = ParametricProblem::parse_sources(
        &["x+2*y-p", "y-q"],
        &["y", "x"],
        &["q", "p"],
        &mut Unbounded,
    )
    .unwrap();
    let bytes = proposal(
        &ordered,
        &[interval(-4, 4), interval(-4, 4)],
        &[interval(-1, 1), interval(-1, 1)],
        &json!(null),
    );
    for (unknowns, parameters) in [(["x", "y"], ["q", "p"]), (["y", "x"], ["p", "q"])] {
        let reordered = ParametricProblem::parse_sources(
            &["x+2*y-p", "y-q"],
            &unknowns,
            &parameters,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(
            replay_parametric_family(
                &bytes,
                &reordered,
                &[interval(-4, 4), interval(-4, 4)],
                &[interval(-1, 1), interval(-1, 1)],
                CONTEXT,
                &mut Unbounded,
            )
            .unwrap_err(),
            ParametricError::Certification(CertificationError::SourceMismatch)
        );
    }
}

#[test]
fn uniform_replay_rechecks_strict_interior_contraction_and_full_parameter_dependence() {
    let source = problem("x-p");
    let original = root_witness();
    let mut zero_preconditioner = RootWire::from(&original);
    zero_preconditioner.preconditioner[0][0] = q(0, 1);
    assert_eq!(
        replay(&proposal(
            &source,
            &original.unknown_domain,
            &original.parameter_domain,
            &WitnessWire::RootBox(zero_preconditioner.clone()),
        ))
        .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::NotContractive))
    );
    let mut bad_center = RootWire::from(&original);
    bad_center.center[0] = q(3, 1);
    let mut foreign_domain = RootWire::from(&original);
    foreign_domain.parameter_domain[0] = RealBall::from_i64(0);
    let mut narrow_unknown = RootWire::from(&original);
    narrow_unknown.unknown_domain[0] = interval(-1, 1);
    for candidate in [
        zero_preconditioner,
        bad_center,
        foreign_domain,
        narrow_unknown,
    ] {
        assert!(
            replay(&proposal(
                &source,
                &original.unknown_domain,
                &original.parameter_domain,
                &WitnessWire::RootBox(candidate),
            ))
            .is_err()
        );
    }
    // Checking only p=0 would falsely certify strict containment for x=p.
    let mut boundary = original;
    boundary.unknown_domain[0] = interval(-1, 1);
    let bytes = proposal(
        &source,
        &boundary.unknown_domain,
        &boundary.parameter_domain,
        &WitnessWire::RootBox(RootWire::from(&boundary)),
    );
    assert!(matches!(
        replay_parametric_family(
            &bytes,
            &source,
            &boundary.unknown_domain,
            &boundary.parameter_domain,
            CONTEXT,
            &mut Unbounded,
        ),
        Err(ParametricError::Certification(CertificationError::Root(_)))
    ));
}

#[test]
fn complete_cover_replay_rejects_missing_reused_foreign_or_invalid_leaves() {
    let (source, accepted) = adaptive();
    let ParametricFamilyReceipt::Cover(receipt) = &accepted else {
        unreachable!();
    };
    let original = CoverWire::from(receipt.witness());
    assert_eq!(
        replay_parametric_family(
            &proposal(
                &source,
                accepted.unknown_domain(),
                accepted.parameter_domain(),
                &WitnessWire::Cover(original.clone()),
            ),
            &source,
            accepted.unknown_domain(),
            accepted.parameter_domain(),
            CONTEXT,
            &mut Unbounded,
        )
        .unwrap(),
        accepted
    );
    let mut omitted = original.clone();
    omitted.partition.pop();
    let mut reused = original.clone();
    for step in &mut reused.partition {
        if let StepWire::Leaf { index } = step {
            *index = 0;
        }
    }
    let mut foreign = original.clone();
    foreign.leaves[0].parameter_domain[0] = interval(10, 11);
    let mut false_contraction = original.clone();
    false_contraction.leaves[0].preconditioner[0][0] = q(0, 1);
    let mut bad_axis = original.clone();
    bad_axis.partition[0] = StepWire::Split { axis: 1 };
    let mut unused = original.clone();
    unused.leaves.push(original.leaves[0].clone());
    let mut wrong_precision = original;
    wrong_precision.leaves[0].precision_digits += 1;
    for candidate in [
        omitted,
        reused,
        foreign,
        false_contraction,
        bad_axis,
        unused,
        wrong_precision,
    ] {
        let bytes = proposal(
            &source,
            accepted.unknown_domain(),
            accepted.parameter_domain(),
            &WitnessWire::Cover(candidate),
        );
        assert!(
            replay_parametric_family(
                &bytes,
                &source,
                accepted.unknown_domain(),
                accepted.parameter_domain(),
                CONTEXT,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn held_original_source_domains_survive_serialization_and_replay() {
    for source in ["x-p+0*(1/p)", "x-p+(1/p)^0-1", "x-p+log(p)-log(p)"] {
        let source = problem(source);
        let witness = root_witness();
        let bytes = proposal(
            &source,
            &witness.unknown_domain,
            &witness.parameter_domain,
            &WitnessWire::RootBox(RootWire::from(&witness)),
        );
        assert_eq!(
            replay(&bytes).unwrap_err(),
            ParametricError::Certification(CertificationError::SourceMismatch)
        );
        assert!(
            replay_parametric_family(
                &bytes,
                &source,
                &witness.unknown_domain,
                &witness.parameter_domain,
                CONTEXT,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn canonical_wire_rejects_metadata_forgery_alternate_scalars_and_unknown_schema() {
    let bytes = export_parametric_family(&accepted_affine(), CONTEXT, &mut Unbounded).unwrap();
    let mut whitespace = b" ".to_vec();
    whitespace.extend_from_slice(&bytes);
    assert_eq!(
        replay(&whitespace).unwrap_err(),
        ParametricError::Certification(CertificationError::NonCanonical)
    );
    for suffix in [b",\"verified\":true}".as_slice(), b",\"schema_version\":1}"] {
        let mut changed = bytes[..bytes.len() - 1].to_vec();
        changed.extend_from_slice(suffix);
        assert!(matches!(
            replay(&changed),
            Err(ParametricError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
    let unsupported = String::from_utf8(bytes.clone()).unwrap().replacen(
        "\"schema_version\":1",
        "\"schema_version\":999",
        1,
    );
    assert!(matches!(
        replay(unsupported.as_bytes()),
        Err(ParametricError::Certification(
            CertificationError::UnsupportedVersion { found: 999, .. }
        ))
    ));
    for scalar in [
        json!([[1, [2]], [1, [2]]]),
        json!([[0, []], [0, []]]),
        json!([[0, [0]], [1, [1]]]),
        json!([[0, []], [-1, [1]]]),
    ] {
        let mut wire =
            serde_json::to_value(WitnessWire::RootBox(RootWire::from(&root_witness()))).unwrap();
        wire["data"]["center"][0] = scalar;
        assert!(
            replay(&proposal(
                &problem("x-p"),
                &[interval(-2, 2)],
                &[interval(-1, 1)],
                &wire,
            ))
            .is_err()
        );
    }
}

#[test]
fn hostile_nesting_lengths_and_scalar_limbs_are_bounded_before_acceptance() {
    assert!(matches!(
        replay(&vec![b' '; MAX_PARAMETRIC_FAMILY_ARTIFACT_BYTES + 1]),
        Err(ParametricError::Certification(
            CertificationError::ByteLimit { .. }
        ))
    ));
    let nesting = "[".repeat(100) + &"]".repeat(100);
    assert!(matches!(
        replay(nesting.as_bytes()),
        Err(ParametricError::Certification(
            CertificationError::DepthLimit { .. }
        ))
    ));
    let valid =
        serde_json::to_value(WitnessWire::RootBox(RootWire::from(&root_witness()))).unwrap();
    let mut long_center = valid.clone();
    long_center["data"]["center"] = json!(vec![q(0, 1); 17]);
    let mut long_rows = valid.clone();
    long_rows["data"]["preconditioner"] = json!(vec![vec![q(0, 1)]; 17]);
    let mut long_columns = valid.clone();
    long_columns["data"]["preconditioner"] = json!([vec![q(0, 1); 17]]);
    let mut huge_scalar = valid;
    huge_scalar["data"]["center"][0] =
        json!([[1, vec![1; 2 * MAX_ROOT_BOX_LIMBS as usize + 1]], [1, [1]]]);
    let too_many_leaves = json!({
        "kind": "cover",
        "data": {"leaves": vec![json!(null); 1025], "partition": [], "precision_digits": 16}
    });
    for witness in [
        long_center,
        long_rows,
        long_columns,
        huge_scalar,
        too_many_leaves,
    ] {
        assert!(
            replay(&proposal(
                &problem("x-p"),
                &[interval(-2, 2)],
                &[interval(-1, 1)],
                &witness,
            ))
            .is_err()
        );
    }
}

#[derive(Default)]
struct CancelAt {
    checkpoints: usize,
    at: Option<usize>,
    stopped: bool,
    charged: u64,
}

impl BudgetMeter for CancelAt {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "work continued after cancellation");
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
        assert!(!self.stopped, "work continued after cancellation");
        self.checkpoints += 1;
        if self.at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn transport_cancellation_stops_immediately_and_plain_budgets_cannot_spend_reserves() {
    let accepted = accepted_affine();
    let source = problem("x-p");
    let bytes = export_parametric_family(&accepted, CONTEXT, &mut Unbounded).unwrap();
    for exporting in [true, false] {
        let run = |meter: &mut CancelAt| -> Result<(), ParametricError> {
            if exporting {
                export_parametric_family(&accepted, CONTEXT, meter).map(|_| ())
            } else {
                replay_parametric_family(
                    &bytes,
                    &source,
                    accepted.unknown_domain(),
                    accepted.parameter_domain(),
                    CONTEXT,
                    meter,
                )
                .map(|_| ())
            }
        };
        let mut probe = CancelAt::default();
        run(&mut probe).unwrap();
        assert!(probe.checkpoints > 1);
        assert!(probe.charged > 0);
        let mut points = vec![1, probe.checkpoints / 2, probe.checkpoints];
        points.sort_unstable();
        points.dedup();
        for at in points {
            let mut meter = CancelAt {
                at: Some(at),
                ..CancelAt::default()
            };
            assert_eq!(
                run(&mut meter).unwrap_err(),
                ParametricError::Certification(CertificationError::Meter(MeterError::Cancelled))
            );
            assert_eq!(meter.checkpoints, at);
        }
    }
    for exporting in [true, false] {
        let mut budget = Budget::new(BudgetLimits::uniform(1, 100_000_000));
        let error = if exporting {
            export_parametric_family(&accepted, CONTEXT, &mut budget).map(|_| ())
        } else {
            replay_parametric_family(
                &bytes,
                &source,
                accepted.unknown_domain(),
                accepted.parameter_domain(),
                CONTEXT,
                &mut budget,
            )
            .map(|_| ())
        }
        .unwrap_err();
        assert!(matches!(
            error,
            ParametricError::Certification(CertificationError::Meter(MeterError::Budget(_)))
        ));
        assert_eq!(budget.verifier_remaining(), 100_000_000);
    }
}
