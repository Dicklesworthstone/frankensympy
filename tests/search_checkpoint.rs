//! Portable root-search candidates remain source-bound and independently checked.

use frankensympy::certification::{CertificationError, ResidualProblem};
use frankensympy::search_checkpoint::{
    MAX_ROOT_SEARCH_CHECKPOINT_BYTES, ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION,
    export_root_search_checkpoint, restore_root_search_checkpoint,
};
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::root_isolation::{
    MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationCounters, RootIsolationOptions,
    RootIsolationSearch, RootIsolationSearchCheckpoint,
};
use fsym_core::{
    BigRational, Expr, MAX_ROOT_ISOLATION_NODES, RealBall, RootBoxWitness, RootIsolationError,
    RootIsolationStep, Symbol, SymbolIdentity,
};
use serde::Serialize;
use serde_json::json;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn problem(source: &str) -> ResidualProblem {
    ResidualProblem::parse_sources(&[source.into()], &["x".into()], &mut Unbounded).unwrap()
}

fn domain() -> Vec<RealBall> {
    vec![RealBall::new(q(0, 1), q(2, 1)).unwrap()]
}

fn options() -> RootIsolationOptions {
    RootIsolationOptions {
        precision_digits: 16,
        max_leaves: 128,
    }
}

fn search(problem: &ResidualProblem) -> RootIsolationSearch {
    problem
        .begin_isolation_search(&domain(), options(), &mut Unbounded)
        .unwrap()
}

fn restored(bytes: &[u8]) -> Result<RootIsolationSearch, CertificationError> {
    restore_root_search_checkpoint(
        bytes,
        &problem("x^2-2"),
        &domain(),
        options(),
        [7; 32],
        &mut Unbounded,
    )
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    Split { axis: usize },
    Exclude,
    Root { index: usize },
}

impl From<RootIsolationStep> for Step {
    fn from(step: RootIsolationStep) -> Self {
        match step {
            RootIsolationStep::Split { axis } => Self::Split { axis },
            RootIsolationStep::Exclude => Self::Exclude,
            RootIsolationStep::Root { index } => Self::Root { index },
        }
    }
}

#[derive(Serialize)]
struct Witness<'a> {
    roots: &'a [RootBoxWitness],
    partition: Vec<Step>,
    precision_digits: u32,
}

#[derive(Serialize)]
struct ProposedCheckpoint<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    options: RootIsolationOptions,
    counters: RootIsolationCounters,
    witness: Witness<'a>,
}

fn proposed_bytes(checkpoint: &RootIsolationSearchCheckpoint) -> Vec<u8> {
    let problem = problem("x^2-2");
    serde_json::to_vec(&ProposedCheckpoint {
        schema_version: ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION,
        context_digest: [7; 32],
        residuals: problem.residuals(),
        variables: problem.variables(),
        domain: &domain(),
        options: checkpoint.options,
        counters: checkpoint.counters,
        witness: Witness {
            roots: &checkpoint.witness.roots,
            partition: checkpoint
                .witness
                .partition
                .iter()
                .copied()
                .map(Step::from)
                .collect(),
            precision_digits: checkpoint.witness.precision_digits,
        },
    })
    .unwrap()
}

#[test]
fn unfinished_checkpoint_resumes_to_the_same_independently_verified_complete_cover() {
    let problem = problem("x^2-2");
    let mut original = search(&problem);
    original.advance(1, &mut Unbounded).unwrap();
    let progress = original.progress();
    assert_eq!(progress.processed_cells, 1);
    assert_eq!(progress.pending_cells, 2);
    assert!(!progress.ready_for_verification);
    let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    let mut resumed = restored(&bytes).unwrap();
    assert_eq!(resumed.progress(), progress);
    assert!(resumed.completed_witness().is_none());
    assert!(resumed.counters().steps > original.counters().steps);
    assert!(resumed.counters().allocation_bytes > original.counters().allocation_bytes);
    assert_eq!(
        resumed.counters().source_bytes,
        original.counters().source_bytes
    );
    original
        .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
        .unwrap();
    resumed
        .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
        .unwrap();
    assert_eq!(resumed.completed_witness(), original.completed_witness());
    let accepted = problem
        .verify_isolation(
            &domain(),
            resumed.completed_witness().unwrap(),
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(accepted.root_count(), 2);
    let mut negative = false;
    let mut positive = false;
    for root in accepted.roots() {
        let enclosure = &root.root_enclosure()[0];
        if enclosure.upper() < q(0, 1) {
            negative = true;
            assert!(enclosure.lower().pow(2).unwrap() > q(2, 1));
            assert!(enclosure.upper().pow(2).unwrap() < q(2, 1));
        } else {
            positive = true;
            assert!(enclosure.lower() > q(0, 1));
            assert!(enclosure.lower().pow(2).unwrap() < q(2, 1));
            assert!(enclosure.upper().pow(2).unwrap() > q(2, 1));
        }
    }
    assert!(negative && positive);
}

#[test]
fn empty_and_completed_candidates_restore_without_becoming_receipts() {
    let problem = problem("x^2-2");
    let mut original = search(&problem);
    for completed in [false, true] {
        if completed {
            original
                .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
                .unwrap();
        }
        let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
        let resumed = restored(&bytes).unwrap();
        assert_eq!(resumed.progress(), original.progress());
        assert_eq!(resumed.progress().ready_for_verification, completed);
        let object: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let fields = object.as_object().unwrap();
        assert_eq!(fields.len(), 8);
        for field in ["accepted", "verified", "root_count", "receipt", "pending"] {
            assert!(!fields.contains_key(field));
        }
    }
}

#[test]
fn repeated_export_retains_snapshot_work_instead_of_resetting_counters() {
    let mut original = search(&problem("x^2-2"));
    original.advance(1, &mut Unbounded).unwrap();
    let first = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    let after_first = original.counters();
    let second = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    let after_second = original.counters();
    assert!(after_second.steps > after_first.steps);
    assert!(after_second.allocation_bytes > after_first.allocation_bytes);
    let first: serde_json::Value = serde_json::from_slice(&first).unwrap();
    let second: serde_json::Value = serde_json::from_slice(&second).unwrap();
    assert_eq!(first["witness"], second["witness"]);
    assert_ne!(first["counters"], second["counters"]);
}

#[test]
fn changed_sources_boxes_options_and_context_refuse_before_counter_or_witness_decode() {
    let mut original = search(&problem("x^2-2"));
    let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    let mut invalid: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    invalid["counters"] = json!("do not decode foreign counters");
    invalid["witness"] = json!("do not decode foreign witness");
    let bytes = serde_json::to_vec(&invalid).unwrap();
    assert!(matches!(
        restore_root_search_checkpoint(
            &bytes,
            &problem("x^2-3"),
            &domain(),
            options(),
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::SourceMismatch)
    ));
    assert!(matches!(
        restore_root_search_checkpoint(
            &bytes,
            &problem("x^2-2"),
            &[RealBall::new(q(0, 1), q(3, 1)).unwrap()],
            options(),
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::DomainMismatch)
    ));
    for different in [
        RootIsolationOptions {
            max_leaves: 64,
            ..options()
        },
        RootIsolationOptions {
            precision_digits: 24,
            ..options()
        },
    ] {
        assert!(matches!(
            restore_root_search_checkpoint(
                &bytes,
                &problem("x^2-2"),
                &domain(),
                different,
                [7; 32],
                &mut Unbounded,
            ),
            Err(CertificationError::Isolation(
                RootIsolationError::InvalidPartition(_)
            ))
        ));
    }
    assert!(matches!(
        restore_root_search_checkpoint(
            &bytes,
            &problem("x^2-2"),
            &domain(),
            options(),
            [8; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::UniverseMismatch { .. })
    ));
}

#[test]
fn same_printed_symbol_with_different_identity_cannot_restore_the_checkpoint() {
    let identity = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let typed = ResidualProblem::new(
        vec![Expr::Sym(identity.clone())],
        vec![identity],
        &mut Unbounded,
    )
    .unwrap();
    let mut original = search(&typed);
    let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    assert!(matches!(
        restore_root_search_checkpoint(
            &bytes,
            &problem("x"),
            &domain(),
            options(),
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::SourceMismatch)
    ));
}

#[test]
fn canonical_forged_exclusions_root_witnesses_and_extra_cover_tokens_are_rechecked() {
    let mut original = search(&problem("x^2-2"));
    original
        .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
        .unwrap();
    let checkpoint = original.checkpoint(&mut Unbounded).unwrap();
    assert!(restored(&proposed_bytes(&checkpoint)).is_ok());
    let mut invalid = checkpoint.clone();
    invalid.witness.roots.clear();
    invalid.witness.partition = vec![RootIsolationStep::Exclude];
    assert!(matches!(
        restored(&proposed_bytes(&invalid)),
        Err(CertificationError::Isolation(
            RootIsolationError::Incomplete(_)
        ))
    ));
    let mut invalid = checkpoint.clone();
    invalid.witness.roots[0].preconditioner[0][0] = q(0, 1);
    assert!(restored(&proposed_bytes(&invalid)).is_err());
    let mut invalid = checkpoint;
    invalid.witness.partition.push(RootIsolationStep::Exclude);
    assert!(restored(&proposed_bytes(&invalid)).is_err());
}

#[test]
fn understated_overflowed_or_substituted_counters_never_replenish_lifetime_limits() {
    let mut original = search(&problem("x^2-2"));
    original.advance(1, &mut Unbounded).unwrap();
    let checkpoint = original.checkpoint(&mut Unbounded).unwrap();
    for field in 0..7 {
        let mut invalid = checkpoint.clone();
        match field {
            0 => invalid.counters.steps = 0,
            1 => invalid.counters.allocation_bytes = 0,
            2 => invalid.counters.source_bytes += 1,
            3 => invalid.counters.evaluations = 0,
            4 => invalid.counters.steps = u64::MAX,
            5 => invalid.counters.allocation_bytes = u64::MAX,
            _ => invalid.counters.evaluations = usize::MAX,
        }
        assert!(
            restored(&proposed_bytes(&invalid)).is_err(),
            "counter mutation {field}"
        );
    }
}

#[test]
fn malformed_fields_noncanonical_bytes_and_unknown_schema_fail_closed() {
    let mut original = search(&problem("x^2-2"));
    let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    for location in ["envelope", "counters", "witness"] {
        // Preserve the exact source/options encodings so each planted field
        // reaches its own decoder instead of failing earlier key-order checks.
        let invalid = match location {
            "envelope" => format!("{{\"accepted\":true,{}", &text[1..]),
            "counters" => text.replacen(
                "\"counters\":{",
                "\"counters\":{\"allowance\":18446744073709551615,",
                1,
            ),
            _ => text.replacen("\"witness\":{", "\"witness\":{\"pending\":[],", 1),
        };
        assert!(matches!(
            restored(invalid.as_bytes()),
            Err(CertificationError::Encoding(_))
        ));
    }
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["schema_version"] = json!(99);
    assert!(matches!(
        restored(&serde_json::to_vec(&value).unwrap()),
        Err(CertificationError::UnsupportedVersion { .. })
    ));
    let duplicate = format!(
        "{{\"schema_version\":1,{}",
        std::str::from_utf8(&bytes[1..]).unwrap()
    );
    assert!(matches!(
        restored(duplicate.as_bytes()),
        Err(CertificationError::Encoding(_))
    ));
    let mut padded = bytes;
    padded.push(b'\n');
    assert!(matches!(
        restored(&padded),
        Err(CertificationError::NonCanonical)
    ));
}

#[test]
fn transport_length_depth_and_partition_bounds_precede_restoration() {
    assert!(matches!(
        restored(&vec![b' '; MAX_ROOT_SEARCH_CHECKPOINT_BYTES + 1]),
        Err(CertificationError::ByteLimit { .. })
    ));
    let nested = format!("{}0{}", "[".repeat(100), "]".repeat(100));
    assert!(matches!(
        restored(nested.as_bytes()),
        Err(CertificationError::DepthLimit { .. })
    ));
    let mut original = search(&problem("x^2-2"));
    let mut checkpoint = original.checkpoint(&mut Unbounded).unwrap();
    checkpoint.witness.partition = vec![RootIsolationStep::Exclude; MAX_ROOT_ISOLATION_NODES + 1];
    assert!(matches!(
        restored(&proposed_bytes(&checkpoint)),
        Err(CertificationError::Encoding(_))
    ));
}

#[test]
fn restoration_charges_receiver_resources_and_cannot_use_its_protected_pool() {
    let mut original = search(&problem("x^2-2"));
    original.advance(1, &mut Unbounded).unwrap();
    let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    let mut receiver = Budget::new(BudgetLimits::uniform(1 << 35, 193));
    let resumed = restore_root_search_checkpoint(
        &bytes,
        &problem("x^2-2"),
        &domain(),
        options(),
        [7; 32],
        &mut receiver,
    )
    .unwrap();
    assert_eq!(resumed.progress(), original.progress());
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(receiver.remaining(dimension) < 1 << 35);
    }
    assert_eq!(receiver.verifier_remaining(), 193);
    let mut limits = BudgetLimits::uniform(1 << 35, 1 << 35);
    limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut exhausted = Budget::new(limits);
    assert!(matches!(
        restore_root_search_checkpoint(
            &bytes,
            &problem("x^2-2"),
            &domain(),
            options(),
            [7; 32],
            &mut exhausted,
        ),
        Err(CertificationError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(exhausted.verifier_remaining(), 1 << 35);
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

#[test]
fn cancellation_preserves_existing_frontier_and_never_returns_a_restored_search() {
    let mut original = search(&problem("x^2-2"));
    original.advance(1, &mut Unbounded).unwrap();
    let before = original.progress();
    for checkpoint in [0, 1, 20] {
        assert!(matches!(
            export_root_search_checkpoint(&mut original, [7; 32], &mut CancelAfter(checkpoint)),
            Err(CertificationError::Meter(MeterError::Cancelled))
        ));
        assert_eq!(original.progress(), before);
    }
    let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    for checkpoint in [0, 1, 20] {
        assert!(matches!(
            restore_root_search_checkpoint(
                &bytes,
                &problem("x^2-2"),
                &domain(),
                options(),
                [7; 32],
                &mut CancelAfter(checkpoint),
            ),
            Err(CertificationError::Meter(MeterError::Cancelled))
        ));
    }
}
