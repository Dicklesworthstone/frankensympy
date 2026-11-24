//! Host-accessible transports retain the native source-bound continuation rules.

use fsym_budget::{Budget, BudgetLimits, Dimension, MeterError, Unbounded};
use fsym_calculus::root_isolation::checkpoint::{
    MAX_ROOT_SEARCH_CHECKPOINT_BYTES, export_root_search_checkpoint, restore_root_search_checkpoint,
};
use fsym_calculus::root_isolation::{
    MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationOptions, RootIsolationSearch,
};
use fsym_core::root_certificate::artifact::{CertificationError, RootCertificateSources};
use fsym_core::root_certificate::isolation_artifact::{
    export_root_isolation, replay_root_isolation,
};
use fsym_core::{BigRational, Expr, RealBall, Symbol, parse_raw, verify_root_isolation};

fn sources(source: &str) -> RootCertificateSources {
    RootCertificateSources::new(
        vec![parse_raw(source).unwrap()],
        vec![Symbol::new("x")],
        &mut Unbounded,
    )
    .unwrap()
}

fn domain() -> Vec<RealBall> {
    vec![RealBall::new(BigRational::from(0), BigRational::from(2)).unwrap()]
}

fn options() -> RootIsolationOptions {
    RootIsolationOptions {
        precision_digits: 16,
        max_leaves: 128,
    }
}

fn search(source: &RootCertificateSources) -> RootIsolationSearch {
    RootIsolationSearch::new(
        source.residuals(),
        source.variables(),
        &domain(),
        options(),
        &mut Unbounded,
    )
    .unwrap()
}

fn restore(
    bytes: &[u8],
    source: &RootCertificateSources,
) -> Result<RootIsolationSearch, CertificationError> {
    restore_root_search_checkpoint(bytes, source, &domain(), options(), [7; 32], &mut Unbounded)
}

#[test]
fn partial_transport_resumes_then_independently_checks_both_roots() {
    let source = sources("x^2-2");
    let mut original = search(&source);
    original.advance(1, &mut Unbounded).unwrap();
    let progress = original.progress();
    assert_eq!(progress.processed_cells, 1);
    assert!(!progress.ready_for_verification);
    let bytes = export_root_search_checkpoint(&mut original, [7; 32], &mut Unbounded).unwrap();
    let mut resumed = restore(&bytes, &source).unwrap();
    assert_eq!(resumed.progress(), progress);
    assert!(resumed.completed_witness().is_none());
    assert!(resumed.counters().steps > original.counters().steps);
    assert!(resumed.counters().allocation_bytes > original.counters().allocation_bytes);
    for candidate in [&mut original, &mut resumed] {
        candidate
            .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
            .unwrap();
    }
    assert_eq!(resumed.completed_witness(), original.completed_witness());
    let receipt = verify_root_isolation(
        source.residuals(),
        source.variables(),
        &domain(),
        resumed.completed_witness().unwrap(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.root_count(), 2);
    let proof = export_root_isolation(&receipt, [7; 32], &mut Unbounded).unwrap();
    let replayed =
        replay_root_isolation(&proof, &source, &domain(), [7; 32], &mut Unbounded).unwrap();
    assert_eq!(replayed, receipt);
}

#[test]
fn complete_candidate_and_root_free_receipt_have_distinct_wire_statements() {
    let source = sources("x^2+1");
    let mut candidate = search(&source);
    candidate
        .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
        .unwrap();
    let checkpoint =
        export_root_search_checkpoint(&mut candidate, [7; 32], &mut Unbounded).unwrap();
    let restored = restore(&checkpoint, &source).unwrap();
    assert!(restored.progress().ready_for_verification);
    let receipt = verify_root_isolation(
        source.residuals(),
        source.variables(),
        &domain(),
        restored.completed_witness().unwrap(),
        &mut Unbounded,
    )
    .unwrap();
    assert!(receipt.is_root_free());
    let proof = export_root_isolation(&receipt, [7; 32], &mut Unbounded).unwrap();
    assert!(restore(&proof, &source).is_err());
    assert!(
        replay_root_isolation(&checkpoint, &source, &domain(), [7; 32], &mut Unbounded).is_err()
    );
    assert!(
        replay_root_isolation(&proof, &source, &domain(), [7; 32], &mut Unbounded)
            .unwrap()
            .is_root_free()
    );
}

#[test]
fn frozen_source_axes_domain_options_and_context_precede_untrusted_witness_decode() {
    let source = sources("x^2-2");
    let mut candidate = search(&source);
    let bytes = export_root_search_checkpoint(&mut candidate, [7; 32], &mut Unbounded).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    // Keep the canonical statement bytes, but make the untrusted payload invalid.
    let witness_start = text.find("\"witness\":").unwrap();
    let malformed = format!("{}\"witness\":false}}", &text[..witness_start]);
    assert!(matches!(
        restore(malformed.as_bytes(), &sources("x^2-3")),
        Err(CertificationError::SourceMismatch)
    ));
    let different_axis = Symbol::with_identity(
        "x",
        fsym_core::SymbolIdentity::from_assumptions_digest([3; 32]),
    );
    let typed = RootCertificateSources::new(
        vec![Expr::Sym(different_axis.clone())],
        vec![different_axis],
        &mut Unbounded,
    )
    .unwrap();
    assert!(matches!(
        restore(malformed.as_bytes(), &typed),
        Err(CertificationError::SourceMismatch)
    ));
    assert!(matches!(
        restore_root_search_checkpoint(
            malformed.as_bytes(),
            &source,
            &[RealBall::new(BigRational::from(0), BigRational::from(3)).unwrap()],
            options(),
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::DomainMismatch)
    ));
    assert!(matches!(
        restore_root_search_checkpoint(
            malformed.as_bytes(),
            &source,
            &domain(),
            RootIsolationOptions {
                max_leaves: 64,
                ..options()
            },
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::Isolation(_))
    ));
    assert!(matches!(
        restore_root_search_checkpoint(
            malformed.as_bytes(),
            &source,
            &domain(),
            options(),
            [8; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::UniverseMismatch { .. })
    ));
}

#[test]
fn forged_exclusion_is_not_promoted_by_valid_encoding_and_high_counters() {
    let source = sources("x");
    let mut candidate = search(&source);
    candidate.advance(1, &mut Unbounded).unwrap();
    let bytes = export_root_search_checkpoint(&mut candidate, [7; 32], &mut Unbounded).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("\"partition\":[{\"root\":{\"index\":0}}]"));
    let forged = text.replace(
        "\"partition\":[{\"root\":{\"index\":0}}]",
        "\"partition\":[\"exclude\"]",
    );
    assert!(restore(forged.as_bytes(), &source).is_err());
    assert!(restore(&bytes, &source).is_ok());
}

#[test]
fn repeated_checkpointing_and_receiver_replay_keep_cumulative_work() {
    let source = sources("x");
    let mut candidate = search(&source);
    let mut budget = Budget::new(BudgetLimits::uniform(1 << 35, 193));
    let first = export_root_search_checkpoint(&mut candidate, [7; 32], &mut budget).unwrap();
    let first_intrinsic = candidate.counters();
    let after_first = budget.remaining(Dimension::ComputeSteps);
    let second = export_root_search_checkpoint(&mut candidate, [7; 32], &mut budget).unwrap();
    assert_ne!(first, second);
    assert!(candidate.counters().steps > first_intrinsic.steps);
    assert!(budget.remaining(Dimension::ComputeSteps) < after_first);
    let before_restore = budget.remaining(Dimension::ComputeSteps);
    let restored = restore_root_search_checkpoint(
        &second,
        &source,
        &domain(),
        options(),
        [7; 32],
        &mut budget,
    )
    .unwrap();
    assert_eq!(restored.progress(), candidate.progress());
    assert!(restored.counters().steps > candidate.counters().steps);
    assert!(budget.remaining(Dimension::ComputeSteps) < before_restore);
    assert_eq!(budget.verifier_remaining(), 193);
    let mut limits = BudgetLimits::uniform(1 << 35, 193);
    limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut exhausted = Budget::new(limits);
    assert!(matches!(
        restore_root_search_checkpoint(
            &second,
            &source,
            &domain(),
            options(),
            [7; 32],
            &mut exhausted,
        ),
        Err(CertificationError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(exhausted.verifier_remaining(), 193);
}

#[test]
fn noncanonical_unknown_and_oversized_checkpoints_are_not_resumable() {
    let source = sources("x");
    let mut candidate = search(&source);
    let bytes = export_root_search_checkpoint(&mut candidate, [7; 32], &mut Unbounded).unwrap();
    let mut padded = bytes.clone();
    padded.push(b'\n');
    assert!(matches!(
        restore(&padded, &source),
        Err(CertificationError::NonCanonical)
    ));
    let planted = format!(
        "{{\"accepted\":true,{}",
        std::str::from_utf8(&bytes[1..]).unwrap()
    );
    assert!(matches!(
        restore(planted.as_bytes(), &source),
        Err(CertificationError::Encoding(_))
    ));
    assert!(matches!(
        restore(&vec![b' '; MAX_ROOT_SEARCH_CHECKPOINT_BYTES + 1], &source),
        Err(CertificationError::ByteLimit { .. })
    ));
}
