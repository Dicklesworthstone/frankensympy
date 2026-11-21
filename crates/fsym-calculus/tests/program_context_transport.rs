//! The shared context envelope transports candidates, never acceptance.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::compile::{CompiledExpr, CompiledOp};
use fsym_calculus::program_artifact::context::{
    COMPILED_PROGRAM_ARTIFACT_SCHEMA_VERSION, MAX_COMPILED_PROGRAM_ARTIFACT_BYTES,
    ProgramTransportError, decode_objective_program, decode_residual_program,
    export_objective_program, export_residual_program,
};
use fsym_calculus::program_artifact::{
    ProgramArtifactError, admit_objective_candidate, admit_residual_candidate,
    encode_objective_candidate, encode_residual_candidate,
};
use fsym_calculus::{VerifiedObjective, VerifiedResidualSystem};
use fsym_core::root_certificate::artifact::{CertificationError, MAX_ROOT_CERTIFICATE_JSON_DEPTH};
use fsym_core::{Expr, Symbol, SymbolIdentity};
use serde::Serialize;
use serde_json::value::RawValue;

const CONTEXT: [u8; 32] = [11; 32];
const AMPLE: u64 = 1 << 40;
const DIMENSIONS: [Dimension; 3] = [
    Dimension::ComputeSteps,
    Dimension::MemoryBytes,
    Dimension::AllocationCount,
];

fn residual_fixture() -> VerifiedResidualSystem {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let y = Symbol::new("y");
    let residuals = [
        Expr::Add(vec![
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
            Expr::Sym(y.clone()),
        ]),
        Expr::Add(vec![
            Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
            Expr::from_i64(-1),
        ]),
    ];
    VerifiedResidualSystem::try_compile(&residuals, &[y, x]).unwrap()
}

fn objective_fixture() -> VerifiedObjective {
    let source = fsym_core::parse("x^2*y+3*y^2").unwrap();
    VerifiedObjective::try_compile(&source, &[Symbol::new("y"), Symbol::new("x")]).unwrap()
}

#[derive(Serialize)]
struct OuterEnvelope<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    candidate: &'a RawValue,
}

fn envelope(candidate: &[u8]) -> Vec<u8> {
    let raw = serde_json::from_slice(candidate).unwrap();
    serde_json::to_vec(&OuterEnvelope {
        schema_version: 1,
        context_digest: CONTEXT,
        candidate: raw,
    })
    .unwrap()
}

#[derive(Serialize)]
struct CandidateEnvelope<'a, T> {
    schema_version: u32,
    kind: &'static str,
    candidate: &'a T,
}

fn forged_envelope<T: Serialize>(kind: &'static str, candidate: &T) -> Vec<u8> {
    envelope(
        &serde_json::to_vec(&CandidateEnvelope {
            schema_version: 1,
            kind,
            candidate,
        })
        .unwrap(),
    )
}

#[test]
fn shared_envelope_keeps_the_existing_exact_wire_bytes_for_both_program_kinds() {
    let residual = residual_fixture();
    let objective = objective_fixture();
    assert_eq!(COMPILED_PROGRAM_ARTIFACT_SCHEMA_VERSION, 1);
    for (actual, inner) in [
        (
            export_residual_program(&residual, CONTEXT, &mut Unbounded).unwrap(),
            encode_residual_candidate(&residual, &mut Unbounded).unwrap(),
        ),
        (
            export_objective_program(&objective, CONTEXT, &mut Unbounded).unwrap(),
            encode_objective_candidate(&objective, &mut Unbounded).unwrap(),
        ),
    ] {
        assert_eq!(actual, envelope(&inner));
        let text = std::str::from_utf8(&actual).unwrap();
        assert!(text.starts_with("{\"schema_version\":1,\"context_digest\":[11,"));
        for forbidden in ["\"receipt\"", "\"verified\"", "\"shared_evaluation\""] {
            assert!(!text.contains(forbidden));
        }
    }
}

#[test]
fn residual_replay_requires_independent_source_admission_and_keeps_typed_axis_order() {
    let original = residual_fixture();
    let bytes = export_residual_program(&original, CONTEXT, &mut Unbounded).unwrap();
    let candidate = decode_residual_program(&bytes, CONTEXT, &mut Unbounded).unwrap();
    assert_eq!(&candidate, original.sparse_jacobian());
    let restored = admit_residual_candidate(
        original.residuals(),
        original.vars(),
        candidate,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(restored.vars(), original.vars());
    assert_eq!(restored.vars()[1].identity, original.vars()[1].identity);
    let values = restored
        .evaluate_sparse_with_meter(&[3.0, 2.0], &mut Unbounded)
        .unwrap();
    assert_eq!(values.residuals, [7.0, 5.0]);
    assert_eq!(values.jacobian, [1.0, 4.0, 2.0, 3.0]);
    assert_eq!(
        restored
            .jacobian_vector_product(&[3.0, 2.0], &[2.0, -1.0])
            .unwrap(),
        [-2.0, 1.0]
    );
    assert_eq!(
        export_residual_program(&restored, CONTEXT, &mut Unbounded).unwrap(),
        bytes
    );
}

#[test]
fn objective_replay_checks_both_derivative_stages_and_preserves_products() {
    let original = objective_fixture();
    let bytes = export_objective_program(&original, CONTEXT, &mut Unbounded).unwrap();
    let candidate = decode_objective_program(&bytes, CONTEXT, &mut Unbounded).unwrap();
    assert_eq!(&candidate, original.candidate());
    let restored = admit_objective_candidate(
        original.objective(),
        original.variables(),
        candidate,
        &mut Unbounded,
    )
    .unwrap();
    let values = restored.evaluate(&[3.0, 2.0]).unwrap();
    assert_eq!(values.value, 39.0);
    assert_eq!(values.gradient, [22.0, 12.0]);
    assert_eq!(values.hessian, [6.0, 4.0, 4.0, 6.0]);
    assert_eq!(
        restored
            .hessian_vector_product(&[3.0, 2.0], &[2.0, -1.0])
            .unwrap(),
        [8.0, 2.0]
    );
    assert_eq!(
        export_objective_program(&restored, CONTEXT, &mut Unbounded).unwrap(),
        bytes
    );
}

#[test]
fn context_and_program_kind_mismatches_keep_distinct_error_variants() {
    let residual = export_residual_program(&residual_fixture(), CONTEXT, &mut Unbounded).unwrap();
    let objective =
        export_objective_program(&objective_fixture(), CONTEXT, &mut Unbounded).unwrap();
    for error in [
        decode_residual_program(&residual, [12; 32], &mut Unbounded).unwrap_err(),
        decode_objective_program(&objective, [12; 32], &mut Unbounded).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            ProgramTransportError::Envelope(CertificationError::UniverseMismatch { found, expected })
                if found == CONTEXT && expected == [12; 32]
        ));
    }
    assert!(matches!(
        decode_residual_program(&objective, CONTEXT, &mut Unbounded),
        Err(ProgramTransportError::Candidate(
            ProgramArtifactError::WrongKind {
                expected: "residual"
            }
        ))
    ));
    assert!(matches!(
        decode_objective_program(&residual, CONTEXT, &mut Unbounded),
        Err(ProgramTransportError::Candidate(
            ProgramArtifactError::WrongKind {
                expected: "objective"
            }
        ))
    ));
}

#[test]
fn canonical_transport_does_not_grant_authority_to_forged_derivative_bytecode() {
    let residual = residual_fixture();
    let mut candidate = residual.sparse_jacobian().clone();
    candidate.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(99.0)]).unwrap();
    let bytes = forged_envelope("residual", &candidate);
    let decoded = decode_residual_program(&bytes, CONTEXT, &mut Unbounded).unwrap();
    assert_eq!(decoded, candidate);
    assert!(
        admit_residual_candidate(
            residual.residuals(),
            residual.vars(),
            decoded,
            &mut Unbounded
        )
        .is_err()
    );

    let objective = objective_fixture();
    let mut candidate = objective.candidate().clone();
    candidate.hessian.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(99.0)]).unwrap();
    let bytes = forged_envelope("objective", &candidate);
    let decoded = decode_objective_program(&bytes, CONTEXT, &mut Unbounded).unwrap();
    assert_eq!(decoded, candidate);
    assert!(
        admit_objective_candidate(
            objective.objective(),
            objective.variables(),
            decoded,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn unknown_schemas_duplicate_fields_and_noncanonical_envelopes_fail_closed() {
    let original = String::from_utf8(
        export_residual_program(&residual_fixture(), CONTEXT, &mut Unbounded).unwrap(),
    )
    .unwrap();
    let future = original.replacen("\"schema_version\":1", "\"schema_version\":2", 1);
    assert!(matches!(
        decode_residual_program(future.as_bytes(), CONTEXT, &mut Unbounded),
        Err(ProgramTransportError::Envelope(
            CertificationError::UnsupportedVersion {
                found: 2,
                supported: 1,
            }
        ))
    ));
    assert!(matches!(
        decode_residual_program(format!(" {original}").as_bytes(), CONTEXT, &mut Unbounded),
        Err(ProgramTransportError::Envelope(
            CertificationError::NonCanonical
        ))
    ));
    for mutant in [
        original.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        ),
        original.replacen(
            "\"context_digest\":",
            "\"accepted\":true,\"context_digest\":",
            1,
        ),
    ] {
        assert!(matches!(
            decode_residual_program(mutant.as_bytes(), CONTEXT, &mut Unbounded),
            Err(ProgramTransportError::Envelope(
                CertificationError::Encoding(_)
            ))
        ));
    }
    let overdeep = format!(
        "{}0{}",
        "[".repeat(MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1),
        "]".repeat(MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1),
    );
    assert!(matches!(
        decode_residual_program(overdeep.as_bytes(), CONTEXT, &mut Unbounded),
        Err(ProgramTransportError::Envelope(
            CertificationError::DepthLimit { .. }
        ))
    ));
    assert!(matches!(
        decode_residual_program(
            &vec![b' '; MAX_COMPILED_PROGRAM_ARTIFACT_BYTES + 1],
            CONTEXT,
            &mut Unbounded,
        ),
        Err(ProgramTransportError::Envelope(
            CertificationError::ByteLimit { .. }
        ))
    ));
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(AMPLE, 777))
}

struct CancelAfter {
    ledger: Budget,
    checkpoints: usize,
}

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.ledger.charge(dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.ledger.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints >= 5 {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn meter_error(error: &ProgramTransportError) -> Option<&MeterError> {
    match error {
        ProgramTransportError::Envelope(CertificationError::Meter(error))
        | ProgramTransportError::Candidate(ProgramArtifactError::Budget(error)) => Some(error),
        _ => None,
    }
}

#[test]
fn envelope_and_candidate_work_share_one_budget_and_preserve_cancellation() {
    let residual = residual_fixture();
    let objective = objective_fixture();
    let mut ledger = budget();
    let residual_bytes = export_residual_program(&residual, CONTEXT, &mut ledger).unwrap();
    decode_residual_program(&residual_bytes, CONTEXT, &mut ledger).unwrap();
    let objective_bytes = export_objective_program(&objective, CONTEXT, &mut ledger).unwrap();
    let before = DIMENSIONS.map(|dimension| ledger.remaining(dimension));
    decode_objective_program(&objective_bytes, CONTEXT, &mut ledger).unwrap();
    for (dimension, previous) in DIMENSIONS.into_iter().zip(before) {
        assert!(ledger.remaining(dimension) < previous);
    }
    assert_eq!(ledger.verifier_remaining(), 777);
    for dimension in DIMENSIONS {
        let mut limits = BudgetLimits::uniform(AMPLE, 777);
        limits.dimensions[dimension.index()] = 0;
        for error in [
            export_residual_program(&residual, CONTEXT, &mut Budget::new(limits)).unwrap_err(),
            decode_residual_program(&residual_bytes, CONTEXT, &mut Budget::new(limits))
                .unwrap_err(),
        ] {
            assert!(matches!(meter_error(&error), Some(MeterError::Budget(_))));
        }
    }
    let mut cancelled = CancelAfter {
        ledger: budget(),
        checkpoints: 0,
    };
    let error = export_residual_program(&residual, CONTEXT, &mut cancelled).unwrap_err();
    assert!(matches!(meter_error(&error), Some(MeterError::Cancelled)));
    assert!(cancelled.ledger.remaining(Dimension::ComputeSteps) < AMPLE);
    let mut cancelled = CancelAfter {
        ledger: budget(),
        checkpoints: 0,
    };
    let error = decode_residual_program(&residual_bytes, CONTEXT, &mut cancelled).unwrap_err();
    assert!(matches!(meter_error(&error), Some(MeterError::Cancelled)));
    assert!(cancelled.ledger.remaining(Dimension::ComputeSteps) < AMPLE);
}
