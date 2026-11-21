//! Saved Python bundles migrate through bounded candidate decoding and fresh
//! source-bound admission; neither the old envelope nor a status flag is proof.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::program_artifact::{
    MAX_PROGRAM_CANDIDATE_BYTES, ProgramAdmissionError, ProgramArtifactError,
    admit_residual_candidate, decode_residual_candidate_with_legacy, encode_residual_candidate,
};
use fsym_calculus::verified::{MAX_VERIFIED_ARTIFACT_BYTES, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol};

const AMPLE: u64 = 1 << 40;

fn fixture() -> VerifiedResidualSystem {
    VerifiedResidualSystem::try_compile(
        &[
            fsym_core::parse("x^2+y-5").unwrap(),
            fsym_core::parse("x-y-1").unwrap(),
        ],
        &[Symbol::new("y"), Symbol::new("x")],
    )
    .unwrap()
}

#[test]
fn historical_and_portable_bundles_preserve_sources_axes_and_checked_values() {
    let original = fixture();
    let portable = encode_residual_candidate(&original, &mut Unbounded).unwrap();
    let legacy = original.to_json_bytes().unwrap();
    for bytes in [&legacy, &portable] {
        let mut meter = Budget::new(BudgetLimits::uniform(AMPLE, 73));
        let candidate = decode_residual_candidate_with_legacy(bytes, &mut meter).unwrap();
        let restored =
            admit_residual_candidate(original.residuals(), original.vars(), candidate, &mut meter)
                .unwrap();
        let values = restored
            .evaluate_sparse_with_meter(&[1.0, 2.0], &mut meter)
            .unwrap();
        assert_eq!(values.residuals, [0.0, 0.0]);
        assert_eq!(values.jacobian, [1.0, 4.0, -1.0, 1.0]);
        assert_eq!(
            encode_residual_candidate(&restored, &mut meter).unwrap(),
            portable
        );
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            assert!(meter.remaining(dimension) < AMPLE);
        }
        assert_eq!(meter.verifier_remaining(), 73);
    }
}

#[test]
fn legacy_whitespace_and_original_size_limit_do_not_relax_portable_canonicality() {
    let program = fixture();
    let legacy = program.to_json_bytes().unwrap();
    let mut padded = vec![b' '; MAX_PROGRAM_CANDIDATE_BYTES + 1];
    padded.extend_from_slice(&legacy);
    let candidate = decode_residual_candidate_with_legacy(&padded, &mut Unbounded).unwrap();
    admit_residual_candidate(
        program.residuals(),
        program.vars(),
        candidate,
        &mut Unbounded,
    )
    .unwrap();

    let portable = encode_residual_candidate(&program, &mut Unbounded).unwrap();
    let mut noncanonical = portable.clone();
    noncanonical.push(b'\n');
    assert!(matches!(
        decode_residual_candidate_with_legacy(&noncanonical, &mut Unbounded),
        Err(ProgramArtifactError::NonCanonical)
    ));
    let mut oversized = vec![b' '; MAX_PROGRAM_CANDIDATE_BYTES + 1];
    oversized.extend_from_slice(&portable);
    assert!(matches!(
        decode_residual_candidate_with_legacy(&oversized, &mut Unbounded),
        Err(ProgramArtifactError::ByteLimit {
            max: MAX_PROGRAM_CANDIDATE_BYTES,
            ..
        })
    ));
    assert!(matches!(
        decode_residual_candidate_with_legacy(
            &vec![b' '; MAX_VERIFIED_ARTIFACT_BYTES + 1],
            &mut Unbounded
        ),
        Err(ProgramArtifactError::ByteLimit {
            max: MAX_VERIFIED_ARTIFACT_BYTES,
            ..
        })
    ));
}

#[test]
fn forged_legacy_bytecode_decodes_as_candidate_but_independent_admission_rejects_it() {
    let program = fixture();
    let mut wire: serde_json::Value =
        serde_json::from_slice(&program.to_json_bytes().unwrap()).unwrap();
    wire["jacobian"]["entries"][0]["compiled_deriv"]["ops"] =
        serde_json::json!([{"LoadConst": 999.0}]);
    let bytes = serde_json::to_vec(&wire).unwrap();
    let candidate = decode_residual_candidate_with_legacy(&bytes, &mut Unbounded).unwrap();
    assert!(matches!(
        admit_residual_candidate(
            program.residuals(),
            program.vars(),
            candidate.clone(),
            &mut Unbounded
        ),
        Err(ProgramAdmissionError::Residual(_))
    ));

    // Bind the receiver's statement before any check of the forged proof.
    let wrong_sources = vec![Expr::from_i64(999); 2];
    assert!(matches!(
        admit_residual_candidate(
            &wrong_sources,
            program.vars(),
            candidate.clone(),
            &mut Unbounded
        ),
        Err(ProgramAdmissionError::SourceMismatch(
            "ordered residual sources"
        ))
    ));
    let mut reversed = program.vars().to_vec();
    reversed.reverse();
    assert!(matches!(
        admit_residual_candidate(program.residuals(), &reversed, candidate, &mut Unbounded),
        Err(ProgramAdmissionError::SourceMismatch(
            "ordered typed variables"
        ))
    ));
}

#[test]
fn mixed_duplicate_unknown_null_and_wrong_kind_envelopes_fail_closed() {
    let program = fixture();
    let original: serde_json::Value =
        serde_json::from_slice(&program.to_json_bytes().unwrap()).unwrap();
    for field in ["kind", "candidate", "verified", "receipt"] {
        let mut wire = original.clone();
        wire[field] = serde_json::Value::Null;
        assert!(
            decode_residual_candidate_with_legacy(
                &serde_json::to_vec(&wire).unwrap(),
                &mut Unbounded
            )
            .is_err()
        );
    }
    for bytes in [
        br#"{"schema_version":1,"schema_version":1,"jacobian":null}"#.as_slice(),
        br#"{"schema_version":0,"jacobian":null}"#.as_slice(),
        br#"{"schema_version":1,"kind":"objective","candidate":null}"#.as_slice(),
        br#"{"schema_version":1,"kind":null,"candidate":null}"#.as_slice(),
        br#"{"schema_version":1}"#.as_slice(),
    ] {
        assert!(decode_residual_candidate_with_legacy(bytes, &mut Unbounded).is_err());
    }
    let mut portable: serde_json::Value =
        serde_json::from_slice(&encode_residual_candidate(&program, &mut Unbounded).unwrap())
            .unwrap();
    portable["jacobian"] = serde_json::Value::Null;
    assert!(
        decode_residual_candidate_with_legacy(
            &serde_json::to_vec(&portable).unwrap(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn every_resource_is_cumulative_and_decoding_never_spends_the_verifier_pool() {
    let bytes = fixture().to_json_bytes().unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(AMPLE, 73);
        limits.dimensions[dimension.index()] = 0;
        let mut meter = Budget::new(limits);
        assert!(matches!(
            decode_residual_candidate_with_legacy(&bytes, &mut meter),
            Err(ProgramArtifactError::Budget(MeterError::Budget(_)))
        ));
        assert_eq!(meter.remaining(dimension), 0);
        assert_eq!(meter.verifier_remaining(), 73);
    }
    let mut meter = Budget::new(BudgetLimits::uniform(AMPLE, 73));
    decode_residual_candidate_with_legacy(&bytes, &mut meter).unwrap();
    let before = meter.remaining(Dimension::ComputeSteps);
    decode_residual_candidate_with_legacy(&bytes, &mut meter).unwrap();
    assert!(meter.remaining(Dimension::ComputeSteps) < before);
    assert_eq!(meter.verifier_remaining(), 73);
}

struct CancelAfter {
    remaining: usize,
    charged: u64,
}

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        self.checkpoint()?;
        self.charged += amount;
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()?;
        self.charged += charges.iter().map(|(_, amount)| amount).sum::<u64>();
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.remaining == 0 {
            return Err(MeterError::Cancelled);
        }
        self.remaining -= 1;
        Ok(())
    }
}

#[test]
fn cancellation_before_and_during_legacy_decoding_returns_no_candidate() {
    let bytes = fixture().to_json_bytes().unwrap();
    for remaining in [0, 5, 40] {
        let mut meter = CancelAfter {
            remaining,
            charged: 0,
        };
        assert!(matches!(
            decode_residual_candidate_with_legacy(&bytes, &mut meter),
            Err(ProgramArtifactError::Budget(MeterError::Cancelled))
        ));
        if remaining != 0 {
            assert!(meter.charged > 0);
        }
    }
}
