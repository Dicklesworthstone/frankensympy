//! Portable existence proofs replay without depending on a candidate generator.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::root_certificate::artifact::{CertificationError, MAX_ROOT_CERTIFICATE_JSON_DEPTH};
use fsym_core::root_certificate::scalar_bracket_artifact::{
    MAX_SCALAR_ROOT_BRACKET_BYTES, SCALAR_ROOT_BRACKET_SCHEMA_VERSION, ScalarBracketError,
    export_scalar_root_bracket, replay_scalar_root_bracket,
};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_SCALAR_ROOT_INPUT_BITS, RealBall, RootBoxError,
    ScalarRootBracketError, ScalarRootBracketReceipt, ScalarRootBracketWitness, Symbol,
    SymbolIdentity, parse_raw, verify_scalar_root_bracket,
};
use serde::Serialize;
use std::error::Error;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn witness(lower: BigRational, upper: BigRational) -> ScalarRootBracketWitness {
    ScalarRootBracketWitness {
        lower,
        upper,
        precision_digits: 24,
    }
}

struct Statement {
    residuals: [Expr; 1],
    variables: [Symbol; 1],
    lower: BigRational,
    upper: BigRational,
    tolerance: BigRational,
    context: [u8; 32],
}

impl Statement {
    fn new(source: &str, lower: BigRational, upper: BigRational) -> Self {
        Self {
            residuals: [parse_raw(source).unwrap()],
            variables: [Symbol::new("x")],
            lower,
            upper,
            tolerance: q(1, 1000),
            context: [7; 32],
        }
    }

    fn checked(&self, proposed: &ScalarRootBracketWitness) -> ScalarRootBracketReceipt {
        verify_scalar_root_bracket(
            &self.residuals[0],
            &self.variables[0],
            &self.lower,
            &self.upper,
            &self.tolerance,
            proposed,
            &mut Unbounded,
        )
        .unwrap()
    }

    fn export(
        &self,
        receipt: &ScalarRootBracketReceipt,
        meter: &mut impl BudgetMeter,
    ) -> Result<Vec<u8>, ScalarBracketError> {
        export_scalar_root_bracket(
            &self.residuals,
            &self.variables,
            receipt,
            self.context,
            meter,
        )
    }

    fn replay(
        &self,
        bytes: &[u8],
        meter: &mut impl BudgetMeter,
    ) -> Result<ScalarRootBracketReceipt, ScalarBracketError> {
        replay_scalar_root_bracket(
            bytes,
            &self.residuals,
            &self.variables,
            &self.lower,
            &self.upper,
            &self.tolerance,
            self.context,
            meter,
        )
    }

    fn candidate<W: Serialize>(&self, proposed: &W) -> Vec<u8> {
        // This independently specified envelope preserves the native Session's
        // original version-1 field order. Serialization grants no acceptance.
        #[derive(Serialize)]
        struct Artifact<'a, W: Serialize> {
            schema_version: u32,
            context_digest: [u8; 32],
            residuals: &'a [Expr],
            variables: &'a [Symbol],
            lower: &'a BigRational,
            upper: &'a BigRational,
            absolute_tolerance: &'a BigRational,
            witness: &'a W,
        }
        serde_json::to_vec(&Artifact {
            schema_version: SCALAR_ROOT_BRACKET_SCHEMA_VERSION,
            context_digest: self.context,
            residuals: &self.residuals,
            variables: &self.variables,
            lower: &self.lower,
            upper: &self.upper,
            absolute_tolerance: &self.tolerance,
            witness: proposed,
        })
        .unwrap()
    }
}

#[test]
fn exact_repeated_nonsmooth_and_endpoint_roots_replay_the_legacy_wire_format() {
    for (source, lower, upper, root) in [
        ("(3*x-1)^3", q(0, 1), q(1, 1), q(1, 3)),
        ("abs(x)+2*x", q(-1, 1), q(1, 1), q(0, 1)),
        ("(x-1)^4", q(0, 1), q(1, 1), q(1, 1)),
        ("abs(x-1/3)", q(1, 3), q(1, 3), q(1, 3)),
    ] {
        let statement = Statement::new(source, lower, upper);
        let proposed = witness(root.clone(), root.clone());
        let receipt = statement.checked(&proposed);
        let encoded = statement.export(&receipt, &mut Unbounded).unwrap();
        assert_eq!(encoded, statement.candidate(&proposed));
        let replayed = statement.replay(&encoded, &mut Unbounded).unwrap();
        assert_eq!(replayed, receipt);
        assert_eq!(replayed.root_enclosure(), &RealBall::exact(root));
        assert_eq!(replayed.expression(), &statement.residuals[0]);
        assert_eq!(replayed.variable(), &statement.variables[0]);
        assert_eq!(replayed.lower(), &statement.lower);
        assert_eq!(replayed.upper(), &statement.upper);
        assert_eq!(replayed.absolute_tolerance(), &statement.tolerance);
        assert_eq!(
            statement.export(&replayed, &mut Unbounded).unwrap(),
            encoded
        );
        let object: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(object.as_object().unwrap().len(), 8);
        for field in ["accepted", "unique", "root_enclosure", "endpoint_values"] {
            assert!(object.get(field).is_none());
        }
    }
}

#[test]
fn a_valid_sign_bracket_can_contain_several_roots_without_proving_uniqueness() {
    let mut statement = Statement::new("(x+1)*x*(x-1)", q(-2, 1), q(2, 1));
    statement.tolerance = q(4, 1);
    let bytes = statement.candidate(&witness(q(-2, 1), q(2, 1)));
    let receipt = statement.replay(&bytes, &mut Unbounded).unwrap();
    for root in [-1, 0, 1] {
        assert!(receipt.root_enclosure().contains(&q(root, 1)));
    }
    assert_eq!(receipt.root_enclosure().width(), q(4, 1));
}

#[test]
fn expected_source_axis_interval_tolerance_and_context_precede_witness_decoding() {
    let mut statement = Statement::new("x", q(0, 1), q(1, 1));
    let malformed = statement.candidate(&serde_json::json!({"lower": "invalid rational"}));
    assert!(matches!(
        statement.replay(&malformed, &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    statement.residuals[0] = parse_raw("x+0").unwrap();
    assert!(matches!(
        statement.replay(&malformed, &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    statement.residuals[0] = parse_raw("x").unwrap();
    statement.variables[0] =
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    assert!(matches!(
        statement.replay(&malformed, &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    statement.variables[0] = Symbol::new("x");
    for (lower, upper, tolerance) in [
        (q(-1, 1), q(1, 1), q(1, 1000)),
        (q(0, 1), q(2, 1), q(1, 1000)),
        (q(0, 1), q(1, 1), q(1, 100)),
    ] {
        statement.lower = lower;
        statement.upper = upper;
        statement.tolerance = tolerance;
        assert!(matches!(
            statement.replay(&malformed, &mut Unbounded),
            Err(ScalarBracketError::StatementMismatch)
        ));
    }
    statement.context = [8; 32];
    assert!(matches!(
        statement.replay(&malformed, &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::UniverseMismatch { .. }
        ))
    ));
}

#[test]
fn narrow_candidates_cannot_hide_original_poles_or_trigger_generator_fallback() {
    for source in ["(x-1/2)/x", "0*(1/x)", "x+(1/x)^0-1"] {
        let statement = Statement::new(source, q(0, 1), q(1, 1));
        let bytes = statement.candidate(&witness(q(1, 2), q(1, 2)));
        assert!(matches!(
            statement.replay(&bytes, &mut Unbounded),
            Err(ScalarBracketError::Bracket(
                ScalarRootBracketError::Enclosure(_)
            ))
        ));
    }
    let statement = Statement::new("(3*x-1)^3", q(0, 1), q(1, 1));
    // The full interval has a root; the supplied point does not. A retry with
    // a generator would incorrectly turn this invalid proof into acceptance.
    let bytes = statement.candidate(&witness(q(1, 2), q(1, 2)));
    assert!(matches!(
        statement.replay(&bytes, &mut Unbounded),
        Err(ScalarBracketError::Bracket(
            ScalarRootBracketError::NoSignChange
        ))
    ));
}

#[test]
fn export_binds_the_receipt_source_and_requires_exactly_one_axis() {
    let statement = Statement::new("x", q(0, 1), q(1, 1));
    let receipt = statement.checked(&witness(q(0, 1), q(0, 1)));
    let mut different = Statement::new("x+0", q(0, 1), q(1, 1));
    assert!(matches!(
        different.export(&receipt, &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    different.residuals = statement.residuals;
    different.variables[0] =
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    assert!(matches!(
        different.export(&receipt, &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    assert!(matches!(
        export_scalar_root_bracket(&[], &different.variables, &receipt, [7; 32], &mut Unbounded),
        Err(ScalarBracketError::ProblemDimension)
    ));
}

#[test]
fn bounded_decoding_and_canonical_bytes_refuse_untrusted_claims() {
    let statement = Statement::new("x", q(0, 1), q(1, 1));
    let bytes = statement.candidate(&witness(q(0, 1), q(0, 1)));
    let mut padded = bytes.clone();
    padded.push(b'\n');
    assert!(matches!(
        statement.replay(&padded, &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::NonCanonical
        ))
    ));
    assert!(matches!(
        statement.replay(
            &vec![b' '; MAX_SCALAR_ROOT_BRACKET_BYTES + 1],
            &mut Unbounded
        ),
        Err(ScalarBracketError::Certification(
            CertificationError::ByteLimit { .. }
        ))
    ));
    assert!(matches!(
        statement.replay(&[b'['; MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1], &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::DepthLimit { .. }
        ))
    ));
    let mut false_evidence = serde_json::to_value(witness(q(0, 1), q(0, 1))).unwrap();
    false_evidence["unique"] = serde_json::json!(true);
    assert!(matches!(
        statement.replay(&statement.candidate(&false_evidence), &mut Unbounded),
        Err(ScalarBracketError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    let huge = BigRational::from_integer(BigInt::from(1) << MAX_SCALAR_ROOT_INPUT_BITS as u32);
    for proposed in [
        witness(q(1, 1), q(0, 1)),
        witness(q(0, 1), huge),
        ScalarRootBracketWitness {
            precision_digits: 129,
            ..witness(q(0, 1), q(0, 1))
        },
    ] {
        assert!(matches!(
            statement.replay(&statement.candidate(&proposed), &mut Unbounded),
            Err(ScalarBracketError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
}

#[derive(Default)]
struct Observed {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Observed {
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
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

#[test]
fn export_replay_and_repeated_requests_consume_one_cumulative_ledger() {
    let statement = Statement::new("x", q(0, 1), q(1, 1));
    let receipt = statement.checked(&witness(q(0, 1), q(0, 1)));
    let mut observed = Observed::default();
    let bytes = statement.export(&receipt, &mut observed).unwrap();
    assert_eq!(statement.replay(&bytes, &mut observed).unwrap(), receipt);
    let mut budget = Budget::new(BudgetLimits {
        dimensions: observed.charges.map(|amount| 2 * amount),
        verifier_pool: 193,
    });
    for _ in 0..2 {
        let encoded = statement.export(&receipt, &mut budget).unwrap();
        assert_eq!(statement.replay(&encoded, &mut budget).unwrap(), receipt);
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(observed.charges[dimension.index()] > 0);
        assert_eq!(budget.remaining(dimension), 0);
    }
    assert!(statement.replay(&bytes, &mut budget).is_err());
    assert_eq!(budget.verifier_remaining(), 193);
}

#[test]
fn cancellation_never_publishes_a_partial_export_or_accepted_replay() {
    let statement = Statement::new("x", q(0, 1), q(1, 1));
    let receipt = statement.checked(&witness(q(0, 1), q(0, 1)));
    let bytes = statement.export(&receipt, &mut Unbounded).unwrap();
    for replay in [false, true] {
        let operation = |meter: &mut Observed| {
            if replay {
                statement.replay(&bytes, meter).map(|_| ())
            } else {
                statement.export(&receipt, meter).map(|_| ())
            }
        };
        let mut observed = Observed::default();
        operation(&mut observed).unwrap();
        assert!(observed.checkpoints > 3);
        for cancel_at in [1, observed.checkpoints / 2, observed.checkpoints] {
            let mut cancelled = Observed {
                cancel_at: Some(cancel_at),
                ..Observed::default()
            };
            let error = operation(&mut cancelled).unwrap_err();
            assert!(matches!(
                error,
                ScalarBracketError::Certification(CertificationError::Meter(MeterError::Cancelled))
                    | ScalarBracketError::Bracket(ScalarRootBracketError::Enclosure(
                        RootBoxError::Meter(MeterError::Cancelled)
                    ))
            ));
            assert_eq!(cancelled.checkpoints, cancel_at);
        }
    }
}

#[test]
fn immediate_operational_causes_remain_visible_to_host_error_mappers() {
    let allocation = ScalarBracketError::Certification(CertificationError::AllocationFailure);
    assert!(matches!(
        allocation
            .source()
            .unwrap()
            .downcast_ref::<CertificationError>(),
        Some(CertificationError::AllocationFailure)
    ));
    let resource = ScalarBracketError::Bracket(ScalarRootBracketError::ResourceLimit("precision"));
    assert!(matches!(
        resource
            .source()
            .unwrap()
            .downcast_ref::<ScalarRootBracketError>(),
        Some(ScalarRootBracketError::ResourceLimit("precision"))
    ));
}
