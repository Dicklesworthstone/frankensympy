//! Shared integral transport replays the real checker against receiver-owned input.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::root_certificate::artifact::{CertificationError, MAX_ROOT_CERTIFICATE_JSON_DEPTH};
use fsym_core::root_certificate::integral_artifact::{
    INTEGRAL_CERTIFICATE_SCHEMA_VERSION, IntegrationError, MAX_INTEGRAL_CERTIFICATE_BYTES,
    export_integral_certificate, replay_integral_certificate,
};
use fsym_core::{
    BigInt, BigRational, Expr, IntegralEnclosureReceipt, MAX_QUADRATURE_INPUT_BITS,
    MAX_QUADRATURE_INTERVALS, QuadratureError, QuadratureWitness, RealBall, RootBoxError, Symbol,
    SymbolIdentity, parse_raw, verify_integral_enclosure,
};
use serde::Serialize;
use std::error::Error;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn witness(partition: Vec<BigRational>) -> QuadratureWitness {
    QuadratureWitness {
        partition,
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

    fn checked(&self, witness: &QuadratureWitness) -> IntegralEnclosureReceipt {
        verify_integral_enclosure(
            &self.residuals[0],
            &self.variables[0],
            &self.lower,
            &self.upper,
            &self.tolerance,
            witness,
            &mut Unbounded,
        )
        .unwrap()
    }

    fn export(
        &self,
        receipt: &IntegralEnclosureReceipt,
        meter: &mut impl BudgetMeter,
    ) -> Result<Vec<u8>, IntegrationError> {
        export_integral_certificate(
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
    ) -> Result<IntegralEnclosureReceipt, IntegrationError> {
        replay_integral_certificate(
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

    fn candidate<W: Serialize>(&self, witness: &W) -> Vec<u8> {
        serde_json::to_vec(&ProposedArtifact {
            schema_version: INTEGRAL_CERTIFICATE_SCHEMA_VERSION,
            context_digest: self.context,
            residuals: &self.residuals,
            variables: &self.variables,
            lower: &self.lower,
            upper: &self.upper,
            absolute_tolerance: &self.tolerance,
            witness,
        })
        .unwrap()
    }
}

// Keep the legacy Session envelope's exact field order. Candidate construction
// supplies no authority: every acceptance below comes from the core checker.
#[derive(Serialize)]
struct ProposedArtifact<'a, W: Serialize> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    lower: &'a BigRational,
    upper: &'a BigRational,
    absolute_tolerance: &'a BigRational,
    witness: &'a W,
}

#[test]
fn exact_polynomial_and_absolute_partitions_roundtrip_with_original_sources() {
    // Polynomial antiderivatives and triangle areas supply these exact values.
    for (source, points, exact) in [
        ("2*x^2+5*x-7", vec![q(-2, 1), q(3, 1)], q(5, 6)),
        ("abs(x-1/3)", vec![q(0, 1), q(1, 3), q(1, 1)], q(5, 18)),
        (
            "abs(abs(x)-1/2)",
            vec![q(-1, 1), q(-1, 2), q(0, 1), q(1, 2), q(1, 1)],
            q(1, 2),
        ),
    ] {
        let mut statement =
            Statement::new(source, points[0].clone(), points.last().unwrap().clone());
        let proposed = witness(points);
        for expected in [exact.clone(), -exact] {
            let receipt = statement.checked(&proposed);
            assert_eq!(receipt.enclosure(), &RealBall::exact(expected));
            let bytes = statement.export(&receipt, &mut Unbounded).unwrap();
            assert_eq!(bytes, statement.candidate(&proposed));
            let replayed = statement.replay(&bytes, &mut Unbounded).unwrap();
            assert_eq!(replayed, receipt);
            assert_eq!(replayed.integrand(), &statement.residuals[0]);
            assert_eq!(replayed.variable(), &statement.variables[0]);
            assert_eq!(replayed.lower(), &statement.lower);
            assert_eq!(replayed.upper(), &statement.upper);
            assert_eq!(replayed.absolute_tolerance(), &statement.tolerance);
            assert_eq!(statement.export(&replayed, &mut Unbounded).unwrap(), bytes);
            let object: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(object.as_object().unwrap().len(), 8);
            for field in ["accepted", "verified", "enclosure", "integral", "receipt"] {
                assert!(object.get(field).is_none());
            }
            std::mem::swap(&mut statement.lower, &mut statement.upper);
        }
    }
}

#[test]
fn equal_endpoints_replay_zero_only_after_admitting_the_original_source() {
    let statement = Statement::new("abs(x)", q(0, 1), q(0, 1));
    let proposed = witness(vec![q(0, 1)]);
    let receipt = statement.checked(&proposed);
    let bytes = statement.export(&receipt, &mut Unbounded).unwrap();
    let replayed = statement.replay(&bytes, &mut Unbounded).unwrap();
    assert_eq!(replayed.enclosure(), &RealBall::from_i64(0));
    assert_eq!(replayed.interval_count(), 0);
    let pole = Statement::new("0*(1/x)", q(0, 1), q(0, 1));
    assert!(matches!(
        pole.replay(&pole.candidate(&proposed), &mut Unbounded),
        Err(IntegrationError::Quadrature(QuadratureError::Enclosure(_)))
    ));
}

#[test]
fn every_receiver_binding_precedes_untrusted_witness_decoding() {
    let mut statement = Statement::new("x^2", q(0, 1), q(1, 1));
    let malformed = statement.candidate(&serde_json::json!({"partition": "not a partition"}));
    assert!(matches!(
        statement.replay(&malformed, &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    for source in ["x^3", "x^2+0"] {
        let other = Statement::new(source, q(0, 1), q(1, 1));
        assert!(matches!(
            other.replay(&malformed, &mut Unbounded),
            Err(IntegrationError::Certification(
                CertificationError::SourceMismatch
            ))
        ));
    }
    statement.variables[0] =
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    assert!(matches!(
        statement.replay(&malformed, &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    statement.variables[0] = Symbol::new("x");
    for (lower, upper, tolerance) in [
        (q(1, 1), q(0, 1), q(1, 1000)),
        (q(0, 1), q(2, 1), q(1, 1000)),
        (q(0, 1), q(1, 1), q(1, 100)),
    ] {
        statement.lower = lower;
        statement.upper = upper;
        statement.tolerance = tolerance;
        assert!(matches!(
            statement.replay(&malformed, &mut Unbounded),
            Err(IntegrationError::StatementMismatch)
        ));
    }
    statement.context = [8; 32];
    assert!(matches!(
        statement.replay(&malformed, &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::UniverseMismatch { .. }
        ))
    ));
}

#[test]
fn export_requires_the_same_original_source_and_typed_axis_as_the_receipt() {
    let statement = Statement::new("x^2", q(0, 1), q(1, 1));
    let receipt = statement.checked(&witness(vec![q(0, 1), q(1, 1)]));
    let mut other = Statement::new("x^2+0", q(0, 1), q(1, 1));
    assert!(matches!(
        other.export(&receipt, &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    other.residuals = statement.residuals;
    other.variables[0] =
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    assert!(matches!(
        other.export(&receipt, &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
    assert!(matches!(
        export_integral_certificate(&[], &other.variables, &receipt, [7; 32], &mut Unbounded),
        Err(IntegrationError::ProblemDimension)
    ));
}

#[test]
fn canonical_candidates_cannot_hide_poles_or_claim_unproved_accuracy() {
    let proposed = witness(vec![q(0, 1), q(1, 1)]);
    for source in ["0*(1/x)", "(1/x)^0", "log(x)-log(x)"] {
        let statement = Statement::new(source, q(0, 1), q(1, 1));
        assert!(
            matches!(
                statement.replay(&statement.candidate(&proposed), &mut Unbounded),
                Err(IntegrationError::Quadrature(QuadratureError::Enclosure(_)))
            ),
            "{source}"
        );
    }
    for source in ["x^4", "abs(x-1/3)"] {
        let statement = Statement::new(source, q(0, 1), q(1, 1));
        assert!(
            matches!(
                statement.replay(&statement.candidate(&proposed), &mut Unbounded),
                Err(IntegrationError::Quadrature(
                    QuadratureError::AccuracyNotReached
                ))
            ),
            "{source}"
        );
    }
    let statement = Statement::new("x^2", q(0, 1), q(1, 1));
    let mut false_claim = serde_json::to_value(&proposed).unwrap();
    false_claim["integral"] = serde_json::json!(0);
    assert!(matches!(
        statement.replay(&statement.candidate(&false_claim), &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
}

#[test]
fn incomplete_overlapping_duplicate_and_reversed_partitions_refuse() {
    let statement = Statement::new("x^2", q(0, 1), q(1, 1));
    for points in [
        vec![],
        vec![q(0, 1)],
        vec![q(0, 1), q(1, 2)],
        vec![q(1, 2), q(1, 1)],
        vec![q(0, 1), q(1, 2), q(1, 2), q(1, 1)],
        vec![q(0, 1), q(3, 4), q(1, 4), q(1, 1)],
        vec![q(1, 1), q(0, 1)],
    ] {
        let bytes = statement.candidate(&witness(points));
        assert!(matches!(
            statement.replay(&bytes, &mut Unbounded),
            Err(IntegrationError::Quadrature(
                QuadratureError::InvalidPartition(_)
            ))
        ));
    }
}

#[test]
fn schema_canonical_encoding_and_decoder_envelopes_fail_closed() {
    let statement = Statement::new("x^2", q(0, 1), q(1, 1));
    let bytes = statement.candidate(&witness(vec![q(0, 1), q(1, 1)]));
    let version = std::str::from_utf8(&bytes).unwrap().replacen(
        "\"schema_version\":1",
        "\"schema_version\":2",
        1,
    );
    assert!(matches!(
        statement.replay(version.as_bytes(), &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::UnsupportedVersion { .. }
        ))
    ));
    let mut claimed_acceptance = bytes[..bytes.len() - 1].to_vec();
    claimed_acceptance.extend_from_slice(b",\"accepted\":true}");
    let duplicate = format!(
        "{{\"schema_version\":1,{}",
        std::str::from_utf8(&bytes[1..]).unwrap()
    );
    for malformed in [claimed_acceptance.as_slice(), duplicate.as_bytes()] {
        assert!(matches!(
            statement.replay(malformed, &mut Unbounded),
            Err(IntegrationError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
    let mut padded = bytes;
    padded.push(b'\n');
    assert!(matches!(
        statement.replay(&padded, &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::NonCanonical
        ))
    ));
    assert!(matches!(
        statement.replay(
            &vec![b' '; MAX_INTEGRAL_CERTIFICATE_BYTES + 1],
            &mut Unbounded
        ),
        Err(IntegrationError::Certification(
            CertificationError::ByteLimit { .. }
        ))
    ));
    assert!(matches!(
        statement.replay(&[b'['; MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1], &mut Unbounded),
        Err(IntegrationError::Certification(
            CertificationError::DepthLimit { .. }
        ))
    ));
    for proposed in [
        witness(vec![q(0, 1); MAX_QUADRATURE_INTERVALS + 2]),
        QuadratureWitness {
            partition: vec![q(0, 1), q(1, 1)],
            precision_digits: 7,
        },
        witness(vec![
            q(0, 1),
            BigRational::from_integer(BigInt::from(1) << MAX_QUADRATURE_INPUT_BITS as u32),
            q(1, 1),
        ]),
    ] {
        assert!(matches!(
            statement.replay(&statement.candidate(&proposed), &mut Unbounded),
            Err(IntegrationError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
}

#[derive(Default)]
struct Observed {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
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
        Ok(())
    }
}

struct CancelAfter(usize);

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.0 == 0 {
            return Err(MeterError::Cancelled);
        }
        self.0 -= 1;
        Ok(())
    }
}

fn assert_cancelled(error: IntegrationError) {
    assert!(matches!(
        error,
        IntegrationError::Certification(CertificationError::Meter(MeterError::Cancelled))
            | IntegrationError::Quadrature(QuadratureError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
    ));
}

#[test]
fn export_and_replay_share_cumulative_compute_memory_and_allocation_allowances() {
    let statement = Statement::new("x^2", q(0, 1), q(1, 1));
    let receipt = statement.checked(&witness(vec![q(0, 1), q(1, 1)]));
    let mut observed = Observed::default();
    let bytes = statement.export(&receipt, &mut observed).unwrap();
    assert_eq!(statement.replay(&bytes, &mut observed).unwrap(), receipt);
    let mut budget = Budget::new(BudgetLimits {
        dimensions: observed.charges.map(|charge| charge * 2),
        verifier_pool: 193,
    });
    for _ in 0..2 {
        assert_eq!(statement.export(&receipt, &mut budget).unwrap(), bytes);
        assert_eq!(statement.replay(&bytes, &mut budget).unwrap(), receipt);
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(observed.charges[dimension.index()] > 0);
        assert_eq!(budget.remaining(dimension), 0);
    }
    assert!(matches!(
        statement.export(&receipt, &mut budget),
        Err(IntegrationError::Certification(CertificationError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert!(matches!(
        statement.replay(&bytes, &mut budget),
        Err(IntegrationError::Certification(CertificationError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.verifier_remaining(), 193);
}

#[test]
fn cancellation_at_early_and_late_transport_safe_points_never_publishes() {
    let statement = Statement::new("x^2", q(0, 1), q(1, 1));
    let receipt = statement.checked(&witness(vec![q(0, 1), q(1, 1)]));
    let mut encoded = Observed::default();
    let bytes = statement.export(&receipt, &mut encoded).unwrap();
    let mut replayed = Observed::default();
    statement.replay(&bytes, &mut replayed).unwrap();
    for before in [0, 1, encoded.checkpoints / 2, encoded.checkpoints - 1] {
        assert_cancelled(
            statement
                .export(&receipt, &mut CancelAfter(before))
                .unwrap_err(),
        );
    }
    for before in [0, 1, replayed.checkpoints / 2, replayed.checkpoints - 1] {
        assert_cancelled(
            statement
                .replay(&bytes, &mut CancelAfter(before))
                .unwrap_err(),
        );
    }
}

#[test]
fn integration_errors_preserve_leaf_errors_for_host_operational_classification() {
    let allocation = IntegrationError::from(CertificationError::AllocationFailure);
    assert_eq!(allocation.to_string(), "certification allocation refused");
    assert!(matches!(
        allocation
            .source()
            .unwrap()
            .downcast_ref::<CertificationError>(),
        Some(CertificationError::AllocationFailure)
    ));
    let resource = IntegrationError::from(QuadratureError::ResourceLimit("partition work"));
    assert!(matches!(
        resource.source().unwrap().downcast_ref::<QuadratureError>(),
        Some(QuadratureError::ResourceLimit("partition work"))
    ));
}
