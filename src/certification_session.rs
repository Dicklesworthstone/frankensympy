//! Session transport for independently checked, original-source real systems.
//!
//! Object labels are typed session-local references, not mathematical content
//! identities. Registration freezes parsed native expressions and ordered axes;
//! workspace binding changes never rewrite that captured problem. Verification
//! uses the protected pool of the same cumulative session ledger.

use crate::certification::{
    CertificationError, ResidualProblem, export_root_certificate, replay_root_certificate,
};
use fsym_budget::{Budget, BudgetError, BudgetMeter, Dimension, MeterError, VerifierLease};
use fsym_core::{
    BigInt, BigRational, RealBall, RootBoxError, RootBoxReceipt, RootBoxWitness,
    RootIsolationError, RootIsolationReceipt, Symbol,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::io::{self, Write};
use thiserror::Error;

pub const CERTIFICATION_REQUEST_SCHEMA: u32 = 1;
/// The single-frame CLI subset leaves room for JSON escaping and the envelope.
pub const MAX_SESSION_ROOT_ARTIFACT_BYTES: usize = 24 * 1024;
pub const MAX_CERTIFICATION_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_SYSTEMS: usize = 32;
const MAX_CERTIFICATES: usize = 128;
const MAX_ISOLATIONS: usize = 32;
const MAX_SCALAR_BYTES: usize = 512;
const MAX_AXES: usize = fsym_core::MAX_ROOT_BOX_VARIABLES;

fn valid_label(value: &str, prefix: &str) -> bool {
    value.len() <= 128
        && value.strip_prefix(prefix).is_some_and(|tail| {
            !tail.is_empty()
                && tail
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
        })
}

/// A resident immutable residual system, scoped to one session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ResidualSystemId(String);

impl ResidualSystemId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "system:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ResidualSystemId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// An independently accepted root certificate, scoped to one session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct RootCertificateId(String);

impl RootCertificateId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "certificate:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for RootCertificateId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// A complete, independently verified root set in one finite closed box.
/// The 104-byte cap leaves room for deterministic child certificate labels.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct RootIsolationId(String);

impl RootIsolationId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if value.len() > 104 || !valid_label(&value, "isolation:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn child_certificate(
        &self,
        index: usize,
    ) -> Result<RootCertificateId, CertificationSessionError> {
        RootCertificateId::new(format!(
            "certificate:{}.root-{index}",
            &self.0["isolation:".len()..],
        ))
    }
}

impl<'de> Deserialize<'de> for RootIsolationId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// One member of a verified complete root set. Its label also supports the
/// ordinary root-certificate export/replay operations.
#[derive(Debug, Serialize)]
pub struct IsolatedRoot {
    pub certificate: RootCertificateId,
    pub domain: Vec<ExactInterval>,
    pub root_enclosure: Vec<ExactInterval>,
    pub contraction_bound: String,
}

/// Exact endpoints: integers, finite decimal strings, or integer fractions.
/// No binary floating-point conversion occurs at this boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactInterval {
    pub lower: String,
    pub upper: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactRootWitness {
    pub domain: Vec<ExactInterval>,
    pub center: Vec<String>,
    pub preconditioner: Vec<Vec<String>>,
    pub precision_digits: u32,
}

/// Bounded certification operations accepted by the native session and NDJSON
/// command. Every payload carries an explicit schema version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", deny_unknown_fields)]
pub enum CertificationRequest {
    RegisterResidualSystem {
        schema_version: u32,
        system: ResidualSystemId,
        residuals: Vec<String>,
        variables: Vec<String>,
    },
    EncloseResidualJacobian {
        schema_version: u32,
        system: ResidualSystemId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
    },
    CertifyRootBox {
        schema_version: u32,
        system: ResidualSystemId,
        certificate: RootCertificateId,
        witness: ExactRootWitness,
    },
    IsolateRealRootsInBox {
        schema_version: u32,
        system: ResidualSystemId,
        isolation: RootIsolationId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
        max_leaves: usize,
    },
    ExportRootCertificate {
        schema_version: u32,
        certificate: RootCertificateId,
    },
    ReplayRootCertificate {
        schema_version: u32,
        system: ResidualSystemId,
        certificate: RootCertificateId,
        artifact: String,
    },
}

impl CertificationRequest {
    fn schema_version(&self) -> u32 {
        match self {
            Self::RegisterResidualSystem { schema_version, .. }
            | Self::EncloseResidualJacobian { schema_version, .. }
            | Self::CertifyRootBox { schema_version, .. }
            | Self::IsolateRealRootsInBox { schema_version, .. }
            | Self::ExportRootCertificate { schema_version, .. }
            | Self::ReplayRootCertificate { schema_version, .. } => *schema_version,
        }
    }
}

/// Preserve raw payload bytes until typed deserialization. Decoding through a
/// `Value` first would silently collapse duplicate keys before serde can refuse
/// them. The caller has already bounded and charged the request envelope.
pub(crate) fn decode_request(
    line: &str,
) -> Result<CertificationRequest, CertificationSessionError> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Envelope<'a> {
        #[serde(default, rename = "id")]
        _id: Option<serde_json::Value>,
        #[serde(rename = "type")]
        kind: String,
        #[serde(borrow)]
        payload: &'a serde_json::value::RawValue,
    }
    #[derive(Serialize)]
    struct Request<'a> {
        #[serde(rename = "type")]
        kind: &'a str,
        payload: &'a serde_json::value::RawValue,
    }
    let invalid = |error: serde_json::Error| {
        CertificationSessionError::Serialization(format!(
            "invalid typed envelope at line {}, column {}",
            error.line(),
            error.column(),
        ))
    };
    let envelope: Envelope<'_> = serde_json::from_str(line).map_err(invalid)?;
    let request = serde_json::to_string(&Request {
        kind: &envelope.kind,
        payload: envelope.payload,
    })
    .map_err(invalid)?;
    serde_json::from_str(&request).map_err(invalid)
}

/// Wire descriptions of claims actually granted by the independent core.
/// Registration alone grants no mathematical evidence.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CertificationResponse {
    ResidualSystemRegistered {
        schema_version: u32,
        system: ResidualSystemId,
        rows: usize,
        variables: Vec<Symbol>,
        context_digest: [u8; 32],
    },
    ResidualJacobianEnclosed {
        schema_version: u32,
        system: ResidualSystemId,
        evidence: &'static str,
        claim: &'static str,
        domain: Vec<ExactInterval>,
        residuals: Vec<ExactInterval>,
        jacobian: Vec<Vec<ExactInterval>>,
        precision_digits: u32,
        context_digest: [u8; 32],
    },
    RootBoxCertified {
        schema_version: u32,
        system: ResidualSystemId,
        certificate: RootCertificateId,
        evidence: &'static str,
        claim: &'static str,
        domain: Vec<ExactInterval>,
        root_enclosure: Vec<ExactInterval>,
        contraction_bound: String,
        precision_digits: u32,
        context_digest: [u8; 32],
    },
    RealRootsIsolated {
        schema_version: u32,
        system: ResidualSystemId,
        isolation: RootIsolationId,
        evidence: &'static str,
        claim: &'static str,
        domain: Vec<ExactInterval>,
        root_count: usize,
        roots: Vec<IsolatedRoot>,
        leaf_count: usize,
        excluded_leaf_count: usize,
        precision_digits: u32,
        context_digest: [u8; 32],
    },
    RootCertificateExported {
        schema_version: u32,
        system: ResidualSystemId,
        certificate: RootCertificateId,
        artifact: String,
    },
}

#[derive(Debug, Error)]
pub enum CertificationSessionError {
    #[error("unsupported certification request schema {0}")]
    UnsupportedVersion(u32),
    #[error(
        "object label must have the proper system:, certificate: or isolation: kind and a bounded identifier"
    )]
    InvalidLabel,
    #[error("object label already exists in this session")]
    DuplicateObject,
    #[error("unknown residual system")]
    UnknownSystem,
    #[error("unknown accepted root certificate")]
    UnknownCertificate,
    #[error("resident certification object limit reached")]
    ObjectLimit,
    #[error(
        "exact scalar must be a bounded integer, decimal, or integer fraction with nonzero denominator"
    )]
    InvalidScalar,
    #[error("box endpoints are reversed or dimensions are inconsistent")]
    InvalidDomain,
    #[error(
        "certificate exceeds the bounded single-frame session transport; use the native artifact API"
    )]
    ArtifactLimit,
    #[error("certification response exceeds its output limit")]
    OutputLimit,
    #[error("certification serialization failed: {0}")]
    Serialization(String),
    #[error("certification resource accounting refused: {0}")]
    Meter(#[from] MeterError),
    #[error("certification refused: {0}")]
    Certification(#[from] CertificationError),
}

impl CertificationSessionError {
    /// Stable machine category; mathematical refusal is distinct from an
    /// unsupported operation, exhausted resources, malformed input or replay.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedVersion(_) => "unsupported_schema",
            Self::InvalidLabel | Self::InvalidScalar | Self::InvalidDomain => "invalid_input",
            Self::DuplicateObject => "duplicate_object",
            Self::UnknownSystem | Self::UnknownCertificate => "unknown_object",
            Self::ObjectLimit | Self::ArtifactLimit => "resource_limit_exceeded",
            Self::OutputLimit => "output_limit_exceeded",
            Self::Serialization(_) => "malformed_request",
            Self::Meter(MeterError::Cancelled) => "cancelled",
            Self::Meter(MeterError::Budget(_)) => "budget_exhausted",
            Self::Certification(error) => certification_error_code(error),
        }
    }
}

fn certification_error_code(error: &CertificationError) -> &'static str {
    match error {
        CertificationError::Meter(MeterError::Cancelled)
        | CertificationError::Root(RootBoxError::Meter(MeterError::Cancelled)) => "cancelled",
        CertificationError::Meter(MeterError::Budget(_))
        | CertificationError::Root(RootBoxError::Meter(MeterError::Budget(_))) => {
            "budget_exhausted"
        }
        CertificationError::UniverseMismatch { .. } => "universe_mismatch",
        CertificationError::SourceMismatch => "source_mismatch",
        CertificationError::UnsupportedVersion { .. } => "unsupported_schema",
        CertificationError::Root(RootBoxError::Domain(_)) => "domain_not_certified",
        CertificationError::Root(RootBoxError::Unsupported(_)) => "unsupported_operation",
        CertificationError::Root(RootBoxError::ResourceLimit(_))
        | CertificationError::SourceLimit(_)
        | CertificationError::ByteLimit { .. }
        | CertificationError::DepthLimit { .. }
        | CertificationError::AllocationFailure => "resource_limit_exceeded",
        CertificationError::Encoding(_) | CertificationError::NonCanonical => "invalid_artifact",
        CertificationError::Isolation(RootIsolationError::Incomplete(_)) => "inconclusive",
        CertificationError::Isolation(RootIsolationError::InvalidPartition(_)) => {
            "invalid_artifact"
        }
        _ => "refused",
    }
}

fn charge<M: BudgetMeter + ?Sized>(
    meter: &mut M,
    steps: usize,
    bytes: usize,
) -> Result<(), CertificationSessionError> {
    meter.checkpoint()?;
    let mut charges = [
        (Dimension::ComputeSteps, steps.max(1) as u64),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 1),
    ];
    if bytes == 0 {
        meter.charge(Dimension::ComputeSteps, steps.max(1) as u64)?;
    } else {
        charges[1].1 = bytes as u64;
        meter.charge_batch(&charges)?;
    }
    Ok(())
}

fn integer_text(value: &str) -> bool {
    let digits = value
        .strip_prefix('-')
        .or_else(|| value.strip_prefix('+'))
        .unwrap_or(value);
    !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit())
}

fn parse_scalar(
    value: &str,
    meter: &mut impl BudgetMeter,
) -> Result<BigRational, CertificationSessionError> {
    if value.is_empty() || value.len() > MAX_SCALAR_BYTES {
        return Err(CertificationSessionError::InvalidScalar);
    }
    // The fixed length bounds integer parsing/gcd. Charge its quadratic limb
    // envelope before constructing or normalizing any arbitrary-precision value.
    let limbs = value.len().div_ceil(16).max(1);
    charge(meter, limbs * limbs + value.len(), value.len() * 16 + 128)?;
    let result = if let Some((numerator, denominator)) = value.split_once('/') {
        if !integer_text(numerator) || !integer_text(denominator) {
            return Err(CertificationSessionError::InvalidScalar);
        }
        let numerator = numerator
            .parse::<BigInt>()
            .map_err(|_| CertificationSessionError::InvalidScalar)?;
        let denominator = denominator
            .parse::<BigInt>()
            .map_err(|_| CertificationSessionError::InvalidScalar)?;
        if denominator.is_zero() {
            return Err(CertificationSessionError::InvalidScalar);
        }
        BigRational::new(numerator, denominator)
    } else if let Some((whole, fractional)) = value.split_once('.') {
        if !integer_text(whole)
            || fractional.is_empty()
            || !fractional.bytes().all(|c| c.is_ascii_digit())
        {
            return Err(CertificationSessionError::InvalidScalar);
        }
        let digits = format!("{whole}{fractional}");
        let numerator = digits
            .parse::<BigInt>()
            .map_err(|_| CertificationSessionError::InvalidScalar)?;
        BigRational::new(numerator, BigInt::from(10).pow(fractional.len() as u32))
    } else {
        if !integer_text(value) {
            return Err(CertificationSessionError::InvalidScalar);
        }
        BigRational::from_integer(
            value
                .parse::<BigInt>()
                .map_err(|_| CertificationSessionError::InvalidScalar)?,
        )
    };
    meter.checkpoint()?;
    Ok(result)
}

fn parse_domain(
    domain: &[ExactInterval],
    dimensions: usize,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<RealBall>, CertificationSessionError> {
    if dimensions == 0 || dimensions > MAX_AXES || domain.len() != dimensions {
        return Err(CertificationSessionError::InvalidDomain);
    }
    charge(
        meter,
        dimensions,
        dimensions * std::mem::size_of::<RealBall>(),
    )?;
    let mut result = Vec::with_capacity(dimensions);
    for interval in domain {
        let lower = parse_scalar(&interval.lower, meter)?;
        let upper = parse_scalar(&interval.upper, meter)?;
        if lower > upper {
            return Err(CertificationSessionError::InvalidDomain);
        }
        let limbs = (lower.numer().bits()
            + lower.denom().bits()
            + upper.numer().bits()
            + upper.denom().bits())
        .div_ceil(64) as usize;
        charge(meter, limbs.saturating_pow(2).max(1), limbs * 64 + 128)?;
        let two = BigRational::from_integer(BigInt::from(2));
        let midpoint = (&lower + &upper) / &two;
        let radius = (upper - lower) / two;
        result.push(
            RealBall::new(midpoint, radius)
                .map_err(|_| CertificationSessionError::InvalidDomain)?,
        );
    }
    Ok(result)
}

fn parse_witness(
    witness: &ExactRootWitness,
    dimensions: usize,
    meter: &mut impl BudgetMeter,
) -> Result<RootBoxWitness, CertificationSessionError> {
    if dimensions == 0
        || dimensions > MAX_AXES
        || witness.center.len() != dimensions
        || witness.preconditioner.len() != dimensions
        || witness
            .preconditioner
            .iter()
            .any(|row| row.len() != dimensions)
    {
        return Err(CertificationSessionError::InvalidDomain);
    }
    let domain = parse_domain(&witness.domain, dimensions, meter)?;
    charge(
        meter,
        dimensions * dimensions + dimensions,
        (dimensions * dimensions + dimensions) * std::mem::size_of::<BigRational>(),
    )?;
    let center = witness
        .center
        .iter()
        .map(|v| parse_scalar(v, meter))
        .collect::<Result<_, _>>()?;
    let preconditioner = witness
        .preconditioner
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| parse_scalar(v, meter))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<_, _>>()?;
    Ok(RootBoxWitness {
        domain,
        center,
        preconditioner,
        precision_digits: witness.precision_digits,
    })
}

fn render_scalar(
    value: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<String, CertificationSessionError> {
    let bits = value.numer().bits().saturating_add(value.denom().bits());
    if bits > 131_072 {
        return Err(CertificationSessionError::OutputLimit);
    }
    let bound = bits as usize + 4;
    charge(meter, bound.saturating_mul(bound).div_ceil(1024), bound)?;
    Ok(if value.denom().is_one() {
        value.numer().to_string()
    } else {
        value.to_string()
    })
}

fn render_domain(
    values: &[RealBall],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<ExactInterval>, CertificationSessionError> {
    charge(
        meter,
        values.len(),
        values.len() * std::mem::size_of::<ExactInterval>(),
    )?;
    values
        .iter()
        .map(|ball| {
            let bits = ball.midpoint().numer().bits()
                + ball.midpoint().denom().bits()
                + ball.radius().numer().bits()
                + ball.radius().denom().bits();
            if bits > 131_072 {
                return Err(CertificationSessionError::OutputLimit);
            }
            charge(
                meter,
                (bits as usize).div_ceil(32).saturating_pow(2).max(1),
                bits as usize + 128,
            )?;
            Ok(ExactInterval {
                lower: render_scalar(&ball.lower(), meter)?,
                upper: render_scalar(&ball.upper(), meter)?,
            })
        })
        .collect()
}

struct OutputCounter(usize);

impl Write for OutputCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .filter(|&n| n <= MAX_CERTIFICATION_RESPONSE_BYTES - 4096)
            .ok_or_else(|| io::Error::other("certification output limit"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn admit_response(
    response: &CertificationResponse,
    meter: &mut impl BudgetMeter,
) -> Result<(), CertificationSessionError> {
    let mut counter = OutputCounter(0);
    serde_json::to_writer(&mut counter, response)
        .map_err(|_| CertificationSessionError::OutputLimit)?;
    // Reserve the response tree and final encoded envelope before publication.
    charge(
        meter,
        counter.0,
        counter.0.saturating_mul(8).saturating_add(8192),
    )?;
    meter.checkpoint()?;
    Ok(())
}

fn root_response(
    receipt: &RootBoxReceipt,
    system: ResidualSystemId,
    certificate: RootCertificateId,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<CertificationResponse, CertificationSessionError> {
    Ok(CertificationResponse::RootBoxCertified {
        schema_version: CERTIFICATION_REQUEST_SCHEMA,
        system,
        certificate,
        evidence: "certificate_verified",
        claim: "unique_nonsingular_real_root_in_box",
        domain: render_domain(&receipt.witness().domain, meter)?,
        root_enclosure: render_domain(receipt.root_enclosure(), meter)?,
        contraction_bound: render_scalar(receipt.contraction_bound(), meter)?,
        precision_digits: receipt.witness().precision_digits,
        context_digest,
    })
}

#[derive(Default)]
pub(crate) struct CertificationState {
    systems: BTreeMap<ResidualSystemId, ResidualProblem>,
    certificates: BTreeMap<RootCertificateId, (ResidualSystemId, RootBoxReceipt)>,
    // References into immutable complete receipts avoid cloning retained source
    // forests and witnesses for each separately exportable root certificate.
    isolated_certificates: BTreeMap<RootCertificateId, (RootIsolationId, usize)>,
    isolations: BTreeMap<RootIsolationId, (ResidualSystemId, RootIsolationReceipt)>,
}

impl CertificationState {
    pub(crate) fn problem(&self, id: &ResidualSystemId) -> Option<&ResidualProblem> {
        self.systems.get(id)
    }
    pub(crate) fn certificate(&self, id: &RootCertificateId) -> Option<&RootBoxReceipt> {
        self.certificate_record(id).map(|(_, receipt)| receipt)
    }

    pub(crate) fn isolation(&self, id: &RootIsolationId) -> Option<&RootIsolationReceipt> {
        self.isolations.get(id).map(|(_, receipt)| receipt)
    }

    fn certificate_record(
        &self,
        id: &RootCertificateId,
    ) -> Option<(&ResidualSystemId, &RootBoxReceipt)> {
        if let Some((system, receipt)) = self.certificates.get(id) {
            return Some((system, receipt));
        }
        let (isolation, index) = self.isolated_certificates.get(id)?;
        let (system, receipt) = self.isolations.get(isolation)?;
        receipt.roots().get(*index).map(|root| (system, root))
    }

    fn certificate_count(&self) -> usize {
        self.certificates.len() + self.isolated_certificates.len()
    }

    fn admit_isolation_id(&self, id: &RootIsolationId) -> Result<(), CertificationSessionError> {
        if self.isolations.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.isolations.len() >= MAX_ISOLATIONS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        Ok(())
    }

    fn publish_isolation(
        &mut self,
        system: ResidualSystemId,
        isolation: RootIsolationId,
        receipt: RootIsolationReceipt,
        context_digest: [u8; 32],
        budget: &mut Budget,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_isolation_id(&isolation)?;
        let count = receipt.root_count();
        if self.certificate_count().saturating_add(count) > MAX_CERTIFICATES {
            return Err(CertificationSessionError::ObjectLimit);
        }
        // Admit both index entries and their bounded labels, response slots,
        // and every associated allocation before constructing them.
        budget.checkpoint()?;
        budget.charge_batch(&[
            (Dimension::ComputeSteps, (count + 1) as u64),
            (Dimension::MemoryBytes, ((count + 1) * 1024) as u64),
            (Dimension::AllocationCount, ((count + 1) * 8) as u64),
        ])?;
        let mut members = Vec::new();
        members
            .try_reserve_exact(count)
            .map_err(|_| CertificationError::AllocationFailure)?;
        let mut roots = Vec::new();
        roots
            .try_reserve_exact(count)
            .map_err(|_| CertificationError::AllocationFailure)?;
        for (index, root) in receipt.roots().iter().enumerate() {
            let certificate = isolation.child_certificate(index)?;
            self.admit_certificate_id(&certificate)?;
            roots.push(IsolatedRoot {
                certificate: certificate.clone(),
                domain: render_domain(&root.witness().domain, budget)?,
                root_enclosure: render_domain(root.root_enclosure(), budget)?,
                contraction_bound: render_scalar(root.contraction_bound(), budget)?,
            });
            members.push((certificate, (isolation.clone(), index)));
        }
        let response = CertificationResponse::RealRootsIsolated {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: system.clone(),
            isolation: isolation.clone(),
            evidence: "certificate_verified",
            claim: "all_real_roots_in_closed_box_are_isolated_and_nonsingular",
            domain: render_domain(receipt.domain(), budget)?,
            root_count: count,
            roots,
            leaf_count: receipt.leaf_count(),
            excluded_leaf_count: receipt.excluded_leaf_count(),
            precision_digits: receipt.precision_digits(),
            context_digest,
        };
        admit_response(&response, budget)?;
        // All fallible validation, preparation and accounting is complete.
        // Both stores publish together; no partially accepted root list escapes.
        self.isolations.insert(isolation, (system, receipt));
        for (certificate, member) in members {
            self.isolated_certificates.insert(certificate, member);
        }
        Ok(response)
    }

    fn admit_certificate_id(
        &self,
        id: &RootCertificateId,
    ) -> Result<(), CertificationSessionError> {
        if self.certificates.contains_key(id) || self.isolated_certificates.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.certificate_count() >= MAX_CERTIFICATES {
            return Err(CertificationSessionError::ObjectLimit);
        }
        Ok(())
    }

    pub(crate) fn handle(
        &mut self,
        request: CertificationRequest,
        context_digest: [u8; 32],
        budget: &mut Budget,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        if request.schema_version() != CERTIFICATION_REQUEST_SCHEMA {
            return Err(CertificationSessionError::UnsupportedVersion(
                request.schema_version(),
            ));
        }
        match request {
            CertificationRequest::RegisterResidualSystem {
                system,
                residuals,
                variables,
                ..
            } => {
                if self.systems.contains_key(&system) {
                    return Err(CertificationSessionError::DuplicateObject);
                }
                if self.systems.len() >= MAX_SYSTEMS {
                    return Err(CertificationSessionError::ObjectLimit);
                }
                let problem = ResidualProblem::parse_sources(&residuals, &variables, budget)?;
                charge(
                    budget,
                    variables.len(),
                    variables.iter().map(|v| v.len() + 128).sum(),
                )?;
                let response = CertificationResponse::ResidualSystemRegistered {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    rows: problem.residuals().len(),
                    variables: problem.variables().to_vec(),
                    system: system.clone(),
                    context_digest,
                };
                admit_response(&response, budget)?;
                self.systems.insert(system, problem);
                Ok(response)
            }
            CertificationRequest::EncloseResidualJacobian {
                system,
                domain,
                precision_digits,
                ..
            } => {
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = problem.enclose(
                    &domain,
                    precision_digits,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                let response = CertificationResponse::ResidualJacobianEnclosed {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system,
                    evidence: "certified_numeric",
                    claim: "whole_box_c1_residual_jacobian_enclosure",
                    domain: render_domain(receipt.domain(), budget)?,
                    residuals: render_domain(receipt.residual_values(), budget)?,
                    jacobian: receipt
                        .jacobian()
                        .iter()
                        .map(|row| render_domain(row, budget))
                        .collect::<Result<_, _>>()?,
                    precision_digits: receipt.precision_digits(),
                    context_digest,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::CertifyRootBox {
                system,
                certificate,
                witness,
                ..
            } => {
                self.admit_certificate_id(&certificate)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let witness = parse_witness(&witness, problem.variables().len(), budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = problem.verify_root(
                    &witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                let response = root_response(
                    &receipt,
                    system.clone(),
                    certificate.clone(),
                    context_digest,
                    budget,
                )?;
                admit_response(&response, budget)?;
                self.certificates.insert(certificate, (system, receipt));
                Ok(response)
            }
            CertificationRequest::IsolateRealRootsInBox {
                system,
                isolation,
                domain,
                precision_digits,
                max_leaves,
                ..
            } => {
                self.admit_isolation_id(&isolation)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let witness = problem.propose_isolation(
                    &domain,
                    fsym_calculus::root_isolation::RootIsolationOptions {
                        precision_digits,
                        max_leaves,
                    },
                    budget,
                )?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = problem.verify_isolation(
                    &domain,
                    &witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_isolation(system, isolation, receipt, context_digest, budget)
            }
            CertificationRequest::ExportRootCertificate { certificate, .. } => {
                let (system, receipt) = self
                    .certificate_record(&certificate)
                    .ok_or(CertificationSessionError::UnknownCertificate)?;
                let bytes = export_root_certificate(receipt, context_digest, budget)?;
                if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let artifact = String::from_utf8(bytes)
                    .map_err(|e| CertificationSessionError::Serialization(e.to_string()))?;
                let response = CertificationResponse::RootCertificateExported {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system: system.clone(),
                    certificate,
                    artifact,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::ReplayRootCertificate {
                system,
                certificate,
                artifact,
                ..
            } => {
                self.admit_certificate_id(&certificate)?;
                if artifact.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_root_certificate(
                    artifact.as_bytes(),
                    problem,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                let response = root_response(
                    &receipt,
                    system.clone(),
                    certificate.clone(),
                    context_digest,
                    budget,
                )?;
                admit_response(&response, budget)?;
                self.certificates.insert(certificate, (system, receipt));
                Ok(response)
            }
        }
    }
}
