//! Session transport for independently checked, original-source real systems.
//!
//! Object labels are typed session-local references, not mathematical content
//! identities. Registration freezes parsed native expressions and ordered axes;
//! workspace binding changes never rewrite that captured problem. Verification
//! uses the protected pool of the same cumulative session ledger.

use crate::artifact_stream::{
    ArtifactChunk, ArtifactExport, ArtifactImport, ArtifactKind, ArtifactManifest,
    ArtifactProgress, ArtifactTransferError, ArtifactTransferId,
};
use crate::certification::{
    CertificationError, ResidualProblem, export_root_certificate, replay_root_certificate,
};
use crate::durable_search::{
    DurableSearchError, RootSearchCheckpointStore, StoredRootSearchCheckpoint,
};
use crate::integration::{
    IntegrationError, export_integral_certificate, replay_integral_certificate,
};
use crate::isolation_transport::{export_root_isolation, replay_root_isolation};
use crate::objective::ObjectiveProblem;
use crate::objective_transport::{export_stationary_certificate, replay_stationary_certificate};
use crate::optimization::{OptimizationError, export_box_optimization, replay_box_optimization};
use crate::repairable_search::{
    RepairableRootSearchCheckpoint, RepairableRootSearchCheckpointStore, RepairableSearchError,
    RootSearchRepairReport,
};
use crate::search_checkpoint::{export_root_search_checkpoint, restore_root_search_checkpoint};
use fsym_budget::{Budget, BudgetError, BudgetMeter, Dimension, MeterError, VerifierLease};
use fsym_calculus::box_optimization::BoxOptimizationOptions;
use fsym_calculus::quadrature::QuadratureOptions;
use fsym_calculus::root_isolation::{
    MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationOptions, RootIsolationProgress,
    RootIsolationSearch,
};
use fsym_core::{
    BigInt, BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationReceipt,
    IntegralEnclosureReceipt, QuadratureError, RealBall, RootBoxError, RootBoxReceipt,
    RootBoxWitness, RootIsolationError, RootIsolationReceipt, StationaryPointKind,
    StationaryPointReceipt, Symbol,
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
const MAX_INTEGRALS: usize = 32;
const MAX_OBJECTIVES: usize = 32;
const MAX_STATIONARY_POINTS: usize = 128;
const MAX_BOX_OPTIMIZATIONS: usize = 32;
/// Imports and exports share this resident bound; releasing a transfer does not
/// refund any work or allocation already charged to the session ledger.
pub const MAX_SESSION_ARTIFACT_TRANSFERS: usize = 8;
/// Pending and complete candidate searches remain resident until released.
/// This bound is independent of accepted complete-isolation receipts.
pub const MAX_SESSION_ROOT_SEARCHES: usize = 8;
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

/// A resumable candidate search, distinct from an accepted root isolation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct RootIsolationSearchId(Box<str>);

impl RootIsolationSearchId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "search:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        // A short label cannot retain an arbitrarily large caller capacity.
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for RootIsolationSearchId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// A source-bound, independently checked proper real integral enclosure.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct IntegralCertificateId(Box<str>);

impl IntegralCertificateId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "integral:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for IntegralCertificateId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// One resident immutable scalar objective, scoped to this session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ObjectiveId(Box<str>);

impl ObjectiveId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "objective:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ObjectiveId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// An accepted local stationary-point claim, never a residual-root claim.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct StationaryPointId(Box<str>);

impl StationaryPointId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "stationary:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for StationaryPointId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// An accepted global value enclosure and tolerance-optimal feasible point.
/// This reference grants neither exact optimizer coordinates nor uniqueness.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct BoxOptimizationId(Box<str>);

impl BoxOptimizationId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "optimization:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for BoxOptimizationId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Local classification granted by the independent interval Hessian checker.
/// `Unclassified` grants no positive or negative extremum assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StationaryPointClassification {
    StrictMinimum,
    StrictMaximum,
    Saddle,
    Unclassified,
}

impl From<StationaryPointKind> for StationaryPointClassification {
    fn from(kind: StationaryPointKind) -> Self {
        match kind {
            StationaryPointKind::StrictMinimum => Self::StrictMinimum,
            StationaryPointKind::StrictMaximum => Self::StrictMaximum,
            StationaryPointKind::Saddle => Self::Saddle,
            StationaryPointKind::Unclassified => Self::Unclassified,
        }
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

/// A typed object to export. Search checkpoints stay candidates; accepted
/// certificates retain their own replay requirements. Transfers grant no evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactSource {
    RootCertificate(RootCertificateId),
    RootIsolation(RootIsolationId),
    IntegralCertificate(IntegralCertificateId),
    RootIsolationSearch(RootIsolationSearchId),
    StationaryCertificate(StationaryPointId),
    BoxOptimization(BoxOptimizationId),
}

/// Bounded certification operations accepted by the native session and NDJSON
/// command. Every payload carries an explicit schema version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", deny_unknown_fields)]
pub enum CertificationRequest {
    RegisterObjective {
        schema_version: u32,
        objective: ObjectiveId,
        source: String,
        variables: Vec<String>,
    },
    EncloseObjective {
        schema_version: u32,
        objective: ObjectiveId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
    },
    CertifyStationaryPoint {
        schema_version: u32,
        objective: ObjectiveId,
        certificate: StationaryPointId,
        witness: ExactRootWitness,
    },
    ExportStationaryCertificate {
        schema_version: u32,
        certificate: StationaryPointId,
    },
    ReplayStationaryCertificate {
        schema_version: u32,
        objective: ObjectiveId,
        certificate: StationaryPointId,
        domain: Vec<ExactInterval>,
        artifact: String,
    },
    ReplayStationaryCertificateFromArtifact {
        schema_version: u32,
        objective: ObjectiveId,
        certificate: StationaryPointId,
        domain: Vec<ExactInterval>,
        transfer: ArtifactTransferId,
    },
    OptimizeBox {
        schema_version: u32,
        objective: ObjectiveId,
        optimization: BoxOptimizationId,
        domain: Vec<ExactInterval>,
        extremum: BoxExtremumKind,
        absolute_tolerance: String,
        precision_digits: u32,
        max_leaves: usize,
    },
    ExportBoxOptimization {
        schema_version: u32,
        optimization: BoxOptimizationId,
    },
    ReplayBoxOptimization {
        schema_version: u32,
        objective: ObjectiveId,
        optimization: BoxOptimizationId,
        domain: Vec<ExactInterval>,
        extremum: BoxExtremumKind,
        absolute_tolerance: String,
        artifact: String,
    },
    ReplayBoxOptimizationFromArtifact {
        schema_version: u32,
        objective: ObjectiveId,
        optimization: BoxOptimizationId,
        domain: Vec<ExactInterval>,
        extremum: BoxExtremumKind,
        absolute_tolerance: String,
        transfer: ArtifactTransferId,
    },
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
    BeginRootIsolationSearch {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
        max_leaves: usize,
    },
    AdvanceRootIsolationSearch {
        schema_version: u32,
        search: RootIsolationSearchId,
        max_cells: usize,
    },
    InspectRootIsolationSearch {
        schema_version: u32,
        search: RootIsolationSearchId,
    },
    FinishRootIsolationSearch {
        schema_version: u32,
        search: RootIsolationSearchId,
        isolation: RootIsolationId,
    },
    ReleaseRootIsolationSearch {
        schema_version: u32,
        search: RootIsolationSearchId,
    },
    ExportRootIsolationSearch {
        schema_version: u32,
        search: RootIsolationSearchId,
    },
    SaveRootIsolationSearch {
        schema_version: u32,
        search: RootIsolationSearchId,
    },
    SaveRepairableRootIsolationSearch {
        schema_version: u32,
        search: RootIsolationSearchId,
    },
    LoadRepairableRootIsolationSearch {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
        max_leaves: usize,
        checkpoint: RepairableRootSearchCheckpoint,
    },
    LoadRootIsolationSearch {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
        max_leaves: usize,
        checkpoint: StoredRootSearchCheckpoint,
    },
    RestoreRootIsolationSearch {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
        max_leaves: usize,
        artifact: String,
    },
    ExportRootIsolation {
        schema_version: u32,
        isolation: RootIsolationId,
    },
    ReplayRootIsolation {
        schema_version: u32,
        system: ResidualSystemId,
        isolation: RootIsolationId,
        domain: Vec<ExactInterval>,
        artifact: String,
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
    CertifyIntegral {
        schema_version: u32,
        system: ResidualSystemId,
        integral: IntegralCertificateId,
        lower: String,
        upper: String,
        absolute_tolerance: String,
        precision_digits: u32,
        max_intervals: usize,
    },
    ExportIntegralCertificate {
        schema_version: u32,
        integral: IntegralCertificateId,
    },
    ReplayIntegralCertificate {
        schema_version: u32,
        system: ResidualSystemId,
        integral: IntegralCertificateId,
        lower: String,
        upper: String,
        absolute_tolerance: String,
        artifact: String,
    },
    BeginArtifactExport {
        schema_version: u32,
        transfer: ArtifactTransferId,
        source: ArtifactSource,
    },
    BeginArtifactImport {
        schema_version: u32,
        transfer: ArtifactTransferId,
        manifest: ArtifactManifest,
    },
    ReadArtifactChunk {
        schema_version: u32,
        transfer: ArtifactTransferId,
        index: usize,
    },
    AppendArtifactChunk {
        schema_version: u32,
        transfer: ArtifactTransferId,
        chunk: ArtifactChunk,
    },
    ReleaseArtifactTransfer {
        schema_version: u32,
        transfer: ArtifactTransferId,
    },
    ReplayRootCertificateFromArtifact {
        schema_version: u32,
        system: ResidualSystemId,
        certificate: RootCertificateId,
        transfer: ArtifactTransferId,
    },
    ReplayRootIsolationFromArtifact {
        schema_version: u32,
        system: ResidualSystemId,
        isolation: RootIsolationId,
        domain: Vec<ExactInterval>,
        transfer: ArtifactTransferId,
    },
    ReplayIntegralCertificateFromArtifact {
        schema_version: u32,
        system: ResidualSystemId,
        integral: IntegralCertificateId,
        lower: String,
        upper: String,
        absolute_tolerance: String,
        transfer: ArtifactTransferId,
    },
    RestoreRootIsolationSearchFromArtifact {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        domain: Vec<ExactInterval>,
        precision_digits: u32,
        max_leaves: usize,
        transfer: ArtifactTransferId,
    },
}

impl CertificationRequest {
    fn schema_version(&self) -> u32 {
        match self {
            Self::RegisterObjective { schema_version, .. }
            | Self::EncloseObjective { schema_version, .. }
            | Self::CertifyStationaryPoint { schema_version, .. }
            | Self::ExportStationaryCertificate { schema_version, .. }
            | Self::ReplayStationaryCertificate { schema_version, .. }
            | Self::ReplayStationaryCertificateFromArtifact { schema_version, .. }
            | Self::OptimizeBox { schema_version, .. }
            | Self::ExportBoxOptimization { schema_version, .. }
            | Self::ReplayBoxOptimization { schema_version, .. }
            | Self::ReplayBoxOptimizationFromArtifact { schema_version, .. }
            | Self::RegisterResidualSystem { schema_version, .. }
            | Self::EncloseResidualJacobian { schema_version, .. }
            | Self::CertifyRootBox { schema_version, .. }
            | Self::IsolateRealRootsInBox { schema_version, .. }
            | Self::BeginRootIsolationSearch { schema_version, .. }
            | Self::AdvanceRootIsolationSearch { schema_version, .. }
            | Self::InspectRootIsolationSearch { schema_version, .. }
            | Self::FinishRootIsolationSearch { schema_version, .. }
            | Self::ReleaseRootIsolationSearch { schema_version, .. }
            | Self::ExportRootIsolationSearch { schema_version, .. }
            | Self::SaveRootIsolationSearch { schema_version, .. }
            | Self::SaveRepairableRootIsolationSearch { schema_version, .. }
            | Self::LoadRepairableRootIsolationSearch { schema_version, .. }
            | Self::LoadRootIsolationSearch { schema_version, .. }
            | Self::RestoreRootIsolationSearch { schema_version, .. }
            | Self::RestoreRootIsolationSearchFromArtifact { schema_version, .. }
            | Self::ExportRootIsolation { schema_version, .. }
            | Self::ReplayRootIsolation { schema_version, .. }
            | Self::ExportRootCertificate { schema_version, .. }
            | Self::ReplayRootCertificate { schema_version, .. }
            | Self::BeginArtifactExport { schema_version, .. }
            | Self::BeginArtifactImport { schema_version, .. }
            | Self::ReadArtifactChunk { schema_version, .. }
            | Self::AppendArtifactChunk { schema_version, .. }
            | Self::ReleaseArtifactTransfer { schema_version, .. }
            | Self::ReplayRootCertificateFromArtifact { schema_version, .. }
            | Self::ReplayRootIsolationFromArtifact { schema_version, .. }
            | Self::CertifyIntegral { schema_version, .. }
            | Self::ExportIntegralCertificate { schema_version, .. }
            | Self::ReplayIntegralCertificate { schema_version, .. }
            | Self::ReplayIntegralCertificateFromArtifact { schema_version, .. } => *schema_version,
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
    ObjectiveRegistered {
        schema_version: u32,
        objective: ObjectiveId,
        variables: Vec<Symbol>,
        context_digest: [u8; 32],
    },
    ObjectiveEnclosed {
        schema_version: u32,
        objective: ObjectiveId,
        evidence: &'static str,
        claim: &'static str,
        domain: Vec<ExactInterval>,
        value: ExactInterval,
        gradient: Vec<ExactInterval>,
        hessian: Vec<Vec<ExactInterval>>,
        precision_digits: u32,
        context_digest: [u8; 32],
    },
    StationaryPointCertified {
        schema_version: u32,
        objective: ObjectiveId,
        certificate: StationaryPointId,
        evidence: &'static str,
        claim: &'static str,
        classification: StationaryPointClassification,
        domain: Vec<ExactInterval>,
        stationary_enclosure: Vec<ExactInterval>,
        objective_enclosure: ExactInterval,
        gradient_enclosure: Vec<ExactInterval>,
        hessian_enclosure: Vec<Vec<ExactInterval>>,
        contraction_bound: String,
        precision_digits: u32,
        context_digest: [u8; 32],
    },
    StationaryCertificateExported {
        schema_version: u32,
        objective: ObjectiveId,
        certificate: StationaryPointId,
        artifact: String,
    },
    BoxOptimized {
        schema_version: u32,
        objective: ObjectiveId,
        optimization: BoxOptimizationId,
        evidence: &'static str,
        claim: &'static str,
        extremum: BoxExtremumKind,
        domain: Vec<ExactInterval>,
        point: Vec<String>,
        value: ExactInterval,
        point_value: ExactInterval,
        absolute_tolerance: String,
        gap: String,
        leaf_count: usize,
        precision_digits: u32,
        context_digest: [u8; 32],
    },
    BoxOptimizationExported {
        schema_version: u32,
        objective: ObjectiveId,
        optimization: BoxOptimizationId,
        artifact: String,
    },
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
    RootIsolationSearchStarted {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        evidence: &'static str,
        progress: RootIsolationProgress,
        context_digest: [u8; 32],
    },
    RootIsolationSearchProgress {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        evidence: &'static str,
        progress: RootIsolationProgress,
        context_digest: [u8; 32],
    },
    RootIsolationSearchReleased {
        schema_version: u32,
        search: RootIsolationSearchId,
    },
    RootIsolationSearchExported {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        evidence: &'static str,
        artifact: String,
    },
    RootIsolationSearchSaved {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        evidence: &'static str,
        checkpoint: StoredRootSearchCheckpoint,
    },
    RootIsolationSearchSavedWithRepair {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        evidence: &'static str,
        checkpoint: RepairableRootSearchCheckpoint,
    },
    RootIsolationSearchRecovered {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        evidence: &'static str,
        progress: RootIsolationProgress,
        context_digest: [u8; 32],
        repair: RootSearchRepairReport,
    },
    RootIsolationSearchRestored {
        schema_version: u32,
        system: ResidualSystemId,
        search: RootIsolationSearchId,
        evidence: &'static str,
        progress: RootIsolationProgress,
        context_digest: [u8; 32],
    },
    RootCertificateExported {
        schema_version: u32,
        system: ResidualSystemId,
        certificate: RootCertificateId,
        artifact: String,
    },
    RootIsolationExported {
        schema_version: u32,
        system: ResidualSystemId,
        isolation: RootIsolationId,
        artifact: String,
    },
    IntegralCertified {
        schema_version: u32,
        system: ResidualSystemId,
        integral: IntegralCertificateId,
        evidence: &'static str,
        claim: &'static str,
        lower: String,
        upper: String,
        absolute_tolerance: String,
        enclosure: ExactInterval,
        interval_count: usize,
        precision_digits: u32,
        context_digest: [u8; 32],
    },
    IntegralCertificateExported {
        schema_version: u32,
        system: ResidualSystemId,
        integral: IntegralCertificateId,
        artifact: String,
    },
    ArtifactExportStarted {
        schema_version: u32,
        transfer: ArtifactTransferId,
        manifest: ArtifactManifest,
    },
    ArtifactImportStarted {
        schema_version: u32,
        transfer: ArtifactTransferId,
        manifest: ArtifactManifest,
        progress: ArtifactProgress,
    },
    ArtifactChunkRead {
        schema_version: u32,
        transfer: ArtifactTransferId,
        chunk: ArtifactChunk,
    },
    ArtifactChunkAppended {
        schema_version: u32,
        transfer: ArtifactTransferId,
        progress: ArtifactProgress,
    },
    ArtifactTransferReleased {
        schema_version: u32,
        transfer: ArtifactTransferId,
    },
}

#[derive(Debug, Error)]
pub enum CertificationSessionError {
    #[error("unsupported certification request schema {0}")]
    UnsupportedVersion(u32),
    #[error(
        "object label must have the proper system:, certificate:, isolation:, search:, integral:, objective:, stationary: or optimization: kind and a bounded identifier"
    )]
    InvalidLabel,
    #[error("object label already exists in this session")]
    DuplicateObject,
    #[error("unknown residual system")]
    UnknownSystem,
    #[error("unknown accepted root certificate")]
    UnknownCertificate,
    #[error("unknown accepted complete root isolation")]
    UnknownIsolation,
    #[error("unknown root isolation search")]
    UnknownSearch,
    #[error("unknown accepted integral certificate")]
    UnknownIntegral,
    #[error("unknown scalar objective")]
    UnknownObjective,
    #[error("unknown accepted stationary-point certificate")]
    UnknownStationaryPoint,
    #[error("unknown accepted global box optimization")]
    UnknownBoxOptimization,
    #[error("root isolation search still has pending cells")]
    SearchNotReady,
    #[error("root search work quantum is outside the supported range")]
    InvalidSearchQuantum,
    #[error("unknown artifact transfer")]
    UnknownTransfer,
    #[error("artifact transfer has the wrong import/export direction")]
    WrongTransferMode,
    #[error("resident certification object limit reached")]
    ObjectLimit,
    #[error(
        "exact scalar must be a bounded integer, decimal, or integer fraction with nonzero denominator"
    )]
    InvalidScalar,
    #[error("box endpoints are reversed or dimensions are inconsistent")]
    InvalidDomain,
    #[error(
        "certificate exceeds the bounded single-frame session transport; use chunked artifact transfer"
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
    #[error("integral certification refused: {0}")]
    Integration(#[from] IntegrationError),
    #[error("global box optimization refused: {0}")]
    Optimization(#[from] OptimizationError),
    #[error("artifact transfer refused: {0}")]
    Transfer(#[from] ArtifactTransferError),
    #[error("durable root-search operation refused: {0}")]
    DurableSearch(#[from] DurableSearchError),
    #[error("repairable root-search operation refused: {0}")]
    RepairableSearch(#[from] RepairableSearchError),
}

impl CertificationSessionError {
    /// Stable machine category; mathematical refusal is distinct from an
    /// unsupported operation, exhausted resources, malformed input or replay.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedVersion(_) => "unsupported_schema",
            Self::InvalidLabel
            | Self::InvalidScalar
            | Self::InvalidDomain
            | Self::InvalidSearchQuantum => "invalid_input",
            Self::DuplicateObject => "duplicate_object",
            Self::UnknownSystem
            | Self::UnknownCertificate
            | Self::UnknownIsolation
            | Self::UnknownSearch
            | Self::UnknownIntegral
            | Self::UnknownObjective
            | Self::UnknownStationaryPoint
            | Self::UnknownBoxOptimization
            | Self::UnknownTransfer => "unknown_object",
            Self::SearchNotReady => "search_not_ready",
            Self::WrongTransferMode => "wrong_transfer_mode",
            Self::ObjectLimit | Self::ArtifactLimit => "resource_limit_exceeded",
            Self::OutputLimit => "output_limit_exceeded",
            Self::Serialization(_) => "malformed_request",
            Self::Meter(MeterError::Cancelled) => "cancelled",
            Self::Meter(MeterError::Budget(_)) => "budget_exhausted",
            Self::Certification(error) => certification_error_code(error),
            Self::Integration(error) => integration_error_code(error),
            Self::Optimization(error) => optimization_error_code(error),
            Self::Transfer(error) => transfer_error_code(error),
            Self::DurableSearch(error) => durable_search_error_code(error),
            Self::RepairableSearch(error) => repairable_search_error_code(error),
        }
    }
}

fn optimization_error_code(error: &OptimizationError) -> &'static str {
    match error {
        OptimizationError::Certification(error) => certification_error_code(error),
        OptimizationError::StatementMismatch => "statement_mismatch",
        OptimizationError::Optimization(BoxOptimizationError::Verification(error)) => match error {
            RootBoxError::Meter(MeterError::Cancelled) => "cancelled",
            RootBoxError::Meter(MeterError::Budget(_)) => "budget_exhausted",
            RootBoxError::Domain(_) => "domain_not_certified",
            RootBoxError::Unsupported(_) => "unsupported_operation",
            RootBoxError::ResourceLimit(_) => "resource_limit_exceeded",
            _ => "refused",
        },
        OptimizationError::Optimization(BoxOptimizationError::InvalidTolerance) => "invalid_input",
        OptimizationError::Optimization(BoxOptimizationError::InvalidPartition(_)) => {
            "invalid_artifact"
        }
        OptimizationError::Optimization(BoxOptimizationError::AccuracyNotReached) => "inconclusive",
        OptimizationError::Optimization(BoxOptimizationError::InconsistentEnclosure) => "refused",
    }
}

fn repairable_search_error_code(error: &RepairableSearchError) -> &'static str {
    match error {
        RepairableSearchError::InvalidLength => "resource_limit_exceeded",
        RepairableSearchError::InvalidManifest(_)
        | RepairableSearchError::RecordMismatch
        | RepairableSearchError::Json(_) => "invalid_artifact",
        RepairableSearchError::Meter(MeterError::Cancelled) => "cancelled",
        RepairableSearchError::Meter(MeterError::Budget(_)) => "budget_exhausted",
        RepairableSearchError::Certification(error) => certification_error_code(error),
        RepairableSearchError::Repair(_) => "repair_failed",
        RepairableSearchError::Storage(fsym_durable::DurableError::Absent) => "checkpoint_absent",
        RepairableSearchError::Storage(fsym_durable::DurableError::RecordTooLarge { .. }) => {
            "resource_limit_exceeded"
        }
        RepairableSearchError::Storage(fsym_durable::DurableError::Backend(_))
        | RepairableSearchError::Cleanup { .. } => "storage_error",
        RepairableSearchError::Storage(_) => "invalid_artifact",
    }
}

fn durable_search_error_code(error: &DurableSearchError) -> &'static str {
    match error {
        DurableSearchError::Disabled => "persistence_disabled",
        DurableSearchError::InvalidLength => "invalid_input",
        DurableSearchError::RecordMismatch => "invalid_artifact",
        DurableSearchError::Meter(MeterError::Cancelled) => "cancelled",
        DurableSearchError::Meter(MeterError::Budget(_)) => "budget_exhausted",
        DurableSearchError::Certification(error) => certification_error_code(error),
        DurableSearchError::Storage(fsym_durable::DurableError::Absent) => "checkpoint_absent",
        DurableSearchError::Storage(fsym_durable::DurableError::RecordTooLarge { .. }) => {
            "resource_limit_exceeded"
        }
        DurableSearchError::Storage(fsym_durable::DurableError::Backend(_))
        | DurableSearchError::Cleanup { .. } => "storage_error",
        DurableSearchError::Storage(_) => "invalid_artifact",
    }
}

fn integration_error_code(error: &IntegrationError) -> &'static str {
    match error {
        IntegrationError::Certification(error) => certification_error_code(error),
        IntegrationError::ProblemDimension => "invalid_input",
        IntegrationError::StatementMismatch => "statement_mismatch",
        IntegrationError::Quadrature(QuadratureError::Enclosure(error)) => {
            // Preserve cancellation, budgets and source-domain failure categories.
            match error {
                RootBoxError::Meter(MeterError::Cancelled) => "cancelled",
                RootBoxError::Meter(MeterError::Budget(_)) => "budget_exhausted",
                RootBoxError::Domain(_) => "domain_not_certified",
                RootBoxError::Unsupported(_) => "unsupported_operation",
                RootBoxError::ResourceLimit(_) => "resource_limit_exceeded",
                _ => "refused",
            }
        }
        IntegrationError::Quadrature(QuadratureError::InvalidTolerance) => "invalid_input",
        IntegrationError::Quadrature(QuadratureError::InvalidPartition(_)) => "invalid_artifact",
        IntegrationError::Quadrature(QuadratureError::ResourceLimit(_)) => {
            "resource_limit_exceeded"
        }
        IntegrationError::Quadrature(QuadratureError::AccuracyNotReached) => "inconclusive",
        IntegrationError::Quadrature(QuadratureError::InconsistentEnclosure) => "refused",
    }
}

fn transfer_error_code(error: &ArtifactTransferError) -> &'static str {
    match error {
        ArtifactTransferError::Meter(MeterError::Cancelled) => "cancelled",
        ArtifactTransferError::Meter(MeterError::Budget(_)) => "budget_exhausted",
        ArtifactTransferError::UnsupportedVersion { .. } => "unsupported_schema",
        ArtifactTransferError::ByteLimit { .. } | ArtifactTransferError::AllocationFailure => {
            "resource_limit_exceeded"
        }
        ArtifactTransferError::Incomplete { .. } => "incomplete_transfer",
        ArtifactTransferError::OutOfOrder { .. } => "out_of_order_chunk",
        ArtifactTransferError::ConflictingChunk => "conflicting_chunk",
        ArtifactTransferError::KindMismatch => "artifact_kind_mismatch",
        ArtifactTransferError::ChunkDigestMismatch | ArtifactTransferError::FinalDigestMismatch => {
            "invalid_artifact"
        }
        ArtifactTransferError::InvalidLabel
        | ArtifactTransferError::InvalidContentId
        | ArtifactTransferError::EmptyArtifact
        | ArtifactTransferError::InvalidIndex { .. }
        | ArtifactTransferError::ChunkLength { .. }
        | ArtifactTransferError::InvalidHex => "invalid_input",
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
        CertificationError::DomainMismatch => "domain_mismatch",
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

fn render_ball(
    value: &RealBall,
    meter: &mut impl BudgetMeter,
) -> Result<ExactInterval, CertificationSessionError> {
    render_domain(std::slice::from_ref(value), meter)?
        .pop()
        .ok_or(CertificationSessionError::OutputLimit)
}

fn render_hessian(
    hessian: &[Vec<RealBall>],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<Vec<ExactInterval>>, CertificationSessionError> {
    charge(
        meter,
        hessian.len(),
        hessian.len() * std::mem::size_of::<Vec<ExactInterval>>(),
    )?;
    hessian
        .iter()
        .map(|row| render_domain(row, meter))
        .collect()
}

fn stationary_response(
    receipt: &StationaryPointReceipt,
    objective: ObjectiveId,
    certificate: StationaryPointId,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<CertificationResponse, CertificationSessionError> {
    Ok(CertificationResponse::StationaryPointCertified {
        schema_version: CERTIFICATION_REQUEST_SCHEMA,
        objective,
        certificate,
        evidence: "certificate_verified",
        claim: "unique_nonsingular_stationary_point_in_box",
        classification: receipt.kind().into(),
        domain: render_domain(&receipt.witness().domain, meter)?,
        stationary_enclosure: render_domain(receipt.stationary_enclosure(), meter)?,
        objective_enclosure: render_ball(receipt.objective_enclosure(), meter)?,
        gradient_enclosure: render_domain(receipt.gradient_enclosure(), meter)?,
        hessian_enclosure: render_hessian(receipt.hessian_enclosure(), meter)?,
        contraction_bound: render_scalar(receipt.contraction_bound(), meter)?,
        precision_digits: receipt.witness().precision_digits,
        context_digest,
    })
}

fn optimization_response(
    receipt: &BoxOptimizationReceipt,
    objective: ObjectiveId,
    optimization: BoxOptimizationId,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<CertificationResponse, CertificationSessionError> {
    charge(
        meter,
        receipt.point().len(),
        receipt.point().len() * std::mem::size_of::<String>(),
    )?;
    Ok(CertificationResponse::BoxOptimized {
        schema_version: CERTIFICATION_REQUEST_SCHEMA,
        objective,
        optimization,
        evidence: "certified_numeric",
        claim: "global_box_optimum_with_absolute_value_gap_bound",
        extremum: receipt.kind(),
        domain: render_domain(receipt.domain(), meter)?,
        point: receipt
            .point()
            .iter()
            .map(|value| render_scalar(value, meter))
            .collect::<Result<_, _>>()?,
        value: render_ball(receipt.value(), meter)?,
        point_value: render_ball(receipt.point_value(), meter)?,
        absolute_tolerance: render_scalar(receipt.absolute_tolerance(), meter)?,
        gap: render_scalar(receipt.gap(), meter)?,
        leaf_count: receipt.leaf_count(),
        precision_digits: receipt.precision_digits(),
        context_digest,
    })
}

enum ResidentArtifact {
    Export(ArtifactExport),
    Import(ArtifactImport),
}

struct ResidentRootSearch {
    system: ResidualSystemId,
    context_digest: [u8; 32],
    search: RootIsolationSearch,
}

struct ResidentIntegral {
    system: ResidualSystemId,
    context_digest: [u8; 32],
    receipt: IntegralEnclosureReceipt,
}

struct ResidentStationaryPoint {
    objective: ObjectiveId,
    context_digest: [u8; 32],
    receipt: StationaryPointReceipt,
}

struct ResidentBoxOptimization {
    objective: ObjectiveId,
    context_digest: [u8; 32],
    receipt: BoxOptimizationReceipt,
}

impl ResidentRootSearch {
    fn response(
        &self,
        id: RootIsolationSearchId,
        progress: RootIsolationProgress,
    ) -> CertificationResponse {
        CertificationResponse::RootIsolationSearchProgress {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: self.system.clone(),
            search: id,
            evidence: "heuristic_candidate",
            progress,
            context_digest: self.context_digest,
        }
    }
}

#[derive(Default)]
pub(crate) struct CertificationState {
    objectives: BTreeMap<ObjectiveId, ObjectiveProblem>,
    stationary_points: BTreeMap<StationaryPointId, ResidentStationaryPoint>,
    optimizations: BTreeMap<BoxOptimizationId, ResidentBoxOptimization>,
    checkpoint_store: Option<RootSearchCheckpointStore>,
    repairable_checkpoint_store: Option<RepairableRootSearchCheckpointStore>,
    systems: BTreeMap<ResidualSystemId, ResidualProblem>,
    certificates: BTreeMap<RootCertificateId, (ResidualSystemId, RootBoxReceipt)>,
    // References into immutable complete receipts avoid cloning retained source
    // forests and witnesses for each separately exportable root certificate.
    isolated_certificates: BTreeMap<RootCertificateId, (RootIsolationId, usize)>,
    isolations: BTreeMap<RootIsolationId, (ResidualSystemId, RootIsolationReceipt)>,
    // These byte containers never participate in accepted-object lookup. An
    // import must pass full independent replay before any certificate exists.
    transfers: BTreeMap<ArtifactTransferId, ResidentArtifact>,
    // Candidate frontiers are not accepted-object stores. Their immutable
    // source/domain and receiving context survive every advance and retry.
    searches: BTreeMap<RootIsolationSearchId, ResidentRootSearch>,
    integrals: BTreeMap<IntegralCertificateId, ResidentIntegral>,
}

impl CertificationState {
    pub(crate) fn box_optimization(
        &self,
        id: &BoxOptimizationId,
    ) -> Option<&BoxOptimizationReceipt> {
        self.optimizations.get(id).map(|slot| &slot.receipt)
    }

    fn admit_optimization_id(
        &self,
        id: &BoxOptimizationId,
        meter: &mut impl BudgetMeter,
    ) -> Result<(), CertificationSessionError> {
        if self.optimizations.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.optimizations.len() >= MAX_BOX_OPTIMIZATIONS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        charge(meter, 1, 4096)?;
        Ok(())
    }

    fn publish_optimization(
        &mut self,
        objective: ObjectiveId,
        optimization: BoxOptimizationId,
        receipt: BoxOptimizationReceipt,
        context_digest: [u8; 32],
        budget: &mut Budget,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        let response = optimization_response(
            &receipt,
            objective.clone(),
            optimization.clone(),
            context_digest,
            budget,
        )?;
        // Full output admission and the last cancellation check precede the
        // only write. A valid certificate with a refused response stays unpublished.
        admit_response(&response, budget)?;
        self.optimizations.insert(
            optimization,
            ResidentBoxOptimization {
                objective,
                context_digest,
                receipt,
            },
        );
        Ok(response)
    }

    pub(crate) fn objective(&self, id: &ObjectiveId) -> Option<&ObjectiveProblem> {
        self.objectives.get(id)
    }

    pub(crate) fn stationary_point(
        &self,
        id: &StationaryPointId,
    ) -> Option<&StationaryPointReceipt> {
        self.stationary_points.get(id).map(|slot| &slot.receipt)
    }

    fn admit_stationary_point_id(
        &self,
        id: &StationaryPointId,
        meter: &mut impl BudgetMeter,
    ) -> Result<(), CertificationSessionError> {
        if self.stationary_points.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.stationary_points.len() >= MAX_STATIONARY_POINTS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        // The checker separately charges its source, witness and receipt.
        // This reserve covers map/label ownership and publication metadata.
        charge(meter, 1, 4096)?;
        Ok(())
    }

    fn publish_stationary_point(
        &mut self,
        objective: ObjectiveId,
        certificate: StationaryPointId,
        receipt: StationaryPointReceipt,
        context_digest: [u8; 32],
        budget: &mut Budget,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        let response = stationary_response(
            &receipt,
            objective.clone(),
            certificate.clone(),
            context_digest,
            budget,
        )?;
        // No accepted handle becomes observable until complete output
        // admission, including final cancellation/resource checks, succeeds.
        admit_response(&response, budget)?;
        self.stationary_points.insert(
            certificate,
            ResidentStationaryPoint {
                objective,
                context_digest,
                receipt,
            },
        );
        Ok(response)
    }

    pub(crate) fn enable_checkpoint_store(
        &mut self,
        store: RootSearchCheckpointStore,
        repairable: RepairableRootSearchCheckpointStore,
    ) {
        self.checkpoint_store = Some(store);
        self.repairable_checkpoint_store = Some(repairable);
    }

    pub(crate) fn problem(&self, id: &ResidualSystemId) -> Option<&ResidualProblem> {
        self.systems.get(id)
    }
    pub(crate) fn certificate(&self, id: &RootCertificateId) -> Option<&RootBoxReceipt> {
        self.certificate_record(id).map(|(_, receipt)| receipt)
    }

    pub(crate) fn isolation(&self, id: &RootIsolationId) -> Option<&RootIsolationReceipt> {
        self.isolations.get(id).map(|(_, receipt)| receipt)
    }

    pub(crate) fn integral(&self, id: &IntegralCertificateId) -> Option<&IntegralEnclosureReceipt> {
        self.integrals.get(id).map(|slot| &slot.receipt)
    }

    fn admit_integral_id(
        &self,
        id: &IntegralCertificateId,
    ) -> Result<(), CertificationSessionError> {
        if self.integrals.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.integrals.len() >= MAX_INTEGRALS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        Ok(())
    }

    fn publish_integral(
        &mut self,
        system: ResidualSystemId,
        integral: IntegralCertificateId,
        receipt: IntegralEnclosureReceipt,
        context_digest: [u8; 32],
        budget: &mut Budget,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_integral_id(&integral)?;
        charge(budget, 1, 4096)?;
        let mut enclosure = render_domain(std::slice::from_ref(receipt.enclosure()), budget)?;
        let response = CertificationResponse::IntegralCertified {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: system.clone(),
            integral: integral.clone(),
            evidence: "certified_numeric",
            claim: "proper_real_definite_integral_with_absolute_enclosure_width_bound",
            lower: render_scalar(receipt.lower(), budget)?,
            upper: render_scalar(receipt.upper(), budget)?,
            absolute_tolerance: render_scalar(receipt.absolute_tolerance(), budget)?,
            enclosure: enclosure
                .pop()
                .expect("one integral enclosure was rendered"),
            interval_count: receipt.interval_count(),
            precision_digits: receipt.witness().precision_digits,
            context_digest,
        };
        admit_response(&response, budget)?;
        // Receipt, response and storage admission precede the only publication.
        self.integrals.insert(
            integral,
            ResidentIntegral {
                system,
                context_digest,
                receipt,
            },
        );
        Ok(response)
    }

    pub(crate) fn search_progress(
        &self,
        id: &RootIsolationSearchId,
    ) -> Option<RootIsolationProgress> {
        self.searches.get(id).map(|slot| slot.search.progress())
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

    fn admit_transfer_id(
        &self,
        id: &ArtifactTransferId,
        meter: &mut impl BudgetMeter,
    ) -> Result<(), CertificationSessionError> {
        if self.transfers.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.transfers.len() >= MAX_SESSION_ARTIFACT_TRANSFERS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        // Bound the resident map entry, labels and response metadata before
        // constructing either direction of a transfer.
        meter.checkpoint()?;
        meter.charge_batch(&[
            (Dimension::ComputeSteps, 1),
            (Dimension::MemoryBytes, 4096),
            (Dimension::AllocationCount, 8),
        ])?;
        Ok(())
    }

    fn admit_search_id(
        &self,
        id: &RootIsolationSearchId,
        meter: &mut impl BudgetMeter,
    ) -> Result<(), CertificationSessionError> {
        if self.searches.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.searches.len() >= MAX_SESSION_ROOT_SEARCHES {
            return Err(CertificationSessionError::ObjectLimit);
        }
        // The engine separately admits its retained source/frontier storage.
        // Reserve bounded map, label and response metadata before construction.
        meter.checkpoint()?;
        meter.charge_batch(&[
            (Dimension::ComputeSteps, 1),
            (Dimension::MemoryBytes, 4096),
            (Dimension::AllocationCount, 8),
        ])?;
        Ok(())
    }

    fn publish_restored_search(
        &mut self,
        system: ResidualSystemId,
        id: RootIsolationSearchId,
        search: RootIsolationSearch,
        context_digest: [u8; 32],
        budget: &mut Budget,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.publish_resumed_search(system, id, search, context_digest, budget, None)
    }

    fn publish_resumed_search(
        &mut self,
        system: ResidualSystemId,
        id: RootIsolationSearchId,
        search: RootIsolationSearch,
        context_digest: [u8; 32],
        budget: &mut Budget,
        repair: Option<RootSearchRepairReport>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        let response = if let Some(repair) = repair {
            CertificationResponse::RootIsolationSearchRecovered {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system: system.clone(),
                search: id.clone(),
                evidence: "heuristic_candidate",
                progress: search.progress(),
                context_digest,
                repair,
            }
        } else {
            CertificationResponse::RootIsolationSearchRestored {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system: system.clone(),
                search: id.clone(),
                evidence: "heuristic_candidate",
                progress: search.progress(),
                context_digest,
            }
        };
        admit_response(&response, budget)?;
        self.searches.insert(
            id,
            ResidentRootSearch {
                system,
                context_digest,
                search,
            },
        );
        Ok(response)
    }

    fn artifact_import(
        &self,
        id: &ArtifactTransferId,
    ) -> Result<&ArtifactImport, CertificationSessionError> {
        match self.transfers.get(id) {
            Some(ResidentArtifact::Import(import)) => Ok(import),
            Some(ResidentArtifact::Export(_)) => Err(CertificationSessionError::WrongTransferMode),
            None => Err(CertificationSessionError::UnknownTransfer),
        }
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
            CertificationRequest::RegisterObjective {
                objective,
                source,
                variables,
                ..
            } => {
                if self.objectives.contains_key(&objective) {
                    return Err(CertificationSessionError::DuplicateObject);
                }
                if self.objectives.len() >= MAX_OBJECTIVES {
                    return Err(CertificationSessionError::ObjectLimit);
                }
                charge(budget, 1, 4096)?;
                let problem = ObjectiveProblem::parse_source(&source, &variables, budget)?;
                charge(
                    budget,
                    variables.len(),
                    variables.iter().map(|v| v.len() + 128).sum(),
                )?;
                let response = CertificationResponse::ObjectiveRegistered {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    variables: problem.variables().to_vec(),
                    objective: objective.clone(),
                    context_digest,
                };
                admit_response(&response, budget)?;
                self.objectives.insert(objective, problem);
                Ok(response)
            }
            CertificationRequest::EncloseObjective {
                objective,
                domain,
                precision_digits,
                ..
            } => {
                let problem = self
                    .objectives
                    .get(&objective)
                    .ok_or(CertificationSessionError::UnknownObjective)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = problem.enclose(
                    &domain,
                    precision_digits,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                let response = CertificationResponse::ObjectiveEnclosed {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    objective,
                    evidence: "certified_numeric",
                    claim: "whole_box_c2_objective_gradient_hessian_enclosure",
                    domain: render_domain(receipt.domain(), budget)?,
                    value: render_ball(receipt.value(), budget)?,
                    gradient: render_domain(receipt.gradient(), budget)?,
                    hessian: render_hessian(receipt.hessian(), budget)?,
                    precision_digits: receipt.precision_digits(),
                    context_digest,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::CertifyStationaryPoint {
                objective,
                certificate,
                witness,
                ..
            } => {
                self.admit_stationary_point_id(&certificate, budget)?;
                let problem = self
                    .objectives
                    .get(&objective)
                    .ok_or(CertificationSessionError::UnknownObjective)?;
                let witness = parse_witness(&witness, problem.variables().len(), budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = problem.verify_stationary(
                    &witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_stationary_point(
                    objective,
                    certificate,
                    receipt,
                    context_digest,
                    budget,
                )
            }
            CertificationRequest::ExportStationaryCertificate { certificate, .. } => {
                let slot = self
                    .stationary_points
                    .get(&certificate)
                    .ok_or(CertificationSessionError::UnknownStationaryPoint)?;
                let bytes =
                    export_stationary_certificate(&slot.receipt, slot.context_digest, budget)?;
                if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let artifact = String::from_utf8(bytes)
                    .map_err(|error| CertificationSessionError::Serialization(error.to_string()))?;
                let response = CertificationResponse::StationaryCertificateExported {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    objective: slot.objective.clone(),
                    certificate,
                    artifact,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::ReplayStationaryCertificate {
                objective,
                certificate,
                domain,
                artifact,
                ..
            } => {
                self.admit_stationary_point_id(&certificate, budget)?;
                if artifact.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let problem = self
                    .objectives
                    .get(&objective)
                    .ok_or(CertificationSessionError::UnknownObjective)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_stationary_certificate(
                    artifact.as_bytes(),
                    problem,
                    &domain,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_stationary_point(
                    objective,
                    certificate,
                    receipt,
                    context_digest,
                    budget,
                )
            }
            CertificationRequest::ReplayStationaryCertificateFromArtifact {
                objective,
                certificate,
                domain,
                transfer,
                ..
            } => {
                self.admit_stationary_point_id(&certificate, budget)?;
                let problem = self
                    .objectives
                    .get(&objective)
                    .ok_or(CertificationSessionError::UnknownObjective)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let bytes = self
                    .artifact_import(&transfer)?
                    .checked_bytes(ArtifactKind::StationaryCertificate, budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_stationary_certificate(
                    bytes,
                    problem,
                    &domain,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_stationary_point(
                    objective,
                    certificate,
                    receipt,
                    context_digest,
                    budget,
                )
            }
            CertificationRequest::OptimizeBox {
                objective,
                optimization,
                domain,
                extremum,
                absolute_tolerance,
                precision_digits,
                max_leaves,
                ..
            } => {
                self.admit_optimization_id(&optimization, budget)?;
                let problem = self
                    .objectives
                    .get(&objective)
                    .ok_or(CertificationSessionError::UnknownObjective)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let tolerance = parse_scalar(&absolute_tolerance, budget)?;
                let witness = problem.propose_box_optimization(
                    &domain,
                    extremum,
                    &tolerance,
                    BoxOptimizationOptions {
                        precision_digits,
                        max_leaves,
                    },
                    budget,
                )?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = problem.verify_box_optimization(
                    &domain,
                    extremum,
                    &tolerance,
                    &witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_optimization(objective, optimization, receipt, context_digest, budget)
            }
            CertificationRequest::ExportBoxOptimization { optimization, .. } => {
                let slot = self
                    .optimizations
                    .get(&optimization)
                    .ok_or(CertificationSessionError::UnknownBoxOptimization)?;
                let bytes = export_box_optimization(&slot.receipt, slot.context_digest, budget)?;
                if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let artifact = String::from_utf8(bytes)
                    .map_err(|error| CertificationSessionError::Serialization(error.to_string()))?;
                let response = CertificationResponse::BoxOptimizationExported {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    objective: slot.objective.clone(),
                    optimization,
                    artifact,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::ReplayBoxOptimization {
                objective,
                optimization,
                domain,
                extremum,
                absolute_tolerance,
                artifact,
                ..
            } => {
                self.admit_optimization_id(&optimization, budget)?;
                if artifact.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let problem = self
                    .objectives
                    .get(&objective)
                    .ok_or(CertificationSessionError::UnknownObjective)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let tolerance = parse_scalar(&absolute_tolerance, budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_box_optimization(
                    artifact.as_bytes(),
                    problem,
                    &domain,
                    extremum,
                    &tolerance,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_optimization(objective, optimization, receipt, context_digest, budget)
            }
            CertificationRequest::ReplayBoxOptimizationFromArtifact {
                objective,
                optimization,
                domain,
                extremum,
                absolute_tolerance,
                transfer,
                ..
            } => {
                self.admit_optimization_id(&optimization, budget)?;
                let problem = self
                    .objectives
                    .get(&objective)
                    .ok_or(CertificationSessionError::UnknownObjective)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let tolerance = parse_scalar(&absolute_tolerance, budget)?;
                let bytes = self
                    .artifact_import(&transfer)?
                    .checked_bytes(ArtifactKind::BoxOptimization, budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_box_optimization(
                    bytes,
                    problem,
                    &domain,
                    extremum,
                    &tolerance,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_optimization(objective, optimization, receipt, context_digest, budget)
            }
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
            CertificationRequest::BeginRootIsolationSearch {
                system,
                search,
                domain,
                precision_digits,
                max_leaves,
                ..
            } => {
                self.admit_search_id(&search, budget)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let candidate = problem.begin_isolation_search(
                    &domain,
                    RootIsolationOptions {
                        precision_digits,
                        max_leaves,
                    },
                    budget,
                )?;
                let response = CertificationResponse::RootIsolationSearchStarted {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system: system.clone(),
                    search: search.clone(),
                    evidence: "heuristic_candidate",
                    progress: candidate.progress(),
                    context_digest,
                };
                admit_response(&response, budget)?;
                self.searches.insert(
                    search,
                    ResidentRootSearch {
                        // Cloning keeps only this bounded label's bytes, not
                        // spare capacity in a caller-owned native String.
                        system: system.clone(),
                        context_digest,
                        search: candidate,
                    },
                );
                Ok(response)
            }
            CertificationRequest::AdvanceRootIsolationSearch {
                search, max_cells, ..
            } => {
                if !(1..=MAX_ROOT_ISOLATION_ADVANCE_CELLS).contains(&max_cells) {
                    return Err(CertificationSessionError::InvalidSearchQuantum);
                }
                let slot = self
                    .searches
                    .get(&search)
                    .ok_or(CertificationSessionError::UnknownSearch)?;
                // Reserve the maximum possible counter widths and the longer
                // Boolean spelling before the engine commits even one cell.
                // Updating these Copy fields afterward cannot allocate or fail.
                let mut response = slot.response(
                    search.clone(),
                    RootIsolationProgress {
                        processed_cells: usize::MAX,
                        pending_cells: usize::MAX,
                        candidate_root_count: usize::MAX,
                        excluded_cells: usize::MAX,
                        leaf_count: usize::MAX,
                        ready_for_verification: false,
                    },
                );
                admit_response(&response, budget)?;
                let slot = self
                    .searches
                    .get_mut(&search)
                    .expect("the admitted candidate search remains resident");
                let advanced = slot
                    .search
                    .advance(max_cells, budget)
                    .map_err(CertificationError::from)?;
                let CertificationResponse::RootIsolationSearchProgress { progress, .. } =
                    &mut response
                else {
                    unreachable!("the prepared response is a candidate progress record");
                };
                *progress = advanced;
                Ok(response)
            }
            CertificationRequest::InspectRootIsolationSearch { search, .. } => {
                let slot = self
                    .searches
                    .get(&search)
                    .ok_or(CertificationSessionError::UnknownSearch)?;
                let response = slot.response(search, slot.search.progress());
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::FinishRootIsolationSearch {
                search, isolation, ..
            } => {
                self.admit_isolation_id(&isolation)?;
                let slot = self
                    .searches
                    .get(&search)
                    .ok_or(CertificationSessionError::UnknownSearch)?;
                if slot.context_digest != context_digest {
                    return Err(CertificationError::UniverseMismatch {
                        found: slot.context_digest,
                        expected: context_digest,
                    }
                    .into());
                }
                let witness = slot
                    .search
                    .completed_witness()
                    .ok_or(CertificationSessionError::SearchNotReady)?;
                let problem = self
                    .systems
                    .get(&slot.system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = problem.verify_isolation(
                    slot.search.domain(),
                    witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                let system = slot.system.clone();
                self.publish_isolation(system, isolation, receipt, context_digest, budget)
            }
            CertificationRequest::ReleaseRootIsolationSearch { search, .. } => {
                if !self.searches.contains_key(&search) {
                    return Err(CertificationSessionError::UnknownSearch);
                }
                let response = CertificationResponse::RootIsolationSearchReleased {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    search: search.clone(),
                };
                admit_response(&response, budget)?;
                self.searches.remove(&search);
                Ok(response)
            }
            CertificationRequest::ExportRootIsolationSearch { search, .. } => {
                let slot = self
                    .searches
                    .get_mut(&search)
                    .ok_or(CertificationSessionError::UnknownSearch)?;
                let bytes =
                    export_root_search_checkpoint(&mut slot.search, slot.context_digest, budget)?;
                if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let artifact = String::from_utf8(bytes)
                    .map_err(|error| CertificationSessionError::Serialization(error.to_string()))?;
                let response = CertificationResponse::RootIsolationSearchExported {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system: slot.system.clone(),
                    search,
                    evidence: "heuristic_candidate",
                    artifact,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::SaveRootIsolationSearch { search, .. } => {
                let store = self
                    .checkpoint_store
                    .as_ref()
                    .ok_or(DurableSearchError::Disabled)?;
                let slot = self
                    .searches
                    .get_mut(&search)
                    .ok_or(CertificationSessionError::UnknownSearch)?;
                let prepared = store.prepare(&mut slot.search, slot.context_digest, budget)?;
                let response = CertificationResponse::RootIsolationSearchSaved {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system: slot.system.clone(),
                    search,
                    evidence: "heuristic_candidate",
                    checkpoint: prepared.key(),
                };
                // Encoding and response resources are admitted before any disk
                // publication. A successful commit is the final effect.
                admit_response(&response, budget)?;
                store.commit(prepared, budget)?;
                Ok(response)
            }
            CertificationRequest::SaveRepairableRootIsolationSearch { search, .. } => {
                let store = self
                    .repairable_checkpoint_store
                    .as_ref()
                    .ok_or(DurableSearchError::Disabled)?;
                let slot = self
                    .searches
                    .get_mut(&search)
                    .ok_or(CertificationSessionError::UnknownSearch)?;
                let prepared = store.prepare(&mut slot.search, slot.context_digest, budget)?;
                let response = CertificationResponse::RootIsolationSearchSavedWithRepair {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system: slot.system.clone(),
                    search,
                    evidence: "heuristic_candidate",
                    checkpoint: prepared.key(),
                };
                admit_response(&response, budget)?;
                store.commit(prepared, budget)?;
                Ok(response)
            }
            CertificationRequest::LoadRepairableRootIsolationSearch {
                system,
                search,
                domain,
                precision_digits,
                max_leaves,
                checkpoint,
                ..
            } => {
                let store = self
                    .repairable_checkpoint_store
                    .as_ref()
                    .ok_or(DurableSearchError::Disabled)?;
                self.admit_search_id(&search, budget)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let (restored, repair) = store.restore(
                    checkpoint,
                    problem,
                    &domain,
                    RootIsolationOptions {
                        precision_digits,
                        max_leaves,
                    },
                    context_digest,
                    budget,
                )?;
                self.publish_resumed_search(
                    system,
                    search,
                    restored,
                    context_digest,
                    budget,
                    Some(repair),
                )
            }
            CertificationRequest::LoadRootIsolationSearch {
                system,
                search,
                domain,
                precision_digits,
                max_leaves,
                checkpoint,
                ..
            } => {
                let store = self
                    .checkpoint_store
                    .as_ref()
                    .ok_or(DurableSearchError::Disabled)?;
                self.admit_search_id(&search, budget)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let restored = store.restore(
                    checkpoint,
                    problem,
                    &domain,
                    RootIsolationOptions {
                        precision_digits,
                        max_leaves,
                    },
                    context_digest,
                    budget,
                )?;
                self.publish_restored_search(system, search, restored, context_digest, budget)
            }
            CertificationRequest::RestoreRootIsolationSearch {
                system,
                search,
                domain,
                precision_digits,
                max_leaves,
                artifact,
                ..
            } => {
                self.admit_search_id(&search, budget)?;
                if artifact.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                // Restoring checked partial work still creates only a candidate.
                // Its receiver-side cost cannot consume the protected finish pool.
                let restored = restore_root_search_checkpoint(
                    artifact.as_bytes(),
                    problem,
                    &domain,
                    RootIsolationOptions {
                        precision_digits,
                        max_leaves,
                    },
                    context_digest,
                    budget,
                )?;
                self.publish_restored_search(system, search, restored, context_digest, budget)
            }
            CertificationRequest::ExportRootIsolation { isolation, .. } => {
                let (system, receipt) = self
                    .isolations
                    .get(&isolation)
                    .ok_or(CertificationSessionError::UnknownIsolation)?;
                let bytes = export_root_isolation(receipt, context_digest, budget)?;
                if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let artifact = String::from_utf8(bytes)
                    .map_err(|error| CertificationSessionError::Serialization(error.to_string()))?;
                let response = CertificationResponse::RootIsolationExported {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system: system.clone(),
                    isolation,
                    artifact,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::ReplayRootIsolation {
                system,
                isolation,
                domain,
                artifact,
                ..
            } => {
                self.admit_isolation_id(&isolation)?;
                if artifact.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_root_isolation(
                    artifact.as_bytes(),
                    problem,
                    &domain,
                    context_digest,
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
            CertificationRequest::CertifyIntegral {
                system,
                integral,
                lower,
                upper,
                absolute_tolerance,
                precision_digits,
                max_intervals,
                ..
            } => {
                self.admit_integral_id(&integral)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let lower = parse_scalar(&lower, budget)?;
                let upper = parse_scalar(&upper, budget)?;
                let tolerance = parse_scalar(&absolute_tolerance, budget)?;
                // Require the protected pool before spending any generation work.
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let witness = problem.propose_integral(
                    &lower,
                    &upper,
                    &tolerance,
                    QuadratureOptions {
                        precision_digits,
                        max_intervals,
                    },
                    budget,
                )?;
                let receipt = problem.verify_integral(
                    &lower,
                    &upper,
                    &tolerance,
                    &witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_integral(system, integral, receipt, context_digest, budget)
            }
            CertificationRequest::ExportIntegralCertificate { integral, .. } => {
                let slot = self
                    .integrals
                    .get(&integral)
                    .ok_or(CertificationSessionError::UnknownIntegral)?;
                let problem = self
                    .systems
                    .get(&slot.system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let bytes = export_integral_certificate(
                    problem,
                    &slot.receipt,
                    slot.context_digest,
                    budget,
                )?;
                if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let artifact = String::from_utf8(bytes)
                    .map_err(|error| CertificationSessionError::Serialization(error.to_string()))?;
                let response = CertificationResponse::IntegralCertificateExported {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    system: slot.system.clone(),
                    integral,
                    artifact,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::ReplayIntegralCertificate {
                system,
                integral,
                lower,
                upper,
                absolute_tolerance,
                artifact,
                ..
            } => {
                self.admit_integral_id(&integral)?;
                if artifact.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let lower = parse_scalar(&lower, budget)?;
                let upper = parse_scalar(&upper, budget)?;
                let tolerance = parse_scalar(&absolute_tolerance, budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_integral_certificate(
                    artifact.as_bytes(),
                    problem,
                    &lower,
                    &upper,
                    &tolerance,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_integral(system, integral, receipt, context_digest, budget)
            }
            CertificationRequest::BeginArtifactExport {
                transfer, source, ..
            } => {
                self.admit_transfer_id(&transfer, budget)?;
                let (kind, bytes) = match source {
                    ArtifactSource::BoxOptimization(optimization) => {
                        let slot = self
                            .optimizations
                            .get(&optimization)
                            .ok_or(CertificationSessionError::UnknownBoxOptimization)?;
                        (
                            ArtifactKind::BoxOptimization,
                            export_box_optimization(&slot.receipt, slot.context_digest, budget)?,
                        )
                    }
                    ArtifactSource::StationaryCertificate(certificate) => {
                        let slot = self
                            .stationary_points
                            .get(&certificate)
                            .ok_or(CertificationSessionError::UnknownStationaryPoint)?;
                        (
                            ArtifactKind::StationaryCertificate,
                            export_stationary_certificate(
                                &slot.receipt,
                                slot.context_digest,
                                budget,
                            )?,
                        )
                    }
                    ArtifactSource::RootCertificate(certificate) => {
                        let (_, receipt) = self
                            .certificate_record(&certificate)
                            .ok_or(CertificationSessionError::UnknownCertificate)?;
                        (
                            ArtifactKind::RootCertificate,
                            export_root_certificate(receipt, context_digest, budget)?,
                        )
                    }
                    ArtifactSource::RootIsolation(isolation) => {
                        let (_, receipt) = self
                            .isolations
                            .get(&isolation)
                            .ok_or(CertificationSessionError::UnknownIsolation)?;
                        (
                            ArtifactKind::RootIsolation,
                            export_root_isolation(receipt, context_digest, budget)?,
                        )
                    }
                    ArtifactSource::IntegralCertificate(integral) => {
                        let slot = self
                            .integrals
                            .get(&integral)
                            .ok_or(CertificationSessionError::UnknownIntegral)?;
                        let problem = self
                            .systems
                            .get(&slot.system)
                            .ok_or(CertificationSessionError::UnknownSystem)?;
                        (
                            ArtifactKind::IntegralCertificate,
                            export_integral_certificate(
                                problem,
                                &slot.receipt,
                                slot.context_digest,
                                budget,
                            )?,
                        )
                    }
                    ArtifactSource::RootIsolationSearch(search) => {
                        let slot = self
                            .searches
                            .get_mut(&search)
                            .ok_or(CertificationSessionError::UnknownSearch)?;
                        (
                            ArtifactKind::RootSearchCheckpoint,
                            export_root_search_checkpoint(
                                &mut slot.search,
                                slot.context_digest,
                                budget,
                            )?,
                        )
                    }
                };
                let export = ArtifactExport::new(kind, bytes, budget)?;
                let response = CertificationResponse::ArtifactExportStarted {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    transfer: transfer.clone(),
                    manifest: export.manifest().clone(),
                };
                admit_response(&response, budget)?;
                self.transfers
                    .insert(transfer, ResidentArtifact::Export(export));
                Ok(response)
            }
            CertificationRequest::BeginArtifactImport {
                transfer, manifest, ..
            } => {
                self.admit_transfer_id(&transfer, budget)?;
                let import = ArtifactImport::new(manifest, budget)?;
                let response = CertificationResponse::ArtifactImportStarted {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    transfer: transfer.clone(),
                    manifest: import.manifest().clone(),
                    progress: import.progress(),
                };
                admit_response(&response, budget)?;
                self.transfers
                    .insert(transfer, ResidentArtifact::Import(import));
                Ok(response)
            }
            CertificationRequest::ReadArtifactChunk {
                transfer, index, ..
            } => {
                let export = match self.transfers.get(&transfer) {
                    Some(ResidentArtifact::Export(export)) => export,
                    Some(ResidentArtifact::Import(_)) => {
                        return Err(CertificationSessionError::WrongTransferMode);
                    }
                    None => return Err(CertificationSessionError::UnknownTransfer),
                };
                let response = CertificationResponse::ArtifactChunkRead {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    chunk: export.chunk(index, budget)?,
                    transfer,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::AppendArtifactChunk {
                transfer, chunk, ..
            } => {
                let progress = self
                    .artifact_import(&transfer)?
                    .preview_append(chunk.index)?;
                let response = CertificationResponse::ArtifactChunkAppended {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    transfer: transfer.clone(),
                    progress,
                };
                // A response-admission refusal cannot advance the staged
                // prefix. A lost reply is recovered by retrying the exact chunk.
                // The core validates and charges before its only write.
                admit_response(&response, budget)?;
                let Some(ResidentArtifact::Import(import)) = self.transfers.get_mut(&transfer)
                else {
                    unreachable!("the checked transfer remains in its import slot");
                };
                import.append(chunk.index, &chunk.digest, &chunk.data_hex, budget)?;
                Ok(response)
            }
            CertificationRequest::ReleaseArtifactTransfer { transfer, .. } => {
                if !self.transfers.contains_key(&transfer) {
                    return Err(CertificationSessionError::UnknownTransfer);
                }
                let response = CertificationResponse::ArtifactTransferReleased {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    transfer: transfer.clone(),
                };
                admit_response(&response, budget)?;
                self.transfers.remove(&transfer);
                Ok(response)
            }
            CertificationRequest::ReplayRootCertificateFromArtifact {
                system,
                certificate,
                transfer,
                ..
            } => {
                self.admit_certificate_id(&certificate)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let bytes = self
                    .artifact_import(&transfer)?
                    .checked_bytes(ArtifactKind::RootCertificate, budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_root_certificate(
                    bytes,
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
            CertificationRequest::ReplayRootIsolationFromArtifact {
                system,
                isolation,
                domain,
                transfer,
                ..
            } => {
                self.admit_isolation_id(&isolation)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let bytes = self
                    .artifact_import(&transfer)?
                    .checked_bytes(ArtifactKind::RootIsolation, budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_root_isolation(
                    bytes,
                    problem,
                    &domain,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_isolation(system, isolation, receipt, context_digest, budget)
            }
            CertificationRequest::ReplayIntegralCertificateFromArtifact {
                system,
                integral,
                lower,
                upper,
                absolute_tolerance,
                transfer,
                ..
            } => {
                self.admit_integral_id(&integral)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let lower = parse_scalar(&lower, budget)?;
                let upper = parse_scalar(&upper, budget)?;
                let tolerance = parse_scalar(&absolute_tolerance, budget)?;
                let bytes = self
                    .artifact_import(&transfer)?
                    .checked_bytes(ArtifactKind::IntegralCertificate, budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = replay_integral_certificate(
                    bytes,
                    problem,
                    &lower,
                    &upper,
                    &tolerance,
                    context_digest,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_integral(system, integral, receipt, context_digest, budget)
            }
            CertificationRequest::RestoreRootIsolationSearchFromArtifact {
                system,
                search,
                domain,
                precision_digits,
                max_leaves,
                transfer,
                ..
            } => {
                self.admit_search_id(&search, budget)?;
                let problem = self
                    .systems
                    .get(&system)
                    .ok_or(CertificationSessionError::UnknownSystem)?;
                let domain = parse_domain(&domain, problem.variables().len(), budget)?;
                let bytes = self
                    .artifact_import(&transfer)?
                    .checked_bytes(ArtifactKind::RootSearchCheckpoint, budget)?;
                let restored = restore_root_search_checkpoint(
                    bytes,
                    problem,
                    &domain,
                    RootIsolationOptions {
                        precision_digits,
                        max_leaves,
                    },
                    context_digest,
                    budget,
                )?;
                self.publish_restored_search(system, search, restored, context_digest, budget)
            }
        }
    }
}
