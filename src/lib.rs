//! # FrankenSymPy
//!
//! Implementation workspace for an independently implemented, memory-safe Rust and Python
//! replacement for named SymPy compatibility profiles.
//!
//! The core implementation is landed and tested, but the compatibility profiles described by
//! the plan are implemented-uncertified: no profile is certified yet.

#![forbid(unsafe_code)]

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Machine-readable implementation status for the current crate.
pub const IMPLEMENTATION_STATUS: &str = "implemented_uncertified";

pub mod artifact_stream;
pub mod certification;
pub mod certification_session;
pub mod durable_search;
pub mod integration;
pub mod isolation_transport;
pub mod objective;
pub mod objective_transport;
pub mod optimization;
pub mod parametric;
pub mod repairable_search;
pub mod search_checkpoint;
pub mod session;

pub use artifact_stream::{
    ARTIFACT_CHUNK_BYTES, ARTIFACT_STREAM_SCHEMA_VERSION, ArtifactChunk, ArtifactContentId,
    ArtifactExport, ArtifactImport, ArtifactKind, ArtifactManifest, ArtifactProgress,
    ArtifactTransferError, ArtifactTransferId, MAX_ARTIFACT_BYTES, artifact_chunk_digest,
    artifact_content_id,
};
pub use certification::{
    CertificationError, ResidualProblem, export_root_certificate, replay_root_certificate,
};
pub use certification_session::{
    ArtifactSource, BoxOptimizationId, CERTIFICATION_REQUEST_SCHEMA, CertificationRequest,
    CertificationResponse, CertificationSessionError, ExactInterval, ExactParametricRootWitness,
    ExactRootWitness, IntegralCertificateId, IsolatedRoot, MAX_SESSION_ARTIFACT_TRANSFERS,
    MAX_SESSION_ROOT_SEARCHES, ObjectiveId, ParametricFamilyId, ParametricFamilyReceipt,
    ParametricProblemId, ParametricSensitivityId, ResidualSystemId, RootCertificateId,
    RootIsolationId, RootIsolationSearchId, StationaryPointClassification, StationaryPointId,
};
pub use durable_search::{
    DURABLE_ROOT_SEARCH_SCHEMA, DurableSearchError, PreparedRootSearchCheckpoint,
    RootSearchCheckpointStore, StoredRootSearchCheckpoint,
};
pub use fsym_budget::{Budget, BudgetLimits, BudgetMeter, BudgetSnapshot, Dimension, MeterError};
pub use fsym_calculus::box_optimization::BoxOptimizationOptions;
pub use fsym_calculus::parametric_cover::ParametricRootCoverOptions;
pub use fsym_calculus::quadrature::QuadratureOptions;
pub use fsym_calculus::root_isolation::{
    MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationCounters, RootIsolationOptions,
    RootIsolationProgress, RootIsolationSearch, RootIsolationSearchCheckpoint,
};
pub use fsym_core::{
    BigInt, BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationReceipt,
    BoxOptimizationWitness, BoxPartitionStep, Expr, IntegralEnclosureReceipt, ObjectiveEnclosure,
    ParametricRootBoxReceipt, ParametricRootBoxWitness, ParametricRootCoverError,
    ParametricRootCoverReceipt, ParametricRootCoverStep, ParametricRootCoverWitness,
    ParametricSensitivityReceipt, QuadratureError, QuadratureWitness, RealBall,
    ResidualJacobianEnclosure, RootBoxReceipt, RootBoxWitness, RootIsolationError,
    RootIsolationReceipt, RootIsolationStep, RootIsolationWitness, StationaryPointKind,
    StationaryPointReceipt, Symbol, SymbolIdentity, metered_enclose_parametric_cover_sensitivity,
    metered_enclose_parametric_root_sensitivity,
};
pub use fsym_proof_kernel::{
    MergeCertError, MergePolicy, SemanticMergeCertificate, WitnessKind, verify_merge_certificate,
};
pub use fsym_runtime::fmap::{FmapBundle, ReplayOutcome, replay_fmap_bundle};
pub use fsym_runtime::protocol::{
    AgentRequest, AgentResponse, ProtocolErrorCode, ProtocolSession, handle_agent_session_ndjson,
};
pub use fsym_runtime::publication::{PublicationError, PublicationGate};
pub use fsym_runtime::workspace::{
    MergeReceipt, SemanticWorkspace, WorkspaceError, WorkspacePatch,
};
pub use integration::{
    INTEGRAL_CERTIFICATE_SCHEMA_VERSION, IntegrationError, MAX_INTEGRAL_CERTIFICATE_BYTES,
    export_integral_certificate, replay_integral_certificate,
};
pub use isolation_transport::{
    ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION, export_root_isolation, replay_root_isolation,
};
pub use objective::ObjectiveProblem;
pub use objective_transport::{
    MAX_STATIONARY_CERTIFICATE_BYTES, STATIONARY_CERTIFICATE_SCHEMA_VERSION,
    export_stationary_certificate, replay_stationary_certificate,
};
pub use optimization::{
    BOX_OPTIMIZATION_ARTIFACT_SCHEMA_VERSION, MAX_BOX_OPTIMIZATION_ARTIFACT_BYTES,
    OptimizationError, export_box_optimization, replay_box_optimization,
};
pub use parametric::{ParametricError, ParametricProblem};
pub use repairable_search::{
    PreparedRepairableRootSearchCheckpoint, RepairableRootSearchCheckpoint,
    RepairableRootSearchCheckpointStore, RepairableSearchError, RootSearchRepairReport,
};
pub use search_checkpoint::{
    MAX_ROOT_SEARCH_CHECKPOINT_BYTES, ROOT_SEARCH_CHECKPOINT_SCHEMA_VERSION,
    export_root_search_checkpoint, restore_root_search_checkpoint,
};

pub use session::{
    ClaimExportPage, ClaimRecord, ClaimRequest, ClaimStatus, MAX_CLAIM_EXPORT_BYTES,
    MAX_ENVELOPE_BYTES, MAX_EXPORT_PAGE_SIZE, MAX_SOURCE_BYTES, Session, SessionBudgets,
    SessionError,
};
