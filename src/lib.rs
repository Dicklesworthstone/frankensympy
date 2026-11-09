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
pub mod isolation_transport;
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
    CERTIFICATION_REQUEST_SCHEMA, CertificationRequest, CertificationResponse,
    CertificationSessionError, ExactInterval, ExactRootWitness, IsolatedRoot, ResidualSystemId,
    RootCertificateId, RootIsolationId,
};
pub use fsym_budget::{Budget, BudgetLimits, BudgetMeter, BudgetSnapshot, Dimension, MeterError};
pub use fsym_calculus::root_isolation::RootIsolationOptions;
pub use fsym_core::{
    BigInt, BigRational, Expr, RealBall, ResidualJacobianEnclosure, RootBoxReceipt, RootBoxWitness,
    RootIsolationError, RootIsolationReceipt, RootIsolationStep, RootIsolationWitness, Symbol,
    SymbolIdentity,
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
pub use isolation_transport::{
    ROOT_ISOLATION_ARTIFACT_SCHEMA_VERSION, export_root_isolation, replay_root_isolation,
};

pub use session::{
    ClaimExportPage, ClaimRecord, ClaimRequest, ClaimStatus, MAX_CLAIM_EXPORT_BYTES,
    MAX_ENVELOPE_BYTES, MAX_EXPORT_PAGE_SIZE, MAX_SOURCE_BYTES, Session, SessionBudgets,
    SessionError,
};
