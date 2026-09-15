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

pub mod session;

pub use fsym_budget::{Budget, BudgetLimits};
pub use fsym_core::{Expr, Symbol};
pub use fsym_proof_kernel::{
    MergeCertError, MergePolicy, SemanticMergeCertificate, WitnessKind, verify_merge_certificate,
};
pub use fsym_runtime::fmap::{FmapBundle, ReplayOutcome, replay_fmap_bundle};
pub use fsym_runtime::protocol::{AgentRequest, AgentResponse, ProtocolErrorCode};
pub use fsym_runtime::publication::{PublicationError, PublicationGate};
pub use fsym_runtime::workspace::{
    MergeReceipt, SemanticWorkspace, WorkspaceError, WorkspacePatch,
};

pub use session::{
    ClaimExportPage, ClaimRecord, ClaimStatus, MAX_ENVELOPE_BYTES, MAX_EXPORT_PAGE_SIZE,
    MAX_SOURCE_BYTES, Session, SessionBudgets, SessionError,
};
