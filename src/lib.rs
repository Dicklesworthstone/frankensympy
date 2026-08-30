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
