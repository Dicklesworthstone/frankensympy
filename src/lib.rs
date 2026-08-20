//! # FrankenSymPy
//!
//! Planning-stage crate for an independently implemented, memory-safe Rust and Python
//! replacement for named SymPy compatibility profiles.
//!
//! The architecture and implementation program are public, but the symbolic engine and
//! compatibility profiles described by that plan are not implemented or certified yet.

#![forbid(unsafe_code)]

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Machine-readable implementation status for the current crate.
pub const IMPLEMENTATION_STATUS: &str = "planning";
