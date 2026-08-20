//! # FrankenSymPy
//!
//! Clean-room, memory-safe Rust reimplementation of SymPy with differential oracle conformance.

#![forbid(unsafe_code)]

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
