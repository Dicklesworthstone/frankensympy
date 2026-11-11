//! Whole original Session dependency modules; source-component validation only.
#![forbid(unsafe_code)]

#[path = "/workspace/scratch/8a453db7a629/frankensympy-integration/crates/fsym-runtime/src/workspace.rs"]
pub mod workspace;
#[path = "/workspace/scratch/8a453db7a629/frankensympy-integration/crates/fsym-runtime/src/protocol.rs"]
pub mod protocol;
#[path = "/workspace/scratch/8a453db7a629/frankensympy-integration/crates/fsym-runtime/src/fmap.rs"]
pub mod fmap;
#[path = "/workspace/scratch/8a453db7a629/frankensympy-integration/crates/fsym-runtime/src/publication.rs"]
pub mod publication;
#[path = "/workspace/scratch/8a453db7a629/frankensympy-integration/crates/fsym-runtime/src/repair.rs"]
pub mod repair;

pub use repair::*;
