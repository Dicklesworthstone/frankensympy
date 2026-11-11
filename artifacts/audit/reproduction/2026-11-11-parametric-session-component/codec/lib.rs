//! Whole pinned donor codec modules; excludes the asynchronous runtime graph.
#![deny(unsafe_code)]
#![cfg_attr(feature = "nightly-outcome-try", feature(try_trait_v2))]
#![cfg_attr(feature = "nightly-outcome-try", feature(try_trait_v2_residual))]
#![cfg_attr(feature = "simd-intrinsics", feature(portable_simd))]
#![recursion_limit = "256"]

#[path = "/workspace/scratch/8a453db7a629/asupersync/src/tracing_compat.rs"]
pub mod tracing_compat;

pub mod raptorq {
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/raptorq/decoder.rs"]
    pub mod decoder;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/raptorq/systematic.rs"]
    pub mod systematic;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/raptorq/gf256.rs"]
    pub mod gf256;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/raptorq/linalg.rs"]
    pub mod linalg;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/raptorq/rfc6330.rs"]
    pub mod rfc6330;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/raptorq/proof.rs"]
    pub mod proof;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/raptorq/decision_contract.rs"]
    pub mod decision_contract;
}

pub mod util {
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/util/det_rng.rs"]
    pub mod det_rng;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/util/det_hash.rs"]
    pub mod det_hash;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/util/entropy.rs"]
    pub mod entropy;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/util/arena.rs"]
    pub mod arena;
    pub use det_rng::DetRng;
    pub use det_hash::DetHasher;
    pub use arena::ArenaIndex;
}

pub mod types {
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/types/id.rs"]
    pub mod id;
    #[path = "/workspace/scratch/8a453db7a629/asupersync/src/types/symbol.rs"]
    pub mod symbol;
    pub use id::TaskId;
    pub use symbol::ObjectId;
}

