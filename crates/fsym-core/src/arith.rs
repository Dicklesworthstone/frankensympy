//! Compatibility façade for the WS03 exact-arithmetic ownership crates.
//!
//! Integer value operations are owned by `fsym-bigint`; modular arithmetic,
//! CRT, rational reconstruction support, and deterministic prime primitives
//! are owned by `fsym-modular`. Existing `fsym_core::arith` callers retain
//! one stable import surface while the L1 crates remain independently usable.

#![forbid(unsafe_code)]

pub use fsym_bigint::{
    BigInt, DEFAULT_STRATEGY_THRESHOLD_BITS, LIMB_BITS, NonZeroBigInt, Strategy as MulStrategy,
    exact_div, extended_gcd, gcd, limb_count_u64, metered_div_rem, metered_div_rem_nonzero,
    metered_exact_div, metered_extended_gcd, metered_gcd, metered_multiply as metered_mul,
    multiply, multiply_with_strategy as mul_with_strategy, select_strategy,
};
pub use fsym_modular::{
    PrimeStream, crt, crt_pair, is_probable_prime, metered_crt, metered_crt_pair,
    metered_is_probable_prime, metered_mod_inverse, metered_rational_reconstruct, mod_inverse,
    rational_reconstruct,
};
