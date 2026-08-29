//! Exact modular arithmetic primitives (WS03).
//!
//! Everything here is pure modular big-integer math with no FFI and no
//! machine-float intermediates behind the [`fsym_bigint::BigInt`] containment boundary.
//! Determinism: identical inputs always produce identical outputs; the prime stream is
//! a fixed deterministic sequence, and primality testing uses a fixed base set.
//!
//! # Primality honesty
//!
//! [`is_probable_prime`] is **deterministic** for `n < 3.317·10²⁴` (the
//! first 13 prime bases are a proven certificate for that range) and only
//! probabilistic beyond it. Callers needing certainty above that bound
//! must supply their own proof (e.g. ECPP later in WS11).

#![forbid(unsafe_code)]

use fsym_bigint::{
    BigInt, NonZeroBigInt, extended_gcd, gcd, metered_add as metered_bigint_add,
    metered_cmp as metered_bigint_cmp, metered_div_rem_nonzero, metered_extended_gcd, metered_gcd,
    metered_multiply as metered_mul, metered_pow as metered_bigint_pow, metered_sqrt_floor,
    metered_subtract as metered_bigint_subtract, sqrt_floor,
};
#[cfg(test)]
use fsym_bigint::{exact_div, metered_exact_div};
use fsym_budget::{BudgetMeter, Dimension, MeterError};

/// Why a governed prime-stream step stopped before publishing a prime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimeStreamError {
    /// Budget exhaustion or structured cancellation from the owning region.
    Meter(MeterError),
    /// The deterministic emitted-table growth calculation exceeded machine size.
    SizeOverflow,
    /// The private replacement table could not reserve its checked logical capacity.
    AllocationFailure,
}

impl std::fmt::Display for PrimeStreamError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Meter(error) => error.fmt(formatter),
            Self::SizeOverflow => formatter.write_str("prime-stream table size overflow"),
            Self::AllocationFailure => {
                formatter.write_str("prime-stream replacement-table allocation failed")
            }
        }
    }
}

impl std::error::Error for PrimeStreamError {}

impl From<MeterError> for PrimeStreamError {
    fn from(error: MeterError) -> Self {
        Self::Meter(error)
    }
}

/// Why a governed finite-field batch inversion stopped before publishing its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchInverseError {
    /// Budget exhaustion or structured cancellation from the owning region.
    Meter(MeterError),
    /// The checked buffer-layout calculation exceeded machine size.
    SizeOverflow,
    /// One of the two private batch buffers could not reserve its checked capacity.
    AllocationFailure,
    /// A private finite-field invariant failed after public inputs were admitted.
    InvariantViolation(&'static str),
}

impl std::fmt::Display for BatchInverseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Meter(error) => error.fmt(formatter),
            Self::SizeOverflow => formatter.write_str("batch-inverse buffer size overflow"),
            Self::AllocationFailure => {
                formatter.write_str("batch-inverse private-buffer allocation failed")
            }
            Self::InvariantViolation(message) => {
                write!(formatter, "batch-inverse invariant violation: {message}")
            }
        }
    }
}

impl std::error::Error for BatchInverseError {}

impl From<MeterError> for BatchInverseError {
    fn from(error: MeterError) -> Self {
        Self::Meter(error)
    }
}

/// Multiplicative inverse of `a` modulo `m` (`m > 0`); `None` when
/// `gcd(a, m) != 1`.
pub fn mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    if !m.is_positive() {
        return None;
    }
    let (g, x, _) = extended_gcd(&(a % m), m);
    if !g.is_one() {
        return None;
    }
    Some(((x % m) + m) % m)
}

/// Cancellation-first multiplicative inverse using metered division and Bézout lanes.
pub fn metered_mod_inverse<M: BudgetMeter>(
    a: &BigInt,
    m: &BigInt,
    meter: &mut M,
) -> Result<Option<BigInt>, MeterError> {
    meter.checkpoint()?;
    if !m.is_positive() {
        return metered_finish(None, meter);
    }
    let Some(modulus) = NonZeroBigInt::new(m) else {
        return metered_finish(None, meter);
    };
    let residue = metered_normalized_remainder(a, modulus, meter)?;
    let (g, x, _) = metered_extended_gcd(&residue, m, meter)?;
    if !g.is_one() {
        return metered_finish(None, meter);
    }
    let inverse = metered_normalized_remainder(&x, modulus, meter)?;
    metered_finish(Some(inverse), meter)
}

/// Solves the two-congruence system `x ≡ rem_i (mod mod_i)`. Returns
/// `(x, lcm(mod_1, mod_2))` with `0 <= x < lcm`, or `None` when the
/// congruences are inconsistent.
pub fn crt_pair(
    rem1: &BigInt,
    mod1: &BigInt,
    rem2: &BigInt,
    mod2: &BigInt,
) -> Option<(BigInt, BigInt)> {
    if !mod1.is_positive() || !mod2.is_positive() {
        return None;
    }
    let g = gcd(mod1, mod2);
    crt_pair_with_gcd(rem1, mod1, rem2, mod2, &g)
}

fn crt_pair_with_gcd(
    rem1: &BigInt,
    mod1: &BigInt,
    rem2: &BigInt,
    mod2: &BigInt,
    g: &BigInt,
) -> Option<(BigInt, BigInt)> {
    debug_assert!(mod1.is_positive());
    debug_assert!(mod2.is_positive());
    debug_assert!(g.is_positive());
    let diff = rem2 - rem1;
    if (&diff % g) != BigInt::zero() {
        return None;
    }
    let lcm = (mod1 / g) * mod2;
    let m1_div_g = mod1 / g;
    let m2_div_g = mod2 / g;
    let (_, u, _) = extended_gcd(&m1_div_g, &m2_div_g);
    let shift = (diff / g) * u * mod1;
    let mut x = (rem1 + shift) % &lcm;
    if x.is_negative() {
        x += &lcm;
    }
    Some((x, lcm))
}

/// Cancellation-first two-congruence CRT using only metered arithmetic lanes.
pub fn metered_crt_pair<M: BudgetMeter>(
    rem1: &BigInt,
    mod1: &BigInt,
    rem2: &BigInt,
    mod2: &BigInt,
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    if !mod1.is_positive() || !mod2.is_positive() {
        return metered_finish(None, meter);
    }

    let g = metered_gcd(mod1, mod2, meter)?;
    let Some(g_divisor) = NonZeroBigInt::new(&g) else {
        return metered_finish(None, meter);
    };
    let diff = metered_subtract(rem2, rem1, meter)?;
    let (diff_over_g, diff_remainder) = metered_div_rem_nonzero(&diff, g_divisor, meter)?;
    if !diff_remainder.is_zero() {
        return metered_finish(None, meter);
    }
    let (m1_div_g, m1_remainder) = metered_div_rem_nonzero(mod1, g_divisor, meter)?;
    let (m2_div_g, m2_remainder) = metered_div_rem_nonzero(mod2, g_divisor, meter)?;
    if !m1_remainder.is_zero() || !m2_remainder.is_zero() {
        return metered_finish(None, meter);
    }

    let lcm = metered_mul(&m1_div_g, mod2, meter)?;
    let (_, u, _) = metered_extended_gcd(&m1_div_g, &m2_div_g, meter)?;
    let scaled_diff = metered_mul(&diff_over_g, &u, meter)?;
    let shift = metered_mul(&scaled_diff, mod1, meter)?;
    let shifted_remainder = metered_add(rem1, &shift, meter)?;
    let Some(lcm_divisor) = NonZeroBigInt::new(&lcm) else {
        return metered_finish(None, meter);
    };
    let x = metered_normalized_remainder(&shifted_remainder, lcm_divisor, meter)?;
    metered_finish(Some((x, lcm)), meter)
}

/// Solves an arbitrary system of simultaneous congruences.
pub fn crt(congruences: &[(BigInt, BigInt)]) -> Option<(BigInt, BigInt)> {
    if congruences
        .iter()
        .any(|(_remainder, modulus)| !modulus.is_positive())
    {
        return None;
    }
    crt_refs(
        congruences
            .iter()
            .map(|(remainder, modulus)| (remainder, modulus)),
        false,
    )
}

/// Solves a split-slice system whose moduli must be pairwise coprime.
///
/// This borrowed lane avoids materializing an owned `(remainder, modulus)` vector for facades
/// that already store the two columns separately. Both empty slices return the CRT identity
/// `(0, 1)`; unequal lengths, non-positive moduli, or non-coprime moduli return `None`.
pub fn crt_coprime_slices(remainders: &[BigInt], moduli: &[BigInt]) -> Option<(BigInt, BigInt)> {
    if remainders.len() != moduli.len() || moduli.iter().any(|modulus| !modulus.is_positive()) {
        return None;
    }
    crt_refs(remainders.iter().zip(moduli), true)
}

fn crt_refs<'a>(
    mut congruences: impl Iterator<Item = (&'a BigInt, &'a BigInt)>,
    require_pairwise_coprime: bool,
) -> Option<(BigInt, BigInt)> {
    let Some((first_remainder, first_modulus)) = congruences.next() else {
        return Some((BigInt::zero(), BigInt::one()));
    };

    let mut x = first_remainder % first_modulus;
    if x.is_negative() {
        x += first_modulus;
    }
    let mut modulus = first_modulus.clone();

    for (r_i, m_i) in congruences {
        let g = gcd(&modulus, m_i);
        if require_pairwise_coprime && !g.is_one() {
            return None;
        }
        let (next_x, next_m) = crt_pair_with_gcd(&x, &modulus, r_i, m_i, &g)?;
        x = next_x;
        modulus = next_m;
    }
    Some((x, modulus))
}

/// Cancellation-first arbitrary CRT fold.
pub fn metered_crt<M: BudgetMeter>(
    congruences: &[(BigInt, BigInt)],
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    let mut congruence_iter = congruences.iter();
    let Some((first_remainder, first_modulus)) = congruence_iter.next() else {
        return metered_finish(Some((BigInt::zero(), BigInt::one())), meter);
    };
    if !first_modulus.is_positive() {
        return metered_finish(None, meter);
    }
    for (_remainder, modulus) in congruence_iter.clone() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        if !modulus.is_positive() {
            return metered_finish(None, meter);
        }
    }
    let Some(first_modulus_divisor) = NonZeroBigInt::new(first_modulus) else {
        return metered_finish(None, meter);
    };
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            first_modulus.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut x = metered_normalized_remainder(first_remainder, first_modulus_divisor, meter)?;
    let mut modulus = first_modulus.clone();
    for (remainder, next_modulus) in congruence_iter {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let Some((next_x, combined_modulus)) =
            metered_crt_pair(&x, &modulus, remainder, next_modulus, meter)?
        else {
            return metered_finish(None, meter);
        };
        x = next_x;
        modulus = combined_modulus;
    }
    metered_finish(Some((x, modulus)), meter)
}

/// Symmetric rational reconstruction: recovers `(r, s)` with `gcd(r, s) == 1`,
/// `s > 0`, and `r · s⁻¹ ≡ n (mod m)`, where `|r|` and `s` do not exceed
/// `floor(sqrt((m - 1) / 2))`. The strict `2 * bound^2 < m` inequality makes the
/// representative unique.
pub fn rational_reconstruct(n: &BigInt, m: &BigInt) -> Option<(BigInt, BigInt)> {
    if *m <= BigInt::one() {
        return None;
    }

    // The symmetric uniqueness condition is 2 * bound^2 < m. Using sqrt(m)
    // admits multiple representatives and can make the result depend on the Euclidean path.
    let bound = sqrt_floor(&((m - 1i64) / 2i64))?;
    if bound.is_zero() {
        return None;
    }

    let residue = (n % m + m) % m;
    if residue.is_zero() {
        return Some((BigInt::zero(), BigInt::one()));
    }

    let (mut r_prev, mut r_cur) = (m.clone(), residue.clone());
    let (mut t_prev, mut t_cur) = (BigInt::zero(), BigInt::one());

    while r_cur.abs() > bound {
        let (q, r_next) = r_prev.div_rem(&r_cur);
        r_prev = r_cur;
        r_cur = r_next;

        let t_next = t_prev - q * &t_cur;
        t_prev = t_cur;
        t_cur = t_next;
    }

    let mut r_out = r_cur;
    if t_cur.is_negative() {
        r_out = -r_out;
        t_cur = -t_cur;
    }
    if !t_cur.is_positive() {
        return None;
    }
    if r_out.abs() > bound || t_cur > bound {
        return None;
    }
    if gcd(&r_out, &t_cur) != BigInt::one() {
        return None;
    }
    if (&r_out - &residue * &t_cur) % m != BigInt::zero() {
        return None;
    }
    Some((r_out, t_cur))
}

/// Cancellation-first symmetric rational reconstruction.
pub fn metered_rational_reconstruct<M: BudgetMeter>(
    n: &BigInt,
    m: &BigInt,
    meter: &mut M,
) -> Result<Option<(BigInt, BigInt)>, MeterError> {
    meter.checkpoint()?;
    if *m <= BigInt::one() {
        return metered_finish(None, meter);
    }
    let one = BigInt::one();
    let two = BigInt::from(2i64);
    let m_minus_one = metered_subtract(m, &one, meter)?;
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return Ok(None);
    };
    let (half, _) = metered_div_rem_nonzero(&m_minus_one, two_divisor, meter)?;
    let Some(bound) = metered_sqrt_floor(&half, meter)? else {
        return metered_finish(None, meter);
    };
    if bound.is_zero() {
        return metered_finish(None, meter);
    }

    let Some(modulus) = NonZeroBigInt::new(m) else {
        return metered_finish(None, meter);
    };
    let residue = metered_normalized_remainder(n, modulus, meter)?;
    if residue.is_zero() {
        return metered_finish(Some((BigInt::zero(), BigInt::one())), meter);
    }

    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            m.limb_count()
                .max(1)
                .saturating_add(residue.limb_count().max(1))
                .saturating_mul(8),
        ),
        (Dimension::AllocationCount, 2),
    ])?;
    let (mut r_prev, mut r_cur) = (m.clone(), residue.clone());
    let (mut t_prev, mut t_cur) = (BigInt::zero(), BigInt::one());

    while metered_greater(&r_cur, &bound, meter)? {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let Some(r_cur_divisor) = NonZeroBigInt::new(&r_cur) else {
            return Ok(None);
        };
        let (q, r_next) = metered_div_rem_nonzero(&r_prev, r_cur_divisor, meter)?;
        r_prev = r_cur;
        r_cur = r_next;

        let q_times_t = metered_mul(&q, &t_cur, meter)?;
        let t_next = metered_subtract(&t_prev, &q_times_t, meter)?;
        t_prev = t_cur;
        t_cur = t_next;
    }

    let mut r_out = r_cur;
    if t_cur.is_negative() {
        r_out = metered_negate(r_out, meter)?;
        t_cur = metered_negate(t_cur, meter)?;
    }
    if !t_cur.is_positive() || metered_greater(&t_cur, &bound, meter)? {
        return metered_finish(None, meter);
    }
    if metered_gcd(&r_out, &t_cur, meter)? != BigInt::one() {
        return metered_finish(None, meter);
    }
    let residue_times_denominator = metered_mul(&residue, &t_cur, meter)?;
    let congruence_delta = metered_subtract(&r_out, &residue_times_denominator, meter)?;
    if !metered_normalized_remainder(&congruence_delta, modulus, meter)?.is_zero() {
        return metered_finish(None, meter);
    }
    metered_finish(Some((r_out, t_cur)), meter)
}

/// Deterministic increasing stream of primes: 2, 3, 5, 7, ...
pub struct PrimeStream {
    emitted: Vec<BigInt>,
    emitted_capacity: usize,
    current: BigInt,
}

fn prime_table_growth_target(
    logical_capacity: usize,
    required_len: usize,
) -> Result<usize, PrimeStreamError> {
    if required_len <= logical_capacity {
        return Ok(logical_capacity);
    }
    let grown = if logical_capacity == 0 {
        1
    } else {
        logical_capacity
            .checked_mul(2)
            .ok_or(PrimeStreamError::SizeOverflow)?
    };
    Ok(grown.max(required_len))
}

fn prime_stream_error<T, M: BudgetMeter>(
    error: PrimeStreamError,
    meter: &mut M,
) -> Result<T, PrimeStreamError> {
    meter.checkpoint()?;
    Err(error)
}

fn reserve_prime_table<M: BudgetMeter>(
    logical_capacity: usize,
    copied_entries: usize,
    meter: &mut M,
) -> Result<Vec<BigInt>, PrimeStreamError> {
    let header_bytes = match u64::try_from(std::mem::size_of::<BigInt>()) {
        Ok(bytes) => bytes,
        Err(_) => return prime_stream_error(PrimeStreamError::SizeOverflow, meter),
    };
    let logical_capacity_u64 = match u64::try_from(logical_capacity) {
        Ok(capacity) => capacity,
        Err(_) => return prime_stream_error(PrimeStreamError::SizeOverflow, meter),
    };
    let copied_entries = match u64::try_from(copied_entries) {
        Ok(entries) => entries,
        Err(_) => return prime_stream_error(PrimeStreamError::SizeOverflow, meter),
    };
    let Some(replacement_bytes) = logical_capacity_u64.checked_mul(header_bytes) else {
        return prime_stream_error(PrimeStreamError::SizeOverflow, meter);
    };

    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, copied_entries),
        (Dimension::MemoryBytes, replacement_bytes),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut replacement = Vec::new();
    if replacement.try_reserve_exact(logical_capacity).is_err() {
        return prime_stream_error(PrimeStreamError::AllocationFailure, meter);
    }
    meter.checkpoint()?;
    Ok(replacement)
}

impl PrimeStream {
    pub fn new() -> Self {
        Self {
            emitted: Vec::new(),
            emitted_capacity: 0,
            current: BigInt::from(2i64),
        }
    }

    /// Returns the next prime through cancellation-first metered arithmetic.
    ///
    /// Refusal leaves the stream cursor and emitted-prime table unchanged, so retrying cannot
    /// silently skip a candidate. Consumed budget is not refunded.
    pub fn next_metered<M: BudgetMeter>(
        &mut self,
        meter: &mut M,
    ) -> Result<BigInt, PrimeStreamError> {
        meter.checkpoint()?;
        meter.charge_batch(&[
            (
                Dimension::MemoryBytes,
                self.current.limb_count().max(1).saturating_mul(8),
            ),
            (Dimension::AllocationCount, 1),
        ])?;
        let mut current = self.current.clone();
        let one = BigInt::one();
        loop {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let candidate = current;
            let next_current = metered_add(&candidate, &one, meter)?;
            let root = metered_sqrt_floor(&candidate, meter)?
                .expect("prime-stream candidates stay positive");
            let mut divides = false;
            for prime in &self.emitted {
                meter.checkpoint()?;
                meter.charge(Dimension::ComputeSteps, 1)?;
                if metered_greater(prime, &root, meter)? {
                    break;
                }
                let Some(prime_divisor) = NonZeroBigInt::new(prime) else {
                    continue;
                };
                let (_, remainder) = metered_div_rem_nonzero(&candidate, prime_divisor, meter)?;
                if remainder.is_zero() {
                    divides = true;
                    break;
                }
            }
            if !divides {
                let Some(required_len) = self.emitted.len().checked_add(1) else {
                    return prime_stream_error(PrimeStreamError::SizeOverflow, meter);
                };
                if required_len > self.emitted_capacity {
                    let target =
                        match prime_table_growth_target(self.emitted_capacity, required_len) {
                            Ok(target) => target,
                            Err(error) => return prime_stream_error(error, meter),
                        };
                    let mut replacement = reserve_prime_table(target, required_len, meter)?;
                    for prime in &self.emitted {
                        replacement.push(metered_clone_bigint(prime, meter)?);
                    }
                    replacement.push(metered_clone_bigint(&candidate, meter)?);
                    meter.checkpoint()?;
                    self.emitted = replacement;
                    self.emitted_capacity = target;
                    self.current = next_current;
                    return Ok(candidate);
                }

                debug_assert!(self.emitted.capacity() >= self.emitted_capacity);
                meter.charge(Dimension::ComputeSteps, 1)?;
                let stored_candidate = metered_clone_bigint(&candidate, meter)?;
                meter.checkpoint()?;
                self.emitted.push(stored_candidate);
                self.current = next_current;
                return Ok(candidate);
            }
            current = next_current;
        }
    }

    /// Returns the next prime without metering, preserving the stream on allocation refusal.
    ///
    /// This is the fallible counterpart to [`Iterator::next`]. Size and table-allocation errors
    /// leave the cursor, emitted primes, and logical capacity unchanged, so callers may retry.
    pub fn try_next(&mut self) -> Result<BigInt, PrimeStreamError> {
        self.try_next_with_reserve(|emitted, additional| {
            emitted
                .try_reserve_exact(additional)
                .map_err(|_| PrimeStreamError::AllocationFailure)
        })
    }

    fn try_next_with_reserve<F>(&mut self, mut reserve: F) -> Result<BigInt, PrimeStreamError>
    where
        F: FnMut(&mut Vec<BigInt>, usize) -> Result<(), PrimeStreamError>,
    {
        let mut current = self.current.clone();
        loop {
            let candidate = current;
            let next_current = &candidate + 1i64;
            let root = sqrt_floor(&candidate).expect("prime-stream candidates stay positive");
            let divides = self.emitted.iter().take_while(|p| **p <= root).any(|p| {
                let remainder = &candidate % p;
                remainder.is_zero()
            });
            if divides {
                current = next_current;
                continue;
            }

            let required_len = self
                .emitted
                .len()
                .checked_add(1)
                .ok_or(PrimeStreamError::SizeOverflow)?;
            let mut next_capacity = self.emitted_capacity;
            if required_len > self.emitted_capacity {
                next_capacity = prime_table_growth_target(self.emitted_capacity, required_len)?;
                let additional = next_capacity
                    .checked_sub(self.emitted.len())
                    .ok_or(PrimeStreamError::SizeOverflow)?;
                reserve(&mut self.emitted, additional)?;
            }

            self.emitted.push(candidate.clone());
            self.emitted_capacity = next_capacity;
            self.current = next_current;
            return Ok(candidate);
        }
    }
}

impl Default for PrimeStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for PrimeStream {
    type Item = BigInt;

    fn next(&mut self) -> Option<BigInt> {
        let prime = self.try_next_with_reserve(|emitted, additional| {
            emitted.reserve_exact(additional);
            Ok(())
        });
        Some(prime.expect("infinite prime-stream iterator cannot exhaust"))
    }
}

const MR_BASES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Miller-Rabin primality with fixed deterministic bases.
pub fn is_probable_prime(n: &BigInt) -> bool {
    if *n < 2i64 {
        return false;
    }
    for base in MR_BASES {
        let b = BigInt::from(i64::from(base));
        if *n == b {
            return true;
        }
        if n % &b == BigInt::zero() {
            return false;
        }
    }

    let n_minus_1 = n - BigInt::one();
    let mut d = n_minus_1.clone();
    let mut s: u32 = 0;
    while (&d % 2i64).is_zero() {
        d /= 2i64;
        s += 1;
    }

    for base in MR_BASES {
        let a = BigInt::from(i64::from(base));
        if &a >= n {
            continue;
        }
        let mut x = mod_pow(&a, &d, n);
        if x.is_one() || x == n_minus_1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            x = mod_pow(&x, &BigInt::from(2i64), n);
            if x == n_minus_1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

/// Cancellation-first Miller-Rabin primality with the same fixed base set.
pub fn metered_is_probable_prime<M: BudgetMeter>(
    n: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    meter.checkpoint()?;
    if *n < 2i64 {
        return metered_finish(false, meter);
    }
    let Some(modulus) = NonZeroBigInt::new(n) else {
        return metered_finish(false, meter);
    };
    for base in MR_BASES {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let divisor = BigInt::from(i64::from(base));
        if *n == divisor {
            return metered_finish(true, meter);
        }
        let Some(divisor) = NonZeroBigInt::new(&divisor) else {
            return Ok(false);
        };
        let (_, remainder) = metered_div_rem_nonzero(n, divisor, meter)?;
        if remainder.is_zero() {
            return metered_finish(false, meter);
        }
    }

    let n_minus_one = metered_subtract(n, &BigInt::one(), meter)?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            n_minus_one.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut d = n_minus_one.clone();
    let two = BigInt::from(2i64);
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return Ok(false);
    };
    let mut s: u32 = 0;
    loop {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let (quotient, remainder) = metered_div_rem_nonzero(&d, two_divisor, meter)?;
        if !remainder.is_zero() {
            break;
        }
        d = quotient;
        s = s.saturating_add(1);
    }

    for base in MR_BASES {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let a = BigInt::from(i64::from(base));
        if metered_greater_or_equal(&a, n, meter)? {
            continue;
        }
        let mut x = metered_mod_pow(&a, &d, modulus, meter)?;
        if x.is_one() || metered_equal(&x, &n_minus_one, meter)? {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let square = metered_mul(&x, &x, meter)?;
            x = metered_normalized_remainder(&square, modulus, meter)?;
            if metered_equal(&x, &n_minus_one, meter)? {
                composite = false;
                break;
            }
        }
        if composite {
            return metered_finish(false, meter);
        }
    }
    metered_finish(true, meter)
}

/// Jacobi symbol `(a / n)` for a positive odd denominator.
///
/// Returns `None` when `n` is nonpositive or even. The symbol is otherwise one
/// of `-1`, `0`, or `1`; a zero result means that `a` and `n` are not coprime.
/// This definition includes `(a / 1) = 1`.
pub fn jacobi_symbol(a: &BigInt, n: &BigInt) -> Option<i8> {
    if !n.is_positive() || (n % 2i64).is_zero() {
        return None;
    }

    let mut numerator = normalized_remainder(a, n);
    let mut denominator = n.clone();
    let mut symbol = 1i8;

    while !numerator.is_zero() {
        while (&numerator % 2i64).is_zero() {
            numerator /= 2i64;
            let denominator_mod_eight = &denominator % 8i64;
            if denominator_mod_eight == 3i64 || denominator_mod_eight == 5i64 {
                symbol = -symbol;
            }
        }

        std::mem::swap(&mut numerator, &mut denominator);
        if (&numerator % 4i64) == 3i64 && (&denominator % 4i64) == 3i64 {
            symbol = -symbol;
        }
        numerator %= &denominator;
    }

    Some(if denominator.is_one() { symbol } else { 0 })
}

/// Cancellation-first Jacobi symbol using metered big-integer division lanes.
pub fn metered_jacobi_symbol<M: BudgetMeter>(
    a: &BigInt,
    n: &BigInt,
    meter: &mut M,
) -> Result<Option<i8>, MeterError> {
    meter.checkpoint()?;
    if !n.is_positive() {
        return metered_finish(None, meter);
    }

    let two = BigInt::from(2i64);
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return metered_finish(None, meter);
    };
    let (_, denominator_parity) = metered_div_rem_nonzero(n, two_divisor, meter)?;
    if denominator_parity.is_zero() {
        return metered_finish(None, meter);
    }
    let Some(denominator_divisor) = NonZeroBigInt::new(n) else {
        return metered_finish(None, meter);
    };

    let mut numerator = metered_normalized_remainder(a, denominator_divisor, meter)?;
    let mut denominator = metered_clone_bigint(n, meter)?;
    let eight = BigInt::from(8i64);
    let Some(eight_divisor) = NonZeroBigInt::new(&eight) else {
        return metered_finish(None, meter);
    };
    let four = BigInt::from(4i64);
    let Some(four_divisor) = NonZeroBigInt::new(&four) else {
        return metered_finish(None, meter);
    };
    let three = BigInt::from(3i64);
    let five = BigInt::from(5i64);
    let mut symbol = 1i8;

    while !numerator.is_zero() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        loop {
            meter.checkpoint()?;
            let (halved, parity) = metered_div_rem_nonzero(&numerator, two_divisor, meter)?;
            if !parity.is_zero() {
                break;
            }
            numerator = halved;
            let (_, denominator_mod_eight) =
                metered_div_rem_nonzero(&denominator, eight_divisor, meter)?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            if denominator_mod_eight == three || denominator_mod_eight == five {
                symbol = -symbol;
            }
        }

        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        std::mem::swap(&mut numerator, &mut denominator);
        let (_, numerator_mod_four) = metered_div_rem_nonzero(&numerator, four_divisor, meter)?;
        let (_, denominator_mod_four) = metered_div_rem_nonzero(&denominator, four_divisor, meter)?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        if numerator_mod_four == three && denominator_mod_four == three {
            symbol = -symbol;
        }
        let Some(next_denominator_divisor) = NonZeroBigInt::new(&denominator) else {
            return metered_finish(None, meter);
        };
        numerator = metered_normalized_remainder(&numerator, next_denominator_divisor, meter)?;
    }

    meter.charge(Dimension::ComputeSteps, denominator.limb_count().max(1))?;
    let result = if denominator.is_one() { symbol } else { 0 };
    metered_finish(Some(result), meter)
}

/// Legendre symbol `(a / p)` for an exactly admitted odd prime `p`.
///
/// The crate's fixed-base Miller-Rabin theorem is exact only below its
/// registered exclusive bound. Therefore this function refuses `p = 2`,
/// composites, and probable-prime candidates at or above that bound.
pub fn legendre_symbol(a: &BigInt, p: &BigInt) -> Option<i8> {
    if *p <= 2i64 || !is_certified_prime(p) {
        return None;
    }
    jacobi_symbol(a, p)
}

/// Cancellation-first Legendre symbol with the same exact-prime admission rule.
pub fn metered_legendre_symbol<M: BudgetMeter>(
    a: &BigInt,
    p: &BigInt,
    meter: &mut M,
) -> Result<Option<i8>, MeterError> {
    meter.checkpoint()?;
    if *p <= 2i64 {
        return metered_finish(None, meter);
    }
    let bound = deterministic_primality_bound();
    if metered_greater_or_equal(p, &bound, meter)? {
        return metered_finish(None, meter);
    }
    if !metered_is_probable_prime(p, meter)? {
        return metered_finish(None, meter);
    }
    let symbol = metered_jacobi_symbol(a, p, meter)?;
    metered_finish(symbol, meter)
}

fn mod_pow(base: &BigInt, exp: &BigInt, modulus: &BigInt) -> BigInt {
    if modulus.is_one() {
        return BigInt::zero();
    }
    let mut res = BigInt::one();
    let mut b = base % modulus;
    let mut e = exp.clone();
    let two = BigInt::from(2i64);

    while e.is_positive() {
        if !(&e % &two).is_zero() {
            res = &(&res * &b) % modulus;
        }
        b = &(&b * &b) % modulus;
        e = &e / &two;
    }
    res
}

fn metered_mod_pow<M: BudgetMeter>(
    base: &BigInt,
    exp: &BigInt,
    modulus: NonZeroBigInt<'_>,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    if modulus.get().is_one() {
        return Ok(BigInt::zero());
    }
    let mut result = BigInt::one();
    let mut base = metered_normalized_remainder(base, modulus, meter)?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            exp.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut exponent = exp.clone();
    let two = BigInt::from(2i64);
    let Some(two_divisor) = NonZeroBigInt::new(&two) else {
        return Ok(BigInt::zero());
    };

    while exponent.is_positive() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let (next_exponent, parity) = metered_div_rem_nonzero(&exponent, two_divisor, meter)?;
        if !parity.is_zero() {
            let product = metered_mul(&result, &base, meter)?;
            result = metered_normalized_remainder(&product, modulus, meter)?;
        }
        exponent = next_exponent;
        if exponent.is_positive() {
            let square = metered_mul(&base, &base, meter)?;
            base = metered_normalized_remainder(&square, modulus, meter)?;
        }
    }
    meter.checkpoint()?;
    Ok(result)
}

fn metered_normalized_remainder<M: BudgetMeter>(
    value: &BigInt,
    modulus: NonZeroBigInt<'_>,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    let (_, remainder) = metered_div_rem_nonzero(value, modulus, meter)?;
    if remainder.is_negative() {
        metered_add(&remainder, modulus.get(), meter)
    } else {
        Ok(remainder)
    }
}

/// Publishes a fully classified value only after one terminal cancellation checkpoint.
fn metered_finish<T, M: BudgetMeter>(value: T, meter: &mut M) -> Result<T, MeterError> {
    meter.checkpoint()?;
    Ok(value)
}

fn metered_equal<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    metered_compare(lhs, rhs, |ordering| ordering.is_eq(), meter)
}

fn metered_greater<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    metered_compare(lhs, rhs, |ordering| ordering.is_gt(), meter)
}

fn metered_greater_or_equal<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<bool, MeterError> {
    metered_compare(lhs, rhs, |ordering| ordering.is_ge(), meter)
}

fn metered_compare<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    predicate: impl FnOnce(std::cmp::Ordering) -> bool,
    meter: &mut M,
) -> Result<bool, MeterError> {
    metered_bigint_cmp(lhs, rhs, meter).map(predicate)
}

fn metered_add<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    metered_bigint_add(lhs, rhs, meter)
}

fn metered_negate<M: BudgetMeter>(value: BigInt, meter: &mut M) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge(Dimension::ComputeSteps, value.limb_count().max(1))?;
    let result = -value;
    meter.checkpoint()?;
    Ok(result)
}

fn metered_subtract<M: BudgetMeter>(
    lhs: &BigInt,
    rhs: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    metered_bigint_subtract(lhs, rhs, meter)
}

/// Exclusive upper bound for the fixed-base Miller-Rabin theorem used by this crate.
/// Values at or above this boundary remain probable-prime candidates, not exact field evidence.
fn deterministic_primality_bound() -> BigInt {
    BigInt::from(3_317_044_064_679_887_385u64) * BigInt::from(1_000_000u64)
        + BigInt::from(961_981u64)
}

fn is_certified_prime(characteristic: &BigInt) -> bool {
    characteristic > &BigInt::one()
        && characteristic < &deterministic_primality_bound()
        && is_probable_prime(characteristic)
}

fn is_canonical_residue(value: &BigInt, modulus: &BigInt) -> bool {
    !value.is_negative() && value < modulus
}

fn normalized_remainder(value: &BigInt, modulus: &BigInt) -> BigInt {
    let remainder = value % modulus;
    if remainder.is_negative() {
        remainder + modulus
    } else {
        remainder
    }
}

fn metered_clone_bigint<M: BudgetMeter>(
    value: &BigInt,
    meter: &mut M,
) -> Result<BigInt, MeterError> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            value.limb_count().max(1).saturating_mul(8),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let cloned = value.clone();
    meter.checkpoint()?;
    Ok(cloned)
}

fn batch_inverse_error<T, M: BudgetMeter>(
    error: BatchInverseError,
    meter: &mut M,
) -> Result<T, BatchInverseError> {
    meter.checkpoint()?;
    Err(error)
}

fn batch_buffer_layout(len: usize) -> Result<u64, BatchInverseError> {
    let prefix_slot = std::mem::size_of::<BigInt>();
    let output_slot = std::mem::size_of::<FiniteFieldElement>();
    let prefix_bytes = len
        .checked_mul(prefix_slot)
        .ok_or(BatchInverseError::SizeOverflow)?;
    let output_bytes = len
        .checked_mul(output_slot)
        .ok_or(BatchInverseError::SizeOverflow)?;
    let layout_ceiling = isize::MAX as usize;
    if prefix_bytes > layout_ceiling || output_bytes > layout_ceiling {
        return Err(BatchInverseError::SizeOverflow);
    }
    let total_bytes = prefix_bytes
        .checked_add(output_bytes)
        .ok_or(BatchInverseError::SizeOverflow)?;
    u64::try_from(total_bytes).map_err(|_| BatchInverseError::SizeOverflow)
}

fn reserve_batch_buffers<M: BudgetMeter>(
    len: usize,
    meter: &mut M,
) -> Result<(Vec<BigInt>, Vec<FiniteFieldElement>), BatchInverseError> {
    reserve_batch_buffers_with(
        len,
        meter,
        |prefixes, capacity| prefixes.try_reserve_exact(capacity).map_err(|_| ()),
        |output, capacity| output.try_reserve_exact(capacity).map_err(|_| ()),
    )
}

fn reserve_batch_buffers_with<M, P, O>(
    len: usize,
    meter: &mut M,
    mut reserve_prefixes: P,
    mut reserve_output: O,
) -> Result<(Vec<BigInt>, Vec<FiniteFieldElement>), BatchInverseError>
where
    M: BudgetMeter,
    P: FnMut(&mut Vec<BigInt>, usize) -> Result<(), ()>,
    O: FnMut(&mut Vec<FiniteFieldElement>, usize) -> Result<(), ()>,
{
    let buffer_bytes = match batch_buffer_layout(len) {
        Ok(bytes) => bytes,
        Err(error) => return batch_inverse_error(error, meter),
    };

    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::MemoryBytes, buffer_bytes),
        (Dimension::AllocationCount, 2),
    ])?;

    let mut prefixes = Vec::new();
    if reserve_prefixes(&mut prefixes, len).is_err() {
        return batch_inverse_error(BatchInverseError::AllocationFailure, meter);
    }
    meter.checkpoint()?;

    let mut output = Vec::new();
    if reserve_output(&mut output, len).is_err() {
        return batch_inverse_error(BatchInverseError::AllocationFailure, meter);
    }
    meter.checkpoint()?;
    Ok((prefixes, output))
}

/// Typed representation of a modular arithmetic residue ring $\mathbb{Z} / m\mathbb{Z}$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModularRing {
    modulus: BigInt,
}

impl ModularRing {
    /// Creates a new modular ring $\mathbb{Z} / m\mathbb{Z}$ for $m > 1$.
    pub fn new(modulus: BigInt) -> Option<Self> {
        if modulus > BigInt::one() {
            Some(Self { modulus })
        } else {
            None
        }
    }

    /// Access the modulus $m$.
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// Constructs a canonical element in $\mathbb{Z} / m\mathbb{Z}$ from an arbitrary integer.
    pub fn element(&self, value: BigInt) -> ModularRingElement {
        let residue = normalized_remainder(&value, &self.modulus);
        ModularRingElement {
            ring: self.clone(),
            value: residue,
        }
    }

    /// Cancellation-first construction of a canonical element from an arbitrary integer.
    pub fn metered_element<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<ModularRingElement, MeterError> {
        let Some(modulus) = NonZeroBigInt::new(&self.modulus) else {
            return metered_finish(self.zero(), meter);
        };
        let residue = metered_normalized_remainder(value, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.modulus, meter)?,
        };
        metered_finish(
            ModularRingElement {
                ring,
                value: residue,
            },
            meter,
        )
    }

    /// The additive identity $0 \pmod m$.
    pub fn zero(&self) -> ModularRingElement {
        self.element(BigInt::zero())
    }

    /// The multiplicative identity $1 \pmod m$.
    pub fn one(&self) -> ModularRingElement {
        self.element(BigInt::one())
    }
}

/// A typed canonical element in a modular ring $\mathbb{Z} / m\mathbb{Z}$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModularRingElement {
    ring: ModularRing,
    value: BigInt,
}

impl ModularRingElement {
    /// The canonical integer value in the range $[0, m)$.
    pub fn value(&self) -> &BigInt {
        &self.value
    }

    /// Reference to the parent modular ring.
    pub fn ring(&self) -> &ModularRing {
        &self.ring
    }

    /// Whether this element is $0 \pmod m$.
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Whether this element is $1 \pmod m$.
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Modular addition: $(a + b) \pmod m$.
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let m = &self.ring.modulus;
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: (&self.value + &other.value) % m,
        })
    }

    /// Modular subtraction: $(a - b) \pmod m$.
    pub fn sub(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let m = &self.ring.modulus;
        let mut diff = (&self.value - &other.value) % m;
        if diff.is_negative() {
            diff += m;
        }
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: diff,
        })
    }

    /// Modular multiplication: $(a \cdot b) \pmod m$.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let m = &self.ring.modulus;
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: (&self.value * &other.value) % m,
        })
    }

    /// Modular negation: $-a \pmod m$.
    pub fn neg(&self) -> Self {
        if self.value.is_zero() {
            self.clone()
        } else {
            ModularRingElement {
                ring: self.ring.clone(),
                value: &self.ring.modulus - &self.value,
            }
        }
    }

    /// Cancellation-first modular negation.
    pub fn metered_neg<M: BudgetMeter>(&self, meter: &mut M) -> Result<Self, MeterError> {
        meter.checkpoint()?;
        let value = if self.value.is_zero() {
            metered_clone_bigint(&self.value, meter)?
        } else {
            metered_subtract(&self.ring.modulus, &self.value, meter)?
        };
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Self { ring, value }, meter)
    }

    /// Multiplicative inverse $a^{-1} \pmod m$; returns `None` when $\gcd(a, m) \neq 1$.
    pub fn inv(&self) -> Option<Self> {
        let inv_val = mod_inverse(&self.value, &self.ring.modulus)?;
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: inv_val,
        })
    }

    /// Exact modular division: $a / b \pmod m \iff a \cdot b^{-1} \pmod m$.
    pub fn div(&self, other: &Self) -> Option<Self> {
        if self.ring != other.ring {
            return None;
        }
        let b_inv = other.inv()?;
        self.mul(&b_inv)
    }

    /// Modular exponentiation $a^e \pmod m$; negative exponents are refused because a ring
    /// element need not be a unit.
    pub fn pow(&self, exp: &BigInt) -> Option<Self> {
        if exp.is_negative() {
            return None;
        }
        Some(ModularRingElement {
            ring: self.ring.clone(),
            value: mod_pow(&self.value, exp, &self.ring.modulus),
        })
    }

    /// Cancellation-first modular addition. A different parent ring is a computed refusal.
    pub fn metered_add<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let sum = metered_add(&self.value, &other.value, meter)?;
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_normalized_remainder(&sum, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular subtraction. A different parent ring is refused.
    pub fn metered_sub<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let difference = metered_subtract(&self.value, &other.value, meter)?;
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_normalized_remainder(&difference, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular multiplication. A different parent ring is refused.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let product = metered_mul(&self.value, &other.value, meter)?;
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_normalized_remainder(&product, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular inverse; non-units produce `Ok(None)`.
    pub fn metered_inv<M: BudgetMeter>(&self, meter: &mut M) -> Result<Option<Self>, MeterError> {
        let Some(value) = metered_mod_inverse(&self.value, &self.ring.modulus, meter)? else {
            return metered_finish(None, meter);
        };
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }

    /// Cancellation-first modular division; mismatched rings and non-unit divisors are refused.
    pub fn metered_div<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(&self.ring.modulus, &other.ring.modulus, meter)? {
            return metered_finish(None, meter);
        }
        let Some(inverse) = other.metered_inv(meter)? else {
            return metered_finish(None, meter);
        };
        self.metered_mul(&inverse, meter)
    }

    /// Cancellation-first nonnegative modular exponentiation.
    pub fn metered_pow<M: BudgetMeter>(
        &self,
        exponent: &BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if exponent.is_negative() {
            return metered_finish(None, meter);
        }
        let Some(modulus) = NonZeroBigInt::new(&self.ring.modulus) else {
            return metered_finish(None, meter);
        };
        let value = metered_mod_pow(&self.value, exponent, modulus, meter)?;
        let ring = ModularRing {
            modulus: metered_clone_bigint(&self.ring.modulus, meter)?,
        };
        metered_finish(Some(Self { ring, value }), meter)
    }
}

/// Typed representation of a prime Galois field $\mathbb{F}_p = \mathbb{Z} / p\mathbb{Z}$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FiniteField {
    characteristic: BigInt,
}

impl FiniteField {
    /// Creates a prime finite field $\mathbb{F}_p$ only when the fixed-base primality theorem
    /// certifies `p` exactly. Larger probable primes are refused rather than promoted to fields.
    pub fn new(characteristic: BigInt) -> Option<Self> {
        if is_certified_prime(&characteristic) {
            Some(Self { characteristic })
        } else {
            None
        }
    }

    /// Cancellation-first exact field admission under the same deterministic theorem bound.
    pub fn metered_new<M: BudgetMeter>(
        characteristic: BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        let bound = deterministic_primality_bound();
        if characteristic <= BigInt::one()
            || metered_greater_or_equal(&characteristic, &bound, meter)?
        {
            return metered_finish(None, meter);
        }
        if !metered_is_probable_prime(&characteristic, meter)? {
            return metered_finish(None, meter);
        }
        metered_finish(Some(Self { characteristic }), meter)
    }

    /// Access the prime characteristic $p$.
    pub fn characteristic(&self) -> &BigInt {
        &self.characteristic
    }

    /// Constructs a canonical element in $\mathbb{F}_p$.
    pub fn element(&self, value: BigInt) -> FiniteFieldElement {
        let residue = normalized_remainder(&value, &self.characteristic);
        FiniteFieldElement {
            field: self.clone(),
            value: residue,
        }
    }

    /// Cancellation-first construction of a canonical field element.
    pub fn metered_element<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<FiniteFieldElement, MeterError> {
        let modulus =
            NonZeroBigInt::new(&self.characteristic).expect("FiniteField characteristic invariant");
        let residue = metered_normalized_remainder(value, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.characteristic, meter)?,
        };
        metered_finish(
            FiniteFieldElement {
                field,
                value: residue,
            },
            meter,
        )
    }

    /// The field additive identity $0$.
    pub fn zero(&self) -> FiniteFieldElement {
        self.element(BigInt::zero())
    }

    /// The field multiplicative identity $1$.
    pub fn one(&self) -> FiniteFieldElement {
        self.element(BigInt::one())
    }

    /// Inverts a batch of nonzero elements with one modular inverse.
    ///
    /// `Ok(None)` means that at least one input belongs to another field or is zero. Buffer-size
    /// and allocation refusals are typed, and no partial output is published.
    pub fn try_batch_inv(
        &self,
        values: &[FiniteFieldElement],
    ) -> Result<Option<Vec<FiniteFieldElement>>, BatchInverseError> {
        self.metered_batch_inv(values, &mut fsym_budget::Unbounded)
    }

    /// Cancellation-first batch inversion using one modular inverse and ordered prefix products.
    ///
    /// Membership and zero admission complete before either private output buffer is allocated.
    /// Cancellation, budget exhaustion, and typed internal failures drop all private work rather
    /// than exposing a partial vector.
    pub fn metered_batch_inv<M: BudgetMeter>(
        &self,
        values: &[FiniteFieldElement],
        meter: &mut M,
    ) -> Result<Option<Vec<FiniteFieldElement>>, BatchInverseError> {
        meter.checkpoint()?;
        if values.is_empty() {
            return metered_finish(Some(Vec::new()), meter).map_err(BatchInverseError::from);
        }

        for value in values {
            if !metered_equal(&self.characteristic, &value.field.characteristic, meter)? {
                return metered_finish(None, meter).map_err(BatchInverseError::from);
            }
        }
        for value in values {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            if value.is_zero() {
                return metered_finish(None, meter).map_err(BatchInverseError::from);
            }
        }

        let Some(modulus) = NonZeroBigInt::new(&self.characteristic) else {
            return batch_inverse_error(
                BatchInverseError::InvariantViolation("field characteristic is zero"),
                meter,
            );
        };
        let (mut prefixes, mut output) = reserve_batch_buffers(values.len(), meter)?;
        prefixes.push(metered_clone_bigint(&values[0].value, meter)?);

        for value in &values[1..] {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let product = metered_mul(
                prefixes
                    .last()
                    .expect("nonempty batch has an admitted prefix"),
                &value.value,
                meter,
            )?;
            prefixes.push(metered_normalized_remainder(&product, modulus, meter)?);
        }

        let Some(mut inverse_accumulator) = metered_mod_inverse(
            prefixes.last().expect("nonempty batch has a final prefix"),
            &self.characteristic,
            meter,
        )?
        else {
            return batch_inverse_error(
                BatchInverseError::InvariantViolation(
                    "nonzero product in a certified field was not invertible",
                ),
                meter,
            );
        };

        for index in (1..values.len()).rev() {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            let inverse_product = metered_mul(&inverse_accumulator, &prefixes[index - 1], meter)?;
            let inverse_value = metered_normalized_remainder(&inverse_product, modulus, meter)?;
            let next_product = metered_mul(&inverse_accumulator, &values[index].value, meter)?;
            let next_accumulator = metered_normalized_remainder(&next_product, modulus, meter)?;
            prefixes[index] = inverse_value;
            inverse_accumulator = next_accumulator;
        }
        prefixes[0] = inverse_accumulator;

        for value in prefixes {
            meter.checkpoint()?;
            meter.charge(Dimension::ComputeSteps, 1)?;
            output.push(FiniteFieldElement {
                field: FiniteField {
                    characteristic: metered_clone_bigint(&self.characteristic, meter)?,
                },
                value,
            });
        }
        metered_finish(Some(output), meter).map_err(BatchInverseError::from)
    }
}

/// A typed canonical element in a prime finite field $\mathbb{F}_p$.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FiniteFieldElement {
    field: FiniteField,
    value: BigInt,
}

impl FiniteFieldElement {
    /// The canonical integer representative in $[0, p)$.
    pub fn value(&self) -> &BigInt {
        &self.value
    }

    /// Reference to the underlying finite field.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Whether this element is $0$.
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Whether this element is $1$.
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Field addition $(a + b) \pmod p$.
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.field != other.field {
            return None;
        }
        let p = &self.field.characteristic;
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: (&self.value + &other.value) % p,
        })
    }

    /// Field subtraction $(a - b) \pmod p$.
    pub fn sub(&self, other: &Self) -> Option<Self> {
        if self.field != other.field {
            return None;
        }
        let p = &self.field.characteristic;
        let mut diff = (&self.value - &other.value) % p;
        if diff.is_negative() {
            diff += p;
        }
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: diff,
        })
    }

    /// Field multiplication $(a \cdot b) \pmod p$.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.field != other.field {
            return None;
        }
        let p = &self.field.characteristic;
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: (&self.value * &other.value) % p,
        })
    }

    /// Field negation $-a \pmod p$.
    pub fn neg(&self) -> Self {
        if self.value.is_zero() {
            self.clone()
        } else {
            FiniteFieldElement {
                field: self.field.clone(),
                value: &self.field.characteristic - &self.value,
            }
        }
    }

    /// Cancellation-first field negation.
    pub fn metered_neg<M: BudgetMeter>(&self, meter: &mut M) -> Result<Self, MeterError> {
        meter.checkpoint()?;
        let value = if self.value.is_zero() {
            metered_clone_bigint(&self.value, meter)?
        } else {
            metered_subtract(&self.field.characteristic, &self.value, meter)?
        };
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Self { field, value }, meter)
    }

    /// Multiplicative inverse $a^{-1} \pmod p$; returns `None` only for $0$.
    pub fn inv(&self) -> Option<Self> {
        if self.value.is_zero() {
            return None;
        }
        let inv_val = mod_inverse(&self.value, &self.field.characteristic)?;
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: inv_val,
        })
    }

    /// Field division $a / b \pmod p$; returns `None` when $b = 0$ or fields mismatch.
    pub fn div(&self, other: &Self) -> Option<Self> {
        if self.field != other.field || other.is_zero() {
            return None;
        }
        let b_inv = other.inv()?;
        self.mul(&b_inv)
    }

    /// Nonnegative exponentiation $a^e \pmod p$. Negative exponents are refused explicitly;
    /// callers may invert a nonzero value and then exponentiate.
    pub fn pow(&self, exp: &BigInt) -> Option<Self> {
        if exp.is_negative() {
            return None;
        }
        Some(FiniteFieldElement {
            field: self.field.clone(),
            value: mod_pow(&self.value, exp, &self.field.characteristic),
        })
    }

    /// Cancellation-first field addition. A different parent field is refused.
    pub fn metered_add<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? {
            return metered_finish(None, meter);
        }
        let sum = metered_add(&self.value, &other.value, meter)?;
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_normalized_remainder(&sum, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first field subtraction. A different parent field is refused.
    pub fn metered_sub<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? {
            return metered_finish(None, meter);
        }
        let difference = metered_subtract(&self.value, &other.value, meter)?;
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_normalized_remainder(&difference, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first field multiplication. A different parent field is refused.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? {
            return metered_finish(None, meter);
        }
        let product = metered_mul(&self.value, &other.value, meter)?;
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_normalized_remainder(&product, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first multiplicative inverse. Zero is a computed refusal.
    pub fn metered_inv<M: BudgetMeter>(&self, meter: &mut M) -> Result<Option<Self>, MeterError> {
        if self.value.is_zero() {
            meter.checkpoint()?;
            return metered_finish(None, meter);
        }
        let Some(value) = metered_mod_inverse(&self.value, &self.field.characteristic, meter)?
        else {
            return metered_finish(None, meter);
        };
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }

    /// Cancellation-first field division. Mismatched fields and zero divisors are refused.
    pub fn metered_div<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if !metered_equal(
            &self.field.characteristic,
            &other.field.characteristic,
            meter,
        )? || other.value.is_zero()
        {
            return metered_finish(None, meter);
        }
        let Some(inverse) = other.metered_inv(meter)? else {
            return metered_finish(None, meter);
        };
        self.metered_mul(&inverse, meter)
    }

    /// Cancellation-first nonnegative field exponentiation.
    pub fn metered_pow<M: BudgetMeter>(
        &self,
        exponent: &BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if exponent.is_negative() {
            return metered_finish(None, meter);
        }
        let modulus = NonZeroBigInt::new(&self.field.characteristic)
            .expect("FiniteField characteristic invariant");
        let value = metered_mod_pow(&self.value, exponent, modulus, meter)?;
        let field = FiniteField {
            characteristic: metered_clone_bigint(&self.field.characteristic, meter)?,
        };
        metered_finish(Some(Self { field, value }), meter)
    }
}

/// Montgomery representation reducer for fast modular arithmetic modulo an odd integer $M$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryReducer {
    modulus: BigInt,
    r: BigInt,
    reduction_bound: BigInt,
    r_shift: u32,
    r2_mod_m: BigInt,
    m_prime: BigInt,
}

impl MontgomeryReducer {
    /// Creates a Montgomery reducer for an odd modulus $M > 1$.
    pub fn new(modulus: BigInt) -> Option<Self> {
        if modulus <= BigInt::one() || (&modulus % &BigInt::from(2i64)).is_zero() {
            return None;
        }
        let bit_len = u32::try_from(modulus.bits()).ok()?;
        let r_shift = bit_len.checked_add(1)?;
        let r = BigInt::one() << r_shift;
        let r_mod_m = &r % &modulus;
        let r2_mod_m = (&r_mod_m * &r_mod_m) % &modulus;
        let m_inv = mod_inverse(&modulus, &r)?;
        let m_prime = &r - m_inv;
        let reduction_bound = &modulus * &r;
        Some(Self {
            modulus,
            r,
            reduction_bound,
            r_shift,
            r2_mod_m,
            m_prime,
        })
    }

    /// Cancellation-first reducer construction. Power-of-two setup uses the governed bigint
    /// binary-power lane, which exposes safe points during each multiply and square.
    pub fn metered_new<M: BudgetMeter>(
        modulus: BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if modulus <= BigInt::one() {
            return metered_finish(None, meter);
        }
        let two = BigInt::from(2i64);
        let two_divisor = NonZeroBigInt::new(&two).expect("two is nonzero");
        let (_, parity) = metered_div_rem_nonzero(&modulus, two_divisor, meter)?;
        if parity.is_zero() {
            return metered_finish(None, meter);
        }
        let Some(r_shift) = u32::try_from(modulus.bits())
            .ok()
            .and_then(|bits| bits.checked_add(1))
        else {
            return metered_finish(None, meter);
        };
        let r = metered_bigint_pow(&two, r_shift, meter)?;
        let (_, r_mod_m) = {
            let modulus_divisor =
                NonZeroBigInt::new(&modulus).expect("admitted Montgomery modulus is nonzero");
            metered_div_rem_nonzero(&r, modulus_divisor, meter)?
        };
        let r_squared = metered_mul(&r_mod_m, &r_mod_m, meter)?;
        let r2_mod_m = {
            let modulus_divisor =
                NonZeroBigInt::new(&modulus).expect("admitted Montgomery modulus is nonzero");
            metered_normalized_remainder(&r_squared, modulus_divisor, meter)?
        };
        let Some(m_inv) = metered_mod_inverse(&modulus, &r, meter)? else {
            return metered_finish(None, meter);
        };
        let m_prime = metered_subtract(&r, &m_inv, meter)?;
        let reduction_bound = metered_mul(&modulus, &r, meter)?;
        let reducer = Self {
            modulus,
            r,
            reduction_bound,
            r_shift,
            r2_mod_m,
            m_prime,
        };
        metered_finish(Some(reducer), meter)
    }

    /// Access the odd modulus.
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// Converts any signed integer into canonical Montgomery form $a \cdot R \pmod M$.
    pub fn to_montgomery(&self, a: &BigInt) -> BigInt {
        let canonical = normalized_remainder(a, &self.modulus);
        self.reduce_admitted(&(&canonical * &self.r2_mod_m))
    }

    /// Cancellation-first conversion into canonical Montgomery form.
    pub fn metered_to_montgomery<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        let modulus = NonZeroBigInt::new(&self.modulus).expect("Montgomery modulus invariant");
        let canonical = metered_normalized_remainder(value, modulus, meter)?;
        let product = metered_mul(&canonical, &self.r2_mod_m, meter)?;
        self.metered_reduce(&product, meter)
    }

    /// Converts a canonical Montgomery residue back to a standard representative.
    pub fn from_montgomery(&self, value: &BigInt) -> Option<BigInt> {
        is_canonical_residue(value, &self.modulus).then(|| self.reduce_admitted(value))
    }

    /// Cancellation-first conversion back from a canonical Montgomery residue.
    pub fn metered_from_montgomery<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if value.is_negative() || metered_greater_or_equal(value, &self.modulus, meter)? {
            return metered_finish(None, meter);
        }
        self.metered_reduce(value, meter)
    }

    /// Montgomery reduction for the required domain `0 <= T < M*R`. Inputs outside that domain
    /// are refused instead of producing a noncanonical or mathematically invalid value.
    pub fn reduce(&self, value: &BigInt) -> Option<BigInt> {
        if value.is_negative() || value >= &self.reduction_bound {
            return None;
        }
        Some(self.reduce_admitted(value))
    }

    fn reduce_admitted(&self, value: &BigInt) -> BigInt {
        let prod = value * &self.m_prime;
        let r_minus_1 = (&self.r) - BigInt::one();
        let m = &prod & &r_minus_1;
        let u = (value + &m * &self.modulus) >> self.r_shift;
        if u >= self.modulus {
            u - &self.modulus
        } else {
            u
        }
    }

    /// Cancellation-first Montgomery reduction over the same admitted domain.
    pub fn metered_reduce<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if value.is_negative() || metered_greater_or_equal(value, &self.reduction_bound, meter)? {
            return metered_finish(None, meter);
        }
        let product = metered_mul(value, &self.m_prime, meter)?;
        let r_divisor = NonZeroBigInt::new(&self.r).expect("Montgomery R invariant");
        let m = metered_normalized_remainder(&product, r_divisor, meter)?;
        let correction = metered_mul(&m, &self.modulus, meter)?;
        let numerator = metered_add(value, &correction, meter)?;
        let (mut reduced, remainder) = metered_div_rem_nonzero(&numerator, r_divisor, meter)?;
        if !remainder.is_zero() {
            return metered_finish(None, meter);
        }
        if metered_greater_or_equal(&reduced, &self.modulus, meter)? {
            reduced = metered_subtract(&reduced, &self.modulus, meter)?;
        }
        metered_finish(Some(reduced), meter)
    }

    /// Montgomery multiplication over two canonical Montgomery residues.
    pub fn mul(&self, lhs: &BigInt, rhs: &BigInt) -> Option<BigInt> {
        if !is_canonical_residue(lhs, &self.modulus) || !is_canonical_residue(rhs, &self.modulus) {
            return None;
        }
        self.reduce(&(lhs * rhs))
    }

    /// Cancellation-first Montgomery multiplication over canonical operands.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        lhs: &BigInt,
        rhs: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if lhs.is_negative()
            || rhs.is_negative()
            || metered_greater_or_equal(lhs, &self.modulus, meter)?
            || metered_greater_or_equal(rhs, &self.modulus, meter)?
        {
            return metered_finish(None, meter);
        }
        let product = metered_mul(lhs, rhs, meter)?;
        self.metered_reduce(&product, meter)
    }
}

/// Barrett reducer for division-free reduction modulo $M$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarrettReducer {
    modulus: BigInt,
    k: u32,
    mu: BigInt,
    modulus_squared: BigInt,
    b_k_minus_one: BigInt,
    b_k_plus_one: BigInt,
}

impl BarrettReducer {
    /// Creates a Barrett reducer for modulus $M > 1$.
    pub fn new(modulus: BigInt) -> Option<Self> {
        if modulus <= BigInt::one() {
            return None;
        }
        let k = u32::try_from(modulus.bits()).ok()?;
        let two_k = k.checked_mul(2)?;
        let k_minus_one = k.checked_sub(1)?;
        let k_plus_one = k.checked_add(1)?;
        let num = BigInt::one() << two_k;
        let mu = num / &modulus;
        let modulus_squared = &modulus * &modulus;
        let b_k_minus_one = BigInt::one() << k_minus_one;
        let b_k_plus_one = BigInt::one() << k_plus_one;
        Some(Self {
            modulus,
            k,
            mu,
            modulus_squared,
            b_k_minus_one,
            b_k_plus_one,
        })
    }

    /// Cancellation-first reducer construction with metered power-of-two precomputation.
    pub fn metered_new<M: BudgetMeter>(
        modulus: BigInt,
        meter: &mut M,
    ) -> Result<Option<Self>, MeterError> {
        meter.checkpoint()?;
        if modulus <= BigInt::one() {
            return metered_finish(None, meter);
        }
        let Some(k) = u32::try_from(modulus.bits()).ok() else {
            return metered_finish(None, meter);
        };
        let Some(two_k) = k.checked_mul(2) else {
            return metered_finish(None, meter);
        };
        let Some(k_minus_one) = k.checked_sub(1) else {
            return metered_finish(None, meter);
        };
        let Some(k_plus_one) = k.checked_add(1) else {
            return metered_finish(None, meter);
        };
        let two = BigInt::from(2i64);
        let numerator = metered_bigint_pow(&two, two_k, meter)?;
        let modulus_divisor =
            NonZeroBigInt::new(&modulus).expect("admitted Barrett modulus is nonzero");
        let (mu, _) = metered_div_rem_nonzero(&numerator, modulus_divisor, meter)?;
        let modulus_squared = metered_mul(&modulus, &modulus, meter)?;
        let b_k_minus_one = metered_bigint_pow(&two, k_minus_one, meter)?;
        let b_k_plus_one = metered_bigint_pow(&two, k_plus_one, meter)?;
        metered_finish(
            Some(Self {
                modulus,
                k,
                mu,
                modulus_squared,
                b_k_minus_one,
                b_k_plus_one,
            }),
            meter,
        )
    }

    /// Access the positive modulus.
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// Reduces `value` modulo $M$ for the admitted Barrett domain `0 <= value < M^2`.
    /// Negative and out-of-range inputs are refused, making the correction loop bounded.
    pub fn reduce(&self, value: &BigInt) -> Option<BigInt> {
        if value.is_negative() || value >= &self.modulus_squared {
            return None;
        }
        if value < &self.modulus {
            return Some(value.clone());
        }
        let q1 = value >> (self.k - 1);
        let q2 = &q1 * &self.mu;
        let q3 = q2 >> (self.k + 1);
        let mut r = value - &q3 * &self.modulus;
        if r.is_negative() {
            return None;
        }
        for _ in 0..3 {
            if r < self.modulus {
                return Some(r);
            }
            r -= &self.modulus;
        }
        (r < self.modulus).then_some(r)
    }

    /// Cancellation-first Barrett reduction over the same bounded domain.
    pub fn metered_reduce<M: BudgetMeter>(
        &self,
        value: &BigInt,
        meter: &mut M,
    ) -> Result<Option<BigInt>, MeterError> {
        meter.checkpoint()?;
        if value.is_negative() || metered_greater_or_equal(value, &self.modulus_squared, meter)? {
            return metered_finish(None, meter);
        }
        if !metered_greater_or_equal(value, &self.modulus, meter)? {
            let cloned = metered_clone_bigint(value, meter)?;
            return metered_finish(Some(cloned), meter);
        }
        let k_minus_divisor =
            NonZeroBigInt::new(&self.b_k_minus_one).expect("Barrett power invariant");
        let (q1, _) = metered_div_rem_nonzero(value, k_minus_divisor, meter)?;
        let q2 = metered_mul(&q1, &self.mu, meter)?;
        let k_plus_divisor =
            NonZeroBigInt::new(&self.b_k_plus_one).expect("Barrett power invariant");
        let (q3, _) = metered_div_rem_nonzero(&q2, k_plus_divisor, meter)?;
        let product = metered_mul(&q3, &self.modulus, meter)?;
        let mut reduced = metered_subtract(value, &product, meter)?;
        if reduced.is_negative() {
            return metered_finish(None, meter);
        }
        for _ in 0..3 {
            meter.checkpoint()?;
            if !metered_greater_or_equal(&reduced, &self.modulus, meter)? {
                return metered_finish(Some(reduced), meter);
            }
            reduced = metered_subtract(&reduced, &self.modulus, meter)?;
        }
        if metered_greater_or_equal(&reduced, &self.modulus, meter)? {
            metered_finish(None, meter)
        } else {
            metered_finish(Some(reduced), meter)
        }
    }
}

/// Classification of unlucky prime failures during modular algorithms (e.g. modular GCD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnluckyPrimeReason {
    /// Candidate is nonpositive, composite, or above the exact fixed-base theorem range.
    InvalidPrimeCandidate,
    /// Prime divides the leading coefficient of one or more input polynomials.
    DividesLeadingCoefficient,
    /// Modular reduction causes degree collapse or degenerate structures.
    DegenerateReduction,
    /// Inconsistent modular residues during CRT combination.
    InconsistentResidues,
    /// Prime characteristic is smaller than algorithm coefficient bound.
    ModulusTooSmall,
}

/// Diagnostic record explaining why a chosen prime is unusable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnluckyPrimeDiagnostic {
    /// Rejected candidate; this is diagnostic data, never prime evidence.
    pub prime: BigInt,
    /// Structured refusal class.
    pub reason: UnluckyPrimeReason,
    /// Index of the offending leading coefficient when the reason is coefficient-specific.
    pub coefficient_index: Option<usize>,
}

fn unlucky_prime_diagnostic(
    prime: &BigInt,
    reason: UnluckyPrimeReason,
    coefficient_index: Option<usize>,
) -> UnluckyPrimeDiagnostic {
    UnluckyPrimeDiagnostic {
        prime: prime.clone(),
        reason,
        coefficient_index,
    }
}

/// Verifies an exactly admitted prime against polynomial leading coefficients.
///
/// This bounded diagnostic check does not establish that a prime will be lucky for every later
/// modular-algorithm phase; it only rejects the declared leading-coefficient obstruction.
pub fn check_lucky_prime(
    prime: &BigInt,
    leading_coefficients: &[BigInt],
) -> Result<(), UnluckyPrimeDiagnostic> {
    if !is_certified_prime(prime) {
        return Err(unlucky_prime_diagnostic(
            prime,
            UnluckyPrimeReason::InvalidPrimeCandidate,
            None,
        ));
    }
    for (idx, coeff) in leading_coefficients.iter().enumerate() {
        if (coeff % prime).is_zero() {
            return Err(unlucky_prime_diagnostic(
                prime,
                UnluckyPrimeReason::DividesLeadingCoefficient,
                Some(idx),
            ));
        }
    }
    Ok(())
}

/// Cancellation-first form of [`check_lucky_prime`].
pub fn metered_check_lucky_prime<M: BudgetMeter>(
    prime: &BigInt,
    leading_coefficients: &[BigInt],
    meter: &mut M,
) -> Result<Result<(), UnluckyPrimeDiagnostic>, MeterError> {
    meter.checkpoint()?;
    let bound = deterministic_primality_bound();
    let invalid = prime <= &BigInt::one()
        || metered_greater_or_equal(prime, &bound, meter)?
        || !metered_is_probable_prime(prime, meter)?;
    if invalid {
        let diagnostic = UnluckyPrimeDiagnostic {
            prime: metered_clone_bigint(prime, meter)?,
            reason: UnluckyPrimeReason::InvalidPrimeCandidate,
            coefficient_index: None,
        };
        return metered_finish(Err(diagnostic), meter);
    }
    let divisor = NonZeroBigInt::new(prime).expect("certified prime is nonzero");
    for (index, coefficient) in leading_coefficients.iter().enumerate() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        let (_, remainder) = metered_div_rem_nonzero(coefficient, divisor, meter)?;
        if remainder.is_zero() {
            let diagnostic = UnluckyPrimeDiagnostic {
                prime: metered_clone_bigint(prime, meter)?,
                reason: UnluckyPrimeReason::DividesLeadingCoefficient,
                coefficient_index: Some(index),
            };
            return metered_finish(Err(diagnostic), meter);
        }
    }
    metered_finish(Ok(()), meter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetError, BudgetLimits, DIMENSION_COUNT, Unbounded};
    use proptest::prelude::*;

    #[derive(Debug, Default)]
    struct CheckpointMeter {
        checkpoints: usize,
        cancel_at: Option<usize>,
        arm_after: Option<usize>,
        armed: bool,
        charged: bool,
    }

    impl CheckpointMeter {
        fn cancelling_at(checkpoint: usize) -> Self {
            Self {
                checkpoints: 0,
                cancel_at: Some(checkpoint.max(1)),
                arm_after: None,
                armed: false,
                charged: false,
            }
        }

        fn arming_after(checkpoint: usize) -> Self {
            Self {
                checkpoints: 0,
                cancel_at: None,
                arm_after: Some(checkpoint),
                armed: false,
                charged: false,
            }
        }
    }

    impl BudgetMeter for CheckpointMeter {
        fn charge(&mut self, _dimension: Dimension, amount: u64) -> Result<(), MeterError> {
            self.charged |= amount != 0;
            Ok(())
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            self.charged |= charges.iter().any(|(_, amount)| *amount != 0);
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.armed {
                return Err(MeterError::Cancelled);
            }
            self.checkpoints = self.checkpoints.saturating_add(1);
            if self.cancel_at == Some(self.checkpoints) {
                Err(MeterError::Cancelled)
            } else {
                if self.arm_after == Some(self.checkpoints) {
                    self.armed = true;
                }
                Ok(())
            }
        }
    }

    #[derive(Debug, Default)]
    struct CountingMeter {
        dimensions: [u64; DIMENSION_COUNT],
        checkpoints: usize,
    }

    impl BudgetMeter for CountingMeter {
        fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
            self.charge_batch(&[(dimension, amount)])
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            let mut updated = self.dimensions;
            for &(dimension, amount) in charges {
                let slot = &mut updated[dimension.index()];
                *slot = slot.checked_add(amount).ok_or(MeterError::Budget(
                    BudgetError::ChargeOverflow { dimension },
                ))?;
            }
            self.dimensions = updated;
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints = self
                .checkpoints
                .checked_add(1)
                .expect("test checkpoint count must fit usize");
            Ok(())
        }
    }

    fn scalar_gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    fn prime_stream_before_four_to_eight_growth() -> PrimeStream {
        let mut stream = PrimeStream::new();
        for expected in [2i64, 3, 5, 7] {
            assert_eq!(stream.next(), Some(BigInt::from(expected)));
        }
        assert_eq!(stream.emitted.len(), 4);
        assert_eq!(stream.emitted_capacity, 4);
        stream
    }

    fn scalar_is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        let mut divisor = 2u64;
        while divisor <= n / divisor {
            if n.is_multiple_of(divisor) {
                return false;
            }
            divisor += 1;
        }
        true
    }

    fn terminal_base_pseudoprime() -> BigInt {
        let factors = [399_165_290_221u64, 798_330_580_441u64];
        assert!(factors.into_iter().all(scalar_is_prime));
        let composite = BigInt::from(factors[0]) * BigInt::from(factors[1]);
        let decimal = BigInt::from(318_665_857_834_031_151u64) * BigInt::from(1_000_000u64)
            + BigInt::from(167_461u64);
        assert_eq!(composite, decimal);
        composite
    }

    fn scalar_mod_pow(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
        let mut result = 1u64 % modulus;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = ((u128::from(result) * u128::from(base)) % u128::from(modulus)) as u64;
            }
            base = ((u128::from(base) * u128::from(base)) % u128::from(modulus)) as u64;
            exponent >>= 1;
        }
        result
    }

    fn scalar_legendre_prime(a: i64, prime: u64) -> i8 {
        debug_assert!(scalar_is_prime(prime));
        debug_assert!(prime > 2);
        let residue = a.rem_euclid(prime as i64) as u64;
        if residue == 0 {
            return 0;
        }
        let criterion = scalar_mod_pow(residue, (prime - 1) / 2, prime);
        if criterion == 1 {
            1
        } else {
            assert_eq!(criterion, prime - 1);
            -1
        }
    }

    fn scalar_jacobi_by_factorization(a: i64, mut denominator: u64) -> i8 {
        debug_assert!(denominator > 0 && denominator % 2 == 1);
        let mut symbol = 1i8;
        let mut prime = 3u64;
        while prime <= denominator / prime {
            while denominator.is_multiple_of(prime) {
                symbol *= scalar_legendre_prime(a, prime);
                denominator /= prime;
            }
            prime += 2;
        }
        if denominator > 1 {
            symbol *= scalar_legendre_prime(a, denominator);
        }
        symbol
    }

    fn assert_terminal_checkpoint<T: std::fmt::Debug + PartialEq>(
        expected: T,
        expected_checkpoints: usize,
        mut operation: impl FnMut(&mut CheckpointMeter) -> Result<T, MeterError>,
    ) {
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            operation(&mut measured).expect("measurement run completes"),
            expected
        );
        assert_eq!(measured.checkpoints, expected_checkpoints);

        let mut cancelled = CheckpointMeter::arming_after(expected_checkpoints - 1);
        assert!(matches!(
            operation(&mut cancelled),
            Err(MeterError::Cancelled)
        ));
    }

    fn assert_uncharged_fast_terminal<T: std::fmt::Debug + PartialEq>(
        expected: T,
        mut operation: impl FnMut(&mut CheckpointMeter) -> Result<T, MeterError>,
    ) {
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            operation(&mut measured).expect("fast terminal path completes"),
            expected
        );
        assert_eq!(measured.checkpoints, 2);
        assert!(!measured.charged);

        let mut cancelled = CheckpointMeter::arming_after(1);
        assert!(matches!(
            operation(&mut cancelled),
            Err(MeterError::Cancelled)
        ));
        assert_eq!(cancelled.checkpoints, 1);
        assert!(!cancelled.charged);
    }

    #[test]
    fn known_gcd_and_bezout_identity() {
        let (g, x, y) = extended_gcd(&BigInt::from(240i64), &BigInt::from(46i64));
        assert_eq!(g, BigInt::from(2i64));
        let lhs = BigInt::from(240i64) * x + BigInt::from(46i64) * y;
        assert_eq!(lhs, g);
        assert_eq!(
            gcd(&BigInt::from(0i64), &BigInt::from(7i64)),
            BigInt::from(7i64)
        );
        assert_eq!(
            gcd(&BigInt::from(0i64), &BigInt::from(0i64)),
            BigInt::from(0i64)
        );
    }

    #[test]
    fn invalid_first_crt_modulus_is_refused_without_division() {
        assert_eq!(crt(&[(1.into(), 0.into())]), None);
        assert_eq!(crt(&[(1.into(), (-7).into())]), None);
    }

    #[test]
    fn crt_preflights_late_moduli_before_owned_numeric_work() {
        let large_remainder = (BigInt::one() << 4_096u32) + 1i64;
        let congruences = [
            (large_remainder, BigInt::from(3)),
            (BigInt::from(1), BigInt::from(5)),
            (BigInt::from(2), BigInt::zero()),
        ];

        assert_eq!(crt(&congruences), None);

        let mut meter = CountingMeter::default();
        assert_eq!(metered_crt(&congruences, &mut meter), Ok(None));
        assert_eq!(
            meter.dimensions,
            [2, 0, 0, 0, 0],
            "modulus preflight must not clone or normalize the large first remainder"
        );
        assert_eq!(meter.checkpoints, 4);

        let mut cancelled = CheckpointMeter::cancelling_at(2);
        assert_eq!(
            metered_crt(&congruences, &mut cancelled),
            Err(MeterError::Cancelled)
        );
    }

    #[test]
    fn coprime_split_crt_preserves_strict_and_generalized_boundaries() {
        let remainders = [BigInt::from(-1), BigInt::from(3), BigInt::from(9)];
        let moduli = [BigInt::from(5), BigInt::from(7), BigInt::from(11)];
        let owned = remainders
            .iter()
            .cloned()
            .zip(moduli.iter().cloned())
            .collect::<Vec<_>>();
        assert_eq!(crt_coprime_slices(&remainders, &moduli), crt(&owned));

        let non_coprime_moduli = [BigInt::from(6), BigInt::from(35), BigInt::from(10)];
        assert_eq!(
            crt_coprime_slices(&remainders, &non_coprime_moduli),
            None,
            "the accumulated modulus must expose a factor shared with any prior modulus"
        );

        let generalized = [
            (BigInt::from(1), BigInt::from(2)),
            (BigInt::from(1), BigInt::from(4)),
        ];
        assert_eq!(crt(&generalized), Some((BigInt::from(1), BigInt::from(4))));
        assert_eq!(
            crt_coprime_slices(
                &[BigInt::from(1), BigInt::from(1)],
                &[BigInt::from(2), BigInt::from(4)]
            ),
            None
        );
        assert_eq!(crt_coprime_slices(&[], &[]), Some((0.into(), 1.into())));
        assert_eq!(crt_coprime_slices(&[0.into()], &[]), None);
        assert_eq!(crt_coprime_slices(&[0.into()], &[0.into()]), None);
    }

    #[test]
    fn rational_reconstruction_handles_zero_and_refuses_degenerate_moduli() {
        assert_eq!(
            rational_reconstruct(&BigInt::zero(), &BigInt::from(101)),
            Some((BigInt::zero(), BigInt::one()))
        );
        assert_eq!(
            rational_reconstruct(&BigInt::from(17), &BigInt::one()),
            None
        );
        assert_eq!(
            rational_reconstruct(&BigInt::from(17), &BigInt::zero()),
            None
        );
    }

    #[test]
    fn rational_reconstruction_zero_respects_the_declared_uniqueness_bound() {
        let zero_mod_two = (BigInt::from(2), BigInt::from(2));
        assert_eq!(rational_reconstruct(&zero_mod_two.0, &zero_mod_two.1), None);

        let mut measured = CheckpointMeter::default();
        assert_eq!(
            metered_rational_reconstruct(&zero_mod_two.0, &zero_mod_two.1, &mut measured).unwrap(),
            None
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            metered_rational_reconstruct(&zero_mod_two.0, &zero_mod_two.1, &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let zero_mod_three = (BigInt::from(3), BigInt::from(3));
        let expected = Some((BigInt::zero(), BigInt::one()));
        assert_eq!(
            rational_reconstruct(&zero_mod_three.0, &zero_mod_three.1),
            expected
        );
        let mut meter = Unbounded;
        assert_eq!(
            metered_rational_reconstruct(&zero_mod_three.0, &zero_mod_three.1, &mut meter).unwrap(),
            expected
        );
    }

    #[test]
    fn published_rational_reconstructions_satisfy_the_declared_bounds() {
        for modulus_value in 2i64..65 {
            let modulus = BigInt::from(modulus_value);
            let bound = sqrt_floor(&((&modulus - 1i64) / 2i64))
                .expect("admitted modulus gives a nonnegative bound radicand");
            for residue_value in -128i64..=128 {
                let residue = BigInt::from(residue_value);
                let result = rational_reconstruct(&residue, &modulus);
                let mut meter = Unbounded;
                assert_eq!(
                    metered_rational_reconstruct(&residue, &modulus, &mut meter).unwrap(),
                    result
                );
                if let Some((numerator, denominator)) = result {
                    assert!(denominator.is_positive());
                    assert!(numerator.abs() <= bound);
                    assert!(denominator <= bound);
                    assert_eq!(gcd(&numerator, &denominator), BigInt::one());
                    assert_eq!(
                        (&numerator - &residue * &denominator) % &modulus,
                        BigInt::zero()
                    );
                }
            }
        }
    }

    #[test]
    fn prime_stream_and_miller_rabin_match_independent_trial_division() {
        let expected: Vec<u64> = (2..)
            .filter(|value| scalar_is_prime(*value))
            .take(100)
            .collect();
        let actual: Vec<u64> = PrimeStream::new()
            .take(100)
            .map(|value| value.to_u64().unwrap())
            .collect();
        assert_eq!(actual, expected);

        let mut metered_stream = PrimeStream::new();
        let mut meter = Unbounded;
        let metered: Vec<u64> = (0..100)
            .map(|_| {
                metered_stream
                    .next_metered(&mut meter)
                    .unwrap()
                    .to_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(metered, expected);

        for value in 0..10_000u64 {
            assert_eq!(
                is_probable_prime(&BigInt::from(value)),
                scalar_is_prime(value),
                "primality mismatch for {value}"
            );
            if value < 1_000 {
                let mut meter = Unbounded;
                assert_eq!(
                    metered_is_probable_prime(&BigInt::from(value), &mut meter).unwrap(),
                    scalar_is_prime(value),
                    "metered primality mismatch for {value}"
                );
            }
        }
        for carmichael in [561u64, 1_105, 1_729, 2_465, 2_821, 6_601] {
            assert!(!is_probable_prime(&BigInt::from(carmichael)));
            let mut meter = Unbounded;
            assert!(!metered_is_probable_prime(&BigInt::from(carmichael), &mut meter).unwrap());
        }
    }

    #[test]
    fn deterministic_primality_rejects_the_terminal_base_adversary() {
        let composite = terminal_base_pseudoprime();
        assert!(composite < deterministic_primality_bound());
        assert!(!is_probable_prime(&composite));

        let mut measured = CountingMeter::default();
        assert!(!metered_is_probable_prime(&composite, &mut measured).unwrap());
        assert_eq!(measured.dimensions, [1_257_341, 236_088, 16_258, 0, 0]);
        assert_eq!(measured.checkpoints, 1_272_866);

        let mut limits = BudgetLimits {
            dimensions: measured.dimensions,
            verifier_pool: 0,
        };
        limits.dimensions[Dimension::ComputeSteps.index()] -= 1;
        let mut budget = Budget::new(limits);
        assert!(matches!(
            metered_is_probable_prime(&composite, &mut budget),
            Err(MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            }))
        ));

        for checkpoint in [
            1,
            measured.checkpoints / 4,
            measured.checkpoints / 2,
            measured.checkpoints * 3 / 4,
            measured.checkpoints - 1,
            measured.checkpoints,
        ] {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                metered_is_probable_prime(&composite, &mut cancelled),
                Err(MeterError::Cancelled),
                "primality result crossed checkpoint {checkpoint}"
            );
        }

        assert_eq!(FiniteField::new(composite.clone()), None);
        let mut meter = Unbounded;
        assert_eq!(
            FiniteField::metered_new(composite.clone(), &mut meter).unwrap(),
            None
        );
        assert_eq!(legendre_symbol(&BigInt::from(2), &composite), None);
        let mut meter = Unbounded;
        assert_eq!(
            metered_legendre_symbol(&BigInt::from(2), &composite, &mut meter).unwrap(),
            None
        );

        let diagnostic = check_lucky_prime(&composite, &[]).unwrap_err();
        assert_eq!(diagnostic.prime, composite);
        assert_eq!(diagnostic.reason, UnluckyPrimeReason::InvalidPrimeCandidate);
        assert_eq!(diagnostic.coefficient_index, None);
        let mut meter = Unbounded;
        let diagnostic = metered_check_lucky_prime(&composite, &[], &mut meter)
            .unwrap()
            .unwrap_err();
        assert_eq!(diagnostic.prime, composite);
        assert_eq!(diagnostic.reason, UnluckyPrimeReason::InvalidPrimeCandidate);
        assert_eq!(diagnostic.coefficient_index, None);
    }

    #[test]
    fn deterministic_primality_cancels_at_every_observed_small_composite_checkpoint() {
        let composite = BigInt::from(2_021);
        let mut measured = CheckpointMeter::default();
        assert!(!metered_is_probable_prime(&composite, &mut measured).unwrap());
        assert_eq!(measured.checkpoints, 2_405);

        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                metered_is_probable_prime(&composite, &mut cancelled),
                Err(MeterError::Cancelled),
                "primality result crossed checkpoint {checkpoint}"
            );
        }
    }

    #[test]
    fn quadratic_symbols_match_independent_factorization_and_euler_oracles() {
        for denominator in (1u64..128).step_by(2) {
            for numerator in -128i64..=128 {
                let expected = scalar_jacobi_by_factorization(numerator, denominator);
                let numerator = BigInt::from(numerator);
                let denominator = BigInt::from(denominator);
                assert_eq!(
                    jacobi_symbol(&numerator, &denominator),
                    Some(expected),
                    "Jacobi mismatch for numerator={numerator}, denominator={denominator}"
                );
                let mut meter = Unbounded;
                assert_eq!(
                    metered_jacobi_symbol(&numerator, &denominator, &mut meter).unwrap(),
                    Some(expected),
                    "metered Jacobi mismatch for numerator={numerator}, denominator={denominator}"
                );
            }
        }

        let prime = 97u64;
        for numerator in -128i64..=128 {
            let expected = scalar_legendre_prime(numerator, prime);
            let numerator = BigInt::from(numerator);
            let prime = BigInt::from(prime);
            assert_eq!(legendre_symbol(&numerator, &prime), Some(expected));
            let mut meter = Unbounded;
            assert_eq!(
                metered_legendre_symbol(&numerator, &prime, &mut meter).unwrap(),
                Some(expected)
            );
        }
    }

    #[test]
    fn jacobi_supports_arbitrary_precision_and_checks_terminal_gcd() {
        let mersenne_prime = (BigInt::one() << 127) - 1i64;
        assert_eq!(jacobi_symbol(&BigInt::from(2), &mersenne_prime), Some(1));
        let mut meter = Unbounded;
        assert_eq!(
            metered_jacobi_symbol(&BigInt::from(2), &mersenne_prime, &mut meter).unwrap(),
            Some(1)
        );

        let composite = BigInt::from(3) * &mersenne_prime;
        assert_eq!(jacobi_symbol(&BigInt::from(3), &composite), Some(0));
        let mut meter = Unbounded;
        assert_eq!(
            metered_jacobi_symbol(&BigInt::from(3), &composite, &mut meter).unwrap(),
            Some(0)
        );
    }

    #[test]
    fn quadratic_symbols_refuse_values_outside_their_exact_domains() {
        for invalid_denominator in [BigInt::from(-3), BigInt::zero(), BigInt::from(2)] {
            assert_eq!(jacobi_symbol(&BigInt::one(), &invalid_denominator), None);
            let mut meter = Unbounded;
            assert_eq!(
                metered_jacobi_symbol(&BigInt::one(), &invalid_denominator, &mut meter).unwrap(),
                None
            );
        }

        let above_bound_prime = (BigInt::one() << 127) - 1i64;
        for refused_prime in [
            BigInt::from(2),
            BigInt::from(91),
            deterministic_primality_bound(),
            above_bound_prime,
        ] {
            assert_eq!(legendre_symbol(&BigInt::from(3), &refused_prime), None);
            let mut meter = Unbounded;
            assert_eq!(
                metered_legendre_symbol(&BigInt::from(3), &refused_prime, &mut meter).unwrap(),
                None
            );
        }
    }

    proptest! {
        #[test]
        fn exact_division_round_trips_broad_operands(
            dividend_bytes in proptest::collection::vec(any::<u8>(), 0..129),
            divisor_bytes in proptest::collection::vec(any::<u8>(), 0..129),
        ) {
            let dividend = BigInt::from_signed_bytes_be(&dividend_bytes);
            let mut divisor = BigInt::from_signed_bytes_be(&divisor_bytes);
            if divisor.is_zero() {
                divisor = BigInt::one();
            }
            let product = &dividend * &divisor;
            prop_assert_eq!(exact_div(&product, &divisor), Some(dividend));

            if divisor.abs() > BigInt::one() {
                prop_assert_eq!(exact_div(&(product + 1i64), &divisor), None);
            }
        }

        #[test]
        fn metered_exact_division_matches_unmetered_lane(
            dividend_bytes in proptest::collection::vec(any::<u8>(), 0..97),
            divisor_bytes in proptest::collection::vec(any::<u8>(), 0..65),
        ) {
            let dividend = BigInt::from_signed_bytes_be(&dividend_bytes);
            let divisor = BigInt::from_signed_bytes_be(&divisor_bytes);
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_exact_div(&dividend, &divisor, &mut meter).unwrap(),
                exact_div(&dividend, &divisor)
            );
        }

        #[test]
        fn modular_inverse_matches_bounded_scalar_oracle(a in -500i64..500, modulus in 1i64..200) {
            let normalized = a.rem_euclid(modulus);
            let expected = (0..modulus)
                .find(|candidate| (normalized * candidate).rem_euclid(modulus) == 1i64.rem_euclid(modulus));
            let actual = mod_inverse(&BigInt::from(a), &BigInt::from(modulus))
                .map(|value| value.to_i64().unwrap());
            prop_assert_eq!(actual, expected);

            let mut meter = Unbounded;
            let metered = metered_mod_inverse(
                &BigInt::from(a),
                &BigInt::from(modulus),
                &mut meter,
            )
            .unwrap()
            .map(|value| value.to_i64().unwrap());
            prop_assert_eq!(metered, expected);
        }

        #[test]
        fn crt_pair_matches_bounded_exhaustive_oracle(
            remainder_a in -500i64..500,
            modulus_a in 1u64..200,
            remainder_b in -500i64..500,
            modulus_b in 1u64..200,
        ) {
            let gcd = scalar_gcd(modulus_a, modulus_b);
            let lcm = (modulus_a / gcd) * modulus_b;
            let normalized_a = remainder_a.rem_euclid(modulus_a as i64) as u64;
            let normalized_b = remainder_b.rem_euclid(modulus_b as i64) as u64;
            let expected = (0..lcm).find(|candidate| {
                candidate % modulus_a == normalized_a && candidate % modulus_b == normalized_b
            });
            let actual = crt_pair(
                &BigInt::from(remainder_a),
                &BigInt::from(modulus_a),
                &BigInt::from(remainder_b),
                &BigInt::from(modulus_b),
            );
            let mut meter = Unbounded;
            let metered = metered_crt_pair(
                &BigInt::from(remainder_a),
                &BigInt::from(modulus_a),
                &BigInt::from(remainder_b),
                &BigInt::from(modulus_b),
                &mut meter,
            )
            .unwrap();
            prop_assert_eq!(&metered, &actual);
            match (actual, expected) {
                (Some((value, combined_modulus)), Some(expected_value)) => {
                    prop_assert_eq!(value.to_u64(), Some(expected_value));
                    prop_assert_eq!(combined_modulus.to_u64(), Some(lcm));
                }
                (None, None) => {}
                (actual, expected) => prop_assert!(false, "CRT mismatch: {actual:?} vs {expected:?}"),
            }
        }

        #[test]
        fn rational_reconstruction_recovers_unique_small_fraction(
            numerator in -50i64..51,
            denominator in 1u64..51,
        ) {
            prop_assume!(scalar_gcd(numerator.unsigned_abs(), denominator) == 1);
            const MODULUS: i64 = 1_000_003;
            let inverse = mod_inverse(&BigInt::from(denominator), &BigInt::from(MODULUS)).unwrap();
            let residue = ((BigInt::from(numerator) * inverse) % BigInt::from(MODULUS)
                + BigInt::from(MODULUS)) % BigInt::from(MODULUS);
            prop_assert_eq!(
                rational_reconstruct(&residue, &BigInt::from(MODULUS)),
                Some((BigInt::from(numerator), BigInt::from(denominator)))
            );
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_rational_reconstruct(
                    &residue,
                    &BigInt::from(MODULUS),
                    &mut meter,
                )
                .unwrap(),
                Some((BigInt::from(numerator), BigInt::from(denominator)))
            );
        }

        #[test]
        fn metered_rational_reconstruction_matches_unmetered_refusals(
            residue in -20_000i64..20_001,
            modulus in 2i64..10_000,
        ) {
            let residue = BigInt::from(residue);
            let modulus = BigInt::from(modulus);
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_rational_reconstruct(&residue, &modulus, &mut meter).unwrap(),
                rational_reconstruct(&residue, &modulus)
            );
        }

        #[test]
        fn metered_crt_fold_matches_unmetered_lane(
            congruences in proptest::collection::vec((-500i64..500, 1u64..80), 0..6),
        ) {
            let congruences: Vec<(BigInt, BigInt)> = congruences
                .into_iter()
                .map(|(remainder, modulus)| (remainder.into(), modulus.into()))
                .collect();
            let mut meter = Unbounded;
            prop_assert_eq!(
                metered_crt(&congruences, &mut meter).unwrap(),
                crt(&congruences)
            );
        }

        #[test]
        fn coprime_split_crt_matches_pairwise_scalar_admission(
            congruences in proptest::collection::vec((-500i64..500, 1u64..80), 0..7),
        ) {
            let pairwise_coprime = congruences.iter().enumerate().all(
                |(index, (_remainder, modulus))| {
                    congruences[..index]
                        .iter()
                        .all(|(_other_remainder, other_modulus)| {
                            scalar_gcd(*modulus, *other_modulus) == 1
                        })
                },
            );
            let remainders = congruences
                .iter()
                .map(|(remainder, _modulus)| BigInt::from(*remainder))
                .collect::<Vec<_>>();
            let moduli = congruences
                .iter()
                .map(|(_remainder, modulus)| BigInt::from(*modulus))
                .collect::<Vec<_>>();
            let owned = remainders
                .iter()
                .cloned()
                .zip(moduli.iter().cloned())
                .collect::<Vec<_>>();
            let expected = if pairwise_coprime {
                crt(&owned)
            } else {
                None
            };
            prop_assert_eq!(crt_coprime_slices(&remainders, &moduli), expected);
        }

        #[test]
        fn metered_gcd_and_bezout_match_unmetered_lanes(
            a_bytes in proptest::collection::vec(any::<u8>(), 0..97),
            b_bytes in proptest::collection::vec(any::<u8>(), 0..97),
        ) {
            let a = BigInt::from_signed_bytes_be(&a_bytes);
            let b = BigInt::from_signed_bytes_be(&b_bytes);
            let mut meter = Unbounded;
            prop_assert_eq!(metered_gcd(&a, &b, &mut meter).unwrap(), gcd(&a, &b));

            let mut meter = Unbounded;
            let (metered_g, x, y) = metered_extended_gcd(&a, &b, &mut meter).unwrap();
            prop_assert_eq!(&metered_g, &gcd(&a, &b));
            prop_assert_eq!(&a * x + &b * y, metered_g);
        }
    }

    #[test]
    fn metered_mul_halts_on_budget_exhaustion() {
        let a = (BigInt::one() << 300) + 12345i64;
        let b = (BigInt::one() << 300) + 67890i64;

        let limits = BudgetLimits::uniform(1, 0);
        let mut budget = Budget::new(limits);
        let err = metered_mul(&a, &b, &mut budget).unwrap_err();
        assert!(matches!(err, MeterError::Budget(_)));

        let limits = BudgetLimits::uniform(1_000_000, 0);
        let mut budget = Budget::new(limits);
        let res = metered_mul(&a, &b, &mut budget).expect("computes within budget");
        assert_eq!(res, &a * &b);
    }

    #[test]
    fn metered_gcd_halts_on_budget_exhaustion() {
        let a = (BigInt::one() << 200) + 1i64;
        let b = (BigInt::one() << 150) + 1i64;

        let limits = BudgetLimits::uniform(2, 0);
        let mut budget = Budget::new(limits);
        let err = metered_gcd(&a, &b, &mut budget).unwrap_err();
        assert!(matches!(err, MeterError::Budget(_)));

        let limits = BudgetLimits::uniform(100, 0);
        let mut budget = Budget::new(limits);
        assert_eq!(
            metered_gcd(&BigInt::zero(), &BigInt::zero(), &mut budget),
            Ok(BigInt::zero())
        );
    }

    #[test]
    fn metered_modular_lanes_refuse_invalid_inputs_without_zero_charges() {
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert_eq!(
            metered_mod_inverse(&BigInt::one(), &BigInt::zero(), &mut budget),
            Ok(None)
        );
        assert_eq!(
            metered_crt_pair(
                &BigInt::zero(),
                &BigInt::zero(),
                &BigInt::zero(),
                &BigInt::one(),
                &mut budget,
            ),
            Ok(None)
        );
        assert_eq!(
            metered_crt(&[(BigInt::zero(), BigInt::zero())], &mut budget),
            Ok(None)
        );
        assert_eq!(
            metered_rational_reconstruct(&BigInt::one(), &BigInt::one(), &mut budget),
            Ok(None)
        );
        assert_eq!(
            metered_is_probable_prime(&BigInt::one(), &mut budget),
            Ok(false)
        );
        assert_eq!(
            metered_jacobi_symbol(&BigInt::one(), &BigInt::zero(), &mut budget),
            Ok(None)
        );
        assert_eq!(
            metered_legendre_symbol(&BigInt::one(), &BigInt::from(2), &mut budget),
            Ok(None)
        );
    }

    #[test]
    fn fast_terminal_classes_checkpoint_after_classification_without_charges() {
        assert_uncharged_fast_terminal(None, |meter| {
            metered_mod_inverse(&BigInt::one(), &BigInt::zero(), meter)
        });
        assert_uncharged_fast_terminal(None, |meter| {
            metered_crt_pair(
                &BigInt::zero(),
                &BigInt::zero(),
                &BigInt::zero(),
                &BigInt::one(),
                meter,
            )
        });
        assert_uncharged_fast_terminal(Some((BigInt::zero(), BigInt::one())), |meter| {
            metered_crt(&[], meter)
        });
        assert_uncharged_fast_terminal(None, |meter| {
            metered_crt(&[(BigInt::zero(), BigInt::zero())], meter)
        });
        assert_uncharged_fast_terminal(None, |meter| {
            metered_rational_reconstruct(&BigInt::one(), &BigInt::one(), meter)
        });
        assert_uncharged_fast_terminal(false, |meter| {
            metered_is_probable_prime(&BigInt::one(), meter)
        });
        assert_uncharged_fast_terminal(None, |meter| {
            metered_jacobi_symbol(&BigInt::one(), &BigInt::zero(), meter)
        });
        assert_uncharged_fast_terminal(None, |meter| {
            metered_legendre_symbol(&BigInt::one(), &BigInt::from(2), meter)
        });
    }

    #[test]
    fn metered_modular_lanes_halt_on_budget_exhaustion() {
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_mod_inverse(&BigInt::from(17), &BigInt::from(101), &mut budget),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_crt_pair(
                &BigInt::from(3),
                &BigInt::from(5),
                &BigInt::from(4),
                &BigInt::from(7),
                &mut budget,
            ),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_rational_reconstruct(&BigInt::from(27), &BigInt::from(1_000_003), &mut budget,),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_is_probable_prime(&BigInt::from(1_000_003), &mut budget),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_jacobi_symbol(&BigInt::from(37), &BigInt::from(101), &mut budget),
            Err(MeterError::Budget(_))
        ));

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            metered_legendre_symbol(&BigInt::from(37), &BigInt::from(101), &mut budget),
            Err(MeterError::Budget(_))
        ));
    }

    #[test]
    fn prime_stream_cancellation_is_retry_safe_and_interleavable() {
        let mut measured_stream = PrimeStream::new();
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            measured_stream.next_metered(&mut measured).unwrap(),
            BigInt::from(2)
        );
        assert!(measured.checkpoints > 1);

        let mut stream = PrimeStream::new();
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            stream.next_metered(&mut cancelled),
            Err(PrimeStreamError::Meter(MeterError::Cancelled))
        );
        assert_eq!(stream.current, BigInt::from(2));
        assert!(stream.emitted.is_empty());

        let mut meter = Unbounded;
        assert_eq!(stream.next_metered(&mut meter).unwrap(), BigInt::from(2));
        assert_eq!(stream.next(), Some(BigInt::from(3)));
        assert_eq!(stream.next_metered(&mut meter).unwrap(), BigInt::from(5));
        assert_eq!(stream.next(), Some(BigInt::from(7)));
    }

    #[test]
    fn prime_stream_unmetered_growth_failure_is_retry_safe() {
        let mut stream = prime_stream_before_four_to_eight_growth();
        let before_emitted = stream.emitted.clone();
        let before_current = stream.current.clone();
        let before_capacity = stream.emitted_capacity;
        let mut reserve_calls = 0;

        // PRIME-ITERATOR-SKIP-ON-RESERVE-FAIL: advancing the public cursor while scanning 8, 9,
        // 10, or before the failing reservation silently skips prime 11 on retry.
        assert_eq!(
            stream.try_next_with_reserve(|_, _| {
                reserve_calls += 1;
                Err(PrimeStreamError::AllocationFailure)
            }),
            Err(PrimeStreamError::AllocationFailure)
        );
        assert_eq!(reserve_calls, 1);
        assert_eq!(stream.emitted, before_emitted);
        assert_eq!(stream.current, before_current);
        assert_eq!(stream.emitted_capacity, before_capacity);

        let mut meter = Unbounded;
        assert_eq!(stream.next_metered(&mut meter).unwrap(), BigInt::from(11));
        assert_eq!(stream.try_next().unwrap(), BigInt::from(13));
    }

    #[test]
    fn prime_stream_growth_is_prepared_before_atomic_publication() {
        let mut stream = prime_stream_before_four_to_eight_growth();
        let before_emitted = stream.emitted.clone();
        let before_current = stream.current.clone();
        let mut measured = CountingMeter::default();
        assert_eq!(
            stream.next_metered(&mut measured).unwrap(),
            BigInt::from(11)
        );
        // PRIME-TABLE-DIRECT-PUSH: the former one-header charge/direct Vec::push path reports a
        // smaller transcript and performs its reallocation after the final checkpoint.
        // Candidates 8 through 11 perform thirteen one-limb Newton comparisons and seven
        // one-limb prime-versus-root comparisons. Both governed owners charge sign, length, and
        // digit work with four safe points, so restoring either opaque comparator changes this
        // transcript.
        assert_eq!(measured.dimensions, [1_485, 1_280, 194, 0, 0]);
        assert_eq!(measured.checkpoints, 1_677);
        assert_eq!(stream.emitted_capacity, 8);
        assert_eq!(stream.emitted.len(), 5);
        assert_eq!(stream.emitted.last(), Some(&BigInt::from(11)));

        let mut stream = prime_stream_before_four_to_eight_growth();
        let mut terminal = CheckpointMeter::arming_after(measured.checkpoints - 1);
        assert_eq!(
            stream.next_metered(&mut terminal),
            Err(PrimeStreamError::Meter(MeterError::Cancelled))
        );
        assert_eq!(stream.emitted, before_emitted);
        assert_eq!(stream.current, before_current);
        assert_eq!(stream.emitted_capacity, 4);
    }

    #[test]
    fn prime_stream_growth_refuses_one_short_budgets_without_state_change() {
        let mut measured_stream = prime_stream_before_four_to_eight_growth();
        let mut measured = CountingMeter::default();
        let expected = measured_stream.next_metered(&mut measured).unwrap();

        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            assert!(measured.dimensions[dimension.index()] > 0);
            let mut limits = BudgetLimits {
                dimensions: measured.dimensions,
                verifier_pool: 0,
            };
            limits.dimensions[dimension.index()] -= 1;

            let mut stream = prime_stream_before_four_to_eight_growth();
            let before_emitted = stream.emitted.clone();
            let before_current = stream.current.clone();
            let mut budget = Budget::new(limits);
            assert!(matches!(
                stream.next_metered(&mut budget),
                Err(PrimeStreamError::Meter(MeterError::Budget(
                    BudgetError::Exhausted {
                        dimension: exhausted,
                        ..
                    }
                ))) if exhausted == dimension
            ));
            assert_eq!(stream.emitted, before_emitted);
            assert_eq!(stream.current, before_current);
            assert_eq!(stream.emitted_capacity, 4);

            let mut retry_meter = Unbounded;
            assert_eq!(stream.next_metered(&mut retry_meter).unwrap(), expected);
        }
    }

    #[test]
    fn prime_stream_growth_cancels_at_every_observed_prepare_checkpoint() {
        let mut measured_stream = prime_stream_before_four_to_eight_growth();
        let mut measured = CheckpointMeter::default();
        let expected = measured_stream.next_metered(&mut measured).unwrap();
        assert!(measured.checkpoints > 10);

        for checkpoint in 1..=measured.checkpoints {
            let mut stream = prime_stream_before_four_to_eight_growth();
            let before_emitted = stream.emitted.clone();
            let before_current = stream.current.clone();
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                stream.next_metered(&mut cancelled),
                Err(PrimeStreamError::Meter(MeterError::Cancelled)),
                "prime table published across checkpoint {checkpoint}"
            );
            assert_eq!(stream.emitted, before_emitted);
            assert_eq!(stream.current, before_current);
            assert_eq!(stream.emitted_capacity, 4);

            let mut retry_meter = Unbounded;
            assert_eq!(stream.next_metered(&mut retry_meter).unwrap(), expected);
        }
    }

    #[test]
    fn prime_stream_table_growth_fails_typed_and_terminally() {
        let overflowing_capacity = usize::MAX / 2 + 1;
        assert_eq!(
            prime_table_growth_target(overflowing_capacity, overflowing_capacity + 1),
            Err(PrimeStreamError::SizeOverflow)
        );

        let header_size = std::mem::size_of::<BigInt>();
        let impossible_capacity = (isize::MAX as usize) / header_size + 1;
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            reserve_prime_table(impossible_capacity, 1, &mut measured),
            Err(PrimeStreamError::AllocationFailure)
        );
        assert_eq!(measured.checkpoints, 2);

        let mut cancelled = CheckpointMeter::arming_after(1);
        assert_eq!(
            reserve_prime_table(impossible_capacity, 1, &mut cancelled),
            Err(PrimeStreamError::Meter(MeterError::Cancelled))
        );
        assert_eq!(cancelled.checkpoints, 1);
    }

    #[test]
    fn primality_supports_late_in_algorithm_cancellation() {
        let candidate = (BigInt::one() << 127) - 1i64;
        let mut measured = CheckpointMeter::default();
        let expected = metered_is_probable_prime(&candidate, &mut measured).unwrap();
        assert!(expected);
        assert!(measured.checkpoints > 100);

        let late_checkpoint = measured.checkpoints.saturating_mul(3) / 4;
        let mut cancelled = CheckpointMeter::cancelling_at(late_checkpoint);
        assert_eq!(
            metered_is_probable_prime(&candidate, &mut cancelled),
            Err(MeterError::Cancelled)
        );
        assert!(cancelled.checkpoints > 75);
    }

    #[test]
    fn quadratic_symbols_support_late_and_terminal_cancellation() {
        let mut previous = BigInt::one();
        let mut current = BigInt::one();
        for _ in 0..219 {
            let next = &previous + &current;
            previous = current;
            current = next;
        }
        assert!(!(&current % 2i64).is_zero());

        let mut measured = CheckpointMeter::default();
        let expected = metered_jacobi_symbol(&previous, &current, &mut measured)
            .unwrap()
            .expect("positive odd denominator has a Jacobi symbol");
        assert_eq!(jacobi_symbol(&previous, &current), Some(expected));
        assert!(measured.checkpoints > 100);

        let mut late_cancelled =
            CheckpointMeter::cancelling_at(measured.checkpoints.saturating_mul(3) / 4);
        assert_eq!(
            metered_jacobi_symbol(&previous, &current, &mut late_cancelled),
            Err(MeterError::Cancelled)
        );
        assert!(late_cancelled.checkpoints > 75);

        let mut terminal_cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            metered_jacobi_symbol(&previous, &current, &mut terminal_cancelled),
            Err(MeterError::Cancelled)
        );

        let numerator = BigInt::from(35);
        let prime = BigInt::from(97);
        let mut measured = CheckpointMeter::default();
        let expected = metered_legendre_symbol(&numerator, &prime, &mut measured).unwrap();
        assert_eq!(expected, legendre_symbol(&numerator, &prime));
        let mut terminal_cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            metered_legendre_symbol(&numerator, &prime, &mut terminal_cancelled),
            Err(MeterError::Cancelled)
        );
    }

    #[test]
    fn every_computed_terminal_class_observes_final_cancellation() {
        assert_terminal_checkpoint(None, 53, |meter| {
            metered_exact_div(&BigInt::from(35), &BigInt::from(6), meter)
        });
        assert_terminal_checkpoint(None, 152, |meter| {
            metered_mod_inverse(&BigInt::from(6), &BigInt::from(9), meter)
        });
        assert_terminal_checkpoint(Some(BigInt::from(6)), 232, |meter| {
            metered_mod_inverse(&BigInt::from(17), &BigInt::from(101), meter)
        });
        assert_terminal_checkpoint(None, 56, |meter| {
            metered_crt_pair(
                &BigInt::zero(),
                &BigInt::from(2),
                &BigInt::one(),
                &BigInt::from(2),
                meter,
            )
        });
        // sqrt(50) performs five one-limb Newton comparisons. The governed comparator charges
        // sign, length, and digit work with four safe points, so an opaque one-step/two-checkpoint
        // comparison would reduce each reconstruction transcript by ten checkpoints. The
        // nonzero-refusal case also performs three governed one-limb reconstruction-bound
        // comparisons, contributing six more checkpoints than the former modular wrapper.
        assert_terminal_checkpoint(Some((BigInt::zero(), BigInt::one())), 605, |meter| {
            metered_rational_reconstruct(&BigInt::zero(), &BigInt::from(101), meter)
        });
        assert_terminal_checkpoint(None, 686, |meter| {
            metered_rational_reconstruct(&BigInt::from(8), &BigInt::from(101), meter)
        });
        assert_terminal_checkpoint(true, 603, |meter| {
            metered_is_probable_prime(&BigInt::from(41), meter)
        });
        // The composite witness path performs three additional governed one-limb comparisons.
        assert_terminal_checkpoint(false, 2_405, |meter| {
            metered_is_probable_prime(&BigInt::from(2_021), meter)
        });
    }

    #[test]
    fn crt_and_reconstruction_support_late_in_algorithm_cancellation() {
        let modulus_a = (BigInt::one() << 127) - 1i64;
        let modulus_b = (BigInt::one() << 89) - 1i64;
        let mut measured_crt = CheckpointMeter::default();
        assert!(
            metered_crt_pair(
                &BigInt::zero(),
                &modulus_a,
                &BigInt::zero(),
                &modulus_b,
                &mut measured_crt,
            )
            .unwrap()
            .is_some()
        );
        assert!(measured_crt.checkpoints > 100);
        let mut cancelled_crt =
            CheckpointMeter::cancelling_at(measured_crt.checkpoints.saturating_mul(3) / 4);
        assert_eq!(
            metered_crt_pair(
                &BigInt::zero(),
                &modulus_a,
                &BigInt::zero(),
                &modulus_b,
                &mut cancelled_crt,
            ),
            Err(MeterError::Cancelled)
        );

        let denominator = BigInt::from(37);
        let inverse = mod_inverse(&denominator, &modulus_a).unwrap();
        let residue = (&BigInt::from(-23) * inverse) % &modulus_a;
        let mut measured_reconstruction = CheckpointMeter::default();
        assert_eq!(
            metered_rational_reconstruct(&residue, &modulus_a, &mut measured_reconstruction,)
                .unwrap(),
            Some((BigInt::from(-23), denominator))
        );
        assert!(measured_reconstruction.checkpoints > 100);
        let mut cancelled_reconstruction = CheckpointMeter::cancelling_at(
            measured_reconstruction.checkpoints.saturating_mul(3) / 4,
        );
        assert_eq!(
            metered_rational_reconstruct(&residue, &modulus_a, &mut cancelled_reconstruction,),
            Err(MeterError::Cancelled)
        );
    }

    #[test]
    fn modular_ring_and_exact_finite_field_preserve_parent_invariants() {
        let ring = ModularRing::new(BigInt::from(12)).expect("modulus > 1");
        let a = ring.element(BigInt::from(7));
        let b = ring.element(BigInt::from(8));
        assert_eq!(a.add(&b).unwrap().value(), &BigInt::from(3));
        assert_eq!(a.sub(&b).unwrap().value(), &BigInt::from(11));
        assert_eq!(a.mul(&b).unwrap().value(), &BigInt::from(8));
        assert_eq!(a.inv().unwrap().value(), &BigInt::from(7));
        assert!(b.inv().is_none());
        assert!(a.pow(&BigInt::from(-1)).is_none());
        let other_ring = ModularRing::new(BigInt::from(13)).unwrap();
        assert!(a.add(&other_ring.one()).is_none());

        let ff = FiniteField::new(BigInt::from(17)).expect("17 is prime");
        assert!(FiniteField::new(BigInt::from(18)).is_none());
        assert!(FiniteField::new(deterministic_primality_bound()).is_none());
        let x = ff.element(BigInt::from(5));
        let y = ff.element(BigInt::from(11));
        assert_eq!(x.add(&y).unwrap().value(), &BigInt::from(16));
        assert_eq!(x.sub(&y).unwrap().value(), &BigInt::from(11));
        assert_eq!(x.mul(&y).unwrap().value(), &BigInt::from(4));
        let x_inv = x.inv().expect("5 is invertible mod 17");
        assert_eq!(x_inv.value(), &BigInt::from(7));
        assert_eq!(y.div(&x).unwrap().value(), y.mul(&x_inv).unwrap().value());
        assert_eq!(x.pow(&BigInt::from(16)).unwrap().value(), &BigInt::one());
        assert!(x.pow(&BigInt::from(-1)).is_none());
        let other_field = FiniteField::new(BigInt::from(19)).unwrap();
        assert!(x.mul(&other_field.one()).is_none());

        let mut meter = Unbounded;
        assert_eq!(
            ring.metered_element(&BigInt::from(-5), &mut meter)
                .unwrap()
                .value(),
            &BigInt::from(7)
        );
        let mut meter = Unbounded;
        assert_eq!(
            a.metered_mul(&b, &mut meter).unwrap().unwrap().value(),
            a.mul(&b).unwrap().value()
        );
        let mut meter = Unbounded;
        assert_eq!(
            FiniteField::metered_new(BigInt::from(17), &mut meter).unwrap(),
            Some(ff.clone())
        );
        let mut meter = Unbounded;
        assert_eq!(
            x.metered_pow(&BigInt::from(16), &mut meter)
                .unwrap()
                .unwrap(),
            x.pow(&BigInt::from(16)).unwrap()
        );
    }

    #[test]
    fn finite_field_batch_inverse_matches_scalar_and_preserves_order() {
        let field = FiniteField::new(BigInt::from(101)).unwrap();
        let values = [2, 3, 5, 7].map(|value| field.element(BigInt::from(value)));
        let expected = values
            .iter()
            .map(|value| value.inv().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            field.try_batch_inv(&values).unwrap(),
            Some(expected.clone())
        );
        let mut meter = Unbounded;
        let actual = field
            .metered_batch_inv(&values, &mut meter)
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
        for (value, inverse) in values.iter().zip(&actual) {
            assert_eq!(value.mul(inverse).unwrap(), field.one());
        }

        let binary_field = FiniteField::new(BigInt::from(2)).unwrap();
        assert_eq!(
            binary_field.try_batch_inv(&[binary_field.one()]).unwrap(),
            Some(vec![binary_field.one()])
        );
        let repeated = vec![field.element(BigInt::from(9)); 3];
        assert_eq!(
            field.try_batch_inv(&repeated).unwrap(),
            Some(repeated.iter().map(|value| value.inv().unwrap()).collect())
        );
    }

    #[test]
    fn finite_field_batch_inverse_preflights_before_allocation() {
        let field = FiniteField::new(BigInt::from(101)).unwrap();
        let foreign = FiniteField::new(BigInt::from(103)).unwrap();

        let mut empty_meter = CountingMeter::default();
        assert_eq!(
            field.metered_batch_inv(&[], &mut empty_meter).unwrap(),
            Some(Vec::new())
        );
        assert_eq!(empty_meter.dimensions, [0; DIMENSION_COUNT]);
        assert_eq!(empty_meter.checkpoints, 2);

        let mut zero_meter = CountingMeter::default();
        assert_eq!(
            field
                .metered_batch_inv(&[field.zero()], &mut zero_meter)
                .unwrap(),
            None
        );
        assert_eq!(zero_meter.dimensions[Dimension::AllocationCount.index()], 0);

        let mut foreign_zero_meter = CountingMeter::default();
        assert_eq!(
            field
                .metered_batch_inv(&[foreign.zero()], &mut foreign_zero_meter)
                .unwrap(),
            None
        );
        let mut foreign_nonzero_meter = CountingMeter::default();
        assert_eq!(
            field
                .metered_batch_inv(&[foreign.one()], &mut foreign_nonzero_meter)
                .unwrap(),
            None
        );
        assert_eq!(
            foreign_zero_meter.dimensions,
            foreign_nonzero_meter.dimensions
        );
        assert_eq!(
            foreign_zero_meter.checkpoints,
            foreign_nonzero_meter.checkpoints
        );
        assert_eq!(
            foreign_zero_meter.dimensions[Dimension::AllocationCount.index()],
            0
        );

        let mut mixed_meter = CountingMeter::default();
        let mixed = [field.one(), foreign.one()];
        assert_eq!(
            field.metered_batch_inv(&mixed, &mut mixed_meter).unwrap(),
            None
        );
        assert_eq!(
            mixed_meter.dimensions[Dimension::AllocationCount.index()],
            0
        );

        for refused in [vec![field.zero()], vec![foreign.one()], mixed.to_vec()] {
            let mut measured = CheckpointMeter::default();
            assert_eq!(
                field.metered_batch_inv(&refused, &mut measured).unwrap(),
                None
            );
            let mut terminal = CheckpointMeter::cancelling_at(measured.checkpoints);
            assert_eq!(
                field.metered_batch_inv(&refused, &mut terminal),
                Err(BatchInverseError::Meter(MeterError::Cancelled))
            );
        }
    }

    #[test]
    fn finite_field_batch_inverse_pins_governed_transcript() {
        let field = FiniteField::new(BigInt::from(101)).unwrap();
        let values = [2, 3, 5, 7].map(|value| field.element(BigInt::from(value)));
        let mut measured = CountingMeter::default();
        let expected = field
            .metered_batch_inv(&values, &mut measured)
            .unwrap()
            .unwrap();

        // BATCH-INVERSE-PER-ELEMENT: replacing the one-inverse prefix/reverse lane with four
        // independent scalar inversions still returns these values but changes this transcript.
        // The four membership checks each use the governed sign/length/digit comparator.
        assert_eq!(measured.dimensions, [581, 1_176, 147, 0, 0]);
        assert_eq!(measured.checkpoints, 756);
        assert_eq!(
            expected,
            values
                .iter()
                .map(|value| value.inv().unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(measured.dimensions[Dimension::DepthLimit.index()], 0);
        assert_eq!(measured.dimensions[Dimension::RandomDraws.index()], 0);
    }

    #[test]
    fn finite_field_batch_inverse_refuses_each_one_short_budget() {
        let field = FiniteField::new(BigInt::from(101)).unwrap();
        let values = [2, 3, 5, 7].map(|value| field.element(BigInt::from(value)));
        let mut measured = CountingMeter::default();
        let expected = field
            .metered_batch_inv(&values, &mut measured)
            .unwrap()
            .unwrap();

        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            assert!(measured.dimensions[dimension.index()] > 0);
            let mut limits = BudgetLimits {
                dimensions: measured.dimensions,
                verifier_pool: 0,
            };
            limits.dimensions[dimension.index()] -= 1;
            let mut budget = Budget::new(limits);
            assert!(matches!(
                field.metered_batch_inv(&values, &mut budget),
                Err(BatchInverseError::Meter(MeterError::Budget(
                    BudgetError::Exhausted {
                        dimension: exhausted,
                        ..
                    }
                ))) if exhausted == dimension
            ));

            let mut retry = Unbounded;
            assert_eq!(
                field.metered_batch_inv(&values, &mut retry).unwrap(),
                Some(expected.clone())
            );
        }
    }

    #[test]
    fn finite_field_batch_inverse_cancels_at_every_checkpoint() {
        let field = FiniteField::new(BigInt::from(101)).unwrap();
        let values = [2, 3, 5, 7].map(|value| field.element(BigInt::from(value)));
        let mut measured = CheckpointMeter::default();
        let expected = field
            .metered_batch_inv(&values, &mut measured)
            .unwrap()
            .unwrap();
        assert!(measured.checkpoints > 20);

        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                field.metered_batch_inv(&values, &mut cancelled),
                Err(BatchInverseError::Meter(MeterError::Cancelled)),
                "batch output crossed checkpoint {checkpoint}"
            );
        }

        let mut retry = Unbounded;
        assert_eq!(
            field.metered_batch_inv(&values, &mut retry).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn finite_field_batch_buffers_fail_typed_and_terminally() {
        let prefix_slot = std::mem::size_of::<BigInt>();
        let output_slot = std::mem::size_of::<FiniteFieldElement>();
        let combined_slot = prefix_slot.checked_add(output_slot).unwrap();
        let u64_limit =
            usize::try_from(u64::MAX / u64::try_from(combined_slot).unwrap()).unwrap_or(usize::MAX);
        let max_len = [
            (isize::MAX as usize) / prefix_slot,
            (isize::MAX as usize) / output_slot,
            usize::MAX / combined_slot,
            u64_limit,
        ]
        .into_iter()
        .min()
        .unwrap();
        let overflowing_len = max_len.checked_add(1).unwrap();
        assert_eq!(
            batch_buffer_layout(max_len),
            Ok(u64::try_from(max_len.checked_mul(combined_slot).unwrap()).unwrap())
        );
        assert_eq!(
            batch_buffer_layout(overflowing_len),
            Err(BatchInverseError::SizeOverflow)
        );
        let mut overflow_meter = CheckpointMeter::default();
        assert_eq!(
            reserve_batch_buffers(overflowing_len, &mut overflow_meter),
            Err(BatchInverseError::SizeOverflow)
        );
        assert_eq!(overflow_meter.checkpoints, 1);
        let mut overflow_cancelled = CheckpointMeter::cancelling_at(1);
        assert_eq!(
            reserve_batch_buffers(overflowing_len, &mut overflow_cancelled),
            Err(BatchInverseError::Meter(MeterError::Cancelled))
        );

        let mut first_calls = 0;
        let mut second_calls = 0;
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            reserve_batch_buffers_with(
                1,
                &mut measured,
                |_, _| {
                    first_calls += 1;
                    Err(())
                },
                |_, _| {
                    second_calls += 1;
                    Ok(())
                },
            ),
            Err(BatchInverseError::AllocationFailure)
        );
        assert_eq!((first_calls, second_calls), (1, 0));
        assert_eq!(measured.checkpoints, 2);

        let mut first_calls = 0;
        let mut second_calls = 0;
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            reserve_batch_buffers_with(
                1,
                &mut measured,
                |_, _| {
                    first_calls += 1;
                    Ok(())
                },
                |_, _| {
                    second_calls += 1;
                    Err(())
                },
            ),
            Err(BatchInverseError::AllocationFailure)
        );
        assert_eq!((first_calls, second_calls), (1, 1));
        assert_eq!(measured.checkpoints, 3);

        let mut cancelled = CheckpointMeter::arming_after(1);
        assert_eq!(
            reserve_batch_buffers_with(1, &mut cancelled, |_, _| Err(()), |_, _| Ok(())),
            Err(BatchInverseError::Meter(MeterError::Cancelled))
        );

        let reserve_calls = std::cell::Cell::<usize>::new(0);
        let mut budget = Budget::new(BudgetLimits {
            dimensions: [u64::MAX, 0, u64::MAX, u64::MAX, u64::MAX],
            verifier_pool: 0,
        });
        assert!(matches!(
            reserve_batch_buffers_with(
                1,
                &mut budget,
                |_, _| {
                    reserve_calls.set(reserve_calls.get() + 1);
                    Ok(())
                },
                |_, _| {
                    reserve_calls.set(reserve_calls.get() + 1);
                    Ok(())
                },
            ),
            Err(BatchInverseError::Meter(MeterError::Budget(
                BudgetError::Exhausted {
                    dimension: Dimension::MemoryBytes,
                    ..
                }
            )))
        ));
        assert_eq!(reserve_calls.get(), 0);
    }

    #[test]
    fn finite_field_batch_inverse_does_not_downgrade_internal_faults() {
        let invalid_field = FiniteField {
            characteristic: BigInt::from(4),
        };
        let invalid_value = FiniteFieldElement {
            field: invalid_field.clone(),
            value: BigInt::from(2),
        };
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            invalid_field.metered_batch_inv(std::slice::from_ref(&invalid_value), &mut measured),
            Err(BatchInverseError::InvariantViolation(
                "nonzero product in a certified field was not invertible"
            ))
        );

        let mut cancelled = CheckpointMeter::arming_after(measured.checkpoints - 1);
        assert_eq!(
            invalid_field.metered_batch_inv(std::slice::from_ref(&invalid_value), &mut cancelled),
            Err(BatchInverseError::Meter(MeterError::Cancelled))
        );
    }

    proptest! {
        #[test]
        fn finite_field_batch_inverse_matches_scalar_over_generated_batches(
            representatives in proptest::collection::vec(1u64..101, 0..24),
        ) {
            let field = FiniteField::new(BigInt::from(101)).unwrap();
            let values = representatives
                .into_iter()
                .map(|value| field.element(BigInt::from(value)))
                .collect::<Vec<_>>();
            let expected = values
                .iter()
                .map(|value| value.inv().unwrap())
                .collect::<Vec<_>>();

            prop_assert_eq!(
                field.try_batch_inv(&values).unwrap(),
                Some(expected.clone())
            );
            let mut meter = Unbounded;
            prop_assert_eq!(
                field.metered_batch_inv(&values, &mut meter).unwrap(),
                Some(expected)
            );
        }
    }

    #[test]
    fn montgomery_and_barrett_reducers_enforce_their_input_domains() {
        let m = BigInt::from(97);
        let mont = MontgomeryReducer::new(m.clone()).expect("valid odd modulus");
        let a = BigInt::from(35);
        let b = BigInt::from(42);
        let expected_prod = (&a * &b) % &m;

        let a_r = mont.to_montgomery(&a);
        let b_r = mont.to_montgomery(&b);
        let prod_r = mont.mul(&a_r, &b_r).unwrap();
        let actual_prod = mont.from_montgomery(&prod_r).unwrap();
        assert_eq!(actual_prod, expected_prod);
        assert_eq!(
            mont.from_montgomery(&mont.to_montgomery(&BigInt::from(-3))),
            Some(BigInt::from(94))
        );
        assert_eq!(mont.reduce(&BigInt::from(-1)), None);
        assert_eq!(mont.reduce(&mont.reduction_bound), None);
        assert_eq!(mont.mul(&BigInt::from(-1), &b_r), None);

        let barrett = BarrettReducer::new(m.clone()).expect("valid modulus");
        for v in [0i64, 1, 35, 96, 97, 100, 500, 9000] {
            let x = BigInt::from(v);
            assert_eq!(barrett.reduce(&x), Some(&x % &m));
        }
        assert_eq!(barrett.reduce(&BigInt::from(-1)), None);
        assert_eq!(barrett.reduce(&barrett.modulus_squared), None);

        let mut meter = Unbounded;
        let metered_mont = MontgomeryReducer::metered_new(m.clone(), &mut meter)
            .unwrap()
            .unwrap();
        assert_eq!(metered_mont, mont);
        let mut meter = Unbounded;
        let a_r_metered = metered_mont
            .metered_to_montgomery(&a, &mut meter)
            .unwrap()
            .unwrap();
        assert_eq!(a_r_metered, a_r);
        let mut meter = Unbounded;
        assert_eq!(
            metered_mont.metered_mul(&a_r, &b_r, &mut meter).unwrap(),
            Some(prod_r)
        );

        let mut meter = Unbounded;
        let metered_barrett = BarrettReducer::metered_new(m.clone(), &mut meter)
            .unwrap()
            .unwrap();
        assert_eq!(metered_barrett, barrett);
        let mut meter = Unbounded;
        assert_eq!(
            metered_barrett
                .metered_reduce(&BigInt::from(9000), &mut meter)
                .unwrap(),
            Some(BigInt::from(76))
        );
    }

    #[test]
    fn metered_reducer_constructors_match_small_and_limb_boundary_moduli() {
        let smallest_barrett =
            BarrettReducer::new(BigInt::from(2)).expect("two is an admitted Barrett modulus");
        let mut meter = Unbounded;
        assert_eq!(
            BarrettReducer::metered_new(BigInt::from(2), &mut meter).unwrap(),
            Some(smallest_barrett)
        );

        let moduli = [
            BigInt::from(3),
            (BigInt::one() << 31u32) + BigInt::one(),
            (BigInt::one() << 32u32) + BigInt::one(),
            (BigInt::one() << 63u32) + BigInt::one(),
            (BigInt::one() << 64u32) + BigInt::one(),
            (BigInt::one() << 127u32) - BigInt::one(),
        ];

        for modulus in moduli {
            let expected_montgomery =
                MontgomeryReducer::new(modulus.clone()).expect("odd modulus is admitted");
            let mut meter = Unbounded;
            let actual_montgomery = MontgomeryReducer::metered_new(modulus.clone(), &mut meter)
                .unwrap()
                .expect("odd modulus is admitted");
            assert_eq!(actual_montgomery, expected_montgomery);

            let expected_barrett =
                BarrettReducer::new(modulus.clone()).expect("positive modulus is admitted");
            let mut meter = Unbounded;
            let actual_barrett = BarrettReducer::metered_new(modulus.clone(), &mut meter)
                .unwrap()
                .expect("positive modulus is admitted");
            assert_eq!(actual_barrett, expected_barrett);

            let largest_admitted = &modulus * &modulus - BigInt::one();
            assert_eq!(
                actual_barrett.reduce(&largest_admitted),
                Some(&largest_admitted % &modulus)
            );
        }
    }

    #[test]
    fn reducer_constructors_cancel_at_every_observed_safe_point() {
        let modulus = BigInt::from(97);

        let mut observed = CheckpointMeter::default();
        assert!(
            MontgomeryReducer::metered_new(modulus.clone(), &mut observed)
                .unwrap()
                .is_some()
        );
        assert!(observed.checkpoints > 1);
        for checkpoint in 1..=observed.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                MontgomeryReducer::metered_new(modulus.clone(), &mut cancelled),
                Err(MeterError::Cancelled),
                "Montgomery constructor published across checkpoint {checkpoint}"
            );
        }

        let mut observed = CheckpointMeter::default();
        assert!(
            BarrettReducer::metered_new(modulus.clone(), &mut observed)
                .unwrap()
                .is_some()
        );
        assert!(observed.checkpoints > 1);
        for checkpoint in 1..=observed.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                BarrettReducer::metered_new(modulus.clone(), &mut cancelled),
                Err(MeterError::Cancelled),
                "Barrett constructor published across checkpoint {checkpoint}"
            );
        }
    }

    #[test]
    fn reducer_constructors_refuse_one_short_owned_budgets() {
        let modulus = BigInt::from(97);
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut measured = CountingMeter::default();
            assert!(
                MontgomeryReducer::metered_new(modulus.clone(), &mut measured)
                    .unwrap()
                    .is_some()
            );
            assert!(measured.dimensions[dimension.index()] > 0);
            let mut limits = BudgetLimits {
                dimensions: measured.dimensions,
                verifier_pool: 0,
            };
            limits.dimensions[dimension.index()] -= 1;
            let mut budget = Budget::new(limits);
            assert!(matches!(
                MontgomeryReducer::metered_new(modulus.clone(), &mut budget),
                Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension: exhausted,
                    ..
                })) if exhausted == dimension
            ));

            let mut measured = CountingMeter::default();
            assert!(
                BarrettReducer::metered_new(modulus.clone(), &mut measured)
                    .unwrap()
                    .is_some()
            );
            assert!(measured.dimensions[dimension.index()] > 0);
            let mut limits = BudgetLimits {
                dimensions: measured.dimensions,
                verifier_pool: 0,
            };
            limits.dimensions[dimension.index()] -= 1;
            let mut budget = Budget::new(limits);
            assert!(matches!(
                BarrettReducer::metered_new(modulus.clone(), &mut budget),
                Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension: exhausted,
                    ..
                })) if exhausted == dimension
            ));
        }
    }

    #[test]
    fn reducer_constructors_use_the_owned_binary_power_transcript() {
        // MOD-POWER-OWNER-RESTORE: restoring modular's former repeated-doubling helper changes
        // both transcripts. Pinning the smallest useful constructors makes that ownership mutant
        // observable while full-struct comparisons separately guard every derived value.
        let mut montgomery = CountingMeter::default();
        assert_eq!(
            MontgomeryReducer::metered_new(BigInt::from(3), &mut montgomery).unwrap(),
            MontgomeryReducer::new(BigInt::from(3))
        );
        assert_eq!(montgomery.dimensions, [290, 480, 89, 0, 0]);
        assert_eq!(montgomery.checkpoints, 393);

        let mut barrett = CountingMeter::default();
        assert_eq!(
            BarrettReducer::metered_new(BigInt::from(2), &mut barrett).unwrap(),
            BarrettReducer::new(BigInt::from(2))
        );
        assert_eq!(barrett.dimensions, [69, 196, 34, 0, 0]);
        assert_eq!(barrett.checkpoints, 110);
    }

    #[test]
    fn unlucky_prime_diagnostics_are_structured_bounded_and_fail_closed() {
        let p_lucky = BigInt::from(17);
        let p_unlucky = BigInt::from(5);
        let leading_coeffs = vec![BigInt::from(15), BigInt::from(28)];

        assert!(check_lucky_prime(&p_lucky, &leading_coeffs).is_ok());
        let diag = check_lucky_prime(&p_unlucky, &leading_coeffs).unwrap_err();
        assert_eq!(diag.reason, UnluckyPrimeReason::DividesLeadingCoefficient);
        assert_eq!(diag.prime, p_unlucky);
        assert_eq!(diag.coefficient_index, Some(0));

        for invalid in [
            BigInt::from(-3),
            BigInt::zero(),
            BigInt::one(),
            BigInt::from(9),
            deterministic_primality_bound(),
        ] {
            let diag = check_lucky_prime(&invalid, &leading_coeffs).unwrap_err();
            assert_eq!(diag.reason, UnluckyPrimeReason::InvalidPrimeCandidate);
            assert_eq!(diag.coefficient_index, None);
        }

        let mut meter = Unbounded;
        assert_eq!(
            metered_check_lucky_prime(&p_lucky, &leading_coeffs, &mut meter).unwrap(),
            Ok(())
        );
        let mut meter = Unbounded;
        assert_eq!(
            metered_check_lucky_prime(&BigInt::zero(), &leading_coeffs, &mut meter)
                .unwrap()
                .unwrap_err()
                .reason,
            UnluckyPrimeReason::InvalidPrimeCandidate
        );
    }

    #[test]
    fn modular_comparison_uses_bigint_digit_accounting_and_safe_points() {
        let deep_left = (BigInt::one() << 2_048u32) + 1i64;
        let deep_right = (BigInt::one() << 2_048u32) + 2i64;
        let shallow_right = (BigInt::from(2i64) << 2_048u32) + 1i64;

        let mut deep_comparison = CountingMeter::default();
        assert!(!metered_equal(&deep_left, &deep_right, &mut deep_comparison).unwrap());
        assert_eq!(deep_comparison.dimensions, [67, 0, 0, 0, 0]);
        assert_eq!(deep_comparison.checkpoints, 68);

        let mut shallow_comparison = CountingMeter::default();
        assert!(!metered_equal(&deep_left, &shallow_right, &mut shallow_comparison).unwrap());
        assert_eq!(shallow_comparison.dimensions, [3, 0, 0, 0, 0]);
        assert_eq!(shallow_comparison.checkpoints, 4);

        let left_ring = ModularRing::new(deep_left).unwrap();
        let right_ring = ModularRing::new(deep_right).unwrap();
        let left = left_ring.one();
        let right = right_ring.one();
        let mut measured = CountingMeter::default();
        assert_eq!(left.metered_add(&right, &mut measured).unwrap(), None);
        // MODULAR-OPAQUE-CMP-RESTORE: the former aggregate charge plus lhs.cmp(rhs) admits this
        // public refusal with 33 compute steps and only four total checkpoints.
        assert_eq!(measured.dimensions, [67, 0, 0, 0, 0]);
        assert_eq!(measured.checkpoints, 70);

        let mut one_short_limits = BudgetLimits {
            dimensions: measured.dimensions,
            verifier_pool: 0,
        };
        one_short_limits.dimensions[Dimension::ComputeSteps.index()] -= 1;
        let mut one_short = Budget::new(one_short_limits);
        assert!(matches!(
            left.metered_add(&right, &mut one_short),
            Err(MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            }))
        ));

        let mut exact = Budget::new(BudgetLimits {
            dimensions: measured.dimensions,
            verifier_pool: 0,
        });
        assert_eq!(left.metered_add(&right, &mut exact).unwrap(), None);

        for checkpoint in 1..=measured.checkpoints {
            let mut cancelled = CheckpointMeter::cancelling_at(checkpoint);
            assert_eq!(
                left.metered_add(&right, &mut cancelled),
                Err(MeterError::Cancelled),
                "mismatched-ring comparison ignored checkpoint {checkpoint}"
            );
        }
    }

    #[test]
    fn modular_comparison_predicates_match_integer_ordering_boundaries() {
        let deep = (BigInt::one() << 2_048u32) + 1i64;
        let cases = [
            (BigInt::zero(), BigInt::zero()),
            (BigInt::from(-1), BigInt::zero()),
            (BigInt::zero(), BigInt::one()),
            (-(BigInt::one() << 96u32), -(BigInt::one() << 64u32)),
            (BigInt::one() << 96u32, BigInt::one() << 64u32),
            (deep.clone(), deep + 1i64),
        ];

        for (lhs, rhs) in cases {
            let expected = lhs.cmp(&rhs);
            let mut meter = Unbounded;
            assert_eq!(
                metered_equal(&lhs, &rhs, &mut meter).unwrap(),
                expected.is_eq()
            );
            let mut meter = Unbounded;
            assert_eq!(
                metered_greater(&lhs, &rhs, &mut meter).unwrap(),
                expected.is_gt()
            );
            let mut meter = Unbounded;
            assert_eq!(
                metered_greater_or_equal(&lhs, &rhs, &mut meter).unwrap(),
                expected.is_ge()
            );
        }
    }

    #[test]
    fn new_metered_types_check_cancellation_before_terminal_publication() {
        let ring = ModularRing::new(BigInt::from(97)).unwrap();
        let lhs = ring.element(BigInt::from(35));
        let rhs = ring.element(BigInt::from(42));
        let mut measured = CheckpointMeter::default();
        assert!(lhs.metered_mul(&rhs, &mut measured).unwrap().is_some());
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            lhs.metered_mul(&rhs, &mut cancelled),
            Err(MeterError::Cancelled)
        );
        let other_ring = ModularRing::new(BigInt::from(101)).unwrap();
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            lhs.metered_add(&other_ring.one(), &mut measured).unwrap(),
            None
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            lhs.metered_add(&other_ring.one(), &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let field = FiniteField::new(BigInt::from(97)).unwrap();
        let value = field.element(BigInt::from(35));
        let mut measured = CheckpointMeter::default();
        assert_eq!(
            value.metered_pow(&BigInt::from(-1), &mut measured).unwrap(),
            None
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            value.metered_pow(&BigInt::from(-1), &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let mut measured = CheckpointMeter::default();
        let reducer = MontgomeryReducer::metered_new(BigInt::from(97), &mut measured)
            .unwrap()
            .unwrap();
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            MontgomeryReducer::metered_new(BigInt::from(97), &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let value = reducer.to_montgomery(&BigInt::from(35));
        let mut measured = CheckpointMeter::default();
        assert!(
            reducer
                .metered_from_montgomery(&value, &mut measured)
                .unwrap()
                .is_some()
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            reducer.metered_from_montgomery(&value, &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let coefficients = [BigInt::from(15), BigInt::from(28)];
        let mut measured = CheckpointMeter::default();
        assert!(
            metered_check_lucky_prime(&BigInt::from(5), &coefficients, &mut measured)
                .unwrap()
                .is_err()
        );
        let mut cancelled = CheckpointMeter::cancelling_at(measured.checkpoints);
        assert_eq!(
            metered_check_lucky_prime(&BigInt::from(5), &coefficients, &mut cancelled),
            Err(MeterError::Cancelled)
        );

        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            MontgomeryReducer::metered_new(BigInt::from(97), &mut budget),
            Err(MeterError::Budget(_))
        ));
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        assert!(matches!(
            BarrettReducer::metered_new(BigInt::from(97), &mut budget),
            Err(MeterError::Budget(_))
        ));
    }

    proptest! {
        #[test]
        fn reducers_match_scalar_remainders_over_their_full_admitted_ranges(
            modulus in 2u64..500,
            value_seed in any::<u64>(),
            lhs in -10_000i64..10_001,
            rhs in -10_000i64..10_001,
        ) {
            let modulus_big = BigInt::from(modulus);
            let modulus_squared = modulus.saturating_mul(modulus);
            let value = value_seed % modulus_squared;
            let barrett = BarrettReducer::new(modulus_big.clone()).unwrap();
            let expected = BigInt::from(value % modulus);
            prop_assert_eq!(barrett.reduce(&BigInt::from(value)), Some(expected.clone()));
            let mut meter = Unbounded;
            prop_assert_eq!(
                barrett.metered_reduce(&BigInt::from(value), &mut meter).unwrap(),
                Some(expected)
            );

            if modulus > 2 && modulus % 2 == 1 {
                let montgomery = MontgomeryReducer::new(modulus_big.clone()).unwrap();
                let lhs_mont = montgomery.to_montgomery(&BigInt::from(lhs));
                let rhs_mont = montgomery.to_montgomery(&BigInt::from(rhs));
                let product_mont = montgomery.mul(&lhs_mont, &rhs_mont).unwrap();
                let product = montgomery.from_montgomery(&product_mont).unwrap();
                let expected_product =
                    BigInt::from(lhs.rem_euclid(modulus as i64))
                        * BigInt::from(rhs.rem_euclid(modulus as i64))
                        % &modulus_big;
                prop_assert_eq!(product, expected_product);
            }
        }
    }
}
