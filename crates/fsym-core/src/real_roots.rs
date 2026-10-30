//! Complete bounded isolation of the real roots of rational polynomials.
//!
//! A generator bisects rational intervals. The reference checker independently
//! counts roots of the original polynomial in every supplied interval, counts
//! the complete real line using a strict Cauchy bound, and derives multiplicity
//! from successive gcd(P, P') layers. It never repeats the generator's search.
//! Only checked receipts are published. Complex-root order, irreducibility and
//! analytic/function domains are not claims of this coefficient-level API.
//!
//! All rational primitives have cancellation safe points, growth preflights,
//! and cumulative work/allocation limits, including the reference checker.
//! Separate caller meters do not by themselves reserve a runtime verifier pool.

use crate::{BigInt, BigRational};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use num_traits::{One, Signed, Zero};
use thiserror::Error;

pub const MAX_REAL_ROOT_DEGREE: usize = 64;
pub const MAX_REAL_ROOT_INPUT_BITS: u64 = 4096;
pub const MAX_REAL_ROOT_DIGITS: u32 = 512;
const MAX_WORK_BITS: u64 = 32_768;
const MAX_WORK: u64 = 100_000_000;
const MAX_ALLOCATION: u64 = 256 * 1024 * 1024;
const MAX_BISECTIONS: usize = 32_768;

type Poly = Vec<BigRational>;

/// One distinct real root, including its multiplicity in the original input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealRootInterval {
    pub lower: BigRational,
    pub upper: BigRational,
    pub multiplicity: usize,
}

/// Untrusted proposal. Constructing this value confers no acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealRootIsolationCertificate {
    pub intervals: Vec<RealRootInterval>,
}

/// Complete, sorted real-root isolation bound to exact input coefficients.
/// Fields are private and there is no unchecked deserializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRealRoots {
    coefficients: Poly,
    certificate: RealRootIsolationCertificate,
}

impl VerifiedRealRoots {
    pub fn coefficients(&self) -> &[BigRational] {
        &self.coefficients
    }
    pub fn intervals(&self) -> &[RealRootInterval] {
        &self.certificate.intervals
    }
    pub fn certificate(&self) -> &RealRootIsolationCertificate {
        &self.certificate
    }
    pub fn count_with_multiplicity(&self) -> usize {
        self.intervals().iter().map(|root| root.multiplicity).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RealRootError {
    #[error("the zero polynomial does not have a finite root set")]
    ZeroPolynomial,
    #[error("real-root resource limit exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("invalid real-root certificate: {0}")]
    InvalidCertificate(&'static str),
    #[error("real-root index is outside the distinct real-root set")]
    InvalidIndex,
    #[error("real-root precision must be between 1 and 512 decimal digits")]
    InvalidPrecision,
    #[error("real-root arithmetic invariant failed: {0}")]
    Arithmetic(&'static str),
    #[error("real-root work stopped: {0}")]
    Meter(#[from] MeterError),
}

struct Work<'a, M: BudgetMeter> {
    meter: &'a mut M,
    compute: u64,
    allocated: u64,
}

fn bits(q: &BigRational) -> u64 {
    q.numer().bits().max(q.denom().bits()).max(1)
}

fn trim(p: &mut Poly) {
    while p.len() > 1 && p.last().is_some_and(Zero::is_zero) {
        p.pop();
    }
    if p.is_empty() {
        p.push(BigRational::zero());
    }
}
fn is_zero(p: &[BigRational]) -> bool {
    p.iter().all(Zero::is_zero)
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Result<Self, RealRootError> {
        meter.checkpoint()?;
        Ok(Self {
            meter,
            compute: 0,
            allocated: 0,
        })
    }
    fn charge(&mut self, steps: u64, bytes: u64) -> Result<(), RealRootError> {
        self.meter.checkpoint()?;
        self.compute = self
            .compute
            .checked_add(steps.max(1))
            .ok_or(RealRootError::ResourceLimit("work"))?;
        self.allocated = self
            .allocated
            .checked_add(bytes.max(1))
            .ok_or(RealRootError::ResourceLimit("allocation"))?;
        if self.compute > MAX_WORK || self.allocated > MAX_ALLOCATION {
            return Err(RealRootError::ResourceLimit("cumulative work/allocation"));
        }
        self.meter.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1)),
            (Dimension::MemoryBytes, bytes.max(1)),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }
    fn scalar(&mut self, q: &BigRational) -> Result<(), RealRootError> {
        if bits(q) > MAX_WORK_BITS {
            return Err(RealRootError::ResourceLimit("intermediate rational bits"));
        }
        self.charge(1, 32 + (q.numer().bits() + q.denom().bits()).div_ceil(8))
    }
    fn input(&mut self, p: &[BigRational]) -> Result<Poly, RealRootError> {
        if p.is_empty() || p.len() > MAX_REAL_ROOT_DEGREE + 1 {
            return Err(RealRootError::ResourceLimit("coefficient count"));
        }
        for q in p {
            if bits(q) > MAX_REAL_ROOT_INPUT_BITS {
                return Err(RealRootError::ResourceLimit("input coefficient bits"));
            }
            self.scalar(q)?;
        }
        let mut out = p.to_vec();
        trim(&mut out);
        if is_zero(&out) {
            return Err(RealRootError::ZeroPolynomial);
        }
        Ok(out)
    }
    fn binary_preflight(&mut self, a: &BigRational, b: &BigRational) -> Result<(), RealRootError> {
        let growth = bits(a).saturating_add(bits(b)).saturating_add(1);
        if growth > MAX_WORK_BITS {
            return Err(RealRootError::ResourceLimit("rational operation growth"));
        }
        let limbs = growth.div_ceil(32).max(1);
        self.charge(limbs.saturating_mul(limbs), 64 + growth.div_ceil(8) * 8)
    }
    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, RealRootError> {
        self.binary_preflight(a, b)?;
        Ok(a + b)
    }
    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, RealRootError> {
        self.binary_preflight(a, b)?;
        Ok(a - b)
    }
    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, RealRootError> {
        self.binary_preflight(a, b)?;
        Ok(a * b)
    }
    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, RealRootError> {
        if b.is_zero() {
            return Err(RealRootError::Arithmetic("division by zero"));
        }
        self.binary_preflight(a, b)?;
        Ok(a / b)
    }
    fn copy_poly(&mut self, p: &[BigRational]) -> Result<Poly, RealRootError> {
        if p.len() > MAX_REAL_ROOT_DEGREE + 1 {
            return Err(RealRootError::ResourceLimit("intermediate degree"));
        }
        for q in p {
            self.scalar(q)?;
        }
        Ok(p.to_vec())
    }
    fn derivative(&mut self, p: &[BigRational]) -> Result<Poly, RealRootError> {
        let mut out = Vec::with_capacity(p.len().saturating_sub(1).max(1));
        self.charge(1, (p.len() as u64 + 1) * 32)?;
        for (i, q) in p.iter().enumerate().skip(1) {
            out.push(self.mul(q, &BigRational::from_integer(BigInt::from(i as u64)))?);
        }
        trim(&mut out);
        Ok(out)
    }
    fn division(
        &mut self,
        a: &[BigRational],
        b: &[BigRational],
    ) -> Result<(Poly, Poly), RealRootError> {
        if is_zero(b) {
            return Err(RealRootError::Arithmetic("zero polynomial divisor"));
        }
        let mut r = self.copy_poly(a)?;
        let mut q = vec![BigRational::zero(); a.len().saturating_sub(b.len()) + 1];
        self.charge(1, q.len() as u64 * 32)?;
        while !is_zero(&r) && r.len() >= b.len() {
            let shift = r.len() - b.len();
            let c = self.div(r.last().unwrap(), b.last().unwrap())?;
            q[shift] = c.clone();
            for (j, bj) in b.iter().enumerate() {
                let product = self.mul(&c, bj)?;
                r[shift + j] = self.sub(&r[shift + j], &product)?;
            }
            trim(&mut r);
        }
        trim(&mut q);
        Ok((q, r))
    }
    fn normalized(&mut self, p: &[BigRational], positive: bool) -> Result<Poly, RealRootError> {
        if is_zero(p) {
            return Ok(vec![BigRational::zero()]);
        }
        let lead = if positive {
            p.last().unwrap().abs()
        } else {
            p.last().unwrap().clone()
        };
        let mut out = Vec::with_capacity(p.len());
        self.charge(1, p.len() as u64 * 32)?;
        for q in p {
            out.push(self.div(q, &lead)?);
        }
        Ok(out)
    }
    fn gcd(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Poly, RealRootError> {
        let mut a = self.normalized(a, false)?;
        let mut b = self.normalized(b, false)?;
        while !is_zero(&b) {
            let (_, r) = self.division(&a, &b)?;
            a = b;
            b = self.normalized(&r, false)?;
        }
        self.normalized(&a, false)
    }
    fn square_free(&mut self, p: &[BigRational]) -> Result<Poly, RealRootError> {
        let d = self.derivative(p)?;
        let g = self.gcd(p, &d)?;
        let (q, r) = self.division(p, &g)?;
        if !is_zero(&r) {
            return Err(RealRootError::Arithmetic("gcd division remainder"));
        }
        self.normalized(&q, false)
    }
    fn sturm(&mut self, square_free: &[BigRational]) -> Result<Vec<Poly>, RealRootError> {
        let p = self.normalized(square_free, true)?;
        if p.len() == 1 {
            return Ok(vec![p]);
        }
        let d = self.derivative(&p)?;
        let d = self.normalized(&d, true)?;
        let mut seq = vec![p, d];
        while seq.last().unwrap().len() > 1 {
            let n = seq.len();
            let (_, r) = self.division(&seq[n - 2], &seq[n - 1])?;
            if is_zero(&r) {
                return Err(RealRootError::Arithmetic("non square-free Sturm input"));
            }
            let mut r = self.normalized(&r, true)?;
            for c in &mut r {
                self.scalar(c)?;
                *c = -c.clone();
            }
            seq.push(r);
        }
        Ok(seq)
    }
    fn eval(&mut self, p: &[BigRational], x: &BigRational) -> Result<BigRational, RealRootError> {
        let mut out = BigRational::zero();
        for c in p.iter().rev() {
            let product = self.mul(&out, x)?;
            out = self.add(&product, c)?;
        }
        Ok(out)
    }
    fn variations(&mut self, seq: &[Poly], x: &BigRational) -> Result<usize, RealRootError> {
        let mut previous = 0i8;
        let mut count = 0;
        for p in seq {
            let q = self.eval(p, x)?;
            let sign = if q.is_positive() {
                1
            } else if q.is_negative() {
                -1
            } else {
                0
            };
            if sign != 0 {
                if previous != 0 && previous != sign {
                    count += 1;
                }
                previous = sign;
            }
        }
        Ok(count)
    }
    fn count(
        &mut self,
        seq: &[Poly],
        lo: &BigRational,
        hi: &BigRational,
    ) -> Result<usize, RealRootError> {
        if lo > hi {
            return Err(RealRootError::InvalidCertificate("reversed interval"));
        }
        let lower_root = self.eval(&seq[0], lo)?.is_zero();
        if lo == hi {
            return Ok(usize::from(lower_root));
        }
        let left = self.variations(seq, lo)?;
        let right = self.variations(seq, hi)?;
        left.checked_sub(right)
            .and_then(|n| n.checked_add(usize::from(lower_root)))
            .ok_or(RealRootError::Arithmetic("negative Sturm count"))
    }
    fn bound(&mut self, p: &[BigRational]) -> Result<BigRational, RealRootError> {
        let lead = p.last().unwrap().abs();
        let mut ratio = BigRational::zero();
        for c in &p[..p.len() - 1] {
            ratio = ratio.max(self.div(&c.abs(), &lead)?);
        }
        self.add(&ratio, &BigRational::from_integer(2.into()))
    }
    fn midpoint(
        &mut self,
        lo: &BigRational,
        hi: &BigRational,
    ) -> Result<BigRational, RealRootError> {
        let sum = self.add(lo, hi)?;
        self.div(&sum, &BigRational::from_integer(2.into()))
    }
    fn layers(&mut self, p: &[BigRational]) -> Result<Vec<Vec<Poly>>, RealRootError> {
        let mut remaining = self.copy_poly(p)?;
        let mut out = Vec::new();
        while remaining.len() > 1 {
            let sf = self.square_free(&remaining)?;
            out.push(self.sturm(&sf)?);
            let derivative = self.derivative(&remaining)?;
            remaining = self.gcd(&remaining, &derivative)?;
        }
        Ok(out)
    }
    fn product(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Poly, RealRootError> {
        let len = a.len() + b.len() - 1;
        if len > MAX_REAL_ROOT_DEGREE + 1 {
            return Err(RealRootError::ResourceLimit("factor product degree"));
        }
        self.charge(1, len as u64 * 32)?;
        let mut out = vec![BigRational::zero(); len];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                let product = self.mul(ai, bj)?;
                out[i + j] = self.add(&out[i + j], &product)?;
            }
        }
        trim(&mut out);
        Ok(out)
    }
}

/// Independent reference verification: no bisection or candidate search runs.
pub fn verify_real_root_isolation(
    coefficients: &[BigRational],
    certificate: &RealRootIsolationCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRoots, RealRootError> {
    let mut work = Work::new(meter)?;
    let p = work.input(coefficients)?;
    if certificate.intervals.len() > p.len() - 1 {
        return Err(RealRootError::InvalidCertificate("too many distinct roots"));
    }
    if p.len() == 1 {
        return Ok(VerifiedRealRoots {
            coefficients: p,
            certificate: certificate.clone(),
        });
    }
    let layers = work.layers(&p)?;
    let bound = work.bound(&p)?;
    let total = work.count(&layers[0], &-bound.clone(), &bound)?;
    if total != certificate.intervals.len() {
        return Err(RealRootError::InvalidCertificate(
            "incomplete distinct real-root set",
        ));
    }
    let mut previous: Option<&RealRootInterval> = None;
    for interval in &certificate.intervals {
        work.scalar(&interval.lower)?;
        work.scalar(&interval.upper)?;
        if interval.multiplicity == 0 || interval.multiplicity > p.len() - 1 {
            return Err(RealRootError::InvalidCertificate("invalid multiplicity"));
        }
        if let Some(before) = previous
            && (before.upper > interval.lower
                || (before.upper == interval.lower
                    && work.eval(&layers[0][0], &interval.lower)?.is_zero()))
        {
            return Err(RealRootError::InvalidCertificate(
                "overlapping or unsorted intervals",
            ));
        }
        if work.count(&layers[0], &interval.lower, &interval.upper)? != 1 {
            return Err(RealRootError::InvalidCertificate(
                "interval must contain exactly one distinct root",
            ));
        }
        let mut multiplicity = 0;
        for seq in &layers {
            multiplicity += work.count(seq, &interval.lower, &interval.upper)?;
        }
        if interval.multiplicity != multiplicity {
            return Err(RealRootError::InvalidCertificate(
                "wrong original multiplicity",
            ));
        }
        previous = Some(interval);
    }
    work.charge(1, (certificate.intervals.len() as u64 + 1) * 96)?;
    Ok(VerifiedRealRoots {
        coefficients: p,
        certificate: certificate.clone(),
    })
}

fn generate<M: BudgetMeter>(
    p: &[BigRational],
    work: &mut Work<'_, M>,
) -> Result<RealRootIsolationCertificate, RealRootError> {
    if p.len() == 1 {
        return Ok(RealRootIsolationCertificate {
            intervals: Vec::new(),
        });
    }
    let layers = work.layers(p)?;
    let seq = &layers[0];
    let bound = work.bound(p)?;
    let lower = -bound.clone();
    let count = work.count(seq, &lower, &bound)?;
    let mut pending = vec![(lower, bound, count)];
    let mut intervals = Vec::new();
    let mut splits = 0;
    while let Some((lo, hi, count)) = pending.pop() {
        work.charge(1, 96)?;
        if count == 0 {
            continue;
        }
        if count == 1 {
            let mut multiplicity = 0;
            for layer in &layers {
                multiplicity += work.count(layer, &lo, &hi)?;
            }
            intervals.push(RealRootInterval {
                lower: lo,
                upper: hi,
                multiplicity,
            });
            continue;
        }
        splits += 1;
        if splits > MAX_BISECTIONS {
            return Err(RealRootError::ResourceLimit("isolation bisections"));
        }
        let mut mid = work.midpoint(&lo, &hi)?;
        // Split at a nonroot so closed child intervals never share a root.
        // At most degree many distinct rational points can be roots.
        if work.eval(&seq[0], &mid)?.is_zero() {
            let width = work.sub(&hi, &lo)?;
            for divisor in 3..=p.len() + 2 {
                let fraction = work.div(
                    &width,
                    &BigRational::from_integer(BigInt::from(divisor as u64)),
                )?;
                mid = work.add(&lo, &fraction)?;
                if !work.eval(&seq[0], &mid)?.is_zero() {
                    break;
                }
            }
            if work.eval(&seq[0], &mid)?.is_zero() {
                return Err(RealRootError::Arithmetic(
                    "could not choose a nonroot split",
                ));
            }
        }
        let left = work.count(seq, &lo, &mid)?;
        let right = count
            .checked_sub(left)
            .ok_or(RealRootError::Arithmetic("partition count"))?;
        pending.push((mid.clone(), hi, right));
        pending.push((lo, mid, left));
    }
    Ok(RealRootIsolationCertificate { intervals })
}

/// Generate and independently check the complete real-root set, preserving
/// multiplicities. Coefficients are in ascending degree. A constant has no
/// roots; the zero polynomial is refused.
pub fn isolate_real_roots(
    coefficients: &[BigRational],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRoots, RealRootError> {
    let mut work = Work::new(generator_meter)?;
    let p = work.input(coefficients)?;
    let certificate = generate(&p, &mut work)?;
    verify_real_root_isolation(coefficients, &certificate, verifier_meter)
}

fn refine_once<M: BudgetMeter>(
    root: &mut RealRootInterval,
    seq: &[Poly],
    work: &mut Work<'_, M>,
) -> Result<(), RealRootError> {
    if root.lower == root.upper {
        return Ok(());
    }
    let mid = work.midpoint(&root.lower, &root.upper)?;
    if work.eval(&seq[0], &mid)?.is_zero() {
        root.lower = mid.clone();
        root.upper = mid;
    } else if work.count(seq, &root.lower, &mid)? == 1 {
        root.upper = mid;
    } else {
        root.lower = mid;
    }
    Ok(())
}

/// Refine one indexed root of an already checked complete isolation.
///
/// The generator only bisects its existing interval. The independent checker
/// subsequently recounts every original root and multiplicity before the new
/// immutable isolation is returned. Both caller meters remain cumulative.
pub fn refine_real_root_isolation(
    roots: &VerifiedRealRoots,
    index: usize,
    bisections: usize,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRoots, RealRootError> {
    let mut work = Work::new(generator_meter)?;
    if bisections > MAX_BISECTIONS {
        return Err(RealRootError::ResourceLimit("refinement bisections"));
    }
    if index >= roots.intervals().len() {
        return Err(RealRootError::InvalidIndex);
    }
    work.charge(1, roots.intervals().len() as u64 * 96)?;
    for interval in roots.intervals() {
        work.scalar(&interval.lower)?;
        work.scalar(&interval.upper)?;
    }
    let mut certificate = roots.certificate().clone();
    let square_free = work.square_free(roots.coefficients())?;
    let sequence = work.sturm(&square_free)?;
    let interval = &mut certificate.intervals[index];
    if bisections != 0 && interval.lower != interval.upper {
        // An independently admitted interval contains exactly one root.
        // Its nonroot endpoints therefore differ by one Sturm variation.
        // Retain the left variation count while bisecting; repeatedly
        // recounting both old endpoints makes high precision needlessly
        // expensive. The full independent checker still runs afterwards.
        if work.eval(&square_free, &interval.lower)?.is_zero() {
            work.scalar(&interval.lower)?;
            interval.upper = interval.lower.clone();
        } else if work.eval(&square_free, &interval.upper)?.is_zero() {
            work.scalar(&interval.upper)?;
            interval.lower = interval.upper.clone();
        } else {
            let mut left_variations = work.variations(&sequence, &interval.lower)?;
            for _ in 0..bisections {
                let midpoint = work.midpoint(&interval.lower, &interval.upper)?;
                if work.eval(&square_free, &midpoint)?.is_zero() {
                    work.scalar(&midpoint)?;
                    interval.lower = midpoint.clone();
                    interval.upper = midpoint;
                    break;
                }
                let middle_variations = work.variations(&sequence, &midpoint)?;
                match left_variations.checked_sub(middle_variations) {
                    Some(1) => interval.upper = midpoint,
                    Some(0) => {
                        interval.lower = midpoint;
                        left_variations = middle_variations;
                    }
                    _ => {
                        return Err(RealRootError::Arithmetic(
                            "refinement interval lost its unique root",
                        ));
                    }
                }
            }
        }
    }
    let result = verify_real_root_isolation(roots.coefficients(), &certificate, verifier_meter)?;
    work.meter.checkpoint()?;
    verifier_meter.checkpoint()?;
    Ok(result)
}

/// Decide exactly whether `polynomial` vanishes at one checked indexed root.
///
/// A gcd restricts the roots to those shared with the defining polynomial;
/// Sturm counting in the already isolating interval then selects the exact
/// embedding. No approximate root proximity or factor irreducibility is used.
/// This reference decision also handles repeated and reducible definitions.
pub fn real_root_polynomial_zero(
    roots: &VerifiedRealRoots,
    index: usize,
    polynomial: &[BigRational],
    meter: &mut impl BudgetMeter,
) -> Result<bool, RealRootError> {
    let mut work = Work::new(meter)?;
    let root = roots
        .intervals()
        .get(index)
        .ok_or(RealRootError::InvalidIndex)?;
    if polynomial.is_empty() || polynomial.len() > MAX_REAL_ROOT_DEGREE + 1 {
        return Err(RealRootError::ResourceLimit("value polynomial degree"));
    }
    let mut polynomial = work.copy_poly(polynomial)?;
    trim(&mut polynomial);
    let result = if root.lower == root.upper {
        work.eval(&polynomial, &root.lower)?.is_zero()
    } else if is_zero(&polynomial) {
        true
    } else {
        let common = work.gcd(roots.coefficients(), &polynomial)?;
        if common.len() == 1 {
            false
        } else {
            let square_free = work.square_free(&common)?;
            let sequence = work.sturm(&square_free)?;
            work.count(&sequence, &root.lower, &root.upper)? == 1
        }
    };
    work.meter.checkpoint()?;
    Ok(result)
}

/// Bounded round-half-to-even serialization shared by indexed-root values.
/// The pair is a decimal mantissa and base-ten exponent; it is an exact
/// rounding decision on the input rational, not an interval certificate.
pub fn metered_round_rational(
    value: &BigRational,
    digits: u32,
    meter: &mut impl BudgetMeter,
) -> Result<(String, i64), RealRootError> {
    let mut work = Work::new(meter)?;
    if !(1..=MAX_REAL_ROOT_DIGITS).contains(&digits) {
        return Err(RealRootError::InvalidPrecision);
    }
    work.scalar(value)?;
    work.charge(
        (bits(value) + u64::from(digits)).saturating_mul(8),
        (bits(value) + u64::from(digits)).saturating_mul(4),
    )?;
    let result = if value.is_zero() {
        ("0".into(), 0)
    } else {
        crate::round_significant(value, digits)
    };
    work.meter.checkpoint()?;
    Ok(result)
}

// Decode the bounded decimal *number* produced by round_significant, not
// an expression or printed symbolic IR. This exact candidate is used only
// to resolve a rounding boundary by polynomial substitution.
fn decimal_rational<M: BudgetMeter>(
    rounded: &(String, i64),
    work: &mut Work<'_, M>,
) -> Result<BigRational, RealRootError> {
    if rounded.0 == "0" {
        return Ok(BigRational::zero());
    }
    let fractional = rounded
        .0
        .split_once('.')
        .map_or(0, |(_, digits)| digits.len());
    let exponent = rounded.1 - fractional as i64;
    let power = u32::try_from(exponent.unsigned_abs())
        .map_err(|_| RealRootError::ResourceLimit("decimal exponent"))?;
    if u64::from(power) * 4 > MAX_WORK_BITS {
        return Err(RealRootError::ResourceLimit("decimal exponent bits"));
    }
    work.charge(
        (u64::from(power) + 1).saturating_mul(8),
        (u64::from(power) + 1).saturating_mul(8),
    )?;
    let text = rounded.0.replace('.', "");
    let mantissa = BigInt::parse_bytes(text.as_bytes(), 10)
        .ok_or(RealRootError::Arithmetic("invalid generated decimal"))?;
    let scale = BigInt::from(10).pow(power);
    let result = if exponent >= 0 {
        BigRational::from_integer(mantissa * scale)
    } else {
        BigRational::new(mantissa, scale)
    };
    work.scalar(&result)?;
    Ok(result)
}

/// Refine one distinct real root until both exact enclosure endpoints round
/// to the same decimal. Reverify the complete root certificate after refining.
/// The returned text has `digits` significant digits, round-half-to-even;
/// it is a rounded approximation, not itself an enclosing interval.
pub fn real_root_decimal(
    coefficients: &[BigRational],
    index: usize,
    digits: u32,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<String, RealRootError> {
    if !(1..=MAX_REAL_ROOT_DIGITS).contains(&digits) {
        return Err(RealRootError::InvalidPrecision);
    }
    let mut work = Work::new(generator_meter)?;
    let p = work.input(coefficients)?;
    let mut certificate = generate(&p, &mut work)?;
    let root = certificate
        .intervals
        .get_mut(index)
        .ok_or(RealRootError::InvalidIndex)?;
    let sf = work.square_free(&p)?;
    let seq = work.sturm(&sf)?;
    // Bisection need not land on zero (or a terminating decimal) when the
    // initial rational bracket has an unrelated denominator. Recognize zero
    // algebraically before asking two rounded endpoints to agree.
    if root.lower <= BigRational::zero() && root.upper >= BigRational::zero() && sf[0].is_zero() {
        root.lower = BigRational::zero();
        root.upper = BigRational::zero();
    }
    let rounded = |q: &BigRational| {
        if q.is_zero() {
            ("0".into(), 0)
        } else {
            crate::round_significant(q, digits)
        }
    };
    let mut result = None;
    for _ in 0..MAX_BISECTIONS {
        // Decimal integer powers/rendering are bounded by coefficient and
        // endpoint bit caps plus the public 512-digit cap.
        work.charge(
            (bits(&root.lower) + bits(&root.upper) + digits as u64).saturating_mul(8),
            (bits(&root.lower) + bits(&root.upper) + digits as u64).saturating_mul(4),
        )?;
        let a = rounded(&root.lower);
        let b = rounded(&root.upper);
        if a == b {
            result = Some(if a.0 == "0" {
                a.0
            } else {
                format!("{}e{}", a.0, a.1)
            });
            break;
        }
        // At an exact rounding tie, arbitrarily narrow intervals can round
        // differently forever. The midpoint of the two rounded numbers is
        // a rational candidate for that tie; accept it only after exact
        // substitution into the square-free input and enclosure membership.
        let ar = decimal_rational(&a, &mut work)?;
        let br = decimal_rational(&b, &mut work)?;
        let boundary = work.midpoint(&ar, &br)?;
        if boundary >= root.lower && boundary <= root.upper && work.eval(&sf, &boundary)?.is_zero()
        {
            root.lower = boundary.clone();
            root.upper = boundary;
        } else {
            refine_once(root, &seq, &mut work)?;
        }
    }
    let result = result.ok_or(RealRootError::ResourceLimit(
        "decimal refinement bisections",
    ))?;
    verify_real_root_isolation(coefficients, &certificate, verifier_meter)?;
    Ok(result)
}

/// Bind each accepted real root to a square-free factor and its increasing
/// real-root index. The monic factor product, including all multiplicities,
/// must equal the original polynomial; floating proximity is never used.
/// This does not establish factor irreducibility.
pub fn identify_real_root_factors(
    roots: &VerifiedRealRoots,
    factors: &[Poly],
    multiplicities: &[usize],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<(usize, usize)>, RealRootError> {
    if factors.len() != multiplicities.len() || factors.len() > MAX_REAL_ROOT_DEGREE {
        return Err(RealRootError::InvalidCertificate("factor list shape"));
    }
    let mut work = Work::new(meter)?;
    let mut product = vec![BigRational::one()];
    let mut sequences = Vec::new();
    let mut degree = 0usize;
    for (factor, &multiplicity) in factors.iter().zip(multiplicities) {
        let p = work.input(factor)?;
        if p.len() == 1 || multiplicity == 0 || multiplicity > MAX_REAL_ROOT_DEGREE {
            return Err(RealRootError::InvalidCertificate(
                "nonconstant factor/multiplicity required",
            ));
        }
        degree = degree
            .checked_add((p.len() - 1) * multiplicity)
            .ok_or(RealRootError::ResourceLimit("factor degree"))?;
        if degree > MAX_REAL_ROOT_DEGREE {
            return Err(RealRootError::ResourceLimit("factor degree"));
        }
        let p = work.normalized(&p, false)?;
        let d = work.derivative(&p)?;
        if work.gcd(&p, &d)?.len() != 1 {
            return Err(RealRootError::InvalidCertificate(
                "factor is not square-free",
            ));
        }
        sequences.push(work.sturm(&p)?);
        for _ in 0..multiplicity {
            product = work.product(&product, &p)?;
        }
    }
    if product != work.normalized(roots.coefficients(), false)? {
        return Err(RealRootError::InvalidCertificate(
            "factor product differs from original polynomial",
        ));
    }
    let mut local_indices = vec![0usize; factors.len()];
    let mut result = Vec::new();
    for root in roots.intervals() {
        let mut owner = None;
        for (i, seq) in sequences.iter().enumerate() {
            if work.count(seq, &root.lower, &root.upper)? == 1 {
                if owner.is_some() {
                    return Err(RealRootError::InvalidCertificate("non-coprime factor list"));
                }
                owner = Some(i);
            }
        }
        let owner = owner.ok_or(RealRootError::InvalidCertificate(
            "root missing from factor list",
        ))?;
        if root.multiplicity != multiplicities[owner] {
            return Err(RealRootError::InvalidCertificate(
                "factor multiplicity differs from original",
            ));
        }
        work.charge(1, 32)?;
        result.push((owner, local_indices[owner]));
        local_indices[owner] += 1;
    }
    Ok(result)
}
