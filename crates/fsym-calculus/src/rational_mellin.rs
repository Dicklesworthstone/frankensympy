//! Checked ordinary inverse Mellin transforms of proper rational functions.
//!
//! For x > 0, t = -log(x) changes the Mellin transform into a bilateral
//! Laplace transform. The independent rational Laplace checker establishes
//! the complete coefficient identity. A separate strip check assigns every
//! pole to the left or right of the supplied open strip, including every
//! original rational guard hidden by cancellation or a zeroth power.
//!
//! A left mode h(t) contributes h(-log(x))*H(1-x); a right mode contributes
//! -h(-log(x))*H(x-1). Real quadratic modes may split across the strip.
//! Repeated poles retain their full polynomial weights. Values at x = 1 use
//! the usual Heaviside convention; a Mellin transform cannot determine an
//! isolated point value. These are bounded in-process receipts, not portable
//! proof objects or a distributional inverse-transform claim.

#![forbid(unsafe_code)]

use crate::rational_laplace::{
    LaplaceBlock, RationalLaplaceCertificate, RationalLaplaceError,
    generate_rational_laplace_certificate, original_rational_guard_polynomials,
    verify_rational_laplace_certificate,
};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::{RationalArithmeticError, metered_mul, metered_sqrt_floor};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{PolyError, metered_complete_factorization};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::sync::Arc;
use thiserror::Error;

const MAX_DEGREE: usize = 32;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_BITS: u64 = 65_536;
const MAX_TEXT: usize = 65_536;
const MAX_OUTPUT_NODES: usize = 8192;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;
type Result<T> = std::result::Result<T, RationalMellinError>;

#[derive(Debug, Error)]
pub enum RationalMellinError {
    #[error("rational inverse Mellin transform unsupported: {0}")]
    Unsupported(&'static str),
    #[error("rational inverse Mellin strip refused: {0}")]
    InvalidStrip(&'static str),
    #[error("rational inverse Mellin source refused: {0}")]
    InvalidSource(&'static str),
    #[error("rational inverse Mellin verification failed: {0}")]
    Verification(&'static str),
    #[error("rational inverse Mellin envelope exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error(transparent)]
    Laplace(#[from] RationalLaplaceError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Polynomial(#[from] PolyError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalMellinError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

impl RationalMellinError {
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
            || matches!(self, Self::Laplace(error) if error.is_unsupported())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MellinStripBound {
    NegativeInfinity,
    Finite(BigRational),
    PositiveInfinity,
}

/// The whole nonempty open strip lower < Re(s) < upper is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MellinStrip {
    pub lower: MellinStripBound,
    pub upper: MellinStripBound,
}

/// Location of a block's poles relative to the supplied strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MellinSupport {
    /// All poles are at or left of the lower boundary: support 0 < x < 1.
    Left,
    /// All poles are at or right of the upper boundary: support x > 1.
    Right,
    /// A real quadratic has its smaller pole left and larger pole right.
    Split,
}

/// Untrusted coefficient identity and ordered strip assignments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalMellinCertificate {
    pub laplace: RationalLaplaceCertificate,
    pub sides: Vec<MellinSupport>,
}

/// Exact inverse on the positive real x axis with absolute convergence on
/// the supplied open strip. No assertion is made about x <= 0.
#[derive(Debug, Clone)]
pub struct VerifiedRationalMellin {
    source: Expr,
    transform_variable: Symbol,
    argument_variable: Symbol,
    strip: MellinStrip,
    certificate: RationalMellinCertificate,
    function: Expr,
    original_guard_count: usize,
}

impl VerifiedRationalMellin {
    pub fn source(&self) -> &Expr {
        &self.source
    }
    pub fn transform_variable(&self) -> &Symbol {
        &self.transform_variable
    }
    pub fn argument_variable(&self) -> &Symbol {
        &self.argument_variable
    }
    pub fn strip(&self) -> &MellinStrip {
        &self.strip
    }
    pub fn certificate(&self) -> &RationalMellinCertificate {
        &self.certificate
    }
    pub fn function(&self) -> &Expr {
        &self.function
    }
    pub fn original_guard_count(&self) -> usize {
        self.original_guard_count
    }
    pub fn into_function(self) -> Expr {
        self.function
    }
}

struct Capped<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = (|| {
            let mut next = self.used;
            for &(dimension, amount) in charges {
                let (index, limit) = match dimension {
                    Dimension::ComputeSteps => (0, MAX_COMPUTE),
                    Dimension::MemoryBytes => (1, MAX_MEMORY),
                    Dimension::AllocationCount => (2, MAX_COMPUTE),
                    _ => continue,
                };
                let remaining = limit.saturating_sub(next[index]);
                if amount > remaining {
                    return Err(MeterError::Budget(BudgetError::Exhausted {
                        dimension,
                        requested: amount,
                        remaining,
                    }));
                }
                next[index] += amount;
            }
            self.inner.charge_batch(charges)?;
            self.used = next;
            Ok(())
        })();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn scalar_ok(value: &BigRational, maximum: u64) -> Result<()> {
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(RationalMellinError::InvalidSource("invalid rational"));
    }
    if bits(value) > maximum {
        return Err(RationalMellinError::ResourceLimit("coefficient height"));
    }
    Ok(())
}

struct Work<'a, M> {
    meter: &'a mut M,
}

impl<M: BudgetMeter> Work<'_, M> {
    fn charge(&mut self, steps: usize, bytes: usize) -> Result<()> {
        self.meter.checkpoint()?;
        self.meter.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }

    fn number(&mut self, value: &BigRational) -> Result<BigRational> {
        scalar_ok(value, MAX_BITS)?;
        self.charge(1, 96 + 2 * bits(value).div_ceil(8) as usize)?;
        Ok(value.clone())
    }

    fn sub(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        let value = left.metered_sub(right, self.meter)?;
        scalar_ok(&value, MAX_BITS)?;
        Ok(value)
    }

    fn mul(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        let value = left.metered_mul(right, self.meter)?;
        scalar_ok(&value, MAX_BITS)?;
        Ok(value)
    }

    fn div(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        if right.is_zero() {
            return Err(RationalMellinError::Verification("zero divisor"));
        }
        let value = left.metered_div(right, self.meter)?;
        scalar_ok(&value, MAX_BITS)?;
        Ok(value)
    }
}

fn validate_inputs(
    variable: &Symbol,
    argument: &Symbol,
    strip: &MellinStrip,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let names = variable.name.len().saturating_add(argument.name.len());
    if names > MAX_TEXT {
        return Err(RationalMellinError::ResourceLimit("typed symbol names"));
    }
    work.charge(2, names + 256)?;
    if variable == argument {
        return Err(RationalMellinError::InvalidSource(
            "transform and argument variables must be distinct",
        ));
    }
    for bound in [&strip.lower, &strip.upper] {
        if let MellinStripBound::Finite(value) = bound {
            scalar_ok(value, MAX_INPUT_BITS)?;
        }
    }
    let ordered = match (&strip.lower, &strip.upper) {
        (MellinStripBound::PositiveInfinity, _) | (_, MellinStripBound::NegativeInfinity) => false,
        (MellinStripBound::Finite(left), MellinStripBound::Finite(right)) => {
            left.metered_cmp(right, work.meter)?.is_lt()
        }
        _ => true,
    };
    if !ordered {
        return Err(RationalMellinError::InvalidStrip(
            "a nonempty open strip with ordered bounds is required",
        ));
    }
    work.meter.checkpoint()?;
    Ok(())
}

/// Compare rate +/- sqrt(magnitude) to a strip boundary without constructing
/// or approximating an algebraic root. None represents the rational rate.
fn root_cmp(
    rate: &BigRational,
    radical: Option<(&BigRational, bool)>,
    bound: &MellinStripBound,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Ordering> {
    let value = match bound {
        MellinStripBound::NegativeInfinity => return Ok(Ordering::Greater),
        MellinStripBound::PositiveInfinity => return Ok(Ordering::Less),
        MellinStripBound::Finite(value) => value,
    };
    let Some((magnitude, positive)) = radical else {
        return Ok(rate.metered_cmp(value, work.meter)?);
    };
    if !magnitude.is_positive() {
        return Err(RationalMellinError::Verification("nonpositive radical"));
    }
    let delta = work.sub(value, rate)?;
    if positive && delta.is_negative() {
        return Ok(Ordering::Greater);
    }
    if !positive && !delta.is_negative() {
        return Ok(Ordering::Less);
    }
    let square = work.mul(&delta, &delta)?;
    let ordering = magnitude.metered_cmp(&square, work.meter)?;
    Ok(if positive {
        ordering
    } else {
        ordering.reverse()
    })
}

fn root_side(
    rate: &BigRational,
    radical: Option<(&BigRational, bool)>,
    strip: &MellinStrip,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<MellinSupport> {
    if !root_cmp(rate, radical, &strip.lower, work)?.is_gt() {
        return Ok(MellinSupport::Left);
    }
    if !root_cmp(rate, radical, &strip.upper, work)?.is_lt() {
        return Ok(MellinSupport::Right);
    }
    Err(RationalMellinError::InvalidStrip(
        "a pole or original rational guard lies inside the open strip",
    ))
}

fn spectral_side(
    rate: &BigRational,
    squared_frequency: Option<&BigRational>,
    strip: &MellinStrip,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<MellinSupport> {
    if let Some(d) = squared_frequency.filter(|d| d.is_negative()) {
        let magnitude = work.sub(&BigRational::zero(), d)?;
        let smaller = root_side(rate, Some((&magnitude, false)), strip, work)?;
        let larger = root_side(rate, Some((&magnitude, true)), strip, work)?;
        return match (smaller, larger) {
            (MellinSupport::Left, MellinSupport::Left) => Ok(MellinSupport::Left),
            (MellinSupport::Right, MellinSupport::Right) => Ok(MellinSupport::Right),
            (MellinSupport::Left, MellinSupport::Right) => Ok(MellinSupport::Split),
            _ => Err(RationalMellinError::Verification(
                "unordered quadratic roots",
            )),
        };
    }
    // Complex conjugate poles share their exact rational real part.
    root_side(rate, None, strip, work)
}

fn block_side(
    block: &LaplaceBlock,
    strip: &MellinStrip,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<MellinSupport> {
    match block {
        LaplaceBlock::Exponential { rate, .. } => spectral_side(rate, None, strip, work),
        LaplaceBlock::Quadratic {
            rate,
            squared_frequency,
            ..
        } => spectral_side(rate, Some(squared_frequency), strip, work),
    }
}

fn check_original_guards(
    source: &Expr,
    variable: &Symbol,
    argument: &Symbol,
    strip: &MellinStrip,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    let guards = original_rational_guard_polynomials(source, variable, argument, work.meter)?;
    for guard in &guards {
        let complete = metered_complete_factorization(guard, work.meter)?;
        for factor in &complete.factors {
            let degree = factor.poly.degree().unwrap_or(0);
            if degree == 0 || degree > 2 {
                return Err(RationalMellinError::Unsupported(
                    "original guard has a factor of degree greater than two",
                ));
            }
            let negative_b = work.sub(&BigRational::zero(), &factor.poly.coeffs[degree - 1])?;
            if degree == 1 {
                spectral_side(&negative_b, None, strip, work)?;
            } else {
                let rate = work.div(&negative_b, &BigRational::from(2i64))?;
                let square = work.mul(&rate, &rate)?;
                let d = work.sub(&factor.poly.coeffs[0], &square)?;
                spectral_side(&rate, Some(&d), strip, work)?;
            }
        }
    }
    work.meter.checkpoint()?;
    Ok(guards.len())
}

fn sum(terms: Vec<Expr>) -> Expr {
    let mut terms: Vec<_> = terms.into_iter().filter(|term| !term.is_zero()).collect();
    match terms.len() {
        0 => Expr::from_i64(0),
        1 => terms.pop().expect("one term"),
        _ => Expr::Add(terms),
    }
}

fn product(factors: Vec<Expr>) -> Expr {
    if factors.iter().any(Expr::is_zero) {
        return Expr::from_i64(0);
    }
    let mut factors: Vec<_> = factors
        .into_iter()
        .filter(|factor| !factor.is_one())
        .collect();
    match factors.len() {
        0 => Expr::from_i64(1),
        1 => factors.pop().expect("one factor"),
        _ => Expr::Mul(factors),
    }
}

fn power(base: Expr, exponent: Expr) -> Expr {
    if exponent.is_zero() {
        Expr::from_i64(1)
    } else if exponent.is_one() {
        base
    } else {
        Expr::Pow(Arc::new(base), Arc::new(exponent))
    }
}

fn polynomial(coefficients: &[BigRational], coordinate: &Expr) -> Expr {
    sum(coefficients
        .iter()
        .enumerate()
        .filter_map(|(degree, coefficient)| {
            if coefficient.is_zero() {
                None
            } else {
                Some(product(vec![
                    rational_expr(coefficient.clone()),
                    power(coordinate.clone(), Expr::from_i64(degree as i64)),
                ]))
            }
        })
        .collect())
}

fn frequency(
    magnitude: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(Expr, Expr)> {
    if !magnitude.is_positive() {
        return Err(RationalMellinError::Verification("nonpositive frequency"));
    }
    let numerator = metered_sqrt_floor(magnitude.numer(), work.meter)?.ok_or(
        RationalMellinError::Verification("negative frequency numerator"),
    )?;
    let denominator = metered_sqrt_floor(magnitude.denom(), work.meter)?.ok_or(
        RationalMellinError::Verification("negative frequency denominator"),
    )?;
    let numerator_square = metered_mul(&numerator, &numerator, work.meter)?;
    let denominator_square = metered_mul(&denominator, &denominator, work.meter)?;
    work.charge(8, 512 + 8 * bits(magnitude).div_ceil(8) as usize)?;
    if numerator_square == *magnitude.numer() && denominator_square == *magnitude.denom() {
        let value = BigRational::metered_new(&numerator, &denominator, work.meter)?;
        let inverse = work.div(&BigRational::one(), &value)?;
        return Ok((rational_expr(value), rational_expr(inverse)));
    }
    let value = power(
        rational_expr(work.number(magnitude)?),
        rational_expr(BigRational::new(1.into(), 2.into())),
    );
    let inverse = power(value.clone(), Expr::from_i64(-1));
    work.meter.checkpoint()?;
    Ok((value, inverse))
}

fn attach_support(value: Expr, argument: &Expr, side: MellinSupport) -> Result<Expr> {
    let (sign, guard) = match side {
        MellinSupport::Left => (
            1,
            sum(vec![
                Expr::from_i64(1),
                product(vec![Expr::from_i64(-1), argument.clone()]),
            ]),
        ),
        MellinSupport::Right => (-1, sum(vec![argument.clone(), Expr::from_i64(-1)])),
        MellinSupport::Split => {
            return Err(RationalMellinError::Verification("unsplit real quadratic"));
        }
    };
    Ok(product(vec![
        Expr::from_i64(sign),
        value,
        Expr::Function("Heaviside".into(), vec![guard]),
    ]))
}

fn render(
    certificate: &RationalMellinCertificate,
    argument: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    let mut cells = 0usize;
    let mut maximum_bits = 1u64;
    for block in &certificate.laplace.blocks {
        match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                cells += coefficients.len();
                maximum_bits = maximum_bits.max(bits(rate));
                for value in coefficients {
                    maximum_bits = maximum_bits.max(bits(value));
                }
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                cells += cosine.len() + sine.len();
                maximum_bits = maximum_bits.max(bits(rate)).max(bits(squared_frequency));
                for value in cosine.iter().chain(sine) {
                    maximum_bits = maximum_bits.max(bits(value));
                }
            }
        }
    }
    let nodes = 1 + 128 * cells + 32 * certificate.laplace.blocks.len();
    if nodes > MAX_OUTPUT_NODES {
        return Err(RationalMellinError::ResourceLimit(
            "output expression nodes",
        ));
    }
    let bytes = nodes
        .checked_mul(256 + 4 * maximum_bits.div_ceil(8) as usize + 4 * argument.name.len())
        .ok_or(RationalMellinError::ResourceLimit("output allocation"))?;
    work.charge(nodes * nodes, bytes)?;
    let x = Expr::Sym(argument.clone());
    let coordinate = product(vec![
        Expr::from_i64(-1),
        Expr::Function("log".into(), vec![x.clone()]),
    ]);
    let mut terms = Vec::with_capacity(certificate.laplace.blocks.len() * 2);
    for (block, &side) in certificate.laplace.blocks.iter().zip(&certificate.sides) {
        work.meter.checkpoint()?;
        let (rate, inner) = match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                (rate, polynomial(coefficients, &coordinate))
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency: d,
                cosine,
                sine,
            } => {
                let magnitude = if d.is_positive() {
                    work.number(d)?
                } else {
                    work.sub(&BigRational::zero(), d)?
                };
                let (frequency, inverse) = frequency(&magnitude, work)?;
                let cosine = polynomial(cosine, &coordinate);
                let sine = product(vec![polynomial(sine, &coordinate), inverse]);
                if side == MellinSupport::Split {
                    if !d.is_negative() {
                        return Err(RationalMellinError::Verification("split nonreal poles"));
                    }
                    let negative_rate = rational_expr(work.sub(&BigRational::zero(), rate)?);
                    let smaller = product(vec![
                        rational_expr(BigRational::new(1.into(), 2.into())),
                        sum(vec![
                            cosine.clone(),
                            product(vec![Expr::from_i64(-1), sine.clone()]),
                        ]),
                        power(
                            x.clone(),
                            sum(vec![negative_rate.clone(), frequency.clone()]),
                        ),
                    ]);
                    let larger = product(vec![
                        rational_expr(BigRational::new(1.into(), 2.into())),
                        sum(vec![cosine, sine]),
                        power(
                            x.clone(),
                            sum(vec![
                                negative_rate,
                                product(vec![Expr::from_i64(-1), frequency]),
                            ]),
                        ),
                    ]);
                    terms.push(attach_support(smaller, &x, MellinSupport::Left)?);
                    terms.push(attach_support(larger, &x, MellinSupport::Right)?);
                    continue;
                }
                let phase = product(vec![frequency, coordinate.clone()]);
                let inner = sum(vec![
                    product(vec![
                        cosine,
                        Expr::Function(
                            if d.is_positive() { "cos" } else { "cosh" }.into(),
                            vec![phase.clone()],
                        ),
                    ]),
                    product(vec![
                        sine,
                        Expr::Function(
                            if d.is_positive() { "sin" } else { "sinh" }.into(),
                            vec![phase],
                        ),
                    ]),
                ]);
                (rate, inner)
            }
        };
        let negative_rate = work.sub(&BigRational::zero(), rate)?;
        let value = product(vec![power(x.clone(), rational_expr(negative_rate)), inner]);
        terms.push(attach_support(value, &x, side)?);
    }
    work.meter.checkpoint()?;
    Ok(sum(terms))
}

fn verify_inner(
    source: &Expr,
    variable: &Symbol,
    argument: &Symbol,
    strip: &MellinStrip,
    certificate: &RationalMellinCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedRationalMellin> {
    validate_inputs(variable, argument, strip, work)?;
    if certificate.sides.len() > MAX_DEGREE
        || certificate.sides.len() != certificate.laplace.blocks.len()
    {
        return Err(RationalMellinError::Verification(
            "incomplete strip assignments",
        ));
    }
    let original_guard_count = check_original_guards(source, variable, argument, strip, work)?;
    // This independent closed-form coefficient checker reconstructs the
    // entire original rational transform; the Mellin generator is not replayed.
    let accepted = verify_rational_laplace_certificate(
        source,
        variable,
        argument,
        &certificate.laplace,
        work.meter,
    )?;
    for (block, &side) in accepted.blocks().iter().zip(&certificate.sides) {
        if block_side(block, strip, work)? != side {
            return Err(RationalMellinError::Verification(
                "incorrect strip assignment",
            ));
        }
    }
    let function = render(certificate, argument, work)?;
    // Source/certificate/typed-owner copies precede immutable publication.
    // Existing Laplace admission bounds every source node and coefficient.
    work.charge(4096 + MAX_DEGREE, 4096 * 2048 + 4 * MAX_TEXT + 96 * 16_512)?;
    let receipt = VerifiedRationalMellin {
        source: source.clone(),
        transform_variable: variable.clone(),
        argument_variable: argument.clone(),
        strip: strip.clone(),
        certificate: certificate.clone(),
        function,
        original_guard_count,
    };
    work.meter.checkpoint()?;
    Ok(receipt)
}

pub fn verify_rational_mellin_certificate(
    source: &Expr,
    transform_variable: &Symbol,
    argument_variable: &Symbol,
    strip: &MellinStrip,
    certificate: &RationalMellinCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalMellin> {
    let mut capped = Capped {
        inner: verifier,
        used: [0; 3],
        stopped: None,
    };
    let result = verify_inner(
        source,
        transform_variable,
        argument_variable,
        strip,
        certificate,
        &mut Work { meter: &mut capped },
    );
    if let Some(error) = capped.stopped {
        return Err(RationalMellinError::Meter(error));
    }
    result
}

pub fn inverse_rational_mellin(
    source: &Expr,
    transform_variable: &Symbol,
    argument_variable: &Symbol,
    strip: &MellinStrip,
) -> Result<VerifiedRationalMellin> {
    metered_inverse_rational_mellin(
        source,
        transform_variable,
        argument_variable,
        strip,
        &mut Unbounded,
        &mut Unbounded,
    )
}

pub fn metered_inverse_rational_mellin(
    source: &Expr,
    transform_variable: &Symbol,
    argument_variable: &Symbol,
    strip: &MellinStrip,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalMellin> {
    let mut generation = Capped {
        inner: generator,
        used: [0; 3],
        stopped: None,
    };
    let mut verification = Capped {
        inner: verifier,
        used: [0; 3],
        stopped: None,
    };
    let result = (|| {
        validate_inputs(
            transform_variable,
            argument_variable,
            strip,
            &mut Work {
                meter: &mut generation,
            },
        )?;
        let laplace = generate_rational_laplace_certificate(
            source,
            transform_variable,
            argument_variable,
            &mut generation,
        )?;
        let mut work = Work {
            meter: &mut generation,
        };
        work.charge(laplace.blocks.len(), laplace.blocks.len() * 8)?;
        let mut sides = Vec::with_capacity(laplace.blocks.len());
        for block in &laplace.blocks {
            sides.push(block_side(block, strip, &mut work)?);
        }
        let certificate = RationalMellinCertificate { laplace, sides };
        generation.checkpoint()?;
        let receipt = verify_inner(
            source,
            transform_variable,
            argument_variable,
            strip,
            &certificate,
            &mut Work {
                meter: &mut verification,
            },
        )?;
        generation.checkpoint()?;
        verification.checkpoint()?;
        Ok(receipt)
    })();
    if let Some(error) = generation.stopped.or(verification.stopped) {
        return Err(RationalMellinError::Meter(error));
    }
    result
}
