//! Exact rational Laurent expansions, leading terms, and directional limits.
//!
//! The generator computes local numerator/denominator coefficients by
//! synthetic division. An independent proof-kernel checker reconstructs both
//! original polynomials before exposing an asymptotic/limit receipt. Finite
//! rational centers and both ends of the real line are supported. The claim
//! is local on the original rational expression's punctured real domain;
//! neither a value at a pole nor convergence of a general series is asserted.
//! A bounded denominator recurrence generates formal coefficients; the
//! proof-kernel checks their truncated convolution and any exactness claim.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_limit::{
    LocalDirection, MAX_LOCAL_COEFFICIENT_BITS, MAX_LOCAL_COEFFICIENTS, MAX_SERIES_COEFFICIENTS,
    RationalDirectionalLimit, RationalLimitPoint, RationalLocalCertificate,
    RationalLocalVerification, RationalLocalVerificationError, RationalSeriesCertificate,
    RationalSeriesVerification, verify_rational_local_order, verify_rational_series,
};
use num_traits::{One, Zero};
use thiserror::Error;

const MAX_INPUT_BITS: u64 = 4_096;
const MAX_INPUT_NODES: usize = 4_096;
const MAX_INPUT_DEPTH: usize = 64;
const MAX_INPUT_FANOUT: usize = 128;
const MAX_INPUT_TEXT: usize = 16_384;
const MAX_WORK: u64 = 1_000_000;

#[derive(Debug, Error)]
pub enum RationalLimitError {
    #[error("rational local analysis refused: {0}")]
    Refused(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
    #[error(transparent)]
    Verification(#[from] RationalLocalVerificationError),
}

impl From<RationalArithmeticError> for RationalLimitError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalLimitError>;

/// Checked analysis and its transportable, untrusted witness. Retained
/// certificates must be independently checked again on reuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalLocalAnalysis {
    certificate: RationalLocalCertificate,
    verification: RationalLocalVerification,
}

impl RationalLocalAnalysis {
    pub fn certificate(&self) -> &RationalLocalCertificate {
        &self.certificate
    }

    pub fn verification(&self) -> &RationalLocalVerification {
        &self.verification
    }
}

/// A bounded formal Laurent expansion and independently checked truncation.
/// The witness itself remains untrusted when serialized or reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalSeriesAnalysis {
    certificate: RationalSeriesCertificate,
    verification: RationalSeriesVerification,
}

impl RationalSeriesAnalysis {
    pub fn certificate(&self) -> &RationalSeriesCertificate {
        &self.certificate
    }

    pub fn verification(&self) -> &RationalSeriesVerification {
        &self.verification
    }
}

fn refused(message: &'static str) -> RationalLimitError {
    RationalLimitError::Refused(message)
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate(poly: &[BigRational], bit_limit: u64) -> Result<()> {
    if poly.is_empty() || poly.len() > MAX_LOCAL_COEFFICIENTS {
        return Err(refused("polynomial degree limit"));
    }
    if poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        return Err(refused("noncanonical polynomial coefficients"));
    }
    if poly
        .iter()
        .any(|c| bits(c) > bit_limit || c.denom().is_zero() || c.denom().is_negative())
    {
        return Err(refused("invalid rational or coefficient height limit"));
    }
    Ok(())
}

fn trim(poly: &mut Vec<BigRational>) {
    while poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        poly.pop();
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    remaining: u64,
}

impl<M: BudgetMeter> Work<'_, M> {
    fn step(&mut self, count: u64) -> Result<()> {
        self.meter.checkpoint()?;
        self.remaining = self
            .remaining
            .checked_sub(count)
            .ok_or_else(|| refused("rational arithmetic work limit"))?;
        self.meter.charge(Dimension::ComputeSteps, count)?;
        Ok(())
    }

    fn reserve(&mut self, count: usize) -> Result<()> {
        if count > MAX_LOCAL_COEFFICIENTS {
            return Err(refused("intermediate polynomial degree limit"));
        }
        self.step(count as u64)?;
        self.meter.charge_batch(&[
            (Dimension::AllocationCount, (2 * count + 1) as u64),
            (
                Dimension::MemoryBytes,
                count as u64 * (64 + 2 * MAX_LOCAL_COEFFICIENT_BITS.div_ceil(8)),
            ),
        ])?;
        Ok(())
    }

    fn copy(&mut self, coefficients: &[BigRational]) -> Result<Vec<BigRational>> {
        self.reserve(coefficients.len())?;
        Ok(coefficients.to_vec())
    }

    fn reserve_series(&mut self, count: usize) -> Result<()> {
        if count > MAX_SERIES_COEFFICIENTS {
            return Err(refused("Laurent output coefficient count limit"));
        }
        self.step(count.max(1) as u64)?;
        self.meter.charge_batch(&[
            (Dimension::AllocationCount, (2 * count + 1) as u64),
            (
                Dimension::MemoryBytes,
                count.max(1) as u64 * (64 + 2 * MAX_LOCAL_COEFFICIENT_BITS.div_ceil(8)),
            ),
        ])?;
        Ok(())
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.step(1)?;
        let result = a.metered_add(b, self.meter)?;
        if bits(&result) > MAX_LOCAL_COEFFICIENT_BITS {
            return Err(refused("intermediate coefficient height limit"));
        }
        Ok(result)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.step(1)?;
        let result = a.metered_mul(b, self.meter)?;
        if bits(&result) > MAX_LOCAL_COEFFICIENT_BITS {
            return Err(refused("intermediate coefficient height limit"));
        }
        Ok(result)
    }

    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.step(1)?;
        let result = a.metered_sub(b, self.meter)?;
        if bits(&result) > MAX_LOCAL_COEFFICIENT_BITS {
            return Err(refused("intermediate coefficient height limit"));
        }
        Ok(result)
    }

    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.step(1)?;
        let result = a.metered_div(b, self.meter)?;
        if bits(&result) > MAX_LOCAL_COEFFICIENT_BITS {
            return Err(refused("intermediate coefficient height limit"));
        }
        Ok(result)
    }

    fn series_is_exact(
        &mut self,
        numerator: &[BigRational],
        denominator: &[BigRational],
        coefficients: &[BigRational],
    ) -> Result<bool> {
        let product_len = if coefficients.is_empty() {
            0
        } else {
            denominator.len() + coefficients.len() - 1
        };
        for degree in 0..numerator.len().max(product_len) {
            let mut value = BigRational::zero();
            for (j, vj) in denominator.iter().enumerate().take(degree + 1) {
                if let Some(coefficient) = coefficients.get(degree - j) {
                    let product = self.mul(vj, coefficient)?;
                    value = self.add(&value, &product)?;
                }
            }
            if value
                != numerator
                    .get(degree)
                    .cloned()
                    .unwrap_or_else(BigRational::zero)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn polynomial_sum(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Vec<BigRational>> {
        let count = a.len().max(b.len());
        self.reserve(count)?;
        let mut result = vec![BigRational::zero(); count];
        for (index, coefficient) in result.iter_mut().enumerate() {
            if let Some(value) = a.get(index) {
                *coefficient = self.add(coefficient, value)?;
            }
            if let Some(value) = b.get(index) {
                *coefficient = self.add(coefficient, value)?;
            }
        }
        trim(&mut result);
        Ok(result)
    }

    fn polynomial_product(
        &mut self,
        a: &[BigRational],
        b: &[BigRational],
    ) -> Result<Vec<BigRational>> {
        let count = a.len() + b.len() - 1;
        self.reserve(count)?;
        let mut result = vec![BigRational::zero(); count];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                let product = self.mul(ai, bj)?;
                result[i + j] = self.add(&result[i + j], &product)?;
            }
        }
        trim(&mut result);
        Ok(result)
    }

    /// Repeated synthetic division by x-a yields the Taylor coefficients
    /// without derivative expression trees or factorial-sized integers.
    fn local_coefficients(
        &mut self,
        original: &[BigRational],
        point: &RationalLimitPoint,
        common_degree: usize,
    ) -> Result<Vec<BigRational>> {
        if let RationalLimitPoint::Finite(center) = point {
            let mut dividend = self.copy(original)?;
            self.reserve(original.len())?;
            let mut local = Vec::with_capacity(original.len());
            while dividend.len() > 1 {
                let degree = dividend.len() - 1;
                self.reserve(degree)?;
                let mut quotient = vec![BigRational::zero(); degree];
                quotient[degree - 1] = dividend[degree].clone();
                for index in (0..degree - 1).rev() {
                    let scaled = self.mul(center, &quotient[index + 1])?;
                    quotient[index] = self.add(&dividend[index + 1], &scaled)?;
                }
                let scaled = self.mul(center, &quotient[0])?;
                local.push(self.add(&dividend[0], &scaled)?);
                dividend = quotient;
            }
            local.push(dividend[0].clone());
            trim(&mut local);
            Ok(local)
        } else {
            self.reserve(common_degree + 1)?;
            let mut local = vec![BigRational::zero(); common_degree + 1 - original.len()];
            local.extend(original.iter().rev().cloned());
            trim(&mut local);
            Ok(local)
        }
    }
}

/// Convenience entry point with explicit unbounded external meters. Hard
/// degree/height/work limits still apply to each phase.
pub fn rational_local_order(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    point: &RationalLimitPoint,
) -> Result<RationalLocalAnalysis> {
    metered_rational_local_order(
        numerator,
        denominator,
        point,
        &mut Unbounded,
        &mut Unbounded,
    )
}

/// Generate and independently check local order. Every scalar arithmetic
/// operation is metered; safe points occur before and after scalar operations
/// and before publication. Generator and verifier reservations stay separate.
pub fn metered_rational_local_order(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    point: &RationalLimitPoint,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<RationalLocalAnalysis> {
    generator_meter.checkpoint()?;
    if numerator.gen_sym != denominator.gen_sym {
        return Err(refused("different polynomial generators"));
    }
    validate(&numerator.coeffs, MAX_INPUT_BITS)?;
    validate(&denominator.coeffs, MAX_INPUT_BITS)?;
    if denominator.is_zero() {
        return Err(refused("identically zero denominator"));
    }
    if let RationalLimitPoint::Finite(center) = point {
        validate(std::slice::from_ref(center), MAX_INPUT_BITS)?;
    }
    let common_degree = numerator.coeffs.len().max(denominator.coeffs.len()) - 1;
    let mut work = Work {
        meter: generator_meter,
        remaining: MAX_WORK,
    };
    let local_numerator = work.local_coefficients(&numerator.coeffs, point, common_degree)?;
    let local_denominator = work.local_coefficients(&denominator.coeffs, point, common_degree)?;
    let numerator_order = local_numerator.iter().position(|value| !value.is_zero());
    let denominator_order = local_denominator
        .iter()
        .position(|value| !value.is_zero())
        .ok_or_else(|| refused("transformed denominator vanished"))?;
    let certificate = RationalLocalCertificate {
        numerator_order,
        denominator_order,
        numerator_unit: work.copy(&local_numerator[numerator_order.unwrap_or(0)..])?,
        denominator_unit: work.copy(&local_denominator[denominator_order..])?,
    };
    let verification = verify_rational_local_order(
        &numerator.coeffs,
        &denominator.coeffs,
        point,
        &certificate,
        verifier_meter,
    )?;
    work.meter.checkpoint()?;
    Ok(RationalLocalAnalysis {
        certificate,
        verification,
    })
}

/// Exact formal coefficients strictly below `absolute_order`, in `t=x-a`
/// at a rational center or `t=1/x` at infinity. At most 256 coefficients
/// beyond the local valuation are requested. No general convergence or
/// pointwise value at an excluded center is claimed.
pub fn rational_series(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    point: &RationalLimitPoint,
    absolute_order: i64,
) -> Result<RationalSeriesAnalysis> {
    metered_rational_series(
        numerator,
        denominator,
        point,
        absolute_order,
        &mut Unbounded,
        &mut Unbounded,
    )
}

/// Metered coefficient recurrence with a separate verifier meter. Local
/// factorization and truncation checks charge the same supplied verifier
/// meter, without replenishing either phase's accounting.
pub fn metered_rational_series(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    point: &RationalLimitPoint,
    absolute_order: i64,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<RationalSeriesAnalysis> {
    let local = metered_rational_local_order(
        numerator,
        denominator,
        point,
        generator_meter,
        verifier_meter,
    )?;
    let mut work = Work {
        meter: generator_meter,
        remaining: MAX_WORK,
    };
    let mut coefficients = Vec::new();
    let mut exact = true;
    if let Some(start_order) = local.verification().order() {
        let required = if absolute_order <= start_order {
            0
        } else {
            absolute_order
                .checked_sub(start_order)
                .and_then(|count| usize::try_from(count).ok())
                .filter(|&count| count <= MAX_SERIES_COEFFICIENTS)
                .ok_or_else(|| {
                    refused("requested Laurent precision exceeds the coefficient bound")
                })?
        };
        work.reserve_series(required)?;
        coefficients.reserve_exact(required);
        let u = &local.certificate.numerator_unit;
        let v = &local.certificate.denominator_unit;
        for degree in 0..required {
            let mut residual = u.get(degree).cloned().unwrap_or_else(BigRational::zero);
            for (j, vj) in v.iter().enumerate().skip(1).take(degree) {
                let product = work.mul(vj, &coefficients[degree - j])?;
                residual = work.sub(&residual, &product)?;
            }
            coefficients.push(work.div(&residual, &v[0])?);
        }
        exact = work.series_is_exact(u, v, &coefficients)?;
        if exact {
            while coefficients.last().is_some_and(Zero::is_zero) {
                coefficients.pop();
            }
        }
    }
    let certificate = RationalSeriesCertificate {
        local: local.certificate,
        coefficients,
        exact,
    };
    let verification = verify_rational_series(
        &numerator.coeffs,
        &denominator.coeffs,
        point,
        absolute_order,
        &certificate,
        verifier_meter,
    )?;
    work.meter.checkpoint()?;
    Ok(RationalSeriesAnalysis {
        certificate,
        verification,
    })
}

fn integer_power(exponent: &Expr) -> Option<i64> {
    match exponent {
        Expr::Integer(value) => value.to_i64(),
        Expr::Rational(value) if value.is_integer() => value.numer().to_i64(),
        _ => None,
    }
}

/// Validate the whole original syntax before rational arithmetic. An
/// unsupported function/parameter returns false; an oversized rational input
/// returns an error and must never re-enter an unbounded fallback.
fn rational_shape(expression: &Expr, variable: &Symbol) -> Result<bool> {
    let mut stack = vec![(expression, 0usize)];
    let mut nodes = 0usize;
    let mut text = variable.name.len();
    let mut rational = true;
    let mut rational_refusal = None;
    while let Some((expr, depth)) = stack.pop() {
        nodes += 1;
        if nodes > MAX_INPUT_NODES || depth > MAX_INPUT_DEPTH || text > MAX_INPUT_TEXT {
            return Err(refused("expression traversal limit"));
        }
        match expr {
            Expr::Integer(value) => {
                if value.bits() > MAX_INPUT_BITS {
                    rational_refusal = Some(refused("input integer height limit"));
                }
            }
            Expr::Rational(value) => {
                if let Err(error) = validate(std::slice::from_ref(value), MAX_INPUT_BITS) {
                    rational_refusal = Some(error);
                }
            }
            Expr::Sym(symbol) => {
                text = text.saturating_add(symbol.name.len());
                rational &= symbol == variable;
            }
            Expr::Const(_) => rational = false,
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = expr {
                    text = text.saturating_add(name.len());
                    rational = false;
                } else if children.is_empty() {
                    return Err(refused("empty arithmetic node"));
                }
                if children.len() > MAX_INPUT_FANOUT
                    || stack.len() + children.len() > MAX_INPUT_NODES
                {
                    return Err(refused("expression fanout limit"));
                }
                stack.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if matches!(exponent.as_ref(), Expr::Integer(_))
                    || matches!(exponent.as_ref(), Expr::Rational(value) if value.is_integer())
                {
                    if integer_power(exponent).is_none_or(|power| power.unsigned_abs() > 128) {
                        rational_refusal = Some(refused("literal exponent limit"));
                    }
                } else {
                    rational = false;
                }
                if stack.len() + 2 > MAX_INPUT_NODES {
                    return Err(refused("expression traversal limit"));
                }
                stack.push((base, depth + 1));
                stack.push((exponent, depth + 1));
            }
        }
    }
    if text > MAX_INPUT_TEXT {
        return Err(refused("expression text limit"));
    }
    // Degree and coefficient limits belong to this rational lane. An
    // otherwise structurally bounded expression such as exp(x^129) must
    // remain eligible for the existing nonrational algorithms.
    if rational && let Some(error) = rational_refusal {
        return Err(error);
    }
    Ok(rational)
}

struct Fraction {
    numerator: Vec<BigRational>,
    denominator: Vec<BigRational>,
}

impl Fraction {
    fn one() -> Self {
        Self {
            numerator: vec![BigRational::one()],
            denominator: vec![BigRational::one()],
        }
    }

    fn multiply(self, other: Self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        Ok(Self {
            numerator: work.polynomial_product(&self.numerator, &other.numerator)?,
            denominator: work.polynomial_product(&self.denominator, &other.denominator)?,
        })
    }

    fn copy(&self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        Ok(Self {
            numerator: work.copy(&self.numerator)?,
            denominator: work.copy(&self.denominator)?,
        })
    }
}

fn parse_fraction(expr: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Fraction> {
    work.step(1)?;
    work.reserve(2)?;
    let constant = |value| Fraction {
        numerator: vec![value],
        denominator: vec![BigRational::one()],
    };
    match expr {
        Expr::Integer(value) => Ok(constant(BigRational::from_integer(value.clone()))),
        Expr::Rational(value) => Ok(constant(value.clone())),
        Expr::Sym(_) => Ok(Fraction {
            numerator: vec![BigRational::zero(), BigRational::one()],
            denominator: vec![BigRational::one()],
        }),
        Expr::Add(children) => {
            let mut result = constant(BigRational::zero());
            for child in children {
                let other = parse_fraction(child, work)?;
                let left = work.polynomial_product(&result.numerator, &other.denominator)?;
                let right = work.polynomial_product(&other.numerator, &result.denominator)?;
                result = Fraction {
                    numerator: work.polynomial_sum(&left, &right)?,
                    denominator: work
                        .polynomial_product(&result.denominator, &other.denominator)?,
                };
            }
            Ok(result)
        }
        Expr::Mul(children) => {
            let mut result = Fraction::one();
            for child in children {
                // Do not short-circuit a zero factor: every original child
                // must have a nonempty punctured rational domain.
                result = result.multiply(parse_fraction(child, work)?, work)?;
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            // Check the original base even for exponent zero, before any
            // cancellation can erase a literal or polynomial zero divisor.
            let mut base = parse_fraction(base, work)?;
            let exponent = integer_power(exponent).ok_or_else(|| refused("noninteger power"))?;
            if base.numerator == [BigRational::zero()] && exponent <= 0 {
                return Err(refused("zero base of a nonpositive power"));
            }
            if exponent < 0 {
                std::mem::swap(&mut base.numerator, &mut base.denominator);
            }
            let mut count = exponent.unsigned_abs();
            let mut result = Fraction::one();
            while count != 0 {
                if count & 1 == 1 {
                    result = result.multiply(base.copy(work)?, work)?;
                }
                count >>= 1;
                if count != 0 {
                    base = base.copy(work)?.multiply(base, work)?;
                }
            }
            Ok(result)
        }
        _ => Err(refused("expression outside rational grammar")),
    }
}

/// `None` means the expression is outside Q(x), so another algorithm may be
/// applicable. A recognized but invalid or exhausted rational input returns
/// an error; callers must not turn that error into heuristic acceptance.
fn rational_expression_parts(
    expression: &Expr,
    variable: &Symbol,
) -> Result<Option<(UnivariatePoly, UnivariatePoly)>> {
    if !rational_shape(expression, variable)? {
        return Ok(None);
    }
    let mut meter = Unbounded;
    let mut work = Work {
        meter: &mut meter,
        remaining: MAX_WORK,
    };
    let fraction = parse_fraction(expression, &mut work)?;
    let numerator = UnivariatePoly::new(variable.clone(), fraction.numerator);
    let denominator = UnivariatePoly::new(variable.clone(), fraction.denominator);
    Ok(Some((numerator, denominator)))
}

pub fn rational_expression_limit(
    expression: &Expr,
    variable: &Symbol,
    point: &RationalLimitPoint,
    direction: LocalDirection,
) -> Result<Option<RationalDirectionalLimit>> {
    let Some((numerator, denominator)) = rational_expression_parts(expression, variable)? else {
        return Ok(None);
    };
    let analysis = rational_local_order(&numerator, &denominator, point)?;
    Ok(Some(analysis.verification().limit(direction)))
}

/// Expand an original rational expression without evaluating away excluded
/// subexpressions. `None` means another, nonrational algorithm may apply;
/// invalid/exhausted rational cases remain errors and cannot fall through.
pub fn rational_expression_series(
    expression: &Expr,
    variable: &Symbol,
    point: &RationalLimitPoint,
    absolute_order: i64,
) -> Result<Option<RationalSeriesAnalysis>> {
    let Some((numerator, denominator)) = rational_expression_parts(expression, variable)? else {
        return Ok(None);
    };
    rational_series(&numerator, &denominator, point, absolute_order).map(Some)
}

fn point_value(
    expression: &Expr,
    center: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    work.step(1)?;
    match expression {
        Expr::Integer(value) => Ok(BigRational::from_integer(value.clone())),
        Expr::Rational(value) => Ok(value.clone()),
        Expr::Sym(_) => Ok(center.clone()),
        Expr::Add(children) => {
            let mut value = BigRational::zero();
            for child in children {
                let other = point_value(child, center, work)?;
                value = work.add(&value, &other)?;
            }
            Ok(value)
        }
        Expr::Mul(children) => {
            let mut value = BigRational::one();
            for child in children {
                let other = point_value(child, center, work)?;
                value = work.mul(&value, &other)?;
            }
            Ok(value)
        }
        Expr::Pow(base, exponent) => {
            let mut base = point_value(base, center, work)?;
            let exponent = integer_power(exponent).ok_or_else(|| refused("noninteger power"))?;
            if base.is_zero() && exponent <= 0 {
                return Err(refused(
                    "the original expression is undefined at the Taylor center",
                ));
            }
            if exponent < 0 {
                base = work.div(&BigRational::one(), &base)?;
            }
            let mut count = exponent.unsigned_abs();
            let mut value = BigRational::one();
            while count != 0 {
                if count & 1 == 1 {
                    value = work.mul(&value, &base)?;
                }
                count >>= 1;
                if count != 0 {
                    base = work.mul(&base, &base)?;
                }
            }
            Ok(value)
        }
        _ => Err(refused("expression outside rational grammar")),
    }
}

pub(crate) fn try_series(
    expression: &Expr,
    variable: &Symbol,
    at: &Expr,
    absolute_order: i64,
) -> Option<Result<crate::SeriesTerms>> {
    let point = match at {
        Expr::Integer(value) => {
            RationalLimitPoint::Finite(BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value) => RationalLimitPoint::Finite(value.clone()),
        Expr::Const(Constant::Infinity | Constant::NegativeInfinity) => {
            RationalLimitPoint::Infinity
        }
        _ => return None,
    };
    let analysis = match rational_expression_series(expression, variable, &point, absolute_order) {
        Ok(Some(analysis)) => analysis,
        Ok(None) => return None,
        Err(error) => return Some(Err(error)),
    };
    let receipt = analysis.verification();
    let negative_infinity = matches!(at, Expr::Const(Constant::NegativeInfinity));
    let terms = receipt
        .coefficients()
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| !coefficient.is_zero())
        .map(|(index, coefficient)| {
            let exponent = receipt.start_order() + index as i64;
            // Public series at -infinity uses -1/x, whereas the certificate
            // uniformly uses 1/x. Substitute t=-u with exact integer parity.
            let coefficient = if negative_infinity && exponent % 2 != 0 {
                -coefficient.clone()
            } else {
                coefficient.clone()
            };
            (
                fsym_core::elementary::rational_expr(coefficient),
                BigRational::from_integer(exponent.into()),
            )
        })
        .collect();
    let remainder = receipt
        .remainder_order()
        .map(|order| BigRational::from_integer(order.into()));
    Some(Ok((terms, remainder)))
}

pub(crate) fn try_taylor(
    expression: &Expr,
    variable: &Symbol,
    at: &Expr,
    degree: usize,
) -> Option<Result<Expr>> {
    let center = match at {
        Expr::Integer(value) => BigRational::from_integer(value.clone()),
        Expr::Rational(value) => value.clone(),
        _ => return None,
    };
    let (numerator, denominator) = match rational_expression_parts(expression, variable) {
        Ok(Some(parts)) => parts,
        Ok(None) => return None,
        Err(error) => return Some(Err(error)),
    };
    Some((|| {
        validate(std::slice::from_ref(&center), MAX_INPUT_BITS)?;
        let absolute_order = degree
            .checked_add(1)
            .and_then(|order| i64::try_from(order).ok())
            .ok_or_else(|| refused("Taylor order overflow"))?;
        let mut meter = Unbounded;
        let mut work = Work {
            meter: &mut meter,
            remaining: MAX_WORK,
        };
        // Formal Laurent expansions permit removable holes. A Taylor
        // polynomial at the point additionally requires every original
        // rational child to be defined there, before any cancellation.
        point_value(expression, &center, &mut work)?;
        let analysis = rational_series(
            &numerator,
            &denominator,
            &RationalLimitPoint::Finite(center.clone()),
            absolute_order,
        )?;
        let receipt = analysis.verification();
        if receipt.start_order() < 0 {
            return Err(refused("a Laurent pole is not a Taylor polynomial"));
        }
        let local = Expr::Sym(variable.clone()) + fsym_core::elementary::rational_expr(-center);
        let mut terms = Vec::new();
        for (index, coefficient) in receipt.coefficients().iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let exponent = receipt.start_order() + index as i64;
            let power = fsym_core::elementary::eval_pow(local.clone(), Expr::from_i64(exponent));
            terms.push(fsym_core::elementary::rational_expr(coefficient.clone()) * power);
        }
        if terms.is_empty() {
            return Ok(Expr::from_i64(0));
        }
        Ok(fsym_simplify::simplify(&Expr::Add(terms)))
    })())
}

pub(crate) fn try_limit(
    expression: &Expr,
    variable: &Symbol,
    to: &Expr,
    direction: crate::gruntz::Direction,
) -> Option<std::result::Result<Expr, crate::gruntz::LimitError>> {
    use crate::gruntz::{Direction, LimitError};
    let (point, direction) = match to {
        Expr::Integer(value) => (
            RationalLimitPoint::Finite(BigRational::from_integer(value.clone())),
            match direction {
                Direction::Plus => LocalDirection::Positive,
                Direction::Minus => LocalDirection::Negative,
                Direction::Both => LocalDirection::Both,
            },
        ),
        Expr::Rational(value) => (
            RationalLimitPoint::Finite(value.clone()),
            match direction {
                Direction::Plus => LocalDirection::Positive,
                Direction::Minus => LocalDirection::Negative,
                Direction::Both => LocalDirection::Both,
            },
        ),
        Expr::Const(Constant::Infinity) => (RationalLimitPoint::Infinity, LocalDirection::Positive),
        Expr::Const(Constant::NegativeInfinity) => {
            (RationalLimitPoint::Infinity, LocalDirection::Negative)
        }
        _ => return None,
    };
    Some(
        match rational_expression_limit(expression, variable, &point, direction) {
            Ok(None) => return None,
            Ok(Some(RationalDirectionalLimit::Finite(value))) => {
                Ok(fsym_core::elementary::rational_expr(value))
            }
            Ok(Some(RationalDirectionalLimit::PositiveInfinity)) => {
                Ok(Expr::Const(Constant::Infinity))
            }
            Ok(Some(RationalDirectionalLimit::NegativeInfinity)) => {
                Ok(Expr::Const(Constant::NegativeInfinity))
            }
            Ok(Some(RationalDirectionalLimit::DoesNotExist)) => Err(LimitError::DoesNotExist(
                "the rational pole has opposite one-sided infinite limits".into(),
            )),
            Err(error) => Err(LimitError::Undetermined(error.to_string())),
        },
    )
}
