//! Independent real endpoint calculus for rational integrals.
//!
//! A checked Hermite reduction is completed by logarithmic derivatives and
//! reciprocal quadratics. This reference lane checks a complete polynomial
//! identity, the original-domain obligations and both endpoint limits. At
//! infinity the logarithms are considered together: their exact coefficient
//! of log(abs(x)) must vanish. Subtracting two divergent endpoint values is
//! never accepted as an ordinary improper integral.
//!
//! The primitive uses real log(abs(p(x))). A quadratic reciprocal uses an
//! arctangent for negative discriminant and a real logarithm for positive
//! discriminant. No complex-log branch or numerical sampling is involved.

use crate::rational_definite::{
    RationalDefiniteIntegralVerificationError, RationalIntegralEndpoint,
    verify_rational_integral_domain,
};
use crate::rational_integral::{
    RationalIntegralReduction, RationalIntegralVerificationError,
    verify_rational_integral_reduction,
};
use crate::rational_limit::{
    LocalDirection, RationalDirectionalLimit, RationalLocalCertificate,
    RationalLocalVerificationError, verify_rational_local_order,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::arith::{RationalArithmeticError, metered_mul, metered_sqrt_floor};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Constant, Expr};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;
use thiserror::Error;

const MAX_COEFFICIENTS: usize = 129;
const MAX_TERMS: usize = 128;
const MAX_BITS: u64 = 65_536;
const MAX_STEPS: u64 = 10_000_000;
const MAX_ALLOCATION: u64 = 512 * 1024 * 1024;

/// Untrusted primitive term a log|p(x)| + b integral(1/p(x), x).
/// The polynomial p is monic and the reciprocal coefficient is permitted
/// only for a nondegenerate quadratic. The pole witness means S = p * cofactor,
/// where S is the checked Hermite remainder denominator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementaryRationalTerm {
    pub denominator: Vec<BigRational>,
    pub pole_cofactor: Vec<BigRational>,
    pub logarithm_coefficient: BigRational,
    pub reciprocal_coefficient: BigRational,
}

/// Untrusted in-process witness. Acceptance must be reconstructed against
/// the caller's original source coefficients and endpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementaryRationalDefiniteIntegralCertificate {
    pub primitive: RationalIntegralReduction,
    pub terms: Vec<ElementaryRationalTerm>,
    pub lower_limit: RationalLocalCertificate,
    pub upper_limit: RationalLocalCertificate,
}

/// Exact real value derived by the checker; there is no unchecked
/// constructor or deserialization path to an accepted result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementaryRationalDefiniteIntegralVerification {
    value: Expr,
}

impl ElementaryRationalDefiniteIntegralVerification {
    pub fn value(&self) -> &Expr {
        &self.value
    }
}

#[derive(Debug, Error)]
pub enum ElementaryRationalDefiniteIntegralVerificationError {
    #[error("invalid elementary rational integral: {0}")]
    Invalid(&'static str),
    #[error("the elementary primitive has no finite endpoint limit")]
    DivergentEndpoint,
    #[error(transparent)]
    Domain(#[from] RationalDefiniteIntegralVerificationError),
    #[error(transparent)]
    Reduction(#[from] RationalIntegralVerificationError),
    #[error(transparent)]
    Endpoint(#[from] RationalLocalVerificationError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for ElementaryRationalDefiniteIntegralVerificationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, ElementaryRationalDefiniteIntegralVerificationError>;

fn invalid(reason: &'static str) -> ElementaryRationalDefiniteIntegralVerificationError {
    ElementaryRationalDefiniteIntegralVerificationError::Invalid(reason)
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn scalar(value: &BigRational) -> Result<()> {
    if !value.denom().is_positive() || bits(value) > MAX_BITS {
        return Err(invalid("rational coefficient or height limit"));
    }
    Ok(())
}

fn polynomial(values: &[BigRational]) -> Result<()> {
    if values.is_empty() || values.len() > MAX_COEFFICIENTS {
        return Err(invalid("polynomial coefficient count limit"));
    }
    if values.len() > 1 && values.last().is_some_and(Zero::is_zero) {
        return Err(invalid("noncanonical trailing zero"));
    }
    for value in values {
        scalar(value)?;
    }
    Ok(())
}

fn trim(values: &mut Vec<BigRational>) {
    while values.len() > 1 && values.last().is_some_and(Zero::is_zero) {
        values.pop();
    }
}

struct Reference<'a, M> {
    meter: &'a mut M,
    steps: u64,
    allocation: u64,
}

impl<M: BudgetMeter> Reference<'_, M> {
    fn work(&mut self, steps: usize) -> Result<()> {
        self.meter.checkpoint()?;
        self.steps = self
            .steps
            .checked_sub(steps.max(1) as u64)
            .ok_or_else(|| invalid("reference work limit"))?;
        self.meter
            .charge(Dimension::ComputeSteps, steps.max(1) as u64)?;
        Ok(())
    }

    fn reserve(&mut self, count: usize, height: u64) -> Result<()> {
        if count > 2 * MAX_COEFFICIENTS || height > 2 * MAX_BITS + 1 {
            return Err(invalid("reference arithmetic growth limit"));
        }
        let bytes = (count as u64).saturating_mul(64 + 2 * height.div_ceil(8));
        self.allocation = self
            .allocation
            .checked_add(bytes)
            .ok_or_else(|| invalid("reference allocation limit"))?;
        if self.allocation > MAX_ALLOCATION {
            return Err(invalid("reference allocation limit"));
        }
        self.work(count)?;
        self.meter.charge_batch(&[
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, 2 * count as u64 + 1),
        ])?;
        Ok(())
    }

    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        scalar(value)?;
        self.reserve(1, bits(value))?;
        Ok(value.clone())
    }

    fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.reserve(8, bits(left) + bits(right) + 1)?;
        let out = left.metered_add(right, self.meter)?;
        scalar(&out)?;
        Ok(out)
    }

    fn sub(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.reserve(8, bits(left) + bits(right) + 1)?;
        let out = left.metered_sub(right, self.meter)?;
        scalar(&out)?;
        Ok(out)
    }

    fn mul(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.reserve(8, bits(left) + bits(right) + 1)?;
        let out = left.metered_mul(right, self.meter)?;
        scalar(&out)?;
        Ok(out)
    }

    fn div(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.reserve(8, bits(left) + bits(right) + 1)?;
        let out = left.metered_div(right, self.meter)?;
        scalar(&out)?;
        Ok(out)
    }

    fn product(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Vec<BigRational>> {
        let count = a.len() + b.len() - 1;
        let height = a.iter().map(bits).max().unwrap_or(1) + b.iter().map(bits).max().unwrap_or(1);
        self.reserve(count, height)?;
        let mut out = vec![BigRational::zero(); count];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                let product = self.mul(ai, bj)?;
                out[i + j] = self.add(&out[i + j], &product)?;
            }
        }
        trim(&mut out);
        Ok(out)
    }

    fn sum(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Vec<BigRational>> {
        let count = a.len().max(b.len());
        let height = 2 * a.iter().chain(b).map(bits).max().unwrap_or(1) + 1;
        self.reserve(count, height)?;
        let mut out = vec![BigRational::zero(); count];
        for (i, value) in out.iter_mut().enumerate() {
            if let Some(ai) = a.get(i) {
                *value = self.add(value, ai)?;
            }
            if let Some(bi) = b.get(i) {
                *value = self.add(value, bi)?;
            }
        }
        trim(&mut out);
        Ok(out)
    }

    fn numerator(&mut self, term: &ElementaryRationalTerm) -> Result<Vec<BigRational>> {
        let count = term.denominator.len() - 1;
        self.reserve(count, MAX_BITS)?;
        let mut out = vec![BigRational::zero(); count];
        for (degree, coefficient) in term.denominator.iter().enumerate().skip(1) {
            let derivative = self.mul(coefficient, &BigRational::from(degree as i64))?;
            out[degree - 1] = self.mul(&term.logarithm_coefficient, &derivative)?;
        }
        out[0] = self.add(&out[0], &term.reciprocal_coefficient)?;
        trim(&mut out);
        Ok(out)
    }

    fn eval(&mut self, polynomial: &[BigRational], point: &BigRational) -> Result<BigRational> {
        let mut value = BigRational::zero();
        for coefficient in polynomial.iter().rev() {
            let product = self.mul(&value, point)?;
            value = self.add(&product, coefficient)?;
        }
        Ok(value)
    }

    fn delta(&mut self, term: &ElementaryRationalTerm) -> Result<BigRational> {
        let four_c = self.mul(&BigRational::from(4), &term.denominator[0])?;
        let b_squared = self.mul(&term.denominator[1], &term.denominator[1])?;
        self.sub(&four_c, &b_squared)
    }

    fn expr_number(&mut self, value: &BigRational) -> Result<Expr> {
        Ok(rational_expr(self.copy(value)?))
    }

    /// Only perfect-square detection is required. Avoid the generic power
    /// simplifier's factor search for an irrational rational square root.
    fn square_root(&mut self, value: &BigRational) -> Result<Expr> {
        if !value.is_positive() {
            return Err(invalid("nonpositive real square root"));
        }
        self.work(1)?;
        let numerator = metered_sqrt_floor(value.numer(), self.meter)?
            .ok_or_else(|| invalid("negative square-root numerator"))?;
        let denominator = metered_sqrt_floor(value.denom(), self.meter)?
            .ok_or_else(|| invalid("negative square-root denominator"))?;
        let numerator_squared = metered_mul(&numerator, &numerator, self.meter)?;
        let denominator_squared = metered_mul(&denominator, &denominator, self.meter)?;
        if &numerator_squared == value.numer() && &denominator_squared == value.denom() {
            let exact = BigRational::metered_new(&numerator, &denominator, self.meter)?;
            return Ok(rational_expr(exact));
        }
        self.reserve(4, bits(value))?;
        Ok(Expr::Pow(
            Arc::new(self.expr_number(value)?),
            Arc::new(rational_expr(BigRational::new(1.into(), 2.into()))),
        ))
    }

    fn reciprocal(&mut self, expr: Expr) -> Result<Expr> {
        self.reserve(3, 1)?;
        match expr {
            Expr::Integer(integer) => {
                let rational = BigRational::from_integer(integer);
                Ok(rational_expr(self.div(&BigRational::one(), &rational)?))
            }
            Expr::Rational(rational) => {
                Ok(rational_expr(self.div(&BigRational::one(), &rational)?))
            }
            expr => Ok(Expr::Pow(Arc::new(expr), Arc::new(Expr::from_i64(-1)))),
        }
    }

    fn log_positive(&mut self, value: &BigRational) -> Result<Expr> {
        if !value.is_positive() {
            return Err(invalid("nonpositive real logarithm"));
        }
        if value.is_one() {
            return Ok(Expr::from_i64(0));
        }
        self.reserve(2, bits(value))?;
        Ok(Expr::Function("log".into(), vec![self.expr_number(value)?]))
    }

    fn arctangent(&mut self, argument: Expr) -> Result<Expr> {
        self.reserve(4, MAX_BITS)?;
        if argument.is_zero() {
            return Ok(Expr::from_i64(0));
        }
        if argument == Expr::from_i64(1) || argument == Expr::from_i64(-1) {
            let numerator = if argument.is_one() { 1 } else { -1 };
            return Ok(rational_expr(BigRational::new(numerator.into(), 4.into()))
                * Expr::Const(Constant::Pi));
        }
        Ok(Expr::Function("atan".into(), vec![argument]))
    }

    fn quadratic_endpoint(
        &mut self,
        term: &ElementaryRationalTerm,
        endpoint: &RationalIntegralEndpoint,
    ) -> Result<Expr> {
        let delta = self.delta(term)?;
        if delta.is_zero() {
            return Err(invalid("degenerate reciprocal quadratic"));
        }
        let magnitude = if delta.is_negative() {
            self.sub(&BigRational::zero(), &delta)?
        } else {
            self.copy(&delta)?
        };
        let root = self.square_root(&magnitude)?;
        let inverse_root = self.reciprocal(root.clone())?;
        let coefficient = self.expr_number(&term.reciprocal_coefficient)?;
        match endpoint {
            RationalIntegralEndpoint::NegativeInfinity
            | RationalIntegralEndpoint::PositiveInfinity => {
                if delta.is_negative() {
                    return Ok(Expr::from_i64(0));
                }
                let sign = if *endpoint == RationalIntegralEndpoint::PositiveInfinity {
                    1
                } else {
                    -1
                };
                Ok(Expr::from_i64(sign) * coefficient * inverse_root * Expr::Const(Constant::Pi))
            }
            RationalIntegralEndpoint::Finite(point) => {
                let two_x = self.mul(&BigRational::from(2), point)?;
                let z = self.add(&two_x, &term.denominator[1])?;
                let z_expr = self.expr_number(&z)?;
                if delta.is_positive() {
                    let argument = z_expr * inverse_root.clone();
                    let angle = self.arctangent(argument)?;
                    Ok(Expr::from_i64(2) * coefficient * inverse_root * angle)
                } else {
                    // For d = sqrt(b^2 - 4c),
                    // (1/d) log |(2x+b-d)/(2x+b+d)| differentiates to 1/p.
                    // Neither argument can vanish because p(point) != 0;
                    // the checked original domain supplies the same fact
                    // throughout the integration interval.
                    self.reserve(20, MAX_BITS)?;
                    let denominator = z_expr.clone() + root.clone();
                    let ratio = (z_expr - root) * self.reciprocal(denominator)?;
                    let absolute = Expr::Function("Abs".into(), vec![ratio]);
                    let logarithm = Expr::Function("log".into(), vec![absolute]);
                    Ok(coefficient * inverse_root * logarithm)
                }
            }
        }
    }

    fn elementary_endpoint(
        &mut self,
        terms: &[ElementaryRationalTerm],
        endpoint: &RationalIntegralEndpoint,
        logarithmic_growth: &BigRational,
    ) -> Result<Expr> {
        if !matches!(endpoint, RationalIntegralEndpoint::Finite(_)) && !logarithmic_growth.is_zero()
        {
            return Err(ElementaryRationalDefiniteIntegralVerificationError::DivergentEndpoint);
        }
        let mut value = Expr::from_i64(0);
        for term in terms {
            self.reserve(32, MAX_BITS)?;
            if let RationalIntegralEndpoint::Finite(point) = endpoint {
                let p = self.eval(&term.denominator, point)?;
                if p.is_zero() {
                    return Err(
                        ElementaryRationalDefiniteIntegralVerificationError::DivergentEndpoint,
                    );
                }
                if !term.logarithm_coefficient.is_zero() {
                    let magnitude = if p.is_negative() {
                        self.sub(&BigRational::zero(), &p)?
                    } else {
                        p
                    };
                    let logarithm = self.log_positive(&magnitude)?;
                    value = value + self.expr_number(&term.logarithm_coefficient)? * logarithm;
                }
            }
            if !term.reciprocal_coefficient.is_zero() {
                value = value + self.quadratic_endpoint(term, endpoint)?;
            }
        }
        Ok(value)
    }
}

/// Check and derive the exact real definite value without invoking an
/// integrator, factorizer, heuristic simplifier or transcendental evaluator.
pub fn verify_elementary_rational_definite_integral(
    numerator: &[BigRational],
    denominator: &[BigRational],
    lower: &RationalIntegralEndpoint,
    upper: &RationalIntegralEndpoint,
    witness: &ElementaryRationalDefiniteIntegralCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<ElementaryRationalDefiniteIntegralVerification> {
    meter.checkpoint()?;
    if witness.terms.len() > MAX_TERMS {
        return Err(invalid("elementary term count limit"));
    }
    for term in &witness.terms {
        polynomial(&term.denominator)?;
        polynomial(&term.pole_cofactor)?;
        scalar(&term.logarithm_coefficient)?;
        scalar(&term.reciprocal_coefficient)?;
        if term.denominator.len() < 2
            || !term.denominator.last().is_some_and(One::is_one)
            || (term.logarithm_coefficient.is_zero() && term.reciprocal_coefficient.is_zero())
            || (!term.reciprocal_coefficient.is_zero() && term.denominator.len() != 3)
        {
            return Err(invalid("invalid elementary primitive term"));
        }
    }
    // Source-domain checks precede all primitive cancellation and endpoint
    // arithmetic, including zero and reversed-interval cases.
    verify_rational_integral_domain(denominator, lower, upper, meter)?;
    verify_rational_integral_reduction(numerator, denominator, &witness.primitive, meter)?;
    let primitive = &witness.primitive;
    let left_point = lower.metered_limit_point(meter)?;
    let left = verify_rational_local_order(
        &primitive.rational_numerator,
        &primitive.rational_denominator,
        &left_point,
        &witness.lower_limit,
        meter,
    )?;
    let right_point = upper.metered_limit_point(meter)?;
    let right = verify_rational_local_order(
        &primitive.rational_numerator,
        &primitive.rational_denominator,
        &right_point,
        &witness.upper_limit,
        meter,
    )?;
    let left_bits = bits(left.leading_coefficient());
    let right_bits = bits(right.leading_coefficient());
    meter.charge_batch(&[
        (Dimension::AllocationCount, 4),
        (
            Dimension::MemoryBytes,
            128 + 2 * left_bits.div_ceil(8) + 2 * right_bits.div_ceil(8),
        ),
    ])?;
    let RationalDirectionalLimit::Finite(left_value) = left.limit(LocalDirection::Both) else {
        return Err(ElementaryRationalDefiniteIntegralVerificationError::DivergentEndpoint);
    };
    let RationalDirectionalLimit::Finite(right_value) = right.limit(LocalDirection::Both) else {
        return Err(ElementaryRationalDefiniteIntegralVerificationError::DivergentEndpoint);
    };
    let mut reference = Reference {
        meter,
        steps: MAX_STEPS,
        allocation: 0,
    };
    let mut product = vec![BigRational::one()];
    let mut sum = vec![BigRational::zero()];
    let mut logarithmic_growth = BigRational::zero();
    for term in &witness.terms {
        if reference.product(&term.denominator, &term.pole_cofactor)?
            != primitive.remainder_denominator
        {
            return Err(invalid("elementary term introduces an unproved pole"));
        }
        if !term.reciprocal_coefficient.is_zero() && reference.delta(term)?.is_zero() {
            return Err(invalid("degenerate reciprocal quadratic"));
        }
        product = reference.product(&product, &term.denominator)?;
        let local_numerator = reference.numerator(term)?;
        let contribution = reference.product(&local_numerator, &term.pole_cofactor)?;
        sum = reference.sum(&sum, &contribution)?;
        let degree = BigRational::from((term.denominator.len() - 1) as i64);
        let growth = reference.mul(&term.logarithm_coefficient, &degree)?;
        logarithmic_growth = reference.add(&logarithmic_growth, &growth)?;
    }
    // S is independently square-free. Product equality therefore also
    // establishes pairwise coprimality, so distinct endpoint poles cannot
    // hide an unexamined cancellation between primitive terms.
    if product != primitive.remainder_denominator || sum != primitive.remainder_numerator {
        return Err(invalid("incomplete elementary derivative identity"));
    }
    let left_elementary =
        reference.elementary_endpoint(&witness.terms, lower, &logarithmic_growth)?;
    let right_elementary =
        reference.elementary_endpoint(&witness.terms, upper, &logarithmic_growth)?;
    let rational_difference = reference.sub(&right_value, &left_value)?;
    reference.reserve(64, MAX_BITS)?;
    let value = reference.expr_number(&rational_difference)? + right_elementary - left_elementary;
    reference.meter.checkpoint()?;
    Ok(ElementaryRationalDefiniteIntegralVerification { value })
}
