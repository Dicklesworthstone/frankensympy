//! Bounded ordinary Fourier transforms in the cycles-per-unit convention.
//!
//! This catalog applies two closed real-line integral identities, with their
//! convergence premises checked using exact rational arithmetic:
//! exp(-a*x^2+b*x+c), a > 0, and exp(c-a*abs(b*x+d)), a > 0, b != 0.
//! Finite sums and rational multiples compose these absolutely integrable
//! functions. Nonzero constants and other distributions are deliberately not
//! ordinary integrals. The angular-frequency catalog in transforms retains
//! its separate API and convention.
//!
//! Original syntax is inspected before normalization, including every child
//! of zero products and powers. No numerical sampling or assumed sign admits
//! a rule. These direct catalog identities are not portable proof-kernel
//! certificates or a general transform/completeness claim.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;
use thiserror::Error;

const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_NAMES: usize = 65_536;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 65_536;
const MAX_POWER: u64 = 16;
const MAX_TERMS: usize = 64;
const MAX_OUTPUT_NODES: usize = 8192;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum FourierError {
    #[error("ordinary Fourier catalog does not support {0}")]
    Unsupported(&'static str),
    #[error("ordinary Fourier transform refused: {0}")]
    Refused(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for FourierError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, FourierError>;
type Polynomial = Vec<BigRational>;

struct Work<'a, M> {
    meter: &'a mut M,
    compute: u64,
    memory: u64,
    terms: usize,
    output_nodes: usize,
    used_bilateral_identity: bool,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            compute: 0,
            memory: 0,
            terms: 0,
            output_nodes: 0,
            used_bilateral_identity: false,
        }
    }

    fn allocate(&mut self, count: usize, bytes: u64) -> Result<()> {
        self.checkpoint()?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, count.max(1) as u64),
            (Dimension::AllocationCount, count.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1)),
        ])?;
        Ok(())
    }

    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        self.allocate(1, 64 + 2 * bits(value).div_ceil(8))?;
        Ok(value.clone())
    }

    fn preflight(&mut self, left: &BigRational, right: &BigRational) -> Result<()> {
        self.checkpoint()?;
        if bits(left).saturating_add(bits(right)).saturating_add(2) > MAX_WORK_BITS {
            return Err(FourierError::Refused("intermediate scalar growth bound"));
        }
        Ok(())
    }

    fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.preflight(left, right)?;
        Ok(left.metered_add(right, self)?)
    }

    fn mul(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.preflight(left, right)?;
        Ok(left.metered_mul(right, self)?)
    }

    fn div(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.preflight(left, right)?;
        if right.is_zero() {
            return Err(FourierError::Refused("undefined rational division"));
        }
        Ok(left.metered_div(right, self)?)
    }

    fn pow(&mut self, base: &BigRational, power: i32) -> Result<BigRational> {
        self.checkpoint()?;
        if base.is_zero() && power <= 0 {
            return Err(FourierError::Refused("undefined zero power"));
        }
        if bits(base).saturating_mul(u64::from(power.unsigned_abs()).max(1)) > MAX_WORK_BITS / 2 {
            return Err(FourierError::Refused("rational power growth bound"));
        }
        Ok(base.metered_pow(power, self)?)
    }

    fn output(&mut self, children: usize) -> Result<()> {
        self.output_nodes += 1;
        if self.output_nodes > MAX_OUTPUT_NODES {
            return Err(FourierError::Refused("output node bound"));
        }
        self.allocate(children + 1, 128 * (children + 1) as u64)
    }

    fn scalar(&mut self, value: &BigRational) -> Result<Expr> {
        self.output(0)?;
        Ok(rational_expr(self.copy(value)?))
    }

    fn symbol(&mut self, symbol: &Symbol) -> Result<Expr> {
        self.output(0)?;
        self.allocate(1, symbol.name.len() as u64)?;
        Ok(Expr::Sym(symbol.clone()))
    }

    fn constant(&mut self, constant: Constant) -> Result<Expr> {
        self.output(0)?;
        Ok(Expr::Const(constant))
    }

    fn sum(&mut self, mut terms: Vec<Expr>) -> Result<Expr> {
        terms.retain(|term| !term.is_zero());
        match terms.len() {
            0 => self.scalar(&BigRational::zero()),
            1 => Ok(terms.pop().expect("one term")),
            count => {
                self.output(count)?;
                Ok(Expr::Add(terms))
            }
        }
    }

    fn product(&mut self, mut factors: Vec<Expr>) -> Result<Expr> {
        // Only generated, finite-on-the-real-axis factors enter this helper.
        if factors.iter().any(Expr::is_zero) {
            return self.scalar(&BigRational::zero());
        }
        factors.retain(|factor| !factor.is_one());
        match factors.len() {
            0 => self.scalar(&BigRational::one()),
            1 => Ok(factors.pop().expect("one factor")),
            count => {
                self.output(count)?;
                Ok(Expr::Mul(factors))
            }
        }
    }

    fn power(&mut self, base: Expr, exponent: BigRational) -> Result<Expr> {
        let exponent = self.scalar(&exponent)?;
        self.output(2)?;
        Ok(Expr::Pow(Arc::new(base), Arc::new(exponent)))
    }

    fn exp(&mut self, argument: Expr) -> Result<Expr> {
        if argument.is_zero() {
            return self.scalar(&BigRational::one());
        }
        self.output(1)?;
        Ok(Expr::Function("exp".into(), vec![argument]))
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        let (mut compute, mut memory) = (self.compute, self.memory);
        for &(dimension, amount) in charges {
            let (used, limit) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_COMPUTE),
                Dimension::MemoryBytes => (&mut memory, MAX_MEMORY),
                _ => continue,
            };
            let remaining = limit.saturating_sub(*used);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *used += amount;
        }
        self.meter.charge_batch(charges)?;
        (self.compute, self.memory) = (compute, memory);
        Ok(())
    }
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn preflight<M: BudgetMeter>(
    source: &Expr,
    variable: &Symbol,
    frequency: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<()> {
    let mut names = variable.name.len().saturating_add(frequency.name.len());
    if names > MAX_NAMES {
        return Err(FourierError::Refused("variable names exceed their bound"));
    }
    if variable == frequency {
        return Err(FourierError::Refused(
            "integration and frequency variables must be distinct",
        ));
    }
    work.allocate(MAX_NODES, MAX_NODES as u64 * 32)?;
    let mut pending = Vec::with_capacity(MAX_NODES);
    pending.push((source, 0));
    let mut visited = 0;
    while let Some((node, depth)) = pending.pop() {
        work.charge(Dimension::ComputeSteps, 1)?;
        visited += 1;
        if depth > MAX_DEPTH || visited > MAX_NODES {
            return Err(FourierError::Refused("original source node/depth bound"));
        }
        match node {
            Expr::Integer(value) if value.bits() > MAX_INPUT_BITS => {
                return Err(FourierError::Refused("input coefficient exceeds 4096 bits"));
            }
            Expr::Rational(value) if bits(value) > MAX_INPUT_BITS => {
                return Err(FourierError::Refused("input coefficient exceeds 4096 bits"));
            }
            Expr::Sym(symbol) => {
                names = names.saturating_add(symbol.name.len());
                if names > MAX_NAMES {
                    return Err(FourierError::Refused("source name bound"));
                }
                work.charge(
                    Dimension::ComputeSteps,
                    symbol.name.len().div_ceil(64).max(1) as u64,
                )?;
                if symbol == frequency {
                    return Err(FourierError::Refused("source depends on the frequency variable"));
                }
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = node {
                    names = names.saturating_add(name.len());
                    if names > MAX_NAMES {
                        return Err(FourierError::Refused("source name bound"));
                    }
                }
                if children.len() > MAX_NODES.saturating_sub(visited + pending.len()) {
                    return Err(FourierError::Refused("original source node bound"));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if let Expr::Integer(power) = exponent.as_ref() {
                    if power.to_i64().is_none_or(|power| power.unsigned_abs() > MAX_POWER) {
                        return Err(FourierError::Refused("original integer power bound"));
                    }
                    if base.is_zero() && !power.is_positive() {
                        return Err(FourierError::Refused("undefined original zero power"));
                    }
                }
                if 2 > MAX_NODES.saturating_sub(visited + pending.len()) {
                    return Err(FourierError::Refused("original source node bound"));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
            }
            Expr::Const(
                Constant::NaN
                | Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity,
            ) => {
                return Err(FourierError::Refused("nonfinite original source"));
            }
            _ => {}
        }
    }
    Ok(())
}

fn number<M: BudgetMeter>(source: &Expr, work: &mut Work<'_, M>) -> Result<Option<BigRational>> {
    work.charge(Dimension::ComputeSteps, 1)?;
    match source {
        Expr::Integer(value) => {
            work.allocate(1, 64 + value.bits().div_ceil(8))?;
            Ok(Some(BigRational::from_integer(value.clone())))
        }
        Expr::Rational(value) => Ok(Some(work.copy(value)?)),
        Expr::Add(children) | Expr::Mul(children) => {
            let addition = matches!(source, Expr::Add(_));
            let mut value = if addition { BigRational::zero() } else { BigRational::one() };
            let mut supported = true;
            for child in children {
                match number(child, work)? {
                    Some(coefficient) => {
                        value = if addition {
                            work.add(&value, &coefficient)?
                        } else {
                            work.mul(&value, &coefficient)?
                        };
                    }
                    None => supported = false,
                }
            }
            Ok(supported.then_some(value))
        }
        Expr::Pow(base, exponent) => {
            let value = number(base, work)?;
            let Expr::Integer(power) = exponent.as_ref() else {
                return Ok(None);
            };
            let power = power.to_i64().ok_or(FourierError::Refused("integer power bound"))?;
            match value {
                Some(value) => Ok(Some(work.pow(&value, power as i32)?)),
                None => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

fn trim(mut polynomial: Polynomial) -> Polynomial {
    while polynomial.len() > 1 && polynomial.last().is_some_and(Zero::is_zero) {
        polynomial.pop();
    }
    polynomial
}

fn polynomial_product<M: BudgetMeter>(
    left: &[BigRational],
    right: &[BigRational],
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    let degree = left.len() + right.len() - 2;
    if degree > 2 {
        return Err(FourierError::Unsupported("exponent degree above two"));
    }
    work.allocate(degree + 1, 64 * (degree + 1) as u64)?;
    let mut result = vec![BigRational::zero(); degree + 1];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            let product = work.mul(a, b)?;
            result[i + j] = work.add(&result[i + j], &product)?;
        }
    }
    Ok(trim(result))
}

fn polynomial<M: BudgetMeter>(
    source: &Expr,
    variable: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    work.charge(Dimension::ComputeSteps, 1)?;
    if let Some(value) = number(source, work)? {
        return Ok(vec![value]);
    }
    match source {
        Expr::Sym(symbol) if symbol == variable => {
            Ok(vec![BigRational::zero(), BigRational::one()])
        }
        Expr::Add(children) => {
            let mut result = vec![BigRational::zero()];
            for child in children {
                let child = polynomial(child, variable, work)?;
                if result.len() < child.len() {
                    work.allocate(child.len(), 64 * child.len() as u64)?;
                    result.resize(child.len(), BigRational::zero());
                }
                for (value, term) in result.iter_mut().zip(child) {
                    *value = work.add(value, &term)?;
                }
            }
            Ok(trim(result))
        }
        Expr::Mul(children) => {
            let mut result = vec![BigRational::one()];
            for child in children {
                let child = polynomial(child, variable, work)?;
                result = polynomial_product(&result, &child, work)?;
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            // Parse the original base even when the outer exponent is zero.
            let base = polynomial(base, variable, work)?;
            let Expr::Integer(power) = exponent.as_ref() else {
                return Err(FourierError::Unsupported("noninteger polynomial powers"));
            };
            let power = power.to_i64().ok_or(FourierError::Refused("integer power bound"))?;
            if power < 0 {
                return Err(FourierError::Unsupported("nonconstant reciprocal exponents"));
            }
            let mut result = vec![BigRational::one()];
            for _ in 0..power {
                result = polynomial_product(&result, &base, work)?;
            }
            Ok(result)
        }
        _ => Err(FourierError::Unsupported("nonrational quadratic exponent")),
    }
}

struct AbsoluteExponent {
    constant: BigRational,
    // coefficient * abs(x - center), with an exact rational center.
    term: Option<(BigRational, BigRational)>,
}

fn absolute_exponent<M: BudgetMeter>(
    source: &Expr,
    variable: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<AbsoluteExponent> {
    work.charge(Dimension::ComputeSteps, 1)?;
    if let Some(constant) = number(source, work)? {
        return Ok(AbsoluteExponent { constant, term: None });
    }
    match source {
        Expr::Function(name, arguments)
            if (name == "Abs" || name == "abs") && arguments.len() == 1 =>
        {
            let affine = polynomial(&arguments[0], variable, work)?;
            if affine.len() == 1 {
                return Ok(AbsoluteExponent { constant: affine[0].abs(), term: None });
            }
            if affine.len() != 2 {
                return Err(FourierError::Unsupported("absolute value of a nonlinear argument"));
            }
            let center = work.div(&(-&affine[0]), &affine[1])?;
            Ok(AbsoluteExponent {
                constant: BigRational::zero(),
                term: Some((affine[1].abs(), center)),
            })
        }
        Expr::Add(children) => {
            let mut result = AbsoluteExponent { constant: BigRational::zero(), term: None };
            for child in children {
                let child = absolute_exponent(child, variable, work)?;
                result.constant = work.add(&result.constant, &child.constant)?;
                if let Some((coefficient, center)) = child.term {
                    match &mut result.term {
                        None => result.term = Some((coefficient, center)),
                        Some((old_coefficient, old_center)) => {
                            if old_center.metered_cmp(&center, work)?.is_ne() {
                                return Err(FourierError::Unsupported("multiple absolute-value centers"));
                            }
                            *old_coefficient = work.add(old_coefficient, &coefficient)?;
                        }
                    }
                }
            }
            Ok(result)
        }
        Expr::Mul(children) => {
            let mut constant = BigRational::one();
            let mut dependent = None;
            for child in children {
                let child = absolute_exponent(child, variable, work)?;
                if child.term.is_none() {
                    constant = work.mul(&constant, &child.constant)?;
                } else if dependent.is_none() {
                    dependent = Some(child);
                } else {
                    return Err(FourierError::Unsupported("products of absolute-value factors"));
                }
            }
            if let Some(mut dependent) = dependent {
                dependent.constant = work.mul(&constant, &dependent.constant)?;
                if let Some((coefficient, _)) = &mut dependent.term {
                    *coefficient = work.mul(&constant, coefficient)?;
                }
                Ok(dependent)
            } else {
                Ok(AbsoluteExponent { constant, term: None })
            }
        }
        _ => Err(FourierError::Unsupported("nonaffine absolute-value exponent")),
    }
}

fn phase<M: BudgetMeter>(
    center: &BigRational,
    frequency: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    let scale = work.mul(&BigRational::from(-2), center)?;
    let scale = work.scalar(&scale)?;
    let pi = work.constant(Constant::Pi)?;
    let imaginary = work.constant(Constant::I)?;
    let k = work.symbol(frequency)?;
    let argument = work.product(vec![scale, pi, imaginary, k])?;
    work.exp(argument)
}

fn gaussian<M: BudgetMeter>(
    coefficients: &[BigRational],
    frequency: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    if coefficients.len() != 3 || !coefficients[2].is_negative() {
        return Err(FourierError::Unsupported(
            "a Gaussian without strictly positive exact decay",
        ));
    }
    // Completing the square gives center=b/(2a), height=c+b^2/(4a).
    let a = -&coefficients[2];
    let inverse_a = work.div(&BigRational::one(), &a)?;
    let twice_a = work.mul(&BigRational::from(2), &a)?;
    let center = work.div(&coefficients[1], &twice_a)?;
    let square = work.mul(&coefficients[1], &coefficients[1])?;
    let four_a = work.mul(&BigRational::from(4), &a)?;
    let offset = work.div(&square, &four_a)?;
    let height = work.add(&coefficients[0], &offset)?;

    let scale = work.scalar(&inverse_a)?;
    let pi = work.constant(Constant::Pi)?;
    let scale_pi = work.product(vec![scale, pi])?;
    let amplitude = work.power(scale_pi, BigRational::new(1.into(), 2.into()))?;
    let pi = work.constant(Constant::Pi)?;
    let pi_squared = work.power(pi, BigRational::from(2))?;
    let k = work.symbol(frequency)?;
    let k_squared = work.power(k, BigRational::from(2))?;
    let negative_scale = work.scalar(&(-inverse_a))?;
    let decay = work.product(vec![negative_scale, pi_squared, k_squared])?;
    let height = work.scalar(&height)?;
    let argument = work.sum(vec![height, decay])?;
    let envelope = work.exp(argument)?;
    let shift = phase(&center, frequency, work)?;
    work.product(vec![amplitude, envelope, shift])
}

fn bilateral_exponential<M: BudgetMeter>(
    exponent: AbsoluteExponent,
    frequency: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    let Some((coefficient, center)) = exponent.term else {
        return Err(FourierError::Unsupported("a nonzero constant integrand"));
    };
    if !coefficient.is_negative() {
        return Err(FourierError::Unsupported(
            "a bilateral exponential without strictly positive exact decay",
        ));
    }
    work.used_bilateral_identity = true;
    let rate = -coefficient;
    let twice_rate = work.mul(&BigRational::from(2), &rate)?;
    let rate_squared = work.mul(&rate, &rate)?;
    let numerator = work.scalar(&twice_rate)?;
    let rate_squared = work.scalar(&rate_squared)?;
    let four = work.scalar(&BigRational::from(4))?;
    let pi = work.constant(Constant::Pi)?;
    let pi_squared = work.power(pi, BigRational::from(2))?;
    let k = work.symbol(frequency)?;
    let k_squared = work.power(k, BigRational::from(2))?;
    let frequency_squared = work.product(vec![four, pi_squared, k_squared])?;
    let denominator = work.sum(vec![rate_squared, frequency_squared])?;
    let reciprocal = work.power(denominator, BigRational::from(-1))?;
    let constant = work.scalar(&exponent.constant)?;
    let amplitude = work.exp(constant)?;
    let shift = phase(&center, frequency, work)?;
    work.product(vec![numerator, reciprocal, amplitude, shift])
}

fn transform<M: BudgetMeter>(
    source: &Expr,
    variable: &Symbol,
    frequency: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    work.checkpoint()?;
    if let Some(value) = number(source, work)? {
        return if value.is_zero() {
            work.scalar(&value)
        } else {
            Err(FourierError::Unsupported("a nonzero constant integrand"))
        };
    }
    match source {
        Expr::Add(children) => {
            work.allocate(children.len(), 128 * children.len() as u64)?;
            let mut values = Vec::with_capacity(children.len());
            for child in children {
                values.push(transform(child, variable, frequency, work)?);
            }
            work.sum(values)
        }
        Expr::Mul(children) => {
            let mut coefficient = BigRational::one();
            let mut dependent = None;
            for child in children {
                if let Some(value) = number(child, work)? {
                    coefficient = work.mul(&coefficient, &value)?;
                } else if dependent.is_none() {
                    dependent = Some(child);
                } else {
                    return Err(FourierError::Unsupported(
                        "products outside rational scalar multiplication",
                    ));
                }
            }
            let dependent = dependent
                .ok_or(FourierError::Unsupported("a nonzero constant integrand"))?;
            // Do not erase an unsupported or undefined factor through zero.
            let value = transform(dependent, variable, frequency, work)?;
            let coefficient = work.scalar(&coefficient)?;
            work.product(vec![coefficient, value])
        }
        Expr::Function(name, arguments) if name == "exp" && arguments.len() == 1 => {
            work.terms += 1;
            if work.terms > MAX_TERMS {
                return Err(FourierError::Refused("more than sixty-four catalog terms"));
            }
            match polynomial(&arguments[0], variable, work) {
                Ok(coefficients) => gaussian(&coefficients, frequency, work),
                Err(FourierError::Unsupported(_)) => {
                    let exponent = absolute_exponent(&arguments[0], variable, work)?;
                    bilateral_exponential(exponent, frequency, work)
                }
                Err(error) => Err(error),
            }
        }
        _ => Err(FourierError::Unsupported("this original source grammar")),
    }
}

/// A direct catalog evaluation, without a portable proof/certification claim.
#[derive(Debug)]
pub struct OrdinaryFourierTransform {
    pub value: Expr,
    /// Which closed integral identity was used, not a convergence condition.
    /// Public compatibility layers may retain a more restrictive oracle
    /// condition than this catalog's exact real-frequency theorem requires.
    pub used_bilateral_identity: bool,
}

/// Ordinary real-line Fourier transform with kernel exp(-2*pi*I*k*x).
///
/// The frequency is a formal real scalar. Gaussian outputs additionally
/// continue to every complex frequency; bilateral outputs need a separate
/// convergence strip off the real axis. The native Symbol identity is not an
/// assumptions context, and does not itself discharge a complex-domain claim.
///
/// The caller's meter and fixed whole-request limits are cumulative across
/// source admission, exact coefficient arithmetic and construction. Exhaustion
/// and cancellation are terminal, never returned as unsupported input.
pub fn metered_fourier_transform_l1<M: BudgetMeter>(
    source: &Expr,
    variable: &Symbol,
    frequency: &Symbol,
    meter: &mut M,
) -> Result<OrdinaryFourierTransform> {
    let mut work = Work::new(meter);
    preflight(source, variable, frequency, &mut work)?;
    let value = transform(source, variable, frequency, &mut work)?;
    work.checkpoint()?;
    Ok(OrdinaryFourierTransform {
        value,
        used_bilateral_identity: work.used_bilateral_identity,
    })
}

pub fn fourier_transform_l1(
    source: &Expr,
    variable: &Symbol,
    frequency: &Symbol,
) -> Result<Expr> {
    Ok(metered_fourier_transform_l1(source, variable, frequency, &mut Unbounded)?.value)
}
