//! Checked polynomial-weighted exponential generating series.
//!
//! For `p in QQ[k]`, finite complex `z`, and an exact integer `m >= 0`,
//! `sum_{k=m}^oo p(k)*z^k/k! = exp(z)*sum_j c_j*z^j - sum_{k=0}^{m-1} p(k)*z^k/k!`,
//! where `p(k) = sum_j c_j*k*(k-1)*...*(k-j+1)`. Generation obtains `c_j`
//! from forward differences. The independent checker reconstructs the
//! original polynomial by multiplying the falling-factorial factors; it
//! never uses the forward-difference generator or sampled equality.
//!
//! Absolute convergence follows from the exponential series after the
//! finite falling-factorial decomposition. A symbolic parameter denotes the
//! resulting entire function, not an assertion about an infinite argument.
//! Explicit infinities, poles, unsupported bounds and source syntax refuse.
//! This is a bounded in-process receipt, not a portable proof-kernel claim.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::elementary::{eval_function, rational_expr};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use num_traits::{One, Zero};
use std::sync::Arc;
use thiserror::Error;

const MAX_DEGREE: usize = 32;
const MAX_LOWER: usize = 256;
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 65_536;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ExponentialSummationError {
    #[error("polynomial-exponential summation does not support {0}")]
    Unsupported(&'static str),
    #[error("polynomial-exponential summation refused: {0}")]
    Refused(&'static str),
    #[error("polynomial-exponential coefficient identity does not hold")]
    InvalidCertificate,
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for ExponentialSummationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, ExponentialSummationError>;
type Coefficients = Vec<BigRational>;

/// Untrusted coefficients in the ascending falling-factorial basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialExponentialCertificate {
    pub falling_coefficients: Coefficients,
}

/// Original-source-bound result for a positive-infinite factorial series.
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialExponentialSum {
    source: Expr,
    index: Symbol,
    lower: Expr,
    parameter: Expr,
    certificate: PolynomialExponentialCertificate,
    value: Expr,
}

impl VerifiedPolynomialExponentialSum {
    pub fn source(&self) -> &Expr {
        &self.source
    }
    pub fn index(&self) -> &Symbol {
        &self.index
    }
    pub fn lower(&self) -> &Expr {
        &self.lower
    }
    pub fn parameter(&self) -> &Expr {
        &self.parameter
    }
    pub fn certificate(&self) -> &PolynomialExponentialCertificate {
        &self.certificate
    }
    pub fn value(&self) -> &Expr {
        &self.value
    }
    pub fn into_value(self) -> Expr {
        self.value
    }
}

/// Cumulative phase limits supplement the caller's cancellation and budget.
struct Work<'a, M> {
    meter: &'a mut M,
    compute: u64,
    memory: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            compute: 0,
            memory: 0,
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
        self.allocate(1, 64 + bits(value).div_ceil(8) * 2)?;
        Ok(value.clone())
    }
    fn preflight(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        self.checkpoint()?;
        if bits(a).saturating_add(bits(b)).saturating_add(2) > MAX_WORK_BITS {
            return Err(ExponentialSummationError::Refused(
                "intermediate scalar growth bound",
            ));
        }
        Ok(())
    }
    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_add(b, self)?)
    }
    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_sub(b, self)?)
    }
    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_mul(b, self)?)
    }
    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        if b.is_zero() {
            return Err(ExponentialSummationError::Refused("division by zero"));
        }
        Ok(a.metered_div(b, self)?)
    }
    fn pow(&mut self, value: &BigRational, exponent: i32) -> Result<BigRational> {
        self.checkpoint()?;
        if value.is_zero() && exponent < 0 {
            return Err(ExponentialSummationError::Refused("undefined zero power"));
        }
        if exponent.unsigned_abs() > 1024
            || bits(value).saturating_mul(u64::from(exponent.unsigned_abs()).max(1))
                > MAX_WORK_BITS / 2
        {
            return Err(ExponentialSummationError::Refused(
                "scalar power growth bound",
            ));
        }
        if exponent == 0 {
            return Ok(BigRational::one());
        }
        Ok(value.metered_pow(exponent, self)?)
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
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
        self.compute = compute;
        self.memory = memory;
        Ok(())
    }
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}
fn integer(value: usize) -> BigRational {
    BigRational::from_integer(BigInt::from(value as u64))
}
fn number(expression: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Option<BigRational>> {
    match expression {
        Expr::Integer(value) => {
            work.allocate(1, 64 + value.bits().div_ceil(8))?;
            Ok(Some(BigRational::from_integer(value.clone())))
        }
        Expr::Rational(value) => Ok(Some(work.copy(value)?)),
        _ => Ok(None),
    }
}

fn copy_parameter(parameter: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Expr> {
    let payload = match parameter {
        Expr::Integer(value) => value.bits().div_ceil(8),
        Expr::Rational(value) => bits(value).div_ceil(8) * 2,
        Expr::Sym(symbol) => symbol.name.len() as u64,
        Expr::Const(Constant::Pi | Constant::E | Constant::I) => 0,
        _ => {
            return Err(ExponentialSummationError::Unsupported(
                "nonconstant exponential bases",
            ));
        }
    };
    work.allocate(1, payload + 128)?;
    Ok(parameter.clone())
}
fn trim(poly: &mut Coefficients) {
    while poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        poly.pop();
    }
}
fn factorial_of(expression: &Expr, index: &Symbol) -> bool {
    matches!(expression, Expr::Function(name, args)
        if name == "factorial" && args.len() == 1
            && matches!(&args[0], Expr::Sym(symbol) if symbol == index))
}

/// Inspect every borrowed source child before normalization, including those
/// under held zero products/powers. The return value is only a family probe.
fn admit(source: &Expr, index: &Symbol, work: &mut Work<'_, impl BudgetMeter>) -> Result<bool> {
    let mut pending = vec![(source, 0usize)];
    let mut visited = 0usize;
    let mut seen = false;
    // Only impose this lane's scalar/domain conditions when its factorial
    // family occurs. No expressions are evaluated during the family probe.
    let mut refusal = (index.name.len() > 4096).then_some("index name bound");
    while let Some((node, depth)) = pending.pop() {
        work.checkpoint()?;
        if visited == MAX_NODES || depth > MAX_DEPTH {
            return Err(ExponentialSummationError::Refused("source shape bound"));
        }
        visited += 1;
        work.allocate(1, 128)?;
        let children = match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_INPUT_BITS {
                    refusal = Some("input scalar bound");
                } else {
                    work.allocate(1, value.bits().div_ceil(8))?;
                }
                None
            }
            Expr::Rational(value) => {
                if bits(value) > MAX_INPUT_BITS {
                    refusal = Some("input scalar bound");
                } else {
                    work.allocate(1, bits(value).div_ceil(8) * 2)?;
                }
                None
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > 4096 {
                    refusal = Some("symbol name bound");
                } else {
                    work.allocate(1, symbol.name.len() as u64)?;
                }
                None
            }
            Expr::Add(children) | Expr::Mul(children) => Some(children),
            Expr::Function(name, children) => {
                if name.len() > 4096 {
                    refusal = Some("function name bound");
                } else {
                    work.allocate(1, name.len() as u64)?;
                }
                seen |= factorial_of(node, index);
                if name == "factorial"
                    && children.len() == 1
                    && matches!(&children[0], Expr::Integer(value) if value.is_negative())
                {
                    refusal = Some("original factorial pole");
                }
                Some(children)
            }
            Expr::Pow(base, exponent) => {
                if base.is_zero()
                    && matches!(exponent.as_ref(), Expr::Integer(value) if value.is_negative())
                {
                    refusal = Some("original zero denominator");
                }
                if visited + pending.len() + 2 > MAX_NODES {
                    return Err(ExponentialSummationError::Refused("source shape bound"));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
                None
            }
            Expr::Const(Constant::Pi | Constant::E | Constant::I) => None,
            Expr::Const(_) => {
                refusal = Some("nonfinite original summand");
                None
            }
        };
        if let Some(children) = children {
            if children.len() > MAX_NODES.saturating_sub(visited + pending.len()) {
                return Err(ExponentialSummationError::Refused("source shape bound"));
            }
            pending.extend(children.iter().map(|child| (child, depth + 1)));
        }
    }
    if seen && let Some(reason) = refusal {
        return Err(ExponentialSummationError::Refused(reason));
    }
    Ok(seen)
}

fn add_poly(
    a: &[BigRational],
    b: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    let size = a.len().max(b.len());
    work.allocate(size, (size * 64) as u64)?;
    let zero = BigRational::zero();
    let mut result = Vec::with_capacity(size);
    for i in 0..size {
        result.push(work.add(a.get(i).unwrap_or(&zero), b.get(i).unwrap_or(&zero))?);
    }
    trim(&mut result);
    Ok(result)
}

fn mul_poly(
    a: &[BigRational],
    b: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    if a.len() + b.len() - 2 > MAX_DEGREE {
        return Err(ExponentialSummationError::Refused(
            "polynomial degree bound",
        ));
    }
    let size = a.len() + b.len() - 1;
    work.allocate(size, (size * 64) as u64)?;
    let mut result = vec![BigRational::zero(); size];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let term = work.mul(x, y)?;
            result[i + j] = work.add(&result[i + j], &term)?;
        }
    }
    trim(&mut result);
    Ok(result)
}

fn polynomial(
    source: &Expr,
    index: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    work.allocate(1, 64)?;
    match source {
        Expr::Integer(_) | Expr::Rational(_) => {
            Ok(vec![number(source, work)?.expect("numeric atom")])
        }
        Expr::Sym(symbol) if symbol == index => Ok(vec![BigRational::zero(), BigRational::one()]),
        Expr::Add(children) | Expr::Mul(children) => {
            let multiply = matches!(source, Expr::Mul(_));
            let mut result = vec![if multiply {
                BigRational::one()
            } else {
                BigRational::zero()
            }];
            for child in children {
                let next = polynomial(child, index, work)?;
                result = if multiply {
                    mul_poly(&result, &next, work)?
                } else {
                    add_poly(&result, &next, work)?
                };
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            // Do not erase an unadmitted base under a held exponent zero.
            let mut base = polynomial(base, index, work)?;
            let power = match exponent.as_ref() {
                Expr::Integer(value) => value.to_i64().and_then(|v| i32::try_from(v).ok()),
                _ => None,
            }
            .ok_or(ExponentialSummationError::Unsupported(
                "noninteger polynomial powers",
            ))?;
            if base.len() == 1 {
                return Ok(vec![work.pow(&base[0], power)?]);
            }
            if power < 0 {
                return Err(ExponentialSummationError::Unsupported(
                    "rational weights with poles",
                ));
            }
            if (base.len() - 1).saturating_mul(power as usize) > MAX_DEGREE {
                return Err(ExponentialSummationError::Refused(
                    "polynomial degree bound",
                ));
            }
            let mut result = vec![BigRational::one()];
            let mut remaining = power as u32;
            while remaining != 0 {
                if remaining & 1 != 0 {
                    result = mul_poly(&result, &base, work)?;
                }
                remaining >>= 1;
                if remaining != 0 {
                    base = mul_poly(&base, &base, work)?;
                }
            }
            Ok(result)
        }
        _ => Err(ExponentialSummationError::Unsupported(
            "coefficients outside QQ or nonpolynomial weights",
        )),
    }
}

fn normalize(
    source: &Expr,
    index: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Option<(Coefficients, Expr)>> {
    if !admit(source, index, work)? {
        return Ok(None);
    }
    let mut pending = vec![source];
    let mut factorials = 0;
    let mut parameter = None;
    let mut weight = vec![BigRational::one()];
    while let Some(part) = pending.pop() {
        work.allocate(1, 64)?;
        if let Expr::Mul(children) = part {
            pending.extend(children.iter());
            continue;
        }
        if let Expr::Pow(base, exponent) = part {
            if factorial_of(base, index) && exponent.as_ref() == &Expr::from_i64(-1) {
                factorials += 1;
                continue;
            }
            if matches!(exponent.as_ref(), Expr::Sym(symbol) if symbol == index) {
                if parameter.is_some() {
                    return Err(ExponentialSummationError::Unsupported(
                        "multiple exponential bases",
                    ));
                }
                match base.as_ref() {
                    Expr::Integer(_)
                    | Expr::Rational(_)
                    | Expr::Const(Constant::Pi | Constant::E | Constant::I) => {}
                    Expr::Sym(symbol) if symbol != index => {}
                    _ => {
                        return Err(ExponentialSummationError::Unsupported(
                            "nonconstant exponential bases",
                        ));
                    }
                }
                parameter = Some(copy_parameter(base, work)?);
                continue;
            }
        }
        let next = polynomial(part, index, work)?;
        weight = mul_poly(&weight, &next, work)?;
    }
    if factorials != 1 {
        return Err(ExponentialSummationError::Unsupported(
            "a single reciprocal index factorial",
        ));
    }
    Ok(Some((
        weight,
        parameter.unwrap_or_else(|| Expr::from_i64(1)),
    )))
}

fn lower_endpoint(lower: &Expr) -> Result<usize> {
    match lower {
        Expr::Integer(value) if value.is_negative() => Err(ExponentialSummationError::Refused(
            "negative factorial endpoint",
        )),
        Expr::Integer(value) => value
            .to_i64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value <= MAX_LOWER)
            .ok_or(ExponentialSummationError::Refused("finite-head bound")),
        _ => Err(ExponentialSummationError::Unsupported(
            "a noninteger or symbolic lower endpoint",
        )),
    }
}

fn evaluate(
    poly: &[BigRational],
    point: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let mut result = BigRational::zero();
    for coefficient in poly.iter().rev() {
        let product = work.mul(&result, point)?;
        result = work.add(&product, coefficient)?;
    }
    Ok(result)
}

/// Newton's forward-difference formula is the candidate generator only.
fn generate(
    poly: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<PolynomialExponentialCertificate> {
    work.allocate(poly.len() * 2, (poly.len() * 128) as u64)?;
    let mut differences = (0..poly.len())
        .map(|point| evaluate(poly, &integer(point), work))
        .collect::<Result<Vec<_>>>()?;
    let mut factorial = BigRational::one();
    let mut coefficients = Vec::with_capacity(poly.len());
    for order in 0..poly.len() {
        coefficients.push(work.div(&differences[0], &factorial)?);
        for i in 0..differences.len() - 1 {
            differences[i] = work.sub(&differences[i + 1], &differences[i])?;
        }
        differences.pop();
        factorial = work.mul(&factorial, &integer(order + 1))?;
    }
    trim(&mut coefficients);
    Ok(PolynomialExponentialCertificate {
        falling_coefficients: coefficients,
    })
}

/// Expand the products k*(k-1)*... directly and compare every coefficient.
/// Neither forward differences nor evaluations at sample points occur here.
fn check_identity(
    poly: &[BigRational],
    certificate: &PolynomialExponentialCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let coefficients = &certificate.falling_coefficients;
    if coefficients.is_empty()
        || coefficients.len() > MAX_DEGREE + 1
        || (coefficients.len() > 1 && coefficients.last().is_some_and(Zero::is_zero))
        || coefficients.iter().any(|value| bits(value) > MAX_WORK_BITS)
    {
        return Err(ExponentialSummationError::InvalidCertificate);
    }
    work.allocate(MAX_DEGREE + 1, ((MAX_DEGREE + 1) * 64) as u64)?;
    let mut reconstructed = vec![BigRational::zero(); coefficients.len()];
    let mut falling = vec![BigRational::one()];
    for (order, coefficient) in coefficients.iter().enumerate() {
        for (power, factor) in falling.iter().enumerate() {
            let term = work.mul(coefficient, factor)?;
            reconstructed[power] = work.add(&reconstructed[power], &term)?;
        }
        if order + 1 < coefficients.len() {
            work.allocate(falling.len() + 1, ((falling.len() + 1) * 64) as u64)?;
            let mut next = vec![BigRational::zero(); falling.len() + 1];
            for (power, factor) in falling.iter().enumerate() {
                let offset = work.mul(&integer(order), factor)?;
                next[power] = work.sub(&next[power], &offset)?;
                next[power + 1] = work.add(&next[power + 1], factor)?;
            }
            falling = next;
        }
    }
    trim(&mut reconstructed);
    if reconstructed != poly {
        return Err(ExponentialSummationError::InvalidCertificate);
    }
    Ok(())
}

fn sum(mut terms: Vec<Expr>) -> Expr {
    terms.retain(|term| !term.is_zero());
    match terms.len() {
        0 => Expr::from_i64(0),
        1 => terms.pop().expect("one term"),
        _ => Expr::Add(terms),
    }
}
fn product(a: Expr, b: Expr) -> Expr {
    if a.is_zero() || b.is_zero() {
        Expr::from_i64(0)
    } else if a.is_one() {
        b
    } else if b.is_one() {
        a
    } else {
        Expr::Mul(vec![a, b])
    }
}

fn render(
    poly: &[BigRational],
    parameter: &Expr,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    if let Some(point) = number(parameter, work)? {
        return Ok(rational_expr(evaluate(poly, &point, work)?));
    }
    work.allocate(poly.len(), (poly.len() * 384) as u64)?;
    let mut terms = Vec::with_capacity(poly.len());
    for (power, coefficient) in poly.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let factor = if power == 0 {
            Expr::from_i64(1)
        } else if power == 1 {
            copy_parameter(parameter, work)?
        } else {
            Expr::Pow(
                Arc::new(copy_parameter(parameter, work)?),
                Arc::new(Expr::from_i64(power as i64)),
            )
        };
        terms.push(product(rational_expr(work.copy(coefficient)?), factor));
    }
    Ok(sum(terms))
}

fn value(
    poly: &[BigRational],
    parameter: &Expr,
    lower: usize,
    certificate: &PolynomialExponentialCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    // After complete source and coefficient admission, z=0 has only its
    // k=0 term. The discrete exponential series uses the exact 0**0=1
    // convention; every strictly positive index contributes zero.
    if parameter.is_zero() {
        return Ok(if lower == 0 {
            rational_expr(work.copy(&poly[0])?)
        } else {
            Expr::from_i64(0)
        });
    }
    let transformed = render(&certificate.falling_coefficients, parameter, work)?;
    let argument = copy_parameter(parameter, work)?;
    let exponential = eval_function("exp", std::slice::from_ref(&argument))
        .unwrap_or_else(|| Expr::Function("exp".into(), vec![argument]));
    work.allocate(lower.max(1), (lower.max(1) * 64) as u64)?;
    let mut head = Vec::with_capacity(lower);
    let mut factorial = BigRational::one();
    for point in 0..lower {
        let weight = evaluate(poly, &integer(point), work)?;
        head.push(work.div(&weight, &factorial)?);
        factorial = work.mul(&factorial, &integer(point + 1))?;
    }
    let head = render(&head, parameter, work)?;
    let complete = product(exponential, transformed);
    if let (Some(total), Some(prefix)) = (number(&complete, work)?, number(&head, work)?) {
        return Ok(rational_expr(work.sub(&total, &prefix)?));
    }
    let negative_head = if let Some(prefix) = number(&head, work)? {
        rational_expr(work.sub(&BigRational::zero(), &prefix)?)
    } else {
        product(Expr::from_i64(-1), head)
    };
    work.allocate(1, 256)?;
    Ok(sum(vec![complete, negative_head]))
}

/// Recheck original syntax, coefficient identity and finite lower endpoint.
pub fn verify_polynomial_exponential_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
    certificate: PolynomialExponentialCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialExponentialSum> {
    let mut work = Work::new(verifier);
    work.checkpoint()?;
    if !matches!(upper, Expr::Const(Constant::Infinity)) {
        return Err(ExponentialSummationError::Unsupported(
            "a finite upper endpoint",
        ));
    }
    let lower_index = lower_endpoint(lower)?;
    let (poly, parameter) = normalize(source, index, &mut work)?.ok_or(
        ExponentialSummationError::Unsupported("summands without the reciprocal index factorial"),
    )?;
    check_identity(&poly, &certificate, &mut work)?;
    let value = value(&poly, &parameter, lower_index, &certificate, &mut work)?;
    // This second bounded walk charges the complete source copy, including
    // scalar payloads and symbol names, before it becomes owned receipt data.
    admit(source, index, &mut work)?;
    work.allocate(1, index.name.len() as u64 + 128)?;
    work.checkpoint()?;
    Ok(VerifiedPolynomialExponentialSum {
        source: source.clone(),
        index: index.clone(),
        lower: lower.clone(),
        parameter,
        certificate,
        value,
    })
}

pub fn metered_polynomial_exponential_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Option<VerifiedPolynomialExponentialSum>> {
    let mut work = Work::new(generator);
    work.checkpoint()?;
    if !matches!(upper, Expr::Const(Constant::Infinity)) {
        return Ok(None);
    }
    let Some((poly, _parameter)) = normalize(source, index, &mut work)? else {
        return Ok(None);
    };
    lower_endpoint(lower)?;
    let certificate = generate(&poly, &mut work)?;
    work.checkpoint()?;
    let checked =
        verify_polynomial_exponential_sum(source, index, lower, upper, certificate, verifier)?;
    work.checkpoint()?;
    verifier.checkpoint()?;
    Ok(Some(checked))
}

pub fn polynomial_exponential_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
) -> Result<Option<VerifiedPolynomialExponentialSum>> {
    metered_polynomial_exponential_sum(source, index, lower, upper, &mut Unbounded, &mut Unbounded)
}
