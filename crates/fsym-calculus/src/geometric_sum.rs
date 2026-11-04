//! Exact polynomial-geometric sums, with a separate antidifference checker.
//!
//! For rational `q != 0` and `p in Q[k]`, generation solves
//! `q*P(k+1)-P(k)=p(k)`. Acceptance independently expands the shift, checks
//! every coefficient, and evaluates the endpoints. It never calls the
//! triangular generator. Infinite sums additionally require `|q| < 1`.
//!
//! This synchronous, in-process receipt is not a portable proof-kernel claim.
//! Each phase has cumulative limits of twenty million compute units and
//! 128 MiB of logical allocation, independently of the caller's meter.
//! Original syntax is bounded before normalization; children of zero products
//! and zero powers are still checked. The supported polynomial degree is 32,
//! input scalars have at most 4096 bits, and work scalars at most 65536 bits.

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;
use thiserror::Error;

const MAX_DEGREE: usize = 32;
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 65_536;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum GeometricSummationError {
    #[error("polynomial-geometric summation does not support {0}")]
    Unsupported(&'static str),
    #[error("polynomial-geometric summation refused: {0}")]
    Refused(&'static str),
    #[error("polynomial-geometric antidifference identity does not hold")]
    InvalidCertificate,
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for GeometricSummationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, GeometricSummationError>;
type Coefficients = Vec<BigRational>;

/// Untrusted ascending coefficients of the proposed polynomial `P`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialGeometricCertificate {
    pub antidifference: Coefficients,
}

/// Bound to the complete original summand, typed index and exact endpoints.
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialGeometricSum {
    source: Expr,
    index: Symbol,
    lower: Expr,
    upper: Expr,
    certificate: PolynomialGeometricCertificate,
    value: Expr,
}

impl VerifiedPolynomialGeometricSum {
    pub fn source(&self) -> &Expr {
        &self.source
    }
    pub fn index(&self) -> &Symbol {
        &self.index
    }
    pub fn lower(&self) -> &Expr {
        &self.lower
    }
    pub fn upper(&self) -> &Expr {
        &self.upper
    }
    pub fn certificate(&self) -> &PolynomialGeometricCertificate {
        &self.certificate
    }
    pub fn value(&self) -> &Expr {
        &self.value
    }
    pub fn into_value(self) -> Expr {
        self.value
    }
}

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
            return Err(GeometricSummationError::Refused(
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
            return Err(GeometricSummationError::Refused("division by zero"));
        }
        Ok(a.metered_div(b, self)?)
    }
    fn pow(&mut self, value: &BigRational, exponent: i32) -> Result<BigRational> {
        self.checkpoint()?;
        if value.is_zero() && exponent <= 0 {
            return Err(GeometricSummationError::Refused("undefined zero power"));
        }
        if bits(value).saturating_mul(u64::from(exponent.unsigned_abs()).max(1)) > MAX_WORK_BITS / 2
        {
            return Err(GeometricSummationError::Refused(
                "endpoint or power growth bound",
            ));
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
fn integer(value: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(value))
}
fn number(expression: &Expr) -> Option<BigRational> {
    match expression {
        Expr::Integer(value) => Some(BigRational::from_integer(value.clone())),
        Expr::Rational(value) => Some(value.clone()),
        _ => None,
    }
}
fn trim(poly: &mut Coefficients) {
    while poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        poly.pop();
    }
}

/// Walk original children even when later normalization makes them vanish.
fn admit(
    expression: &Expr,
    depth: usize,
    nodes: &mut usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.checkpoint()?;
    *nodes += 1;
    if depth > MAX_DEPTH || *nodes > MAX_NODES {
        return Err(GeometricSummationError::Refused("source shape bound"));
    }
    work.allocate(1, 128)?;
    match expression {
        Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => {
            work.allocate(1, value.bits().div_ceil(8))?;
        }
        Expr::Rational(value) if bits(value) <= MAX_INPUT_BITS => {
            work.allocate(1, bits(value).div_ceil(8) * 2)?;
        }
        Expr::Integer(_) | Expr::Rational(_) => {
            return Err(GeometricSummationError::Refused("input scalar bound"));
        }
        Expr::Sym(symbol) if symbol.name.len() <= 4096 => {
            work.allocate(1, symbol.name.len() as u64)?;
        }
        Expr::Add(children) | Expr::Mul(children) => {
            if children.len() > MAX_NODES {
                return Err(GeometricSummationError::Refused("source arity bound"));
            }
            for child in children {
                admit(child, depth + 1, nodes, work)?;
            }
        }
        Expr::Pow(base, exponent) => {
            admit(base, depth + 1, nodes, work)?;
            admit(exponent, depth + 1, nodes, work)?;
        }
        _ => {
            return Err(GeometricSummationError::Unsupported(
                "the original summand syntax",
            ));
        }
    }
    Ok(())
}

fn add_poly(
    a: &[BigRational],
    b: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    work.allocate(a.len().max(b.len()), (a.len().max(b.len()) * 64) as u64)?;
    let zero = BigRational::zero();
    let mut result = Vec::with_capacity(a.len().max(b.len()));
    for i in 0..a.len().max(b.len()) {
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
        return Err(GeometricSummationError::Refused("polynomial degree bound"));
    }
    let size = a.len() + b.len() - 1;
    work.allocate(size, (size * 64) as u64)?;
    let mut result = vec![BigRational::zero(); size];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let value = work.mul(x, y)?;
            result[i + j] = work.add(&result[i + j], &value)?;
        }
    }
    trim(&mut result);
    Ok(result)
}

fn polynomial(
    expression: &Expr,
    index: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    work.allocate(1, 64)?;
    match expression {
        Expr::Integer(_) | Expr::Rational(_) => {
            Ok(vec![number(expression).expect("admitted scalar")])
        }
        Expr::Sym(symbol) if symbol == index => Ok(vec![BigRational::zero(), BigRational::one()]),
        Expr::Add(children) | Expr::Mul(children) => {
            let multiply = matches!(expression, Expr::Mul(_));
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
            // Parse the base first, including held children under exponent zero.
            let mut base = polynomial(base, index, work)?;
            let power = match exponent.as_ref() {
                Expr::Integer(value) => value.to_i64().and_then(|v| i32::try_from(v).ok()),
                _ => None,
            }
            .ok_or(GeometricSummationError::Unsupported(
                "noninteger polynomial powers",
            ))?;
            if base.len() == 1 {
                return Ok(vec![work.pow(&base[0], power)?]);
            }
            if power < 0 {
                return Err(GeometricSummationError::Unsupported(
                    "rational weights with poles",
                ));
            }
            if (base.len() - 1).saturating_mul(power as usize) > MAX_DEGREE {
                return Err(GeometricSummationError::Refused("polynomial degree bound"));
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
        _ => Err(GeometricSummationError::Unsupported(
            "coefficients outside Q or nonpolynomial weights",
        )),
    }
}

fn factors<'a>(
    source: &'a Expr,
    depth: usize,
    nodes: &mut usize,
    output: &mut Vec<&'a Expr>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.checkpoint()?;
    *nodes += 1;
    work.charge(Dimension::ComputeSteps, 1)?;
    if depth > MAX_DEPTH || *nodes > MAX_NODES {
        return Err(GeometricSummationError::Refused("source product bound"));
    }
    if let Expr::Mul(children) = source {
        if children.len() > MAX_NODES {
            return Err(GeometricSummationError::Refused("source arity bound"));
        }
        for child in children {
            factors(child, depth + 1, nodes, output, work)?;
        }
    } else {
        work.allocate(1, 8)?;
        output.push(source);
    }
    Ok(())
}

fn normalize(
    source: &Expr,
    index: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Option<(Coefficients, BigRational)>> {
    let mut parts = Vec::new();
    factors(source, 0, &mut 0, &mut parts, work)?;
    // This probe is deliberately cheap and cannot expand, simplify or drop a child.
    if !parts.iter().any(|part| {
        matches!(part, Expr::Pow(base, exponent)
        if matches!(base.as_ref(), Expr::Integer(_) | Expr::Rational(_))
        && !matches!(exponent.as_ref(), Expr::Integer(_)))
    }) {
        return Ok(None);
    }
    admit(source, 0, &mut 0, work)?;
    if index.name.len() > 4096 {
        return Err(GeometricSummationError::Refused("index name bound"));
    }
    let mut ratio = BigRational::one();
    let mut weight = vec![BigRational::one()];
    let mut seen = false;
    for part in parts {
        if let Expr::Pow(base, exponent) = part
            && let Some(base) = number(base)
            && !matches!(exponent.as_ref(), Expr::Integer(_))
        {
            if base.is_zero() {
                return Err(GeometricSummationError::Unsupported("zero geometric bases"));
            }
            let affine = polynomial(exponent, index, work)?;
            if affine.len() != 2 || affine.iter().any(|c| !c.is_integer()) {
                return Err(GeometricSummationError::Unsupported(
                    "nonaffine integer exponents",
                ));
            }
            let slope = affine[1]
                .numer()
                .to_i64()
                .and_then(|v| i32::try_from(v).ok())
                .ok_or(GeometricSummationError::Refused("geometric slope bound"))?;
            let offset = affine[0]
                .numer()
                .to_i64()
                .and_then(|v| i32::try_from(v).ok())
                .ok_or(GeometricSummationError::Refused("geometric offset bound"))?;
            let next_ratio = work.pow(&base, slope)?;
            ratio = work.mul(&ratio, &next_ratio)?;
            let constant = work.pow(&base, offset)?;
            weight = mul_poly(&weight, &[constant], work)?;
            seen = true;
        } else {
            let next = polynomial(part, index, work)?;
            weight = mul_poly(&weight, &next, work)?;
        }
    }
    Ok(seen.then_some((weight, ratio)))
}

fn binomial(n: usize, r: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<BigRational> {
    let mut result = BigRational::one();
    for i in 0..r {
        result = work.mul(&result, &integer((n - i) as i64))?;
        result = work.div(&result, &integer((i + 1) as i64))?;
    }
    Ok(result)
}

fn generate(
    p: &[BigRational],
    q: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<PolynomialGeometricCertificate> {
    let unit = q.is_one();
    let size = p.len() + usize::from(unit);
    work.allocate(size, (size * 64) as u64)?;
    let mut candidate = vec![BigRational::zero(); size];
    let diagonal = work.sub(q, &BigRational::one())?;
    for j in (0..p.len()).rev() {
        let column = j + usize::from(unit);
        let mut residual = work.copy(&p[j])?;
        for (i, value) in candidate.iter().enumerate().skip(column + 1) {
            let choose = binomial(i, j, work)?;
            let coefficient = work.mul(q, &choose)?;
            let term = work.mul(&coefficient, value)?;
            residual = work.sub(&residual, &term)?;
        }
        candidate[column] = if unit {
            work.div(&residual, &integer(column as i64))?
        } else {
            work.div(&residual, &diagonal)?
        };
    }
    trim(&mut candidate);
    Ok(PolynomialGeometricCertificate {
        antidifference: candidate,
    })
}

/// Independent shift expansion uses Pascal rows, not coefficient elimination.
fn check_identity(
    p: &[BigRational],
    q: &BigRational,
    certificate: &PolynomialGeometricCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let candidate = &certificate.antidifference;
    if candidate.is_empty()
        || candidate.len() > MAX_DEGREE + 2
        || (candidate.len() > 1 && candidate.last().is_some_and(Zero::is_zero))
        || candidate.iter().any(|c| bits(c) > MAX_WORK_BITS / 2)
    {
        return Err(GeometricSummationError::InvalidCertificate);
    }
    work.allocate(candidate.len() * 3, (candidate.len() * 192) as u64)?;
    let mut shifted = vec![BigRational::zero(); candidate.len()];
    let mut row = vec![BigRational::one()];
    for coefficient in candidate {
        for (power, choose) in row.iter().enumerate() {
            let term = work.mul(coefficient, choose)?;
            shifted[power] = work.add(&shifted[power], &term)?;
        }
        work.allocate(row.len() + 1, ((row.len() + 1) * 64) as u64)?;
        let mut next = vec![BigRational::one(); row.len() + 1];
        for j in 1..row.len() {
            next[j] = work.add(&row[j - 1], &row[j])?;
        }
        row = next;
    }
    let zero = BigRational::zero();
    for j in 0..candidate.len().max(p.len()) {
        let shifted = work.mul(q, shifted.get(j).unwrap_or(&zero))?;
        let actual = work.sub(&shifted, candidate.get(j).unwrap_or(&zero))?;
        if &actual != p.get(j).unwrap_or(&zero) {
            return Err(GeometricSummationError::InvalidCertificate);
        }
    }
    Ok(())
}

fn admit_bound(
    bound: &Expr,
    depth: usize,
    nodes: &mut usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.checkpoint()?;
    *nodes += 1;
    if depth > MAX_DEPTH || *nodes > MAX_NODES {
        return Err(GeometricSummationError::Refused("endpoint syntax bound"));
    }
    work.allocate(1, 128)?;
    match bound {
        Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => {
            work.allocate(1, value.bits().div_ceil(8))?;
        }
        Expr::Sym(symbol) if symbol.name.len() <= 4096 => {
            work.allocate(1, symbol.name.len() as u64)?;
        }
        Expr::Add(children) | Expr::Mul(children) => {
            if children.len() > MAX_NODES {
                return Err(GeometricSummationError::Refused("endpoint arity bound"));
            }
            for child in children {
                admit_bound(child, depth + 1, nodes, work)?;
            }
        }
        _ => {
            return Err(GeometricSummationError::Unsupported(
                "endpoints other than integers or formal integer-bound polynomials",
            ));
        }
    }
    Ok(())
}

/// Copy only already-admitted syntax, prepaying the actual scalar/name bytes.
fn copy_expression(expression: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Expr> {
    work.allocate(1, 128)?;
    Ok(match expression {
        Expr::Integer(value) => {
            work.allocate(1, value.bits().max(1).div_ceil(8))?;
            Expr::Integer(value.clone())
        }
        Expr::Rational(value) => rational_expr(work.copy(value)?),
        Expr::Sym(symbol) => {
            work.allocate(1, symbol.name.len() as u64)?;
            Expr::Sym(symbol.clone())
        }
        Expr::Add(children) | Expr::Mul(children) => {
            work.allocate(1, (children.len() * std::mem::size_of::<Expr>()) as u64)?;
            let copied = children
                .iter()
                .map(|child| copy_expression(child, work))
                .collect::<Result<Vec<_>>>()?;
            if matches!(expression, Expr::Add(_)) {
                Expr::Add(copied)
            } else {
                Expr::Mul(copied)
            }
        }
        Expr::Pow(base, exponent) => Expr::Pow(
            Arc::new(copy_expression(base, work)?),
            Arc::new(copy_expression(exponent, work)?),
        ),
        Expr::Const(Constant::Infinity) => Expr::Const(Constant::Infinity),
        _ => {
            return Err(GeometricSummationError::Unsupported(
                "expression copying outside admitted syntax",
            ));
        }
    })
}

fn numeric_endpoint(
    poly: &[BigRational],
    q: &BigRational,
    endpoint: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let power = endpoint
        .to_i64()
        .and_then(|v| i32::try_from(v).ok())
        .ok_or(GeometricSummationError::Refused("numeric endpoint bound"))?;
    let point = BigRational::from_integer(endpoint.clone());
    let mut value = BigRational::zero();
    for coefficient in poly.iter().rev() {
        value = work.mul(&value, &point)?;
        value = work.add(&value, coefficient)?;
    }
    let exponential = work.pow(q, power)?;
    work.mul(&value, &exponential)
}

fn symbolic_endpoint(
    poly: &[BigRational],
    q: &BigRational,
    endpoint: &Expr,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    // Retain a compact Horner form; every copied endpoint is charged before use.
    work.allocate(
        poly.len() * 4,
        (poly.len() * 4 * std::mem::size_of::<Expr>()) as u64,
    )?;
    let mut value = Expr::from_i64(0);
    for coefficient in poly.iter().rev() {
        let coefficient = rational_expr(work.copy(coefficient)?);
        value = if value.is_zero() {
            coefficient
        } else {
            Expr::Add(vec![
                Expr::Mul(vec![value, copy_expression(endpoint, work)?]),
                coefficient,
            ])
        };
    }
    if value.is_zero() || q.is_one() {
        return Ok(value);
    }
    Ok(Expr::Mul(vec![
        value,
        Expr::Pow(
            Arc::new(rational_expr(work.copy(q)?)),
            Arc::new(copy_expression(endpoint, work)?),
        ),
    ]))
}

fn endpoint(
    poly: &[BigRational],
    q: &BigRational,
    endpoint: &Expr,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    if let Expr::Integer(point) = endpoint {
        Ok(rational_expr(numeric_endpoint(poly, q, point, work)?))
    } else {
        symbolic_endpoint(poly, q, endpoint, work)
    }
}

/// Verify against original syntax and return only a complete checked result.
pub fn verify_polynomial_geometric_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
    certificate: PolynomialGeometricCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialGeometricSum> {
    let mut work = Work::new(verifier);
    let (p, q) = normalize(source, index, &mut work)?.ok_or(
        GeometricSummationError::Unsupported("summands without a rational geometric factor"),
    )?;
    check_identity(&p, &q, &certificate, &mut work)?;
    let mut nodes = 0;
    admit_bound(lower, 0, &mut nodes, &mut work)?;
    let infinite = matches!(upper, Expr::Const(Constant::Infinity));
    if infinite {
        if !matches!(lower, Expr::Integer(_))
            || (!p.iter().all(Zero::is_zero) && q.abs() >= BigRational::one())
        {
            return Err(GeometricSummationError::Unsupported(
                "a divergent or nonnumeric-lower-bound infinite series",
            ));
        }
    } else {
        admit_bound(upper, 0, &mut nodes, &mut work)?;
    }
    // For q=1 a valid antidifference may contain an arbitrary constant. The
    // limit of that constant is not zero; a zero source has sum zero directly.
    let left = if infinite && p.iter().all(Zero::is_zero) {
        Expr::from_i64(0)
    } else {
        endpoint(&certificate.antidifference, &q, lower, &mut work)?
    };
    let right = if infinite {
        Expr::from_i64(0)
    } else {
        let next = if let Expr::Integer(value) = upper {
            let number = work.add(
                &BigRational::from_integer(value.clone()),
                &BigRational::one(),
            )?;
            Expr::Integer(number.to_integer())
        } else {
            Expr::Add(vec![copy_expression(upper, &mut work)?, Expr::from_i64(1)])
        };
        endpoint(&certificate.antidifference, &q, &next, &mut work)?
    };
    let value = match (number(&right), number(&left)) {
        (Some(a), Some(b)) => rational_expr(work.sub(&a, &b)?),
        _ => Expr::Add(vec![right, Expr::Mul(vec![Expr::from_i64(-1), left])]),
    };
    let source = copy_expression(source, &mut work)?;
    let lower = copy_expression(lower, &mut work)?;
    let upper = copy_expression(upper, &mut work)?;
    work.allocate(
        1,
        index.name.len() as u64 + std::mem::size_of::<Symbol>() as u64,
    )?;
    work.checkpoint()?;
    Ok(VerifiedPolynomialGeometricSum {
        source,
        index: index.clone(),
        lower,
        upper,
        certificate,
        value,
    })
}

pub fn metered_polynomial_geometric_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Option<VerifiedPolynomialGeometricSum>> {
    let mut work = Work::new(generator);
    let Some((p, q)) = normalize(source, index, &mut work)? else {
        return Ok(None);
    };
    let certificate = generate(&p, &q, &mut work)?;
    work.checkpoint()?;
    let checked =
        verify_polynomial_geometric_sum(source, index, lower, upper, certificate, verifier)?;
    work.checkpoint()?;
    verifier.checkpoint()?;
    Ok(Some(checked))
}

pub fn polynomial_geometric_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
) -> Result<Option<VerifiedPolynomialGeometricSum>> {
    metered_polynomial_geometric_sum(source, index, lower, upper, &mut Unbounded, &mut Unbounded)
}
