//! Checked causal inverse Laplace transforms of proper rational functions.
//!
//! A generator factors the original denominator and solves for polynomial
//! exponential, trigonometric and hyperbolic blocks. A separate reference
//! lane computes each block's transform from closed binomial coefficients,
//! then checks the complete numerator identity against the original source.
//! It never searches for residues and cannot accept an omitted pole.
//!
//! The returned function applies for t >= 0; a public causal surface must
//! attach Heaviside(t). The abscissa is a sufficient rational half-plane
//! bound, not a claimed optimal convergence boundary. All original held
//! rational children are admitted before cancellation or zero shortcuts.
//! Receipts are independently checked in process, not portable proof objects.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::{RationalArithmeticError, metered_mul, metered_sqrt_floor};
use fsym_core::elementary::{eval_pow, rational_expr};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{PolyError, UnivariatePoly, metered_complete_factorization};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;
use thiserror::Error;

const MAX_DEGREE: usize = 32;
const MAX_INTERNAL_DEGREE: usize = 96;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_BITS: u64 = 65_536;
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_TEXT: usize = 65_536;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

type Coefficients = Vec<BigRational>;
type Result<T> = std::result::Result<T, RationalLaplaceError>;

#[derive(Debug, Error)]
pub enum RationalLaplaceError {
    #[error("rational Laplace transform unsupported: {0}")]
    Unsupported(&'static str),
    #[error("rational Laplace source refused: {0}")]
    InvalidSource(&'static str),
    #[error("rational Laplace verification failed: {0}")]
    Verification(&'static str),
    #[error("rational Laplace envelope exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Polynomial(#[from] PolyError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalLaplaceError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

impl RationalLaplaceError {
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }
}

/// Coefficients are in ascending powers of the time variable.
///
/// For d > 0, Quadratic means exp(rate*t) times
/// cosine(t)*cos(sqrt(d)*t) + sine(t)*sin(sqrt(d)*t)/sqrt(d).
/// For d < 0, replace cos/sin by cosh/sinh and sqrt(d) by sqrt(-d).
/// This normalized basis keeps every stored coefficient rational.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaplaceBlock {
    Exponential {
        rate: BigRational,
        coefficients: Coefficients,
    },
    Quadratic {
        rate: BigRational,
        squared_frequency: BigRational,
        cosine: Coefficients,
        sine: Coefficients,
    },
}

/// Untrusted data: callers must verify it against the exact original source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalLaplaceCertificate {
    pub blocks: Vec<LaplaceBlock>,
}

/// Complete causal inverse on a sufficient right half-plane.
#[derive(Debug, Clone)]
pub struct VerifiedRationalLaplace {
    source: Expr,
    transform_variable: Symbol,
    time_variable: Symbol,
    certificate: RationalLaplaceCertificate,
    function: Expr,
    sufficient_abscissa: BigRational,
}

impl VerifiedRationalLaplace {
    pub fn source(&self) -> &Expr {
        &self.source
    }
    pub fn transform_variable(&self) -> &Symbol {
        &self.transform_variable
    }
    pub fn time_variable(&self) -> &Symbol {
        &self.time_variable
    }
    pub fn certificate(&self) -> &RationalLaplaceCertificate {
        &self.certificate
    }
    pub fn blocks(&self) -> &[LaplaceBlock] {
        &self.certificate.blocks
    }
    pub fn function(&self) -> &Expr {
        &self.function
    }
    pub fn sufficient_abscissa(&self) -> &BigRational {
        &self.sufficient_abscissa
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
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

fn q(value: i64) -> BigRational {
    BigRational::from(value)
}

fn bit_size(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn scalar_ok(value: &BigRational, limit: u64) -> Result<()> {
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(RationalLaplaceError::InvalidSource(
            "invalid rational coefficient",
        ));
    }
    if bit_size(value) > limit {
        return Err(RationalLaplaceError::ResourceLimit("coefficient height"));
    }
    Ok(())
}

fn trim(values: &mut Coefficients) {
    while values.len() > 1 && values.last().is_some_and(Zero::is_zero) {
        values.pop();
    }
}

fn is_zero(values: &[BigRational]) -> bool {
    values.iter().all(Zero::is_zero)
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
        self.charge(1, 96 + 2 * bit_size(value).div_ceil(8) as usize)?;
        Ok(value.clone())
    }

    fn add_number(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        let value = a.metered_add(b, self.meter)?;
        scalar_ok(&value, MAX_BITS)?;
        Ok(value)
    }

    fn sub_number(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        let value = a.metered_sub(b, self.meter)?;
        scalar_ok(&value, MAX_BITS)?;
        Ok(value)
    }

    fn mul_number(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        let value = a.metered_mul(b, self.meter)?;
        scalar_ok(&value, MAX_BITS)?;
        Ok(value)
    }

    fn div_number(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        if b.is_zero() {
            return Err(RationalLaplaceError::Verification(
                "zero arithmetic divisor",
            ));
        }
        let value = a.metered_div(b, self.meter)?;
        scalar_ok(&value, MAX_BITS)?;
        Ok(value)
    }

    fn zeros(&mut self, length: usize) -> Result<Coefficients> {
        if length == 0 || length > MAX_INTERNAL_DEGREE + 1 {
            return Err(RationalLaplaceError::ResourceLimit("polynomial degree"));
        }
        self.charge(length, length * 128)?;
        Ok(vec![BigRational::zero(); length])
    }

    fn copy(&mut self, values: &[BigRational]) -> Result<Coefficients> {
        if values.is_empty() || values.len() > MAX_INTERNAL_DEGREE + 1 {
            return Err(RationalLaplaceError::ResourceLimit("polynomial degree"));
        }
        let mut bytes = values.len() * 96;
        for value in values {
            scalar_ok(value, MAX_BITS)?;
            bytes += 2 * bit_size(value).div_ceil(8) as usize;
        }
        self.charge(values.len(), bytes)?;
        Ok(values.to_vec())
    }

    fn add(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Coefficients> {
        let mut out = self.zeros(a.len().max(b.len()))?;
        for (i, value) in out.iter_mut().enumerate() {
            let zero = BigRational::zero();
            *value = self.add_number(a.get(i).unwrap_or(&zero), b.get(i).unwrap_or(&zero))?;
        }
        trim(&mut out);
        Ok(out)
    }

    fn sub(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Coefficients> {
        let mut out = self.zeros(a.len().max(b.len()))?;
        for (i, value) in out.iter_mut().enumerate() {
            let zero = BigRational::zero();
            *value = self.sub_number(a.get(i).unwrap_or(&zero), b.get(i).unwrap_or(&zero))?;
        }
        trim(&mut out);
        Ok(out)
    }

    fn scale(&mut self, a: &[BigRational], c: &BigRational) -> Result<Coefficients> {
        let mut out = self.zeros(a.len())?;
        for (value, original) in out.iter_mut().zip(a) {
            *value = self.mul_number(original, c)?;
        }
        trim(&mut out);
        Ok(out)
    }

    fn mul(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Coefficients> {
        let mut out = self.zeros(a.len() + b.len() - 1)?;
        for (i, av) in a.iter().enumerate() {
            for (j, bv) in b.iter().enumerate() {
                let term = self.mul_number(av, bv)?;
                out[i + j] = self.add_number(&out[i + j], &term)?;
            }
        }
        trim(&mut out);
        Ok(out)
    }

    fn pow(&mut self, values: &[BigRational], mut exponent: usize) -> Result<Coefficients> {
        if exponent > MAX_DEGREE {
            return Err(RationalLaplaceError::ResourceLimit("polynomial power"));
        }
        let mut base = self.copy(values)?;
        let mut out = self.copy(&[BigRational::one()])?;
        while exponent != 0 {
            self.meter.checkpoint()?;
            if exponent & 1 != 0 {
                out = self.mul(&out, &base)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = self.mul(&base, &base)?;
            }
        }
        Ok(out)
    }

    fn derivative(&mut self, values: &[BigRational]) -> Result<Coefficients> {
        let mut out = self.zeros(values.len().saturating_sub(1).max(1))?;
        for i in 1..values.len() {
            out[i - 1] = self.mul_number(&values[i], &q(i as i64))?;
        }
        trim(&mut out);
        Ok(out)
    }

    fn quotient(&mut self, top: &[BigRational], bottom: &[BigRational]) -> Result<Coefficients> {
        if is_zero(bottom) {
            return Err(RationalLaplaceError::Verification(
                "zero polynomial divisor",
            ));
        }
        let mut remainder = self.copy(top)?;
        let mut quotient = self.zeros(top.len().saturating_sub(bottom.len()) + 1)?;
        while !is_zero(&remainder) && remainder.len() >= bottom.len() {
            self.meter.checkpoint()?;
            let position = remainder.len() - bottom.len();
            let coefficient = self.div_number(
                remainder.last().expect("nonempty"),
                bottom.last().expect("nonempty"),
            )?;
            quotient[position] = self.number(&coefficient)?;
            for (j, value) in bottom.iter().enumerate() {
                let product = self.mul_number(&coefficient, value)?;
                remainder[position + j] = self.sub_number(&remainder[position + j], &product)?;
            }
            trim(&mut remainder);
        }
        if !is_zero(&remainder) {
            return Err(RationalLaplaceError::Verification(
                "transform denominator does not divide original denominator",
            ));
        }
        trim(&mut quotient);
        Ok(quotient)
    }
}

fn integer_power(expression: &Expr) -> Option<i64> {
    match expression {
        Expr::Integer(value) => value.to_i64(),
        Expr::Rational(value) if value.is_integer() => value.numer().to_i64(),
        _ => None,
    }
}

fn admit(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    let mut text = variable.name.len().saturating_add(time.name.len());
    if text > MAX_TEXT {
        return Err(RationalLaplaceError::ResourceLimit("typed symbol names"));
    }
    if variable == time {
        return Err(RationalLaplaceError::InvalidSource(
            "transform and time variables must be distinct",
        ));
    }
    work.charge(1, MAX_NODES * 24)?;
    let mut pending = vec![(source, 0usize)];
    let mut visited = 0usize;
    let mut unsupported = false;
    while let Some((node, depth)) = pending.pop() {
        work.meter.checkpoint()?;
        visited += 1;
        if visited > MAX_NODES || depth > MAX_DEPTH {
            return Err(RationalLaplaceError::ResourceLimit(
                "source expression shape",
            ));
        }
        match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_INPUT_BITS {
                    return Err(RationalLaplaceError::ResourceLimit("source integer height"));
                }
            }
            Expr::Rational(value) => scalar_ok(value, MAX_INPUT_BITS)?,
            Expr::Sym(symbol) => {
                text = text.saturating_add(symbol.name.len());
                if text > MAX_TEXT {
                    return Err(RationalLaplaceError::ResourceLimit("source symbol names"));
                }
                unsupported |= symbol != variable;
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if matches!(node, Expr::Function(_, _)) {
                    unsupported = true;
                }
                if children.is_empty() || children.len() + pending.len() + visited > MAX_NODES {
                    return Err(RationalLaplaceError::ResourceLimit(
                        "source expression fanout",
                    ));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                match integer_power(exponent) {
                    Some(power) if power.unsigned_abs() <= MAX_DEGREE as u64 => {}
                    Some(_) => {
                        return Err(RationalLaplaceError::ResourceLimit("source integer power"));
                    }
                    None => unsupported = true,
                }
                if pending.len() + visited + 2 > MAX_NODES {
                    return Err(RationalLaplaceError::ResourceLimit(
                        "source expression fanout",
                    ));
                }
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
            }
            Expr::Const(_) => unsupported = true,
        }
        if text > MAX_TEXT {
            return Err(RationalLaplaceError::ResourceLimit("source symbol names"));
        }
    }
    work.charge(visited, 1)?;
    if unsupported {
        return Err(RationalLaplaceError::Unsupported(
            "exact rational functions in one transform variable are required",
        ));
    }
    Ok(visited)
}

struct Fraction {
    numerator: Coefficients,
    denominator: Coefficients,
    // A reciprocal or zeroth power can remove an original denominator from
    // the final rational value. Preserve all such original-domain exclusions.
    excluded_abscissa: Option<BigRational>,
}

fn maximum_bound(
    left: Option<BigRational>,
    right: Option<BigRational>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Option<BigRational>> {
    match (left, right) {
        (Some(left), Some(right)) => {
            let ordering = left.metered_cmp(&right, work.meter)?;
            Ok(Some(if ordering.is_lt() { right } else { left }))
        }
        (Some(value), None) | (None, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

fn parse_source(source: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Fraction> {
    work.charge(1, 64)?;
    let constant = |numerator| Fraction {
        numerator,
        denominator: vec![BigRational::one()],
        excluded_abscissa: None,
    };
    let result = match source {
        Expr::Integer(value) => {
            work.charge(1, 128 + value.bits().div_ceil(8) as usize)?;
            constant(vec![BigRational::from_integer(value.clone())])
        }
        Expr::Rational(value) => constant(vec![work.number(value)?]),
        Expr::Sym(_) => constant(work.copy(&[BigRational::zero(), BigRational::one()])?),
        Expr::Add(children) => {
            let mut out = constant(work.copy(&[BigRational::zero()])?);
            for child in children {
                let rhs = parse_source(child, work)?;
                let left = work.mul(&out.numerator, &rhs.denominator)?;
                let right = work.mul(&rhs.numerator, &out.denominator)?;
                out = Fraction {
                    numerator: work.add(&left, &right)?,
                    denominator: work.mul(&out.denominator, &rhs.denominator)?,
                    excluded_abscissa: maximum_bound(
                        out.excluded_abscissa,
                        rhs.excluded_abscissa,
                        work,
                    )?,
                };
            }
            out
        }
        Expr::Mul(children) => {
            let mut out = constant(work.copy(&[BigRational::one()])?);
            for child in children {
                // Even a zero product admits every original denominator.
                let rhs = parse_source(child, work)?;
                out = Fraction {
                    numerator: work.mul(&out.numerator, &rhs.numerator)?,
                    denominator: work.mul(&out.denominator, &rhs.denominator)?,
                    excluded_abscissa: maximum_bound(
                        out.excluded_abscissa,
                        rhs.excluded_abscissa,
                        work,
                    )?,
                };
            }
            out
        }
        Expr::Pow(base, exponent) => {
            let base = parse_source(base, work)?;
            let exponent = integer_power(exponent).ok_or(RationalLaplaceError::Unsupported(
                "noninteger rational power",
            ))?;
            if exponent <= 0 && is_zero(&base.numerator) {
                return Err(RationalLaplaceError::InvalidSource(
                    "zero base of a nonpositive power",
                ));
            }
            let mut excluded_abscissa = base.excluded_abscissa;
            if exponent <= 0 && base.numerator.len() > 1 {
                let bound = cauchy_bound(&base.numerator, work)?;
                excluded_abscissa = maximum_bound(excluded_abscissa, Some(bound), work)?;
            }
            let (numerator, denominator) = if exponent < 0 {
                (base.denominator, base.numerator)
            } else {
                (base.numerator, base.denominator)
            };
            Fraction {
                numerator: work.pow(&numerator, exponent.unsigned_abs() as usize)?,
                denominator: work.pow(&denominator, exponent.unsigned_abs() as usize)?,
                excluded_abscissa,
            }
        }
        _ => return Err(RationalLaplaceError::Unsupported("nonrational source")),
    };
    if is_zero(&result.denominator) {
        return Err(RationalLaplaceError::InvalidSource(
            "identically zero denominator",
        ));
    }
    if result.numerator.len() > MAX_DEGREE + 1 || result.denominator.len() > MAX_DEGREE + 1 {
        return Err(RationalLaplaceError::ResourceLimit(
            "source polynomial degree",
        ));
    }
    Ok(result)
}

struct BlockPlan {
    rate: BigRational,
    squared_frequency: Option<BigRational>,
    multiplicity: usize,
    column: usize,
}

fn write_column(
    matrix: &mut [Coefficients],
    column: usize,
    numerator: &[BigRational],
    denominator: &[BigRational],
    original_denominator: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let complement = work.quotient(original_denominator, denominator)?;
    let expanded = work.mul(numerator, &complement)?;
    if expanded.len() > matrix.len() {
        return Err(RationalLaplaceError::Verification(
            "nonproper transform basis",
        ));
    }
    for (row, value) in expanded.iter().enumerate() {
        matrix[row][column] = work.number(value)?;
    }
    Ok(())
}

/// Generator recurrence for (-d/ds)^n(A/B):
/// next numerator = power*A*B' - A'*B, next denominator = B^(power+1).
/// The independent checker below uses binomial sums instead.
fn next_numerator(
    numerator: &[BigRational],
    base: &[BigRational],
    power: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    let base_derivative = work.derivative(base)?;
    let top_derivative = work.derivative(numerator)?;
    let first = work.mul(numerator, &base_derivative)?;
    let first = work.scale(&first, &q(power as i64))?;
    let second = work.mul(&top_derivative, base)?;
    work.sub(&first, &second)
}

fn solve_coefficients(
    mut matrix: Vec<Coefficients>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    let n = matrix.len();
    for column in 0..n {
        work.meter.checkpoint()?;
        let pivot = (column..n)
            .find(|&row| !matrix[row][column].is_zero())
            .ok_or(RationalLaplaceError::Verification(
                "singular complete transform basis",
            ))?;
        matrix.swap(column, pivot);
        let divisor = work.number(&matrix[column][column])?;
        for value in &mut matrix[column][column..=n] {
            *value = work.div_number(value, &divisor)?;
        }
        let normalized = work.copy(&matrix[column])?;
        for (row, values) in matrix.iter_mut().enumerate() {
            if row == column || values[column].is_zero() {
                continue;
            }
            let factor = work.number(&values[column])?;
            for j in column..=n {
                let product = work.mul_number(&factor, &normalized[j])?;
                values[j] = work.sub_number(&values[j], &product)?;
            }
        }
    }
    let mut solution = work.zeros(n)?;
    for (out, row) in solution.iter_mut().zip(&matrix) {
        *out = work.number(&row[n])?;
    }
    Ok(solution)
}

fn generate(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalLaplaceCertificate> {
    admit(source, variable, time, work)?;
    let original = parse_source(source, work)?;
    if is_zero(&original.numerator) {
        return Ok(RationalLaplaceCertificate { blocks: Vec::new() });
    }
    if original.numerator.len() >= original.denominator.len() {
        return Err(RationalLaplaceError::Unsupported(
            "improper transforms require distributional terms",
        ));
    }
    let n = original.denominator.len() - 1;
    work.charge(n * n, n * n * 32 + 128 + variable.name.len())?;
    let denominator = UnivariatePoly::new(variable.clone(), work.copy(&original.denominator)?);
    let factorization = metered_complete_factorization(&denominator, work.meter)?;
    let mut matrix = Vec::with_capacity(n);
    for _ in 0..n {
        matrix.push(work.zeros(n + 1)?);
    }
    for (i, value) in original.numerator.iter().enumerate() {
        matrix[i][n] = work.number(value)?;
    }
    let mut plans = Vec::new();
    let mut column = 0usize;
    for factor in &factorization.factors {
        work.meter.checkpoint()?;
        let degree = factor.poly.degree().unwrap_or(0);
        if degree == 0 || degree > 2 {
            return Err(RationalLaplaceError::Unsupported(
                "denominator contains a factor of degree greater than two",
            ));
        }
        let multiplicity = factor.multiplicity;
        let next_column = degree
            .checked_mul(multiplicity)
            .and_then(|width| column.checked_add(width));
        if multiplicity == 0
            || next_column.is_none_or(|end| end > n)
            || factor.poly.gen_sym != *variable
            || factor.poly.coeffs.last() != Some(&BigRational::one())
        {
            return Err(RationalLaplaceError::Verification(
                "invalid complete factorization shape",
            ));
        }
        let base = work.copy(&factor.poly.coeffs)?;
        let minus_b = work.sub_number(&BigRational::zero(), &base[degree - 1])?;
        let rate = if degree == 1 {
            minus_b
        } else {
            work.div_number(&minus_b, &q(2))?
        };
        let squared_frequency = if degree == 2 {
            let rate_squared = work.mul_number(&rate, &rate)?;
            let d = work.sub_number(&base[0], &rate_squared)?;
            if d.is_zero() {
                return Err(RationalLaplaceError::Verification(
                    "repeated linear pole was not factored",
                ));
            }
            Some(d)
        } else {
            None
        };
        work.charge(1, 128)?;
        let plan = BlockPlan {
            rate,
            squared_frequency,
            multiplicity,
            column,
        };
        let mut first = if degree == 1 {
            work.copy(&[BigRational::one()])?
        } else {
            let minus_rate = work.sub_number(&BigRational::zero(), &plan.rate)?;
            work.copy(&[minus_rate, BigRational::one()])?
        };
        let mut second = work.copy(&[BigRational::one()])?;
        for power in 1..=multiplicity {
            let denominator = work.pow(&base, power)?;
            write_column(
                &mut matrix,
                column,
                &first,
                &denominator,
                &original.denominator,
                work,
            )?;
            column += 1;
            if degree == 2 {
                write_column(
                    &mut matrix,
                    column,
                    &second,
                    &denominator,
                    &original.denominator,
                    work,
                )?;
                column += 1;
            }
            if power != multiplicity {
                first = next_numerator(&first, &base, power, work)?;
                if degree == 2 {
                    second = next_numerator(&second, &base, power, work)?;
                }
            }
        }
        plans.push(plan);
    }
    if column != n {
        return Err(RationalLaplaceError::Verification(
            "incomplete transform basis",
        ));
    }
    let solution = solve_coefficients(matrix, work)?;
    let mut blocks = Vec::with_capacity(plans.len());
    work.charge(plans.len(), plans.len() * 128)?;
    for plan in plans {
        if let Some(squared_frequency) = plan.squared_frequency {
            let mut cosine = work.zeros(plan.multiplicity)?;
            let mut sine = work.zeros(plan.multiplicity)?;
            for k in 0..plan.multiplicity {
                cosine[k] = work.number(&solution[plan.column + 2 * k])?;
                sine[k] = work.number(&solution[plan.column + 2 * k + 1])?;
            }
            trim(&mut cosine);
            trim(&mut sine);
            if !is_zero(&cosine) || !is_zero(&sine) {
                blocks.push(LaplaceBlock::Quadratic {
                    rate: plan.rate,
                    squared_frequency,
                    cosine,
                    sine,
                });
            }
        } else {
            let mut coefficients =
                work.copy(&solution[plan.column..plan.column + plan.multiplicity])?;
            trim(&mut coefficients);
            if !is_zero(&coefficients) {
                blocks.push(LaplaceBlock::Exponential {
                    rate: plan.rate,
                    coefficients,
                });
            }
        }
    }
    Ok(RationalLaplaceCertificate { blocks })
}

fn canonical_coefficients(values: &[BigRational]) -> Result<()> {
    if values.is_empty()
        || values.len() > MAX_DEGREE
        || (values.len() > 1 && values.last().is_some_and(Zero::is_zero))
    {
        return Err(RationalLaplaceError::Verification(
            "noncanonical time polynomial",
        ));
    }
    for value in values {
        scalar_ok(value, MAX_BITS)?;
    }
    Ok(())
}

fn validate_certificate(
    certificate: &RationalLaplaceCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    if certificate.blocks.len() > MAX_DEGREE {
        return Err(RationalLaplaceError::ResourceLimit(
            "too many transform blocks",
        ));
    }
    let mut cells = 0usize;
    for block in &certificate.blocks {
        work.meter.checkpoint()?;
        match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                scalar_ok(rate, MAX_BITS)?;
                canonical_coefficients(coefficients)?;
                if is_zero(coefficients) {
                    return Err(RationalLaplaceError::Verification(
                        "zero transform block is not canonical",
                    ));
                }
                cells += coefficients.len();
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                scalar_ok(rate, MAX_BITS)?;
                scalar_ok(squared_frequency, MAX_BITS)?;
                if squared_frequency.is_zero() {
                    return Err(RationalLaplaceError::Verification(
                        "zero quadratic frequency",
                    ));
                }
                canonical_coefficients(cosine)?;
                canonical_coefficients(sine)?;
                if is_zero(cosine) && is_zero(sine) {
                    return Err(RationalLaplaceError::Verification(
                        "zero transform block is not canonical",
                    ));
                }
                cells += 2 * cosine.len().max(sine.len());
            }
        }
        if cells > MAX_DEGREE {
            return Err(RationalLaplaceError::ResourceLimit(
                "time polynomial dimension",
            ));
        }
    }
    work.charge(cells.max(1), certificate.blocks.len() * 128 + 1)?;
    Ok(cells)
}

#[derive(Clone, Copy)]
enum Component {
    Exponential,
    Cosine,
    Sine,
}

/// Independent closed transform table in a rational coefficient basis.
/// For quadratic blocks this is the real/normalized-imaginary part of
/// n!/(z-i*sqrt(d))^(n+1), expanded by the binomial theorem. The same
/// rational identity holds for d < 0 and the real hyperbolic basis.
fn reference_basis(
    rate: &BigRational,
    squared_frequency: Option<&BigRational>,
    degree: usize,
    component: Component,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Fraction> {
    let minus_rate = work.sub_number(&BigRational::zero(), rate)?;
    let shifted = work.copy(&[minus_rate, BigRational::one()])?;
    let mut factorial = BigRational::one();
    for k in 2..=degree {
        factorial = work.mul_number(&factorial, &q(k as i64))?;
    }
    if matches!(component, Component::Exponential) {
        return Ok(Fraction {
            numerator: vec![factorial],
            denominator: work.pow(&shifted, degree + 1)?,
            excluded_abscissa: None,
        });
    }
    let d = squared_frequency.ok_or(RationalLaplaceError::Verification(
        "missing quadratic frequency",
    ))?;
    let square = work.mul(&shifted, &shifted)?;
    let base = work.add(&square, std::slice::from_ref(d))?;
    let denominator = work.pow(&base, degree + 1)?;
    let mut numerator = work.copy(&[BigRational::zero()])?;
    let mut binomial = BigRational::one();
    for j in 0..=degree + 1 {
        work.meter.checkpoint()?;
        let include = match component {
            Component::Cosine => j % 2 == 0,
            Component::Sine => j % 2 == 1,
            Component::Exponential => false,
        };
        if include {
            let half = j / 2;
            let d_power = work.pow(std::slice::from_ref(d), half)?;
            let mut coefficient = work.mul_number(&factorial, &binomial)?;
            coefficient = work.mul_number(&coefficient, &d_power[0])?;
            if half % 2 == 1 {
                coefficient = work.sub_number(&BigRational::zero(), &coefficient)?;
            }
            let power = work.pow(&shifted, degree + 1 - j)?;
            let term = work.scale(&power, &coefficient)?;
            numerator = work.add(&numerator, &term)?;
        }
        if j < degree + 1 {
            let top = work.mul_number(&binomial, &q((degree + 1 - j) as i64))?;
            binomial = work.div_number(&top, &q((j + 1) as i64))?;
        }
    }
    Ok(Fraction {
        numerator,
        denominator,
        excluded_abscissa: None,
    })
}

fn check_component(
    accumulator: &mut Coefficients,
    original_denominator: &[BigRational],
    rate: &BigRational,
    frequency: Option<&BigRational>,
    coefficients: &[BigRational],
    component: Component,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    for (degree, coefficient) in coefficients.iter().enumerate() {
        work.meter.checkpoint()?;
        if coefficient.is_zero() {
            continue;
        }
        let transform = reference_basis(rate, frequency, degree, component, work)?;
        let complement = work.quotient(original_denominator, &transform.denominator)?;
        let numerator = work.mul(&transform.numerator, &complement)?;
        let numerator = work.scale(&numerator, coefficient)?;
        *accumulator = work.add(accumulator, &numerator)?;
    }
    Ok(())
}

fn cauchy_bound(
    denominator: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    if denominator.len() == 1 {
        return Ok(BigRational::zero());
    }
    let leading = denominator.last().expect("nonempty denominator");
    let mut maximum = BigRational::zero();
    for coefficient in &denominator[..denominator.len() - 1] {
        let ratio = work.div_number(coefficient, leading)?;
        let absolute = if ratio.is_negative() {
            work.sub_number(&BigRational::zero(), &ratio)?
        } else {
            ratio
        };
        if absolute.metered_cmp(&maximum, work.meter)?.is_gt() {
            maximum = absolute;
        }
    }
    work.add_number(&maximum, &BigRational::one())
}

fn sum_expression(terms: Vec<Expr>) -> Expr {
    match terms.len() {
        0 => Expr::from_i64(0),
        1 => terms.into_iter().next().expect("one expression"),
        _ => Expr::Add(terms),
    }
}

fn product_expression(factors: Vec<Expr>) -> Expr {
    let factors: Vec<_> = factors
        .into_iter()
        .filter(|factor| !factor.is_one())
        .collect();
    match factors.len() {
        0 => Expr::from_i64(1),
        1 => factors.into_iter().next().expect("one expression"),
        _ => Expr::Mul(factors),
    }
}

fn time_polynomial(coefficients: &[BigRational], time: &Symbol) -> Expr {
    let terms = coefficients
        .iter()
        .enumerate()
        .filter_map(|(power, coefficient)| {
            if coefficient.is_zero() {
                return None;
            }
            let coefficient = rational_expr(coefficient.clone());
            if power == 0 {
                return Some(coefficient);
            }
            let variable = Expr::Sym(time.clone());
            let monomial = if power == 1 {
                variable
            } else {
                eval_pow(variable, Expr::from_i64(power as i64))
            };
            Some(product_expression(vec![coefficient, monomial]))
        })
        .collect();
    sum_expression(terms)
}

/// Exact rational square roots use governed integer arithmetic. A nonsquare
/// stays a held radical; presentation never starts an unmetered factor search.
fn lifted_frequency(
    magnitude: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(Expr, Expr)> {
    if !magnitude.is_positive() {
        return Err(RationalLaplaceError::Verification(
            "nonpositive frequency magnitude",
        ));
    }
    let numerator = metered_sqrt_floor(magnitude.numer(), work.meter)?.ok_or(
        RationalLaplaceError::Verification("negative frequency numerator"),
    )?;
    let denominator = metered_sqrt_floor(magnitude.denom(), work.meter)?.ok_or(
        RationalLaplaceError::Verification("negative frequency denominator"),
    )?;
    let numerator_square = metered_mul(&numerator, &numerator, work.meter)?;
    let denominator_square = metered_mul(&denominator, &denominator, work.meter)?;
    work.charge(8, 512 + 8 * bit_size(magnitude).div_ceil(8) as usize)?;
    if numerator_square == *magnitude.numer() && denominator_square == *magnitude.denom() {
        let frequency = BigRational::metered_new(&numerator, &denominator, work.meter)?;
        let inverse = work.div_number(&BigRational::one(), &frequency)?;
        return Ok((rational_expr(frequency), rational_expr(inverse)));
    }
    let frequency = Expr::Pow(
        Arc::new(rational_expr(work.number(magnitude)?)),
        Arc::new(rational_expr(BigRational::new(1.into(), 2.into()))),
    );
    let inverse = Expr::Pow(Arc::new(frequency.clone()), Arc::new(Expr::from_i64(-1)));
    work.meter.checkpoint()?;
    Ok((frequency, inverse))
}

fn lifted_function(
    certificate: &RationalLaplaceCertificate,
    time: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    let mut cells = 0usize;
    let mut bits = 1u64;
    for block in &certificate.blocks {
        match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                cells += coefficients.len();
                bits = bits.max(bit_size(rate));
                for coefficient in coefficients {
                    bits = bits.max(bit_size(coefficient));
                }
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                cells += cosine.len() + sine.len();
                bits = bits.max(bit_size(rate)).max(bit_size(squared_frequency));
                for coefficient in cosine.iter().chain(sine) {
                    bits = bits.max(bit_size(coefficient));
                }
            }
        }
    }
    let nodes = 1 + 24 * cells + 16 * certificate.blocks.len();
    if nodes > MAX_NODES {
        return Err(RationalLaplaceError::ResourceLimit(
            "lifted expression nodes",
        ));
    }
    // Cover exact coefficient/name copies, elementary rational-power
    // construction, and transient term vectors before allocating them.
    let bytes = nodes
        .checked_mul(256 + 4 * bits.div_ceil(8) as usize + 4 * time.name.len())
        .ok_or(RationalLaplaceError::ResourceLimit(
            "lifted expression allocation",
        ))?;
    work.charge(nodes * nodes, bytes)?;
    let mut terms = Vec::with_capacity(certificate.blocks.len());
    for block in &certificate.blocks {
        let (rate, inner) = match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                (rate, time_polynomial(coefficients, time))
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency: d,
                cosine,
                sine,
            } => {
                let positive = d.is_positive();
                let magnitude = if positive {
                    work.number(d)?
                } else {
                    work.sub_number(&BigRational::zero(), d)?
                };
                let (frequency, inverse_frequency) = lifted_frequency(&magnitude, work)?;
                let argument = product_expression(vec![frequency.clone(), Expr::Sym(time.clone())]);
                let mut pair = Vec::new();
                if !is_zero(cosine) {
                    let function = Expr::Function(
                        if positive { "cos" } else { "cosh" }.into(),
                        vec![argument.clone()],
                    );
                    pair.push(product_expression(vec![
                        time_polynomial(cosine, time),
                        function,
                    ]));
                }
                if !is_zero(sine) {
                    let function = Expr::Function(
                        if positive { "sin" } else { "sinh" }.into(),
                        vec![argument],
                    );
                    pair.push(product_expression(vec![
                        time_polynomial(sine, time),
                        inverse_frequency,
                        function,
                    ]));
                }
                (rate, sum_expression(pair))
            }
        };
        if inner.is_zero() {
            continue;
        }
        let value = if rate.is_zero() {
            inner
        } else {
            let argument =
                product_expression(vec![rational_expr(rate.clone()), Expr::Sym(time.clone())]);
            product_expression(vec![Expr::Function("exp".into(), vec![argument]), inner])
        };
        terms.push(value);
    }
    work.meter.checkpoint()?;
    Ok(sum_expression(terms))
}

fn verify_inner(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    certificate: &RationalLaplaceCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedRationalLaplace> {
    let nodes = admit(source, variable, time, work)?;
    let original = parse_source(source, work)?;
    if !is_zero(&original.numerator) && original.numerator.len() >= original.denominator.len() {
        return Err(RationalLaplaceError::Unsupported(
            "improper transforms require distributional terms",
        ));
    }
    let cells = validate_certificate(certificate, work)?;
    let mut reconstructed = work.copy(&[BigRational::zero()])?;
    for block in &certificate.blocks {
        match block {
            LaplaceBlock::Exponential { rate, coefficients } => check_component(
                &mut reconstructed,
                &original.denominator,
                rate,
                None,
                coefficients,
                Component::Exponential,
                work,
            )?,
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                check_component(
                    &mut reconstructed,
                    &original.denominator,
                    rate,
                    Some(squared_frequency),
                    cosine,
                    Component::Cosine,
                    work,
                )?;
                check_component(
                    &mut reconstructed,
                    &original.denominator,
                    rate,
                    Some(squared_frequency),
                    sine,
                    Component::Sine,
                    work,
                )?;
            }
        }
    }
    if reconstructed != original.numerator {
        return Err(RationalLaplaceError::Verification(
            "complete rational transform identity",
        ));
    }
    let denominator_bound = cauchy_bound(&original.denominator, work)?;
    let sufficient_abscissa =
        maximum_bound(Some(denominator_bound), original.excluded_abscissa, work)?
            .expect("denominator bound is always present");
    let function = lifted_function(certificate, time, work)?;
    // Complete original AST and certificate copies are charged before the
    // immutable receipt is assembled. No untrusted boolean creates acceptance.
    work.charge(
        nodes + cells,
        nodes * 2048 + 4 * MAX_TEXT + (cells + certificate.blocks.len() * 2) * 16_512,
    )?;
    let receipt = VerifiedRationalLaplace {
        source: source.clone(),
        transform_variable: variable.clone(),
        time_variable: time.clone(),
        certificate: certificate.clone(),
        function,
        sufficient_abscissa,
    };
    work.meter.checkpoint()?;
    Ok(receipt)
}

pub fn verify_rational_laplace_certificate(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    certificate: &RationalLaplaceCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalLaplace> {
    let mut capped = Capped {
        inner: verifier,
        used: [0; 3],
        stopped: None,
    };
    verify_inner(
        source,
        transform_variable,
        time_variable,
        certificate,
        &mut Work { meter: &mut capped },
    )
}

pub fn inverse_rational_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
) -> Result<VerifiedRationalLaplace> {
    metered_inverse_rational_laplace(
        source,
        transform_variable,
        time_variable,
        &mut Unbounded,
        &mut Unbounded,
    )
}

pub fn metered_inverse_rational_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalLaplace> {
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
    let candidate = generate(
        source,
        transform_variable,
        time_variable,
        &mut Work {
            meter: &mut generation,
        },
    );
    // The polynomial factorizer carries its legacy error type. Preserve any
    // underlying meter interruption rather than reclassifying it as algebra.
    if let Some(error) = generation.stopped {
        return Err(RationalLaplaceError::Meter(error));
    }
    let candidate = candidate?;
    generation.checkpoint()?;
    let result = verify_inner(
        source,
        transform_variable,
        time_variable,
        &candidate,
        &mut Work {
            meter: &mut verification,
        },
    )?;
    generation.checkpoint()?;
    verification.checkpoint()?;
    Ok(result)
}
