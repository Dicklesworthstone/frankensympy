//! Complete bounded rational constant-coefficient ODE systems.
//!
//! Generation uses Faddeev--LeVerrier for a rational resolvent, whose full
//! polynomial matrix identity is checked independently. Existing checked
//! inverse Laplace transforms propose exponential/trigonometric blocks.
//! Acceptance then checks Phi' = A*Phi, Phi(0) = I, p' = A*p + f(t0+tau),
//! and p(0) = 0 directly in a rational coefficient basis. This second check
//! establishes the smooth global real-time solution family, beyond the causal
//! transform's half-plane claim. No expression simplifier is a proof oracle.
//!
//! Matrices have dimension at most eight. Polynomial forcing has degree at
//! most sixteen; rational linear and quadratic spectral factors are supported.
//! Other spectral factors, exhausted work, and invalid certificates refuse.
//! Generator and verifier each retain one cumulative cap across every entry.
//! Receipts are immutable in-process evidence, not portable kernel proofs.

use super::{Capped, product, sum};
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_laplace::{
    LaplaceBlock, RationalLaplaceCertificate, RationalLaplaceError,
    metered_inverse_rational_laplace, verify_rational_laplace_certificate,
};
use fsym_core::arith::{RationalArithmeticError, metered_mul, metered_sqrt_floor};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use num_traits::{One, Signed, Zero};
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;

const MAX_DIMENSION: usize = 8;
const MAX_FORCING_DEGREE: usize = 16;
const MAX_BLOCK_CELLS: usize = 32;
const MAX_INPUT_BITS: u64 = 1024;
const MAX_DERIVED_BITS: u64 = 16_384;
const MAX_TRANSFORM_BITS: u64 = 4096;
const MAX_TEXT: usize = 16_384;

/// Independently checked rational exponential and trigonometric forcing.
pub mod forcing;

pub type RationalMatrix = Vec<Vec<BigRational>>;
pub type PolynomialVector = Vec<Vec<BigRational>>;
pub type ExpressionMatrix = Vec<Vec<Expr>>;
type PolynomialMatrix = Vec<Vec<Vec<BigRational>>>;
type PolynomialFraction = (Vec<BigRational>, Vec<BigRational>);
type Result<T> = std::result::Result<T, GeneralOdeSystemError>;

#[derive(Debug, Error)]
pub enum GeneralOdeSystemError {
    #[error("general linear ODE system unsupported: {0}")]
    Unsupported(&'static str),
    #[error("invalid general linear ODE system: {0}")]
    InvalidSource(&'static str),
    #[error("general linear ODE system resource limit: {0}")]
    ResourceLimit(&'static str),
    #[error("general linear ODE system verification failed: {0}")]
    Verification(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
    #[error(transparent)]
    Transform(RationalLaplaceError),
}

impl GeneralOdeSystemError {
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }
}

impl From<RationalArithmeticError> for GeneralOdeSystemError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

impl From<RationalLaplaceError> for GeneralOdeSystemError {
    fn from(error: RationalLaplaceError) -> Self {
        match error {
            RationalLaplaceError::Meter(error) => Self::Meter(error),
            RationalLaplaceError::Unsupported(reason) => Self::Unsupported(reason),
            error => Self::Transform(error),
        }
    }
}

/// Untrusted block coefficients in the local coordinate tau = t - t0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralOdeSystemCertificate {
    pub fundamental: Vec<Vec<RationalLaplaceCertificate>>,
    pub particular: Vec<RationalLaplaceCertificate>,
}

/// A complete affine solution family, normalized at the original initial point.
#[derive(Debug, Clone)]
pub struct VerifiedGeneralOdeSystem {
    coefficients: RationalMatrix,
    forcing: PolynomialVector,
    time: Symbol,
    initial_point: BigRational,
    initial_values: Vec<Expr>,
    certificate: GeneralOdeSystemCertificate,
    fundamental: ExpressionMatrix,
    particular: Vec<Expr>,
    solutions: Vec<Expr>,
}

impl VerifiedGeneralOdeSystem {
    pub fn coefficients(&self) -> &[Vec<BigRational>] {
        &self.coefficients
    }
    pub fn forcing(&self) -> &[Vec<BigRational>] {
        &self.forcing
    }
    pub fn time(&self) -> &Symbol {
        &self.time
    }
    pub fn initial_point(&self) -> &BigRational {
        &self.initial_point
    }
    pub fn initial_values(&self) -> &[Expr] {
        &self.initial_values
    }
    pub fn certificate(&self) -> &GeneralOdeSystemCertificate {
        &self.certificate
    }
    pub fn fundamental(&self) -> &[Vec<Expr>] {
        &self.fundamental
    }
    pub fn particular(&self) -> &[Expr] {
        &self.particular
    }
    pub fn solutions(&self) -> &[Expr] {
        &self.solutions
    }
    pub fn into_solutions(self) -> Vec<Expr> {
        self.solutions
    }

    /// Evaluate the accepted fundamental matrix at an exact absolute time.
    ///
    /// Only the checked block coefficients are rendered. Rational Horner
    /// evaluation and governed radicals avoid generic expression substitution.
    /// A caller may pass its existing verifier meter to charge this consumer
    /// to the same cumulative ledger as generation and verification.
    pub fn fundamental_at(
        &self,
        point: &BigRational,
        meter: &mut impl BudgetMeter,
    ) -> Result<ExpressionMatrix> {
        let mut capped = Capped {
            inner: meter,
            used: [0; 3],
        };
        let mut work = Work { meter: &mut capped };
        work.meter.checkpoint()?;
        validate_number(point, MAX_INPUT_BITS)?;
        let local = work.sub(point, &self.initial_point)?;
        let n = self.coefficients.len();
        work.charge(n * n + n, 256 * (n * n + n))?;
        let mut result = Vec::with_capacity(n);
        if local.is_zero() {
            // The complete initial matrix was checked independently.
            for row in 0..n {
                result.push(
                    (0..n)
                        .map(|column| Expr::from_i64(i64::from(row == column)))
                        .collect(),
                );
            }
        } else {
            let coordinate = rational_expr(local);
            for row in &self.certificate.fundamental {
                let mut values = Vec::with_capacity(n);
                for entry in row {
                    values.push(render(entry, &coordinate, 0, &mut work)?.0);
                }
                result.push(values);
            }
        }
        work.meter.checkpoint()?;
        Ok(result)
    }
}

fn q(value: usize) -> BigRational {
    // Callers use admitted dimensions or polynomial degrees, all bounded by 32.
    BigRational::from_integer((value as u64).into())
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate_number(value: &BigRational, limit: u64) -> Result<()> {
    if !value.denom().is_positive() || bits(value) > limit {
        return Err(GeneralOdeSystemError::ResourceLimit(
            "rational coefficient height or denominator",
        ));
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
            (Dimension::AllocationCount, steps.max(1) as u64),
        ])?;
        Ok(())
    }

    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        validate_number(value, MAX_DERIVED_BITS)?;
        self.charge(1, 128 + 2 * bits(value).div_ceil(8) as usize)?;
        Ok(value.clone())
    }

    fn preflight(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        self.meter.checkpoint()?;
        if bits(a).saturating_add(bits(b)).saturating_add(2) > MAX_DERIVED_BITS {
            return Err(GeneralOdeSystemError::ResourceLimit(
                "intermediate rational coefficient growth",
            ));
        }
        Ok(())
    }

    fn finish(&mut self, value: BigRational) -> Result<BigRational> {
        validate_number(&value, MAX_DERIVED_BITS)?;
        self.charge(1, 128 + 2 * bits(&value).div_ceil(8) as usize)?;
        Ok(value)
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        let value = a.metered_add(b, self.meter)?;
        self.finish(value)
    }

    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        let value = a.metered_sub(b, self.meter)?;
        self.finish(value)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        let value = a.metered_mul(b, self.meter)?;
        self.finish(value)
    }

    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        let value = a.metered_div(b, self.meter)?;
        self.finish(value)
    }

    fn zeros(&mut self, length: usize) -> Result<Vec<BigRational>> {
        self.charge(length, length * 128 + 64)?;
        Ok(vec![BigRational::zero(); length])
    }

    fn matrix(&mut self, dimension: usize) -> Result<RationalMatrix> {
        self.charge(dimension, dimension * 64)?;
        (0..dimension).map(|_| self.zeros(dimension)).collect()
    }
}

fn admit_source(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    forcing: &[Vec<BigRational>],
    initial: &[Expr],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    work.meter.checkpoint()?;
    let n = matrix.len();
    if !(1..=MAX_DIMENSION).contains(&n)
        || matrix.iter().any(|row| row.len() != n)
        || forcing.len() != n
        || initial.len() != n
    {
        return Err(GeneralOdeSystemError::InvalidSource(
            "square dimension one through eight and matching coordinate counts",
        ));
    }
    if time.name.len() > MAX_TEXT {
        return Err(GeneralOdeSystemError::ResourceLimit("time symbol text"));
    }
    validate_number(point, MAX_INPUT_BITS)?;
    for value in matrix.iter().flatten() {
        validate_number(value, MAX_INPUT_BITS)?;
    }
    for polynomial in forcing {
        if polynomial.is_empty()
            || polynomial.len() > MAX_FORCING_DEGREE + 1
            || (polynomial.len() > 1 && polynomial.last().is_some_and(Zero::is_zero))
        {
            return Err(GeneralOdeSystemError::InvalidSource(
                "canonical forcing coefficients of degree at most sixteen",
            ));
        }
        for value in polynomial {
            validate_number(value, MAX_INPUT_BITS)?;
        }
    }
    for (index, value) in initial.iter().enumerate() {
        match value {
            Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => {}
            Expr::Rational(value) => validate_number(value, MAX_INPUT_BITS)?,
            Expr::Sym(symbol)
                if symbol != time
                    && symbol.name.len() <= MAX_TEXT
                    && !initial[..index].iter().any(
                        |previous| matches!(previous, Expr::Sym(other) if other == symbol),
                    ) => {}
            _ => {
                return Err(GeneralOdeSystemError::InvalidSource(
                    "initial values require rationals or distinct fresh constant symbols",
                ));
            }
        }
    }
    work.charge(
        n * n + n * 20,
        4096 + 512 * (n * n + n * 20) + (n + 1) * MAX_TEXT,
    )?;
    Ok(n)
}

fn internal_symbol(
    prefix: &str,
    time: &Symbol,
    initial: &[Expr],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Symbol> {
    for suffix in 0..=MAX_DIMENSION + 1 {
        work.charge(1, 128)?;
        let name = format!("{prefix}_{suffix}");
        if name != time.name
            && !initial
                .iter()
                .any(|value| matches!(value, Expr::Sym(s) if s.name == name))
        {
            return Ok(Symbol::new(name));
        }
    }
    Err(GeneralOdeSystemError::ResourceLimit(
        "fresh internal symbol",
    ))
}

fn multiply(
    a: &[Vec<BigRational>],
    b: &[Vec<BigRational>],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalMatrix> {
    let n = a.len();
    let mut result = work.matrix(n)?;
    for (row, values) in result.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            for (index, coefficient) in a[row].iter().enumerate() {
                if coefficient.is_zero() || b[index][column].is_zero() {
                    continue;
                }
                let term = work.mul(coefficient, &b[index][column])?;
                *value = work.add(value, &term)?;
            }
        }
    }
    Ok(result)
}

/// Candidate adjugate and determinant, in ascending powers of s.
fn resolvent(
    matrix: &[Vec<BigRational>],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(PolynomialMatrix, Vec<BigRational>)> {
    let n = matrix.len();
    work.charge(n * n * n, n * n * n * 128 + 512)?;
    let mut numerator = vec![vec![vec![BigRational::zero(); n]; n]; n];
    let mut denominator = work.zeros(n + 1)?;
    denominator[n] = BigRational::one();
    let mut b = work.matrix(n)?;
    for (index, row) in b.iter_mut().enumerate() {
        row[index] = BigRational::one();
    }
    for step in 1..=n {
        for (row, entries) in b.iter().enumerate() {
            for (column, value) in entries.iter().enumerate() {
                numerator[row][column][n - step] = work.copy(value)?;
            }
        }
        let mut next = multiply(matrix, &b, work)?;
        let mut trace = BigRational::zero();
        for (index, row) in next.iter().enumerate() {
            trace = work.add(&trace, &row[index])?;
        }
        let negative_trace = work.sub(&BigRational::zero(), &trace)?;
        let coefficient = work.div(&negative_trace, &q(step))?;
        denominator[n - step] = work.copy(&coefficient)?;
        for (index, row) in next.iter_mut().enumerate() {
            row[index] = work.add(&row[index], &coefficient)?;
        }
        b = next;
    }
    Ok((numerator, denominator))
}

/// Independently multiply (s I - A) by every proposed adjugate column.
fn check_resolvent(
    matrix: &[Vec<BigRational>],
    numerator: &PolynomialMatrix,
    denominator: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let n = matrix.len();
    for (row, matrix_row) in matrix.iter().enumerate() {
        for (column, polynomial) in numerator[row].iter().enumerate() {
            for (degree, expected_diagonal) in denominator.iter().enumerate() {
                let mut coefficient = if degree == 0 {
                    BigRational::zero()
                } else {
                    work.copy(&polynomial[degree - 1])?
                };
                if degree < n {
                    for (index, value) in matrix_row.iter().enumerate() {
                        let term = work.mul(value, &numerator[index][column][degree])?;
                        coefficient = work.sub(&coefficient, &term)?;
                    }
                }
                let valid = if row == column {
                    &coefficient == expected_diagonal
                } else {
                    coefficient.is_zero()
                };
                if !valid {
                    return Err(GeneralOdeSystemError::Verification(
                        "complete resolvent polynomial identity",
                    ));
                }
            }
        }
    }
    work.meter.checkpoint()?;
    Ok(())
}

/// Generator translation by the binomial theorem.
fn shift_forcing(
    forcing: &[Vec<BigRational>],
    point: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<PolynomialVector> {
    let maximum = forcing.iter().map(Vec::len).max().unwrap_or(1);
    let mut powers = work.zeros(maximum)?;
    powers[0] = BigRational::one();
    for degree in 1..maximum {
        powers[degree] = work.mul(&powers[degree - 1], point)?;
    }
    work.charge(forcing.len(), forcing.len() * 64)?;
    let mut shifted = Vec::with_capacity(forcing.len());
    for original in forcing {
        let mut result = work.zeros(original.len())?;
        for (degree, coefficient) in original.iter().enumerate() {
            let mut binomial = BigRational::one();
            for (index, value) in result.iter_mut().enumerate().take(degree + 1) {
                let scaled = work.mul(coefficient, &binomial)?;
                let term = work.mul(&scaled, &powers[degree - index])?;
                *value = work.add(value, &term)?;
                if index < degree {
                    let next = work.mul(&binomial, &q(degree - index))?;
                    binomial = work.div(&next, &q(index + 1))?;
                }
            }
        }
        shifted.push(result);
    }
    Ok(shifted)
}

/// The checker independently translates the original polynomial by Horner
/// multiplication, without using the generator's binomial table.
fn check_shifted_forcing(
    forcing: &[Vec<BigRational>],
    point: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<PolynomialVector> {
    work.charge(forcing.len(), forcing.len() * 64)?;
    let mut shifted = Vec::with_capacity(forcing.len());
    for original in forcing {
        let mut result = work.zeros(1)?;
        for coefficient in original.iter().rev() {
            let mut next = work.zeros(result.len() + 1)?;
            for (degree, value) in result.iter().enumerate() {
                let scaled = work.mul(value, point)?;
                next[degree] = work.add(&next[degree], &scaled)?;
                next[degree + 1] = work.add(&next[degree + 1], value)?;
            }
            next[0] = work.add(&next[0], coefficient)?;
            while next.len() > 1 && next.last().is_some_and(Zero::is_zero) {
                next.pop();
            }
            result = next;
        }
        shifted.push(result);
    }
    Ok(shifted)
}

fn transform_polynomial(
    coefficients: &[BigRational],
    variable: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    for value in coefficients {
        validate_number(value, MAX_TRANSFORM_BITS)?;
    }
    let nodes = 1 + 6 * coefficients.len();
    work.charge(nodes * nodes, nodes * (1536 + variable.name.len()))?;
    Ok(sum(coefficients
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_zero())
        .map(|(degree, value)| {
            let power = match degree {
                0 => Expr::from_i64(1),
                1 => Expr::Sym(variable.clone()),
                _ => Expr::Pow(
                    Arc::new(Expr::Sym(variable.clone())),
                    Arc::new(Expr::from_i64(degree as i64)),
                ),
            };
            product(vec![rational_expr(value.clone()), power])
        })
        .collect()))
}

fn transform_entry(
    numerator: &[BigRational],
    denominator: &[BigRational],
    variable: &Symbol,
    local_time: &Symbol,
    generator: &mut Work<'_, impl BudgetMeter>,
    verifier: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalLaplaceCertificate> {
    if numerator.iter().all(Zero::is_zero) {
        generator.charge(1, 64)?;
        return Ok(RationalLaplaceCertificate { blocks: Vec::new() });
    }
    let top = transform_polynomial(numerator, variable, generator)?;
    let bottom = transform_polynomial(denominator, variable, generator)?;
    generator.charge(4, 512)?;
    let source = product(vec![
        top,
        Expr::Pow(Arc::new(bottom), Arc::new(Expr::from_i64(-1))),
    ]);
    // A monomial denominator has a complete polynomial inverse table. Avoid
    // factoring the same high power of s and solving a dense partial-fraction
    // system for every nilpotent entry. The independent checker still sees
    // the entire original internal fraction, including uncancelled powers.
    let checked = if let Some(candidate) = monomial_inverse(numerator, denominator, generator)? {
        verify_rational_laplace_certificate(
            &source,
            variable,
            local_time,
            &candidate,
            &mut *verifier.meter,
        )?
    } else if let Some((top, bottom)) =
        cancelled_transform_fraction(numerator, denominator, generator)?
    {
        // A matrix entry can have a much smaller denominator than the full
        // characteristic polynomial. Search on that exact reduced fraction,
        // but accept only against the entire original source above. In
        // particular, its original excluded points are not replaced by the
        // reduced fraction's domain or convergence bound.
        let reduced_top = transform_polynomial(&top, variable, generator)?;
        let reduced_bottom = transform_polynomial(&bottom, variable, generator)?;
        generator.charge(4, 512)?;
        let reduced_source = product(vec![
            reduced_top,
            Expr::Pow(Arc::new(reduced_bottom), Arc::new(Expr::from_i64(-1))),
        ]);
        let proposed = metered_inverse_rational_laplace(
            &reduced_source,
            variable,
            local_time,
            &mut *generator.meter,
            &mut *verifier.meter,
        )?;
        verify_rational_laplace_certificate(
            &source,
            variable,
            local_time,
            proposed.certificate(),
            &mut *verifier.meter,
        )?
    } else {
        metered_inverse_rational_laplace(
            &source,
            variable,
            local_time,
            &mut *generator.meter,
            &mut *verifier.meter,
        )?
    };
    // The transform's own bounded block envelope is at most 32 rational
    // coordinates, each no wider than 65,536 bits. Our verifier applies its
    // stricter coefficient bound before using or publishing this copy.
    generator.charge(64, 64 * 16_512)?;
    Ok(checked.certificate().clone())
}

fn trim_transform_polynomial(values: &mut Vec<BigRational>) {
    while values.len() > 1 && values.last().is_some_and(Zero::is_zero) {
        values.pop();
    }
}

fn copy_transform_polynomial(
    values: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<BigRational>> {
    let mut copy = work.zeros(values.len())?;
    for (target, source) in copy.iter_mut().zip(values) {
        *target = work.copy(source)?;
    }
    trim_transform_polynomial(&mut copy);
    Ok(copy)
}

/// Exact dense Euclidean division for the generator's bounded resolvent data.
/// Every rational operation and transient coefficient vector uses the caller's
/// existing ledger. The independent transform verifier does not use this code.
fn divide_transform_polynomials(
    numerator: &[BigRational],
    denominator: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<PolynomialFraction> {
    let mut remainder = copy_transform_polynomial(numerator, work)?;
    if denominator.is_empty() || denominator.last().is_some_and(Zero::is_zero) {
        return Err(GeneralOdeSystemError::Verification(
            "noncanonical internal transform divisor",
        ));
    }
    let mut quotient = work.zeros(remainder.len().saturating_sub(denominator.len()) + 1)?;
    while remainder.len() >= denominator.len() && !remainder.iter().all(Zero::is_zero) {
        let previous = remainder.len();
        let shift = previous - denominator.len();
        let coefficient = work.div(
            remainder.last().expect("nonempty internal polynomial"),
            denominator.last().expect("admitted nonzero divisor"),
        )?;
        quotient[shift] = work.copy(&coefficient)?;
        for (index, value) in denominator.iter().enumerate() {
            let multiple = work.mul(&coefficient, value)?;
            remainder[shift + index] = work.sub(&remainder[shift + index], &multiple)?;
        }
        trim_transform_polynomial(&mut remainder);
        if !remainder.iter().all(Zero::is_zero) && remainder.len() >= previous {
            return Err(GeneralOdeSystemError::Verification(
                "internal transform division did not decrease degree",
            ));
        }
    }
    trim_transform_polynomial(&mut quotient);
    work.meter.checkpoint()?;
    Ok((quotient, remainder))
}

/// Cancel exact common factors before searching for inverse-transform modes.
/// This is an optimization of candidate generation only: the original fraction
/// is retained by transform_entry and independently checked after generation.
fn cancelled_transform_fraction(
    numerator: &[BigRational],
    denominator: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Option<PolynomialFraction>> {
    work.charge(numerator.len() + denominator.len(), 1)?;
    // Linear/quadratic denominators already use a direct exact table. A
    // nonzero constant numerator cannot have a nonconstant common factor.
    if denominator
        .iter()
        .rposition(|value| !value.is_zero())
        .unwrap_or(0)
        <= 2
        || numerator
            .iter()
            .rposition(|value| !value.is_zero())
            .unwrap_or(0)
            == 0
    {
        return Ok(None);
    }
    let mut left = copy_transform_polynomial(numerator, work)?;
    let mut right = copy_transform_polynomial(denominator, work)?;
    while !right.iter().all(Zero::is_zero) {
        let (_, remainder) = divide_transform_polynomials(&left, &right, work)?;
        left = right;
        right = remainder;
    }
    if left.len() == 1 {
        return Ok(None);
    }
    let (top, top_remainder) = divide_transform_polynomials(numerator, &left, work)?;
    let (bottom, bottom_remainder) = divide_transform_polynomials(denominator, &left, work)?;
    if !top_remainder.iter().all(Zero::is_zero) || !bottom_remainder.iter().all(Zero::is_zero) {
        return Err(GeneralOdeSystemError::Verification(
            "inexact internal transform cancellation",
        ));
    }
    work.meter.checkpoint()?;
    Ok(Some((top, bottom)))
}

/// Untrusted inverse of N(s)/(c*s^m), checked by the separate complete
/// rational transform verifier before use. Each a_j*s^j contributes
/// a_j*t^(m-j-1)/(c*(m-j-1)!). This is a class-wide exact table, independent
/// of matrix dimension, triangular structure, forcing sparsity or fixtures.
fn monomial_inverse(
    numerator: &[BigRational],
    denominator: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Option<RationalLaplaceCertificate>> {
    work.charge(numerator.len() + denominator.len(), 1)?;
    let Some(degree) = denominator.iter().rposition(|value| !value.is_zero()) else {
        return Err(GeneralOdeSystemError::Verification(
            "zero internal transform denominator",
        ));
    };
    if degree == 0
        || degree > MAX_BLOCK_CELLS
        || denominator[..degree].iter().any(|value| !value.is_zero())
        || numerator.iter().skip(degree).any(|value| !value.is_zero())
    {
        return Ok(None);
    }
    let mut factorials = work.zeros(degree)?;
    factorials[0] = work.copy(&denominator[degree])?;
    for index in 1..degree {
        factorials[index] = work.mul(&factorials[index - 1], &q(index))?;
    }
    let mut coefficients = work.zeros(degree)?;
    for (index, value) in numerator.iter().enumerate().take(degree) {
        if !value.is_zero() {
            let power = degree - index - 1;
            coefficients[power] = work.div(value, &factorials[power])?;
        }
    }
    while coefficients.len() > 1 && coefficients.last().is_some_and(Zero::is_zero) {
        coefficients.pop();
    }
    work.charge(1, 128)?;
    let blocks = if coefficients.iter().all(Zero::is_zero) {
        Vec::new()
    } else {
        vec![LaplaceBlock::Exponential {
            rate: BigRational::zero(),
            coefficients,
        }]
    };
    work.meter.checkpoint()?;
    Ok(Some(RationalLaplaceCertificate { blocks }))
}

fn generate(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    forcing: &[Vec<BigRational>],
    initial: &[Expr],
    generator: &mut Work<'_, impl BudgetMeter>,
    verifier: &mut Work<'_, impl BudgetMeter>,
) -> Result<GeneralOdeSystemCertificate> {
    let n = matrix.len();
    let (numerator, denominator) = resolvent(matrix, generator)?;
    check_resolvent(matrix, &numerator, &denominator, verifier)?;
    let variable = internal_symbol("_fsym_ode_transform", time, initial, generator)?;
    let local_time = internal_symbol("_fsym_ode_local", time, initial, generator)?;
    generator.charge(n * n + n, 128 * (n * n + n))?;
    let mut fundamental = Vec::with_capacity(n);
    for row in &numerator {
        let mut entries = Vec::with_capacity(n);
        for polynomial in row {
            entries.push(transform_entry(
                polynomial,
                &denominator,
                &variable,
                &local_time,
                generator,
                verifier,
            )?);
        }
        fundamental.push(entries);
    }

    let shifted = shift_forcing(forcing, point, generator)?;
    let maximum = shifted.iter().map(Vec::len).max().unwrap_or(1);
    let mut factorials = generator.zeros(maximum)?;
    factorials[0] = BigRational::one();
    for degree in 1..maximum {
        factorials[degree] = generator.mul(&factorials[degree - 1], &q(degree))?;
    }
    generator.charge(n, n * 64)?;
    let mut forcing_numerator = Vec::with_capacity(n);
    for polynomial in &shifted {
        let mut values = generator.zeros(maximum)?;
        for (degree, coefficient) in polynomial.iter().enumerate() {
            values[maximum - 1 - degree] = generator.mul(coefficient, &factorials[degree])?;
        }
        forcing_numerator.push(values);
    }
    let mut particular_denominator = generator.zeros(denominator.len() + maximum)?;
    for (degree, value) in denominator.iter().enumerate() {
        particular_denominator[degree + maximum] = generator.copy(value)?;
    }
    let mut particular = Vec::with_capacity(n);
    for row in &numerator {
        let mut values = generator.zeros(n + maximum - 1)?;
        for (polynomial, forcing_polynomial) in row.iter().zip(&forcing_numerator) {
            for (left, a) in polynomial.iter().enumerate() {
                if a.is_zero() {
                    continue;
                }
                for (right, b) in forcing_polynomial.iter().enumerate() {
                    if b.is_zero() {
                        continue;
                    }
                    let term = generator.mul(a, b)?;
                    values[left + right] = generator.add(&values[left + right], &term)?;
                }
            }
        }
        particular.push(transform_entry(
            &values,
            &particular_denominator,
            &variable,
            &local_time,
            generator,
            verifier,
        )?);
    }
    generator.meter.checkpoint()?;
    Ok(GeneralOdeSystemCertificate {
        fundamental,
        particular,
    })
}

fn admit_polynomial(values: &[BigRational]) -> Result<()> {
    if values.is_empty()
        || values.len() > MAX_BLOCK_CELLS
        || (values.len() > 1 && values.last().is_some_and(Zero::is_zero))
    {
        return Err(GeneralOdeSystemError::Verification(
            "noncanonical block polynomial",
        ));
    }
    for value in values {
        validate_number(value, MAX_DERIVED_BITS)?;
    }
    Ok(())
}

fn admit_blocks(
    certificate: &RationalLaplaceCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    if certificate.blocks.len() > MAX_BLOCK_CELLS {
        return Err(GeneralOdeSystemError::ResourceLimit(
            "too many solution blocks",
        ));
    }
    let mut cells = 0;
    for block in &certificate.blocks {
        work.meter.checkpoint()?;
        match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                validate_number(rate, MAX_DERIVED_BITS)?;
                admit_polynomial(coefficients)?;
                if coefficients.iter().all(Zero::is_zero) {
                    return Err(GeneralOdeSystemError::Verification("zero solution block"));
                }
                cells += coefficients.len();
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                validate_number(rate, MAX_DERIVED_BITS)?;
                validate_number(squared_frequency, MAX_DERIVED_BITS)?;
                if squared_frequency.is_zero() {
                    return Err(GeneralOdeSystemError::Verification(
                        "zero quadratic frequency",
                    ));
                }
                admit_polynomial(cosine)?;
                admit_polynomial(sine)?;
                if cosine.iter().chain(sine).all(Zero::is_zero) {
                    return Err(GeneralOdeSystemError::Verification("zero solution block"));
                }
                cells += 2 * cosine.len().max(sine.len());
            }
        }
        if cells > MAX_BLOCK_CELLS {
            return Err(GeneralOdeSystemError::ResourceLimit(
                "solution block coefficient dimension",
            ));
        }
    }
    work.charge(
        cells + 1,
        256 + (cells + 2 * certificate.blocks.len()) * 4224,
    )?;
    Ok(cells)
}

fn admit_certificate(
    candidate: &GeneralOdeSystemCertificate,
    dimension: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.meter.checkpoint()?;
    if candidate.fundamental.len() != dimension
        || candidate
            .fundamental
            .iter()
            .any(|row| row.len() != dimension)
        || candidate.particular.len() != dimension
    {
        return Err(GeneralOdeSystemError::Verification(
            "fundamental and particular coordinate counts",
        ));
    }
    for certificate in candidate
        .fundamental
        .iter()
        .flatten()
        .chain(&candidate.particular)
    {
        admit_blocks(certificate, work)?;
    }
    Ok(())
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Basis {
    Exponential(BigRational),
    Quadratic(BigRational, BigRational),
}

type BlockMap = BTreeMap<Basis, [Vec<BigRational>; 2]>;

fn accumulate(
    target: &mut BlockMap,
    basis: Basis,
    first: &[BigRational],
    second: &[BigRational],
    scale: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    if scale.is_zero() {
        return Ok(());
    }
    let length = first.len().max(second.len());
    let entry = match target.entry(basis) {
        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::btree_map::Entry::Vacant(entry) => {
            work.charge(1, 8192)?;
            let coefficients = [work.zeros(length)?, work.zeros(length)?];
            entry.insert(coefficients)
        }
    };
    for (values, source) in entry.iter_mut().zip([first, second]) {
        if values.len() < length {
            work.charge(length - values.len(), 128 * (length - values.len()))?;
            values.resize(length, BigRational::zero());
        }
        for (index, coefficient) in source.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let scaled = work.mul(coefficient, scale)?;
            values[index] = work.add(&values[index], &scaled)?;
        }
    }
    Ok(())
}

fn add_blocks(
    target: &mut BlockMap,
    certificate: &RationalLaplaceCertificate,
    scale: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    if scale.is_zero() {
        return Ok(());
    }
    let zero = [BigRational::zero()];
    for block in &certificate.blocks {
        match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                let basis = Basis::Exponential(work.copy(rate)?);
                accumulate(target, basis, coefficients, &zero, scale, work)?;
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                let basis = Basis::Quadratic(work.copy(rate)?, work.copy(squared_frequency)?);
                accumulate(target, basis, cosine, sine, scale, work)?;
            }
        }
    }
    Ok(())
}

fn polynomial_derivative(
    polynomial: &[BigRational],
    rate: &BigRational,
    length: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<BigRational>> {
    let mut result = work.zeros(length)?;
    for (degree, value) in result.iter_mut().enumerate() {
        if let Some(coefficient) = polynomial.get(degree) {
            *value = work.mul(rate, coefficient)?;
        }
        if let Some(coefficient) = polynomial.get(degree + 1) {
            let term = work.mul(&q(degree + 1), coefficient)?;
            *value = work.add(value, &term)?;
        }
    }
    Ok(result)
}

fn add_derivative(
    target: &mut BlockMap,
    certificate: &RationalLaplaceCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let zero = [BigRational::zero()];
    let one = BigRational::one();
    for block in &certificate.blocks {
        match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                let derivative =
                    polynomial_derivative(coefficients, rate, coefficients.len(), work)?;
                let basis = Basis::Exponential(work.copy(rate)?);
                accumulate(target, basis, &derivative, &zero, &one, work)?;
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                let length = cosine.len().max(sine.len());
                let mut first = polynomial_derivative(cosine, rate, length, work)?;
                let mut second = polynomial_derivative(sine, rate, length, work)?;
                for (degree, value) in first.iter_mut().enumerate() {
                    if let Some(coefficient) = sine.get(degree) {
                        *value = work.add(value, coefficient)?;
                    }
                }
                for (degree, value) in second.iter_mut().enumerate() {
                    if let Some(coefficient) = cosine.get(degree) {
                        let term = work.mul(squared_frequency, coefficient)?;
                        *value = work.sub(value, &term)?;
                    }
                }
                let basis = Basis::Quadratic(work.copy(rate)?, work.copy(squared_frequency)?);
                accumulate(target, basis, &first, &second, &one, work)?;
            }
        }
    }
    Ok(())
}

fn initial_value(
    certificate: &RationalLaplaceCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let mut result = BigRational::zero();
    for block in &certificate.blocks {
        let value = match block {
            LaplaceBlock::Exponential { coefficients, .. } => &coefficients[0],
            LaplaceBlock::Quadratic { cosine, .. } => &cosine[0],
        };
        result = work.add(&result, value)?;
    }
    Ok(result)
}

fn all_zero(values: &BlockMap) -> bool {
    values.values().flatten().flatten().all(Zero::is_zero)
}

/// Coefficient identities suffice even if a candidate repeats basis keys.
/// No linear-independence assertion about unequal basis keys is needed.
fn check_identities(
    matrix: &[Vec<BigRational>],
    point: &BigRational,
    forcing: &[Vec<BigRational>],
    candidate: &GeneralOdeSystemCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let shifted = check_shifted_forcing(forcing, point, work)?;
    for (row, matrix_row) in matrix.iter().enumerate() {
        for (column, entry) in candidate.fundamental[row].iter().enumerate() {
            let initial = initial_value(entry, work)?;
            if initial != q(usize::from(row == column)) {
                return Err(GeneralOdeSystemError::Verification(
                    "fundamental initial matrix",
                ));
            }
            let mut residual = BlockMap::new();
            add_derivative(&mut residual, entry, work)?;
            for (index, coefficient) in matrix_row.iter().enumerate() {
                let negative = work.sub(&BigRational::zero(), coefficient)?;
                add_blocks(
                    &mut residual,
                    &candidate.fundamental[index][column],
                    &negative,
                    work,
                )?;
            }
            if !all_zero(&residual) {
                return Err(GeneralOdeSystemError::Verification(
                    "fundamental differential identity",
                ));
            }
        }
        if !initial_value(&candidate.particular[row], work)?.is_zero() {
            return Err(GeneralOdeSystemError::Verification(
                "particular initial vector",
            ));
        }
        let mut residual = BlockMap::new();
        add_derivative(&mut residual, &candidate.particular[row], work)?;
        for (index, coefficient) in matrix_row.iter().enumerate() {
            let negative = work.sub(&BigRational::zero(), coefficient)?;
            add_blocks(&mut residual, &candidate.particular[index], &negative, work)?;
        }
        accumulate(
            &mut residual,
            Basis::Exponential(BigRational::zero()),
            &shifted[row],
            &[BigRational::zero()],
            &-BigRational::one(),
            work,
        )?;
        if !all_zero(&residual) {
            return Err(GeneralOdeSystemError::Verification(
                "particular differential identity",
            ));
        }
    }
    work.meter.checkpoint()?;
    Ok(())
}

/// Exact squares use governed integer arithmetic. Nonsquares remain held
/// radicals, so presentation cannot start an unmetered integer factor search.
fn frequency(
    magnitude: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(Expr, Expr)> {
    let numerator = metered_sqrt_floor(magnitude.numer(), work.meter)?.ok_or(
        GeneralOdeSystemError::Verification("negative frequency numerator"),
    )?;
    let denominator = metered_sqrt_floor(magnitude.denom(), work.meter)?.ok_or(
        GeneralOdeSystemError::Verification("negative frequency denominator"),
    )?;
    let numerator_square = metered_mul(&numerator, &numerator, work.meter)?;
    let denominator_square = metered_mul(&denominator, &denominator, work.meter)?;
    work.charge(8, 512 + 8 * bits(magnitude).div_ceil(8) as usize)?;
    if numerator_square == *magnitude.numer() && denominator_square == *magnitude.denom() {
        let value = BigRational::metered_new(&numerator, &denominator, work.meter)?;
        let inverse = work.div(&BigRational::one(), &value)?;
        return Ok((rational_expr(value), rational_expr(inverse)));
    }
    let value = Expr::Pow(
        Arc::new(rational_expr(work.copy(magnitude)?)),
        Arc::new(rational_expr(BigRational::new(1.into(), 2.into()))),
    );
    let inverse = Expr::Pow(Arc::new(value.clone()), Arc::new(Expr::from_i64(-1)));
    work.meter.checkpoint()?;
    Ok((value, inverse))
}

fn time_polynomial(
    coefficients: &[BigRational],
    local: &Expr,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    let point = match local {
        Expr::Integer(value) => {
            work.charge(1, 128 + 2 * value.bits().div_ceil(8) as usize)?;
            Some(BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value) => Some(work.copy(value)?),
        _ => None,
    };
    if let Some(point) = point {
        let mut value = BigRational::zero();
        for coefficient in coefficients.iter().rev() {
            let product = work.mul(&value, &point)?;
            value = work.add(&product, coefficient)?;
        }
        return Ok(rational_expr(value));
    }
    Ok(sum(coefficients
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_zero())
        .map(|(degree, value)| {
            let power = match degree {
                0 => Expr::from_i64(1),
                1 => local.clone(),
                _ => Expr::Pow(
                    Arc::new(local.clone()),
                    Arc::new(Expr::from_i64(degree as i64)),
                ),
            };
            product(vec![rational_expr(value.clone()), power])
        })
        .collect()))
}

/// Literal identities need no transcendental approximation or factor search.
fn basis_function(name: &str, argument: Expr) -> Expr {
    if argument.is_zero() {
        return Expr::from_i64(if name == "sin" || name == "sinh" {
            0
        } else {
            1
        });
    }
    if name == "exp" && argument.is_one() {
        return Expr::Const(Constant::E);
    }
    Expr::Function(name.into(), vec![argument])
}

/// Rendering uses exactly the basis already checked: s_d' = c_d and
/// c_d' = -d*s_d, for both signs of nonzero rational d.
fn render(
    certificate: &RationalLaplaceCertificate,
    local: &Expr,
    text: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(Expr, usize)> {
    let cells = admit_blocks(certificate, work)?;
    let nodes = 1 + 32 * cells + 24 * certificate.blocks.len();
    let node_bytes = 512 + 4 * MAX_DERIVED_BITS.div_ceil(8) as usize + 4 * text;
    work.charge(nodes * nodes, nodes * node_bytes)?;
    let mut terms = Vec::with_capacity(certificate.blocks.len());
    for block in &certificate.blocks {
        let (rate, inner) = match block {
            LaplaceBlock::Exponential { rate, coefficients } => {
                (rate, time_polynomial(coefficients, local, work)?)
            }
            LaplaceBlock::Quadratic {
                rate,
                squared_frequency,
                cosine,
                sine,
            } => {
                let magnitude = if squared_frequency.is_positive() {
                    work.copy(squared_frequency)?
                } else {
                    work.sub(&BigRational::zero(), squared_frequency)?
                };
                let (frequency, inverse) = frequency(&magnitude, work)?;
                let argument = product(vec![frequency, local.clone()]);
                let positive = squared_frequency.is_positive();
                let c = basis_function(if positive { "cos" } else { "cosh" }, argument.clone());
                let s = basis_function(if positive { "sin" } else { "sinh" }, argument);
                (
                    rate,
                    sum(vec![
                        product(vec![time_polynomial(cosine, local, work)?, c]),
                        product(vec![time_polynomial(sine, local, work)?, inverse, s]),
                    ]),
                )
            }
        };
        let term = if rate.is_zero() {
            inner
        } else {
            let argument = product(vec![rational_expr(rate.clone()), local.clone()]);
            product(vec![basis_function("exp", argument), inner])
        };
        terms.push(term);
    }
    work.meter.checkpoint()?;
    Ok((sum(terms), nodes))
}

#[allow(clippy::too_many_arguments)]
fn verify(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    forcing: &[Vec<BigRational>],
    initial: &[Expr],
    candidate: &GeneralOdeSystemCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedGeneralOdeSystem> {
    let n = admit_source(matrix, time, point, forcing, initial, work)?;
    admit_certificate(candidate, n, work)?;
    check_identities(matrix, point, forcing, candidate, work)?;
    let negative_point = work.sub(&BigRational::zero(), point)?;
    work.charge(4, 1024 + time.name.len())?;
    let local = sum(vec![Expr::Sym(time.clone()), rational_expr(negative_point)]);
    work.charge(n * n + 4 * n, (n * n + 4 * n) * 128)?;
    let mut fundamental = Vec::with_capacity(n);
    let mut counts = Vec::with_capacity(n);
    for row in &candidate.fundamental {
        let mut values = Vec::with_capacity(n);
        let mut sizes = Vec::with_capacity(n);
        for entry in row {
            let (value, size) = render(entry, &local, time.name.len(), work)?;
            values.push(value);
            sizes.push(size);
        }
        fundamental.push(values);
        counts.push(sizes);
    }
    let mut particular = Vec::with_capacity(n);
    let mut solutions = Vec::with_capacity(n);
    let initial_text = initial
        .iter()
        .filter_map(|value| match value {
            Expr::Sym(symbol) => Some(symbol.name.len()),
            _ => None,
        })
        .sum::<usize>();
    let node_bytes =
        512 + 4 * MAX_DERIVED_BITS.div_ceil(8) as usize + 4 * time.name.len() + initial_text;
    for (row, (entries, sizes)) in fundamental.iter().zip(&counts).enumerate() {
        let (part, size) = render(&candidate.particular[row], &local, time.name.len(), work)?;
        let nodes = size + 1 + sizes.iter().map(|size| size + 3).sum::<usize>();
        if nodes > 16_384 {
            return Err(GeneralOdeSystemError::ResourceLimit(
                "assembled solution nodes",
            ));
        }
        work.charge(nodes, nodes * node_bytes)?;
        let mut terms = Vec::with_capacity(n + 1);
        terms.push(part.clone());
        for (column, value) in initial.iter().enumerate() {
            if !value.is_zero() {
                terms.push(product(vec![entries[column].clone(), value.clone()]));
            }
        }
        particular.push(part);
        solutions.push(sum(terms));
    }
    // Borrowed admission prepaid all original-data and certificate copies.
    // Every output is assembled only from independently accepted blocks.
    work.meter.checkpoint()?;
    let receipt = VerifiedGeneralOdeSystem {
        coefficients: matrix.to_vec(),
        forcing: forcing.to_vec(),
        time: time.clone(),
        initial_point: point.clone(),
        initial_values: initial.to_vec(),
        certificate: candidate.clone(),
        fundamental,
        particular,
        solutions,
    };
    work.meter.checkpoint()?;
    Ok(receipt)
}

/// Independently admit and check an externally supplied coefficient candidate.
#[allow(clippy::too_many_arguments)]
pub fn verify_general_ode_system(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    forcing: &[Vec<BigRational>],
    initial: &[Expr],
    candidate: &GeneralOdeSystemCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedGeneralOdeSystem> {
    let mut capped = Capped {
        inner: verifier,
        used: [0; 3],
    };
    verify(
        matrix,
        time,
        point,
        forcing,
        initial,
        candidate,
        &mut Work { meter: &mut capped },
    )
}

/// Generate and independently check one complete affine family. Forcing is in
/// ascending powers of the original time variable, not the shifted coordinate.
#[allow(clippy::too_many_arguments)]
pub fn metered_general_ode_system(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    forcing: &[Vec<BigRational>],
    initial: &[Expr],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedGeneralOdeSystem> {
    let mut generation = Capped {
        inner: generator,
        used: [0; 3],
    };
    let mut verification = Capped {
        inner: verifier,
        used: [0; 3],
    };
    let mut generate_work = Work {
        meter: &mut generation,
    };
    let mut verify_work = Work {
        meter: &mut verification,
    };
    admit_source(matrix, time, point, forcing, initial, &mut generate_work)?;
    let candidate = generate(
        matrix,
        time,
        point,
        forcing,
        initial,
        &mut generate_work,
        &mut verify_work,
    )?;
    let result = verify(
        matrix,
        time,
        point,
        forcing,
        initial,
        &candidate,
        &mut verify_work,
    )?;
    generate_work.meter.checkpoint()?;
    verify_work.meter.checkpoint()?;
    Ok(result)
}

pub fn general_ode_system(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    forcing: &[Vec<BigRational>],
    initial: &[Expr],
) -> Result<VerifiedGeneralOdeSystem> {
    metered_general_ode_system(
        matrix,
        time,
        point,
        forcing,
        initial,
        &mut Unbounded,
        &mut Unbounded,
    )
}
