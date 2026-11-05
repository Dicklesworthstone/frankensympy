//! Bounded exact minimal polynomials of rational matrices.
//!
//! A generator interpolates successive Krylov columns vec(I), vec(A), ... on
//! an increasing set of matrix coordinates. A nonzero residual adds another
//! independent coordinate; the first zero residual gives a monic annihilator.
//!
//! The independent checker does not repeat that search. It reconstructs the
//! selected minor of the lower powers from the original matrix and checks a
//! supplied left inverse. This proves that no lower-degree polynomial can
//! annihilate the matrix. Separate reference Horner evaluation proves that the
//! proposed polynomial does annihilate it. Merely returning the characteristic
//! polynomial or checking an arbitrary annihilator cannot establish minimality.
//!
//! Inputs are square literal-rational matrices through dimension sixteen.
//! Input coefficients have at most 4096 bits and intermediate coefficients at
//! most 8192 bits. Generation and verification each retain a cumulative 200M
//! step / 512MiB logical-allocation envelope, including accepted publication.
//! This is an in-process checked mathematical result, not a new portable
//! proof-kernel certificate. No eigenvalue or factorization algorithm is used.

#![forbid(unsafe_code)]

use crate::{Matrix, MatrixError, SparseMatrix, numeric_value};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use num_traits::{One, Zero};

pub const MAX_MINIMAL_POLYNOMIAL_DIMENSION: usize = 16;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_BITS: u64 = 8192;
const MAX_STEPS: u64 = 200_000_000;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
type Result<T> = std::result::Result<T, MatrixError>;

/// Untrusted evidence for a monic minimal polynomial of degree d.
///
/// `coefficients` are ascending. `coordinate_rows` selects d distinct row-major
/// cells of the original matrix; column k of their minor contains those cells
/// of A^k, for 0 <= k < d. `minor_inverse` is its row-major d by d left inverse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimalPolynomialCandidate {
    pub coefficients: Vec<BigRational>,
    pub coordinate_rows: Vec<usize>,
    pub minor_inverse: Vec<BigRational>,
}

/// A monic polynomial proved to have the least possible annihilating degree.
/// The exact source and checked evidence remain immutable behind accessors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedMinimalPolynomial {
    matrix: Matrix,
    certificate: MinimalPolynomialCandidate,
}

impl VerifiedMinimalPolynomial {
    pub fn matrix(&self) -> &Matrix {
        &self.matrix
    }
    pub fn coefficients(&self) -> &[BigRational] {
        &self.certificate.coefficients
    }
    pub fn degree(&self) -> usize {
        self.certificate.coefficients.len() - 1
    }
    pub fn certificate(&self) -> &MinimalPolynomialCandidate {
        &self.certificate
    }
    pub fn into_coefficients(self) -> Vec<BigRational> {
        self.certificate.coefficients
    }
}

fn limit(reason: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("exact matrix minimal polynomial: {reason}"))
}
fn invalid(reason: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("exact matrix minimal polynomial: {reason}"))
}
fn stopped(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => MatrixError::ResourceLimit(error.to_string()),
    }
}
fn arithmetic(error: RationalArithmeticError) -> MatrixError {
    match error {
        RationalArithmeticError::Meter(error) => stopped(error),
        RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
        RationalArithmeticError::InvariantViolation(reason) => {
            MatrixError::ArithmeticInvariantViolation(reason)
        }
        error => MatrixError::ResourceLimit(error.to_string()),
    }
}
fn number_ok(value: &BigRational, maximum: u64) -> Result<()> {
    if value.numer().bits().max(value.denom().bits()) > maximum {
        return Err(limit("rational coefficient height"));
    }
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(invalid("invalid rational denominator"));
    }
    Ok(())
}

struct Work<'a, M> {
    meter: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
}
impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; 3],
            stopped: None,
        }
    }
    fn storage(&mut self, steps: usize, bytes: usize) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(stopped)
    }
    fn tick(&mut self, count: usize) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge(Dimension::ComputeSteps, count.max(1) as u64)
            .map_err(stopped)
    }
    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| limit("container size overflow"))?;
        self.storage(1, bytes)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| limit("container allocation refused"))?;
        Ok(result)
    }
    fn zero(&mut self) -> Result<BigRational> {
        self.storage(1, 64)?;
        Ok(BigRational::zero())
    }
    fn one(&mut self) -> Result<BigRational> {
        self.storage(1, 64)?;
        Ok(BigRational::one())
    }
    fn zeros(&mut self, count: usize) -> Result<Vec<BigRational>> {
        let mut values = self.vector(count)?;
        self.storage(count, count.saturating_mul(64))?;
        values.resize_with(count, BigRational::zero);
        Ok(values)
    }
    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        number_ok(value, MAX_BITS)?;
        self.storage(
            1,
            64 + 8 * (value.numer().limb_count() + value.denom().limb_count()) as usize,
        )?;
        Ok(value.clone())
    }
    fn equal(&mut self, left: &BigRational, right: &BigRational) -> Result<bool> {
        number_ok(left, MAX_BITS)?;
        number_ok(right, MAX_BITS)?;
        self.checkpoint().map_err(stopped)?;
        Ok(left.metered_cmp(right, self).map_err(arithmetic)?.is_eq())
    }
    fn combine(
        &mut self,
        left: &BigRational,
        right: &BigRational,
        operation: Operation,
    ) -> Result<BigRational> {
        number_ok(left, MAX_BITS)?;
        number_ok(right, MAX_BITS)?;
        let (an, ad, bn, bd) = (
            left.numer().bits(),
            left.denom().bits(),
            right.numer().bits(),
            right.denom().bits(),
        );
        let (numerator, denominator) = match operation {
            Operation::Add | Operation::Subtract => (
                an.saturating_add(bd)
                    .max(bn.saturating_add(ad))
                    .saturating_add(1),
                ad.saturating_add(bd),
            ),
            Operation::Multiply => (an.saturating_add(bn), ad.saturating_add(bd)),
            Operation::Divide => (an.saturating_add(bd), ad.saturating_add(bn)),
        };
        if numerator > MAX_BITS || denominator > MAX_BITS {
            return Err(limit("arithmetic intermediate coefficient height"));
        }
        self.tick(1)?;
        let value = match operation {
            Operation::Add => left.metered_add(right, self),
            Operation::Subtract => left.metered_sub(right, self),
            Operation::Multiply => left.metered_mul(right, self),
            Operation::Divide => left.metered_div(right, self),
        }
        .map_err(arithmetic)?;
        number_ok(&value, MAX_BITS)?;
        Ok(value)
    }
    fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Add)
    }
    fn subtract(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Subtract)
    }
    fn multiply(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Multiply)
    }
    fn divide(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Divide)
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
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
                let (index, maximum) = match dimension {
                    Dimension::ComputeSteps => (0, MAX_STEPS),
                    Dimension::MemoryBytes => (1, MAX_BYTES),
                    Dimension::AllocationCount => (2, MAX_STEPS),
                    _ => continue,
                };
                let remaining = maximum.saturating_sub(next[index]);
                if amount > remaining {
                    return Err(MeterError::Budget(BudgetError::Exhausted {
                        dimension,
                        requested: amount,
                        remaining,
                    }));
                }
                next[index] += amount;
            }
            self.meter.charge_batch(charges)?;
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
        let result = self.meter.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

fn shape(rows: usize, columns: usize) -> Result<usize> {
    if rows != columns {
        return Err(MatrixError::NotSquare(rows, columns));
    }
    if rows > MAX_MINIMAL_POLYNOMIAL_DIMENSION {
        return Err(limit("matrix dimension exceeds sixteen"));
    }
    Ok(rows)
}
fn input(value: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
    work.tick(1)?;
    match value {
        Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => Ok(()),
        Expr::Integer(_) => Err(limit("input integer exceeds 4096 bits")),
        Expr::Rational(value) => number_ok(value, MAX_INPUT_BITS),
        _ => Err(MatrixError::UnsupportedCertificateDomain),
    }
}
fn admit(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<usize> {
    work.tick(1)?;
    matrix.validate_shape()?;
    let n = shape(matrix.rows, matrix.cols)?;
    // Admit every original literal cell before a zero/scalar shortcut or copy.
    for value in &matrix.data {
        input(value, work)?;
    }
    Ok(n)
}
fn entry_copy(value: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Expr> {
    let limbs = match value {
        Expr::Integer(value) => value.limb_count() + 1,
        Expr::Rational(value) => value.numer().limb_count() + value.denom().limb_count(),
        _ => return Err(MatrixError::UnsupportedCertificateDomain),
    };
    work.storage(1, 128 + 8 * limbs as usize)?;
    Ok(value.clone())
}
fn scalar_value(value: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<BigRational> {
    let limbs = match value {
        Expr::Integer(value) => value.limb_count() + 1,
        Expr::Rational(value) => value.numer().limb_count() + value.denom().limb_count(),
        _ => return Err(MatrixError::UnsupportedCertificateDomain),
    };
    work.storage(1, 64 + 8 * limbs as usize)?;
    numeric_value(value).ok_or(MatrixError::UnsupportedCertificateDomain)
}
fn values(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Vec<BigRational>> {
    let mut result = work.vector(matrix.data.len())?;
    for value in &matrix.data {
        result.push(scalar_value(value, work)?);
    }
    Ok(result)
}
fn identity(n: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Vec<BigRational>> {
    let mut result = work.zeros(n * n)?;
    for i in 0..n {
        result[i * n + i] = work.one()?;
    }
    Ok(result)
}

// Generator multiplication accumulates scaled rows. The checker below uses
// a separate direct-dot traversal and never invokes this routine.
fn generated_product(
    left: &[BigRational],
    right: &[BigRational],
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<BigRational>> {
    let mut result = work.zeros(n * n)?;
    for row in 0..n {
        for inner in 0..n {
            for column in 0..n {
                let value = work.multiply(&left[row * n + inner], &right[inner * n + column])?;
                let position = row * n + column;
                result[position] = work.add(&result[position], &value)?;
            }
        }
    }
    Ok(result)
}

fn generate(
    matrix: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<MinimalPolynomialCandidate> {
    let n = admit(matrix, work)?;
    let zero = work.zero()?;
    let one = work.one()?;
    let mut coordinates = work.vector(n)?;
    let mut inverse = work.vector(n * n)?;
    if n == 0 {
        let mut coefficients = work.vector(1)?;
        coefficients.push(work.copy(&one)?);
        work.tick(1)?;
        return Ok(MinimalPolynomialCandidate {
            coefficients,
            coordinate_rows: coordinates,
            minor_inverse: inverse,
        });
    }
    let original = values(matrix, work)?;
    let mut powers = work.vector(n)?;
    powers.push(identity(n, work)?);
    coordinates.push(0); // The (0,0) cell of I is one.
    inverse.push(work.copy(&one)?);
    let mut current = work.vector(original.len())?;
    for value in &original {
        current.push(work.copy(value)?);
    }

    for degree in 1..=n {
        let mut interpolation = work.zeros(degree)?;
        for (row, coefficient) in interpolation.iter_mut().enumerate() {
            for (column, &coordinate) in coordinates.iter().enumerate() {
                let value = work.multiply(&inverse[row * degree + column], &current[coordinate])?;
                *coefficient = work.add(coefficient, &value)?;
            }
        }
        let mut pivot = None;
        for (coordinate, entry) in current.iter().enumerate() {
            let mut predicted = work.zero()?;
            for (column, coefficient) in interpolation.iter().enumerate() {
                let value = work.multiply(coefficient, &powers[column][coordinate])?;
                predicted = work.add(&predicted, &value)?;
            }
            let residual = work.subtract(entry, &predicted)?;
            if !residual.is_zero() {
                pivot = Some((coordinate, residual));
                break;
            }
        }
        let Some((coordinate, residual)) = pivot else {
            let mut coefficients = work.vector(degree + 1)?;
            for coefficient in &interpolation {
                coefficients.push(work.subtract(&zero, coefficient)?);
            }
            coefficients.push(work.copy(&one)?);
            work.tick(1)?;
            return Ok(MinimalPolynomialCandidate {
                coefficients,
                coordinate_rows: coordinates,
                minor_inverse: inverse,
            });
        };
        if degree == n || coordinates.contains(&coordinate) {
            return Err(MatrixError::ArithmeticInvariantViolation(
                "matrix Krylov interpolation did not produce an independent next coordinate",
            ));
        }

        // The nonzero residual is the Schur complement of the enlarged
        // selected minor. Extend its inverse without an n^2-row solve.
        let mut bottom = work.zeros(degree)?;
        for (column, value) in bottom.iter_mut().enumerate() {
            for (inner, power) in powers.iter().enumerate() {
                let term = work.multiply(&power[coordinate], &inverse[inner * degree + column])?;
                *value = work.add(value, &term)?;
            }
        }
        let reciprocal = work.divide(&one, &residual)?;
        let mut scaled = work.vector(degree)?;
        for coefficient in &interpolation {
            scaled.push(work.multiply(coefficient, &reciprocal)?);
        }
        let mut enlarged = work.vector((degree + 1) * (degree + 1))?;
        for (row, scaled_value) in scaled.iter().enumerate() {
            for (column, bottom_value) in bottom.iter().enumerate() {
                let correction = work.multiply(scaled_value, bottom_value)?;
                enlarged.push(work.add(&inverse[row * degree + column], &correction)?);
            }
            enlarged.push(work.subtract(&zero, scaled_value)?);
        }
        for value in &bottom {
            let scaled_value = work.multiply(value, &reciprocal)?;
            enlarged.push(work.subtract(&zero, &scaled_value)?);
        }
        enlarged.push(reciprocal);
        inverse = enlarged;
        coordinates.push(coordinate);
        powers.push(current);
        current = generated_product(&original, &powers[degree], n, work)?;
    }
    Err(MatrixError::ArithmeticInvariantViolation(
        "matrix Krylov search exceeded the dimension bound",
    ))
}

// Independent reference evaluation: direct scalar dot products, with no
// generator interpolation, pivot search, inverse update or matrix methods.
fn reference_dot<'a>(
    entries: impl Iterator<Item = (&'a BigRational, &'a BigRational)>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let mut value = work.zero()?;
    for (left, right) in entries {
        let term = work.multiply(left, right)?;
        value = work.add(&value, &term)?;
    }
    Ok(value)
}
fn reference_product(
    left: &[BigRational],
    right: &[BigRational],
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<BigRational>> {
    let mut result = work.vector(n * n)?;
    for row in 0..n {
        for column in 0..n {
            result.push(reference_dot(
                (0..n).map(|inner| (&left[row * n + inner], &right[inner * n + column])),
                work,
            )?);
        }
    }
    Ok(result)
}

fn verify(
    matrix: &Matrix,
    candidate: &MinimalPolynomialCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedMinimalPolynomial> {
    let n = admit(matrix, work)?;
    if candidate.coefficients.is_empty() || candidate.coefficients.len() > n + 1 {
        return Err(invalid(
            "polynomial degree must be between zero and the matrix dimension",
        ));
    }
    let degree = candidate.coefficients.len() - 1;
    if candidate.coordinate_rows.len() != degree || candidate.minor_inverse.len() != degree * degree
    {
        return Err(invalid("independence minor has the wrong dimension"));
    }
    for value in candidate
        .coefficients
        .iter()
        .chain(&candidate.minor_inverse)
    {
        work.tick(1)?;
        number_ok(value, MAX_BITS)?;
    }
    if !candidate.coefficients[degree].is_one() {
        return Err(invalid("the proposed polynomial is not monic"));
    }
    for (index, &coordinate) in candidate.coordinate_rows.iter().enumerate() {
        work.tick(degree + 1)?;
        if coordinate >= n * n || candidate.coordinate_rows[..index].contains(&coordinate) {
            return Err(invalid(
                "independence coordinates are out of range or repeated",
            ));
        }
    }
    let original = values(matrix, work)?;
    let mut power = identity(n, work)?;
    let mut minor = work.zeros(degree * degree)?;
    for column in 0..degree {
        for (row, &coordinate) in candidate.coordinate_rows.iter().enumerate() {
            minor[row * degree + column] = work.copy(&power[coordinate])?;
        }
        if column + 1 < degree {
            power = reference_product(&power, &original, n, work)?;
        }
    }
    // A left inverse of a square rational minor proves that all lower
    // powers are linearly independent. This is the minimality obligation.
    for row in 0..degree {
        for column in 0..degree {
            let value = reference_dot(
                (0..degree).map(|inner| {
                    (
                        &candidate.minor_inverse[row * degree + inner],
                        &minor[inner * degree + column],
                    )
                }),
                work,
            )?;
            if (row == column && !value.is_one()) || (row != column && !value.is_zero()) {
                return Err(invalid("lower matrix powers are not certified independent"));
            }
        }
    }

    // Separate Horner evaluation binds annihilation to every original cell,
    // including cells outside the selected independence coordinates.
    let mut evaluated = identity(n, work)?;
    for coefficient in candidate.coefficients.iter().rev().skip(1) {
        evaluated = reference_product(&original, &evaluated, n, work)?;
        for diagonal in 0..n {
            let position = diagonal * n + diagonal;
            evaluated[position] = work.add(&evaluated[position], coefficient)?;
        }
    }
    work.tick(evaluated.len())?;
    if evaluated.iter().any(|value| !value.is_zero()) {
        return Err(invalid(
            "the proposed polynomial does not annihilate the original matrix",
        ));
    }

    let mut entries = work.vector(matrix.data.len())?;
    for entry in &matrix.data {
        entries.push(entry_copy(entry, work)?);
    }
    let original_matrix = Matrix::new(n, n, entries)?;
    let mut coefficients = work.vector(candidate.coefficients.len())?;
    for coefficient in &candidate.coefficients {
        coefficients.push(work.copy(coefficient)?);
    }
    let mut coordinate_rows = work.vector(degree)?;
    work.tick(degree)?;
    coordinate_rows.extend_from_slice(&candidate.coordinate_rows);
    let mut minor_inverse = work.vector(candidate.minor_inverse.len())?;
    for value in &candidate.minor_inverse {
        minor_inverse.push(work.copy(value)?);
    }
    work.tick(1)?;
    Ok(VerifiedMinimalPolynomial {
        matrix: original_matrix,
        certificate: MinimalPolynomialCandidate {
            coefficients,
            coordinate_rows,
            minor_inverse,
        },
    })
}

/// Independently prove annihilation and least degree against the original matrix.
/// No generator, inverse solver, characteristic polynomial or eigensolver is used.
pub fn verify_minimal_polynomial_candidate(
    matrix: &Matrix,
    candidate: &MinimalPolynomialCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedMinimalPolynomial> {
    let mut work = Work::new(verifier);
    verify(matrix, candidate, &mut work)
}

impl Matrix {
    /// Monic QQ minimal polynomial, ascending coefficients; 0 by 0 returns one.
    pub fn minimal_polynomial(&self) -> Result<VerifiedMinimalPolynomial> {
        self.minimal_polynomial_with_meters(&mut Unbounded, &mut Unbounded)
    }

    /// Both phases keep their cumulative bounds through admission and publication.
    pub fn minimal_polynomial_with_meters(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedMinimalPolynomial> {
        let mut generation = Work::new(generator);
        let mut verification = Work::new(verifier);
        let candidate = generate(self, &mut generation)?;
        let result = verify(self, &candidate, &mut verification)?;
        generation.checkpoint().map_err(stopped)?;
        verification.checkpoint().map_err(stopped)?;
        Ok(result)
    }

    /// Generate untrusted evidence; only the independent checker accepts it.
    pub fn minimal_polynomial_candidate_with_meter(
        &self,
        generator: &mut impl BudgetMeter,
    ) -> Result<MinimalPolynomialCandidate> {
        generate(self, &mut Work::new(generator))
    }
}

fn sparse_dense(source: &SparseMatrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
    work.tick(1)?;
    let n = shape(source.rows(), source.cols())?;
    if source.entries().len() > n * n {
        return Err(invalid("sparse source has too many stored coordinates"));
    }
    for (&(row, column), value) in source.entries() {
        if row >= n || column >= n {
            return Err(invalid("sparse source coordinate is out of bounds"));
        }
        input(value, work)?;
    }
    // The complete stored source and dimension are admitted before bounded
    // dense fill. This path never calls SparseMatrix::to_dense.
    let mut entries = work.vector(n * n)?;
    work.storage(n * n, 128 * n * n)?;
    entries.resize_with(n * n, || Expr::from_i64(0));
    for (&(row, column), value) in source.entries() {
        entries[row * n + column] = entry_copy(value, work)?;
    }
    Matrix::new(n, n, entries)
}

fn verify_sparse_lift(
    source: &SparseMatrix,
    dense: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick(1)?;
    let n = shape(source.rows(), source.cols())?;
    if source.entries().len() > n * n || dense.rows != n || dense.cols != n {
        return Err(invalid(
            "sparse source and bounded lift have different dimensions",
        ));
    }
    for (&(row, column), value) in source.entries() {
        if row >= n || column >= n {
            return Err(invalid("sparse source coordinate is out of bounds"));
        }
        input(value, work)?;
    }
    let dense_values = values(dense, work)?;
    let zero = work.zero()?;
    for (coordinate, value) in dense_values.iter().enumerate() {
        if let Some(original) = source.entries().get(&(coordinate / n, coordinate % n)) {
            let original = scalar_value(original, work)?;
            if !work.equal(value, &original)? {
                return Err(invalid("bounded lift changed a stored sparse coefficient"));
            }
        } else if !work.equal(value, &zero)? {
            return Err(invalid("bounded lift changed an implicit sparse zero"));
        }
    }
    work.tick(1)
}

impl SparseMatrix {
    /// Checked minimal polynomial after dimension-bounded sparse admission.
    /// The accepted matrix accessor contains the independently bound dense view.
    pub fn minimal_polynomial(&self) -> Result<VerifiedMinimalPolynomial> {
        self.minimal_polynomial_with_meters(&mut Unbounded, &mut Unbounded)
    }

    pub fn minimal_polynomial_with_meters(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedMinimalPolynomial> {
        let mut generation = Work::new(generator);
        let mut verification = Work::new(verifier);
        let dense = sparse_dense(self, &mut generation)?;
        let candidate = generate(&dense, &mut generation)?;
        verify_sparse_lift(self, &dense, &mut verification)?;
        let result = verify(&dense, &candidate, &mut verification)?;
        generation.checkpoint().map_err(stopped)?;
        verification.checkpoint().map_err(stopped)?;
        Ok(result)
    }
}
