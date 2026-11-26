//! Exact constant-coefficient systems with polynomial right-hand sides.
//!
//! A proposed left inverse and polynomial solution are checked against the
//! original sources. L A = I establishes uniqueness; an independent
//! coefficient-dot calculation checks X = L B and A X = B. No numerical
//! sampling, symbolic simplifier, or generator rank flag grants acceptance.
//!
//! This is a bounded in-process reference check over Q(i)[x_1, ..., x_k],
//! not a portable proof-kernel certificate. Symbols are formal commuting
//! coordinates, keyed by their complete typed identity.

use super::{
    BudgetMeter, Constant, Expr, Gaussian, GaussianMatrix, Matrix, MatrixError, Operation, Result,
    SparseMatrix, Work, check_bits, limit, propose,
};
use fsym_core::{BigRational, Symbol};
use fsym_proof_kernel::hermitian_inertia::{
    MAX_GAUSSIAN_RATIONAL_INPUT_BITS, MAX_GAUSSIAN_SOURCE_DEPTH, MAX_GAUSSIAN_SOURCE_NODES,
    MAX_GAUSSIAN_SOURCE_POWER, MAX_HERMITIAN_INERTIA_DIMENSION,
};
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const MAX_GENERATORS: usize = 8;
const MAX_SYMBOL_BYTES: usize = 1024;
const MAX_TERMS: usize = 256;
const MAX_TOTAL_TERMS: usize = 16_384;
const MAX_DEGREE: u32 = 64;

type Monomial = Vec<u32>;
type Polynomial = BTreeMap<Monomial, Gaussian>;

/// Untrusted witness for a unique polynomial solution of constant Q(i) A.
///
/// The left inverse may differ from the Moore-Penrose inverse. Acceptance
/// checks its exact identity and every original polynomial coefficient.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialLinearSystemCandidate {
    pub left_inverse: Matrix,
    pub solution: Matrix,
}

struct Source<'a> {
    symbols: BTreeSet<&'a Symbol>,
    nodes: usize,
    supported: bool,
}

impl<'a> Source<'a> {
    fn inspect(
        &mut self,
        expression: &'a Expr,
        depth: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<bool> {
        work.tick(1)?;
        self.nodes += 1;
        if self.nodes > MAX_GAUSSIAN_SOURCE_NODES || depth > MAX_GAUSSIAN_SOURCE_DEPTH {
            return Err(limit("polynomial RHS source expression envelope"));
        }
        match expression {
            Expr::Integer(value) => {
                if value.bits() > MAX_GAUSSIAN_RATIONAL_INPUT_BITS {
                    return Err(limit("polynomial RHS literal exceeds 4096 bits"));
                }
                work.tick(value.limb_count() + 1)?;
                Ok(false)
            }
            Expr::Rational(value) => {
                check_bits(value, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
                work.tick(value.numer().limb_count() + value.denom().limb_count() + 1)?;
                Ok(false)
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > MAX_SYMBOL_BYTES {
                    return Err(limit("polynomial RHS symbol name exceeds 1024 bytes"));
                }
                work.tick((self.symbols.len() + 1) as u64 * (symbol.name.len() + 33) as u64)?;
                if !self.symbols.contains(symbol) {
                    if self.symbols.len() == MAX_GENERATORS {
                        return Err(limit("polynomial RHS exceeds eight typed coordinates"));
                    }
                    work.storage(64)?;
                    self.symbols.insert(symbol);
                }
                Ok(true)
            }
            Expr::Const(Constant::I) => Ok(false),
            Expr::Const(_) => {
                self.supported = false;
                Ok(false)
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = expression {
                    if name.len() > MAX_SYMBOL_BYTES {
                        return Err(limit(
                            "polynomial RHS function name exceeds its source bound",
                        ));
                    }
                    self.supported = false;
                }
                if children.len() > MAX_GAUSSIAN_SOURCE_NODES.saturating_sub(self.nodes) {
                    return Err(limit("polynomial RHS source expression envelope"));
                }
                let mut symbolic = false;
                // Inspect even unsupported or zero-multiplied children. A
                // resource refusal must never become a new fallback request.
                for child in children {
                    symbolic |= self.inspect(child, depth + 1, work)?;
                }
                Ok(symbolic)
            }
            Expr::Pow(base, exponent) => {
                let symbolic_base = self.inspect(base, depth + 1, work)?;
                let symbolic_exponent = self.inspect(exponent, depth + 1, work)?;
                if let Expr::Integer(value) = exponent.as_ref() {
                    let power = value
                        .to_i64()
                        .filter(|power| {
                            (-MAX_GAUSSIAN_SOURCE_POWER..=MAX_GAUSSIAN_SOURCE_POWER).contains(power)
                        })
                        .ok_or_else(|| limit("polynomial RHS integer power exceeds 64"))?;
                    // Do not turn a symbolic denominator into a constant by
                    // cancelling its source first, even inside a zero product.
                    if power <= 0 && symbolic_base {
                        self.supported = false;
                    }
                } else {
                    self.supported = false;
                }
                Ok(symbolic_base || symbolic_exponent)
            }
        }
    }
}

fn scan<'a>(
    roots: impl IntoIterator<Item = &'a Expr>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Source<'a>> {
    let mut source = Source {
        symbols: BTreeSet::new(),
        nodes: 0,
        supported: true,
    };
    for expression in roots {
        source.inspect(expression, 0, work)?;
    }
    work.tick(1)?;
    Ok(source)
}

fn axes(rows: usize, columns: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
    work.tick(1)?;
    if rows > MAX_HERMITIAN_INERTIA_DIMENSION || columns > MAX_HERMITIAN_INERTIA_DIMENSION {
        return Err(limit("polynomial RHS matrix dimension exceeds 32"));
    }
    Ok(())
}

fn dense_source<'a>(
    matrix: &'a Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Source<'a>> {
    matrix.validate_shape()?;
    axes(matrix.rows, matrix.cols, work)?;
    scan(matrix.data(), work)
}

fn sparse_source<'a>(
    matrix: &'a SparseMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Source<'a>> {
    let (rows, columns) = (matrix.rows(), matrix.cols());
    axes(rows, columns, work)?;
    if matrix.entries().len() > rows * columns {
        return Err(MatrixError::InvalidCertificate(
            "too many original sparse polynomial RHS cells".into(),
        ));
    }
    for (&(row, column), value) in matrix.entries() {
        work.tick(1)?;
        if row >= rows || column >= columns {
            return Err(MatrixError::OutOfBounds(row, column, rows, columns));
        }
        if value.is_zero() {
            return Err(MatrixError::InvalidCertificate(
                "sparse polynomial RHS has an explicit zero".into(),
            ));
        }
    }
    scan(matrix.entries().values(), work)
}

fn copy_symbols<'a>(
    values: impl ExactSizeIterator<Item = &'a Symbol>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<Symbol>> {
    let mut copied = work.vector(values.len())?;
    for symbol in values {
        work.tick(symbol.name.len() as u64 + 1)?;
        work.storage(symbol.name.len() as u64 + 64)?;
        copied.push(symbol.clone());
    }
    Ok(copied)
}

fn monomial(values: &[u32], work: &mut Work<'_, impl BudgetMeter>) -> Result<Monomial> {
    let mut copied = work.vector(values.len())?;
    work.tick(values.len() as u64 + 1)?;
    copied.extend_from_slice(values);
    Ok(copied)
}

fn zero_monomial(width: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Monomial> {
    let mut result = work.vector(width)?;
    work.tick(width as u64 + 1)?;
    result.resize(width, 0);
    Ok(result)
}

fn zero(value: &Gaussian) -> bool {
    value.0.is_zero() && value.1.is_zero()
}

fn copy_coefficient(value: &Gaussian, work: &mut Work<'_, impl BudgetMeter>) -> Result<Gaussian> {
    Ok((work.clone_scalar(&value.0)?, work.clone_scalar(&value.1)?))
}

fn accumulate(
    result: &mut Polynomial,
    powers: &[u32],
    mut value: Gaussian,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick((u64::from(result.len().max(1).ilog2()) + 2) * (powers.len() as u64 + 1))?;
    if zero(&value) {
        return Ok(());
    }
    if let Some(previous) = result.get(powers) {
        value = work.add(previous, &value)?;
    } else if result.len() == MAX_TERMS {
        return Err(limit("polynomial RHS exceeds 256 terms in one cell"));
    }
    if zero(&value) {
        result.remove(powers);
    } else {
        work.storage(192)?;
        result.insert(monomial(powers, work)?, value);
    }
    Ok(())
}

fn constant(
    value: Gaussian,
    width: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Polynomial> {
    let mut result = Polynomial::new();
    let powers = zero_monomial(width, work)?;
    accumulate(&mut result, &powers, value, work)?;
    Ok(result)
}

fn one(width: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Polynomial> {
    work.storage(128)?;
    constant((BigRational::one(), BigRational::zero()), width, work)
}

fn add_into(
    target: &mut Polynomial,
    source: &Polynomial,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    for (powers, value) in source {
        let value = copy_coefficient(value, work)?;
        accumulate(target, powers, value, work)?;
    }
    work.tick(1)
}

fn product(
    left: &Polynomial,
    right: &Polynomial,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Polynomial> {
    let mut result = Polynomial::new();
    for (left_powers, left_value) in left {
        for (right_powers, right_value) in right {
            work.tick(left_powers.len() as u64 + 1)?;
            let mut powers = work.vector(left_powers.len())?;
            let mut degree = 0u32;
            for (&left_power, &right_power) in left_powers.iter().zip(right_powers) {
                let power = left_power
                    .checked_add(right_power)
                    .ok_or_else(|| limit("polynomial RHS exponent overflow"))?;
                degree = degree
                    .checked_add(power)
                    .filter(|degree| *degree <= MAX_DEGREE)
                    .ok_or_else(|| limit("polynomial RHS total degree exceeds 64"))?;
                powers.push(power);
            }
            let value = work.multiply(left_value, right_value)?;
            accumulate(&mut result, &powers, value, work)?;
        }
    }
    work.tick(1)?;
    Ok(result)
}

fn power(
    mut base: Polynomial,
    exponent: i64,
    width: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Polynomial> {
    if exponent <= 0 && base.is_empty() {
        return Err(MatrixError::DivisionByZero);
    }
    if exponent < 0 {
        if base.len() != 1
            || base
                .keys()
                .any(|powers| powers.iter().any(|power| *power != 0))
        {
            return Err(MatrixError::UnsupportedCertificateDomain);
        }
        let value = base.values().next().ok_or(MatrixError::DivisionByZero)?;
        let real_square = work.rational(&value.0, &value.0, Operation::Multiply)?;
        let imaginary_square = work.rational(&value.1, &value.1, Operation::Multiply)?;
        let norm = work.rational(&real_square, &imaginary_square, Operation::Add)?;
        let negative_imaginary = work.negate(&value.1)?;
        let inverse = (
            work.rational(&value.0, &norm, Operation::Divide)?,
            work.rational(&negative_imaginary, &norm, Operation::Divide)?,
        );
        base = constant(inverse, width, work)?;
    }
    let degree = base
        .keys()
        .map(|powers| powers.iter().copied().sum::<u32>())
        .max()
        .unwrap_or(0);
    if u64::from(degree).saturating_mul(exponent.unsigned_abs()) > u64::from(MAX_DEGREE) {
        return Err(limit("polynomial RHS total degree exceeds 64"));
    }
    let mut remaining = exponent.unsigned_abs();
    let mut value = one(width, work)?;
    while remaining != 0 {
        work.tick(1)?;
        if remaining & 1 != 0 {
            value = product(&value, &base, work)?;
        }
        remaining >>= 1;
        if remaining != 0 {
            base = product(&base, &base, work)?;
        }
    }
    Ok(value)
}

fn read_polynomial(
    expression: &Expr,
    generators: &[Symbol],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Polynomial> {
    // The complete original source has already passed the bounded scan.
    // Recursive interpretation still visits every child before reduction.
    work.tick(1)?;
    match expression {
        Expr::Integer(value) => {
            work.storage(128 + 8 * value.limb_count())?;
            constant(
                (
                    BigRational::from_integer(value.clone()),
                    BigRational::zero(),
                ),
                generators.len(),
                work,
            )
        }
        Expr::Rational(value) => {
            let value = work.clone_scalar(value)?;
            work.storage(64)?;
            constant((value, BigRational::zero()), generators.len(), work)
        }
        Expr::Const(Constant::I) => {
            work.storage(128)?;
            constant(
                (BigRational::zero(), BigRational::one()),
                generators.len(),
                work,
            )
        }
        Expr::Sym(symbol) => {
            work.tick((generators.len() + 1) as u64 * (symbol.name.len() + 33) as u64)?;
            let coordinate = generators
                .iter()
                .position(|candidate| candidate == symbol)
                .ok_or_else(|| {
                    MatrixError::InvalidCertificate(
                        "polynomial witness changed its typed ring".into(),
                    )
                })?;
            let mut powers = zero_monomial(generators.len(), work)?;
            powers[coordinate] = 1;
            let mut result = Polynomial::new();
            work.storage(128)?;
            accumulate(
                &mut result,
                &powers,
                (BigRational::one(), BigRational::zero()),
                work,
            )?;
            Ok(result)
        }
        Expr::Add(children) => {
            let mut result = Polynomial::new();
            for child in children {
                let value = read_polynomial(child, generators, work)?;
                add_into(&mut result, &value, work)?;
            }
            Ok(result)
        }
        Expr::Mul(children) => {
            let mut result = one(generators.len(), work)?;
            for child in children {
                let value = read_polynomial(child, generators, work)?;
                result = product(&result, &value, work)?;
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            let base = read_polynomial(base, generators, work)?;
            let Expr::Integer(exponent) = exponent.as_ref() else {
                return Err(MatrixError::UnsupportedCertificateDomain);
            };
            let exponent = exponent
                .to_i64()
                .ok_or_else(|| limit("polynomial RHS power exceeds its admitted bound"))?;
            power(base, exponent, generators.len(), work)
        }
        _ => Err(MatrixError::UnsupportedCertificateDomain),
    }
}

fn total_terms(count: &mut usize, added: usize) -> Result<()> {
    *count = count
        .checked_add(added)
        .filter(|count| *count <= MAX_TOTAL_TERMS)
        .ok_or_else(|| limit("polynomial RHS exceeds 16384 total terms"))?;
    Ok(())
}

struct PolynomialMatrix {
    rows: usize,
    cols: usize,
    generators: Vec<Symbol>,
    entries: Vec<Polynomial>,
}

impl PolynomialMatrix {
    fn read(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let source = dense_source(matrix, work)?;
        if !source.supported {
            return Err(MatrixError::UnsupportedCertificateDomain);
        }
        let generators = copy_symbols(source.symbols.into_iter(), work)?;
        Self::interpret(matrix, generators, work)
    }

    fn read_in_ring(
        matrix: &Matrix,
        generators: &[Symbol],
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<Self> {
        let source = dense_source(matrix, work)?;
        if !source.supported {
            return Err(MatrixError::UnsupportedCertificateDomain);
        }
        for symbol in source.symbols {
            work.tick((generators.len() + 1) as u64 * (symbol.name.len() + 33) as u64)?;
            if !generators.contains(symbol) {
                return Err(MatrixError::InvalidCertificate(
                    "polynomial solution introduced another typed coordinate".into(),
                ));
            }
        }
        let generators = copy_symbols(generators.iter(), work)?;
        Self::interpret(matrix, generators, work)
    }

    fn interpret(
        matrix: &Matrix,
        generators: Vec<Symbol>,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<Self> {
        let mut entries = work.vector(matrix.data.len())?;
        let mut count = 0usize;
        for expression in matrix.data() {
            let value = read_polynomial(expression, &generators, work)?;
            total_terms(&mut count, value.len())?;
            entries.push(value);
        }
        work.tick(1)?;
        Ok(Self {
            rows: matrix.rows,
            cols: matrix.cols,
            generators,
            entries,
        })
    }

    fn get(&self, row: usize, column: usize) -> &Polynomial {
        &self.entries[row * self.cols + column]
    }

    fn left_multiply(
        &self,
        left: &GaussianMatrix,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<Self> {
        if left.cols != self.rows {
            return Err(MatrixError::ShapeMismatch(
                left.rows, left.cols, self.rows, self.cols,
            ));
        }
        let generators = copy_symbols(self.generators.iter(), work)?;
        let mut entries = work.vector(left.rows * self.cols)?;
        let mut count = 0usize;
        for row in 0..left.rows {
            for column in 0..self.cols {
                let mut value = Polynomial::new();
                for inner in 0..left.cols {
                    work.tick(1)?;
                    for (powers, coefficient) in self.get(inner, column) {
                        let coefficient = work.multiply(left.get(row, inner), coefficient)?;
                        accumulate(&mut value, powers, coefficient, work)?;
                    }
                }
                total_terms(&mut count, value.len())?;
                entries.push(value);
            }
        }
        work.tick(1)?;
        Ok(Self {
            rows: left.rows,
            cols: self.cols,
            generators,
            entries,
        })
    }

    fn into_matrix(self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
        let mut entries = work.vector(self.entries.len())?;
        for polynomial in self.entries {
            let mut terms = work.vector(polynomial.len())?;
            for (powers, coefficient) in polynomial {
                work.tick(1)?;
                check_bits(&coefficient.0, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
                check_bits(&coefficient.1, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
                let mut factors = work.vector(self.generators.len() + 1)?;
                if !coefficient.0.is_one()
                    || !coefficient.1.is_zero()
                    || powers.iter().all(|power| *power == 0)
                {
                    factors.push(coefficient_expression(coefficient, work)?);
                }
                for (symbol, exponent) in self.generators.iter().zip(powers) {
                    if exponent == 0 {
                        continue;
                    }
                    work.tick(symbol.name.len() as u64 + 1)?;
                    work.storage(symbol.name.len() as u64 + 256)?;
                    let variable = Expr::Sym(symbol.clone());
                    factors.push(if exponent == 1 {
                        variable
                    } else {
                        Expr::Pow(
                            Arc::new(variable),
                            Arc::new(Expr::from_i64(i64::from(exponent))),
                        )
                    });
                }
                terms.push(match factors.len() {
                    0 => Expr::from_i64(1),
                    1 => factors.pop().ok_or_else(|| {
                        MatrixError::ArithmeticInvariantViolation("empty polynomial factor".into())
                    })?,
                    _ => Expr::Mul(factors),
                });
            }
            entries.push(match terms.len() {
                0 => Expr::from_i64(0),
                1 => terms.pop().ok_or_else(|| {
                    MatrixError::ArithmeticInvariantViolation("empty polynomial term".into())
                })?,
                _ => Expr::Add(terms),
            });
        }
        work.tick(1)?;
        Matrix::new(self.rows, self.cols, entries)
    }
}

fn coefficient_expression(
    (real, imaginary): Gaussian,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    if imaginary.is_zero() {
        return work.scalar_expression(real);
    }
    work.storage(256)?;
    let imaginary = if imaginary.is_one() {
        Expr::Const(Constant::I)
    } else {
        Expr::Mul(vec![
            work.scalar_expression(imaginary)?,
            Expr::Const(Constant::I),
        ])
    };
    Ok(if real.is_zero() {
        imaginary
    } else {
        Expr::Add(vec![work.scalar_expression(real)?, imaginary])
    })
}

fn same_coefficient(
    left: &Gaussian,
    right: &Gaussian,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    let limbs = [&left.0, &left.1, &right.0, &right.1]
        .iter()
        .map(|value| value.numer().limb_count() + value.denom().limb_count())
        .sum::<u64>();
    work.tick(limbs + 1)?;
    Ok(left == right)
}

/// Reference acceptance computes one coefficient dot product at a time.
/// It does not call the generator's polynomial scatter/add multiplication.
fn check_product(
    left: &GaussianMatrix,
    right: &PolynomialMatrix,
    expected: &PolynomialMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    work.tick(
        right
            .generators
            .iter()
            .chain(&expected.generators)
            .map(|symbol| symbol.name.len() as u64 + 33)
            .sum::<u64>()
            + 1,
    )?;
    if left.cols != right.rows
        || left.rows != expected.rows
        || right.cols != expected.cols
        || right.generators != expected.generators
    {
        return Ok(false);
    }
    for column in 0..right.cols {
        let mut powers: BTreeSet<&Monomial> = BTreeSet::new();
        for row in 0..right.rows {
            for key in right.get(row, column).keys() {
                work.tick((u64::from(powers.len().max(1).ilog2()) + 2) * (key.len() as u64 + 1))?;
                work.storage(64)?;
                powers.insert(key);
            }
        }
        for row in 0..expected.rows {
            for key in expected.get(row, column).keys() {
                work.tick((u64::from(powers.len().max(1).ilog2()) + 2) * (key.len() as u64 + 1))?;
                work.storage(64)?;
                powers.insert(key);
            }
        }
        for row in 0..left.rows {
            for key in &powers {
                work.storage(128)?;
                let mut sum = (BigRational::zero(), BigRational::zero());
                for inner in 0..left.cols {
                    work.tick(key.len() as u64 + 1)?;
                    if let Some(coefficient) = right.get(inner, column).get(key.as_slice()) {
                        let term = work.multiply(left.get(row, inner), coefficient)?;
                        sum = work.add(&sum, &term)?;
                    }
                }
                if let Some(value) = expected.get(row, column).get(key.as_slice()) {
                    if !same_coefficient(&sum, value, work)? {
                        return Ok(false);
                    }
                } else if !zero(&sum) {
                    return Ok(false);
                }
            }
        }
    }
    work.tick(1)?;
    Ok(true)
}

fn identity(matrix: &GaussianMatrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<bool> {
    if matrix.rows != matrix.cols {
        return Ok(false);
    }
    for row in 0..matrix.rows {
        for column in 0..matrix.cols {
            let value = matrix.get(row, column);
            work.tick(value.0.numer().limb_count() + value.1.numer().limb_count() + 1)?;
            if !value.1.is_zero()
                || if row == column {
                    !value.0.is_one()
                } else {
                    !value.0.is_zero()
                }
            {
                return Ok(false);
            }
        }
    }
    work.tick(1)?;
    Ok(true)
}

fn generate(
    matrix: &Matrix,
    rhs: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<PolynomialLinearSystemCandidate> {
    let a = GaussianMatrix::read(matrix, work)?;
    let b = PolynomialMatrix::read(rhs, work)?;
    if a.rows != b.rows {
        return Err(MatrixError::ShapeMismatch(a.rows, a.cols, b.rows, b.cols));
    }
    let left_inverse = propose(matrix, work)?;
    let left = GaussianMatrix::read(&left_inverse, work)?;
    let solution = b.left_multiply(&left, work)?.into_matrix(work)?;
    work.tick(1)?;
    Ok(PolynomialLinearSystemCandidate {
        left_inverse,
        solution,
    })
}

enum Check {
    Accepted,
    NoLeftInverse,
    Inconsistent,
}

fn check(
    matrix: &Matrix,
    rhs: &Matrix,
    candidate: &PolynomialLinearSystemCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Check> {
    // Reread both original graphs and both untrusted witnesses. Generator
    // coefficient maps, realification and rank decisions are not retained.
    let a = GaussianMatrix::read(matrix, work)?;
    let b = PolynomialMatrix::read(rhs, work)?;
    let left = GaussianMatrix::read(&candidate.left_inverse, work)?;
    let x = PolynomialMatrix::read_in_ring(&candidate.solution, &b.generators, work)?;
    if a.rows != b.rows
        || left.rows != a.cols
        || left.cols != a.rows
        || x.rows != a.cols
        || x.cols != b.cols
    {
        return Err(MatrixError::InvalidCertificate(
            "polynomial exact linear-system source or witness shape".into(),
        ));
    }
    if !identity(&left.multiply(&a, work)?, work)? {
        return Ok(Check::NoLeftInverse);
    }
    if !check_product(&left, &b, &x, work)? {
        return Err(MatrixError::InvalidCertificate(
            "polynomial solution does not equal the original left-inverse RHS product".into(),
        ));
    }
    if !check_product(&a, &x, &b, work)? {
        return Ok(Check::Inconsistent);
    }
    work.tick(1)?;
    Ok(Check::Accepted)
}

impl Matrix {
    /// Inspect complete borrowed RHS syntax before selecting a polynomial lane.
    ///
    /// Numeric constants also satisfy this grammar. Selection grants no
    /// mathematical admission: the separate source reader checks every domain
    /// and bounded polynomial expansion, including held zero-power children.
    pub fn has_polynomial_rhs_syntax_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<bool> {
        let mut work = Work::new(meter);
        let source = dense_source(self, &mut work)?;
        work.tick(1)?;
        Ok(source.supported)
    }

    /// Admit all original RHS cells over bounded Q(i) polynomial coordinates.
    pub fn validate_polynomial_rhs_source_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        let mut work = Work::new(meter);
        PolynomialMatrix::read(self, &mut work)?;
        work.tick(1)
    }

    /// Generate an untrusted source-relative polynomial solution proposal.
    pub fn polynomial_linear_system_candidate_with_meter(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<PolynomialLinearSystemCandidate> {
        generate(self, rhs, &mut Work::new(meter))
    }

    /// Check uniqueness and the full original polynomial residual separately
    /// from proposal generation. This is an in-process reference certificate.
    pub fn verify_polynomial_linear_system_candidate(
        &self,
        rhs: &Self,
        candidate: &PolynomialLinearSystemCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        match check(self, rhs, candidate, &mut Work::new(meter))? {
            Check::Accepted => Ok(()),
            Check::NoLeftInverse => Err(MatrixError::InvalidCertificate(
                "polynomial solve left inverse does not establish uniqueness".into(),
            )),
            Check::Inconsistent => Err(MatrixError::InvalidCertificate(
                "polynomial solution has a nonzero original-source residual".into(),
            )),
        }
    }

    /// Solve constant Q(i) A X=B with a complete polynomial RHS over Q(i).
    ///
    /// Each axis is at most 32. There are at most eight complete typed symbols,
    /// degree 64, 256 terms per cell and 16384 terms per matrix. Source nodes,
    /// depth and scalar height share the Gaussian reader's intrinsic bounds.
    /// Source admission, proposal, independent coefficient checking and final
    /// publication use one cumulative 20M-step/256MiB ledger and caller meter.
    /// Singular or inconsistent requests refuse; no projection is published.
    pub fn metered_polynomial_solve(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self> {
        let mut work = Work::new(meter);
        let candidate = generate(self, rhs, &mut work)?;
        match check(self, rhs, &candidate, &mut work)? {
            Check::Accepted => {
                work.tick(1)?;
                Ok(candidate.solution)
            }
            Check::NoLeftInverse => Err(MatrixError::SingularMatrix),
            Check::Inconsistent => Err(MatrixError::InconsistentLinearSystem),
        }
    }
}

impl SparseMatrix {
    /// Inspect original sparse RHS storage without allocating a dense view.
    pub fn has_polynomial_rhs_syntax_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<bool> {
        let mut work = Work::new(meter);
        let source = sparse_source(self, &mut work)?;
        work.tick(1)?;
        Ok(source.supported)
    }

    /// Admit original sparse polynomial cells before bounded densification.
    pub fn validate_polynomial_rhs_source_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        let mut work = Work::new(meter);
        let source = sparse_source(self, &mut work)?;
        if !source.supported {
            return Err(MatrixError::UnsupportedCertificateDomain);
        }
        let generators = copy_symbols(source.symbols.into_iter(), &mut work)?;
        let mut count = 0usize;
        for value in self.entries().values() {
            let polynomial = read_polynomial(value, &generators, &mut work)?;
            total_terms(&mut count, polynomial.len())?;
        }
        work.tick(1)
    }
}
