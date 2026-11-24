//! Positive-definite Hermitian LDL and Cholesky over exact Q(i).
//!
//! The generator uses unpivoted Schur updates. An independent reference pass
//! reinterprets the original source and all proposed factors, requires a unit
//! lower triangle and a positive real diagonal, and checks every coefficient
//! of A = L D L^H. This establishes positive definiteness as well as the exact
//! decomposition. No permutation or indefinite block factorization is implied.
//!
//! Cholesky proposals additionally contain positive square-root scales and the
//! published factor. The checker independently reads its finite radical grammar
//! and checks component signs and squared values against L_ij sqrt(D_jj).
//! A negative or imaginary column phase cannot pass merely because F F^H = A.
//!
//! All phases share the Gaussian lane's 32-axis, 4096-bit source/LDL output,
//! 8192-bit rational-work, 20M-step and 256MiB cumulative limits. Radical output
//! uses the existing QR normalization grammar and its explicit bounds. This is
//! synchronous same-crate checking, not portable proof-kernel certification.

use super::{
    BigRational, BudgetMeter, Gaussian, GaussianMatrix, Matrix, MatrixError, Operation, Result,
    Work,
};
use fsym_core::{Constant, Expr};
use num_traits::{One, Signed, Zero};

/// Untrusted Hermitian positive-definite factorization A = L D L^H.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianLdlCandidate {
    pub l: Matrix,
    pub d: Matrix,
}

/// Untrusted positive-column normalization of an LDL proposal.
/// Every field is checked against the separately supplied original source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianCholeskyCandidate {
    pub ldl: GaussianLdlCandidate,
    pub scales: Vec<Expr>,
    pub factor: Matrix,
}

fn invalid(reason: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("Hermitian LDL/Cholesky: {reason}"))
}

fn is_zero(value: &Gaussian) -> bool {
    value.0.is_zero() && value.1.is_zero()
}

fn copy(value: &Gaussian, work: &mut Work<'_, impl BudgetMeter>) -> Result<Gaussian> {
    Ok((work.clone_scalar(&value.0)?, work.clone_scalar(&value.1)?))
}

fn source(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<GaussianMatrix> {
    work.tick(1)?;
    matrix.validate_shape()?;
    if matrix.rows != matrix.cols {
        return Err(MatrixError::NotSquare(matrix.rows, matrix.cols));
    }
    // Interpret every original operand before making a structural or zero
    // decision, including entries outside the lower triangle.
    let original = GaussianMatrix::read(matrix, work)?;
    if !original.is_hermitian(work)? {
        return Err(MatrixError::NotHermitian);
    }
    Ok(original)
}

fn propose(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<GaussianLdlCandidate> {
    let original = source(matrix, work)?;
    let size = original.rows;
    let mut lower = GaussianMatrix::zeros(size, size, work)?;
    let mut diagonal = GaussianMatrix::zeros(size, size, work)?;
    for column in 0..size {
        work.tick(1)?;
        work.storage(64)?;
        lower.entries[column * size + column].0 = BigRational::one();
        let mut pivot = work.clone_scalar(&original.get(column, column).0)?;
        for inner in 0..column {
            work.tick(1)?;
            let value = lower.get(column, inner);
            if is_zero(value) {
                continue;
            }
            let real = work.rational(&value.0, &value.0, Operation::Multiply)?;
            let imaginary = work.rational(&value.1, &value.1, Operation::Multiply)?;
            let norm = work.rational(&real, &imaginary, Operation::Add)?;
            let term = work.rational(&norm, &diagonal.get(inner, inner).0, Operation::Multiply)?;
            pivot = work.rational(&pivot, &term, Operation::Subtract)?;
        }
        if !pivot.is_positive() {
            return Err(MatrixError::NotPositiveDefinite);
        }
        diagonal.entries[column * size + column].0 = work.clone_scalar(&pivot)?;
        for row in column + 1..size {
            let mut residual = copy(original.get(row, column), work)?;
            for inner in 0..column {
                work.tick(1)?;
                let left = lower.get(row, inner);
                let right = lower.get(column, inner);
                if is_zero(left) || is_zero(right) {
                    continue;
                }
                let conjugate = (work.clone_scalar(&right.0)?, work.negate(&right.1)?);
                let product = work.multiply(left, &conjugate)?;
                let diagonal = &diagonal.get(inner, inner).0;
                let real = work.rational(&product.0, diagonal, Operation::Multiply)?;
                let imaginary = work.rational(&product.1, diagonal, Operation::Multiply)?;
                residual.0 = work.rational(&residual.0, &real, Operation::Subtract)?;
                residual.1 = work.rational(&residual.1, &imaginary, Operation::Subtract)?;
            }
            lower.entries[row * size + column] = (
                work.rational(&residual.0, &pivot, Operation::Divide)?,
                work.rational(&residual.1, &pivot, Operation::Divide)?,
            );
        }
    }
    Ok(GaussianLdlCandidate {
        l: lower.into_matrix(work)?,
        d: diagonal.into_matrix(work)?,
    })
}

/// Return independently read factors for subsequent normalization checking.
fn verify(
    matrix: &Matrix,
    candidate: &GaussianLdlCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(GaussianMatrix, Vec<BigRational>)> {
    let original = source(matrix, work)?;
    let size = original.rows;
    candidate.l.validate_shape()?;
    candidate.d.validate_shape()?;
    if candidate.l.rows != size
        || candidate.l.cols != size
        || candidate.d.rows != size
        || candidate.d.cols != size
    {
        return Err(invalid("factor shape"));
    }
    let lower = GaussianMatrix::read(&candidate.l, work)?;
    let diagonal = GaussianMatrix::read(&candidate.d, work)?;
    let mut pivots = work.vector(size)?;
    for row in 0..size {
        for column in 0..size {
            work.tick(1)?;
            let l = lower.get(row, column);
            if (row == column && (!l.0.is_one() || !l.1.is_zero())) || (column > row && !is_zero(l))
            {
                return Err(invalid("L is not unit lower triangular"));
            }
            let d = diagonal.get(row, column);
            if row != column && !is_zero(d) {
                return Err(invalid("D is not diagonal"));
            }
            if row == column {
                if !d.1.is_zero() || !d.0.is_positive() {
                    return Err(invalid("D does not have a strictly positive real diagonal"));
                }
                pivots.push(work.clone_scalar(&d.0)?);
            }
        }
    }

    // This direct convolution does not reuse Schur updates or the generator's
    // conjugation/product helper. Unit lower structure bounds the inner sum.
    // Every source/candidate cell was admitted before zero factors can skip work.
    for row in 0..size {
        for column in 0..size {
            work.storage(128)?;
            let mut real = BigRational::zero();
            let mut imaginary = BigRational::zero();
            for (inner, pivot) in pivots.iter().enumerate().take(row.min(column) + 1) {
                work.tick(1)?;
                let a = lower.get(row, inner);
                let b = lower.get(column, inner);
                if is_zero(a) || is_zero(b) {
                    continue;
                }
                let ac = work.rational(&a.0, &b.0, Operation::Multiply)?;
                let bd = work.rational(&a.1, &b.1, Operation::Multiply)?;
                let bc = work.rational(&a.1, &b.0, Operation::Multiply)?;
                let ad = work.rational(&a.0, &b.1, Operation::Multiply)?;
                let re = work.rational(&ac, &bd, Operation::Add)?;
                let im = work.rational(&bc, &ad, Operation::Subtract)?;
                let re = work.rational(&re, pivot, Operation::Multiply)?;
                let im = work.rational(&im, pivot, Operation::Multiply)?;
                real = work.rational(&real, &re, Operation::Add)?;
                imaginary = work.rational(&imaginary, &im, Operation::Add)?;
            }
            let expected = original.get(row, column);
            let limbs = [&real, &imaginary, &expected.0, &expected.1]
                .iter()
                .map(|value| value.numer().limb_count() + value.denom().limb_count())
                .sum::<u64>();
            work.tick(limbs + 1)?;
            if real != expected.0 || imaginary != expected.1 {
                return Err(invalid("L D L^H differs from the original source"));
            }
        }
    }
    work.tick(1)?;
    Ok((lower, pivots))
}

fn propose_cholesky(
    matrix: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianCholeskyCandidate> {
    let ldl = propose(matrix, work)?;
    let size = ldl.l.rows;
    let mut scales = work.vector(size)?;
    for index in 0..size {
        let value = &ldl.d.data[index * size + index];
        let rational = match value {
            Expr::Integer(value) => {
                work.storage(128 + 8 * value.limb_count())?;
                BigRational::from_integer(value.clone())
            }
            Expr::Rational(value) => work.clone_scalar(value)?,
            _ => return Err(invalid("generated diagonal is not rational")),
        };
        scales.push(super::qr::square_root(&rational, work)?);
    }
    let mut entries = work.vector(ldl.l.data.len())?;
    for row in 0..size {
        for (column, scale) in scales.iter().enumerate() {
            let value = &ldl.l.data[row * size + column];
            entries.push(super::qr::normalization_product(value, scale, work)?);
        }
    }
    let factor = Matrix::new(size, size, entries)?;
    Ok(GaussianCholeskyCandidate {
        ldl,
        scales,
        factor,
    })
}

/// A finite normalized expression equal to coefficient * sqrt(radicand).
/// The radicand is positive; this positive branch is essential for sign checks.
struct ScaledScalar {
    coefficient: Gaussian,
    radicand: BigRational,
}

fn read_scaled(expression: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<ScaledScalar> {
    work.tick(1)?;
    work.storage(256)?;
    match expression {
        Expr::Integer(integer) => {
            work.tick(integer.limb_count() + 1)?;
            work.storage(64 + 8 * integer.limb_count())?;
            Ok(ScaledScalar {
                coefficient: (
                    BigRational::from_integer(integer.clone()),
                    BigRational::zero(),
                ),
                radicand: BigRational::one(),
            })
        }
        Expr::Rational(value) => Ok(ScaledScalar {
            coefficient: (work.clone_scalar(value)?, BigRational::zero()),
            radicand: BigRational::one(),
        }),
        Expr::Const(Constant::I) => Ok(ScaledScalar {
            coefficient: (BigRational::zero(), BigRational::one()),
            radicand: BigRational::one(),
        }),
        Expr::Pow(base, _) => {
            // The complete finite grammar is checked before this recursive read.
            let Expr::Integer(base) = base.as_ref() else {
                return Err(invalid("normalized radical base"));
            };
            work.tick(base.limb_count() + 1)?;
            work.storage(64 + 8 * base.limb_count())?;
            Ok(ScaledScalar {
                coefficient: (BigRational::one(), BigRational::zero()),
                radicand: BigRational::from_integer(base.clone()),
            })
        }
        Expr::Add(terms) => {
            let mut left = read_scaled(&terms[0], work)?;
            let right = read_scaled(&terms[1], work)?;
            if is_zero(&left.coefficient) {
                return Ok(right);
            }
            if is_zero(&right.coefficient) {
                return Ok(left);
            }
            work.tick(
                left.radicand.numer().limb_count() + right.radicand.numer().limb_count() + 1,
            )?;
            if left.radicand != right.radicand {
                return Err(invalid("normalized sum has different radical scales"));
            }
            left.coefficient = work.add(&left.coefficient, &right.coefficient)?;
            Ok(left)
        }
        Expr::Mul(factors) => {
            let mut result = ScaledScalar {
                coefficient: (BigRational::one(), BigRational::zero()),
                radicand: BigRational::one(),
            };
            for factor in factors {
                let next = read_scaled(factor, work)?;
                result.coefficient = work.multiply(&result.coefficient, &next.coefficient)?;
                if result.radicand.is_one() {
                    result.radicand = next.radicand;
                } else if !next.radicand.is_one() {
                    return Err(invalid("normalized product has multiple radicals"));
                }
            }
            Ok(result)
        }
        _ => Err(invalid("normalized scalar source")),
    }
}

fn component_matches(
    actual: &BigRational,
    radicand: &BigRational,
    expected: &BigRational,
    diagonal: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    work.tick(actual.numer().limb_count() + expected.numer().limb_count() + 1)?;
    if actual.is_zero() != expected.is_zero() || actual.is_positive() != expected.is_positive() {
        return Ok(false);
    }
    if actual.is_zero() {
        return Ok(true);
    }
    // Both radicals denote their positive roots. Equal signs plus equal
    // squared real components prove equality, without a floating comparison
    // or a call back into the normalization product generator.
    let left_square = work.rational(actual, actual, Operation::Multiply)?;
    let right_square = work.rational(expected, expected, Operation::Multiply)?;
    let left = work.rational(&left_square, radicand, Operation::Multiply)?;
    let right = work.rational(&right_square, diagonal, Operation::Multiply)?;
    work.tick(
        left.numer().limb_count()
            + left.denom().limb_count()
            + right.numer().limb_count()
            + right.denom().limb_count()
            + 1,
    )?;
    Ok(left == right)
}

fn verify_cholesky(
    matrix: &Matrix,
    candidate: &GaussianCholeskyCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let (lower, diagonal) = verify(matrix, &candidate.ldl, work)?;
    let size = lower.rows;
    candidate.factor.validate_shape()?;
    if candidate.factor.rows != size
        || candidate.factor.cols != size
        || candidate.scales.len() != size
    {
        return Err(invalid("Cholesky factor or scale shape"));
    }
    // Preflight the entire proposed publication forest before its first scalar
    // arithmetic. Original LDL/source parsing above remains independent.
    for expression in candidate.scales.iter().chain(candidate.factor.data.iter()) {
        super::qr::validate_normalized_scalar(expression, work)?;
    }
    for (index, scale) in candidate.scales.iter().enumerate() {
        let square = super::qr::positive_scale_square(scale, 0, work)?;
        work.tick(
            square.numer().limb_count()
                + square.denom().limb_count()
                + diagonal[index].numer().limb_count()
                + diagonal[index].denom().limb_count()
                + 1,
        )?;
        if square != diagonal[index] {
            return Err(invalid(
                "positive scale does not square to the checked diagonal",
            ));
        }
    }
    for row in 0..size {
        for (column, pivot) in diagonal.iter().enumerate() {
            let entry = &candidate.factor.data[row * size + column];
            let actual = read_scaled(entry, work)?;
            let expected = lower.get(row, column);
            if !component_matches(
                &actual.coefficient.0,
                &actual.radicand,
                &expected.0,
                pivot,
                work,
            )? || !component_matches(
                &actual.coefficient.1,
                &actual.radicand,
                &expected.1,
                pivot,
                work,
            )? {
                return Err(invalid(
                    "published Cholesky entry has the wrong value or branch",
                ));
            }
        }
    }
    work.tick(1)
}

impl Matrix {
    /// Propose positive-definite Hermitian LDL without granting evidence.
    pub fn gaussian_ldl_candidate_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianLdlCandidate> {
        propose(self, &mut Work::new(meter))
    }

    /// Check the separately supplied original and every LDL factor entry.
    pub fn verify_gaussian_ldl_candidate(
        &self,
        candidate: &GaussianLdlCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        verify(self, candidate, &mut Work::new(meter)).map(|_| ())
    }

    /// Return Hermitian LDL only after its complete independent reference pass.
    pub fn metered_gaussian_ldl(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianLdlCandidate> {
        let mut work = Work::new(meter);
        let candidate = propose(self, &mut work)?;
        verify(self, &candidate, &mut work)?;
        work.tick(1)?;
        Ok(candidate)
    }

    /// Propose a positive-diagonal Cholesky factor without granting evidence.
    pub fn gaussian_cholesky_candidate_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianCholeskyCandidate> {
        propose_cholesky(self, &mut Work::new(meter))
    }

    /// Check LDL, every positive scale, and every normalized output component.
    pub fn verify_gaussian_cholesky_candidate(
        &self,
        candidate: &GaussianCholeskyCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        verify_cholesky(self, candidate, &mut Work::new(meter))
    }

    /// Return positive-diagonal Cholesky after independent original-source and
    /// normalized-component checks, retaining one cumulative request ledger.
    pub fn metered_gaussian_cholesky(&self, meter: &mut impl BudgetMeter) -> Result<Matrix> {
        let mut work = Work::new(meter);
        let candidate = propose_cholesky(self, &mut work)?;
        verify_cholesky(self, &candidate, &mut work)?;
        work.tick(1)?;
        Ok(candidate.factor)
    }
}
