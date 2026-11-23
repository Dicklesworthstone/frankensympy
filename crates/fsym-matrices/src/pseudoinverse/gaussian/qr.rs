//! Exact thin QR over Q(i), including rectangular and deficient sources.
//!
//! Modified Gram-Schmidt proposes unnormalized orthogonal columns. A separate
//! reference pass reinterprets the original source and checks reconstruction,
//! the complete Hermitian Gram matrix and a unit row-echelon minor of R.
//! Positive Gram entries and that minor prove that both factors have the
//! reported rank; no numerical zero decision or omitted-column assumption is
//! trusted. Optional normalization uses positive exact rational square roots.
//!
//! This synchronous same-crate cross-check is not a portable proof-kernel
//! certificate. Source admission, generation, reference checking and output
//! share the Gaussian lane's 32-axis, 20M-step and 256MiB cumulative bounds.
//! Original and unnormalized Q(i) components have at most 4096 bits, and
//! rational reference arithmetic has an 8192-bit temporary bound. Normalized
//! expressions use a separately checked finite grammar: at most 16 nodes,
//! one positive radical with an 8192-bit radicand, and a conservative 16384-bit
//! aggregate rational-coefficient bound. Budget limits can refuse a smaller
//! admitted input before normalization finishes.

use super::{
    BigRational, BudgetMeter, Gaussian, GaussianMatrix, MAX_WORK_BITS, Matrix, MatrixError,
    Operation, Result, Work, check_bits,
};
use fsym_budget::Dimension;
use fsym_core::Expr;
use fsym_core::elementary::sqrt_rational;
use fsym_proof_kernel::hermitian_inertia::MAX_GAUSSIAN_RATIONAL_INPUT_BITS;
use num_traits::{One, Signed, Zero};

/// Untrusted thin factorization candidate, with full-rank witnesses for R.
///
/// q has mutually Hermitian-orthogonal columns; its positive squared norms
/// are supplied separately. The row at each declared pivot in r begins with
/// one, with zero entries before it. Normalization is intentionally separate
/// so every candidate value remains in Q(i) and is checked by exact arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianQrCandidate {
    pub q: Matrix,
    pub r: Matrix,
    pub pivots: Vec<usize>,
    pub norm_squared: Vec<BigRational>,
}

fn invalid(reason: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("Gaussian QR: {reason}"))
}

fn copy(value: &Gaussian, work: &mut Work<'_, impl BudgetMeter>) -> Result<Gaussian> {
    Ok((work.clone_scalar(&value.0)?, work.clone_scalar(&value.1)?))
}

fn inner(
    left: &[Gaussian],
    right: &[Gaussian],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Gaussian> {
    let mut sum = (BigRational::zero(), BigRational::zero());
    work.storage(128)?;
    for (left, right) in left.iter().zip(right) {
        let ac = work.rational(&left.0, &right.0, Operation::Multiply)?;
        let bd = work.rational(&left.1, &right.1, Operation::Multiply)?;
        let ad = work.rational(&left.0, &right.1, Operation::Multiply)?;
        let bc = work.rational(&left.1, &right.0, Operation::Multiply)?;
        let real = work.rational(&ac, &bd, Operation::Add)?;
        let imaginary = work.rational(&ad, &bc, Operation::Subtract)?;
        sum.0 = work.rational(&sum.0, &real, Operation::Add)?;
        sum.1 = work.rational(&sum.1, &imaginary, Operation::Add)?;
    }
    Ok(sum)
}

fn propose(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<GaussianQrCandidate> {
    let source = GaussianMatrix::read(matrix, work)?;
    let maximum_rank = source.rows.min(source.cols);
    let mut columns: Vec<Vec<Gaussian>> = work.vector(maximum_rank)?;
    let mut upper: Vec<Vec<Gaussian>> = work.vector(maximum_rank)?;
    let mut pivots = work.vector(maximum_rank)?;
    let mut norm_squared = work.vector(maximum_rank)?;
    for column in 0..source.cols {
        work.tick(1)?;
        let mut residual = work.vector(source.rows)?;
        for row in 0..source.rows {
            residual.push(copy(source.get(row, column), work)?);
        }
        for index in 0..columns.len() {
            let dot = inner(&columns[index], &residual, work)?;
            let projection = (
                work.rational(&dot.0, &norm_squared[index], Operation::Divide)?,
                work.rational(&dot.1, &norm_squared[index], Operation::Divide)?,
            );
            upper[index][column] = copy(&projection, work)?;
            for (value, basis) in residual.iter_mut().zip(&columns[index]) {
                let term = work.multiply(basis, &projection)?;
                value.0 = work.rational(&value.0, &term.0, Operation::Subtract)?;
                value.1 = work.rational(&value.1, &term.1, Operation::Subtract)?;
            }
        }
        let norm = inner(&residual, &residual, work)?;
        if !norm.1.is_zero() || norm.0.is_negative() {
            return Err(invalid(
                "generator produced a nonreal or negative squared norm",
            ));
        }
        if norm.0.is_zero() {
            continue;
        }
        if columns.len() == maximum_rank {
            return Err(invalid("generator exceeded the matrix rank bound"));
        }
        check_bits(&norm.0, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
        let mut row = work.vector(source.cols)?;
        work.storage(128 * source.cols as u64)?;
        row.resize_with(source.cols, || (BigRational::zero(), BigRational::zero()));
        row[column] = (BigRational::one(), BigRational::zero());
        upper.push(row);
        columns.push(residual);
        pivots.push(column);
        norm_squared.push(norm.0);
    }
    let rank = columns.len();
    let mut q = GaussianMatrix::zeros(source.rows, rank, work)?;
    for row in 0..source.rows {
        for (column, values) in columns.iter().enumerate() {
            q.entries[row * rank + column] = copy(&values[row], work)?;
        }
    }
    let mut r = GaussianMatrix::zeros(rank, source.cols, work)?;
    for (row, values) in upper.into_iter().enumerate() {
        for (column, value) in values.into_iter().enumerate() {
            work.tick(1)?;
            r.entries[row * source.cols + column] = value;
        }
    }
    let result = GaussianQrCandidate {
        q: q.into_matrix(work)?,
        r: r.into_matrix(work)?,
        pivots,
        norm_squared,
    };
    work.tick(1)?;
    Ok(result)
}

fn verify(
    matrix: &Matrix,
    candidate: &GaussianQrCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick(1)?;
    matrix.validate_shape()?;
    candidate.q.validate_shape()?;
    candidate.r.validate_shape()?;
    let rank = candidate.q.cols;
    if rank > matrix.rows.min(matrix.cols)
        || candidate.q.rows != matrix.rows
        || candidate.r.rows != rank
        || candidate.r.cols != matrix.cols
        || candidate.pivots.len() != rank
        || candidate.norm_squared.len() != rank
    {
        return Err(invalid("factor, pivot or norm shape"));
    }
    // Reinterpret all three original expression graphs independently. This
    // admission runs even for rank zero and before any identity can shortcut.
    let source = GaussianMatrix::read(matrix, work)?;
    let q = GaussianMatrix::read(&candidate.q, work)?;
    let r = GaussianMatrix::read(&candidate.r, work)?;
    for (row, &pivot) in candidate.pivots.iter().enumerate() {
        work.tick(1)?;
        if pivot >= matrix.cols || (row != 0 && pivot <= candidate.pivots[row - 1]) {
            return Err(invalid("pivot order or bound"));
        }
        for column in 0..=pivot {
            let value = r.get(row, column);
            work.tick(1)?;
            let real_ok = if column == pivot {
                value.0.is_one()
            } else {
                value.0.is_zero()
            };
            if !real_ok || !value.1.is_zero() {
                return Err(invalid("R unit row-echelon pivot minor"));
            }
        }
        let norm = &candidate.norm_squared[row];
        check_bits(norm, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
        if !norm.is_positive() {
            return Err(invalid("nonpositive squared norm"));
        }
    }
    // Reference Gram reconstruction does not call the projection helper,
    // recurrence, generator, matrix adjoint or expression simplifier.
    for left in 0..rank {
        for right in 0..rank {
            let mut real = BigRational::zero();
            let mut imaginary = BigRational::zero();
            work.storage(128)?;
            for row in 0..source.rows {
                let a = q.get(row, left);
                let b = q.get(row, right);
                let rr = work.rational(&a.0, &b.0, Operation::Multiply)?;
                let ii = work.rational(&a.1, &b.1, Operation::Multiply)?;
                let ri = work.rational(&a.0, &b.1, Operation::Multiply)?;
                let ir = work.rational(&a.1, &b.0, Operation::Multiply)?;
                real = work.rational(&real, &rr, Operation::Add)?;
                real = work.rational(&real, &ii, Operation::Add)?;
                imaginary = work.rational(&imaginary, &ri, Operation::Add)?;
                imaginary = work.rational(&imaginary, &ir, Operation::Subtract)?;
            }
            let real_ok = if left == right {
                work.tick(
                    real.numer().limb_count()
                        + real.denom().limb_count()
                        + candidate.norm_squared[left].numer().limb_count()
                        + candidate.norm_squared[left].denom().limb_count()
                        + 1,
                )?;
                real == candidate.norm_squared[left]
            } else {
                real.is_zero()
            };
            if !real_ok || !imaginary.is_zero() {
                return Err(invalid("Q Hermitian Gram matrix"));
            }
        }
    }
    if !q.multiply(&r, work)?.equal(&source, work)? {
        return Err(invalid("Q R does not reconstruct the original matrix"));
    }
    work.tick(1)
}

/// The canonical square-root helper performs at most 5000 small trial
/// divisions plus one extraction per pair of radicand bits. Precharge its
/// full bounded arithmetic/allocation envelope, with cancellation checks on
/// both sides; no asynchronous or sub-scalar cancellation bound is claimed.
fn square_root(value: &BigRational, work: &mut Work<'_, impl BudgetMeter>) -> Result<Expr> {
    check_bits(value, MAX_GAUSSIAN_RATIONAL_INPUT_BITS)?;
    if !value.is_positive() {
        return Err(invalid("normalization requires a positive squared norm"));
    }
    if value.is_one() {
        work.tick(1)?;
        return Ok(Expr::from_i64(1));
    }
    let bits = value.numer().bits() + value.denom().bits();
    let limbs = bits.div_ceil(64).max(1);
    let trials = if bits < 28 {
        (1u64 << bits.div_ceil(2)).min(10_000) / 2 + 2
    } else {
        5_001
    };
    let rounds = trials + bits;
    work.tick(rounds * limbs * limbs * 8 + (bits + 1) * limbs * limbs)?;
    work.charge_batch(&[
        (
            Dimension::MemoryBytes,
            rounds * (512 + 32 * limbs) * limbs + (bits + 1) * (128 + 16 * limbs),
        ),
        (
            Dimension::AllocationCount,
            rounds * (32 + 4 * limbs) + (bits + 1) * 32,
        ),
    ])
    .map_err(super::stopped)?;
    let result = sqrt_rational(value);
    let square = positive_scale_square(&result, 0, work)?;
    work.tick(1)?;
    if square != *value {
        return Err(invalid("normalization square-root identity"));
    }
    Ok(result)
}

/// Independently interpret only the positive exact scalar grammar emitted by
/// normalization. No square-root generator, factorization or simplifier is
/// invoked. Positivity fixes the principal branch; exact squaring checks the
/// value, rather than trusting a successful generator return.
fn positive_scale_square(
    expression: &Expr,
    depth: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    work.tick(1)?;
    if depth > 4 {
        return Err(invalid("normalization scalar depth"));
    }
    match expression {
        Expr::Integer(integer) => {
            if integer.bits() > MAX_WORK_BITS {
                return Err(invalid("normalization scalar bit limit"));
            }
            work.tick(integer.limb_count() + 1)?;
            work.storage(128 + 8 * integer.limb_count())?;
            let value = BigRational::from_integer(integer.clone());
            if !value.is_positive() {
                return Err(invalid("normalization scalar is not positive"));
            }
            work.rational(&value, &value, Operation::Multiply)
        }
        Expr::Rational(value) => {
            if !value.is_positive() {
                return Err(invalid("normalization scalar is not positive"));
            }
            work.rational(value, value, Operation::Multiply)
        }
        Expr::Pow(base, exponent) => {
            let Expr::Rational(exponent) = exponent.as_ref() else {
                return Err(invalid("normalization scalar exponent"));
            };
            if !exponent.numer().is_one() || exponent.denom() != &fsym_core::BigInt::from(2) {
                return Err(invalid("normalization scalar exponent"));
            }
            let Expr::Integer(integer) = base.as_ref() else {
                return Err(invalid("normalization radical base"));
            };
            if !integer.is_positive() || integer.bits() > MAX_WORK_BITS {
                return Err(invalid("normalization radical base"));
            }
            work.tick(integer.limb_count() + 1)?;
            work.storage(128 + 8 * integer.limb_count())?;
            Ok(BigRational::from_integer(integer.clone()))
        }
        Expr::Mul(factors) if (2..=4).contains(&factors.len()) => {
            let mut result = BigRational::one();
            work.storage(128)?;
            for factor in factors {
                let square = positive_scale_square(factor, depth + 1, work)?;
                result = work.rational(&result, &square, Operation::Multiply)?;
            }
            Ok(result)
        }
        _ => Err(invalid("normalization scalar grammar")),
    }
}

const MAX_NORMALIZED_COEFFICIENT_BITS: u64 = 4 * MAX_GAUSSIAN_RATIONAL_INPUT_BITS;
const MAX_NORMALIZED_NODES: usize = 16;

#[derive(Default)]
struct NormalizedShape {
    nodes: usize,
    coefficient_bits: u64,
    radicals: usize,
    radicand_bits: u64,
    sums: usize,
    imaginary_units: usize,
}

/// Inspect the closed output grammar without allocating or simplifying.
/// A source Gaussian scalar has at most two rational leaves. Its scale has
/// at most one rational leaf and one radical; inverse scaling can double
/// the scale coefficient's bit length. Summing all coefficient bit lengths
/// therefore conservatively bounds every canonicalization cross-product.
fn normalized_shape(
    expression: &Expr,
    depth: usize,
    shape: &mut NormalizedShape,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick(1)?;
    shape.nodes += 1;
    if depth > 4 || shape.nodes > MAX_NORMALIZED_NODES {
        return Err(super::limit("QR normalization expression size"));
    }
    match expression {
        Expr::Integer(integer) => {
            shape.coefficient_bits += integer.bits();
        }
        Expr::Rational(value) => {
            shape.coefficient_bits += value.numer().bits().max(value.denom().bits());
        }
        Expr::Const(fsym_core::Constant::I) => {
            shape.imaginary_units += 1;
        }
        Expr::Pow(base, exponent) => {
            let (Expr::Integer(base), Expr::Rational(exponent)) =
                (base.as_ref(), exponent.as_ref())
            else {
                return Err(invalid("normalized radical grammar"));
            };
            if !base.is_positive()
                || base.bits() > MAX_WORK_BITS
                || !exponent.numer().is_one()
                || exponent.denom() != &fsym_core::BigInt::from(2)
            {
                return Err(invalid("normalized radical grammar"));
            }
            shape.nodes += 2;
            shape.radicals += 1;
            shape.radicand_bits = base.bits();
        }
        Expr::Add(terms) if terms.len() == 2 => {
            shape.sums += 1;
            for term in terms {
                normalized_shape(term, depth + 1, shape, work)?;
            }
        }
        Expr::Mul(factors) if (2..=4).contains(&factors.len()) => {
            for factor in factors {
                normalized_shape(factor, depth + 1, shape, work)?;
            }
        }
        _ => return Err(invalid("normalized scalar grammar")),
    }
    if shape.nodes > MAX_NORMALIZED_NODES
        || shape.coefficient_bits > MAX_NORMALIZED_COEFFICIENT_BITS
        || shape.radicals > 1
        || shape.sums > 1
        || shape.imaginary_units > 1
    {
        return Err(super::limit(
            "QR normalization coefficient or grammar bound",
        ));
    }
    Ok(())
}

fn normalization_product(
    left: &Expr,
    right: &Expr,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    // Inspect both operands together before any cloning or arithmetic. There
    // is at most one radical, so canonicalization cannot combine radicals or
    // start another root extraction. At most one two-term Gaussian sum can
    // distribute; the aggregate coefficient bound pays for both branches.
    let mut shape = NormalizedShape::default();
    normalized_shape(left, 0, &mut shape, work)?;
    normalized_shape(right, 0, &mut shape, work)?;
    let bits = shape.coefficient_bits.max(1);
    let limbs = bits.div_ceil(64);
    // The finite grammar needs fewer than 32 rational operations. For each,
    // reserve two Euclidean divisions per coefficient bit, quadratic limb
    // work, and the cloned operands/results of every division. This is a
    // deliberately conservative cumulative allocation envelope, not a
    // count of simultaneously live allocations.
    let rounds = 32 * (2 * bits + 1);
    let rendered_bits = bits + shape.radicand_bits + 1;
    let rendered_limbs = rendered_bits.div_ceil(64);
    // Core factor ordering can render the radical and Gaussian sum. Pay for
    // decimal conversion and temporary strings as well as rational work.
    let formatting = 8 * shape.nodes as u64 * rendered_bits * rendered_limbs;
    work.tick(rounds * limbs * limbs + formatting + 64 * shape.nodes as u64)?;
    work.charge_batch(&[
        (
            Dimension::MemoryBytes,
            rounds * (128 + 32 * limbs) + 32 * shape.nodes as u64 * (128 + rendered_bits),
        ),
        (
            Dimension::AllocationCount,
            8 * rounds + 16 * shape.nodes as u64 * (rendered_limbs + 1),
        ),
    ])
    .map_err(super::stopped)?;
    let result = left.clone() * right.clone();
    normalized_shape(&result, 0, &mut NormalizedShape::default(), work)?;
    work.tick(1)?;
    Ok(result)
}

fn normalize(
    candidate: GaussianQrCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(Matrix, Matrix)> {
    let rank = candidate.q.cols;
    let mut scales = work.vector(rank)?;
    let mut inverse_scales = work.vector(rank)?;
    for norm in &candidate.norm_squared {
        let inverse = work.rational(&BigRational::one(), norm, Operation::Divide)?;
        let scale = square_root(norm, work)?;
        let inverse = work.scalar_expression(inverse)?;
        inverse_scales.push(normalization_product(&inverse, &scale, work)?);
        scales.push(scale);
    }
    let mut q = work.vector(candidate.q.data.len())?;
    let mut r = work.vector(candidate.r.data.len())?;
    for row in 0..candidate.q.rows {
        for (column, scale) in inverse_scales.iter().enumerate() {
            let value = &candidate.q.data[row * rank + column];
            q.push(normalization_product(value, scale, work)?);
        }
    }
    for (row, scale) in scales.iter().enumerate() {
        for column in 0..candidate.r.cols {
            let value = &candidate.r.data[row * candidate.r.cols + column];
            r.push(normalization_product(value, scale, work)?);
        }
    }
    let q = Matrix::new(candidate.q.rows, rank, q)?;
    let r = Matrix::new(rank, candidate.r.cols, r)?;
    work.tick(1)?;
    Ok((q, r))
}

impl Matrix {
    /// Propose an exact thin Q(i) QR factorization without granting evidence.
    /// The caller must independently check it against the original source.
    pub fn gaussian_qr_candidate_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianQrCandidate> {
        propose(self, &mut Work::new(meter))
    }

    /// Independently verify the original source, all Hermitian Gram entries,
    /// reconstruction and the unit pivot minor establishing complete rank.
    pub fn verify_gaussian_qr_candidate(
        &self,
        candidate: &GaussianQrCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        verify(self, candidate, &mut Work::new(meter))
    }

    /// Return exact unnormalized orthogonal factors after independent checking.
    /// Both phases share one cumulative caller meter and intrinsic envelope.
    pub fn metered_gaussian_qr(&self, meter: &mut impl BudgetMeter) -> Result<GaussianQrCandidate> {
        let mut work = Work::new(meter);
        let candidate = propose(self, &mut work)?;
        verify(self, &candidate, &mut work)?;
        work.tick(1)?;
        Ok(candidate)
    }

    /// Return thin orthonormal Q and echelon R, with A = Q R and Q* Q = I.
    /// Positive exact rational square roots fix the normalization convention.
    /// All phases, including normalized output, share one cumulative meter.
    pub fn metered_gaussian_qr_normalized(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<(Self, Self)> {
        let mut work = Work::new(meter);
        let candidate = propose(self, &mut work)?;
        verify(self, &candidate, &mut work)?;
        normalize(candidate, &mut work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;
    use std::sync::Arc;

    #[test]
    fn independent_scale_checker_accepts_exact_nonsquare_and_fractional_anchors() {
        let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
        let root_two = Expr::Pow(Arc::new(Expr::from_i64(2)), Arc::new(half.clone()));
        let root_thirteen = Expr::Pow(Arc::new(Expr::from_i64(13)), Arc::new(half));
        for (scale, square) in [
            (Expr::from_i64(5), BigRational::from(25)),
            (root_two, BigRational::from(2)),
            (
                Expr::Mul(vec![Expr::rational(1, 3).unwrap(), root_thirteen]),
                BigRational::new(13.into(), 9.into()),
            ),
        ] {
            let mut meter = Unbounded;
            let mut work = Work::new(&mut meter);
            assert_eq!(positive_scale_square(&scale, 0, &mut work).unwrap(), square);
            assert_eq!(
                positive_scale_square(&sqrt_rational(&square), 0, &mut work).unwrap(),
                square
            );
        }
    }

    #[test]
    fn independent_scale_checker_rejects_wrong_branches_and_unknown_scalar_shapes() {
        let root_two = Expr::Pow(
            Arc::new(Expr::from_i64(2)),
            Arc::new(Expr::rational(1, 2).unwrap()),
        );
        for forged in [
            Expr::from_i64(0),
            Expr::from_i64(-1),
            Expr::Mul(vec![Expr::from_i64(-1), root_two]),
            Expr::Pow(
                Arc::new(Expr::from_i64(-2)),
                Arc::new(Expr::rational(1, 2).unwrap()),
            ),
            Expr::Pow(
                Arc::new(Expr::from_i64(2)),
                Arc::new(Expr::rational(1, 3).unwrap()),
            ),
            Expr::symbol("positive_but_unproved"),
        ] {
            assert!(matches!(
                positive_scale_square(&forged, 0, &mut Work::new(&mut Unbounded)),
                Err(MatrixError::InvalidCertificate(_))
            ));
        }
        // A positive but incorrect candidate remains distinguishable by its
        // exact square: positive syntax alone never validates a norm.
        assert_ne!(
            positive_scale_square(&Expr::from_i64(3), 0, &mut Work::new(&mut Unbounded)).unwrap(),
            BigRational::from(2),
        );
    }
    #[test]
    fn normalization_products_bound_coefficients_radicals_and_grammar_before_arithmetic() {
        let huge = Expr::Integer(fsym_core::BigInt::from(1) << 8_192u32);
        let left = Expr::Mul(vec![huge.clone(), Expr::Const(fsym_core::Constant::I)]);
        assert!(matches!(
            normalization_product(&left, &huge, &mut Work::new(&mut Unbounded)),
            Err(MatrixError::ResourceLimit(_))
        ));
        let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
        let radical = Expr::Pow(Arc::new(Expr::from_i64(2)), Arc::new(half));
        assert!(matches!(
            normalization_product(&radical, &radical, &mut Work::new(&mut Unbounded)),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(
            normalization_product(
                &Expr::symbol("x"),
                &Expr::from_i64(1),
                &mut Work::new(&mut Unbounded),
            )
            .is_err()
        );
    }
}
