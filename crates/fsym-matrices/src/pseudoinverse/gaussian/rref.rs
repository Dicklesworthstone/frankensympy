//! Exact Q(i) row reduction and complete affine families.
//!
//! Elimination proposes both directions of its row transform. A separate
//! reference pass reinterprets the original source, checks U A = R and
//! U V = V U = I, and checks every canonical RREF condition. Rank and
//! nullity therefore do not depend on a generator's pivot report.
//! Affine acceptance checks original residuals and an identity block on
//! free coordinates, establishing the complete kernel.
//!
//! Sources, candidates, verification and parameter publication share the
//! existing cumulative Gaussian ledger. Every matrix axis is at most 32,
//! independently for coefficients and RHS. This is synchronous same-crate
//! reference checking, not a portable proof-kernel certificate or protected
//! verifier-resource reservation.

use super::{
    BigRational, BudgetMeter, Constant, Expr, Gaussian, GaussianMatrix, Matrix, MatrixError,
    Operation, Result, Work, limit,
};
use num_traits::{One, Zero};
use std::fmt::Write;

/// Untrusted exact row-reduction proposal for one original matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianRrefCandidate {
    pub reduced: Matrix,
    pub transform: Matrix,
    pub inverse_transform: Matrix,
    pub pivots: Vec<usize>,
}

/// An untrusted affine proposal. Both optional values are absent exactly
/// when a checked zero row of U A has a nonzero entry in U B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianAffineCandidate {
    pub reduction: GaussianRrefCandidate,
    pub particular: Option<Matrix>,
    pub kernel: Option<Matrix>,
}

/// All solutions are particular + kernel * T, with one independent complex
/// parameter per free coordinate and RHS column. Free rows of the particular
/// are zero; free rows of the kernel are the identity. Sources are normalized
/// only after original-domain admission. Private fields prevent constructing
/// an acceptance receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedGaussianAffineSolution {
    coefficients: Matrix,
    rhs: Matrix,
    particular: Matrix,
    kernel: Matrix,
    free_columns: Vec<usize>,
}

impl VerifiedGaussianAffineSolution {
    pub fn coefficients(&self) -> &Matrix {
        &self.coefficients
    }

    pub fn rhs(&self) -> &Matrix {
        &self.rhs
    }

    pub fn particular(&self) -> &Matrix {
        &self.particular
    }

    pub fn kernel(&self) -> &Matrix {
        &self.kernel
    }

    pub fn free_columns(&self) -> &[usize] {
        &self.free_columns
    }

    pub fn into_parts(self) -> (Matrix, Matrix, Vec<usize>) {
        (self.particular, self.kernel, self.free_columns)
    }
}

struct CheckedReduction {
    original: GaussianMatrix,
    reduced: GaussianMatrix,
    transform: GaussianMatrix,
    pivots: Vec<usize>,
}

fn invalid(message: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("Gaussian row reduction: {message}"))
}

fn is_zero(value: &Gaussian) -> bool {
    value.0.is_zero() && value.1.is_zero()
}

fn is_one(value: &Gaussian) -> bool {
    value.0.is_one() && value.1.is_zero()
}

fn copy(value: &Gaussian, work: &mut Work<'_, impl BudgetMeter>) -> Result<Gaussian> {
    Ok((work.clone_scalar(&value.0)?, work.clone_scalar(&value.1)?))
}

fn negative(value: &Gaussian, work: &mut Work<'_, impl BudgetMeter>) -> Result<Gaussian> {
    Ok((work.negate(&value.0)?, work.negate(&value.1)?))
}

fn subtract(
    left: &Gaussian,
    right: &Gaussian,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Gaussian> {
    Ok((
        work.rational(&left.0, &right.0, Operation::Subtract)?,
        work.rational(&left.1, &right.1, Operation::Subtract)?,
    ))
}

fn reciprocal(value: &Gaussian, work: &mut Work<'_, impl BudgetMeter>) -> Result<Gaussian> {
    if is_zero(value) {
        return Err(MatrixError::DivisionByZero);
    }
    let real_squared = work.rational(&value.0, &value.0, Operation::Multiply)?;
    let imaginary_squared = work.rational(&value.1, &value.1, Operation::Multiply)?;
    let magnitude = work.rational(&real_squared, &imaginary_squared, Operation::Add)?;
    let imaginary = work.negate(&value.1)?;
    Ok((
        work.rational(&value.0, &magnitude, Operation::Divide)?,
        work.rational(&imaginary, &magnitude, Operation::Divide)?,
    ))
}

fn identity(size: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<GaussianMatrix> {
    let mut result = GaussianMatrix::zeros(size, size, work)?;
    for index in 0..size {
        work.tick(1)?;
        work.storage(64)?;
        result.entries[index * size + index].0 = BigRational::one();
    }
    Ok(result)
}

fn require_identity(matrix: &GaussianMatrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
    if matrix.rows != matrix.cols {
        return Err(invalid("row-transform product is not square"));
    }
    for row in 0..matrix.rows {
        for column in 0..matrix.cols {
            work.tick(1)?;
            let entry = matrix.get(row, column);
            let accepted = if row == column {
                is_one(entry)
            } else {
                is_zero(entry)
            };
            if !accepted {
                return Err(invalid("row transforms are not inverses"));
            }
        }
    }
    work.tick(1)
}

fn require_zero(matrix: &GaussianMatrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
    for entry in &matrix.entries {
        work.tick(1)?;
        if !is_zero(entry) {
            return Err(invalid("original kernel residual is nonzero"));
        }
    }
    work.tick(1)
}

fn swap_rows(
    matrix: &mut GaussianMatrix,
    first: usize,
    second: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    for column in 0..matrix.cols {
        work.tick(1)?;
        matrix
            .entries
            .swap(first * matrix.cols + column, second * matrix.cols + column);
    }
    Ok(())
}

fn scale_row(
    matrix: &mut GaussianMatrix,
    row: usize,
    factor: &Gaussian,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    for column in 0..matrix.cols {
        work.tick(1)?;
        let value = work.multiply(matrix.get(row, column), factor)?;
        matrix.entries[row * matrix.cols + column] = value;
    }
    Ok(())
}

fn eliminate_row(
    matrix: &mut GaussianMatrix,
    target: usize,
    pivot: usize,
    factor: &Gaussian,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    for column in 0..matrix.cols {
        work.tick(1)?;
        let multiple = work.multiply(factor, matrix.get(pivot, column))?;
        let value = subtract(matrix.get(target, column), &multiple, work)?;
        matrix.entries[target * matrix.cols + column] = value;
    }
    Ok(())
}

fn generate_reduction(
    mut reduced: GaussianMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianRrefCandidate> {
    let (rows, columns) = (reduced.rows, reduced.cols);
    let mut transform = identity(rows, work)?;
    let mut inverse = identity(rows, work)?;
    let mut pivots = work.vector(rows.min(columns))?;
    for column in 0..columns {
        work.tick(1)?;
        let pivot = pivots.len();
        if pivot == rows {
            break;
        }
        let mut selected = None;
        for row in pivot..rows {
            work.tick(1)?;
            if !is_zero(reduced.get(row, column)) {
                selected = Some(row);
                break;
            }
        }
        let Some(selected) = selected else {
            continue;
        };
        if selected != pivot {
            swap_rows(&mut reduced, pivot, selected, work)?;
            swap_rows(&mut transform, pivot, selected, work)?;
            // U <- P U and V <- V P preserve V=U^-1.
            for row in 0..rows {
                work.tick(1)?;
                inverse
                    .entries
                    .swap(row * rows + pivot, row * rows + selected);
            }
        }
        let divisor = copy(reduced.get(pivot, column), work)?;
        let inverse_divisor = reciprocal(&divisor, work)?;
        scale_row(&mut reduced, pivot, &inverse_divisor, work)?;
        scale_row(&mut transform, pivot, &inverse_divisor, work)?;
        for row in 0..rows {
            work.tick(1)?;
            let value = work.multiply(inverse.get(row, pivot), &divisor)?;
            inverse.entries[row * rows + pivot] = value;
        }
        for row in 0..rows {
            work.tick(1)?;
            if row == pivot || is_zero(reduced.get(row, column)) {
                continue;
            }
            let factor = copy(reduced.get(row, column), work)?;
            eliminate_row(&mut reduced, row, pivot, &factor, work)?;
            eliminate_row(&mut transform, row, pivot, &factor, work)?;
            // U <- (I-f*e_row*e_pivot^T) U, so V's pivot column
            // gains f times its target column. No row of V is changed.
            for inverse_row in 0..rows {
                work.tick(1)?;
                let multiple = work.multiply(inverse.get(inverse_row, row), &factor)?;
                let value = work.add(inverse.get(inverse_row, pivot), &multiple)?;
                inverse.entries[inverse_row * rows + pivot] = value;
            }
        }
        pivots.push(column);
    }
    let candidate = GaussianRrefCandidate {
        reduced: reduced.into_matrix(work)?,
        transform: transform.into_matrix(work)?,
        inverse_transform: inverse.into_matrix(work)?,
        pivots,
    };
    work.tick(1)?;
    Ok(candidate)
}

fn check_reduction(
    matrix: &Matrix,
    candidate: &GaussianRrefCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<CheckedReduction> {
    // Interpret the original before examining a zero/rank shortcut. Neither
    // cancellation nor a supplied rank can erase a hidden original pole.
    let original = GaussianMatrix::read(matrix, work)?;
    let (rows, columns) = (original.rows, original.cols);
    if candidate.reduced.rows != rows
        || candidate.reduced.cols != columns
        || candidate.transform.rows != rows
        || candidate.transform.cols != rows
        || candidate.inverse_transform.rows != rows
        || candidate.inverse_transform.cols != rows
        || candidate.pivots.len() > rows.min(columns)
    {
        return Err(invalid("source or witness shape"));
    }
    let reduced = GaussianMatrix::read(&candidate.reduced, work)?;
    let transform = GaussianMatrix::read(&candidate.transform, work)?;
    let inverse = GaussianMatrix::read(&candidate.inverse_transform, work)?;
    let rank = candidate.pivots.len();
    let mut previous = None;
    let mut pivots = work.vector(rank)?;
    for (row, &column) in candidate.pivots.iter().enumerate() {
        work.tick(1)?;
        if column >= columns || previous.is_some_and(|old| old >= column) {
            return Err(invalid("pivot columns are not strictly increasing"));
        }
        previous = Some(column);
        for before in 0..column {
            work.tick(1)?;
            if !is_zero(reduced.get(row, before)) {
                return Err(invalid("a pivot is not the first nonzero row entry"));
            }
        }
        for other in 0..rows {
            work.tick(1)?;
            let value = reduced.get(other, column);
            let accepted = if other == row {
                is_one(value)
            } else {
                is_zero(value)
            };
            if !accepted {
                return Err(invalid("a pivot column is not a canonical unit column"));
            }
        }
        pivots.push(column);
    }
    for row in rank..rows {
        for column in 0..columns {
            work.tick(1)?;
            if !is_zero(reduced.get(row, column)) {
                return Err(invalid("an unreported nonzero row remains"));
            }
        }
    }
    // These reference products do not use elimination, pivot selection,
    // symbolic simplification or the generator's elementary row updates.
    require_identity(&transform.multiply(&inverse, work)?, work)?;
    require_identity(&inverse.multiply(&transform, work)?, work)?;
    if !transform.multiply(&original, work)?.equal(&reduced, work)? {
        return Err(invalid("row transform does not reduce the original source"));
    }
    work.tick(1)?;
    Ok(CheckedReduction {
        original,
        reduced,
        transform,
        pivots,
    })
}

fn free_columns(
    columns: usize,
    pivots: &[usize],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<usize>> {
    let mut free = work.vector(columns - pivots.len())?;
    let mut pivot_index = 0;
    for column in 0..columns {
        work.tick(1)?;
        if pivots.get(pivot_index) == Some(&column) {
            pivot_index += 1;
        } else {
            free.push(column);
        }
    }
    Ok(free)
}

fn kernel_from_reduction(
    reduced: &GaussianMatrix,
    pivots: &[usize],
    free: &[usize],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianMatrix> {
    let mut kernel = GaussianMatrix::zeros(reduced.cols, free.len(), work)?;
    for (index, &column) in free.iter().enumerate() {
        work.tick(1)?;
        work.storage(64)?;
        kernel.entries[column * free.len() + index].0 = BigRational::one();
        for (row, &pivot) in pivots.iter().enumerate() {
            work.tick(1)?;
            kernel.entries[pivot * free.len() + index] = negative(reduced.get(row, column), work)?;
        }
    }
    Ok(kernel)
}

fn check_kernel(
    original: &GaussianMatrix,
    kernel: &GaussianMatrix,
    free: &[usize],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    if kernel.rows != original.cols || kernel.cols != free.len() {
        return Err(invalid("kernel shape does not match the checked nullity"));
    }
    // A*K=0 alone accepts empty/dependent lists. Independently established
    // rank and the free-coordinate identity prove complete independence.
    for (row, &coordinate) in free.iter().enumerate() {
        for column in 0..free.len() {
            work.tick(1)?;
            let value = kernel.get(coordinate, column);
            let accepted = if row == column {
                is_one(value)
            } else {
                is_zero(value)
            };
            if !accepted {
                return Err(invalid("kernel free-coordinate block is not the identity"));
            }
        }
    }
    require_zero(&original.multiply(kernel, work)?, work)
}

fn consistent(
    transformed_rhs: &GaussianMatrix,
    rank: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    for row in rank..transformed_rhs.rows {
        for column in 0..transformed_rhs.cols {
            work.tick(1)?;
            if !is_zero(transformed_rhs.get(row, column)) {
                return Ok(false);
            }
        }
    }
    work.tick(1)?;
    Ok(true)
}

fn generate_affine(
    matrix: &Matrix,
    rhs: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianAffineCandidate> {
    // Read A and B separately: augmentation would incorrectly impose the
    // 32-axis bound on unknowns+RHS columns.
    let original = GaussianMatrix::read(matrix, work)?;
    let b = GaussianMatrix::read(rhs, work)?;
    if original.rows != b.rows {
        return Err(MatrixError::ShapeMismatch(
            original.rows,
            original.cols,
            b.rows,
            b.cols,
        ));
    }
    let reduction = generate_reduction(original, work)?;
    let reduced = GaussianMatrix::read(&reduction.reduced, work)?;
    let transform = GaussianMatrix::read(&reduction.transform, work)?;
    let transformed_rhs = transform.multiply(&b, work)?;
    if !consistent(&transformed_rhs, reduction.pivots.len(), work)? {
        work.tick(1)?;
        return Ok(GaussianAffineCandidate {
            reduction,
            particular: None,
            kernel: None,
        });
    }
    let free = free_columns(reduced.cols, &reduction.pivots, work)?;
    let kernel = kernel_from_reduction(&reduced, &reduction.pivots, &free, work)?;
    let mut particular = GaussianMatrix::zeros(reduced.cols, b.cols, work)?;
    for (row, &pivot) in reduction.pivots.iter().enumerate() {
        for column in 0..b.cols {
            work.tick(1)?;
            particular.entries[pivot * b.cols + column] =
                copy(transformed_rhs.get(row, column), work)?;
        }
    }
    let result = GaussianAffineCandidate {
        reduction,
        particular: Some(particular.into_matrix(work)?),
        kernel: Some(kernel.into_matrix(work)?),
    };
    work.tick(1)?;
    Ok(result)
}

fn check_affine(
    matrix: &Matrix,
    rhs: &Matrix,
    candidate: &GaussianAffineCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Option<VerifiedGaussianAffineSolution>> {
    let reduction = check_reduction(matrix, &candidate.reduction, work)?;
    let b = GaussianMatrix::read(rhs, work)?;
    if reduction.original.rows != b.rows {
        return Err(MatrixError::ShapeMismatch(
            reduction.original.rows,
            reduction.original.cols,
            b.rows,
            b.cols,
        ));
    }
    let transformed_rhs = reduction.transform.multiply(&b, work)?;
    if !consistent(&transformed_rhs, reduction.pivots.len(), work)? {
        if candidate.particular.is_some() || candidate.kernel.is_some() {
            return Err(invalid("an inconsistent original system has proposed values"));
        }
        work.tick(1)?;
        return Ok(None);
    }
    let (Some(particular), Some(kernel)) = (&candidate.particular, &candidate.kernel) else {
        return Err(invalid(
            "a consistent original system is missing its complete family",
        ));
    };
    let free = free_columns(reduction.original.cols, &reduction.pivots, work)?;
    if particular.rows != reduction.original.cols
        || particular.cols != b.cols
        || kernel.rows != reduction.original.cols
        || kernel.cols != free.len()
    {
        return Err(invalid("affine output shapes"));
    }
    let x = GaussianMatrix::read(particular, work)?;
    let kernel = GaussianMatrix::read(kernel, work)?;
    for &coordinate in &free {
        for column in 0..x.cols {
            work.tick(1)?;
            if !is_zero(x.get(coordinate, column)) {
                return Err(invalid("particular has a nonzero free coordinate"));
            }
        }
    }
    if !reduction.original.multiply(&x, work)?.equal(&b, work)? {
        return Err(invalid("particular fails the original source residual"));
    }
    check_kernel(&reduction.original, &kernel, &free, work)?;
    let accepted = VerifiedGaussianAffineSolution {
        coefficients: reduction.original.into_matrix(work)?,
        rhs: b.into_matrix(work)?,
        particular: x.into_matrix(work)?,
        kernel: kernel.into_matrix(work)?,
        free_columns: free,
    };
    work.tick(1)?;
    Ok(Some(accepted))
}

fn scalar_expression(value: &Gaussian, work: &mut Work<'_, impl BudgetMeter>) -> Result<Expr> {
    let real = work.clone_scalar(&value.0)?;
    let imaginary = work.clone_scalar(&value.1)?;
    if imaginary.is_zero() {
        return work.scalar_expression(real);
    }
    let imaginary = if imaginary.is_one() {
        Expr::Const(Constant::I)
    } else {
        let mut factors = work.vector(2)?;
        factors.push(work.scalar_expression(imaginary)?);
        factors.push(Expr::Const(Constant::I));
        Expr::Mul(factors)
    };
    if real.is_zero() {
        Ok(imaginary)
    } else {
        let mut terms = work.vector(2)?;
        terms.push(work.scalar_expression(real)?);
        terms.push(imaginary);
        Ok(Expr::Add(terms))
    }
}

fn parameter(index: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Expr> {
    let digits = if index == 0 {
        1
    } else {
        index.ilog10() as usize + 1
    };
    work.tick((digits + 3) as u64)?;
    work.storage((digits + 3 + std::mem::size_of::<fsym_core::Symbol>()) as u64)?;
    let mut name = String::new();
    name.try_reserve_exact(digits + 3)
        .map_err(|_| limit("affine parameter name allocation refused"))?;
    write!(name, "tau{index}").map_err(|_| limit("affine parameter name construction failed"))?;
    Ok(Expr::symbol(name))
}

fn ordered_sum(
    mut terms: Vec<Expr>,
    maximum_bits: u64,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    work.tick(1)?;
    match terms.len() {
        0 => {
            work.storage(64)?;
            Ok(Expr::from_i64(0))
        }
        1 => terms
            .pop()
            .ok_or_else(|| invalid("symbolic sum lost its only term")),
        count => {
            // At most 32 independent parameter terms and one closed scalar;
            // coefficients have at most two Q(i) components. Charge the
            // comparator's scalar/name temporaries before this bounded sort.
            let levels = u64::from(usize::BITS - (count - 1).leading_zeros());
            let comparisons = 8 * count as u64 * levels;
            let limbs = maximum_bits.max(1).div_ceil(64) + 1;
            work.tick(comparisons * (limbs * limbs + 32))?;
            work.charge_batch(&[
                (
                    fsym_budget::Dimension::MemoryBytes,
                    comparisons * (8 * maximum_bits.max(1) + 256),
                ),
                (
                    fsym_budget::Dimension::AllocationCount,
                    comparisons * (4 * limbs + 16),
                ),
            ])
            .map_err(super::stopped)?;
            fsym_core::canonicalize_add_args(&mut terms);
            work.tick(1)?;
            Ok(Expr::Add(terms))
        }
    }
}

fn render_affine(
    family: VerifiedGaussianAffineSolution,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(Matrix, Matrix, Vec<usize>)> {
    let particular = GaussianMatrix::read(&family.particular, work)?;
    let kernel = GaussianMatrix::read(&family.kernel, work)?;
    let rhs_columns = particular.cols;
    let mut parameters = work.vector(kernel.cols * rhs_columns)?;
    for index in 0..kernel.cols * rhs_columns {
        parameters.push(parameter(index, work)?);
    }
    let parameters = Matrix::new(kernel.cols, rhs_columns, parameters)?;
    let mut solution = work.vector(particular.rows * rhs_columns)?;
    for row in 0..particular.rows {
        for column in 0..rhs_columns {
            work.tick(1)?;
            let mut terms = work.vector(kernel.cols + 1)?;
            let value = particular.get(row, column);
            let mut maximum_bits = value
                .0
                .numer()
                .bits()
                .max(value.0.denom().bits())
                .max(value.1.numer().bits())
                .max(value.1.denom().bits());
            if !is_zero(value) {
                terms.push(scalar_expression(value, work)?);
            }
            for index in 0..kernel.cols {
                work.tick(1)?;
                let coefficient = kernel.get(row, index);
                if is_zero(coefficient) {
                    continue;
                }
                let symbol = parameter(index * rhs_columns + column, work)?;
                if is_one(coefficient) {
                    terms.push(symbol);
                } else {
                    maximum_bits = maximum_bits
                        .max(coefficient.0.numer().bits())
                        .max(coefficient.0.denom().bits())
                        .max(coefficient.1.numer().bits())
                        .max(coefficient.1.denom().bits());
                    let mut factors = work.vector(2)?;
                    factors.push(scalar_expression(coefficient, work)?);
                    factors.push(symbol);
                    terms.push(Expr::Mul(factors));
                }
            }
            solution.push(ordered_sum(terms, maximum_bits, work)?);
        }
    }
    let solution = Matrix::new(particular.rows, rhs_columns, solution)?;
    work.tick(1)?;
    Ok((solution, parameters, family.free_columns))
}

impl Matrix {
    /// Generate an untrusted exact Q(i) RREF and invertible row-transform
    /// witness. Use the independent verifier before accepting its pivots.
    pub fn gaussian_rref_candidate_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianRrefCandidate> {
        let mut work = Work::new(meter);
        let source = GaussianMatrix::read(self, &mut work)?;
        generate_reduction(source, &mut work)
    }

    /// Independently check original row equivalence, inverse transforms and
    /// every RREF/pivot condition without running elimination again.
    pub fn verify_gaussian_rref_candidate(
        &self,
        candidate: &GaussianRrefCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        let mut work = Work::new(meter);
        check_reduction(self, candidate, &mut work)?;
        work.tick(1)
    }

    /// Bounded exact Q(i) RREF, accepted against the original matrix before
    /// returning its canonical pivot columns. One cumulative ledger covers
    /// generation, independent verification and output construction.
    pub fn metered_gaussian_rref(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<(Self, Vec<usize>)> {
        let mut work = Work::new(meter);
        let source = GaussianMatrix::read(self, &mut work)?;
        let candidate = generate_reduction(source, &mut work)?;
        let checked = check_reduction(self, &candidate, &mut work)?;
        let reduced = checked.reduced.into_matrix(&mut work)?;
        work.tick(1)?;
        Ok((reduced, checked.pivots))
    }

    /// Complete exact Q(i) nullspace with a canonical unit block on free
    /// coordinates. Original residuals, rank and independence are checked.
    pub fn metered_gaussian_nullspace(&self, meter: &mut impl BudgetMeter) -> Result<Vec<Self>> {
        let mut work = Work::new(meter);
        let source = GaussianMatrix::read(self, &mut work)?;
        let candidate = generate_reduction(source, &mut work)?;
        let checked = check_reduction(self, &candidate, &mut work)?;
        let free = free_columns(self.cols, &checked.pivots, &mut work)?;
        let kernel = kernel_from_reduction(&checked.reduced, &checked.pivots, &free, &mut work)?;
        check_kernel(&checked.original, &kernel, &free, &mut work)?;
        let mut result = work.vector(free.len())?;
        for index in 0..free.len() {
            let mut vector = GaussianMatrix::zeros(self.cols, 1, &mut work)?;
            for row in 0..self.cols {
                vector.entries[row] = copy(kernel.get(row, index), &mut work)?;
            }
            result.push(vector.into_matrix(&mut work)?);
        }
        work.tick(1)?;
        Ok(result)
    }

    /// Propose either a complete exact affine family or an inconsistency
    /// witness, retaining the same independently checkable RREF certificate.
    pub fn gaussian_affine_candidate_with_meter(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianAffineCandidate> {
        generate_affine(self, rhs, &mut Work::new(meter))
    }

    /// Check complete-family or inconsistency claims against both originals.
    /// Residual-only or incomplete kernel witnesses cannot be accepted.
    pub fn verify_gaussian_affine_candidate(
        &self,
        rhs: &Self,
        candidate: &GaussianAffineCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<Option<VerifiedGaussianAffineSolution>> {
        check_affine(self, rhs, candidate, &mut Work::new(meter))
    }

    /// Return the canonical particular and complete kernel, or independently
    /// checked inconsistency, under a single bounded work ledger.
    pub fn metered_gaussian_affine_solution(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<Option<VerifiedGaussianAffineSolution>> {
        let mut work = Work::new(meter);
        let candidate = generate_affine(self, rhs, &mut work)?;
        check_affine(self, rhs, &candidate, &mut work)
    }

    /// Complete multi-RHS Q(i) family with row-major tau0, tau1, ... symbols.
    /// Parameters and all output terms are constructed only after independent
    /// acceptance and within the original cumulative work ledger.
    pub fn metered_gaussian_gauss_jordan_solve(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<Option<(Self, Self, Vec<usize>)>> {
        let mut work = Work::new(meter);
        let candidate = generate_affine(self, rhs, &mut work)?;
        let Some(family) = check_affine(self, rhs, &candidate, &mut work)? else {
            return Ok(None);
        };
        render_affine(family, &mut work).map(Some)
    }
}
