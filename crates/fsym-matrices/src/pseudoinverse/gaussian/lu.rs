//! Rectangular exact LU over Q(i), including deficient and empty matrices.
//!
//! Elimination selects the first nonzero row in each successive nonzero
//! column and records its row swaps. A separate reference pass reinterprets
//! the original source, reconstructs the permutation from those swaps and
//! checks P A = L U. Unit lower-triangular L and the complete upper-echelon
//! structure of U independently establish the reported rank. No numerical
//! pivot decision or generator-reported rank is trusted.
//!
//! P and L are square with the original row count, and U retains the original
//! rectangular shape. Sources, candidates, checks and publication share the
//! Gaussian lane's 32-axis, 4096-bit input/output, 8192-bit intermediate,
//! 20M-step and 256MiB cumulative envelopes. This is same-crate reference
//! checking, not a portable proof or protected verifier-budget reservation.

use super::{
    BigRational, BudgetMeter, Gaussian, GaussianMatrix, Matrix, MatrixError, Operation, Result,
    Work,
};
use num_traits::{One, Zero};

/// Untrusted P A = L U proposal. Applying `row_swaps` to I must give P.
///
/// L is unit lower triangular. The nonzero rows of U come first, and the
/// leading nonzero column of each row is the corresponding `pivots` entry.
/// The strict echelon conditions and invertible P/L prove complete rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianLuCandidate {
    pub p: Matrix,
    pub l: Matrix,
    pub u: Matrix,
    pub row_swaps: Vec<(usize, usize)>,
    pub pivots: Vec<usize>,
}

fn invalid(reason: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("Gaussian LU: {reason}"))
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

fn identity(size: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<GaussianMatrix> {
    let mut matrix = GaussianMatrix::zeros(size, size, work)?;
    for index in 0..size {
        work.tick(1)?;
        work.storage(64)?;
        matrix.entries[index * size + index].0 = BigRational::one();
    }
    Ok(matrix)
}

fn quotient(
    numerator: &Gaussian,
    denominator: &Gaussian,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Gaussian> {
    if is_zero(denominator) {
        return Err(MatrixError::DivisionByZero);
    }
    // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i)/(c*c+d*d).
    let ac = work.rational(&numerator.0, &denominator.0, Operation::Multiply)?;
    let bd = work.rational(&numerator.1, &denominator.1, Operation::Multiply)?;
    let bc = work.rational(&numerator.1, &denominator.0, Operation::Multiply)?;
    let ad = work.rational(&numerator.0, &denominator.1, Operation::Multiply)?;
    let cc = work.rational(&denominator.0, &denominator.0, Operation::Multiply)?;
    let dd = work.rational(&denominator.1, &denominator.1, Operation::Multiply)?;
    let norm = work.rational(&cc, &dd, Operation::Add)?;
    let real = work.rational(&ac, &bd, Operation::Add)?;
    let imaginary = work.rational(&bc, &ad, Operation::Subtract)?;
    Ok((
        work.rational(&real, &norm, Operation::Divide)?,
        work.rational(&imaginary, &norm, Operation::Divide)?,
    ))
}

fn propose(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<GaussianLuCandidate> {
    let mut upper = GaussianMatrix::read(matrix, work)?;
    let (rows, columns) = (upper.rows, upper.cols);
    let mut lower = identity(rows, work)?;
    let mut permutation = identity(rows, work)?;
    let mut row_swaps = work.vector(rows.min(columns))?;
    let mut pivots = work.vector(rows.min(columns))?;
    let mut pivot_row = 0;

    for column in 0..columns {
        if pivot_row == rows {
            break;
        }
        let mut selected = None;
        for row in pivot_row..rows {
            work.tick(1)?;
            if !is_zero(upper.get(row, column)) {
                selected = Some(row);
                break;
            }
        }
        let Some(selected) = selected else {
            continue;
        };
        if selected != pivot_row {
            work.tick((columns + rows + pivot_row) as u64 + 1)?;
            for index in 0..columns {
                upper
                    .entries
                    .swap(pivot_row * columns + index, selected * columns + index);
            }
            for index in 0..rows {
                permutation
                    .entries
                    .swap(pivot_row * rows + index, selected * rows + index);
            }
            // Previously stored multipliers follow the swapped original
            // rows. The unit diagonal and future multiplier columns stay put.
            for index in 0..pivot_row {
                lower
                    .entries
                    .swap(pivot_row * rows + index, selected * rows + index);
            }
            row_swaps.push((pivot_row, selected));
        }
        let pivot = copy(upper.get(pivot_row, column), work)?;
        for row in (pivot_row + 1)..rows {
            work.tick(1)?;
            if is_zero(upper.get(row, column)) {
                continue;
            }
            let multiplier = quotient(upper.get(row, column), &pivot, work)?;
            for index in column..columns {
                let product = work.multiply(&multiplier, upper.get(pivot_row, index))?;
                let value = upper.get(row, index);
                let real = work.rational(&value.0, &product.0, Operation::Subtract)?;
                let imaginary = work.rational(&value.1, &product.1, Operation::Subtract)?;
                upper.entries[row * columns + index] = (real, imaginary);
            }
            lower.entries[row * rows + pivot_row] = multiplier;
        }
        pivots.push(column);
        pivot_row += 1;
    }
    let candidate = GaussianLuCandidate {
        p: permutation.into_matrix(work)?,
        l: lower.into_matrix(work)?,
        u: upper.into_matrix(work)?,
        row_swaps,
        pivots,
    };
    work.tick(1)?;
    Ok(candidate)
}

fn verify(
    matrix: &Matrix,
    candidate: &GaussianLuCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let original = GaussianMatrix::read(matrix, work)?;
    let (rows, columns) = (original.rows, original.cols);
    // Bound all untrusted containers before interpreting their cells or
    // indexing a pivot. Matrix construction alone admits larger dimensions.
    if candidate.p.rows != rows
        || candidate.p.cols != rows
        || candidate.l.rows != rows
        || candidate.l.cols != rows
        || candidate.u.rows != rows
        || candidate.u.cols != columns
        || candidate.pivots.len() > rows.min(columns)
        || candidate.row_swaps.len() > candidate.pivots.len()
    {
        return Err(invalid("factor, permutation or pivot shape"));
    }
    work.tick((candidate.pivots.len() + candidate.row_swaps.len()) as u64 + 1)?;
    if candidate.pivots.iter().any(|&column| column >= columns)
        || candidate.pivots.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid("pivot columns are not strictly increasing"));
    }
    let permutation = GaussianMatrix::read(&candidate.p, work)?;
    let lower = GaussianMatrix::read(&candidate.l, work)?;
    let upper = GaussianMatrix::read(&candidate.u, work)?;

    // A scalar row-label permutation replay is independent of the generator's
    // matrix swaps. It proves exactly one unit in every row and column of P,
    // binds the public swap pairs, and prevents a singular P erasing source
    // rows while satisfying the otherwise vacuous P A = L U identity.
    let mut labels = work.vector(rows)?;
    labels.extend(0..rows);
    let mut previous = None;
    for &(first, second) in &candidate.row_swaps {
        work.tick(1)?;
        if first >= candidate.pivots.len()
            || second >= rows
            || first >= second
            || previous.is_some_and(|row| first <= row)
        {
            return Err(invalid("malformed ordered row swaps"));
        }
        labels.swap(first, second);
        previous = Some(first);
    }
    for (row, &label) in labels.iter().enumerate() {
        for column in 0..rows {
            work.tick(1)?;
            let entry = permutation.get(row, column);
            if (column == label && !is_one(entry)) || (column != label && !is_zero(entry)) {
                return Err(invalid("permutation disagrees with its exact row swaps"));
            }
            let entry = lower.get(row, column);
            if (row == column && !is_one(entry)) || (row < column && !is_zero(entry)) {
                return Err(invalid("L is not unit lower triangular"));
            }
        }
    }
    for row in 0..rows {
        let expected = candidate.pivots.get(row).copied();
        let mut first_nonzero = None;
        for column in 0..columns {
            work.tick(1)?;
            if !is_zero(upper.get(row, column)) && first_nonzero.is_none() {
                first_nonzero = Some(column);
            }
        }
        if first_nonzero != expected {
            return Err(invalid(
                "U does not have the complete declared echelon pivots",
            ));
        }
    }

    // Check every original P A = L U coefficient directly. The independently
    // checked row labels select P A without cubic multiplication by a one-hot
    // matrix. The checked lower triangle bounds the convolution. All original
    // and proposed cells have already passed complete domain admission, so an
    // exact zero factor may now skip arithmetic without hiding a held pole.
    // No elimination, quotient helper, expression simplifier or numeric LU
    // participates in this reference identity check.
    for (row, &label) in labels.iter().enumerate() {
        for column in 0..columns {
            work.storage(128)?;
            let mut sum = (BigRational::zero(), BigRational::zero());
            for inner in 0..=row {
                work.tick(1)?;
                let left = lower.get(row, inner);
                let right = upper.get(inner, column);
                if is_zero(left) || is_zero(right) {
                    continue;
                }
                let product = work.multiply(left, right)?;
                sum = work.add(&sum, &product)?;
            }
            let expected = original.get(label, column);
            let limbs = [&sum.0, &sum.1, &expected.0, &expected.1]
                .iter()
                .map(|value| value.numer().limb_count() + value.denom().limb_count())
                .sum::<u64>();
            work.tick(limbs + 1)?;
            if &sum != expected {
                return Err(invalid("P A differs from L U for the original source"));
            }
        }
    }
    work.tick(1)
}

impl Matrix {
    /// Propose bounded rectangular Q(i) LU without granting evidence.
    pub fn gaussian_lu_candidate_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianLuCandidate> {
        propose(self, &mut Work::new(meter))
    }

    /// Independently check original-source reconstruction, exact permutation,
    /// unit triangular L and complete echelon rank of U.
    pub fn verify_gaussian_lu_candidate(
        &self,
        candidate: &GaussianLuCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        verify(self, candidate, &mut Work::new(meter))
    }

    /// Return rectangular LU only after independent reference checks, using
    /// one cumulative ledger through source parsing and accepted publication.
    pub fn metered_gaussian_lu(&self, meter: &mut impl BudgetMeter) -> Result<GaussianLuCandidate> {
        let mut work = Work::new(meter);
        let candidate = propose(self, &mut work)?;
        verify(self, &candidate, &mut work)?;
        work.tick(1)?;
        Ok(candidate)
    }

    /// Exact square Q(i) determinant from an independently checked LU.
    ///
    /// The exact permutation and unit lower-triangular factor establish
    /// det(A) = (-1)^swaps product(diag(U)). A deficient result becomes zero
    /// only after complete source admission and the full reconstruction/rank
    /// check. The empty determinant is one. All phases and scalar publication
    /// retain one cumulative caller ledger and the existing Gaussian limits.
    pub fn metered_gaussian_det(&self, meter: &mut impl BudgetMeter) -> Result<super::Expr> {
        let mut work = Work::new(meter);
        work.tick(1)?;
        self.validate_shape()?;
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let candidate = propose(self, &mut work)?;
        verify(self, &candidate, &mut work)?;
        let upper = GaussianMatrix::read(&candidate.u, &mut work)?;
        work.storage(128)?;
        let mut determinant = if candidate.pivots.len() == self.rows {
            (BigRational::one(), BigRational::zero())
        } else {
            (BigRational::zero(), BigRational::zero())
        };
        if candidate.pivots.len() == self.rows {
            for index in 0..self.rows {
                work.tick(1)?;
                determinant = work.multiply(&determinant, upper.get(index, index))?;
            }
        }
        if candidate.row_swaps.len() % 2 == 1 {
            determinant.0 = work.negate(&determinant.0)?;
            determinant.1 = work.negate(&determinant.1)?;
        }
        // Reuse the bounded exact Gaussian-to-Expr publication grammar.
        let mut scalar = GaussianMatrix::zeros(1, 1, &mut work)?;
        scalar.entries[0] = determinant;
        let expression = scalar
            .into_matrix(&mut work)?
            .data
            .into_iter()
            .next()
            .ok_or_else(|| invalid("determinant scalar publication lost its entry"))?;
        work.tick(1)?;
        Ok(expression)
    }
}
