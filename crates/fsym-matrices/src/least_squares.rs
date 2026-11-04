//! Complete bounded exact-rational least-squares families.
//!
//! A generator constructs a minimum-norm particular X and an orthogonal
//! projector P onto ker(A). Every least-squares solution is X + P W.
//! Acceptance independently checks a complete homogeneous affine witness for
//! the original A, then AP=0, P=P^T, Pk=k on that complete kernel basis,
//! A^T(AX-B)=0 and k^T X=0. The last two identities establish least squares
//! and the unique minimum norm; the projector checks establish completeness.
//! No generated Gram matrix, inverse, or normal-equation solve is trusted.
//!
//! Inputs have at most 32 rows, unknowns and RHS columns, with literal rational
//! coefficients of at most 4096 bits. Intermediate values have an 8192-bit
//! bound. Both phases retain independent cumulative budgets through every
//! delegated affine operation and final publication. This is a composite
//! in-process receipt, not a new portable proof-kernel claim family.

use crate::{
    AffineSolutionCandidate, Matrix, MatrixError, RationalLinearSolution, VerifiedAffineSolution,
    from_rational, numeric_value, verify_affine_solution_candidate,
};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use num_traits::{One, Zero};

const MAX_DIMENSION: usize = 32;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_BITS: u64 = 8192;
const MAX_STEPS: u64 = 200_000_000;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
type Result<T> = std::result::Result<T, MatrixError>;

/// Untrusted proposal. Only the independent verifier creates an accepted family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeastSquaresCandidate {
    pub minimum_norm: Matrix,
    pub projector: Matrix,
    pub kernel: AffineSolutionCandidate,
}

/// Complete family X + P W, tied to the exact immutable A and B checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLeastSquaresFamily {
    coefficients: Matrix,
    rhs: Matrix,
    minimum_norm: Matrix,
    projector: Matrix,
    kernel: VerifiedAffineSolution,
}

impl VerifiedLeastSquaresFamily {
    pub fn coefficients(&self) -> &Matrix {
        &self.coefficients
    }
    pub fn rhs(&self) -> &Matrix {
        &self.rhs
    }
    pub fn minimum_norm(&self) -> &Matrix {
        &self.minimum_norm
    }
    pub fn projector(&self) -> &Matrix {
        &self.projector
    }
    pub fn rank(&self) -> usize {
        self.kernel.rank()
    }
    pub fn kernel(&self) -> &VerifiedAffineSolution {
        &self.kernel
    }
    pub fn into_parts(self) -> (Matrix, Matrix) {
        (self.minimum_norm, self.projector)
    }
}

fn limit(reason: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("exact least squares: {reason}"))
}
fn invalid(reason: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("exact least squares: {reason}"))
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
    fn charge_storage(&mut self, count: usize, bytes: usize) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, count.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(stopped)
    }
    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>> {
        self.charge_storage(1, count.saturating_mul(std::mem::size_of::<T>()))?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| limit("container allocation"))?;
        Ok(result)
    }
    fn copy_number(&mut self, value: &BigRational) -> Result<BigRational> {
        number_ok(value, MAX_BITS)?;
        self.charge_storage(
            1,
            64 + 8 * (value.numer().limb_count() + value.denom().limb_count()) as usize,
        )?;
        Ok(value.clone())
    }
    fn equal(&mut self, a: &BigRational, b: &BigRational) -> Result<bool> {
        number_ok(a, MAX_BITS)?;
        number_ok(b, MAX_BITS)?;
        self.checkpoint().map_err(stopped)?;
        Ok(a.metered_cmp(b, self).map_err(arithmetic)?.is_eq())
    }
    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.combine(a, b, false)
    }
    fn subtract(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.combine(a, b, true)
    }
    fn combine(&mut self, a: &BigRational, b: &BigRational, subtract: bool) -> Result<BigRational> {
        let numerator = (a.numer().bits() + b.denom().bits())
            .max(b.numer().bits() + a.denom().bits())
            .saturating_add(1);
        let denominator = a.denom().bits() + b.denom().bits();
        if numerator > MAX_BITS || denominator > MAX_BITS {
            return Err(limit("additive intermediate coefficient height"));
        }
        self.checkpoint().map_err(stopped)?;
        let value = if subtract {
            a.metered_sub(b, self)
        } else {
            a.metered_add(b, self)
        }
        .map_err(arithmetic)?;
        number_ok(&value, MAX_BITS)?;
        Ok(value)
    }
    fn multiply(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        if a.numer().bits() + b.numer().bits() > MAX_BITS
            || a.denom().bits() + b.denom().bits() > MAX_BITS
        {
            return Err(limit("multiplicative intermediate coefficient height"));
        }
        self.checkpoint().map_err(stopped)?;
        let value = a.metered_mul(b, self).map_err(arithmetic)?;
        number_ok(&value, MAX_BITS)?;
        Ok(value)
    }
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

fn admit(matrix: &Matrix, maximum: u64, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
    work.checkpoint().map_err(stopped)?;
    matrix.validate_shape()?;
    if matrix.rows > MAX_DIMENSION || matrix.cols > MAX_DIMENSION {
        return Err(limit(
            "at most 32 rows, unknowns and right-hand sides are supported",
        ));
    }
    for entry in &matrix.data {
        work.charge(Dimension::ComputeSteps, 1).map_err(stopped)?;
        match entry {
            Expr::Integer(value) if value.bits() <= maximum => {}
            Expr::Integer(_) => return Err(limit("integer coefficient height")),
            Expr::Rational(value) => number_ok(value, maximum)?,
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        }
    }
    work.checkpoint().map_err(stopped)?;
    Ok(())
}

struct Dense {
    rows: usize,
    cols: usize,
    values: Vec<BigRational>,
}
impl Dense {
    fn zeros(rows: usize, cols: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let mut values = work.vector(rows * cols)?;
        work.charge_storage(rows * cols, rows * cols * 64)?;
        values.resize_with(rows * cols, BigRational::zero);
        Ok(Self { rows, cols, values })
    }
    fn from_matrix(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let mut values = work.vector(matrix.data.len())?;
        for entry in &matrix.data {
            let limbs = match entry {
                Expr::Integer(value) => value.limb_count() + 1,
                Expr::Rational(value) => value.numer().limb_count() + value.denom().limb_count(),
                _ => return Err(MatrixError::UnsupportedCertificateDomain),
            };
            work.charge_storage(1, 64 + 8 * limbs as usize)?;
            values.push(numeric_value(entry).ok_or(MatrixError::UnsupportedCertificateDomain)?);
        }
        Ok(Self {
            rows: matrix.rows,
            cols: matrix.cols,
            values,
        })
    }
    fn get(&self, row: usize, column: usize) -> &BigRational {
        &self.values[row * self.cols + column]
    }
    fn into_matrix(self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
        let mut entries = work.vector(self.values.len())?;
        for value in self.values {
            // Integer lifting may copy the owned numerator; cover that copy.
            work.charge_storage(1, 64 + 8 * value.numer().limb_count() as usize)?;
            entries.push(from_rational(value));
        }
        Matrix::new(self.rows, self.cols, entries)
    }
    fn column(&self, column: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
        let mut entries = work.vector(self.rows)?;
        for row in 0..self.rows {
            entries.push(from_rational(work.copy_number(self.get(row, column))?));
        }
        Matrix::new(self.rows, 1, entries)
    }
}

// Generator products accumulate scaled rows. The reference lane below uses
// direct dot products instead of calling these multiplication helpers.
fn product(a: &Dense, b: &Dense, work: &mut Work<'_, impl BudgetMeter>) -> Result<Dense> {
    if a.cols != b.rows {
        return Err(invalid("generator product shape"));
    }
    let mut result = Dense::zeros(a.rows, b.cols, work)?;
    for row in 0..a.rows {
        for inner in 0..a.cols {
            for column in 0..b.cols {
                let term = work.multiply(a.get(row, inner), b.get(inner, column))?;
                let index = row * b.cols + column;
                result.values[index] = work.add(&result.values[index], &term)?;
            }
        }
    }
    Ok(result)
}
fn transpose(a: &Dense, work: &mut Work<'_, impl BudgetMeter>) -> Result<Dense> {
    let mut values = work.vector(a.values.len())?;
    for column in 0..a.cols {
        for row in 0..a.rows {
            values.push(work.copy_number(a.get(row, column))?);
        }
    }
    Ok(Dense {
        rows: a.cols,
        cols: a.rows,
        values,
    })
}
fn basis(solution: &RationalLinearSolution) -> Result<&[Vec<BigRational>]> {
    match solution {
        RationalLinearSolution::Unique { .. } => Ok(&[]),
        RationalLinearSolution::Affine { basis, .. } => Ok(basis),
        RationalLinearSolution::Inconsistent { .. } => {
            Err(invalid("homogeneous system is inconsistent"))
        }
    }
}
fn particular(solution: RationalLinearSolution) -> Result<Vec<BigRational>> {
    match solution {
        RationalLinearSolution::Unique { values } => Ok(values),
        RationalLinearSolution::Affine { particular, .. } => Ok(particular),
        RationalLinearSolution::Inconsistent { .. } => {
            Err(invalid("generated normal system is inconsistent"))
        }
    }
}
fn generate(
    a: &Matrix,
    b: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<LeastSquaresCandidate> {
    admit(a, MAX_INPUT_BITS, work)?;
    admit(b, MAX_INPUT_BITS, work)?;
    if a.rows != b.rows {
        return Err(MatrixError::ShapeMismatch(a.rows, a.cols, b.rows, b.cols));
    }
    let original = Dense::from_matrix(a, work)?;
    let rhs = Dense::from_matrix(b, work)?;
    let zero = Dense::zeros(a.rows, 1, work)?.into_matrix(work)?;
    let kernel = a.affine_solution_candidate_with_meter(&zero, work)?;
    let directions = basis(&kernel.witness.solution)?;
    let nullity = directions.len();
    let mut k = Dense::zeros(a.cols, nullity, work)?;
    for (column, vector) in directions.iter().enumerate() {
        for (row, entry) in vector.iter().enumerate() {
            k.values[row * nullity + column] = work.copy_number(entry)?;
        }
    }
    let kt = transpose(&k, work)?;
    let gram = product(&kt, &k, work)?.into_matrix(work)?;
    let mut inverse = Dense::zeros(nullity, nullity, work)?;
    for column in 0..nullity {
        let mut unit = Dense::zeros(nullity, 1, work)?;
        unit.values[column] = work.copy_number(&BigRational::one())?;
        let proposal = gram.affine_solution_candidate_with_meter(&unit.into_matrix(work)?, work)?;
        let RationalLinearSolution::Unique { values } = proposal.witness.solution else {
            return Err(invalid("kernel Gram matrix is not invertible"));
        };
        for (row, entry) in values.into_iter().enumerate() {
            inverse.values[row * nullity + column] = entry;
        }
    }
    let projector = product(&product(&k, &inverse, work)?, &kt, work)?;
    let at = transpose(&original, work)?;
    let normal = product(&at, &original, work)?.into_matrix(work)?;
    let normal_rhs = product(&at, &rhs, work)?;
    let mut first = Dense::zeros(a.cols, b.cols, work)?;
    for column in 0..b.cols {
        let proposal =
            normal.affine_solution_candidate_with_meter(&normal_rhs.column(column, work)?, work)?;
        for (row, entry) in particular(proposal.witness.solution)?
            .into_iter()
            .enumerate()
        {
            first.values[row * b.cols + column] = entry;
        }
    }
    let projected = product(&projector, &first, work)?;
    for (entry, component) in first.values.iter_mut().zip(&projected.values) {
        *entry = work.subtract(entry, component)?;
    }
    let result = LeastSquaresCandidate {
        minimum_norm: first.into_matrix(work)?,
        projector: projector.into_matrix(work)?,
        kernel,
    };
    work.checkpoint().map_err(stopped)?;
    Ok(result)
}

fn dot<'a>(
    pairs: impl Iterator<Item = (&'a BigRational, &'a BigRational)>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let mut value = BigRational::zero();
    for (left, right) in pairs {
        let term = work.multiply(left, right)?;
        value = work.add(&value, &term)?;
    }
    Ok(value)
}
fn copy_matrix(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
    work.charge_storage(
        matrix.data.len(),
        matrix.data.len() * std::mem::size_of::<Expr>(),
    )?;
    for entry in &matrix.data {
        let limbs = match entry {
            Expr::Integer(value) => value.limb_count(),
            Expr::Rational(value) => value.numer().limb_count() + value.denom().limb_count(),
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        };
        work.charge_storage(1, 64 + 8 * limbs as usize)?;
    }
    Ok(matrix.clone())
}
fn verify(
    a: &Matrix,
    b: &Matrix,
    candidate: LeastSquaresCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedLeastSquaresFamily> {
    admit(a, MAX_INPUT_BITS, work)?;
    admit(b, MAX_INPUT_BITS, work)?;
    admit(&candidate.minimum_norm, MAX_BITS, work)?;
    admit(&candidate.projector, MAX_BITS, work)?;
    if a.rows != b.rows
        || candidate.minimum_norm.rows != a.cols
        || candidate.minimum_norm.cols != b.cols
        || candidate.projector.rows != a.cols
        || candidate.projector.cols != a.cols
    {
        return Err(invalid("source or output shapes"));
    }
    let source = &candidate.kernel.system;
    if source.rows != a.rows
        || source.columns != a.cols
        || source.coefficients.len() != a.data.len()
        || source.rhs.len() != a.rows
    {
        return Err(invalid("kernel witness is bound to a different shape"));
    }
    let a_values = Dense::from_matrix(a, work)?;
    let b_values = Dense::from_matrix(b, work)?;
    for (original, claimed) in a_values.values.iter().zip(&source.coefficients) {
        if !work.equal(original, claimed)? {
            return Err(invalid("kernel witness is bound to different coefficients"));
        }
    }
    for value in &source.rhs {
        number_ok(value, MAX_BITS)?;
        work.charge(Dimension::ComputeSteps, 1).map_err(stopped)?;
        if !value.is_zero() {
            return Err(invalid("kernel witness is not homogeneous"));
        }
    }
    let kernel = verify_affine_solution_candidate(candidate.kernel, work)?;
    let directions = basis(kernel.solution())?;
    let x = Dense::from_matrix(&candidate.minimum_norm, work)?;
    let p = Dense::from_matrix(&candidate.projector, work)?;
    for row in 0..p.rows {
        for column in 0..row {
            if !work.equal(p.get(row, column), p.get(column, row))? {
                return Err(invalid("kernel projector is not symmetric"));
            }
        }
    }
    for row in 0..a.rows {
        for column in 0..a.cols {
            let value = dot(
                (0..a.cols).map(|inner| (a_values.get(row, inner), p.get(inner, column))),
                work,
            )?;
            if !value.is_zero() {
                return Err(invalid("projector range is not in the original kernel"));
            }
        }
    }
    for vector in directions {
        for (row, expected) in vector.iter().enumerate() {
            let value = dot(
                vector
                    .iter()
                    .enumerate()
                    .map(|(inner, entry)| (p.get(row, inner), entry)),
                work,
            )?;
            if !work.equal(&value, expected)? {
                return Err(invalid("projector omits a complete kernel direction"));
            }
        }
        for column in 0..b.cols {
            let value = dot(
                vector
                    .iter()
                    .enumerate()
                    .map(|(row, entry)| (entry, x.get(row, column))),
                work,
            )?;
            if !value.is_zero() {
                return Err(invalid("particular solution is not minimum norm"));
            }
        }
    }
    let mut residual = Dense::zeros(a.rows, b.cols, work)?;
    for column in 0..b.cols {
        for row in 0..a.rows {
            let value = dot(
                (0..a.cols).map(|inner| (a_values.get(row, inner), x.get(inner, column))),
                work,
            )?;
            residual.values[row * b.cols + column] =
                work.subtract(&value, b_values.get(row, column))?;
        }
    }
    for column in 0..b.cols {
        for unknown in 0..a.cols {
            let normal_residual = dot(
                (0..a.rows).map(|row| (a_values.get(row, unknown), residual.get(row, column))),
                work,
            )?;
            if !normal_residual.is_zero() {
                return Err(invalid(
                    "particular solution does not minimize the residual",
                ));
            }
        }
    }
    let result = VerifiedLeastSquaresFamily {
        coefficients: copy_matrix(a, work)?,
        rhs: copy_matrix(b, work)?,
        minimum_norm: candidate.minimum_norm,
        projector: candidate.projector,
        kernel,
    };
    work.checkpoint().map_err(stopped)?;
    Ok(result)
}

/// Verify the complete original least-squares claim without calling its generator.
pub fn verify_least_squares_candidate(
    a: &Matrix,
    b: &Matrix,
    candidate: LeastSquaresCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedLeastSquaresFamily> {
    let mut work = Work::new(verifier);
    let result = verify(a, b, candidate, &mut work);
    if let Some(error) = work.stopped {
        return Err(stopped(error));
    }
    result
}

impl Matrix {
    /// Complete least-squares family, including singular and rectangular inputs.
    pub fn least_squares_family(&self, rhs: &Matrix) -> Result<VerifiedLeastSquaresFamily> {
        self.least_squares_family_with_meters(rhs, &mut Unbounded, &mut Unbounded)
    }
    pub fn least_squares_candidate_with_meter(
        &self,
        rhs: &Matrix,
        meter: &mut impl BudgetMeter,
    ) -> Result<LeastSquaresCandidate> {
        let mut work = Work::new(meter);
        let result = generate(self, rhs, &mut work);
        if let Some(error) = work.stopped {
            return Err(stopped(error));
        }
        result
    }
    /// Independent cumulative generator/verifier meters, with intrinsic caps.
    pub fn least_squares_family_with_meters(
        &self,
        rhs: &Matrix,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedLeastSquaresFamily> {
        let mut generation = Work::new(generator);
        let mut verification = Work::new(verifier);
        let result = (|| {
            let candidate = generate(self, rhs, &mut generation)?;
            let checked = verify(self, rhs, candidate, &mut verification)?;
            generation.checkpoint().map_err(stopped)?;
            verification.checkpoint().map_err(stopped)?;
            Ok(checked)
        })();
        if let Some(error) = generation.stopped.or(verification.stopped) {
            return Err(stopped(error));
        }
        result
    }
}
