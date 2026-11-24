//! PyO3 bindings exposing `fsym_matrices::Matrix` to Python (WS05, WS10).

#![forbid(unsafe_code)]

mod integer_system;
mod modular_inverse;

use fsym_budget::{BudgetError, BudgetMeter, DIMENSION_COUNT, MeterError};
use fsym_core::{BigInt, BigRational, Constant, Expr};
use fsym_matrices::{
    Matrix, MatrixError, MatrixExponentialError, RationalLinearSolution, SparseMatrix,
    VerifiedHermiteNormalForm, VerifiedHermitianInertia, VerifiedRationalInertia,
    VerifiedSmithNormalForm,
};
use fsym_proof_kernel::integer_hermite::IntegerHermiteWitness;
use fsym_proof_kernel::integer_normal_form::IntegerSmithWitness;
use fsym_proof_kernel::sparse_rref::RationalSparseMatrix;
use fsym_runtime::{Budget, BudgetLimits, Dimension};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyIndexError, PyNotImplementedError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList, PyTuple};

use crate::callbacks::PySupervisedCallbackSession;
use crate::expr::{PyExpr, exact_python_integer};
use crate::rational_laplace::validate_held_transform_domains;
use crate::to_value_error;

/// Particular solution, one nullspace vector per free column, and free columns.
/// These values are exposed only after the independent exact checker succeeds.
type AffineSolutionParts = (Vec<PyExpr>, Vec<Vec<PyExpr>>, Vec<usize>);

/// Complete sparse affine values, parameter matrix, and free columns.
type SparseAffineSolutionParts = (PySparseMatrix, PySparseMatrix, Vec<usize>);

/// Complete dense Gaussian-rational family and its ordered free coordinates.
type GaussianAffineSolutionParts = (PyMatrix, PyMatrix, Vec<usize>);

/// Checked permutation, triangular factors, row swaps, and pivot columns.
type GaussianLuParts = (
    PyMatrix,
    PyMatrix,
    PyMatrix,
    Vec<(usize, usize)>,
    Vec<usize>,
);

/// Both inertia phases consume the same caller-owned symbolic request ledger.
/// Native phase meters still enforce their own memory and allocation caps.
/// Keep the original Python refusal: MeterError only stops native work, and
/// must never relabel an exhausted or ownership-invalid request as cancellation.
struct InertiaRequestMeter<'py, 'session> {
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    failure: Option<PyErr>,
}

impl<'py, 'session> InertiaRequestMeter<'py, 'session> {
    fn new(py: Python<'py>, session: Option<&'session PySupervisedCallbackSession>) -> Self {
        Self {
            py,
            session,
            failure: None,
        }
    }

    fn forward(&mut self, steps: u64) -> Result<(), MeterError> {
        if self.failure.is_some() {
            return Err(MeterError::Cancelled);
        }
        if let Some(session) = self.session
            && let Err(error) = session.charge_nested_steps_atomic(self.py, steps)
        {
            self.failure = Some(error);
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }

    fn refuse(&mut self, error: PyErr) -> Result<(), MeterError> {
        // The first error retains its budget ValueError. Subsequent use sees
        // the cancelled session; no counter reset or failed-charge spending.
        self.failure = Some(match self.session {
            Some(session) => session.cancel(self.py).err().unwrap_or(error),
            None => error,
        });
        Err(MeterError::Cancelled)
    }
}

impl BudgetMeter for InertiaRequestMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.forward(if dimension == Dimension::ComputeSteps {
            amount
        } else {
            0
        })
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let mut steps = 0u64;
        for &(dimension, amount) in charges {
            if dimension == Dimension::ComputeSteps {
                let Some(total) = steps.checked_add(amount) else {
                    return self
                        .refuse(PyValueError::new_err("matrix request step charge overflow"));
                };
                steps = total;
            }
        }
        self.forward(steps)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.forward(0)
    }
}

fn checked_inertia_counts(
    result: Result<VerifiedRationalInertia, MatrixError>,
    generator: InertiaRequestMeter<'_, '_>,
    verifier: InertiaRequestMeter<'_, '_>,
) -> PyResult<(usize, usize, usize)> {
    if let Some(error) = generator.failure.or(verifier.failure) {
        return Err(error);
    }
    let checked = result.map_err(matrix_err)?;
    let receipt = checked.receipt();
    Ok((receipt.positive(), receipt.negative(), receipt.zero()))
}

fn checked_hermitian_inertia_counts(
    result: Result<VerifiedHermitianInertia, MatrixError>,
    mut generator: InertiaRequestMeter<'_, '_>,
    mut verifier: InertiaRequestMeter<'_, '_>,
) -> PyResult<(usize, usize, usize)> {
    let output = (|| {
        let checked = result.map_err(matrix_err)?;
        // Keep publication on the same caller-owned ledger as source
        // interpretation, candidate generation and independent verification.
        // Only these three bounded counts cross the Python boundary.
        generator
            .charge(Dimension::ComputeSteps, 1)
            .map_err(to_value_error)?;
        verifier.checkpoint().map_err(to_value_error)?;
        let receipt = checked.receipt();
        Ok((receipt.positive(), receipt.negative(), receipt.zero()))
    })();
    // The native meter reports cancellation to stop work after any session
    // refusal. Preserve its original budget/cancellation/ownership exception.
    match generator.failure.or(verifier.failure) {
        Some(error) => Err(error),
        None => output,
    }
}

/// Keep two private envelopes through output lifting while both phases debit
/// the same active Python request. Native generation cannot borrow verification
/// capacity, and lifting cannot restart the generation allowance.
struct NormalFormRequestMeter<'py, 'session> {
    request: InertiaRequestMeter<'py, 'session>,
    budget: Budget,
}

impl<'py, 'session> NormalFormRequestMeter<'py, 'session> {
    fn new(py: Python<'py>, session: Option<&'session PySupervisedCallbackSession>) -> Self {
        let mut limits = BudgetLimits::uniform(256 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = 20_000_000;
        limits.dimensions[Dimension::AllocationCount.index()] = 20_000_000;
        Self {
            request: InertiaRequestMeter::new(py, session),
            budget: Budget::new(limits),
        }
    }
}

impl BudgetMeter for NormalFormRequestMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        // Preflight every private dimension before forwarding the request
        // charge. Either refusal leaves both ledgers' counters unchanged.
        // The private ledger is exclusively owned, so this exact batch cannot
        // fail after the session has accepted it.
        let mut totals = [0u64; DIMENSION_COUNT];
        for &(dimension, amount) in charges {
            if amount == 0 {
                return Err(BudgetError::ZeroCharge.into());
            }
            let index = dimension.index();
            totals[index] = totals[index]
                .checked_add(amount)
                .ok_or(BudgetError::ChargeOverflow { dimension })?;
        }
        for dimension in Dimension::ALL {
            let amount = totals[dimension.index()];
            let remaining = self.budget.remaining(dimension);
            if amount > remaining {
                return Err(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }
                .into());
            }
        }
        self.request.charge_batch(charges)?;
        self.budget.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.request.checkpoint()
    }
}

enum IntegerMatrixSource<'a> {
    Dense(&'a Matrix),
    Sparse(&'a SparseMatrix),
}

impl IntegerMatrixSource<'_> {
    fn compute(
        self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedSmithNormalForm, MatrixError> {
        match self {
            Self::Dense(source) => source.metered_smith_normal_form(generator, verifier),
            Self::Sparse(source) => source.metered_smith_normal_form(generator, verifier),
        }
    }

    fn compute_hermite(
        self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedHermiteNormalForm, MatrixError> {
        match self {
            Self::Dense(source) => source.metered_hermite_normal_form(generator, verifier),
            Self::Sparse(source) => source.metered_hermite_normal_form(generator, verifier),
        }
    }
}

fn checked_smith_output<'py, 'session, T>(
    source: IntegerMatrixSource<'_>,
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    lift: impl FnOnce(IntegerSmithWitness, &mut NormalFormRequestMeter<'py, 'session>) -> PyResult<T>,
) -> PyResult<T> {
    let mut generator = NormalFormRequestMeter::new(py, session);
    let mut verifier = NormalFormRequestMeter::new(py, session);
    let result = (|| {
        let accepted = source
            .compute(&mut generator, &mut verifier)
            .map_err(matrix_err)?;
        let (_, witness, _) = accepted.into_parts();
        let output = lift(witness, &mut generator)?;
        generator.checkpoint().map_err(to_value_error)?;
        verifier.checkpoint().map_err(to_value_error)?;
        Ok(output)
    })();
    // Preserve the session's original exhausted/cancelled/ownership exception.
    match generator.request.failure.or(verifier.request.failure) {
        Some(error) => Err(error),
        None => result,
    }
}

fn checked_hermite_output<'py, 'session, T>(
    source: IntegerMatrixSource<'_>,
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
    lift: impl FnOnce(IntegerHermiteWitness, &mut NormalFormRequestMeter<'py, 'session>) -> PyResult<T>,
) -> PyResult<T> {
    let mut generator = NormalFormRequestMeter::new(py, session);
    let mut verifier = NormalFormRequestMeter::new(py, session);
    let result = (|| {
        let accepted = source
            .compute_hermite(&mut generator, &mut verifier)
            .map_err(matrix_err)?;
        let (_, witness, _) = accepted.into_parts();
        let output = lift(witness, &mut generator)?;
        generator.checkpoint().map_err(to_value_error)?;
        verifier.checkpoint().map_err(to_value_error)?;
        Ok(output)
    })();
    match generator.request.failure.or(verifier.request.failure) {
        Some(error) => Err(error),
        None => result,
    }
}

fn charge_integer_matrix_output(
    matrix: &RationalSparseMatrix,
    dense: bool,
    meter: &mut impl BudgetMeter,
) -> PyResult<usize> {
    let cells = matrix
        .rows
        .checked_mul(matrix.columns)
        .ok_or_else(|| PyValueError::new_err("integer normal-form output shape overflow"))?;
    if matrix.rows > 64 || matrix.columns > 64 || matrix.entries.len() > cells {
        return Err(PyValueError::new_err(
            "integer normal-form output exceeds its shape envelope",
        ));
    }
    let count = if dense { cells } else { matrix.entries.len() };
    let mut bytes = count.saturating_mul(std::mem::size_of::<Expr>() + 64);
    let mut steps = count + 1;
    for entry in &matrix.entries {
        if !entry.value.is_integer() || entry.value.numer().bits() > 8192 {
            return Err(PyValueError::new_err(
                "integer normal-form output has invalid coefficients",
            ));
        }
        let limbs = entry.value.numer().limb_count().max(1) as usize;
        bytes += limbs * 8;
        steps += limbs;
    }
    meter.checkpoint().map_err(to_value_error)?;
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, steps as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, matrix.entries.len() as u64 + 1),
        ])
        .map_err(to_value_error)?;
    Ok(cells)
}

fn integer_dense_output(
    matrix: RationalSparseMatrix,
    meter: &mut impl BudgetMeter,
) -> PyResult<PyMatrix> {
    let cells = charge_integer_matrix_output(&matrix, true, meter)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(cells)
        .map_err(|_| PyValueError::new_err("integer normal-form output allocation refused"))?;
    entries.resize_with(cells, || Expr::from_i64(0));
    for entry in matrix.entries {
        entries[entry.row * matrix.columns + entry.column] =
            Expr::Integer(entry.value.numer().clone());
    }
    Ok(PyMatrix {
        inner: Matrix::new(matrix.rows, matrix.columns, entries).map_err(matrix_err)?,
    })
}

fn integer_sparse_output(
    matrix: RationalSparseMatrix,
    meter: &mut impl BudgetMeter,
) -> PyResult<PySparseMatrix> {
    charge_integer_matrix_output(&matrix, false, meter)?;
    let entries = matrix
        .entries
        .into_iter()
        .map(|entry| {
            (
                (entry.row, entry.column),
                Expr::Integer(entry.value.numer().clone()),
            )
        })
        .collect();
    Ok(PySparseMatrix {
        inner: SparseMatrix::new(matrix.rows, matrix.columns, entries).map_err(matrix_err)?,
    })
}

/// Borrow and bound the Python coordinate list before cloning native atoms.
/// The matrix engine independently checks every typed identity and duplicate.
fn affine_coordinates(
    variables: Option<&Bound<'_, PyAny>>,
    unknowns: usize,
) -> PyResult<Option<Vec<Expr>>> {
    let Some(variables) = variables else {
        return Ok(None);
    };
    if !variables.is_exact_instance_of::<PyList>() && !variables.is_exact_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(
            "affine coordinates require an exact list or tuple of native symbols",
        ));
    }
    let count = variables.len()?;
    if count != unknowns || count > 8192 {
        return Err(PyValueError::new_err(
            "affine coordinate count must match the bounded matrix columns",
        ));
    }
    let mut total_bytes = 0usize;
    for value in variables.try_iter()? {
        let value = value?;
        if !value.is_exact_instance_of::<PyExpr>() {
            return Err(PyTypeError::new_err(
                "affine coordinates require exact native symbol expressions",
            ));
        }
        let value = value.extract::<PyRef<'_, PyExpr>>()?;
        let Expr::Sym(symbol) = &value.inner else {
            return Err(PyValueError::new_err(
                "affine coordinates must all be symbol atoms",
            ));
        };
        total_bytes = total_bytes
            .checked_add(symbol.name.len())
            .filter(|&bytes| {
                symbol.name.len() <= fsym_matrices::MAX_SPARSE_AFFINE_PARAMETER_NAME_BYTES
                    && bytes <= fsym_matrices::MAX_SPARSE_AFFINE_PARAMETER_TOTAL_BYTES
            })
            .ok_or_else(|| PyValueError::new_err("affine coordinate names exceed their limit"))?;
    }
    let mut coordinates = Vec::new();
    coordinates
        .try_reserve_exact(count)
        .map_err(|_| PyValueError::new_err("affine coordinates exceed the allocation limit"))?;
    // Exact Python containers and exact native atoms have no callback between
    // these passes. Admit the full original list before copying any symbol.
    for value in variables.try_iter()? {
        let value = value?;
        let value = value.extract::<PyRef<'_, PyExpr>>()?;
        coordinates.push(value.inner.clone());
    }
    Ok(Some(coordinates))
}

fn rational_expr(value: &BigRational) -> PyExpr {
    PyExpr::from_expr(if value.is_integer() {
        Expr::Integer(value.to_integer())
    } else {
        Expr::Rational(value.clone())
    })
}

fn matrix_err(err: MatrixError) -> PyErr {
    match err {
        MatrixError::OutOfBounds(r, c, rows, cols) => PyIndexError::new_err(format!(
            "Index ({r}, {c}) out of bounds for shape {rows}x{cols}"
        )),
        MatrixError::ShapeMismatch(r1, c1, r2, c2) => {
            PyValueError::new_err(format!("Shape mismatch: {r1}x{c1} vs {r2}x{c2}"))
        }
        MatrixError::NotSquare(r, c) => {
            PyValueError::new_err(format!("Matrix must be square: shape is {r}x{c}"))
        }
        MatrixError::SingularMatrix => PyValueError::new_err("Matrix is singular"),
        MatrixError::DivisionByZero => PyValueError::new_err("Exact matrix division by zero"),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn hermitian_source_err(error: MatrixError) -> PyErr {
    match error {
        MatrixError::UnsupportedCertificateDomain => PyNotImplementedError::new_err(
            "Hermitian inertia requires rational/I arithmetic with bounded integer powers",
        ),
        MatrixError::DivisionByZero => {
            PyZeroDivisionError::new_err("Hermitian inertia source contains division by zero")
        }
        error => matrix_err(error),
    }
}

fn matrix_exponential_err(error: MatrixExponentialError) -> PyErr {
    if error.is_unsupported() {
        PyNotImplementedError::new_err(error.to_string())
    } else {
        to_value_error(error)
    }
}

fn nonfinite(expression: &Expr) -> bool {
    matches!(
        expression,
        Expr::Const(
            Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity
                | Constant::NaN
        )
    )
}

/// Inspect the complete borrowed parameter forest before normalization. This
/// guard runs even when the accepted kernel projector will annihilate W.
fn admit_least_squares_parameters(matrix: &Matrix, meter: &mut impl BudgetMeter) -> PyResult<()> {
    meter.checkpoint().map_err(to_value_error)?;
    if matrix.rows() > 32 || matrix.cols() > 32 {
        return Err(PyValueError::new_err(
            "least-squares parameter dimensions exceed 32",
        ));
    }
    admit_least_squares_parameter_roots(matrix.data().iter(), None, meter).map(|_| ())
}

fn least_squares_parameter_meter<'py, 'session>(
    py: Python<'py>,
    session: Option<&'session PySupervisedCallbackSession>,
) -> NormalFormRequestMeter<'py, 'session> {
    let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 200_000_000;
    // Reuse the atomic composition of a private envelope and the supplied
    // request. Node counts alone cannot account for retained-graph equality,
    // symbolic simplification or original held-domain arithmetic.
    NormalFormRequestMeter {
        request: InertiaRequestMeter::new(py, session),
        budget: Budget::new(limits),
    }
}

fn admit_least_squares_parameter_roots<'a>(
    entries: impl ExactSizeIterator<Item = &'a Expr> + Clone,
    retained: Option<&Bound<'_, PyList>>,
    meter: &mut impl BudgetMeter,
) -> PyResult<Vec<usize>> {
    const MAX_NODES: usize = 4096;
    meter.checkpoint().map_err(to_value_error)?;
    if entries.len() > MAX_NODES {
        return Err(PyValueError::new_err(
            "least-squares parameters exceed their bounded expression envelope",
        ));
    }
    meter
        .charge_batch(&[
            (
                Dimension::MemoryBytes,
                (MAX_NODES
                    * (std::mem::size_of::<(&Expr, usize, usize)>()
                        + std::mem::size_of::<&Expr>()
                        + std::mem::size_of::<usize>())) as u64,
            ),
            (Dimension::AllocationCount, 3),
        ])
        .map_err(to_value_error)?;
    let mut pending = Vec::with_capacity(MAX_NODES);
    let mut functions = Vec::with_capacity(MAX_NODES);
    let mut root_sizes = vec![0usize; entries.len()];
    pending.extend(
        entries
            .clone()
            .enumerate()
            .map(|(index, entry)| (entry, 0usize, index)),
    );
    let mut visited = 0usize;
    let mut text_bytes = 0usize;
    while let Some((node, depth, root)) = pending.pop() {
        visited += 1;
        root_sizes[root] += 1;
        if visited > MAX_NODES || depth > 64 {
            return Err(PyValueError::new_err(
                "least-squares parameters exceed their bounded expression envelope",
            ));
        }
        meter.checkpoint().map_err(to_value_error)?;
        meter
            .charge(Dimension::ComputeSteps, 1)
            .map_err(to_value_error)?;
        if nonfinite(node) {
            return Err(PyValueError::new_err(
                "least-squares parameters must be finite scalar expressions",
            ));
        }
        match node {
            Expr::Integer(value) if value.bits() > 4096 => {
                return Err(PyValueError::new_err(
                    "least-squares parameter coefficient exceeds 4096 bits",
                ));
            }
            Expr::Rational(value) if value.numer().bits().max(value.denom().bits()) > 4096 => {
                return Err(PyValueError::new_err(
                    "least-squares parameter coefficient exceeds 4096 bits",
                ));
            }
            Expr::Sym(symbol) => text_bytes = text_bytes.saturating_add(symbol.name.len()),
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = node {
                    text_bytes = text_bytes.saturating_add(name.len());
                    if name == "__fsymFloat" {
                        let bits = match children.as_slice() {
                            [Expr::Integer(value)] => value.to_u64(),
                            _ => None,
                        };
                        if !bits.is_some_and(|bits| f64::from_bits(bits).is_finite()) {
                            return Err(PyValueError::new_err(
                                "least-squares parameters must be finite scalar expressions",
                            ));
                        }
                    }
                    functions.push(node);
                }
                if children.len() > MAX_NODES.saturating_sub(visited + pending.len()) {
                    return Err(PyValueError::new_err(
                        "least-squares parameters exceed their bounded expression envelope",
                    ));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1, root)));
            }
            Expr::Pow(base, exponent) => {
                if 2 > MAX_NODES.saturating_sub(visited + pending.len()) {
                    return Err(PyValueError::new_err(
                        "least-squares parameters exceed their bounded expression envelope",
                    ));
                }
                pending.push((base.as_ref(), depth + 1, root));
                pending.push((exponent.as_ref(), depth + 1, root));
            }
            _ => {}
        }
        if text_bytes > 65_536 {
            return Err(PyValueError::new_err(
                "least-squares parameter names exceed their bounded envelope",
            ));
        }
    }
    // Original retained child identities must agree with these native roots
    // before either algebra or a zero projector can erase a supplied node.
    if let Some(retained) = retained {
        crate::fourier::check_retained(retained, meter)?;
    }
    // Inspect original function nodes separately, before a parent zero could
    // erase a known pole such as held log(0) or log(x-x). Unknown symbolic
    // function values remain formal parameters; this is not a domain solver.
    for function in functions.into_iter().rev() {
        let evaluated = fsym_simplify::simplify_with(function, meter).map_err(to_value_error)?;
        // The algebraic simplifier deliberately retains log(0), rather than
        // choosing an extended-real/complex infinity convention. It can still
        // prove that a held argument such as x - x is zero. That is already
        // enough to reject this parameter's original logarithm domain, before
        // either a parent zero or the nullspace projector erases the node.
        let logarithm_pole = matches!(
            &evaluated,
            Expr::Function(name, arguments)
                if matches!(name.as_str(), "log" | "ln")
                    && matches!(arguments.as_slice(), [argument] if argument.is_zero())
        );
        if nonfinite(&evaluated) || logarithm_pole {
            return Err(PyValueError::new_err(
                "least-squares parameter contains an undefined held function",
            ));
        }
    }
    // Known function poles above are terminal even below a zeroth power whose
    // general symbolic base this checker cannot otherwise resolve. Preserve
    // every raw inverse/zeroth-power child and its polynomial zero obligation.
    for entry in entries {
        validate_held_transform_domains(entry, meter)?;
    }
    meter
        .charge(Dimension::ComputeSteps, 1)
        .map_err(to_value_error)?;
    meter.checkpoint().map_err(to_value_error)?;
    Ok(root_sizes)
}

/// Native matrix type exposing exact algebraic matrix operations.
#[pyclass(name = "Matrix", module = "fsym_python")]
#[derive(Clone)]
pub struct PyMatrix {
    pub(crate) inner: Matrix,
}

#[pymethods]
impl PyMatrix {
    /// Construct a new Matrix with the given dimensions and flat element sequence.
    #[new]
    #[pyo3(signature = (rows, cols, data))]
    pub fn new(rows: usize, cols: usize, data: Vec<PyExpr>) -> PyResult<Self> {
        let raw: Vec<Expr> = data.into_iter().map(|e| e.inner).collect();
        let m = Matrix::new(rows, cols, raw).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Construct an identity matrix of size N x N.
    #[staticmethod]
    pub fn eye(n: usize) -> PyResult<Self> {
        let m = Matrix::eye(n).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Construct a zero matrix of size `rows x cols`.
    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        let m = Matrix::zeros(rows, cols).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Construct a diagonal matrix from the given diagonal elements.
    #[staticmethod]
    pub fn diag(entries: Vec<PyExpr>) -> PyResult<Self> {
        let raw: Vec<Expr> = entries.into_iter().map(|e| e.inner).collect();
        let m = Matrix::diag(raw).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Matrix shape as a `(rows, cols)` tuple.
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.rows(), self.inner.cols())
    }

    /// Number of rows.
    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns.
    #[getter]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Number of elements.
    pub fn __len__(&self) -> usize {
        self.inner.rows() * self.inner.cols()
    }

    /// Whether the matrix is square.
    #[getter]
    pub fn is_square(&self) -> bool {
        self.inner.rows() == self.inner.cols()
    }

    /// Whether every stored entry belongs to the exact rational domain.
    #[getter]
    pub fn is_exact_rational(&self) -> bool {
        self.inner
            .data()
            .iter()
            .all(|entry| matches!(entry, Expr::Integer(_) | Expr::Rational(_)))
    }

    /// Whether the matrix is symmetric.
    #[getter]
    pub fn is_symmetric(&self) -> bool {
        self.inner.is_symmetric()
    }

    /// Whether the matrix is diagonal.
    #[getter]
    pub fn is_diagonal(&self) -> bool {
        self.inner.is_diagonal()
    }

    /// Whether the matrix is upper triangular.
    #[getter]
    pub fn is_upper_triangular(&self) -> bool {
        self.inner.is_upper_triangular()
    }

    /// Whether the matrix is lower triangular.
    #[getter]
    pub fn is_lower_triangular(&self) -> bool {
        self.inner.is_lower_triangular()
    }

    /// Whether the matrix is skew-symmetric (anti-symmetric).
    #[getter]
    pub fn is_skew_symmetric(&self) -> bool {
        self.inner.is_skew_symmetric()
    }

    /// Element indexing: `m[r, c]`.
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        if let Ok(tuple) = key.cast::<PyTuple>()
            && tuple.len() == 2
        {
            let r: usize = tuple.get_item(0)?.extract()?;
            let c: usize = tuple.get_item(1)?.extract()?;
            let entry = self.inner.get(r, c).map_err(matrix_err)?;
            return Ok(PyExpr::from_expr(entry.clone()));
        }
        if let Ok(flat_idx) = key.extract::<usize>() {
            let cols = self.inner.cols();
            if cols == 0 {
                return Err(PyIndexError::new_err("matrix has 0 columns"));
            }
            let r = flat_idx / cols;
            let c = flat_idx % cols;
            let entry = self.inner.get(r, c).map_err(matrix_err)?;
            return Ok(PyExpr::from_expr(entry.clone()));
        }
        Err(PyTypeError::new_err(
            "Matrix indices must be integers or (row, col) integer pairs",
        ))
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    /// Transpose property shorthand `T`.
    #[getter]
    pub fn t(&self) -> Self {
        self.transpose()
    }

    /// Matrix trace: sum of diagonal elements.
    pub fn trace(&self) -> PyResult<PyExpr> {
        let tr = self.inner.trace().map_err(matrix_err)?;
        Ok(PyExpr::from_expr(tr))
    }

    /// Matrix determinant.
    pub fn det(&self) -> PyResult<PyExpr> {
        let d = self.inner.det().map_err(matrix_err)?;
        Ok(PyExpr::from_expr(d))
    }

    /// Exact Q(i) determinant from independently checked LU, with cumulative
    /// source, factor, reference-check and scalar-publication accounting.
    #[pyo3(signature = (*, session=None))]
    pub fn gaussian_det(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<PyExpr> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let determinant = self
                .inner
                .metered_gaussian_det(&mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 1)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok(PyExpr::from_expr(determinant))
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Borrowed rational-source admission before public snapshot copies.
    pub fn validate_rational_inertia_source(&self) -> PyResult<()> {
        self.inner
            .validate_rational_inertia_source()
            .map_err(matrix_err)
    }

    /// Interpret bounded original Gaussian-rational cells before shell copies.
    #[pyo3(signature = (*, session=None))]
    pub fn validate_hermitian_inertia_source(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<()> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .validate_hermitian_inertia_source_with_meter(&mut meter)
            .map_err(hermitian_source_err);
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Bound and interpret original rectangular Q(i) cells before shell copies.
    #[pyo3(signature = (*, session=None))]
    pub fn validate_gaussian_pseudoinverse_source(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<()> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .validate_gaussian_pseudoinverse_source_with_meter(&mut meter)
            .map_err(matrix_err);
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Borrowed integer admission before any public source copying.
    pub fn validate_integer_normal_form_source(&self) -> PyResult<()> {
        self.inner
            .validate_integer_normal_form_source()
            .map_err(matrix_err)
    }

    /// Canonical positive integer Smith form, after independent acceptance.
    #[pyo3(signature = (*, session=None))]
    pub fn smith_normal_form(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        checked_smith_output(
            IntegerMatrixSource::Dense(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| integer_dense_output(witness.diagonal, meter),
        )
    }

    /// Checked `(D, U, V)` with positive invariant factors and `D = U A V`.
    #[pyo3(signature = (*, session=None))]
    pub fn smith_normal_decomposition(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Self, Self)> {
        checked_smith_output(
            IntegerMatrixSource::Dense(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| {
                Ok((
                    integer_dense_output(witness.diagonal, meter)?,
                    integer_dense_output(witness.left, meter)?,
                    integer_dense_output(witness.right, meter)?,
                ))
            },
        )
    }

    /// Canonical basis of the integer column lattice, independently checked.
    #[pyo3(signature = (*, session=None))]
    pub fn hermite_normal_form(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        checked_hermite_output(
            IntegerMatrixSource::Dense(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| integer_dense_output(witness.normal_form, meter),
        )
    }

    /// Checked `(H, V, V_inverse)` with `A V = [0 | H]` over the integers.
    #[pyo3(signature = (*, session=None))]
    pub fn hermite_normal_decomposition(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Self, Self)> {
        checked_hermite_output(
            IntegerMatrixSource::Dense(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| {
                Ok((
                    integer_dense_output(witness.normal_form, meter)?,
                    integer_dense_output(witness.right, meter)?,
                    integer_dense_output(witness.right_inverse, meter)?,
                ))
            },
        )
    }

    /// Checked inverse over Z/mZ, including composite moduli. None is returned
    /// only after a nonunit Smith factor has been independently verified.
    #[pyo3(signature = (modulus, *, session=None))]
    pub fn inverse_mod(
        &self,
        modulus: &Bound<'_, PyAny>,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Option<PyMatrix>> {
        modular_inverse::inverse((&self.inner).into(), modulus, py, session.as_deref())
    }

    /// Complete integer family `(particular, kernel)` or checked inconsistency.
    #[pyo3(signature = (rhs, *, session=None))]
    pub fn solve_integer(
        &self,
        rhs: &Bound<'_, PyAny>,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Option<(Self, Self)>> {
        integer_system::solve_dense(&self.inner, rhs, py, session.as_deref())
    }

    /// Independently checked inertia of the real rational quadratic form.
    #[pyo3(signature = (*, session=None))]
    pub fn quadratic_form_inertia(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(usize, usize, usize)> {
        let mut generator = InertiaRequestMeter::new(py, session.as_deref());
        let mut verifier = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .inertia_with_meters(&mut generator, &mut verifier);
        checked_inertia_counts(result, generator, verifier)
    }

    /// Complete checked signs of Re(z* A z) for exact Gaussian-rational A.
    /// Non-Hermitian input is classified by its Hermitian part, not its spectrum.
    #[pyo3(signature = (*, session=None))]
    pub fn hermitian_form_inertia(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(usize, usize, usize)> {
        let mut generator = InertiaRequestMeter::new(py, session.as_deref());
        let mut verifier = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .hermitian_inertia_with_meters(&mut generator, &mut verifier);
        checked_hermitian_inertia_counts(result, generator, verifier)
    }

    /// Ascending coefficients of the checked monic minimal polynomial over QQ.
    /// The complete call shares two budgets through bounded result publication.
    pub fn minimal_polynomial(&self) -> PyResult<Vec<PyExpr>> {
        let phase_budget = || {
            let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
            limits.dimensions[Dimension::ComputeSteps.index()] = 200_000_000;
            Budget::new(limits)
        };
        let mut generator = phase_budget();
        let mut verifier = phase_budget();
        let coefficients = self
            .inner
            .minimal_polynomial_with_meters(&mut generator, &mut verifier)
            .map_err(matrix_err)?
            .into_coefficients();
        if coefficients.is_empty() || coefficients.len() > 17 {
            return Err(PyValueError::new_err(
                "checked matrix minimal polynomial has an invalid degree",
            ));
        }
        // Native verification bounds both numerator and denominator by 8192
        // bits. This covers all wrappers and integer canonicalization copies.
        crate::ideal_operations::charge(
            &mut generator,
            coefficients.len(),
            coefficients.len() * 8192,
        )?;
        let values = coefficients
            .into_iter()
            .map(|value| PyExpr::from_expr(crate::rational_to_expr(value)))
            .collect();
        crate::ideal_operations::charge(&mut generator, 1, 1)?;
        crate::ideal_operations::charge(&mut verifier, 1, 1)?;
        Ok(values)
    }

    /// Select the bounded rational lane without copying source coefficients.
    #[getter]
    pub fn is_rational_jordan_input(&self) -> bool {
        self.inner.rows() == self.inner.cols()
            && self.inner.rows() <= fsym_matrices::MAX_RATIONAL_JORDAN_DIMENSION
            && self.inner.data().iter().all(|entry| match entry {
                Expr::Integer(integer) => integer.bits() <= 4096,
                Expr::Rational(rational) => {
                    rational.numer().bits() <= 4096 && rational.denom().bits() <= 4096
                }
                _ => false,
            })
    }

    /// Bound the original rational source before any public spectral search.
    pub fn validate_rational_jordan_source(&self) -> PyResult<()> {
        if self.inner.rows() != self.inner.cols() {
            return Err(PyValueError::new_err(
                "Jordan form requires a square matrix",
            ));
        }
        if self.inner.rows() > fsym_matrices::MAX_RATIONAL_JORDAN_DIMENSION {
            return Err(PyValueError::new_err(
                "rational Jordan dimension exceeds 16",
            ));
        }
        for entry in self.inner.data() {
            let (numerator_bits, denominator_bits) = match entry {
                Expr::Integer(integer) => (integer.bits(), 1),
                Expr::Rational(rational) => (rational.numer().bits(), rational.denom().bits()),
                _ => {
                    return Err(PyValueError::new_err(
                        "rational Jordan requires literal rational entries",
                    ));
                }
            };
            if numerator_bits > 4096 || denominator_bits > 4096 {
                return Err(PyValueError::new_err(
                    "Jordan source coefficient exceeds 4096 bits",
                ));
            }
        }
        Ok(())
    }

    /// Complete Jordan decomposition over a supplied rational spectrum.
    /// Hints are proposals: the independent checker establishes A*P=P*J
    /// and an exact inverse of P against the original matrix before return.
    pub fn checked_rational_jordan(&self, spectrum: &Bound<'_, PyAny>) -> PyResult<(Self, Self)> {
        self.validate_rational_jordan_source()?;
        let dimension = self.inner.rows();
        if !spectrum.is_exact_instance_of::<PyList>() && !spectrum.is_exact_instance_of::<PyTuple>()
        {
            return Err(PyTypeError::new_err(
                "Jordan spectrum requires an exact list or tuple",
            ));
        }
        let count = spectrum.len()?;
        if count > dimension {
            return Err(PyValueError::new_err(
                "Jordan spectrum exceeds matrix dimension",
            ));
        }
        // Borrow the entire bounded input before cloning arbitrary-precision
        // values. Exact built-in containers cannot invoke iterator callbacks.
        for item in spectrum.try_iter()? {
            let item = item?;
            if !item.is_exact_instance_of::<PyTuple>() || item.len()? != 2 {
                return Err(PyTypeError::new_err(
                    "Jordan spectrum entries must be pairs",
                ));
            }
            let value = item.get_item(0)?;
            if !value.is_exact_instance_of::<PyExpr>() {
                return Err(PyTypeError::new_err(
                    "Jordan eigenvalues require exact native atoms",
                ));
            }
            let value = value.extract::<PyRef<'_, PyExpr>>()?;
            let (numerator_bits, denominator_bits) = match &value.inner {
                Expr::Integer(integer) => (integer.bits(), 1),
                Expr::Rational(rational) => (rational.numer().bits(), rational.denom().bits()),
                _ => {
                    return Err(PyValueError::new_err(
                        "Jordan eigenvalues must be literal rationals",
                    ));
                }
            };
            if numerator_bits > 4096 || denominator_bits > 4096 {
                return Err(PyValueError::new_err("Jordan eigenvalue exceeds 4096 bits"));
            }
            let multiplicity = item.get_item(1)?;
            if !multiplicity.is_exact_instance_of::<PyInt>() {
                return Err(PyTypeError::new_err(
                    "Jordan multiplicities require exact integers",
                ));
            }
            let multiplicity = multiplicity.extract::<usize>()?;
            if multiplicity == 0 || multiplicity > dimension {
                return Err(PyValueError::new_err(
                    "Jordan multiplicity is outside matrix dimension",
                ));
            }
        }
        let phase_budget = || {
            let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
            limits.dimensions[Dimension::ComputeSteps.index()] = 200_000_000;
            Budget::new(limits)
        };
        let mut generator = phase_budget();
        let mut verifier = phase_budget();
        crate::ideal_operations::charge(&mut generator, count, count * 2048)?;
        let mut hints = Vec::with_capacity(count);
        for item in spectrum.try_iter()? {
            let item = item?;
            let value = item.get_item(0)?;
            let value = value.extract::<PyRef<'_, PyExpr>>()?;
            let value = match &value.inner {
                Expr::Integer(integer) => BigRational::from_integer(integer.clone()),
                Expr::Rational(rational) => rational.clone(),
                _ => unreachable!("all spectrum atoms were admitted before copying"),
            };
            hints.push((value, item.get_item(1)?.extract::<usize>()?));
        }
        let verified = self
            .inner
            .metered_rational_jordan(&hints, &mut generator, &mut verifier)
            .map_err(matrix_err)?;
        // Publication moves the checked matrices without recomputation or
        // arbitrary Python scalar callbacks. Both phase meters remain live.
        crate::ideal_operations::charge(&mut generator, dimension * dimension * 2, 4096)?;
        crate::ideal_operations::charge(&mut verifier, 1, 1)?;
        let (transform, jordan, _inverse) = verified.into_parts();
        Ok((Self { inner: transform }, Self { inner: jordan }))
    }

    /// Independently checked bounded exact rational matrix exponential.
    pub fn exp(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.exp().map_err(matrix_exponential_err)?,
        })
    }

    /// Independently checked integer power over bounded exact Q(i).
    /// Source admission, negative-power inversion, both power passes and
    /// publication retain the same active request and its terminal state.
    #[pyo3(signature = (exponent, *, session=None))]
    pub fn gaussian_power(
        &self,
        py: Python<'_>,
        exponent: i64,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let power = self
                .inner
                .metered_gaussian_power(exponent, &mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 1)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok(Self { inner: power })
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Bound the closed Q(i) syntax before choosing an inverse lane.
    /// False selects the existing symbolic path before any exact algorithm;
    /// admission, budget and cancellation errors are never converted to false.
    #[pyo3(signature = (*, session=None))]
    pub fn has_gaussian_rational_syntax(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<bool> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .has_gaussian_rational_syntax_with_meter(&mut meter)
            .map_err(matrix_err);
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Matrix inverse: A^-1, with both exact products checked over Q(i).
    #[pyo3(signature = (*, session=None))]
    pub fn inv(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let inverse = if session.is_none() {
                self.inner.inverse()
            } else {
                self.inner.metered_gaussian_inverse(&mut meter)
            }
            .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 1)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok(Self { inner: inverse })
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Matrix inverse alias.
    #[pyo3(signature = (*, session=None))]
    pub fn inverse(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        self.inv(py, session)
    }

    /// Moore-Penrose pseudoinverse, with exact conjugate identities over Q(i).
    /// A supplied request keeps source, generation, checking and publication
    /// on one cumulative ledger through the bounded Gaussian-rational lane.
    #[pyo3(signature = (*, session=None))]
    pub fn pinv(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let inverse = if session.is_none() && self.is_exact_rational() {
                self.inner.pseudoinverse()
            } else {
                self.inner.metered_gaussian_pseudoinverse(&mut meter)
            }
            .map_err(matrix_err)?;
            // Move the accepted immutable native matrix into its wrapper;
            // no Python scalar reconstruction or callback is required.
            meter
                .charge(Dimension::ComputeSteps, 1)
                .map_err(to_value_error)?;
            Ok(Self { inner: inverse })
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Adjugate (classical adjoint) matrix.
    pub fn adjugate(&self) -> PyResult<Self> {
        let adj = self.inner.adjugate().map_err(matrix_err)?;
        Ok(Self { inner: adj })
    }

    /// Cofactor C(i, j).
    pub fn cofactor(&self, i: usize, j: usize) -> PyResult<PyExpr> {
        let c = self.inner.cofactor(i, j).map_err(matrix_err)?;
        Ok(PyExpr::from_expr(c))
    }

    /// Squared Frobenius norm.
    pub fn frobenius_norm_squared(&self) -> PyResult<PyExpr> {
        let norm_sq = self.inner.frobenius_norm_squared().map_err(matrix_err)?;
        Ok(PyExpr::from_expr(norm_sq))
    }

    /// Matrix rank.
    pub fn rank(&self) -> PyResult<usize> {
        self.inner.rank().map_err(matrix_err)
    }

    /// Reduced row echelon form (RREF) returning `(rref_matrix, pivot_columns)`.
    pub fn rref(&self) -> PyResult<(Self, Vec<usize>)> {
        let (rref_m, pivots) = self.inner.rref().map_err(matrix_err)?;
        Ok((Self { inner: rref_m }, pivots))
    }

    /// Basis vectors for the nullspace (kernel).
    pub fn nullspace(&self) -> PyResult<Vec<Self>> {
        let bases = self.inner.nullspace().map_err(matrix_err)?;
        Ok(bases.into_iter().map(|m| Self { inner: m }).collect())
    }

    /// Independently checked reduced row echelon form over exact Q(i).
    /// Original source admission, row-equivalence checking and publication
    /// consume the same active request, including terminal refusal states.
    #[pyo3(signature = (*, session=None))]
    pub fn gaussian_rref(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Vec<usize>)> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let (reduced, pivots) = self
                .inner
                .metered_gaussian_rref(&mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 1)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok((Self { inner: reduced }, pivots))
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Complete exact Q(i) kernel basis in increasing free-column order.
    #[pyo3(signature = (*, session=None))]
    pub fn gaussian_nullspace(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Vec<Self>> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let basis = self
                .inner
                .metered_gaussian_nullspace(&mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, basis.len() as u64 + 1)
                .map_err(to_value_error)?;
            let result = basis.into_iter().map(|inner| Self { inner }).collect();
            meter.checkpoint().map_err(to_value_error)?;
            Ok(result)
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Complete Q(i) affine family, or an independently checked inconsistency.
    /// Parameters and all right-hand sides share the original request ledger.
    #[pyo3(signature = (rhs, *, session=None))]
    pub fn gaussian_gauss_jordan_solve(
        &self,
        rhs: &Self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Option<GaussianAffineSolutionParts>> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let family = self
                .inner
                .metered_gaussian_gauss_jordan_solve(&rhs.inner, &mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 2)
                .map_err(to_value_error)?;
            let result = family.map(|(solution, parameters, free_columns)| {
                (
                    Self { inner: solution },
                    Self { inner: parameters },
                    free_columns,
                )
            });
            meter.checkpoint().map_err(to_value_error)?;
            Ok(result)
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Exact eigenvalues of the matrix.
    pub fn eigenvalues(&self) -> PyResult<Vec<PyExpr>> {
        let evals = self.inner.eigenvalues().map_err(matrix_err)?;
        Ok(evals.into_iter().map(PyExpr::from_expr).collect())
    }

    /// Characteristic polynomial coefficients in descending degree order.
    pub fn char_poly(&self) -> PyResult<Vec<PyExpr>> {
        let coeffs = self.inner.char_poly().map_err(matrix_err)?;
        Ok(coeffs.into_iter().map(PyExpr::from_expr).collect())
    }

    /// Reference-checked rational characteristic coefficients, descending.
    ///
    /// The existing independent checker compares against exact shifted
    /// determinants at n + 1 distinct points. Original entries and their
    /// aggregate coefficient height are admitted before generator arithmetic.
    pub fn checked_char_poly(&self) -> PyResult<Vec<PyExpr>> {
        let dimension = self.inner.rows();
        if dimension != self.inner.cols() {
            return Err(matrix_err(MatrixError::NotSquare(
                dimension,
                self.inner.cols(),
            )));
        }
        // Keep the Python transport within the native characteristic-polynomial
        // envelope before scanning coefficient payloads or constructing outputs.
        if dimension > 32 {
            return Err(matrix_err(MatrixError::ResourceLimit(
                "checked characteristic polynomial supports dimensions up to 32".into(),
            )));
        }
        let mut input_bits = 0u64;
        for entry in self.inner.data() {
            let bits = match entry {
                Expr::Integer(value) => value.bits().max(1),
                Expr::Rational(value) => value
                    .numer()
                    .bits()
                    .max(1)
                    .saturating_add(value.denom().bits()),
                _ => return Err(matrix_err(MatrixError::UnsupportedCertificateDomain)),
            };
            input_bits = input_bits.saturating_add(bits);
            if input_bits > 65_536 {
                return Err(matrix_err(MatrixError::ResourceLimit(
                    "exact characteristic-polynomial input exceeds 65536 coefficient bits".into(),
                )));
            }
        }
        let certificate = self
            .inner
            .char_poly_with_certificate("_fsym_eigenvalue")
            .map_err(matrix_err)?;
        Ok(certificate
            .poly
            .coeffs
            .iter()
            .rev()
            .map(rational_expr)
            .collect())
    }

    /// Kronecker product.
    pub fn kron(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.kron(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Exact LU decomposition with partial row pivoting returning `(P, L, U)`.
    pub fn lu(&self) -> PyResult<(Self, Self, Self)> {
        let cert = self.inner.lu().map_err(matrix_err)?;
        Ok((
            Self { inner: cert.p },
            Self { inner: cert.l },
            Self { inner: cert.u },
        ))
    }

    /// Rectangular exact Q(i) LU with checked row swaps and complete pivots.
    /// Admission, generation, verification and publication share one request.
    #[pyo3(signature = (*, session=None))]
    pub fn gaussian_lu(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<GaussianLuParts> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let factors = self
                .inner
                .metered_gaussian_lu(&mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(
                    Dimension::ComputeSteps,
                    (factors.row_swaps.len() + factors.pivots.len() + 3) as u64,
                )
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok((
                Self { inner: factors.p },
                Self { inner: factors.l },
                Self { inner: factors.u },
                factors.row_swaps,
                factors.pivots,
            ))
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Exact QR decomposition with orthogonal columns returning `(Q, R)`.
    pub fn qr(&self) -> PyResult<(Self, Self)> {
        let cert = self.inner.qr().map_err(matrix_err)?;
        Ok((Self { inner: cert.q }, Self { inner: cert.r }))
    }

    /// Checked exact thin Q(i) QR, optionally normalized to orthonormal columns.
    /// The original source, reference pass and output share the active request.
    #[pyo3(signature = (*, normalize=true, session=None))]
    pub fn gaussian_qr(
        &self,
        py: Python<'_>,
        normalize: bool,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Self)> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let (q, r) = if normalize {
                self.inner
                    .metered_gaussian_qr_normalized(&mut meter)
                    .map_err(matrix_err)?
            } else {
                let candidate = self
                    .inner
                    .metered_gaussian_qr(&mut meter)
                    .map_err(matrix_err)?;
                (candidate.q, candidate.r)
            };
            meter
                .charge(Dimension::ComputeSteps, 2)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok((Self { inner: q }, Self { inner: r }))
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Legacy exact-rational LDL^T, optionally requiring a positive diagonal.
    /// The optional check inspects the independently accepted exact D.
    /// Active requests use gaussian_ldl instead of this standalone legacy lane.
    #[pyo3(signature = (*, positive=false))]
    pub fn ldl(&self, positive: bool) -> PyResult<(Self, Self)> {
        let cert = self.inner.ldl().map_err(matrix_err)?;
        if positive {
            let zero = BigInt::from(0);
            for index in 0..cert.d.rows() {
                let entry = cert.d.get(index, index).map_err(matrix_err)?;
                let is_positive = match entry {
                    Expr::Integer(value) => value > &zero,
                    Expr::Rational(value) => value.numer() > &zero && value.denom() > &zero,
                    _ => return Err(matrix_err(MatrixError::UnsupportedCertificateDomain)),
                };
                if !is_positive {
                    return Err(matrix_err(MatrixError::NotPositiveDefinite));
                }
            }
        }
        Ok((Self { inner: cert.l }, Self { inner: cert.d }))
    }

    /// Checked positive-definite Hermitian Q(i) LDL with one cumulative request.
    #[pyo3(signature = (*, session=None))]
    pub fn gaussian_ldl(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Self)> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result: PyResult<(Self, Self)> = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let factors = self
                .inner
                .metered_gaussian_ldl(&mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 3)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok((Self { inner: factors.l }, Self { inner: factors.d }))
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Checked positive-diagonal Hermitian Cholesky over exact Q(i).
    /// Original source, LDL, positive roots and all output components are checked.
    #[pyo3(signature = (*, session=None))]
    pub fn gaussian_cholesky(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result: PyResult<Self> = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let factor = self
                .inner
                .metered_gaussian_cholesky(&mut meter)
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 2)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok(Self { inner: factor })
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Minor submatrix obtained by removing row `rem_r` and column `rem_c`.
    pub fn minor_submatrix(&self, rem_r: usize, rem_c: usize) -> PyResult<Self> {
        let m = self.inner.minor_matrix(rem_r, rem_c).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Solves `self * X = B` exactly and uniquely, including bounded Q(i).
    /// A complex least-squares projection is never an exact solution.
    #[pyo3(signature = (b, *, session=None))]
    pub fn solve(
        &self,
        py: Python<'_>,
        b: &Self,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let solution = if session.is_none() {
                self.inner.solve(&b.inner)
            } else {
                self.inner.metered_gaussian_solve(&b.inner, &mut meter)
            }
            .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 1)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok(Self { inner: solution })
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Complete exact-rational solution family for rectangular or singular A.
    /// `None` means inconsistency was independently checked. Otherwise the
    /// result is `(particular, nullspace_basis, free_columns)`, with one basis
    /// vector for each free column. Unsupported domains and exhausted budgets
    /// are explicit errors; neither becomes an empty solution set.
    pub fn solve_affine(&self, b: &Self) -> PyResult<Option<AffineSolutionParts>> {
        let checked = self.inner.solve_affine(&b.inner).map_err(matrix_err)?;
        Ok(match checked.solution() {
            RationalLinearSolution::Unique { values } => {
                Some((values.iter().map(rational_expr).collect(), vec![], vec![]))
            }
            RationalLinearSolution::Affine {
                particular,
                basis,
                free_columns,
            } => Some((
                particular.iter().map(rational_expr).collect(),
                basis
                    .iter()
                    .map(|vector| vector.iter().map(rational_expr).collect())
                    .collect(),
                free_columns.clone(),
            )),
            RationalLinearSolution::Inconsistent { .. } => None,
        })
    }

    /// Solves the exact Hermitian least-squares system `A* A X = A* B` for `X`.
    #[pyo3(signature = (b, *, session=None))]
    pub fn solve_least_squares(
        &self,
        py: Python<'_>,
        b: &Self,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Self> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let solution =
                if session.is_none() && self.is_exact_rational() && b.is_exact_rational() {
                    self.inner.solve_least_squares(&b.inner)
                } else {
                    self.inner
                        .metered_gaussian_solve_least_squares(&b.inner, &mut meter)
                }
                .map_err(matrix_err)?;
            meter
                .charge(Dimension::ComputeSteps, 1)
                .map_err(to_value_error)?;
            Ok(Self { inner: solution })
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Checked minimum-norm particular and complete orthogonal kernel projector.
    /// All least-squares solutions are `particular + projector * arbitrary`.
    #[pyo3(signature = (b, *, session=None))]
    pub fn least_squares_family(
        &self,
        py: Python<'_>,
        b: &Self,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Self)> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            let (minimum_norm, projector) =
                if session.is_none() && self.is_exact_rational() && b.is_exact_rational() {
                    self.inner
                        .least_squares_family(&b.inner)
                        .map_err(matrix_err)?
                        .into_parts()
                } else {
                    self.inner
                        .metered_gaussian_least_squares_family(&b.inner, &mut meter)
                        .map_err(matrix_err)?
                        .into_parts()
                };
            // Source admission, generation, independent Hermitian checking
            // and both result wrappers share the active request's ledger.
            meter
                .charge(Dimension::ComputeSteps, 2)
                .map_err(to_value_error)?;
            meter.checkpoint().map_err(to_value_error)?;
            Ok((
                Self {
                    inner: minimum_norm,
                },
                Self { inner: projector },
            ))
        })();
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Bound and inspect original supplied parameters before any projection.
    #[pyo3(signature = (*, session=None))]
    pub fn validate_least_squares_parameters(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<()> {
        let mut meter = least_squares_parameter_meter(py, session.as_deref());
        let result = admit_least_squares_parameters(&self.inner, &mut meter);
        match meter.request.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Borrow original dense W before exporting cells or converting to sparse.
    /// The sparse family admits large thin dense parameter matrices within the
    /// same stored-forest bound, independently of the dense solver's 32 cap.
    #[pyo3(signature = (retained=None, *, session=None))]
    pub fn validate_sparse_least_squares_parameters(
        &self,
        py: Python<'_>,
        retained: Option<&Bound<'_, PyList>>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Vec<usize>> {
        let mut meter = least_squares_parameter_meter(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            if self.inner.rows() > 8192 || self.inner.cols() > 8192 {
                return Err(PyValueError::new_err(
                    "sparse least-squares parameter dimensions exceed 8192",
                ));
            }
            admit_least_squares_parameter_roots(self.inner.data().iter(), retained, &mut meter)
        })();
        match meter.request.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => {
                if let Ok(other_mat) = other.extract::<PyRef<Self>>() {
                    Ok(self.inner == other_mat.inner)
                } else {
                    Ok(false)
                }
            }
            CompareOp::Ne => {
                if let Ok(other_mat) = other.extract::<PyRef<Self>>() {
                    Ok(self.inner != other_mat.inner)
                } else {
                    Ok(true)
                }
            }
            _ => Err(PyTypeError::new_err(
                "Ordering comparisons not supported for Matrix",
            )),
        }
    }

    /// Matrix addition.
    pub fn __add__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.add(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Matrix subtraction.
    pub fn __sub__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.sub(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Matrix negation: -self.
    pub fn __neg__(&self) -> PyResult<Self> {
        let neg_one = Expr::from_i64(-1);
        let res = self.inner.scalar_mul(&neg_one).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Matrix multiplication: self @ other.
    pub fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.matmul(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Scalar or matrix multiplication: self * other.
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_mat) = other.extract::<PyRef<Self>>() {
            let res = self.inner.matmul(&other_mat.inner).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(scalar_expr) = other.extract::<PyRef<PyExpr>>() {
            let res = self
                .inner
                .scalar_mul(&scalar_expr.inner)
                .map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(int_val) = other.extract::<i64>() {
            let scalar = Expr::from_i64(int_val);
            let res = self.inner.scalar_mul(&scalar).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(bigint) = exact_python_integer(other, "scalar") {
            let scalar = Expr::Integer(bigint);
            let res = self.inner.scalar_mul(&scalar).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        Err(PyTypeError::new_err(
            "Multiplication unsupported between Matrix and the given operand",
        ))
    }

    /// Right scalar multiplication: other * self.
    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    /// Scalar division: self / other.
    pub fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar_recip = if let Ok(scalar_expr) = other.extract::<PyRef<PyExpr>>() {
            Expr::pow(scalar_expr.inner.clone(), Expr::from_i64(-1))
        } else if let Ok(int_val) = other.extract::<i64>() {
            if int_val == 0 {
                return Err(PyZeroDivisionError::new_err("division by zero"));
            }
            Expr::rational(1, int_val).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(bigint) = exact_python_integer(other, "divisor") {
            if bigint == BigInt::from(0) {
                return Err(PyZeroDivisionError::new_err("division by zero"));
            }
            Expr::Rational(BigRational::new(BigInt::from(1), bigint))
        } else {
            return Err(PyTypeError::new_err(
                "Division unsupported between Matrix and the given operand",
            ));
        };
        let res = self.inner.scalar_mul(&scalar_recip).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Matrix integer power: self ** n.
    pub fn __pow__(&self, n: usize, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let res = self.inner.pow(n).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Hadamard (elementwise) product.
    pub fn hadamard(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.hadamard(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Returns a flat list of all entries.
    pub fn flat(&self) -> Vec<PyExpr> {
        self.inner
            .data()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    /// Returns the matrix as nested rows: `List[List[Expr]]`.
    pub fn to_list(&self) -> Vec<Vec<PyExpr>> {
        let rows = self.inner.rows();
        let cols = self.inner.cols();
        let data = self.inner.data();
        let mut result = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(PyExpr::from_expr(data[r * cols + c].clone()));
            }
            result.push(row);
        }
        result
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Matrix({}x{}, {:?})",
            self.inner.rows(),
            self.inner.cols(),
            self.inner.data()
        )
    }

    pub fn __str__(&self) -> String {
        let rows = self.inner.rows();
        let cols = self.inner.cols();
        let data = self.inner.data();
        let mut lines = Vec::with_capacity(rows);
        for r in 0..rows {
            let row_strs: Vec<String> = (0..cols)
                .map(|c| format!("{}", data[r * cols + c]))
                .collect();
            lines.push(format!("[{}]", row_strs.join(", ")));
        }
        format!("Matrix([\n  {}\n])", lines.join(",\n  "))
    }

    pub fn _repr_latex_(&self) -> PyResult<String> {
        let rows = self.inner.rows();
        let cols = self.inner.cols();
        let data = self.inner.data();
        let mut row_strs = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut entries = Vec::with_capacity(cols);
            for c in 0..cols {
                let s = fsym_printing::latex(&data[r * cols + c])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                entries.push(s);
            }
            row_strs.push(entries.join(" & "));
        }
        Ok(format!(
            "\\left[\\begin{{matrix}}{}\\end{{matrix}}\\right]",
            row_strs.join(" \\\\ ")
        ))
    }

    /// Convert to a sparse matrix representation.
    pub fn to_sparse(&self) -> PyResult<PySparseMatrix> {
        let sparse = self.inner.to_sparse().map_err(matrix_err)?;
        Ok(PySparseMatrix { inner: sparse })
    }

    /// Exact Jacobian matrix: J[i, j] = d(exprs[i]) / d(vars[j]).
    #[staticmethod]
    pub fn jacobian(exprs: Vec<PyExpr>, vars: Vec<String>) -> PyResult<Self> {
        let raw_exprs: Vec<Expr> = exprs.into_iter().map(|e| e.inner).collect();
        let symbols: Vec<fsym_core::Symbol> =
            vars.into_iter().map(fsym_core::Symbol::new).collect();
        let m = Matrix::jacobian(&raw_exprs, &symbols).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }
}

/// Native sparse matrix type exposing exact algebraic sparse matrix operations.
#[pyclass(name = "SparseMatrix", module = "fsym_python")]
#[derive(Clone)]
pub struct PySparseMatrix {
    pub(crate) inner: SparseMatrix,
}

#[pymethods]
impl PySparseMatrix {
    /// Construct a new SparseMatrix with the given dimensions and optional coordinate-value map.
    #[new]
    #[pyo3(signature = (rows, cols, entries=None))]
    pub fn new(
        rows: usize,
        cols: usize,
        entries: Option<std::collections::BTreeMap<(usize, usize), PyExpr>>,
    ) -> PyResult<Self> {
        let raw_entries = entries
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, v.inner))
            .collect();
        let sm = SparseMatrix::new(rows, cols, raw_entries).map_err(matrix_err)?;
        Ok(Self { inner: sm })
    }

    /// Construct a sparse identity matrix of size N x N.
    #[staticmethod]
    pub fn eye(n: usize) -> PyResult<Self> {
        let sm = SparseMatrix::eye(n).map_err(matrix_err)?;
        Ok(Self { inner: sm })
    }

    /// Construct a sparse zero matrix of size `rows x cols`.
    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            inner: SparseMatrix::zeros(rows, cols),
        }
    }

    /// Matrix shape as a `(rows, cols)` tuple.
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.rows(), self.inner.cols())
    }

    /// Number of rows.
    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns.
    #[getter]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Number of stored non-zero entries.
    #[getter]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Whether every stored entry belongs to the exact rational lane.
    #[getter]
    pub fn is_exact_rational(&self) -> bool {
        self.inner
            .entries()
            .values()
            .all(|entry| matches!(entry, Expr::Integer(_) | Expr::Rational(_)))
    }

    /// Stored entries only; implicit zeros never require a dense traversal.
    pub fn todok(&self) -> std::collections::BTreeMap<(usize, usize), PyExpr> {
        self.inner
            .entries()
            .iter()
            .map(|(&coordinate, value)| (coordinate, PyExpr::from_expr(value.clone())))
            .collect()
    }

    /// Inspect stored original syntax without densifying symbolic sparse inputs.
    #[pyo3(signature = (*, session=None))]
    pub fn has_gaussian_rational_syntax(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<bool> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .has_gaussian_rational_syntax_with_meter(&mut meter)
            .map_err(matrix_err);
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Admit bounded original rectangular sparse Q(i) cells before dense copying.
    #[pyo3(signature = (*, session=None))]
    pub fn validate_gaussian_pseudoinverse_source(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<()> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .validate_gaussian_pseudoinverse_source_with_meter(&mut meter)
            .map_err(matrix_err);
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Number of elements in full shape.
    pub fn __len__(&self) -> usize {
        self.inner.rows() * self.inner.cols()
    }

    /// Element access: `m[r, c]`.
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        if let Ok(tuple) = key.cast::<PyTuple>()
            && tuple.len() == 2
        {
            let r: usize = tuple.get_item(0)?.extract()?;
            let c: usize = tuple.get_item(1)?.extract()?;
            let entry = self.inner.get(r, c).map_err(matrix_err)?;
            return Ok(PyExpr::from_expr(entry));
        }
        if let Ok(flat_idx) = key.extract::<usize>() {
            let cols = self.inner.cols();
            if cols == 0 {
                return Err(PyIndexError::new_err("sparse matrix has 0 columns"));
            }
            let r = flat_idx / cols;
            let c = flat_idx % cols;
            let entry = self.inner.get(r, c).map_err(matrix_err)?;
            return Ok(PyExpr::from_expr(entry));
        }
        Err(PyTypeError::new_err(
            "SparseMatrix indices must be integers or (row, col) integer pairs",
        ))
    }

    /// Convert to dense Matrix.
    pub fn to_dense(&self) -> PyResult<PyMatrix> {
        let dense = self.inner.to_dense().map_err(matrix_err)?;
        Ok(PyMatrix { inner: dense })
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> PyResult<Self> {
        let tr = self.inner.transpose().map_err(matrix_err)?;
        Ok(Self { inner: tr })
    }

    /// Trace of square sparse matrix.
    pub fn trace(&self) -> PyResult<PyExpr> {
        let tr = self.inner.trace().map_err(matrix_err)?;
        Ok(PyExpr::from_expr(tr))
    }

    /// Checked rational exponential with bounded sparse result fill.
    pub fn exp(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.exp().map_err(matrix_exponential_err)?,
        })
    }

    /// Exact rational determinant without allocating the dense matrix.
    pub fn det(&self) -> PyResult<PyExpr> {
        self.inner.det().map(PyExpr::from_expr).map_err(matrix_err)
    }

    /// Borrowed sparse admission before any dense inertia storage.
    pub fn validate_rational_inertia_source(&self) -> PyResult<()> {
        self.inner
            .validate_rational_inertia_source()
            .map_err(matrix_err)
    }

    /// Interpret bounded original sparse Gaussian-rational cells before copying.
    #[pyo3(signature = (*, session=None))]
    pub fn validate_hermitian_inertia_source(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<()> {
        let mut meter = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .validate_hermitian_inertia_source_with_meter(&mut meter)
            .map_err(hermitian_source_err);
        match meter.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Borrowed integer admission before any public sparse source copying.
    pub fn validate_integer_normal_form_source(&self) -> PyResult<()> {
        self.inner
            .validate_integer_normal_form_source()
            .map_err(matrix_err)
    }

    /// Canonical positive integer Smith form as a dense matrix.
    #[pyo3(signature = (*, session=None))]
    pub fn smith_normal_form(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<PyMatrix> {
        checked_smith_output(
            IntegerMatrixSource::Sparse(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| integer_dense_output(witness.diagonal, meter),
        )
    }

    /// Checked sparse `(D, U, V)`, retaining bounded sparse output storage.
    #[pyo3(signature = (*, session=None))]
    pub fn smith_normal_decomposition(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Self, Self)> {
        checked_smith_output(
            IntegerMatrixSource::Sparse(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| {
                Ok((
                    integer_sparse_output(witness.diagonal, meter)?,
                    integer_sparse_output(witness.left, meter)?,
                    integer_sparse_output(witness.right, meter)?,
                ))
            },
        )
    }

    /// Canonical integer column-lattice basis as a mutable dense matrix.
    #[pyo3(signature = (*, session=None))]
    pub fn hermite_normal_form(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<PyMatrix> {
        checked_hermite_output(
            IntegerMatrixSource::Sparse(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| integer_dense_output(witness.normal_form, meter),
        )
    }

    /// Checked sparse `(H, V, V_inverse)` with `A V = [0 | H]`.
    #[pyo3(signature = (*, session=None))]
    pub fn hermite_normal_decomposition(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(Self, Self, Self)> {
        checked_hermite_output(
            IntegerMatrixSource::Sparse(&self.inner),
            py,
            session.as_deref(),
            |witness, meter| {
                Ok((
                    integer_sparse_output(witness.normal_form, meter)?,
                    integer_sparse_output(witness.right, meter)?,
                    integer_sparse_output(witness.right_inverse, meter)?,
                ))
            },
        )
    }

    /// Checked inverse over Z/mZ, including composite moduli. None is returned
    /// only after a nonunit Smith factor has been independently verified.
    #[pyo3(signature = (modulus, *, session=None))]
    pub fn inverse_mod(
        &self,
        modulus: &Bound<'_, PyAny>,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Option<PyMatrix>> {
        modular_inverse::inverse((&self.inner).into(), modulus, py, session.as_deref())
    }

    /// Complete integer family with dense or sparse RHS, preserving sparse output.
    #[pyo3(signature = (rhs, *, session=None))]
    pub fn solve_integer(
        &self,
        rhs: &Bound<'_, PyAny>,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Option<(Self, Self)>> {
        integer_system::solve_sparse(&self.inner, rhs, py, session.as_deref())
    }

    /// Independently checked inertia without public dense conversion.
    #[pyo3(signature = (*, session=None))]
    pub fn quadratic_form_inertia(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(usize, usize, usize)> {
        let mut generator = InertiaRequestMeter::new(py, session.as_deref());
        let mut verifier = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .inertia_with_meters(&mut generator, &mut verifier);
        checked_inertia_counts(result, generator, verifier)
    }

    /// Complete checked Hermitian-part inertia, retaining original sparse cells.
    #[pyo3(signature = (*, session=None))]
    pub fn hermitian_form_inertia(
        &self,
        py: Python<'_>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<(usize, usize, usize)> {
        let mut generator = InertiaRequestMeter::new(py, session.as_deref());
        let mut verifier = InertiaRequestMeter::new(py, session.as_deref());
        let result = self
            .inner
            .hermitian_inertia_with_meters(&mut generator, &mut verifier);
        checked_hermitian_inertia_counts(result, generator, verifier)
    }

    /// Unique exact rational solution, retaining sparse inputs and outputs.
    pub fn solve(&self, rhs: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.solve(&rhs.inner).map_err(matrix_err)?,
        })
    }

    /// Unique exact rational least squares with bounded sparse intermediates.
    pub fn solve_least_squares(&self, rhs: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .solve_least_squares(&rhs.inner)
                .map_err(matrix_err)?,
        })
    }

    /// Exact rational Moore-Penrose inverse, independently checked in the kernel.
    pub fn pinv(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.pseudoinverse().map_err(matrix_err)?,
        })
    }

    /// Unique minimum-norm least-squares solution, including deficient rank.
    pub fn minimum_norm_least_squares(&self, rhs: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .minimum_norm_least_squares(&rhs.inner)
                .map_err(matrix_err)?,
        })
    }

    /// Independently checked minimum-norm particular and complete kernel projector.
    pub fn least_squares_family(&self, rhs: &Self) -> PyResult<(Self, Self)> {
        let (minimum_norm, projector) = self
            .inner
            .least_squares_family(&rhs.inner)
            .map_err(matrix_err)?
            .into_parts();
        Ok((
            Self {
                inner: minimum_norm,
            },
            Self { inner: projector },
        ))
    }

    /// Admit original stored parameter roots without visiting implicit zeros.
    #[pyo3(signature = (retained=None, *, session=None))]
    pub fn validate_least_squares_parameters(
        &self,
        py: Python<'_>,
        retained: Option<&Bound<'_, PyList>>,
        session: Option<PyRef<'_, PySupervisedCallbackSession>>,
    ) -> PyResult<Vec<usize>> {
        let mut meter = least_squares_parameter_meter(py, session.as_deref());
        let result = (|| {
            meter.checkpoint().map_err(to_value_error)?;
            if self.inner.rows() > 8192 || self.inner.cols() > 8192 {
                return Err(PyValueError::new_err(
                    "sparse least-squares parameter dimensions exceed 8192",
                ));
            }
            admit_least_squares_parameter_roots(self.inner.entries().values(), retained, &mut meter)
        })();
        match meter.request.failure {
            Some(error) => Err(error),
            None => result,
        }
    }

    /// Complete checked affine family, rendered under the same native budget.
    pub fn gauss_jordan_solve(&self, rhs: &Self) -> PyResult<Option<SparseAffineSolutionParts>> {
        self.inner
            .gauss_jordan_solve(&rhs.inner)
            .map(|family| {
                family.map(|(solution, parameters, free_columns)| {
                    (
                        Self { inner: solution },
                        Self { inner: parameters },
                        free_columns,
                    )
                })
            })
            .map_err(matrix_err)
    }

    /// Render one checked affine family with caller-owned typed coordinates.
    #[pyo3(signature = (rhs, variables=None))]
    pub fn solve_affine_parameters(
        &self,
        rhs: &Self,
        variables: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<SparseAffineSolutionParts>> {
        let coordinates = affine_coordinates(variables, self.inner.cols())?;
        self.inner
            .solve_affine_parameters(&rhs.inner, coordinates.as_deref())
            .map(|family| {
                family.map(|(solution, parameters, free_columns)| {
                    (
                        Self { inner: solution },
                        Self { inner: parameters },
                        free_columns,
                    )
                })
            })
            .map_err(matrix_err)
    }

    /// Split an augmented source sparsely within the same native work ledger.
    #[pyo3(signature = (variables=None))]
    pub fn solve_augmented_affine_parameters(
        &self,
        variables: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<SparseAffineSolutionParts>> {
        let unknowns = self.inner.cols().checked_sub(1).ok_or_else(|| {
            PyValueError::new_err("an augmented linear system requires a right-hand-side column")
        })?;
        let coordinates = affine_coordinates(variables, unknowns)?;
        self.inner
            .solve_augmented_affine_parameters(coordinates.as_deref())
            .map(|family| {
                family.map(|(solution, parameters, free_columns)| {
                    (
                        Self { inner: solution },
                        Self { inner: parameters },
                        free_columns,
                    )
                })
            })
            .map_err(matrix_err)
    }

    /// Independently checked rational RREF, retaining sparse native storage.
    pub fn rref(&self) -> PyResult<(Self, Vec<usize>)> {
        let (reduced, pivots) = self.inner.rref().map_err(matrix_err)?;
        Ok((Self { inner: reduced }, pivots))
    }

    /// Exact rank without allocating the full matrix area.
    pub fn rank(&self) -> PyResult<usize> {
        self.inner.rank().map_err(matrix_err)
    }

    /// Complete rational nullspace, with one sparse vector per free column.
    pub fn nullspace(&self) -> PyResult<Vec<Self>> {
        Ok(self
            .inner
            .nullspace()
            .map_err(matrix_err)?
            .into_iter()
            .map(|inner| Self { inner })
            .collect())
    }

    /// Original pivot columns selected by independently checked reduction.
    pub fn columnspace(&self) -> PyResult<Vec<Self>> {
        Ok(self
            .inner
            .columnspace()
            .map_err(matrix_err)?
            .into_iter()
            .map(|inner| Self { inner })
            .collect())
    }

    /// Sparse matrix addition.
    pub fn __add__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.add(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Sparse matrix subtraction.
    pub fn __sub__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.sub(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Sparse matrix multiplication.
    pub fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.matmul(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Scalar or matrix multiplication.
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_mat) = other.extract::<PyRef<Self>>() {
            let res = self.inner.matmul(&other_mat.inner).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(scalar_expr) = other.extract::<PyRef<PyExpr>>() {
            let res = self
                .inner
                .scalar_mul(&scalar_expr.inner)
                .map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(int_val) = other.extract::<i64>() {
            let scalar = Expr::from_i64(int_val);
            let res = self.inner.scalar_mul(&scalar).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(bigint) = exact_python_integer(other, "scalar") {
            let scalar = Expr::Integer(bigint);
            let res = self.inner.scalar_mul(&scalar).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        Err(PyTypeError::new_err(
            "Multiplication unsupported between SparseMatrix and the given operand",
        ))
    }

    /// Right scalar multiplication.
    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => {
                if let Ok(other_mat) = other.extract::<PyRef<Self>>() {
                    Ok(self.inner == other_mat.inner)
                } else {
                    Ok(false)
                }
            }
            CompareOp::Ne => {
                if let Ok(other_mat) = other.extract::<PyRef<Self>>() {
                    Ok(self.inner != other_mat.inner)
                } else {
                    Ok(true)
                }
            }
            _ => Err(PyTypeError::new_err(
                "Ordering comparisons not supported for SparseMatrix",
            )),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SparseMatrix({}x{}, nnz={})",
            self.inner.rows(),
            self.inner.cols(),
            self.inner.nnz()
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}
