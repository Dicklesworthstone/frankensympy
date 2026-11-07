//! Typed Python access to independently checked sparse second derivatives.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::PyExpr;
use crate::numeric::{
    ball_bounds, check_certified_sources, check_source, donor_float_option, donor_newton_guards,
    exact_box, exact_rational, exact_vector, extract_variables, input_values_metered, rational_expr,
};
use crate::numeric_budget::{
    PyEvaluationBudget, evaluation_buffer, with_certification_meter, with_evaluation_meter,
    with_evaluation_meter_attached,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::VerifiedObjective;
use fsym_calculus::box_optimization::{BoxOptimizationOptions, propose_box_optimization};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxExtremumReceipt, BoxOptimizationReceipt, Expr,
    MAX_BOX_OPTIMIZATION_LEAVES, ObjectiveEnclosure, RootBoxWitness, StationaryPointKind,
    StationaryPointReceipt, Symbol, metered_enclose_objective, verify_box_extremum,
    verify_box_optimization, verify_stationary_point,
};
use fsym_numeric_consumer::{
    MAX_MINIMIZE_VARIABLES, MinimizationProposal, MinimizeOptions, minimize_with_meter,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

fn exact_preconditioner(
    values: &Bound<'_, PyAny>,
    dimension: usize,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Vec<Vec<BigRational>>> {
    let py = values.py();
    if !values.is_exact_instance(&py.get_type::<PyList>())
        && !values.is_exact_instance(&py.get_type::<PyTuple>())
    {
        return Err(PyTypeError::new_err(
            "preconditioner must be an exact built-in list or tuple",
        ));
    }
    if dimension > 16 || values.len()? != dimension {
        return Err(PyValueError::new_err(
            "preconditioner has the wrong dimension",
        ));
    }
    let mut matrix =
        with_evaluation_meter_attached(budget, |meter| evaluation_buffer(dimension, meter))?;
    for index in 0..dimension {
        matrix.push(exact_vector(&values.get_item(index)?, dimension, budget)?);
    }
    Ok(matrix)
}

fn box_extremum_kind(kind: &str) -> PyResult<BoxExtremumKind> {
    match kind {
        "minimum" => Ok(BoxExtremumKind::Minimum),
        "maximum" => Ok(BoxExtremumKind::Maximum),
        _ => Err(PyValueError::new_err("kind must be 'minimum' or 'maximum'")),
    }
}

/// Both source snapshots and compiled objectives use the same witness parsing,
/// original-source verification and final budget checkpoint.
fn certify_source_box_extremum(
    source: &Expr,
    variables: &[Symbol],
    domain: &Bound<'_, PyAny>,
    point: &Bound<'_, PyAny>,
    kind: &str,
    precision_digits: u32,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<PyBoxExtremumReceipt> {
    let kind = box_extremum_kind(kind)?;
    let py = domain.py();
    let domain = exact_box(domain, variables.len(), budget)?;
    let point = exact_vector(point, variables.len(), budget)?;
    let receipt = with_certification_meter(py, budget, |meter| {
        verify_box_extremum(
            source,
            variables,
            &domain,
            &point,
            kind,
            precision_digits,
            meter,
        )
    })?;
    let result = PyBoxExtremumReceipt { receipt };
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}

fn optimize_source_box(
    source: &Expr,
    variables: &[Symbol],
    domain: &Bound<'_, PyAny>,
    tolerance: &Bound<'_, PyAny>,
    kind: &str,
    options: BoxOptimizationOptions,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<PyBoxOptimizationReceipt> {
    let kind = box_extremum_kind(kind)?;
    if !(8..=128).contains(&options.precision_digits) {
        return Err(PyValueError::new_err("precision must be 8 through 128"));
    }
    if !(1..=MAX_BOX_OPTIMIZATION_LEAVES).contains(&options.max_leaves) {
        return Err(PyValueError::new_err("max_leaves must be 1 through 1024"));
    }
    let py = domain.py();
    let domain = exact_box(domain, variables.len(), budget)?;
    let tolerance = exact_rational(tolerance, budget)?;
    let receipt = with_certification_meter(py, budget, |meter| {
        let proposed =
            propose_box_optimization(source, variables, &domain, kind, &tolerance, options, meter)?;
        // One ledger spans search and independent complete-cover replay. The
        // candidate supplies no trusted values, derivatives or pruning claims.
        verify_box_optimization(
            source, variables, &domain, kind, &tolerance, &proposed, meter,
        )
    })?;
    let result = PyBoxOptimizationReceipt { receipt };
    with_evaluation_meter_attached(budget, |meter| meter.checkpoint())?;
    Ok(result)
}

/// An exact source snapshot for independent second-order interval checks.
/// Capturing a source is not a derivative or smoothness certificate.
#[pyclass(name = "SourceObjective", module = "fsym_python", frozen)]
pub struct PySourceObjective {
    source: Expr,
    variables: Vec<Symbol>,
}

#[pymethods]
impl PySourceObjective {
    #[new]
    fn new(source: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !source.is_exact_instance(&source.py().get_type::<PyExpr>()) {
            return Err(PyTypeError::new_err(
                "objective must be an exact native Expr object",
            ));
        }
        let source = source.extract::<PyRef<'_, PyExpr>>()?;
        let variables = extract_variables(variables)?;
        check_certified_sources(std::slice::from_ref(&source.inner), &variables)?;
        Ok(Self {
            source: source.inner.clone(),
            variables,
        })
    }

    #[getter]
    fn source(&self) -> PyExpr {
        PyExpr::from_expr(self.source.clone())
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.variables.len()
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.variables
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[pyo3(signature = (domain, precision_digits=32, *, budget=None))]
    fn enclose(
        &self,
        domain: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyObjectiveEnclosure> {
        let py = domain.py();
        let domain = exact_box(domain, self.variables.len(), budget.as_deref())?;
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            metered_enclose_objective(
                &self.source,
                &self.variables,
                &domain,
                precision_digits,
                meter,
            )
        })?;
        let result = PyObjectiveEnclosure { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    #[pyo3(signature = (domain, center, preconditioner, precision_digits=32, *, budget=None))]
    fn certify_stationary_point(
        &self,
        domain: &Bound<'_, PyAny>,
        center: &Bound<'_, PyAny>,
        preconditioner: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyStationaryPointReceipt> {
        let dimension = self.variables.len();
        let py = domain.py();
        let witness = RootBoxWitness {
            domain: exact_box(domain, dimension, budget.as_deref())?,
            center: exact_vector(center, dimension, budget.as_deref())?,
            preconditioner: exact_preconditioner(preconditioner, dimension, budget.as_deref())?,
            precision_digits,
        };
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            verify_stationary_point(&self.source, &self.variables, &witness, meter)
        })?;
        let result = PyStationaryPointReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    #[pyo3(signature = (domain, point, kind="minimum", precision_digits=32, *, budget=None))]
    fn certify_box_extremum(
        &self,
        domain: &Bound<'_, PyAny>,
        point: &Bound<'_, PyAny>,
        kind: &str,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyBoxExtremumReceipt> {
        certify_source_box_extremum(
            &self.source,
            &self.variables,
            domain,
            point,
            kind,
            precision_digits,
            budget.as_deref(),
        )
    }

    #[pyo3(signature = (domain, tolerance, kind="minimum", precision_digits=32, max_leaves=512, *, budget=None))]
    fn optimize_box(
        &self,
        domain: &Bound<'_, PyAny>,
        tolerance: &Bound<'_, PyAny>,
        kind: &str,
        precision_digits: u32,
        max_leaves: usize,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyBoxOptimizationReceipt> {
        optimize_source_box(
            &self.source,
            &self.variables,
            domain,
            tolerance,
            kind,
            BoxOptimizationOptions {
                precision_digits,
                max_leaves,
            },
            budget.as_deref(),
        )
    }
}

/// Checked symbolic gradient and Hessian with finite binary64 evaluation.
#[pyclass(name = "VerifiedObjective", module = "fsym_python", frozen)]
pub struct PyVerifiedObjective {
    objective: VerifiedObjective,
}

#[pymethods]
impl PyVerifiedObjective {
    #[new]
    fn new(source: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !source.is_exact_instance(&source.py().get_type::<PyExpr>()) {
            return Err(PyTypeError::new_err(
                "objective must be an exact native Expr object",
            ));
        }
        let source = source.extract::<PyRef<'_, PyExpr>>()?;
        let variables = extract_variables(variables)?;
        let mut nodes = 0;
        check_source(&source.inner, &mut nodes)?;
        let objective = VerifiedObjective::try_compile(&source.inner, &variables)
            .map_err(crate::to_value_error)?;
        Ok(Self { objective })
    }

    /// The immutable original typed objective, before symbolic differentiation.
    #[getter]
    fn source(&self) -> PyExpr {
        PyExpr::from_expr(self.objective.objective().clone())
    }

    /// Independently enclose the original source and both derivative orders.
    #[pyo3(signature = (domain, precision_digits=32, *, budget=None))]
    fn enclose(
        &self,
        domain: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyObjectiveEnclosure> {
        let py = domain.py();
        let domain = exact_box(domain, self.objective.variables().len(), budget.as_deref())?;
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            metered_enclose_objective(
                self.objective.objective(),
                self.objective.variables(),
                &domain,
                precision_digits,
                meter,
            )
        })?;
        let result = PyObjectiveEnclosure { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Check an untrusted stationary-point witness against the original source.
    #[pyo3(signature = (domain, center, preconditioner, precision_digits=32, *, budget=None))]
    fn certify_stationary_point(
        &self,
        domain: &Bound<'_, PyAny>,
        center: &Bound<'_, PyAny>,
        preconditioner: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyStationaryPointReceipt> {
        let dimension = self.objective.variables().len();
        let py = domain.py();
        let domain = exact_box(domain, dimension, budget.as_deref())?;
        let center = exact_vector(center, dimension, budget.as_deref())?;
        let preconditioner = exact_preconditioner(preconditioner, dimension, budget.as_deref())?;
        let witness = RootBoxWitness {
            domain,
            center,
            preconditioner,
            precision_digits,
        };
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            verify_stationary_point(
                self.objective.objective(),
                self.objective.variables(),
                &witness,
                meter,
            )
        })?;
        let result = PyStationaryPointReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Check global convex/concave optimality of an exact point on a closed box.
    #[pyo3(signature = (domain, point, kind="minimum", precision_digits=32, *, budget=None))]
    fn certify_box_extremum(
        &self,
        domain: &Bound<'_, PyAny>,
        point: &Bound<'_, PyAny>,
        kind: &str,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyBoxExtremumReceipt> {
        certify_source_box_extremum(
            self.objective.objective(),
            self.objective.variables(),
            domain,
            point,
            kind,
            precision_digits,
            budget.as_deref(),
        )
    }

    /// Search for a feasible point and independently certify its global
    /// objective-value error on the complete declared box.
    #[pyo3(signature = (domain, tolerance, kind="minimum", precision_digits=32, max_leaves=512, *, budget=None))]
    fn optimize_box(
        &self,
        domain: &Bound<'_, PyAny>,
        tolerance: &Bound<'_, PyAny>,
        kind: &str,
        precision_digits: u32,
        max_leaves: usize,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyBoxOptimizationReceipt> {
        optimize_source_box(
            self.objective.objective(),
            self.objective.variables(),
            domain,
            tolerance,
            kind,
            BoxOptimizationOptions {
                precision_digits,
                max_leaves,
            },
            budget.as_deref(),
        )
    }

    /// Propose a finite stationary point using bounded damped Newton descent.
    /// This is approximate evidence, including at a stationary maximum or saddle.
    #[pyo3(signature = (initial, *, gtol=None, maxiter=None, max_step=None, budget=None))]
    fn minimize(
        slf: &Bound<'_, Self>,
        initial: &Bound<'_, PyAny>,
        gtol: Option<&Bound<'_, PyAny>>,
        maxiter: Option<&Bound<'_, PyAny>>,
        max_step: Option<&Bound<'_, PyAny>>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Py<PyNumericMinimizationProposal>> {
        // Admit every control before any initial-coordinate conversion or work.
        let guards = donor_newton_guards(None, maxiter, max_step)?;
        let gradient_tolerance = donor_float_option(
            gtol,
            MinimizeOptions::default().gradient_tolerance,
            "gtol",
            true,
        )?;
        let objective = slf.borrow();
        let dimension = objective.objective.variables().len();
        if !(1..=MAX_MINIMIZE_VARIABLES).contains(&dimension) {
            return Err(PyValueError::new_err("minimization requires 1..16 variables"));
        }
        let inputs = input_values_metered(initial, dimension, budget.as_deref())?;
        let options = MinimizeOptions {
            max_iterations: guards.max_iterations,
            gradient_tolerance,
            max_step_norm: guards.max_step_norm,
        };
        // Capture only the immutable Rust program across GIL detachment; the
        // Python borrow guard and owner remain on this attached stack.
        let program = &objective.objective;
        let py = slf.py();
        let proposal = with_evaluation_meter(py, budget.as_deref(), |meter| {
            minimize_with_meter(program, &inputs, &options, meter)
        })?;
        let result = Py::new(
            py,
            PyNumericMinimizationProposal {
                proposal,
                objective: slf.clone().unbind(),
            },
        )?;
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Return value, gradient and ordered sparse Hessian values atomically.
    #[pyo3(signature = (values, *, budget=None))]
    fn evaluate(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
        let inputs =
            input_values_metered(values, self.objective.variables().len(), budget.as_deref())?;
        let result = with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            self.objective.evaluate_with_meter(&inputs, meter)
        })?;
        Ok((result.value, result.gradient, result.hessian))
    }

    /// Multiply by the Hessian without allocating its dense representation.
    #[pyo3(signature = (values, direction, *, budget=None))]
    fn hessian_vector_product(
        &self,
        values: &Bound<'_, PyAny>,
        direction: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<f64>> {
        let size = self.objective.variables().len();
        let inputs = input_values_metered(values, size, budget.as_deref())?;
        let direction = input_values_metered(direction, size, budget.as_deref())?;
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            self.objective
                .hessian_vector_product_with_meter(&inputs, &direction, meter)
        })
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.objective.variables().len()
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.objective
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn hessian_pattern(&self) -> Vec<(usize, usize)> {
        self.objective.hessian_pattern().to_vec()
    }

    #[getter]
    fn hessian_nonzeros(&self) -> usize {
        self.objective.receipt().hessian_nonzeros
    }

    /// Formal derivative proofs; this does not certify numerical rounding.
    #[getter]
    fn proofs_verified(&self) -> usize {
        let receipt = self.objective.receipt();
        receipt.gradient_proofs_verified + receipt.hessian_proofs_verified
    }

    #[getter]
    fn zero_entries_checked(&self) -> usize {
        self.objective.receipt().zero_entries_checked
    }
}

/// An immutable approximate proposal tied to its exact compiled source owner.
/// There is no public constructor and no minimum or rounding-error certificate.
#[pyclass(name = "NumericMinimizationProposal", module = "fsym_python", frozen)]
pub struct PyNumericMinimizationProposal {
    proposal: MinimizationProposal,
    objective: Py<PyVerifiedObjective>,
}

#[pymethods]
impl PyNumericMinimizationProposal {
    #[getter]
    fn objective(&self, py: Python<'_>) -> Py<PyVerifiedObjective> {
        self.objective.clone_ref(py)
    }

    #[getter]
    fn point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.proposal.point.iter().copied())
    }

    #[getter]
    fn value(&self) -> f64 {
        self.proposal.value
    }

    #[getter]
    fn gradient<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.proposal.gradient.iter().copied())
    }

    #[getter]
    fn gradient_norm(&self) -> f64 {
        self.proposal.gradient_norm
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.proposal.iterations
    }

    #[getter]
    fn evaluations(&self) -> usize {
        self.proposal.evaluations
    }

    #[getter]
    fn regularizations(&self) -> usize {
        self.proposal.regularizations
    }

    #[getter]
    fn evidence(&self) -> &'static str {
        "approximate"
    }
}

/// Accepted second-order source enclosures, with no public constructor.
#[pyclass(name = "ObjectiveEnclosure", module = "fsym_python", frozen)]
pub struct PyObjectiveEnclosure {
    receipt: ObjectiveEnclosure,
}

#[pymethods]
impl PyObjectiveEnclosure {
    #[getter]
    fn objective(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.objective().clone())
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.domain())
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.precision_digits()
    }

    #[getter]
    fn value(&self) -> (PyExpr, PyExpr) {
        (
            rational_expr(&self.receipt.value().lower()),
            rational_expr(&self.receipt.value().upper()),
        )
    }

    #[getter]
    fn gradient(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.gradient())
    }

    #[getter]
    fn hessian(&self) -> Vec<Vec<(PyExpr, PyExpr)>> {
        self.receipt
            .hessian()
            .iter()
            .map(|row| ball_bounds(row))
            .collect()
    }
}

/// Accepted local stationary-point evidence, with no public constructor.
#[pyclass(name = "StationaryPointReceipt", module = "fsym_python", frozen)]
pub struct PyStationaryPointReceipt {
    receipt: StationaryPointReceipt,
}

#[pymethods]
impl PyStationaryPointReceipt {
    #[getter]
    fn objective(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.objective().clone())
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(&self.receipt.witness().domain)
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.witness().precision_digits
    }

    #[getter]
    fn center(&self) -> Vec<PyExpr> {
        self.receipt
            .witness()
            .center
            .iter()
            .map(rational_expr)
            .collect()
    }

    #[getter]
    fn preconditioner(&self) -> Vec<Vec<PyExpr>> {
        self.receipt
            .witness()
            .preconditioner
            .iter()
            .map(|row| row.iter().map(rational_expr).collect())
            .collect()
    }

    #[getter]
    fn stationary_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.stationary_enclosure())
    }

    #[getter]
    fn objective_enclosure(&self) -> (PyExpr, PyExpr) {
        (
            rational_expr(&self.receipt.objective_enclosure().lower()),
            rational_expr(&self.receipt.objective_enclosure().upper()),
        )
    }

    #[getter]
    fn gradient_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.gradient_enclosure())
    }

    #[getter]
    fn hessian_enclosure(&self) -> Vec<Vec<(PyExpr, PyExpr)>> {
        self.receipt
            .hessian_enclosure()
            .iter()
            .map(|row| ball_bounds(row))
            .collect()
    }

    #[getter]
    fn contraction_bound(&self) -> PyExpr {
        rational_expr(self.receipt.contraction_bound())
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.receipt.kind() {
            StationaryPointKind::StrictMinimum => "strict_minimum",
            StationaryPointKind::StrictMaximum => "strict_maximum",
            StationaryPointKind::Saddle => "saddle",
            StationaryPointKind::Unclassified => "unclassified",
        }
    }
}

/// Accepted global-on-box optimality, with no public constructor.
#[pyclass(name = "BoxExtremumReceipt", module = "fsym_python", frozen)]
pub struct PyBoxExtremumReceipt {
    receipt: BoxExtremumReceipt,
}

#[pymethods]
impl PyBoxExtremumReceipt {
    #[getter]
    fn objective(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.objective().clone())
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.domain())
    }

    #[getter]
    fn point(&self) -> Vec<PyExpr> {
        self.receipt.point().iter().map(rational_expr).collect()
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.precision_digits()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.receipt.kind() {
            BoxExtremumKind::Minimum => "minimum",
            BoxExtremumKind::Maximum => "maximum",
        }
    }

    #[getter]
    fn value(&self) -> (PyExpr, PyExpr) {
        (
            rational_expr(&self.receipt.value().lower()),
            rational_expr(&self.receipt.value().upper()),
        )
    }

    #[getter]
    fn gradient(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.gradient())
    }

    #[getter]
    fn hessian(&self) -> Vec<Vec<(PyExpr, PyExpr)>> {
        self.receipt
            .hessian()
            .iter()
            .map(|row| ball_bounds(row))
            .collect()
    }

    #[getter]
    fn uniqueness_proved(&self) -> bool {
        self.receipt.uniqueness_proved()
    }
}

/// Global value enclosure and feasible-point suboptimality, without a public
/// constructor. This is not an exact optimizer or coordinate-distance claim.
#[pyclass(name = "BoxOptimizationReceipt", module = "fsym_python", frozen)]
pub struct PyBoxOptimizationReceipt {
    receipt: BoxOptimizationReceipt,
}

#[pymethods]
impl PyBoxOptimizationReceipt {
    #[getter]
    fn objective(&self) -> PyExpr {
        PyExpr::from_expr(self.receipt.objective().clone())
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .variables()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.domain())
    }

    #[getter]
    fn point(&self) -> Vec<PyExpr> {
        self.receipt.point().iter().map(rational_expr).collect()
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.precision_digits()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.receipt.kind() {
            BoxExtremumKind::Minimum => "minimum",
            BoxExtremumKind::Maximum => "maximum",
        }
    }

    #[getter]
    fn value(&self) -> (PyExpr, PyExpr) {
        (
            rational_expr(&self.receipt.value().lower()),
            rational_expr(&self.receipt.value().upper()),
        )
    }

    #[getter]
    fn point_value(&self) -> (PyExpr, PyExpr) {
        (
            rational_expr(&self.receipt.point_value().lower()),
            rational_expr(&self.receipt.point_value().upper()),
        )
    }

    #[getter]
    fn gap(&self) -> PyExpr {
        rational_expr(self.receipt.gap())
    }

    #[getter]
    fn absolute_tolerance(&self) -> PyExpr {
        rational_expr(self.receipt.absolute_tolerance())
    }

    #[getter]
    fn leaf_count(&self) -> usize {
        self.receipt.leaf_count()
    }
}
