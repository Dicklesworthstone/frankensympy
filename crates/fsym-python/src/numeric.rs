//! Reusable typed numerical programs for the Python bridge.
//!
//! These bounded synchronous programs evaluate finite real binary64 values.
//! Source/bytecode checks and symbolic derivative proofs do not certify the
//! rounding error of numerical outputs. No Python callback or source-code
//! evaluation is used during compilation or execution.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::{PyExpr, PySymbol};
use crate::matrix::{PyMatrix, PySparseMatrix};
use crate::numeric_budget::{
    PyEvaluationBudget, evaluation_buffer, with_certification_meter, with_evaluation_meter,
    with_evaluation_meter_attached,
};
use fsym_budget::{BudgetMeter, Dimension};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, EvalError, MAX_COMPILE_DEPTH, MAX_COMPILE_NAME_BYTES};
use fsym_calculus::parametric_cover::{ParametricRootCoverOptions, propose_parametric_root_cover};
use fsym_calculus::root_isolation::{RootIsolationOptions, propose_root_isolation};
use fsym_calculus::sparse_jacobian::{compiled_expression_matches, plan_sparse_sources};
use fsym_calculus::verified::{
    MAX_VERIFIED_ARTIFACT_BYTES, ResidualEvaluationError, VerifiedArtifactError,
};
use fsym_core::{
    BigRational, Expr, ParametricRootBoxReceipt, ParametricRootBoxWitness,
    ParametricRootCoverReceipt, ParametricSensitivityReceipt, RealBall, ResidualJacobianEnclosure,
    RootBoxReceipt, RootBoxWitness, RootIsolationReceipt, Symbol,
    metered_enclose_parametric_cover_sensitivity, metered_enclose_parametric_root_sensitivity,
    metered_enclose_residual_jacobian, verify_parametric_root_box, verify_parametric_root_cover,
    verify_root_box, verify_root_isolation,
};
use fsym_numeric_consumer::{
    LeastSquaresOptions, LeastSquaresTermination, MAX_CONSUMER_NEWTON_ITERATIONS,
    MAX_CONSUMER_VARIABLES, MAX_LEAST_SQUARES_RESIDUALS, MAX_SPARSE_LEAST_SQUARES_RESIDUALS,
    MAX_SPARSE_LEAST_SQUARES_VARIABLES, MAX_SPARSE_NEWTON_VARIABLES, NewtonGuards,
    SystemConsumerError, frankenscipy_analytic_system_with_meter,
    frankenscipy_fsolve_system_with_meter, least_squares_with_meter, newton_refine_with_meter,
    sparse_least_squares_with_meter, sparse_newton_refine_with_meter,
};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFloat, PyInt, PyList, PyTuple};
use std::collections::HashMap;

const MAX_VARIABLES: usize = 256;
const MAX_OUTPUTS: usize = 4096;
const MAX_TOTAL_OPS: usize = 65_536;
const MAX_SOURCE_NODES: usize = 65_536;
const MAX_LITERAL_BITS: u64 = 4096;
const MAX_CERTIFIED_VARIABLES: usize = 16;
const MAX_BATCH_ROWS: usize = 4096;
const MAX_BATCH_VALUES: usize = 262_144;

type ResidualValues = (Vec<f64>, Vec<Vec<f64>>);

pub(crate) fn donor_float_option(
    value: Option<&Bound<'_, PyAny>>,
    default: f64,
    name: &'static str,
    allow_zero: bool,
) -> PyResult<f64> {
    let value = if let Some(value) = value {
        let py = value.py();
        if !value.is_exact_instance(&py.get_type::<PyInt>())
            && !value.is_exact_instance(&py.get_type::<PyFloat>())
        {
            return Err(PyTypeError::new_err(format!(
                "{name} must be an exact built-in int or float"
            )));
        }
        value.extract::<f64>()?
    } else {
        default
    };
    if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and {}",
            if allow_zero {
                "nonnegative"
            } else {
                "positive"
            }
        )));
    }
    Ok(value)
}

pub(crate) fn donor_newton_guards(
    tol: Option<&Bound<'_, PyAny>>,
    maxiter: Option<&Bound<'_, PyAny>>,
    max_step_norm: Option<&Bound<'_, PyAny>>,
) -> PyResult<NewtonGuards> {
    let defaults = NewtonGuards::default();
    let max_iterations = if let Some(value) = maxiter {
        if !value.is_exact_instance(&value.py().get_type::<PyInt>()) {
            return Err(PyTypeError::new_err(
                "maxiter must be an exact built-in integer",
            ));
        }
        value
            .extract::<usize>()
            .map_err(|_| PyValueError::new_err("FrankenNumPy maxiter must be between 0 and 1024"))?
    } else {
        defaults.max_iterations
    };
    if max_iterations > MAX_CONSUMER_NEWTON_ITERATIONS {
        return Err(PyValueError::new_err(
            "FrankenNumPy maxiter must be between 0 and 1024",
        ));
    }
    Ok(NewtonGuards {
        max_iterations,
        residual_tolerance: donor_float_option(tol, defaults.residual_tolerance, "tol", true)?,
        max_step_norm: donor_float_option(
            max_step_norm,
            defaults.max_step_norm,
            "max_step_norm",
            false,
        )?,
    })
}

pub(crate) fn exact_rational(
    value: &Bound<'_, PyAny>,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<BigRational> {
    if !value.is_exact_instance(&value.py().get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "exact bounds require native rational Expr objects",
        ));
    }
    let value = value.extract::<PyRef<'_, PyExpr>>()?;
    match &value.inner {
        Expr::Integer(n) if n.bits() <= MAX_LITERAL_BITS => {
            charge_exact_copy(budget, n.bits().div_ceil(64) + 1)?;
            Ok(BigRational::from_integer(n.clone()))
        }
        Expr::Rational(q) if q.numer().bits().max(q.denom().bits()) <= MAX_LITERAL_BITS => {
            charge_exact_copy(
                budget,
                q.numer().bits().div_ceil(64) + q.denom().bits().div_ceil(64),
            )?;
            Ok(q.clone())
        }
        _ => Err(PyValueError::new_err(
            "exact bounds must be rational literals of at most 4096 bits",
        )),
    }
}

fn charge_exact_copy(budget: Option<&PyEvaluationBudget>, limbs: u64) -> PyResult<()> {
    // The bounded limb payload and two integer slots are conservative logical
    // copy reservations, independent of small-integer storage optimizations.
    with_evaluation_meter_attached(budget, |meter| {
        meter.charge_batch(&[
            (Dimension::ComputeSteps, limbs + 1),
            (Dimension::MemoryBytes, 8 * limbs.max(1)),
            (Dimension::AllocationCount, 2),
        ])
    })
}

fn exact_buffer<T>(size: usize, budget: Option<&PyEvaluationBudget>) -> PyResult<Vec<T>> {
    with_evaluation_meter_attached(budget, |meter| evaluation_buffer(size, meter))
}

pub(crate) fn exact_vector(
    values: &Bound<'_, PyAny>,
    expected: usize,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Vec<BigRational>> {
    if container_length(values, MAX_CERTIFIED_VARIABLES, "exact vector")? != expected {
        return Err(PyValueError::new_err(
            "exact vector has the wrong dimension",
        ));
    }
    let mut result = exact_buffer(expected, budget)?;
    for index in 0..expected {
        result.push(exact_rational(&values.get_item(index)?, budget)?);
    }
    Ok(result)
}

pub(crate) fn exact_box(
    values: &Bound<'_, PyAny>,
    expected: usize,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Vec<RealBall>> {
    if container_length(values, MAX_CERTIFIED_VARIABLES, "box")? != expected {
        return Err(PyValueError::new_err("box has the wrong dimension"));
    }
    let mut result = exact_buffer(expected, budget)?;
    for index in 0..expected {
        let coordinate = values.get_item(index)?;
        if container_length(&coordinate, 2, "coordinate bounds")? != 2 {
            return Err(PyValueError::new_err(
                "coordinate bounds require two endpoints",
            ));
        }
        let lo = exact_rational(&coordinate.get_item(0)?, budget)?;
        let hi = exact_rational(&coordinate.get_item(1)?, budget)?;
        let bits = lo
            .numer()
            .bits()
            .max(lo.denom().bits())
            .max(hi.numer().bits())
            .max(hi.denom().bits());
        let limbs = bits.div_ceil(64) + 1;
        with_evaluation_meter_attached(budget, |meter| {
            // Reserve comparison and midpoint/radius arithmetic before any
            // cross-product or temporary rational is constructed.
            meter.charge_batch(&[
                (Dimension::ComputeSteps, 320 * limbs * limbs),
                (Dimension::MemoryBytes, 640 * limbs),
                (Dimension::AllocationCount, 80),
            ])
        })?;
        if lo > hi {
            return Err(PyValueError::new_err("box lower bound exceeds upper bound"));
        }
        let two = BigRational::from_integer(2.into());
        result.push(
            RealBall::new((&lo + &hi) / &two, (hi - lo) / two).map_err(crate::to_value_error)?,
        );
    }
    Ok(result)
}

pub(crate) fn rational_expr(value: &BigRational) -> PyExpr {
    PyExpr::from_expr(if value.is_integer() {
        Expr::Integer(value.to_integer())
    } else {
        Expr::Rational(value.clone())
    })
}

pub(crate) fn ball_bounds(values: &[RealBall]) -> Vec<(PyExpr, PyExpr)> {
    values
        .iter()
        .map(|value| (rational_expr(&value.lower()), rational_expr(&value.upper())))
        .collect()
}

/// Accepted enclosures can only be constructed by the independent core checker.
#[pyclass(name = "ResidualJacobianEnclosure", module = "fsym_python", frozen)]
pub struct PyResidualJacobianEnclosure {
    receipt: ResidualJacobianEnclosure,
}

#[pymethods]
impl PyResidualJacobianEnclosure {
    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.receipt
            .residuals()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
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
    fn residual_values(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.residual_values())
    }

    #[getter]
    fn jacobian(&self) -> Vec<Vec<(PyExpr, PyExpr)>> {
        self.receipt
            .jacobian()
            .iter()
            .map(|row| ball_bounds(row))
            .collect()
    }
}

/// An immutable, source-bound accepted local root claim, with no public constructor.
#[pyclass(name = "RootBoxReceipt", module = "fsym_python", frozen)]
pub struct PyRootBoxReceipt {
    receipt: RootBoxReceipt,
}

#[pymethods]
impl PyRootBoxReceipt {
    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.receipt
            .residuals()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
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
    fn precision_digits(&self) -> u32 {
        self.receipt.witness().precision_digits
    }

    #[getter]
    fn root_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.root_enclosure())
    }

    #[getter]
    fn contraction_bound(&self) -> PyExpr {
        rational_expr(self.receipt.contraction_bound())
    }
}

/// Complete original-source root coverage on a finite box, with no public constructor.
#[pyclass(name = "RootIsolationReceipt", module = "fsym_python", frozen)]
pub struct PyRootIsolationReceipt {
    receipt: RootIsolationReceipt,
}

#[pymethods]
impl PyRootIsolationReceipt {
    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.receipt
            .residuals()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
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
    fn roots(&self) -> Vec<PyRootBoxReceipt> {
        self.receipt
            .roots()
            .iter()
            .cloned()
            .map(|receipt| PyRootBoxReceipt { receipt })
            .collect()
    }

    #[getter]
    fn root_count(&self) -> usize {
        self.receipt.root_count()
    }

    #[getter]
    fn leaf_count(&self) -> usize {
        self.receipt.leaf_count()
    }

    #[getter]
    fn excluded_leaf_count(&self) -> usize {
        self.receipt.excluded_leaf_count()
    }

    #[getter]
    fn is_root_free(&self) -> bool {
        self.receipt.is_root_free()
    }
}

/// One unique unknown root for every parameter value, with no public constructor.
#[pyclass(name = "ParametricRootBoxReceipt", module = "fsym_python", frozen)]
pub struct PyParametricRootBoxReceipt {
    receipt: ParametricRootBoxReceipt,
}

#[pymethods]
impl PyParametricRootBoxReceipt {
    /// Independently bound the implicit solution Jacobian on this entire family.
    /// The accepted root claim supplies existence; no new root proposal or
    /// numerical derivative is used. All source work shares the caller ledger.
    #[pyo3(signature = (*, budget=None))]
    fn enclose_sensitivity(
        &self,
        py: Python<'_>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyParametricSensitivityReceipt> {
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            metered_enclose_parametric_root_sensitivity(&self.receipt, meter)
        })?;
        let result = PyParametricSensitivityReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.receipt
            .residuals()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .unknowns()
            .iter()
            .chain(self.receipt.parameters())
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn unknown_variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .unknowns()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn parameters(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .parameters()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        let mut domain = ball_bounds(&self.receipt.witness().unknown_domain);
        domain.extend(ball_bounds(&self.receipt.witness().parameter_domain));
        domain
    }

    #[getter]
    fn unknown_domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(&self.receipt.witness().unknown_domain)
    }

    #[getter]
    fn parameter_domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(&self.receipt.witness().parameter_domain)
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
    fn precision_digits(&self) -> u32 {
        self.receipt.witness().precision_digits
    }

    #[getter]
    fn root_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.root_enclosure())
    }

    #[getter]
    fn contraction_bound(&self) -> PyExpr {
        rational_expr(self.receipt.contraction_bound())
    }
}

/// Complete parameter coverage with a unique root in one common unknown box.
#[pyclass(name = "ParametricRootCoverReceipt", module = "fsym_python", frozen)]
pub struct PyParametricRootCoverReceipt {
    receipt: ParametricRootCoverReceipt,
}

#[pymethods]
impl PyParametricRootCoverReceipt {
    /// Independently bound every cell and take the complete-cover derivative hull.
    /// An operational refusal returns no partial sensitivity receipt and does
    /// not invalidate the already accepted root cover.
    #[pyo3(signature = (*, budget=None))]
    fn enclose_sensitivity(
        &self,
        py: Python<'_>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyParametricSensitivityReceipt> {
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            metered_enclose_parametric_cover_sensitivity(&self.receipt, meter)
        })?;
        let result = PyParametricSensitivityReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.receipt
            .residuals()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .unknowns()
            .iter()
            .chain(self.receipt.parameters())
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn unknown_variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .unknowns()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn parameters(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .parameters()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        let mut domain = ball_bounds(self.receipt.unknown_domain());
        domain.extend(ball_bounds(self.receipt.parameter_domain()));
        domain
    }

    #[getter]
    fn unknown_domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.unknown_domain())
    }

    #[getter]
    fn parameter_domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.parameter_domain())
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.precision_digits()
    }

    #[getter]
    fn leaves(&self) -> Vec<PyParametricRootBoxReceipt> {
        self.receipt
            .leaves()
            .iter()
            .cloned()
            .map(|receipt| PyParametricRootBoxReceipt { receipt })
            .collect()
    }

    #[getter]
    fn leaf_count(&self) -> usize {
        self.receipt.leaf_count()
    }

    #[getter]
    fn root_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.root_enclosure())
    }
}

/// Source-bound implicit derivative enclosures, with no unchecked constructor.
/// Rows follow unknown_variables; columns follow parameters. The private core
/// receipt binds the whole accepted parameter domain, including cover faces.
#[pyclass(name = "ParametricSensitivityReceipt", module = "fsym_python", frozen)]
pub struct PyParametricSensitivityReceipt {
    receipt: ParametricSensitivityReceipt,
}

#[pymethods]
impl PyParametricSensitivityReceipt {
    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.receipt
            .residuals()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    #[getter]
    fn variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .unknowns()
            .iter()
            .chain(self.receipt.parameters())
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn unknown_variables(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .unknowns()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn parameters(&self) -> Vec<PySymbolBinding> {
        self.receipt
            .parameters()
            .iter()
            .map(PySymbolBinding::from_symbol)
            .collect()
    }

    #[getter]
    fn domain(&self) -> Vec<(PyExpr, PyExpr)> {
        let mut domain = ball_bounds(self.receipt.unknown_domain());
        domain.extend(ball_bounds(self.receipt.parameter_domain()));
        domain
    }

    #[getter]
    fn unknown_domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.unknown_domain())
    }

    #[getter]
    fn parameter_domain(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.parameter_domain())
    }

    #[getter]
    fn precision_digits(&self) -> u32 {
        self.receipt.precision_digits()
    }

    #[getter]
    fn root_enclosure(&self) -> Vec<(PyExpr, PyExpr)> {
        ball_bounds(self.receipt.root_enclosure())
    }

    #[getter]
    fn solution_jacobian(&self) -> Vec<Vec<(PyExpr, PyExpr)>> {
        self.receipt
            .solution_jacobian()
            .iter()
            .map(|row| ball_bounds(row))
            .collect()
    }

    #[getter]
    fn contraction_bound(&self) -> PyExpr {
        rational_expr(self.receipt.contraction_bound())
    }

    #[getter]
    fn leaf_count(&self) -> usize {
        self.receipt.leaf_count()
    }
}

fn container_length(values: &Bound<'_, PyAny>, maximum: usize, name: &str) -> PyResult<usize> {
    let py = values.py();
    if !values.is_exact_instance(&py.get_type::<PyList>())
        && !values.is_exact_instance(&py.get_type::<PyTuple>())
    {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an exact built-in list or tuple"
        )));
    }
    let length = values.len()?;
    if length > maximum {
        return Err(PyValueError::new_err(format!(
            "{name} has {length} items; the limit is {maximum}"
        )));
    }
    Ok(length)
}

fn items<'py>(
    values: &Bound<'py, PyAny>,
    maximum: usize,
    name: &str,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let length = container_length(values, maximum, name)?;
    (0..length).map(|index| values.get_item(index)).collect()
}

pub(crate) fn extract_variables(values: &Bound<'_, PyAny>) -> PyResult<Vec<Symbol>> {
    let py = values.py();
    items(values, MAX_VARIABLES, "variables")?
        .iter()
        .map(|value| {
            if !value.is_exact_instance(&py.get_type::<PySymbolBinding>())
                && !value.is_exact_instance(&py.get_type::<PySymbol>())
                && !value.is_exact_instance(&py.get_type::<PyExpr>())
            {
                return Err(PyTypeError::new_err(
                    "variables must be typed SymbolBinding or native symbol objects",
                ));
            }
            crate::extract_symbol_arg(value)
        })
        .collect()
}

/// Check the original tree before cloning or formatting large literals.
pub(crate) fn check_source(expr: &Expr, nodes: &mut usize) -> PyResult<()> {
    let mut pending = vec![(expr, 0usize)];
    while let Some((node, depth)) = pending.pop() {
        *nodes += 1;
        if *nodes > MAX_SOURCE_NODES || depth > MAX_COMPILE_DEPTH {
            return Err(PyValueError::new_err(
                "numeric source exceeds the node or depth limit",
            ));
        }
        let children = match node {
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => args.len(),
            Expr::Pow(_, _) => 2,
            _ => 0,
        };
        if children > MAX_SOURCE_NODES.saturating_sub(*nodes + pending.len()) {
            return Err(PyValueError::new_err(
                "numeric source exceeds the node limit",
            ));
        }
        match node {
            Expr::Sym(symbol) if symbol.name.len() > MAX_COMPILE_NAME_BYTES => {
                return Err(PyValueError::new_err(
                    "numeric symbol name exceeds the 4096-byte limit",
                ));
            }
            Expr::Function(name, _) if name.len() > MAX_COMPILE_NAME_BYTES => {
                return Err(PyValueError::new_err(
                    "numeric function name exceeds the 4096-byte limit",
                ));
            }
            Expr::Integer(value) if value.bits() > MAX_LITERAL_BITS => {
                return Err(PyValueError::new_err(
                    "numeric integer exceeds the literal limit",
                ));
            }
            Expr::Rational(value)
                if value.numer().bits().max(value.denom().bits()) > MAX_LITERAL_BITS =>
            {
                return Err(PyValueError::new_err(
                    "numeric rational exceeds the literal limit",
                ));
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                pending.extend(args.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            _ => {}
        }
    }
    Ok(())
}

fn extract_expressions(values: &Bound<'_, PyAny>) -> PyResult<Vec<Expr>> {
    let py = values.py();
    let mut nodes = 0;
    items(values, MAX_OUTPUTS, "expressions")?
        .iter()
        .map(|value| {
            if !value.is_exact_instance(&py.get_type::<PyExpr>()) {
                return Err(PyTypeError::new_err(
                    "expressions must be exact native Expr objects",
                ));
            }
            let value = value.extract::<PyRef<'_, PyExpr>>()?;
            check_source(&value.inner, &mut nodes)?;
            Ok(value.inner.clone())
        })
        .collect()
}

/// Validate the complete exact source without generating floating-point code.
/// This is ingress validation only: the original-source interval checker still
/// establishes every domain, derivative and certificate claim on the given box.
pub(crate) fn check_certified_sources(sources: &[Expr], variables: &[Symbol]) -> PyResult<()> {
    if !(1..=MAX_CERTIFIED_VARIABLES).contains(&variables.len()) {
        return Err(PyValueError::new_err(
            "certified sources require 1..16 variables",
        ));
    }
    let bindings = variable_map(variables)?;
    let mut nodes = 0;
    for source in sources {
        check_source(source, &mut nodes)?;
    }
    let mut pending: Vec<(&Expr, bool)> = sources.iter().map(|source| (source, false)).collect();
    while let Some((source, condition)) = pending.pop() {
        if condition {
            match source {
                Expr::Sym(symbol)
                    if symbol.is_plain() && matches!(symbol.name.as_str(), "True" | "False") => {}
                Expr::Function(name, arguments)
                    if matches!(
                        name.as_str(),
                        "Equality"
                            | "Unequality"
                            | "StrictLessThan"
                            | "LessThan"
                            | "StrictGreaterThan"
                            | "GreaterThan"
                    ) && arguments.len() == 2 =>
                {
                    pending.extend(arguments.iter().map(|argument| (argument, false)));
                }
                Expr::Function(name, arguments) if matches!(name.as_str(), "And" | "Or") => {
                    pending.extend(arguments.iter().map(|argument| (argument, true)));
                }
                Expr::Function(name, arguments) if name == "Not" && arguments.len() == 1 => {
                    pending.push((&arguments[0], true));
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "unsupported certified Piecewise condition",
                    ));
                }
            }
            continue;
        }
        match source {
            Expr::Integer(_)
            | Expr::Rational(_)
            | Expr::Const(fsym_core::Constant::Pi | fsym_core::Constant::E) => {}
            Expr::Sym(symbol) if bindings.contains_key(symbol) => {}
            Expr::Sym(_) => {
                return Err(PyValueError::new_err(
                    "unbound certified source symbol identity",
                ));
            }
            Expr::Add(children) | Expr::Mul(children) => {
                pending.extend(children.iter().map(|child| (child, false)));
            }
            Expr::Pow(base, exponent) => {
                pending.push((base, false));
                pending.push((exponent, false));
            }
            Expr::Function(name, branches) if name == "Piecewise" && !branches.is_empty() => {
                for branch in branches {
                    let Expr::Function(tag, pair) = branch else {
                        return Err(PyValueError::new_err(
                            "malformed certified Piecewise branch",
                        ));
                    };
                    if tag != "Tuple" || pair.len() != 2 {
                        return Err(PyValueError::new_err(
                            "malformed certified Piecewise branch",
                        ));
                    }
                    pending.push((&pair[0], false));
                    pending.push((&pair[1], true));
                }
            }
            Expr::Function(name, arguments)
                if matches!(
                    name.as_str(),
                    "sin"
                        | "cos"
                        | "tan"
                        | "exp"
                        | "log"
                        | "ln"
                        | "sqrt"
                        | "asin"
                        | "acos"
                        | "atan"
                        | "sinh"
                        | "cosh"
                        | "tanh"
                ) && arguments.len() == 1 =>
            {
                pending.push((&arguments[0], false));
            }
            _ => return Err(PyValueError::new_err("unsupported certified scalar source")),
        }
    }
    Ok(())
}

pub(crate) fn input_values(values: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<f64>> {
    input_values_bounded(values, expected, MAX_VARIABLES)
}

fn input_values_bounded(
    values: &Bound<'_, PyAny>,
    expected: usize,
    maximum: usize,
) -> PyResult<Vec<f64>> {
    let py = values.py();
    let actual = container_length(values, maximum, "numeric inputs")?;
    if actual != expected {
        return Err(PyTypeError::new_err(format!(
            "expected {expected} numeric inputs, got {actual}"
        )));
    }
    // Reserve the charged size once. Fallible iterator collection can grow a
    // vector through uncharged reallocations, and copying handles first would
    // allocate an additional buffer for no numerical work.
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(expected)
        .map_err(|_| crate::numeric_budget::evaluation_error(EvalError::AllocationFailure))?;
    for index in 0..expected {
        let value = values.get_item(index)?;
        if !value.is_exact_instance(&py.get_type::<PyFloat>())
            && !value.is_exact_instance(&py.get_type::<PyInt>())
        {
            return Err(PyTypeError::new_err(
                "numeric inputs must be exact built-in int or float values",
            ));
        }
        let value = value.extract::<f64>()?;
        if !value.is_finite() {
            return Err(PyValueError::new_err("numeric inputs must be finite"));
        }
        inputs.push(value);
    }
    Ok(inputs)
}

fn charge_input_storage(
    budget: Option<&PyEvaluationBudget>,
    entries: usize,
    rows: usize,
    batched: bool,
) -> PyResult<()> {
    // Native f64 snapshots and the outer batch vector have exact reservation
    // requests. Python handles stay on the stack; CPython objects and the
    // caller's original containers do not belong to this native ledger.
    let bytes = entries * std::mem::size_of::<f64>()
        + if batched {
            rows * std::mem::size_of::<Vec<f64>>()
        } else {
            0
        };
    let allocations = if entries != 0 { rows } else { 0 } + usize::from(batched && rows != 0);
    with_evaluation_meter_attached(budget, |meter| {
        if bytes != 0 {
            meter.charge_batch(&[
                (Dimension::MemoryBytes, bytes as u64),
                (Dimension::AllocationCount, allocations as u64),
                (Dimension::ComputeSteps, entries.max(rows) as u64),
            ])?;
        }
        Ok::<_, fsym_budget::MeterError>(())
    })
}

pub(crate) fn input_values_metered(
    values: &Bound<'_, PyAny>,
    expected: usize,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Vec<f64>> {
    input_values_metered_bounded(values, expected, MAX_VARIABLES, budget)
}

fn input_values_metered_bounded(
    values: &Bound<'_, PyAny>,
    expected: usize,
    maximum: usize,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Vec<f64>> {
    let actual = container_length(values, maximum, "numeric inputs")?;
    if actual != expected {
        return Err(PyTypeError::new_err(format!(
            "expected {expected} numeric inputs, got {actual}"
        )));
    }
    charge_input_storage(budget, expected, 1, false)?;
    input_values_bounded(values, expected, maximum)
}

fn evaluate_programs(
    programs: &[CompiledExpr],
    inputs: &[f64],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<f64>, EvalError> {
    let mut values = evaluation_buffer(programs.len(), meter)?;
    for program in programs {
        meter.charge(Dimension::ComputeSteps, 1)?;
        values.push(program.try_eval_finite_with_meter(inputs, meter)?);
    }
    meter.checkpoint()?;
    Ok(values)
}

fn jacobian_rows(
    flat: &[f64],
    rows: usize,
    columns: usize,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<Vec<f64>>, EvalError> {
    let mut result = evaluation_buffer(rows, meter)?;
    for row in 0..rows {
        let mut values = evaluation_buffer(columns, meter)?;
        if columns != 0 {
            meter.charge(Dimension::ComputeSteps, columns as u64)?;
        }
        values.extend_from_slice(&flat[row * columns..(row + 1) * columns]);
        result.push(values);
    }
    meter.checkpoint()?;
    Ok(result)
}

fn batch_inputs(
    rows: &Bound<'_, PyAny>,
    variables: usize,
    outputs: usize,
    budget: Option<&PyEvaluationBudget>,
) -> PyResult<Vec<Vec<f64>>> {
    let count = container_length(rows, MAX_BATCH_ROWS, "numeric batch")?;
    if count.saturating_mul(variables.max(outputs).max(1)) > MAX_BATCH_VALUES {
        return Err(PyValueError::new_err(
            "numeric batch exceeds the total input/output value limit",
        ));
    }
    charge_input_storage(budget, count * variables, count, true)?;
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(count)
        .map_err(|_| crate::to_value_error(EvalError::AllocationFailure))?;
    for index in 0..count {
        inputs.push(input_values(&rows.get_item(index)?, variables)?);
    }
    Ok(inputs)
}

fn variable_map(vars: &[Symbol]) -> PyResult<HashMap<Symbol, usize>> {
    let mut map = HashMap::new();
    for (index, var) in vars.iter().enumerate() {
        if map.insert(var.clone(), index).is_some() {
            return Err(PyValueError::new_err("duplicate typed numerical variable"));
        }
    }
    Ok(map)
}

/// Immutable reusable real binary64 programs, checked against typed sources.
#[pyclass(name = "CompiledFunction", module = "fsym_python", frozen)]
pub struct PyCompiledFunction {
    programs: Vec<CompiledExpr>,
    variable_count: usize,
    operation_count: usize,
}

impl PyCompiledFunction {
    fn compile(exprs: &[Expr], vars: &[Symbol]) -> PyResult<Self> {
        let map = variable_map(vars)?;
        let mut programs = Vec::with_capacity(exprs.len());
        let mut operation_count = 0;
        for source in exprs {
            let program = CompiledExpr::try_compile(source, &map).map_err(crate::to_value_error)?;
            operation_count += program.ops().len();
            if operation_count > MAX_TOTAL_OPS {
                return Err(PyValueError::new_err(
                    "numeric program exceeds the total operation limit",
                ));
            }
            if !compiled_expression_matches(source, &map, &program)
                .map_err(crate::to_value_error)?
            {
                return Err(PyValueError::new_err(
                    "numeric bytecode provenance verification failed",
                ));
            }
            programs.push(program);
        }
        Ok(Self {
            programs,
            variable_count: vars.len(),
            operation_count,
        })
    }

    #[cfg(test)]
    fn evaluate_values(&self, values: &[f64]) -> PyResult<Vec<f64>> {
        if values.len() != self.variable_count || values.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "invalid numeric input dimension or nonfinite value",
            ));
        }
        evaluate_programs(&self.programs, values, &mut fsym_budget::Unbounded)
            .map_err(crate::numeric_budget::evaluation_error)
    }
}

#[pymethods]
impl PyCompiledFunction {
    /// Preflight a borrowed native source before Python traverses cloned children.
    /// This is admission only; it neither executes nor certifies the expression.
    #[staticmethod]
    fn _validate_source(source: &Bound<'_, PyAny>) -> PyResult<()> {
        if !source.is_exact_instance(&source.py().get_type::<PyExpr>()) {
            return Err(PyTypeError::new_err(
                "numeric source must be an exact native Expr object",
            ));
        }
        let source = source.extract::<PyRef<'_, PyExpr>>()?;
        check_source(&source.inner, &mut 0)
    }

    /// Check all borrowed matrix entries before Python snapshots native cells.
    #[staticmethod]
    fn _validate_matrix_source(source: &Bound<'_, PyAny>) -> PyResult<()> {
        fn dimensions(rows: usize, columns: usize) -> PyResult<()> {
            if rows > MAX_OUTPUTS
                || columns > MAX_OUTPUTS
                || rows
                    .checked_mul(columns)
                    .is_none_or(|cells| cells > MAX_OUTPUTS)
            {
                return Err(PyValueError::new_err(
                    "numeric matrix exceeds the output limit",
                ));
            }
            Ok(())
        }

        let mut nodes = 0;
        if source.is_exact_instance(&source.py().get_type::<PyMatrix>()) {
            let source = source.extract::<PyRef<'_, PyMatrix>>()?;
            dimensions(source.inner.rows(), source.inner.cols())?;
            for cell in source.inner.data() {
                check_source(cell, &mut nodes)?;
            }
        } else if source.is_exact_instance(&source.py().get_type::<PySparseMatrix>()) {
            let source = source.extract::<PyRef<'_, PySparseMatrix>>()?;
            dimensions(source.inner.rows(), source.inner.cols())?;
            // Missing sparse cells are scalar zeros. Account for them before
            // traversing the borrowed, explicitly stored expressions.
            nodes = source.inner.rows() * source.inner.cols() - source.inner.entries().len();
            for cell in source.inner.entries().values() {
                check_source(cell, &mut nodes)?;
            }
        } else {
            return Err(PyTypeError::new_err(
                "numeric matrix source must be an exact native Matrix or SparseMatrix",
            ));
        }
        Ok(())
    }

    #[new]
    fn new(expressions: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exprs = extract_expressions(expressions)?;
        let vars = extract_variables(variables)?;
        Self::compile(&exprs, &vars)
    }

    /// Evaluate all outputs atomically. Results are approximate binary64 values.
    #[pyo3(signature = (values, *, budget=None))]
    fn evaluate(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<f64>> {
        let inputs = input_values_metered(values, self.variable_count, budget.as_deref())?;
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            evaluate_programs(&self.programs, &inputs, meter)
        })
    }

    /// Evaluate a bounded batch using one ledger and publish no partial rows.
    #[pyo3(signature = (rows, *, budget=None))]
    fn evaluate_batch(
        &self,
        rows: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let inputs = batch_inputs(
            rows,
            self.variable_count,
            self.programs.len(),
            budget.as_deref(),
        )?;
        with_evaluation_meter(rows.py(), budget.as_deref(), |meter| {
            let mut results = evaluation_buffer(inputs.len(), meter)?;
            for values in &inputs {
                results.push(evaluate_programs(&self.programs, values, meter)?);
            }
            Ok::<_, EvalError>(results)
        })
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.variable_count
    }

    #[getter]
    fn output_count(&self) -> usize {
        self.programs.len()
    }

    #[getter]
    fn operation_count(&self) -> usize {
        self.operation_count
    }
}

/// Immutable exact sources for independent interval checks. Construction makes
/// no mathematical claim and does not compile approximate values or derivatives.
#[pyclass(name = "SourceResidualSystem", module = "fsym_python", frozen)]
pub struct PySourceResidualSystem {
    residuals: Vec<Expr>,
    variables: Vec<Symbol>,
}

#[pymethods]
impl PySourceResidualSystem {
    #[new]
    fn new(residuals: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        let count = container_length(residuals, 64, "certified residuals")?;
        if count == 0 {
            return Err(PyValueError::new_err(
                "certified sources require 1..64 residuals",
            ));
        }
        let residuals = extract_expressions(residuals)?;
        let variables = extract_variables(variables)?;
        check_certified_sources(&residuals, &variables)?;
        Ok(Self {
            residuals,
            variables,
        })
    }

    #[getter]
    fn residual_count(&self) -> usize {
        self.residuals.len()
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.variables.len()
    }

    #[getter]
    fn residuals(&self) -> Vec<PyExpr> {
        self.residuals
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
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
    ) -> PyResult<PyResidualJacobianEnclosure> {
        let py = domain.py();
        let domain = exact_box(domain, self.variables.len(), budget.as_deref())?;
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            metered_enclose_residual_jacobian(
                &self.residuals,
                &self.variables,
                &domain,
                precision_digits,
                meter,
            )
        })?;
        let result = PyResidualJacobianEnclosure { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    #[pyo3(signature = (domain, center, preconditioner, precision_digits=32, *, budget=None))]
    fn certify_root_box(
        &self,
        domain: &Bound<'_, PyAny>,
        center: &Bound<'_, PyAny>,
        preconditioner: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyRootBoxReceipt> {
        let dimension = self.variables.len();
        let py = domain.py();
        let domain = exact_box(domain, dimension, budget.as_deref())?;
        let center = exact_vector(center, dimension, budget.as_deref())?;
        if container_length(preconditioner, MAX_CERTIFIED_VARIABLES, "preconditioner")? != dimension
        {
            return Err(PyValueError::new_err(
                "preconditioner has the wrong dimension",
            ));
        }
        let mut matrix = exact_buffer(dimension, budget.as_deref())?;
        for index in 0..dimension {
            matrix.push(exact_vector(
                &preconditioner.get_item(index)?,
                dimension,
                budget.as_deref(),
            )?);
        }
        let witness = RootBoxWitness {
            domain,
            center,
            preconditioner: matrix,
            precision_digits,
        };
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            verify_root_box(&self.residuals, &self.variables, &witness, meter)
        })?;
        let result = PyRootBoxReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }
}

/// Independently verified symbolic Jacobian with finite binary64 evaluation.
#[pyclass(name = "VerifiedResidualSystem", module = "fsym_python", frozen)]
pub struct PyVerifiedResidualSystem {
    system: VerifiedResidualSystem,
}

impl PyVerifiedResidualSystem {
    fn check_requested_source(residuals: &[Expr], variables: &[Symbol]) -> PyResult<()> {
        variable_map(variables)?;
        let mut source_nodes = 0;
        for residual in residuals {
            check_source(residual, &mut source_nodes)?;
        }
        let source_plan =
            plan_sparse_sources(residuals, variables).map_err(crate::to_value_error)?;
        if source_plan.source_nodes > MAX_SOURCE_NODES
            || source_plan.differentiation_units > MAX_TOTAL_OPS
        {
            return Err(PyValueError::new_err(
                "verified system exceeds the differentiation work limit",
            ));
        }
        Ok(())
    }

    fn admit(system: VerifiedResidualSystem) -> PyResult<Self> {
        if system.stored_operation_count() > MAX_TOTAL_OPS {
            return Err(PyValueError::new_err(
                "verified numeric system exceeds the total operation limit",
            ));
        }
        Ok(Self { system })
    }

    fn transport_error(error: VerifiedArtifactError) -> PyErr {
        match error {
            VerifiedArtifactError::AllocationFailure => PyMemoryError::new_err(error.to_string()),
            _ => crate::to_value_error(error),
        }
    }

    fn evaluate_values(
        &self,
        inputs: &[f64],
        meter: &mut impl BudgetMeter,
    ) -> Result<ResidualValues, ResidualEvaluationError> {
        let result = self.system.try_eval_dense_with_meter(inputs, meter)?;
        let rows = jacobian_rows(
            &result.jacobian,
            self.system.num_residuals(),
            self.system.num_vars(),
            meter,
        )?;
        Ok((result.residuals, rows))
    }
}

#[pymethods]
impl PyVerifiedResidualSystem {
    #[new]
    fn new(residuals: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        let residuals = extract_expressions(residuals)?;
        let variables = extract_variables(variables)?;
        Self::check_requested_source(&residuals, &variables)?;
        let system = VerifiedResidualSystem::try_compile(&residuals, &variables)
            .map_err(crate::to_value_error)?;
        Self::admit(system)
    }

    /// Generate an approximate root through an explicitly selected native consumer.
    /// Native buffers, donor work and every source evaluation share one ledger.
    /// No convergence flag or residual tolerance grants a root certificate.
    #[pyo3(signature = (initial, *, backend="frankennumpy", tol=None, maxiter=None, max_step_norm=None, budget=None))]
    fn solve_candidate<'py>(
        &self,
        initial: &Bound<'py, PyAny>,
        backend: &str,
        tol: Option<&Bound<'_, PyAny>>,
        maxiter: Option<&Bound<'_, PyAny>>,
        max_step_norm: Option<&Bound<'_, PyAny>>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let guards = match backend {
            "frankennumpy" | "sparse" => Some(donor_newton_guards(tol, maxiter, max_step_norm)?),
            "frankenscipy" | "frankenscipy_analytic" => {
                if tol.is_some() || maxiter.is_some() || max_step_norm.is_some() {
                    return Err(PyValueError::new_err(
                        "FrankenSciPy has fixed solver controls; omit tol, maxiter and max_step_norm",
                    ));
                }
                None
            }
            _ => {
                return Err(PyValueError::new_err(
                    "candidate backend must be frankennumpy, frankenscipy, frankenscipy_analytic or sparse",
                ));
            }
        };
        let variables = self.system.vars().len();
        let residuals = self.system.residuals().len();
        if backend == "sparse" {
            if !(1..=MAX_SPARSE_NEWTON_VARIABLES).contains(&variables) || residuals != variables {
                return Err(crate::to_value_error(
                    SystemConsumerError::UnsupportedSparseRootShape {
                        variables,
                        residuals,
                    },
                ));
            }
        } else if !(1..=MAX_CONSUMER_VARIABLES).contains(&variables) || residuals != variables {
            return Err(crate::to_value_error(
                SystemConsumerError::UnsupportedShape {
                    variables,
                    residuals,
                },
            ));
        }
        let inputs = input_values_metered(initial, variables, budget.as_deref())?;
        let py = initial.py();
        let proposal = with_evaluation_meter(py, budget.as_deref(), |meter| match &guards {
            Some(guards) if backend == "sparse" => {
                sparse_newton_refine_with_meter(&self.system, &inputs, guards, meter)
            }
            Some(guards) => newton_refine_with_meter(&self.system, &inputs, guards, meter),
            None if backend == "frankenscipy_analytic" => {
                frankenscipy_analytic_system_with_meter(&self.system, &inputs, meter)
            }
            None => frankenscipy_fsolve_system_with_meter(&self.system, &inputs, meter),
        })?;
        // Convert the complete result privately. A cancellation while waiting
        // to reattach the interpreter cannot publish a successful proposal.
        let result =
            (proposal.root, proposal.residual_norm, proposal.iterations).into_pyobject(py)?;
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Generate an approximate least-squares stationary point, not a root.
    /// All residual rows and derivative domains share the same native ledger.
    /// The sparse route retains sparse Jacobian values throughout LSQR; neither
    /// route silently substitutes another solver when its own work refuses.
    #[pyo3(signature = (initial, *, backend="dense", gtol=None, maxiter=None, max_step_norm=None, budget=None))]
    fn solve_least_squares<'py>(
        &self,
        initial: &Bound<'py, PyAny>,
        backend: &str,
        gtol: Option<&Bound<'_, PyAny>>,
        maxiter: Option<&Bound<'_, PyAny>>,
        max_step_norm: Option<&Bound<'_, PyAny>>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let sparse = match backend {
            "dense" => false,
            "sparse" => true,
            _ => {
                return Err(PyValueError::new_err(
                    "least-squares backend must be dense or sparse",
                ));
            }
        };
        let defaults = LeastSquaresOptions::default();
        let shared_guards = donor_newton_guards(None, maxiter, max_step_norm)?;
        let gradient_tolerance =
            donor_float_option(gtol, defaults.gradient_tolerance, "gtol", true)?;
        let options = LeastSquaresOptions {
            max_iterations: shared_guards.max_iterations,
            gradient_tolerance,
            max_step_norm: shared_guards.max_step_norm,
        };
        let variables = self.system.vars().len();
        let residuals = self.system.residuals().len();
        if sparse {
            if !(1..=MAX_SPARSE_LEAST_SQUARES_VARIABLES).contains(&variables)
                || !(1..=MAX_SPARSE_LEAST_SQUARES_RESIDUALS).contains(&residuals)
            {
                return Err(PyValueError::new_err(
                    "sparse least squares requires 1..=256 variables and 1..=4096 residuals",
                ));
            }
        } else if !(1..=MAX_CONSUMER_VARIABLES).contains(&variables)
            || !(variables..=MAX_LEAST_SQUARES_RESIDUALS).contains(&residuals)
        {
            return Err(PyValueError::new_err(
                "least squares requires 1..=16 variables and variable_count..=64 residuals",
            ));
        }
        let inputs = input_values_metered(initial, variables, budget.as_deref())?;
        let py = initial.py();
        let proposal = with_evaluation_meter(py, budget.as_deref(), |meter| {
            if sparse {
                sparse_least_squares_with_meter(&self.system, &inputs, &options, meter)
            } else {
                least_squares_with_meter(&self.system, &inputs, &options, meter)
            }
        })?;
        let termination = match proposal.termination {
            LeastSquaresTermination::GradientTolerance => "gradient_tolerance",
        };
        let values = (
            proposal.point,
            proposal.residuals,
            proposal.cost,
            proposal.gradient_norm,
            proposal.iterations,
            proposal.evaluations,
            termination,
        );
        let result = values.into_pyobject(py)?;
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Encode the checked source, derivatives, proofs and bytecode as bounded JSON bytes.
    /// Receipts, Python objects and execution budgets are not serialized.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let encoded = py
            .detach(|| self.system.to_json_bytes())
            .map_err(Self::transport_error)?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Replay proof admission and bind to the caller's exact expected typed source.
    /// Only immutable built-in bytes are accepted; admission is outside EvaluationBudget.
    #[staticmethod]
    #[pyo3(signature = (data, *, residuals, variables))]
    fn from_bytes(
        data: &Bound<'_, PyAny>,
        residuals: &Bound<'_, PyAny>,
        variables: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let py = data.py();
        if !data.is_exact_instance(&py.get_type::<PyBytes>()) {
            return Err(PyTypeError::new_err(
                "verified program data must be exact built-in bytes",
            ));
        }
        let encoded = data.cast::<PyBytes>()?.as_bytes();
        if encoded.len() > MAX_VERIFIED_ARTIFACT_BYTES {
            return Err(PyValueError::new_err(
                "verified program data exceeds the 16 MiB byte limit",
            ));
        }
        let residuals = extract_expressions(residuals)?;
        let variables = extract_variables(variables)?;
        Self::check_requested_source(&residuals, &variables)?;
        py.detach(|| {
            // Replay the retained certificate; never replace it with fresh
            // derivative generation or accept a serialized verification flag.
            let system =
                VerifiedResidualSystem::from_json_bytes(encoded).map_err(Self::transport_error)?;
            if system.vars() != variables.as_slice() {
                return Err(PyValueError::new_err(
                    "verified program variables do not match the expected typed order",
                ));
            }
            if system.residuals() != residuals.as_slice() {
                return Err(PyValueError::new_err(
                    "verified program residuals do not match the expected exact source",
                ));
            }
            // Expected-source extraction already enforced Python bridge
            // axis, literal, node and depth limits. Check the loaded bytecode
            // limit as well, since native transport permits a wider envelope.
            Self::admit(system)
        })
    }

    /// Independently enclose the original residuals and their derivatives.
    #[pyo3(signature = (domain, precision_digits=32, *, budget=None))]
    fn enclose(
        &self,
        domain: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyResidualJacobianEnclosure> {
        let source = self.system.sparse_jacobian();
        let py = domain.py();
        let domain = exact_box(domain, source.vars.len(), budget.as_deref())?;
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            metered_enclose_residual_jacobian(
                &source.residuals,
                &source.vars,
                &domain,
                precision_digits,
                meter,
            )
        })?;
        let result = PyResidualJacobianEnclosure { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Check an untrusted rational Krawczyk witness against the original source.
    #[pyo3(signature = (domain, center, preconditioner, precision_digits=32, *, budget=None))]
    fn certify_root_box(
        &self,
        domain: &Bound<'_, PyAny>,
        center: &Bound<'_, PyAny>,
        preconditioner: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyRootBoxReceipt> {
        let source = self.system.sparse_jacobian();
        let dimension = source.vars.len();
        let py = domain.py();
        let domain = exact_box(domain, dimension, budget.as_deref())?;
        let center = exact_vector(center, dimension, budget.as_deref())?;
        if container_length(preconditioner, MAX_CERTIFIED_VARIABLES, "preconditioner")? != dimension
        {
            return Err(PyValueError::new_err(
                "preconditioner has the wrong dimension",
            ));
        }
        let mut matrix = exact_buffer(dimension, budget.as_deref())?;
        for index in 0..dimension {
            matrix.push(exact_vector(
                &preconditioner.get_item(index)?,
                dimension,
                budget.as_deref(),
            )?);
        }
        let witness = RootBoxWitness {
            domain,
            center,
            preconditioner: matrix,
            precision_digits,
        };
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            verify_root_box(&source.residuals, &source.vars, &witness, meter)
        })?;
        let result = PyRootBoxReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Discover every nonsingular real root in a finite box, or prove it root-free.
    /// Search proposes a complete cover; independent original-source replay
    /// alone grants the global receipt. Both phases spend the same ledger.
    #[pyo3(signature = (domain, precision_digits=32, max_leaves=512, *, budget=None))]
    fn isolate_roots(
        &self,
        domain: &Bound<'_, PyAny>,
        precision_digits: u32,
        max_leaves: usize,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyRootIsolationReceipt> {
        if !(8..=128).contains(&precision_digits) {
            return Err(PyValueError::new_err(
                "root isolation precision must be between 8 and 128",
            ));
        }
        if !(1..=1024).contains(&max_leaves) {
            return Err(PyValueError::new_err(
                "root isolation max_leaves must be between 1 and 1024",
            ));
        }
        let source = self.system.sparse_jacobian();
        if !(1..=MAX_CERTIFIED_VARIABLES).contains(&source.vars.len())
            || source.residuals.len() != source.vars.len()
        {
            return Err(PyValueError::new_err(
                "root isolation requires a square system with 1..16 variables",
            ));
        }
        let py = domain.py();
        let domain = exact_box(domain, source.vars.len(), budget.as_deref())?;
        let options = RootIsolationOptions {
            precision_digits,
            max_leaves,
        };
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            let witness =
                propose_root_isolation(&source.residuals, &source.vars, &domain, options, meter)?;
            verify_root_isolation(&source.residuals, &source.vars, &domain, &witness, meter)
        })?;
        let result = PyRootIsolationReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Prove a unique unknown root for every value in the full parameter box.
    /// The first residual_count variables are unknowns; all others are parameters.
    #[pyo3(signature = (unknown_domain, parameter_domain, center, preconditioner, precision_digits=32, *, budget=None))]
    fn certify_parametric_root_box(
        &self,
        unknown_domain: &Bound<'_, PyAny>,
        parameter_domain: &Bound<'_, PyAny>,
        center: &Bound<'_, PyAny>,
        preconditioner: &Bound<'_, PyAny>,
        precision_digits: u32,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyParametricRootBoxReceipt> {
        if !(8..=128).contains(&precision_digits) {
            return Err(PyValueError::new_err(
                "parametric root precision must be between 8 and 128",
            ));
        }
        let source = self.system.sparse_jacobian();
        let dimension = source.residuals.len();
        let total = source.vars.len();
        if dimension == 0 || dimension >= total || total > MAX_CERTIFIED_VARIABLES {
            return Err(PyValueError::new_err(
                "parametric roots require one unknown per residual, at least one parameter, and at most 16 total variables",
            ));
        }
        let py = unknown_domain.py();
        let unknown_domain = exact_box(unknown_domain, dimension, budget.as_deref())?;
        let parameter_domain = exact_box(parameter_domain, total - dimension, budget.as_deref())?;
        let center = exact_vector(center, dimension, budget.as_deref())?;
        if container_length(preconditioner, MAX_CERTIFIED_VARIABLES, "preconditioner")? != dimension
        {
            return Err(PyValueError::new_err(
                "preconditioner has the wrong dimension",
            ));
        }
        let mut matrix = exact_buffer(dimension, budget.as_deref())?;
        for index in 0..dimension {
            matrix.push(exact_vector(
                &preconditioner.get_item(index)?,
                dimension,
                budget.as_deref(),
            )?);
        }
        let witness = ParametricRootBoxWitness {
            unknown_domain,
            parameter_domain,
            center,
            preconditioner: matrix,
            precision_digits,
        };
        let (unknowns, parameters) = source.vars.split_at(dimension);
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            verify_parametric_root_box(&source.residuals, unknowns, parameters, &witness, meter)
        })?;
        let result = PyParametricRootBoxReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Search a complete parameter cover, then independently verify its entire partition.
    #[pyo3(signature = (unknown_domain, parameter_domain, precision_digits=32, max_leaves=512, *, budget=None))]
    fn cover_parametric_roots(
        &self,
        unknown_domain: &Bound<'_, PyAny>,
        parameter_domain: &Bound<'_, PyAny>,
        precision_digits: u32,
        max_leaves: usize,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<PyParametricRootCoverReceipt> {
        if !(8..=128).contains(&precision_digits) {
            return Err(PyValueError::new_err(
                "parametric cover precision must be between 8 and 128",
            ));
        }
        if !(1..=1024).contains(&max_leaves) {
            return Err(PyValueError::new_err(
                "parametric cover max_leaves must be between 1 and 1024",
            ));
        }
        let source = self.system.sparse_jacobian();
        let dimension = source.residuals.len();
        let total = source.vars.len();
        if dimension == 0 || dimension >= total || total > MAX_CERTIFIED_VARIABLES {
            return Err(PyValueError::new_err(
                "parametric roots require one unknown per residual, at least one parameter, and at most 16 total variables",
            ));
        }
        let py = unknown_domain.py();
        let unknown_domain = exact_box(unknown_domain, dimension, budget.as_deref())?;
        let parameter_domain = exact_box(parameter_domain, total - dimension, budget.as_deref())?;
        let (unknowns, parameters) = source.vars.split_at(dimension);
        let options = ParametricRootCoverOptions {
            precision_digits,
            max_leaves,
        };
        let receipt = with_certification_meter(py, budget.as_deref(), |meter| {
            let witness = propose_parametric_root_cover(
                &source.residuals,
                unknowns,
                parameters,
                &unknown_domain,
                &parameter_domain,
                options,
                meter,
            )?;
            verify_parametric_root_cover(
                &source.residuals,
                unknowns,
                parameters,
                &unknown_domain,
                &parameter_domain,
                &witness,
                meter,
            )
        })?;
        let result = PyParametricRootCoverReceipt { receipt };
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Return residuals and a row-major nested Jacobian after all values pass.
    #[pyo3(signature = (values, *, budget=None))]
    fn evaluate(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<ResidualValues> {
        self.system
            .dense_jacobian_len()
            .map_err(crate::to_value_error)?;
        let inputs = input_values_metered(values, self.system.num_vars(), budget.as_deref())?;
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            self.evaluate_values(&inputs, meter)
        })
    }

    /// Evaluate every original residual and structural derivative without dense output.
    #[pyo3(signature = (values, *, budget=None))]
    fn evaluate_sparse<'py>(
        &self,
        values: &Bound<'py, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let py = values.py();
        let inputs = input_values_metered(values, self.system.vars().len(), budget.as_deref())?;
        let result = with_evaluation_meter(py, budget.as_deref(), |meter| {
            self.system.evaluate_sparse_with_meter(&inputs, meter)
        })?;
        let result = (result.residuals, result.jacobian).into_pyobject(py)?;
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Evaluate J(x) times a direction, retaining source and derivative domain checks.
    #[pyo3(signature = (values, direction, *, budget=None))]
    fn jacobian_vector_product<'py>(
        &self,
        values: &Bound<'py, PyAny>,
        direction: &Bound<'py, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let py = values.py();
        let size = self.system.vars().len();
        let inputs = input_values_metered(values, size, budget.as_deref())?;
        let direction = input_values_metered(direction, size, budget.as_deref())?;
        let result = with_evaluation_meter(py, budget.as_deref(), |meter| {
            self.system
                .jacobian_vector_product_with_meter(&inputs, &direction, meter)
        })?;
        let result = PyList::new(py, result)?;
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Evaluate J(x) transposed times residual weights without allocating dense rows.
    #[pyo3(signature = (values, weights, *, budget=None))]
    fn jacobian_transpose_vector_product<'py>(
        &self,
        values: &Bound<'py, PyAny>,
        weights: &Bound<'py, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let py = values.py();
        let inputs = input_values_metered(values, self.system.vars().len(), budget.as_deref())?;
        // A rectangular residual system may have more rows than variables.
        // Weight admission follows the residual bound, not the variable bound.
        let weights = input_values_metered_bounded(
            weights,
            self.system.residuals().len(),
            MAX_OUTPUTS,
            budget.as_deref(),
        )?;
        let result = with_evaluation_meter(py, budget.as_deref(), |meter| {
            self.system
                .jacobian_transpose_vector_product_with_meter(&inputs, &weights, meter)
        })?;
        let result = PyList::new(py, result)?;
        with_evaluation_meter_attached(budget.as_deref(), |meter| meter.checkpoint())?;
        Ok(result)
    }

    /// Evaluate every residual/Jacobian pair before publishing a batch.
    #[pyo3(signature = (rows, *, budget=None))]
    fn evaluate_batch(
        &self,
        rows: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<ResidualValues>> {
        self.system
            .dense_jacobian_len()
            .map_err(crate::to_value_error)?;
        let variables = self.system.num_vars();
        let outputs = self.system.num_residuals().saturating_mul(variables + 1);
        let inputs = batch_inputs(rows, variables, outputs, budget.as_deref())?;
        with_evaluation_meter(rows.py(), budget.as_deref(), |meter| {
            let mut results = evaluation_buffer(inputs.len(), meter)?;
            for values in &inputs {
                results.push(self.evaluate_values(values, meter)?);
            }
            Ok::<_, ResidualEvaluationError>(results)
        })
    }

    /// Evaluate residuals without requiring the derivatives to be finite.
    #[pyo3(signature = (values, *, budget=None))]
    fn residuals(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<f64>> {
        let inputs = input_values_metered(values, self.system.num_vars(), budget.as_deref())?;
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            evaluate_programs(self.system.residual_programs(), &inputs, meter)
        })
    }

    /// Evaluate the independently verified symbolic Jacobian in row-major order.
    #[pyo3(signature = (values, *, budget=None))]
    fn jacobian(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.system
            .dense_jacobian_len()
            .map_err(crate::to_value_error)?;
        let inputs = input_values_metered(values, self.system.num_vars(), budget.as_deref())?;
        // Derivatives can erase source-domain restrictions (including a held
        // zero times log(x)). Admit every original residual before publishing
        // a derivative-only result.
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            self.evaluate_values(&inputs, meter)
                .map(|(_, jacobian)| jacobian)
        })
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.system.num_vars()
    }

    #[getter]
    fn residual_count(&self) -> usize {
        self.system.num_residuals()
    }

    /// Canonical structural nonzeros in residual-row and variable-column order.
    #[getter]
    fn jacobian_pattern(&self) -> Vec<(usize, usize)> {
        self.system.jacobian_pattern().to_vec()
    }

    /// Counts symbolic derivative proofs; numerical rounding is not certified.
    #[getter]
    fn proofs_verified(&self) -> usize {
        self.system.receipt().proofs_verified
    }

    #[getter]
    fn nonzero_derivatives(&self) -> usize {
        self.system.receipt().nnz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::{SymbolIdentity, parse};

    #[test]
    fn same_printed_variables_have_independent_slots() {
        let a = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
        let b = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
        let source = Expr::Add(vec![
            Expr::Sym(a.clone()),
            Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(b.clone())]),
        ]);
        let program = PyCompiledFunction::compile(&[source], &[a, b]).unwrap();
        assert_eq!(program.evaluate_values(&[3.0, 7.0]).unwrap(), [17.0]);
    }

    #[test]
    fn constants_and_nonlinear_functions_use_the_native_program() {
        let program = PyCompiledFunction::compile(
            &[parse("x^2 + sin(y)").unwrap(), Expr::from_i64(7)],
            &[Symbol::new("x"), Symbol::new("y")],
        )
        .unwrap();
        assert_eq!(program.evaluate_values(&[3.0, 0.0]).unwrap(), [9.0, 7.0]);
        assert!(program.operation_count > 2);
    }

    #[test]
    fn unmapped_duplicate_and_unsupported_inputs_refuse() {
        let x = Symbol::new("x");
        assert!(PyCompiledFunction::compile(&[parse("x").unwrap()], &[]).is_err());
        assert!(PyCompiledFunction::compile(&[], &[x.clone(), x]).is_err());
        assert!(PyCompiledFunction::compile(&[parse("unknown(1)").unwrap()], &[]).is_err());
    }

    #[test]
    fn dimensions_domain_failures_and_nonfinite_outputs_refuse() {
        let program =
            PyCompiledFunction::compile(&[parse("log(x)").unwrap()], &[Symbol::new("x")]).unwrap();
        assert!(program.evaluate_values(&[]).is_err());
        assert!(program.evaluate_values(&[f64::INFINITY]).is_err());
        assert!(program.evaluate_values(&[-1.0]).is_err());
        assert!(program.evaluate_values(&[0.0]).is_err());
        assert_eq!(program.evaluate_values(&[1.0]).unwrap(), [0.0]);
        let reciprocal =
            PyCompiledFunction::compile(&[parse("1 / log(x)").unwrap()], &[Symbol::new("x")])
                .unwrap();
        // An infinite logarithm cannot be hidden by a subsequent reciprocal.
        assert!(reciprocal.evaluate_values(&[0.0]).is_err());
    }

    #[pyclass(extends = PyExpr)]
    struct SourceSubclass;

    #[test]
    fn borrowed_source_preflight_checks_exact_types_and_original_tree_limits() {
        Python::initialize();
        Python::attach(|py| {
            let source = Py::new(py, PyExpr::from_expr(Expr::from_i64(7))).unwrap();
            PyCompiledFunction::_validate_source(source.bind(py).as_any()).unwrap();
            let subclass = Py::new(
                py,
                PyClassInitializer::from(PyExpr::from_expr(Expr::from_i64(7)))
                    .add_subclass(SourceSubclass),
            )
            .unwrap();
            let error =
                PyCompiledFunction::_validate_source(subclass.bind(py).as_any()).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(PyCompiledFunction::_validate_source(PyList::empty(py).as_any()).is_err());

            let mut deep = Expr::symbol("x");
            for _ in 0..=MAX_COMPILE_DEPTH {
                deep = Expr::Function("sin".into(), vec![deep]);
            }
            let big = fsym_core::BigInt::one() << MAX_LITERAL_BITS as u32;
            let invalid = [
                Expr::Integer(big.clone()),
                Expr::Rational(BigRational::new(&big + 1i64, big)),
                Expr::Add(vec![Expr::from_i64(0); MAX_SOURCE_NODES]),
                deep,
                Expr::symbol("x".repeat(MAX_COMPILE_NAME_BYTES + 1)),
                Expr::Function("λ".repeat(MAX_COMPILE_NAME_BYTES / 2 + 1), vec![]),
            ];
            for source in invalid {
                let source = Py::new(py, PyExpr::from_expr(source)).unwrap();
                let error =
                    PyCompiledFunction::_validate_source(source.bind(py).as_any()).unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
            }

            // Preflight admits a bounded tree; it does not certify compiler support or
            // floating range. Both of these values fail later numerical compilation.
            for source in [
                Expr::Integer(fsym_core::BigInt::one() << (MAX_LITERAL_BITS - 1) as u32),
                Expr::Function("unregistered".into(), vec![Expr::from_i64(1)]),
            ] {
                let wrapped = Py::new(py, PyExpr::from_expr(source.clone())).unwrap();
                PyCompiledFunction::_validate_source(wrapped.bind(py).as_any()).unwrap();
                assert!(PyCompiledFunction::compile(&[source], &[]).is_err());
            }
        });
    }

    #[test]
    fn borrowed_matrix_preflight_bounds_shape_and_all_cells_before_snapshotting() {
        use fsym_matrices::{Matrix, SparseMatrix};
        use std::collections::BTreeMap;

        Python::initialize();
        Python::attach(|py| {
            let check_dense = |inner| {
                let source = Py::new(py, PyMatrix { inner }).unwrap();
                PyCompiledFunction::_validate_matrix_source(source.bind(py).as_any())
            };
            let check_sparse = |inner| {
                let source = Py::new(py, PySparseMatrix { inner }).unwrap();
                PyCompiledFunction::_validate_matrix_source(source.bind(py).as_any())
            };
            check_dense(Matrix::zeros(2, 2).unwrap()).unwrap();
            check_sparse(SparseMatrix::zeros(2, 2)).unwrap();
            for (rows, columns) in [(MAX_OUTPUTS + 1, 0), (0, MAX_OUTPUTS + 1), (65, 64)] {
                assert!(check_dense(Matrix::zeros(rows, columns).unwrap()).is_err());
                assert!(check_sparse(SparseMatrix::zeros(rows, columns)).is_err());
            }
            assert!(check_sparse(SparseMatrix::zeros(usize::MAX, 0)).is_err());
            let invalid_cell = Expr::Integer(fsym_core::BigInt::one() << MAX_LITERAL_BITS as u32);
            assert!(check_dense(Matrix::new(1, 1, vec![invalid_cell.clone()]).unwrap()).is_err());
            assert!(
                check_sparse(
                    SparseMatrix::new(1, 1, BTreeMap::from([((0, 0), invalid_cell)])).unwrap()
                )
                .is_err()
            );

            // Each cell individually fits, but the matrix must share one node limit.
            let cell = Expr::Add(vec![Expr::symbol("x"); MAX_SOURCE_NODES / 2]);
            let wrapped = Py::new(py, PyExpr::from_expr(cell.clone())).unwrap();
            PyCompiledFunction::_validate_source(wrapped.bind(py).as_any()).unwrap();
            assert!(
                check_dense(Matrix::new(1, 2, vec![cell.clone(), cell.clone()]).unwrap()).is_err()
            );
            assert!(
                check_sparse(
                    SparseMatrix::new(
                        1,
                        2,
                        BTreeMap::from([((0, 0), cell.clone()), ((0, 1), cell)]),
                    )
                    .unwrap(),
                )
                .is_err()
            );
            // Omitted sparse zeros are still output cells and consume source nodes.
            let boundary = Expr::Add(vec![Expr::symbol("x"); MAX_SOURCE_NODES - 1]);
            let wrapped = Py::new(py, PyExpr::from_expr(boundary.clone())).unwrap();
            PyCompiledFunction::_validate_source(wrapped.bind(py).as_any()).unwrap();
            assert!(
                check_sparse(
                    SparseMatrix::new(1, 2, BTreeMap::from([((0, 0), boundary)])).unwrap(),
                )
                .is_err()
            );
        });
    }

    fn bridge_module(py: Python<'_>) -> Bound<'_, PyModule> {
        let module = PyModule::new(py, "numeric_bridge_tests").unwrap();
        crate::numeric_budget::register(&module).unwrap();
        module.add_class::<PyCompiledFunction>().unwrap();
        module.add_class::<PyVerifiedResidualSystem>().unwrap();
        module
    }

    fn fixture(py: Python<'_>, source: Expr) -> Py<PyExpr> {
        Py::new(py, PyExpr::from_expr(source)).unwrap()
    }

    fn run_bridge(module: &Bound<'_, PyModule>, source: &str) {
        let namespace = module.dict();
        module
            .py()
            .run(
                std::ffi::CString::new(source).unwrap().as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .unwrap();
    }

    #[test]
    fn actual_python_numeric_entrypoints_share_budgets_and_keep_domain_refusals() {
        Python::initialize();
        Python::attach(|py| {
            let module = bridge_module(py);
            let x = Symbol::new("x");
            module
                .add(
                    "x_binding",
                    Py::new(py, PySymbolBinding::from_symbol(&x)).unwrap(),
                )
                .unwrap();
            module
                .add("source", fixture(py, parse("x + 1").unwrap()))
                .unwrap();
            module
                .add("guarded", fixture(py, parse("1 / log(x)").unwrap()))
                .unwrap();
            run_bridge(
                &module,
                r#"
def raises(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError('expected ' + kind.__name__)

program = CompiledFunction([source], [x_binding])
budget = EvaluationBudget()
assert program.evaluate([2], budget=budget) == [3.0]
first_remaining = budget.remaining_steps
assert program.evaluate_batch([[3], [4]], budget=budget) == [[4.0], [5.0]]
assert budget.remaining_steps < first_remaining < 10_000_000
blocked = EvaluationBudget(max_steps=100, max_memory_bytes=0, max_allocations=10)
raises(EvaluationBudgetExceeded, lambda: program.evaluate([2], budget=blocked))
assert (blocked.remaining_steps, blocked.remaining_memory_bytes, blocked.remaining_allocations) == (100, 0, 10)
raises(EvaluationBudgetExceeded, lambda: program.evaluate([2], budget=EvaluationBudget(max_steps=0)))
budget.cancel()
raises(EvaluationCancelled, lambda: program.evaluate_batch([[1]], budget=budget))
domain = CompiledFunction([guarded], [x_binding])
assert domain.evaluate([2])[0] > 0
for value in (0, -1):
    raises(ValueError, lambda: domain.evaluate([value]))
sentinel = object()
result = sentinel
try:
    result = domain.evaluate_batch([[2], [3], [0]])
except ValueError:
    pass
else:
    raise AssertionError('nonfinite intermediate hidden by reciprocal')
assert result is sentinel
assert program.evaluate([2]) == [3.0]
"#,
            );
        });
    }

    #[test]
    fn actual_python_transport_replays_evidence_and_certifies_the_bound_source() {
        Python::initialize();
        Python::attach(|py| {
            let module = bridge_module(py);
            let source = parse("x^2 - 2").unwrap();
            let x = Symbol::new("x");
            module
                .add(
                    "x_binding",
                    Py::new(py, PySymbolBinding::from_symbol(&x)).unwrap(),
                )
                .unwrap();
            module.add("source", fixture(py, source.clone())).unwrap();
            module
                .add("different", fixture(py, parse("x^2 - 3").unwrap()))
                .unwrap();
            for (name, numerator, denominator) in [
                ("lo", 7, 5),
                ("hi", 3, 2),
                ("center", 29, 20),
                ("inverse", 10, 29),
            ] {
                module
                    .add(
                        name,
                        fixture(
                            py,
                            Expr::Rational(BigRational::new(numerator.into(), denominator.into())),
                        ),
                    )
                    .unwrap();
            }
            run_bridge(
                &module,
                r#"
import json
def raises(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError('expected ' + kind.__name__)

program = VerifiedResidualSystem([source], [x_binding])
data = program.to_bytes()
restored = VerifiedResidualSystem.from_bytes(data, residuals=[source], variables=[x_binding])
assert restored.to_bytes() == data
assert restored.evaluate([1.5]) == ([0.25], [[3.0]])
raises(ValueError, lambda: VerifiedResidualSystem.from_bytes(data, residuals=[different], variables=[x_binding]))
document = json.loads(data)
document['jacobian']['entries'][0]['compiled_deriv']['ops'] = [{'LoadConst': 0.0}]
changed = json.dumps(document).encode()
raises(ValueError, lambda: VerifiedResidualSystem.from_bytes(changed, residuals=[source], variables=[x_binding]))
assert restored.evaluate([1.5]) == ([0.25], [[3.0]])
budget = EvaluationBudget(max_steps=2**45, max_memory_bytes=2**40, max_allocations=2**30)
receipt = restored.certify_root_box([(lo, hi)], [center], [[inverse]], 16, budget=budget)
assert receipt.precision_digits == 16
assert receipt.variables[0].name == 'x'
assert budget.remaining_steps < 2**45
budget.cancel()
raises(EvaluationCancelled, lambda: restored.enclose([(lo, hi)], 16, budget=budget))
assert restored.to_bytes() == data
"#,
            );
            let receipt = module.getattr("receipt").unwrap();
            let receipt = receipt.extract::<PyRef<'_, PyRootBoxReceipt>>().unwrap();
            assert_eq!(receipt.receipt.residuals(), &[source]);
            assert_eq!(receipt.receipt.variables(), &[x]);
            assert!(receipt.receipt.contraction_bound() < &BigRational::from_integer(1.into()));
            let enclosure = &receipt.receipt.root_enclosure()[0];
            assert!(&enclosure.lower() * &enclosure.lower() <= BigRational::from_integer(2.into()));
            assert!(&enclosure.upper() * &enclosure.upper() >= BigRational::from_integer(2.into()));
        });
    }
}
