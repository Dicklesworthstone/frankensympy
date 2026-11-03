//! Reusable typed numerical programs for the Python bridge.
//!
//! These bounded synchronous programs evaluate finite real binary64 values.
//! Source/bytecode checks and symbolic derivative proofs do not certify the
//! rounding error of numerical outputs. No Python callback or source-code
//! evaluation is used during compilation or execution.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::{PyExpr, PySymbol};
use crate::numeric_budget::{
    PyEvaluationBudget, evaluation_buffer, with_evaluation_meter, with_evaluation_meter_attached,
};
use fsym_budget::{BudgetMeter, Dimension};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, EvalError, MAX_COMPILE_DEPTH};
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_core::{
    BigRational, Expr, RealBall, ResidualJacobianEnclosure, RootBoxReceipt, RootBoxWitness, Symbol,
    metered_enclose_residual_jacobian, verify_root_box,
};
use fsym_runtime::{Budget, BudgetLimits};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt, PyList, PyTuple};
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

fn exact_rational(value: &Bound<'_, PyAny>) -> PyResult<BigRational> {
    if !value.is_exact_instance(&value.py().get_type::<PyExpr>()) {
        return Err(PyTypeError::new_err(
            "exact bounds require native rational Expr objects",
        ));
    }
    let value = value.extract::<PyRef<'_, PyExpr>>()?;
    match &value.inner {
        Expr::Integer(n) if n.bits() <= MAX_LITERAL_BITS => {
            Ok(BigRational::from_integer(n.clone()))
        }
        Expr::Rational(q) if q.numer().bits().max(q.denom().bits()) <= MAX_LITERAL_BITS => {
            Ok(q.clone())
        }
        _ => Err(PyValueError::new_err(
            "exact bounds must be rational literals of at most 4096 bits",
        )),
    }
}

fn exact_vector(values: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<BigRational>> {
    let values = items(values, MAX_CERTIFIED_VARIABLES, "exact vector")?;
    if values.len() != expected {
        return Err(PyValueError::new_err(
            "exact vector has the wrong dimension",
        ));
    }
    values.iter().map(exact_rational).collect()
}

fn exact_box(values: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<RealBall>> {
    let coordinates = items(values, MAX_CERTIFIED_VARIABLES, "box")?;
    if coordinates.len() != expected {
        return Err(PyValueError::new_err("box has the wrong dimension"));
    }
    coordinates
        .iter()
        .map(|coordinate| {
            let endpoints = items(coordinate, 2, "coordinate bounds")?;
            if endpoints.len() != 2 {
                return Err(PyValueError::new_err(
                    "coordinate bounds require two endpoints",
                ));
            }
            let lo = exact_rational(&endpoints[0])?;
            let hi = exact_rational(&endpoints[1])?;
            if lo > hi {
                return Err(PyValueError::new_err("box lower bound exceeds upper bound"));
            }
            let two = BigRational::from_integer(2.into());
            RealBall::new((&lo + &hi) / &two, (hi - lo) / two).map_err(crate::to_value_error)
        })
        .collect()
}

fn certificate_budget() -> Budget {
    // Bounded synchronous work. Core also enforces its own cumulative limits;
    // this is not a claim of supervised runtime cancellation through Python.
    Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0))
}

fn rational_expr(value: &BigRational) -> PyExpr {
    PyExpr::from_expr(if value.is_integer() {
        Expr::Integer(value.to_integer())
    } else {
        Expr::Rational(value.clone())
    })
}

fn ball_bounds(values: &[RealBall]) -> Vec<(PyExpr, PyExpr)> {
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

pub(crate) fn input_values(values: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<f64>> {
    let py = values.py();
    let actual = container_length(values, MAX_VARIABLES, "numeric inputs")?;
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
        .map_err(|_| crate::to_value_error(EvalError::AllocationFailure))?;
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
    let actual = container_length(values, MAX_VARIABLES, "numeric inputs")?;
    if actual != expected {
        return Err(PyTypeError::new_err(format!(
            "expected {expected} numeric inputs, got {actual}"
        )));
    }
    charge_input_storage(budget, expected, 1, false)?;
    input_values(values, expected)
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

/// Independently verified symbolic Jacobian with finite binary64 evaluation.
#[pyclass(name = "VerifiedResidualSystem", module = "fsym_python", frozen)]
pub struct PyVerifiedResidualSystem {
    system: VerifiedResidualSystem,
}

impl PyVerifiedResidualSystem {
    fn evaluate_values(
        &self,
        inputs: &[f64],
        meter: &mut impl BudgetMeter,
    ) -> Result<ResidualValues, EvalError> {
        let compiled = self.system.compiled();
        let residuals = evaluate_programs(&compiled.compiled_residuals, inputs, meter)?;
        let jacobian = evaluate_programs(&compiled.compiled_jacobian, inputs, meter)?;
        let rows = jacobian_rows(&jacobian, compiled.num_residuals, compiled.num_vars, meter)?;
        Ok((residuals, rows))
    }
}

#[pymethods]
impl PyVerifiedResidualSystem {
    #[new]
    fn new(residuals: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Self> {
        let residuals = extract_expressions(residuals)?;
        let variables = extract_variables(variables)?;
        variable_map(&variables)?;
        let mut source_nodes = 0;
        for residual in &residuals {
            check_source(residual, &mut source_nodes)?;
        }
        if source_nodes.saturating_mul(variables.len().max(1)) > MAX_TOTAL_OPS {
            return Err(PyValueError::new_err(
                "verified system exceeds the differentiation work limit",
            ));
        }
        let system = VerifiedResidualSystem::try_compile(&residuals, &variables)
            .map_err(crate::to_value_error)?;
        let compiled = system.compiled();
        let operations: usize = compiled
            .compiled_residuals
            .iter()
            .chain(&compiled.compiled_jacobian)
            .map(|program| program.ops().len())
            .sum();
        if operations > MAX_TOTAL_OPS {
            return Err(PyValueError::new_err(
                "verified numeric system exceeds the total operation limit",
            ));
        }
        Ok(Self { system })
    }

    /// Independently enclose the original residuals and their derivatives.
    #[pyo3(signature = (domain, precision_digits=32))]
    fn enclose(
        &self,
        domain: &Bound<'_, PyAny>,
        precision_digits: u32,
    ) -> PyResult<PyResidualJacobianEnclosure> {
        let source = self.system.sparse_jacobian();
        let domain = exact_box(domain, source.vars.len())?;
        let receipt = metered_enclose_residual_jacobian(
            &source.residuals,
            &source.vars,
            &domain,
            precision_digits,
            &mut certificate_budget(),
        )
        .map_err(crate::to_value_error)?;
        Ok(PyResidualJacobianEnclosure { receipt })
    }

    /// Check an untrusted rational Krawczyk witness against the original source.
    #[pyo3(signature = (domain, center, preconditioner, precision_digits=32))]
    fn certify_root_box(
        &self,
        domain: &Bound<'_, PyAny>,
        center: &Bound<'_, PyAny>,
        preconditioner: &Bound<'_, PyAny>,
        precision_digits: u32,
    ) -> PyResult<PyRootBoxReceipt> {
        let source = self.system.sparse_jacobian();
        let dimension = source.vars.len();
        let domain = exact_box(domain, dimension)?;
        let center = exact_vector(center, dimension)?;
        let rows = items(preconditioner, MAX_CERTIFIED_VARIABLES, "preconditioner")?;
        if rows.len() != dimension {
            return Err(PyValueError::new_err(
                "preconditioner has the wrong dimension",
            ));
        }
        let preconditioner = rows
            .iter()
            .map(|row| exact_vector(row, dimension))
            .collect::<PyResult<Vec<_>>>()?;
        let witness = RootBoxWitness {
            domain,
            center,
            preconditioner,
            precision_digits,
        };
        let receipt = verify_root_box(
            &source.residuals,
            &source.vars,
            &witness,
            &mut certificate_budget(),
        )
        .map_err(crate::to_value_error)?;
        Ok(PyRootBoxReceipt { receipt })
    }

    /// Return residuals and a row-major nested Jacobian after all values pass.
    #[pyo3(signature = (values, *, budget=None))]
    fn evaluate(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<ResidualValues> {
        let compiled = self.system.compiled();
        let inputs = input_values_metered(values, compiled.num_vars, budget.as_deref())?;
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            self.evaluate_values(&inputs, meter)
        })
    }

    /// Evaluate every residual/Jacobian pair before publishing a batch.
    #[pyo3(signature = (rows, *, budget=None))]
    fn evaluate_batch(
        &self,
        rows: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<ResidualValues>> {
        let compiled = self.system.compiled();
        let outputs = compiled.num_residuals.saturating_mul(compiled.num_vars + 1);
        let inputs = batch_inputs(rows, compiled.num_vars, outputs, budget.as_deref())?;
        with_evaluation_meter(rows.py(), budget.as_deref(), |meter| {
            let mut results = evaluation_buffer(inputs.len(), meter)?;
            for values in &inputs {
                results.push(self.evaluate_values(values, meter)?);
            }
            Ok::<_, EvalError>(results)
        })
    }

    /// Evaluate residuals without requiring the derivatives to be finite.
    #[pyo3(signature = (values, *, budget=None))]
    fn residuals(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<f64>> {
        let compiled = self.system.compiled();
        let inputs = input_values_metered(values, compiled.num_vars, budget.as_deref())?;
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            evaluate_programs(&compiled.compiled_residuals, &inputs, meter)
        })
    }

    /// Evaluate the independently verified symbolic Jacobian in row-major order.
    #[pyo3(signature = (values, *, budget=None))]
    fn jacobian(
        &self,
        values: &Bound<'_, PyAny>,
        budget: Option<PyRef<'_, PyEvaluationBudget>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let compiled = self.system.compiled();
        let inputs = input_values_metered(values, compiled.num_vars, budget.as_deref())?;
        // Derivatives can erase source-domain restrictions (including a held
        // zero times log(x)). Admit every original residual before publishing
        // a derivative-only result.
        with_evaluation_meter(values.py(), budget.as_deref(), |meter| {
            evaluate_programs(&compiled.compiled_residuals, &inputs, meter)?;
            let flat = evaluate_programs(&compiled.compiled_jacobian, &inputs, meter)?;
            jacobian_rows(&flat, compiled.num_residuals, compiled.num_vars, meter)
        })
    }

    #[getter]
    fn variable_count(&self) -> usize {
        self.system.compiled().num_vars
    }

    #[getter]
    fn residual_count(&self) -> usize {
        self.system.compiled().num_residuals
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
}
