//! Reusable typed numerical programs for the Python bridge.
//!
//! These bounded synchronous programs evaluate finite real binary64 values.
//! Source/bytecode checks and symbolic derivative proofs do not certify the
//! rounding error of numerical outputs. No Python callback or source-code
//! evaluation is used during compilation or execution.

#![forbid(unsafe_code)]

use crate::binding::PySymbolBinding;
use crate::expr::{PyExpr, PySymbol};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, MAX_COMPILE_DEPTH};
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_core::{Expr, Symbol};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt, PyList, PyTuple};
use std::collections::HashMap;

const MAX_VARIABLES: usize = 256;
const MAX_OUTPUTS: usize = 4096;
const MAX_TOTAL_OPS: usize = 65_536;
const MAX_SOURCE_NODES: usize = 65_536;
const MAX_LITERAL_BITS: u64 = 4096;

fn items<'py>(
    values: &Bound<'py, PyAny>,
    maximum: usize,
    name: &str,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
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
    (0..length).map(|index| values.get_item(index)).collect()
}

fn extract_variables(values: &Bound<'_, PyAny>) -> PyResult<Vec<Symbol>> {
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
fn check_source(expr: &Expr, nodes: &mut usize) -> PyResult<()> {
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

fn input_values(values: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<f64>> {
    let py = values.py();
    let values = items(values, MAX_VARIABLES, "numeric inputs")?;
    if values.len() != expected {
        return Err(PyTypeError::new_err(format!(
            "expected {expected} numeric inputs, got {}",
            values.len()
        )));
    }
    values
        .iter()
        .map(|value| {
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
            Ok(value)
        })
        .collect()
}

fn finite_results(values: Vec<f64>) -> PyResult<Vec<f64>> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "numerical evaluation left the finite real domain or overflowed",
        ));
    }
    Ok(values)
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

    fn evaluate_values(&self, values: &[f64]) -> PyResult<Vec<f64>> {
        if values.len() != self.variable_count || values.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "invalid numeric input dimension or nonfinite value",
            ));
        }
        let result = self
            .programs
            .iter()
            .map(|program| {
                program
                    .try_eval_finite(values)
                    .map_err(crate::to_value_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        finite_results(result)
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
    fn evaluate(&self, values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        self.evaluate_values(&input_values(values, self.variable_count)?)
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

    /// Return residuals and a row-major nested Jacobian after all values pass.
    fn evaluate(&self, values: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let compiled = self.system.compiled();
        let values = input_values(values, compiled.num_vars)?;
        let residuals = compiled
            .compiled_residuals
            .iter()
            .map(|program| {
                program
                    .try_eval_finite(&values)
                    .map_err(crate::to_value_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let jacobian = compiled
            .compiled_jacobian
            .iter()
            .map(|program| {
                program
                    .try_eval_finite(&values)
                    .map_err(crate::to_value_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let residuals = finite_results(residuals)?;
        let jacobian = finite_results(jacobian)?;
        let rows = (0..compiled.num_residuals)
            .map(|row| jacobian[row * compiled.num_vars..(row + 1) * compiled.num_vars].to_vec())
            .collect();
        Ok((residuals, rows))
    }

    /// Evaluate residuals without requiring the derivatives to be finite.
    fn residuals(&self, values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let compiled = self.system.compiled();
        let values = input_values(values, compiled.num_vars)?;
        compiled
            .compiled_residuals
            .iter()
            .map(|program| {
                program
                    .try_eval_finite(&values)
                    .map_err(crate::to_value_error)
            })
            .collect()
    }

    /// Evaluate the independently verified symbolic Jacobian in row-major order.
    fn jacobian(&self, values: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let compiled = self.system.compiled();
        let values = input_values(values, compiled.num_vars)?;
        let flat = compiled
            .compiled_jacobian
            .iter()
            .map(|program| {
                program
                    .try_eval_finite(&values)
                    .map_err(crate::to_value_error)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((0..compiled.num_residuals)
            .map(|row| flat[row * compiled.num_vars..(row + 1) * compiled.num_vars].to_vec())
            .collect())
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
