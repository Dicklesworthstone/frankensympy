//! Fast symbolic-to-numeric compilation for residual systems and symbolically differentiated
//! Jacobians (WS12).

#![forbid(unsafe_code)]

use crate::diff;
use fsym_core::{Constant, Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Maximum recursion depth allowed during expression compilation to avoid stack overflow.
pub const MAX_COMPILE_DEPTH: usize = 128;

/// Maximum number of compiled bytecode operations allowed for a single expression.
pub const MAX_COMPILE_OPS: usize = 8192;

/// Errors emitted during symbolic-to-numeric expression and system compilation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CompileError {
    #[error("Variable list contains the duplicate symbol {0}")]
    DuplicateVariable(Symbol),
    #[error("Unmapped variable in expression: {0}")]
    UnmappedVariable(Symbol),
    #[error("Unsupported expression variant for compilation: {0}")]
    UnsupportedExpression(String),
    #[error("Unsupported or non-numeric power exponent: {0}")]
    UnsupportedExponent(String),
    #[error("Unsupported elementary function `{0}` or invalid arity")]
    UnsupportedFunction(String),
    #[error("Numeric conversion error or non-finite literal: {0}")]
    NumericConversion(String),
    #[error("Expression nesting depth {depth} exceeds maximum compile limit {max}")]
    DepthLimitExceeded { depth: usize, max: usize },
    #[error("Compiled bytecode size {ops} exceeds maximum operation limit {max}")]
    OpLimitExceeded { ops: usize, max: usize },
    #[error("Residual-system dimensions overflow the platform size type")]
    DimensionOverflow,
    #[error("Allocation failed during symbolic-to-numeric compilation")]
    AllocationFailure,
}

/// Errors emitted during compiled numerical evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EvalError {
    #[error("Operand stack underflow during bytecode evaluation")]
    StackUnderflow,
    #[error("Operand stack had {0} remaining elements after evaluation (expected 1)")]
    StackRemaining(usize),
    #[error("Variable index {index} out of bounds for input slice of length {slice_len}")]
    VariableOutOfBounds { index: usize, slice_len: usize },
    #[error("Residual buffer length mismatch: expected {expected}, got {actual}")]
    ResidualBufferMismatch { expected: usize, actual: usize },
    #[error("Jacobian buffer length mismatch: expected {expected}, got {actual}")]
    JacobianBufferMismatch { expected: usize, actual: usize },
    #[error("Compiled {lane} expression count mismatch: expected {expected}, got {actual}")]
    CompiledExpressionCountMismatch {
        lane: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Compiled bytecode size {ops} exceeds maximum operation limit {max}")]
    OpLimitExceeded { ops: usize, max: usize },
    #[error("Residual-system dimensions overflow the platform size type")]
    DimensionOverflow,
    #[error("Allocation failed during compiled numerical evaluation")]
    AllocationFailure,
}

/// Fast compiled expression evaluator targeting numerical arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompiledOp {
    LoadConst(f64),
    LoadVar(usize),
    Add(usize),
    Mul(usize),
    Neg,
    Pow(f64),
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Sqrt,
    Abs,
}

/// Linear sequence of operations evaluating an expression to an f64 value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledExpr {
    pub ops: Vec<CompiledOp>,
}

impl CompiledExpr {
    /// Compiles a symbolic expression into bytecode given a variable index lookup map.
    /// Fails closed if any unmapped symbol, unsupported function, or complexity limit is reached.
    pub fn try_compile(
        expr: &Expr,
        var_map: &HashMap<Symbol, usize>,
    ) -> Result<Self, CompileError> {
        let mut ops = Vec::new();
        Self::compile_recursive(expr, var_map, &mut ops, 0)?;
        Ok(Self { ops })
    }

    /// Convenience infallible compiler that panics on malformed or unmapped expressions.
    pub fn compile(expr: &Expr, var_map: &HashMap<Symbol, usize>) -> Self {
        Self::try_compile(expr, var_map).expect("expression compilation failed")
    }

    fn compile_recursive(
        expr: &Expr,
        var_map: &HashMap<Symbol, usize>,
        ops: &mut Vec<CompiledOp>,
        depth: usize,
    ) -> Result<(), CompileError> {
        if depth > MAX_COMPILE_DEPTH {
            return Err(CompileError::DepthLimitExceeded {
                depth,
                max: MAX_COMPILE_DEPTH,
            });
        }
        if ops.len() >= MAX_COMPILE_OPS {
            return Err(CompileError::OpLimitExceeded {
                ops: ops.len().saturating_add(1),
                max: MAX_COMPILE_OPS,
            });
        }

        match expr {
            Expr::Integer(n) => {
                let s = n.to_string();
                let val = s
                    .parse::<f64>()
                    .map_err(|_| CompileError::NumericConversion(s.clone()))?;
                if !val.is_finite() {
                    return Err(CompileError::NumericConversion(s));
                }
                Self::push_op(ops, CompiledOp::LoadConst(val))?;
            }
            Expr::Rational(r) => {
                let numer_s = r.numer().to_string();
                let denom_s = r.denom().to_string();
                let numer = numer_s
                    .parse::<f64>()
                    .map_err(|_| CompileError::NumericConversion(numer_s))?;
                let denom = denom_s
                    .parse::<f64>()
                    .map_err(|_| CompileError::NumericConversion(denom_s))?;
                if denom == 0.0 {
                    return Err(CompileError::NumericConversion("division by zero".into()));
                }
                let val = numer / denom;
                if !val.is_finite() {
                    return Err(CompileError::NumericConversion(format!("{numer}/{denom}")));
                }
                Self::push_op(ops, CompiledOp::LoadConst(val))?;
            }
            Expr::Const(Constant::Pi) => {
                Self::push_op(ops, CompiledOp::LoadConst(std::f64::consts::PI))?
            }
            Expr::Const(Constant::E) => {
                Self::push_op(ops, CompiledOp::LoadConst(std::f64::consts::E))?
            }
            Expr::Const(Constant::Infinity) => {
                Self::push_op(ops, CompiledOp::LoadConst(f64::INFINITY))?
            }
            Expr::Const(Constant::NegativeInfinity) => {
                Self::push_op(ops, CompiledOp::LoadConst(f64::NEG_INFINITY))?
            }
            Expr::Const(Constant::ComplexInfinity)
            | Expr::Const(Constant::I)
            | Expr::Const(Constant::NaN) => {
                return Err(CompileError::UnsupportedExpression(format!("{expr:?}")));
            }
            Expr::Sym(s) => {
                if let Some(&idx) = var_map.get(s) {
                    Self::push_op(ops, CompiledOp::LoadVar(idx))?;
                } else {
                    return Err(CompileError::UnmappedVariable(s.clone()));
                }
            }
            Expr::Add(terms) => {
                if terms.is_empty() {
                    Self::push_op(ops, CompiledOp::LoadConst(0.0))?;
                } else {
                    for t in terms {
                        Self::compile_recursive(t, var_map, ops, depth + 1)?;
                    }
                    Self::push_op(ops, CompiledOp::Add(terms.len()))?;
                }
            }
            Expr::Mul(factors) => {
                if factors.is_empty() {
                    Self::push_op(ops, CompiledOp::LoadConst(1.0))?;
                } else {
                    for f in factors {
                        Self::compile_recursive(f, var_map, ops, depth + 1)?;
                    }
                    Self::push_op(ops, CompiledOp::Mul(factors.len()))?;
                }
            }
            Expr::Pow(base, exp) => {
                Self::compile_recursive(base, var_map, ops, depth + 1)?;
                match exp.as_ref() {
                    Expr::Integer(n) => {
                        let p = n
                            .to_string()
                            .parse::<f64>()
                            .map_err(|_| CompileError::UnsupportedExponent(n.to_string()))?;
                        if !p.is_finite() {
                            return Err(CompileError::UnsupportedExponent(n.to_string()));
                        }
                        Self::push_op(ops, CompiledOp::Pow(p))?;
                    }
                    Expr::Rational(r) => {
                        let numer = r
                            .numer()
                            .to_string()
                            .parse::<f64>()
                            .map_err(|_| CompileError::UnsupportedExponent(r.to_string()))?;
                        let denom = r
                            .denom()
                            .to_string()
                            .parse::<f64>()
                            .map_err(|_| CompileError::UnsupportedExponent(r.to_string()))?;
                        if denom == 0.0 {
                            return Err(CompileError::UnsupportedExponent("0 denominator".into()));
                        }
                        let exponent = numer / denom;
                        if !exponent.is_finite() {
                            return Err(CompileError::UnsupportedExponent(r.to_string()));
                        }
                        Self::push_op(ops, CompiledOp::Pow(exponent))?;
                    }
                    other => {
                        return Err(CompileError::UnsupportedExponent(format!("{other}")));
                    }
                }
            }
            Expr::Function(name, args) => {
                if args.len() != 1 {
                    return Err(CompileError::UnsupportedFunction(format!(
                        "{name} with {} args",
                        args.len()
                    )));
                }
                Self::compile_recursive(&args[0], var_map, ops, depth + 1)?;
                match name.as_str() {
                    "sin" => Self::push_op(ops, CompiledOp::Sin)?,
                    "cos" => Self::push_op(ops, CompiledOp::Cos)?,
                    "tan" => Self::push_op(ops, CompiledOp::Tan)?,
                    "sinh" => Self::push_op(ops, CompiledOp::Sinh)?,
                    "cosh" => Self::push_op(ops, CompiledOp::Cosh)?,
                    "tanh" => Self::push_op(ops, CompiledOp::Tanh)?,
                    "exp" => Self::push_op(ops, CompiledOp::Exp)?,
                    "ln" | "log" => Self::push_op(ops, CompiledOp::Ln)?,
                    "sqrt" => Self::push_op(ops, CompiledOp::Sqrt)?,
                    "abs" => Self::push_op(ops, CompiledOp::Abs)?,
                    other => {
                        return Err(CompileError::UnsupportedFunction(other.to_string()));
                    }
                }
            }
        }
        Ok(())
    }

    fn push_op(ops: &mut Vec<CompiledOp>, op: CompiledOp) -> Result<(), CompileError> {
        let attempted = ops
            .len()
            .checked_add(1)
            .ok_or(CompileError::OpLimitExceeded {
                ops: usize::MAX,
                max: MAX_COMPILE_OPS,
            })?;
        if attempted > MAX_COMPILE_OPS {
            return Err(CompileError::OpLimitExceeded {
                ops: attempted,
                max: MAX_COMPILE_OPS,
            });
        }
        ops.try_reserve(1)
            .map_err(|_| CompileError::AllocationFailure)?;
        ops.push(op);
        Ok(())
    }

    /// Evaluates the compiled expression using a verified operand stack with bounds checks.
    pub fn try_eval(&self, vars: &[f64]) -> Result<f64, EvalError> {
        if self.ops.len() > MAX_COMPILE_OPS {
            return Err(EvalError::OpLimitExceeded {
                ops: self.ops.len(),
                max: MAX_COMPILE_OPS,
            });
        }
        let mut stack = Vec::new();
        stack
            .try_reserve_exact(self.ops.len())
            .map_err(|_| EvalError::AllocationFailure)?;
        for op in &self.ops {
            match op {
                CompiledOp::LoadConst(c) => stack.push(*c),
                CompiledOp::LoadVar(idx) => {
                    if let Some(&v) = vars.get(*idx) {
                        stack.push(v);
                    } else {
                        return Err(EvalError::VariableOutOfBounds {
                            index: *idx,
                            slice_len: vars.len(),
                        });
                    }
                }
                CompiledOp::Add(count) => {
                    let mut sum = 0.0;
                    for _ in 0..*count {
                        sum += stack.pop().ok_or(EvalError::StackUnderflow)?;
                    }
                    stack.push(sum);
                }
                CompiledOp::Mul(count) => {
                    let mut prod = 1.0;
                    for _ in 0..*count {
                        prod *= stack.pop().ok_or(EvalError::StackUnderflow)?;
                    }
                    stack.push(prod);
                }
                CompiledOp::Neg => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(-v);
                }
                CompiledOp::Pow(p) => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.powf(*p));
                }
                CompiledOp::Sin => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.sin());
                }
                CompiledOp::Cos => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.cos());
                }
                CompiledOp::Tan => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.tan());
                }
                CompiledOp::Sinh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.sinh());
                }
                CompiledOp::Cosh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.cosh());
                }
                CompiledOp::Tanh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.tanh());
                }
                CompiledOp::Exp => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.exp());
                }
                CompiledOp::Ln => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.ln());
                }
                CompiledOp::Sqrt => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.sqrt());
                }
                CompiledOp::Abs => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.abs());
                }
            }
        }
        if stack.len() != 1 {
            return Err(EvalError::StackRemaining(stack.len()));
        }
        stack.pop().ok_or(EvalError::StackUnderflow)
    }

    /// Evaluates the compiled expression, returning `f64::NAN` on failure.
    pub fn eval(&self, vars: &[f64]) -> f64 {
        self.try_eval(vars).unwrap_or(f64::NAN)
    }
}

/// Compiled multi-dimensional residual system and symbolically differentiated Jacobian matrix.
#[derive(Debug, Clone)]
pub struct CompiledResidualSystem {
    pub vars: Vec<Symbol>,
    pub num_residuals: usize,
    pub num_vars: usize,
    pub compiled_residuals: Vec<CompiledExpr>,
    pub compiled_jacobian: Vec<CompiledExpr>,
}

impl CompiledResidualSystem {
    /// Compiles a system of equations $\mathbf{f}(\mathbf{x}) = \mathbf{0}$ and its symbolic
    /// Jacobian.
    /// Fails closed if any expression contains unmapped symbols or unsupported functions.
    pub fn try_compile(exprs: &[Expr], vars: &[Symbol]) -> Result<Self, CompileError> {
        let mut var_map = HashMap::new();
        var_map
            .try_reserve(vars.len())
            .map_err(|_| CompileError::AllocationFailure)?;
        for (i, v) in vars.iter().enumerate() {
            if var_map.insert(v.clone(), i).is_some() {
                return Err(CompileError::DuplicateVariable(v.clone()));
            }
        }

        let num_residuals = exprs.len();
        let num_vars = vars.len();
        let jacobian_len = num_residuals
            .checked_mul(num_vars)
            .ok_or(CompileError::DimensionOverflow)?;

        let mut compiled_residuals = Vec::new();
        compiled_residuals
            .try_reserve_exact(num_residuals)
            .map_err(|_| CompileError::AllocationFailure)?;
        for e in exprs {
            compiled_residuals.push(CompiledExpr::try_compile(e, &var_map)?);
        }

        let mut compiled_jacobian = Vec::new();
        compiled_jacobian
            .try_reserve_exact(jacobian_len)
            .map_err(|_| CompileError::AllocationFailure)?;
        for e in exprs {
            for v in vars {
                let d = diff(e, v);
                compiled_jacobian.push(CompiledExpr::try_compile(&d, &var_map)?);
            }
        }

        Ok(Self {
            vars: vars.to_vec(),
            num_residuals,
            num_vars,
            compiled_residuals,
            compiled_jacobian,
        })
    }

    /// Convenience infallible compiler that panics on compile error.
    pub fn compile(exprs: &[Expr], vars: &[Symbol]) -> Self {
        Self::try_compile(exprs, vars).expect("residual system compilation failed")
    }

    /// Evaluates the residual vector $\mathbf{f}(\mathbf{x})$ in-place with buffer length validation.
    pub fn try_eval_residuals(&self, x: &[f64], out_res: &mut [f64]) -> Result<(), EvalError> {
        if out_res.len() != self.num_residuals {
            return Err(EvalError::ResidualBufferMismatch {
                expected: self.num_residuals,
                actual: out_res.len(),
            });
        }
        let evaluated =
            Self::evaluate_batch(&self.compiled_residuals, x, self.num_residuals, "residual")?;
        out_res.copy_from_slice(&evaluated);
        Ok(())
    }

    /// Infallible residual evaluator for backward compatibility.
    pub fn eval_residuals(&self, x: &[f64], out_res: &mut [f64]) {
        self.try_eval_residuals(x, out_res)
            .expect("residual evaluation failed");
    }

    /// Evaluates the flat Jacobian matrix $J_{i, j}$ in-place (row-major: $i \times n + j$).
    pub fn try_eval_jacobian(&self, x: &[f64], out_jac: &mut [f64]) -> Result<(), EvalError> {
        let expected = self
            .num_residuals
            .checked_mul(self.num_vars)
            .ok_or(EvalError::DimensionOverflow)?;
        if out_jac.len() != expected {
            return Err(EvalError::JacobianBufferMismatch {
                expected,
                actual: out_jac.len(),
            });
        }
        let evaluated = Self::evaluate_batch(&self.compiled_jacobian, x, expected, "Jacobian")?;
        out_jac.copy_from_slice(&evaluated);
        Ok(())
    }

    /// Infallible Jacobian evaluator for backward compatibility.
    pub fn eval_jacobian(&self, x: &[f64], out_jac: &mut [f64]) {
        self.try_eval_jacobian(x, out_jac)
            .expect("jacobian evaluation failed");
    }

    /// Evaluates both residuals and Jacobian simultaneously.
    pub fn try_eval_system(
        &self,
        x: &[f64],
        out_res: &mut [f64],
        out_jac: &mut [f64],
    ) -> Result<(), EvalError> {
        if out_res.len() != self.num_residuals {
            return Err(EvalError::ResidualBufferMismatch {
                expected: self.num_residuals,
                actual: out_res.len(),
            });
        }
        let jacobian_len = self
            .num_residuals
            .checked_mul(self.num_vars)
            .ok_or(EvalError::DimensionOverflow)?;
        if out_jac.len() != jacobian_len {
            return Err(EvalError::JacobianBufferMismatch {
                expected: jacobian_len,
                actual: out_jac.len(),
            });
        }
        let residuals =
            Self::evaluate_batch(&self.compiled_residuals, x, self.num_residuals, "residual")?;
        let jacobian = Self::evaluate_batch(&self.compiled_jacobian, x, jacobian_len, "Jacobian")?;
        out_res.copy_from_slice(&residuals);
        out_jac.copy_from_slice(&jacobian);
        Ok(())
    }

    /// Infallible system evaluator for backward compatibility.
    pub fn eval_system(&self, x: &[f64], out_res: &mut [f64], out_jac: &mut [f64]) {
        self.try_eval_system(x, out_res, out_jac)
            .expect("system evaluation failed");
    }

    /// Runs a fail-closed approximate consistency check against central finite differences.
    ///
    /// This diagnostic is neither an exact verifier nor mathematical evidence for the compiled
    /// Jacobian. It rejects non-finite inputs, outputs, steps, and tolerances.
    pub fn check_with_finite_differences(&self, x: &[f64], eps: f64, tol: f64) -> bool {
        if x.len() != self.num_vars
            || self.vars.len() != self.num_vars
            || !eps.is_finite()
            || eps <= 0.0
            || !tol.is_finite()
            || tol < 0.0
            || x.iter().any(|value| !value.is_finite())
        {
            return false;
        }

        let Some(jacobian_len) = self.num_residuals.checked_mul(self.num_vars) else {
            return false;
        };
        let Ok(jac) = Self::evaluate_batch(&self.compiled_jacobian, x, jacobian_len, "Jacobian")
        else {
            return false;
        };
        if jac.iter().any(|value| !value.is_finite()) {
            return false;
        }

        let Ok(baseline) =
            Self::evaluate_batch(&self.compiled_residuals, x, self.num_residuals, "residual")
        else {
            return false;
        };
        if baseline.iter().any(|value| !value.is_finite()) {
            return false;
        }

        let mut x_plus = Vec::new();
        let mut x_minus = Vec::new();
        if x_plus.try_reserve_exact(x.len()).is_err() || x_minus.try_reserve_exact(x.len()).is_err()
        {
            return false;
        }
        x_plus.extend_from_slice(x);
        x_minus.extend_from_slice(x);

        for j in 0..self.num_vars {
            x_plus[j] = x[j] + eps;
            x_minus[j] = x[j] - eps;
            let denominator = x_plus[j] - x_minus[j];
            if !x_plus[j].is_finite()
                || !x_minus[j].is_finite()
                || !denominator.is_finite()
                || denominator == 0.0
            {
                return false;
            }

            let Ok(res_plus) = Self::evaluate_batch(
                &self.compiled_residuals,
                &x_plus,
                self.num_residuals,
                "residual",
            ) else {
                return false;
            };
            let Ok(res_minus) = Self::evaluate_batch(
                &self.compiled_residuals,
                &x_minus,
                self.num_residuals,
                "residual",
            ) else {
                return false;
            };

            for i in 0..self.num_residuals {
                let numerical_deriv = (res_plus[i] - res_minus[i]) / denominator;
                let compiled_deriv = jac[i * self.num_vars + j];
                let difference = (numerical_deriv - compiled_deriv).abs();
                if !res_plus[i].is_finite()
                    || !res_minus[i].is_finite()
                    || !numerical_deriv.is_finite()
                    || !difference.is_finite()
                    || difference > tol
                {
                    return false;
                }
            }

            x_plus[j] = x[j];
            x_minus[j] = x[j];
        }
        true
    }

    fn evaluate_batch(
        expressions: &[CompiledExpr],
        x: &[f64],
        expected: usize,
        lane: &'static str,
    ) -> Result<Vec<f64>, EvalError> {
        if expressions.len() != expected {
            return Err(EvalError::CompiledExpressionCountMismatch {
                lane,
                expected,
                actual: expressions.len(),
            });
        }
        let mut evaluated = Vec::new();
        evaluated
            .try_reserve_exact(expected)
            .map_err(|_| EvalError::AllocationFailure)?;
        for expression in expressions {
            evaluated.push(expression.try_eval(x)?);
        }
        Ok(evaluated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_compile_rejection_of_unmapped_symbols() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let mut var_map = HashMap::new();
        var_map.insert(x, 0);

        let expr = Expr::Add(vec![Expr::symbol("x"), Expr::symbol("y")]);
        assert_eq!(
            CompiledExpr::try_compile(&expr, &var_map),
            Err(CompileError::UnmappedVariable(y))
        );
    }

    #[test]
    fn test_compile_rejection_of_unsupported_functions() {
        let x = Symbol::new("x");
        let mut var_map = HashMap::new();
        var_map.insert(x, 0);

        let expr = Expr::Function("custom_bessel".to_string(), vec![Expr::symbol("x")]);
        assert!(matches!(
            CompiledExpr::try_compile(&expr, &var_map),
            Err(CompileError::UnsupportedFunction(_))
        ));
    }

    #[test]
    fn test_compile_rejection_of_symbolic_powers() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let mut var_map = HashMap::new();
        var_map.insert(x.clone(), 0);
        var_map.insert(y.clone(), 1);

        // x^y is not supported in compiled fast array evaluator (only numeric constant powers)
        let expr = Expr::Pow(Arc::new(Expr::Sym(x)), Arc::new(Expr::Sym(y)));
        assert!(matches!(
            CompiledExpr::try_compile(&expr, &var_map),
            Err(CompileError::UnsupportedExponent(_))
        ));
    }

    #[test]
    fn test_eval_buffer_mismatch_and_bounds() {
        let x = Symbol::new("x");
        let sys = CompiledResidualSystem::try_compile(&[Expr::symbol("x")], &[x]).unwrap();

        let mut wrong_res = [0.0; 2];
        assert!(matches!(
            sys.try_eval_residuals(&[1.0], &mut wrong_res),
            Err(EvalError::ResidualBufferMismatch { .. })
        ));

        let mut wrong_jac = [0.0; 2];
        assert!(matches!(
            sys.try_eval_jacobian(&[1.0], &mut wrong_jac),
            Err(EvalError::JacobianBufferMismatch { .. })
        ));
    }

    #[test]
    fn compile_operation_limit_includes_parent_operations() {
        let boundary = Expr::Add(vec![Expr::from_i64(1); MAX_COMPILE_OPS - 1]);
        assert_eq!(
            CompiledExpr::try_compile(&boundary, &HashMap::new())
                .unwrap()
                .ops
                .len(),
            MAX_COMPILE_OPS
        );

        let expression = Expr::Add(vec![Expr::from_i64(1); MAX_COMPILE_OPS]);
        assert!(matches!(
            CompiledExpr::try_compile(&expression, &HashMap::new()),
            Err(CompileError::OpLimitExceeded { ops, max })
                if ops == MAX_COMPILE_OPS + 1 && max == MAX_COMPILE_OPS
        ));
    }

    #[test]
    fn non_finite_exponents_are_refused() {
        let huge = fsym_core::BigInt::parse_bytes("9".repeat(400).as_bytes(), 10).unwrap();
        let base = Arc::new(Expr::from_i64(2));

        for exponent in [
            Expr::Integer(huge.clone()),
            Expr::Rational(fsym_core::BigRational::new(
                huge,
                fsym_core::BigInt::from(1),
            )),
        ] {
            assert!(matches!(
                CompiledExpr::try_compile(
                    &Expr::Pow(base.clone(), Arc::new(exponent)),
                    &HashMap::new(),
                ),
                Err(CompileError::UnsupportedExponent(_))
            ));
        }
    }

    #[test]
    fn failed_batch_evaluation_does_not_publish_partial_outputs() {
        let system = CompiledResidualSystem {
            vars: Vec::new(),
            num_residuals: 2,
            num_vars: 1,
            compiled_residuals: vec![
                CompiledExpr {
                    ops: vec![CompiledOp::LoadConst(1.0)],
                },
                CompiledExpr {
                    ops: vec![CompiledOp::Add(1)],
                },
            ],
            compiled_jacobian: vec![
                CompiledExpr {
                    ops: vec![CompiledOp::LoadConst(1.0)],
                },
                CompiledExpr {
                    ops: vec![CompiledOp::Add(1)],
                },
            ],
        };

        let mut residuals = [9.0, 9.0];
        assert_eq!(
            system.try_eval_residuals(&[], &mut residuals),
            Err(EvalError::StackUnderflow)
        );
        assert_eq!(residuals, [9.0, 9.0]);

        let mut jacobian = [9.0, 9.0];
        assert_eq!(
            system.try_eval_jacobian(&[], &mut jacobian),
            Err(EvalError::StackUnderflow)
        );
        assert_eq!(jacobian, [9.0, 9.0]);

        assert_eq!(
            system.try_eval_system(&[], &mut residuals, &mut jacobian),
            Err(EvalError::StackUnderflow)
        );
        assert_eq!(residuals, [9.0, 9.0]);
        assert_eq!(jacobian, [9.0, 9.0]);
    }

    #[test]
    fn finite_difference_diagnostic_rejects_invalid_numeric_policy() {
        let x = Symbol::new("x");
        let system = CompiledResidualSystem::try_compile(&[Expr::symbol("x")], &[x]).unwrap();

        assert!(!system.check_with_finite_differences(&[1.0], 0.0, 1e-5));
        assert!(!system.check_with_finite_differences(&[1.0], 1e-6, f64::NAN));
        assert!(!system.check_with_finite_differences(&[f64::NAN], 1e-6, 1e-5));
    }

    #[test]
    fn duplicate_variables_and_oversized_bytecode_are_typed_refusals() {
        let x = Symbol::new("x");
        assert!(matches!(
            CompiledResidualSystem::try_compile(&[Expr::symbol("x")], &[x.clone(), x.clone()],),
            Err(CompileError::DuplicateVariable(ref duplicate)) if duplicate == &x
        ));

        let expression = CompiledExpr {
            ops: vec![CompiledOp::LoadConst(1.0); MAX_COMPILE_OPS + 1],
        };
        assert_eq!(
            expression.try_eval(&[]),
            Err(EvalError::OpLimitExceeded {
                ops: MAX_COMPILE_OPS + 1,
                max: MAX_COMPILE_OPS,
            })
        );
    }

    #[test]
    fn inconsistent_public_system_shape_is_refused_without_output_mutation() {
        let system = CompiledResidualSystem {
            vars: Vec::new(),
            num_residuals: 1,
            num_vars: 0,
            compiled_residuals: Vec::new(),
            compiled_jacobian: Vec::new(),
        };
        let mut output = [7.0];
        assert!(matches!(
            system.try_eval_residuals(&[], &mut output),
            Err(EvalError::CompiledExpressionCountMismatch {
                lane: "residual",
                expected: 1,
                actual: 0,
            })
        ));
        assert_eq!(output, [7.0]);
    }
}
