//! Fast symbolic-to-numeric compilation for residual systems and exact Jacobians (WS12).

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
        if ops.is_empty() {
            ops.push(CompiledOp::LoadConst(0.0));
        }
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
        if ops.len() > MAX_COMPILE_OPS {
            return Err(CompileError::OpLimitExceeded {
                ops: ops.len(),
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
                ops.push(CompiledOp::LoadConst(val));
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
                ops.push(CompiledOp::LoadConst(val));
            }
            Expr::Const(Constant::Pi) => ops.push(CompiledOp::LoadConst(std::f64::consts::PI)),
            Expr::Const(Constant::E) => ops.push(CompiledOp::LoadConst(std::f64::consts::E)),
            Expr::Const(Constant::Infinity) => ops.push(CompiledOp::LoadConst(f64::INFINITY)),
            Expr::Const(Constant::NegativeInfinity) => {
                ops.push(CompiledOp::LoadConst(f64::NEG_INFINITY))
            }
            Expr::Const(Constant::ComplexInfinity)
            | Expr::Const(Constant::I)
            | Expr::Const(Constant::NaN) => {
                return Err(CompileError::UnsupportedExpression(format!("{expr:?}")));
            }
            Expr::Sym(s) => {
                if let Some(&idx) = var_map.get(s) {
                    ops.push(CompiledOp::LoadVar(idx));
                } else {
                    return Err(CompileError::UnmappedVariable(s.clone()));
                }
            }
            Expr::Add(terms) => {
                if terms.is_empty() {
                    ops.push(CompiledOp::LoadConst(0.0));
                } else {
                    for t in terms {
                        Self::compile_recursive(t, var_map, ops, depth + 1)?;
                    }
                    ops.push(CompiledOp::Add(terms.len()));
                }
            }
            Expr::Mul(factors) => {
                if factors.is_empty() {
                    ops.push(CompiledOp::LoadConst(1.0));
                } else {
                    for f in factors {
                        Self::compile_recursive(f, var_map, ops, depth + 1)?;
                    }
                    ops.push(CompiledOp::Mul(factors.len()));
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
                        ops.push(CompiledOp::Pow(p));
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
                        ops.push(CompiledOp::Pow(numer / denom));
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
                    "sin" => ops.push(CompiledOp::Sin),
                    "cos" => ops.push(CompiledOp::Cos),
                    "tan" => ops.push(CompiledOp::Tan),
                    "sinh" => ops.push(CompiledOp::Sinh),
                    "cosh" => ops.push(CompiledOp::Cosh),
                    "tanh" => ops.push(CompiledOp::Tanh),
                    "exp" => ops.push(CompiledOp::Exp),
                    "ln" | "log" => ops.push(CompiledOp::Ln),
                    "sqrt" => ops.push(CompiledOp::Sqrt),
                    "abs" => ops.push(CompiledOp::Abs),
                    other => {
                        return Err(CompileError::UnsupportedFunction(other.to_string()));
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluates the compiled expression using a verified operand stack with bounds checks.
    pub fn try_eval(&self, vars: &[f64]) -> Result<f64, EvalError> {
        let mut stack = Vec::with_capacity(16);
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
        Ok(stack.pop().unwrap())
    }

    /// Evaluates the compiled expression, returning `f64::NAN` on failure.
    pub fn eval(&self, vars: &[f64]) -> f64 {
        self.try_eval(vars).unwrap_or(f64::NAN)
    }
}

/// Compiled multi-dimensional residual system and exact Jacobian matrix.
#[derive(Debug, Clone)]
pub struct CompiledResidualSystem {
    pub vars: Vec<Symbol>,
    pub num_residuals: usize,
    pub num_vars: usize,
    pub compiled_residuals: Vec<CompiledExpr>,
    pub compiled_jacobian: Vec<CompiledExpr>,
}

impl CompiledResidualSystem {
    /// Compiles a system of equations $\mathbf{f}(\mathbf{x}) = \mathbf{0}$ and its exact Jacobian.
    /// Fails closed if any expression contains unmapped symbols or unsupported functions.
    pub fn try_compile(exprs: &[Expr], vars: &[Symbol]) -> Result<Self, CompileError> {
        let mut var_map = HashMap::new();
        for (i, v) in vars.iter().enumerate() {
            var_map.insert(v.clone(), i);
        }

        let num_residuals = exprs.len();
        let num_vars = vars.len();

        let mut compiled_residuals = Vec::with_capacity(num_residuals);
        for e in exprs {
            compiled_residuals.push(CompiledExpr::try_compile(e, &var_map)?);
        }

        let mut compiled_jacobian = Vec::with_capacity(num_residuals * num_vars);
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
        for (i, expr) in self.compiled_residuals.iter().enumerate() {
            out_res[i] = expr.try_eval(x)?;
        }
        Ok(())
    }

    /// Infallible residual evaluator for backward compatibility.
    pub fn eval_residuals(&self, x: &[f64], out_res: &mut [f64]) {
        self.try_eval_residuals(x, out_res)
            .expect("residual evaluation failed");
    }

    /// Evaluates the flat Jacobian matrix $J_{i, j}$ in-place (row-major: $i \times n + j$).
    pub fn try_eval_jacobian(&self, x: &[f64], out_jac: &mut [f64]) -> Result<(), EvalError> {
        let expected = self.num_residuals * self.num_vars;
        if out_jac.len() != expected {
            return Err(EvalError::JacobianBufferMismatch {
                expected,
                actual: out_jac.len(),
            });
        }
        for (idx, expr) in self.compiled_jacobian.iter().enumerate() {
            out_jac[idx] = expr.try_eval(x)?;
        }
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
        self.try_eval_residuals(x, out_res)?;
        self.try_eval_jacobian(x, out_jac)?;
        Ok(())
    }

    /// Infallible system evaluator for backward compatibility.
    pub fn eval_system(&self, x: &[f64], out_res: &mut [f64], out_jac: &mut [f64]) {
        self.try_eval_system(x, out_res, out_jac)
            .expect("system evaluation failed");
    }

    /// Verifies exact compiled Jacobian against numerical central finite differences.
    pub fn verify_with_finite_differences(&self, x: &[f64], eps: f64, tol: f64) -> bool {
        let mut jac = vec![0.0; self.num_residuals * self.num_vars];
        if self.try_eval_jacobian(x, &mut jac).is_err() {
            return false;
        }

        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        let mut res_plus = vec![0.0; self.num_residuals];
        let mut res_minus = vec![0.0; self.num_residuals];

        for j in 0..self.num_vars {
            x_plus[j] = x[j] + eps;
            x_minus[j] = x[j] - eps;

            if self.try_eval_residuals(&x_plus, &mut res_plus).is_err() {
                return false;
            }
            if self.try_eval_residuals(&x_minus, &mut res_minus).is_err() {
                return false;
            }

            for i in 0..self.num_residuals {
                let numerical_deriv = (res_plus[i] - res_minus[i]) / (2.0 * eps);
                let exact_deriv = jac[i * self.num_vars + j];
                if (numerical_deriv - exact_deriv).abs() > tol {
                    return false;
                }
            }

            x_plus[j] = x[j];
            x_minus[j] = x[j];
        }
        true
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
}
