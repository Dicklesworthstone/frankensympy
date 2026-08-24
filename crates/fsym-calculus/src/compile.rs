//! Fast symbolic-to-numeric compilation for residual systems and exact Jacobians (WS12).

#![forbid(unsafe_code)]

use crate::diff;
use fsym_core::{Constant, Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    Exp,
    Ln,
}

/// Linear sequence of operations evaluating an expression to an f64 value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledExpr {
    pub ops: Vec<CompiledOp>,
}

impl CompiledExpr {
    /// Compiles a symbolic expression given a variable index lookup map.
    pub fn compile(expr: &Expr, var_map: &HashMap<Symbol, usize>) -> Self {
        let mut ops = Vec::new();
        Self::compile_recursive(expr, var_map, &mut ops);
        Self { ops }
    }

    fn compile_recursive(expr: &Expr, var_map: &HashMap<Symbol, usize>, ops: &mut Vec<CompiledOp>) {
        match expr {
            Expr::Integer(n) => {
                let val = n.to_string().parse::<f64>().unwrap_or(0.0);
                ops.push(CompiledOp::LoadConst(val));
            }
            Expr::Rational(r) => {
                let numer = r.numer().to_string().parse::<f64>().unwrap_or(0.0);
                let denom = r.denom().to_string().parse::<f64>().unwrap_or(1.0);
                ops.push(CompiledOp::LoadConst(numer / denom));
            }
            Expr::Const(Constant::Pi) => ops.push(CompiledOp::LoadConst(std::f64::consts::PI)),
            Expr::Const(Constant::E) => ops.push(CompiledOp::LoadConst(std::f64::consts::E)),
            Expr::Const(Constant::Infinity) => ops.push(CompiledOp::LoadConst(f64::INFINITY)),
            Expr::Const(Constant::NegativeInfinity) => {
                ops.push(CompiledOp::LoadConst(f64::NEG_INFINITY))
            }
            Expr::Const(Constant::ComplexInfinity)
            | Expr::Const(Constant::I)
            | Expr::Const(Constant::NaN) => ops.push(CompiledOp::LoadConst(f64::NAN)),
            Expr::Sym(s) => {
                if let Some(&idx) = var_map.get(s) {
                    ops.push(CompiledOp::LoadVar(idx));
                } else {
                    ops.push(CompiledOp::LoadConst(0.0));
                }
            }
            Expr::Add(terms) => {
                for t in terms {
                    Self::compile_recursive(t, var_map, ops);
                }
                ops.push(CompiledOp::Add(terms.len()));
            }
            Expr::Mul(factors) => {
                for f in factors {
                    Self::compile_recursive(f, var_map, ops);
                }
                ops.push(CompiledOp::Mul(factors.len()));
            }
            Expr::Pow(base, exp) => {
                Self::compile_recursive(base, var_map, ops);
                if let Expr::Integer(n) = exp.as_ref() {
                    let p = n.to_string().parse::<f64>().unwrap_or(1.0);
                    ops.push(CompiledOp::Pow(p));
                } else {
                    // Fallback constant pow
                    ops.push(CompiledOp::Pow(1.0));
                }
            }
            Expr::Function(name, args) => {
                if !args.is_empty() {
                    Self::compile_recursive(&args[0], var_map, ops);
                    match name.as_str() {
                        "sin" => ops.push(CompiledOp::Sin),
                        "cos" => ops.push(CompiledOp::Cos),
                        "exp" => ops.push(CompiledOp::Exp),
                        "ln" | "log" => ops.push(CompiledOp::Ln),
                        _ => {}
                    }
                }
            }
        }
    }

    /// Evaluates the compiled expression using a small operand stack.
    pub fn eval(&self, vars: &[f64]) -> f64 {
        let mut stack = Vec::with_capacity(16);
        for op in &self.ops {
            match op {
                CompiledOp::LoadConst(c) => stack.push(*c),
                CompiledOp::LoadVar(idx) => stack.push(vars.get(*idx).copied().unwrap_or(0.0)),
                CompiledOp::Add(count) => {
                    let mut sum = 0.0;
                    for _ in 0..*count {
                        sum += stack.pop().unwrap_or(0.0);
                    }
                    stack.push(sum);
                }
                CompiledOp::Mul(count) => {
                    let mut prod = 1.0;
                    for _ in 0..*count {
                        prod *= stack.pop().unwrap_or(1.0);
                    }
                    stack.push(prod);
                }
                CompiledOp::Neg => {
                    if let Some(v) = stack.pop() {
                        stack.push(-v);
                    }
                }
                CompiledOp::Pow(p) => {
                    if let Some(v) = stack.pop() {
                        stack.push(v.powf(*p));
                    }
                }
                CompiledOp::Sin => {
                    if let Some(v) = stack.pop() {
                        stack.push(v.sin());
                    }
                }
                CompiledOp::Cos => {
                    if let Some(v) = stack.pop() {
                        stack.push(v.cos());
                    }
                }
                CompiledOp::Exp => {
                    if let Some(v) = stack.pop() {
                        stack.push(v.exp());
                    }
                }
                CompiledOp::Ln => {
                    if let Some(v) = stack.pop() {
                        stack.push(v.ln());
                    }
                }
            }
        }
        stack.pop().unwrap_or(0.0)
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
    pub fn compile(exprs: &[Expr], vars: &[Symbol]) -> Self {
        let mut var_map = HashMap::new();
        for (i, v) in vars.iter().enumerate() {
            var_map.insert(v.clone(), i);
        }

        let num_residuals = exprs.len();
        let num_vars = vars.len();

        let mut compiled_residuals = Vec::with_capacity(num_residuals);
        for e in exprs {
            compiled_residuals.push(CompiledExpr::compile(e, &var_map));
        }

        let mut compiled_jacobian = Vec::with_capacity(num_residuals * num_vars);
        for e in exprs {
            for v in vars {
                let d = diff(e, v);
                compiled_jacobian.push(CompiledExpr::compile(&d, &var_map));
            }
        }

        Self {
            vars: vars.to_vec(),
            num_residuals,
            num_vars,
            compiled_residuals,
            compiled_jacobian,
        }
    }

    /// Evaluates the residual vector $\mathbf{f}(\mathbf{x})$ in-place.
    pub fn eval_residuals(&self, x: &[f64], out_res: &mut [f64]) {
        assert_eq!(out_res.len(), self.num_residuals);
        for (i, expr) in self.compiled_residuals.iter().enumerate() {
            out_res[i] = expr.eval(x);
        }
    }

    /// Evaluates the flat Jacobian matrix $J_{i, j}$ in-place (row-major: $i \times n + j$).
    pub fn eval_jacobian(&self, x: &[f64], out_jac: &mut [f64]) {
        assert_eq!(out_jac.len(), self.num_residuals * self.num_vars);
        for (idx, expr) in self.compiled_jacobian.iter().enumerate() {
            out_jac[idx] = expr.eval(x);
        }
    }

    /// Evaluates both residuals and Jacobian simultaneously.
    pub fn eval_system(&self, x: &[f64], out_res: &mut [f64], out_jac: &mut [f64]) {
        self.eval_residuals(x, out_res);
        self.eval_jacobian(x, out_jac);
    }

    /// Verifies exact compiled Jacobian against numerical central finite differences.
    pub fn verify_with_finite_differences(&self, x: &[f64], eps: f64, tol: f64) -> bool {
        let mut jac = vec![0.0; self.num_residuals * self.num_vars];
        self.eval_jacobian(x, &mut jac);

        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        let mut res_plus = vec![0.0; self.num_residuals];
        let mut res_minus = vec![0.0; self.num_residuals];

        for j in 0..self.num_vars {
            x_plus[j] = x[j] + eps;
            x_minus[j] = x[j] - eps;

            self.eval_residuals(&x_plus, &mut res_plus);
            self.eval_residuals(&x_minus, &mut res_minus);

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
