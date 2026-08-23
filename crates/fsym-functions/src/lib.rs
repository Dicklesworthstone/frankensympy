//! # fsym-functions
//!
//! Elementary and special mathematical functions: trigonometric, hyperbolic, exponential,
//! logarithmic, gamma, zeta, error functions, Bessel, and orthogonal polynomials.

#![forbid(unsafe_code)]

use fsym_core::{Constant, Expr};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FunctionError {
    #[error("Invalid argument count for function {0}: expected {1}, got {2}")]
    ArgumentCountMismatch(String, usize, usize),
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

/// Create a sine function expression: sin(x).
pub fn sin(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("sin".to_string(), vec![arg])
}

/// Create a cosine function expression: cos(x).
pub fn cos(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("cos".to_string(), vec![arg])
}

/// Create an exponential function expression: exp(x).
pub fn exp(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("exp".to_string(), vec![arg])
}

/// Create a natural logarithm function expression: log(x) or ln(x).
pub fn log(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(0);
    }
    if arg == Expr::Const(Constant::E) {
        return Expr::from_i64(1);
    }
    Expr::Function("log".to_string(), vec![arg])
}

/// Create a Gamma function expression: Γ(x).
pub fn gamma(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(1);
    }
    Expr::Function("gamma".to_string(), vec![arg])
}

/// Create a Riemann Zeta function expression: ζ(s).
pub fn zeta(arg: Expr) -> Expr {
    Expr::Function("zeta".to_string(), vec![arg])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementary_evaluations() {
        assert_eq!(sin(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(cos(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(exp(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(log(Expr::from_i64(1)), Expr::from_i64(0));
    }
}
