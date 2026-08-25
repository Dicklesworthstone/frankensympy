//! # fsym-functions
//!
//! Constructors for sine, cosine, exponential, logarithm, gamma, and zeta
//! expressions, with a small catalog of exact identity values.

#![forbid(unsafe_code)]

use fsym_core::{Constant, Expr};

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

/// Create a natural logarithm function expression: log(x).
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
