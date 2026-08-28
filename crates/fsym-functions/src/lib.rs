//! # fsym-functions
//!
//! Constructors for sine, cosine, tangent, cotangent, secant, cosecant,
//! inverse trigonometric, exponential, logarithm, hyperbolic, gamma,
//! zeta, factorial, binomial, and fibonacci expressions, with exact identity values.

#![forbid(unsafe_code)]

use fsym_core::{BigInt, Constant, Expr};

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

/// Create a tangent function expression: tan(x).
pub fn tan(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("tan".to_string(), vec![arg])
}

/// Create a cotangent function expression: cot(x).
pub fn cot(arg: Expr) -> Expr {
    Expr::Function("cot".to_string(), vec![arg])
}

/// Create a secant function expression: sec(x).
pub fn sec(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("sec".to_string(), vec![arg])
}

/// Create a cosecant function expression: csc(x).
pub fn csc(arg: Expr) -> Expr {
    Expr::Function("csc".to_string(), vec![arg])
}

/// Create an arcsine function expression: asin(x).
pub fn asin(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("asin".to_string(), vec![arg])
}

/// Create an arccosine function expression: acos(x).
pub fn acos(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(0);
    }
    Expr::Function("acos".to_string(), vec![arg])
}

/// Create an arctangent function expression: atan(x).
pub fn atan(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("atan".to_string(), vec![arg])
}

/// Create a hyperbolic sine function expression: sinh(x).
pub fn sinh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("sinh".to_string(), vec![arg])
}

/// Create a hyperbolic cosine function expression: cosh(x).
pub fn cosh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("cosh".to_string(), vec![arg])
}

/// Create a hyperbolic tangent function expression: tanh(x).
pub fn tanh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("tanh".to_string(), vec![arg])
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
    if arg == Expr::from_i64(2) {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(n) = &arg
        && n > &BigInt::from(0)
        && n <= &BigInt::from(100)
    {
        let n_u64 = n.to_u64().unwrap_or(0);
        if n_u64 >= 1 {
            let mut acc = BigInt::from(1);
            for i in 1..n_u64 {
                acc *= BigInt::from(i);
            }
            return Expr::Integer(acc);
        }
    }
    Expr::Function("gamma".to_string(), vec![arg])
}

/// Create a factorial function expression: n!.
pub fn factorial(arg: Expr) -> Expr {
    if arg.is_zero() || arg.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(n) = &arg
        && n > &BigInt::from(0)
        && n <= &BigInt::from(100)
    {
        let n_u64 = n.to_u64().unwrap_or(0);
        let mut acc = BigInt::from(1);
        for i in 1..=n_u64 {
            acc *= BigInt::from(i);
        }
        return Expr::Integer(acc);
    }
    Expr::Function("factorial".to_string(), vec![arg])
}

/// Create a binomial coefficient expression: (n choose k).
pub fn binomial(n: Expr, k: Expr) -> Expr {
    if k.is_zero() {
        return Expr::from_i64(1);
    }
    if n == k {
        return Expr::from_i64(1);
    }
    if let (Expr::Integer(ni), Expr::Integer(ki)) = (&n, &k)
        && ki >= &BigInt::from(0)
        && ni >= ki
        && ni <= &BigInt::from(100)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let k_val = ki.to_u64().unwrap_or(0);
        let k_opt = k_val.min(n_val - k_val);
        let mut result = BigInt::from(1);
        for i in 0..k_opt {
            result = (result * BigInt::from(n_val - i)) / BigInt::from(i + 1);
        }
        return Expr::Integer(result);
    }
    Expr::Function("binomial".to_string(), vec![n, k])
}

/// Create a Fibonacci number expression: F_n.
pub fn fibonacci(n: Expr) -> Expr {
    if n.is_zero() {
        return Expr::from_i64(0);
    }
    if n.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(ni) = &n
        && ni > &BigInt::from(0)
        && ni <= &BigInt::from(200)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let mut a = BigInt::from(0);
        let mut b = BigInt::from(1);
        for _ in 2..=n_val {
            let c = &a + &b;
            a = b;
            b = c;
        }
        return Expr::Integer(b);
    }
    Expr::Function("fibonacci".to_string(), vec![n])
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
        assert_eq!(tan(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(sec(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(asin(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(acos(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(atan(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(sinh(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(cosh(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(tanh(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(exp(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(log(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(log(Expr::Const(Constant::E)), Expr::from_i64(1));
    }

    #[test]
    fn test_combinatorial_and_special() {
        assert_eq!(factorial(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(factorial(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(factorial(Expr::from_i64(5)), Expr::from_i64(120));

        assert_eq!(gamma(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(gamma(Expr::from_i64(2)), Expr::from_i64(1));
        assert_eq!(gamma(Expr::from_i64(5)), Expr::from_i64(24));

        assert_eq!(
            binomial(Expr::from_i64(5), Expr::from_i64(2)),
            Expr::from_i64(10)
        );
        assert_eq!(
            binomial(Expr::from_i64(10), Expr::from_i64(0)),
            Expr::from_i64(1)
        );
        assert_eq!(
            binomial(Expr::from_i64(10), Expr::from_i64(10)),
            Expr::from_i64(1)
        );

        assert_eq!(fibonacci(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(fibonacci(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(fibonacci(Expr::from_i64(2)), Expr::from_i64(1));
        assert_eq!(fibonacci(Expr::from_i64(10)), Expr::from_i64(55));
    }
}
