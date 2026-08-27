//! # fsym-core
//!
//! Core symbolic expression AST, canonicalization primitives, and symbol registry
//! for FrankenSymPy.

#![forbid(unsafe_code)]

pub use fsym_bigint::BigInt;
pub use fsym_rational::BigRational;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

pub mod algebraic;
pub mod arith;
pub mod ball;
pub mod canonical;
pub mod dag;
mod parser;

pub use algebraic::*;
pub use ball::*;
pub use dag::*;
pub use parser::parse;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    #[error("Division by zero in symbolic expression")]
    DivisionByZero,
    #[error("Invalid operation on symbolic expression: {0}")]
    InvalidOperation(String),
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),
    #[error("Failed to parse expression: {0}")]
    ParseError(String),
}

/// Fundamental symbol definition with name and assumptions metadata.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
}

impl Symbol {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Core symbolic expression enum.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Expr {
    /// Symbolic atomic variable.
    Sym(Symbol),
    /// Arbitrary-precision integer literal.
    Integer(BigInt),
    /// Exact rational number.
    Rational(BigRational),
    /// Named mathematical constants: e.g., Pi, E, I, Infinity, NegativeInfinity.
    Const(Constant),
    /// N-ary addition of expressions: Σ a_i.
    Add(Vec<Expr>),
    /// N-ary multiplication of expressions: Π a_i.
    Mul(Vec<Expr>),
    /// Power expression: base ^ exp.
    Pow(Arc<Expr>, Arc<Expr>),
    /// Named function application: name(args...).
    Function(String, Vec<Expr>),
}

/// Mathematical constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Constant {
    Pi,
    E,
    I, // Imaginary unit
    Infinity,
    NegativeInfinity,
    ComplexInfinity,
    NaN,
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Pi => write!(f, "pi"),
            Constant::E => write!(f, "E"),
            Constant::I => write!(f, "I"),
            Constant::Infinity => write!(f, "oo"),
            Constant::NegativeInfinity => write!(f, "-oo"),
            Constant::ComplexInfinity => write!(f, "zoo"),
            Constant::NaN => write!(f, "nan"),
        }
    }
}

impl Expr {
    /// Create a symbol expression.
    pub fn symbol(name: impl Into<String>) -> Self {
        Expr::Sym(Symbol::new(name))
    }

    /// Create an integer expression from i64.
    pub fn from_i64(n: i64) -> Self {
        Expr::Integer(BigInt::from(n))
    }

    /// Create a rational expression from numer/denom.
    pub fn rational(numer: i64, denom: i64) -> Result<Self, CoreError> {
        if denom == 0 {
            return Err(CoreError::DivisionByZero);
        }
        let r = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        if r.is_integer() {
            Ok(Expr::Integer(r.to_integer()))
        } else {
            Ok(Expr::Rational(r))
        }
    }

    /// Power expression.
    pub fn pow(self, exp: Expr) -> Self {
        Expr::Pow(Arc::new(self), Arc::new(exp))
    }

    /// Check if expression is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Expr::Integer(n) => n.is_zero(),
            Expr::Rational(r) => r.is_zero(),
            _ => false,
        }
    }

    /// Check if expression is one.
    pub fn is_one(&self) -> bool {
        match self {
            Expr::Integer(n) => n.is_one(),
            Expr::Rational(r) => r.is_one(),
            _ => false,
        }
    }

    /// Substitute symbol mappings in expression.
    pub fn subs(&self, map: &HashMap<Symbol, Expr>) -> Expr {
        match self {
            Expr::Sym(s) => map.get(s).cloned().unwrap_or_else(|| Expr::Sym(s.clone())),
            Expr::Integer(n) => Expr::Integer(n.clone()),
            Expr::Rational(r) => Expr::Rational(r.clone()),
            Expr::Const(c) => Expr::Const(*c),
            // Re-fold through the arithmetic operators so fully numeric
            // combinations canonicalize (e.g. x + 10 with x -> 3 gives 13).
            Expr::Add(terms) => terms
                .iter()
                .map(|t| t.subs(map))
                .reduce(|a, b| a + b)
                .unwrap_or(Expr::from_i64(0)),
            Expr::Mul(factors) => factors
                .iter()
                .map(|f| f.subs(map))
                .reduce(|a, b| a * b)
                .unwrap_or(Expr::from_i64(1)),
            Expr::Pow(b, e) => Expr::Pow(Arc::new(b.subs(map)), Arc::new(e.subs(map))),
            Expr::Function(name, args) => {
                Expr::Function(name.clone(), args.iter().map(|a| a.subs(map)).collect())
            }
        }
    }

    /// Collect all free symbols in this expression.
    pub fn free_symbols(&self) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        self.collect_symbols(&mut symbols);
        symbols.sort();
        symbols.dedup();
        symbols
    }

    fn collect_symbols(&self, acc: &mut Vec<Symbol>) {
        match self {
            Expr::Sym(s) => acc.push(s.clone()),
            Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
            Expr::Add(terms) | Expr::Mul(terms) | Expr::Function(_, terms) => {
                for t in terms {
                    t.collect_symbols(acc);
                }
            }
            Expr::Pow(b, e) => {
                b.collect_symbols(acc);
                e.collect_symbols(acc);
            }
        }
    }
}
impl Expr {
    /// Value of a fully constant integer subexpression, including integer
    /// powers with bounded non-negative exponents. `None` if symbolic or
    /// the exponent is too large to fold safely.
    pub fn const_integer_value(&self) -> Option<BigInt> {
        match self {
            Expr::Integer(n) => Some(n.clone()),
            Expr::Pow(b, e) => match (b.as_ref(), e.as_ref()) {
                (Expr::Integer(base), Expr::Integer(exp)) if exp.bits() <= 11 => {
                    let exp_i = exp.to_i64()?;
                    // Result budget: base bits x exponent capped so a
                    // hostile (huge_base, huge_exp) pair cannot allocate
                    // gigabytes inside a fold. 4096 result bits max.
                    let result_bits = base.bits() * u64::try_from(exp_i).ok()?;
                    if !(0..=1024).contains(&exp_i) || result_bits > 4096 {
                        return None;
                    }
                    Some(base.pow(exp_i as u32))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Numeric evaluation to `f64`.
    ///
    /// Fails on free symbols, non-real constants, and functions outside the
    /// supported single-argument set (`sin`, `cos`, `tan`, `exp`, `log`,
    /// `ln`, `sqrt`).
    pub fn evalf(&self) -> Result<f64, CoreError> {
        match self {
            Expr::Integer(n) => n.to_f64().ok_or_else(|| {
                CoreError::InvalidOperation(format!("integer out of f64 range: {n}"))
            }),
            Expr::Rational(r) => r.to_f64().ok_or_else(|| {
                CoreError::InvalidOperation("rational out of f64 range".to_string())
            }),
            Expr::Const(c) => Ok(match c {
                Constant::Pi => std::f64::consts::PI,
                Constant::E => std::f64::consts::E,
                Constant::Infinity => f64::INFINITY,
                Constant::NegativeInfinity => f64::NEG_INFINITY,
                Constant::NaN => f64::NAN,
                Constant::I | Constant::ComplexInfinity => {
                    return Err(CoreError::InvalidOperation(format!(
                        "{c} is not real-valued"
                    )));
                }
            }),
            Expr::Add(terms) => {
                let mut acc = 0.0;
                for t in terms {
                    acc += t.evalf()?;
                }
                Ok(acc)
            }
            Expr::Mul(factors) => {
                let mut acc = 1.0;
                for f in factors {
                    acc *= f.evalf()?;
                }
                Ok(acc)
            }
            Expr::Pow(b, e) => Ok(b.evalf()?.powf(e.evalf()?)),
            Expr::Function(name, args) => {
                let x = match args.as_slice() {
                    [arg] => Some(arg.evalf()?),
                    _ => None,
                };
                match (name.as_str(), x) {
                    ("sin", Some(x)) => Ok(x.sin()),
                    ("cos", Some(x)) => Ok(x.cos()),
                    ("tan", Some(x)) => Ok(x.tan()),
                    ("exp", Some(x)) => Ok(x.exp()),
                    ("log" | "ln", Some(x)) => Ok(x.ln()),
                    ("sqrt", Some(x)) => Ok(x.sqrt()),
                    _ => Err(CoreError::InvalidOperation(format!(
                        "cannot evaluate function `{name}` numerically"
                    ))),
                }
            }
            Expr::Sym(s) => Err(CoreError::InvalidOperation(format!(
                "free symbol `{s}` cannot be evaluated numerically"
            ))),
        }
    }
}

impl std::ops::Add for Expr {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        if let (Some(x), Some(y)) = (self.const_integer_value(), other.const_integer_value()) {
            return Expr::Integer(x + y);
        }
        match (self, other) {
            (Expr::Add(mut terms_a), Expr::Add(terms_b)) => {
                terms_a.extend(terms_b);
                Expr::Add(terms_a)
            }
            (Expr::Add(mut terms), single) | (single, Expr::Add(mut terms)) => {
                terms.push(single);
                Expr::Add(terms)
            }
            (a, b) => Expr::Add(vec![a, b]),
        }
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        match (self, other) {
            (Expr::Integer(a), Expr::Integer(b)) => Expr::Integer(a * b),
            (Expr::Mul(mut factors_a), Expr::Mul(factors_b)) => {
                factors_a.extend(factors_b);
                Expr::Mul(factors_a)
            }
            (Expr::Mul(mut factors), single) | (single, Expr::Mul(mut factors)) => {
                factors.push(single);
                Expr::Mul(factors)
            }
            (a, b) => Expr::Mul(vec![a, b]),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Sym(s) => write!(f, "{}", s),
            Expr::Integer(n) => write!(f, "{}", n),
            Expr::Rational(r) => write!(f, "{}", r),
            Expr::Const(c) => write!(f, "{}", c),
            Expr::Add(terms) => {
                f.write_str("(")?;
                for (index, term) in terms.iter().enumerate() {
                    if index > 0 {
                        f.write_str(" + ")?;
                    }
                    term.fmt(f)?;
                }
                f.write_str(")")
            }
            Expr::Mul(factors) => {
                for (index, factor) in factors.iter().enumerate() {
                    if index > 0 {
                        f.write_str("*")?;
                    }
                    factor.fmt(f)?;
                }
                Ok(())
            }
            Expr::Pow(b, e) => write!(f, "({}**{})", b, e),
            Expr::Function(name, args) => {
                f.write_str(name)?;
                f.write_str("(")?;
                for (index, argument) in args.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    argument.fmt(f)?;
                }
                f.write_str(")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_creation() {
        let x = Expr::symbol("x");
        assert_eq!(format!("{}", x), "x");
    }

    #[test]
    fn test_addition_and_multiplication() {
        let x = Expr::symbol("x");
        let two = Expr::from_i64(2);
        let expr = x.clone() * two + Expr::from_i64(5);
        let free = expr.free_symbols();
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].name, "x");
    }

    #[test]
    fn test_substitution() {
        let x = Symbol::new("x");
        let expr = Expr::symbol("x") + Expr::from_i64(10);
        let mut map = HashMap::new();
        map.insert(x, Expr::from_i64(3));
        let res = expr.subs(&map);
        assert_eq!(res, Expr::from_i64(3) + Expr::from_i64(10));
    }

    #[test]
    fn rational_evalf_uses_balanced_owner_conversion() {
        let scale = &BigInt::one() << 2000u32;
        let value = BigRational::new(&scale + 1i64, &scale - 1i64);
        assert_eq!(Expr::Rational(value).evalf(), Ok(1.0));
    }

    #[test]
    fn test_certified_real_ball_arithmetic() {
        // b1 = [1.5 ± 0.5] = [1.0, 2.0]
        let b1 = RealBall::new(
            BigRational::new(BigInt::from(3), BigInt::from(2)),
            BigRational::new(BigInt::from(1), BigInt::from(2)),
        )
        .unwrap();

        // b2 = [2.0 ± 0.2] = [1.8, 2.2]
        let b2 = RealBall::new(
            BigRational::from_integer(BigInt::from(2)),
            BigRational::new(BigInt::from(1), BigInt::from(5)),
        )
        .unwrap();

        let sum = b1.add(&b2);
        assert_eq!(
            sum.lower(),
            BigRational::new(BigInt::from(14), BigInt::from(5))
        ); // 2.8
        assert_eq!(
            sum.upper(),
            BigRational::new(BigInt::from(21), BigInt::from(5))
        ); // 4.2

        let prod = b1.mul(&b2);
        // Product [1.0, 2.0] * [1.8, 2.2] = [1.8, 4.4]
        assert!(prod.lower() <= BigRational::new(BigInt::from(18), BigInt::from(10)));
        assert!(prod.upper() >= BigRational::new(BigInt::from(44), BigInt::from(10)));

        assert!(b1.is_positive());
        assert!(!b1.contains_zero());
    }

    #[test]
    fn test_algebraic_number_root_refinement() {
        // sqrt(2): P(x) = x^2 - 2, isolating interval [1, 2] -> [1.5 ± 0.5]
        let p_sqrt2 = vec![
            BigRational::from_integer(BigInt::from(-2)),
            BigRational::zero(),
            BigRational::one(),
        ];
        let initial_ball = RealBall::new(
            BigRational::new(BigInt::from(3), BigInt::from(2)),
            BigRational::new(BigInt::from(1), BigInt::from(2)),
        )
        .unwrap();

        let mut alpha = AlgebraicNumber::new(p_sqrt2, initial_ball).unwrap();
        assert_eq!(alpha.degree(), 2);
        assert_eq!(alpha.sign(), 1);

        // Refine radius to <= 1/1000
        let target = BigRational::new(BigInt::from(1), BigInt::from(1000));
        alpha.refine_to_radius(&target).unwrap();
        assert!(alpha.isolating_ball().radius() <= &target);

        // Check certified root enclosure: P(lower) <= 0 and P(upper) >= 0 for x^2 - 2
        let low = alpha.isolating_ball().lower();
        let high = alpha.isolating_ball().upper();
        let p_low = &low * &low - BigRational::from_integer(BigInt::from(2));
        let p_high = &high * &high - BigRational::from_integer(BigInt::from(2));
        assert!(p_low <= BigRational::zero(), "P(lower) must be <= 0");
        assert!(p_high >= BigRational::zero(), "P(upper) must be >= 0");
    }

    #[test]
    fn test_algebraic_number_rejects_non_isolating_intervals() {
        // P(x) = x^3 - x = x(x-1)(x+1) with roots at -1, 0, 1
        let p_cubic = vec![
            BigRational::zero(),
            BigRational::from_integer(BigInt::from(-1)),
            BigRational::zero(),
            BigRational::one(),
        ];

        // Interval [-2, 2] contains 3 roots -> must be REJECTED!
        let multi_root_ball = RealBall::new(
            BigRational::zero(),
            BigRational::from_integer(BigInt::from(2)),
        )
        .unwrap();
        assert!(matches!(
            AlgebraicNumber::new(p_cubic.clone(), multi_root_ball),
            Err(crate::algebraic::AlgebraicError::InvalidIsolatingInterval(
                _
            ))
        ));

        // Interval [2, 3] contains 0 roots -> must be REJECTED!
        let zero_root_ball = RealBall::new(
            BigRational::new(BigInt::from(5), BigInt::from(2)),
            BigRational::new(BigInt::from(1), BigInt::from(2)),
        )
        .unwrap();
        assert!(matches!(
            AlgebraicNumber::new(p_cubic.clone(), zero_root_ball),
            Err(crate::algebraic::AlgebraicError::InvalidIsolatingInterval(
                _
            ))
        ));

        // Interval [0.5, 1.5] contains exactly root x=1 -> ACCEPTED!
        let isolating_ball = RealBall::new(
            BigRational::one(),
            BigRational::new(BigInt::from(1), BigInt::from(2)),
        )
        .unwrap();
        let mut root1 = AlgebraicNumber::new(p_cubic, isolating_ball).unwrap();
        assert_eq!(root1.sign(), 1);

        // Negative refine target radius must be rejected
        let neg_target = BigRational::from_integer(BigInt::from(-1));
        assert!(matches!(
            root1.refine_to_radius(&neg_target),
            Err(crate::algebraic::AlgebraicError::NegativeTargetRadius(_))
        ));
    }
}
