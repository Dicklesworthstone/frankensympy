//! Scoped symbol assumption facts for algorithm lanes.
//!
//! String-keyed algorithm lanes (integration, limits, series) receive the
//! caller's declared symbol facts (`positive`, `nonzero`, ...) as an explicit
//! scope: [`with_facts`] installs them for the duration of one call on the
//! current thread and removes them on exit (including unwinding). Queries
//! derive only exact structural consequences of the declared facts; an
//! unknown fact is never promoted to true or false.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::HashMap;

use crate::{Constant, Expr};
use num_traits::{Signed, Zero};

/// Declared facts about one symbol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    pub positive: bool,
    pub negative: bool,
    pub nonzero: bool,
    pub nonnegative: bool,
    pub nonpositive: bool,
    pub real: bool,
    pub integer: bool,
}

impl Facts {
    /// Facts from upstream assumption names; unknown names are ignored.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Self {
        let mut f = Facts::default();
        for n in names {
            match n.as_ref() {
                "positive" => f.positive = true,
                "negative" => f.negative = true,
                "nonzero" => f.nonzero = true,
                "nonnegative" => f.nonnegative = true,
                "nonpositive" => f.nonpositive = true,
                "real" | "extended_real" => f.real = true,
                "integer" => f.integer = true,
                _ => {}
            }
        }
        // Implied facts.
        if f.positive || f.negative {
            f.nonzero = true;
            f.real = true;
        }
        if f.positive {
            f.nonnegative = true;
        }
        if f.negative {
            f.nonpositive = true;
        }
        if f.integer || f.nonnegative || f.nonpositive {
            f.real = true;
        }
        f
    }
}

thread_local! {
    static CONTEXT: RefCell<Vec<HashMap<String, Facts>>> = const { RefCell::new(Vec::new()) };
}

struct ScopeGuard;

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CONTEXT.with(|c| {
            c.borrow_mut().pop();
        });
    }
}

/// Runs `f` with `facts` in scope on this thread.
pub fn with_facts<R>(facts: HashMap<String, Facts>, f: impl FnOnce() -> R) -> R {
    CONTEXT.with(|c| c.borrow_mut().push(facts));
    let _guard = ScopeGuard;
    f()
}

/// Facts declared for `name` in the innermost scope that mentions it.
pub fn facts_of(name: &str) -> Option<Facts> {
    CONTEXT.with(|c| c.borrow().iter().rev().find_map(|m| m.get(name).copied()))
}

/// Exact sign under the active facts: `Some(1)`, `Some(-1)`, `Some(0)`, or
/// `None` when the facts do not decide it.
pub fn sign(e: &Expr) -> Option<i8> {
    match e {
        Expr::Integer(n) => Some(if n.is_zero() {
            0
        } else if n.is_negative() {
            -1
        } else {
            1
        }),
        Expr::Rational(r) => Some(if r.is_zero() {
            0
        } else if r.is_negative() {
            -1
        } else {
            1
        }),
        Expr::Const(Constant::Pi | Constant::E | Constant::Infinity) => Some(1),
        Expr::Const(Constant::NegativeInfinity) => Some(-1),
        Expr::Const(_) => None,
        Expr::Sym(s) => {
            let f = facts_of(&s.name)?;
            if f.positive {
                Some(1)
            } else if f.negative {
                Some(-1)
            } else {
                None
            }
        }
        Expr::Mul(xs) => xs.iter().try_fold(1i8, |acc, x| Some(acc * sign(x)?)),
        Expr::Add(xs) => {
            let signs: Option<Vec<i8>> = xs.iter().map(sign).collect();
            let signs = signs?;
            if signs.iter().all(|s| *s >= 0) && signs.contains(&1) {
                Some(1)
            } else if signs.iter().all(|s| *s <= 0) && signs.contains(&-1) {
                Some(-1)
            } else if signs.iter().all(|s| *s == 0) {
                Some(0)
            } else {
                None
            }
        }
        Expr::Pow(b, x) => {
            let sb = sign(b)?;
            match (sb, x.as_ref()) {
                (1, _) if is_real(x) => Some(1),
                (-1, Expr::Integer(k)) => Some(if (k % num_bigint_two()).is_zero() {
                    1
                } else {
                    -1
                }),
                (0, Expr::Integer(k)) if k.is_positive() => Some(0),
                _ => None,
            }
        }
        Expr::Function(name, args) if name == "exp" && args.len() == 1 && is_real(&args[0]) => {
            Some(1)
        }
        Expr::Function(_, _) => None,
    }
}

fn num_bigint_two() -> crate::BigInt {
    crate::BigInt::from(2)
}

/// Whether `e` is real under the active facts (structural, exact).
pub fn is_real(e: &Expr) -> bool {
    match e {
        Expr::Integer(_) | Expr::Rational(_) => true,
        Expr::Const(c) => matches!(c, Constant::Pi | Constant::E),
        Expr::Sym(s) => facts_of(&s.name).is_some_and(|f| f.real),
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().all(is_real),
        Expr::Pow(b, x) => {
            is_real(b)
                && match x.as_ref() {
                    Expr::Integer(_) => true,
                    _ => sign(b) == Some(1) && is_real(x),
                }
        }
        Expr::Function(name, args) => {
            args.len() == 1
                && is_real(&args[0])
                && matches!(
                    name.as_str(),
                    "exp" | "sin" | "cos" | "atan" | "sinh" | "cosh" | "tanh" | "Abs"
                )
        }
    }
}

/// Whether `e` is nonzero under the active facts (structural, exact).
pub fn is_nonzero(e: &Expr) -> bool {
    if let Some(s) = sign(e) {
        return s != 0;
    }
    match e {
        Expr::Sym(s) => facts_of(&s.name).is_some_and(|f| f.nonzero),
        Expr::Mul(xs) => xs.iter().all(is_nonzero),
        Expr::Pow(b, x) => {
            is_nonzero(b) && matches!(x.as_ref(), Expr::Integer(_) | Expr::Rational(_))
        }
        Expr::Function(name, args) if name == "exp" && args.len() == 1 => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(pairs: &[(&str, &[&str])]) -> HashMap<String, Facts> {
        pairs
            .iter()
            .map(|(n, fs)| (n.to_string(), Facts::from_names(fs)))
            .collect()
    }

    #[test]
    fn scoped_facts_decide_signs_and_vanish_after_scope() {
        let a = Expr::symbol("a");
        let e = Expr::Mul(vec![Expr::from_i64(-2), a.clone()]);
        let inside = with_facts(facts(&[("a", &["positive"])]), || {
            (sign(&e), is_nonzero(&a))
        });
        assert_eq!(inside, (Some(-1), true));
        assert_eq!(sign(&e), None);
        assert!(!is_nonzero(&a));
    }

    #[test]
    fn unknown_facts_stay_unknown() {
        let a = Expr::symbol("a");
        let r = with_facts(facts(&[("a", &["real"])]), || {
            (sign(&a), is_nonzero(&a), is_real(&a))
        });
        assert_eq!(r, (None, false, true));
    }

    #[test]
    fn nested_scopes_prefer_innermost() {
        let a = Expr::symbol("a");
        let r = with_facts(facts(&[("a", &["positive"])]), || {
            with_facts(facts(&[("a", &["negative"])]), || sign(&a))
        });
        assert_eq!(r, Some(-1));
    }
}
