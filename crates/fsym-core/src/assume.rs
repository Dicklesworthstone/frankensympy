//! Scoped symbol assumption facts for algorithm lanes.
//!
//! Algorithm lanes (integration, limits, series) receive the caller's declared
//! symbol facts (`positive`, `nonzero`, ...) as an explicit scope:
//! [`with_symbol_facts`] binds them to exact symbol identities for one call on
//! the current thread and removes them on exit (including unwinding). The
//! legacy [`with_facts`] entry point binds only plain, untagged symbols.
//! Queries derive only exact structural consequences of consistent declared
//! facts; unknown or contradictory facts cannot discharge a side condition.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::HashMap;

use crate::{Constant, Expr};
use num_traits::{Signed, Zero};

/// Declared positive facts about one symbol; a false field means no declaration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    pub positive: bool,
    pub negative: bool,
    pub nonzero: bool,
    pub nonnegative: bool,
    pub nonpositive: bool,
    pub real: bool,
    /// Membership in the extended real line does not establish finiteness.
    pub extended_real: bool,
    pub integer: bool,
}

impl Facts {
    /// Whether these declarations have incompatible exact sign constraints.
    ///
    /// Both nonnegative and nonpositive consistently describe zero, unless
    /// nonzero is also declared. A positive/negative declaration conflicts
    /// with the opposite closed half-line even if this record was built
    /// directly instead of through [`Self::from_names`].
    ///
    /// Scoped records retain contradictions for diagnostics. The legacy
    /// Boolean/Option predicate helpers cannot express a contradictory
    /// outcome, so they refuse to discharge facts from such a record. Call
    /// this method on [`facts_of_symbol`]'s result to distinguish that refusal
    /// from an absent or merely unknown declaration.
    pub fn is_contradictory(self) -> bool {
        (self.positive && (self.negative || self.nonpositive))
            || (self.negative && self.nonnegative)
            || (self.nonzero && self.nonnegative && self.nonpositive)
    }

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
                "real" => f.real = true,
                "extended_real" => f.extended_real = true,
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
        if f.integer || f.nonzero || f.nonnegative || f.nonpositive {
            f.real = true;
        }
        if f.real {
            f.extended_real = true;
        }
        f
    }
}

thread_local! {
    static CONTEXT: RefCell<Vec<HashMap<crate::Symbol, Facts>>> = const { RefCell::new(Vec::new()) };
}

struct ScopeGuard;

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CONTEXT.with(|c| {
            c.borrow_mut().pop();
        });
    }
}

/// Runs `f` with `facts` keyed by full [`crate::Symbol`] (including [`crate::SymbolIdentity`]) in scope on this thread.
pub fn with_symbol_facts<R>(facts: HashMap<crate::Symbol, Facts>, f: impl FnOnce() -> R) -> R {
    CONTEXT.with(|c| c.borrow_mut().push(facts));
    let _guard = ScopeGuard;
    f()
}

/// Runs `f` with `facts` for plain, untagged symbols in scope on this thread.
///
/// A name does not authorize facts about a distinct identity-bearing atom
/// with the same printed name; use [`with_symbol_facts`] to name that atom.
pub fn with_facts<R>(facts: HashMap<String, Facts>, f: impl FnOnce() -> R) -> R {
    let sym_facts = facts
        .into_iter()
        .map(|(name, fact)| (crate::Symbol::new(name), fact))
        .collect();
    with_symbol_facts(sym_facts, f)
}

/// Facts declared for this exact `symbol` in the innermost scope that mentions it.
///
/// The raw record is retained even when [`Facts::is_contradictory`] is true.
/// Neither a plain name nor another identity-bearing atom with the same name
/// can shadow or supply this symbol's facts.
pub fn facts_of_symbol(symbol: &crate::Symbol) -> Option<Facts> {
    CONTEXT.with(|c| c.borrow().iter().rev().find_map(|m| m.get(symbol).copied()))
}

fn consistent_facts_of_symbol(symbol: &crate::Symbol) -> Option<Facts> {
    facts_of_symbol(symbol).filter(|facts| !facts.is_contradictory())
}

/// Facts declared for `name` in the innermost scope that mentions it.
pub fn facts_of(name: &str) -> Option<Facts> {
    let sym = crate::Symbol::new(name);
    CONTEXT.with(|c| c.borrow().iter().rev().find_map(|m| m.get(&sym).copied()))
}

// Predicate helpers cannot turn a held undefined intermediate into a fact.
// Analyze each child once: independently recursive finiteness/sign/nonzero
// guards otherwise revisit nested powers exponentially. These unmetered
// convenience queries have a fixed envelope and fail closed outside it.
const MAX_FACT_DEPTH: usize = 128;
const MAX_FACT_NODES: usize = 16_384;

#[derive(Clone, Copy, Default)]
struct StructuralFacts {
    sign: Option<i8>,
    real: bool,
    integer: bool,
    nonzero: bool,
    finite: bool,
}

fn inferred_facts(e: &Expr) -> StructuralFacts {
    let mut remaining = MAX_FACT_NODES;
    structural_facts(e, 0, &mut remaining).unwrap_or_default()
}

fn structural_facts(e: &Expr, depth: usize, remaining: &mut usize) -> Option<StructuralFacts> {
    if depth > MAX_FACT_DEPTH || *remaining == 0 {
        return None;
    }
    *remaining -= 1;
    let facts = match e {
        Expr::Integer(_) | Expr::Rational(_) => {
            let negative = match e {
                Expr::Integer(n) => n.is_negative(),
                Expr::Rational(r) => r.is_negative(),
                _ => unreachable!(),
            };
            StructuralFacts {
                sign: Some(if e.is_zero() {
                    0
                } else if negative {
                    -1
                } else {
                    1
                }),
                real: true,
                integer: matches!(e, Expr::Integer(_)),
                nonzero: !e.is_zero(),
                finite: true,
            }
        }
        Expr::Const(Constant::Pi | Constant::E) => StructuralFacts {
            sign: Some(1),
            real: true,
            finite: true,
            nonzero: true,
            ..StructuralFacts::default()
        },
        Expr::Const(Constant::I) => StructuralFacts {
            finite: true,
            nonzero: true,
            ..StructuralFacts::default()
        },
        // Limits need these literal signs. They grant no finite-domain fact
        // to a containing expression, such as 0*oo or oo**-1.
        Expr::Const(Constant::Infinity | Constant::NegativeInfinity) => StructuralFacts {
            sign: Some(if matches!(e, Expr::Const(Constant::Infinity)) {
                1
            } else {
                -1
            }),
            ..StructuralFacts::default()
        },
        Expr::Const(_) => StructuralFacts::default(),
        Expr::Sym(s) => {
            let Some(f) = consistent_facts_of_symbol(s) else {
                return Some(StructuralFacts::default());
            };
            let sign = if f.positive {
                Some(1)
            } else if f.negative {
                Some(-1)
            } else {
                None
            };
            let real = f.real
                || f.integer
                || f.nonzero
                || f.positive
                || f.negative
                || f.nonnegative
                || f.nonpositive;
            StructuralFacts {
                sign,
                real,
                integer: f.integer,
                nonzero: f.nonzero || sign.is_some(),
                finite: real,
            }
        }
        Expr::Add(args) | Expr::Mul(args) => {
            let product = matches!(e, Expr::Mul(_));
            let mut result = StructuralFacts {
                sign: Some(if product { 1 } else { 0 }),
                real: true,
                integer: true,
                nonzero: product,
                finite: true,
            };
            let (mut positive, mut negative, mut unknown) = (false, false, false);
            for arg in args {
                let f = structural_facts(arg, depth + 1, remaining)?;
                if !f.finite {
                    return Some(StructuralFacts::default());
                }
                result.real &= f.real;
                result.integer &= f.integer;
                if product {
                    result.sign = result.sign.zip(f.sign).map(|(a, b)| a * b);
                    result.nonzero &= f.nonzero;
                } else {
                    positive |= f.sign == Some(1);
                    negative |= f.sign == Some(-1);
                    unknown |= f.sign.is_none();
                }
            }
            if !product {
                result.sign = if unknown || positive && negative {
                    None
                } else if positive {
                    Some(1)
                } else if negative {
                    Some(-1)
                } else {
                    Some(0)
                };
                result.nonzero = matches!(result.sign, Some(-1 | 1));
            }
            result
        }
        Expr::Pow(base, exponent) => {
            let b = structural_facts(base, depth + 1, remaining)?;
            let x = structural_facts(exponent, depth + 1, remaining)?;
            if !b.finite || !x.finite {
                return Some(StructuralFacts::default());
            }
            let integer_exponent = match exponent.as_ref() {
                Expr::Integer(k) => Some(k),
                _ => None,
            };
            let zero_exponent = integer_exponent.is_some_and(Zero::is_zero);
            let nonnegative_integer = integer_exponent.is_some_and(|k| !k.is_negative());
            if !nonnegative_integer
                && !b.nonzero
                && !(b.sign == Some(0) && x.real && x.sign == Some(1))
            {
                return Some(StructuralFacts::default());
            }
            let sign = if zero_exponent || b.sign == Some(1) && x.real {
                Some(1)
            } else if b.sign == Some(-1) {
                integer_exponent.map(|k| {
                    if (k % crate::BigInt::from(2)).is_zero() {
                        1
                    } else {
                        -1
                    }
                })
            } else if b.sign == Some(0) && x.real && x.sign == Some(1) {
                Some(0)
            } else {
                None
            };
            StructuralFacts {
                sign,
                real: zero_exponent
                    || b.real && integer_exponent.is_some()
                    || b.sign == Some(1) && x.real
                    || sign == Some(0),
                integer: zero_exponent || b.integer && nonnegative_integer,
                nonzero: b.nonzero || zero_exponent,
                finite: true,
            }
        }
        Expr::Function(name, args) => {
            let [arg] = args.as_slice() else {
                return Some(StructuralFacts::default());
            };
            let a = structural_facts(arg, depth + 1, remaining)?;
            if !a.finite {
                return Some(StructuralFacts::default());
            }
            let finite = match name.as_str() {
                "exp" | "sin" | "cos" | "sinh" | "cosh" | "Abs" | "abs" | "sqrt" | "erf"
                | "erfc" | "sinc" | "re" | "im" | "conjugate" => true,
                "log" | "ln" => a.nonzero,
                "atan" | "tanh" => a.real,
                _ => false,
            };
            if !finite {
                return Some(StructuralFacts::default());
            }
            let real = match name.as_str() {
                "Abs" | "abs" | "re" | "im" => true,
                "log" | "ln" => a.sign == Some(1),
                "sqrt" => matches!(a.sign, Some(0 | 1)),
                _ => a.real,
            };
            let sign = match name.as_str() {
                "exp" if real => Some(1),
                "Abs" | "abs" if a.nonzero => Some(1),
                "Abs" | "abs" | "sqrt" if a.sign == Some(0) => Some(0),
                "sqrt" if a.sign == Some(1) => Some(1),
                "conjugate" => a.sign,
                _ => None,
            };
            StructuralFacts {
                sign,
                real,
                finite: true,
                nonzero: name == "exp"
                    || matches!(name.as_str(), "Abs" | "abs" | "sqrt" | "conjugate") && a.nonzero,
                ..StructuralFacts::default()
            }
        }
    };
    Some(facts)
}

/// Exact sign under the active facts: `Some(1)`, `Some(-1)`, `Some(0)`, or
/// `None` when the facts do not decide it. Literal real infinities keep their
/// signs, but a nonfinite or undefined child cannot establish an arithmetic sign.
pub fn sign(e: &Expr) -> Option<i8> {
    inferred_facts(e).sign
}

/// Whether `e` is defined and finite real under the active facts.
pub fn is_real(e: &Expr) -> bool {
    inferred_facts(e).real
}

/// Whether `e` is a defined integer under the active facts.
pub fn is_integer(e: &Expr) -> bool {
    inferred_facts(e).integer
}

/// Whether `e` is defined, finite and nonzero under the active facts.
pub fn is_nonzero(e: &Expr) -> bool {
    inferred_facts(e).nonzero
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
