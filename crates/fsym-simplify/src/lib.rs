//! # fsym-simplify
//!
//! Algebraic simplification engine, rewrite pipelines, expansion, and rational canonicalization.

#![forbid(unsafe_code)]

use fsym_core::Expr;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;

/// Exact numeric view of an expression leaf.
fn numeric_of(e: &Expr) -> Option<BigRational> {
    match e {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn rational_expr(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Integer(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

/// Split an additive term into `(coefficient, symbolic key)`, normalizing
/// key factor order so `y*x` and `x*y` collect together.
fn split_coeff(term: &Expr) -> (BigRational, Expr) {
    match term {
        Expr::Integer(n) => (BigRational::from_integer(n.clone()), Expr::from_i64(1)),
        Expr::Rational(r) => (r.clone(), Expr::from_i64(1)),
        Expr::Mul(factors) => {
            let mut coeff = BigRational::one();
            let mut rest: Vec<Expr> = Vec::new();
            for f in factors {
                match numeric_of(f) {
                    Some(q) => coeff *= q,
                    None => rest.push(f.clone()),
                }
            }
            if rest.len() > 1 {
                rest.sort();
            }
            let key = match rest.len() {
                0 => Expr::from_i64(1),
                1 => rest.pop().expect("len checked"),
                _ => Expr::Mul(rest),
            };
            (coeff, key)
        }
        other => (BigRational::one(), other.clone()),
    }
}

/// Canonicalize an additive term list: collect like terms by symbolic key
/// with exact rational coefficients, dropping zeros; numeric leaves merge
/// into a trailing constant.
fn collect_terms(mut terms: Vec<Expr>) -> Expr {
    let mut collected: BTreeMap<Expr, BigRational> = BTreeMap::new();
    let mut constant = BigRational::zero();
    for t in terms.drain(..) {
        let (coeff, key) = split_coeff(&t);
        if coeff.is_zero() {
            continue;
        }
        if key == Expr::from_i64(1) {
            constant += coeff;
        } else {
            *collected.entry(key).or_insert_with(BigRational::zero) += coeff;
        }
    }
    let mut out: Vec<Expr> = Vec::new();
    for (key, coeff) in collected {
        if coeff.is_zero() {
            continue;
        }
        if coeff.is_one() {
            out.push(key);
        } else {
            out.push(Expr::Mul(vec![rational_expr(coeff), key]));
        }
    }
    if !constant.is_zero() || out.is_empty() {
        out.push(rational_expr(constant));
    }
    if out.len() == 1 {
        out.pop().expect("len checked")
    } else {
        Expr::Add(out)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SimplifyError {
    #[error("Simplification error: {0}")]
    General(String),
}

/// Simplify an algebraic expression recursively.
///
/// Canonical form: like terms collected with exact rational coefficients,
/// factors sorted with identical factors folded into powers, numeric
/// constants merged (trailing in sums, leading in products).
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let simplified: Vec<Expr> = terms.iter().map(simplify).collect();
            collect_terms(simplified)
        }
        Expr::Mul(factors) => {
            let simplified: Vec<Expr> = factors.iter().map(simplify).collect();
            let mut coeff = BigRational::one();
            let mut rest: Vec<Expr> = Vec::new();
            for f in simplified {
                match numeric_of(&f) {
                    Some(q) => coeff *= q,
                    None => rest.push(f),
                }
            }
            if coeff.is_zero() {
                return Expr::from_i64(0);
            }
            if rest.is_empty() {
                return rational_expr(coeff);
            }
            let mut parts: Vec<Expr> = Vec::new();
            if !coeff.is_one() {
                parts.push(rational_expr(coeff));
            }
            rest.sort();
            let mut iter = rest.into_iter().peekable();
            while let Some(f) = iter.next() {
                let mut count = 1usize;
                while iter.peek() == Some(&f) {
                    count += 1;
                    iter.next();
                }
                if count == 1 {
                    parts.push(f);
                } else {
                    parts.push(simplify(&Expr::Pow(
                        Arc::new(f),
                        Arc::new(Expr::from_i64(count as i64)),
                    )));
                }
            }
            if parts.len() == 1 {
                parts.pop().expect("len checked")
            } else {
                Expr::Mul(parts)
            }
        }
        Expr::Pow(base, exp) => {
            let b = simplify(base);
            let e = simplify(exp);
            if e.is_zero() {
                Expr::from_i64(1)
            } else if e.is_one() {
                b
            } else if let (Some(bv), Some(ev)) = (b.const_integer_value(), e.const_integer_value())
            {
                // Bounded constant-power fold: exponent must be a
                // non-negative small integer.
                match usize::try_from(ev) {
                    Ok(n) => Expr::Integer(num_traits::pow::pow(bv, n)),
                    Err(_) => Expr::Pow(Arc::new(b), Arc::new(e)),
                }
            } else {
                Expr::Pow(Arc::new(b), Arc::new(e))
            }
        }
        other => other.clone(),
    }
}

/// Additive term list of an expanded subexpression.
fn expanded_terms(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Add(terms) => terms.iter().flat_map(expanded_terms).collect(),
        Expr::Mul(factors) => {
            let mut acc: Vec<Expr> = vec![Expr::from_i64(1)];
            for f in factors {
                let f_terms = expanded_terms(f);
                let mut next = Vec::with_capacity(acc.len() * f_terms.len());
                for a in &acc {
                    for b in &f_terms {
                        next.push(simplify(&(a.clone() * b.clone())));
                    }
                }
                acc = next;
            }
            acc
        }
        Expr::Pow(base, exp) => {
            let b_terms = expanded_terms(base);
            let e_simplified = simplify(exp);
            match e_simplified
                .const_integer_value()
                .and_then(|v| usize::try_from(v).ok())
            {
                // Distribute small non-negative integer powers over sums.
                Some(n) if n <= 8 && b_terms.len() > 1 => {
                    let mut acc: Vec<Expr> = vec![Expr::from_i64(1)];
                    for _ in 0..n {
                        let mut next = Vec::with_capacity(acc.len() * b_terms.len());
                        for a in &acc {
                            for b in &b_terms {
                                next.push(simplify(&(a.clone() * b.clone())));
                            }
                        }
                        acc = next;
                    }
                    acc
                }
                _ => vec![Expr::Pow(Arc::new(simplify(base)), Arc::new(e_simplified))],
            }
        }
        other => vec![other.clone()],
    }
}

/// Expand products of sums and bounded powers over sums, collecting the
/// result into canonical polynomial form.
pub fn expand(expr: &Expr) -> Expr {
    collect_terms(expanded_terms(expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_basic() {
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let e = Expr::Add(vec![x.clone(), zero, Expr::from_i64(5), Expr::from_i64(3)]);
        let s = simplify(&e);
        assert_eq!(s, Expr::Add(vec![x, Expr::from_i64(8)]));
    }

    #[test]
    fn test_simplify_mul_zero() {
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let e = Expr::Mul(vec![x, zero]);
        let s = simplify(&e);
        assert_eq!(s, Expr::from_i64(0));
    }

    #[test]
    fn test_like_term_collection() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        // x + x = 2x
        assert_eq!(
            simplify(&Expr::Add(vec![x.clone(), x.clone()])),
            Expr::Mul(vec![Expr::from_i64(2), x.clone()])
        );
        // 3y + 2y - y = 4y (negative via -1 factor head)
        let e = Expr::Mul(vec![Expr::from_i64(-1), y.clone()]);
        let sum = Expr::Add(vec![
            Expr::Mul(vec![Expr::from_i64(3), y.clone()]),
            Expr::Mul(vec![Expr::from_i64(2), y.clone()]),
            e,
        ]);
        assert_eq!(
            simplify(&sum),
            Expr::Mul(vec![Expr::from_i64(4), y.clone()])
        );
        // Commutative keys: x*y + y*x = 2xy
        let xy = Expr::Mul(vec![x.clone(), y.clone()]);
        let yx = Expr::Mul(vec![y.clone(), x.clone()]);
        assert_eq!(
            simplify(&Expr::Add(vec![xy, yx])),
            Expr::Mul(vec![Expr::from_i64(2), Expr::Mul(vec![x, y])])
        );
    }

    #[test]
    fn test_expand_product_of_sums_structural() {
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let c = Expr::symbol("c");
        let d = Expr::symbol("d");
        let e = Expr::Mul(vec![
            Expr::Add(vec![a.clone(), b]),
            Expr::Add(vec![c.clone(), d]),
        ]);
        match expand(&e) {
            Expr::Add(terms) => {
                assert_eq!(terms.len(), 4, "ac + ad + bc + bd, got {terms:?}");
            }
            other => panic!("expected Add, got {other}"),
        }
    }

    /// Metamorphic relation: expansion preserves numeric value at probe points.
    fn assert_expansion_equivalent(original: &Expr, probes: &[(i64, i64)]) {
        let expanded = expand(original);
        for (px, py) in probes {
            let env = std::collections::HashMap::from([
                (fsym_core::Symbol::new("x"), Expr::from_i64(*px)),
                (fsym_core::Symbol::new("y"), Expr::from_i64(*py)),
            ]);
            let before = original.subs(&env).evalf().unwrap();
            let after = expanded.subs(&env).evalf().unwrap();
            assert!(
                (before - after).abs() < 1e-9,
                "expansion changed value at ({px},{py}): {before} vs {after}"
            );
        }
    }

    #[test]
    fn test_expand_square_of_sum_is_value_preserving() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let original = Expr::Pow(
            Arc::new(Expr::Add(vec![x.clone(), y.clone()])),
            Arc::new(Expr::from_i64(2)),
        );
        assert_expansion_equivalent(&original, &[(2, 3), (-1, 4), (0, 0), (7, -5)]);
        // Canonical polynomial form has exactly three collected terms.
        match expand(&original) {
            Expr::Add(terms) => assert_eq!(terms.len(), 3),
            other => panic!("expected Add, got {other}"),
        }
    }

    #[test]
    fn test_expand_cubic_mixed_product_is_value_preserving() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let sum = Expr::Add(vec![x.clone(), Expr::from_i64(1)]);
        let diff = Expr::Add(vec![y.clone(), Expr::from_i64(-1)]);
        let original = Expr::Mul(vec![sum.clone(), sum.clone(), diff]);
        assert_expansion_equivalent(&original, &[(3, 2), (-2, 5), (1, 1), (10, -10)]);
    }
}
