//! # fsym-simplify
//!
//! Algebraic simplification engine, rewrite pipelines, expansion, and rational canonicalization.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::Expr;
use num_bigint::BigInt;
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
            // Prepend the coefficient to the key's own factor list: a
            // product stays flat (Mul(-1, b, c)), never nested
            // (Mul(-1, Mul(b, c))). Matches the SymPy object model the
            // compatibility profile pins.
            match key {
                Expr::Mul(factors) => {
                    let mut parts = Vec::with_capacity(factors.len() + 1);
                    parts.push(rational_expr(coeff));
                    parts.extend(factors);
                    out.push(Expr::Mul(parts));
                }
                other => out.push(Expr::Mul(vec![rational_expr(coeff), other])),
            }
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
    #[error("Evaluation budget exhausted: {0}")]
    BudgetExhausted(BudgetError),
    #[error("Recursion depth limit exceeded: depth {0} > {MAX_RECURSION_DEPTH}")]
    DepthLimitExceeded(usize),
    #[error("Evaluation cancelled by owning region")]
    Cancelled,
}

impl SimplifyError {
    fn from_meter(e: MeterError) -> Self {
        match e {
            MeterError::Budget(b) => SimplifyError::BudgetExhausted(b),
            MeterError::Cancelled => SimplifyError::Cancelled,
        }
    }
}
/// Maximum syntactic nesting the recursive evaluation descent will traverse
/// before refusing with [`SimplifyError::DepthLimitExceeded`]. This bounds
/// native stack usage independently of the step budget, on every path
/// including the unbounded convenience forms (no parser-side nesting bound
/// exists yet). Sized to stay safe on default 2 MiB secondary threads.
pub const MAX_RECURSION_DEPTH: usize = 1024;

/// Work units charged per visited expression node.
const NODE_STEP: u64 = 1;

/// Shared completion rule for the unbounded convenience forms: only the
/// structural depth guard can fire under [`Unbounded`].
fn unwrap_unbounded(result: Result<Expr, SimplifyError>) -> Expr {
    match result {
        Ok(e) => e,
        Err(SimplifyError::DepthLimitExceeded(d)) => panic!(
            "expression nesting depth {d} exceeds MAX_RECURSION_DEPTH ({MAX_RECURSION_DEPTH}); \
             use simplify_with/expand_with for a typed refusal"
        ),
        Err(e) => unreachable!("unbounded meter cannot refuse: {e}"),
    }
}

/// Simplify an algebraic expression recursively.
///
/// Canonical form: like terms collected with exact rational coefficients,
/// factors sorted with identical factors folded into powers, numeric
/// constants merged (trailing in sums, leading in products).
///
/// Unbounded convenience form: runs the metered core under [`Unbounded`].
/// Callers owning an execution region should use [`simplify_with`].
pub fn simplify(expr: &Expr) -> Expr {
    unwrap_unbounded(simplify_with(expr, &mut Unbounded))
}
/// Simplify under a caller-owned budget/cancellation meter.
///
/// Charges [`Dimension::ComputeSteps`] per visited node, checks the region
/// safe point at every node, and refuses beyond [`MAX_RECURSION_DEPTH`]
/// nesting. A refusal aborts the whole evaluation: no partial result is
/// published.
pub fn simplify_with<M: BudgetMeter>(expr: &Expr, meter: &mut M) -> Result<Expr, SimplifyError> {
    simplify_at(expr, 0, meter)
}

fn simplify_at<M: BudgetMeter>(
    expr: &Expr,
    depth: usize,
    m: &mut M,
) -> Result<Expr, SimplifyError> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(SimplifyError::DepthLimitExceeded(depth));
    }
    m.checkpoint().map_err(SimplifyError::from_meter)?;
    m.charge(Dimension::ComputeSteps, NODE_STEP)
        .map_err(SimplifyError::from_meter)?;
    match expr {
        Expr::Add(terms) => {
            let mut simplified = Vec::with_capacity(terms.len());
            for t in terms {
                simplified.push(simplify_at(t, depth + 1, m)?);
            }
            Ok(collect_terms(simplified))
        }
        Expr::Mul(factors) => {
            let mut simplified = Vec::with_capacity(factors.len());
            for f in factors {
                simplified.push(simplify_at(f, depth + 1, m)?);
            }
            let mut coeff = BigRational::one();
            let mut rest: Vec<Expr> = Vec::new();
            for f in simplified {
                match numeric_of(&f) {
                    Some(q) => coeff *= q,
                    None => rest.push(f),
                }
            }
            // 0 * f is 0 only when f is defined; a negative-power factor may
            // be an undefined reciprocal (0^-1), so keep the structure for
            // the limit engine to classify.
            let has_neg_power = rest.iter().any(|f| {
                matches!(
                    f,
                    Expr::Pow(_, e) if matches!(e.as_ref(), Expr::Integer(n) if *n < BigInt::from(0))
                )
            });
            if coeff.is_zero() && !has_neg_power {
                return Ok(Expr::from_i64(0));
            }
            if rest.is_empty() {
                return Ok(rational_expr(coeff));
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
                    let folded = Expr::Pow(Arc::new(f), Arc::new(Expr::from_i64(count as i64)));
                    parts.push(simplify_at(&folded, depth + 1, m)?);
                }
            }
            if parts.len() == 1 {
                Ok(parts.pop().expect("len checked"))
            } else {
                Ok(Expr::Mul(parts))
            }
        }
        Expr::Pow(base, exp) => {
            let b = simplify_at(base, depth + 1, m)?;
            let e = simplify_at(exp, depth + 1, m)?;
            if e.is_zero() {
                Ok(Expr::from_i64(1))
            } else if e.is_one() {
                Ok(b)
            } else if let (Some(bv), Some(ev)) = (b.const_integer_value(), e.const_integer_value())
            {
                // Bounded constant-power fold: exponent must be a
                // non-negative small integer.
                match usize::try_from(ev) {
                    Ok(n) => Ok(Expr::Integer(num_traits::pow::pow(bv, n))),
                    Err(_) => Ok(Expr::Pow(Arc::new(b), Arc::new(e))),
                }
            } else {
                Ok(Expr::Pow(Arc::new(b), Arc::new(e)))
            }
        }
        Expr::Function(name, args) => {
            let mut simplified_args = Vec::with_capacity(args.len());
            for a in args {
                simplified_args.push(simplify_at(a, depth + 1, m)?);
            }
            // Exact values at rational points fold to rationals.
            if simplified_args.len() == 1
                && let Some(v) = numeric_of(&simplified_args[0])
            {
                let folded = match name.as_str() {
                    "sin" | "tan" if v.is_zero() => Some(BigRational::zero()),
                    "cos" | "exp" if v.is_zero() => Some(BigRational::one()),
                    "log" | "ln" if v == BigRational::one() => Some(BigRational::zero()),
                    _ => None,
                };
                if let Some(r) = folded {
                    return Ok(rational_expr(r));
                }
            }
            Ok(Expr::Function(name.clone(), simplified_args))
        }
        other => Ok(other.clone()),
    }
}

/// Additive term list of an expanded subexpression, metered like
/// [`simplify_at`]: every product/inner simplification charges the region.
fn expanded_terms_in<M: BudgetMeter>(
    expr: &Expr,
    depth: usize,
    m: &mut M,
) -> Result<Vec<Expr>, SimplifyError> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(SimplifyError::DepthLimitExceeded(depth));
    }
    m.checkpoint().map_err(SimplifyError::from_meter)?;
    m.charge(Dimension::ComputeSteps, NODE_STEP)
        .map_err(SimplifyError::from_meter)?;
    match expr {
        Expr::Add(terms) => {
            let mut out = Vec::new();
            for t in terms {
                out.extend(expanded_terms_in(t, depth + 1, m)?);
            }
            Ok(out)
        }
        Expr::Mul(factors) => {
            let mut acc: Vec<Expr> = vec![Expr::from_i64(1)];
            for f in factors {
                let f_terms = expanded_terms_in(f, depth + 1, m)?;
                let mut next = Vec::with_capacity(acc.len() * f_terms.len());
                for a in &acc {
                    for b in &f_terms {
                        let product = a.clone() * b.clone();
                        next.push(simplify_at(&product, depth + 1, m)?);
                    }
                }
                acc = next;
            }
            Ok(acc)
        }
        Expr::Pow(base, exp) => {
            let b_terms = expanded_terms_in(base, depth + 1, m)?;
            let e_simplified = simplify_at(exp, depth + 1, m)?;
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
                                let product = a.clone() * b.clone();
                                next.push(simplify_at(&product, depth + 1, m)?);
                            }
                        }
                        acc = next;
                    }
                    Ok(acc)
                }
                _ => {
                    let b = simplify_at(base, depth + 1, m)?;
                    Ok(vec![Expr::Pow(Arc::new(b), Arc::new(e_simplified))])
                }
            }
        }
        other => Ok(vec![other.clone()]),
    }
}

/// Expand under a caller-owned budget/cancellation meter. A refusal aborts
/// the expansion; no partial polynomial is published.
pub fn expand_with<M: BudgetMeter>(expr: &Expr, meter: &mut M) -> Result<Expr, SimplifyError> {
    let terms = expanded_terms_in(expr, 0, meter)?;
    Ok(collect_terms(terms))
}

/// Expand products of sums and bounded powers over sums, collecting the
/// result into canonical polynomial form.
///
/// Unbounded convenience form over [`expand_with`].
pub fn expand(expr: &Expr) -> Expr {
    unwrap_unbounded(expand_with(expr, &mut Unbounded))
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
            // Flat canonical Mul: SymPy keeps 2*x*y un-nested.
            Expr::Mul(vec![Expr::from_i64(2), x, y])
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
    /// Meter that refuses everything after `cancel_after` successful
    /// checkpoints: models an owning region cancelling mid-evaluation.
    struct CancellingMeter {
        remaining_steps: u64,
        cancel_after: u64,
        checkpoints_seen: usize,
    }

    impl CancellingMeter {
        fn new(remaining_steps: u64, cancel_after: u64) -> Self {
            Self {
                remaining_steps,
                cancel_after,
                checkpoints_seen: 0,
            }
        }
    }

    impl BudgetMeter for CancellingMeter {
        fn charge(&mut self, _d: Dimension, amount: u64) -> Result<(), MeterError> {
            if amount > self.remaining_steps {
                return Err(MeterError::Budget(fsym_budget::BudgetError::Exhausted {
                    dimension: Dimension::ComputeSteps,
                    requested: amount,
                    remaining: self.remaining_steps,
                }));
            }
            self.remaining_steps -= amount;
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints_seen += 1;
            if self.checkpoints_seen as u64 > self.cancel_after {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    fn product_of_sums() -> Expr {
        let (a, b, x, y) = (
            Expr::symbol("a"),
            Expr::symbol("b"),
            Expr::symbol("x"),
            Expr::symbol("y"),
        );
        Expr::Mul(vec![Expr::Add(vec![x, y]), Expr::Add(vec![a, b])])
    }

    #[test]
    fn budgeted_simplify_reports_exhaustion_atomically() {
        // Seven nodes must be charged for product_of_sums; four refuse first.
        let mut budget = fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(4, 0));
        let err = simplify_with(&product_of_sums(), &mut budget).unwrap_err();
        assert_eq!(
            err,
            SimplifyError::BudgetExhausted(fsym_budget::BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                requested: 1,
                remaining: 0,
            })
        );
        // Charges before the refusal are real ledger state, not rolled back
        // fiction: exactly the accepted steps were consumed.
        assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    }

    #[test]
    fn cancelled_region_stops_evaluation_at_safe_point() {
        // Cancel right after entering: far fewer checkpoints than the
        // expression has nodes.
        let mut m = CancellingMeter::new(10_000, 2);
        let err = simplify_with(&product_of_sums(), &mut m).unwrap_err();
        assert_eq!(err, SimplifyError::Cancelled);
        assert!(
            m.checkpoints_seen <= 3,
            "evaluation kept running after cancellation: {} checkpoints",
            m.checkpoints_seen
        );
    }

    #[test]
    fn expand_budget_refusal_publishes_no_partial_result() {
        let mut budget = fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(2, 0));
        let err = expand_with(&product_of_sums(), &mut budget).unwrap_err();
        assert!(matches!(
            err,
            SimplifyError::BudgetExhausted(fsym_budget::BudgetError::Exhausted { .. })
        ));
    }

    #[test]
    fn depth_limit_refuses_deep_nesting_at_structural_bound() {
        // Deep recursion needs a big-stack thread: this test exists precisely
        // because default 2 MiB stacks cannot carry MAX_RECURSION_DEPTH frames.
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let build = |n: usize| {
                    let mut deep = Expr::from_i64(1);
                    for _ in 0..n {
                        deep = Expr::Add(vec![deep, Expr::from_i64(1)]);
                    }
                    deep
                };
                // A chain nesting exactly MAX_RECURSION_DEPTH levels is legal.
                let ok = build(MAX_RECURSION_DEPTH);
                assert!(simplify_with(&ok, &mut Unbounded).is_ok());
                let _ = simplify(&ok);
                // One level deeper refuses with the typed structural error on
                // every entry point, before any budget is consulted.
                let too_deep = build(MAX_RECURSION_DEPTH + 1);
                let err = simplify_with(&too_deep, &mut Unbounded).unwrap_err();
                assert!(matches!(err, SimplifyError::DepthLimitExceeded(_)));
                let mut budget =
                    fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(10_000, 0));
                assert!(simplify_with(&too_deep, &mut budget).is_err());
            })
            .expect("spawn")
            .join()
            .expect("depth test thread");
    }

    #[test]
    fn legacy_wrappers_match_metered_results_on_small_inputs() {
        let e = product_of_sums();
        let mut budget = fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(100_000, 0));
        assert_eq!(simplify(&e), simplify_with(&e, &mut budget).unwrap());
        assert_eq!(expand(&e), expand_with(&e, &mut budget).unwrap());
        // Every node of both traversals was charged.
        assert!(budget.remaining(Dimension::ComputeSteps) < 100_000);
    }

    #[test]
    fn coefficient_times_multi_factor_key_stays_flat() {
        // Regression: the fra-4rm collection rebuild wrapped coefficients
        // around an already-multiplied key, emitting Mul(-1, Mul(b, c)).
        // Canonical products are flat, matching the SymPy object model and
        // every consumer that pins product shape (e.g. matrix determinants).
        let (a, b, c, d) = (
            Expr::symbol("a"),
            Expr::symbol("b"),
            Expr::symbol("c"),
            Expr::symbol("d"),
        );
        let ad_bc = Expr::Add(vec![
            Expr::Mul(vec![a.clone(), d.clone()]),
            Expr::Mul(vec![Expr::from_i64(-1), b.clone(), c.clone()]),
        ]);
        assert_eq!(
            simplify(&ad_bc),
            Expr::Add(vec![
                Expr::Mul(vec![a, d]),
                Expr::Mul(vec![Expr::from_i64(-1), b, c]),
            ])
        );
        // Rational coefficients flatten identically.
        let two_x_y = Expr::Mul(vec![
            Expr::from_i64(2),
            Expr::symbol("x"),
            Expr::symbol("y"),
        ]);
        assert_eq!(simplify(&two_x_y), two_x_y);
    }
}
