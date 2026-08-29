//! # fsym-simplify
//!
//! Algebraic simplification engine, rewrite pipelines, expansion, and an explicitly verified
//! simplification entry point (WS07). Ordinary convenience functions return values only;
//! `verified_simplify` additionally emits an independently replayed kernel derivation.

#![forbid(unsafe_code)]

pub mod rewrite;

pub use rewrite::*;

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Expr};
use fsym_evidence::{EvidenceEnvelope, VerificationReceipt};
use fsym_id::ReceiptId;
use fsym_outcome::EvidenceClass;
use fsym_proof_kernel::{Claim, ProofKernel, verify_derivation_independent};
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

pub(crate) fn rational_expr(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Integer(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

/// Evaluates `0**exponent` only when the exponent's relevant class is explicit.
///
/// An uninterpreted expression may later be positive, negative, zero, or non-real,
/// so treating every non-literal-negative exponent as positive would turn a
/// conditional power into an unconditional zero.
fn zero_base_power_value(exponent: &Expr) -> Option<Expr> {
    use fsym_core::Constant;

    match exponent {
        Expr::Integer(value) => Some(if value.is_negative() {
            Expr::Const(Constant::ComplexInfinity)
        } else if value.is_zero() {
            Expr::from_i64(1)
        } else {
            Expr::from_i64(0)
        }),
        Expr::Rational(value) => Some(if value.numer().is_negative() {
            Expr::Const(Constant::ComplexInfinity)
        } else if value.is_zero() {
            Expr::from_i64(1)
        } else {
            Expr::from_i64(0)
        }),
        Expr::Const(Constant::Pi | Constant::E | Constant::Infinity) => Some(Expr::from_i64(0)),
        Expr::Const(Constant::NegativeInfinity) => Some(Expr::Const(Constant::ComplexInfinity)),
        Expr::Const(Constant::I | Constant::ComplexInfinity | Constant::NaN) => {
            Some(Expr::Const(Constant::NaN))
        }
        _ => None,
    }
}

fn is_explicitly_non_finite_exponent(exponent: &Expr) -> bool {
    matches!(
        exponent,
        Expr::Const(
            fsym_core::Constant::Infinity
                | fsym_core::Constant::NegativeInfinity
                | fsym_core::Constant::ComplexInfinity
                | fsym_core::Constant::NaN
        )
    )
}

pub(crate) fn is_total_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) => true,
        Expr::Const(fsym_core::Constant::Pi | fsym_core::Constant::E | fsym_core::Constant::I) => {
            true
        }
        Expr::Const(_) => false,
        Expr::Function(name, args) => {
            matches!(name.as_str(), "exp" | "sin" | "cos" | "sinh" | "cosh")
                && args.iter().all(is_total_expr)
        }
        Expr::Add(terms) | Expr::Mul(terms) => terms.iter().all(is_total_expr),
        Expr::Pow(base, exponent) => {
            is_total_expr(base)
                && matches!(
                    exponent.as_ref(),
                    Expr::Integer(value) if !value.is_negative()
                )
        }
    }
}

/// Split an additive term into `(coefficient, symbolic key)`, normalizing
/// key factor order so `y*x` and `x*y` collect together.
pub(crate) fn split_coeff(term: &Expr) -> (BigRational, Expr) {
    match term {
        Expr::Integer(n) => (BigRational::from_integer(n.clone()), Expr::from_i64(1)),
        Expr::Rational(r) => (r.clone(), Expr::from_i64(1)),
        Expr::Mul(factors) => {
            let mut coeff = BigRational::one();
            let mut rest: Vec<Expr> = Vec::new();
            let mut stack: Vec<&Expr> = factors.iter().collect();
            while let Some(f) = stack.pop() {
                match f {
                    Expr::Mul(nested) => stack.extend(nested.iter()),
                    _ => match numeric_of(f) {
                        Some(q) => coeff *= q,
                        None => rest.push(f.clone()),
                    },
                }
            }
            // A zero coefficient annihilates only the total polynomial fragment. Preserve the
            // original term as an opaque additive key when a partial/indeterminate factor is
            // present; otherwise a surrounding Add would still erase `0 * Infinity`, `0/x`,
            // or `0 * log(x)` after the Mul simplifier correctly retained it.
            if coeff.is_zero() && !rest.iter().all(is_total_expr) {
                return (BigRational::one(), term.clone());
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
fn collect_terms(terms: Vec<Expr>) -> Expr {
    let mut collected: BTreeMap<Expr, BigRational> = BTreeMap::new();
    let mut constant = BigRational::zero();
    let mut stack: Vec<Expr> = terms;
    while let Some(t) = stack.pop() {
        match t {
            Expr::Add(nested) => stack.extend(nested),
            _ => {
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
        }
    }
    let mut out_pure: Vec<Expr> = Vec::new();
    let mut out_scaled: Vec<Expr> = Vec::new();
    for (key, coeff) in collected {
        if coeff.is_zero() {
            continue;
        }
        if coeff.is_one() {
            out_pure.push(key);
        } else {
            match key {
                Expr::Mul(factors) => {
                    let mut parts = Vec::with_capacity(factors.len() + 1);
                    parts.push(rational_expr(coeff));
                    parts.extend(factors);
                    out_scaled.push(Expr::Mul(parts));
                }
                other => out_scaled.push(Expr::Mul(vec![rational_expr(coeff), other])),
            }
        }
    }
    let mut out = out_pure;
    out.extend(out_scaled);
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
    #[error("Verification proof error: {0}")]
    ProofFailed(String),
}

impl SimplifyError {
    fn from_meter(e: MeterError) -> Self {
        match e {
            MeterError::Budget(b) => SimplifyError::BudgetExhausted(b),
            MeterError::Cancelled => SimplifyError::Cancelled,
        }
    }
}

pub const MAX_RECURSION_DEPTH: usize = 1024;
const NODE_STEP: u64 = 1;
const MAX_INPUT_FANOUT: usize = 262_144;
const MAX_EXPANDED_TERMS: usize = 4_096;

fn check_fanout(actual: usize) -> Result<(), SimplifyError> {
    if actual > MAX_INPUT_FANOUT {
        Err(SimplifyError::General(format!(
            "expression fanout {actual} exceeds the limit of {MAX_INPUT_FANOUT}"
        )))
    } else {
        Ok(())
    }
}

fn unwrap_legacy(result: Result<Expr, SimplifyError>, fallback: &Expr) -> Expr {
    match result {
        Ok(e) => e,
        Err(_) => fallback.clone(),
    }
}

/// Simplify an algebraic expression recursively, returning resource and shape refusals.
pub fn try_simplify(expr: &Expr) -> Result<Expr, SimplifyError> {
    simplify_with(expr, &mut Unbounded)
}

/// Simplify an algebraic expression recursively.
///
/// This compatibility convenience falls back to the original unsimplified expression
/// when fixed structural limits reject the input. Trust-boundary callers should use
/// [`try_simplify`] or [`simplify_with`] for typed refusals.
pub fn simplify(expr: &Expr) -> Expr {
    unwrap_legacy(try_simplify(expr), expr)
}

/// Simplify under a caller-owned budget/cancellation meter.
pub fn simplify_with<M: BudgetMeter>(expr: &Expr, meter: &mut M) -> Result<Expr, SimplifyError> {
    simplify_counting_folds(expr, meter).map(|(simplified, _)| simplified)
}

fn simplify_counting_folds<M: BudgetMeter>(
    expr: &Expr,
    m: &mut M,
) -> Result<(Expr, u64), SimplifyError> {
    let mut folds = 0u64;
    let simplified = simplify_at(expr, 0, m, &mut folds)?;
    Ok((simplified, folds))
}

fn simplify_at<M: BudgetMeter>(
    expr: &Expr,
    depth: usize,
    m: &mut M,
    folds: &mut u64,
) -> Result<Expr, SimplifyError> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(SimplifyError::DepthLimitExceeded(depth));
    }
    m.checkpoint().map_err(SimplifyError::from_meter)?;
    m.charge(Dimension::ComputeSteps, NODE_STEP)
        .map_err(SimplifyError::from_meter)?;
    match expr {
        Expr::Add(terms) => {
            check_fanout(terms.len())?;
            let mut simplified = Vec::with_capacity(terms.len());
            for t in terms {
                simplified.push(simplify_at(t, depth + 1, m, folds)?);
            }
            let collected = collect_terms(simplified);
            let folded = match &collected {
                Expr::Add(fold_terms) => {
                    rewrite::fold_pythagorean_terms(fold_terms).map(|folded| {
                        *folds += 1;
                        collect_terms(folded)
                    })
                }
                _ => None,
            };
            Ok(folded.unwrap_or(collected))
        }
        Expr::Mul(factors) => {
            check_fanout(factors.len())?;
            let mut simplified = Vec::with_capacity(factors.len());
            for f in factors {
                simplified.push(simplify_at(f, depth + 1, m, folds)?);
            }
            let mut coeff = BigRational::one();
            let mut rest: Vec<Expr> = Vec::new();
            let mut stack: Vec<Expr> = simplified;
            while let Some(f) = stack.pop() {
                match f {
                    Expr::Mul(nested) => stack.extend(nested),
                    _ => match numeric_of(&f) {
                        Some(q) => coeff *= q,
                        None => rest.push(f),
                    },
                }
            }
            if coeff.is_zero() && rest.iter().all(is_total_expr) {
                return Ok(Expr::from_i64(0));
            }

            // Combine exponential factors: exp(a) * exp(b) -> exp(a + b)
            let mut exp_args: Vec<Expr> = Vec::new();
            let mut other_factors: Vec<Expr> = Vec::new();
            for f in rest {
                if let Expr::Function(name, args) = &f
                    && name == "exp"
                    && args.len() == 1
                {
                    exp_args.push(args[0].clone());
                } else {
                    other_factors.push(f);
                }
            }
            if exp_args.len() > 1 {
                let sum_args = collect_terms(exp_args);
                let simplified_exp = simplify_at(
                    &Expr::Function("exp".to_string(), vec![sum_args]),
                    depth + 1,
                    m,
                    folds,
                )?;
                if !simplified_exp.is_one() {
                    other_factors.push(simplified_exp);
                }
                rest = other_factors;
            } else if exp_args.len() == 1 {
                other_factors.push(Expr::Function("exp".to_string(), exp_args));
                rest = other_factors;
            } else {
                rest = other_factors;
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
                    parts.push(simplify_at(&folded, depth + 1, m, folds)?);
                }
            }
            if parts.len() == 1 {
                Ok(parts.pop().expect("len checked"))
            } else {
                Ok(Expr::Mul(parts))
            }
        }
        Expr::Pow(base, exp) => {
            let b = simplify_at(base, depth + 1, m, folds)?;
            let e = simplify_at(exp, depth + 1, m, folds)?;
            if e.is_zero() {
                Ok(Expr::from_i64(1))
            } else if e.is_one() {
                Ok(b)
            } else if b.is_zero() {
                Ok(
                    zero_base_power_value(&e)
                        .unwrap_or_else(|| Expr::Pow(Arc::new(b), Arc::new(e))),
                )
            } else if b.is_one() {
                if is_explicitly_non_finite_exponent(&e) {
                    Ok(Expr::Const(fsym_core::Constant::NaN))
                } else {
                    Ok(Expr::from_i64(1))
                }
            } else {
                match (b, e) {
                    (Expr::Integer(bn), Expr::Integer(en)) => {
                        if en.is_negative() {
                            if let Ok(exp_abs) = usize::try_from(&-en.clone())
                                && exp_abs <= 1000
                            {
                                let exponent = u32::try_from(exp_abs).map_err(|_| {
                                    SimplifyError::General(
                                        "bounded exponent failed u32 conversion".to_string(),
                                    )
                                })?;
                                let denom = bn.pow(exponent);
                                if denom.is_zero() {
                                    return Ok(Expr::Const(fsym_core::Constant::ComplexInfinity));
                                }
                                return Ok(Expr::Rational(BigRational::new(
                                    fsym_core::BigInt::from(1),
                                    denom,
                                )));
                            }
                        } else if let Ok(exp_usize) = usize::try_from(&en)
                            && exp_usize <= 1000
                        {
                            let exponent = u32::try_from(exp_usize).map_err(|_| {
                                SimplifyError::General(
                                    "bounded exponent failed u32 conversion".to_string(),
                                )
                            })?;
                            return Ok(Expr::Integer(bn.pow(exponent)));
                        }
                        Ok(Expr::Pow(
                            Arc::new(Expr::Integer(bn)),
                            Arc::new(Expr::Integer(en)),
                        ))
                    }
                    (sb, se) => Ok(Expr::Pow(Arc::new(sb), Arc::new(se))),
                }
            }
        }
        Expr::Function(name, args) => {
            check_fanout(args.len())?;
            let mut simplified_args = Vec::with_capacity(args.len());
            for a in args {
                simplified_args.push(simplify_at(a, depth + 1, m, folds)?);
            }
            if simplified_args.len() == 1 && simplified_args[0].is_zero() {
                match name.as_str() {
                    "sin" | "tan" | "sinh" | "tanh" | "asin" | "atan" | "asinh" | "atanh" => {
                        return Ok(Expr::from_i64(0));
                    }
                    "cos" | "cosh" | "exp" => return Ok(Expr::from_i64(1)),
                    _ => {}
                }
            }
            if simplified_args.len() == 1 && simplified_args[0].is_one() {
                match name.as_str() {
                    "acos" | "acosh" | "ln" | "log" => return Ok(Expr::from_i64(0)),
                    _ => {}
                }
            }
            Ok(Expr::Function(name.clone(), simplified_args))
        }
        _ => Ok(expr.clone()),
    }
}

/// Simplifies an expression and produces an independently replayed derivation receipt (WS07).
pub fn verified_simplify<M: BudgetMeter>(
    expr: &Expr,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    receipt_id: ReceiptId,
    meter: &mut M,
) -> Result<(Expr, EvidenceEnvelope), SimplifyError> {
    let (simplified, fold_count) = simplify_counting_folds(expr, meter)?;
    let claim = Claim::equality(expr.clone(), simplified.clone());
    let mut kernel = ProofKernel::new((**context).clone());

    let step_id = if expr == &simplified {
        kernel
            .prove_reflexivity(expr.clone(), meter)
            .map_err(|e| SimplifyError::ProofFailed(e.to_string()))?
    } else {
        let rule_name = if fold_count > 0 {
            // Pythagorean pair cancellation is the only assumption-free
            // identity reduction simplify_at performs beyond like-term
            // collection, so its presence selects the dedicated kernel rule.
            "pythagorean_identity"
        } else {
            match expr {
                Expr::Function(name, args)
                    if args.len() == 1
                        && args[0].is_zero()
                        && matches!(
                            name.as_str(),
                            "sin"
                                | "cos"
                                | "tan"
                                | "sinh"
                                | "cosh"
                                | "tanh"
                                | "exp"
                                | "asin"
                                | "atan"
                                | "asinh"
                                | "atanh"
                        ) =>
                {
                    "elementary_zero_eval"
                }
                Expr::Function(name, args)
                    if args.len() == 1
                        && args[0].is_one()
                        && matches!(name.as_str(), "acos" | "acosh" | "ln" | "log") =>
                {
                    "elementary_one_eval"
                }
                _ => "simplify_normal_form",
            }
        };
        kernel
            .prove_definitional_reduction(expr.clone(), simplified.clone(), rule_name, meter)
            .map_err(|e| SimplifyError::ProofFailed(e.to_string()))?
    };

    let derivation_tree = kernel
        .export_derivation(step_id)
        .map_err(|e| SimplifyError::ProofFailed(e.to_string()))?;

    // Independent reference verification check
    let verified_claim = verify_derivation_independent(&derivation_tree, context)
        .map_err(|e| SimplifyError::ProofFailed(format!("Independent verifier rejected: {e}")))?;
    if verified_claim != claim {
        return Err(SimplifyError::ProofFailed(format!(
            "Independent verifier established `{verified_claim}`, expected `{claim}`"
        )));
    }

    let receipt = VerificationReceipt::issue(
        receipt_id,
        &claim,
        EvidenceClass::KernelProved,
        "fsym-simplify.v1",
        receipt_id.raw(),
        Some(derivation_tree.digest()),
    );

    let envelope = EvidenceEnvelope::new(
        claim,
        EvidenceClass::KernelProved,
        receipt,
        Some(derivation_tree),
    );
    if !envelope.verify_integrity() {
        return Err(SimplifyError::ProofFailed(
            "constructed evidence envelope failed its structural integrity check".to_string(),
        ));
    }
    Ok((simplified, envelope))
}

/// Expand polynomial products and powers, returning resource and shape refusals.
pub fn try_expand(expr: &Expr) -> Result<Expr, SimplifyError> {
    expand_with(expr, &mut Unbounded)
}

/// Expand polynomial products and powers into sum-of-products normal form.
///
/// This compatibility convenience falls back to the original unexpanded expression
/// when fixed structural limits reject the input. Trust-boundary callers should use
/// [`try_expand`] or [`expand_with`] for typed refusals.
pub fn expand(expr: &Expr) -> Expr {
    unwrap_legacy(try_expand(expr), expr)
}

/// Expand under a caller-owned budget/cancellation meter.
pub fn expand_with<M: BudgetMeter>(expr: &Expr, meter: &mut M) -> Result<Expr, SimplifyError> {
    expand_at(expr, 0, meter)
}

fn expand_at<M: BudgetMeter>(expr: &Expr, depth: usize, m: &mut M) -> Result<Expr, SimplifyError> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(SimplifyError::DepthLimitExceeded(depth));
    }
    m.checkpoint().map_err(SimplifyError::from_meter)?;
    m.charge(Dimension::ComputeSteps, NODE_STEP)
        .map_err(SimplifyError::from_meter)?;

    match expr {
        Expr::Add(terms) => {
            check_fanout(terms.len())?;
            let mut expanded_terms = Vec::with_capacity(terms.len());
            for t in terms {
                expanded_terms.push(expand_at(t, depth + 1, m)?);
            }
            Ok(collect_terms(expanded_terms))
        }
        Expr::Mul(factors) => {
            check_fanout(factors.len())?;
            let mut current = vec![Expr::from_i64(1)];
            for f in factors {
                let ef = expand_at(f, depth + 1, m)?;
                let next_terms = match ef {
                    Expr::Add(ts) => ts,
                    other => vec![other],
                };
                let product_count =
                    current.len().checked_mul(next_terms.len()).ok_or_else(|| {
                        SimplifyError::General(
                            "expanded term-count multiplication overflowed".to_string(),
                        )
                    })?;
                if product_count > MAX_EXPANDED_TERMS {
                    return Err(SimplifyError::General(format!(
                        "expansion exceeds the term limit of {MAX_EXPANDED_TERMS}"
                    )));
                }
                let mut product_terms = Vec::with_capacity(product_count);
                for a in &current {
                    for b in &next_terms {
                        product_terms
                            .push(simplify_with(&Expr::Mul(vec![a.clone(), b.clone()]), m)?);
                    }
                }
                current = product_terms;
            }
            Ok(collect_terms(current))
        }
        Expr::Pow(base, exp) => {
            let eb = expand_at(base, depth + 1, m)?;
            let ee = expand_at(exp, depth + 1, m)?;
            if let Expr::Integer(n) = &ee
                && let Ok(k) = usize::try_from(n)
                && (2..=16).contains(&k)
            {
                let mut factors = Vec::with_capacity(k);
                for _ in 0..k {
                    factors.push(eb.clone());
                }
                expand_at(&Expr::Mul(factors), depth + 1, m)
            } else {
                Ok(Expr::Pow(Arc::new(eb), Arc::new(ee)))
            }
        }
        _ => Ok(expr.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_assumptions::{AssumptionsContext, Predicate};
    use fsym_core::Symbol;
    use fsym_proof_kernel::ProofRule;

    #[test]
    fn test_simplify_basic() {
        let x = Expr::symbol("x");
        let expr = Expr::Add(vec![
            x.clone(),
            x.clone(),
            Expr::from_i64(3),
            Expr::from_i64(2),
        ]);
        let s = simplify(&expr);
        assert_eq!(
            s,
            Expr::Add(vec![
                Expr::Mul(vec![Expr::from_i64(2), x.clone()]),
                Expr::from_i64(5),
            ])
        );
    }

    #[test]
    fn test_simplify_mul_zero() {
        let x = Expr::symbol("x");
        let expr = Expr::Mul(vec![Expr::from_i64(0), x.clone()]);
        assert_eq!(simplify(&expr), Expr::from_i64(0));
    }

    #[test]
    fn power_simplification_requires_a_known_zero_base_exponent() {
        let zero = Expr::from_i64(0);
        let one = Expr::from_i64(1);
        let x = Expr::symbol("x");

        let conditional = Expr::Pow(Arc::new(zero.clone()), Arc::new(x.clone()));
        assert_eq!(try_simplify(&conditional).unwrap(), conditional);
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let (verified, envelope) = verified_simplify(
            &conditional,
            &context,
            ReceiptId::new(6).unwrap(),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(verified, conditional);
        assert!(
            verify_derivation_independent(envelope.derivation.as_ref().unwrap(), &context).is_ok()
        );

        for (exponent, expected) in [
            (Expr::from_i64(0), one.clone()),
            (Expr::from_i64(3), zero.clone()),
            (Expr::rational(1, 2).unwrap(), zero.clone()),
            (Expr::Const(fsym_core::Constant::Pi), zero.clone()),
            (Expr::Const(fsym_core::Constant::Infinity), zero.clone()),
            (
                Expr::from_i64(-3),
                Expr::Const(fsym_core::Constant::ComplexInfinity),
            ),
            (
                Expr::rational(-1, 2).unwrap(),
                Expr::Const(fsym_core::Constant::ComplexInfinity),
            ),
            (
                Expr::Const(fsym_core::Constant::NegativeInfinity),
                Expr::Const(fsym_core::Constant::ComplexInfinity),
            ),
            (
                Expr::Const(fsym_core::Constant::I),
                Expr::Const(fsym_core::Constant::NaN),
            ),
            (
                Expr::Const(fsym_core::Constant::ComplexInfinity),
                Expr::Const(fsym_core::Constant::NaN),
            ),
            (
                Expr::Const(fsym_core::Constant::NaN),
                Expr::Const(fsym_core::Constant::NaN),
            ),
        ] {
            let input = Expr::Pow(Arc::new(zero.clone()), Arc::new(exponent));
            assert_eq!(try_simplify(&input).unwrap(), expected, "input: {input}");
        }

        assert_eq!(
            try_simplify(&Expr::Pow(Arc::new(one.clone()), Arc::new(x),)).unwrap(),
            one
        );
        for exponent in [
            fsym_core::Constant::Infinity,
            fsym_core::Constant::NegativeInfinity,
            fsym_core::Constant::ComplexInfinity,
            fsym_core::Constant::NaN,
        ] {
            let input = Expr::Pow(Arc::new(Expr::from_i64(1)), Arc::new(Expr::Const(exponent)));
            assert_eq!(
                try_simplify(&input).unwrap(),
                Expr::Const(fsym_core::Constant::NaN),
                "input: {input}"
            );
        }
    }

    #[test]
    fn simplify_does_not_erase_partial_or_indeterminate_zero_factors() {
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        for partial in [
            x.clone().pow(Expr::from_i64(-1)),
            Expr::Const(fsym_core::Constant::Infinity),
            Expr::Const(fsym_core::Constant::NaN),
            Expr::Function("log".to_string(), vec![x.clone()]),
        ] {
            let input = Expr::Mul(vec![zero.clone(), partial]);
            assert_eq!(simplify(&input), input);

            let wrapped = Expr::Add(vec![Expr::from_i64(1), input.clone()]);
            assert_eq!(
                simplify(&wrapped),
                Expr::Add(vec![input, Expr::from_i64(1)]),
                "an additive parent must not erase the retained partial product"
            );
        }
    }

    #[test]
    fn test_expand_square_of_sum() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let sum = Expr::Add(vec![x.clone(), y.clone()]);
        let sq = Expr::Pow(Arc::new(sum), Arc::new(Expr::from_i64(2)));
        let exp = expand(&sq);

        // Expected: x^2 + y^2 + 2*x*y
        let expected = Expr::Add(vec![
            Expr::Pow(Arc::new(x.clone()), Arc::new(Expr::from_i64(2))),
            Expr::Pow(Arc::new(y.clone()), Arc::new(Expr::from_i64(2))),
            Expr::Mul(vec![Expr::from_i64(2), x.clone(), y.clone()]),
        ]);
        assert_eq!(exp, expected);
    }

    #[test]
    fn expansion_refuses_combinatorial_term_growth_before_allocation() {
        let left = Expr::Add(
            (0..65)
                .map(|index| Expr::symbol(format!("x{index}")))
                .collect(),
        );
        let right = Expr::Add(
            (0..65)
                .map(|index| Expr::symbol(format!("y{index}")))
                .collect(),
        );

        let oversized = Expr::Mul(vec![left, right]);
        assert!(matches!(
            expand_with(&oversized, &mut Unbounded),
            Err(SimplifyError::General(message)) if message.contains("term limit")
        ));
        assert!(matches!(
            try_expand(&oversized),
            Err(SimplifyError::General(message)) if message.contains("term limit")
        ));
    }

    #[test]
    fn verified_simplify_produces_valid_receipt_and_independent_verification() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut meter = Unbounded;
        let x = Expr::symbol("x");
        let expr = Expr::Add(vec![x.clone(), x.clone()]);

        let (simplified, envelope) =
            verified_simplify(&expr, &context, ReceiptId::new(1).unwrap(), &mut meter).unwrap();
        assert_eq!(simplified, Expr::Mul(vec![Expr::from_i64(2), x]));

        assert_eq!(envelope.claim.lhs().unwrap(), &expr);
        assert_eq!(envelope.claim.rhs().unwrap(), &simplified);
        assert!(
            verify_derivation_independent(envelope.derivation.as_ref().unwrap(), &context).is_ok()
        );
    }

    #[test]
    fn verified_simplify_uses_the_dedicated_trig_zero_rule() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());

        for (name, expected) in [
            ("sin", Expr::from_i64(0)),
            ("cos", Expr::from_i64(1)),
            ("tan", Expr::from_i64(0)),
        ] {
            let expr = Expr::Function(name.to_string(), vec![Expr::from_i64(0)]);
            let (simplified, envelope) = verified_simplify(
                &expr,
                &context,
                ReceiptId::new(match name {
                    "sin" => 2,
                    "cos" => 3,
                    _ => 4,
                })
                .unwrap(),
                &mut Unbounded,
            )
            .unwrap();

            assert_eq!(simplified, expected);
            assert!(
                verify_derivation_independent(envelope.derivation.as_ref().unwrap(), &context)
                    .is_ok()
            );
        }
    }

    #[test]
    fn normal_form_is_idempotent() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let expr = Expr::Add(vec![
            Expr::Mul(vec![Expr::from_i64(3), x.clone()]),
            Expr::Mul(vec![Expr::from_i64(2), y.clone()]),
            Expr::Mul(vec![Expr::from_i64(5), x.clone()]),
        ]);

        let s1 = simplify(&expr);
        let s2 = simplify(&s1);
        let s3 = simplify(&s2);
        assert_eq!(s1, s2);
        assert_eq!(s2, s3);
    }

    #[test]
    fn rewrite_catalog_applies_rules() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let rules = standard_rules();

        let x = Expr::symbol("x");
        let expr_add_zero = Expr::Add(vec![x.clone(), Expr::from_i64(0)]);
        let (out, rule) = apply_step(&expr_add_zero, &rules, &context).unwrap();
        assert_eq!(out, x);
        assert!(matches!(rule, ProofRule::DefinitionalReduction { .. }));

        let expr_mul_one = Expr::Mul(vec![x.clone(), Expr::from_i64(1)]);
        let (out_mul, _) = apply_step(&expr_mul_one, &rules, &context).unwrap();
        assert_eq!(out_mul, x);

        let expr_mul_zero_total = Expr::Mul(vec![x.clone(), Expr::from_i64(0)]);
        let (out_mul_zero, rule_zero) = apply_step(&expr_mul_zero_total, &rules, &context).unwrap();
        assert_eq!(out_mul_zero, Expr::from_i64(0));
        assert!(matches!(rule_zero, ProofRule::DefinitionalReduction { .. }));

        let expr_mul_zero_partial = Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Const(fsym_core::Constant::Infinity),
        ]);
        assert!(
            apply_step(&expr_mul_zero_partial, &rules, &context).is_none(),
            "0 * Infinity must not be annihilated by mul_zero_annihilator"
        );
    }

    #[test]
    fn mutant_tampered_envelope_claim_fails_verification() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut meter = Unbounded;
        let x = Expr::symbol("x");
        let expr = Expr::Add(vec![x.clone(), x.clone()]);

        let (_, envelope) =
            verified_simplify(&expr, &context, ReceiptId::new(5).unwrap(), &mut meter).unwrap();
        let mut tree = envelope.derivation.unwrap();

        // Mutate the final step claim to forged claim x + x = x
        tree.steps.last_mut().unwrap().claim = Claim::equality(expr.clone(), x.clone());

        let res = verify_derivation_independent(&tree, &context);
        assert!(
            res.is_err(),
            "Independent verifier must kill mutant tampered claim"
        );
    }

    #[test]
    fn budgeted_simplify_stops_atomically_at_safe_points() {
        let limits = fsym_budget::BudgetLimits::uniform(1, 0);
        let mut budget = fsym_budget::Budget::new(limits);
        let x = Expr::symbol("x");
        let complex = Expr::Add(vec![
            Expr::Mul(vec![x.clone(), x.clone(), x.clone()]),
            Expr::Mul(vec![x.clone(), x.clone()]),
            Expr::Mul(vec![x.clone(), Expr::from_i64(5)]),
        ]);

        let res = simplify_with(&complex, &mut budget);
        assert!(matches!(res, Err(SimplifyError::BudgetExhausted(_))));
    }

    #[test]
    fn verified_simplify_supports_elementary_functions() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut meter = Unbounded;

        // 1. asin(0) -> 0
        let asin_zero = Expr::Function("asin".into(), vec![Expr::from_i64(0)]);
        let (res_asin, env_asin) = verified_simplify(
            &asin_zero,
            &context,
            ReceiptId::new(10).unwrap(),
            &mut meter,
        )
        .unwrap();
        assert_eq!(res_asin, Expr::from_i64(0));
        assert!(env_asin.verify_integrity());

        // 2. exp(0) -> 1
        let exp_zero = Expr::Function("exp".into(), vec![Expr::from_i64(0)]);
        let (res_exp, env_exp) =
            verified_simplify(&exp_zero, &context, ReceiptId::new(11).unwrap(), &mut meter)
                .unwrap();
        assert_eq!(res_exp, Expr::from_i64(1));
        assert!(env_exp.verify_integrity());

        // 3. ln(1) -> 0
        let ln_one = Expr::Function("ln".into(), vec![Expr::from_i64(1)]);
        let (res_ln, env_ln) =
            verified_simplify(&ln_one, &context, ReceiptId::new(12).unwrap(), &mut meter).unwrap();
        assert_eq!(res_ln, Expr::from_i64(0));
        assert!(env_ln.verify_integrity());
    }
    fn pythagorean_square(f: &str, u: &Expr) -> Expr {
        Expr::Pow(
            Arc::new(Expr::Function(f.to_string(), vec![u.clone()])),
            Arc::new(Expr::from_i64(2)),
        )
    }

    #[test]
    fn simplify_folds_pythagorean_pairs() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let coeff = |c: i64, t: Expr| Expr::Mul(vec![Expr::from_i64(c), t]);

        // sin^2 + cos^2 -> 1
        assert_eq!(
            simplify(&Expr::Add(vec![
                pythagorean_square("sin", &x),
                pythagorean_square("cos", &x)
            ])),
            Expr::from_i64(1)
        );
        // scaled pair -> common coefficient
        assert_eq!(
            simplify(&Expr::Add(vec![
                coeff(3, pythagorean_square("sin", &x)),
                coeff(3, pythagorean_square("cos", &x))
            ])),
            Expr::from_i64(3)
        );
        // pair plus an existing constant
        assert_eq!(
            simplify(&Expr::Add(vec![
                Expr::from_i64(1),
                pythagorean_square("sin", &x),
                pythagorean_square("cos", &x)
            ])),
            Expr::from_i64(2)
        );
        // pair inside a Mul child folds through recursion
        assert_eq!(
            simplify(&Expr::Mul(vec![
                Expr::Add(vec![
                    pythagorean_square("sin", &x),
                    pythagorean_square("cos", &x)
                ]),
                y.clone()
            ])),
            y.clone()
        );
        // sec^2 - tan^2 -> 1
        assert_eq!(
            simplify(&Expr::Add(vec![
                pythagorean_square("sec", &x),
                coeff(-1, pythagorean_square("tan", &x))
            ])),
            Expr::from_i64(1)
        );
        // csc^2 - cot^2 -> 1
        assert_eq!(
            simplify(&Expr::Add(vec![
                pythagorean_square("csc", &x),
                coeff(-1, pythagorean_square("cot", &x))
            ])),
            Expr::from_i64(1)
        );
        // cosh^2 - sinh^2 -> 1
        assert_eq!(
            simplify(&Expr::Add(vec![
                pythagorean_square("cosh", &x),
                coeff(-1, pythagorean_square("sinh", &x))
            ])),
            Expr::from_i64(1)
        );
    }

    #[test]
    fn simplify_preserves_non_pythagorean_adds() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let coeff = |c: i64, t: Expr| Expr::Mul(vec![Expr::from_i64(c), t]);

        // Collection alone may reorder terms; a fold would change the term
        // set, so compare order-insensitively.
        let assert_same_terms = |expr: &Expr, expected_terms: &mut Vec<Expr>| {
            let simplified = simplify(expr);
            let mut actual = match simplified {
                Expr::Add(terms) => terms,
                other => vec![other],
            };
            actual.sort();
            expected_terms.sort();
            assert_eq!(actual, *expected_terms);
        };

        // unequal coefficients never fold
        assert_same_terms(
            &Expr::Add(vec![
                coeff(3, pythagorean_square("sin", &x)),
                coeff(2, pythagorean_square("cos", &x)),
            ]),
            &mut vec![
                coeff(3, pythagorean_square("sin", &x)),
                coeff(2, pythagorean_square("cos", &x)),
            ],
        );
        // different arguments never fold
        assert_same_terms(
            &Expr::Add(vec![
                pythagorean_square("sin", &x),
                pythagorean_square("cos", &y),
            ]),
            &mut vec![pythagorean_square("sin", &x), pythagorean_square("cos", &y)],
        );
        // same-family wrong sign never folds
        assert_same_terms(
            &Expr::Add(vec![
                pythagorean_square("sec", &x),
                pythagorean_square("tan", &x),
            ]),
            &mut vec![pythagorean_square("sec", &x), pythagorean_square("tan", &x)],
        );
        // exponent other than exactly two never folds
        assert_same_terms(
            &Expr::Add(vec![
                Expr::Pow(
                    Arc::new(Expr::Function("sin".to_string(), vec![x.clone()])),
                    Arc::new(Expr::from_i64(4)),
                ),
                pythagorean_square("cos", &x),
            ]),
            &mut vec![
                Expr::Pow(
                    Arc::new(Expr::Function("sin".to_string(), vec![x.clone()])),
                    Arc::new(Expr::from_i64(4)),
                ),
                pythagorean_square("cos", &x),
            ],
        );
    }

    #[test]
    fn verified_simplify_proves_pythagorean_folds_with_dedicated_rule() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");
        let expr = Expr::Add(vec![
            pythagorean_square("sin", &x),
            pythagorean_square("cos", &x),
        ]);

        let (simplified, envelope) =
            verified_simplify(&expr, &context, ReceiptId::new(30).unwrap(), &mut Unbounded)
                .unwrap();
        assert_eq!(simplified, Expr::from_i64(1));
        assert_eq!(
            verify_derivation_independent(envelope.derivation.as_ref().unwrap(), &context).unwrap(),
            Claim::equality(expr, simplified)
        );

        // A nested fold behind a constant verifies end to end as well.
        let nested = Expr::Add(vec![
            Expr::from_i64(1),
            pythagorean_square("sin", &x),
            pythagorean_square("cos", &x),
        ]);
        let (simplified_nested, envelope_nested) = verified_simplify(
            &nested,
            &context,
            ReceiptId::new(31).unwrap(),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(simplified_nested, Expr::from_i64(2));
        assert!(
            verify_derivation_independent(envelope_nested.derivation.as_ref().unwrap(), &context)
                .is_ok()
        );
    }

    #[test]
    fn rewrite_catalog_carries_pythagorean_and_inverse_rules() {
        let rules = standard_rules();
        let empty = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let x = Expr::symbol("x");

        // Root-level identity rule emits a typed DefinitionalReduction.
        let (out, rule) = apply_step(
            &Expr::Add(vec![
                pythagorean_square("sin", &x),
                pythagorean_square("cos", &x),
            ]),
            &rules,
            &empty,
        )
        .unwrap();
        assert_eq!(out, Expr::from_i64(1));
        assert!(matches!(
            rule,
            ProofRule::DefinitionalReduction { ref rule_name, .. }
                if rule_name == "pythagorean_identity"
        ));

        // exp(log(u)) folds only when u is provably Positive.
        let log_x = Expr::Function("log".to_string(), vec![x.clone()]);
        let exp_log_x = Expr::Function("exp".to_string(), vec![log_x]);
        assert!(apply_step(&exp_log_x, &rules, &empty).is_none());

        let mut positive_context = AssumptionsContext::new();
        positive_context
            .assume(Symbol::new("x"), Predicate::Positive)
            .unwrap();
        let positive = Arc::new(positive_context.snapshot());
        let (out, _) = apply_step(&exp_log_x, &rules, &positive).unwrap();
        assert_eq!(out, x);

        // Literal positive arguments fold via inherent facts alone.
        let exp_log_two = Expr::Function(
            "exp".to_string(),
            vec![Expr::Function("log".to_string(), vec![Expr::from_i64(2)])],
        );
        let (out, _) = apply_step(&exp_log_two, &rules, &empty).unwrap();
        assert_eq!(out, Expr::from_i64(2));

        // log(exp(u)) folds only when u is provably Real.
        let exp_x = Expr::Function("exp".to_string(), vec![x.clone()]);
        let log_exp_x = Expr::Function("ln".to_string(), vec![exp_x]);
        assert!(apply_step(&log_exp_x, &rules, &empty).is_none());

        let mut real_context = AssumptionsContext::new();
        real_context
            .assume(Symbol::new("x"), Predicate::Real)
            .unwrap();
        let real = Arc::new(real_context.snapshot());
        let (out, _) = apply_step(&log_exp_x, &rules, &real).unwrap();
        assert_eq!(out, x);
    }
}
