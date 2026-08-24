//! # fsym-simplify
//!
//! Algebraic simplification engine, rewrite pipelines, expansion, and verified simplification (WS07).
//! Emits proof kernel derivations and independent verification receipts for all transformations.

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
pub fn simplify(expr: &Expr) -> Expr {
    unwrap_unbounded(simplify_with(expr, &mut Unbounded))
}

/// Simplify under a caller-owned budget/cancellation meter.
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
            let has_neg_power = rest.iter().any(|f| {
                matches!(
                    f,
                    Expr::Pow(_, e) if matches!(e.as_ref(), Expr::Integer(n) if n.is_negative())
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
            } else {
                match (b, e) {
                    (Expr::Integer(bn), Expr::Integer(en)) => match usize::try_from(&en) {
                        Ok(exp_usize) if exp_usize <= 1000 => {
                            Ok(Expr::Integer(bn.pow(exp_usize as u32)))
                        }
                        _ => Ok(Expr::Pow(
                            Arc::new(Expr::Integer(bn)),
                            Arc::new(Expr::Integer(en)),
                        )),
                    },
                    (sb, se) => Ok(Expr::Pow(Arc::new(sb), Arc::new(se))),
                }
            }
        }
        Expr::Function(name, args) => {
            if name == "sin" && args.len() == 1 && args[0].is_zero() {
                return Ok(Expr::from_i64(0));
            }
            if name == "cos" && args.len() == 1 && args[0].is_zero() {
                return Ok(Expr::from_i64(1));
            }
            if name == "tan" && args.len() == 1 && args[0].is_zero() {
                return Ok(Expr::from_i64(0));
            }
            let mut simplified_args = Vec::with_capacity(args.len());
            for a in args {
                simplified_args.push(simplify_at(a, depth + 1, m)?);
            }
            Ok(Expr::Function(name.clone(), simplified_args))
        }
        _ => Ok(expr.clone()),
    }
}

/// Simplifies an expression and produces a cryptographically verified derivation receipt (WS07).
pub fn verified_simplify<M: BudgetMeter>(
    expr: &Expr,
    context: &Arc<ImmutableAssumptionsSnapshot>,
    meter: &mut M,
) -> Result<(Expr, EvidenceEnvelope), SimplifyError> {
    let simplified = simplify_with(expr, meter)?;
    let claim = Claim::equality(expr.clone(), simplified.clone());
    let mut kernel = ProofKernel::new((**context).clone());

    let step_id = if expr == &simplified {
        kernel
            .prove_reflexivity(expr.clone(), meter)
            .map_err(|e| SimplifyError::ProofFailed(e.to_string()))?
    } else {
        let rule_name = match expr {
            Expr::Function(name, args)
                if args.len() == 1
                    && args[0].is_zero()
                    && matches!(name.as_str(), "sin" | "cos" | "tan") =>
            {
                "trig_zero_eval"
            }
            _ => "simplify_normal_form",
        };
        kernel
            .prove_definitional_reduction(expr.clone(), simplified.clone(), rule_name, meter)
            .map_err(|e| SimplifyError::ProofFailed(e.to_string()))?
    };

    let derivation_tree = kernel
        .export_derivation(step_id)
        .map_err(|e| SimplifyError::ProofFailed(e.to_string()))?;

    // Independent reference verification check
    verify_derivation_independent(&derivation_tree, context)
        .map_err(|e| SimplifyError::ProofFailed(format!("Independent verifier rejected: {e}")))?;

    let receipt_id = ReceiptId::new(1).map_err(|e| SimplifyError::ProofFailed(e.to_string()))?;
    let receipt = VerificationReceipt::issue(
        receipt_id,
        &claim,
        EvidenceClass::KernelProved,
        "fsym-simplify.v1",
        1,
        Some(derivation_tree.digest()),
    );

    let envelope = EvidenceEnvelope::new(
        claim,
        EvidenceClass::KernelProved,
        receipt,
        Some(derivation_tree),
    );
    Ok((simplified, envelope))
}

/// Expand polynomial products and powers into sum-of-products normal form.
pub fn expand(expr: &Expr) -> Expr {
    unwrap_unbounded(expand_with(expr, &mut Unbounded))
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
            let mut expanded_terms = Vec::with_capacity(terms.len());
            for t in terms {
                expanded_terms.push(expand_at(t, depth + 1, m)?);
            }
            Ok(collect_terms(expanded_terms))
        }
        Expr::Mul(factors) => {
            let mut current = vec![Expr::from_i64(1)];
            for f in factors {
                let ef = expand_at(f, depth + 1, m)?;
                let next_terms = match ef {
                    Expr::Add(ts) => ts,
                    other => vec![other],
                };
                let mut product_terms = Vec::with_capacity(current.len() * next_terms.len());
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
    fn test_expand_square_of_sum() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let sum = Expr::Add(vec![x.clone(), y.clone()]);
        let sq = Expr::Pow(Arc::new(sum), Arc::new(Expr::from_i64(2)));
        let exp = expand(&sq);

        // Expected: 2*x*y + x^2 + y^2
        let expected = Expr::Add(vec![
            Expr::Mul(vec![Expr::from_i64(2), x.clone(), y.clone()]),
            Expr::Pow(Arc::new(x.clone()), Arc::new(Expr::from_i64(2))),
            Expr::Pow(Arc::new(y.clone()), Arc::new(Expr::from_i64(2))),
        ]);
        assert_eq!(exp, expected);
    }

    #[test]
    fn verified_simplify_produces_valid_receipt_and_independent_verification() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut meter = Unbounded;
        let x = Expr::symbol("x");
        let expr = Expr::Add(vec![x.clone(), x.clone()]);

        let (simplified, envelope) = verified_simplify(&expr, &context, &mut meter).unwrap();
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
            let (simplified, envelope) =
                verified_simplify(&expr, &context, &mut Unbounded).unwrap();

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
    }

    #[test]
    fn mutant_tampered_envelope_claim_fails_verification() {
        let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
        let mut meter = Unbounded;
        let x = Expr::symbol("x");
        let expr = Expr::Add(vec![x.clone(), x.clone()]);

        let (_, envelope) = verified_simplify(&expr, &context, &mut meter).unwrap();
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
}
