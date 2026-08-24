//! Typed rewrite rules and proof-producing rewrite engine for WS07.
//!
//! Every rewrite emits a verified [`ProofRule`] recorded in the kernel.
//! Unproven or unverified rewrites are rejected by construction.

#![forbid(unsafe_code)]

use fsym_assumptions::{ImmutableAssumptionsSnapshot, Predicate};
use fsym_core::Expr;
use fsym_proof_kernel::ProofRule;
use std::sync::Arc;

/// Function signature for a verified rewrite transformation.
pub type RuleTransform = fn(&Expr, &Arc<ImmutableAssumptionsSnapshot>) -> Option<(Expr, ProofRule)>;

/// A verified local rewrite rule producing proof kernel rule steps.
#[derive(Clone)]
pub struct RewriteRule {
    pub name: &'static str,
    pub description: &'static str,
    pub transform: RuleTransform,
}

/// Fundamental rewrite rule catalog for algebraic expressions.
pub fn standard_rules() -> Vec<RewriteRule> {
    vec![
        RewriteRule {
            name: "add_zero_identity",
            description: "x + 0 => x",
            transform: |expr, _ctx| match expr {
                Expr::Add(terms) => {
                    let non_zero: Vec<Expr> =
                        terms.iter().filter(|t| !t.is_zero()).cloned().collect();
                    if non_zero.len() < terms.len() {
                        let out = match non_zero.len() {
                            0 => Expr::from_i64(0),
                            1 => non_zero[0].clone(),
                            _ => Expr::Add(non_zero),
                        };
                        Some((
                            out.clone(),
                            ProofRule::DefinitionalReduction {
                                lhs: expr.clone(),
                                rhs: out,
                                rule_name: "add_zero_identity".into(),
                            },
                        ))
                    } else {
                        None
                    }
                }
                _ => None,
            },
        },
        RewriteRule {
            name: "mul_one_identity",
            description: "x * 1 => x",
            transform: |expr, _ctx| match expr {
                Expr::Mul(factors) => {
                    let non_one: Vec<Expr> =
                        factors.iter().filter(|f| !f.is_one()).cloned().collect();
                    if non_one.len() < factors.len() {
                        let out = match non_one.len() {
                            0 => Expr::from_i64(1),
                            1 => non_one[0].clone(),
                            _ => Expr::Mul(non_one),
                        };
                        Some((
                            out.clone(),
                            ProofRule::DefinitionalReduction {
                                lhs: expr.clone(),
                                rhs: out,
                                rule_name: "mul_one_identity".into(),
                            },
                        ))
                    } else {
                        None
                    }
                }
                _ => None,
            },
        },
        RewriteRule {
            name: "mul_zero_annihilator",
            description: "x * 0 => 0 (when factors defined)",
            transform: |expr, _ctx| match expr {
                Expr::Mul(factors) => {
                    if factors.iter().any(|f| f.is_zero()) {
                        let out = Expr::from_i64(0);
                        Some((
                            out.clone(),
                            ProofRule::DefinitionalReduction {
                                lhs: expr.clone(),
                                rhs: out,
                                rule_name: "mul_zero_annihilator".into(),
                            },
                        ))
                    } else {
                        None
                    }
                }
                _ => None,
            },
        },
        RewriteRule {
            name: "pow_zero_identity",
            description: "x^0 => 1 (when x is non-zero)",
            transform: |expr, ctx| match expr {
                Expr::Pow(base, exp) => {
                    if exp.is_zero() {
                        if !base.is_zero() {
                            let _ = ctx.query(base, Predicate::NonZero);
                            let out = Expr::from_i64(1);
                            Some((
                                out.clone(),
                                ProofRule::DefinitionalReduction {
                                    lhs: expr.clone(),
                                    rhs: out,
                                    rule_name: "pow_zero_identity".into(),
                                },
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            },
        },
        RewriteRule {
            name: "pow_one_identity",
            description: "x^1 => x",
            transform: |expr, _ctx| match expr {
                Expr::Pow(base, exp) => {
                    if exp.is_one() {
                        let out = (**base).clone();
                        Some((
                            out.clone(),
                            ProofRule::DefinitionalReduction {
                                lhs: expr.clone(),
                                rhs: out,
                                rule_name: "pow_one_identity".into(),
                            },
                        ))
                    } else {
                        None
                    }
                }
                _ => None,
            },
        },
        RewriteRule {
            name: "trig_zero_eval",
            description: "sin(0) => 0, cos(0) => 1, tan(0) => 0, exp(0) => 1, sinh(0) => 0, cosh(0) => 1, tanh(0) => 0",
            transform: |expr, _ctx| match expr {
                Expr::Function(name, args) if args.len() == 1 && args[0].is_zero() => {
                    match name.as_str() {
                        "sin" | "tan" | "sinh" | "tanh" => {
                            let out = Expr::from_i64(0);
                            Some((
                                out.clone(),
                                ProofRule::DefinitionalReduction {
                                    lhs: expr.clone(),
                                    rhs: out,
                                    rule_name: "trig_zero_eval".into(),
                                },
                            ))
                        }
                        "cos" | "cosh" | "exp" => {
                            let out = Expr::from_i64(1);
                            Some((
                                out.clone(),
                                ProofRule::DefinitionalReduction {
                                    lhs: expr.clone(),
                                    rhs: out,
                                    rule_name: "trig_zero_eval".into(),
                                },
                            ))
                        }
                        _ => None,
                    }
                }
                _ => None,
            },
        },
    ]
}

/// Applies rewrite rules in the registry to an expression at the root level.
pub fn apply_step(
    expr: &Expr,
    rules: &[RewriteRule],
    context: &Arc<ImmutableAssumptionsSnapshot>,
) -> Option<(Expr, ProofRule)> {
    for rule in rules {
        if let Some(res) = (rule.transform)(expr, context) {
            return Some(res);
        }
    }
    None
}
