//! Differentiation proofs and independent derivation verifier (WS12).

#![forbid(unsafe_code)]

use crate::{diff, diff_unsimplified};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, KernelError, ProofRule, StepId};
use std::sync::Arc;

/// Standard differentiation derivation rule identifiers.
pub const RULE_DIFF_CONST: &str = "diff_constant";
pub const RULE_DIFF_VAR_SELF: &str = "diff_var_self";
pub const RULE_DIFF_VAR_OTHER: &str = "diff_var_other";
pub const RULE_DIFF_SUM: &str = "diff_sum";
pub const RULE_DIFF_PROD: &str = "diff_product";
pub const RULE_DIFF_POW_INT: &str = "diff_power_integer";
pub const RULE_DIFF_SIN: &str = "diff_sin";
pub const RULE_DIFF_COS: &str = "diff_cos";
pub const RULE_DIFF_TAN: &str = "diff_tan";
pub const RULE_DIFF_EXP: &str = "diff_exp";
pub const RULE_DIFF_SINH: &str = "diff_sinh";
pub const RULE_DIFF_COSH: &str = "diff_cosh";
pub const RULE_DIFF_TANH: &str = "diff_tanh";
pub const RULE_DIFF_LOG: &str = "diff_log";
pub const RULE_DIFF_ASIN: &str = "diff_asin";
pub const RULE_DIFF_ACOS: &str = "diff_acos";
pub const RULE_DIFF_ATAN: &str = "diff_atan";
pub const RULE_DIFF_ERF: &str = "diff_erf";
pub const RULE_DIFF_ERFC: &str = "diff_erfc";
pub const RULE_DIFF_SINC: &str = "diff_sinc";
pub const RULE_DIFF_GENERAL: &str = "diff_general";

/// Constructs the canonical diff application node: $\frac{\partial}{\partial \text{var}}(\text{expr})$.
pub fn make_diff_term(expr: &Expr, var: &Symbol) -> Expr {
    Expr::Function(
        "diff".to_string(),
        vec![expr.clone(), Expr::Sym(var.clone())],
    )
}

/// Identifies the differentiation rule appropriate for a given expression and differentiation variable.
pub fn classify_diff_rule(expr: &Expr, var: &Symbol) -> &'static str {
    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => RULE_DIFF_CONST,
        Expr::Sym(s) if s == var => RULE_DIFF_VAR_SELF,
        Expr::Sym(_) => RULE_DIFF_VAR_OTHER,
        Expr::Add(_) => RULE_DIFF_SUM,
        Expr::Mul(_) => RULE_DIFF_PROD,
        Expr::Pow(_, exp) => {
            if let Expr::Integer(_) = exp.as_ref() {
                RULE_DIFF_POW_INT
            } else {
                RULE_DIFF_GENERAL
            }
        }
        Expr::Function(name, args) => match (name.as_str(), args.len()) {
            ("sin", 1) => RULE_DIFF_SIN,
            ("cos", 1) => RULE_DIFF_COS,
            ("tan", 1) => RULE_DIFF_TAN,
            ("exp", 1) => RULE_DIFF_EXP,
            ("sinh", 1) => RULE_DIFF_SINH,
            ("cosh", 1) => RULE_DIFF_COSH,
            ("tanh", 1) => RULE_DIFF_TANH,
            ("log" | "ln", 1) => RULE_DIFF_LOG,
            ("asin", 1) => RULE_DIFF_ASIN,
            ("acos", 1) => RULE_DIFF_ACOS,
            ("atan", 1) => RULE_DIFF_ATAN,
            ("erf", 1) => RULE_DIFF_ERF,
            ("erfc", 1) => RULE_DIFF_ERFC,
            ("sinc", 1) => RULE_DIFF_SINC,
            _ => RULE_DIFF_GENERAL,
        },
    }
}

/// Computes the symbolic derivative and generates a typed, verifiable derivation proof tree
/// establishing the claim: $\vdash \text{diff}(\text{expr}, \text{var}) = \text{deriv}$.
pub fn verified_diff(expr: &Expr, var: &Symbol) -> (Expr, DerivationTree) {
    let deriv = diff_unsimplified(expr, var);
    let diff_term = make_diff_term(expr, var);
    let rule_name = classify_diff_rule(expr, var);

    let step = DerivationStep {
        id: StepId(1),
        rule: ProofRule::DefinitionalReduction {
            lhs: diff_term.clone(),
            rhs: deriv.clone(),
            rule_name: rule_name.to_string(),
        },
        claim: Claim::equality(diff_term, deriv.clone()),
    };

    let tree = DerivationTree {
        steps: vec![step],
        root: StepId(1),
    };

    (deriv, tree)
}

/// Replays a serialized differentiation receipt.
///
/// The claim is *recomputed* from the expression and variable, checked by the
/// independent derivation verifier, and only then compared against the text
/// the receipt carried. Nothing is accepted because the receipt says so: a
/// tampered rule or derivative text is refused, and a receipt for a different
/// variable/expression fails the verifier's rule semantics.
pub fn verify_diff_receipt(
    expr: &Expr,
    var: &Symbol,
    rule_name: &str,
    rhs_text: &str,
    deriv_text: &str,
) -> Result<(), KernelError> {
    let deriv = diff_unsimplified(expr, var);
    let diff_term = make_diff_term(expr, var);
    let tree = DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(1),
            rule: ProofRule::DefinitionalReduction {
                lhs: diff_term.clone(),
                rhs: deriv.clone(),
                rule_name: rule_name.to_string(),
            },
            claim: Claim::equality(diff_term, deriv.clone()),
        }],
        root: StepId(1),
    };
    verify_diff_derivation(&tree, expr, var, &deriv)?;
    let rendered = deriv.to_string();
    if rendered != deriv_text || rendered != rhs_text {
        return Err(KernelError::RuleMismatch(format!(
            "receipt text disagrees with the recomputed derivation: recomputed {rendered:?}, \
             receipt rhs {rhs_text:?}, receipt derivative {deriv_text:?}"
        )));
    }
    Ok(())
}

/// Independent verifier checking that the differentiation derivation tree correctly proves
/// the claim $\vdash \text{diff}(\text{expr}, \text{var}) = \text{deriv}$.
pub fn verify_diff_derivation(
    tree: &DerivationTree,
    expr: &Expr,
    var: &Symbol,
    deriv: &Expr,
) -> Result<(), KernelError> {
    if tree.steps.is_empty() {
        return Err(KernelError::UnknownStep(tree.root));
    }

    let root_step = tree
        .steps
        .iter()
        .find(|s| s.id == tree.root)
        .ok_or(KernelError::UnknownStep(tree.root))?;

    let diff_term = make_diff_term(expr, var);
    let expected_claim = Claim::equality(diff_term.clone(), deriv.clone());

    if root_step.claim != expected_claim {
        return Err(KernelError::ClaimDiscrepancy {
            expected: Box::new(expected_claim),
            derived: Box::new(root_step.claim.clone()),
        });
    }

    let expected_rule_name = classify_diff_rule(expr, var);

    match &root_step.rule {
        ProofRule::DefinitionalReduction {
            rule_name,
            lhs,
            rhs,
        } => {
            if rule_name != expected_rule_name {
                return Err(KernelError::RuleMismatch(format!(
                    "Expected rule {expected_rule_name}, got {rule_name}"
                )));
            }
            if lhs != &diff_term {
                return Err(KernelError::RuleMismatch(format!(
                    "LHS mismatch: expected `{diff_term}`, got `{lhs}`"
                )));
            }
            if rhs != deriv {
                return Err(KernelError::RuleMismatch(format!(
                    "RHS mismatch: expected `{deriv}`, got `{rhs}`"
                )));
            }

            // Independent structural verification of the reduction rule
            verify_rule_reduction_semantics(expr, var, deriv, expected_rule_name)?;
        }
        other => {
            return Err(KernelError::RuleMismatch(format!(
                "Expected DefinitionalReduction for differentiation, got {other:?}"
            )));
        }
    }

    Ok(())
}

/// Validates that the RHS formula corresponds to the mathematical differentiation rule.
fn verify_rule_reduction_semantics(
    expr: &Expr,
    var: &Symbol,
    deriv: &Expr,
    rule_name: &str,
) -> Result<(), KernelError> {
    match rule_name {
        RULE_DIFF_CONST => {
            if !deriv.is_zero() {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!("Constant derivative must be 0, got {deriv}"),
                });
            }
        }
        RULE_DIFF_VAR_SELF => {
            if !deriv.is_one() {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!("Self variable derivative must be 1, got {deriv}"),
                });
            }
        }
        RULE_DIFF_VAR_OTHER => {
            if !deriv.is_zero() {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!("Other variable derivative must be 0, got {deriv}"),
                });
            }
        }
        RULE_DIFF_SUM => {
            if let Expr::Add(terms) = expr {
                let expected_terms: Vec<Expr> = terms.iter().map(|t| diff(t, var)).collect();
                let expected_rhs = Expr::Add(expected_terms);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Sum derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected Add expression for diff_sum".to_string(),
                });
            }
        }
        RULE_DIFF_PROD => {
            if let Expr::Mul(factors) = expr {
                let mut add_terms = Vec::new();
                for i in 0..factors.len() {
                    let mut prod_factors = Vec::new();
                    for (j, factor) in factors.iter().enumerate() {
                        if i == j {
                            prod_factors.push(diff(factor, var));
                        } else {
                            prod_factors.push(factor.clone());
                        }
                    }
                    add_terms.push(Expr::Mul(prod_factors));
                }
                let expected_rhs = Expr::Add(add_terms);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Product derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected Mul expression for diff_product".to_string(),
                });
            }
        }
        RULE_DIFF_POW_INT => {
            if let Expr::Pow(base, exp) = expr
                && let Expr::Integer(n) = exp.as_ref()
            {
                let n_minus_1 = n - BigInt::from(1);
                let du = diff(base, var);
                let expected_rhs = Expr::Mul(vec![
                    Expr::Integer(n.clone()),
                    Expr::Pow(base.clone(), Arc::new(Expr::Integer(n_minus_1))),
                    du,
                ]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Power derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected Pow expression with integer exponent for diff_power_integer"
                        .to_string(),
                });
            }
        }
        RULE_DIFF_SIN => {
            if let Expr::Function(name, args) = expr
                && name == "sin"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let expected_rhs =
                    Expr::Mul(vec![Expr::Function("cos".to_string(), vec![u.clone()]), du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Sin derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            }
        }
        RULE_DIFF_COS => {
            if let Expr::Function(name, args) = expr
                && name == "cos"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let expected_rhs = Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("sin".to_string(), vec![u.clone()]),
                    du,
                ]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Cos derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            }
        }
        RULE_DIFF_EXP => {
            if let Expr::Function(name, args) = expr
                && name == "exp"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let expected_rhs =
                    Expr::Mul(vec![Expr::Function("exp".to_string(), vec![u.clone()]), du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Exp derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            }
        }
        RULE_DIFF_SINH => {
            if let Expr::Function(name, args) = expr
                && name == "sinh"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let expected_rhs = Expr::Mul(vec![
                    Expr::Function("cosh".to_string(), vec![u.clone()]),
                    du,
                ]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Sinh derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            }
        }
        RULE_DIFF_COSH => {
            if let Expr::Function(name, args) = expr
                && name == "cosh"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let expected_rhs = Expr::Mul(vec![
                    Expr::Function("sinh".to_string(), vec![u.clone()]),
                    du,
                ]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Cosh derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            }
        }
        RULE_DIFF_LOG => {
            if let Expr::Function(name, args) = expr
                && (name == "log" || name == "ln")
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let expected_rhs = Expr::Mul(vec![
                    Expr::Pow(Arc::new(u.clone()), Arc::new(Expr::from_i64(-1))),
                    du,
                ]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Log derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            }
        }
        RULE_DIFF_TAN => {
            if let Expr::Function(name, args) = expr
                && name == "tan"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let tan_u_sq = Expr::pow(
                    Expr::Function("tan".to_string(), vec![u.clone()]),
                    Expr::from_i64(2),
                );
                let expected_rhs =
                    Expr::Mul(vec![Expr::Add(vec![Expr::from_i64(1), tan_u_sq]), du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Tan derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected tan(u) for diff_tan".to_string(),
                });
            }
        }
        RULE_DIFF_TANH => {
            if let Expr::Function(name, args) = expr
                && name == "tanh"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let tanh_u_sq = Expr::pow(
                    Expr::Function("tanh".to_string(), vec![u.clone()]),
                    Expr::from_i64(2),
                );
                let expected_rhs = Expr::Mul(vec![
                    Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::Mul(vec![Expr::from_i64(-1), tanh_u_sq]),
                    ]),
                    du,
                ]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Tanh derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected tanh(u) for diff_tanh".to_string(),
                });
            }
        }
        RULE_DIFF_ASIN => {
            if let Expr::Function(name, args) = expr
                && name == "asin"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let one_minus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]),
                ]);
                let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                let expected_rhs = Expr::Mul(vec![Expr::pow(one_minus_u_sq, neg_half), du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Asin derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected asin(u) for diff_asin".to_string(),
                });
            }
        }
        RULE_DIFF_ACOS => {
            if let Expr::Function(name, args) = expr
                && name == "acos"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let one_minus_u_sq = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]),
                ]);
                let neg_half = Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                let expected_rhs = Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(one_minus_u_sq, neg_half),
                    du,
                ]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Acos derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected acos(u) for diff_acos".to_string(),
                });
            }
        }
        RULE_DIFF_ATAN => {
            if let Expr::Function(name, args) = expr
                && name == "atan"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let denom = Expr::Add(vec![
                    Expr::from_i64(1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                let expected_rhs = Expr::Mul(vec![Expr::pow(denom, Expr::from_i64(-1)), du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Atan derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected atan(u) for diff_atan".to_string(),
                });
            }
        }
        RULE_DIFF_ERF => {
            if let Expr::Function(name, args) = expr
                && name == "erf"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let pi = Expr::Const(Constant::Pi);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_pi = Expr::pow(pi, half);
                let inv_sqrt_pi = Expr::pow(sqrt_pi, Expr::from_i64(-1));
                let neg_u_sq = Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                let exp_neg_u_sq = Expr::Function("exp".to_string(), vec![neg_u_sq]);
                let expected_rhs =
                    Expr::Mul(vec![Expr::from_i64(2), inv_sqrt_pi, exp_neg_u_sq, du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Erf derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected erf(u) for diff_erf".to_string(),
                });
            }
        }
        RULE_DIFF_ERFC => {
            if let Expr::Function(name, args) = expr
                && name == "erfc"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let pi = Expr::Const(Constant::Pi);
                let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                let sqrt_pi = Expr::pow(pi, half);
                let inv_sqrt_pi = Expr::pow(sqrt_pi, Expr::from_i64(-1));
                let neg_u_sq = Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::pow(u.clone(), Expr::from_i64(2)),
                ]);
                let exp_neg_u_sq = Expr::Function("exp".to_string(), vec![neg_u_sq]);
                let expected_rhs =
                    Expr::Mul(vec![Expr::from_i64(-2), inv_sqrt_pi, exp_neg_u_sq, du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Erfc derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected erfc(u) for diff_erfc".to_string(),
                });
            }
        }
        RULE_DIFF_SINC => {
            if let Expr::Function(name, args) = expr
                && name == "sinc"
                && args.len() == 1
            {
                let u = &args[0];
                let du = diff(u, var);
                let cos_u = Expr::Function("cos".to_string(), vec![u.clone()]);
                let sin_u = Expr::Function("sin".to_string(), vec![u.clone()]);
                let inv_u = Expr::pow(u.clone(), Expr::from_i64(-1));
                let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                let term1 = Expr::Mul(vec![cos_u, inv_u]);
                let term2 = Expr::Mul(vec![Expr::from_i64(-1), sin_u, inv_u_sq]);
                let expected_rhs = Expr::Mul(vec![Expr::Add(vec![term1, term2]), du]);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Sinc derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected sinc(u) for diff_sinc".to_string(),
                });
            }
        }
        RULE_DIFF_GENERAL => {
            // Unsupported derivatives are represented by the unevaluated
            // derivative term.  This establishes only reflexive equality; it
            // must never certify an arbitrary caller-supplied right-hand side.
            let expected_rhs = make_diff_term(expr, var);
            if deriv != &expected_rhs {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!(
                        "General derivative must remain unevaluated as {expected_rhs}, got {deriv}"
                    ),
                });
            }
        }
        unknown => {
            return Err(KernelError::InvalidDefinitionalReduction {
                rule_name: unknown.to_string(),
                reason: "Unknown differentiation reduction rule".to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verified_differentiation_polynomial() {
        let x = Symbol::new("x");
        let expr = Expr::Add(vec![
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(3))),
            Expr::Mul(vec![Expr::from_i64(2), Expr::symbol("x")]),
            Expr::from_i64(5),
        ]);

        let (deriv, tree) = verified_diff(&expr, &x);
        assert!(verify_diff_derivation(&tree, &expr, &x, &deriv).is_ok());
    }

    #[test]
    fn test_verified_differentiation_trig_and_transcendental() {
        let x = Symbol::new("x");
        for expr in [
            Expr::Function("sin".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("cos".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("tan".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("exp".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("sinh".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("cosh".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("tanh".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("log".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("asin".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("acos".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("atan".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("erf".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("erfc".to_string(), vec![Expr::symbol("x")]),
            Expr::Function("sinc".to_string(), vec![Expr::symbol("x")]),
        ] {
            let (deriv, tree) = verified_diff(&expr, &x);
            assert!(verify_diff_derivation(&tree, &expr, &x, &deriv).is_ok());
        }
    }

    #[test]
    fn test_mutant_rejection_on_tampered_derivative_claim() {
        let x = Symbol::new("x");
        let expr = Expr::Mul(vec![Expr::symbol("x"), Expr::from_i64(5)]);
        let (deriv, tree) = verified_diff(&expr, &x);

        // Positive check
        assert!(verify_diff_derivation(&tree, &expr, &x, &deriv).is_ok());

        // Mutant 1: forged deriv value
        let forged_deriv = Expr::from_i64(42);
        assert!(verify_diff_derivation(&tree, &expr, &x, &forged_deriv).is_err());

        // Mutant 2: tampered claim inside derivation step
        let mut tampered_tree = tree.clone();
        tampered_tree.steps[0].claim = Claim::equality(expr.clone(), deriv.clone());
        assert!(verify_diff_derivation(&tampered_tree, &expr, &x, &deriv).is_err());

        // Mutant 3: forged rule name
        let mut tampered_tree_rule = tree.clone();
        tampered_tree_rule.steps[0].rule = ProofRule::DefinitionalReduction {
            lhs: make_diff_term(&expr, &x),
            rhs: deriv.clone(),
            rule_name: RULE_DIFF_CONST.to_string(),
        };
        assert!(verify_diff_derivation(&tampered_tree_rule, &expr, &x, &deriv).is_err());
    }

    #[test]
    fn test_general_rule_only_verifies_the_unevaluated_derivative() {
        let x = Symbol::new("x");
        let expr = Expr::Function("unsupported_fn".to_string(), vec![Expr::symbol("x")]);
        let diff_term = make_diff_term(&expr, &x);

        let (unevaluated, tree) = verified_diff(&expr, &x);
        assert_eq!(unevaluated, diff_term);
        assert!(verify_diff_derivation(&tree, &expr, &x, &unevaluated).is_ok());

        let forged_deriv = Expr::from_i64(42);
        let forged_tree = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(1),
                rule: ProofRule::DefinitionalReduction {
                    lhs: diff_term.clone(),
                    rhs: forged_deriv.clone(),
                    rule_name: RULE_DIFF_GENERAL.to_string(),
                },
                claim: Claim::equality(diff_term, forged_deriv.clone()),
            }],
            root: StepId(1),
        };

        assert!(matches!(
            verify_diff_derivation(&forged_tree, &expr, &x, &forged_deriv),
            Err(KernelError::InvalidDefinitionalReduction { .. })
        ));
    }

    #[test]
    fn test_verified_diff_independent_of_other_variables() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let expr = Expr::symbol("y");

        let (deriv, tree) = verified_diff(&expr, &x);
        assert_eq!(deriv, Expr::from_i64(0));
        assert!(verify_diff_derivation(&tree, &expr, &x, &deriv).is_ok());

        let (deriv_y, tree_y) = verified_diff(&expr, &y);
        assert_eq!(deriv_y, Expr::from_i64(1));
        assert!(verify_diff_derivation(&tree_y, &expr, &y, &deriv_y).is_ok());
    }

    #[test]
    fn test_receipt_replay_accepts_the_original_and_refuses_tampering() {
        let x = Symbol::new("x");
        let expr = Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("x")]);
        let (deriv, _tree) = verified_diff(&expr, &x);
        let rule = classify_diff_rule(&expr, &x);
        let text = deriv.to_string();

        // The untampered receipt replays.
        assert!(verify_diff_receipt(&expr, &x, rule, &text, &text).is_ok());

        // A replaced derivative text or rule is refused.
        assert!(verify_diff_receipt(&expr, &x, rule, "0", &text).is_err());
        assert!(verify_diff_receipt(&expr, &x, rule, &text, "0").is_err());
        assert!(verify_diff_receipt(&expr, &x, RULE_DIFF_SUM, &text, &text).is_err());

        // A receipt replayed against another expression is refused as well.
        let y = Symbol::new("y");
        assert!(verify_diff_receipt(&Expr::Sym(y), &x, rule, &text, &text).is_err());
    }
}
