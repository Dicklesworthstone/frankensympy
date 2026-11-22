//! Differentiation proofs and independent derivation verifier (WS12).

#![forbid(unsafe_code)]

use crate::diff_unsimplified;
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
/// Formal power/chain rule on the common smooth domain of its two sides.
/// This does not prove differentiability at zero, choose a new real branch,
/// or certify a numerical enclosure. Finite numerical consumers must still
/// reject points where the original function or its derivative is undefined.
pub const RULE_DIFF_POW_RATIONAL: &str = "diff_power_rational";
/// Formal variable-power chain rule on the common smooth domain. This receipt
/// does not prove positivity, select complex branches, or enclose rounding.
/// Finite-real compiled consumers separately require a positive source base.
pub const RULE_DIFF_POW_GENERAL: &str = "diff_power_general";
pub const RULE_DIFF_SIN: &str = "diff_sin";
pub const RULE_DIFF_COS: &str = "diff_cos";
pub const RULE_DIFF_TAN: &str = "diff_tan";
/// Reciprocal-function chain rules hold on the common smooth domain. They
/// do not discharge the original sine/cosine denominator obligations.
pub const RULE_DIFF_COT: &str = "diff_cot";
pub const RULE_DIFF_SEC: &str = "diff_sec";
pub const RULE_DIFF_CSC: &str = "diff_csc";
pub const RULE_DIFF_EXP: &str = "diff_exp";
pub const RULE_DIFF_SINH: &str = "diff_sinh";
pub const RULE_DIFF_COSH: &str = "diff_cosh";
pub const RULE_DIFF_TANH: &str = "diff_tanh";
pub const RULE_DIFF_COTH: &str = "diff_coth";
pub const RULE_DIFF_SECH: &str = "diff_sech";
pub const RULE_DIFF_CSCH: &str = "diff_csch";
pub const RULE_DIFF_LOG: &str = "diff_log";
pub const RULE_DIFF_ASIN: &str = "diff_asin";
pub const RULE_DIFF_ACOS: &str = "diff_acos";
pub const RULE_DIFF_ATAN: &str = "diff_atan";
/// Formal binary chain rule on the common smooth domain. Original-source
/// consumers separately reject the origin and principal negative-axis cut.
pub const RULE_DIFF_ATAN2: &str = "diff_atan2";
/// Principal-branch rules on their common smooth domains. These derivations
/// do not discharge whole-box domain or differentiability obligations.
pub const RULE_DIFF_ASINH: &str = "diff_asinh";
pub const RULE_DIFF_ACOSH: &str = "diff_acosh";
pub const RULE_DIFF_ATANH: &str = "diff_atanh";
pub const RULE_DIFF_ERF: &str = "diff_erf";
pub const RULE_DIFF_ERFC: &str = "diff_erfc";
pub const RULE_DIFF_SINC: &str = "diff_sinc";
/// Exact chain rule for the private entire sinc first-derivative node.
pub const RULE_DIFF_SINC_D1: &str = "diff_sinc_d1";
/// Numeric-only real chain rules on the open domain u != 0. Original source
/// and private jet evaluation must discharge that domain at every invocation;
/// these rules do not claim a derivative or subgradient at the cusp.
pub const RULE_DIFF_ABS: &str = "diff_abs_nonzero_real";
pub const RULE_DIFF_ABS_D1: &str = "diff_abs_d1_nonzero_real";
pub const RULE_DIFF_SQRT: &str = "diff_sqrt";
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
            } else if let Expr::Rational(_) = exp.as_ref() {
                RULE_DIFF_POW_RATIONAL
            } else {
                RULE_DIFF_POW_GENERAL
            }
        }
        Expr::Function(name, args) => match (name.as_str(), args.len()) {
            ("sin", 1) => RULE_DIFF_SIN,
            ("cos", 1) => RULE_DIFF_COS,
            ("tan", 1) => RULE_DIFF_TAN,
            ("cot", 1) => RULE_DIFF_COT,
            ("sec", 1) => RULE_DIFF_SEC,
            ("csc", 1) => RULE_DIFF_CSC,
            ("exp", 1) => RULE_DIFF_EXP,
            ("sinh", 1) => RULE_DIFF_SINH,
            ("cosh", 1) => RULE_DIFF_COSH,
            ("tanh", 1) => RULE_DIFF_TANH,
            ("coth", 1) => RULE_DIFF_COTH,
            ("sech", 1) => RULE_DIFF_SECH,
            ("csch", 1) => RULE_DIFF_CSCH,
            ("log" | "ln", 1) => RULE_DIFF_LOG,
            ("asin", 1) => RULE_DIFF_ASIN,
            ("acos", 1) => RULE_DIFF_ACOS,
            ("atan", 1) => RULE_DIFF_ATAN,
            ("atan2", 2) => RULE_DIFF_ATAN2,
            ("asinh", 1) => RULE_DIFF_ASINH,
            ("acosh", 1) => RULE_DIFF_ACOSH,
            ("atanh", 1) => RULE_DIFF_ATANH,
            ("erf", 1) => RULE_DIFF_ERF,
            ("erfc", 1) => RULE_DIFF_ERFC,
            ("sinc", 1) => RULE_DIFF_SINC,
            ("_fsym_sinc_d1", 1) => RULE_DIFF_SINC_D1,
            ("sqrt", 1) => RULE_DIFF_SQRT,
            _ => RULE_DIFF_GENERAL,
        },
    }
}

fn classify_diff_rule_with_form(expr: &Expr, var: &Symbol, form: DerivativeForm) -> &'static str {
    // Preserve public symbolic and persisted legacy rule names. The new real
    // absolute-value interpretation belongs only to the current numeric lane.
    if form == DerivativeForm::Numeric
        && let Expr::Function(name, arguments) = expr
        && arguments.len() == 1
    {
        match name.as_str() {
            "abs" | "Abs" => return RULE_DIFF_ABS,
            "_fsym_abs_d1" => return RULE_DIFF_ABS_D1,
            _ => {}
        }
    }
    classify_diff_rule(expr, var)
}

/// Computes the symbolic derivative and generates a typed, verifiable derivation proof tree
/// establishing the claim: $\vdash \text{diff}(\text{expr}, \text{var}) = \text{deriv}$.
pub fn verified_diff(expr: &Expr, var: &Symbol) -> (Expr, DerivationTree) {
    let deriv = diff_unsimplified(expr, var);
    derivative_tree(expr, var, deriv, DerivativeForm::Symbolic)
}

/// Internal numeric programs use stable tanh/coth, entire sinc jets and literal
/// identity-power derivatives, plus guarded real absolute-value jets, without
/// changing public symbolic receipts.
pub(crate) fn verified_numeric_diff(expr: &Expr, var: &Symbol) -> (Expr, DerivationTree) {
    let deriv = crate::numeric_diff_unsimplified(expr, var);
    derivative_tree(expr, var, deriv, DerivativeForm::Numeric)
}

fn derivative_tree(
    expr: &Expr,
    var: &Symbol,
    deriv: Expr,
    form: DerivativeForm,
) -> (Expr, DerivationTree) {
    let diff_term = make_diff_term(expr, var);
    let rule_name = classify_diff_rule_with_form(expr, var, form);

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
    let deriv = reference_unsimplified_diff(expr, var);
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DerivativeForm {
    Symbolic,
    Numeric,
    // Persisted schema-1 programs used stable coth but 1 - tanh(u)^2.
    // Replay checks that exact older recipe without rewriting stored programs.
    LegacyNumeric,
}

/// Independent verifier checking that the differentiation derivation tree correctly proves
/// the claim $\vdash \text{diff}(\text{expr}, \text{var}) = \text{deriv}$.
pub fn verify_diff_derivation(
    tree: &DerivationTree,
    expr: &Expr,
    var: &Symbol,
    deriv: &Expr,
) -> Result<(), KernelError> {
    verify_diff_derivation_with_form(tree, expr, var, deriv, DerivativeForm::Symbolic)
}

pub(crate) fn verify_numeric_diff_derivation(
    tree: &DerivationTree,
    expr: &Expr,
    var: &Symbol,
    deriv: &Expr,
) -> Result<(), KernelError> {
    verify_diff_derivation_with_form(tree, expr, var, deriv, DerivativeForm::Numeric)
}

/// Replay the older numeric rule only after the common proof envelope has
/// passed and the current rule returned InvalidDefinitionalReduction.
/// The caller charges the second semantic pass without rescanning that envelope.
pub(crate) fn verify_legacy_numeric_diff_reduction(
    expr: &Expr,
    var: &Symbol,
    deriv: &Expr,
) -> Result<(), KernelError> {
    verify_rule_reduction_semantics(
        expr,
        var,
        deriv,
        classify_diff_rule_with_form(expr, var, DerivativeForm::LegacyNumeric),
        DerivativeForm::LegacyNumeric,
    )
}

fn verify_diff_derivation_with_form(
    tree: &DerivationTree,
    expr: &Expr,
    var: &Symbol,
    deriv: &Expr,
    form: DerivativeForm,
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

    let expected_rule_name = classify_diff_rule_with_form(expr, var, form);

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
            verify_rule_reduction_semantics(expr, var, deriv, expected_rule_name, form)?;
        }
        other => {
            return Err(KernelError::RuleMismatch(format!(
                "Expected DefinitionalReduction for differentiation, got {other:?}"
            )));
        }
    }

    Ok(())
}

/// Verifier-local zero-product elimination (independent of `crate::eliminate_zero_products`).
fn reference_eliminate_zero_products(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let non_zero: Vec<Expr> = terms
                .iter()
                .map(reference_eliminate_zero_products)
                .filter(|t| !t.is_zero())
                .collect();
            match non_zero.len() {
                0 => Expr::from_i64(0),
                1 => non_zero.into_iter().next().expect("len checked"),
                _ => Expr::Add(non_zero),
            }
        }
        Expr::Mul(factors) => {
            let mapped: Vec<Expr> = factors
                .iter()
                .map(reference_eliminate_zero_products)
                .collect();
            if mapped.iter().any(|f| f.is_zero()) {
                Expr::from_i64(0)
            } else {
                match mapped.len() {
                    0 => Expr::from_i64(1),
                    1 => mapped.into_iter().next().expect("len checked"),
                    _ => Expr::Mul(mapped),
                }
            }
        }
        Expr::Pow(base, exp) => Expr::Pow(
            Arc::new(reference_eliminate_zero_products(base)),
            Arc::new(reference_eliminate_zero_products(exp)),
        ),
        Expr::Function(name, args) => Expr::Function(
            name.clone(),
            args.iter().map(reference_eliminate_zero_products).collect(),
        ),
        _ => expr.clone(),
    }
}

/// Verifier-local reference sub-derivative (never invokes `crate::diff` or `crate::diff_unsimplified`).
pub(crate) fn reference_numeric_sub_derivative(expr: &Expr, var: &Symbol) -> Expr {
    reference_sub_derivative_with_form(expr, var, DerivativeForm::Numeric)
}

pub(crate) fn reference_legacy_numeric_sub_derivative(expr: &Expr, var: &Symbol) -> Expr {
    reference_sub_derivative_with_form(expr, var, DerivativeForm::LegacyNumeric)
}

fn reference_sub_derivative_with_form(expr: &Expr, var: &Symbol, form: DerivativeForm) -> Expr {
    // Native Expr has no binders: absence from its structural free-symbol
    // set proves constancy with respect to this variable, including powers
    // and functions outside the reference rule table.
    if !expr.free_symbols().contains(var) {
        return Expr::from_i64(0);
    }
    let raw = reference_unsimplified_diff_with_form(expr, var, form);
    let cleaned = reference_eliminate_zero_products(&raw);
    fsym_simplify::simplify(&cleaned)
}

/// Verifier-local formal differentiation rule table (independent of `crate::diff_unsimplified`).
fn reference_unsimplified_diff(expr: &Expr, var: &Symbol) -> Expr {
    reference_unsimplified_diff_with_form(expr, var, DerivativeForm::Symbolic)
}

fn reference_unsimplified_diff_with_form(expr: &Expr, var: &Symbol, form: DerivativeForm) -> Expr {
    let reference_sub_derivative =
        |child: &Expr, variable: &Symbol| reference_sub_derivative_with_form(child, variable, form);
    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => Expr::from_i64(0),
        Expr::Sym(s) if s == var => Expr::from_i64(1),
        Expr::Sym(_) => Expr::from_i64(0),
        Expr::Add(terms) => Expr::Add(
            terms
                .iter()
                .map(|t| reference_sub_derivative(t, var))
                .collect(),
        ),
        Expr::Mul(factors) => {
            let mut add_terms = Vec::with_capacity(factors.len());
            for i in 0..factors.len() {
                let mut prod_factors = Vec::with_capacity(factors.len());
                for (j, factor) in factors.iter().enumerate() {
                    if i == j {
                        prod_factors.push(reference_sub_derivative(factor, var));
                    } else {
                        prod_factors.push(factor.clone());
                    }
                }
                add_terms.push(Expr::Mul(prod_factors));
            }
            Expr::Add(add_terms)
        }
        Expr::Pow(base, exp) => {
            // Reconstruct the numeric-only identity cases from exact source
            // literals, independently of the optimizing differentiator.
            if form != DerivativeForm::Symbolic {
                match exp.as_ref() {
                    Expr::Integer(n) if n.is_zero() => return Expr::from_i64(0),
                    Expr::Rational(r) if r.numer().is_zero() => return Expr::from_i64(0),
                    Expr::Integer(n) if n.is_one() => {
                        return reference_sub_derivative(base, var);
                    }
                    Expr::Rational(r) if r.numer() == r.denom() => {
                        return reference_sub_derivative(base, var);
                    }
                    _ => {}
                }
            }
            if let Expr::Integer(n) = exp.as_ref() {
                let n_minus_1 = n - BigInt::from(1);
                let du = reference_sub_derivative(base, var);
                if du.is_zero() {
                    Expr::from_i64(0)
                } else {
                    Expr::Mul(vec![
                        Expr::Integer(n.clone()),
                        Expr::Pow(base.clone(), Arc::new(Expr::Integer(n_minus_1))),
                        du,
                    ])
                }
            } else if let Expr::Rational(rational) = exp.as_ref() {
                let du = reference_sub_derivative(base, var);
                if du.is_zero() {
                    Expr::from_i64(0)
                } else {
                    // Derive (p-q)/q directly from the exact source payload;
                    // never reuse the generator's exponent transformation or
                    // round an exponent through floating point.
                    let shifted = BigRational::new(
                        rational.numer() - rational.denom(),
                        rational.denom().clone(),
                    );
                    Expr::Mul(vec![
                        Expr::Rational(rational.clone()),
                        Expr::Pow(
                            base.clone(),
                            Arc::new(fsym_core::elementary::rational_expr(shifted)),
                        ),
                        du,
                    ])
                }
            } else {
                // Reconstruct both chain factors with the verifier's own
                // recurrence; never invoke the optimizing differentiator.
                let du = reference_sub_derivative(base, var);
                let dv = reference_sub_derivative(exp, var);
                if du.is_zero() && dv.is_zero() {
                    Expr::from_i64(0)
                } else if dv.is_zero() {
                    Expr::Mul(vec![
                        exp.as_ref().clone(),
                        Expr::pow(
                            base.as_ref().clone(),
                            Expr::Add(vec![exp.as_ref().clone(), Expr::from_i64(-1)]),
                        ),
                        du,
                    ])
                } else if du.is_zero() {
                    Expr::Mul(vec![
                        expr.clone(),
                        Expr::Function("log".to_string(), vec![base.as_ref().clone()]),
                        dv,
                    ])
                } else {
                    Expr::Mul(vec![
                        expr.clone(),
                        Expr::Add(vec![
                            Expr::Mul(vec![
                                dv,
                                Expr::Function("log".to_string(), vec![base.as_ref().clone()]),
                            ]),
                            Expr::Mul(vec![
                                exp.as_ref().clone(),
                                du,
                                Expr::pow(base.as_ref().clone(), Expr::from_i64(-1)),
                            ]),
                        ]),
                    ])
                }
            }
        }
        Expr::Function(name, args) if name == "atan2" && args.len() == 2 => {
            let y = &args[0];
            let x = &args[1];
            let dy = reference_sub_derivative(y, var);
            let dx = reference_sub_derivative(x, var);
            if dy.is_zero() && dx.is_zero() {
                Expr::from_i64(0)
            } else {
                Expr::Mul(vec![
                    Expr::Add(vec![
                        Expr::Mul(vec![x.clone(), dy]),
                        Expr::Mul(vec![Expr::from_i64(-1), y.clone(), dx]),
                    ]),
                    Expr::pow(
                        Expr::Add(vec![
                            Expr::pow(x.clone(), Expr::from_i64(2)),
                            Expr::pow(y.clone(), Expr::from_i64(2)),
                        ]),
                        Expr::from_i64(-1),
                    ),
                ])
            }
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let u = &args[0];
            let du = reference_sub_derivative(u, var);
            if name != "Derivative" && name != "diff" && du.is_zero() {
                return Expr::from_i64(0);
            }
            match name.as_str() {
                "sin" => Expr::Mul(vec![Expr::Function("cos".to_string(), vec![u.clone()]), du]),
                "cos" => Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("sin".to_string(), vec![u.clone()]),
                    du,
                ]),
                "tan" => {
                    let tan_u_sq = Expr::pow(
                        Expr::Function("tan".to_string(), vec![u.clone()]),
                        Expr::from_i64(2),
                    );
                    Expr::Mul(vec![Expr::Add(vec![Expr::from_i64(1), tan_u_sq]), du])
                }
                "cot" => {
                    let square = Expr::pow(
                        Expr::Function("cot".to_string(), vec![u.clone()]),
                        Expr::from_i64(2),
                    );
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::Add(vec![Expr::from_i64(1), square]),
                        du,
                    ])
                }
                "sec" => Expr::Mul(vec![
                    Expr::Function("sec".to_string(), vec![u.clone()]),
                    Expr::Function("tan".to_string(), vec![u.clone()]),
                    du,
                ]),
                "csc" => Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("csc".to_string(), vec![u.clone()]),
                    Expr::Function("cot".to_string(), vec![u.clone()]),
                    du,
                ]),
                "exp" => Expr::Mul(vec![Expr::Function("exp".to_string(), vec![u.clone()]), du]),
                "sinh" => Expr::Mul(vec![
                    Expr::Function("cosh".to_string(), vec![u.clone()]),
                    du,
                ]),
                "cosh" => Expr::Mul(vec![
                    Expr::Function("sinh".to_string(), vec![u.clone()]),
                    du,
                ]),
                "tanh" => {
                    if form == DerivativeForm::Numeric {
                        return Expr::Mul(vec![
                            Expr::pow(
                                Expr::Function("sech".to_string(), vec![u.clone()]),
                                Expr::from_i64(2),
                            ),
                            du,
                        ]);
                    }
                    let tanh_u_sq = Expr::pow(
                        Expr::Function("tanh".to_string(), vec![u.clone()]),
                        Expr::from_i64(2),
                    );
                    Expr::Mul(vec![
                        Expr::Add(vec![
                            Expr::from_i64(1),
                            Expr::Mul(vec![Expr::from_i64(-1), tanh_u_sq]),
                        ]),
                        du,
                    ])
                }
                "coth" => {
                    if form != DerivativeForm::Symbolic {
                        return Expr::Mul(vec![
                            Expr::from_i64(-1),
                            Expr::pow(
                                Expr::Function("csch".to_string(), vec![u.clone()]),
                                Expr::from_i64(2),
                            ),
                            du,
                        ]);
                    }
                    let square = Expr::pow(
                        Expr::Function("coth".to_string(), vec![u.clone()]),
                        Expr::from_i64(2),
                    );
                    Expr::Mul(vec![
                        Expr::Add(vec![
                            Expr::from_i64(1),
                            Expr::Mul(vec![Expr::from_i64(-1), square]),
                        ]),
                        du,
                    ])
                }
                "sech" => Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("sech".to_string(), vec![u.clone()]),
                    Expr::Function("tanh".to_string(), vec![u.clone()]),
                    du,
                ]),
                "csch" => Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Function("csch".to_string(), vec![u.clone()]),
                    Expr::Function("coth".to_string(), vec![u.clone()]),
                    du,
                ]),
                "sqrt" => Expr::Mul(vec![
                    Expr::Rational(BigRational::new(1.into(), 2.into())),
                    Expr::pow(
                        u.clone(),
                        Expr::Rational(BigRational::new((-1).into(), 2.into())),
                    ),
                    du,
                ]),
                "log" | "ln" => Expr::Mul(vec![
                    Expr::Pow(Arc::new(u.clone()), Arc::new(Expr::from_i64(-1))),
                    du,
                ]),
                "asin" => {
                    let one_minus_u_sq = Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::Mul(vec![
                            Expr::from_i64(-1),
                            Expr::pow(u.clone(), Expr::from_i64(2)),
                        ]),
                    ]);
                    let neg_half =
                        Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                    Expr::Mul(vec![Expr::pow(one_minus_u_sq, neg_half), du])
                }
                "acos" => {
                    let one_minus_u_sq = Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::Mul(vec![
                            Expr::from_i64(-1),
                            Expr::pow(u.clone(), Expr::from_i64(2)),
                        ]),
                    ]);
                    let neg_half =
                        Expr::Rational(BigRational::new(BigInt::from(-1), BigInt::from(2)));
                    Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(one_minus_u_sq, neg_half),
                        du,
                    ])
                }
                "atan" => {
                    let denom = Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]);
                    Expr::Mul(vec![Expr::pow(denom, Expr::from_i64(-1)), du])
                }
                "asinh" => {
                    let denominator = Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]);
                    let exponent = Expr::Rational(BigRational::new((-1).into(), 2.into()));
                    Expr::Mul(vec![Expr::pow(denominator, exponent), du])
                }
                "acosh" => {
                    let exponent = Expr::Rational(BigRational::new((-1).into(), 2.into()));
                    Expr::Mul(vec![
                        Expr::pow(
                            Expr::Add(vec![u.clone(), Expr::from_i64(-1)]),
                            exponent.clone(),
                        ),
                        Expr::pow(Expr::Add(vec![u.clone(), Expr::from_i64(1)]), exponent),
                        du,
                    ])
                }
                "atanh" => {
                    let denominator = Expr::Add(vec![
                        Expr::from_i64(1),
                        Expr::Mul(vec![
                            Expr::from_i64(-1),
                            Expr::pow(u.clone(), Expr::from_i64(2)),
                        ]),
                    ]);
                    Expr::Mul(vec![Expr::pow(denominator, Expr::from_i64(-1)), du])
                }
                "erf" => {
                    let pi = Expr::Const(Constant::Pi);
                    let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                    let sqrt_pi = Expr::pow(pi, half);
                    let inv_sqrt_pi = Expr::pow(sqrt_pi, Expr::from_i64(-1));
                    let neg_u_sq = Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]);
                    let exp_neg_u_sq = Expr::Function("exp".to_string(), vec![neg_u_sq]);
                    Expr::Mul(vec![Expr::from_i64(2), inv_sqrt_pi, exp_neg_u_sq, du])
                }
                "erfc" => {
                    let pi = Expr::Const(Constant::Pi);
                    let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
                    let sqrt_pi = Expr::pow(pi, half);
                    let inv_sqrt_pi = Expr::pow(sqrt_pi, Expr::from_i64(-1));
                    let neg_u_sq = Expr::Mul(vec![
                        Expr::from_i64(-1),
                        Expr::pow(u.clone(), Expr::from_i64(2)),
                    ]);
                    let exp_neg_u_sq = Expr::Function("exp".to_string(), vec![neg_u_sq]);
                    Expr::Mul(vec![Expr::from_i64(-2), inv_sqrt_pi, exp_neg_u_sq, du])
                }
                "sinc" => {
                    if form == DerivativeForm::Numeric {
                        return Expr::Mul(vec![
                            Expr::Function("_fsym_sinc_d1".to_string(), vec![u.clone()]),
                            du,
                        ]);
                    }
                    let cos_u = Expr::Function("cos".to_string(), vec![u.clone()]);
                    let sin_u = Expr::Function("sin".to_string(), vec![u.clone()]);
                    let inv_u = Expr::pow(u.clone(), Expr::from_i64(-1));
                    let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                    let term1 = Expr::Mul(vec![cos_u, inv_u]);
                    let term2 = Expr::Mul(vec![Expr::from_i64(-1), sin_u, inv_u_sq]);
                    Expr::Mul(vec![Expr::Add(vec![term1, term2]), du])
                }
                "_fsym_sinc_d1" => Expr::Mul(vec![
                    Expr::Function("_fsym_sinc_d2".to_string(), vec![u.clone()]),
                    du,
                ]),
                "abs" | "Abs" if form == DerivativeForm::Numeric => Expr::Mul(vec![
                    Expr::Function("_fsym_abs_d1".to_string(), vec![u.clone()]),
                    du,
                ]),
                "_fsym_abs_d1" if form == DerivativeForm::Numeric => Expr::Mul(vec![
                    Expr::Function("_fsym_abs_d2".to_string(), vec![u.clone()]),
                    du,
                ]),
                _ => make_diff_term(expr, var),
            }
        }
        _ => make_diff_term(expr, var),
    }
}

/// Validates that the RHS formula corresponds to the mathematical differentiation rule.
fn verify_rule_reduction_semantics(
    expr: &Expr,
    var: &Symbol,
    deriv: &Expr,
    rule_name: &str,
    form: DerivativeForm,
) -> Result<(), KernelError> {
    let reference_sub_derivative =
        |child: &Expr, variable: &Symbol| reference_sub_derivative_with_form(child, variable, form);
    let reference_unsimplified_diff = |source: &Expr, variable: &Symbol| {
        reference_unsimplified_diff_with_form(source, variable, form)
    };
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
                let expected_terms: Vec<Expr> = terms
                    .iter()
                    .map(|t| reference_sub_derivative(t, var))
                    .collect();
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
                            prod_factors.push(reference_sub_derivative(factor, var));
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
                let du = reference_sub_derivative(base, var);
                let expected_rhs = Expr::Mul(vec![
                    Expr::Integer(n.clone()),
                    Expr::Pow(base.clone(), Arc::new(Expr::Integer(n_minus_1))),
                    du.clone(),
                ]);
                // Preserve old persisted raw power proofs while admitting
                // only the independently derived numeric identity form.
                let identity = if form != DerivativeForm::Symbolic && n.is_zero() {
                    Some(Expr::from_i64(0))
                } else if form != DerivativeForm::Symbolic && n.is_one() {
                    Some(du)
                } else {
                    None
                };
                if deriv != &expected_rhs && identity.as_ref() != Some(deriv) {
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
        RULE_DIFF_POW_RATIONAL => {
            if let Expr::Pow(base, exponent) = expr
                && let Expr::Rational(rational) = exponent.as_ref()
            {
                let expected_rhs = reference_unsimplified_diff(expr, var);
                // Older numeric artifacts retain the literal 0*u^-1*u' or
                // 1*u^0*u' form. Accept exactly that independently rebuilt
                // form as well; arbitrary equivalent expressions still fail.
                let legacy_identity = if form != DerivativeForm::Symbolic
                    && (rational.numer().is_zero() || rational.numer() == rational.denom())
                {
                    let shifted = BigRational::new(
                        rational.numer() - rational.denom(),
                        rational.denom().clone(),
                    );
                    Some(Expr::Mul(vec![
                        Expr::Rational(rational.clone()),
                        Expr::Pow(
                            base.clone(),
                            Arc::new(fsym_core::elementary::rational_expr(shifted)),
                        ),
                        reference_sub_derivative(base, var),
                    ]))
                } else {
                    None
                };
                if deriv != &expected_rhs && legacy_identity.as_ref() != Some(deriv) {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!(
                            "Rational power derivative mismatch: expected {expected_rhs}, got {deriv}"
                        ),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected Pow expression with exact rational exponent".to_string(),
                });
            }
        }
        RULE_DIFF_POW_GENERAL => {
            if let Expr::Pow(_, exponent) = expr
                && !matches!(exponent.as_ref(), Expr::Integer(_) | Expr::Rational(_))
            {
                let expected_rhs = reference_unsimplified_diff(expr, var);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason:
                            "Variable power derivative disagrees with the independent chain rule"
                                .to_string(),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected Pow expression with nonliteral exponent".to_string(),
                });
            }
        }
        RULE_DIFF_SIN => {
            if let Expr::Function(name, args) = expr
                && name == "sin"
                && args.len() == 1
            {
                let u = &args[0];
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let expected_rhs = reference_unsimplified_diff(expr, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
        RULE_DIFF_COT | RULE_DIFF_SEC | RULE_DIFF_CSC | RULE_DIFF_COTH | RULE_DIFF_SECH
        | RULE_DIFF_CSCH => {
            let expected_name = match rule_name {
                RULE_DIFF_COT => "cot",
                RULE_DIFF_SEC => "sec",
                RULE_DIFF_CSC => "csc",
                RULE_DIFF_COTH => "coth",
                RULE_DIFF_SECH => "sech",
                _ => "csch",
            };
            if !matches!(expr, Expr::Function(name, args) if name == expected_name && args.len() == 1)
            {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!("Expected unary {expected_name} for {rule_name}"),
                });
            }
            let expected_rhs = reference_unsimplified_diff(expr, var);
            if deriv != &expected_rhs {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!(
                        "{expected_name} derivative mismatch: expected {expected_rhs}, got {deriv}"
                    ),
                });
            }
        }
        RULE_DIFF_ATAN2 => {
            if !matches!(expr, Expr::Function(name, args) if name == "atan2" && args.len() == 2) {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected binary atan2(y, x) for diff_atan2".to_string(),
                });
            }
            let expected_rhs = reference_unsimplified_diff(expr, var);
            if deriv != &expected_rhs {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!(
                        "Atan2 derivative mismatch: expected {expected_rhs}, got {deriv}"
                    ),
                });
            }
        }
        RULE_DIFF_ASINH | RULE_DIFF_ACOSH | RULE_DIFF_ATANH => {
            let expected_name = match rule_name {
                RULE_DIFF_ASINH => "asinh",
                RULE_DIFF_ACOSH => "acosh",
                _ => "atanh",
            };
            if !matches!(expr, Expr::Function(name, args) if name == expected_name && args.len() == 1)
            {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!("Expected unary {expected_name} for {rule_name}"),
                });
            }
            // The reference recurrence is independent of diff_unsimplified;
            // it rechecks the whole chain rule, including the inner source.
            let expected_rhs = reference_unsimplified_diff(expr, var);
            if deriv != &expected_rhs {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!(
                        "{expected_name} derivative mismatch: expected {expected_rhs}, got {deriv}"
                    ),
                });
            }
        }
        RULE_DIFF_ERF => {
            if let Expr::Function(name, args) = expr
                && name == "erf"
                && args.len() == 1
            {
                let u = &args[0];
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
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
                let du = reference_sub_derivative(u, var);
                let expected_rhs = if du.is_zero() {
                    Expr::from_i64(0)
                } else if form == DerivativeForm::Numeric {
                    // The entire derivative has the convergent power series
                    // sum_{n>=1} (-1)^n 2n*u^(2n-1)/(2n+1)!, whose value
                    // at zero is zero. This is an exact semantic node; the
                    // floating approximation is checked separately.
                    Expr::Mul(vec![
                        Expr::Function("_fsym_sinc_d1".to_string(), vec![u.clone()]),
                        du,
                    ])
                } else {
                    let cos_u = Expr::Function("cos".to_string(), vec![u.clone()]);
                    let sin_u = Expr::Function("sin".to_string(), vec![u.clone()]);
                    let inv_u = Expr::pow(u.clone(), Expr::from_i64(-1));
                    let inv_u_sq = Expr::pow(u.clone(), Expr::from_i64(-2));
                    let term1 = Expr::Mul(vec![cos_u, inv_u]);
                    let term2 = Expr::Mul(vec![Expr::from_i64(-1), sin_u, inv_u_sq]);
                    Expr::Mul(vec![Expr::Add(vec![term1, term2]), du])
                };
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
        RULE_DIFF_SINC_D1 => {
            let Expr::Function(name, arguments) = expr else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected the entire sinc first derivative".to_string(),
                });
            };
            if name != "_fsym_sinc_d1" || arguments.len() != 1 {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected unary _fsym_sinc_d1".to_string(),
                });
            }
            // Differentiating the entire first-derivative series term by
            // term gives the second jet, with its exact value -1/3 at zero.
            let inner = reference_sub_derivative(&arguments[0], var);
            let expected = if inner.is_zero() {
                Expr::from_i64(0)
            } else {
                Expr::Mul(vec![
                    Expr::Function("_fsym_sinc_d2".to_string(), vec![arguments[0].clone()]),
                    inner,
                ])
            };
            if deriv != &expected {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason:
                        "Entire sinc second derivative disagrees with the independent chain rule"
                            .to_string(),
                });
            }
        }
        RULE_DIFF_ABS | RULE_DIFF_ABS_D1 => {
            let Expr::Function(name, arguments) = expr else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected a guarded real absolute-value derivative".to_string(),
                });
            };
            let valid_source = if rule_name == RULE_DIFF_ABS {
                matches!(name.as_str(), "abs" | "Abs")
            } else {
                name == "_fsym_abs_d1"
            };
            if form != DerivativeForm::Numeric || arguments.len() != 1 || !valid_source {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Absolute-value jet rules require their exact unary numeric source"
                        .to_string(),
                });
            }
            // For real u > 0, |u| = u; for u < 0, |u| = -u. Hence the
            // first derivative is +/-1, and its derivative is zero on each
            // open branch. Retain the private second jet instead of erasing
            // its nonzero-domain obligation. Constancy is checked separately,
            // matching the generator's early-zero rule without invoking it.
            let inner = reference_sub_derivative(&arguments[0], var);
            let expected = if inner.is_zero() {
                Expr::from_i64(0)
            } else {
                let jet = if rule_name == RULE_DIFF_ABS {
                    "_fsym_abs_d1"
                } else {
                    "_fsym_abs_d2"
                };
                Expr::Mul(vec![
                    Expr::Function(jet.to_string(), vec![arguments[0].clone()]),
                    inner,
                ])
            };
            if deriv != &expected {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason:
                        "Absolute-value derivative disagrees with the independent real chain rule"
                            .to_string(),
                });
            }
        }
        RULE_DIFF_SQRT => {
            if let Expr::Function(name, args) = expr
                && name == "sqrt"
                && args.len() == 1
            {
                // This is the principal-root formal derivative, not a claim
                // that its real value is defined at zero or across a branch
                // boundary. Numeric admission checks those original domains.
                let expected_rhs = reference_unsimplified_diff(expr, var);
                if deriv != &expected_rhs {
                    return Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: "Square-root derivative disagrees with the independent chain rule"
                            .to_string(),
                    });
                }
            } else {
                return Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "Expected unary sqrt expression for diff_sqrt".to_string(),
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
