//! Typed rewrite rules and proof-producing rewrite engine for WS07.
//!
//! Every rewrite emits a verified [`ProofRule`] recorded in the kernel.
//! Unproven or unverified rewrites are rejected by construction.

#![forbid(unsafe_code)]

use fsym_assumptions::{ImmutableAssumptionsSnapshot, Predicate};
use fsym_core::{BigRational, Expr};
use fsym_proof_kernel::ProofRule;
use num_traits::Zero;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Function signature for a verified rewrite transformation.
pub type RuleTransform = fn(&Expr, &Arc<ImmutableAssumptionsSnapshot>) -> Option<(Expr, ProofRule)>;

/// The sub-expression a side condition constrains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideConditionOperand {
    /// The base of a `Pow` expression.
    Base,
    /// The n-th argument of a function application.
    Arg(usize),
    /// The `inner`-th argument of the `outer`-th argument's function
    /// application (e.g. for `exp(log(u))`, the constrained `u` is
    /// `NestedArg { outer: 0, inner: 0 }`).
    NestedArg { outer: usize, inner: usize },
}

/// A typed side condition declared at rule registration time.
///
/// A `Entailed` condition fires only when the assumption context
/// (including syntactic inherent facts such as literal rational values)
/// entails the predicate for the operand. A rule with an undischarged
/// side condition stays guarded: the engine refuses to apply it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideCondition {
    Unconditional,
    Entailed {
        operand: SideConditionOperand,
        predicate: Predicate,
    },
}

/// A verified local rewrite rule producing proof kernel rule steps.
#[derive(Clone)]
pub struct RewriteRule {
    pub name: &'static str,
    pub description: &'static str,
    /// Typed side conditions, all of which must be discharged before the
    /// engine may apply the rule. Declared here so the guard is registry
    /// data, not merely the transform's private discipline.
    pub side_conditions: &'static [SideCondition],
    pub transform: RuleTransform,
}

/// Resolves a side-condition operand to the sub-expression it constrains.
fn resolve_operand<'a>(expr: &'a Expr, operand: &SideConditionOperand) -> Option<&'a Expr> {
    match (operand, expr) {
        (SideConditionOperand::Base, Expr::Pow(base, _)) => Some(base),
        (SideConditionOperand::Arg(index), Expr::Function(_, args)) => args.get(*index),
        (SideConditionOperand::NestedArg { outer, inner }, Expr::Function(_, outer_args)) => {
            outer_args
                .get(*outer)
                .and_then(|outer_expr| match outer_expr {
                    Expr::Function(_, inner_args) => inner_args.get(*inner),
                    _ => None,
                })
        }
        _ => None,
    }
}

/// True when every declared side condition of `rule` is discharged for
/// `expr` under `context`. Undischarged conditions keep the rule guarded.
pub fn side_conditions_discharged(
    rule: &RewriteRule,
    expr: &Expr,
    context: &Arc<ImmutableAssumptionsSnapshot>,
) -> bool {
    rule.side_conditions
        .iter()
        .all(|condition| match condition {
            SideCondition::Unconditional => true,
            SideCondition::Entailed { operand, predicate } => resolve_operand(expr, operand)
                .is_some_and(|target| context.query(target, *predicate).is_entailed_true()),
        })
}

/// Fundamental rewrite rule catalog for algebraic expressions.
pub fn standard_rules() -> Vec<RewriteRule> {
    vec![
        RewriteRule {
            name: "add_zero_identity",
            description: "x + 0 => x",
            side_conditions: &[],
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
            side_conditions: &[],
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
            side_conditions: &[],
            transform: |expr, _ctx| match expr {
                Expr::Mul(factors) => {
                    if factors.iter().any(|f| f.is_zero())
                        && factors.iter().all(crate::is_total_expr)
                    {
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
            side_conditions: &[SideCondition::Entailed {
                operand: SideConditionOperand::Base,
                predicate: Predicate::NonZero,
            }],
            transform: |expr, ctx| match expr {
                Expr::Pow(base, exp) => {
                    if exp.is_zero() {
                        // Syntactic non-zero constants discharge inherently
                        // (inherent facts); symbolic bases need an entailed
                        // NonZero from the context. Undischarged: stay guarded.
                        let discharged = !base.is_zero()
                            && (matches!(base.as_ref(), Expr::Integer(_) | Expr::Rational(_))
                                || ctx.query(base, Predicate::NonZero).is_entailed_true());
                        if discharged {
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
            side_conditions: &[],
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
            description: "sin(0) => 0, cos(0) => 1, tan(0) => 0, exp(0) => 1, sinh(0) => 0, cosh(0) => 1, tanh(0) => 0, asin(0) => 0, atan(0) => 0, asinh(0) => 0, atanh(0) => 0, sec(0) => 1, sech(0) => 1, sinc(0) => 1, erf(0) => 0, erfc(0) => 1",
            side_conditions: &[],
            transform: |expr, _ctx| match expr {
                Expr::Function(name, args) if args.len() == 1 && args[0].is_zero() => {
                    match name.as_str() {
                        "sin" | "tan" | "sinh" | "tanh" | "asin" | "atan" | "asinh" | "atanh"
                        | "erf" => {
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
                        "cos" | "cosh" | "exp" | "sec" | "sech" | "sinc" | "erfc" => {
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
        RewriteRule {
            name: "elementary_one_eval",
            description: "acos(1) => 0, acosh(1) => 0, asec(1) => 0, asech(1) => 0, ln(1) => 0, log(1) => 0",
            side_conditions: &[],
            transform: |expr, _ctx| match expr {
                Expr::Function(name, args) if args.len() == 1 && args[0].is_one() => {
                    match name.as_str() {
                        "acos" | "acosh" | "asec" | "asech" | "ln" | "log" => {
                            let out = Expr::from_i64(0);
                            Some((
                                out.clone(),
                                ProofRule::DefinitionalReduction {
                                    lhs: expr.clone(),
                                    rhs: out,
                                    rule_name: "elementary_one_eval".into(),
                                },
                            ))
                        }
                        _ => None,
                    }
                }
                _ => None,
            },
        },
        RewriteRule {
            name: "pythagorean_identity",
            description: "c*sin(u)^2 + c*cos(u)^2 => c; c*sec(u)^2 - c*tan(u)^2 => c; c*csc(u)^2 - c*cot(u)^2 => c; c*cosh(u)^2 - c*sinh(u)^2 => c (exact coefficient pairing, root Add level)",
            side_conditions: &[],
            transform: |expr, _ctx| match expr {
                Expr::Add(terms) => fold_pythagorean_terms(terms).map(|folded| {
                    let out = rebuilt_pythagorean_add(folded);
                    (
                        out.clone(),
                        ProofRule::DefinitionalReduction {
                            lhs: expr.clone(),
                            rhs: out,
                            rule_name: "pythagorean_identity".into(),
                        },
                    )
                }),
                _ => None,
            },
        },
        RewriteRule {
            name: "exp_log_inverse",
            description: "exp(log(u)) => u and exp(ln(u)) => u (only when u is provably Positive)",
            side_conditions: &[SideCondition::Entailed {
                operand: SideConditionOperand::NestedArg { outer: 0, inner: 0 },
                predicate: Predicate::Positive,
            }],
            transform: |expr, ctx| match expr {
                Expr::Function(name, args) if name == "exp" && args.len() == 1 => {
                    let Expr::Function(inner_name, inner_args) = &args[0] else {
                        return None;
                    };
                    if !matches!(inner_name.as_str(), "log" | "ln") || inner_args.len() != 1 {
                        return None;
                    }
                    if !ctx
                        .query(&inner_args[0], Predicate::Positive)
                        .is_entailed_true()
                    {
                        return None;
                    }
                    let out = inner_args[0].clone();
                    Some((
                        out.clone(),
                        ProofRule::DefinitionalReduction {
                            lhs: expr.clone(),
                            rhs: out,
                            rule_name: "exp_log_inverse".into(),
                        },
                    ))
                }
                _ => None,
            },
        },
        RewriteRule {
            name: "log_exp_inverse",
            description: "log(exp(u)) => u and ln(exp(u)) => u (only when u is provably Real)",
            side_conditions: &[SideCondition::Entailed {
                operand: SideConditionOperand::NestedArg { outer: 0, inner: 0 },
                predicate: Predicate::Real,
            }],
            transform: |expr, ctx| match expr {
                Expr::Function(name, args)
                    if matches!(name.as_str(), "log" | "ln") && args.len() == 1 =>
                {
                    let Expr::Function(inner_name, inner_args) = &args[0] else {
                        return None;
                    };
                    if inner_name != "exp" || inner_args.len() != 1 {
                        return None;
                    }
                    if !ctx
                        .query(&inner_args[0], Predicate::Real)
                        .is_entailed_true()
                    {
                        return None;
                    }
                    let out = inner_args[0].clone();
                    Some((
                        out.clone(),
                        ProofRule::DefinitionalReduction {
                            lhs: expr.clone(),
                            rhs: out,
                            rule_name: "log_exp_inverse".into(),
                        },
                    ))
                }
                _ => None,
            },
        },
    ]
}

/// Family table for the four Pythagorean identity pairs.
///
/// Family 0 satisfies `sin^2 + cos^2 = 1`, so its term coefficients must
/// match exactly; families 1, 2, and 3 satisfy `sec^2 - tan^2 = 1`,
/// `csc^2 - cot^2 = 1`, and `cosh^2 - sinh^2 = 1`, so their term
/// coefficients must be exact opposites.
pub(crate) fn pythagorean_family_slot(name: &str) -> Option<(u8, u8)> {
    match name {
        "sin" => Some((0, 0)),
        "cos" => Some((0, 1)),
        "sec" => Some((1, 0)),
        "tan" => Some((1, 1)),
        "csc" => Some((2, 0)),
        "cot" => Some((2, 1)),
        "cosh" => Some((3, 0)),
        "sinh" => Some((3, 1)),
        _ => None,
    }
}

/// Whether coefficients `a` and `b` complete one identity fold for `family`.
pub(crate) fn pythagorean_coefficients_fold(family: u8, a: &BigRational, b: &BigRational) -> bool {
    if family == 0 {
        a == b
    } else {
        // `a == -b`, expressed additively because the exact rational
        // substrate guarantees checked addition rather than a Neg impl.
        let mut sum = a.clone();
        sum += b.clone();
        sum.is_zero()
    }
}

/// Classifies one canonical additive term `coeff * f(arg)^2` into its
/// Pythagorean identity family.
///
/// Returns `(family, slot, argument, coefficient)`; the exponent must be the
/// exact integer 2, the function must be single-argument, and the coefficient
/// must be a nonzero exact rational.
fn classify_pythagorean_square(term: &Expr) -> Option<(u8, u8, Expr, BigRational)> {
    let (coeff, key) = crate::split_coeff(term);
    if coeff.is_zero() {
        return None;
    }
    let Expr::Pow(base, exponent) = &key else {
        return None;
    };
    if **exponent != Expr::from_i64(2) {
        return None;
    }
    let Expr::Function(name, args) = &**base else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let (family, slot) = pythagorean_family_slot(name)?;
    Some((family, slot, args[0].clone(), coeff))
}

/// Fold candidate slots keyed by `(family, argument)`; `None` marks an
/// ambiguous duplicate shape that must never fold.
type PythagoreanSlots = BTreeMap<(u8, Expr), Option<[Option<(BigRational, usize)>; 2]>>;

/// Cancels Pythagorean identity pairs across canonical additive terms.
///
/// Returns `None` when no pair folds. Surviving terms keep their order and
/// each completed pair contributes its common coefficient as an additive
/// constant. Input terms are expected in `collect_terms` output shape; the
/// fold is root-level over the given slice only and never descends.
pub(crate) fn fold_pythagorean_terms(terms: &[Expr]) -> Option<Vec<Expr>> {
    let mut slots = PythagoreanSlots::new();

    for (index, term) in terms.iter().enumerate() {
        let Some((family, slot, arg, coeff)) = classify_pythagorean_square(term) else {
            continue;
        };
        match slots.entry((family, arg)) {
            std::collections::btree_map::Entry::Vacant(vacant) => {
                let mut pair = [None, None];
                pair[slot as usize] = Some((coeff, index));
                vacant.insert(Some(pair));
            }
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                let conflict =
                    matches!(occupied.get(), Some(pair) if pair[slot as usize].is_some());
                if conflict {
                    occupied.insert(None);
                } else if let Some(pair) = occupied.get_mut() {
                    pair[slot as usize] = Some((coeff, index));
                }
            }
        }
    }

    let mut remove: Vec<usize> = Vec::new();
    let mut constants: Vec<BigRational> = Vec::new();
    for ((family, _arg), pair) in &slots {
        let Some(pair) = pair else { continue };
        let (Some((a_coeff, a_index)), Some((b_coeff, b_index))) =
            (pair[0].clone(), pair[1].clone())
        else {
            continue;
        };
        if pythagorean_coefficients_fold(*family, &a_coeff, &b_coeff) {
            remove.push(a_index);
            remove.push(b_index);
            constants.push(a_coeff);
        }
    }
    if remove.is_empty() {
        return None;
    }
    let mut out: Vec<Expr> = terms
        .iter()
        .enumerate()
        .filter(|(index, _)| !remove.contains(index))
        .map(|(_, term)| term.clone())
        .collect();
    for constant in constants {
        if !constant.is_zero() {
            out.push(crate::rational_expr(constant));
        }
    }
    Some(out)
}

/// Rebuilds an additive expression from folded terms.
pub(crate) fn rebuilt_pythagorean_add(folded: Vec<Expr>) -> Expr {
    match folded.len() {
        0 => Expr::from_i64(0),
        1 => folded.into_iter().next().expect("non-empty checked"),
        _ => Expr::Add(folded),
    }
}

/// Applies rewrite rules in the registry to an expression at the root level.
pub fn apply_step(
    expr: &Expr,
    rules: &[RewriteRule],
    context: &Arc<ImmutableAssumptionsSnapshot>,
) -> Option<(Expr, ProofRule)> {
    for rule in rules {
        // Registry-level guard: a rule whose declared side conditions are
        // not discharged stays guarded even if its transform would fire.
        if !side_conditions_discharged(rule, expr, context) {
            continue;
        }
        if let Some(res) = (rule.transform)(expr, context) {
            return Some(res);
        }
    }
    None
}

#[cfg(test)]
mod guard_tests {
    use super::*;
    use fsym_assumptions::AssumptionsContext;
    use fsym_core::Symbol;

    fn empty_context() -> Arc<ImmutableAssumptionsSnapshot> {
        Arc::new(ImmutableAssumptionsSnapshot::clone(
            &ImmutableAssumptionsSnapshot::empty(),
        ))
    }

    fn context_with(symbol: &str, predicate: Predicate) -> Arc<ImmutableAssumptionsSnapshot> {
        let mut context = AssumptionsContext::new();
        context
            .assume(Symbol::new(symbol), predicate)
            .expect("assumption");
        Arc::new(context.snapshot())
    }

    fn rules() -> Vec<RewriteRule> {
        standard_rules()
    }

    /// Positive observable: literal non-zero bases discharge inherently.
    #[test]
    fn pow_zero_literal_discharges_without_assumptions() {
        let five_pow_zero = parse("5^0");
        let (out, _) = apply_step(&five_pow_zero, &rules(), &empty_context()).expect("fires");
        assert_eq!(out, Expr::from_i64(1));
    }

    /// Positive observable: an entailed assumption discharges the condition.
    #[test]
    fn pow_zero_entailed_nonzero_discharges() {
        let x_pow_zero = parse("x^0");
        let context = context_with("x", Predicate::NonZero);
        let (out, _) = apply_step(&x_pow_zero, &rules(), &context).expect("fires");
        assert_eq!(out, Expr::from_i64(1));
    }

    /// Planted negative: with the NonZero side condition UNDISCHARGED, the
    /// conditional rewrite stays guarded - it must not fire.
    #[test]
    fn pow_zero_with_undischarged_condition_stays_guarded() {
        let x_pow_zero = parse("x^0");
        assert!(
            apply_step(&x_pow_zero, &rules(), &empty_context()).is_none(),
            "x^0 must stay guarded when NonZero is not entailed"
        );
        // Entailed-false (x assumed Zero) also stays guarded.
        let context = context_with("x", Predicate::Zero);
        assert!(apply_step(&x_pow_zero, &rules(), &context).is_none());
    }

    /// Planted negative for the inverse rule: an entailed-false Positive
    /// condition keeps exp(log(u)) guarded even though the transform's own
    /// pattern matches.
    #[test]
    fn exp_log_inverse_with_entailed_false_positive_stays_guarded() {
        let u = Symbol::new("u");
        let exp_log_u = Expr::Function(
            "exp".to_string(),
            vec![Expr::Function(
                "log".to_string(),
                vec![Expr::Sym(u.clone())],
            )],
        );
        let context = context_with("u", Predicate::Negative);
        assert!(
            apply_step(&exp_log_u, &rules(), &context).is_none(),
            "exp(log(u)) must stay guarded when Positive is entailed-false"
        );
    }

    /// Positive discharge for exp_log_inverse: an entailed Positive
    /// condition fires the rule.
    #[test]
    fn exp_log_positive_discharges_when_entailed() {
        let u = Symbol::new("u");
        let exp_log_u = Expr::Function(
            "exp".to_string(),
            vec![Expr::Function(
                "log".to_string(),
                vec![Expr::Sym(u.clone())],
            )],
        );
        let context = context_with("u", Predicate::Positive);
        let (out, _) = apply_step(&exp_log_u, &rules(), &context).expect("fires");
        assert_eq!(out, Expr::Sym(u));
    }

    /// Positive discharge for log_exp_inverse: an entailed Real condition
    /// fires the rule.
    #[test]
    fn log_exp_positive_discharges_when_entailed() {
        let u = Symbol::new("u");
        let log_exp_u = Expr::Function(
            "log".to_string(),
            vec![Expr::Function(
                "exp".to_string(),
                vec![Expr::Sym(u.clone())],
            )],
        );
        let context = context_with("u", Predicate::Real);
        let (out, _) = apply_step(&log_exp_u, &rules(), &context).expect("fires");
        assert_eq!(out, Expr::Sym(u));
    }

    /// Planted negative for log_exp_inverse: without a Real entailment the
    /// conditional rewrite stays guarded even though the transform's own
    /// pattern matches.
    #[test]
    fn log_exp_undischarged_condition_stays_guarded() {
        let u = Symbol::new("u");
        let log_exp_u = Expr::Function(
            "log".to_string(),
            vec![Expr::Function(
                "exp".to_string(),
                vec![Expr::Sym(u.clone())],
            )],
        );
        assert!(
            apply_step(&log_exp_u, &rules(), &empty_context()).is_none(),
            "log(exp(u)) must stay guarded when Real is not entailed"
        );
    }

    /// The assumptions lattice derives Negative ⊨ Real, so an entailed
    /// Negative still discharges the Real condition and the rule fires —
    /// log(exp(u)) = u holds for every real u, negative included. (There
    /// is no expressible entailed-false-Real predicate: every sign
    /// assumption in the lattice implies Real.)
    #[test]
    fn log_exp_negative_entailment_implies_real_and_fires() {
        let u = Symbol::new("u");
        let log_exp_u = Expr::Function(
            "log".to_string(),
            vec![Expr::Function(
                "exp".to_string(),
                vec![Expr::Sym(u.clone())],
            )],
        );
        let context = context_with("u", Predicate::Negative);
        let (out, _) = apply_step(&log_exp_u, &rules(), &context).expect("fires");
        assert_eq!(out, Expr::Sym(u));
    }

    /// Registry-level guard mutation: a rule whose transform fires
    /// unconditionally must still be refused by the engine when its
    /// declared side condition is undischarged.
    #[test]
    fn registry_guard_refuses_undisciplined_transform() {
        let rules = vec![RewriteRule {
            name: "rogue_rule",
            description: "always fires, but declares an undischarged condition",
            side_conditions: &[SideCondition::Entailed {
                operand: SideConditionOperand::Base,
                predicate: Predicate::NonZero,
            }],
            transform: |expr, _ctx| {
                Some((
                    Expr::from_i64(1),
                    ProofRule::DefinitionalReduction {
                        lhs: expr.clone(),
                        rhs: Expr::from_i64(1),
                        rule_name: "rogue_rule".into(),
                    },
                ))
            },
        }];
        // A Pow with a symbolic base: the Base operand resolves, but NonZero
        // is undischarged on the empty context.
        let x_pow = parse("x^1");
        assert!(
            apply_step(&x_pow, &rules, &empty_context()).is_none(),
            "the registry guard must refuse an undisciplined transform"
        );
        // With the condition discharged, the same rule fires.
        let context = context_with("x", Predicate::NonZero);
        assert!(apply_step(&x_pow, &rules, &context).is_some());
    }

    use fsym_proof_kernel::ProofRule;

    fn parse(source: &str) -> Expr {
        fsym_core::parse(source).expect("parses")
    }
}
