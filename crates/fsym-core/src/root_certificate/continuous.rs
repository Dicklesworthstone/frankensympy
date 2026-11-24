//! Private, source-preserving continuous value enclosures.
//!
//! Continuity on the complete closed box is sufficient for the intermediate
//! value theorem and interval-range integral bounds. It does not imply bounded
//! slopes, a smooth neighborhood, a derivative enclosure or a Krawczyk claim.
//! This evaluator has no derivative fields and cannot construct a smooth jet.
//!
//! The closed real domains of sqrt, positive literal rational powers,
//! asin/acos and acosh include their finite boundary values. Every original
//! child is visited before an algebraic shortcut, and Piecewise guards retain
//! the existing strict neighborhood-stability check. Poles, branch cuts,
//! non-real values and unsupported expressions never become removable by
//! evaluating a continuous outer function.

use super::{
    ABSOLUTE_VALUE_CORNER, MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_NODES,
    RootBoxError, Work, checked,
};
use crate::{BigRational, Constant, Expr, RealBall, Symbol};
use fsym_budget::BudgetMeter;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// Value range of an independently evaluated continuous original expression.
/// No derivative, Lipschitz or smoothness property is carried by this type.
pub(super) struct ContinuousEnclosure {
    pub(super) value: RealBall,
}

/// Only these existing regularity refusals can justify a separate continuity
/// attempt. That attempt must still recheck the entire original expression:
/// the same derivative refusal can also describe a non-real argument. No
/// arithmetic, unsupported-expression, resource or cancellation error retries.
pub(super) fn is_continuity_only_boundary(error: &RootBoxError) -> bool {
    match error {
        RootBoxError::Domain(reason) => {
            *reason == ABSOLUTE_VALUE_CORNER
                || matches!(
                    *reason,
                    "square root is not C1 on this box"
                        | "objective square root is not C2 on this box"
                        | "noninteger power base is not strictly positive"
                        | "objective noninteger power base is not strictly positive"
                        | "inverse circular argument is not strictly inside (-1, 1)"
                        | "acosh argument is not strictly above one"
                )
        }
        _ => false,
    }
}

/// The caller admits the source, ordered typed axes and complete box first.
/// The same Work ledger is used across any earlier failed derivative attempt;
/// this evaluator never resets its node, arithmetic or allocation counters.
pub(super) fn evaluate_continuous(
    expression: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ContinuousEnclosure, RootBoxError> {
    Ok(ContinuousEnclosure {
        value: evaluate_inner(expression, variables, domain, 0, work)?,
    })
}

fn evaluate_inner(
    expression: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    depth: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RealBall, RootBoxError> {
    work.step()?;
    if depth > MAX_ROOT_BOX_DEPTH || work.nodes >= MAX_ROOT_BOX_NODES * 2 {
        return Err(RootBoxError::ResourceLimit(
            "continuous expression depth or node count",
        ));
    }
    work.charge(1, 512, 8)?;
    work.nodes += 1;
    let result = match expression {
        Expr::Integer(value) => {
            if value.limb_count() > MAX_ROOT_BOX_LIMBS {
                return Err(RootBoxError::ResourceLimit("integer height"));
            }
            work.charge(1, 128 + value.limb_count() * 16, 2)?;
            RealBall::exact(BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value) => {
            work.arithmetic(&[], &[value])?;
            RealBall::exact(value.clone())
        }
        Expr::Const(Constant::Pi) => {
            work.elementary(&RealBall::from_i64(1))?;
            RealBall::pi(work.precision)?
        }
        Expr::Const(Constant::E) => work.exp(&RealBall::from_i64(1))?,
        Expr::Const(_) => return Err(RootBoxError::Domain("non-real or non-finite constant")),
        Expr::Sym(symbol) => {
            let index = variables
                .iter()
                .position(|variable| variable == symbol)
                .ok_or(RootBoxError::Unsupported("unbound symbol identity"))?;
            work.arithmetic(&[&domain[index]], &[])?;
            domain[index].clone()
        }
        Expr::Add(children) | Expr::Mul(children) => {
            if children.len() > MAX_ROOT_BOX_NODES {
                return Err(RootBoxError::ResourceLimit("variadic expression width"));
            }
            let sum = matches!(expression, Expr::Add(_));
            let mut value = RealBall::from_i64(if sum { 0 } else { 1 });
            for child in children {
                let next = evaluate_inner(child, variables, domain, depth + 1, work)?;
                value = if sum {
                    work.add(&value, &next)?
                } else {
                    work.mul(&value, &next)?
                };
            }
            value
        }
        Expr::Pow(base, exponent) => {
            let base_value = evaluate_inner(base, variables, domain, depth + 1, work)?;
            // In particular, 1^exponent cannot erase an undefined exponent.
            let exponent_value = evaluate_inner(exponent, variables, domain, depth + 1, work)?;
            let integer = match exponent.as_ref() {
                Expr::Integer(value) => Some(value),
                Expr::Rational(value) if value.denom().is_one() => Some(value.numer()),
                _ => None,
            };
            if let Some(integer) = integer {
                let power = integer
                    .to_i32()
                    .filter(|power| (-64..=64).contains(power))
                    .ok_or(RootBoxError::ResourceLimit("integer exponent"))?;
                // Retains the existing zero-base/zero-exponent refusal.
                work.power(&base_value, power)?
            } else if let Expr::Rational(power) = exponent.as_ref()
                && power.is_positive()
            {
                positive_rational_power(&base_value, power, work)?
            } else {
                if !base_value.is_positive() {
                    return Err(RootBoxError::Domain(
                        "noninteger power base is not strictly positive",
                    ));
                }
                work.elementary(&base_value)?;
                let logarithm = checked(base_value.ln(work.precision)?)?;
                let exponent = work.mul(&exponent_value, &logarithm)?;
                work.exp(&exponent)?
            }
        }
        Expr::Function(name, arguments) => {
            if name == "Piecewise" {
                let selected = work.select_stable_piecewise(arguments, variables, domain, depth)?;
                return evaluate_inner(selected, variables, domain, depth + 2, work);
            }
            if name == "atan2" {
                let [y, x] = arguments.as_slice() else {
                    return Err(RootBoxError::Unsupported("atan2 requires two arguments"));
                };
                let y = evaluate_inner(y, variables, domain, depth + 1, work)?;
                let x = evaluate_inner(x, variables, domain, depth + 1, work)?;
                // Retain the existing complete-rectangle branch-cut policy.
                let (value, _, _) = work.atan2(&y, &x)?;
                return checked(value);
            }
            if name == "LambertW" {
                let (argument, branch) = super::lambert::argument(arguments)?;
                let argument = evaluate_inner(argument, variables, domain, depth + 1, work)?;
                let (value, _) = work.lambert_w(&argument, branch)?;
                return checked(value);
            }
            let [argument] = arguments.as_slice() else {
                return Err(RootBoxError::Unsupported("non-unary function"));
            };
            let argument = evaluate_inner(argument, variables, domain, depth + 1, work)?;
            unary_value(name, &argument, work)?
        }
    };
    checked(result)
}

/// x^q is continuous and increasing for exact q>0 on the closed nonnegative
/// real axis. Its zero endpoint is exact; a positive endpoint uses the
/// existing directed log/exp kernels. Negative bases never enter this rule,
/// including odd rational denominators: no principal branch is changed.
fn positive_rational_power(
    base: &RealBall,
    power: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RealBall, RootBoxError> {
    work.arithmetic(&[base], &[power])?;
    let lower = base.lower();
    let upper = base.upper();
    if lower.is_negative() {
        return Err(RootBoxError::Domain(
            "positive rational power base is not nonnegative",
        ));
    }
    // The caller has already distinguished the literal noninteger case.
    // Square roots retain exact rational endpoints where available.
    if power.numer().is_one() && power.denom().to_u32() == Some(2) {
        work.elementary(base)?;
        return checked(base.sqrt(work.precision.saturating_mul(4).saturating_add(64))?);
    }
    if upper.is_zero() {
        return Ok(RealBall::from_i64(0));
    }
    if lower.is_zero() {
        let endpoint = RealBall::exact(upper);
        work.elementary(&endpoint)?;
        let logarithm = checked(endpoint.ln(work.precision)?)?;
        let exponent = work.mul(&RealBall::exact(power.clone()), &logarithm)?;
        let endpoint_value = work.exp(&exponent)?;
        work.arithmetic(&[&endpoint_value], &[])?;
        let upper_value = endpoint_value.upper();
        work.arithmetic(&[], &[&upper_value])?;
        let half = upper_value / BigRational::from_integer(2.into());
        work.arithmetic(&[], &[&half])?;
        return checked(RealBall::new(half.clone(), half)?);
    }
    work.elementary(base)?;
    let logarithm = checked(base.ln(work.precision)?)?;
    let exponent = work.mul(&RealBall::exact(power.clone()), &logarithm)?;
    work.exp(&exponent)
}

fn unary_value(
    name: &str,
    argument: &RealBall,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RealBall, RootBoxError> {
    match name {
        "abs" | "Abs" => {
            work.arithmetic(&[argument], &[])?;
            checked(argument.abs())
        }
        "sqrt" => {
            work.elementary(argument)?;
            // The complete argument must be nonnegative; no clipping.
            checked(argument.sqrt(work.precision.saturating_mul(4).saturating_add(64))?)
        }
        "asin" | "acos" => {
            work.arithmetic(&[argument], &[])?;
            if argument.lower() < -BigRational::one() || argument.upper() > BigRational::one() {
                // Range dependency can overshoot a valid source's domain.
                // Keep that refusal refinable by the adaptive partitioner;
                // actual ball precision/resource failures remain terminal.
                return Err(RootBoxError::Domain(
                    "inverse circular argument is not contained in [-1, 1]",
                ));
            }
            // Match the existing conservative reservation for both monotone
            // endpoint evaluations, their roots, atan/pi and enclosing hull.
            for _ in 0..8 {
                work.elementary(argument)?;
            }
            checked(if name == "acos" {
                argument.acos(work.precision)?
            } else {
                argument.asin(work.precision)?
            })
        }
        "acosh" => {
            work.arithmetic(&[argument], &[])?;
            if argument.lower() < BigRational::one() {
                return Err(RootBoxError::Domain(
                    "acosh argument is not contained in [1, infinity)",
                ));
            }
            // The closed-domain ball kernel admits one and rejects all input
            // below one. No inverse derivative is evaluated at its endpoint.
            for _ in 0..8 {
                work.elementary(argument)?;
            }
            checked(argument.acosh(work.precision)?)
        }
        _ => {
            // These pre-existing strict-domain rules are conservative for
            // continuity. Their derivative outputs have no authority here.
            let (value, _) = work.smooth_unary_rule(name, argument)?;
            checked(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BigInt, SymbolIdentity};
    use fsym_budget::{BudgetError, Dimension, MeterError, Unbounded};
    use std::cell::Cell;
    use std::sync::Arc;

    fn q(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
    }

    fn interval(lower: i64, upper: i64) -> RealBall {
        RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
    }

    fn integer(value: i64) -> Expr {
        Expr::Integer(value.into())
    }

    fn function(name: &str, argument: Expr) -> Expr {
        Expr::Function(name.into(), vec![argument])
    }

    fn power(base: Expr, exponent: Expr) -> Expr {
        Expr::Pow(Arc::new(base), Arc::new(exponent))
    }

    fn enclose(
        source: &Expr,
        variables: &[Symbol],
        domain: &[RealBall],
    ) -> Result<RealBall, RootBoxError> {
        let mut meter = Unbounded;
        let mut work = Work::new(&mut meter, 16)?;
        super::super::validate_inputs(std::slice::from_ref(source), variables, domain, &mut work)?;
        Ok(evaluate_continuous(source, variables, domain, &mut work)?.value)
    }

    fn scalar(source: &Expr, domain: RealBall) -> Result<RealBall, RootBoxError> {
        enclose(source, &[Symbol::new("x")], &[domain])
    }

    #[test]
    fn square_root_admits_closed_endpoints_and_continuous_compositions() {
        let source = function("sqrt", Expr::symbol("x"));
        assert_eq!(scalar(&source, interval(0, 4)).unwrap(), interval(0, 2));
        assert_eq!(scalar(&source, interval(0, 0)).unwrap(), interval(0, 0));
        let nested = function("sin", source);
        let value = scalar(&nested, interval(0, 1)).unwrap();
        assert!(value.contains(&q(0, 1)));
        assert!(value.upper() > q(0, 1));
        assert!(scalar(&nested, interval(-1, 1)).is_err());
    }

    #[test]
    fn positive_literal_rational_powers_include_the_exact_zero_endpoint() {
        for (numerator, denominator, upper, endpoint) in [(1, 3, 8, 2), (2, 3, 27, 9)] {
            let source = power(Expr::symbol("x"), Expr::Rational(q(numerator, denominator)));
            let value = scalar(&source, interval(0, upper)).unwrap();
            assert_eq!(value.lower(), q(0, 1));
            assert!(value.contains(&q(endpoint, 1)));
            assert_eq!(scalar(&source, interval(0, 0)).unwrap(), interval(0, 0));
            assert!(scalar(&source, interval(-1, upper)).is_err());
        }
        let half = power(Expr::symbol("x"), Expr::Rational(q(1, 2)));
        assert_eq!(scalar(&half, interval(0, 4)).unwrap(), interval(0, 2));
    }

    #[test]
    fn nonliteral_and_nonpositive_powers_keep_the_strict_positive_branch() {
        let dynamic = power(Expr::symbol("x"), Expr::Add(vec![Expr::Rational(q(1, 2))]));
        let negative = power(Expr::symbol("x"), Expr::Rational(q(-1, 2)));
        for source in [dynamic, negative] {
            assert!(matches!(
                scalar(&source, interval(0, 1)),
                Err(RootBoxError::Domain(
                    "noninteger power base is not strictly positive"
                ))
            ));
            assert!(scalar(&source, interval(1, 4)).is_ok());
        }
    }

    #[test]
    fn integer_power_limits_and_zero_to_zero_are_unchanged() {
        for exponent in [integer(65), Expr::Rational(q(65, 1))] {
            let source = power(integer(1), exponent);
            assert!(matches!(
                scalar(&source, interval(0, 1)),
                Err(RootBoxError::ResourceLimit("integer exponent"))
            ));
        }
        let zero_power = power(Expr::symbol("x"), integer(0));
        assert!(scalar(&zero_power, interval(0, 1)).is_err());
        assert_eq!(
            scalar(&zero_power, interval(1, 2)).unwrap(),
            RealBall::from_i64(1)
        );
    }

    #[test]
    fn inverse_function_values_include_only_their_closed_real_domains() {
        let x = Expr::symbol("x");
        let asin = function("asin", x.clone());
        let positive = scalar(&asin, interval(1, 1)).unwrap();
        assert!(positive.lower() > q(1, 1));
        assert!(positive.upper() < q(2, 1));
        let negative = scalar(&asin, interval(-1, -1)).unwrap();
        assert!(negative.lower() > q(-2, 1));
        assert!(negative.upper() < q(-1, 1));
        assert!(scalar(&asin, interval(-1, 2)).is_err());
        let acos = function("acos", x.clone());
        assert_eq!(scalar(&acos, interval(1, 1)).unwrap(), interval(0, 0));
        assert!(scalar(&acos, interval(-2, 1)).is_err());
        let acosh = function("acosh", x);
        assert_eq!(scalar(&acosh, interval(1, 1)).unwrap(), interval(0, 0));
        let value = scalar(&acosh, interval(1, 2)).unwrap();
        assert!(value.contains(&q(0, 1)));
        assert!(value.contains(&q(1, 1)));
        assert!(scalar(&acosh, interval(0, 2)).is_err());
    }

    #[test]
    fn raw_children_preserve_hidden_poles_and_nonreal_domains() {
        let pole = power(Expr::symbol("x"), integer(-1));
        let cases = [
            Expr::Mul(vec![integer(0), pole.clone()]),
            power(integer(1), pole.clone()),
            power(pole.clone(), integer(0)),
            function("sqrt", Expr::Mul(vec![integer(0), pole])),
            Expr::Mul(vec![integer(0), function("log", Expr::symbol("x"))]),
            function("sqrt", Expr::Add(vec![Expr::symbol("x"), integer(-1)])),
            function("atanh", Expr::symbol("x")),
        ];
        for source in cases {
            assert!(scalar(&source, interval(0, 1)).is_err());
        }
        let branch_cut = Expr::Function("atan2".into(), vec![integer(0), integer(-1)]);
        assert!(scalar(&branch_cut, interval(0, 1)).is_err());
    }

    #[test]
    fn typed_axis_identity_and_coordinate_order_are_preserved() {
        let first = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
        let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
        assert!(matches!(
            enclose(&Expr::Sym(other), &[first], &[interval(0, 1)],),
            Err(RootBoxError::Unsupported("unbound symbol identity"))
        ));
        let source = Expr::Add(vec![function("sqrt", Expr::symbol("y")), Expr::symbol("x")]);
        assert_eq!(
            enclose(
                &source,
                &[Symbol::new("y"), Symbol::new("x")],
                &[interval(9, 9), interval(2, 2)],
            )
            .unwrap(),
            RealBall::from_i64(5)
        );
    }

    #[test]
    fn piecewise_retains_guard_stability_and_inactive_branch_policy() {
        let branch = |value, condition| Expr::Function("Tuple".into(), vec![value, condition]);
        let pole = power(integer(0), integer(-1));
        let stable = Expr::Function(
            "Piecewise".into(),
            vec![
                branch(function("sqrt", Expr::symbol("x")), Expr::symbol("True")),
                branch(pole, Expr::symbol("True")),
            ],
        );
        assert_eq!(scalar(&stable, interval(0, 1)).unwrap(), interval(0, 1));
        let unstable = Expr::Function(
            "Piecewise".into(),
            vec![
                branch(
                    function("sqrt", Expr::symbol("x")),
                    Expr::Function("LessThan".into(), vec![Expr::symbol("x"), integer(0)]),
                ),
                branch(integer(1), Expr::symbol("True")),
            ],
        );
        assert!(scalar(&unstable, interval(0, 1)).is_err());
    }

    #[test]
    fn continuous_endpoints_never_acquire_first_or_second_derivatives() {
        let cases = [
            (function("sqrt", Expr::symbol("x")), interval(0, 1)),
            (function("asin", Expr::symbol("x")), interval(0, 1)),
            (function("acos", Expr::symbol("x")), interval(0, 1)),
            (function("acosh", Expr::symbol("x")), interval(1, 2)),
            (
                power(Expr::symbol("x"), Expr::Rational(q(1, 3))),
                interval(0, 1),
            ),
        ];
        for (source, domain) in cases {
            let variables = [Symbol::new("x")];
            let domain = [domain];
            let mut meter = Unbounded;
            let mut work = Work::new(&mut meter, 16).unwrap();
            super::super::validate_inputs(
                std::slice::from_ref(&source),
                &variables,
                &domain,
                &mut work,
            )
            .unwrap();
            assert!(evaluate_continuous(&source, &variables, &domain, &mut work).is_ok());
            assert!(work.eval(&source, &variables, &domain, 0).is_err());
            assert!(
                super::super::objective::evaluate_objective(
                    &source, &variables, &domain, &mut work,
                )
                .is_err()
            );
        }
    }

    struct StoppingMeter<'a> {
        stop: &'a Cell<bool>,
        failure: MeterError,
    }

    impl BudgetMeter for StoppingMeter<'_> {
        fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
            self.checkpoint()
        }

        fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            self.checkpoint()
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.stop.get() {
                Err(self.failure)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn cancellation_and_budget_failures_propagate_after_source_admission() {
        for failure in [
            MeterError::Cancelled,
            MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                requested: 1,
                remaining: 0,
            }),
        ] {
            let stop = Cell::new(false);
            let mut meter = StoppingMeter {
                stop: &stop,
                failure,
            };
            let mut work = Work::new(&mut meter, 16).unwrap();
            let source = function("sqrt", Expr::symbol("x"));
            let variables = [Symbol::new("x")];
            let domain = [interval(0, 1)];
            super::super::validate_inputs(
                std::slice::from_ref(&source),
                &variables,
                &domain,
                &mut work,
            )
            .unwrap();
            stop.set(true);
            assert!(matches!(
                evaluate_continuous(&source, &variables, &domain, &mut work),
                Err(RootBoxError::Meter(error)) if error == failure
            ));
        }
    }

    #[test]
    fn continuation_uses_existing_node_and_work_counters() {
        let mut meter = Unbounded;
        let mut work = Work::new(&mut meter, 16).unwrap();
        work.nodes = MAX_ROOT_BOX_NODES * 2;
        assert!(matches!(
            evaluate_continuous(
                &integer(1),
                &[Symbol::new("x")],
                &[interval(0, 1)],
                &mut work,
            ),
            Err(RootBoxError::ResourceLimit(
                "continuous expression depth or node count"
            ))
        ));
        assert_eq!(work.nodes, MAX_ROOT_BOX_NODES * 2);
    }

    #[test]
    fn only_specific_regularity_refusals_allow_a_continuity_attempt() {
        for reason in [
            ABSOLUTE_VALUE_CORNER,
            "square root is not C1 on this box",
            "objective square root is not C2 on this box",
            "noninteger power base is not strictly positive",
            "objective noninteger power base is not strictly positive",
            "inverse circular argument is not strictly inside (-1, 1)",
            "acosh argument is not strictly above one",
        ] {
            assert!(is_continuity_only_boundary(&RootBoxError::Domain(reason)));
        }
        for error in [
            RootBoxError::Domain("logarithm argument is not strictly positive"),
            RootBoxError::Domain("atanh argument is not strictly inside (-1, 1)"),
            RootBoxError::Unsupported("unbound symbol identity"),
            RootBoxError::ResourceLimit("cumulative arithmetic work"),
            RootBoxError::Meter(MeterError::Cancelled),
            RootBoxError::Meter(MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                requested: 1,
                remaining: 0,
            })),
        ] {
            assert!(!is_continuity_only_boundary(&error));
        }
    }
}
