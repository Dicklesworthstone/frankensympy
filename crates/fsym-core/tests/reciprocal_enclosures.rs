//! Fixed exact trigonometric/hyperbolic identities exercise the independent
//! original-source C1/C2 rules, including both second-order chain terms.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Constant, Expr, RealBall, RootBoxError, Symbol, enclose_objective,
    enclose_residual_jacobian, metered_enclose_objective, metered_enclose_residual_jacobian,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

struct ExactCase {
    name: &'static str,
    argument: Expr,
    value: BigRational,
    slope: BigRational,
    curvature: BigRational,
}

fn cases() -> Vec<ExactCase> {
    // sin(atan(3/4))=3/5, cos(atan(3/4))=4/5;
    // sinh(log(2))=3/4, cosh(log(2))=5/4.
    // These fixed rational values do not use the symbolic differentiator or
    // generated derivative programs that the independent checker must verify.
    let circular = function("atan", Expr::Rational(q(3, 4)));
    let hyperbolic = function("log", Expr::from_i64(2));
    [
        ("cot", circular.clone(), q(4, 3), q(-25, 9), q(200, 27)),
        ("sec", circular.clone(), q(5, 4), q(15, 16), q(85, 32)),
        ("csc", circular, q(5, 3), q(-20, 9), q(205, 27)),
        ("coth", hyperbolic.clone(), q(5, 3), q(-16, 9), q(160, 27)),
        ("sech", hyperbolic.clone(), q(4, 5), q(-12, 25), q(-28, 125)),
        ("csch", hyperbolic, q(4, 3), q(-20, 9), q(164, 27)),
    ]
    .into_iter()
    .map(|(name, argument, value, slope, curvature)| ExactCase {
        name,
        argument,
        value,
        slope,
        curvature,
    })
    .collect()
}

#[test]
fn original_source_values_and_both_derivatives_match_exact_identities() {
    let x = Symbol::new("reciprocal_x");
    for case in cases() {
        for negative in [false, true] {
            let even = matches!(case.name, "sec" | "sech");
            let sign = if negative { -1 } else { 1 };
            let argument = Expr::Add(vec![
                Expr::Sym(x.clone()),
                Expr::from_i64(sign) * case.argument.clone(),
            ]);
            let source = function(case.name, argument);
            let value = if negative && !even {
                -case.value.clone()
            } else {
                case.value.clone()
            };
            let slope = if negative && even {
                -case.slope.clone()
            } else {
                case.slope.clone()
            };
            let curvature = if negative && !even {
                -case.curvature.clone()
            } else {
                case.curvature.clone()
            };
            let first = enclose_residual_jacobian(
                std::slice::from_ref(&source),
                std::slice::from_ref(&x),
                &[RealBall::from_i64(0)],
                24,
            )
            .unwrap_or_else(|error| panic!("{} C1: {error}", case.name));
            assert!(
                first.residual_values()[0].contains(&value),
                "{} value",
                case.name
            );
            assert!(
                first.jacobian()[0][0].contains(&slope),
                "{} slope",
                case.name
            );
            let second = enclose_objective(
                &source,
                std::slice::from_ref(&x),
                &[RealBall::from_i64(0)],
                24,
            )
            .unwrap_or_else(|error| panic!("{} C2: {error}", case.name));
            assert!(second.value().contains(&value), "{} value", case.name);
            assert!(second.gradient()[0].contains(&slope), "{} slope", case.name);
            assert!(
                second.hessian()[0][0].contains(&curvature),
                "{} curvature",
                case.name
            );
            assert_eq!(second.objective(), &source);
        }
    }
}

#[test]
fn finite_width_boxes_enclose_the_known_interior_values_and_derivatives() {
    let x = Symbol::new("reciprocal_box_x");
    // Both atan(3/4) and log(2) are in [3/5, 7/10].
    let domain = [RealBall::new(q(13, 20), q(1, 20)).unwrap()];
    for case in cases() {
        let source = function(case.name, Expr::Sym(x.clone()));
        let first = enclose_residual_jacobian(
            std::slice::from_ref(&source),
            std::slice::from_ref(&x),
            &domain,
            24,
        )
        .unwrap_or_else(|error| panic!("{} C1: {error}", case.name));
        let second = enclose_objective(&source, std::slice::from_ref(&x), &domain, 24)
            .unwrap_or_else(|error| panic!("{} C2: {error}", case.name));
        assert!(first.residual_values()[0].contains(&case.value));
        assert!(first.jacobian()[0][0].contains(&case.slope));
        assert!(second.value().contains(&case.value));
        assert!(second.gradient()[0].contains(&case.slope));
        assert!(second.hessian()[0][0].contains(&case.curvature));
    }
}

#[test]
fn mixed_nonlinear_arguments_keep_inner_hessian_and_outer_curvature() {
    let variables = [Symbol::new("reciprocal_x"), Symbol::new("reciprocal_y")];
    let product = Expr::Sym(variables[0].clone()) * Expr::Sym(variables[1].clone());
    let domain = [RealBall::from_i64(0), RealBall::from_i64(2)];
    for case in cases() {
        let source = function(case.name, product.clone() + case.argument);
        let second = enclose_objective(&source, &variables, &domain, 24)
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert!(second.gradient()[0].contains(&(q(2, 1) * &case.slope)));
        assert!(second.gradient()[1].contains(&q(0, 1)));
        assert!(second.hessian()[0][0].contains(&(q(4, 1) * &case.curvature)));
        assert!(second.hessian()[0][1].contains(&case.slope));
        assert!(second.hessian()[1][0].contains(&case.slope));
        assert!(second.hessian()[1][1].contains(&q(0, 1)));
    }
}

#[test]
fn secant_and_hyperbolic_secant_are_smooth_across_zero() {
    let x = Symbol::new("reciprocal_origin");
    let domain = [RealBall::new(q(0, 1), q(1, 4)).unwrap()];
    for (name, curvature) in [("sec", 1), ("sech", -1)] {
        let source = function(name, Expr::Sym(x.clone()));
        let result = enclose_objective(&source, std::slice::from_ref(&x), &domain, 24).unwrap();
        assert!(result.value().contains(&q(1, 1)));
        assert!(result.gradient()[0].contains(&q(0, 1)));
        assert!(result.hessian()[0][0].contains(&q(curvature, 1)));
    }
}

#[test]
fn poles_and_original_undefined_children_survive_zero_factors_and_powers() {
    let x = Symbol::new("reciprocal_pole");
    let domain = [RealBall::new(q(0, 1), q(1, 20)).unwrap()];
    for name in ["cot", "sec", "csc", "coth", "csch"] {
        let shift = if name == "sec" {
            Expr::Rational(q(1, 2)) * Expr::Const(Constant::Pi)
        } else {
            Expr::from_i64(0)
        };
        let pole = function(name, Expr::Sym(x.clone()) + shift);
        for source in [
            pole.clone(),
            Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
            power(pole, 0),
        ] {
            assert!(
                enclose_residual_jacobian(
                    std::slice::from_ref(&source),
                    std::slice::from_ref(&x),
                    &domain,
                    24,
                )
                .is_err(),
                "{name}: C1 pole was hidden"
            );
            assert!(
                enclose_objective(&source, std::slice::from_ref(&x), &domain, 24).is_err(),
                "{name}: C2 pole was hidden"
            );
        }
    }
    for case in cases() {
        let bad = Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![
                Expr::from_i64(0),
                function("log", Expr::Sym(x.clone())),
            ]),
        ]);
        let source = function(case.name, bad);
        assert!(
            enclose_residual_jacobian(
                std::slice::from_ref(&source),
                std::slice::from_ref(&x),
                &domain,
                24,
            )
            .is_err()
        );
        assert!(enclose_objective(&source, std::slice::from_ref(&x), &domain, 24).is_err());
    }
}

#[test]
fn scalar_ball_dispatch_preserves_exact_values_and_rejects_hyperbolic_poles() {
    for case in cases() {
        let source = function(case.name, case.argument);
        assert!(
            source.evalf_ball(24).unwrap().contains(&case.value),
            "{}",
            case.name
        );
    }
    for name in ["cot", "csc", "coth", "csch"] {
        assert!(function(name, Expr::from_i64(0)).evalf_ball(24).is_err());
    }
}

#[derive(Default)]
struct Cancel {
    visits: usize,
    stop: Option<usize>,
}

impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let visit = self.visits;
        self.visits += 1;
        if self.stop == Some(visit) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn reciprocal_interval_work_retains_cancellation_and_cumulative_budgets() {
    let x = Symbol::new("reciprocal_budget");
    let source = function("csch", Expr::Sym(x.clone()));
    let domain = [RealBall::exact(q(1, 1))];
    for second_order in [false, true] {
        let evaluate = |meter: &mut _| {
            if second_order {
                metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, meter)
                    .map(|_| ())
            } else {
                metered_enclose_residual_jacobian(
                    std::slice::from_ref(&source),
                    std::slice::from_ref(&x),
                    &domain,
                    24,
                    meter,
                )
                .map(|_| ())
            }
        };
        let mut count = Cancel::default();
        evaluate(&mut count).unwrap();
        for stop in [0, count.visits / 2, count.visits - 1] {
            let mut cancelled = Cancel {
                visits: 0,
                stop: Some(stop),
            };
            assert_eq!(
                evaluate(&mut cancelled),
                Err(RootBoxError::Meter(MeterError::Cancelled))
            );
        }
    }
    let limit = 512 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, &mut budget).unwrap();
    let after_first = budget.remaining(Dimension::ComputeSteps);
    metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, &mut budget).unwrap();
    assert!(after_first < limit && budget.remaining(Dimension::ComputeSteps) < after_first);
    let mut empty = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, &mut empty),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
}
