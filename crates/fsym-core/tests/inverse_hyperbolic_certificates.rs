//! Original-source inverse-hyperbolic acceptance with exact independent oracles.
//! The log(2) series and rational derivatives below are never verifier inputs.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BallError, BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness,
    StationaryPointKind, Symbol, certify_root_box, certify_stationary_point, enclose_objective,
    enclose_residual_jacobian, metered_enclose_objective, metered_enclose_residual_jacobian,
};
use num_traits::Zero;
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

/// log(2) = 2 * sum(1 / ((2n+1)*3^(2n+1))). Positivity and a
/// geometric tail give a rational bracket independent of production logs.
fn log_two_bounds() -> (BigRational, BigRational) {
    let mut power = q(1, 3);
    let mut lower = BigRational::zero();
    for index in 0..96 {
        lower += q(2, 2 * index + 1) * &power;
        power /= q(9, 1);
    }
    let tail = q(2, 193) * power / (q(1, 1) - q(1, 9));
    (lower.clone(), lower + tail)
}

fn contains_bounds(ball: &RealBall, lower: &BigRational, upper: &BigRational) {
    assert!(ball.lower() <= *lower, "lower bound: {ball} versus {lower}");
    assert!(ball.upper() >= *upper, "upper bound: {ball} versus {upper}");
}

fn value(name: &str, argument: &RealBall, digits: u32) -> Result<RealBall, BallError> {
    match name {
        "asinh" => argument.asinh(digits),
        "acosh" => argument.acosh(digits),
        "atanh" => argument.atanh(digits),
        _ => unreachable!(),
    }
}

#[test]
fn rational_log_two_anchors_and_odd_symmetry_have_independent_enclosures() {
    let (lower, upper) = log_two_bounds();
    for (name, point) in [("asinh", q(3, 4)), ("acosh", q(5, 4)), ("atanh", q(3, 5))] {
        let result = value(name, &RealBall::exact(point.clone()), 18).unwrap();
        contains_bounds(&result, &lower, &upper);
        assert!(result.radius() <= &q(1, 1_000_000_000_000_000_000));
        let expression = function(name, Expr::Rational(point.clone()));
        let evaluated = expression.eval_real_ball(18).unwrap();
        contains_bounds(&evaluated, &lower, &upper);
        if name != "acosh" {
            let negative = value(name, &RealBall::exact(-point), 18).unwrap();
            contains_bounds(&negative, &(-upper.clone()), &(-lower.clone()));
            assert_eq!(negative, result.neg());
        }
    }
    assert_eq!(
        RealBall::from_i64(0).asinh(18).unwrap(),
        RealBall::from_i64(0)
    );
    assert_eq!(
        RealBall::from_i64(0).atanh(18).unwrap(),
        RealBall::from_i64(0)
    );
    assert_eq!(
        RealBall::from_i64(1).acosh(18).unwrap(),
        RealBall::from_i64(0)
    );
}

#[test]
fn monotone_interval_ranges_keep_both_endpoints_and_cross_zero_safely() {
    let (lower, upper) = log_two_bounds();
    let tolerance = q(1, 1_000_000_000_000_000);
    for (name, radius) in [("asinh", q(3, 4)), ("atanh", q(3, 5))] {
        let argument = RealBall::new(q(0, 1), radius).unwrap();
        let result = value(name, &argument, 18).unwrap();
        contains_bounds(&result, &(-upper.clone()), &upper);
        assert!(result.lower() >= -&upper - &tolerance);
        assert!(result.upper() <= &upper + &tolerance);
        assert!(result.contains(&q(0, 1)));
    }
    let result = interval(q(1, 1), q(5, 4)).acosh(18).unwrap();
    assert!(result.contains(&q(0, 1)));
    assert!(result.upper() >= upper);
    assert!(result.upper() <= lower + tolerance);
    // Squaring [-2,2] as two independent factors must not invent a negative
    // square and refuse an asinh interval that is smooth everywhere.
    let variables = [Symbol::new("x")];
    let source = function("asinh", Expr::Sym(variables[0].clone()));
    let domain = [RealBall::new(q(0, 1), q(2, 1)).unwrap()];
    let c1 =
        enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 18).unwrap();
    assert!(c1.jacobian()[0][0].contains(&q(1, 1)));
    assert!(c1.jacobian()[0][0].lower() > q(0, 1));
    let c2 = enclose_objective(&source, &variables, &domain, 18).unwrap();
    assert!(c2.hessian()[0][0].contains(&q(0, 1)));
    assert!(c2.hessian()[0][0].contains(&q(48, 125)));
    assert!(c2.hessian()[0][0].contains(&q(-48, 125)));
}

#[test]
fn exact_domain_and_resource_edges_refuse_before_rounding_or_large_work() {
    let epsilon = BigRational::new(BigInt::one(), BigInt::one() << 120u32);
    for argument in [
        RealBall::from_i64(-1),
        RealBall::from_i64(1),
        RealBall::exact(q(1, 1) + &epsilon),
        RealBall::new(q(0, 1), q(1, 1)).unwrap(),
    ] {
        assert!(argument.atanh(18).is_err());
    }
    assert!(RealBall::exact(q(1, 1) - &epsilon).acosh(18).is_err());
    assert!(RealBall::new(q(1, 1), epsilon).unwrap().acosh(18).is_err());
    for name in ["asinh", "acosh", "atanh"] {
        let point = RealBall::exact(if name == "acosh" { q(5, 4) } else { q(3, 5) });
        for precision in [
            0,
            RealBall::MAX_INVERSE_HYPERBOLIC_PRECISION_DIGITS + 1,
            u32::MAX,
        ] {
            assert!(matches!(
                value(name, &point, precision),
                Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
            ));
        }
        let huge_denominator = BigInt::one() << 140_000u32;
        let point = if name == "acosh" {
            q(1, 1) + BigRational::new(BigInt::one(), huge_denominator)
        } else {
            BigRational::new(BigInt::one(), huge_denominator)
        };
        assert!(matches!(
            value(name, &RealBall::exact(point), 18),
            Err(BallError::ResourceLimitExceeded { .. })
        ));
    }
}

#[test]
fn exact_anchor_derivatives_keep_their_signs_and_magnitudes() {
    let variables = [Symbol::new("x")];
    for (name, point, first, second) in [
        ("asinh", q(3, 4), q(4, 5), q(-48, 125)),
        ("asinh", q(-3, 4), q(4, 5), q(48, 125)),
        ("acosh", q(5, 4), q(4, 3), q(-80, 27)),
        ("atanh", q(3, 5), q(25, 16), q(375, 128)),
        ("atanh", q(-3, 5), q(25, 16), q(-375, 128)),
    ] {
        let source = function(name, Expr::Sym(variables[0].clone()));
        let domain = [RealBall::exact(point)];
        let c1 = enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 18)
            .unwrap();
        assert_eq!(c1.residuals(), std::slice::from_ref(&source));
        assert_eq!(c1.domain(), domain);
        assert!(
            c1.jacobian()[0][0].contains(&first),
            "{name} first derivative"
        );
        assert!(!c1.jacobian()[0][0].contains(&(-first.clone())));
        let c2 = enclose_objective(&source, &variables, &domain, 18).unwrap();
        assert!(c2.gradient()[0].contains(&first), "{name} gradient");
        assert!(c2.hessian()[0][0].contains(&second), "{name} Hessian");
        assert!(!c2.hessian()[0][0].contains(&(-second)));
    }
}

#[test]
fn nonlinear_multivariate_chain_keeps_inner_and_outer_curvature_separate() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let inner = Expr::Add(vec![
        power(Expr::Sym(variables[0].clone()), 2),
        Expr::Sym(variables[1].clone()),
    ]);
    for (name, y, first, second) in [
        ("asinh", q(1, 2), q(4, 5), q(-48, 125)),
        ("acosh", q(1, 1), q(4, 3), q(-80, 27)),
        ("atanh", q(7, 20), q(25, 16), q(375, 128)),
    ] {
        let domain = [RealBall::exact(q(1, 2)), RealBall::exact(y)];
        let source = function(name, inner.clone());
        let c2 = enclose_objective(&source, &variables, &domain, 18).unwrap();
        assert!(c2.gradient().iter().all(|entry| entry.contains(&first)));
        assert!(c2.hessian()[0][0].contains(&(&second + q(2, 1) * &first)));
        assert!(!c2.hessian()[0][0].contains(&second));
        for (row, column) in [(0, 1), (1, 0), (1, 1)] {
            assert!(c2.hessian()[row][column].contains(&second));
        }
    }
}

#[test]
fn smooth_domain_checks_cannot_be_erased_by_zero_products_or_held_powers() {
    let variables = [Symbol::new("x")];
    for (name, domains) in [
        (
            "acosh",
            vec![
                RealBall::from_i64(1),
                interval(q(1, 1), q(5, 4)),
                interval(q(0, 1), q(2, 1)),
            ],
        ),
        (
            "atanh",
            vec![
                RealBall::from_i64(-1),
                RealBall::from_i64(1),
                interval(q(-1, 1), q(1, 1)),
                interval(q(0, 1), q(2, 1)),
            ],
        ),
    ] {
        let inverse = function(name, Expr::Sym(variables[0].clone()));
        for domain in domains {
            for source in [
                inverse.clone(),
                Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
                power(inverse.clone(), 0),
            ] {
                assert!(
                    matches!(
                        enclose_residual_jacobian(
                            std::slice::from_ref(&source),
                            &variables,
                            std::slice::from_ref(&domain),
                            18
                        ),
                        Err(RootBoxError::Domain(_))
                    ),
                    "{name}, {domain}, {source}"
                );
                assert!(matches!(
                    enclose_objective(&source, &variables, std::slice::from_ref(&domain), 18),
                    Err(RootBoxError::Domain(_))
                ));
            }
        }
    }
}

#[test]
fn strict_domain_points_smaller_than_a_binary64_ulp_remain_admissible() {
    let variables = [Symbol::new("x")];
    let epsilon = BigRational::new(BigInt::one(), BigInt::one() << 120u32);
    for (name, center) in [("acosh", q(1, 1) + &epsilon), ("atanh", q(1, 1) - &epsilon)] {
        let source = function(name, Expr::Sym(variables[0].clone()));
        let domain = [RealBall::new(center, &epsilon / q(2, 1)).unwrap()];
        let c1 = enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 18)
            .unwrap();
        assert!(c1.jacobian()[0][0].lower() > q(1, 1));
        let c2 = enclose_objective(&source, &variables, &domain, 18).unwrap();
        if name == "acosh" {
            assert!(c2.hessian()[0][0].upper() < q(0, 1));
        } else {
            assert!(c2.hessian()[0][0].lower() > q(0, 1));
        }
    }
}

fn anchor_cases() -> [(&'static str, BigRational, BigRational); 3] {
    [
        ("asinh", q(3, 4), q(4, 5)),
        ("acosh", q(5, 4), q(4, 3)),
        ("atanh", q(3, 5), q(25, 16)),
    ]
}

fn root_source(name: &str, variable: &Symbol) -> Expr {
    function(name, Expr::Sym(variable.clone())) - function("log", Expr::from_i64(2))
}

fn witness(center: BigRational, inverse: BigRational) -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(center.clone(), q(1, 100)).unwrap()],
        center: vec![center],
        preconditioner: vec![vec![inverse]],
        precision_digits: 18,
    }
}

#[test]
fn independent_root_witnesses_certify_all_three_functions_and_reject_mutations() {
    let variables = [Symbol::new("x")];
    for (name, center, slope) in anchor_cases() {
        let source = vec![root_source(name, &variables[0])];
        let proposed = witness(center.clone(), q(1, 1) / slope);
        let accepted = certify_root_box(&source, &variables, &proposed).unwrap();
        assert_eq!(accepted.residuals(), source);
        assert_eq!(accepted.variables(), variables);
        assert_eq!(accepted.witness(), &proposed);
        assert!(accepted.root_enclosure()[0].contains(&center));
        assert!(accepted.root_enclosure()[0].lower() > proposed.domain[0].lower());
        assert!(accepted.root_enclosure()[0].upper() < proposed.domain[0].upper());
        assert!(accepted.contraction_bound() < &q(1, 1));
        let shifted = [source[0].clone() + Expr::from_i64(1)];
        assert!(certify_root_box(&shifted, &variables, &proposed).is_err());
        let mut wrong = proposed;
        wrong.preconditioner[0][0] = -wrong.preconditioner[0][0].clone();
        assert_eq!(
            certify_root_box(&source, &variables, &wrong),
            Err(RootBoxError::NotContractive)
        );
    }
}

#[test]
fn squared_residuals_have_source_checked_strict_minima() {
    let variables = [Symbol::new("x")];
    for (name, center, slope) in anchor_cases() {
        let source = power(root_source(name, &variables[0]), 2);
        let proposed = witness(center.clone(), q(1, 1) / (q(2, 1) * &slope * slope));
        let accepted = certify_stationary_point(&source, &variables, &proposed).unwrap();
        assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
        assert_eq!(accepted.objective(), &source);
        assert_eq!(accepted.witness(), &proposed);
        assert!(accepted.stationary_enclosure()[0].contains(&center));
        assert!(accepted.objective_enclosure().contains(&q(0, 1)));
        assert!(accepted.hessian_enclosure()[0][0].lower() > q(0, 1));
        let shifted = source + Expr::Sym(variables[0].clone());
        assert!(certify_stationary_point(&shifted, &variables, &proposed).is_err());
    }
}

fn metered_case(
    name: &str,
    second: bool,
    meter: &mut impl BudgetMeter,
) -> Result<(), RootBoxError> {
    let variables = [Symbol::new("x")];
    let source = function(name, Expr::Sym(variables[0].clone()));
    let point = match name {
        "asinh" => q(3, 4),
        "acosh" => q(5, 4),
        _ => q(3, 5),
    };
    let domain = [RealBall::exact(point)];
    if second {
        metered_enclose_objective(&source, &variables, &domain, 8, meter).map(|_| ())
    } else {
        metered_enclose_residual_jacobian(&[source], &variables, &domain, 8, meter).map(|_| ())
    }
}

struct Checkpoints {
    visits: usize,
    stop: Option<usize>,
    stopped: bool,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "work after terminal cancellation");
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "work after terminal cancellation");
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "work after terminal cancellation");
        self.visits += 1;
        if self.stop == Some(self.visits) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn interval_pipelines_propagate_cancellation_through_all_composition_stages() {
    for name in ["asinh", "acosh", "atanh"] {
        for second in [false, true] {
            let mut count = Checkpoints {
                visits: 0,
                stop: None,
                stopped: false,
            };
            metered_case(name, second, &mut count).unwrap();
            assert!(count.visits > 4);
            for stop in [
                1,
                count.visits / 4,
                count.visits / 2,
                3 * count.visits / 4,
                count.visits,
            ] {
                let mut meter = Checkpoints {
                    visits: 0,
                    stop: Some(stop),
                    stopped: false,
                };
                assert_eq!(
                    metered_case(name, second, &mut meter),
                    Err(RootBoxError::Meter(MeterError::Cancelled))
                );
                assert_eq!(meter.visits, stop);
                assert!(meter.stopped);
            }
        }
    }
}

#[test]
fn interval_pipelines_charge_exact_limits_without_spending_reserved_verifier_budget() {
    const LIMIT: u64 = 1_000_000_000_000;
    const RESERVE: u64 = 37;
    for name in ["asinh", "acosh", "atanh"] {
        for second in [false, true] {
            let mut measured = Budget::new(BudgetLimits::uniform(LIMIT, RESERVE));
            metered_case(name, second, &mut measured).unwrap();
            for dimension in [
                Dimension::ComputeSteps,
                Dimension::MemoryBytes,
                Dimension::AllocationCount,
            ] {
                let used = LIMIT - measured.remaining(dimension);
                assert!(used > 0);
                let mut limits = BudgetLimits::uniform(LIMIT, RESERVE);
                limits.dimensions[dimension.index()] = used;
                let mut exact = Budget::new(limits);
                metered_case(name, second, &mut exact).unwrap();
                assert_eq!(exact.remaining(dimension), 0);
                assert!(matches!(
                    metered_case(name, second, &mut exact),
                    Err(RootBoxError::Meter(MeterError::Budget(_)))
                ));
                assert_eq!(exact.verifier_remaining(), RESERVE);
                limits.dimensions[dimension.index()] = used - 1;
                let mut short = Budget::new(limits);
                assert!(matches!(
                    metered_case(name, second, &mut short),
                    Err(RootBoxError::Meter(MeterError::Budget(_)))
                ));
                assert_eq!(short.verifier_remaining(), RESERVE);
            }
        }
    }
}
