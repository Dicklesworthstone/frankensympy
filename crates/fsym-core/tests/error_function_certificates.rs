//! Error-function values and derivatives checked against independent formulas.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BallError, BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness,
    StationaryPointKind, Symbol, certify_root_box, certify_stationary_point, enclose_objective,
    enclose_residual_jacobian, metered_enclose_objective, metered_enclose_residual_jacobian,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn decimal_unit(digits: u32) -> BigRational {
    BigRational::new(BigInt::one(), BigInt::from(10).pow(digits))
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn value(name: &str, argument: &RealBall, digits: u32) -> Result<RealBall, BallError> {
    match name {
        "erf" => argument.erf(digits),
        "erfc" => argument.erfc(digits),
        _ => unreachable!(),
    }
}

/// Independent positive series, DLMF 7.6.2 (not the production alternating
/// series or continued fraction). Bound the remaining positive terms by a
/// geometric series once their ratios have fallen below one.
fn positive_series_reference(x: &BigRational) -> RealBall {
    let square = x * x;
    let mut term = q(1, 1);
    let mut sum = q(0, 1);
    for n in 0..192 {
        sum += &term;
        term *= q(2, 2 * n + 3) * &square;
    }
    let ratio = q(2, 387) * &square;
    assert!(ratio < q(1, 1));
    let tail = term / (q(1, 1) - ratio);
    let series = interval(sum.clone(), sum + tail);
    let root_pi = RealBall::pi(70).unwrap().sqrt(344).unwrap();
    let denominator = root_pi.mul(&RealBall::exact(square).exp(70).unwrap());
    series
        .mul(&RealBall::exact(q(2, 1) * x))
        .div(&denominator)
        .unwrap()
}

#[test]
fn scalar_values_match_an_independent_positive_series_across_both_algorithms() {
    for x in [q(1, 2), q(1, 1), q(3, 1), q(4, 1)] {
        let reference = positive_series_reference(&x);
        let actual = RealBall::exact(x.clone()).erf(24).unwrap();
        assert!(!actual.is_disjoint(&reference), "erf({x})");
        assert!(actual.radius() <= &decimal_unit(26));
        let complement = RealBall::exact(x.clone()).erfc(24).unwrap();
        assert!(!complement.is_disjoint(&RealBall::from_i64(1).sub(&reference)));
        assert!(complement.lower() > q(0, 1));
        assert!(complement.width() <= complement.lower() * decimal_unit(26));
        for name in ["erf", "erfc"] {
            let expression = function(name, Expr::Rational(x.clone()));
            assert!(
                !expression
                    .eval_real_ball(24)
                    .unwrap()
                    .is_disjoint(&value(name, &RealBall::exact(x.clone()), 24).unwrap())
            );
        }
    }
    assert_eq!(
        RealBall::from_i64(0).erf(24).unwrap(),
        RealBall::from_i64(0)
    );
    assert_eq!(
        RealBall::from_i64(0).erfc(24).unwrap(),
        RealBall::from_i64(1)
    );
}

#[test]
fn tiny_nonzero_arguments_keep_relative_accuracy_and_exact_symmetries() {
    let x = BigRational::new(BigInt::one(), BigInt::one() << 1000u32);
    let positive = RealBall::exact(x.clone()).erf(32).unwrap();
    assert!(positive.lower() > x);
    assert!(positive.upper() < q(2, 1) * &x);
    assert!(positive.width() <= positive.lower() * decimal_unit(34));
    assert_eq!(RealBall::exact(-x.clone()).erf(32).unwrap(), positive.neg());
    for x in [q(1, 2), q(3, 1), q(4, 1), q(20, 1)] {
        let point = RealBall::exact(x.clone());
        let erf = point.erf(18).unwrap();
        let erfc = point.erfc(18).unwrap();
        assert!(erf.add(&erfc).contains(&q(1, 1)));
        let negative = RealBall::exact(-x).erfc(18).unwrap();
        assert_eq!(negative, RealBall::from_i64(2).sub(&erfc));
    }
}

/// DLMF 7.12.1: on the positive real axis the asymptotic remainder has the
/// sign of its first omitted term and magnitude no larger than that term.
/// Sixteen terms give a positive next term and an independent tail bracket.
fn asymptotic_tail_reference(x: i64) -> RealBall {
    let mut term = q(1, 1);
    let mut sum = q(0, 1);
    for n in 0..16 {
        if n % 2 == 0 {
            sum += &term;
        } else {
            sum -= &term;
        }
        term *= q(2 * n + 1, 2 * x * x);
    }
    let scale = RealBall::from_i64(x)
        .mul(&RealBall::pi(80).unwrap().sqrt(400).unwrap())
        .mul(&RealBall::from_i64(x * x).exp(80).unwrap());
    interval(sum.clone(), sum + term).div(&scale).unwrap()
}

#[test]
fn direct_positive_tails_preserve_hundreds_of_orders_below_absolute_precision() {
    for x in [20, 32] {
        let actual = RealBall::from_i64(x).erfc(24).unwrap();
        let reference = asymptotic_tail_reference(x);
        assert!(actual.lower() >= reference.lower());
        assert!(actual.upper() <= reference.upper());
        assert!(actual.lower() > q(0, 1));
        assert!(actual.upper() < decimal_unit(100));
        assert!(actual.width() <= actual.lower() * decimal_unit(26));
    }
    let huge = RealBall::exact(BigRational::from_integer(BigInt::from(10).pow(1000)));
    let tail = huge.erfc(256).unwrap();
    assert_eq!(tail.lower(), q(0, 1));
    assert!(tail.upper() < decimal_unit(258));
    let saturated = huge.erf(256).unwrap();
    assert!(saturated.contains(&q(1, 1)));
    assert!(saturated.radius() < &decimal_unit(258));
}

#[test]
fn maximum_precision_at_the_continued_fraction_boundary_is_checked_exactly() {
    let maximum = RealBall::from_i64(4).erfc(256).unwrap();
    assert!(maximum.lower() > q(0, 1));
    assert!(maximum.width() <= maximum.lower() * decimal_unit(258));
    for name in ["erf", "erfc"] {
        for digits in [
            0,
            RealBall::MAX_ERROR_FUNCTION_PRECISION_DIGITS + 1,
            u32::MAX,
        ] {
            assert!(matches!(
                value_for_precision(name, digits),
                Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
            ));
        }
        let oversized =
            RealBall::exact(BigRational::new(BigInt::one(), BigInt::one() << 70_000u32));
        assert!(matches!(
            value(name, &oversized, 18),
            Err(BallError::ResourceLimitExceeded { .. })
        ));
    }
}

fn value_for_precision(name: &str, digits: u32) -> Result<RealBall, BallError> {
    value(name, &RealBall::from_i64(1), digits)
}

#[test]
fn monotone_hulls_keep_both_real_endpoints_including_zero_and_both_tails() {
    let input = interval(q(-4, 1), q(20, 1));
    let low = RealBall::from_i64(-4);
    let high = RealBall::from_i64(20);
    let erf = input.erf(24).unwrap();
    assert_eq!(erf.lower(), low.erf(24).unwrap().lower());
    assert_eq!(erf.upper(), high.erf(24).unwrap().upper());
    assert!(erf.contains(&q(0, 1)));
    let erfc = input.erfc(24).unwrap();
    assert_eq!(erfc.lower(), high.erfc(24).unwrap().lower());
    assert_eq!(erfc.upper(), low.erfc(24).unwrap().upper());
    assert!(erfc.contains(&q(1, 1)));
}

fn gaussian_slope(point: i64) -> RealBall {
    RealBall::from_i64(2)
        .div(&RealBall::pi(70).unwrap().sqrt(344).unwrap())
        .unwrap()
        .div(&RealBall::from_i64(point * point).exp(70).unwrap())
        .unwrap()
}

#[test]
fn original_source_first_and_second_derivatives_keep_complement_signs() {
    let variables = [Symbol::new("x")];
    for point in [-1, 0, 1, 20] {
        for name in ["erf", "erfc"] {
            let source = function(name, Expr::Sym(variables[0].clone()));
            let domain = [RealBall::from_i64(point)];
            let mut slope = gaussian_slope(point);
            if name == "erfc" {
                slope = slope.neg();
            }
            let curvature = slope.mul(&RealBall::from_i64(-2 * point));
            let c1 =
                enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24)
                    .unwrap();
            assert!(!c1.jacobian()[0][0].is_disjoint(&slope));
            assert!(!c1.jacobian()[0][0].contains_zero());
            let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
            assert!(!c2.gradient()[0].is_disjoint(&slope));
            assert!(!c2.hessian()[0][0].is_disjoint(&curvature));
            if point == 0 {
                assert_eq!(c2.hessian()[0][0], RealBall::from_i64(0));
            }
        }
    }
}

#[test]
fn nonlinear_multivariate_chain_retains_the_inner_hessian() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let inner = Expr::Add(vec![
        power(Expr::Sym(variables[0].clone()), 2),
        Expr::Sym(variables[1].clone()),
    ]);
    let domain = [RealBall::exact(q(1, 2)), RealBall::exact(q(-1, 4))];
    for name in ["erf", "erfc"] {
        let source = function(name, inner.clone());
        let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
        let mut slope = gaussian_slope(0);
        if name == "erfc" {
            slope = slope.neg();
        }
        assert!(c2.gradient().iter().all(|entry| !entry.is_disjoint(&slope)));
        assert!(!c2.hessian()[0][0].is_disjoint(&slope.mul(&RealBall::from_i64(2))));
        assert!(!c2.hessian()[0][0].contains_zero());
        for (row, column) in [(0, 1), (1, 0), (1, 1)] {
            assert_eq!(c2.hessian()[row][column], RealBall::from_i64(0));
        }
    }
}

#[test]
fn gaussian_quantiles_and_squared_residual_minima_accept_independent_witnesses() {
    let variables = [Symbol::new("x")];
    for name in ["erf", "erfc"] {
        let source = function(name, Expr::Sym(variables[0].clone())) - Expr::Rational(q(1, 2));
        let proposal = RootBoxWitness {
            domain: vec![interval(q(9, 20), q(11, 20))],
            center: vec![q(1, 2)],
            preconditioner: vec![vec![q(if name == "erf" { 1 } else { -1 }, 1)]],
            precision_digits: 24,
        };
        let accepted =
            certify_root_box(std::slice::from_ref(&source), &variables, &proposal).unwrap();
        assert!(accepted.contraction_bound() < &q(1, 1));
        assert!(accepted.root_enclosure()[0].lower() > q(9, 20));
        assert!(accepted.root_enclosure()[0].upper() < q(11, 20));
        let changed = [source.clone() + Expr::from_i64(1)];
        assert!(certify_root_box(&changed, &variables, &proposal).is_err());
        let objective = power(source, 2);
        let mut stationary = proposal;
        stationary.preconditioner[0][0] = q(2, 3);
        let accepted = certify_stationary_point(&objective, &variables, &stationary).unwrap();
        assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
        assert!(accepted.hessian_enclosure()[0][0].lower() > q(0, 1));
    }
}

#[test]
fn tiny_complementary_tail_roots_keep_a_nonzero_certified_jacobian() {
    let variables = [Symbol::new("x")];
    // Relative Gaussian enclosures keep this tail and its derivative even
    // though both are far smaller than the 24-digit absolute working target.
    let source =
        function("erfc", Expr::Sym(variables[0].clone())) - function("erfc", Expr::from_i64(20));
    let proposal = RootBoxWitness {
        domain: vec![RealBall::new(q(20, 1), q(1, 10_000)).unwrap()],
        center: vec![q(20, 1)],
        preconditioner: vec![vec![BigRational::from_integer(
            -BigInt::from(46) * BigInt::from(10).pow(172),
        )]],
        precision_digits: 24,
    };
    let receipt = certify_root_box(&[source], &variables, &proposal).unwrap();
    assert!(receipt.root_enclosure()[0].contains(&q(20, 1)));
    assert!(receipt.contraction_bound() < &q(1, 1));
}

#[test]
fn error_functions_cannot_hide_original_inner_domains_with_zero_or_zero_power() {
    let variables = [Symbol::new("x")];
    let inner = function("log", Expr::Sym(variables[0].clone()));
    let domain = [interval(q(-1, 1), q(1, 1))];
    for name in ["erf", "erfc"] {
        let function = function(name, inner.clone());
        for source in [
            function.clone(),
            Expr::Mul(vec![Expr::from_i64(0), function.clone()]),
            power(function, 0),
        ] {
            assert!(matches!(
                enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 18),
                Err(RootBoxError::Domain(_))
            ));
            assert!(matches!(
                enclose_objective(&source, &variables, &domain, 18),
                Err(RootBoxError::Domain(_))
            ));
        }
    }
}

fn metered_case(second: bool, meter: &mut impl BudgetMeter) -> Result<(), RootBoxError> {
    let variables = [Symbol::new("x")];
    let source = function("erfc", Expr::Sym(variables[0].clone()));
    let domain = [RealBall::from_i64(4)];
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
        assert!(!self.stopped);
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped);
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped);
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
fn cumulative_limits_and_cancellation_refuse_without_publishing_partial_receipts() {
    const LIMIT: u64 = 1_000_000_000_000;
    const RESERVE: u64 = 37;
    for second in [false, true] {
        let mut measured = Budget::new(BudgetLimits::uniform(LIMIT, RESERVE));
        metered_case(second, &mut measured).unwrap();
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
            metered_case(second, &mut exact).unwrap();
            assert_eq!(exact.remaining(dimension), 0);
            limits.dimensions[dimension.index()] = used - 1;
            let mut short = Budget::new(limits);
            assert!(matches!(
                metered_case(second, &mut short),
                Err(RootBoxError::Meter(_))
            ));
        }
        let mut count = Checkpoints {
            visits: 0,
            stop: None,
            stopped: false,
        };
        metered_case(second, &mut count).unwrap();
        for stop in [1, count.visits / 2, count.visits] {
            let mut meter = Checkpoints {
                visits: 0,
                stop: Some(stop),
                stopped: false,
            };
            assert_eq!(
                metered_case(second, &mut meter),
                Err(RootBoxError::Meter(MeterError::Cancelled))
            );
            assert!(meter.stopped);
            assert_eq!(meter.visits, stop);
        }
    }
}
