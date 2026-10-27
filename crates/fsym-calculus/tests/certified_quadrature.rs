use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::quadrature::{
    QuadratureOptions, certified_integral, metered_certified_integral,
};
use fsym_core::{BigInt, BigRational, Expr, QuadratureError, RootBoxError, Symbol};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(value: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(value), Arc::new(Expr::from_i64(exponent)))
}
fn options() -> QuadratureOptions {
    QuadratureOptions {
        precision_digits: 16,
        max_intervals: 512,
    }
}

#[test]
fn adaptive_quadratic_and_shifted_cubic_integrals_contain_independent_exact_values() {
    let x = Symbol::new("x");
    for (source, lower, upper, expected) in [
        (power(Expr::Sym(x.clone()), 2), q(0, 1), q(1, 1), q(1, 3)),
        (power(Expr::Sym(x.clone()), 3), q(1, 2), q(3, 2), q(5, 4)),
    ] {
        let result =
            certified_integral(&source, &x, &lower, &upper, &q(1, 1000), options()).unwrap();
        assert!(result.enclosure().contains(&expected));
        assert!(result.enclosure().width() <= q(1, 1000));
        assert!(result.interval_count() > 1);
        assert_eq!(result.witness().partition.first(), Some(&lower));
        assert_eq!(result.witness().partition.last(), Some(&upper));
    }
}

fn gaussian_partial_sum(last: i64) -> BigRational {
    let mut factorial = BigInt::from(1);
    let mut sum = q(0, 1);
    for n in 0..=last {
        if n != 0 {
            factorial *= BigInt::from(n);
        }
        sum += BigRational::new(
            BigInt::from(if n % 2 == 0 { 1 } else { -1 }),
            &factorial * BigInt::from(2 * n + 1),
        );
    }
    sum
}

#[test]
fn non_elementary_gaussian_integral_agrees_with_independent_alternating_series_bounds() {
    let x = Symbol::new("x");
    let source = Expr::Function("exp".into(), vec![-power(Expr::Sym(x.clone()), 2)]);
    let result =
        certified_integral(&source, &x, &q(0, 1), &q(1, 1), &q(1, 1000), options()).unwrap();
    // Integrated alternating Taylor terms give rigorous external-to-the-
    // evaluator bounds: odd truncations below, even truncations above.
    assert!(result.enclosure().contains(&gaussian_partial_sum(11)));
    assert!(result.enclosure().contains(&gaussian_partial_sum(12)));
    assert!(result.enclosure().width() <= q(1, 1000));
    assert!(result.interval_count() > 1);
}

#[test]
fn reversed_intervals_negate_the_identical_checked_enclosure() {
    let x = Symbol::new("x");
    let source = Expr::Function("sin".into(), vec![power(Expr::Sym(x.clone()), 2)]);
    let forward =
        certified_integral(&source, &x, &q(0, 1), &q(1, 1), &q(1, 1000), options()).unwrap();
    let backward =
        certified_integral(&source, &x, &q(1, 1), &q(0, 1), &q(1, 1000), options()).unwrap();
    assert_eq!(forward.enclosure().neg(), *backward.enclosure());
    assert_eq!(forward.witness(), backward.witness());
    // Independent alternating Fresnel-series bound, through its fourth term.
    let lower = q(1, 3) - q(1, 42) + q(1, 1320) - q(1, 75600);
    let upper = lower.clone() + q(1, 6_894_720);
    assert!(forward.enclosure().contains(&lower));
    assert!(forward.enclosure().contains(&upper));
}

#[test]
fn proper_domain_failures_and_unsupported_functions_are_never_returned_as_results() {
    let x = Symbol::new("x");
    let sx = Expr::Sym(x.clone());
    for source in [
        power(sx.clone(), -1),
        Expr::Mul(vec![Expr::from_i64(0), power(sx.clone(), -1)]),
        Expr::Function("sqrt".into(), vec![sx.clone()]),
        Expr::Function("gamma".into(), vec![sx.clone()]),
        Expr::Sym(Symbol::new("unbound")),
    ] {
        assert!(
            certified_integral(&source, &x, &q(0, 1), &q(1, 1), &q(1, 1000), options()).is_err()
        );
    }
    let interior_pole = power(sx - Expr::Rational(q(1, 3)), -1);
    assert!(
        certified_integral(
            &interior_pole,
            &x,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            QuadratureOptions {
                max_intervals: 32,
                ..options()
            },
        )
        .is_err()
    );
}

#[test]
fn subdivision_resolves_range_dependency_without_cancelling_original_denominators() {
    let x = Symbol::new("x");
    let sx = Expr::Sym(x.clone());
    // On [-1,1], the raw interval product x*x encloses [-1,1], so the
    // first reciprocal enclosure sees zero in 1+x*x. The actual source
    // is defined everywhere. Smaller closed cells must discharge that
    // obligation; simplifying this held ratio to one is not the proof.
    let denominator = Expr::Add(vec![Expr::from_i64(1), Expr::Mul(vec![sx.clone(), sx])]);
    let source = Expr::Mul(vec![denominator.clone(), power(denominator, -1)]);
    let result =
        certified_integral(&source, &x, &q(-1, 1), &q(1, 1), &q(1, 100), options()).unwrap();
    assert!(result.enclosure().contains(&q(2, 1)));
    assert!(result.enclosure().width() <= q(1, 100));
    assert!(result.interval_count() > 1);
    assert_eq!(result.integrand(), &source);
}

#[test]
fn subdivision_accuracy_and_input_limits_refuse_without_partial_receipts() {
    let x = Symbol::new("x");
    let source = power(Expr::Sym(x.clone()), 2);
    assert!(matches!(
        certified_integral(
            &source,
            &x,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            QuadratureOptions {
                max_intervals: 1,
                ..options()
            },
        ),
        Err(QuadratureError::AccuracyNotReached)
    ));
    for max_intervals in [0, 1025, usize::MAX] {
        assert!(matches!(
            certified_integral(
                &source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 1000),
                QuadratureOptions {
                    max_intervals,
                    ..options()
                },
            ),
            Err(QuadratureError::ResourceLimit(_))
        ));
    }
    for precision_digits in [7, 129, u32::MAX] {
        assert!(matches!(
            certified_integral(
                &source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 1000),
                QuadratureOptions {
                    precision_digits,
                    ..options()
                },
            ),
            Err(QuadratureError::Enclosure(RootBoxError::InvalidPrecision))
        ));
    }
    let point =
        certified_integral(&source, &x, &q(2, 1), &q(2, 1), &q(1, 1000), options()).unwrap();
    assert_eq!(point.enclosure().lower(), q(0, 1));
    assert_eq!(point.interval_count(), 0);
}

struct Cancelled;
impl BudgetMeter for Cancelled {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn generator_and_verifier_have_separate_metered_success_and_cancellation_paths() {
    let x = Symbol::new("x");
    let source = power(Expr::Sym(x.clone()), 2);
    let mut generator = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let result = metered_certified_integral(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 100),
        options(),
        &mut generator,
        &mut verifier,
    )
    .unwrap();
    assert!(result.enclosure().contains(&q(1, 3)));
    assert!(generator.remaining(Dimension::ComputeSteps) < 512 * 1024 * 1024);
    assert!(verifier.remaining(Dimension::ComputeSteps) < 512 * 1024 * 1024);
    for cancel_generator in [true, false] {
        let result = if cancel_generator {
            metered_certified_integral(
                &source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                options(),
                &mut Cancelled,
                &mut Unbounded,
            )
        } else {
            metered_certified_integral(
                &source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                options(),
                &mut Unbounded,
                &mut Cancelled,
            )
        };
        assert!(matches!(
            result,
            Err(QuadratureError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        ));
    }
}
