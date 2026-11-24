//! End-to-end root-existence and proper-integral checks at continuous,
//! non-Lipschitz endpoints. Every expected value below is exact; no floating
//! solver status, sampled derivative or tolerance estimate authorizes a receipt.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::quadrature::{
    QuadratureOptions, certified_integral, metered_certified_integral,
};
use fsym_calculus::scalar_bracket::{
    ScalarRootBracketOptions, certified_scalar_root_bracket, metered_certified_scalar_root_bracket,
};
use fsym_core::{
    BigRational, Expr, QuadratureError, RealBall, RootBoxError, ScalarRootBracketError, Symbol,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: BigRational) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::Rational(exponent)))
}

fn root_options() -> ScalarRootBracketOptions {
    ScalarRootBracketOptions {
        precision_digits: 16,
        max_bisections: 128,
    }
}

fn integral_options() -> QuadratureOptions {
    QuadratureOptions {
        precision_digits: 16,
        max_intervals: 256,
    }
}

#[test]
fn square_root_boundary_and_interior_roots_have_independently_checked_brackets() {
    let variable = Symbol::new("closed_sqrt_x");
    let root = function("sqrt", Expr::Sym(variable.clone()));
    for (source, lower, upper, expected) in [
        (root.clone(), q(0, 1), q(1, 1), q(0, 1)),
        (root.clone(), q(0, 1), q(0, 1), q(0, 1)),
        (
            Expr::Add(vec![root, Expr::Rational(q(-1, 2))]),
            q(0, 1),
            q(1, 1),
            q(1, 4),
        ),
    ] {
        let tolerance = q(1, 1000);
        let receipt = certified_scalar_root_bracket(
            &source,
            &variable,
            &lower,
            &upper,
            &tolerance,
            root_options(),
        )
        .unwrap();
        assert_eq!(receipt.root_enclosure(), &RealBall::exact(expected));
        assert!(receipt.root_enclosure().width() <= tolerance);
        assert_eq!(receipt.expression(), &source);
        assert_eq!(receipt.variable(), &variable);
    }
}

#[test]
fn inverse_function_endpoint_roots_do_not_require_endpoint_derivatives() {
    let variable = Symbol::new("closed_inverse_x");
    let x = Expr::Sym(variable.clone());
    for (source, lower, upper, expected) in [
        (function("asin", x.clone()), q(-1, 1), q(1, 1), q(0, 1)),
        (function("acos", x.clone()), q(-1, 1), q(1, 1), q(1, 1)),
        (function("acosh", x), q(1, 1), q(2, 1), q(1, 1)),
    ] {
        let receipt = certified_scalar_root_bracket(
            &source,
            &variable,
            &lower,
            &upper,
            &q(1, 100),
            root_options(),
        )
        .unwrap();
        assert_eq!(receipt.root_enclosure(), &RealBall::exact(expected));
        assert_eq!(receipt.expression(), &source);
    }
}

#[test]
fn fractional_power_refinement_contains_the_exact_nondyadic_root() {
    let variable = Symbol::new("fractional_root_x");
    let source = Expr::Add(vec![
        power(Expr::Sym(variable.clone()), q(1, 3)),
        Expr::Rational(q(-3, 10)),
    ]);
    let tolerance = q(1, 1000);
    let receipt = certified_scalar_root_bracket(
        &source,
        &variable,
        &q(0, 1),
        &q(1, 1),
        &tolerance,
        root_options(),
    )
    .unwrap();
    assert!(receipt.root_enclosure().contains(&q(27, 1000)));
    assert!(receipt.root_enclosure().width() <= tolerance);
    assert!(receipt.root_enclosure().width() > q(0, 1));
    assert!(receipt.endpoint_values()[0].is_negative());
    assert!(receipt.endpoint_values()[1].is_positive());
}

#[test]
fn adaptive_endpoint_integrals_contain_independent_exact_antiderivative_values() {
    let variable = Symbol::new("closed_integral_x");
    let x = Expr::Sym(variable.clone());
    for (source, expected) in [
        (function("sqrt", x.clone()), q(2, 3)),
        (power(x.clone(), q(1, 3)), q(3, 4)),
        (power(x.clone(), q(3, 2)), q(2, 5)),
        (function("acos", x), q(1, 1)),
    ] {
        let tolerance = q(1, 64);
        let receipt = certified_integral(
            &source,
            &variable,
            &q(0, 1),
            &q(1, 1),
            &tolerance,
            integral_options(),
        )
        .unwrap();
        assert!(receipt.enclosure().contains(&expected));
        assert!(receipt.enclosure().width() <= tolerance);
        assert_eq!(receipt.integrand(), &source);
        assert_eq!(receipt.variable(), &variable);
        assert_eq!(receipt.witness().partition.first(), Some(&q(0, 1)));
        assert_eq!(receipt.witness().partition.last(), Some(&q(1, 1)));
        assert!(receipt.interval_count() > 1);
        assert!(receipt.interval_count() <= integral_options().max_intervals);
    }
}

#[test]
fn reversed_and_zero_width_integrals_keep_exact_source_endpoint_obligations() {
    let variable = Symbol::new("orientation_endpoint_x");
    let source = function("sqrt", Expr::Sym(variable.clone()));
    let tolerance = q(1, 64);
    let forward = certified_integral(
        &source,
        &variable,
        &q(0, 1),
        &q(1, 1),
        &tolerance,
        integral_options(),
    )
    .unwrap();
    let reverse = certified_integral(
        &source,
        &variable,
        &q(1, 1),
        &q(0, 1),
        &tolerance,
        integral_options(),
    )
    .unwrap();
    assert_eq!(reverse.enclosure(), &forward.enclosure().neg());
    assert_eq!(reverse.witness(), forward.witness());
    assert!(reverse.enclosure().contains(&q(-2, 3)));
    let point = certified_integral(
        &source,
        &variable,
        &q(0, 1),
        &q(0, 1),
        &tolerance,
        integral_options(),
    )
    .unwrap();
    assert_eq!(point.enclosure(), &RealBall::from_i64(0));
    assert_eq!(point.interval_count(), 0);
    let undefined = power(source, q(-1, 1));
    assert!(
        certified_integral(
            &undefined,
            &variable,
            &q(0, 1),
            &q(0, 1),
            &tolerance,
            integral_options(),
        )
        .is_err()
    );
}

#[test]
fn original_poles_and_outside_domain_values_do_not_gain_continuity_receipts() {
    let variable = Symbol::new("original_domain_x");
    let x = Expr::Sym(variable.clone());
    let negative = Expr::Mul(vec![Expr::from_i64(-1), x.clone()]);
    let hidden_pole = Expr::Mul(vec![Expr::from_i64(0), power(x.clone(), q(-1, 1))]);
    let sources = [
        function("sqrt", negative.clone()),
        power(negative, q(1, 3)),
        function("asin", Expr::Add(vec![x.clone(), Expr::from_i64(1)])),
        function("acos", Expr::Add(vec![x.clone(), Expr::from_i64(-2)])),
        function("acosh", x.clone()),
        function("log", x.clone()),
        power(function("sqrt", x.clone()), q(-1, 1)),
        Expr::Add(vec![function("sqrt", x), hidden_pole]),
    ];
    for source in sources {
        assert!(
            certified_scalar_root_bracket(
                &source,
                &variable,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                root_options(),
            )
            .is_err()
        );
        assert!(
            certified_integral(
                &source,
                &variable,
                &q(0, 1),
                &q(1, 1),
                &q(1, 64),
                integral_options(),
            )
            .is_err()
        );
    }
}

#[test]
fn adaptive_partitioning_resolves_closed_domain_range_dependency() {
    let variable = Symbol::new("range_dependency_x");
    let x = Expr::Sym(variable.clone());
    // The first raw interval range is [0,2], although 2*x*(1-x)
    // belongs to [0,1/2] on [0,1]. Both inverse circular functions
    // therefore need subdivision before the original child is admitted.
    let circular_argument = Expr::Mul(vec![
        Expr::from_i64(2),
        x.clone(),
        Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![Expr::from_i64(-1), x.clone()]),
        ]),
    ]);
    // Raw x*x ranges over [-1,1] on [-1,1]; its actual values are
    // nonnegative. A square-power simplification is deliberately absent.
    let square = Expr::Mul(vec![x.clone(), x]);
    let hyperbolic_argument = Expr::Add(vec![Expr::from_i64(1), square.clone()]);
    for (child, lower, upper) in [
        (
            function("asin", circular_argument.clone()),
            q(0, 1),
            q(1, 1),
        ),
        (function("acos", circular_argument), q(0, 1), q(1, 1)),
        (function("acosh", hyperbolic_argument), q(-1, 1), q(1, 1)),
        (function("sqrt", square), q(-1, 1), q(1, 1)),
    ] {
        // The exact value is zero only after every original child's
        // complete-cell domain has been established independently.
        let source = Expr::Mul(vec![Expr::from_i64(0), child]);
        let receipt = certified_integral(
            &source,
            &variable,
            &lower,
            &upper,
            &q(1, 64),
            integral_options(),
        )
        .unwrap();
        assert_eq!(receipt.enclosure(), &RealBall::from_i64(0));
        assert!(receipt.interval_count() > 1);
        assert!(receipt.interval_count() <= integral_options().max_intervals);
        assert_eq!(receipt.integrand(), &source);
    }
}

#[test]
fn endpoint_support_does_not_relax_precision_or_accuracy_limits() {
    let variable = Symbol::new("precision_endpoint_x");
    let source = function("sqrt", Expr::Sym(variable.clone()));
    for precision_digits in [7, 129] {
        assert!(matches!(
            certified_scalar_root_bracket(
                &source,
                &variable,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                ScalarRootBracketOptions {
                    precision_digits,
                    ..root_options()
                },
            ),
            Err(ScalarRootBracketError::Enclosure(
                RootBoxError::InvalidPrecision
            ))
        ));
        assert!(matches!(
            certified_integral(
                &source,
                &variable,
                &q(0, 1),
                &q(1, 1),
                &q(1, 64),
                QuadratureOptions {
                    precision_digits,
                    ..integral_options()
                },
            ),
            Err(QuadratureError::Enclosure(RootBoxError::InvalidPrecision))
        ));
    }
    assert!(matches!(
        certified_integral(
            &source,
            &variable,
            &q(0, 1),
            &q(1, 1),
            &q(1, 64),
            QuadratureOptions {
                max_intervals: 1,
                ..integral_options()
            },
        ),
        Err(QuadratureError::AccuracyNotReached)
    ));
}

#[test]
fn endpoint_generators_and_verifiers_share_each_callers_cumulative_budget() {
    let variable = Symbol::new("budget_endpoint_x");
    let source = function("sqrt", Expr::Sym(variable.clone()));
    let limit = 1024 * 1024 * 1024;
    let mut generator = Budget::new(BudgetLimits::uniform(limit, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(limit, 0));
    metered_certified_scalar_root_bracket(
        &source,
        &variable,
        &q(0, 1),
        &q(1, 1),
        &q(1, 100),
        root_options(),
        &mut generator,
        &mut verifier,
    )
    .unwrap();
    let remaining = [
        generator.remaining(Dimension::ComputeSteps),
        verifier.remaining(Dimension::ComputeSteps),
    ];
    assert!(remaining.iter().all(|amount| *amount < limit));
    metered_certified_integral(
        &source,
        &variable,
        &q(0, 1),
        &q(1, 1),
        &q(1, 64),
        integral_options(),
        &mut generator,
        &mut verifier,
    )
    .unwrap();
    assert!(generator.remaining(Dimension::ComputeSteps) < remaining[0]);
    assert!(verifier.remaining(Dimension::ComputeSteps) < remaining[1]);
    for stop_generator in [true, false] {
        let generation_limit = if stop_generator { 1 } else { limit };
        let verification_limit = if stop_generator { limit } else { 1 };
        let mut generator = Budget::new(BudgetLimits::uniform(generation_limit, 0));
        let mut verifier = Budget::new(BudgetLimits::uniform(verification_limit, 0));
        assert!(matches!(
            metered_certified_scalar_root_bracket(
                &source,
                &variable,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                root_options(),
                &mut generator,
                &mut verifier,
            ),
            Err(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
                MeterError::Budget(_)
            )))
        ));
        let mut generator = Budget::new(BudgetLimits::uniform(generation_limit, 0));
        let mut verifier = Budget::new(BudgetLimits::uniform(verification_limit, 0));
        assert!(matches!(
            metered_certified_integral(
                &source,
                &variable,
                &q(0, 1),
                &q(0, 1),
                &q(1, 64),
                integral_options(),
                &mut generator,
                &mut verifier,
            ),
            Err(QuadratureError::Enclosure(RootBoxError::Meter(
                MeterError::Budget(_)
            )))
        ));
    }
}

#[derive(Default)]
struct Checkpoints {
    calls: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        self.checkpoint()
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.calls += 1;
        if self.cancel_at == Some(self.calls) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_during_either_phase_prevents_an_endpoint_receipt() {
    let variable = Symbol::new("cancel_endpoint_x");
    let source = function("sqrt", Expr::Sym(variable.clone()));
    for integral in [false, true] {
        let run = |generator: &mut Checkpoints, verifier: &mut Checkpoints| {
            if integral {
                metered_certified_integral(
                    &source,
                    &variable,
                    &q(0, 1),
                    &q(0, 1),
                    &q(1, 64),
                    integral_options(),
                    generator,
                    verifier,
                )
                .map(|_| ())
                .map_err(|error| match error {
                    QuadratureError::Enclosure(error) => error,
                    other => panic!("unexpected quadrature failure: {other}"),
                })
            } else {
                metered_certified_scalar_root_bracket(
                    &source,
                    &variable,
                    &q(0, 1),
                    &q(1, 1),
                    &q(1, 100),
                    root_options(),
                    generator,
                    verifier,
                )
                .map(|_| ())
                .map_err(|error| match error {
                    ScalarRootBracketError::Enclosure(error) => error,
                    other => panic!("unexpected bracket failure: {other}"),
                })
            }
        };
        let mut generator = Checkpoints::default();
        let mut verifier = Checkpoints::default();
        run(&mut generator, &mut verifier).unwrap();
        for (cancel_generator, calls) in [(true, generator.calls), (false, verifier.calls)] {
            for cancel_at in [calls / 2, calls] {
                assert!(cancel_at > 0);
                let mut generator = Checkpoints {
                    cancel_at: cancel_generator.then_some(cancel_at),
                    ..Checkpoints::default()
                };
                let mut verifier = Checkpoints {
                    cancel_at: (!cancel_generator).then_some(cancel_at),
                    ..Checkpoints::default()
                };
                assert!(matches!(
                    run(&mut generator, &mut verifier),
                    Err(RootBoxError::Meter(MeterError::Cancelled))
                ));
            }
        }
    }
}
