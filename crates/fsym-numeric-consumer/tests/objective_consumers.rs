//! Independent objective formulas, domain adversaries and publication gates.
//!
//! Floating stationarity is deliberately not a minimum certificate. The tests
//! include stationary maxima/saddles and keep source-domain failures separate
//! from cancellation and exhausted work during unsuccessful line-search trials.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedObjective;
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_numeric_consumer::{
    MinimizeOptions, ObjectiveConsumerError, minimize, minimize_with_meter,
};
use std::error::Error;
use std::sync::Arc;

fn objective(source: &str, variables: &[&str]) -> VerifiedObjective {
    let source = parse(source).unwrap();
    let variables: Vec<_> = variables.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedObjective::try_compile(&source, &variables).unwrap()
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected} by more than {tolerance}"
    );
}

fn meter_cause(error: &ObjectiveConsumerError) -> Option<MeterError> {
    let mut current: Option<&(dyn Error + 'static)> = Some(error);
    while let Some(cause) = current {
        if let Some(meter) = cause.downcast_ref::<MeterError>() {
            return Some(*meter);
        }
        current = cause.source();
    }
    None
}

#[test]
fn coupled_quadratic_publishes_original_value_and_gradient_at_known_minimizer() {
    // The positive-definite Hessian is [[6, 2], [2, 8]]. The additive
    // constant separates objective minimization from solving objective = 0.
    let source = objective("3*(x-2)^2+2*(x-2)*(y+1)+4*(y+1)^2+7", &["x", "y"]);
    let result = minimize(&source, &[-4.0, 6.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 2.0, 1e-10);
    near(result.point[1], -1.0, 1e-10);
    near(result.value, 7.0, 1e-12);
    let a = result.point[0] - 2.0;
    let b = result.point[1] + 1.0;
    near(result.gradient[0], 6.0 * a + 2.0 * b, 1e-12);
    near(result.gradient[1], 2.0 * a + 8.0 * b, 1e-12);
    assert!(result.gradient_norm <= 1e-10);
    assert!(result.iterations > 0);
    assert!(result.evaluations >= result.iterations + 2);
}

#[test]
fn one_three_and_sixteen_variable_objectives_retain_every_coordinate() {
    for n in [1, 3, 16] {
        let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
        let source = variables
            .iter()
            .enumerate()
            .fold(Expr::from_i64(5), |sum, (i, var)| {
                let offset = Expr::Sym(var.clone()) - Expr::from_i64((i + 1) as i64);
                sum + Expr::from_i64((i + 1) as i64)
                    * Expr::Pow(Arc::new(offset), Arc::new(Expr::from_i64(2)))
            });
        let source = VerifiedObjective::try_compile(&source, &variables).unwrap();
        let result = minimize(&source, &vec![0.0; n], &MinimizeOptions::default()).unwrap();
        assert_eq!(result.point.len(), n);
        assert_eq!(result.gradient.len(), n);
        for (i, coordinate) in result.point.iter().enumerate() {
            near(*coordinate, (i + 1) as f64, 1e-10);
            near(result.gradient[i], 0.0, 1e-10);
        }
        near(result.value, 5.0, 1e-12);
    }
}

#[test]
fn rosenbrock_descent_reaches_the_known_nonlinear_minimum() {
    let source = objective("(1-x)^2+100*(y-x^2)^2", &["x", "y"]);
    let result = minimize(&source, &[-1.2, 1.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-8);
    near(result.point[1], 1.0, 1e-8);
    assert!((0.0..1e-16).contains(&result.value));
    assert!(result.gradient_norm <= 1e-10);
    let x = result.point[0];
    let y = result.point[1];
    near(
        result.gradient[0],
        2.0 * (x - 1.0) - 400.0 * x * (y - x * x),
        1e-11,
    );
    near(result.gradient[1], 200.0 * (y - x * x), 1e-11);
}

#[test]
fn negative_curvature_direction_is_regularized_before_armijo_descent() {
    // At 0.1, f'' < 0 and unmodified Newton heads toward the maximum at
    // zero. A descending direction instead reaches the positive well.
    let source = objective("(x^2-1)^2", &["x"]);
    let result = minimize(&source, &[0.1], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-8);
    assert!((0.0..1e-16).contains(&result.value));
    assert!(result.gradient_norm <= 1e-10);
    assert!(result.regularizations > 0);
}

#[test]
fn zero_hessian_with_nonzero_gradient_uses_a_bounded_regularized_step() {
    // f'(x) = 4*x^3 - 1, hence the minimizer is 4^(-1/3), while
    // f''(0) = 0. The independently derived value there is -3*x/4.
    let source = objective("x^4-x", &["x"]);
    let result = minimize(&source, &[0.0], &MinimizeOptions::default()).unwrap();
    let expected = 4.0_f64.cbrt().recip();
    near(result.point[0], expected, 1e-8);
    near(result.value, -0.75 * expected, 1e-12);
    assert!(result.gradient_norm <= 1e-10);
    assert!(result.regularizations > 0);
}

#[test]
fn stationary_maximum_and_saddle_are_only_numerical_candidates() {
    for (source, initial, expected_value) in [
        (objective("3-x^2", &["x"]), vec![0.0], 3.0),
        (objective("x^2-y^2", &["x", "y"]), vec![0.0, 0.0], 0.0),
    ] {
        let result = minimize(&source, &initial, &MinimizeOptions::default()).unwrap();
        assert_eq!(result.point, initial);
        assert_eq!(result.value, expected_value);
        assert_eq!(result.gradient_norm, 0.0);
        assert!(result.gradient.iter().all(|value| *value == 0.0));
        assert_eq!(result.iterations, 0);
        assert!(result.evaluations >= 2);
    }
}

#[test]
fn flat_objective_and_unused_coordinate_do_not_forge_uniqueness() {
    let flat = objective("17", &["x", "y"]);
    let initial = [2.0, -3.0];
    let result = minimize(&flat, &initial, &MinimizeOptions::default()).unwrap();
    assert_eq!(result.point, initial);
    assert_eq!(result.value, 17.0);
    assert_eq!(result.gradient, [0.0, 0.0]);
    assert_eq!(result.iterations, 0);

    let unused = objective("(x-2)^2", &["x", "y"]);
    let result = minimize(&unused, &[0.0, 7.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 2.0, 1e-9);
    assert_eq!(result.point[1], 7.0);
    assert_eq!(result.gradient[1], 0.0);
    assert!(result.gradient_norm <= 1e-10);
}

#[test]
fn ill_scaled_coupled_hessian_preserves_original_coordinate_units() {
    // Both squares vanish at (10^6, 2). Hessian entries span twelve
    // orders of magnitude, with a nonzero coupling term.
    let source = objective("(x/1000000+y-3)^2+(2*x/1000000-y)^2", &["x", "y"]);
    let result = minimize(&source, &[0.0, 0.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 1_000_000.0, 1e-4);
    near(result.point[1], 2.0, 1e-10);
    assert!((0.0..1e-20).contains(&result.value));
    assert!(result.gradient_norm <= 1e-10);
}

#[test]
fn logarithmic_trial_holes_backtrack_without_becoming_success() {
    // At x=10 the full Newton step is -90, outside log's domain.
    // The unique stationary point on x>0 is x=1, with value 1.
    let source = objective("x-log(x)", &["x"]);
    let result = minimize(&source, &[10.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-9);
    near(result.value, 1.0, 1e-12);
    near(result.gradient[0], 1.0 - result.point[0].recip(), 1e-12);
    assert!(result.gradient_norm <= 1e-10);
    assert!(result.evaluations > result.iterations + 2);
}

#[test]
fn overflowing_trial_values_backtrack_to_a_finite_original_objective() {
    // At -8 the full Newton step takes exp far beyond binary64's range.
    let source = objective("exp(x)-x", &["x"]);
    let result = minimize(&source, &[-8.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 0.0, 1e-9);
    near(result.value, 1.0, 1e-12);
    near(result.gradient[0], result.point[0].exp() - 1.0, 1e-12);
    assert!(result.gradient_norm <= 1e-10);
    assert!(result.evaluations > result.iterations + 2);
}

#[test]
fn original_holes_hidden_by_zero_products_and_powers_fail_before_stationarity() {
    let x = Symbol::new("x");
    let logarithm = Expr::Function("log".into(), vec![Expr::Sym(x.clone())]);
    let held_product = Expr::Mul(vec![Expr::from_i64(0), logarithm.clone()]);
    let held_power = Expr::Pow(Arc::new(logarithm), Arc::new(Expr::from_i64(0)));
    for hidden in [held_product, held_power] {
        let source = Expr::Add(vec![
            Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
            hidden,
        ]);
        let source = VerifiedObjective::try_compile(&source, std::slice::from_ref(&x)).unwrap();
        assert!(minimize(&source, &[0.0], &MinimizeOptions::default()).is_err());
    }
    let constant_hole = objective("log(-1)", &["x"]);
    assert!(minimize(&constant_hole, &[0.0], &MinimizeOptions::default()).is_err());
}

#[test]
fn finite_values_cannot_hide_undefined_first_or_second_derivatives() {
    for formula in ["x^(1/2)", "x^(3/2)"] {
        let source = objective(formula, &["x"]);
        assert!(
            minimize(&source, &[0.0], &MinimizeOptions::default()).is_err(),
            "published at the derivative-domain boundary of {formula}"
        );
    }
    // The source coefficient rounds to finite binary64, but its second
    // derivative contains 2*10^308. Admission must reject that literal before
    // the numerical consumer is reachable; it is not a runtime-overflow case.
    let overflowing_derivative = parse("10^308*x^2").unwrap();
    assert!(matches!(
        VerifiedObjective::try_compile(&overflowing_derivative, &[Symbol::new("x")]),
        Err(
            fsym_calculus::sparse_hessian::ObjectiveCompilationError::Compilation(
                fsym_calculus::compile::CompileError::NumericConversion(_)
            )
        )
    ));

    // Here the source and gradient stay finite while the Hessian's x^4
    // intermediate overflows. All compiled literals are small and admissible.
    let point = 1e100_f64;
    assert!(point.powi(3).sin().is_finite());
    assert!((3.0 * point.powi(2) * point.powi(3).cos()).is_finite());
    let oscillatory = objective("sin(x^3)", &["x"]);
    assert!(minimize(&oscillatory, &[point], &MinimizeOptions::default()).is_err());

    for (formula, point) in [("exp(x)", 800.0), ("x^4", 1e100)] {
        let source = objective(formula, &["x"]);
        assert!(minimize(&source, &[point], &MinimizeOptions::default()).is_err());
    }
}

#[test]
fn same_printed_names_keep_distinct_typed_bindings_and_argument_order() {
    let first = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let a = Expr::Sym(first.clone()) - Expr::from_i64(2);
    let b = Expr::Sym(second.clone()) + Expr::from_i64(3);
    let source = Expr::Pow(Arc::new(a), Arc::new(Expr::from_i64(2)))
        + Expr::from_i64(3) * Expr::Pow(Arc::new(b), Arc::new(Expr::from_i64(2)));
    let variables = [second, first];
    let source = VerifiedObjective::try_compile(&source, &variables).unwrap();
    let result = minimize(&source, &[0.0, 0.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], -3.0, 1e-10);
    near(result.point[1], 2.0, 1e-10);
    near(result.value, 0.0, 1e-20);
    assert_eq!(source.variables(), &variables);
}

#[test]
fn clipped_steps_remain_bounded_and_exhausted_iterations_do_not_publish() {
    let source = objective("(x-3)^2", &["x"]);
    let options = MinimizeOptions {
        max_step_norm: 0.25,
        ..MinimizeOptions::default()
    };
    let result = minimize(&source, &[0.0], &options).unwrap();
    near(result.point[0], 3.0, 1e-9);
    assert!(result.iterations >= 12);
    let short = MinimizeOptions {
        max_iterations: 2,
        ..options
    };
    assert!(minimize(&source, &[0.0], &short).is_err());

    let unrepresentable = MinimizeOptions {
        max_step_norm: f64::from_bits(1),
        gradient_tolerance: 0.0,
        ..MinimizeOptions::default()
    };
    assert!(minimize(&source, &[2.0], &unrepresentable).is_err());
}

#[test]
fn zero_iterations_checks_the_actual_initial_value_and_derivatives() {
    let source = objective("(x-1)^2+2", &["x"]);
    let options = MinimizeOptions {
        max_iterations: 0,
        ..MinimizeOptions::default()
    };
    let result = minimize(&source, &[1.0], &options).unwrap();
    assert_eq!(result.point, [1.0]);
    assert_eq!(result.value, 2.0);
    assert_eq!(result.gradient, [0.0]);
    assert_eq!(result.iterations, 0);
    assert!(result.evaluations >= 2);
    assert!(minimize(&source, &[0.0], &options).is_err());
}

#[test]
fn shape_nonfinite_inputs_and_invalid_controls_refuse_before_dense_work() {
    let defaults = MinimizeOptions::default();
    for n in [0, 17] {
        let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
        let source = VerifiedObjective::try_compile(&Expr::from_i64(1), &variables).unwrap();
        let mut meter = RecordingMeter::default();
        assert!(minimize_with_meter(&source, &vec![0.0; n], &defaults, &mut meter).is_err());
        assert_eq!(
            meter.bytes, 0,
            "dense buffers allocated before shape admission"
        );
        assert_eq!(meter.allocations, 0);
    }
    let source = objective("(x-1)^2", &["x"]);
    for options in [
        MinimizeOptions {
            max_iterations: 1025,
            ..defaults.clone()
        },
        MinimizeOptions {
            max_iterations: usize::MAX,
            ..defaults.clone()
        },
        MinimizeOptions {
            gradient_tolerance: -1.0,
            ..defaults.clone()
        },
        MinimizeOptions {
            gradient_tolerance: f64::NAN,
            ..defaults.clone()
        },
        MinimizeOptions {
            gradient_tolerance: f64::INFINITY,
            ..defaults.clone()
        },
        MinimizeOptions {
            max_step_norm: 0.0,
            ..defaults.clone()
        },
        MinimizeOptions {
            max_step_norm: -1.0,
            ..defaults.clone()
        },
        MinimizeOptions {
            max_step_norm: f64::NAN,
            ..defaults.clone()
        },
        MinimizeOptions {
            max_step_norm: f64::INFINITY,
            ..defaults.clone()
        },
    ] {
        let mut meter = RecordingMeter::default();
        assert!(minimize_with_meter(&source, &[0.0], &options, &mut meter).is_err());
        assert_eq!(meter.bytes, 0, "buffers allocated before guard admission");
        assert_eq!(meter.allocations, 0);
    }
    for point in [
        vec![],
        vec![0.0, 0.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![f64::NEG_INFINITY],
    ] {
        let mut meter = RecordingMeter::default();
        assert!(minimize_with_meter(&source, &point, &defaults, &mut meter).is_err());
        assert_eq!(meter.bytes, 0, "buffers allocated before input admission");
        assert_eq!(meter.allocations, 0);
    }
}

#[derive(Default)]
struct RecordingMeter {
    checkpoints: usize,
    charges: usize,
    cancel_at: Option<usize>,
    exhaust_at: Option<usize>,
    stopped: bool,
    compute: u64,
    bytes: u64,
    allocations: u64,
}

impl RecordingMeter {
    fn before_charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(
            !self.stopped,
            "work continued after a terminal meter refusal"
        );
        self.charges += 1;
        if self.exhaust_at == Some(self.charges) {
            self.stopped = true;
            return Err(BudgetError::Exhausted {
                dimension,
                requested: amount,
                remaining: 0,
            }
            .into());
        }
        Ok(())
    }

    fn accumulate(&mut self, dimension: Dimension, amount: u64) {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.bytes += amount,
            Dimension::AllocationCount => self.allocations += amount,
            _ => {}
        }
    }
}

impl BudgetMeter for RecordingMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.before_charge(dimension, amount)?;
        self.accumulate(dimension, amount);
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if let Some(&(dimension, amount)) = charges.first() {
            // A batch is all-or-nothing, including at an injected refusal.
            self.before_charge(dimension, amount)?;
        }
        for &(dimension, amount) in charges {
            self.accumulate(dimension, amount);
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(
            !self.stopped,
            "work continued after a terminal meter refusal"
        );
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.stopped = true;
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

#[test]
fn cancellation_at_every_safe_point_including_rejected_trials_and_publication_is_terminal() {
    let source = objective("x-log(x)", &["x"]);
    let options = MinimizeOptions::default();
    let mut complete = RecordingMeter::default();
    let proposal = minimize_with_meter(&source, &[10.0], &options, &mut complete).unwrap();
    assert!(proposal.evaluations > proposal.iterations + 2);
    assert!(complete.compute > 0 && complete.bytes > 0 && complete.allocations > 0);
    for checkpoint in 1..=complete.checkpoints {
        let mut meter = RecordingMeter {
            cancel_at: Some(checkpoint),
            ..RecordingMeter::default()
        };
        let error = minimize_with_meter(&source, &[10.0], &options, &mut meter)
            .expect_err("published after an injected cancellation");
        assert_eq!(
            meter_cause(&error),
            Some(MeterError::Cancelled),
            "checkpoint {checkpoint}: {error}"
        );
        assert!(meter.stopped);
        assert_eq!(meter.checkpoints, checkpoint);
    }
}

#[test]
fn every_exhausted_charge_remains_terminal_through_domain_backtracking() {
    let source = objective("x-log(x)", &["x"]);
    let options = MinimizeOptions::default();
    let mut complete = RecordingMeter::default();
    minimize_with_meter(&source, &[10.0], &options, &mut complete).unwrap();
    for charge in 1..=complete.charges {
        let mut meter = RecordingMeter {
            exhaust_at: Some(charge),
            ..RecordingMeter::default()
        };
        let error = minimize_with_meter(&source, &[10.0], &options, &mut meter)
            .expect_err("published after an exhausted charge");
        assert!(
            matches!(
                meter_cause(&error),
                Some(MeterError::Budget(BudgetError::Exhausted { .. }))
            ),
            "charge {charge}: {error}"
        );
        assert!(meter.stopped);
        assert_eq!(meter.charges, charge);
    }
}

#[test]
fn rejected_trials_and_repeated_calls_retain_cumulative_budget_and_verifier_reserve() {
    let source = objective("x-log(x)", &["x"]);
    let options = MinimizeOptions::default();
    let mut measured = RecordingMeter::default();
    minimize_with_meter(&source, &[10.0], &options, &mut measured).unwrap();
    let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
    limits.dimensions[Dimension::ComputeSteps.index()] = measured.compute;
    let mut budget = Budget::new(limits);
    minimize_with_meter(&source, &[10.0], &options, &mut budget).unwrap();
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(budget.verifier_remaining(), 37);
    let error = minimize_with_meter(&source, &[10.0], &options, &mut budget).unwrap_err();
    assert!(matches!(
        meter_cause(&error),
        Some(MeterError::Budget(BudgetError::Exhausted {
            dimension: Dimension::ComputeSteps,
            ..
        }))
    ));
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(budget.verifier_remaining(), 37);

    for (dimension, total) in [
        (Dimension::ComputeSteps, measured.compute),
        (Dimension::MemoryBytes, measured.bytes),
        (Dimension::AllocationCount, measured.allocations),
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
        limits.dimensions[dimension.index()] = total - 1;
        let mut budget = Budget::new(limits);
        let error = minimize_with_meter(&source, &[10.0], &options, &mut budget)
            .expect_err("published without the final required resource unit");
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted {
                dimension: exhausted,
                ..
            })) if exhausted == dimension
        ));
        assert_eq!(budget.verifier_remaining(), 37);
    }
}
