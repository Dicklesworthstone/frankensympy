//! Rectangular numerical fitting must retain every original residual row.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_core::{Expr, Symbol, parse};
use fsym_numeric_consumer::{
    LeastSquaresOptions, LeastSquaresTermination, SystemConsumerError, least_squares,
    least_squares_with_meter,
};
use std::sync::Arc;

fn system(sources: &[&str], variables: &[&str]) -> VerifiedResidualSystem {
    let sources: Vec<_> = sources.iter().map(|value| parse(value).unwrap()).collect();
    let variables: Vec<_> = variables.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedResidualSystem::try_compile(&sources, &variables).unwrap()
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected}"
    );
}

#[test]
fn all_three_residuals_determine_the_inconsistent_hero_fit() {
    let source = system(&["x-1", "y-1", "x+y-4"], &["x", "y"]);
    let result = least_squares(&source, &[0.0, 0.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 5.0 / 3.0, 1e-12);
    near(result.point[1], 5.0 / 3.0, 1e-12);
    assert_eq!(result.residuals.len(), 3);
    for (actual, expected) in result
        .residuals
        .iter()
        .zip([2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0])
    {
        near(*actual, expected, 1e-12);
    }
    near(result.cost, 2.0 / 3.0, 1e-12);
    assert!(result.gradient_norm <= 1e-10);
    assert_eq!(
        result.termination,
        LeastSquaresTermination::GradientTolerance
    );
    assert!(result.iterations > 0);
    assert!(result.evaluations >= result.iterations + 2);
}

#[test]
fn consistent_tall_systems_support_one_three_and_sixteen_coordinates() {
    for n in [1, 3, 16] {
        let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
        let mut sources: Vec<_> = variables
            .iter()
            .enumerate()
            .map(|(index, variable)| {
                Expr::Sym(variable.clone()) - Expr::from_i64((index + 1) as i64)
            })
            .collect();
        sources.push(
            sources
                .iter()
                .cloned()
                .fold(Expr::from_i64(0), |a, b| a + b),
        );
        let source = VerifiedResidualSystem::try_compile(&sources, &variables).unwrap();
        let result =
            least_squares(&source, &vec![0.0; n], &LeastSquaresOptions::default()).unwrap();
        assert_eq!(result.residuals.len(), n + 1);
        for (index, coordinate) in result.point.iter().enumerate() {
            near(*coordinate, (index + 1) as f64, 1e-10);
        }
        assert!(result.cost < 1e-18);
    }
}

#[test]
fn nonlinear_inconsistent_fit_reaches_the_known_stationary_point() {
    // The gradient is (x+1)(2*x*x-2*x-1), so the positive minimizer is
    // (1+sqrt(3))/2, with a nonzero original residual vector.
    let source = system(&["x^2-2", "x-1"], &["x"]);
    let result = least_squares(&source, &[1.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], (1.0 + 3.0_f64.sqrt()) / 2.0, 1e-8);
    assert!(result.cost > 0.0);
    assert!(result.gradient_norm <= 1e-10);
    near(
        result.residuals[0],
        result.point[0] * result.point[0] - 2.0,
        1e-14,
    );
    near(result.residuals[1], result.point[0] - 1.0, 1e-14);
}

#[test]
fn zero_gradient_is_only_stationarity_even_at_a_nonminimum() {
    // x=0 is a local maximum of (x*x-1)^2/2. The API promises numerical
    // stationarity, never a root or minimum certificate.
    let source = system(&["x^2-1"], &["x"]);
    let result = least_squares(&source, &[0.0], &LeastSquaresOptions::default()).unwrap();
    assert_eq!(result.point, [0.0]);
    assert_eq!(result.residuals, [-1.0]);
    assert_eq!(result.cost, 0.5);
    assert_eq!(result.gradient_norm, 0.0);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.evaluations, 2);
}

#[test]
fn rank_deficiency_is_explicit_and_does_not_publish_a_failed_direction() {
    let source = system(&["x+y-1", "2*x+2*y-2", "3*x+3*y-3"], &["x", "y"]);
    assert!(matches!(
        least_squares(&source, &[0.0, 0.0], &LeastSquaresOptions::default()),
        Err(SystemConsumerError::LinearSolve(_))
    ));
    let at_root = least_squares(&source, &[1.0, 0.0], &LeastSquaresOptions::default()).unwrap();
    assert_eq!(at_root.cost, 0.0);
    let unused = system(&["x-1", "2*x-2"], &["x", "y"]);
    assert!(matches!(
        least_squares(&unused, &[0.0, 0.0], &LeastSquaresOptions::default()),
        Err(SystemConsumerError::LinearSolve(_))
    ));
}

#[test]
fn column_scaling_preserves_coordinates_without_forming_normal_equations() {
    let source = system(
        &["x/1000000+y-3", "2*x/1000000-y", "3*x/1000000+y-5"],
        &["x", "y"],
    );
    let result = least_squares(&source, &[0.0, 0.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 1_000_000.0, 1e-5);
    near(result.point[1], 2.0, 1e-10);
    assert!(result.cost < 1e-18);
}

#[test]
fn armijo_backtracking_and_clipped_steps_keep_one_actual_objective() {
    let source = system(&["x^2-2", "2*x^2-4"], &["x"]);
    let result = least_squares(&source, &[0.01], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 2.0_f64.sqrt(), 1e-10);
    assert!(result.evaluations > result.iterations + 2);

    let linear = system(&["x-3", "2*x-6"], &["x"]);
    let options = LeastSquaresOptions {
        max_step_norm: 0.25,
        ..LeastSquaresOptions::default()
    };
    let clipped = least_squares(&linear, &[0.0], &options).unwrap();
    near(clipped.point[0], 3.0, 1e-10);
    assert!(clipped.iterations >= 12);
}

#[test]
fn logarithmic_trials_backtrack_without_losing_any_residual_rows() {
    let source = system(&["log(x)", "2*log(x)"], &["x"]);
    // At x=10 the full Gauss-Newton trial is 10*(1-log(10)) < 0.
    // Smaller trials are valid and must still use both original rows.
    let result = least_squares(&source, &[10.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-10);
    assert!(result.evaluations > result.iterations + 2);
    assert!(result.gradient_norm <= 1e-10);
    near(result.residuals[0], result.point[0].ln(), 1e-14);
    near(result.residuals[1], 2.0 * result.point[0].ln(), 1e-14);
    near(
        result.cost,
        0.5 * result
            .residuals
            .iter()
            .map(|value| value * value)
            .sum::<f64>(),
        1e-24,
    );
    for invalid in [0.0, -1.0] {
        assert!(matches!(
            least_squares(&source, &[invalid], &LeastSquaresOptions::default()),
            Err(SystemConsumerError::SparseEvaluation(
                ResidualEvaluationError::Evaluation(EvalError::NonFiniteValue)
            ))
        ));
    }
}

#[test]
fn derivative_only_trial_holes_backtrack_but_initial_holes_still_refuse() {
    let source = system(&["x^(1/2)-1"], &["x"]);
    // The initial Newton trial is exactly zero: its residual is finite, but
    // its derivative is not. A half step keeps the full source C1 domain.
    let result = least_squares(&source, &[4.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-9);
    assert!(result.evaluations > result.iterations + 2);
    assert!(result.gradient_norm <= 1e-10);
    assert!(least_squares(&source, &[0.0], &LeastSquaresOptions::default()).is_err());
}

#[test]
fn overflowing_source_and_squared_cost_trials_can_recover() {
    let source = system(&["exp(x)-1", "2*exp(x)-2"], &["x"]);
    // The first step from -7 exceeds 1000. Repeated halvings encounter both
    // exp overflow and finite residuals whose squared cost overflows.
    let result = least_squares(&source, &[-7.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 0.0, 1e-10);
    assert!(result.evaluations > result.iterations + 2);
    assert!(result.cost < 1e-20);
    assert!(result.gradient_norm <= 1e-10);
    assert!(matches!(
        least_squares(&source, &[550.0], &LeastSquaresOptions::default()),
        Err(SystemConsumerError::NonFinite(
            "least-squares squared residual cost"
        ))
    ));
}

#[test]
fn overflowing_coordinate_sum_backtracks_before_source_evaluation() {
    let source = system(&["exp(x/(10^308))-5"], &["x"]);
    let options = LeastSquaresOptions {
        gradient_tolerance: 1e-320,
        max_step_norm: f64::MAX,
        ..LeastSquaresOptions::default()
    };
    // The unscaled direction and initial point are separately finite, but
    // 1e308 + (5/e - 1)*1e308 overflows. The actual root fits in binary64.
    let result = least_squares(&source, &[1e308], &options).unwrap();
    assert!(result.point[0].is_finite());
    near(result.point[0] / 1e308, 5.0_f64.ln(), 1e-11);
    assert!(result.gradient_norm <= options.gradient_tolerance);
    assert!(result.residuals[0].abs() < 1e-11);
}

#[test]
fn rejected_hidden_source_holes_cannot_turn_into_zero_gradient_success() {
    let x = Symbol::new("x");
    let hidden = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
    ]);
    let source =
        VerifiedResidualSystem::try_compile(&[Expr::Sym(x.clone()), hidden], &[x]).unwrap();
    let options = LeastSquaresOptions {
        max_iterations: 4,
        gradient_tolerance: 0.0,
        ..LeastSquaresOptions::default()
    };
    // Each full trial is the algebraic zero, which is outside the retained
    // logarithm's domain. Valid half steps must not publish that invalid zero.
    assert!(matches!(
        least_squares(&source, &[1.0], &options),
        Err(SystemConsumerError::NonlinearSolve(_))
    ));
}

#[test]
fn trailing_original_holes_and_derivative_domains_are_never_discarded() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let logarithm = Expr::Function("log".into(), vec![Expr::Sym(x.clone())]);
    let held = Expr::Mul(vec![Expr::from_i64(0), logarithm.clone()]);
    let power = Expr::Add(vec![
        Expr::Pow(Arc::new(logarithm), Arc::new(Expr::from_i64(0))),
        Expr::from_i64(-1),
    ]);
    for hidden in [held, power] {
        let source = VerifiedResidualSystem::try_compile(
            &[Expr::Sym(x.clone()), Expr::Sym(y.clone()), hidden],
            &[x.clone(), y.clone()],
        )
        .unwrap();
        assert!(least_squares(&source, &[0.0, 0.0], &LeastSquaresOptions::default()).is_err());
    }
    let derivative = system(&["x^(1/2)", "x"], &["x"]);
    assert!(least_squares(&derivative, &[0.0], &LeastSquaresOptions::default()).is_err());
    let empty_pattern = system(&["log(-1)", "1"], &["x"]);
    assert!(least_squares(&empty_pattern, &[0.0], &LeastSquaresOptions::default()).is_err());
}

#[test]
fn finite_components_cannot_hide_nonfinite_cost_or_gradient_arithmetic() {
    for source in [
        system(&["10^308"], &["x"]),
        system(&["10^154", "10^154"], &["x"]),
        system(&["10^308*x+10"], &["x"]),
    ] {
        assert!(matches!(
            least_squares(&source, &[0.0], &LeastSquaresOptions::default()),
            Err(SystemConsumerError::NonFinite(_))
        ));
    }
}

#[test]
fn transport_preserves_original_typed_coordinate_order_and_all_rows() {
    let source = system(&["x-1", "y-1", "x+y-4"], &["y", "x"]);
    let restored =
        VerifiedResidualSystem::from_json_bytes(&source.to_json_bytes().unwrap()).unwrap();
    let original = least_squares(&source, &[1.0, 2.0], &LeastSquaresOptions::default()).unwrap();
    let replayed = least_squares(&restored, &[1.0, 2.0], &LeastSquaresOptions::default()).unwrap();
    assert_eq!(original, replayed);
    assert_eq!(restored.vars(), source.vars());
    assert_eq!(restored.residuals(), source.residuals());
}

#[test]
fn shape_options_and_nonstationary_zero_iteration_requests_refuse() {
    let defaults = LeastSquaresOptions::default();
    for source in [
        system(&[], &[]),
        system(&["x"], &["x", "y"]),
        system(&["x"; 65], &["x"]),
    ] {
        assert!(matches!(
            least_squares(&source, &vec![0.0; source.vars().len()], &defaults),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
    let source = system(&["x-1", "2*x-2"], &["x"]);
    for options in [
        LeastSquaresOptions {
            max_iterations: 1025,
            ..defaults.clone()
        },
        LeastSquaresOptions {
            gradient_tolerance: -1.0,
            ..defaults.clone()
        },
        LeastSquaresOptions {
            gradient_tolerance: f64::NAN,
            ..defaults.clone()
        },
        LeastSquaresOptions {
            max_step_norm: 0.0,
            ..defaults.clone()
        },
        LeastSquaresOptions {
            max_step_norm: f64::INFINITY,
            ..defaults.clone()
        },
    ] {
        assert!(matches!(
            least_squares(&source, &[0.0], &options),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
    let no_steps = LeastSquaresOptions {
        max_iterations: 0,
        ..defaults.clone()
    };
    assert!(matches!(
        least_squares(&source, &[0.0], &no_steps),
        Err(SystemConsumerError::NonlinearSolve(_))
    ));
    assert!(least_squares(&source, &[1.0], &no_steps).is_ok());
    for values in [vec![], vec![0.0, 0.0], vec![f64::INFINITY]] {
        assert!(least_squares(&source, &values, &defaults).is_err());
    }
    let stationary = system(&["x-1"], &["x"]);
    let subnormal_step = LeastSquaresOptions {
        max_step_norm: f64::from_bits(1),
        gradient_tolerance: 0.0,
        ..defaults
    };
    assert!(matches!(
        least_squares(&stationary, &[2.0], &subnormal_step),
        Err(SystemConsumerError::NonlinearSolve(_))
    ));
}

#[derive(Default)]
struct CancelAt {
    checkpoints: usize,
    stop: Option<usize>,
    cancelled: bool,
    charges: usize,
    exhaust_at: Option<usize>,
    exhausted: bool,
    compute: u64,
    bytes: u64,
    allocations: u64,
}

impl CancelAt {
    fn before_charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(
            !self.cancelled && !self.exhausted,
            "work continued after refusal"
        );
        self.charges += 1;
        if self.exhaust_at == Some(self.charges) {
            self.exhausted = true;
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

impl BudgetMeter for CancelAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.before_charge(dimension, amount)?;
        self.accumulate(dimension, amount);
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if let Some(&(dimension, amount)) = charges.first() {
            // An injected refusal must precede every member of an atomic batch.
            self.before_charge(dimension, amount)?;
        }
        for &(dimension, amount) in charges {
            self.accumulate(dimension, amount);
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(
            !self.cancelled && !self.exhausted,
            "work continued after refusal"
        );
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.cancelled = true;
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

#[test]
fn cancellation_at_every_safe_point_prevents_partial_publication() {
    let source = system(&["x-1", "y-1", "x+y-4"], &["x", "y"]);
    let options = LeastSquaresOptions::default();
    let mut complete = CancelAt::default();
    least_squares_with_meter(&source, &[0.0; 2], &options, &mut complete).unwrap();
    assert!(complete.compute > 0 && complete.bytes > 0 && complete.allocations > 0);
    for checkpoint in 1..=complete.checkpoints {
        let mut meter = CancelAt {
            stop: Some(checkpoint),
            ..CancelAt::default()
        };
        assert!(
            least_squares_with_meter(&source, &[0.0; 2], &options, &mut meter).is_err(),
            "published after cancellation at checkpoint {checkpoint}"
        );
        assert!(meter.cancelled);
    }
}

#[test]
fn rejected_trials_and_repeated_calls_consume_the_shared_budget() {
    let source = system(&["x^2-2", "2*x^2-4"], &["x"]);
    let options = LeastSquaresOptions::default();
    let mut measured = CancelAt::default();
    let result = least_squares_with_meter(&source, &[0.01], &options, &mut measured).unwrap();
    assert!(result.evaluations > result.iterations + 2);
    let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
    limits.dimensions[Dimension::ComputeSteps.index()] = measured.compute;
    let mut budget = Budget::new(limits);
    least_squares_with_meter(&source, &[0.01], &options, &mut budget).unwrap();
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(budget.verifier_remaining(), 37);
    assert!(least_squares_with_meter(&source, &[0.01], &options, &mut budget).is_err());
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
}

fn meter_cause(error: &SystemConsumerError) -> Option<MeterError> {
    match error {
        SystemConsumerError::Meter(error)
        | SystemConsumerError::Evaluation(EvalError::Meter(error))
        | SystemConsumerError::SparseEvaluation(ResidualEvaluationError::Evaluation(
            EvalError::Meter(error),
        )) => Some(error.clone()),
        _ => None,
    }
}

#[test]
fn backtracked_domain_trials_never_swallow_cancellation_or_exhaustion() {
    let source = system(&["log(x)"], &["x"]);
    let options = LeastSquaresOptions::default();
    let mut complete = CancelAt::default();
    let result = least_squares_with_meter(&source, &[10.0], &options, &mut complete).unwrap();
    assert!(result.evaluations > result.iterations + 2);
    for checkpoint in 1..=complete.checkpoints {
        let mut meter = CancelAt {
            stop: Some(checkpoint),
            ..CancelAt::default()
        };
        let error = least_squares_with_meter(&source, &[10.0], &options, &mut meter)
            .expect_err("published after cancellation during domain backtracking");
        assert_eq!(meter_cause(&error), Some(MeterError::Cancelled));
        assert!(meter.cancelled);
        assert_eq!(meter.checkpoints, checkpoint);
    }
    for charge in 1..=complete.charges {
        let mut meter = CancelAt {
            exhaust_at: Some(charge),
            ..CancelAt::default()
        };
        let error = least_squares_with_meter(&source, &[10.0], &options, &mut meter)
            .expect_err("published after resource exhaustion during domain backtracking");
        assert!(
            matches!(
                meter_cause(&error),
                Some(MeterError::Budget(BudgetError::Exhausted { .. }))
            ),
            "charge {charge}: {error}"
        );
        assert!(meter.exhausted);
        assert_eq!(meter.charges, charge);
    }
}

#[test]
fn failed_domain_trials_are_charged_to_every_cumulative_budget_dimension() {
    let source = system(&["log(x)"], &["x"]);
    let options = LeastSquaresOptions::default();
    let mut measured = CancelAt::default();
    let result = least_squares_with_meter(&source, &[10.0], &options, &mut measured).unwrap();
    assert!(result.evaluations > result.iterations + 2);
    for (dimension, total) in [
        (Dimension::ComputeSteps, measured.compute),
        (Dimension::MemoryBytes, measured.bytes),
        (Dimension::AllocationCount, measured.allocations),
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
        limits.dimensions[dimension.index()] = total;
        let mut budget = Budget::new(limits);
        least_squares_with_meter(&source, &[10.0], &options, &mut budget).unwrap();
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);
        assert!(least_squares_with_meter(&source, &[10.0], &options, &mut budget).is_err());
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);

        limits.dimensions[dimension.index()] -= 1;
        let mut short = Budget::new(limits);
        let error = least_squares_with_meter(&source, &[10.0], &options, &mut short)
            .expect_err("published without the final required resource unit");
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted {
                dimension: exhausted,
                ..
            })) if exhausted == dimension
        ));
        assert_eq!(short.verifier_remaining(), 37);
    }
}
