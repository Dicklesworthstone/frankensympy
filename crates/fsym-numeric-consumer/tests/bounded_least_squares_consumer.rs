//! Box-constrained proposals retain original domains and distinguish KKT from
//! unconstrained stationarity, root finding and minimum certification.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_core::{Expr, Symbol, parse};
use fsym_numeric_consumer::{
    LeastSquaresOptions, SystemConsumerError, sparse_bounded_least_squares,
    sparse_bounded_least_squares_with_meter, sparse_least_squares,
};

fn system(sources: &[&str], variables: &[&str]) -> VerifiedResidualSystem {
    let sources: Vec<_> = sources
        .iter()
        .map(|source| parse(source).unwrap())
        .collect();
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
fn interior_solution_preserves_the_original_unconstrained_fit() {
    let source = system(&["x-1", "y-1", "x+y-4"], &["x", "y"]);
    let controls = LeastSquaresOptions::default();
    let bounded = sparse_bounded_least_squares(
        &source,
        &[0.0, 0.0],
        &[-10.0, -10.0],
        &[10.0, 10.0],
        &controls,
    )
    .unwrap();
    let free = sparse_least_squares(&source, &[0.0, 0.0], &controls).unwrap();
    for (&actual, &expected) in bounded.point.iter().zip(&free.point) {
        near(actual, expected, 1e-10);
        near(actual, 5.0 / 3.0, 1e-10);
    }
    near(bounded.cost, 2.0 / 3.0, 1e-12);
    near(bounded.cost, free.cost, 1e-12);
    assert_eq!(bounded.gradient_norm, bounded.projected_gradient_norm);
    assert!(bounded.projected_gradient_norm <= controls.gradient_tolerance);
    assert_eq!(bounded.residuals.len(), 3);
}

#[test]
fn active_upper_and_lower_bounds_admit_nonzero_original_gradients() {
    for (residual, lower, upper, initial, target, gradient, cost) in [
        ("x-2", -1.0, 1.0, 0.0, 1.0, 1.0, 0.5),
        ("x+2", 0.0, f64::INFINITY, 1.0, 0.0, 2.0, 2.0),
    ] {
        let source = system(&[residual], &["x"]);
        let result = sparse_bounded_least_squares(
            &source,
            &[initial],
            &[lower],
            &[upper],
            &LeastSquaresOptions::default(),
        )
        .unwrap();
        assert_eq!(result.point, [target]);
        assert_eq!(result.projected_gradient_norm, 0.0);
        near(result.gradient_norm, gradient, 1e-12);
        near(result.cost, cost, 1e-12);
        assert!(result.residuals[0].abs() >= 1.0);
    }
}

#[test]
fn inward_gradient_releases_a_coordinate_at_either_endpoint() {
    for (residual, lower, upper, initial, target) in
        [("x-2", 0.0, 3.0, 0.0, 2.0), ("x+2", -3.0, 1.0, 1.0, -2.0)]
    {
        let source = system(&[residual], &["x"]);
        let result = sparse_bounded_least_squares(
            &source,
            &[initial],
            &[lower],
            &[upper],
            &LeastSquaresOptions::default(),
        )
        .unwrap();
        near(result.point[0], target, 1e-10);
        assert!(result.iterations > 0);
    }
}

#[test]
fn fixed_coordinate_changes_the_complete_coupled_fit() {
    let source = system(&["x+y-7", "2*y-4"], &["x", "y"]);
    let result = sparse_bounded_least_squares(
        &source,
        &[3.0, 0.0],
        &[3.0, f64::NEG_INFINITY],
        &[3.0, f64::INFINITY],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    assert_eq!(result.point[0], 3.0);
    near(result.point[1], 12.0 / 5.0, 1e-10);
    near(result.residuals[0], -8.0 / 5.0, 1e-10);
    near(result.residuals[1], 4.0 / 5.0, 1e-10);
    near(result.cost, 8.0 / 5.0, 1e-10);
    near(result.gradient_norm, 8.0 / 5.0, 1e-10);
    assert!(result.projected_gradient_norm <= 1e-10);
}

#[test]
fn coupled_fit_reoptimizes_after_a_bound_becomes_active() {
    // The unconstrained solution is (-3,1), whose simple clipping to (0,1)
    // is not stationary. The constrained minimizer is exactly (0,-1/5).
    let source = system(&["x+2*y+1", "y-1"], &["x", "y"]);
    for initial in [[1.0, 0.0], [2.0_f64.powi(-60), 0.0], [0.0, 2.0]] {
        let result = sparse_bounded_least_squares(
            &source,
            &initial,
            &[0.0, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::INFINITY],
            &LeastSquaresOptions::default(),
        )
        .unwrap();
        assert_eq!(result.point[0], 0.0);
        near(result.point[1], -0.2, 1e-10);
        near(result.residuals[0], 0.6, 1e-10);
        near(result.residuals[1], -1.2, 1e-10);
        near(result.cost, 0.9, 1e-12);
        near(result.gradient_norm, 0.6, 1e-10);
        assert!(result.projected_gradient_norm <= 1e-10);
    }
}

#[test]
fn projected_gradient_step_recovers_when_all_damped_model_directions_face_outward() {
    // g=(1,1e-8) to rounding, while every admitted damped model direction
    // has negative x and positive y. Projection erases the x descent at this
    // tiny positive interior coordinate, leaving ascent for every model trial.
    // The separate projected-gradient direction must retain inward progress.
    let source = system(&["x+2*y+1", "y-2+1/100000000"], &["x", "y"]);
    let result = sparse_bounded_least_squares(
        &source,
        &[2.0_f64.powi(-200), 0.0],
        &[0.0, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    assert_eq!(result.point[0], 0.0);
    near(result.point[1], -1.0 / 500_000_000.0, 2e-11);
    near(result.gradient_norm, 1.0 - 1.0 / 250_000_000.0, 1e-10);
    assert!(result.projected_gradient_norm <= 1e-10);
    assert!(result.iterations > 0);
}

#[test]
fn multiple_bound_corners_and_unused_fixed_coordinates_preserve_layout() {
    let source = system(&["x-4", "y+3", "x+y-2"], &["y", "unused", "x"]);
    let result = sparse_bounded_least_squares(
        &source,
        &[0.0, 7.0, 0.0],
        &[-1.0, 7.0, 0.0],
        &[2.0, 7.0, 1.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    assert_eq!(result.point, [-1.0, 7.0, 1.0]);
    assert_eq!(result.residuals, [-3.0, 2.0, -2.0]);
    assert_eq!(result.cost, 8.5);
    assert_eq!(result.gradient_norm, 5.0);
    assert_eq!(result.projected_gradient_norm, 0.0);
}

#[test]
fn underdetermined_and_rank_deficient_systems_keep_all_original_rows() {
    let source = system(&["x+y-3"], &["x", "y", "unused"]);
    let result = sparse_bounded_least_squares(
        &source,
        &[0.0, 0.0, 9.0],
        &[0.0, 0.0, f64::NEG_INFINITY],
        &[1.0, 3.0, f64::INFINITY],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(result.point[0] + result.point[1], 3.0, 1e-10);
    assert!((0.0..=1.0).contains(&result.point[0]));
    assert_eq!(result.point[2], 9.0);
    let redundant = system(&["x+y-1", "2*x+2*y-2", "3*x+3*y-4"], &["x", "y"]);
    let result = sparse_bounded_least_squares(
        &redundant,
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[0.25, 2.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(result.point[0] + result.point[1], 17.0 / 14.0, 1e-10);
    near(result.cost, 5.0 / 28.0, 1e-12);
    assert_eq!(result.residuals.len(), 3);
}

#[test]
fn nonlinear_fit_relinearizes_inside_the_box() {
    let source = system(&["x*x", "x-3"], &["x"]);
    for (lower, upper, target, expected_cost) in [(0.0, 2.0, 1.0, 2.5), (0.0, 0.5, 0.5, 3.15625)] {
        let result = sparse_bounded_least_squares(
            &source,
            &[0.1],
            &[lower],
            &[upper],
            &LeastSquaresOptions::default(),
        )
        .unwrap();
        near(result.point[0], target, 1e-9);
        near(result.cost, expected_cost, 1e-10);
        assert!(result.projected_gradient_norm <= 1e-10);
    }
}

#[test]
fn domain_invalid_projected_trials_backtrack_within_bounds() {
    let source = system(&["log(x)", "2*log(x)"], &["x"]);
    let result = sparse_bounded_least_squares(
        &source,
        &[10.0],
        &[0.0],
        &[20.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(result.point[0], 1.0, 1e-9);
    assert!(result.evaluations > result.iterations + 2);
}

#[test]
fn fixed_or_blocked_coordinates_do_not_erase_source_and_derivative_holes() {
    for (residual, point) in [("log(x)", 0.0), ("x^(1/2)-1", 0.0), ("log(-1)", 2.0)] {
        let source = system(&[residual], &["x"]);
        assert!(
            sparse_bounded_least_squares(
                &source,
                &[point],
                &[point],
                &[point],
                &LeastSquaresOptions::default(),
            )
            .is_err()
        );
    }
    let source = system(&["x-1", "log(y)"], &["x", "y"]);
    assert!(
        sparse_bounded_least_squares(
            &source,
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &LeastSquaresOptions::default(),
        )
        .is_err()
    );
}

#[test]
fn constant_rows_and_stationary_maxima_remain_approximate_proposals() {
    let source = system(&["x*x-1", "3"], &["x"]);
    let result = sparse_bounded_least_squares(
        &source,
        &[0.0],
        &[-2.0],
        &[2.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    assert_eq!(result.point, [0.0]);
    assert_eq!(result.residuals, [-1.0, 3.0]);
    assert_eq!(result.cost, 5.0);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.evaluations, 2);
    let fixed = sparse_bounded_least_squares(
        &source,
        &[2.0],
        &[2.0],
        &[2.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    assert_eq!(fixed.gradient_norm, 12.0);
    assert_eq!(fixed.projected_gradient_norm, 0.0);
}

#[test]
fn invalid_bounds_and_initial_feasibility_refuse_without_clipping() {
    let source = system(&["x-1"], &["x"]);
    for (initial, lower, upper) in [
        (0.0, 1.0, 2.0),
        (3.0, 1.0, 2.0),
        (0.0, 2.0, 1.0),
        (0.0, f64::NAN, 2.0),
        (0.0, 0.0, f64::NAN),
        (0.0, f64::INFINITY, f64::INFINITY),
        (0.0, f64::NEG_INFINITY, f64::NEG_INFINITY),
    ] {
        assert!(matches!(
            sparse_bounded_least_squares(
                &source,
                &[initial],
                &[lower],
                &[upper],
                &LeastSquaresOptions::default(),
            ),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
    for initial in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            sparse_bounded_least_squares(
                &source,
                &[initial],
                &[f64::NEG_INFINITY],
                &[f64::INFINITY],
                &LeastSquaresOptions::default(),
            ),
            Err(SystemConsumerError::NonFinite(_))
        ));
    }
    for (initial, lower, upper) in [
        (vec![], vec![0.0], vec![2.0]),
        (vec![0.0], vec![], vec![2.0]),
        (vec![0.0], vec![0.0], vec![2.0, 3.0]),
    ] {
        assert!(matches!(
            sparse_bounded_least_squares(
                &source,
                &initial,
                &lower,
                &upper,
                &LeastSquaresOptions::default(),
            ),
            Err(SystemConsumerError::Dimension { .. })
        ));
    }
}

#[test]
fn tiny_interior_distance_or_small_step_never_authorizes_stationarity() {
    let source = system(&["x+1"], &["x"]);
    let no_steps = LeastSquaresOptions {
        max_iterations: 0,
        ..LeastSquaresOptions::default()
    };
    let error =
        sparse_bounded_least_squares(&source, &[2.0_f64.powi(-60)], &[0.0], &[1.0], &no_steps)
            .unwrap_err();
    assert!(matches!(
        error,
        SystemConsumerError::BoundedLeastSquaresNoConvergence {
            iterations: 0,
            projected_gradient_norm: 1.0,
        }
    ));
    assert!(sparse_bounded_least_squares(&source, &[0.0], &[0.0], &[1.0], &no_steps).is_ok());
    let tiny_step = LeastSquaresOptions {
        max_step_norm: f64::from_bits(1),
        gradient_tolerance: 0.0,
        ..LeastSquaresOptions::default()
    };
    assert!(matches!(
        sparse_bounded_least_squares(&source, &[0.5], &[0.0], &[1.0], &tiny_step),
        Err(SystemConsumerError::BoundedLeastSquaresNoConvergence { .. })
    ));
}

#[test]
fn step_clipping_and_iteration_limit_keep_their_original_meaning() {
    let source = system(&["x-2"], &["x"]);
    let controls = LeastSquaresOptions {
        max_step_norm: 0.25,
        ..LeastSquaresOptions::default()
    };
    let result = sparse_bounded_least_squares(&source, &[0.0], &[0.0], &[1.0], &controls).unwrap();
    assert_eq!(result.point, [1.0]);
    assert!(result.iterations >= 4);
    for controls in [
        LeastSquaresOptions {
            max_iterations: 1025,
            ..controls.clone()
        },
        LeastSquaresOptions {
            gradient_tolerance: -1.0,
            ..controls.clone()
        },
        LeastSquaresOptions {
            gradient_tolerance: f64::INFINITY,
            ..controls.clone()
        },
        LeastSquaresOptions {
            gradient_tolerance: f64::NAN,
            ..controls.clone()
        },
        LeastSquaresOptions {
            max_step_norm: 0.0,
            ..controls.clone()
        },
        LeastSquaresOptions {
            max_step_norm: f64::INFINITY,
            ..controls.clone()
        },
    ] {
        assert!(matches!(
            sparse_bounded_least_squares(&source, &[0.0], &[0.0], &[1.0], &controls),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
}

#[test]
fn sparse_storage_and_bound_activity_scale_beyond_dense_dimensions() {
    let n = 256;
    let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
    let residuals: Vec<_> = variables
        .iter()
        .map(|variable| Expr::Add(vec![Expr::Sym(variable.clone()), Expr::from_i64(-2)]))
        .collect();
    let source = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
    let mut meter = Recorder::default();
    let result = sparse_bounded_least_squares_with_meter(
        &source,
        &vec![0.0; n],
        &vec![0.0; n],
        &vec![1.0; n],
        &LeastSquaresOptions::default(),
        &mut meter,
    )
    .unwrap();
    assert_eq!(result.point, vec![1.0; n]);
    assert_eq!(result.residuals, vec![-1.0; n]);
    assert_eq!(result.cost, 128.0);
    assert_eq!(result.gradient_norm, 1.0);
    assert_eq!(result.projected_gradient_norm, 0.0);
    assert!(meter.max_memory_charge <= 32 * n as u64);
}

#[test]
fn immutable_artifact_replay_preserves_bounds_and_coordinate_order() {
    let source = system(&["x-3", "y+2", "x+y-1"], &["y", "x"]);
    let restored =
        VerifiedResidualSystem::from_json_bytes(&source.to_json_bytes().unwrap()).unwrap();
    let run = |source: &VerifiedResidualSystem| {
        sparse_bounded_least_squares(
            source,
            &[0.0, 0.0],
            &[-1.0, 0.0],
            &[2.0, 1.0],
            &LeastSquaresOptions::default(),
        )
        .unwrap()
    };
    assert_eq!(run(&source), run(&restored));
    assert_eq!(source.vars(), restored.vars());
    assert_eq!(source.residuals(), restored.residuals());
}

#[derive(Default)]
struct Recorder {
    checkpoints: usize,
    stop: Option<usize>,
    charges: usize,
    exhaust_at: Option<usize>,
    stopped: bool,
    compute: u64,
    bytes: u64,
    allocations: u64,
    max_memory_charge: u64,
}

impl Recorder {
    fn before_charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "continued after an operational refusal");
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
            Dimension::MemoryBytes => {
                self.bytes += amount;
                self.max_memory_charge = self.max_memory_charge.max(amount);
            }
            Dimension::AllocationCount => self.allocations += amount,
            _ => {}
        }
    }
}

impl BudgetMeter for Recorder {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.before_charge(dimension, amount)?;
        self.accumulate(dimension, amount);
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if let Some(&(dimension, amount)) = charges.first() {
            self.before_charge(dimension, amount)?;
        }
        for &(dimension, amount) in charges {
            self.accumulate(dimension, amount);
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "continued after an operational refusal");
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.stopped = true;
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

fn meter_cause(error: &SystemConsumerError) -> Option<MeterError> {
    match error {
        SystemConsumerError::Meter(error)
        | SystemConsumerError::Evaluation(EvalError::Meter(error))
        | SystemConsumerError::SparseEvaluation(ResidualEvaluationError::Evaluation(
            EvalError::Meter(error),
        )) => Some(*error),
        _ => None,
    }
}

#[test]
fn cancellation_and_exhaustion_remain_terminal_through_final_publication() {
    let source = system(&["x-2"], &["x"]);
    let controls = LeastSquaresOptions::default();
    let mut complete = Recorder::default();
    sparse_bounded_least_squares_with_meter(
        &source,
        &[0.0],
        &[0.0],
        &[1.0],
        &controls,
        &mut complete,
    )
    .unwrap();
    for checkpoint in 1..=complete.checkpoints {
        let mut meter = Recorder {
            stop: Some(checkpoint),
            ..Recorder::default()
        };
        let error = sparse_bounded_least_squares_with_meter(
            &source,
            &[0.0],
            &[0.0],
            &[1.0],
            &controls,
            &mut meter,
        )
        .expect_err("published after cancellation");
        assert_eq!(meter_cause(&error), Some(MeterError::Cancelled));
        assert_eq!(meter.checkpoints, checkpoint);
        assert!(meter.stopped);
    }
    for charge in 1..=complete.charges {
        let mut meter = Recorder {
            exhaust_at: Some(charge),
            ..Recorder::default()
        };
        let error = sparse_bounded_least_squares_with_meter(
            &source,
            &[0.0],
            &[0.0],
            &[1.0],
            &controls,
            &mut meter,
        )
        .expect_err("published after exhaustion");
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted { .. }))
        ));
        assert_eq!(meter.charges, charge);
        assert!(meter.stopped);
    }
}

#[test]
fn all_budget_dimensions_accumulate_across_rejected_trials_and_repeat_calls() {
    let source = system(&["log(x)"], &["x"]);
    let controls = LeastSquaresOptions::default();
    let mut measured = Recorder::default();
    let result = sparse_bounded_least_squares_with_meter(
        &source,
        &[10.0],
        &[0.0],
        &[20.0],
        &controls,
        &mut measured,
    )
    .unwrap();
    assert!(result.evaluations > result.iterations + 2);
    for (dimension, total) in [
        (Dimension::ComputeSteps, measured.compute),
        (Dimension::MemoryBytes, measured.bytes),
        (Dimension::AllocationCount, measured.allocations),
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
        limits.dimensions[dimension.index()] = total;
        let mut budget = Budget::new(limits);
        sparse_bounded_least_squares_with_meter(
            &source,
            &[10.0],
            &[0.0],
            &[20.0],
            &controls,
            &mut budget,
        )
        .unwrap();
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);
        assert!(
            sparse_bounded_least_squares_with_meter(
                &source,
                &[10.0],
                &[0.0],
                &[20.0],
                &controls,
                &mut budget,
            )
            .is_err()
        );
        assert_eq!(budget.remaining(dimension), 0);
        limits.dimensions[dimension.index()] -= 1;
        let mut short = Budget::new(limits);
        let error = sparse_bounded_least_squares_with_meter(
            &source,
            &[10.0],
            &[0.0],
            &[20.0],
            &controls,
            &mut short,
        )
        .expect_err("published without the last required resource unit");
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted { dimension: exhausted, .. })) if exhausted == dimension
        ));
        assert_eq!(short.verifier_remaining(), 37);
    }
}

#[test]
fn published_gradients_preserve_active_fixed_and_free_components_in_variable_order() {
    let source = system(&["x-3", "y+4", "z-8", "2"], &["x", "y", "z"]);
    let initial = [1.0, -1.0, 2.0];
    let lower = [f64::NEG_INFINITY, -1.0, 2.0];
    for (upper_x, tolerance, projected_x) in [(1.0, 0.0, 0.0), (4.0, 2.0, -2.0)] {
        let controls = LeastSquaresOptions {
            max_iterations: 0,
            gradient_tolerance: tolerance,
            ..LeastSquaresOptions::default()
        };
        let fit = sparse_bounded_least_squares(
            &source,
            &initial,
            &lower,
            &[upper_x, f64::INFINITY, 2.0],
            &controls,
        )
        .unwrap();
        assert_eq!(fit.point, initial);
        assert_eq!(fit.residuals, [-2.0, 3.0, -6.0, 2.0]);
        assert_eq!(fit.gradient, [-2.0, 3.0, -6.0]);
        assert_eq!(fit.projected_gradient, [projected_x, 0.0, 0.0]);
        assert_eq!(fit.gradient_norm, 6.0);
        assert_eq!(fit.projected_gradient_norm, projected_x.abs());
        assert_eq!(fit.iterations, 0);
    }
}

#[test]
fn fixed_zero_residuals_still_require_the_original_derivative_domain() {
    for (formula, primitive) in [("x^(1/2)", "power"), ("abs(x)", "abs")] {
        let source = system(&[formula], &["x"]);
        assert_eq!(
            source.residual_programs()[0]
                .try_eval_finite(&[0.0])
                .unwrap(),
            0.0
        );
        let result = sparse_bounded_least_squares(
            &source,
            &[0.0],
            &[0.0],
            &[0.0],
            &LeastSquaresOptions::default(),
        );
        assert!(
            matches!(result, Err(SystemConsumerError::SparseEvaluation(
            ResidualEvaluationError::Evaluation(EvalError::NonsmoothSource(name))
        )) if name == primitive),
            "{formula}: {result:?}"
        );
    }
}
