//! Sparse numerical fitting retains all source rows and never certifies a minimum.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_core::{Expr, Symbol, parse};
use fsym_numeric_consumer::{
    LeastSquaresOptions, LeastSquaresTermination, SystemConsumerError, sparse_least_squares,
    sparse_least_squares_with_meter,
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
fn admitted_sparse_systems_reach_sixty_four_and_two_hundred_fifty_six_variables() {
    for n in [64, 256] {
        let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
        let residuals: Vec<_> = variables
            .iter()
            .enumerate()
            .map(|(i, variable)| {
                Expr::Add(vec![
                    Expr::Sym(variable.clone()),
                    Expr::from_i64(-((i % 7 + 1) as i64)),
                ])
            })
            .collect();
        let source = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
        assert_eq!(source.jacobian_pattern().len(), n);
        let result =
            sparse_least_squares(&source, &vec![0.0; n], &LeastSquaresOptions::default()).unwrap();
        for (i, &value) in result.point.iter().enumerate() {
            near(value, (i % 7 + 1) as f64, 1e-10);
        }
        assert!(result.gradient_norm <= 1e-10);
        assert!(result.cost <= 1e-18);
        assert_eq!(result.residuals.len(), n);
        assert_eq!(
            result.termination,
            LeastSquaresTermination::GradientTolerance
        );
    }
}

#[test]
fn coupled_banded_equations_use_all_sparse_directions() {
    let n = 64;
    let names: Vec<_> = (0..n).map(|i| format!("x{i}")).collect();
    let variables: Vec<_> = names.iter().map(|name| Symbol::new(name.clone())).collect();
    let residuals: Vec<_> = (0..n)
        .map(|i| {
            let mut text = format!("4*{}", names[i]);
            if i > 0 {
                text.push_str(&format!("-{}", names[i - 1]));
            }
            if i + 1 < n {
                text.push_str(&format!("-{}", names[i + 1]));
            }
            text.push_str(if i == 0 || i + 1 == n { "-3" } else { "-2" });
            parse(&text).unwrap()
        })
        .collect();
    let source = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
    let result =
        sparse_least_squares(&source, &vec![0.0; n], &LeastSquaresOptions::default()).unwrap();
    for &value in &result.point {
        near(value, 1.0, 1e-9);
    }
    assert!(result.gradient_norm <= 1e-10);
}

#[test]
fn four_thousand_residual_rows_determine_the_inconsistent_fit_without_square_buffers() {
    let n = 16;
    let m = 4096;
    let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
    let residuals: Vec<_> = (0..m)
        .map(|row| {
            let column = row % n;
            let offset = if (row / n) % 2 == 0 { -1 } else { 1 };
            let target = (column + 1) as i64 + offset;
            Expr::Add(vec![
                Expr::Sym(variables[column].clone()),
                Expr::from_i64(-target),
            ])
        })
        .collect();
    let source = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
    let mut meter = Recorder::default();
    let result = sparse_least_squares_with_meter(
        &source,
        &[0.0; 16],
        &LeastSquaresOptions::default(),
        &mut meter,
    )
    .unwrap();
    for (column, &value) in result.point.iter().enumerate() {
        near(value, (column + 1) as f64, 1e-10);
    }
    assert_eq!(result.residuals.len(), m);
    for (row, &residual) in result.residuals.iter().enumerate() {
        near(residual, if (row / n) % 2 == 0 { 1.0 } else { -1.0 }, 1e-10);
    }
    near(result.cost, 2048.0, 1e-8);
    assert!(result.gradient_norm <= 1e-10);
    // A dense m-by-m Q alone would require 128 MiB. No single solver or
    // evaluator buffer reservation here may exceed the sparse/vector envelope.
    assert!(meter.max_memory_charge <= 128 * 1024);
}

#[test]
fn redundant_columns_reach_the_nonzero_residual_stationary_family() {
    let source = system(&["x+y-1", "2*x+2*y-2", "3*x+3*y-4"], &["x", "y"]);
    let result =
        sparse_least_squares(&source, &[0.0, 0.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0] + result.point[1], 17.0 / 14.0, 1e-11);
    near(result.cost, 5.0 / 28.0, 1e-12);
    assert!(result.gradient_norm <= 1e-10);
    assert!(result.residuals.iter().any(|value| value.abs() > 0.1));
}

#[test]
fn underdetermined_systems_and_unused_coordinates_can_progress() {
    let source = system(&["x+y-3"], &["x", "y", "z"]);
    let result =
        sparse_least_squares(&source, &[0.0, 0.0, 7.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0] + result.point[1], 3.0, 1e-11);
    assert_eq!(result.point[2], 7.0);
    assert!(result.gradient_norm <= 1e-10);
    let unused = system(&["x-2", "2*x-4"], &["x", "y"]);
    let result =
        sparse_least_squares(&unused, &[0.0, -9.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 2.0, 1e-12);
    assert_eq!(result.point[1], -9.0);
}

#[test]
fn column_equilibration_preserves_physical_units_and_the_original_cost() {
    let source = system(&["1000000000*(x-1)", "(y-2)/1000000000"], &["x", "y"]);
    let result =
        sparse_least_squares(&source, &[0.0, 0.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-12);
    near(result.point[1], 2.0, 1e-8);
    assert!(result.gradient_norm <= 1e-10);
}

#[test]
fn nonlinear_overdetermined_and_inconsistent_problems_use_fresh_linearizations() {
    let source = system(&["x*x-1", "y-x*x", "x+y-2"], &["x", "y"]);
    let result =
        sparse_least_squares(&source, &[2.0, 0.0], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-9);
    near(result.point[1], 1.0, 1e-9);
    assert!(result.gradient_norm <= 1e-10);

    // The cost is (x^4 + (x-3)^2)/2. Its derivative 2*x^3+x-3
    // vanishes at one, and its second derivative is positive everywhere.
    let inconsistent = system(&["x*x", "x-3"], &["x"]);
    let result =
        sparse_least_squares(&inconsistent, &[0.1], &LeastSquaresOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-9);
    near(result.cost, 2.5, 1e-10);
    assert!(result.gradient_norm <= 1e-10);
}

#[test]
fn logarithmic_square_root_and_exponential_trials_backtrack() {
    for (residuals, initial, target) in [
        (vec!["log(x)", "2*log(x)"], 10.0, 1.0),
        (vec!["x^(1/2)-1", "2*x^(1/2)-2"], 4.0, 1.0),
        (vec!["exp(x)-1", "2*exp(x)-2"], -7.0, 0.0),
    ] {
        let source = system(&residuals, &["x"]);
        let result =
            sparse_least_squares(&source, &[initial], &LeastSquaresOptions::default()).unwrap();
        near(result.point[0], target, 1e-8);
        assert!(result.gradient_norm <= 1e-10);
        assert!(result.evaluations > result.iterations + 2);
    }
}

#[test]
fn invalid_initial_values_and_derivative_only_holes_remain_terminal() {
    for (residuals, point) in [
        (vec!["log(x)"], -1.0),
        (vec!["log(x)"], 0.0),
        (vec!["x^(1/2)-1"], 0.0),
        (vec!["log(-1)"], 2.0),
    ] {
        let source = system(&residuals, &["x"]);
        assert!(sparse_least_squares(&source, &[point], &LeastSquaresOptions::default()).is_err());
    }
    let overflow = system(&["x"], &["x"]);
    assert!(matches!(
        sparse_least_squares(&overflow, &[1e200], &LeastSquaresOptions::default()),
        Err(SystemConsumerError::NonFinite(_))
    ));
    let gradient = system(&["exp(x)*y+1"], &["x", "y"]);
    assert!(matches!(
        sparse_least_squares(&gradient, &[500.0, 1e-64], &LeastSquaresOptions::default()),
        Err(SystemConsumerError::NonFinite(_))
    ));
}

#[test]
fn held_original_holes_are_never_erased_by_zero_gradient_or_zero_columns() {
    let x = Symbol::new("x");
    let hidden = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
    ]);
    let source = VerifiedResidualSystem::try_compile(
        &[Expr::Sym(x.clone()), hidden],
        &[x, Symbol::new("unused")],
    )
    .unwrap();
    assert!(sparse_least_squares(&source, &[0.0, 2.0], &LeastSquaresOptions::default()).is_err());
    let options = LeastSquaresOptions {
        max_iterations: 4,
        gradient_tolerance: 0.0,
        ..LeastSquaresOptions::default()
    };
    assert!(matches!(
        sparse_least_squares(&source, &[1.0, 2.0], &options),
        Err(SystemConsumerError::SparseLeastSquaresNoConvergence { .. })
    ));
}

#[test]
fn stationarity_does_not_imply_a_root_or_minimum() {
    let source = system(&["x*x-1"], &["x", "unused"]);
    let result =
        sparse_least_squares(&source, &[0.0, 7.0], &LeastSquaresOptions::default()).unwrap();
    assert_eq!(result.point, [0.0, 7.0]);
    assert_eq!(result.cost, 0.5);
    assert_eq!(result.gradient_norm, 0.0);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.evaluations, 2);
}

#[test]
fn transport_and_typed_coordinate_order_preserve_the_sparse_solve() {
    let source = system(&["x-1", "y-1", "x+y-4"], &["y", "x"]);
    let restored =
        VerifiedResidualSystem::from_json_bytes(&source.to_json_bytes().unwrap()).unwrap();
    let original =
        sparse_least_squares(&source, &[1.0, 2.0], &LeastSquaresOptions::default()).unwrap();
    let replayed =
        sparse_least_squares(&restored, &[1.0, 2.0], &LeastSquaresOptions::default()).unwrap();
    assert_eq!(original, replayed);
    assert_eq!(restored.vars(), source.vars());
    assert_eq!(restored.residuals(), source.residuals());
}

#[test]
fn shape_option_and_initial_point_bounds_precede_numerical_work() {
    let too_many_variables: Vec<_> = (0..257).map(|i| Symbol::new(format!("x{i}"))).collect();
    for (residuals, variables) in [
        (vec![], vec![Symbol::new("x")]),
        (vec![Expr::from_i64(0)], vec![]),
        (vec![Expr::from_i64(0)], too_many_variables),
        (vec![Expr::from_i64(0); 4097], vec![Symbol::new("x")]),
    ] {
        let source = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
        assert!(matches!(
            sparse_least_squares(
                &source,
                &vec![0.0; variables.len()],
                &LeastSquaresOptions::default(),
            ),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
    let source = system(&["x-1"], &["x"]);
    let defaults = LeastSquaresOptions::default();
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
            sparse_least_squares(&source, &[0.0], &options),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
    for point in [vec![], vec![0.0, 0.0], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(sparse_least_squares(&source, &point, &defaults).is_err());
    }
}

#[test]
fn clipping_and_iteration_limits_do_not_authorize_false_success() {
    let source = system(&["x-2"], &["x"]);
    let clipped = LeastSquaresOptions {
        max_step_norm: 0.25,
        ..LeastSquaresOptions::default()
    };
    let result = sparse_least_squares(&source, &[0.0], &clipped).unwrap();
    near(result.point[0], 2.0, 1e-10);
    assert!(result.iterations >= 8);
    let no_steps = LeastSquaresOptions {
        max_iterations: 0,
        ..LeastSquaresOptions::default()
    };
    assert!(matches!(
        sparse_least_squares(&source, &[0.0], &no_steps),
        Err(SystemConsumerError::SparseLeastSquaresNoConvergence { iterations: 0, .. })
    ));
    assert!(sparse_least_squares(&source, &[2.0], &no_steps).is_ok());
    let tiny_step = LeastSquaresOptions {
        max_step_norm: f64::from_bits(1),
        gradient_tolerance: 0.0,
        ..LeastSquaresOptions::default()
    };
    assert!(matches!(
        sparse_least_squares(&source, &[1.0], &tiny_step),
        Err(SystemConsumerError::SparseLeastSquaresNoConvergence { .. })
    ));
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
        assert!(
            !self.stopped,
            "continued after a terminal operational refusal"
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
        assert!(
            !self.stopped,
            "continued after a terminal operational refusal"
        );
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
        )) => Some(error.clone()),
        _ => None,
    }
}

#[test]
fn every_sparse_iteration_and_trial_safe_point_preserves_typed_operational_failure() {
    let source = system(&["log(x)", "2*log(x)"], &["x", "unused"]);
    let options = LeastSquaresOptions::default();
    let mut complete = Recorder::default();
    let result =
        sparse_least_squares_with_meter(&source, &[10.0, 3.0], &options, &mut complete).unwrap();
    assert!(result.evaluations > result.iterations + 2);
    for checkpoint in 1..=complete.checkpoints {
        let mut meter = Recorder {
            stop: Some(checkpoint),
            ..Recorder::default()
        };
        let error = sparse_least_squares_with_meter(&source, &[10.0, 3.0], &options, &mut meter)
            .expect_err("published a candidate after cancellation");
        assert_eq!(meter_cause(&error), Some(MeterError::Cancelled));
        assert!(meter.stopped);
        assert_eq!(meter.checkpoints, checkpoint);
    }
    for charge in 1..=complete.charges {
        let mut meter = Recorder {
            exhaust_at: Some(charge),
            ..Recorder::default()
        };
        let error = sparse_least_squares_with_meter(&source, &[10.0, 3.0], &options, &mut meter)
            .expect_err("published a candidate after exhaustion");
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted { .. }))
        ));
        assert!(meter.stopped);
        assert_eq!(meter.charges, charge);
    }
}

#[test]
fn rejected_trials_and_repeated_solves_share_each_budget_dimension() {
    let source = system(&["log(x)"], &["x"]);
    let options = LeastSquaresOptions::default();
    let mut measured = Recorder::default();
    let result =
        sparse_least_squares_with_meter(&source, &[10.0], &options, &mut measured).unwrap();
    assert!(result.evaluations > result.iterations + 2);
    for (dimension, total) in [
        (Dimension::ComputeSteps, measured.compute),
        (Dimension::MemoryBytes, measured.bytes),
        (Dimension::AllocationCount, measured.allocations),
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
        limits.dimensions[dimension.index()] = total;
        let mut budget = Budget::new(limits);
        sparse_least_squares_with_meter(&source, &[10.0], &options, &mut budget).unwrap();
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);
        assert!(sparse_least_squares_with_meter(&source, &[10.0], &options, &mut budget).is_err());
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);

        limits.dimensions[dimension.index()] -= 1;
        let mut short = Budget::new(limits);
        let error = sparse_least_squares_with_meter(&source, &[10.0], &options, &mut short)
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
