//! Sparse objective descent against independent formulas and refusal cases.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedObjective;
use fsym_core::{Expr, Symbol, parse};
use fsym_numeric_consumer::{
    MinimizeOptions, ObjectiveConsumerError, minimize, sparse_minimize, sparse_minimize_with_meter,
};
use std::error::Error;
use std::sync::Arc;

fn objective(source: &str, variables: &[&str]) -> VerifiedObjective {
    let axes = variables
        .iter()
        .map(|name| Symbol::new(*name))
        .collect::<Vec<_>>();
    VerifiedObjective::try_compile(&parse(source).unwrap(), &axes).unwrap()
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
fn sixty_four_variable_sparse_problem_exceeds_the_dense_contract() {
    let variables = (0..64)
        .map(|i| Symbol::new(format!("x{i}")))
        .collect::<Vec<_>>();
    // A flat source stays within the unchanged objective admission bound.
    let source = Expr::Add(
        variables
            .iter()
            .map(|axis| {
                Expr::Pow(
                    Arc::new(Expr::Sym(axis.clone())),
                    Arc::new(Expr::from_i64(2)),
                )
            })
            .collect(),
    );
    let program = VerifiedObjective::try_compile(&source, &variables).unwrap();
    assert_eq!(program.hessian_pattern().len(), 64);
    let initial = (0..64).map(|i| i as f64 - 31.0).collect::<Vec<_>>();
    assert!(matches!(
        minimize(&program, &initial, &MinimizeOptions::default()),
        Err(ObjectiveConsumerError::InvalidRequest(_))
    ));
    let result = sparse_minimize(&program, &initial, &MinimizeOptions::default()).unwrap();
    assert_eq!(result.point.len(), 64);
    for (&point, &gradient) in result.point.iter().zip(&result.gradient) {
        near(point, 0.0, 1e-10);
        near(gradient, 2.0 * point, 1e-12);
    }
    near(
        result.value,
        result.point.iter().map(|x| x * x).sum(),
        1e-20,
    );
    assert!(result.gradient_norm <= 1e-10);
    assert!(result.evaluations >= result.iterations + 2);
}

#[test]
fn coupled_sparse_quadratic_uses_both_mixed_partial_orientations() {
    let program = objective("3*(x-2)^2+2*(x-2)*(y+1)+4*(y+1)^2+7", &["x", "y"]);
    let result = sparse_minimize(&program, &[-4.0, 6.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 2.0, 1e-10);
    near(result.point[1], -1.0, 1e-10);
    near(result.value, 7.0, 1e-12);
    near(
        result.gradient[0],
        6.0 * (result.point[0] - 2.0) + 2.0 * (result.point[1] + 1.0),
        1e-12,
    );
    near(
        result.gradient[1],
        2.0 * (result.point[0] - 2.0) + 8.0 * (result.point[1] + 1.0),
        1e-12,
    );
}

#[test]
fn sparse_nonlinear_descent_and_curvature_regularization() {
    let rosenbrock = objective("(1-x)^2+100*(y-x^2)^2", &["x", "y"]);
    let result = sparse_minimize(&rosenbrock, &[-1.2, 1.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 1.0, 1e-8);
    near(result.point[1], 1.0, 1e-8);
    assert!(result.gradient_norm <= 1e-10);
    for (formula, start, expected) in [
        ("(x^2-1)^2", 0.1, 1.0),
        ("x^4-x", 0.0, 4.0_f64.cbrt().recip()),
    ] {
        let program = objective(formula, &["x"]);
        let result = sparse_minimize(&program, &[start], &MinimizeOptions::default()).unwrap();
        near(result.point[0], expected, 1e-8);
        assert!(result.regularizations > 0);
        assert!(result.gradient_norm <= 1e-10);
    }
}

#[test]
fn sparse_descent_retains_units_and_unused_coordinates() {
    let scaled = objective("(x/1000000+y-3)^2+(2*x/1000000-y)^2", &["x", "y"]);
    let result = sparse_minimize(&scaled, &[0.0, 0.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 1_000_000.0, 1e-4);
    near(result.point[1], 2.0, 1e-10);
    let unused = objective("(x-2)^2", &["x", "y"]);
    let result = sparse_minimize(&unused, &[0.0, 7.0], &MinimizeOptions::default()).unwrap();
    near(result.point[0], 2.0, 1e-10);
    assert_eq!(result.point[1], 7.0);
    assert_eq!(result.gradient[1], 0.0);
}

#[test]
fn domain_invalid_trials_backtrack_but_original_holes_never_disappear() {
    for (formula, start, expected) in [("x-log(x)", 10.0, 1.0), ("exp(x)-x", -8.0, 0.0)] {
        let program = objective(formula, &["x"]);
        let result = sparse_minimize(&program, &[start], &MinimizeOptions::default()).unwrap();
        near(result.point[0], expected, 1e-9);
        assert!(result.evaluations > result.iterations + 2);
    }
    let x = Symbol::new("x");
    let source = Expr::Add(vec![
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
        ]),
    ]);
    let program = VerifiedObjective::try_compile(&source, &[x]).unwrap();
    assert!(sparse_minimize(&program, &[0.0], &MinimizeOptions::default()).is_err());
    for formula in ["x^(1/2)", "x^(3/2)", "atan2(x,-1)"] {
        let program = objective(formula, &["x"]);
        assert!(sparse_minimize(&program, &[0.0], &MinimizeOptions::default()).is_err());
    }
}

#[test]
fn stationary_maxima_and_saddles_remain_proposals() {
    for formula in ["3-x^2-y^2", "x^2-y^2", "17"] {
        let program = objective(formula, &["x", "y"]);
        let result = sparse_minimize(&program, &[0.0, 0.0], &MinimizeOptions::default()).unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.gradient_norm, 0.0);
        assert_eq!(result.evaluations, 2);
    }
}

#[test]
fn invalid_options_and_iteration_limits_do_not_publish() {
    let program = objective("(x-2)^2", &["x"]);
    let options = MinimizeOptions {
        max_iterations: 0,
        ..MinimizeOptions::default()
    };
    assert!(matches!(
        sparse_minimize(&program, &[0.0], &options),
        Err(ObjectiveConsumerError::NoConvergence { iterations: 0, .. })
    ));
    for options in [
        MinimizeOptions {
            gradient_tolerance: f64::NAN,
            ..MinimizeOptions::default()
        },
        MinimizeOptions {
            max_step_norm: 0.0,
            ..MinimizeOptions::default()
        },
        MinimizeOptions {
            max_iterations: 1025,
            ..MinimizeOptions::default()
        },
    ] {
        assert!(matches!(
            sparse_minimize(&program, &[0.0], &options),
            Err(ObjectiveConsumerError::InvalidRequest(_))
        ));
    }
    assert!(sparse_minimize(&program, &[f64::INFINITY], &MinimizeOptions::default()).is_err());
    assert!(sparse_minimize(&program, &[], &MinimizeOptions::default()).is_err());
}

struct CancelAfter {
    remaining: u64,
    consumed: u64,
}

impl BudgetMeter for CancelAfter {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.remaining == 0 {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()?;
        self.consumed += charges.iter().map(|(_, n)| n).sum::<u64>();
        self.remaining -= 1;
        Ok(())
    }
}

#[test]
fn exhausted_and_cancelled_sparse_work_keeps_the_consumed_ledger() {
    let program = objective("(1-x)^2+100*(y-x^2)^2", &["x", "y"]);
    let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
    limits.dimensions[Dimension::ComputeSteps.index()] = 1000;
    let mut budget = Budget::new(limits);
    let error = sparse_minimize_with_meter(
        &program,
        &[-1.2, 1.0],
        &MinimizeOptions::default(),
        &mut budget,
    )
    .unwrap_err();
    assert!(matches!(
        meter_cause(&error),
        Some(MeterError::Budget(BudgetError::Exhausted {
            dimension: Dimension::ComputeSteps,
            ..
        }))
    ));
    assert!(budget.remaining(Dimension::ComputeSteps) < 1000);
    let before = budget.remaining(Dimension::ComputeSteps);
    assert!(
        sparse_minimize_with_meter(
            &program,
            &[-1.2, 1.0],
            &MinimizeOptions::default(),
            &mut budget
        )
        .is_err()
    );
    assert!(budget.remaining(Dimension::ComputeSteps) <= before);
    assert_eq!(budget.verifier_remaining(), 37);
    for remaining in [1, 20, 200] {
        let mut meter = CancelAfter {
            remaining,
            consumed: 0,
        };
        let result = sparse_minimize_with_meter(
            &program,
            &[-1.2, 1.0],
            &MinimizeOptions::default(),
            &mut meter,
        );
        assert_eq!(
            meter_cause(&result.unwrap_err()),
            Some(MeterError::Cancelled)
        );
        assert!(meter.consumed > 0);
    }
}
