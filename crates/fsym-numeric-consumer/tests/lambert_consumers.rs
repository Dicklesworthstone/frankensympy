//! Real-branch line searches must reject mathematical trial points without
//! swallowing bounded-kernel, budget or cancellation failures.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::{VerifiedObjective, VerifiedResidualSystem};
use fsym_core::{Expr, LambertWError, Symbol};
use fsym_numeric_consumer::{
    LeastSquaresOptions, MinimizeOptions, NewtonGuards, frankenscipy_analytic_system,
    least_squares, least_squares_with_meter, minimize, sparse_least_squares, sparse_minimize,
    sparse_newton_refine, sparse_newton_refine_with_meter,
};
use std::error::Error;

fn residual(branch: i32) -> Expr {
    let w = Expr::Function(
        "LambertW".into(),
        vec![Expr::symbol("x"), Expr::from_i64(branch.into())],
    );
    w + if branch == 0 {
        Expr::rational(1, 2).unwrap()
    } else {
        Expr::from_i64(5)
    }
}

fn system(branch: i32) -> VerifiedResidualSystem {
    VerifiedResidualSystem::try_compile(&[residual(branch)], &[Symbol::new("x")]).unwrap()
}

fn objective() -> VerifiedObjective {
    VerifiedObjective::try_compile(&residual(0).pow(Expr::from_i64(2)), &[Symbol::new("x")])
        .unwrap()
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected} by more than {tolerance}"
    );
}

fn cause<'a, E: Error + 'static>(error: &'a (dyn Error + 'static)) -> Option<&'a E> {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(found) = error.downcast_ref::<E>() {
            return Some(found);
        }
        current = error.source();
    }
    None
}

#[test]
fn both_real_branches_backtrack_to_roots_from_invalid_first_newton_trials() {
    for (branch, initial, root) in [
        (0, 1.0, -0.5 * (-0.5_f64).exp()),
        (-1, -0.3, -5.0 * (-5.0_f64).exp()),
    ] {
        let source = system(branch);
        let first = source.evaluate_sparse(&[initial]).unwrap();
        let trial = initial - first.residuals[0] / first.jacobian[0];
        let refusal = source.evaluate_sparse(&[trial]).unwrap_err();
        assert_eq!(
            cause::<LambertWError>(&refusal),
            Some(&LambertWError::OutsideRealDomain { branch })
        );

        let refined = sparse_newton_refine(&source, &[initial], &NewtonGuards::default()).unwrap();
        near(refined.root[0], root, 1e-10);
        assert!(refined.residual_norm <= 1e-12);
        assert!(refined.iterations > 0);

        for fit in [
            least_squares(&source, &[initial], &LeastSquaresOptions::default()).unwrap(),
            sparse_least_squares(&source, &[initial], &LeastSquaresOptions::default()).unwrap(),
        ] {
            near(fit.point[0], root, 1e-9);
            assert!(fit.cost < 1e-18);
            assert!(fit.gradient_norm <= 1e-10);
            assert!(fit.evaluations > fit.iterations + 2);
            assert_eq!(
                source.evaluate_sparse(&fit.point).unwrap().residuals,
                fit.residuals
            );
        }
    }
}

#[test]
fn dense_and_sparse_objectives_recover_after_lambert_domain_trials() {
    let source = objective();
    // At x=0 the value is 1/4, gradient is 1, and Hessian is zero.
    // Regularized Newton proposes a large negative step outside W0's domain.
    let origin = source.evaluate(&[0.0]).unwrap();
    assert_eq!(origin.value, 0.25);
    assert_eq!(origin.gradient, [1.0]);
    assert!(origin.hessian.iter().all(|value| *value == 0.0));
    for result in [
        minimize(&source, &[0.0], &MinimizeOptions::default()).unwrap(),
        sparse_minimize(&source, &[0.0], &MinimizeOptions::default()).unwrap(),
    ] {
        near(result.point[0], -0.5 * (-0.5_f64).exp(), 1e-9);
        assert!(result.value < 1e-18);
        assert!(result.gradient_norm <= 1e-10);
        assert!(result.evaluations > result.iterations + 2);
        let final_state = source.evaluate(&result.point).unwrap();
        assert_eq!(final_state.value, result.value);
        assert_eq!(final_state.gradient, result.gradient);
    }
}

#[test]
fn analytic_donor_preserves_domain_refusal_without_a_trial_error_channel() {
    // The pinned donor accepts only Vec<f64> residual callbacks. Its Armijo
    // driver falls back to a full step when interpolation sees an infinite
    // merit, so neither an infinite sentinel nor an invented finite residual
    // can represent a safely rejected domain trial. This explicit backend
    // retains the first typed failure instead of selecting another solver.
    for (branch, initial) in [(0, 1.0), (-1, -0.3)] {
        let error = frankenscipy_analytic_system(&system(branch), &[initial]).unwrap_err();
        assert_eq!(
            cause::<LambertWError>(&error),
            Some(&LambertWError::OutsideRealDomain { branch })
        );
    }
}

#[test]
fn initial_domain_failures_and_held_zero_domains_remain_terminal() {
    for (branch, point) in [(0, -0.5), (-1, 0.0)] {
        let source = system(branch);
        let error = sparse_newton_refine(&source, &[point], &NewtonGuards::default()).unwrap_err();
        assert_eq!(
            cause::<LambertWError>(&error),
            Some(&LambertWError::OutsideRealDomain { branch })
        );
        let error = least_squares(&source, &[point], &LeastSquaresOptions::default()).unwrap_err();
        assert_eq!(
            cause::<LambertWError>(&error),
            Some(&LambertWError::OutsideRealDomain { branch })
        );
        let held = Expr::Mul(vec![Expr::from_i64(0), residual(branch)]);
        let objective = VerifiedObjective::try_compile(&held, &[Symbol::new("x")]).unwrap();
        for error in [
            minimize(&objective, &[point], &MinimizeOptions::default()).unwrap_err(),
            sparse_minimize(&objective, &[point], &MinimizeOptions::default()).unwrap_err(),
        ] {
            assert_eq!(
                cause::<LambertWError>(&error),
                Some(&LambertWError::OutsideRealDomain { branch })
            );
        }
    }
}

#[derive(Default)]
struct Checkpoints {
    calls: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.calls += 1;
        if self.cancel_at.is_some_and(|at| self.calls >= at) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn backtracking_keeps_the_callers_budget_and_cancellation_terminal() {
    let source = system(0);
    let guards = NewtonGuards::default();
    let mut baseline = Checkpoints::default();
    sparse_newton_refine_with_meter(&source, &[1.0], &guards, &mut baseline).unwrap();
    for cancel_at in [1, baseline.calls / 2, baseline.calls] {
        let mut meter = Checkpoints {
            calls: 0,
            cancel_at: Some(cancel_at),
        };
        let error =
            sparse_newton_refine_with_meter(&source, &[1.0], &guards, &mut meter).unwrap_err();
        assert_eq!(cause::<MeterError>(&error), Some(&MeterError::Cancelled));
    }

    let allowance = u64::MAX;
    let mut measured = Budget::new(BudgetLimits::uniform(allowance, 0));
    least_squares_with_meter(
        &source,
        &[1.0],
        &LeastSquaresOptions::default(),
        &mut measured,
    )
    .unwrap();
    let used = allowance - measured.remaining(Dimension::ComputeSteps);
    let mut limits = BudgetLimits::uniform(allowance, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = used - 1;
    let mut meter = Budget::new(limits);
    let error =
        least_squares_with_meter(&source, &[1.0], &LeastSquaresOptions::default(), &mut meter)
            .unwrap_err();
    assert!(matches!(
        cause::<MeterError>(&error),
        Some(MeterError::Budget(_))
    ));
    assert!(meter.remaining(Dimension::ComputeSteps) < used / 2);
}
