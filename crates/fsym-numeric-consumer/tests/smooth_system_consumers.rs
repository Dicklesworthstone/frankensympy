//! Derivative-based consumers must retain the original source's smooth domain.
//! Finite scalar values and a zero residual do not establish differentiability.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_numeric_consumer::{
    NewtonGuards, SystemConsumerError, frankenscipy_analytic_system, newton_refine,
    newton_refine_with_meter, propose_root_box_with_meter, sparse_newton_refine,
};

fn angle(x: i64) -> Expr {
    Expr::Function("atan2".into(), vec![Expr::symbol("y"), Expr::from_i64(x)])
}

fn system(source: Expr) -> VerifiedResidualSystem {
    VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("y")]).unwrap()
}

fn nonsmooth(error: &SystemConsumerError) -> bool {
    matches!(
        error,
        SystemConsumerError::Evaluation(EvalError::NonsmoothAngle)
            | SystemConsumerError::SparseEvaluation(ResidualEvaluationError::Evaluation(
                EvalError::NonsmoothAngle
            ))
    )
}

#[test]
fn exact_initial_roots_on_the_angle_cut_refuse_in_all_derivative_consumers() {
    let source = system(angle(-1) - Expr::Const(Constant::Pi));
    for zero in [0.0_f64, -0.0] {
        // The scalar principal value is pi, and its formal derivative formula
        // is finite here. Nevertheless, neither side has a common smooth
        // neighborhood across this source cut.
        assert_eq!(
            source.residual_programs()[0]
                .try_eval_finite(&[zero])
                .unwrap(),
            0.0
        );
        for result in [
            newton_refine(&source, &[zero], &NewtonGuards::default()),
            frankenscipy_analytic_system(&source, &[zero]),
            sparse_newton_refine(&source, &[zero], &NewtonGuards::default()),
        ] {
            let error =
                result.expect_err("a cut value cannot authorize a derivative-based proposal");
            assert!(nonsmooth(&error), "{error:?}");
        }
    }
}

#[test]
fn a_held_zero_cannot_hide_a_cut_reached_after_a_valid_newton_step() {
    // F(y)=y has a unit formal Jacobian, but its retained held source contains
    // 0*atan2(y,-1). The first step from y=1 reaches its excluded cut at zero.
    let source = system(Expr::Add(vec![
        Expr::symbol("y"),
        Expr::Mul(vec![Expr::from_i64(0), angle(-1)]),
    ]));
    let initial = source.try_eval_dense(&[1.0]).unwrap();
    assert_eq!(initial.residuals, [1.0]);
    assert_eq!(initial.jacobian, [1.0]);
    let error = newton_refine(&source, &[1.0], &NewtonGuards::default())
        .expect_err("a later zero residual cannot discard the original smooth-domain guard");
    assert!(nonsmooth(&error), "{error:?}");
}

#[test]
fn root_box_proposals_refuse_a_finite_inverse_at_a_nonsmooth_source_point() {
    let source = system(angle(-1) - Expr::Const(Constant::Pi));
    let mut budget = Budget::new(BudgetLimits::uniform(10_000_000, 137));
    let radius = BigRational::new(1.into(), 10.into());
    let error = propose_root_box_with_meter(&source, &[0.0], &[radius], 24, &mut budget)
        .expect_err("a finite formal Jacobian inverse cannot bypass source-domain admission");
    assert!(nonsmooth(&error), "{error:?}");
    assert_eq!(budget.verifier_remaining(), 137);
}

#[test]
fn both_open_sides_and_the_positive_axis_remain_usable() {
    for (target, initial, expected) in [(3, 0.8, 1.0), (-3, -0.8, -1.0)] {
        let source = system(
            angle(-1)
                - Expr::Rational(BigRational::new(target.into(), 4.into()))
                    * Expr::Const(Constant::Pi),
        );
        for result in [
            newton_refine(&source, &[initial], &NewtonGuards::default()),
            frankenscipy_analytic_system(&source, &[initial]),
            sparse_newton_refine(&source, &[initial], &NewtonGuards::default()),
        ] {
            let proposal = result.unwrap();
            assert!((proposal.root[0] - expected).abs() < 1e-9);
            assert!(proposal.residual_norm <= 1e-10);
        }
    }
    let source = system(angle(1));
    let proposal = newton_refine(&source, &[0.0], &NewtonGuards::default()).unwrap();
    assert_eq!(proposal.root, [0.0]);
    assert_eq!(proposal.residual_norm, 0.0);
}

struct InterruptAt {
    budget: Budget,
    checkpoints: usize,
    stop: Option<usize>,
    stopped: bool,
}

impl InterruptAt {
    fn new(stop: Option<usize>) -> Self {
        Self {
            budget: Budget::new(BudgetLimits::uniform(10_000_000, 137)),
            checkpoints: 0,
            stop,
            stopped: false,
        }
    }
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "work continued after cancellation");
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "work continued after cancellation");
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "a failed checkpoint was ignored");
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_checked_evaluation_safe_point_preserves_typed_cancellation_and_the_verifier_pool() {
    let source = system(angle(1));
    let mut complete = InterruptAt::new(None);
    newton_refine_with_meter(&source, &[0.0], &NewtonGuards::default(), &mut complete).unwrap();
    assert!(complete.checkpoints > 10);
    for stop in 1..=complete.checkpoints {
        let mut interrupted = InterruptAt::new(Some(stop));
        let error =
            newton_refine_with_meter(&source, &[0.0], &NewtonGuards::default(), &mut interrupted)
                .expect_err("cancellation must not publish an approximate root");
        assert!(matches!(
            error,
            SystemConsumerError::Meter(MeterError::Cancelled)
                | SystemConsumerError::Evaluation(EvalError::Meter(MeterError::Cancelled))
        ));
        assert_eq!(interrupted.checkpoints, stop);
        assert_eq!(interrupted.budget.verifier_remaining(), 137);
    }
    let result = newton_refine(&source, &[0.0], &NewtonGuards::default()).unwrap();
    assert_eq!(result.root, [0.0]);
}
