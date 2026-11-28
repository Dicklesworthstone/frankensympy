//! Independently derived least-squares values, full curvature and source guards.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::{ObjectiveCompilationError, VerifiedResidualSystem};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxWitness, StationaryPointKind, Symbol, parse,
    verify_stationary_point,
};
use std::error::Error;
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn system(sources: &[&str], variables: &[&str]) -> VerifiedResidualSystem {
    let sources: Vec<_> = sources
        .iter()
        .map(|source| parse(source).unwrap())
        .collect();
    let variables: Vec<_> = variables.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedResidualSystem::try_compile(&sources, &variables).unwrap()
}

fn witness(center: i64, radius: BigRational, inverse: BigRational) -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(q(center, 1), radius).unwrap()],
        center: vec![q(center, 1)],
        preconditioner: vec![vec![inverse]],
        precision_digits: 24,
    }
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(1 << 40, 1 << 40))
}

#[test]
fn inconsistent_data_has_a_certified_minimum_without_a_root_claim() {
    // The two equations cannot both vanish. Their squared error is
    // (x-2)^2+1, with an exact unique strict minimum at x=2.
    let residuals = system(&["x-1", "x-3"], &["x"]);
    let objective = residuals.least_squares_objective().unwrap();
    let values = objective.evaluate(&[2.0]).unwrap();
    assert_eq!(values.value, 1.0);
    assert_eq!(values.gradient, [0.0]);
    assert_eq!(values.hessian, [2.0]);
    let receipt = verify_stationary_point(
        objective.objective(),
        objective.variables(),
        &witness(2, q(1, 4), q(1, 2)),
        &mut budget(),
    )
    .unwrap();
    assert_eq!(receipt.kind(), StationaryPointKind::StrictMinimum);
    assert!(receipt.stationary_enclosure()[0].contains(&q(2, 1)));
    assert!(receipt.objective_enclosure().contains(&q(1, 1)));
    assert_eq!(
        residuals.evaluate_sparse(&[2.0]).unwrap().residuals,
        [1.0, -1.0]
    );
}

#[test]
fn nonlinear_curvature_includes_the_residual_hessian_term() {
    let objective = system(&["x^2-1"], &["x"])
        .least_squares_objective()
        .unwrap();
    // F''=6*x^2-2, whereas J-transpose-J alone is 4*x^2.
    for (point, value, gradient, hessian) in [(0.0, 0.5, 0.0, -2.0), (2.0, 4.5, 12.0, 22.0)] {
        let result = objective.evaluate(&[point]).unwrap();
        assert_eq!(result.value, value);
        assert_eq!(result.gradient, [gradient]);
        assert_eq!(result.hessian, [hessian]);
    }
    // A stationary least-squares proposal can be a maximum. Preserve that
    // independently proved classification instead of calling every fit a min.
    let receipt = verify_stationary_point(
        objective.objective(),
        objective.variables(),
        &witness(0, q(1, 10), q(-1, 2)),
        &mut budget(),
    )
    .unwrap();
    assert_eq!(receipt.kind(), StationaryPointKind::StrictMaximum);
}

#[test]
fn mixed_curvature_and_products_follow_the_original_coordinate_order() {
    let system = system(&["x*y-1", "x+y-3"], &["y", "x"]);
    let objective = system.least_squares_objective().unwrap();
    assert_eq!(objective.variables(), system.vars());
    assert_eq!(
        objective.hessian_pattern(),
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    let values = objective.evaluate(&[1.0, 2.0]).unwrap();
    assert_eq!(values.value, 0.5);
    assert_eq!(values.gradient, [2.0, 1.0]);
    assert_eq!(values.hessian, [5.0, 4.0, 4.0, 2.0]);
    assert_eq!(
        objective
            .hessian_vector_product(&[1.0, 2.0], &[2.0, -1.0])
            .unwrap(),
        [6.0, 6.0]
    );
}

#[test]
fn duplicate_and_constant_rows_keep_their_error_weight_and_source_order() {
    let system = system(&["x-1", "x-1", "3"], &["x"]);
    let objective = system.least_squares_objective().unwrap();
    let values = objective.evaluate(&[2.0]).unwrap();
    assert_eq!(values.value, 5.5);
    assert_eq!(values.gradient, [2.0]);
    assert_eq!(values.hessian, [2.0]);
    let Expr::Mul(factors) = objective.objective() else {
        panic!("held sum source")
    };
    let Expr::Add(squares) = &factors[1] else {
        panic!("held ordered rows")
    };
    assert_eq!(squares.len(), 3);
    for (square, original) in squares.iter().zip(system.residuals()) {
        let Expr::Pow(base, _) = square else {
            panic!("squared residual")
        };
        assert_eq!(base.as_ref(), original);
    }
}

#[test]
fn original_poles_survive_zero_rows_and_zero_exponents() {
    let x = Symbol::new("x");
    let pole = Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-1)));
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
    ] {
        let system = VerifiedResidualSystem::try_compile(
            &[Expr::Sym(x.clone()), hidden],
            std::slice::from_ref(&x),
        )
        .unwrap();
        let objective = system.least_squares_objective().unwrap();
        assert!(objective.evaluate(&[1.0]).is_ok());
        assert!(objective.evaluate(&[0.0]).is_err());
        assert!(
            verify_stationary_point(
                objective.objective(),
                objective.variables(),
                &witness(0, q(1, 10), q(1, 1)),
                &mut budget(),
            )
            .is_err()
        );
    }
}

#[test]
fn independent_admission_rejects_omitted_reordered_and_rescaled_rows() {
    let receiver = system(&["x-1", "x-3"], &["x"]);
    for wrong in [
        system(&["x-1"], &["x"]),
        system(&["x-3", "x-1"], &["x"]),
        system(&["2*(x-1)", "x-3"], &["x"]),
    ] {
        let candidate = wrong
            .propose_least_squares_objective_with_meter(&mut Unbounded)
            .unwrap();
        assert!(matches!(
            receiver.verify_least_squares_objective_with_meter(candidate, &mut Unbounded),
            Err(ObjectiveCompilationError::SourceMismatch(
                "objective source"
            ))
        ));
    }
    let source = system(&["x*y-1"], &["x", "y"]);
    let reordered = system(&["x*y-1"], &["y", "x"]);
    let candidate = reordered
        .propose_least_squares_objective_with_meter(&mut Unbounded)
        .unwrap();
    assert!(matches!(
        source.verify_least_squares_objective_with_meter(candidate, &mut Unbounded),
        Err(ObjectiveCompilationError::SourceMismatch(
            "typed variable order"
        ))
    ));
}

#[test]
fn objective_bounds_apply_to_the_total_source_before_copying_rows() {
    let x = Symbol::new("x");
    let too_many = VerifiedResidualSystem::try_compile(
        &vec![Expr::Sym(x.clone()); 85],
        std::slice::from_ref(&x),
    )
    .unwrap();
    assert!(matches!(
        too_many.least_squares_objective(),
        Err(ObjectiveCompilationError::ResourceLimit { .. })
    ));
    let mut deep = Expr::Sym(x.clone());
    for _ in 0..14 {
        deep = Expr::Pow(Arc::new(deep), Arc::new(Expr::from_i64(1)));
    }
    let deep = VerifiedResidualSystem::try_compile(&[deep], &[x]).unwrap();
    assert!(matches!(
        deep.least_squares_objective(),
        Err(ObjectiveCompilationError::ResourceLimit {
            resource: "expression depth",
            ..
        })
    ));
}

#[test]
fn protected_verification_and_repeated_calls_keep_cumulative_costs() {
    let source = system(&["x-1", "x-3"], &["x"]);
    let mut ledger = budget();
    let protected = ledger.verifier_remaining();
    let candidate = source
        .propose_least_squares_objective_with_meter(&mut ledger)
        .unwrap();
    assert_eq!(ledger.verifier_remaining(), protected);
    let ordinary = ledger.remaining(Dimension::ComputeSteps);
    let lease = ledger.verifier_lease().unwrap();
    source
        .verify_least_squares_objective_with_meter(
            candidate,
            &mut ledger.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
    assert_eq!(ledger.remaining(Dimension::ComputeSteps), ordinary);
    assert!(ledger.verifier_remaining() < protected);

    let mut measured = budget();
    source
        .least_squares_objective_with_meter(&mut measured)
        .unwrap();
    let cost = (1 << 40) - measured.remaining(Dimension::ComputeSteps);
    let mut limits = BudgetLimits::uniform(1 << 40, 37);
    limits.dimensions[Dimension::ComputeSteps.index()] = cost;
    let mut exact = Budget::new(limits);
    source
        .least_squares_objective_with_meter(&mut exact)
        .unwrap();
    assert_eq!(exact.remaining(Dimension::ComputeSteps), 0);
    assert!(
        source
            .least_squares_objective_with_meter(&mut exact)
            .is_err()
    );
    assert_eq!(exact.verifier_remaining(), 37);
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    stop: Option<usize>,
}

impl BudgetMeter for Interrupt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.stop.is_some_and(|stop| self.checkpoints >= stop) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn cancelled(mut error: &(dyn Error + 'static)) -> bool {
    loop {
        if error.downcast_ref::<MeterError>() == Some(&MeterError::Cancelled) {
            return true;
        }
        match error.source() {
            Some(source) => error = source,
            None => return false,
        }
    }
}

#[test]
fn cancellation_during_composition_or_admission_returns_no_objective() {
    let source = system(&["x-1", "x-3"], &["x"]);
    let mut measured = Interrupt {
        budget: budget(),
        checkpoints: 0,
        stop: None,
    };
    source
        .least_squares_objective_with_meter(&mut measured)
        .unwrap();
    for stop in [1, measured.checkpoints / 2, measured.checkpoints] {
        let mut meter = Interrupt {
            budget: budget(),
            checkpoints: 0,
            stop: Some(stop),
        };
        let error = source
            .least_squares_objective_with_meter(&mut meter)
            .unwrap_err();
        assert!(cancelled(&error), "{error}");
    }
}

#[test]
fn exact_weights_bind_value_curvature_and_independent_admission() {
    let residuals = system(&["x^2-1", "x-3"], &["x"]);
    let weights = [q(3, 2), q(1, 2)];
    let objective = residuals
        .weighted_least_squares_objective_with_meter(&weights, &mut Unbounded)
        .unwrap();
    let eval = objective.evaluate(&[0.0]).unwrap();
    assert_eq!(
        (eval.value, eval.gradient, eval.hessian),
        (3.0, vec![-1.5], vec![-2.5])
    );
    let candidate = residuals
        .propose_squared_loss_with_meter(Some(&weights), &mut Unbounded)
        .unwrap();
    assert!(matches!(
        residuals.verify_squared_loss_with_meter(
            Some(&[q(1, 2), q(3, 2)]),
            candidate,
            &mut Unbounded
        ),
        Err(ObjectiveCompilationError::SourceMismatch(_))
    ));
    let ordinary = residuals.least_squares_objective().unwrap();
    let unit = residuals
        .weighted_least_squares_objective_with_meter(&[q(1, 1), q(1, 1)], &mut Unbounded)
        .unwrap();
    assert_eq!(ordinary.objective(), unit.objective());
}

#[test]
fn zero_weight_preserves_domains_and_invalid_weights_refuse() {
    let residuals = system(&["x", "1/x"], &["x"]);
    let loss = residuals
        .weighted_least_squares_objective_with_meter(&[q(2, 1), q(0, 1)], &mut Unbounded)
        .unwrap();
    assert!(loss.evaluate(&[0.0]).is_err());
    assert_eq!(loss.evaluate(&[2.0]).unwrap().value, 4.0);
    for weights in [vec![], vec![q(1, 1), q(-1, 2)]] {
        assert!(matches!(
            residuals.weighted_least_squares_objective_with_meter(&weights, &mut Unbounded),
            Err(ObjectiveCompilationError::InvalidWeights(_))
        ));
    }
}
