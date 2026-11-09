//! The pinned nonlinear driver consumes checked sparse derivative values.
//! These tests exercise real donor calls and independently checked root boxes.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_core::{BigInt, BigRational, Expr, RootBoxError, Symbol, parse};
use fsym_numeric_consumer::{
    SystemConsumerError, frankenscipy_analytic_system, frankenscipy_analytic_system_root_box,
    frankenscipy_analytic_system_root_box_with_meter, frankenscipy_analytic_system_with_meter,
};

fn linear_system(n: usize) -> VerifiedResidualSystem {
    let variables: Vec<_> = (0..n)
        .map(|index| Symbol::new(format!("analytic_{index}")))
        .collect();
    let residuals: Vec<_> = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            let target = i64::try_from(index + 1).unwrap();
            let mut residual =
                Expr::from_i64(2) * Expr::Sym(variable.clone()) - Expr::from_i64(2 * target);
            if index + 1 < n {
                residual =
                    residual + Expr::Sym(variables[index + 1].clone()) - Expr::from_i64(target + 1);
            }
            residual
        })
        .collect();
    VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap()
}

fn reordered_system() -> VerifiedResidualSystem {
    let variables = [Symbol::new("z"), Symbol::new("x"), Symbol::new("y")];
    let residuals = [
        parse("x+y-3").unwrap(),
        parse("2*y+z-7").unwrap(),
        parse("3*x-z").unwrap(),
    ];
    VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap()
}

fn ordinary_budget() -> Budget {
    Budget::new(BudgetLimits::uniform(1_000_000_000, 137))
}

fn meter_error(error: &SystemConsumerError) -> Option<&MeterError> {
    match error {
        SystemConsumerError::Meter(error)
        | SystemConsumerError::Evaluation(EvalError::Meter(error))
        | SystemConsumerError::SparseEvaluation(ResidualEvaluationError::Evaluation(
            EvalError::Meter(error),
        ))
        | SystemConsumerError::Certification(RootBoxError::Meter(error)) => Some(error),
        _ => None,
    }
}

fn exhausted_dimension(error: &SystemConsumerError) -> Option<Dimension> {
    match meter_error(error) {
        Some(MeterError::Budget(BudgetError::Exhausted { dimension, .. })) => Some(*dimension),
        _ => None,
    }
}

#[test]
fn analytic_donor_solves_one_three_and_sixteen_dimensions_with_shared_limits() {
    for n in [1, 3, 16] {
        let system = linear_system(n);
        let mut budget = ordinary_budget();
        let proposal =
            frankenscipy_analytic_system_with_meter(&system, &vec![0.0; n], &mut budget).unwrap();
        assert_eq!(proposal.root.len(), n);
        assert!(proposal.residual_norm <= 1e-10);
        assert!(proposal.iterations > 0 && proposal.iterations <= 200);
        for (index, value) in proposal.root.iter().enumerate() {
            let expected = f64::from(u32::try_from(index + 1).unwrap());
            assert!((value - expected).abs() < 1e-10);
        }
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            assert!(budget.remaining(dimension) < 1_000_000_000);
        }
        assert_eq!(budget.verifier_remaining(), 137);
    }
}

#[test]
fn nonlinear_and_transported_nonsymmetric_sources_keep_variable_order() {
    let x = Symbol::new("x");
    let nonlinear =
        VerifiedResidualSystem::try_compile(&[parse("x^2-2").unwrap()], std::slice::from_ref(&x))
            .unwrap();
    let result = frankenscipy_analytic_system(&nonlinear, &[1.0]).unwrap();
    assert!((result.root[0] - 2.0_f64.sqrt()).abs() < 1e-10);
    assert!(result.residual_norm <= 1e-10);

    let original = reordered_system();
    let restored =
        VerifiedResidualSystem::from_json_bytes(&original.to_json_bytes().unwrap()).unwrap();
    assert_eq!(
        restored.jacobian_pattern(),
        &[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    );
    let result = frankenscipy_analytic_system(&restored, &[0.0; 3]).unwrap();
    for (actual, expected) in result.root.iter().zip([3.0, 1.0, 2.0]) {
        assert!((actual - expected).abs() < 1e-10);
    }
    assert!(result.residual_norm <= 1e-10);
    assert_eq!(restored.residuals(), original.residuals());
    assert_eq!(restored.vars(), original.vars());
}

#[test]
fn derivative_domain_failure_wins_even_at_an_exact_initial_root() {
    let x = Symbol::new("x");
    let radical = parse("x^(1/2)").unwrap();
    for source in [radical.clone(), radical - Expr::from_i64(1)] {
        let system =
            VerifiedResidualSystem::try_compile(&[source], std::slice::from_ref(&x)).unwrap();
        // Both original residuals are finite at zero. The first is exactly
        // zero, but the pinned driver still sets up its analytic Jacobian.
        assert!(
            system.residual_programs()[0]
                .try_eval_finite(&[0.0])
                .is_ok()
        );
        assert!(matches!(
            frankenscipy_analytic_system(&system, &[0.0]),
            Err(SystemConsumerError::SparseEvaluation(
                ResidualEvaluationError::Evaluation(EvalError::NonFiniteValue)
            ))
        ));
    }
}

#[test]
fn finite_zero_jacobian_at_a_root_is_only_an_approximate_proposal() {
    let x = Symbol::new("x");
    let system = VerifiedResidualSystem::try_compile(&[parse("x^2").unwrap()], &[x]).unwrap();
    let result = frankenscipy_analytic_system(&system, &[0.0]).unwrap();
    assert_eq!(result.root, [0.0]);
    assert_eq!(result.residual_norm, 0.0);
    assert_eq!(result.iterations, 0);
    assert!(matches!(
        frankenscipy_analytic_system_root_box(&system, &[0.0], &[0.1], 24),
        Err(SystemConsumerError::LinearSolve(_))
    ));
}

#[test]
fn held_original_domains_cannot_be_hidden_by_zero_merit_or_empty_patterns() {
    let x = Symbol::new("x");
    let logarithm = Expr::Function("log".into(), vec![Expr::Sym(x.clone())]);
    let held = Expr::Mul(vec![Expr::from_i64(0), logarithm.clone()]);
    let power = Expr::Add(vec![
        Expr::Pow(
            std::sync::Arc::new(logarithm),
            std::sync::Arc::new(Expr::from_i64(0)),
        ),
        Expr::from_i64(-1),
    ]);
    for source in [held, power] {
        let system =
            VerifiedResidualSystem::try_compile(&[source], std::slice::from_ref(&x)).unwrap();
        for point in [0.0, -1.0] {
            assert!(matches!(
                frankenscipy_analytic_system(&system, &[point]),
                Err(SystemConsumerError::Evaluation(EvalError::NonFiniteValue))
            ));
        }
    }
    let constant_hole = Expr::Function("log".into(), vec![Expr::from_i64(-1)]);
    let system = VerifiedResidualSystem::try_compile(&[constant_hole], &[x]).unwrap();
    assert!(system.jacobian_pattern().is_empty());
    assert!(matches!(
        frankenscipy_analytic_system(&system, &[1.0]),
        Err(SystemConsumerError::Evaluation(EvalError::NonFiniteValue))
    ));
}

#[test]
fn finite_residual_components_do_not_authorize_overflowing_armijo_cost() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let multiply =
        VerifiedResidualSystem::try_compile(&[parse("10^308").unwrap()], &variables[..1]).unwrap();
    let accumulate = VerifiedResidualSystem::try_compile(
        &[parse("10^154").unwrap(), parse("10^154").unwrap()],
        &variables,
    )
    .unwrap();
    for (system, point) in [(&multiply, vec![0.0]), (&accumulate, vec![0.0; 2])] {
        assert!(matches!(
            frankenscipy_analytic_system(system, &point),
            Err(SystemConsumerError::NonFinite(
                "analytic residual squared cost"
            ))
        ));
    }
}

#[test]
fn linear_solver_failure_is_not_replaced_by_empty_sentinel_success() {
    let x = Symbol::new("x");
    let system = VerifiedResidualSystem::try_compile(&[Expr::from_i64(1)], &[x]).unwrap();
    assert!(system.jacobian_pattern().is_empty());
    assert!(matches!(
        frankenscipy_analytic_system(&system, &[0.0]),
        Err(SystemConsumerError::LinearSolve(_))
    ));
}

#[test]
fn shape_point_and_certificate_admission_remain_bounded() {
    let too_large = linear_system(17);
    assert!(matches!(
        frankenscipy_analytic_system(&too_large, &[0.0; 17]),
        Err(SystemConsumerError::UnsupportedShape { .. })
    ));
    let empty = VerifiedResidualSystem::try_compile(&[], &[]).unwrap();
    assert!(matches!(
        frankenscipy_analytic_system(&empty, &[]),
        Err(SystemConsumerError::UnsupportedShape { .. })
    ));
    let x = Symbol::new("x");
    let rectangular =
        VerifiedResidualSystem::try_compile(&[Expr::Sym(x.clone()), Expr::Sym(x.clone())], &[x])
            .unwrap();
    assert!(matches!(
        frankenscipy_analytic_system(&rectangular, &[0.0]),
        Err(SystemConsumerError::UnsupportedShape { .. })
    ));
    let system = linear_system(3);
    for point in [vec![], vec![0.0; 2], vec![0.0; 4]] {
        assert!(matches!(
            frankenscipy_analytic_system(&system, &point),
            Err(SystemConsumerError::Dimension { .. })
        ));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            frankenscipy_analytic_system(&system, &[value, 0.0, 0.0]),
            Err(SystemConsumerError::NonFinite(_))
        ));
    }
    for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            frankenscipy_analytic_system_root_box(&system, &[0.0; 3], &[radius; 3], 24).is_err()
        );
    }
    for precision in [0, 7, 129, u32::MAX] {
        assert!(matches!(
            frankenscipy_analytic_system_root_box(&system, &[0.0; 3], &[0.01; 3], precision),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
}

#[derive(Default)]
struct CancelAt {
    checkpoints: usize,
    stop: Option<usize>,
    cancelled: bool,
}

impl BudgetMeter for CancelAt {
    fn charge(&mut self, _dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.cancelled, "work continued after cancellation");
        assert_ne!(amount, 0);
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.cancelled, "work continued after cancellation");
        assert!(charges.iter().all(|(_, amount)| *amount != 0));
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.cancelled, "a failed checkpoint was ignored");
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.cancelled = true;
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

#[test]
fn every_safe_point_preserves_cancellation_through_the_infallible_donor_trait() {
    let system = linear_system(1);
    let mut measured = CancelAt::default();
    frankenscipy_analytic_system_with_meter(&system, &[0.0], &mut measured).unwrap();
    assert!(measured.checkpoints > 20);
    let mut sparse_stops = 0;
    for stop in 1..=measured.checkpoints {
        let mut meter = CancelAt {
            stop: Some(stop),
            ..CancelAt::default()
        };
        let error =
            frankenscipy_analytic_system_with_meter(&system, &[0.0], &mut meter).unwrap_err();
        assert!(
            matches!(meter_error(&error), Some(MeterError::Cancelled)),
            "checkpoint {stop}: {error:?}"
        );
        if matches!(error, SystemConsumerError::SparseEvaluation(_)) {
            sparse_stops += 1;
        }
        assert_eq!(meter.checkpoints, stop);
    }
    assert!(
        sparse_stops > 0,
        "the sweep must enter checked sparse capture"
    );
}

#[test]
fn all_three_budget_dimensions_are_cumulative_and_keep_the_verifier_reserve() {
    let system = linear_system(1);
    let mut measured = ordinary_budget();
    frankenscipy_analytic_system_with_meter(&system, &[0.0], &mut measured).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let cost = 1_000_000_000 - measured.remaining(dimension);
        assert!(cost > 0);
        let mut limits = BudgetLimits::uniform(1_000_000_000, 137);
        limits.dimensions[dimension.index()] = cost;
        let mut shared = Budget::new(limits);
        frankenscipy_analytic_system_with_meter(&system, &[0.0], &mut shared).unwrap();
        assert_eq!(shared.remaining(dimension), 0);
        let error =
            frankenscipy_analytic_system_with_meter(&system, &[0.0], &mut shared).unwrap_err();
        assert_eq!(exhausted_dimension(&error), Some(dimension));
        assert_eq!(shared.verifier_remaining(), 137);

        limits.dimensions[dimension.index()] = cost / 2;
        let mut stopped = Budget::new(limits);
        let error =
            frankenscipy_analytic_system_with_meter(&system, &[0.0], &mut stopped).unwrap_err();
        assert_eq!(exhausted_dimension(&error), Some(dimension));
        assert_eq!(stopped.verifier_remaining(), 137);
    }
}

#[test]
fn analytic_candidate_receipt_replays_against_the_original_nonsymmetric_source() {
    let original = reordered_system();
    let system =
        VerifiedResidualSystem::from_json_bytes(&original.to_json_bytes().unwrap()).unwrap();
    let outcome =
        frankenscipy_analytic_system_root_box(&system, &[0.0; 3], &[1e-5; 3], 24).unwrap();
    for (actual, expected) in outcome.approximation().iter().zip([3.0, 1.0, 2.0]) {
        assert!((actual - expected).abs() < 1e-10);
    }
    let receipt = outcome.certificate();
    assert_eq!(receipt.residuals(), original.residuals());
    assert_eq!(receipt.variables(), original.vars());
    assert_eq!(
        fsym_core::certify_root_box(receipt.residuals(), receipt.variables(), receipt.witness())
            .unwrap(),
        *receipt
    );
    for (index, value) in [3_i64, 1, 2].into_iter().enumerate() {
        let exact = BigRational::from_integer(BigInt::from(value));
        assert!(receipt.root_enclosure()[index].contains(&exact));
    }
}

#[test]
fn a_small_residual_still_cannot_certify_a_nonexistent_root_or_invalid_box() {
    let x = Symbol::new("x");
    let exponential =
        VerifiedResidualSystem::try_compile(&[parse("exp(x)").unwrap()], std::slice::from_ref(&x))
            .unwrap();
    let proposal = frankenscipy_analytic_system(&exponential, &[-40.0]).unwrap();
    assert!(proposal.residual_norm <= 1e-10);
    assert!(matches!(
        frankenscipy_analytic_system_root_box(&exponential, &[-40.0], &[1e-5], 24),
        Err(SystemConsumerError::Certification(_))
    ));
    let logarithm = VerifiedResidualSystem::try_compile(&[parse("log(x)").unwrap()], &[x]).unwrap();
    assert!(matches!(
        frankenscipy_analytic_system_root_box(&logarithm, &[1.0], &[2.0], 24),
        Err(SystemConsumerError::Certification(RootBoxError::Domain(_)))
    ));
}

#[test]
fn candidate_and_original_source_verification_share_one_budget_and_cancellation_token() {
    let system = linear_system(1);
    let mut budget = ordinary_budget();
    let identity = budget.ledger_identity();
    frankenscipy_analytic_system_root_box_with_meter(&system, &[0.0], &[0.01], 24, &mut budget)
        .unwrap();
    assert_eq!(budget.ledger_identity(), identity);
    assert_eq!(budget.verifier_remaining(), 137);
    assert!(budget.remaining(Dimension::MemoryBytes) < 1_000_000_000 - 64 * 1024 * 1024);

    let mut limits = BudgetLimits::uniform(1_000_000_000, 137);
    limits.dimensions[Dimension::MemoryBytes.index()] = 64 * 1024 * 1024 - 1;
    let mut exhausted = Budget::new(limits);
    let error = frankenscipy_analytic_system_root_box_with_meter(
        &system,
        &[0.0],
        &[0.01],
        24,
        &mut exhausted,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SystemConsumerError::Certification(RootBoxError::Meter(_))
    ));
    assert_eq!(exhausted_dimension(&error), Some(Dimension::MemoryBytes));
    assert_eq!(exhausted.verifier_remaining(), 137);

    let mut measured = CancelAt::default();
    frankenscipy_analytic_system_root_box_with_meter(&system, &[0.0], &[0.01], 24, &mut measured)
        .unwrap();
    for stop in [
        1,
        measured.checkpoints / 2,
        measured.checkpoints - 1,
        measured.checkpoints,
    ] {
        let mut meter = CancelAt {
            stop: Some(stop),
            ..CancelAt::default()
        };
        let error = frankenscipy_analytic_system_root_box_with_meter(
            &system,
            &[0.0],
            &[0.01],
            24,
            &mut meter,
        )
        .unwrap_err();
        assert!(matches!(meter_error(&error), Some(MeterError::Cancelled)));
        assert_eq!(meter.checkpoints, stop);
    }
}
