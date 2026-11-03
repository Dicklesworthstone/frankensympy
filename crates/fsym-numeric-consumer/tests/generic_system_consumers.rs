//! Actual pinned consumers, independent original-source receipts, and shared
//! resource/cancellation behavior across dimensions beyond the old 2x2 lane.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_core::{BigInt, BigRational, Expr, RootBoxError, Symbol};
use fsym_numeric_consumer::{
    MAX_CONSUMER_NEWTON_ITERATIONS, NewtonGuards, SystemConsumerError, certify_root,
    certify_root_with_meter, frankenscipy_fsolve_system, frankenscipy_fsolve_system_root_box,
    frankenscipy_fsolve_system_with_meter, newton_refine, newton_refine_root_box,
    newton_refine_root_box_with_meter, newton_refine_with_meter,
};

fn linear_system(n: usize) -> VerifiedResidualSystem {
    let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
    let residuals: Vec<_> = variables
        .iter()
        .enumerate()
        .map(|(i, variable)| {
            let target = i64::try_from(i + 1).unwrap();
            // Upper-triangular coupling makes row/column orientation observable.
            let mut residual =
                Expr::from_i64(2) * Expr::Sym(variable.clone()) - Expr::from_i64(2 * target);
            if i + 1 < n {
                residual =
                    residual + Expr::Sym(variables[i + 1].clone()) - Expr::from_i64(target + 1);
            }
            residual
        })
        .collect();
    VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap()
}

fn ordinary_budget() -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 137);
    limits.dimensions[Dimension::ComputeSteps.index()] = 10_000_000;
    limits.dimensions[Dimension::MemoryBytes.index()] = 256 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 100_000;
    Budget::new(limits)
}

fn is_cancelled(error: &SystemConsumerError) -> bool {
    matches!(
        error,
        SystemConsumerError::Meter(MeterError::Cancelled)
            | SystemConsumerError::Evaluation(EvalError::Meter(MeterError::Cancelled))
            | SystemConsumerError::Certification(RootBoxError::Meter(MeterError::Cancelled))
    )
}

fn exhausted_dimension(error: &SystemConsumerError) -> Option<Dimension> {
    let meter = match error {
        SystemConsumerError::Meter(meter)
        | SystemConsumerError::Evaluation(EvalError::Meter(meter))
        | SystemConsumerError::Certification(RootBoxError::Meter(meter)) => meter,
        _ => return None,
    };
    match meter {
        MeterError::Budget(BudgetError::Exhausted { dimension, .. }) => Some(*dimension),
        _ => None,
    }
}

#[test]
fn both_real_consumers_solve_one_three_and_sixteen_dimensions_with_ordinary_limits() {
    for n in [1, 3, 16] {
        let system = linear_system(n);
        let initial = vec![0.0; n];
        let mut newton_budget = ordinary_budget();
        let mut scipy_budget = ordinary_budget();
        let newton = newton_refine_with_meter(
            &system,
            &initial,
            &NewtonGuards::default(),
            &mut newton_budget,
        )
        .unwrap();
        let scipy =
            frankenscipy_fsolve_system_with_meter(&system, &initial, &mut scipy_budget).unwrap();
        for proposal in [&newton, &scipy] {
            assert_eq!(proposal.root.len(), n);
            assert!(proposal.residual_norm <= 1e-10);
            for (coordinate, value) in proposal.root.iter().enumerate() {
                let expected = f64::from(u32::try_from(coordinate + 1).unwrap());
                assert!((value - expected).abs() < 1e-8);
            }
        }
        for budget in [&newton_budget, &scipy_budget] {
            assert!(budget.remaining(Dimension::ComputeSteps) < 10_000_000);
            assert!(budget.remaining(Dimension::MemoryBytes) < 256 * 1024 * 1024);
            assert_eq!(budget.verifier_remaining(), 137);
        }
    }
}

fn mixed_three_dimensional_system() -> VerifiedResidualSystem {
    let variables = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let [x, y, z] = variables.clone().map(Expr::Sym);
    let residuals = [
        Expr::Function("exp".into(), vec![x.clone()]) + y.clone() - Expr::from_i64(2),
        x + Expr::Function("sin".into(), vec![y.clone()])
            - Expr::Function("sin".into(), vec![Expr::from_i64(1)]),
        z.clone() * z + y - Expr::from_i64(5),
    ];
    // The coordinate order differs from the equation order and the Jacobian
    // is nonsymmetric, so transposed inverse proposals cannot pass unnoticed.
    let reversed = [
        variables[2].clone(),
        variables[1].clone(),
        variables[0].clone(),
    ];
    VerifiedResidualSystem::try_compile(&residuals, &reversed).unwrap()
}

#[test]
fn transported_mixed_source_drives_both_consumers_and_replayable_three_dimensional_proofs() {
    let original = mixed_three_dimensional_system();
    let restored =
        VerifiedResidualSystem::from_json_bytes(&original.to_json_bytes().unwrap()).unwrap();
    let initial = [1.8, 0.9, -0.1];
    let radii = [1e-5; 3];
    let newton =
        newton_refine_root_box(&restored, &initial, &NewtonGuards::default(), &radii, 24).unwrap();
    let scipy = frankenscipy_fsolve_system_root_box(&restored, &initial, &radii, 24).unwrap();
    for outcome in [&newton, &scipy] {
        let receipt = outcome.certificate();
        assert_eq!(receipt.residuals(), original.residuals());
        assert_eq!(receipt.variables(), original.vars());
        assert_eq!(
            fsym_core::certify_root_box(
                receipt.residuals(),
                receipt.variables(),
                receipt.witness()
            )
            .unwrap(),
            *receipt
        );
        assert!(receipt.contraction_bound() < &BigRational::from_integer(BigInt::from(1)));
        for (i, expected) in [2_i64, 1, 0].into_iter().enumerate() {
            assert!(
                receipt.root_enclosure()[i]
                    .contains(&BigRational::from_integer(BigInt::from(expected)))
            );
            assert!(receipt.root_enclosure()[i].lower() > receipt.witness().domain[i].lower());
            assert!(receipt.root_enclosure()[i].upper() < receipt.witness().domain[i].upper());
        }
    }
    for (left, right) in newton.approximation().iter().zip(scipy.approximation()) {
        assert!((left - right).abs() < 1e-7);
    }
}

#[test]
fn maximum_dimension_has_a_source_bound_root_receipt() {
    let system = linear_system(16);
    let root: Vec<_> = (1_u32..=16).map(f64::from).collect();
    let outcome = certify_root(&system, &root, &[1e-5; 16], 16).unwrap();
    assert_eq!(outcome.approximation(), root);
    assert_eq!(outcome.certificate().variables(), system.vars());
    assert_eq!(outcome.certificate().root_enclosure().len(), 16);
    assert_eq!(
        outcome.certificate().contraction_bound(),
        &BigRational::from_integer(BigInt::from(0))
    );
}

#[test]
fn a_small_residual_does_not_certify_a_nonexistent_or_singular_root() {
    let x = Symbol::new("x");
    let no_root = VerifiedResidualSystem::try_compile(
        &[Expr::Function("exp".into(), vec![Expr::Sym(x.clone())])],
        std::slice::from_ref(&x),
    )
    .unwrap();
    let proposal = newton_refine(&no_root, &[-40.0], &NewtonGuards::default()).unwrap();
    assert_eq!(proposal.iterations, 0);
    assert!(matches!(
        newton_refine_root_box(&no_root, &[-40.0], &NewtonGuards::default(), &[1e-5], 24),
        Err(SystemConsumerError::Certification(_))
    ));
    let singular =
        VerifiedResidualSystem::try_compile(&[Expr::Sym(x.clone()).pow(Expr::from_i64(2))], &[x])
            .unwrap();
    assert_eq!(
        newton_refine(&singular, &[0.0], &NewtonGuards::default())
            .unwrap()
            .iterations,
        0
    );
    assert!(matches!(
        certify_root(&singular, &[0.0], &[0.1], 24),
        Err(SystemConsumerError::LinearSolve(_))
    ));
}

#[test]
fn original_domain_checks_survive_zero_residuals_and_whole_box_verification() {
    let x = Symbol::new("x");
    let logarithm = Expr::Function("log".into(), vec![Expr::Sym(x.clone())]);
    let held_zero = Expr::Mul(vec![Expr::from_i64(0), logarithm.clone()]);
    let held_power = Expr::Add(vec![
        Expr::Pow(
            std::sync::Arc::new(logarithm.clone()),
            std::sync::Arc::new(Expr::from_i64(0)),
        ),
        Expr::from_i64(-1),
    ]);
    let reciprocal = Expr::Pow(
        std::sync::Arc::new(logarithm.clone()),
        std::sync::Arc::new(Expr::from_i64(-1)),
    );
    // 1/log(0) would misleadingly become finite -0.0 in the legacy evaluator.
    for residual in [held_zero, held_power, reciprocal] {
        let system =
            VerifiedResidualSystem::try_compile(&[residual], std::slice::from_ref(&x)).unwrap();
        assert!(matches!(
            newton_refine(&system, &[0.0], &NewtonGuards::default()),
            Err(SystemConsumerError::Evaluation(EvalError::NonFiniteValue))
        ));
        assert!(matches!(
            frankenscipy_fsolve_system(&system, &[0.0]),
            Err(SystemConsumerError::Evaluation(EvalError::NonFiniteValue))
        ));
    }
    let logarithm = VerifiedResidualSystem::try_compile(&[logarithm], &[x]).unwrap();
    assert!(matches!(
        certify_root(&logarithm, &[1.0], &[2.0], 24),
        Err(SystemConsumerError::Certification(RootBoxError::Domain(_)))
    ));
    certify_root(&logarithm, &[1.0], &[0.01], 24).unwrap();
}

#[test]
fn dimensions_guards_and_box_requests_are_bounded_before_donor_execution() {
    let too_large = linear_system(17);
    assert!(matches!(
        newton_refine(&too_large, &[0.0; 17], &NewtonGuards::default()),
        Err(SystemConsumerError::UnsupportedShape { .. })
    ));
    let x = Symbol::new("x");
    let rectangular =
        VerifiedResidualSystem::try_compile(&[Expr::Sym(x.clone()), Expr::Sym(x.clone())], &[x])
            .unwrap();
    assert!(matches!(
        frankenscipy_fsolve_system(&rectangular, &[0.0]),
        Err(SystemConsumerError::UnsupportedShape { .. })
    ));
    let system = linear_system(3);
    for point in [vec![], vec![0.0; 2], vec![0.0; 4]] {
        assert!(matches!(
            newton_refine(&system, &point, &NewtonGuards::default()),
            Err(SystemConsumerError::Dimension { .. })
        ));
    }
    let guards = NewtonGuards {
        max_iterations: MAX_CONSUMER_NEWTON_ITERATIONS + 1,
        ..NewtonGuards::default()
    };
    assert!(matches!(
        newton_refine(&system, &[1.0, 2.0, 3.0], &guards),
        Err(SystemConsumerError::InvalidRequest(_))
    ));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            frankenscipy_fsolve_system(&system, &[value, 0.0, 0.0]),
            Err(SystemConsumerError::NonFinite(_))
        ));
    }
    for radii in [vec![], vec![0.1; 2], vec![0.1; 4]] {
        assert!(matches!(
            certify_root(&system, &[1.0, 2.0, 3.0], &radii, 24),
            Err(SystemConsumerError::Dimension { .. })
        ));
    }
    for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(frankenscipy_fsolve_system_root_box(&system, &[0.0; 3], &[radius; 3], 24).is_err());
    }
    for precision in [0, 7, 129, u32::MAX] {
        assert!(matches!(
            newton_refine_root_box(
                &system,
                &[0.0; 3],
                &NewtonGuards::default(),
                &[0.1; 3],
                precision
            ),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
}

#[test]
fn combined_root_search_keeps_one_ledger_and_cannot_reset_it_for_verification() {
    let system = linear_system(3);
    let mut budget = Budget::new(BudgetLimits::uniform(1_000_000_000, 251));
    let identity = budget.ledger_identity();
    newton_refine_root_box_with_meter(
        &system,
        &[0.0; 3],
        &NewtonGuards::default(),
        &[1e-5; 3],
        24,
        &mut budget,
    )
    .unwrap();
    assert!(budget.remaining(Dimension::MemoryBytes) < 1_000_000_000 - 64 * 1024 * 1024);
    assert!(budget.remaining(Dimension::ComputeSteps) < 1_000_000_000);
    assert_eq!(budget.ledger_identity(), identity);
    assert_eq!(budget.verifier_remaining(), 251);

    // This can afford proposal work but cannot admit the independent checker's
    // fixed 64 MiB reservation. Resetting the ledger at that boundary is wrong.
    let mut limits = BudgetLimits::uniform(1_000_000_000, 251);
    limits.dimensions[Dimension::MemoryBytes.index()] = 64 * 1024 * 1024 - 1;
    let mut exhausted = Budget::new(limits);
    let result = newton_refine_root_box_with_meter(
        &system,
        &[0.0; 3],
        &NewtonGuards::default(),
        &[1e-5; 3],
        24,
        &mut exhausted,
    );
    let error = result.unwrap_err();
    assert_eq!(exhausted_dimension(&error), Some(Dimension::MemoryBytes));
    assert!(matches!(
        error,
        SystemConsumerError::Certification(RootBoxError::Meter(_))
    ));
    assert!(exhausted.remaining(Dimension::MemoryBytes) < 64 * 1024 * 1024 - 1);
    assert_eq!(exhausted.verifier_remaining(), 251);
}

#[test]
fn all_three_exhaustion_dimensions_survive_scipy_callback_translation_and_reuse() {
    let system = linear_system(1);
    let mut measured = Budget::new(BudgetLimits::uniform(1_000_000_000, 53));
    frankenscipy_fsolve_system_with_meter(&system, &[0.0], &mut measured).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let cost = 1_000_000_000 - measured.remaining(dimension);
        let mut limits = BudgetLimits::uniform(1_000_000_000, 53);
        limits.dimensions[dimension.index()] = cost;
        let mut budget = Budget::new(limits);
        frankenscipy_fsolve_system_with_meter(&system, &[0.0], &mut budget).unwrap();
        assert_eq!(budget.remaining(dimension), 0);
        let error =
            frankenscipy_fsolve_system_with_meter(&system, &[0.0], &mut budget).unwrap_err();
        assert_eq!(exhausted_dimension(&error), Some(dimension));
        assert_eq!(budget.verifier_remaining(), 53);

        // A limit halfway through the observed cost reaches donor callbacks
        // rather than just entry admission. The original typed refusal survives.
        limits.dimensions[dimension.index()] = cost / 2;
        let mut interrupted = Budget::new(limits);
        let error =
            frankenscipy_fsolve_system_with_meter(&system, &[0.0], &mut interrupted).unwrap_err();
        assert_eq!(exhausted_dimension(&error), Some(dimension));
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
fn every_numerical_safe_point_cancels_without_publishing_a_proposal() {
    let system = linear_system(1);
    for scipy in [false, true] {
        let run = |meter: &mut CancelAt| {
            if scipy {
                frankenscipy_fsolve_system_with_meter(&system, &[0.0], meter)
            } else {
                newton_refine_with_meter(&system, &[0.0], &NewtonGuards::default(), meter)
            }
        };
        let mut measured = CancelAt::default();
        run(&mut measured).unwrap();
        assert!(measured.checkpoints > 10);
        for stop in 1..=measured.checkpoints {
            let mut meter = CancelAt {
                stop: Some(stop),
                ..CancelAt::default()
            };
            let error = run(&mut meter).unwrap_err();
            assert!(is_cancelled(&error), "checkpoint {stop}: {error:?}");
            assert_eq!(meter.checkpoints, stop);
        }
    }
}

#[test]
fn verification_and_final_publication_observe_the_same_cancellation_token() {
    let system = linear_system(1);
    let mut measured = CancelAt::default();
    certify_root_with_meter(&system, &[1.0], &[0.01], 24, &mut measured).unwrap();
    for stop in [
        1,
        2,
        measured.checkpoints / 2,
        measured.checkpoints - 1,
        measured.checkpoints,
    ] {
        let mut meter = CancelAt {
            stop: Some(stop),
            ..CancelAt::default()
        };
        let error = certify_root_with_meter(&system, &[1.0], &[0.01], 24, &mut meter).unwrap_err();
        assert!(is_cancelled(&error));
        assert_eq!(meter.checkpoints, stop);
    }
}
