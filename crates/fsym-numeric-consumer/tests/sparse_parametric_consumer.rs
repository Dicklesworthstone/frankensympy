//! Fixed inputs stay outside rectangular sparse solves without disappearing
//! from source evaluation, derivative-domain checks, or resource accounting.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{Expr, Symbol, SymbolIdentity, parse_raw};
use fsym_numeric_consumer::{
    LeastSquaresOptions, NewtonGuards, SystemConsumerError, sparse_least_squares_parametric,
    sparse_least_squares_parametric_with_meter, sparse_newton_refine_parametric,
    sparse_newton_refine_parametric_with_meter,
};
use std::error::Error;

const AMPLE: u64 = 1 << 40;

fn system(sources: &[&str], axes: &[&str]) -> VerifiedResidualSystem {
    let sources: Vec<_> = sources
        .iter()
        .map(|value| parse_raw(value).unwrap())
        .collect();
    let axes: Vec<_> = axes.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedResidualSystem::try_compile(&sources, &axes).unwrap()
}

fn near(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(AMPLE, 73))
}

fn meter_cause(error: &(dyn Error + 'static)) -> Option<MeterError> {
    let mut cause = Some(error);
    while let Some(current) = cause {
        if let Some(meter) = current.downcast_ref::<MeterError>() {
            return Some(*meter);
        }
        cause = current.source();
    }
    None
}

#[test]
fn rectangular_roots_reuse_the_original_program_at_distinct_parameters() {
    let source = system(&["x^2-p", "2*x^2-2*p"], &["x", "p"]);
    let encoded = source.to_json_bytes().unwrap();
    for parameter in [2.0_f64, 9.0] {
        let parameters = [parameter];
        let result =
            sparse_newton_refine_parametric(&source, &[1.0], &parameters, &NewtonGuards::default())
                .unwrap();
        near(result.root[0], parameter.sqrt());
        let values = source
            .evaluate_sparse(&[result.root[0], parameter])
            .unwrap();
        assert!(values.residuals.iter().all(|value| value.abs() <= 1e-10));
        assert_eq!(parameters[0].to_bits(), parameter.to_bits());
    }
    assert_eq!(source.to_json_bytes().unwrap(), encoded);
}

#[test]
fn underdetermined_root_preserves_unused_unknowns_and_parameter_suffix() {
    let source = system(&["2*x+y-p-q"], &["x", "y", "unused", "q", "p"]);
    let result = sparse_newton_refine_parametric(
        &source,
        &[0.0, 0.0, 17.0],
        &[2.0, 3.0],
        &NewtonGuards::default(),
    )
    .unwrap();
    near(2.0 * result.root[0] + result.root[1], 5.0);
    assert_eq!(result.root[2], 17.0);
    assert_eq!(result.root.len(), 3);
}

#[test]
fn least_squares_keeps_inconsistent_rows_and_optimizes_unknowns_only() {
    let original = system(&["x-p", "2*x-q", "p"], &["x", "p", "q"]);
    let source =
        VerifiedResidualSystem::from_json_bytes(&original.to_json_bytes().unwrap()).unwrap();
    let parameters = [1.0, 4.0];
    let result = sparse_least_squares_parametric(
        &source,
        &[0.0],
        &parameters,
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(result.point[0], 9.0 / 5.0);
    for (actual, expected) in result.residuals.iter().zip([0.8, -0.4, 1.0]) {
        near(*actual, expected);
    }
    near(result.cost, 0.9);
    assert!(result.gradient_norm <= 1e-10);
    assert_eq!(parameters, [1.0, 4.0]);
    let checked = source
        .evaluate_sparse(&[result.point[0], 1.0, 4.0])
        .unwrap();
    assert_eq!(checked.residuals, result.residuals);
    // J_p^T r is deliberately nonzero: parameters are not optimized.
    near(1.0 - result.residuals[0], 0.2);
    assert!(
        sparse_newton_refine_parametric(&source, &[0.0], &parameters, &NewtonGuards::default(),)
            .is_err()
    );
}

#[test]
fn nonlinear_fit_and_rank_deficient_families_retain_fixed_data() {
    let nonlinear = system(&["x^2-p", "x^2-q"], &["x", "p", "q"]);
    let result = sparse_least_squares_parametric(
        &nonlinear,
        &[1.0],
        &[2.0, 4.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(result.point[0], 3.0_f64.sqrt());
    near(result.cost, 1.0);

    let source = system(&["x+y-p", "2*x+2*y-2*p"], &["x", "y", "z", "p"]);
    let result = sparse_least_squares_parametric(
        &source,
        &[0.0, 0.0, -7.0],
        &[3.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(result.point[0] + result.point[1], 3.0);
    assert_eq!(result.point[2], -7.0);
}

#[test]
fn identical_names_retain_typed_unknown_and_parameter_identity() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let p = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let residual = Expr::Sym(x.clone()) - Expr::from_i64(2) * Expr::Sym(p.clone());
    let source = VerifiedResidualSystem::try_compile(&[residual], &[x, p]).unwrap();
    let root =
        sparse_newton_refine_parametric(&source, &[0.0], &[3.0], &NewtonGuards::default()).unwrap();
    let fit =
        sparse_least_squares_parametric(&source, &[0.0], &[3.0], &LeastSquaresOptions::default())
            .unwrap();
    near(root.root[0], 6.0);
    near(fit.point[0], 6.0);
    assert_ne!(source.vars()[0], source.vars()[1]);
}

#[test]
fn two_hundred_fifty_five_fixed_coordinates_do_not_enter_the_linear_solve() {
    let mut axes = vec![Symbol::new("x")];
    axes.extend((0..255).map(|index| Symbol::new(format!("p{index}"))));
    let residual = Expr::Sym(axes[0].clone()) - Expr::Sym(axes[255].clone());
    let source = VerifiedResidualSystem::try_compile(&[residual], &axes).unwrap();
    let parameters = vec![2.0; 255];
    let root =
        sparse_newton_refine_parametric(&source, &[0.0], &parameters, &NewtonGuards::default())
            .unwrap();
    let fit = sparse_least_squares_parametric(
        &source,
        &[0.0],
        &parameters,
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(root.root[0], 2.0);
    near(fit.point[0], 2.0);
}

#[test]
fn parameter_only_rows_and_empty_projected_jacobians_have_distinct_root_and_fit_results() {
    let source = system(&["p"], &["x", "p"]);
    let fit =
        sparse_least_squares_parametric(&source, &[9.0], &[2.0], &LeastSquaresOptions::default())
            .unwrap();
    assert_eq!(fit.point, [9.0]);
    assert_eq!(fit.gradient_norm, 0.0);
    assert_eq!(fit.cost, 2.0);
    assert_eq!(fit.iterations, 0);
    assert!(matches!(
        sparse_newton_refine_parametric(&source, &[9.0], &[2.0], &NewtonGuards::default(),),
        Err(SystemConsumerError::NoConvergence { .. })
    ));
}

#[test]
fn parameter_derivatives_and_held_source_domains_are_checked_before_projection() {
    for (formula, initial, parameter) in [
        ("x-sqrt(p)", 0.0, 0.0),
        ("x+0*log(p)", 0.0, 0.0),
        ("x+0*(1/p)", 0.0, 0.0),
        ("x+log(p)^0-1", 0.0, 0.0),
        ("atan2(x,p)-pi", 0.0, -1.0),
    ] {
        let source = system(&[formula], &["x", "p"]);
        assert!(
            sparse_newton_refine_parametric(
                &source,
                &[initial],
                &[parameter],
                &NewtonGuards::default(),
            )
            .is_err(),
            "root accepted {formula}"
        );
        assert!(
            sparse_least_squares_parametric(
                &source,
                &[initial],
                &[parameter],
                &LeastSquaresOptions::default(),
            )
            .is_err(),
            "fit accepted {formula}"
        );
    }
}

#[test]
fn invalid_shapes_and_nonfinite_parameters_refuse_before_reserving_storage() {
    let source = system(&["x-p"], &["x", "p"]);
    for (initial, parameters) in [
        (vec![], vec![1.0, 2.0]),
        (vec![0.0, 0.0], vec![]),
        (vec![0.0], vec![]),
        (vec![0.0], vec![1.0, 2.0]),
        (vec![f64::NAN], vec![1.0]),
        (vec![0.0], vec![f64::INFINITY]),
    ] {
        let mut meter = budget();
        assert!(
            sparse_newton_refine_parametric_with_meter(
                &source,
                &initial,
                &parameters,
                &NewtonGuards::default(),
                &mut meter,
            )
            .is_err()
        );
        assert_eq!(meter.remaining(Dimension::MemoryBytes), AMPLE);
        assert_eq!(meter.remaining(Dimension::AllocationCount), AMPLE);
        let mut meter = budget();
        assert!(
            sparse_least_squares_parametric_with_meter(
                &source,
                &initial,
                &parameters,
                &LeastSquaresOptions::default(),
                &mut meter,
            )
            .is_err()
        );
        assert_eq!(meter.remaining(Dimension::MemoryBytes), AMPLE);
        assert_eq!(meter.remaining(Dimension::AllocationCount), AMPLE);
    }
}

#[test]
fn sparse_projection_and_final_publication_share_the_callers_complete_budget() {
    let source = system(&["x-p", "2*x-2*p"], &["x", "p"]);
    for least_squares in [false, true] {
        let run = |meter: &mut Budget| {
            if least_squares {
                sparse_least_squares_parametric_with_meter(
                    &source,
                    &[0.0],
                    &[2.0],
                    &LeastSquaresOptions::default(),
                    meter,
                )
                .map(|_| ())
            } else {
                sparse_newton_refine_parametric_with_meter(
                    &source,
                    &[0.0],
                    &[2.0],
                    &NewtonGuards::default(),
                    meter,
                )
                .map(|_| ())
            }
        };
        let mut probe = budget();
        run(&mut probe).unwrap();
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let spent = AMPLE - probe.remaining(dimension);
            for allowance in [0, spent - 1] {
                let mut limits = BudgetLimits::uniform(AMPLE, 73);
                limits.dimensions[dimension.index()] = allowance;
                let mut meter = Budget::new(limits);
                let error = run(&mut meter).unwrap_err();
                assert!(matches!(meter_cause(&error),
                    Some(MeterError::Budget(BudgetError::Exhausted { dimension: found, .. })) if found == dimension));
                assert_eq!(meter.verifier_remaining(), 73);
            }
            let mut limits = BudgetLimits::uniform(AMPLE, 73);
            limits.dimensions[dimension.index()] = spent;
            let mut meter = Budget::new(limits);
            run(&mut meter).unwrap();
            assert_eq!(meter.remaining(dimension), 0);
            assert!(run(&mut meter).is_err());
            assert_eq!(meter.verifier_remaining(), 73);
        }
    }
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    cancel_at: usize,
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
        if self.checkpoints >= self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_during_projection_iteration_or_final_publication_never_returns_a_proposal() {
    let source = system(&["x^2-p", "2*x^2-2*p"], &["x", "p"]);
    for least_squares in [false, true] {
        let run = |meter: &mut Interrupt| {
            if least_squares {
                sparse_least_squares_parametric_with_meter(
                    &source,
                    &[1.0],
                    &[2.0],
                    &LeastSquaresOptions::default(),
                    meter,
                )
                .map(|_| ())
            } else {
                sparse_newton_refine_parametric_with_meter(
                    &source,
                    &[1.0],
                    &[2.0],
                    &NewtonGuards::default(),
                    meter,
                )
                .map(|_| ())
            }
        };
        let mut probe = Interrupt {
            budget: budget(),
            checkpoints: 0,
            cancel_at: usize::MAX,
        };
        run(&mut probe).unwrap();
        for cancel_at in [1, 4, probe.checkpoints / 2, probe.checkpoints] {
            let mut meter = Interrupt {
                budget: budget(),
                checkpoints: 0,
                cancel_at,
            };
            assert_eq!(
                meter_cause(&run(&mut meter).unwrap_err()),
                Some(MeterError::Cancelled)
            );
            assert_eq!(meter.budget.verifier_remaining(), 73);
        }
    }
}
