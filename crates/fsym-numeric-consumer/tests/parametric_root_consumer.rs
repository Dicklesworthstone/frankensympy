//! Fixed parameters retain their source roles throughout bounded Newton work.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{Expr, Symbol, SymbolIdentity, parse_raw};
use fsym_numeric_consumer::{
    NewtonGuards, SystemConsumerError, newton_refine_parametric,
    newton_refine_parametric_with_meter,
};
use std::error::Error;

const AMPLE: u64 = 1 << 40;

fn system(residuals: &[&str], variables: &[&str]) -> VerifiedResidualSystem {
    let residuals: Vec<_> = residuals
        .iter()
        .map(|source| parse_raw(source).unwrap())
        .collect();
    let variables: Vec<_> = variables.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap()
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(AMPLE, 73))
}

fn meter_cause(error: &(dyn Error + 'static)) -> Option<MeterError> {
    let mut cause = Some(error);
    while let Some(current) = cause {
        if let Some(meter) = current.downcast_ref::<MeterError>() {
            return Some(meter.clone());
        }
        cause = current.source();
    }
    None
}

#[test]
fn one_checked_program_solves_nonlinear_roots_for_distinct_fixed_parameters() {
    let source = system(&["x^2-p"], &["x", "p"]);
    let guards = NewtonGuards::default();
    let mut meter = budget();
    for parameter in [2.0_f64, 9.0] {
        let parameters = [parameter];
        let result =
            newton_refine_parametric_with_meter(&source, &[1.0], &parameters, &guards, &mut meter)
                .unwrap();
        assert_eq!(result.root.len(), 1);
        assert!((result.root[0] - parameter.sqrt()).abs() < 1e-9);
        let residual = result.root[0] * result.root[0] - parameter;
        assert!(residual.abs() <= guards.residual_tolerance);
        assert_eq!(result.residual_norm, residual.abs());
        assert_eq!(parameters[0].to_bits(), parameter.to_bits());
        let checked = source
            .evaluate_sparse(&[result.root[0], parameter])
            .unwrap();
        assert_eq!(checked.residuals, [residual]);
    }
    assert!(meter.remaining(Dimension::ComputeSteps) < AMPLE);
    assert_eq!(meter.verifier_remaining(), 73);
}

#[test]
fn nonsymmetric_unknown_block_preserves_coordinate_and_parameter_order_after_transport() {
    let original = system(&["2*x+y-p", "x-3*y-q"], &["y", "x", "q", "p"]);
    let source =
        VerifiedResidualSystem::from_json_bytes(&original.to_json_bytes().unwrap()).unwrap();
    for (parameters, expected) in [([-1.0, 5.0], [1.0, 2.0]), ([-2.0, 10.0], [2.0, 4.0])] {
        let result =
            newton_refine_parametric(&source, &[0.0, 0.0], &parameters, &NewtonGuards::default())
                .unwrap();
        for (actual, expected) in result.root.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(result.residual_norm <= 1e-10);
        assert_eq!(result.iterations, 1);
    }
    assert_eq!(source.vars(), original.vars());
    assert_eq!(source.residuals(), original.residuals());
}

#[test]
fn identical_printed_names_keep_distinct_unknown_and_parameter_identities() {
    let unknown = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let parameter = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let residual = Expr::Sym(unknown.clone()) - Expr::from_i64(2) * Expr::Sym(parameter.clone());
    let source = VerifiedResidualSystem::try_compile(&[residual], &[unknown, parameter]).unwrap();
    let result =
        newton_refine_parametric(&source, &[0.0], &[3.0], &NewtonGuards::default()).unwrap();
    assert_eq!(result.root, [6.0]);
    assert_eq!(result.residual_norm, 0.0);
    assert_ne!(source.vars()[0], source.vars()[1]);
}

#[test]
fn maximum_total_dimension_and_many_parameter_suffixes_do_not_become_unknowns() {
    let unknowns: Vec<_> = (0..15)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    let parameter = Symbol::new("p");
    let mut variables = unknowns.clone();
    variables.push(parameter.clone());
    let residuals: Vec<_> = unknowns
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            let target = i64::try_from(index + 1).unwrap();
            let mut residual = Expr::from_i64(2) * Expr::Sym(variable.clone())
                - Expr::from_i64(2 * target) * Expr::Sym(parameter.clone());
            if index + 1 < unknowns.len() {
                residual = residual + Expr::Sym(unknowns[index + 1].clone())
                    - Expr::from_i64(target + 1) * Expr::Sym(parameter.clone());
            }
            residual
        })
        .collect();
    let source = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
    let result =
        newton_refine_parametric(&source, &[0.0; 15], &[2.0], &NewtonGuards::default()).unwrap();
    for (actual, expected) in result.root.iter().zip(1_u32..=15) {
        assert!((actual - 2.0 * f64::from(expected)).abs() < 1e-11);
    }
    assert!(result.residual_norm <= 1e-10);

    let x = Symbol::new("x");
    let parameters: Vec<_> = (0..15)
        .map(|index| Symbol::new(format!("p{index}")))
        .collect();
    let residual = parameters
        .iter()
        .fold(Expr::Sym(x.clone()), |sum, parameter| {
            sum - Expr::Sym(parameter.clone())
        });
    let mut variables = vec![x];
    variables.extend(parameters);
    let source = VerifiedResidualSystem::try_compile(&[residual], &variables).unwrap();
    let values: Vec<_> = (1_u32..=15).map(f64::from).collect();
    let result =
        newton_refine_parametric(&source, &[0.0], &values, &NewtonGuards::default()).unwrap();
    assert_eq!(result.root, [120.0]);
}

#[test]
fn original_parameter_poles_branch_cuts_and_derivative_domains_are_never_skipped() {
    for (formula, initial, parameter) in [
        ("x+0*log(p)", 0.0, 0.0),
        ("x+0*(1/p)", 0.0, 0.0),
        ("x+log(p)^0-1", 0.0, 0.0),
        ("x-sqrt(p)", 0.0, 0.0),
        ("atan2(x,p)-pi", 0.0, -1.0),
    ] {
        let source = system(&[formula], &["x", "p"]);
        let mut meter = budget();
        assert!(
            newton_refine_parametric_with_meter(
                &source,
                &[initial],
                &[parameter],
                &NewtonGuards::default(),
                &mut meter,
            )
            .is_err(),
            "accepted a source/derivative boundary in {formula}"
        );
        assert_eq!(meter.verifier_remaining(), 73);
    }
    let source = system(&["x-sqrt(p)"], &["x", "p"]);
    assert_eq!(
        newton_refine_parametric(&source, &[0.0], &[4.0], &NewtonGuards::default())
            .unwrap()
            .root,
        [2.0]
    );
}

#[test]
fn singular_unknown_block_cannot_be_repaired_by_changing_a_parameter() {
    let source = system(&["x^2-p"], &["x", "p"]);
    // The full 1x2 Jacobian has rank one, but its unknown block is singular.
    assert!(matches!(
        newton_refine_parametric(&source, &[0.0], &[1.0], &NewtonGuards::default()),
        Err(SystemConsumerError::LinearSolve(_))
    ));
    // An already-zero residual remains an approximate proposal, with no
    // nonsingularity or uniqueness claim and no linear solve required.
    let result =
        newton_refine_parametric(&source, &[0.0], &[0.0], &NewtonGuards::default()).unwrap();
    assert_eq!(result.root, [0.0]);
    assert_eq!(result.iterations, 0);
}

#[test]
fn invalid_shapes_inputs_and_controls_refuse_before_allocating_solver_buffers() {
    let defaults = NewtonGuards::default();
    let square = system(&["x-1"], &["x"]);
    let variables: Vec<_> = (0..17)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    let too_wide = VerifiedResidualSystem::try_compile(&[Expr::from_i64(0)], &variables).unwrap();
    for source in [&square, &too_wide] {
        let mut meter = budget();
        assert!(matches!(
            newton_refine_parametric_with_meter(source, &[0.0], &[1.0], &defaults, &mut meter),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
        assert_eq!(meter.remaining(Dimension::MemoryBytes), AMPLE);
        assert_eq!(meter.remaining(Dimension::AllocationCount), AMPLE);
    }
    let source = system(&["x-p"], &["x", "p"]);
    for (initial, parameters) in [
        (vec![], vec![1.0]),
        (vec![0.0, 0.0], vec![1.0]),
        (vec![0.0], vec![]),
        (vec![0.0], vec![1.0, 2.0]),
        (vec![f64::NAN], vec![1.0]),
        (vec![0.0], vec![f64::INFINITY]),
        (vec![f64::NEG_INFINITY], vec![1.0]),
    ] {
        let mut meter = budget();
        assert!(
            newton_refine_parametric_with_meter(
                &source,
                &initial,
                &parameters,
                &defaults,
                &mut meter,
            )
            .is_err()
        );
        assert_eq!(meter.remaining(Dimension::MemoryBytes), AMPLE);
        assert_eq!(meter.verifier_remaining(), 73);
    }
    for guards in [
        NewtonGuards {
            max_iterations: 1025,
            ..defaults.clone()
        },
        NewtonGuards {
            residual_tolerance: -1.0,
            ..defaults.clone()
        },
        NewtonGuards {
            residual_tolerance: f64::NAN,
            ..defaults.clone()
        },
        NewtonGuards {
            max_step_norm: 0.0,
            ..defaults.clone()
        },
        NewtonGuards {
            max_step_norm: f64::INFINITY,
            ..defaults.clone()
        },
    ] {
        let mut meter = budget();
        assert!(matches!(
            newton_refine_parametric_with_meter(&source, &[0.0], &[1.0], &guards, &mut meter),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
        assert_eq!(meter.remaining(Dimension::MemoryBytes), AMPLE);
        assert_eq!(meter.verifier_remaining(), 73);
    }
}

#[test]
fn iteration_step_limits_and_final_residual_failure_return_no_proposal() {
    let linear = system(&["x-p"], &["x", "p"]);
    let no_steps = NewtonGuards {
        max_iterations: 0,
        ..NewtonGuards::default()
    };
    assert!(matches!(
        newton_refine_parametric(&linear, &[0.0], &[1.0], &no_steps),
        Err(SystemConsumerError::NoConvergence { iterations: 0, .. })
    ));
    assert_eq!(
        newton_refine_parametric(&linear, &[1.0], &[1.0], &no_steps)
            .unwrap()
            .iterations,
        0
    );
    let short_step = NewtonGuards {
        max_step_norm: 0.5,
        ..NewtonGuards::default()
    };
    assert!(matches!(
        newton_refine_parametric(&linear, &[0.0], &[1.0], &short_step),
        Err(SystemConsumerError::NoConvergence { iterations: 0, .. })
    ));
    let nonlinear = system(&["x^2-p"], &["x", "p"]);
    let one_step = NewtonGuards {
        max_iterations: 1,
        ..NewtonGuards::default()
    };
    assert!(matches!(
        newton_refine_parametric(&nonlinear, &[1.0], &[2.0], &one_step),
        Err(SystemConsumerError::NoConvergence { iterations: 1, residual_norm })
            if residual_norm == 0.25
    ));
}

#[test]
fn ordinary_resources_are_cumulative_and_never_borrow_protected_fuel() {
    let source = system(&["x-p"], &["x", "p"]);
    let guards = NewtonGuards::default();
    let mut probe = budget();
    newton_refine_parametric_with_meter(&source, &[0.0], &[2.0], &guards, &mut probe).unwrap();
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
            let error =
                newton_refine_parametric_with_meter(&source, &[0.0], &[2.0], &guards, &mut meter)
                    .unwrap_err();
            assert!(matches!(
                meter_cause(&error),
                Some(MeterError::Budget(BudgetError::Exhausted { dimension: found, .. }))
                    if found == dimension
            ));
            assert_eq!(meter.verifier_remaining(), 73);
        }
        let mut limits = BudgetLimits::uniform(AMPLE, 73);
        limits.dimensions[dimension.index()] = spent;
        let mut meter = Budget::new(limits);
        newton_refine_parametric_with_meter(&source, &[0.0], &[2.0], &guards, &mut meter).unwrap();
        assert_eq!(meter.remaining(dimension), 0);
        let error =
            newton_refine_parametric_with_meter(&source, &[0.0], &[2.0], &guards, &mut meter)
                .unwrap_err();
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted { dimension: found, .. }))
                if found == dimension
        ));
        assert_eq!(meter.remaining(dimension), 0);
        assert_eq!(meter.verifier_remaining(), 73);
    }
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    cancel_at: usize,
}

impl Interrupt {
    fn new(cancel_at: usize) -> Self {
        Self {
            budget: budget(),
            checkpoints: 0,
            cancel_at,
        }
    }
}

impl BudgetMeter for Interrupt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let index = self.checkpoints;
        self.checkpoints += 1;
        if index >= self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_solve_safe_point_never_returns_a_partial_root() {
    let source = system(&["x-p"], &["x", "p"]);
    let guards = NewtonGuards::default();
    let mut measured = Interrupt::new(usize::MAX);
    newton_refine_parametric_with_meter(&source, &[0.0], &[2.0], &guards, &mut measured).unwrap();
    assert!(measured.checkpoints > 10);
    for cancel_at in 0..measured.checkpoints {
        let mut meter = Interrupt::new(cancel_at);
        let error =
            newton_refine_parametric_with_meter(&source, &[0.0], &[2.0], &guards, &mut meter)
                .unwrap_err();
        assert_eq!(
            meter_cause(&error),
            Some(MeterError::Cancelled),
            "safe point {cancel_at}"
        );
        assert_eq!(meter.budget.verifier_remaining(), 73);
        assert_eq!(meter.checkpoints, cancel_at + 1);
    }
}
