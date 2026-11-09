//! Sparse root proposals must satisfy every original residual, not stationarity.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_numeric_consumer::{
    LeastSquaresOptions, NewtonGuards, RootProposal, SystemConsumerError, newton_refine,
    sparse_least_squares, sparse_newton_refine, sparse_newton_refine_with_meter,
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

fn check_original_residuals(source: &VerifiedResidualSystem, proposal: &RootProposal, tol: f64) {
    let mut norm = 0.0_f64;
    for program in source.residual_programs() {
        let value = program.try_eval_finite(&proposal.root).unwrap();
        norm = norm.max(value.abs());
    }
    assert_eq!(norm, proposal.residual_norm);
    assert!(norm <= tol, "original residual norm {norm} exceeds {tol}");
}

#[test]
fn full_sparse_variable_bound_solves_without_dense_storage() {
    let n = 256;
    let variables: Vec<_> = (0..n).map(|i| Symbol::new(format!("x{i}"))).collect();
    let sources: Vec<_> = variables
        .iter()
        .enumerate()
        .map(|(i, variable)| {
            Expr::Add(vec![
                Expr::Sym(variable.clone()),
                Expr::from_i64(-((i % 7 + 1) as i64)),
            ])
        })
        .collect();
    let source = VerifiedResidualSystem::try_compile(&sources, &variables).unwrap();
    assert_eq!(source.jacobian_pattern().len(), n);
    let mut measured = Recorder::default();
    let guards = NewtonGuards::default();
    let result =
        sparse_newton_refine_with_meter(&source, &vec![0.0; n], &guards, &mut measured).unwrap();
    for (i, &value) in result.root.iter().enumerate() {
        near(value, (i % 7 + 1) as f64, 1e-12);
    }
    check_original_residuals(&source, &result, guards.residual_tolerance);
    // One dense Jacobian alone would reserve 512 KiB. Every measured buffer
    // fits the vector/nonzero envelope of this diagonal system instead.
    assert!(measured.max_memory_charge <= 32 * 1024);
    assert!(matches!(
        newton_refine(&source, &vec![0.0; n], &guards),
        Err(SystemConsumerError::UnsupportedShape { .. })
    ));
}

#[test]
fn coupled_banded_systems_reach_the_root_at_the_full_variable_bound() {
    let n = 256;
    let names: Vec<_> = (0..n).map(|i| format!("x{i}")).collect();
    let variables: Vec<_> = names.iter().map(|name| Symbol::new(name.clone())).collect();
    let sources: Vec<_> = (0..n)
        .map(|i| {
            let mut expression = format!("4*{}", names[i]);
            if i > 0 {
                expression.push_str(&format!("-{}", names[i - 1]));
            }
            if i + 1 < n {
                expression.push_str(&format!("-{}", names[i + 1]));
            }
            expression.push_str(if i == 0 || i + 1 == n { "-3" } else { "-2" });
            parse(&expression).unwrap()
        })
        .collect();
    let source = VerifiedResidualSystem::try_compile(&sources, &variables).unwrap();
    assert_eq!(source.jacobian_pattern().len(), 3 * n - 2);
    let guards = NewtonGuards::default();
    let result = sparse_newton_refine(&source, &vec![0.0; n], &guards).unwrap();
    for &coordinate in &result.root {
        near(coordinate, 1.0, 1e-11);
    }
    check_original_residuals(&source, &result, guards.residual_tolerance);
}

#[test]
fn a_small_cost_gradient_cannot_substitute_for_a_small_root_residual() {
    let source = system(&["(x-1)/100000000"], &["x"]);
    let stationary =
        sparse_least_squares(&source, &[0.0], &LeastSquaresOptions::default()).unwrap();
    assert_eq!(stationary.iterations, 0);
    assert_eq!(stationary.point, [0.0]);
    assert!(stationary.residuals[0].abs() > NewtonGuards::default().residual_tolerance);
    let guards = NewtonGuards::default();
    let root = sparse_newton_refine(&source, &[0.0], &guards).unwrap();
    assert!(root.iterations > 0);
    near(root.root[0], 1.0, 1e-12);
    check_original_residuals(&source, &root, guards.residual_tolerance);
}

#[test]
fn stationary_nonroots_and_inconsistent_singular_systems_refuse() {
    for source in [
        system(&["x*x+1"], &["x"]),
        system(&["x+y-1", "x+y-3"], &["x", "y"]),
        system(&["1"], &["x"]),
    ] {
        let error = sparse_newton_refine(
            &source,
            &vec![0.0; source.num_vars()],
            &NewtonGuards::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SystemConsumerError::NoConvergence { residual_norm, .. } if residual_norm >= 1.0)
        );
    }
}

#[test]
fn consistent_rank_deficiency_and_unused_coordinates_do_not_require_an_inverse() {
    let source = system(&["x+y-3", "2*x+2*y-6", "0"], &["x", "y", "unused"]);
    let guards = NewtonGuards::default();
    let result = sparse_newton_refine(&source, &[0.0, 0.0, 7.0], &guards).unwrap();
    near(result.root[0] + result.root[1], 3.0, 1e-12);
    assert_eq!(result.root[2], 7.0);
    check_original_residuals(&source, &result, guards.residual_tolerance);
    // There is no uniqueness assertion, and an already satisfied singular
    // root must not demand a linear solve merely to publish its approximation.
    let exact = sparse_newton_refine(
        &source,
        &[1.0, 2.0, -9.0],
        &NewtonGuards {
            max_iterations: 0,
            ..guards
        },
    )
    .unwrap();
    assert_eq!(exact.root, [1.0, 2.0, -9.0]);
    assert_eq!(exact.iterations, 0);
}

#[test]
fn nonlinear_coupling_and_column_units_preserve_the_original_equations() {
    let guards = NewtonGuards::default();
    for (source, initial, expected) in [
        (
            system(&["x*x+y*y-5", "x-y-1"], &["x", "y"]),
            [1.8, 0.8],
            [2.0, 1.0],
        ),
        (
            system(&["1000000000*(x-1)", "(y-2)/1000000000"], &["x", "y"]),
            [0.0, 0.0],
            [1.0, 2.0],
        ),
    ] {
        let result = sparse_newton_refine(&source, &initial, &guards).unwrap();
        for (&coordinate, expected) in result.root.iter().zip(expected) {
            near(coordinate, expected, 1e-9);
        }
        check_original_residuals(&source, &result, guards.residual_tolerance);
    }
}

#[test]
fn root_merit_scaling_retains_tiny_and_large_finite_residuals() {
    let huge = format!("1{}", "0".repeat(200));
    let large = format!("{huge}*(x-1)");
    let small = format!("(x-1)/{huge}");
    let guards = NewtonGuards {
        residual_tolerance: 0.0,
        ..NewtonGuards::default()
    };
    for expression in [&large, &small] {
        let source = system(&[expression], &["x"]);
        let result = sparse_newton_refine(&source, &[0.0], &guards).unwrap();
        assert_eq!(result.root, [1.0]);
        assert!(result.iterations > 0);
        check_original_residuals(&source, &result, 0.0);
    }

    // The smaller row may disappear in the first normalized linear model,
    // but the unscaled root criterion must force a new linearization for it.
    let small_y = format!("(y-2)/{huge}");
    let source = system(&[&large, &small_y], &["x", "y"]);
    let result = sparse_newton_refine(&source, &[0.0, 0.0], &guards).unwrap();
    assert_eq!(result.root, [1.0, 2.0]);
    check_original_residuals(&source, &result, 0.0);
}

#[test]
fn logarithmic_radical_and_overflowing_exponential_trials_can_backtrack() {
    for (expression, initial, target) in [
        ("log(x)", 10.0, 1.0),
        ("x^(1/2)-1", 4.0, 1.0),
        ("exp(x)-1", -7.0, 0.0),
    ] {
        let source = system(&[expression], &["x"]);
        let guards = NewtonGuards::default();
        let result = sparse_newton_refine(&source, &[initial], &guards).unwrap();
        near(result.root[0], target, 1e-10);
        check_original_residuals(&source, &result, guards.residual_tolerance);
    }
}

#[test]
fn invalid_initial_sources_and_derivative_domains_are_terminal() {
    for (expression, initial) in [
        ("log(x)", -1.0),
        ("log(x)", 0.0),
        ("x^(1/2)-1", 0.0),
        ("log(-1)", 2.0),
    ] {
        let source = system(&[expression], &["x"]);
        assert!(matches!(
            sparse_newton_refine(&source, &[initial], &NewtonGuards::default()),
            Err(SystemConsumerError::SparseEvaluation(
                ResidualEvaluationError::Evaluation(_)
            ))
        ));
    }
}

#[test]
fn held_domain_holes_survive_zero_coefficients_and_empty_jacobians() {
    let x = Symbol::new("x");
    let hidden = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
    ]);
    let source = VerifiedResidualSystem::try_compile(&[hidden], &[x]).unwrap();
    assert!(sparse_newton_refine(&source, &[-1.0], &NewtonGuards::default()).is_err());
    let valid = sparse_newton_refine(&source, &[1.0], &NewtonGuards::default()).unwrap();
    assert_eq!(valid.root, [1.0]);
    assert_eq!(valid.iterations, 0);
    assert_eq!(valid.residual_norm, 0.0);

    let constant = system(&["log(-1)"], &["unused"]);
    assert!(constant.jacobian_pattern().is_empty());
    assert!(sparse_newton_refine(&constant, &[7.0], &NewtonGuards::default()).is_err());
}

#[test]
fn artifact_replay_and_same_named_typed_coordinates_preserve_the_solve() {
    let plain = Symbol::new("x");
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32]));
    let variables = [typed.clone(), plain.clone()];
    let sources = [
        Expr::Add(vec![Expr::Sym(plain), Expr::from_i64(-2)]),
        Expr::Add(vec![Expr::Sym(typed), Expr::from_i64(-3)]),
    ];
    let source = VerifiedResidualSystem::try_compile(&sources, &variables).unwrap();
    let restored =
        VerifiedResidualSystem::from_json_bytes(&source.to_json_bytes().unwrap()).unwrap();
    let original = sparse_newton_refine(&source, &[0.0, 0.0], &NewtonGuards::default()).unwrap();
    let replay = sparse_newton_refine(&restored, &[0.0, 0.0], &NewtonGuards::default()).unwrap();
    assert_eq!(original, replay);
    near(replay.root[0], 3.0, 1e-12);
    near(replay.root[1], 2.0, 1e-12);
}

#[test]
fn unsupported_shapes_are_rejected_before_any_numerical_charges() {
    for (rows, columns) in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (257, 257)] {
        let variables: Vec<_> = (0..columns).map(|i| Symbol::new(format!("x{i}"))).collect();
        let source =
            VerifiedResidualSystem::try_compile(&vec![Expr::from_i64(0); rows], &variables)
                .unwrap();
        let mut meter = Recorder::default();
        assert!(
            matches!(sparse_newton_refine_with_meter(&source, &vec![0.0; columns], &NewtonGuards::default(), &mut meter),
            Err(SystemConsumerError::UnsupportedSparseRootShape { variables, residuals }) if variables == columns && residuals == rows)
        );
        assert_eq!(meter.charges, 0);
    }
}

#[test]
fn invalid_controls_and_coordinates_never_start_the_search() {
    let source = system(&["x-1"], &["x"]);
    let defaults = NewtonGuards::default();
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
        assert!(matches!(
            sparse_newton_refine(&source, &[0.0], &guards),
            Err(SystemConsumerError::InvalidRequest(_))
        ));
    }
    for point in [vec![], vec![0.0, 1.0], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(sparse_newton_refine(&source, &point, &defaults).is_err());
    }
}

#[test]
fn clipped_steps_and_iteration_limits_do_not_authorize_false_success() {
    let source = system(&["x-2"], &["x"]);
    let guards = NewtonGuards {
        max_step_norm: 0.25,
        ..NewtonGuards::default()
    };
    let root = sparse_newton_refine(&source, &[0.0], &guards).unwrap();
    near(root.root[0], 2.0, 1e-12);
    assert!(root.iterations >= 8);
    let zero = NewtonGuards {
        max_iterations: 0,
        ..NewtonGuards::default()
    };
    assert!(matches!(
        sparse_newton_refine(&source, &[0.0], &zero),
        Err(SystemConsumerError::NoConvergence { iterations: 0, .. })
    ));
    assert!(sparse_newton_refine(&source, &[2.0], &zero).is_ok());
    let tiny = NewtonGuards {
        max_step_norm: f64::from_bits(1),
        residual_tolerance: 0.0,
        ..NewtonGuards::default()
    };
    assert!(matches!(
        sparse_newton_refine(&source, &[1.0], &tiny),
        Err(SystemConsumerError::NoConvergence { .. })
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
        )) => Some(*error),
        _ => None,
    }
}

#[test]
fn cancellation_and_exhaustion_are_terminal_at_every_safe_point_and_charge() {
    let source = system(&["log(x)"], &["x"]);
    let guards = NewtonGuards::default();
    let mut complete = Recorder::default();
    sparse_newton_refine_with_meter(&source, &[10.0], &guards, &mut complete).unwrap();
    for checkpoint in 1..=complete.checkpoints {
        let mut meter = Recorder {
            stop: Some(checkpoint),
            ..Recorder::default()
        };
        let error = sparse_newton_refine_with_meter(&source, &[10.0], &guards, &mut meter)
            .expect_err("published after cancellation");
        assert_eq!(meter_cause(&error), Some(MeterError::Cancelled));
        assert!(meter.stopped);
        assert_eq!(meter.checkpoints, checkpoint);
    }
    for charge in 1..=complete.charges {
        let mut meter = Recorder {
            exhaust_at: Some(charge),
            ..Recorder::default()
        };
        let error = sparse_newton_refine_with_meter(&source, &[10.0], &guards, &mut meter)
            .expect_err("published after exhaustion");
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted { .. }))
        ));
        assert!(meter.stopped);
        assert_eq!(meter.charges, charge);
    }
}

#[test]
fn rejected_trials_and_final_publication_share_the_callers_cumulative_budget() {
    let source = system(&["log(x)"], &["x"]);
    let guards = NewtonGuards::default();
    let mut measured = Recorder::default();
    sparse_newton_refine_with_meter(&source, &[10.0], &guards, &mut measured).unwrap();
    for (dimension, total) in [
        (Dimension::ComputeSteps, measured.compute),
        (Dimension::MemoryBytes, measured.bytes),
        (Dimension::AllocationCount, measured.allocations),
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000_000, 37);
        limits.dimensions[dimension.index()] = total;
        let mut budget = Budget::new(limits);
        sparse_newton_refine_with_meter(&source, &[10.0], &guards, &mut budget).unwrap();
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);
        assert!(sparse_newton_refine_with_meter(&source, &[10.0], &guards, &mut budget).is_err());
        assert_eq!(budget.remaining(dimension), 0);
        assert_eq!(budget.verifier_remaining(), 37);
        limits.dimensions[dimension.index()] -= 1;
        let mut short = Budget::new(limits);
        let error = sparse_newton_refine_with_meter(&source, &[10.0], &guards, &mut short)
            .expect_err("published without final resource unit");
        assert!(
            matches!(meter_cause(&error), Some(MeterError::Budget(BudgetError::Exhausted { dimension: exhausted, .. })) if exhausted == dimension)
        );
        assert_eq!(short.verifier_remaining(), 37);
    }
}
