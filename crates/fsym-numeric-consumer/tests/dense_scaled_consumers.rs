//! Disparate physical units can make a nonsingular Jacobian fail the donor's
//! global pivot threshold. Approximate proposals still use original residuals.

use fsym_budget::{Budget, BudgetLimits, Dimension, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{BigRational, Symbol, parse_raw};
use fsym_numeric_consumer::{
    NewtonGuards, SystemConsumerError, newton_refine, newton_refine_parametric,
    newton_refine_with_meter, propose_root_box_with_meter,
};

fn system(rows: &[&str], names: &[&str]) -> VerifiedResidualSystem {
    let rows = rows
        .iter()
        .map(|row| parse_raw(row).unwrap())
        .collect::<Vec<_>>();
    let vars = names
        .iter()
        .map(|name| Symbol::new(*name))
        .collect::<Vec<_>>();
    VerifiedResidualSystem::try_compile(&rows, &vars).unwrap()
}

#[test]
fn newton_and_independent_root_box_allow_disparate_equation_units() {
    let problem = system(&["2^60*(y-2)", "x+y-3"], &["x", "y"]);
    assert!(matches!(
        fnp_linalg::solve_nxn(&[0.0, 2_f64.powi(60), 1.0, 1.0], &[0.0, 1.0], 2),
        Err(fnp_linalg::LinAlgError::SolverSingularity)
    ));
    let proposal = newton_refine(&problem, &[0.0, 0.0], &NewtonGuards::default()).unwrap();
    assert_eq!(proposal.root, [1.0, 2.0]);
    assert_eq!(proposal.residual_norm, 0.0);
    let radius = BigRational::new(1.into(), 8.into());
    let witness = propose_root_box_with_meter(
        &problem,
        &proposal.root,
        &[radius.clone(), radius],
        24,
        &mut Unbounded,
    )
    .unwrap();
    let inverse_scale = BigRational::new(1.into(), (1_i64 << 60).into());
    assert_eq!(witness.preconditioner[0][0], -inverse_scale.clone());
    assert_eq!(witness.preconditioner[1][0], inverse_scale);
    fsym_core::certify_root_box(problem.residuals(), problem.vars(), &witness).unwrap();
}

#[test]
fn variable_units_and_fixed_parameters_use_the_same_checked_unknown_block() {
    let problem = system(&["2^60*x+y-p", "2^60*x-y+p"], &["x", "y", "p"]);
    let proposal =
        newton_refine_parametric(&problem, &[0.0, 0.0], &[3.0], &NewtonGuards::default()).unwrap();
    assert_eq!(proposal.root, [0.0, 3.0]);
    assert_eq!(proposal.iterations, 1);
    assert_eq!(proposal.residual_norm, 0.0);
}

#[test]
fn genuine_rank_deficiency_still_refuses_after_equilibration() {
    let problem = system(&["2^60*(x+y-1)", "x+y-2"], &["x", "y"]);
    assert!(matches!(
        newton_refine(&problem, &[0.0; 2], &NewtonGuards::default()),
        Err(SystemConsumerError::LinearSolve(_))
    ));
}

#[test]
fn a_retry_and_its_publication_consume_one_ordinary_ledger() {
    let problem = system(&["2^60*(x-1)", "y-2"], &["x", "y"]);
    let limits = BudgetLimits::uniform(10_000_000, 113);
    let mut measured = Budget::new(limits);
    let run = |budget: &mut Budget| {
        newton_refine_with_meter(&problem, &[0.0; 2], &NewtonGuards::default(), budget)
    };
    assert_eq!(run(&mut measured).unwrap().root, [1.0, 2.0]);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let used = limits.dimensions[dimension.index()] - measured.remaining(dimension);
        let mut short = limits;
        short.dimensions[dimension.index()] = used - 1;
        let mut budget = Budget::new(short);
        assert!(run(&mut budget).is_err());
        assert_eq!(budget.verifier_remaining(), 113);
    }
}
