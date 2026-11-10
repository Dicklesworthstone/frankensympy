//! Magnitude-sensitive exponential reservations and shared endpoint ledgers.
//! Expensive kernels must refuse prepayment before doing the mathematical work.

use fsym_budget::{Budget, BudgetLimits, Dimension, MeterError};
use fsym_core::{
    BigRational, Expr, RealBall, RootBoxError, Symbol, metered_enclose_residual_jacobian,
};

const LIMIT: u64 = 1_000_000_000;
const RESERVE: u64 = 37;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn evaluate(domain: &RealBall, budget: &mut Budget) -> Result<(), RootBoxError> {
    let variable = Symbol::new("x");
    let source = Expr::Function("exp".into(), vec![Expr::Sym(variable.clone())]);
    metered_enclose_residual_jacobian(
        &[source],
        &[variable],
        std::slice::from_ref(domain),
        24,
        budget,
    )
    .map(|_| ())
}

fn measured(domain: &RealBall) -> (u64, u64) {
    let mut budget = Budget::new(BudgetLimits::uniform(LIMIT, RESERVE));
    evaluate(domain, &mut budget).unwrap();
    (
        LIMIT - budget.remaining(Dimension::ComputeSteps),
        LIMIT - budget.remaining(Dimension::AllocationCount),
    )
}

#[test]
fn positive_magnitude_guard_refuses_expensive_work_before_the_kernel_runs() {
    let (ordinary_steps, _) = measured(&RealBall::from_i64(1));
    let mut limits = BudgetLimits::uniform(LIMIT, RESERVE);
    limits.dimensions[Dimension::ComputeSteps.index()] = ordinary_steps;
    let mut budget = Budget::new(limits);
    // The source scalars 1 and 400 occupy the same number of limbs. Their
    // absolute-exp Taylor schedules differ substantially. The 400 schedule
    // exceeds the unchanged intrinsic work ceiling and is refused before
    // invoking RealBall::exp, rather than after constructing its huge value.
    assert!(matches!(
        evaluate(&RealBall::from_i64(400), &mut budget),
        Err(RootBoxError::ResourceLimit("cumulative arithmetic work"))
    ));
    let consumed = ordinary_steps - budget.remaining(Dimension::ComputeSteps);
    assert!(consumed > 0);
    assert!(consumed < ordinary_steps);
    assert_eq!(budget.verifier_remaining(), RESERVE);
}

#[test]
fn a_wide_exponential_pays_for_both_endpoints_on_the_original_ledger() {
    let lower = measured(&RealBall::from_i64(1));
    let upper = measured(&RealBall::from_i64(2));
    let domain = RealBall::new(q(3, 2), q(1, 2)).unwrap();
    let wide = measured(&domain);
    assert!(wide.0 > lower.0.max(upper.0));
    assert!(wide.1 > lower.1.max(upper.1));
    // Enough for either endpoint in isolation is not enough for both on
    // the same ledger. A failed upper-endpoint reservation retains charges
    // already consumed by source admission and the lower endpoint.
    let available = lower.0.max(upper.0);
    let mut limits = BudgetLimits::uniform(LIMIT, RESERVE);
    limits.dimensions[Dimension::ComputeSteps.index()] = available;
    let mut budget = Budget::new(limits);
    assert!(matches!(
        evaluate(&domain, &mut budget),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    assert!(budget.remaining(Dimension::ComputeSteps) < available);
    assert!(budget.remaining(Dimension::ComputeSteps) > 0);
    assert_eq!(budget.verifier_remaining(), RESERVE);
}
