#[path = "baseline.rs"]
mod baseline;

use frankensympy::{BoxOptimizationId, CertificationRequest, ExactInterval, ObjectiveId, Session};
use fsym_budget::{Budget, BudgetLimits, Dimension, Unbounded};
use fsym_core::{BigRational, BoxExtremumKind, RealBall};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 10_000_000;
    limits
}

fn report(label: &str, budget: &Budget) {
    println!("{label}: compute={} memory={} allocations={} verifier={}",
        limits().dimensions[Dimension::ComputeSteps.index()] - budget.remaining(Dimension::ComputeSteps),
        limits().dimensions[Dimension::MemoryBytes.index()] - budget.remaining(Dimension::MemoryBytes),
        limits().dimensions[Dimension::AllocationCount.index()] - budget.remaining(Dimension::AllocationCount),
        limits().verifier_pool - budget.verifier_remaining());
}

fn main() {
    let source = format!("x{}", "+0*x".repeat(600));
    let problem = frankensympy::ObjectiveProblem::parse_source(&source, &["x".into()], &mut Unbounded).unwrap();
    let domain = [RealBall::new(0.into(), 1.into()).unwrap()];
    let tolerance = BigRational::new(1.into(), 1000.into());
    let mut old_budget = Budget::new(limits());
    let old_lease = old_budget.verifier_lease().unwrap();
    let old = baseline::propose_box_optimization(problem.objective(), problem.variables(), &domain,
        BoxExtremumKind::Minimum, &tolerance,
        baseline::BoxOptimizationOptions { precision_digits: 24, max_leaves: 256 }, &mut old_budget).unwrap();
    report("13363cb generator", &old_budget);
    problem.verify_box_optimization(&domain, BoxExtremumKind::Minimum, &tolerance, &old,
        &mut old_budget.verifier_meter(&old_lease).unwrap()).unwrap();
    report("13363cb generator + same independent verifier", &old_budget);

    let mut new_budget = Budget::new(limits());
    let new_lease = new_budget.verifier_lease().unwrap();
    let new = fsym_calculus::box_optimization::propose_box_optimization(problem.objective(), problem.variables(), &domain,
        BoxExtremumKind::Minimum, &tolerance,
        fsym_calculus::box_optimization::BoxOptimizationOptions { precision_digits: 24, max_leaves: 256 }, &mut new_budget).unwrap();
    assert_eq!(new, old);
    report("new generator", &new_budget);
    problem.verify_box_optimization(&domain, BoxExtremumKind::Minimum, &tolerance, &new,
        &mut new_budget.verifier_meter(&new_lease).unwrap()).unwrap();
    report("new generator + same independent verifier", &new_budget);

    let mut session = Session::with_budget_limits(limits(), 256);
    let objective = ObjectiveId::new("objective:global").unwrap();
    session.certification_request(CertificationRequest::RegisterObjective {
        schema_version: 1, objective: objective.clone(), source, variables: vec!["x".into()]
    }).unwrap();
    let before = session.budget_snapshot();
    println!("Session registration allocations={}", limits().dimensions[Dimension::AllocationCount.index()] - before.dimensions[Dimension::AllocationCount.index()]);
    let result = session.certification_request(CertificationRequest::OptimizeBox {
        schema_version: 1, objective, optimization: BoxOptimizationId::new("optimization:sent").unwrap(),
        domain: vec![ExactInterval{lower:"-1".into(),upper:"1".into()}],
        extremum: BoxExtremumKind::Minimum, absolute_tolerance:"1/1000".into(), precision_digits:24, max_leaves:256
    }).unwrap();
    let after = session.budget_snapshot();
    println!("Session through accepted OptimizeBox allocations={} compute={} memory={} verifier={}",
        limits().dimensions[Dimension::AllocationCount.index()] - after.dimensions[Dimension::AllocationCount.index()],
        limits().dimensions[Dimension::ComputeSteps.index()] - after.dimensions[Dimension::ComputeSteps.index()],
        limits().dimensions[Dimension::MemoryBytes.index()] - after.dimensions[Dimension::MemoryBytes.index()],
        limits().verifier_pool - after.verifier_pool);
    println!("response={}", serde_json::to_string(&result).unwrap());
}
