//! Analytic checks of original-source monotone face bounds. The expected
//! extrema come from elementary polynomial identities, not a search proposal.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded,
};
use fsym_core::{
    BigRational, BoxCellEnclosure, BoxExtremumKind, BoxOptimizationError, BoxOptimizationWitness,
    BoxOptimizationWorkspace, BoxPartitionStep, Expr, RealBall, RootBoxError, Symbol,
    certify_box_optimization, verify_box_optimization,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn bounds(lower: i64, upper: i64) -> RealBall {
    interval(q(lower, 1), q(upper, 1))
}

fn power(source: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(source), Arc::new(Expr::from_i64(exponent)))
}

fn negative(source: Expr) -> Expr {
    Expr::Mul(vec![Expr::from_i64(-1), source])
}

fn quadratic(variable: &Symbol) -> Expr {
    // Keep the repeated variable visible to interval evaluation.
    Expr::Add(vec![
        Expr::Sym(variable.clone()),
        negative(power(Expr::Sym(variable.clone()), 2)),
    ])
}

fn witness(point: Vec<BigRational>) -> BoxOptimizationWitness {
    BoxOptimizationWitness {
        partition: vec![BoxPartitionStep::Leaf],
        point,
        precision_digits: 24,
    }
}

fn enclosure(source: &Expr, variables: &[Symbol], domain: &[RealBall]) -> BoxCellEnclosure {
    BoxOptimizationWorkspace::new(source, variables, domain, 24, &mut Unbounded)
        .unwrap()
        .enclose_cell(domain)
        .unwrap()
}

#[test]
fn repeated_variable_monotone_quadratic_has_exact_endpoint_bounds() {
    let variables = [Symbol::new("x")];
    let source = quadratic(&variables[0]);
    let domain = [bounds(1, 2)];
    // x-x^2 decreases from 0 to -2; its centered Taylor upper bound is 1/4.
    assert_eq!(
        enclosure(&source, &variables, &domain).range(),
        &bounds(-2, 0)
    );
    for (kind, point, expected) in [
        (BoxExtremumKind::Minimum, q(2, 1), q(-2, 1)),
        (BoxExtremumKind::Maximum, q(1, 1), q(0, 1)),
    ] {
        let receipt = certify_box_optimization(
            &source,
            &variables,
            &domain,
            kind,
            &q(1, 1_000_000),
            &witness(vec![point]),
        )
        .unwrap();
        assert_eq!(receipt.value(), &RealBall::exact(expected));
        assert_eq!(receipt.gap(), &q(0, 1));
        assert_eq!(receipt.leaf_count(), 1);
    }
}

#[test]
fn nonstrict_signs_include_zero_at_an_endpoint_and_an_identically_zero_derivative() {
    let variables = [Symbol::new("x")];
    let domain = [interval(q(1, 2), q(1, 1))];
    let source = quadratic(&variables[0]);
    let decreasing = enclosure(&source, &variables, &domain);
    assert_eq!(decreasing.gradient()[0].upper(), q(0, 1));
    assert_eq!(decreasing.range(), &interval(q(0, 1), q(1, 4)));
    let increasing = enclosure(&negative(source), &variables, &domain);
    assert_eq!(increasing.gradient()[0].lower(), q(0, 1));
    assert_eq!(increasing.range(), &interval(q(-1, 4), q(0, 1)));

    let flat = Expr::Add(vec![
        Expr::Sym(variables[0].clone()),
        negative(Expr::Sym(variables[0].clone())),
        Expr::from_i64(7),
    ]);
    let enclosed = enclosure(&flat, &variables, &[bounds(-5, 9)]);
    assert_eq!(enclosed.gradient()[0], RealBall::from_i64(0));
    assert_eq!(enclosed.range(), &RealBall::from_i64(7));
}

#[test]
fn mixed_monotone_axes_keep_uncertain_and_fixed_coordinates_intact() {
    let variables = [
        Symbol::new("x"),
        Symbol::new("y"),
        Symbol::new("z"),
        Symbol::new("u"),
    ];
    let source = Expr::Add(vec![
        quadratic(&variables[0]),
        negative(quadratic(&variables[1])),
        Expr::Sym(variables[2].clone()),
        power(Expr::Sym(variables[3].clone()), 2),
    ]);
    let domain = [bounds(1, 2), bounds(1, 2), bounds(3, 3), bounds(-1, 1)];
    let enclosed = enclosure(&source, &variables, &domain);
    // Minimum: x=2,y=1,u=0. Maximum: x=1,y=2,u=+/-1. Reducing u
    // to its center with the monotone coordinates would lose the true max 6.
    assert_eq!(enclosed.range(), &bounds(1, 6));
    assert!(enclosed.gradient()[3].lower() < q(0, 1));
    assert!(enclosed.gradient()[3].upper() > q(0, 1));
    for (kind, point, expected) in [
        (
            BoxExtremumKind::Minimum,
            vec![q(2, 1), q(1, 1), q(3, 1), q(0, 1)],
            1,
        ),
        (
            BoxExtremumKind::Maximum,
            vec![q(1, 1), q(2, 1), q(3, 1), q(1, 1)],
            6,
        ),
    ] {
        let receipt = certify_box_optimization(
            &source,
            &variables,
            &domain,
            kind,
            &q(1, 1000),
            &witness(point),
        )
        .unwrap();
        assert_eq!(receipt.value(), &RealBall::from_i64(expected));
    }
}

#[test]
fn a_sign_changing_derivative_cannot_replace_an_interior_extremum_with_endpoint_values() {
    let variables = [Symbol::new("x")];
    let source = quadratic(&variables[0]);
    let domain = [bounds(0, 1)];
    let enclosed = enclosure(&source, &variables, &domain);
    assert_eq!(enclosed.gradient()[0], bounds(-1, 1));
    assert!(enclosed.range().contains(&q(1, 4)));
    // Both endpoints have value zero, but the exact maximum is 1/4 at 1/2.
    assert_eq!(
        certify_box_optimization(
            &source,
            &variables,
            &domain,
            BoxExtremumKind::Maximum,
            &q(1, 10),
            &witness(vec![q(0, 1)]),
        ),
        Err(BoxOptimizationError::AccuracyNotReached),
    );
    let receipt = certify_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Maximum,
        &q(1, 1000),
        &witness(vec![q(1, 2)]),
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::exact(q(1, 4)));
}

#[test]
fn original_source_poles_and_hidden_domains_are_checked_before_any_face_restriction() {
    let variables = [Symbol::new("x")];
    let interior_zero = Expr::Add(vec![
        Expr::Sym(variables[0].clone()),
        Expr::Rational(q(-3, 2)),
    ]);
    let hidden_sources = [
        Expr::Mul(vec![Expr::from_i64(0), power(interior_zero.clone(), -1)]),
        power(Expr::Function("log".into(), vec![interior_zero]), 0),
    ];
    for hidden in hidden_sources {
        let source = Expr::Add(vec![quadratic(&variables[0]), hidden]);
        assert!(
            BoxOptimizationWorkspace::new(&source, &variables, &[bounds(1, 2)], 24, &mut Unbounded)
                .unwrap()
                .enclose_cell(&[bounds(1, 2)])
                .is_err(),
        );
    }
}

#[derive(Default)]
struct Cancel {
    visited: usize,
    at: Option<usize>,
}

impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let here = self.visited;
        self.visited += 1;
        if self.at == Some(here) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_monotone_evaluation_checkpoint_can_cancel_before_publication() {
    let variables = [Symbol::new("x")];
    let source = quadratic(&variables[0]);
    let domain = [bounds(1, 2)];
    let proposed = witness(vec![q(1, 1)]);
    let mut count = Cancel::default();
    verify_box_optimization(
        &source,
        &variables,
        &domain,
        BoxExtremumKind::Maximum,
        &q(1, 1000),
        &proposed,
        &mut count,
    )
    .unwrap();
    for at in 0..count.visited {
        let mut meter = Cancel {
            visited: 0,
            at: Some(at),
        };
        assert_eq!(
            verify_box_optimization(
                &source,
                &variables,
                &domain,
                BoxExtremumKind::Maximum,
                &q(1, 1000),
                &proposed,
                &mut meter,
            ),
            Err(BoxOptimizationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            ))),
            "checkpoint {at}",
        );
    }
}

#[test]
fn additional_face_evaluations_consume_cumulative_memory_without_touching_reserved_verifier_work() {
    let variables = [Symbol::new("x")];
    let source = quadratic(&variables[0]);
    let uncertain_domain = [bounds(0, 1)];
    let monotone_domain = [bounds(1, 2)];
    let allowance = 100_000_000;
    let mut uncertain_budget = Budget::new(BudgetLimits::uniform(allowance, 17));
    BoxOptimizationWorkspace::new(
        &source,
        &variables,
        &uncertain_domain,
        24,
        &mut uncertain_budget,
    )
    .unwrap()
    .enclose_cell(&uncertain_domain)
    .unwrap();
    let ordinary_memory = allowance - uncertain_budget.remaining(Dimension::MemoryBytes);
    let mut limits = BudgetLimits::uniform(allowance, 17);
    limits.dimensions[Dimension::MemoryBytes.index()] = ordinary_memory;
    let mut constrained = Budget::new(limits);
    let result =
        BoxOptimizationWorkspace::new(&source, &variables, &monotone_domain, 24, &mut constrained)
            .unwrap()
            .enclose_cell(&monotone_domain);
    assert!(matches!(
        result,
        Err(BoxOptimizationError::Verification(RootBoxError::Meter(
            MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::MemoryBytes,
                ..
            },)
        ))),
    ));
    assert!(constrained.remaining(Dimension::MemoryBytes) < ordinary_memory);
    assert!(constrained.remaining(Dimension::ComputeSteps) < allowance);
    assert_eq!(constrained.verifier_remaining(), 17);
}
