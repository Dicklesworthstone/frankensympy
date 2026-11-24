//! Independent source, derivative, branch and request-accounting regressions.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Constant, Expr, RealBall, RootBoxError, RootBoxWitness,
    StationaryPointKind, Symbol, certify_root_box, certify_stationary_point, enclose_objective,
    enclose_residual_jacobian, metered_enclose_objective, metered_enclose_residual_jacobian,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lo: BigRational, hi: BigRational) -> RealBall {
    RealBall::new((&lo + &hi) / q(2, 1), (hi - lo) / q(2, 1)).unwrap()
}

fn w(argument: Expr, branch: Option<i64>) -> Expr {
    let mut arguments = vec![argument];
    if let Some(branch) = branch {
        arguments.push(Expr::from_i64(branch));
    }
    Expr::Function("LambertW".into(), arguments)
}

fn power(argument: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(argument), Arc::new(Expr::from_i64(exponent)))
}

fn source() -> (Symbol, Expr) {
    let variable = Symbol::new("x");
    let expression = Expr::Sym(variable.clone());
    (variable, expression)
}

#[test]
fn principal_zero_has_exact_analytic_value_gradient_and_curvature() {
    let (variable, x) = source();
    for branch in [None, Some(0)] {
        let expression = w(x.clone(), branch);
        let c1 = enclose_residual_jacobian(
            std::slice::from_ref(&expression),
            std::slice::from_ref(&variable),
            &[RealBall::from_i64(0)],
            18,
        )
        .unwrap();
        let c2 = enclose_objective(
            &expression,
            std::slice::from_ref(&variable),
            &[RealBall::from_i64(0)],
            18,
        )
        .unwrap();
        assert_eq!(c1.residual_values(), &[RealBall::from_i64(0)]);
        assert_eq!(c1.jacobian(), &[vec![RealBall::from_i64(1)]]);
        assert_eq!(c2.value(), &RealBall::from_i64(0));
        assert_eq!(c2.gradient(), &[RealBall::from_i64(1)]);
        assert_eq!(c2.hessian(), &[vec![RealBall::from_i64(-2)]]);
        assert_eq!(c2.objective(), &expression);
    }
}

#[test]
fn both_branches_enclose_independently_derived_forward_map_jets() {
    let (variable, x) = source();
    for (value, branch) in [(q(1, 2), 0), (q(-1, 2), 0), (q(-2, 1), -1)] {
        let point = RealBall::exact(value.clone());
        // Use the pre-existing forward exponential, not the inverse algorithm
        // or its reported derivative: z = w*exp(w).
        let argument = point.mul(&point.exp(36).unwrap());
        let expression = w(x.clone(), Some(branch));
        let result = enclose_objective(
            &expression,
            std::slice::from_ref(&variable),
            std::slice::from_ref(&argument),
            16,
        )
        .unwrap();
        let denominator = RealBall::exact(&value + q(1, 1));
        let exponential = RealBall::exact(-value.clone()).exp(36).unwrap();
        let slope = exponential.div(&denominator).unwrap();
        let curvature = exponential
            .pow(2)
            .unwrap()
            .mul(&RealBall::exact(&value + q(2, 1)))
            .div(&denominator.pow(3).unwrap())
            .unwrap()
            .neg();
        assert!(result.value().contains(&value));
        assert!(result.gradient()[0].contains_ball(&slope));
        assert!(result.hessian()[0][0].contains_ball(&curvature));
        assert_eq!(result.domain(), &[argument]);
        if branch == -1 {
            assert!(result.gradient()[0].is_negative());
        } else {
            assert!(result.gradient()[0].is_positive());
        }
    }
}

#[test]
fn multivariate_composition_retains_inner_hessian_and_mixed_terms() {
    let variables = [Symbol::new("u"), Symbol::new("v")];
    let u = Expr::Sym(variables[0].clone());
    let v = Expr::Sym(variables[1].clone());
    let inner = Expr::Add(vec![
        u.clone(),
        Expr::Mul(vec![Expr::from_i64(2), v.clone()]),
        power(u.clone(), 2),
        Expr::Mul(vec![u, v]),
    ]);
    let objective = w(inner, Some(0));
    let result = enclose_objective(
        &objective,
        &variables,
        &[RealBall::from_i64(0), RealBall::from_i64(0)],
        18,
    )
    .unwrap();
    assert_eq!(
        result.gradient(),
        &[RealBall::from_i64(1), RealBall::from_i64(2)]
    );
    assert_eq!(
        result.hessian(),
        &[
            vec![RealBall::from_i64(0), RealBall::from_i64(-3)],
            vec![RealBall::from_i64(-3), RealBall::from_i64(-8)],
        ]
    );
}

#[test]
fn wide_principal_interval_keeps_its_image_and_derivatives_across_zero() {
    let (variable, x) = source();
    let domain = interval(q(-1, 8), q(1, 8));
    let result = enclose_objective(&w(x, None), &[variable], &[domain], 16).unwrap();
    assert!(result.value().contains(&q(-1, 10)));
    assert!(result.value().contains(&q(1, 10)));
    assert!(result.value().width() > q(1, 5));
    assert!(result.gradient()[0].contains(&q(1, 1)));
    assert!(result.gradient()[0].is_positive());
    assert!(result.hessian()[0][0].contains(&q(-2, 1)));
    assert!(result.hessian()[0][0].is_negative());
}

#[test]
fn lower_branch_wide_interval_keeps_negative_slope_and_curvature_variation() {
    let (variable, x) = source();
    let domain = interval(q(-3, 10), q(-1, 5));
    let result = enclose_objective(&w(x, Some(-1)), &[variable], &[domain], 16).unwrap();
    assert!(result.value().contains(&q(-2, 1)));
    assert!(result.value().contains(&q(-5, 2)));
    assert!(result.gradient()[0].is_negative());
    // W'' changes sign at W=-2, inside this whole argument interval.
    assert!(result.hessian()[0][0].contains(&q(0, 1)));
    assert!(result.hessian()[0][0].lower() < q(0, 1));
    assert!(result.hessian()[0][0].upper() > q(0, 1));
}

#[test]
fn singular_and_invalid_original_sources_survive_zero_multipliers() {
    let (variable, x) = source();
    let pole = Expr::Mul(vec![x.clone(), power(x, -1)]);
    let branchpoint = Expr::Mul(vec![
        Expr::from_i64(-1),
        power(Expr::Const(Constant::E), -1),
    ]);
    for source in [
        w(pole, None),
        w(branchpoint, None),
        w(Expr::from_i64(0), Some(-1)),
    ] {
        for expression in [
            source.clone(),
            Expr::Mul(vec![Expr::from_i64(0), source.clone()]),
            power(source, 0),
        ] {
            let variables = std::slice::from_ref(&variable);
            let domain = &[RealBall::from_i64(0)];
            assert!(
                enclose_residual_jacobian(
                    std::slice::from_ref(&expression),
                    variables,
                    domain,
                    16,
                )
                .is_err()
            );
            assert!(enclose_objective(&expression, variables, domain, 16).is_err());
        }
    }
}

#[test]
fn whole_argument_domain_is_checked_before_midpoint_or_chain_shortcuts() {
    let (variable, x) = source();
    for (branch, domain) in [
        (0, interval(q(-2, 5), q(0, 1))),
        (-1, interval(q(-2, 5), q(-1, 5))),
        (-1, interval(q(-1, 4), q(0, 1))),
        (-1, RealBall::from_i64(1)),
    ] {
        let expression = w(x.clone(), Some(branch));
        assert!(
            enclose_objective(&expression, std::slice::from_ref(&variable), &[domain], 16,)
                .is_err()
        );
    }
}

#[test]
fn branches_are_fixed_literal_real_indices_and_malformed_calls_fail_closed() {
    let (variable, x) = source();
    let malformed = [
        vec![],
        vec![x.clone(), Expr::from_i64(0), Expr::from_i64(0)],
        vec![x.clone(), Expr::from_i64(1)],
        vec![x.clone(), Expr::from_i64(-2)],
        vec![x.clone(), Expr::Integer(BigInt::from(1) << 256u32)],
        vec![x.clone(), x.clone()],
        vec![x.clone(), Expr::Rational(q(-1, 2))],
        vec![
            x,
            Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
        ],
    ];
    for arguments in malformed {
        let expression = Expr::Function("LambertW".into(), arguments);
        assert!(matches!(
            enclose_objective(
                &expression,
                std::slice::from_ref(&variable),
                &[RealBall::from_i64(0)],
                16,
            ),
            Err(RootBoxError::Unsupported(_))
        ));
    }
    let private = Expr::Function(
        "_fsym_lambertw_d1".into(),
        vec![Expr::from_i64(0), Expr::from_i64(0)],
    );
    assert!(matches!(
        enclose_objective(&private, &[variable], &[RealBall::from_i64(0)], 16),
        Err(RootBoxError::Unsupported(_))
    ));
}

#[test]
fn principal_root_and_squared_minimum_are_independently_certified() {
    let (variable, x) = source();
    let expression = w(x, None);
    let mut witness = RootBoxWitness {
        domain: vec![interval(q(-1, 40), q(1, 40))],
        center: vec![q(0, 1)],
        preconditioner: vec![vec![q(1, 1)]],
        precision_digits: 16,
    };
    let root = certify_root_box(
        std::slice::from_ref(&expression),
        std::slice::from_ref(&variable),
        &witness,
    )
    .unwrap();
    assert_eq!(root.residuals(), std::slice::from_ref(&expression));
    assert!(root.root_enclosure()[0].contains(&q(0, 1)));
    let wrong = Expr::Add(vec![expression.clone(), Expr::from_i64(1)]);
    assert!(certify_root_box(&[wrong], std::slice::from_ref(&variable), &witness).is_err());
    witness.preconditioner[0][0] = q(1, 2);
    let objective = power(expression, 2);
    let minimum = certify_stationary_point(&objective, &[variable], &witness).unwrap();
    assert_eq!(minimum.kind(), StationaryPointKind::StrictMinimum);
    assert!(minimum.stationary_enclosure()[0].contains(&q(0, 1)));
    assert!(minimum.hessian_enclosure()[0][0].is_positive());
}

#[test]
fn lower_branch_root_uses_negative_jacobian_and_rejects_branch_substitution() {
    let (variable, x) = source();
    let expression = Expr::Add(vec![w(x.clone(), Some(-1)), Expr::from_i64(2)]);
    let witness = RootBoxWitness {
        domain: vec![interval(q(-28, 100), q(-26, 100))],
        center: vec![q(-27, 100)],
        preconditioner: vec![vec![q(-27, 200)]],
        precision_digits: 16,
    };
    let root = certify_root_box(
        std::slice::from_ref(&expression),
        std::slice::from_ref(&variable),
        &witness,
    )
    .unwrap();
    let exact_root = RealBall::from_i64(-2)
        .exp(36)
        .unwrap()
        .mul(&RealBall::from_i64(-2));
    assert!(root.root_enclosure()[0].contains_ball(&exact_root));
    let wrong_branch = Expr::Add(vec![w(x, Some(0)), Expr::from_i64(2)]);
    assert!(certify_root_box(&[wrong_branch], &[variable], &witness).is_err());
}

#[derive(Default)]
struct CountOrCancel {
    checkpoints: usize,
    cancel_at: Option<usize>,
    steps: u64,
}

impl BudgetMeter for CountOrCancel {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            if dimension == Dimension::ComputeSteps {
                self.steps += amount;
            }
        }
        Ok(())
    }
}

#[test]
fn cancellation_inside_inverse_and_at_publication_never_returns_a_receipt() {
    let (variable, x) = source();
    let expression = w(x, None);
    let domain = [RealBall::from_i64(1)];
    let mut probe = CountOrCancel::default();
    metered_enclose_objective(
        &expression,
        std::slice::from_ref(&variable),
        &domain,
        12,
        &mut probe,
    )
    .unwrap();
    assert!(probe.checkpoints > 100);
    for cancel_at in [1, 100, probe.checkpoints] {
        let mut meter = CountOrCancel {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            metered_enclose_objective(
                &expression,
                std::slice::from_ref(&variable),
                &domain,
                12,
                &mut meter,
            ),
            Err(RootBoxError::Meter(MeterError::Cancelled))
        );
        if cancel_at > 1 {
            assert!(meter.steps > 0);
        }
    }
}

#[test]
fn caller_budgets_accumulate_across_inverse_evaluations() {
    let (variable, x) = source();
    let expression = w(x, None);
    let domain = [RealBall::from_i64(1)];
    let mut probe = CountOrCancel::default();
    metered_enclose_residual_jacobian(
        std::slice::from_ref(&expression),
        std::slice::from_ref(&variable),
        &domain,
        12,
        &mut probe,
    )
    .unwrap();
    let mut limits = BudgetLimits::uniform(u64::MAX, 37);
    limits.dimensions[Dimension::ComputeSteps.index()] = probe.steps;
    let mut budget = Budget::new(limits);
    metered_enclose_residual_jacobian(
        std::slice::from_ref(&expression),
        std::slice::from_ref(&variable),
        &domain,
        12,
        &mut budget,
    )
    .unwrap();
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(budget.verifier_remaining(), 37);
    assert!(matches!(
        metered_enclose_residual_jacobian(&[expression], &[variable], &domain, 12, &mut budget,),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn nonlinear_exponential_sources_have_independent_exact_chain_anchors() {
    let (variable, x) = source();
    for (argument, branch, point, value, slope, curvature) in [
        (
            Expr::Function("exp".into(), vec![x.clone()]),
            0,
            1,
            q(1, 1),
            q(1, 2),
            q(1, 8),
        ),
        (
            Expr::Mul(vec![
                Expr::from_i64(-2),
                Expr::Function("exp".into(), vec![x.clone()]),
            ]),
            -1,
            -2,
            q(-2, 1),
            q(2, 1),
            q(2, 1),
        ),
    ] {
        let result = enclose_objective(
            &w(argument, Some(branch)),
            std::slice::from_ref(&variable),
            &[RealBall::from_i64(point)],
            16,
        )
        .unwrap();
        assert!(result.value().contains(&value));
        assert!(result.gradient()[0].contains(&slope));
        assert!(result.hessian()[0][0].contains(&curvature));
    }
}

#[test]
fn lower_branch_curvature_is_positive_and_negative_on_opposite_sides_of_inflection() {
    let (variable, x) = source();
    let expression = w(x, Some(-1));
    for (argument, positive) in [(q(-1, 3), true), (q(-1, 4), false)] {
        let result = enclose_objective(
            &expression,
            std::slice::from_ref(&variable),
            &[RealBall::exact(argument)],
            16,
        )
        .unwrap();
        if positive {
            assert!(result.hessian()[0][0].is_positive());
        } else {
            assert!(result.hessian()[0][0].is_negative());
        }
    }
}
