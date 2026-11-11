//! Search and independent replay must exploit proved whole-cell monotonicity
//! without spending a subdivision or relaxing the requested objective gap.

use fsym_budget::{Budget, BudgetLimits, Unbounded};
use fsym_calculus::box_optimization::{BoxOptimizationOptions, propose_box_optimization};
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationReceipt, BoxPartitionStep, Expr, RealBall, Symbol,
    verify_box_optimization,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn power(value: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(value), Arc::new(Expr::from_i64(exponent)))
}

fn negative(value: Expr) -> Expr {
    Expr::Mul(vec![Expr::from_i64(-1), value])
}

fn checked_one_leaf(
    source: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    kind: BoxExtremumKind,
) -> BoxOptimizationReceipt {
    let tolerance = q(1, 1_000_000_000_000);
    let mut meter = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 91));
    let proposed = propose_box_optimization(
        source,
        variables,
        domain,
        kind,
        &tolerance,
        BoxOptimizationOptions {
            precision_digits: 24,
            max_leaves: 1,
        },
        &mut meter,
    )
    .unwrap();
    assert_eq!(proposed.partition, [BoxPartitionStep::Leaf]);
    let receipt = verify_box_optimization(
        source, variables, domain, kind, &tolerance, &proposed, &mut meter,
    )
    .unwrap();
    assert_eq!(receipt.objective(), source);
    assert_eq!(receipt.domain(), domain);
    assert_eq!(receipt.leaf_count(), 1);
    assert!(receipt.gap() <= &tolerance);
    assert_eq!(meter.verifier_remaining(), 91);
    receipt
}

#[test]
fn exponential_minus_its_argument_certifies_both_endpoints_without_subdivision() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let source = Expr::Add(vec![
        Expr::Function("exp".into(), vec![x.clone()]),
        negative(x),
    ]);
    let domain = [interval(q(1, 1), q(2, 1))];
    // exp(x)-1 > 0 on [1,2]. These coarse exact brackets for e-1 and e^2-2
    // follow independently from the positive Taylor series and its geometric
    // tail bound; the receipt must retain its much narrower rounding gap.
    for (kind, point, lower, upper) in [
        (
            BoxExtremumKind::Minimum,
            1,
            q(1_718_281_828, 1_000_000_000),
            q(1_718_281_829, 1_000_000_000),
        ),
        (
            BoxExtremumKind::Maximum,
            2,
            q(5_389_056_098, 1_000_000_000),
            q(5_389_056_099, 1_000_000_000),
        ),
    ] {
        let receipt = checked_one_leaf(&source, &variables, &domain, kind);
        assert_eq!(receipt.point(), [q(point, 1)]);
        assert!(receipt.value().lower() > lower);
        assert!(receipt.value().upper() < upper);
        assert!(receipt.gap() > &q(0, 1));
    }
}

#[test]
fn shifted_monotone_quadratic_family_accepts_zero_derivative_at_a_boundary() {
    let variables = [Symbol::new("x")];
    for coefficient in [-5, -2, 1, 4, 7] {
        let x = Expr::Sym(variables[0].clone());
        let source = Expr::Add(vec![
            Expr::Mul(vec![Expr::from_i64(coefficient), x.clone()]),
            negative(power(x, 2)),
        ]);
        let left = q(coefficient, 2);
        let right = &left + q(1, 1);
        let domain = [interval(left.clone(), right.clone())];
        let peak = q(coefficient * coefficient, 4);
        for (kind, point, value) in [
            (BoxExtremumKind::Maximum, left, peak.clone()),
            (BoxExtremumKind::Minimum, right, peak - q(1, 1)),
        ] {
            let receipt = checked_one_leaf(&source, &variables, &domain, kind);
            assert_eq!(receipt.point(), [point]);
            assert_eq!(receipt.value(), &RealBall::exact(value));
            assert_eq!(receipt.gap(), &q(0, 1));
        }
    }
}

#[test]
fn mixed_decreasing_increasing_and_fixed_axes_have_exact_one_leaf_optima() {
    let variables = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let quadratic = |variable: &Symbol| {
        Expr::Add(vec![
            Expr::Sym(variable.clone()),
            negative(power(Expr::Sym(variable.clone()), 2)),
        ])
    };
    let source = Expr::Add(vec![
        quadratic(&variables[0]),
        negative(quadratic(&variables[1])),
        Expr::Sym(variables[2].clone()),
    ]);
    let domain = [
        interval(q(1, 1), q(2, 1)),
        interval(q(1, 1), q(2, 1)),
        RealBall::from_i64(4),
    ];
    for (kind, point, expected) in [
        (BoxExtremumKind::Minimum, vec![q(2, 1), q(1, 1), q(4, 1)], 2),
        (BoxExtremumKind::Maximum, vec![q(1, 1), q(2, 1), q(4, 1)], 6),
    ] {
        let receipt = checked_one_leaf(&source, &variables, &domain, kind);
        assert_eq!(receipt.point(), point);
        assert_eq!(receipt.value(), &RealBall::from_i64(expected));
    }
}

#[test]
fn a_good_endpoint_cannot_bypass_a_pole_in_the_original_cell() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let source = Expr::Add(vec![
        x.clone(),
        negative(power(x.clone(), 2)),
        Expr::Mul(vec![
            Expr::from_i64(0),
            power(Expr::Add(vec![x, Expr::Rational(q(-3, 2))]), -1),
        ]),
    ]);
    assert!(
        propose_box_optimization(
            &source,
            &variables,
            &[interval(q(1, 1), q(2, 1))],
            BoxExtremumKind::Maximum,
            &q(1, 1000),
            BoxOptimizationOptions {
                precision_digits: 24,
                max_leaves: 1
            },
            &mut Unbounded,
        )
        .is_err(),
    );
}
