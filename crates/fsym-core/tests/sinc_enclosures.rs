//! Independent original-source checks for the analytic sinc extension.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness, StationaryPointKind, Symbol,
    certify_stationary_point, enclose_objective, enclose_residual_jacobian,
    metered_enclose_objective,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn decimal(value: &str) -> BigRational {
    let (integer, fractional) = value.split_once('.').unwrap();
    let numerator: BigInt = format!("{integer}{fractional}").parse().unwrap();
    BigRational::new(numerator, BigInt::from(10).pow(fractional.len() as u32))
}

fn sinc(argument: Expr) -> Expr {
    Expr::Function("sinc".into(), vec![argument])
}

#[test]
fn exact_origin_has_value_one_zero_slope_and_negative_third_curvature() {
    let x = Symbol::new("sinc_origin");
    let source = sinc(Expr::Sym(x.clone()));
    let domain = [RealBall::from_i64(0)];
    let c1 = enclose_residual_jacobian(
        std::slice::from_ref(&source),
        std::slice::from_ref(&x),
        &domain,
        32,
    )
    .unwrap();
    let c2 = enclose_objective(&source, std::slice::from_ref(&x), &domain, 32).unwrap();
    assert_eq!(c1.residual_values(), [RealBall::from_i64(1)]);
    assert_eq!(c1.jacobian(), [vec![RealBall::from_i64(0)]]);
    assert_eq!(c2.value(), &RealBall::from_i64(1));
    assert_eq!(c2.gradient(), [RealBall::from_i64(0)]);
    assert_eq!(c2.hessian(), [vec![RealBall::exact(q(-1, 3))]]);
    assert_eq!(
        sinc(Expr::from_i64(0)).evalf_ball(32).unwrap(),
        RealBall::from_i64(1)
    );
}

#[test]
fn point_jets_agree_with_independent_high_precision_reference_values() {
    // Computed with mpmath 1.3.0 at 90 decimal digits, separate from the
    // candidate. These numerical fixtures test implementation accuracy;
    // the production enclosure proof is the exact integral/Taylor bound.
    let cases = [
        (
            q(1, 2),
            [
                "0.958851077208406000546575870431142776163606735881201350377233",
                "-0.162537030636066568860588575654626248343923077542914594759244",
                "-0.308702954664139725104221567812637782787914425709542971340254",
            ],
        ),
        (
            q(1, 1),
            [
                "0.841470984807896506652502321630298999622563060798371065672751",
                "-0.301168678939756789251565714187322395890252640180448838002654",
                "-0.239133626928382928149370893255654207842057780437473389667442",
            ],
        ),
        (
            q(2, 1),
            [
                "0.454648713412840847698009932955872421351127485723945134189486",
                "-0.435397774979991617347789081228317305558564128399745012472318",
                "-0.019250938432849230350220851727555115792563357324200121717168",
            ],
        ),
    ];
    let x = Symbol::new("sinc_reference");
    let source = sinc(Expr::Sym(x.clone()));
    for (point, expected) in cases {
        for sign in [-1, 1] {
            let result = enclose_objective(
                &source,
                std::slice::from_ref(&x),
                &[RealBall::exact(&point * q(sign, 1))],
                32,
            )
            .unwrap();
            for (order, value) in [
                result.value(),
                &result.gradient()[0],
                &result.hessian()[0][0],
            ]
            .into_iter()
            .enumerate()
            {
                let expected = decimal(expected[order]) * q(if order == 1 { sign } else { 1 }, 1);
                assert!(
                    value.contains(&expected),
                    "order {order}, point {point}, sign {sign}"
                );
                assert!(value.radius() < &BigRational::new(1.into(), BigInt::from(10).pow(33)));
            }
        }
    }
}

#[test]
fn tiny_nonzero_slopes_are_not_erased_by_absolute_rounding() {
    let x = Symbol::new("sinc_tiny");
    let tiny = BigRational::new(1.into(), BigInt::from(10).pow(80));
    for sign in [-1, 1] {
        let point = &tiny * q(sign, 1);
        let result = enclose_objective(
            &sinc(Expr::Sym(x.clone())),
            std::slice::from_ref(&x),
            &[RealBall::exact(point.clone())],
            40,
        )
        .unwrap();
        let slope = &result.gradient()[0];
        assert!(!slope.contains_zero());
        assert!(slope.contains(&(-&point / q(3, 1))));
        assert!(slope.width() < &tiny * BigRational::new(1.into(), BigInt::from(10).pow(40)));
        assert!(result.hessian()[0][0].contains(&q(-1, 3)));
    }
}

#[test]
fn whole_boxes_crossing_zero_retain_global_bounds_and_strict_concavity() {
    let x = Symbol::new("sinc_box");
    let source = sinc(Expr::Sym(x.clone()));
    for radius in [q(1, 100), q(1, 4), q(2, 1), q(100, 1)] {
        let domain = [RealBall::new(q(0, 1), radius.clone()).unwrap()];
        let result = enclose_objective(&source, std::slice::from_ref(&x), &domain, 24).unwrap();
        assert!(result.value().contains(&q(1, 1)));
        assert!(result.gradient()[0].contains(&q(0, 1)));
        assert!(result.hessian()[0][0].contains(&q(-1, 3)));
        assert!(result.value().lower() >= q(-1, 1) && result.value().upper() <= q(1, 1));
        assert!(result.gradient()[0].lower() >= q(-1, 2));
        assert!(result.gradient()[0].upper() <= q(1, 2));
        assert!(result.hessian()[0][0].lower() >= q(-1, 3));
        assert!(result.hessian()[0][0].upper() <= q(1, 3));
        if radius <= q(1, 4) {
            assert!(result.hessian()[0][0].is_negative());
        }
    }
}

#[test]
fn second_order_chain_rule_keeps_all_mixed_and_inner_derivative_terms() {
    let vars = [Symbol::new("sinc_x"), Symbol::new("sinc_y")];
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let source = sinc(Expr::from_i64(2) * x.clone() + y.clone() + x * y);
    let result = enclose_objective(
        &source,
        &vars,
        &[RealBall::from_i64(0), RealBall::from_i64(0)],
        32,
    )
    .unwrap();
    assert_eq!(
        result.gradient(),
        [RealBall::from_i64(0), RealBall::from_i64(0)]
    );
    assert_eq!(
        result.hessian(),
        [
            vec![RealBall::exact(q(-4, 3)), RealBall::exact(q(-2, 3))],
            vec![RealBall::exact(q(-2, 3)), RealBall::exact(q(-1, 3))],
        ]
    );
    // At (0, 1/2) the inner mixed second derivative has a nonzero slope
    // multiplier, independently checked using the uncomposed point jet.
    let point = [RealBall::from_i64(0), RealBall::exact(q(1, 2))];
    let actual = enclose_objective(&source, &vars, &point, 32).unwrap();
    let anchor = enclose_objective(
        &sinc(Expr::Sym(vars[0].clone())),
        &vars[..1],
        &point[1..],
        40,
    )
    .unwrap();
    let mixed = anchor.hessian()[0][0]
        .mul(&RealBall::exact(q(5, 2)))
        .add(&anchor.gradient()[0]);
    assert!(!actual.hessian()[0][1].is_disjoint(&mixed));
    assert_eq!(actual.hessian()[0][1], actual.hessian()[1][0]);
}

#[test]
fn original_undefined_children_cannot_hide_behind_the_sinc_extension() {
    let x = Symbol::new("sinc_domain");
    let pole = Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-1)));
    let sources = [
        sinc(pole.clone()),
        sinc(Expr::Mul(vec![Expr::from_i64(0), pole])),
        sinc(Expr::Function("log".into(), vec![Expr::Sym(x.clone())])),
        Expr::Function("sinc".into(), vec![]),
        Expr::Function("sinc".into(), vec![Expr::Sym(x.clone()), Expr::from_i64(1)]),
        Expr::Function("_fsym_sinc_d1".into(), vec![Expr::Sym(x.clone())]),
    ];
    for source in sources {
        assert!(
            enclose_objective(
                &source,
                std::slice::from_ref(&x),
                &[RealBall::from_i64(0)],
                24
            )
            .is_err()
        );
        assert!(
            enclose_residual_jacobian(
                std::slice::from_ref(&source),
                std::slice::from_ref(&x),
                &[RealBall::from_i64(0)],
                24,
            )
            .is_err()
        );
    }
}

#[test]
fn sinc_origin_is_a_certified_strict_local_maximum_and_negation_a_minimum() {
    let x = Symbol::new("sinc_stationary");
    for sign in [-1, 1] {
        let source = Expr::from_i64(sign) * sinc(Expr::Sym(x.clone()));
        let witness = RootBoxWitness {
            center: vec![q(0, 1)],
            domain: vec![RealBall::new(q(0, 1), q(1, 4)).unwrap()],
            preconditioner: vec![vec![q(-3 * sign, 1)]],
            precision_digits: 24,
        };
        let receipt =
            certify_stationary_point(&source, std::slice::from_ref(&x), &witness).unwrap();
        assert_eq!(
            receipt.kind(),
            if sign == 1 {
                StationaryPointKind::StrictMaximum
            } else {
                StationaryPointKind::StrictMinimum
            }
        );
        assert!(receipt.stationary_enclosure()[0].contains(&q(0, 1)));
        assert!(receipt.contraction_bound() < &q(1, 1));
    }
}

#[derive(Default)]
struct Cancel {
    visits: usize,
    stop: Option<usize>,
}

impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let at = self.visits;
        self.visits += 1;
        if self.stop == Some(at) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn series_and_publication_share_cancellation_and_cumulative_budget() {
    let x = Symbol::new("sinc_budget");
    let source = sinc(Expr::Sym(x.clone()));
    let domain = [RealBall::new(q(1, 2), q(1, 100)).unwrap()];
    let mut count = Cancel::default();
    metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, &mut count).unwrap();
    for stop in [0, count.visits / 2, count.visits - 1] {
        let mut cancelled = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            metered_enclose_objective(
                &source,
                std::slice::from_ref(&x),
                &domain,
                24,
                &mut cancelled,
            ),
            Err(RootBoxError::Meter(MeterError::Cancelled))
        );
    }
    let limit = 512 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, &mut budget).unwrap();
    let remaining = budget.remaining(Dimension::ComputeSteps);
    metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, &mut budget).unwrap();
    assert!(budget.remaining(Dimension::ComputeSteps) < remaining && remaining < limit);
    let mut empty = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_enclose_objective(&source, std::slice::from_ref(&x), &domain, 24, &mut empty,),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn unsupported_precision_refuses_before_claiming_a_value() {
    for precision in [0, RealBall::MAX_SINC_PRECISION_DIGITS + 1, u32::MAX] {
        assert!(RealBall::from_i64(0).sinc(precision).is_err());
    }
}
