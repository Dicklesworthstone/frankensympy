//! Original-source principal-angle C1/C2 rules, independently derived exact
//! derivatives, accepted root/stationary claims and terminal resource failures.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Constant, Expr, RealBall, RootBoxError, RootBoxWitness,
    StationaryPointKind, Symbol, certify_root_box, certify_stationary_point, enclose_objective,
    enclose_residual_jacobian, metered_enclose_objective, metered_enclose_residual_jacobian,
    verify_root_box, verify_stationary_point,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn atan2(y: Expr, x: Expr) -> Expr {
    Expr::Function("atan2".into(), vec![y, x])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn coordinate_partials_and_hessians_have_the_exact_quadrant_independent_signs() {
    let variables = [Symbol::new("y"), Symbol::new("x")];
    let source = atan2(Expr::Sym(variables[0].clone()), Expr::Sym(variables[1].clone()));
    for (y, x) in [(2, 3), (2, -3), (-2, 3), (-2, -3), (0, 3), (2, 0)] {
        let domain = [RealBall::from_i64(y), RealBall::from_i64(x)];
        let denominator = x * x + y * y;
        let square = denominator * denominator;
        let first = [
            RealBall::exact(q(x, denominator)),
            RealBall::exact(q(-y, denominator)),
        ];
        let second = [
            vec![
                RealBall::exact(q(-2 * x * y, square)),
                RealBall::exact(q(y * y - x * x, square)),
            ],
            vec![
                RealBall::exact(q(y * y - x * x, square)),
                RealBall::exact(q(2 * x * y, square)),
            ],
        ];
        let c1 =
            enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24)
                .unwrap();
        let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
        assert_eq!(c1.jacobian()[0], first);
        assert_eq!(c2.gradient(), first);
        assert_eq!(c2.hessian(), second);
        assert_eq!(c1.residual_values()[0], *c2.value());
        assert_eq!(c1.residuals(), std::slice::from_ref(&source));
        assert_eq!(c2.objective(), &source);
        assert_eq!(c2.domain(), domain);
    }
}

#[test]
fn nonlinear_argument_hessians_and_both_mixed_chain_terms_are_retained() {
    let variables = [Symbol::new("u"), Symbol::new("v")];
    let u = Expr::Sym(variables[0].clone());
    let v = Expr::Sym(variables[1].clone());
    // y=2+u+3v+u^2+2uv, x=3+4u-v+2v^2 at (u,v)=(0,0).
    // Hand-derived gradient=(-5/13,11/13), Hessian=(218,-61;-61,-170)/169.
    let y = Expr::Add(vec![
        Expr::from_i64(2),
        u.clone(),
        Expr::Mul(vec![Expr::from_i64(3), v.clone()]),
        power(u.clone(), 2),
        Expr::Mul(vec![Expr::from_i64(2), u.clone(), v.clone()]),
    ]);
    let x = Expr::Add(vec![
        Expr::from_i64(3),
        Expr::Mul(vec![Expr::from_i64(4), u]),
        Expr::Mul(vec![Expr::from_i64(-1), v.clone()]),
        Expr::Mul(vec![Expr::from_i64(2), power(v, 2)]),
    ]);
    let source = atan2(y, x);
    let domain = [RealBall::from_i64(0), RealBall::from_i64(0)];
    let c1 =
        enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24).unwrap();
    let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
    assert_eq!(
        c2.gradient(),
        [RealBall::exact(q(-5, 13)), RealBall::exact(q(11, 13))]
    );
    assert_eq!(c1.jacobian()[0], c2.gradient());
    assert_eq!(
        c2.hessian(),
        [
            vec![RealBall::exact(q(218, 169)), RealBall::exact(q(-61, 169))],
            vec![RealBall::exact(q(-61, 169)), RealBall::exact(q(-170, 169))],
        ]
    );
}

#[test]
fn complete_rectangle_derivative_bounds_include_independent_rational_anchors() {
    let variables = [Symbol::new("y"), Symbol::new("x")];
    let source = atan2(Expr::Sym(variables[0].clone()), Expr::Sym(variables[1].clone()));
    let domain = [interval(q(1, 1), q(2, 1)), interval(q(-2, 1), q(2, 1))];
    let c1 =
        enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24).unwrap();
    let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
    assert_eq!(c1.residual_values()[0], *c2.value());
    for y in [q(1, 1), q(3, 2), q(2, 1)] {
        for x in [q(-2, 1), q(-1, 1), q(0, 1), q(1, 1), q(2, 1)] {
            let d = &x * &x + &y * &y;
            let d2 = &d * &d;
            let expected_gradient = [&x / &d, -(&y / &d)];
            let yy = q(-2, 1) * &x * &y / &d2;
            let mixed = (&y * &y - &x * &x) / &d2;
            let xx = -yy.clone();
            for (i, expected) in expected_gradient.iter().enumerate() {
                assert!(c1.jacobian()[0][i].contains(expected));
                assert!(c2.gradient()[i].contains(expected));
            }
            assert!(c2.hessian()[0][0].contains(&yy));
            assert!(c2.hessian()[0][1].contains(&mixed));
            assert!(c2.hessian()[1][0].contains(&mixed));
            assert!(c2.hessian()[1][1].contains(&xx));
        }
    }
}

#[test]
fn cut_origin_and_held_shortcuts_cannot_publish_differentiability_receipts() {
    let variables = [Symbol::new("y"), Symbol::new("x")];
    let angle = atan2(Expr::Sym(variables[0].clone()), Expr::Sym(variables[1].clone()));
    let sources = [
        angle.clone(),
        Expr::Mul(vec![Expr::from_i64(0), angle.clone()]),
        power(angle, 0),
    ];
    let zero = RealBall::from_i64(0);
    let crossing = interval(q(-1, 1), q(1, 1));
    for domain in [
        [zero.clone(), RealBall::from_i64(-1)],
        [crossing.clone(), RealBall::from_i64(-1)],
        [interval(q(0, 1), q(1, 1)), RealBall::from_i64(-1)],
        [interval(q(-1, 1), q(0, 1)), RealBall::from_i64(-1)],
        [zero.clone(), zero],
        [crossing.clone(), crossing],
    ] {
        for source in &sources {
            assert!(matches!(
                enclose_residual_jacobian(std::slice::from_ref(source), &variables, &domain, 24),
                Err(RootBoxError::Domain(_))
            ));
            assert!(matches!(
                enclose_objective(source, &variables, &domain, 24),
                Err(RootBoxError::Domain(_))
            ));
        }
    }
}

#[test]
fn both_original_children_and_exact_arity_are_admitted_before_angle_identities() {
    let variables = [Symbol::new("t")];
    let domain = [RealBall::from_i64(0)];
    let invalid = Expr::Function("log".into(), vec![Expr::from_i64(-1)]);
    for source in [
        atan2(Expr::from_i64(0), invalid.clone()),
        atan2(invalid.clone(), Expr::from_i64(1)),
        atan2(Expr::from_i64(1), Expr::Mul(vec![Expr::from_i64(0), invalid])),
    ] {
        assert!(matches!(
            enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24),
            Err(RootBoxError::Domain(_))
        ));
        assert!(matches!(
            enclose_objective(&source, &variables, &domain, 24),
            Err(RootBoxError::Domain(_))
        ));
    }
    for arity in [0, 1, 3] {
        let source = Expr::Function("atan2".into(), vec![Expr::from_i64(1); arity]);
        assert!(matches!(
            enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24),
            Err(RootBoxError::Unsupported(_))
        ));
        assert!(matches!(
            enclose_objective(&source, &variables, &domain, 24),
            Err(RootBoxError::Unsupported(_))
        ));
    }
}

fn witness(preconditioner: BigRational, precision_digits: u32) -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(q(0, 1), q(1, 20)).unwrap()],
        center: vec![q(0, 1)],
        preconditioner: vec![vec![preconditioner]],
        precision_digits,
    }
}

#[test]
fn angle_root_and_squared_angle_minimum_use_original_source_verifiers() {
    let variables = [Symbol::new("t")];
    let source = atan2(Expr::Sym(variables[0].clone()), Expr::from_i64(1));
    let proposal = witness(q(1, 1), 24);
    let receipt = certify_root_box(std::slice::from_ref(&source), &variables, &proposal).unwrap();
    assert_eq!(receipt.residuals(), std::slice::from_ref(&source));
    assert_eq!(receipt.variables(), variables);
    assert_eq!(receipt.witness(), &proposal);
    assert!(receipt.root_enclosure()[0].contains(&q(0, 1)));
    assert!(receipt.contraction_bound() < &q(1, 1));
    let shifted = source.clone() + Expr::from_i64(1);
    assert!(certify_root_box(&[shifted], &variables, &proposal).is_err());
    assert!(
        certify_root_box(
            std::slice::from_ref(&source),
            &variables,
            &witness(q(-1, 1), 24)
        )
        .is_err()
    );

    let objective = power(source, 2);
    let proposal = witness(q(1, 2), 24);
    let minimum = certify_stationary_point(&objective, &variables, &proposal).unwrap();
    assert_eq!(minimum.kind(), StationaryPointKind::StrictMinimum);
    assert_eq!(minimum.objective(), &objective);
    assert_eq!(minimum.witness(), &proposal);
    assert!(minimum.stationary_enclosure()[0].contains(&q(0, 1)));
    assert!(minimum.objective_enclosure().contains(&q(0, 1)));
    assert!(minimum.hessian_enclosure()[0][0].lower() > q(0, 1));
}

#[test]
fn a_coupled_polar_system_retains_typed_variable_order_and_rejects_false_angles() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(variables[0].clone());
    let y = Expr::Sym(variables[1].clone());
    let quarter_pi = Expr::Mul(vec![Expr::Rational(q(1, 4)), Expr::Const(Constant::Pi)]);
    let sources = [
        atan2(y.clone(), x.clone()) - quarter_pi,
        x + y - Expr::from_i64(2),
    ];
    let proposal = RootBoxWitness {
        domain: vec![RealBall::new(q(1, 1), q(1, 20)).unwrap(); 2],
        center: vec![q(1, 1); 2],
        preconditioner: vec![vec![q(-1, 1), q(1, 2)], vec![q(1, 1), q(1, 2)]],
        precision_digits: 24,
    };
    let receipt = certify_root_box(&sources, &variables, &proposal).unwrap();
    assert_eq!(receipt.residuals(), sources);
    assert_eq!(receipt.variables(), variables);
    assert!(
        receipt
            .root_enclosure()
            .iter()
            .all(|coordinate| coordinate.contains(&q(1, 1)))
    );
    let mut false_sources = sources.clone();
    false_sources[0] = false_sources[0].clone() + Expr::from_i64(1);
    assert!(certify_root_box(&false_sources, &variables, &proposal).is_err());
}

#[derive(Clone, Copy, Debug)]
enum Lane {
    First,
    Second,
    Root,
    Stationary,
}

const LANES: [Lane; 4] = [Lane::First, Lane::Second, Lane::Root, Lane::Stationary];

fn metered_case(lane: Lane, meter: &mut impl BudgetMeter) -> Result<(), RootBoxError> {
    let variables = [Symbol::new("t")];
    let source = atan2(Expr::Sym(variables[0].clone()), Expr::from_i64(1));
    let domain = [RealBall::exact(q(3, 5))];
    match lane {
        Lane::First => {
            metered_enclose_residual_jacobian(&[source], &variables, &domain, 8, meter).map(|_| ())
        }
        Lane::Second => {
            metered_enclose_objective(&source, &variables, &domain, 8, meter).map(|_| ())
        }
        Lane::Root => {
            verify_root_box(&[source], &variables, &witness(q(1, 1), 8), meter).map(|_| ())
        }
        Lane::Stationary => {
            verify_stationary_point(&power(source, 2), &variables, &witness(q(1, 2), 8), meter)
                .map(|_| ())
        }
    }
}

struct Checkpoints {
    visits: usize,
    stop: Option<usize>,
    stopped: bool,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after a terminal cancellation");
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after a terminal cancellation");
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "continued after a terminal cancellation");
        self.visits += 1;
        if self.stop == Some(self.visits) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_angle_safe_point_refuses_without_publishing_a_receipt() {
    for lane in LANES {
        let mut measured = Checkpoints {
            visits: 0,
            stop: None,
            stopped: false,
        };
        metered_case(lane, &mut measured).unwrap();
        assert!(measured.visits > 0);
        for stop in 1..=measured.visits {
            let mut meter = Checkpoints {
                visits: 0,
                stop: Some(stop),
                stopped: false,
            };
            assert_eq!(
                metered_case(lane, &mut meter),
                Err(RootBoxError::Meter(MeterError::Cancelled)),
                "{lane:?}, checkpoint {stop}"
            );
            assert!(meter.stopped);
            assert_eq!(meter.visits, stop);
        }
    }
}

#[test]
fn cumulative_phase_limits_include_publication_and_leave_the_verifier_reserve_untouched() {
    const LIMIT: u64 = 1_000_000_000_000;
    const RESERVE: u64 = 37;
    for lane in LANES {
        let mut measured = Budget::new(BudgetLimits::uniform(LIMIT, RESERVE));
        metered_case(lane, &mut measured).unwrap();
        assert_eq!(measured.verifier_remaining(), RESERVE);
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let used = LIMIT - measured.remaining(dimension);
            assert!(used > 0);
            let mut limits = BudgetLimits::uniform(LIMIT, RESERVE);
            limits.dimensions[dimension.index()] = used;
            let mut exact = Budget::new(limits);
            metered_case(lane, &mut exact).unwrap();
            assert_eq!(exact.remaining(dimension), 0);
            let after = exact.snapshot();
            assert!(matches!(
                metered_case(lane, &mut exact),
                Err(RootBoxError::Meter(MeterError::Budget(BudgetError::Exhausted {
                    dimension: exhausted, ..
                }))) if exhausted == dimension
            ));
            assert_eq!(exact.snapshot(), after);
            assert_eq!(exact.verifier_remaining(), RESERVE);

            limits.dimensions[dimension.index()] = used - 1;
            let mut short = Budget::new(limits);
            assert!(matches!(
                metered_case(lane, &mut short),
                Err(RootBoxError::Meter(MeterError::Budget(BudgetError::Exhausted {
                    dimension: exhausted, ..
                }))) if exhausted == dimension
            ));
            assert_eq!(short.verifier_remaining(), RESERVE);
        }
    }
}
