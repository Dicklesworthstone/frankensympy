//! Original-source absolute-value rules on strictly sign-separated real boxes.
//! Exact analytic derivatives are test oracles, never checker inputs. Touching
//! or crossing zero remains a refusal, including under held outer shortcuts.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness, StationaryPointKind, Symbol,
    certify_root_box, certify_stationary_point, enclose_objective, enclose_residual_jacobian,
    metered_enclose_objective, metered_enclose_residual_jacobian, verify_root_box,
    verify_stationary_point,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn asymmetric_intervals_retain_exact_values_signs_and_source_binding() {
    let variables = [Symbol::new("x")];
    for tag in ["abs", "Abs"] {
        let source = function(tag, Expr::Sym(variables[0].clone()));
        for (lower, upper, slope, expected_lower, expected_upper) in [
            (q(1, 3), q(7, 2), 1, q(1, 3), q(7, 2)),
            (q(-7, 2), q(-1, 3), -1, q(1, 3), q(7, 2)),
        ] {
            let domain = [interval(lower, upper)];
            let c1 =
                enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24)
                    .unwrap();
            let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
            assert_eq!(c2.value().lower(), expected_lower);
            assert_eq!(c2.value().upper(), expected_upper);
            assert_eq!(c1.residual_values()[0], *c2.value());
            assert_eq!(c1.jacobian()[0], [RealBall::from_i64(slope)]);
            assert_eq!(c2.gradient(), [RealBall::from_i64(slope)]);
            assert_eq!(c2.hessian(), [vec![RealBall::from_i64(0)]]);
            assert_eq!(c1.residuals(), std::slice::from_ref(&source));
            assert_eq!(c1.variables(), variables);
            assert_eq!(c2.objective(), &source);
            assert_eq!(c2.variables(), variables);
            assert_eq!(c2.domain(), domain);
        }
    }
}

#[test]
fn negative_inner_hessians_and_mixed_derivatives_are_negated_together() {
    let variables = [Symbol::new("u"), Symbol::new("v")];
    let u = Expr::Sym(variables[0].clone());
    let v = Expr::Sym(variables[1].clone());
    // g=2+u^2+3uv+2v^2 at (1,-1): g=2, Dg=(-1,-1),
    // D2g=((2,3),(3,4)). Both |g| and |-g| agree locally.
    let inner = Expr::Add(vec![
        Expr::from_i64(2),
        power(u.clone(), 2),
        Expr::Mul(vec![Expr::from_i64(3), u, v.clone()]),
        Expr::Mul(vec![Expr::from_i64(2), power(v, 2)]),
    ]);
    let domain = [RealBall::from_i64(1), RealBall::from_i64(-1)];
    for sign in [1, -1] {
        let source = function("abs", Expr::Mul(vec![Expr::from_i64(sign), inner.clone()]));
        let c1 = enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24)
            .unwrap();
        let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
        assert_eq!(c2.value(), &RealBall::from_i64(2));
        assert_eq!(c2.gradient(), vec![RealBall::from_i64(-1); 2]);
        assert_eq!(c1.jacobian()[0], c2.gradient());
        assert_eq!(
            c2.hessian(),
            [
                vec![RealBall::from_i64(2), RealBall::from_i64(3)],
                vec![RealBall::from_i64(3), RealBall::from_i64(4)],
            ]
        );
    }
}

#[test]
fn whole_box_bounds_cover_independent_rational_derivative_values() {
    let variables = [Symbol::new("u"), Symbol::new("v")];
    let u = Expr::Sym(variables[0].clone());
    let v = Expr::Sym(variables[1].clone());
    let inner = Expr::Add(vec![
        Expr::from_i64(10),
        power(u.clone(), 2),
        Expr::Mul(vec![u, v.clone()]),
        Expr::Mul(vec![Expr::from_i64(2), power(v, 2)]),
    ]);
    let source = function("Abs", Expr::Mul(vec![Expr::from_i64(-1), inner]));
    let domain = [interval(q(-1, 1), q(1, 1)), interval(q(-1, 2), q(3, 2))];
    let c1 =
        enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24).unwrap();
    let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
    for u in [q(-1, 1), q(0, 1), q(1, 1)] {
        for v in [q(-1, 2), q(1, 2), q(3, 2)] {
            let value = q(10, 1) + &u * &u + &u * &v + q(2, 1) * &v * &v;
            let gradient = [q(2, 1) * &u + &v, &u + q(4, 1) * &v];
            assert!(c1.residual_values()[0].contains(&value));
            assert!(c2.value().contains(&value));
            for (axis, derivative) in gradient.iter().enumerate() {
                assert!(c1.jacobian()[0][axis].contains(derivative));
                assert!(c2.gradient()[axis].contains(derivative));
            }
        }
    }
    assert_eq!(
        c2.hessian(),
        [
            vec![RealBall::from_i64(2), RealBall::from_i64(1)],
            vec![RealBall::from_i64(1), RealBall::from_i64(4)],
        ]
    );
}

#[test]
fn nested_absolute_logarithm_and_square_root_preserve_negative_branch_derivatives() {
    let variables = [Symbol::new("x")];
    let absolute = function("Abs", Expr::Sym(variables[0].clone()));
    for (source, point, value, first, second) in [
        (
            function("abs", absolute.clone()),
            -2,
            Some(q(2, 1)),
            q(-1, 1),
            q(0, 1),
        ),
        (
            function("log", absolute.clone()),
            -2,
            None,
            q(-1, 2),
            q(-1, 4),
        ),
        (
            function("sqrt", absolute),
            -4,
            Some(q(2, 1)),
            q(-1, 4),
            q(-1, 32),
        ),
    ] {
        let domain = [RealBall::from_i64(point)];
        let c1 = enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24)
            .unwrap();
        let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
        if let Some(value) = value {
            assert!(c1.residual_values()[0].contains(&value));
            assert!(c2.value().contains(&value));
        }
        assert!(c1.jacobian()[0][0].contains(&first));
        assert!(c2.gradient()[0].contains(&first));
        assert!(c2.hessian()[0][0].contains(&second));
    }
}

#[test]
fn exact_nonzero_constants_and_subnormal_signs_need_no_binary64_conversion() {
    let variables = [Symbol::new("x")];
    let small = BigRational::new(BigInt::from(1), BigInt::from(1) << 1100u32);
    for value in [q(7, 3), q(-7, 3), small.clone(), -small.clone()] {
        let source = function("abs", Expr::Rational(value.clone()));
        let domain = [RealBall::from_i64(0)];
        let c1 = enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24)
            .unwrap();
        let c2 = enclose_objective(&source, &variables, &domain, 24).unwrap();
        let expected = if value < q(0, 1) { -value } else { value };
        assert_eq!(c2.value(), &RealBall::exact(expected));
        assert_eq!(c1.residual_values()[0], *c2.value());
        assert_eq!(c1.jacobian()[0], [RealBall::from_i64(0)]);
        assert_eq!(c2.gradient(), [RealBall::from_i64(0)]);
        assert_eq!(c2.hessian(), [vec![RealBall::from_i64(0)]]);
    }
    for (point, sign) in [(small.clone(), 1), (-small, -1)] {
        let source = function("abs", Expr::Sym(variables[0].clone()));
        let c2 = enclose_objective(&source, &variables, &[RealBall::exact(point)], 24).unwrap();
        assert_eq!(c2.gradient(), [RealBall::from_i64(sign)]);
    }
}

#[test]
fn zero_crossings_touched_endpoints_and_held_shortcuts_never_grant_smoothness() {
    let variables = [Symbol::new("x")];
    let absolute = function("abs", Expr::Sym(variables[0].clone()));
    let sources = [
        absolute.clone(),
        Expr::Mul(vec![Expr::from_i64(0), absolute.clone()]),
        power(absolute, 0),
        function("Abs", Expr::from_i64(0)),
    ];
    for domain in [
        [RealBall::from_i64(0)],
        [interval(q(0, 1), q(3, 1))],
        [interval(q(-3, 1), q(0, 1))],
        [interval(q(-1, 3), q(5, 1))],
        [interval(q(-5, 1), q(1, 3))],
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
    // This conservative operation-wise checker does not infer that |x^2| is
    // smooth at zero by simplifying its original source to x^2.
    let source = function("abs", power(Expr::Sym(variables[0].clone()), 2));
    assert!(matches!(
        enclose_objective(&source, &variables, &[interval(q(-1, 1), q(1, 1))], 24),
        Err(RootBoxError::Domain(_))
    ));
}

#[test]
fn original_child_domains_and_arity_are_checked_before_absolute_value() {
    let variables = [Symbol::new("x")];
    let domain = [RealBall::from_i64(1)];
    let invalid = function("log", Expr::from_i64(-1));
    for argument in [
        invalid.clone(),
        Expr::Mul(vec![Expr::from_i64(0), invalid.clone()]),
        power(invalid, 0),
    ] {
        let source = function("abs", argument);
        assert!(matches!(
            enclose_residual_jacobian(std::slice::from_ref(&source), &variables, &domain, 24),
            Err(RootBoxError::Domain(_))
        ));
        assert!(matches!(
            enclose_objective(&source, &variables, &domain, 24),
            Err(RootBoxError::Domain(_))
        ));
    }
    for tag in ["abs", "Abs"] {
        for arity in [0, 2, 3] {
            let source = Expr::Function(tag.into(), vec![Expr::from_i64(1); arity]);
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
}

fn witness(center: i64, preconditioner: BigRational) -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(q(center, 1), q(1, 4)).unwrap()],
        center: vec![q(center, 1)],
        preconditioner: vec![vec![preconditioner]],
        precision_digits: 24,
    }
}

#[test]
fn each_smooth_absolute_branch_can_certify_a_root_and_reject_false_witnesses() {
    let variables = [Symbol::new("x")];
    let source = function("abs", Expr::Sym(variables[0].clone())) - Expr::from_i64(2);
    for (center, sign) in [(2, 1), (-2, -1)] {
        let proposal = witness(center, q(sign, 1));
        let accepted =
            certify_root_box(std::slice::from_ref(&source), &variables, &proposal).unwrap();
        assert_eq!(accepted.root_enclosure(), [RealBall::from_i64(center)]);
        assert_eq!(accepted.contraction_bound(), &q(0, 1));
        assert_eq!(accepted.residuals(), std::slice::from_ref(&source));
        assert_eq!(accepted.witness(), &proposal);
        assert!(
            certify_root_box(&[source.clone() + Expr::from_i64(1)], &variables, &proposal).is_err()
        );
        assert!(
            certify_root_box(
                std::slice::from_ref(&source),
                &variables,
                &witness(center, q(-sign, 1))
            )
            .is_err()
        );
    }
    assert!(matches!(
        certify_root_box(&[source], &variables, &witness(0, q(1, 1))),
        Err(RootBoxError::Domain(_))
    ));
}

#[test]
fn squared_absolute_residual_has_two_separate_certified_local_minima() {
    let variables = [Symbol::new("x")];
    let residual = function("Abs", Expr::Sym(variables[0].clone())) - Expr::from_i64(2);
    let source = power(residual, 2);
    for center in [2, -2] {
        let proposal = witness(center, q(1, 2));
        let accepted = certify_stationary_point(&source, &variables, &proposal).unwrap();
        assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
        assert_eq!(
            accepted.stationary_enclosure(),
            [RealBall::from_i64(center)]
        );
        assert_eq!(accepted.hessian_enclosure(), [vec![RealBall::from_i64(2)]]);
        assert_eq!(accepted.objective(), &source);
        assert_eq!(accepted.witness(), &proposal);
        assert!(certify_stationary_point(&source, &variables, &witness(center, q(-1, 2))).is_err());
    }
    assert!(matches!(
        certify_stationary_point(&source, &variables, &witness(0, q(1, 2))),
        Err(RootBoxError::Domain(_))
    ));
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
    let variables = [Symbol::new("x")];
    let source = function("abs", Expr::Sym(variables[0].clone())) - Expr::from_i64(2);
    let domain = [interval(q(-9, 4), q(-7, 4))];
    match lane {
        Lane::First => {
            metered_enclose_residual_jacobian(&[source], &variables, &domain, 24, meter).map(|_| ())
        }
        Lane::Second => {
            metered_enclose_objective(&source, &variables, &domain, 24, meter).map(|_| ())
        }
        Lane::Root => {
            verify_root_box(&[source], &variables, &witness(-2, q(-1, 1)), meter).map(|_| ())
        }
        Lane::Stationary => {
            verify_stationary_point(&power(source, 2), &variables, &witness(-2, q(1, 2)), meter)
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
        assert!(!self.stopped, "charged after terminal cancellation");
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after terminal cancellation");
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "continued after terminal cancellation");
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
fn cancellation_at_each_safe_point_prevents_receipt_publication() {
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
fn cumulative_limits_include_admission_arithmetic_and_publication() {
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
            assert!(matches!(metered_case(lane, &mut exact),
                Err(RootBoxError::Meter(MeterError::Budget(BudgetError::Exhausted { dimension: exhausted, .. })))
                if exhausted == dimension));
            assert_eq!(exact.snapshot(), after);
            assert_eq!(exact.verifier_remaining(), RESERVE);
            limits.dimensions[dimension.index()] = used - 1;
            let mut short = Budget::new(limits);
            assert!(matches!(metered_case(lane, &mut short),
                Err(RootBoxError::Meter(MeterError::Budget(BudgetError::Exhausted { dimension: exhausted, .. })))
                if exhausted == dimension));
            assert_eq!(short.verifier_remaining(), RESERVE);
        }
    }
}
