use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BallError, BigInt, BigRational, Constant, Expr, LambertWError, RealBall};
use num_traits::Zero;
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn tolerance(digits: u32) -> BigRational {
    BigRational::new(BigInt::one(), BigInt::from(10).pow(digits))
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn lambert(argument: Expr, branch: Option<i64>) -> Expr {
    let mut arguments = vec![argument];
    if let Some(branch) = branch {
        arguments.push(Expr::from_i64(branch));
    }
    Expr::Function("LambertW".into(), arguments)
}

fn exp(argument: Expr) -> Expr {
    Expr::Function("exp".into(), vec![argument])
}

#[test]
fn both_real_branches_invert_independent_forward_enclosures() {
    for (value, branch) in [
        (q(1, 2), 0),
        (q(1, 1), 0),
        (q(3, 1), 0),
        (q(-1, 2), 0),
        (q(-2, 1), -1),
        (q(-3, 1), -1),
    ] {
        let point = RealBall::exact(value.clone());
        // Use the pre-existing absolute exp kernel for the forward map.
        let original = point.mul(&point.exp(48).unwrap());
        let recovered = original.lambert_w(branch, 24).unwrap();
        assert!(recovered.contains(&value), "branch={branch}, value={value}");
        assert!(recovered.width() < tolerance(23));
        if branch == -1 {
            assert!(recovered.upper() < q(-1, 1));
        }
    }
}

#[test]
fn rational_arguments_have_tight_brackets_with_independent_residual_signs() {
    for (argument, branch, coarse_low, coarse_high) in [
        (q(1, 1), 0, q(56, 100), q(57, 100)),
        (q(-1, 4), 0, q(-1, 2), q(-1, 3)),
        (q(-1, 4), -1, q(-3, 1), q(-2, 1)),
    ] {
        let result = RealBall::exact(argument.clone())
            .lambert_w(branch, 24)
            .unwrap();
        assert!(result.lower() > coarse_low);
        assert!(result.upper() < coarse_high);
        assert!(result.width() <= tolerance(26));
        let residual = |value: BigRational| {
            let point = RealBall::exact(value);
            point
                .mul(&point.exp(80).unwrap())
                .sub(&RealBall::exact(argument.clone()))
        };
        let lower = residual(result.lower());
        let upper = residual(result.upper());
        if branch == 0 {
            assert!(lower.upper() < BigRational::zero());
            assert!(upper.lower() > BigRational::zero());
        } else {
            assert!(lower.lower() > BigRational::zero());
            assert!(upper.upper() < BigRational::zero());
        }
    }
}

#[test]
fn principal_value_reaches_the_declared_precision_ceiling() {
    let digits = RealBall::MAX_LAMBERT_W_PRECISION_DIGITS;
    let result = RealBall::from_i64(1).lambert_w(0, digits).unwrap();
    assert!(result.lower() > q(56, 100));
    assert!(result.upper() < q(57, 100));
    assert!(result.width() <= tolerance(digits + 2));

    // This pre-existing absolute exponential is independent of Lambert's
    // relative-exp inverse iteration. Strict endpoint signs and monotonicity
    // on the positive branch establish a root inside the narrow interval.
    let residual = |value: BigRational| {
        let point = RealBall::exact(value);
        point
            .mul(&point.exp(220).unwrap())
            .sub(&RealBall::from_i64(1))
    };
    assert!(residual(result.lower()).upper() < BigRational::zero());
    assert!(residual(result.upper()).lower() > BigRational::zero());
}

#[test]
fn wide_inputs_retain_the_complete_monotone_branch_image() {
    let low = q(-3, 10);
    let high = q(-1, 5);
    let input = interval(low.clone(), high.clone());
    for branch in [0, -1] {
        let result = input.lambert_w(branch, 18).unwrap();
        let first = RealBall::exact(low.clone()).lambert_w(branch, 18).unwrap();
        let last = RealBall::exact(high.clone()).lambert_w(branch, 18).unwrap();
        assert!(result.contains_ball(&first));
        assert!(result.contains_ball(&last));
        assert!(result.width() > q(1, 10));
        if branch == 0 {
            assert_eq!(result.lower(), first.lower());
            assert_eq!(result.upper(), last.upper());
        } else {
            assert_eq!(result.lower(), last.lower());
            assert_eq!(result.upper(), first.upper());
        }
    }
    let crossing = interval(q(-1, 4), q(1, 4)).lambert_w(0, 18).unwrap();
    assert!(crossing.contains_zero());
}

#[test]
fn real_domain_admission_uses_every_input_value() {
    for branch in [0, -1] {
        for invalid in [
            RealBall::exact(q(-1, 1)),
            RealBall::exact(q(-368, 1000)),
            interval(q(-368, 1000), q(-36, 100)),
        ] {
            assert!(invalid.lambert_w(branch, 18).is_err());
        }
        let near = RealBall::exact(q(-367879, 1_000_000))
            .lambert_w(branch, 18)
            .unwrap();
        if branch == 0 {
            assert!(near.lower() > q(-1, 1));
        } else {
            assert!(near.upper() < q(-1, 1));
        }
    }
    for invalid in [
        RealBall::from_i64(0),
        RealBall::from_i64(1),
        interval(q(-1, 4), q(1, 100)),
        interval(q(-1, 4), q(0, 1)),
    ] {
        assert!(invalid.lambert_w(-1, 18).is_err());
    }
    assert_eq!(
        RealBall::from_i64(0).lambert_w(0, 18).unwrap(),
        RealBall::from_i64(0)
    );
}

#[test]
fn tiny_principal_values_keep_relative_accuracy_and_lower_tails_remain_finite() {
    let tiny = tolerance(200);
    for sign in [1, -1] {
        let argument = tiny.clone() * q(sign, 1);
        let result = RealBall::exact(argument).lambert_w(0, 30).unwrap();
        assert!(!result.contains_zero());
        assert!(result.width() <= &tiny * tolerance(32));
    }
    let lower = RealBall::exact(-tolerance(100)).lambert_w(-1, 18).unwrap();
    assert!(lower.lower() > q(-240, 1));
    assert!(lower.upper() < q(-230, 1));
    assert!(lower.width() <= tolerance(20));
}

#[test]
fn exact_branchpoint_and_named_specials_survive_native_expression_evaluation() {
    let branchpoints = [
        Expr::Mul(vec![Expr::from_i64(-1), exp(Expr::from_i64(-1))]),
        Expr::Mul(vec![
            Expr::from_i64(-1),
            Expr::Pow(
                Arc::new(Expr::Const(Constant::E)),
                Arc::new(Expr::from_i64(-1)),
            ),
        ]),
    ];
    for argument in branchpoints {
        for branch in [None, Some(0), Some(-1)] {
            assert_eq!(
                lambert(argument.clone(), branch)
                    .eval_real_ball(24)
                    .unwrap(),
                RealBall::from_i64(-1)
            );
        }
    }
    assert_eq!(
        lambert(Expr::Const(Constant::E), None)
            .eval_real_ball(24)
            .unwrap(),
        RealBall::from_i64(1)
    );
    assert_eq!(
        lambert(Expr::from_i64(0), Some(0))
            .evalf_decimal(15)
            .unwrap(),
        "0"
    );
    assert!(
        lambert(Expr::from_i64(0), Some(-1))
            .eval_real_ball(24)
            .is_err()
    );
}

#[test]
fn native_decimal_and_complex_real_dispatch_support_both_real_branches() {
    let principal = lambert(Expr::from_i64(1), None);
    assert_eq!(principal.evalf_decimal(12).unwrap(), "5.67143290410e-1");
    assert_eq!(
        principal.eval_real_ball(24).unwrap(),
        lambert(Expr::from_i64(1), Some(0))
            .eval_real_ball(24)
            .unwrap()
    );
    let lower = lambert(Expr::Rational(q(-1, 4)), Some(-1));
    let complex = lower.eval_complex_ball(18).unwrap();
    assert_eq!(complex.imag(), &RealBall::from_i64(0));
    assert_eq!(complex.real(), &lower.eval_real_ball(18).unwrap());
    assert!(
        lambert(Expr::Const(Constant::I), None)
            .eval_complex_ball(18)
            .is_err()
    );
    assert!(
        lambert(Expr::from_i64(1), Some(2))
            .eval_complex_ball(18)
            .is_err()
    );
}

#[test]
fn exact_special_folding_cannot_discard_held_invalid_argument_subtrees() {
    let pole = Expr::Function("log".into(), vec![Expr::from_i64(0)]);
    for argument in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        Expr::Pow(Arc::new(pole), Arc::new(Expr::from_i64(0))),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Const(Constant::I)]),
    ] {
        for branch in [None, Some(0), Some(-1)] {
            assert!(
                lambert(argument.clone(), branch)
                    .eval_real_ball(18)
                    .is_err()
            );
        }
    }
    let nonliteral_branch = Expr::Function(
        "LambertW".into(),
        vec![Expr::from_i64(1), Expr::Add(vec![Expr::from_i64(0)])],
    );
    assert!(nonliteral_branch.eval_real_ball(18).is_err());
}

#[test]
fn unsupported_branches_precisions_and_numeric_heights_refuse() {
    let point = RealBall::from_i64(1);
    for branch in [-2, 1, 2, i32::MIN, i32::MAX] {
        assert!(point.lambert_w(branch, 18).is_err());
    }
    for digits in [0, RealBall::MAX_LAMBERT_W_PRECISION_DIGITS + 1] {
        assert!(point.lambert_w(0, digits).is_err());
    }
    let oversized = RealBall::exact(BigRational::from_integer(BigInt::one() << 4096u32));
    assert!(matches!(
        oversized.metered_lambert_w(0, 18, &mut Unbounded),
        Err(LambertWError::Ball(
            BallError::ArgumentOutsideDeclaredEnvelope(_)
        ))
    ));
}

#[derive(Default)]
struct Checkpoints {
    calls: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.calls += 1;
        if self.cancel_at.is_some_and(|at| self.calls >= at) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn meters_are_shared_across_endpoints_and_repeated_calls() {
    let input = interval(q(1, 1), q(2, 1));
    let allowance = 1_000_000_000u64;
    let mut measured = Budget::new(BudgetLimits::uniform(allowance, 0));
    input.metered_lambert_w(0, 8, &mut measured).unwrap();
    let used = allowance - measured.remaining(Dimension::ComputeSteps);
    assert!(used > 0);
    assert!(measured.remaining(Dimension::MemoryBytes) < allowance);
    assert!(measured.remaining(Dimension::AllocationCount) < allowance);

    let mut limits = BudgetLimits::uniform(allowance, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 2 * used - 1;
    let mut shared = Budget::new(limits);
    input.metered_lambert_w(0, 8, &mut shared).unwrap();
    assert!(matches!(
        input.metered_lambert_w(0, 8, &mut shared),
        Err(LambertWError::Meter(MeterError::Budget(_)))
    ));
    let mut insufficient = Budget::new(BudgetLimits::uniform(1, 0));
    let before = insufficient.snapshot();
    assert!(matches!(
        input.metered_lambert_w(0, 8, &mut insufficient),
        Err(LambertWError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(insufficient.snapshot(), before);
}

#[test]
fn cancellation_before_during_and_at_publication_never_returns_a_partial_image() {
    let input = interval(q(-3, 10), q(-1, 5));
    let mut baseline = Checkpoints::default();
    input.metered_lambert_w(-1, 8, &mut baseline).unwrap();
    assert!(baseline.calls > 3);
    for cancel_at in [1, baseline.calls / 2, baseline.calls] {
        let mut meter = Checkpoints {
            calls: 0,
            cancel_at: Some(cancel_at),
        };
        assert!(matches!(
            input.metered_lambert_w(-1, 8, &mut meter),
            Err(LambertWError::Meter(MeterError::Cancelled))
        ));
    }
}
