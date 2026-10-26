//! Exact leading-term, directional-limit, original-domain, and budget checks.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::gruntz::{Direction, LimitError};
use fsym_calculus::rational_limits::{
    RationalLimitError, metered_rational_local_order, rational_expression_limit,
    rational_local_order,
};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_limit::{
    LocalDirection, RationalDirectionalLimit, RationalLimitPoint, RationalLocalVerificationError,
    verify_rational_local_order,
};
use num_traits::{One, Zero};
use std::sync::Arc;

fn rational(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn poly(values: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(
        Symbol::new("x"),
        values.iter().map(|&v| rational(v, 1)).collect(),
    )
}

fn raw_power(base: Expr, power: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(power)))
}

fn raw_fraction(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, raw_power(denominator, -1)])
}

fn point(value: BigRational) -> RationalLimitPoint {
    RationalLimitPoint::Finite(value)
}

#[test]
fn rational_local_orders_include_shifted_cancellation_and_zero() {
    // Independently build N=(x-a)^n*(2x+3), D=(x-a)^d*(x+5).
    // At the selected centers the units never vanish.
    for center in [rational(-2, 1), rational(0, 1), rational(3, 2)] {
        let factor =
            UnivariatePoly::new(Symbol::new("x"), vec![-center.clone(), BigRational::one()]);
        for n in 0..=4u32 {
            for d in 0..=4u32 {
                let numerator = factor.pow(n).unwrap().mul(&poly(&[3, 2])).unwrap();
                let denominator = factor.pow(d).unwrap().mul(&poly(&[5, 1])).unwrap();
                let analysis =
                    rational_local_order(&numerator, &denominator, &point(center.clone())).unwrap();
                assert_eq!(
                    analysis.verification().order(),
                    Some(i64::from(n) - i64::from(d))
                );
                let expected = (rational(2, 1) * center.clone() + rational(3, 1))
                    / (center.clone() + rational(5, 1));
                assert_eq!(analysis.verification().leading_coefficient(), &expected);
                verify_rational_local_order(
                    &numerator.coeffs,
                    &denominator.coeffs,
                    &point(center.clone()),
                    analysis.certificate(),
                    &mut Unbounded,
                )
                .unwrap();
            }
        }
    }
    let zero = rational_local_order(&poly(&[0]), &poly(&[-2, 1]), &point(rational(2, 1))).unwrap();
    assert_eq!(zero.verification().order(), None);
    assert_eq!(
        zero.verification().limit(LocalDirection::Both),
        RationalDirectionalLimit::Finite(BigRational::zero())
    );
}

#[test]
fn public_limits_distinguish_one_sided_poles_and_two_sided_failure() {
    let x = Symbol::new("x");
    let center = Expr::Rational(rational(3, 2));
    let base = Expr::Add(vec![Expr::Sym(x.clone()), Expr::Rational(rational(-3, 2))]);
    for power in [1, 2, 3, 4] {
        for sign in [-1, 1] {
            let expression = raw_fraction(Expr::from_i64(sign), raw_power(base.clone(), power));
            let right = if sign < 0 {
                Constant::NegativeInfinity
            } else {
                Constant::Infinity
            };
            let left = if (sign < 0) ^ (power % 2 != 0) {
                Constant::NegativeInfinity
            } else {
                Constant::Infinity
            };
            assert_eq!(
                fsym_calculus::limit_dir(&expression, &x, &center, Direction::Plus).unwrap(),
                Expr::Const(right)
            );
            assert_eq!(
                fsym_calculus::limit_dir(&expression, &x, &center, Direction::Minus).unwrap(),
                Expr::Const(left)
            );
            let both = fsym_calculus::gruntz::limit(&expression, &x, &center, Direction::Both);
            if power % 2 == 0 {
                assert_eq!(both.unwrap(), Expr::Const(right));
            } else {
                assert!(matches!(both, Err(LimitError::DoesNotExist(_))));
            }
        }
    }
}

#[test]
fn infinity_uses_exact_degree_and_negative_axis_parity() {
    let x = Symbol::new("x");
    let cases = [
        (
            poly(&[1, 2]),
            poly(&[3, 0, 1]),
            Expr::from_i64(0),
            Expr::from_i64(0),
        ),
        (
            poly(&[1, 2, 6]),
            poly(&[-1, 0, 4]),
            Expr::Rational(rational(3, 2)),
            Expr::Rational(rational(3, 2)),
        ),
        (
            poly(&[0, 0, 0, -2]),
            poly(&[1, 0, 1]),
            Expr::Const(Constant::NegativeInfinity),
            Expr::Const(Constant::Infinity),
        ),
        (
            poly(&[0, 0, 0, 0, -2]),
            poly(&[1, 0, 1]),
            Expr::Const(Constant::NegativeInfinity),
            Expr::Const(Constant::NegativeInfinity),
        ),
    ];
    for (n, d, positive, negative) in cases {
        let expression = raw_fraction(n.to_expr(), d.to_expr());
        assert_eq!(
            fsym_calculus::limit(&expression, &x, &Expr::Const(Constant::Infinity)).unwrap(),
            positive
        );
        assert_eq!(
            fsym_calculus::limit(&expression, &x, &Expr::Const(Constant::NegativeInfinity))
                .unwrap(),
            negative
        );
    }
}

#[test]
fn high_order_rational_cancellation_does_not_use_a_truncated_series() {
    let x = Symbol::new("x");
    let base = Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(-1)]);
    // These cancellation orders exceed the general series engine's 64-term
    // envelope. Exact local factorization needs no heuristic precision retry.
    for (n, d, expected) in [
        (81, 80, Expr::from_i64(0)),
        (80, 80, Expr::from_i64(1)),
        (80, 81, Expr::Const(Constant::Infinity)),
    ] {
        let expression = raw_fraction(raw_power(base.clone(), n), raw_power(base.clone(), d));
        assert_eq!(
            fsym_calculus::limit(&expression, &x, &Expr::from_i64(1)).unwrap(),
            expected
        );
    }
}

#[test]
fn exact_tiny_coefficients_survive_cancellation_and_determine_pole_sign() {
    let x = Symbol::new("x");
    let epsilon = BigRational::new(BigInt::one(), BigInt::from(10).pow(100));
    let numerator = Expr::Add(vec![
        Expr::Rational(-epsilon.clone()),
        Expr::Sym(x.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(x.clone())]),
    ]);
    let expression = raw_fraction(numerator, Expr::Sym(x.clone()));
    assert_eq!(
        fsym_calculus::limit(&expression, &x, &Expr::from_i64(0)).unwrap(),
        Expr::Const(Constant::NegativeInfinity)
    );
    let analysis = rational_local_order(
        &UnivariatePoly::new(x, vec![-epsilon.clone()]),
        &poly(&[0, 1]),
        &point(BigRational::zero()),
    )
    .unwrap();
    assert_eq!(analysis.verification().leading_coefficient(), &-epsilon);
}

#[test]
fn original_undefined_terms_cannot_disappear_behind_zero_or_power_zero() {
    let x = Symbol::new("x");
    let zero_polynomial = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(x.clone())]),
    ]);
    let invalid = [
        raw_fraction(Expr::from_i64(1), zero_polynomial.clone()),
        Expr::Mul(vec![
            Expr::from_i64(0),
            raw_power(zero_polynomial.clone(), -1),
        ]),
        raw_power(raw_power(zero_polynomial, -1), 0),
        raw_power(Expr::from_i64(0), 0),
        raw_power(raw_fraction(Expr::from_i64(0), Expr::from_i64(0)), 0),
    ];
    for expression in invalid {
        assert!(
            rational_expression_limit(
                &expression,
                &x,
                &point(BigRational::zero()),
                LocalDirection::Positive
            )
            .is_err()
        );
        assert!(
            fsym_calculus::gruntz::limit(&expression, &x, &Expr::from_i64(0), Direction::Plus)
                .is_err()
        );
        assert!(fsym_calculus::limit(&expression, &x, &Expr::from_i64(0)).is_err());
    }
    // 0/x is defined on a punctured neighborhood even though x=0 itself
    // stays excluded. No value at the excluded point is being claimed.
    let valid = raw_fraction(Expr::from_i64(0), Expr::Sym(x.clone()));
    assert_eq!(
        fsym_calculus::limit_dir(&valid, &x, &Expr::from_i64(0), Direction::Both).unwrap(),
        Expr::from_i64(0)
    );
}

#[test]
fn malformed_and_resource_exhausted_rational_inputs_refuse_without_fallback() {
    let x = Symbol::new("x");
    let malformed = UnivariatePoly {
        gen_sym: x.clone(),
        coeffs: vec![],
    };
    assert!(rational_local_order(&malformed, &poly(&[1]), &point(BigRational::zero())).is_err());
    assert!(rational_local_order(&poly(&[1]), &poly(&[0]), &point(BigRational::zero())).is_err());
    let other_variable = UnivariatePoly::one(Symbol::new("y"));
    assert!(
        rational_local_order(&poly(&[1]), &other_variable, &point(BigRational::zero())).is_err()
    );
    let huge = UnivariatePoly::new(
        x.clone(),
        vec![BigRational::from_integer(BigInt::one() << 4097u32)],
    );
    assert!(rational_local_order(&huge, &poly(&[1]), &point(BigRational::zero())).is_err());
    let too_many = UnivariatePoly::new(x.clone(), vec![BigRational::one(); 130]);
    assert!(rational_local_order(&too_many, &poly(&[1]), &point(BigRational::zero())).is_err());
    for expression in [
        raw_power(Expr::Sym(x.clone()), 129),
        raw_power(raw_power(Expr::Sym(x.clone()), 64), 64),
        Expr::Add(vec![]),
    ] {
        assert!(
            rational_expression_limit(
                &expression,
                &x,
                &point(BigRational::zero()),
                LocalDirection::Positive
            )
            .is_err()
        );
        assert!(fsym_calculus::limit(&expression, &x, &Expr::from_i64(0)).is_err());
    }
    let mut deep = Expr::Sym(x.clone());
    for _ in 0..66 {
        deep = Expr::Mul(vec![deep]);
    }
    assert!(
        rational_expression_limit(
            &deep,
            &x,
            &point(BigRational::zero()),
            LocalDirection::Positive
        )
        .is_err()
    );
    let symbolic = Expr::Sym(Symbol::new("a"));
    assert!(
        rational_expression_limit(
            &symbolic,
            &x,
            &point(BigRational::zero()),
            LocalDirection::Positive
        )
        .unwrap()
        .is_none()
    );
    let sine_quotient = raw_fraction(
        Expr::Function("sin".into(), vec![Expr::Sym(x.clone())]),
        Expr::Sym(x.clone()),
    );
    assert_eq!(
        fsym_calculus::limit(&sine_quotient, &x, &Expr::from_i64(0)).unwrap(),
        Expr::from_i64(1)
    );
    let nonrational_large_power =
        Expr::Function("exp".into(), vec![raw_power(Expr::Sym(x.clone()), 129)]);
    assert!(
        rational_expression_limit(
            &nonrational_large_power,
            &x,
            &point(BigRational::zero()),
            LocalDirection::Positive,
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(
        fsym_calculus::limit(&nonrational_large_power, &x, &Expr::from_i64(0)).unwrap(),
        Expr::from_i64(1),
    );
}

#[derive(Default)]
struct CancelMeter {
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CancelMeter {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        assert_ne!(amount, 0);
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn generation_and_verification_stop_without_publishing_a_partial_analysis() {
    let n = poly(&[8, -28, 30, -13, 2]);
    let d = poly(&[28, -32, 11, -1]);
    let center = point(rational(2, 1));
    let mut generation = CancelMeter::default();
    let mut verification = CancelMeter::default();
    metered_rational_local_order(&n, &d, &center, &mut generation, &mut verification).unwrap();
    for checkpoint in [1, generation.checkpoints / 2, generation.checkpoints] {
        let mut cancelled = CancelMeter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_rational_local_order(&n, &d, &center, &mut cancelled, &mut Unbounded),
            Err(RationalLimitError::Meter(MeterError::Cancelled))
        ));
    }
    for checkpoint in [1, verification.checkpoints / 2, verification.checkpoints] {
        let mut cancelled = CancelMeter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_rational_local_order(&n, &d, &center, &mut Unbounded, &mut cancelled),
            Err(RationalLimitError::Verification(
                RationalLocalVerificationError::Meter(MeterError::Cancelled)
            ))
        ));
    }
    let mut generation_budget = Budget::new(BudgetLimits::uniform(1, 0));
    let mut unused_verifier = CancelMeter::default();
    assert!(
        metered_rational_local_order(
            &n,
            &d,
            &center,
            &mut generation_budget,
            &mut unused_verifier
        )
        .is_err()
    );
    assert_eq!(unused_verifier.checkpoints, 0);
    let mut verifier_budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_rational_local_order(&n, &d, &center, &mut Unbounded, &mut verifier_budget),
        Err(RationalLimitError::Verification(
            RationalLocalVerificationError::Meter(_)
        ))
    ));
}
