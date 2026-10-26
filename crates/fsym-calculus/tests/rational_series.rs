//! Exact rational series coefficients, public dispatch, and trust boundaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_limits::{
    RationalLimitError, metered_rational_series, rational_expression_series, rational_series,
};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::rational_limit::{
    RationalLimitPoint, RationalLocalVerificationError, verify_rational_series,
};
use num_traits::{One, Zero};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn poly(values: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(Symbol::new("x"), values.iter().map(|&v| q(v, 1)).collect())
}
fn raw_power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn fraction(n: Expr, d: Expr) -> Expr {
    Expr::Mul(vec![n, raw_power(d, -1)])
}
fn origin() -> RationalLimitPoint {
    RationalLimitPoint::Finite(BigRational::zero())
}
fn coeff(expr: &Expr) -> BigRational {
    match expr {
        Expr::Integer(n) => BigRational::from_integer(n.clone()),
        Expr::Rational(r) => r.clone(),
        _ => panic!("expected exact rational coefficient: {expr}"),
    }
}

#[test]
fn public_rational_series_and_taylor_exceed_the_old_precision_caps() {
    let x = Symbol::new("x");
    let expression = fraction(poly(&[1]).to_expr(), poly(&[1, -1]).to_expr());
    for order in [80, 256] {
        let (terms, remainder) =
            fsym_calculus::series_expansion(&expression, &x, &Expr::from_i64(0), order).unwrap();
        assert_eq!(terms.len(), order as usize);
        assert_eq!(remainder, Some(q(order, 1)));
        for (index, (coefficient, exponent)) in terms.iter().enumerate() {
            assert_eq!(coeff(coefficient), BigRational::one());
            assert_eq!(exponent, &q(index as i64, 1));
        }
    }
    let taylor = fsym_calculus::taylor(&expression, &x, &Expr::from_i64(0), 80).unwrap();
    let actual = UnivariatePoly::from_expr(&taylor, &x).unwrap();
    assert_eq!(actual.coeffs, vec![BigRational::one(); 81]);
    assert!(fsym_calculus::series_expansion(&expression, &x, &Expr::from_i64(0), 257).is_err());
}

#[test]
fn repeated_pole_recurrence_matches_closed_binomial_coefficients() {
    let numerator = poly(&[1]);
    let denominator = poly(&[1, -1]).pow(3).unwrap();
    let analysis = rational_series(&numerator, &denominator, &origin(), 100).unwrap();
    let expected: Vec<BigRational> = (0..100).map(|k| q((k + 1) * (k + 2), 2)).collect();
    assert_eq!(analysis.verification().coefficients(), expected);
    assert_eq!(analysis.verification().remainder_order(), Some(100));
    verify_rational_series(
        &numerator.coeffs,
        &denominator.coeffs,
        &origin(),
        100,
        analysis.certificate(),
        &mut Unbounded,
    )
    .unwrap();
    let fibonacci = rational_series(&poly(&[1]), &poly(&[1, -1, -1]), &origin(), 12).unwrap();
    assert_eq!(
        fibonacci.verification().coefficients(),
        poly(&[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]).coeffs
    );
}

#[test]
fn shifted_rational_centers_and_both_infinity_coordinates_are_exact() {
    let x = Symbol::new("x");
    let expression = fraction(Expr::from_i64(1), poly(&[-2, 1]).to_expr());
    let (terms, remainder) =
        fsym_calculus::series_expansion(&expression, &x, &Expr::Rational(q(3, 2)), 12).unwrap();
    assert_eq!(remainder, Some(q(12, 1)));
    for (index, (coefficient, exponent)) in terms.iter().enumerate() {
        assert_eq!(
            coeff(coefficient),
            BigRational::from_integer(-(BigInt::one() << (index + 1) as u32))
        );
        assert_eq!(exponent, &q(index as i64, 1));
    }
    let infinity_expression = fraction(Expr::from_i64(1), poly(&[-1, 1]).to_expr());
    for (at, negative) in [
        (Constant::Infinity, false),
        (Constant::NegativeInfinity, true),
    ] {
        let (terms, remainder) =
            fsym_calculus::series_expansion(&infinity_expression, &x, &Expr::Const(at), 8).unwrap();
        assert_eq!(terms.len(), 7);
        assert_eq!(remainder, Some(q(8, 1)));
        for (index, (coefficient, exponent)) in terms.iter().enumerate() {
            let k = index as i64 + 1;
            assert_eq!(
                coeff(coefficient),
                q(if negative && k % 2 != 0 { -1 } else { 1 }, 1)
            );
            assert_eq!(exponent, &q(k, 1));
        }
    }
    // Negative exponents also change parity under t=-u: at -infinity,
    // x^3/(1+x) = u^-2/(1-u), where u=-1/x.
    let growing = fraction(poly(&[0, 0, 0, 1]).to_expr(), poly(&[1, 1]).to_expr());
    let (terms, remainder) =
        fsym_calculus::series_expansion(&growing, &x, &Expr::Const(Constant::NegativeInfinity), 4)
            .unwrap();
    assert_eq!(
        terms,
        (-2..4)
            .map(|k| (Expr::from_i64(1), q(k, 1)))
            .collect::<Vec<_>>()
    );
    assert_eq!(remainder, Some(q(4, 1)));
}

#[test]
fn laurent_poles_negative_orders_and_high_cancellation_keep_requested_precision() {
    let x = Symbol::new("x");
    let expression = fraction(poly(&[2, 3]).to_expr(), poly(&[0, 0, 1, -1]).to_expr());
    let (terms, order) =
        fsym_calculus::series_expansion(&expression, &x, &Expr::from_i64(0), 3).unwrap();
    assert_eq!(order, Some(q(3, 1)));
    assert_eq!(
        terms
            .iter()
            .map(|(c, e)| (coeff(c), e.clone()))
            .collect::<Vec<_>>(),
        (0..5)
            .map(|i| (q(if i == 0 { 2 } else { 5 }, 1), q(i - 2, 1)))
            .collect::<Vec<_>>()
    );
    let (terms, order) =
        fsym_calculus::series_expansion(&expression, &x, &Expr::from_i64(0), -1).unwrap();
    assert_eq!(terms, vec![(Expr::from_i64(2), q(-2, 1))]);
    assert_eq!(order, Some(q(-1, 1)));
    let (terms, order) =
        fsym_calculus::series_expansion(&expression, &x, &Expr::from_i64(0), -2).unwrap();
    assert!(terms.is_empty());
    assert_eq!(order, Some(q(-2, 1)));

    let local = Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(-1)]);
    let cancelled = fraction(
        raw_power(local.clone(), 80),
        Expr::Mul(vec![
            raw_power(local.clone(), 81),
            Expr::Add(vec![
                Expr::from_i64(1),
                Expr::Mul(vec![Expr::from_i64(-1), local]),
            ]),
        ]),
    );
    let (terms, order) =
        fsym_calculus::series_expansion(&cancelled, &x, &Expr::from_i64(1), 5).unwrap();
    assert_eq!(order, Some(q(5, 1)));
    assert_eq!(
        terms,
        (-1..5)
            .map(|k| (Expr::from_i64(1), q(k, 1)))
            .collect::<Vec<_>>()
    );
}

#[test]
fn exact_finite_expansions_and_pointwise_taylor_domains_stay_distinct() {
    let x = Symbol::new("x");
    let removable = fraction(poly(&[-1, 0, 1]).to_expr(), poly(&[-1, 1]).to_expr());
    let (terms, order) =
        fsym_calculus::series_expansion(&removable, &x, &Expr::from_i64(1), 8).unwrap();
    assert_eq!(
        terms,
        vec![(Expr::from_i64(2), q(0, 1)), (Expr::from_i64(1), q(1, 1))]
    );
    assert_eq!(order, None);
    assert!(fsym_calculus::taylor(&removable, &x, &Expr::from_i64(1), 8).is_err());
    // A nested reciprocal has an original hole even though its flattened
    // fraction is the polynomial x. Taylor must inspect original children.
    let nested_reciprocal = raw_power(raw_power(Expr::Sym(x.clone()), -1), -1);
    let (terms, order) =
        fsym_calculus::series_expansion(&nested_reciprocal, &x, &Expr::from_i64(0), 8).unwrap();
    assert_eq!(terms, vec![(Expr::from_i64(1), q(1, 1))]);
    assert_eq!(order, None);
    assert!(fsym_calculus::taylor(&nested_reciprocal, &x, &Expr::from_i64(0), 8).is_err());
    let exact_zero = fraction(Expr::from_i64(0), Expr::Sym(x.clone()));
    let (terms, order) =
        fsym_calculus::series_expansion(&exact_zero, &x, &Expr::from_i64(0), -5).unwrap();
    assert!(terms.is_empty());
    assert_eq!(order, None);
    let monomial = raw_power(Expr::Sym(x.clone()), 80);
    let (terms, order) =
        fsym_calculus::series_expansion(&monomial, &x, &Expr::from_i64(0), 81).unwrap();
    assert_eq!(terms, vec![(Expr::from_i64(1), q(80, 1))]);
    assert_eq!(order, None);
}

#[test]
fn original_invalid_syntax_and_unsupported_nonrational_fallthrough_are_preserved() {
    let x = Symbol::new("x");
    let zero = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(x.clone())]),
    ]);
    for invalid in [
        Expr::Mul(vec![Expr::from_i64(0), raw_power(zero.clone(), -1)]),
        raw_power(raw_power(zero, -1), 0),
        raw_power(Expr::from_i64(0), 0),
    ] {
        assert!(fsym_calculus::series_expansion(&invalid, &x, &Expr::from_i64(0), 4).is_err());
        assert!(fsym_calculus::taylor(&invalid, &x, &Expr::from_i64(0), 4).is_err());
    }
    let nonrational = Expr::Function("exp".into(), vec![raw_power(Expr::Sym(x.clone()), 129)]);
    assert!(
        rational_expression_series(&nonrational, &x, &origin(), 2)
            .unwrap()
            .is_none()
    );
    let (terms, remainder) =
        fsym_calculus::series_expansion(&nonrational, &x, &Expr::from_i64(0), 2).unwrap();
    assert_eq!(terms, vec![(Expr::from_i64(1), q(0, 1))]);
    assert!(remainder.is_some());
    let sine = Expr::Function("sin".into(), vec![Expr::Sym(x.clone())]);
    let (terms, _) = fsym_calculus::series_expansion(&sine, &x, &Expr::from_i64(0), 6).unwrap();
    assert_eq!(
        terms
            .iter()
            .map(|(c, e)| (coeff(c), e.clone()))
            .collect::<Vec<_>>(),
        vec![
            (q(1, 1), q(1, 1)),
            (q(-1, 6), q(3, 1)),
            (q(1, 120), q(5, 1))
        ]
    );
}

#[test]
fn precision_and_coefficient_growth_refuse_without_reducing_requested_order() {
    assert!(rational_series(&poly(&[1]), &poly(&[1, -1]), &origin(), 257).is_err());
    assert!(rational_series(&poly(&[1]), &poly(&[1, -1]), &origin(), i64::MAX).is_err());
    let mut pole = vec![BigRational::zero(); 129];
    pole[128] = BigRational::one();
    assert!(
        rational_series(
            &poly(&[1]),
            &UnivariatePoly::new(Symbol::new("x"), pole),
            &origin(),
            129
        )
        .is_err()
    );
    let large = BigInt::one() << 1000u32;
    let denominator = UnivariatePoly::new(
        Symbol::new("x"),
        vec![BigRational::one(), BigRational::from_integer(-large)],
    );
    assert!(rational_series(&poly(&[1]), &denominator, &origin(), 18).is_err());
    let mut oversized = vec![BigRational::zero(); 130];
    oversized[129] = BigRational::one();
    assert!(
        rational_series(
            &poly(&[1]),
            &UnivariatePoly::new(Symbol::new("x"), oversized),
            &origin(),
            2
        )
        .is_err()
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
fn series_generation_and_verification_cancellation_never_publish_partial_coefficients() {
    let n = poly(&[1]);
    let d = poly(&[1, -1, -1]);
    let mut generation = CancelMeter::default();
    let mut verification = CancelMeter::default();
    metered_rational_series(&n, &d, &origin(), 20, &mut generation, &mut verification).unwrap();
    for checkpoint in [1, generation.checkpoints / 2, generation.checkpoints] {
        let mut cancelled = CancelMeter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_rational_series(&n, &d, &origin(), 20, &mut cancelled, &mut Unbounded),
            Err(RationalLimitError::Meter(MeterError::Cancelled))
        ));
    }
    for checkpoint in [1, verification.checkpoints / 2, verification.checkpoints] {
        let mut cancelled = CancelMeter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_rational_series(&n, &d, &origin(), 20, &mut Unbounded, &mut cancelled),
            Err(RationalLimitError::Verification(
                RationalLocalVerificationError::Meter(MeterError::Cancelled)
            ))
        ));
    }
    let mut tiny_generator = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched_verifier = CancelMeter::default();
    assert!(
        metered_rational_series(
            &n,
            &d,
            &origin(),
            20,
            &mut tiny_generator,
            &mut untouched_verifier
        )
        .is_err()
    );
    assert_eq!(untouched_verifier.checkpoints, 0);
    let mut tiny_verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_rational_series(&n, &d, &origin(), 20, &mut Unbounded, &mut tiny_verifier),
        Err(RationalLimitError::Verification(
            RationalLocalVerificationError::Meter(_)
        ))
    ));
}
