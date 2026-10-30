//! Exact elementary rational definite integration and adversarial rechecks.
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::elementary_rational_definite::{
    ElementaryRationalDefiniteIntegralError as Error, elementary_rational_definite_integral,
    metered_elementary_rational_definite_integral,
};
use fsym_calculus::integrate::definite;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_proof_kernel::elementary_rational_definite::verify_elementary_rational_definite_integral;
use fsym_proof_kernel::rational_definite::RationalIntegralEndpoint as Endpoint;
use num_traits::{One, Zero};
use std::sync::Arc;

fn q(value: i64) -> BigRational {
    BigRational::from(value)
}
fn p(values: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(Symbol::new("x"), values.iter().copied().map(q).collect())
}
fn finite(value: i64) -> Endpoint {
    Endpoint::Finite(q(value))
}
fn log(value: i64) -> Expr {
    Expr::Function("log".into(), vec![Expr::from_i64(value)])
}
fn fraction(numerator: &UnivariatePoly, denominator: &UnivariatePoly) -> Expr {
    Expr::Mul(vec![
        numerator.to_expr(),
        Expr::Pow(
            Arc::new(denominator.to_expr()),
            Arc::new(Expr::from_i64(-1)),
        ),
    ])
}
fn same_value(actual: &Expr, expected: &Expr) {
    assert!(
        fsym_simplify::simplify(&(actual.clone() - expected.clone())).is_zero(),
        "actual {actual}; expected {expected}"
    );
}
fn checked(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &Endpoint,
    upper: &Endpoint,
    expected: &Expr,
) {
    let result =
        elementary_rational_definite_integral(numerator, denominator, lower, upper).unwrap();
    same_value(result.value(), expected);
    let rechecked = verify_elementary_rational_definite_integral(
        &numerator.coeffs,
        &denominator.coeffs,
        lower,
        upper,
        result.certificate(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(rechecked.value(), result.value());
}

#[test]
fn coupled_logarithms_at_infinity_have_exact_finite_limits() {
    let numerator = p(&[1]);
    let denominator = p(&[2, 3, 1]);
    checked(
        &numerator,
        &denominator,
        &finite(0),
        &Endpoint::PositiveInfinity,
        &log(2),
    );
    checked(
        &numerator,
        &denominator,
        &Endpoint::PositiveInfinity,
        &finite(0),
        &(-log(2)),
    );
    checked(
        &numerator,
        &denominator,
        &Endpoint::NegativeInfinity,
        &finite(-3),
        &log(2),
    );
    checked(
        &numerator,
        &p(&[6, 9, 3]),
        &finite(0),
        &Endpoint::PositiveInfinity,
        &(rational_expr(BigRational::new(1.into(), 3.into())) * log(2)),
    );
}

#[test]
fn logarithmic_growth_uses_factor_degrees_and_both_infinities() {
    // 2x/(x^2+1) - 2/(x+1): log(x^2+1) - 2log|x+1|.
    checked(
        &p(&[-2, 2]),
        &p(&[1, 1, 1, 1]),
        &finite(1),
        &Endpoint::PositiveInfinity,
        &log(2),
    );
    checked(
        &p(&[-2, 2, 2]),
        &p(&[2, 2, 3, 2, 1]),
        &Endpoint::NegativeInfinity,
        &Endpoint::PositiveInfinity,
        &Expr::from_i64(0),
    );
}

#[test]
fn mixed_and_repeated_factors_combine_rational_log_and_atan_parts() {
    checked(
        &p(&[3, 2]),
        &p(&[2, 5, 4, 1]),
        &finite(0),
        &Endpoint::PositiveInfinity,
        &(Expr::from_i64(1) + log(2)),
    );
    checked(
        &p(&[3, 3, 2]),
        &p(&[2, 3, 3, 3, 1]),
        &finite(0),
        &Endpoint::PositiveInfinity,
        &(rational_expr(BigRational::new(1.into(), 2.into())) * Expr::Const(Constant::Pi) + log(2)),
    );
    checked(
        &p(&[1]),
        &p(&[1, 0, 2, 0, 1]),
        &finite(0),
        &Endpoint::PositiveInfinity,
        &(rational_expr(BigRational::new(1.into(), 4.into())) * Expr::Const(Constant::Pi)),
    );
    checked(
        &p(&[1]),
        &p(&[1, 0, 2, 0, 1]),
        &finite(0),
        &finite(1),
        &(rational_expr(BigRational::new(1.into(), 4.into()))
            + rational_expr(BigRational::new(1.into(), 8.into())) * Expr::Const(Constant::Pi)),
    );
}

#[test]
fn real_logarithm_branches_and_higher_degree_logarithmic_derivatives() {
    checked(&p(&[1]), &p(&[0, 1]), &finite(-2), &finite(-1), &(-log(2)));
    checked(
        &p(&[1]),
        &p(&[2, 3, 1]),
        &finite(-4),
        &finite(-3),
        &(Expr::from_i64(2) * log(2) - log(3)),
    );
    checked(
        &p(&[1, 0, 3]),
        &p(&[1, 1, 0, 1]),
        &finite(0),
        &finite(1),
        &log(3),
    );
}

#[test]
fn real_quadratic_poles_use_real_logarithms_not_complex_branches() {
    let result = elementary_rational_definite_integral(
        &p(&[1]),
        &p(&[-2, 0, 1]),
        &finite(2),
        &Endpoint::PositiveInfinity,
    )
    .unwrap();
    // Independent positive series from 1/(x^2-2) = sum 2^k/x^(2k+2).
    // Integrating gives sum 1/((2k+1)*2^(k+1)), with geometric tail.
    let count = 48u32;
    let mut lower = BigRational::zero();
    for k in 0..count {
        lower += BigRational::new(
            BigInt::one(),
            BigInt::from(2 * k + 1) * (BigInt::one() << (k + 1)),
        );
    }
    let tail = BigRational::new(
        BigInt::one(),
        BigInt::from(2 * count + 1) * (BigInt::one() << count),
    );
    let enclosure = result.value().evalf_ball(32).unwrap();
    assert!(enclosure.lower() > lower);
    assert!(enclosure.upper() < lower + tail);
    assert!(!result.value().to_string().contains("I"));
}

#[test]
fn zero_and_rational_only_primitives_use_the_same_checked_entrypoint() {
    checked(
        &p(&[0]),
        &p(&[1, 0, 1]),
        &finite(-1),
        &finite(2),
        &Expr::from_i64(0),
    );
    checked(
        &p(&[1]),
        &p(&[1, 2, 1]),
        &finite(0),
        &Endpoint::PositiveInfinity,
        &Expr::from_i64(1),
    );
    checked(
        &p(&[0, 2]),
        &p(&[1]),
        &finite(1),
        &finite(3),
        &Expr::from_i64(8),
    );
    checked(
        &p(&[1]),
        &p(&[1, 0, 1]),
        &finite(1),
        &finite(1),
        &Expr::from_i64(0),
    );
}

#[test]
fn nonconvergent_endpoints_never_receive_a_finite_or_principal_value_receipt() {
    for (numerator, denominator, lower, upper) in [
        (p(&[1]), p(&[1, 1]), finite(0), Endpoint::PositiveInfinity),
        (
            p(&[0, 1]),
            p(&[1, 0, 1]),
            Endpoint::NegativeInfinity,
            Endpoint::PositiveInfinity,
        ),
        (p(&[1]), p(&[0, 1]), finite(0), finite(1)),
        (p(&[1]), p(&[0, 0, 1]), finite(0), finite(1)),
        (p(&[1]), p(&[1]), finite(0), Endpoint::PositiveInfinity),
    ] {
        assert!(matches!(
            elementary_rational_definite_integral(&numerator, &denominator, &lower, &upper),
            Err(Error::DivergentEndpoint)
        ));
    }
    assert!(
        elementary_rational_definite_integral(&p(&[1]), &p(&[0, 1]), &finite(-1), &finite(1),)
            .is_err()
    );
}

#[test]
fn original_interior_poles_survive_rational_cancellation() {
    let numerator = p(&[-1, 1]);
    let denominator = p(&[-1, 1, -1, 1]);
    assert!(
        elementary_rational_definite_integral(&numerator, &denominator, &finite(0), &finite(2),)
            .is_err()
    );
    assert!(
        definite(
            &fraction(&numerator, &denominator),
            &Symbol::new("x"),
            &Expr::from_i64(0),
            &Expr::from_i64(2),
        )
        .is_none()
    );
    checked(
        &numerator,
        &denominator,
        &finite(0),
        &finite(1),
        &(rational_expr(BigRational::new(1.into(), 4.into())) * Expr::Const(Constant::Pi)),
    );
}

#[test]
fn incomplete_primitive_and_mutated_certificates_refuse() {
    assert!(matches!(
        elementary_rational_definite_integral(&p(&[1]), &p(&[1, 1, 0, 1]), &finite(0), &finite(1),),
        Err(Error::UnsupportedPrimitive)
    ));
    let numerator = p(&[3, 3, 2]);
    let denominator = p(&[2, 3, 3, 3, 1]);
    let result = elementary_rational_definite_integral(
        &numerator,
        &denominator,
        &finite(0),
        &Endpoint::PositiveInfinity,
    )
    .unwrap();
    let good = result.certificate();
    let mut missing = good.clone();
    missing.terms.pop();
    let mut coefficient = good.clone();
    coefficient.terms[0].logarithm_coefficient += q(1);
    let mut overlap = good.clone();
    overlap.terms.push(good.terms[0].clone());
    let mut cofactor = good.clone();
    cofactor.terms[0].pole_cofactor[0] += q(1);
    let mut rational = good.clone();
    rational.primitive.rational_numerator[0] += q(1);
    for mutant in [missing, coefficient, overlap, cofactor, rational] {
        assert!(
            verify_elementary_rational_definite_integral(
                &numerator.coeffs,
                &denominator.coeffs,
                &finite(0),
                &Endpoint::PositiveInfinity,
                &mutant,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

struct Cancelled;
impl BudgetMeter for Cancelled {
    fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn generator_verifier_cancellation_and_real_budgets_never_publish() {
    let numerator = p(&[1]);
    let denominator = p(&[2, 3, 1]);
    assert!(matches!(
        metered_elementary_rational_definite_integral(
            &numerator,
            &denominator,
            &finite(0),
            &Endpoint::PositiveInfinity,
            &mut Cancelled,
            &mut Unbounded,
        ),
        Err(Error::Meter(MeterError::Cancelled))
    ));
    assert!(
        metered_elementary_rational_definite_integral(
            &numerator,
            &denominator,
            &finite(0),
            &Endpoint::PositiveInfinity,
            &mut Unbounded,
            &mut Cancelled,
        )
        .is_err()
    );
    let mut generator = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_elementary_rational_definite_integral(
            &numerator,
            &denominator,
            &finite(0),
            &Endpoint::PositiveInfinity,
            &mut generator,
            &mut Unbounded,
        )
        .is_err()
    );
    let mut verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_elementary_rational_definite_integral(
            &numerator,
            &denominator,
            &finite(0),
            &Endpoint::PositiveInfinity,
            &mut Unbounded,
            &mut verifier,
        )
        .is_err()
    );
}

#[test]
fn input_limits_generator_identity_and_ordinary_dispatch_remain_explicit() {
    let too_large = UnivariatePoly::new(
        Symbol::new("x"),
        vec![BigRational::from_integer(BigInt::one() << 4096u32)],
    );
    assert!(matches!(
        elementary_rational_definite_integral(&too_large, &p(&[1]), &finite(0), &finite(1),),
        Err(Error::Refused(_))
    ));
    let different = UnivariatePoly::one(Symbol::new("same_print_but_different"));
    assert!(matches!(
        elementary_rational_definite_integral(&p(&[1]), &different, &finite(0), &finite(1),),
        Err(Error::Refused(_))
    ));
    let numerator = p(&[1]);
    let denominator = p(&[2, 3, 1]);
    let value = definite(
        &fraction(&numerator, &denominator),
        &Symbol::new("x"),
        &Expr::from_i64(0),
        &Expr::Const(Constant::Infinity),
    )
    .unwrap();
    same_value(&value, &log(2));
}

#[test]
fn convenience_api_keeps_its_intrinsic_budget_and_ordinary_polynomials_keep_their_route() {
    let mut coefficients = vec![q(0); 33];
    coefficients[32] = q(1);
    let numerator = UnivariatePoly::new(Symbol::new("x"), coefficients);
    assert!(matches!(
        elementary_rational_definite_integral(&numerator, &p(&[1]), &finite(0), &finite(1)),
        Err(Error::Meter(_))
    ));
    let value = definite(
        &numerator.to_expr(),
        &Symbol::new("x"),
        &Expr::from_i64(0),
        &Expr::from_i64(1),
    )
    .unwrap();
    same_value(
        &value,
        &rational_expr(BigRational::new(1.into(), 33.into())),
    );
}
