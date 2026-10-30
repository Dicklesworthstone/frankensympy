use fsym_budget::{Budget, BudgetLimits, Unbounded};
use fsym_core::{BigRational, Constant, Expr};
use fsym_proof_kernel::elementary_rational_definite::{
    ElementaryRationalDefiniteIntegralCertificate,
    ElementaryRationalDefiniteIntegralVerification,
    ElementaryRationalDefiniteIntegralVerificationError as Error,
    ElementaryRationalTerm, verify_elementary_rational_definite_integral,
};
use fsym_proof_kernel::rational_definite::RationalIntegralEndpoint as Endpoint;
use fsym_proof_kernel::rational_integral::RationalIntegralReduction;
use fsym_proof_kernel::rational_limit::RationalLocalCertificate;

type Result = std::result::Result<ElementaryRationalDefiniteIntegralVerification, Error>;

fn q(value: i64) -> BigRational {
    BigRational::from(value)
}
fn coefficients(values: &[i64]) -> Vec<BigRational> {
    values.iter().copied().map(q).collect()
}
fn zero_limit() -> RationalLocalCertificate {
    RationalLocalCertificate {
        numerator_order: None,
        denominator_order: 0,
        numerator_unit: coefficients(&[0]),
        denominator_unit: coefficients(&[1]),
    }
}
fn log_two() -> ElementaryRationalDefiniteIntegralCertificate {
    // Hand-constructed independently of the calculus generator:
    // 1/((x+1)(x+2)) = (log|x+1| - log|x+2|)'.
    // -4*(x^2+3x+2) + (2x+3)*(2x+3) = 1.
    ElementaryRationalDefiniteIntegralCertificate {
        primitive: RationalIntegralReduction {
            rational_numerator: coefficients(&[0]),
            rational_denominator: coefficients(&[1]),
            remainder_numerator: coefficients(&[1]),
            remainder_denominator: coefficients(&[2, 3, 1]),
            rational_pole_cofactor: coefficients(&[2, 3, 1]),
            remainder_pole_cofactor: coefficients(&[1]),
            square_free_u: coefficients(&[-4]),
            square_free_v: coefficients(&[3, 2]),
        },
        terms: vec![
            ElementaryRationalTerm {
                denominator: coefficients(&[1, 1]),
                pole_cofactor: coefficients(&[2, 1]),
                logarithm_coefficient: q(1),
                reciprocal_coefficient: q(0),
            },
            ElementaryRationalTerm {
                denominator: coefficients(&[2, 1]),
                pole_cofactor: coefficients(&[1, 1]),
                logarithm_coefficient: q(-1),
                reciprocal_coefficient: q(0),
            },
        ],
        lower_limit: zero_limit(),
        upper_limit: zero_limit(),
    }
}
fn check_log(certificate: &ElementaryRationalDefiniteIntegralCertificate) -> Result {
    verify_elementary_rational_definite_integral(
        &coefficients(&[1]), &coefficients(&[2, 3, 1]),
        &Endpoint::Finite(q(0)), &Endpoint::PositiveInfinity,
        certificate, &mut Unbounded,
    )
}

#[test]
fn independent_logarithmic_infinity_cancellation_and_negative_branch() {
    let certificate = log_two();
    let logarithm = Expr::Function("log".into(), vec![Expr::from_i64(2)]);
    assert_eq!(check_log(&certificate).unwrap().value(), &logarithm);
    let reversed = verify_elementary_rational_definite_integral(
        &coefficients(&[1]), &coefficients(&[2, 3, 1]),
        &Endpoint::PositiveInfinity, &Endpoint::Finite(q(0)),
        &certificate, &mut Unbounded,
    ).unwrap();
    assert_eq!(reversed.value(), &(-logarithm));
    let negative = verify_elementary_rational_definite_integral(
        &coefficients(&[1]), &coefficients(&[2, 3, 1]),
        &Endpoint::Finite(q(-4)), &Endpoint::Finite(q(-3)),
        &certificate, &mut Unbounded,
    ).unwrap();
    let expected = Expr::from_i64(2)
        * Expr::Function("log".into(), vec![Expr::from_i64(2)])
        - Expr::Function("log".into(), vec![Expr::from_i64(3)]);
    assert_eq!(negative.value(), &expected);
}

#[test]
fn derivative_completeness_poles_sources_and_limits_are_not_optional() {
    let good = log_two();
    let mut mutants = Vec::new();
    let mut missing = good.clone();
    missing.terms.pop();
    mutants.push(missing);
    let mut coefficient = good.clone();
    coefficient.terms[1].logarithm_coefficient = q(-2);
    mutants.push(coefficient);
    let mut cofactor = good.clone();
    cofactor.terms[0].pole_cofactor[0] = q(3);
    mutants.push(cofactor);
    let mut extra_pole = good.clone();
    extra_pole.terms[0].denominator[0] = q(-1);
    mutants.push(extra_pole);
    let mut nonquadratic = good.clone();
    nonquadratic.terms[0].reciprocal_coefficient = q(1);
    mutants.push(nonquadratic);
    let mut overlap = good.clone();
    overlap.terms.push(good.terms[0].clone());
    mutants.push(overlap);
    let mut limit = good.clone();
    limit.upper_limit.denominator_unit[0] = q(2);
    mutants.push(limit);
    let mut rational = good.clone();
    rational.primitive.rational_numerator[0] = q(1);
    mutants.push(rational);
    for mutant in mutants { assert!(check_log(&mutant).is_err()); }
    assert!(verify_elementary_rational_definite_integral(
        &coefficients(&[2]), &coefficients(&[2, 3, 1]),
        &Endpoint::Finite(q(0)), &Endpoint::PositiveInfinity,
        &good, &mut Unbounded,
    ).is_err());
    assert!(verify_elementary_rational_definite_integral(
        &coefficients(&[1]), &coefficients(&[2, 3, 1]),
        &Endpoint::Finite(q(-3)), &Endpoint::Finite(q(0)),
        &good, &mut Unbounded,
    ).is_err());
}

fn quadratic(
    numerator: Vec<BigRational>, logarithm: BigRational, reciprocal: BigRational,
) -> ElementaryRationalDefiniteIntegralCertificate {
    ElementaryRationalDefiniteIntegralCertificate {
        primitive: RationalIntegralReduction {
            rational_numerator: coefficients(&[0]),
            rational_denominator: coefficients(&[1]),
            remainder_numerator: numerator,
            remainder_denominator: coefficients(&[1, 0, 1]),
            rational_pole_cofactor: coefficients(&[1, 0, 1]),
            remainder_pole_cofactor: coefficients(&[1]),
            square_free_u: coefficients(&[1]),
            square_free_v: vec![q(0), BigRational::new((-1).into(), 2.into())],
        },
        terms: vec![ElementaryRationalTerm {
            denominator: coefficients(&[1, 0, 1]),
            pole_cofactor: coefficients(&[1]),
            logarithm_coefficient: logarithm,
            reciprocal_coefficient: reciprocal,
        }],
        lower_limit: zero_limit(),
        upper_limit: zero_limit(),
    }
}

#[test]
fn independent_arctangent_limit_and_divergent_logarithmic_tail() {
    let certificate = quadratic(coefficients(&[1]), q(0), q(1));
    let value = verify_elementary_rational_definite_integral(
        &coefficients(&[1]), &coefficients(&[1, 0, 1]),
        &Endpoint::NegativeInfinity, &Endpoint::PositiveInfinity,
        &certificate, &mut Unbounded,
    ).unwrap();
    assert_eq!(value.value(), &Expr::Const(Constant::Pi));
    // x/(1+x^2) has divergent limits at both infinities. Equal
    // logarithmic divergences cannot be subtracted into a false zero.
    let divergent = quadratic(
        coefficients(&[0, 1]), BigRational::new(1.into(), 2.into()), q(0),
    );
    assert!(matches!(verify_elementary_rational_definite_integral(
        &coefficients(&[0, 1]), &coefficients(&[1, 0, 1]),
        &Endpoint::NegativeInfinity, &Endpoint::PositiveInfinity,
        &divergent, &mut Unbounded,
    ), Err(Error::DivergentEndpoint)));
}

#[test]
fn malformed_shapes_and_actual_verifier_budget_refuse() {
    let good = log_two();
    let mut empty = good.clone();
    empty.terms[0].denominator.clear();
    assert!(check_log(&empty).is_err());
    let mut noncanonical = good.clone();
    noncanonical.terms[0].denominator.push(q(0));
    assert!(check_log(&noncanonical).is_err());
    let mut oversized = good.clone();
    oversized.terms = vec![good.terms[0].clone(); 129];
    assert!(check_log(&oversized).is_err());
    let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(verify_elementary_rational_definite_integral(
        &coefficients(&[1]), &coefficients(&[2, 3, 1]),
        &Endpoint::Finite(q(0)), &Endpoint::PositiveInfinity,
        &good, &mut budget,
    ).is_err());
}
