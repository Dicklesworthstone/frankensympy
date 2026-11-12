//! Exact moment identities and adversaries for factorial-weighted sums.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::exponential_sum::{
    ExponentialSummationError, PolynomialExponentialCertificate,
    metered_polynomial_exponential_sum, polynomial_exponential_sum,
    verify_polynomial_exponential_sum,
};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use std::sync::Arc;

fn e(value: i64) -> Expr {
    Expr::from_i64(value)
}
fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}
fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}
fn factorial(index: &Symbol) -> Expr {
    Expr::Function("factorial".into(), vec![Expr::Sym(index.clone())])
}
fn source(coefficients: &[BigRational], parameter: &Expr, index: &Symbol) -> Expr {
    let weight = Expr::Add(
        coefficients
            .iter()
            .enumerate()
            .map(|(degree, coefficient)| {
                Expr::Mul(vec![
                    rational_expr(coefficient.clone()),
                    power(Expr::Sym(index.clone()), e(degree as i64)),
                ])
            })
            .collect(),
    );
    Expr::Mul(vec![
        weight,
        power(parameter.clone(), Expr::Sym(index.clone())),
        power(factorial(index), e(-1)),
    ])
}
fn equal(actual: &Expr, expected: &Expr) {
    let residual = fsym_simplify::try_expand(&(actual.clone() - expected.clone())).unwrap();
    let residual = fsym_simplify::try_simplify(&residual).unwrap();
    assert!(residual.is_zero(), "{actual} != {expected}: {residual}");
}
fn infinity() -> Expr {
    Expr::Const(Constant::Infinity)
}

#[test]
fn bare_reciprocal_factorial_has_the_exact_exponential_sum() {
    let k = Symbol::new("k");
    let summand = power(factorial(&k), e(-1));
    let receipt = polynomial_exponential_sum(&summand, &k, &e(0), &infinity())
        .unwrap()
        .unwrap();
    assert_eq!(receipt.value(), &Expr::Const(Constant::E));
    assert_eq!(receipt.source(), &summand);
    assert_eq!(receipt.index(), &k);
    assert_eq!(receipt.lower(), &e(0));
    assert_eq!(receipt.parameter(), &e(1));
    assert_eq!(receipt.certificate().falling_coefficients, vec![q(1, 1)]);
}

#[test]
fn independently_known_moments_match_through_degree_six() {
    let k = Symbol::new("k");
    let z = Expr::Sym(Symbol::new("z"));
    // Coefficients are the ordinary exact moments, fixed independently of
    // the candidate's forward-difference implementation.
    let expected = [
        "exp(z)",
        "z*exp(z)",
        "(z**2+z)*exp(z)",
        "(z**3+3*z**2+z)*exp(z)",
        "(z**4+6*z**3+7*z**2+z)*exp(z)",
        "(z**5+10*z**4+25*z**3+15*z**2+z)*exp(z)",
        "(z**6+15*z**5+65*z**4+90*z**3+31*z**2+z)*exp(z)",
    ];
    for (degree, expected) in expected.into_iter().enumerate() {
        let mut coefficients = vec![q(0, 1); degree + 1];
        coefficients[degree] = q(1, 1);
        let summand = source(&coefficients, &z, &k);
        let receipt = polynomial_exponential_sum(&summand, &k, &e(0), &infinity())
            .unwrap()
            .unwrap();
        equal(receipt.value(), &parse(expected).unwrap());
        equal(
            &fsym_calculus::summation::checked_summation(&summand, &k, &e(0), &infinity())
                .unwrap()
                .unwrap(),
            receipt.value(),
        );
    }
}

#[test]
fn rational_polynomial_weights_and_nonzero_lower_endpoints_keep_the_finite_head() {
    let k = Symbol::new("k");
    let z = Expr::Sym(Symbol::new("z"));
    let summand = source(&[q(5, 7), q(-3, 5), q(2, 3)], &z, &k);
    let receipt = polynomial_exponential_sum(&summand, &k, &e(3), &infinity())
        .unwrap()
        .unwrap();
    equal(
        receipt.value(),
        &parse("(5/7+z/15+2*z**2/3)*exp(z)-5/7-82*z/105-229*z**2/210").unwrap(),
    );
    assert_eq!(
        receipt.certificate().falling_coefficients,
        vec![q(5, 7), q(1, 15), q(2, 3)]
    );
}

#[test]
fn finite_complex_and_zero_parameters_do_not_need_a_geometric_convergence_bound() {
    let k = Symbol::new("k");
    for parameter in [
        e(-3),
        e(0),
        e(1),
        e(4),
        rational_expr(q(2, 3)),
        Expr::Const(Constant::I),
    ] {
        let summand = source(&[q(1, 1), q(1, 1)], &parameter, &k);
        let result = polynomial_exponential_sum(&summand, &k, &e(2), &infinity())
            .unwrap()
            .unwrap();
        let exponential = if parameter.is_zero() {
            e(1)
        } else if parameter.is_one() {
            Expr::Const(Constant::E)
        } else {
            Expr::Function("exp".into(), vec![parameter.clone()])
        };
        let expected = (e(1) + parameter.clone()) * exponential - e(1) - e(2) * parameter;
        equal(result.value(), &expected);
    }
    let zero = source(&[q(7, 1), q(2, 1)], &e(0), &k);
    assert_eq!(
        polynomial_exponential_sum(&zero, &k, &e(0), &infinity())
            .unwrap()
            .unwrap()
            .value(),
        &e(7)
    );
    assert_eq!(
        polynomial_exponential_sum(&zero, &k, &e(1), &infinity())
            .unwrap()
            .unwrap()
            .value(),
        &e(0)
    );
}

#[test]
fn typed_index_and_same_spelling_parameter_keep_separate_owners() {
    let k = Symbol::with_identity("k", SymbolIdentity::from_assumptions_digest([11; 32]));
    let z = Symbol::with_identity("k", SymbolIdentity::from_assumptions_digest([12; 32]));
    let summand = source(&[q(0, 1), q(1, 1)], &Expr::Sym(z.clone()), &k);
    let receipt = polynomial_exponential_sum(&summand, &k, &e(0), &infinity())
        .unwrap()
        .unwrap();
    assert_eq!(
        receipt
            .value()
            .free_symbols()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![z]
    );
    let rebound = Symbol::new("k");
    assert!(
        verify_polynomial_exponential_sum(
            &summand,
            &rebound,
            &e(0),
            &infinity(),
            receipt.certificate().clone(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn coefficient_changes_and_malformed_certificates_cannot_verify() {
    let k = Symbol::new("k");
    let z = Expr::Sym(Symbol::new("z"));
    let summand = source(&[q(0, 1), q(0, 1), q(1, 1)], &z, &k);
    let receipt = polynomial_exponential_sum(&summand, &k, &e(0), &infinity())
        .unwrap()
        .unwrap();
    for coefficient in 0..receipt.certificate().falling_coefficients.len() {
        let mut corrupted = receipt.certificate().clone();
        corrupted.falling_coefficients[coefficient] += q(1, 1_000_000_007);
        assert!(matches!(
            verify_polynomial_exponential_sum(
                &summand,
                &k,
                &e(0),
                &infinity(),
                corrupted,
                &mut Unbounded
            ),
            Err(ExponentialSummationError::InvalidCertificate)
        ));
    }
    for coefficients in [
        vec![],
        vec![q(0, 1); 34],
        vec![q(0, 1), q(1, 1), q(1, 1), q(0, 1)],
    ] {
        assert!(matches!(
            verify_polynomial_exponential_sum(
                &summand,
                &k,
                &e(0),
                &infinity(),
                PolynomialExponentialCertificate {
                    falling_coefficients: coefficients
                },
                &mut Unbounded
            ),
            Err(ExponentialSummationError::InvalidCertificate)
        ));
    }
    let different_weight = source(&[q(1, 1), q(0, 1), q(1, 1)], &z, &k);
    assert!(matches!(
        verify_polynomial_exponential_sum(
            &different_weight,
            &k,
            &e(0),
            &infinity(),
            receipt.certificate().clone(),
            &mut Unbounded
        ),
        Err(ExponentialSummationError::InvalidCertificate)
    ));
}

#[test]
fn zero_weight_cannot_hide_original_domain_or_degree_failures() {
    let k = Symbol::new("k");
    let summand = source(&[q(1, 1)], &e(2), &k);
    let pole = Expr::Function("factorial".into(), vec![e(-1)]);
    let zero_denominator = power(Expr::Add(vec![e(1), e(-1)]), e(-1));
    for invalid in [
        power(e(0), e(-1)),
        pole,
        zero_denominator,
        power(Expr::Sym(k.clone()), e(33)),
    ] {
        for erased in [Expr::Mul(vec![e(0), invalid.clone()]), power(invalid, e(0))] {
            let original = Expr::Mul(vec![erased, summand.clone()]);
            assert!(polynomial_exponential_sum(&original, &k, &e(0), &infinity()).is_err());
            assert!(
                fsym_calculus::summation::checked_summation(&original, &k, &e(0), &infinity())
                    .is_err()
            );
        }
    }
}

#[test]
fn unsupported_and_nonfinite_families_do_not_publish_a_formula() {
    let k = Symbol::new("k");
    let reciprocal = power(factorial(&k), e(-1));
    assert!(
        polynomial_exponential_sum(&reciprocal, &k, &e(0), &e(10))
            .unwrap()
            .is_none()
    );
    assert!(polynomial_exponential_sum(&reciprocal, &k, &e(-1), &infinity()).is_err());
    assert!(
        polynomial_exponential_sum(&reciprocal, &k, &Expr::Sym(Symbol::new("m")), &infinity())
            .is_err()
    );
    let squared = power(factorial(&k), e(-2));
    assert!(polynomial_exponential_sum(&squared, &k, &e(0), &infinity()).is_err());
    for parameter in [
        Constant::Infinity,
        Constant::NegativeInfinity,
        Constant::ComplexInfinity,
        Constant::NaN,
    ] {
        let original = source(&[q(1, 1)], &Expr::Const(parameter), &k);
        assert!(polynomial_exponential_sum(&original, &k, &e(0), &infinity()).is_err());
    }
}

#[test]
fn degree_head_and_scalar_boundaries_are_checked_before_expansion() {
    let k = Symbol::new("k");
    let summand = source(&[q(1, 1)], &e(2), &k);
    assert!(polynomial_exponential_sum(&summand, &k, &e(257), &infinity()).is_err());
    let huge_base = Expr::Integer(BigInt::from(1) << 4096_u32);
    let original = source(&[q(1, 1)], &huge_base, &k);
    assert!(polynomial_exponential_sum(&original, &k, &e(0), &infinity()).is_err());
    let oversized = Expr::Mul(vec![power(e(2), e(1025)), summand.clone()]);
    assert!(polynomial_exponential_sum(&oversized, &k, &e(0), &infinity()).is_err());
    let mut boundary = vec![q(0, 1); 33];
    boundary[32] = q(1, 1);
    assert!(
        polynomial_exponential_sum(&source(&boundary, &e(0), &k), &k, &e(256), &infinity()).is_ok()
    );
}

struct Stop {
    charges: usize,
    stop_after: usize,
}
impl BudgetMeter for Stop {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if self.charges >= self.stop_after {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
        self.checkpoint()?;
        self.charges += 1;
        Ok(())
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
}

#[test]
fn generator_and_verifier_failures_remain_terminal() {
    let k = Symbol::new("k");
    let summand = source(&[q(3, 1), q(-2, 1), q(7, 1)], &e(2), &k);
    for stop_after in [0, 10, 80] {
        let mut generator = Stop {
            charges: 0,
            stop_after,
        };
        assert!(matches!(
            metered_polynomial_exponential_sum(
                &summand,
                &k,
                &e(8),
                &infinity(),
                &mut generator,
                &mut Unbounded
            ),
            Err(ExponentialSummationError::Meter(MeterError::Cancelled))
        ));
        let mut verifier = Stop {
            charges: 0,
            stop_after,
        };
        assert!(matches!(
            metered_polynomial_exponential_sum(
                &summand,
                &k,
                &e(8),
                &infinity(),
                &mut Unbounded,
                &mut verifier
            ),
            Err(ExponentialSummationError::Meter(MeterError::Cancelled))
        ));
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000, 0);
        limits.dimensions[dimension.index()] = 0;
        let mut generator = Budget::new(limits.clone());
        let mut verifier = Budget::new(limits);
        assert!(matches!(
            metered_polynomial_exponential_sum(
                &summand,
                &k,
                &e(8),
                &infinity(),
                &mut generator,
                &mut Unbounded
            ),
            Err(ExponentialSummationError::Meter(MeterError::Budget(_)))
        ));
        assert!(matches!(
            metered_polynomial_exponential_sum(
                &summand,
                &k,
                &e(8),
                &infinity(),
                &mut Unbounded,
                &mut verifier
            ),
            Err(ExponentialSummationError::Meter(MeterError::Budget(_)))
        ));
    }
}
