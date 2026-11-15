use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_product::{
    RationalProductError, metered_rational_product_source, rational_product_source,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_proof_kernel::rational_product::{
    RationalProductCertificate, RationalProductVerificationError,
};
use num_traits::{One, Zero};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn integer(value: i64) -> Expr {
    Expr::from_i64(value)
}
fn index() -> Symbol {
    Symbol::new("product_modular_k")
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(integer(exponent)))
}
fn shifted(offset: i64) -> Expr {
    Expr::Add(vec![Expr::Sym(index()), integer(offset)])
}
fn cubic(offset: i64) -> Expr {
    Expr::Add(vec![power(shifted(offset), 3), shifted(offset), integer(1)])
}
fn quartic(offset: i64, constant: BigRational) -> Expr {
    Expr::Add(vec![power(shifted(offset), 4), Expr::Rational(constant)])
}
fn ratio(top: Expr, bottom: Expr) -> Expr {
    Expr::Mul(vec![top, power(bottom, -1)])
}
fn quartic_value(point: i64, constant: &BigRational) -> BigRational {
    BigRational::from_integer(BigInt::from(point).pow(4)) + constant
}
fn value(source: &Expr, lower: i64, upper: i64) -> BigRational {
    rational_product_source(source, &index(), &lower.into(), &upper.into())
        .unwrap()
        .expect("supported modular telescope")
        .into_value()
}

#[test]
fn billion_term_cubic_telescope_has_a_source_bound_modular_domain() {
    let source = ratio(cubic(1), cubic(0));
    let upper = 1_000_000_000;
    let receipt = rational_product_source(&source, &index(), &1.into(), &upper.into())
        .unwrap()
        .unwrap();
    let endpoint = BigInt::from(upper + 1);
    let expected = BigRational::new(endpoint.pow(3) + &endpoint + BigInt::one(), 3.into());
    assert_eq!(receipt.value(), &expected);
    assert_eq!(receipt.source().source(), &source);
    let RationalProductCertificate::Telescoping { domain, .. } = receipt.certificate() else {
        panic!("a billion-term product cannot use the direct witness");
    };
    assert!(!domain.denominator.modular.is_empty());
    assert!(!domain.guards.is_empty());
}

#[test]
fn quartic_telescopes_without_a_full_degree_irreducibility_claim() {
    // x^4+1 has no roots modulo 3, although it is reducible over that field.
    // Integer-domain checking needs no assertion that its reduction is irreducible.
    let source = ratio(quartic(1, q(1, 1)), quartic(0, q(1, 1)));
    assert_eq!(value(&source, 0, 10_000), quartic_value(10_001, &q(1, 1)));
}

#[test]
fn repeated_quartic_factors_and_missing_integer_shifts_telescope() {
    let source = power(ratio(quartic(3, q(1, 1)), quartic(0, q(1, 1))), 2);
    let upper = 5000;
    let mut expected = BigRational::one();
    for offset in 1..=3 {
        expected =
            expected * quartic_value(upper + offset, &q(1, 1)) / quartic_value(offset, &q(1, 1));
    }
    let expected = &expected * &expected;
    assert_eq!(value(&source, 1, upper), expected);
}

#[test]
fn mixed_linear_quadratic_and_modular_factors_keep_the_complete_domain() {
    let polynomial = |offset| {
        Expr::Mul(vec![
            shifted(offset),
            Expr::Add(vec![power(shifted(offset), 2), integer(1)]),
            quartic(offset, q(1, 1)),
        ])
    };
    let source = ratio(polynomial(1), polynomial(0));
    let endpoint = BigInt::from(10_001);
    let expected = BigRational::new(
        &endpoint * (endpoint.pow(2) + BigInt::one()) * (endpoint.pow(4) + BigInt::one()),
        4.into(),
    );
    assert_eq!(value(&source, 1, 10_000), expected);
}

#[test]
fn a_coarse_domain_witness_does_not_lose_distinct_quadratic_shift_orbits() {
    let quadratic =
        |offset, constant| Expr::Add(vec![power(shifted(offset), 2), integer(constant)]);
    let source = ratio(
        Expr::Mul(vec![quadratic(1, 1), quadratic(2, 2)]),
        Expr::Mul(vec![quadratic(0, 1), quadratic(0, 2)]),
    );
    let first = BigInt::from(5001).pow(2);
    let second = BigInt::from(5002).pow(2);
    let expected = BigRational::new(
        (&first + BigInt::one()) * (&first + BigInt::from(2)) * (second + BigInt::from(2)),
        36.into(),
    );
    assert_eq!(value(&source, 1, 5000), expected);
}

#[test]
fn rational_and_negative_coefficients_use_only_good_modular_denominators() {
    for numerator in [-1, 1] {
        let constant = q(numerator, 2);
        let source = ratio(quartic(1, constant.clone()), quartic(0, constant.clone()));
        let expected = quartic_value(10_001, &constant) / constant;
        assert_eq!(value(&source, 0, 10_000), expected);
    }
}

#[test]
fn negative_and_reversed_integer_ranges_keep_exact_endpoints() {
    let source = ratio(quartic(1, q(1, 1)), quartic(0, q(1, 1)));
    let denominator = quartic_value(-10_000, &q(1, 1));
    assert_eq!(
        value(&source, -10_000, -1),
        BigRational::one() / &denominator
    );
    assert_eq!(value(&source, 0, -10_001), denominator);
}

#[test]
fn held_zero_cancelled_and_zeroth_power_sources_retain_modular_guards() {
    let denominator = quartic(0, q(1, 1));
    let inverse = power(denominator.clone(), -1);
    let sources = [
        (
            Expr::Mul(vec![integer(0), inverse.clone()]),
            BigRational::zero(),
        ),
        (
            Expr::Mul(vec![denominator, inverse.clone()]),
            BigRational::one(),
        ),
        (power(inverse, 0), BigRational::one()),
    ];
    for (source, expected) in sources {
        let receipt = rational_product_source(&source, &index(), &1.into(), &10_000.into())
            .unwrap()
            .unwrap();
        assert_eq!(receipt.value(), &expected);
        assert!(!receipt.source().guards().is_empty());
    }
}

#[test]
fn a_real_integer_pole_cannot_hide_inside_a_mixed_higher_degree_guard() {
    let denominator = Expr::Mul(vec![shifted(-5), quartic(0, q(1, 1))]);
    let inverse = power(denominator.clone(), -1);
    let sources = [
        Expr::Mul(vec![integer(0), inverse.clone()]),
        Expr::Mul(vec![denominator, inverse.clone()]),
        power(inverse, 0),
    ];
    for source in sources {
        assert!(rational_product_source(&source, &index(), &1.into(), &10_000.into()).is_err());
    }
}

#[test]
fn a_missing_bounded_modular_witness_keeps_the_product_unevaluated() {
    // k^3+138*k+165 has a root modulo every admitted prime <=97. Its sole
    // real root lies strictly between -2 and -1, so it has no rational root.
    // The bounded modular proof remains unavailable despite the valid domain.
    let polynomial = |offset| {
        Expr::Add(vec![
            power(shifted(offset), 3),
            Expr::Mul(vec![integer(138), shifted(offset)]),
            integer(165),
        ])
    };
    let source = ratio(polynomial(1), polynomial(0));
    assert!(
        rational_product_source(&source, &index(), &1.into(), &10_000.into())
            .unwrap()
            .is_none()
    );
}

#[test]
fn expensive_factor_search_is_a_terminal_refusal_after_a_missing_modular_witness() {
    // All admitted primes divide M, so none can exclude the integer roots of
    // k^3+k+M by modular nonvanishing. The separate factor-search cost exceeds
    // the unchanged cumulative20M cap and must not trigger an unmetered retry.
    let mut constant = BigInt::one();
    for prime in [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97,
    ] {
        constant *= BigInt::from(prime);
    }
    let polynomial = |offset| {
        Expr::Add(vec![
            power(shifted(offset), 3),
            shifted(offset),
            Expr::Integer(constant.clone()),
        ])
    };
    let source = ratio(polynomial(1), polynomial(0));
    assert!(matches!(
        rational_product_source(&source, &index(), &1.into(), &10_000.into()),
        Err(RationalProductError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn higher_degree_orbits_do_not_escape_the_existing_shift_bound() {
    let source = ratio(quartic(65, q(1, 1)), quartic(0, q(1, 1)));
    assert!(
        rational_product_source(&source, &index(), &1.into(), &10_000.into())
            .unwrap()
            .is_none()
    );
}

#[derive(Default)]
struct StopAfter {
    checkpoints: usize,
    stop: Option<usize>,
}

impl BudgetMeter for StopAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.stop == Some(self.checkpoints) {
            return Err(MeterError::Cancelled);
        }
        self.checkpoints += 1;
        Ok(())
    }
}

#[test]
fn modular_generation_and_reference_verification_remain_cancellable() {
    let source = ratio(quartic(1, q(1, 1)), quartic(0, q(1, 1)));
    let mut generation = StopAfter::default();
    let mut verification = StopAfter::default();
    metered_rational_product_source(
        &source,
        &index(),
        &1.into(),
        &10_000.into(),
        &mut generation,
        &mut verification,
    )
    .unwrap()
    .unwrap();
    for checkpoint in [generation.checkpoints / 2, generation.checkpoints - 1] {
        let mut cancelled = StopAfter {
            stop: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_rational_product_source(
                &source,
                &index(),
                &1.into(),
                &10_000.into(),
                &mut cancelled,
                &mut Unbounded,
            ),
            Err(RationalProductError::Meter(MeterError::Cancelled))
        ));
    }
    for checkpoint in [verification.checkpoints / 2, verification.checkpoints - 1] {
        let mut cancelled = StopAfter {
            stop: Some(checkpoint),
            ..Default::default()
        };
        let error = metered_rational_product_source(
            &source,
            &index(),
            &1.into(),
            &10_000.into(),
            &mut Unbounded,
            &mut cancelled,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RationalProductError::Meter(MeterError::Cancelled)
                | RationalProductError::Verification(RationalProductVerificationError::Meter(
                    MeterError::Cancelled
                ))
        ));
    }
}
