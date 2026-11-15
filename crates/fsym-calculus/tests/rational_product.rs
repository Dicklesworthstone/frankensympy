use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_product::{
    RationalProductError, metered_rational_product_source, rational_product_source,
};
use fsym_calculus::summation::checked_product;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_proof_kernel::rational_product::{RationalProductCertificate, verify_rational_product};
use num_traits::{One, Zero};
use std::sync::Arc;

fn integer(value: i64) -> Expr {
    Expr::from_i64(value)
}
fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn index() -> Symbol {
    Symbol::new("k")
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(integer(exponent)))
}
fn shifted(variable: &Symbol, value: i64) -> Expr {
    Expr::Add(vec![Expr::Sym(variable.clone()), integer(value)])
}
fn ratio(top: Expr, bottom: Expr) -> Expr {
    Expr::Mul(vec![top, power(bottom, -1)])
}
fn telescope(variable: &Symbol) -> Expr {
    ratio(Expr::Sym(variable.clone()), shifted(variable, 1))
}
fn result(source: &Expr, lower: i64, upper: i64) -> BigRational {
    rational_product_source(source, &index(), &lower.into(), &upper.into())
        .unwrap()
        .unwrap()
        .into_value()
}

#[test]
fn finite_rational_products_use_exact_direct_evaluation() {
    let k = index();
    assert_eq!(result(&telescope(&k), 1, 25), q(1, 26));
    assert_eq!(result(&Expr::Sym(k), 2, 7), q(5040, 1));
}

#[test]
fn long_rational_products_telescope_without_enumerating_the_range() {
    let k = index();
    let source = telescope(&k);
    let receipt = rational_product_source(&source, &k, &1.into(), &1_000_000.into())
        .unwrap()
        .unwrap();
    assert_eq!(receipt.value(), &q(1, 1_000_001));
    assert_eq!(receipt.source().source(), &source);
    assert_eq!(receipt.source().variables(), std::slice::from_ref(&k));
    assert_eq!(receipt.lower(), &BigInt::from(1));
    assert_eq!(receipt.upper(), &BigInt::from(1_000_000));
    assert!(matches!(
        receipt.certificate(),
        RationalProductCertificate::Telescoping { .. }
    ));
}

#[test]
fn mixed_positive_and_negative_shift_multiplicities_are_complete() {
    let k = index();
    let source = ratio(
        Expr::Mul(vec![Expr::Sym(k.clone()), shifted(&k, 2)]),
        power(shifted(&k, 1), 2),
    );
    assert_eq!(result(&source, 1, 1_000_000), q(1_000_002, 2_000_002));
    let inverse = power(source, -1);
    assert_eq!(result(&inverse, 1, 1_000_000), q(2_000_002, 1_000_002));
}

#[test]
fn irreducible_quadratic_shift_orbits_telescope_exactly() {
    let k = index();
    let bottom = Expr::Add(vec![power(Expr::Sym(k.clone()), 2), integer(1)]);
    let top = Expr::Add(vec![power(shifted(&k, 1), 2), integer(1)]);
    assert_eq!(result(&ratio(top, bottom), 1, 10_000), q(100_020_002, 2));
}

#[test]
fn fractional_offsets_and_higher_multiplicity_are_retained() {
    let k = index();
    let bottom = Expr::Add(vec![Expr::Sym(k.clone()), Expr::Rational(q(1, 2))]);
    let top = Expr::Add(vec![Expr::Sym(k.clone()), Expr::Rational(q(3, 2))]);
    let source = power(ratio(top, bottom), 2);
    assert_eq!(result(&source, 0, 10_000), q(400_120_009, 1));
}

#[test]
fn scalar_powers_are_part_of_the_checked_identity() {
    let k = index();
    let source = Expr::Mul(vec![integer(-1), telescope(&k)]);
    assert_eq!(result(&source, 1, 10_000), q(1, 10_001));
    assert_eq!(result(&source, 1, 10_001), q(-1, 10_002));
    let source = Expr::Mul(vec![integer(2), telescope(&k)]);
    let expected = BigRational::new(BigInt::from(2).pow(2100), 2101.into());
    assert_eq!(result(&source, 1, 2100), expected);
}

#[test]
fn reversed_and_empty_ranges_follow_reciprocal_conventions() {
    let k = index();
    assert_eq!(result(&telescope(&k), 10, 5), q(5, 3));
    assert_eq!(result(&telescope(&k), 10_001, 0), q(10_001, 1));
    assert_eq!(result(&Expr::Sym(k), 5, 4), BigRational::one());
    assert_eq!(result(&integer(3), 5, 1), q(1, 27));
}

#[test]
fn empty_range_still_requires_a_well_formed_original_source() {
    let k = index();
    let undefined = power(integer(0), -1);
    assert!(rational_product_source(&undefined, &k, &5.into(), &4.into()).is_err());
    assert_eq!(result(&power(Expr::Sym(k), -1), 1, 0), BigRational::one());
}

#[test]
fn signed_machine_word_boundaries_never_overflow() {
    let k = index();
    let lo = BigInt::from(i64::MAX - 1);
    let hi = BigInt::from(i64::MAX);
    let receipt = rational_product_source(&telescope(&k), &k, &lo, &hi)
        .unwrap()
        .unwrap();
    assert_eq!(
        receipt.value(),
        &BigRational::new(lo.clone(), &hi + BigInt::one())
    );
    assert_eq!(result(&integer(-1), i64::MIN, i64::MAX), BigRational::one());
    assert_eq!(result(&integer(-1), i64::MAX, i64::MIN), BigRational::one());
}

#[test]
fn arbitrary_precision_integer_endpoints_are_exact() {
    let k = index();
    let lo = BigInt::from(10).pow(40);
    let hi = &lo + BigInt::from(2);
    let receipt = rational_product_source(&telescope(&k), &k, &lo, &hi)
        .unwrap()
        .unwrap();
    assert_eq!(
        receipt.value(),
        &BigRational::new(lo.clone(), &lo + BigInt::from(3))
    );
}

#[test]
fn negative_pole_free_intervals_are_supported() {
    let k = index();
    assert_eq!(result(&telescope(&k), -20_000, -10_001), q(2, 1));
    assert_eq!(result(&telescope(&k), -10_000, -20_001), q(1, 2));
}

#[test]
fn a_zero_factor_is_valid_only_after_all_original_domains_are_checked() {
    let k = index();
    assert_eq!(result(&shifted(&k, -5), 1, 10_000), BigRational::zero());
    assert_eq!(result(&integer(0), 1, 10_000), BigRational::zero());
    let source = ratio(shifted(&k, -2), shifted(&k, -5));
    for upper in [10, 10_000] {
        assert!(rational_product_source(&source, &k, &1.into(), &upper.into()).is_err());
    }
    assert!(rational_product_source(&shifted(&k, -5), &k, &10_001.into(), &0.into()).is_err());
}

#[test]
fn held_cancellation_zero_products_and_zero_powers_do_not_erase_poles() {
    let k = index();
    let inverse = power(shifted(&k, -5), -1);
    let sources = [
        Expr::Mul(vec![shifted(&k, -5), inverse.clone()]),
        Expr::Mul(vec![integer(0), inverse.clone()]),
        power(inverse, 0),
    ];
    for source in sources {
        for upper in [10, 10_000] {
            assert!(
                rational_product_source(&source, &k, &1.into(), &upper.into()).is_err(),
                "{source}"
            );
        }
    }
}

#[test]
fn missing_source_guard_and_altered_value_certificates_are_rejected() {
    let k = index();
    let source = power(power(shifted(&k, -5), -1), 0);
    let receipt = rational_product_source(&source, &k, &6.into(), &20_000.into())
        .unwrap()
        .unwrap();
    let mut certificate = receipt.certificate().clone();
    let RationalProductCertificate::Telescoping { domain, .. } = &mut certificate else {
        panic!()
    };
    domain.guards.clear();
    assert!(
        verify_rational_product(
            &[q(1, 1)],
            &[q(1, 1)],
            &[vec![q(-5, 1), q(1, 1)]],
            &6.into(),
            &20_000.into(),
            &certificate,
            &mut Unbounded
        )
        .is_err()
    );
    let wrong = RationalProductCertificate::Direct { value: q(2, 1) };
    assert!(
        verify_rational_product(
            &[q(1, 1)],
            &[q(1, 1)],
            &[],
            &6.into(),
            &8.into(),
            &wrong,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn unsupported_long_nontelescoping_products_and_symbols_stay_unevaluated() {
    let k = index();
    assert!(
        rational_product_source(&Expr::Sym(k.clone()), &k, &1.into(), &10_000.into())
            .unwrap()
            .is_none()
    );
    assert!(
        rational_product_source(&Expr::symbol("foreign"), &k, &1.into(), &2.into())
            .unwrap()
            .is_none()
    );
    assert!(
        checked_product(
            &Expr::Function("mystery".into(), vec![Expr::Sym(k.clone())]),
            &k,
            &integer(1),
            &integer(10_000)
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn scalar_growth_and_original_hidden_expansion_are_terminal_refusals() {
    let k = index();
    assert!(rational_product_source(&integer(2), &k, &1.into(), &1_000_000.into()).is_err());
    let hidden = Expr::Mul(vec![integer(0), power(Expr::Sym(k.clone()), 257)]);
    assert!(checked_product(&hidden, &k, &integer(1), &integer(2)).is_err());
}

#[derive(Default)]
struct Counting {
    steps: usize,
    cancel_at: Option<usize>,
    compute: u64,
}
impl BudgetMeter for Counting {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.compute += amount;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if self.cancel_at == Some(self.steps) {
            return Err(MeterError::Cancelled);
        }
        self.steps += 1;
        Ok(())
    }
}

#[test]
fn generator_and_verifier_cancellation_remain_terminal() {
    let k = index();
    let source = telescope(&k);
    let mut generation = Counting::default();
    let mut verification = Counting::default();
    assert!(
        metered_rational_product_source(
            &source,
            &k,
            &1.into(),
            &10_000.into(),
            &mut generation,
            &mut verification
        )
        .unwrap()
        .is_some()
    );
    for checkpoint in [0, 1, generation.steps / 2, generation.steps - 1] {
        let mut cancelled = Counting {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_rational_product_source(
                &source,
                &k,
                &1.into(),
                &10_000.into(),
                &mut cancelled,
                &mut Unbounded
            ),
            Err(RationalProductError::Meter(MeterError::Cancelled))
        ));
    }
    for checkpoint in [0, 1, verification.steps / 2, verification.steps - 1] {
        let mut cancelled = Counting {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        let error = metered_rational_product_source(
            &source,
            &k,
            &1.into(),
            &10_000.into(),
            &mut Unbounded,
            &mut cancelled,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RationalProductError::Meter(MeterError::Cancelled)
                | RationalProductError::Verification(
                    fsym_proof_kernel::rational_product::RationalProductVerificationError::Meter(
                        MeterError::Cancelled
                    )
                )
        ));
    }
}

#[test]
fn exhausted_either_phase_never_publishes_a_product() {
    let k = index();
    let source = telescope(&k);
    let mut exhausted = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_rational_product_source(
            &source,
            &k,
            &1.into(),
            &10_000.into(),
            &mut exhausted,
            &mut Unbounded
        )
        .is_err()
    );
    let mut exhausted = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_rational_product_source(
            &source,
            &k,
            &1.into(),
            &10_000.into(),
            &mut Unbounded,
            &mut exhausted
        )
        .is_err()
    );
}

#[test]
fn long_range_cost_is_independent_of_the_integer_count() {
    let k = index();
    let source = telescope(&k);
    let mut first = Counting::default();
    let mut second = Counting::default();
    metered_rational_product_source(
        &source,
        &k,
        &1.into(),
        &10_000.into(),
        &mut first,
        &mut Unbounded,
    )
    .unwrap()
    .unwrap();
    metered_rational_product_source(
        &source,
        &k,
        &1.into(),
        &1_000_000_000.into(),
        &mut second,
        &mut Unbounded,
    )
    .unwrap()
    .unwrap();
    assert_eq!(first.compute, second.compute);
}
