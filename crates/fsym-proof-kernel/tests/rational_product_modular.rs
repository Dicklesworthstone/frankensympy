use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::rational_product::{
    RationalProductCertificate, RationalProductDomain, RationalProductFactors,
    RationalProductModularFactor, RationalProductVerificationError, verify_rational_product,
};
use num_traits::One;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}
fn polynomial(constant: BigRational) -> Vec<BigRational> {
    vec![constant, q(0), q(0), q(0), q(1)]
}
fn shifted(constant: BigRational) -> Vec<BigRational> {
    vec![constant + q(1), q(4), q(6), q(4), q(1)]
}
fn factors(coefficients: Vec<BigRational>, prime: u32) -> RationalProductFactors {
    RationalProductFactors {
        modular: vec![RationalProductModularFactor {
            coefficients,
            multiplicity: 1,
            prime,
        }],
        ..Default::default()
    }
}
fn certificate(constant: BigRational, prime: u32) -> RationalProductCertificate {
    let factor = factors(polynomial(constant.clone()), prime);
    RationalProductCertificate::Telescoping {
        scale: q(1),
        numerator: polynomial(constant),
        denominator: vec![q(1)],
        domain: Box::new(RationalProductDomain {
            denominator: factor.clone(),
            guards: vec![],
        }),
        numerator_factors: factor,
        denominator_factors: RationalProductFactors::default(),
    }
}
fn verify(
    candidate: &RationalProductCertificate,
) -> Result<BigRational, RationalProductVerificationError> {
    verify_rational_product(
        &shifted(q(1)),
        &polynomial(q(1)),
        &[],
        &0.into(),
        &10_000.into(),
        candidate,
        &mut Unbounded,
    )
}

#[test]
fn hand_written_quartic_witness_checks_without_irreducibility_or_factor_search() {
    let candidate = certificate(q(1), 3);
    let expected = BigRational::from_integer(BigInt::from(10_001).pow(4) + BigInt::one());
    assert_eq!(verify(&candidate).unwrap(), expected);
    assert_eq!(
        verify_rational_product(
            &shifted(q(1)),
            &polynomial(q(1)),
            &[],
            &10_001.into(),
            &(-1).into(),
            &candidate,
            &mut Unbounded,
        )
        .unwrap(),
        BigRational::one() / expected
    );
}

#[test]
fn invalid_composite_oversized_and_root_bearing_moduli_are_rejected() {
    // Modulo 4 this quartic also has no zeros: accepting that composite would
    // still violate the declared prime-witness shape, so check it explicitly.
    for prime in [0, 1, 2, 4, 9, 101, u32::MAX] {
        assert!(
            verify(&certificate(q(1), prime)).is_err(),
            "modulus {prime}"
        );
    }
}

#[test]
fn noninvertible_original_coefficient_denominators_are_not_normalized_away() {
    let constant = BigRational::new(1.into(), 2.into());
    for (prime, valid) in [(2, false), (5, true)] {
        let candidate = certificate(constant.clone(), prime);
        let checked = verify_rational_product(
            &shifted(constant.clone()),
            &polynomial(constant.clone()),
            &[],
            &0.into(),
            &10_000.into(),
            &candidate,
            &mut Unbounded,
        );
        if valid {
            let expected = BigRational::from_integer(
                BigInt::from(2) * BigInt::from(10_001).pow(4) + BigInt::one(),
            );
            assert_eq!(checked.unwrap(), expected);
        } else {
            assert!(checked.is_err());
        }
    }
}

#[test]
fn negative_wide_coefficients_have_their_exact_signed_residues_checked() {
    // -2^301 is 1 modulo 3; the high bits and sign both matter.
    let constant = BigRational::from_integer(-BigInt::from(2).pow(301));
    let candidate = certificate(constant.clone(), 3);
    let actual = verify_rational_product(
        &shifted(constant.clone()),
        &polynomial(constant.clone()),
        &[],
        &0.into(),
        &10_000.into(),
        &candidate,
        &mut Unbounded,
    )
    .unwrap();
    let expected = (BigRational::from_integer(BigInt::from(10_001).pow(4)) + &constant) / constant;
    assert_eq!(actual, expected);
}

#[test]
fn modularly_equal_factors_cannot_replace_the_original_rational_factor() {
    let mut candidate = certificate(q(1), 3);
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
        domain.denominator.modular[0].coefficients[0] = q(4);
    }
    // x^4+4 has the same zero-free image modulo 3 as x^4+1, but it is not
    // the original denominator. The complete rational identity must fail.
    assert!(verify(&candidate).is_err());
    let mut candidate = certificate(q(1), 3);
    if let RationalProductCertificate::Telescoping {
        numerator_factors, ..
    } = &mut candidate
    {
        numerator_factors.modular[0].coefficients[0] = q(4);
    }
    assert!(verify(&candidate).is_err());
}

#[test]
fn changed_and_extreme_multiplicities_refuse_before_expansion() {
    for multiplicity in [0, 2, u32::MAX] {
        let mut candidate = certificate(q(1), 3);
        if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
            domain.denominator.modular[0].multiplicity = multiplicity;
        }
        assert!(verify(&candidate).is_err());
    }
}

#[test]
fn omitted_and_nonmonic_modular_factors_do_not_establish_a_complete_domain() {
    let mut omitted = certificate(q(1), 3);
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut omitted {
        domain.denominator.modular.clear();
    }
    assert!(verify(&omitted).is_err());
    let mut nonmonic = certificate(q(1), 3);
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut nonmonic {
        domain.denominator.modular[0].coefficients[4] = q(2);
    }
    assert!(verify(&nonmonic).is_err());
}

#[test]
fn a_zero_product_cannot_use_a_fake_modular_proof_to_hide_an_integer_pole() {
    let denominator = polynomial(q(-16)); // integer poles at -2 and 2
    let candidate = RationalProductCertificate::Zero {
        at: 1.into(),
        domain: RationalProductDomain {
            denominator: factors(denominator.clone(), 3),
            guards: vec![],
        },
    };
    assert!(
        verify_rational_product(
            &[q(0)],
            &denominator,
            &[],
            &1.into(),
            &10_000.into(),
            &candidate,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn every_original_higher_degree_guard_is_checked_in_its_original_position() {
    let guards = vec![polynomial(q(1)), shifted(q(1))];
    let mut candidate = certificate(q(1), 3);
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
        domain.guards = guards
            .iter()
            .cloned()
            .map(|poly| factors(poly, 3))
            .collect();
    }
    assert!(
        verify_rational_product(
            &shifted(q(1)),
            &polynomial(q(1)),
            &guards,
            &0.into(),
            &10_000.into(),
            &candidate,
            &mut Unbounded,
        )
        .is_ok()
    );
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
        domain.guards.swap(0, 1);
    }
    assert!(
        verify_rational_product(
            &shifted(q(1)),
            &polynomial(q(1)),
            &guards,
            &0.into(),
            &10_000.into(),
            &candidate,
            &mut Unbounded,
        )
        .is_err()
    );
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
        domain.guards.pop();
    }
    assert!(
        verify_rational_product(
            &shifted(q(1)),
            &polynomial(q(1)),
            &guards,
            &0.into(),
            &10_000.into(),
            &candidate,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn complete_reconstruction_preserves_the_original_leading_scale() {
    let numerator: Vec<_> = shifted(q(1)).into_iter().map(|c| c * q(2)).collect();
    let denominator: Vec<_> = polynomial(q(1)).into_iter().map(|c| c * q(2)).collect();
    let candidate = certificate(q(1), 3);
    assert_eq!(
        verify_rational_product(
            &numerator,
            &denominator,
            &[],
            &0.into(),
            &10_000.into(),
            &candidate,
            &mut Unbounded,
        )
        .unwrap(),
        verify(&candidate).unwrap()
    );
}

#[test]
fn insufficient_reference_budget_cannot_publish_a_modular_domain() {
    let mut limits = BudgetLimits::uniform(128 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 100;
    assert!(matches!(
        verify_rational_product(
            &shifted(q(1)),
            &polynomial(q(1)),
            &[],
            &0.into(),
            &10_000.into(),
            &certificate(q(1), 3),
            &mut Budget::new(limits),
        ),
        Err(RationalProductVerificationError::Meter(_))
    ));
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
fn reference_cancellation_remains_terminal_through_modular_acceptance() {
    let candidate = certificate(q(1), 3);
    let mut baseline = StopAfter::default();
    verify_rational_product(
        &shifted(q(1)),
        &polynomial(q(1)),
        &[],
        &0.into(),
        &10_000.into(),
        &candidate,
        &mut baseline,
    )
    .unwrap();
    for checkpoint in [0, baseline.checkpoints / 2, baseline.checkpoints - 1] {
        let mut cancelled = StopAfter {
            stop: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            verify_rational_product(
                &shifted(q(1)),
                &polynomial(q(1)),
                &[],
                &0.into(),
                &10_000.into(),
                &candidate,
                &mut cancelled,
            ),
            Err(RationalProductVerificationError::Meter(
                MeterError::Cancelled
            ))
        ));
    }
}
