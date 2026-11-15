use fsym_budget::{Budget, BudgetLimits, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::rational_product::{
    RationalProductCertificate, RationalProductDomain, RationalProductFactors,
    verify_rational_product,
};
use fsym_proof_kernel::rational_sum::{RationalSumPole, RationalSumQuadratic};
use num_traits::{One, Zero};

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}
fn root(value: i64) -> RationalProductFactors {
    RationalProductFactors {
        roots: vec![RationalSumPole {
            root: q(value),
            multiplicity: 1,
        }],
        quadratics: vec![],
        modular: vec![],
    }
}
fn witness() -> RationalProductCertificate {
    // (k+1)/k = R(k+1)/R(k), with R(k)=k.
    RationalProductCertificate::Telescoping {
        scale: q(1),
        numerator: vec![q(0), q(1)],
        denominator: vec![q(1)],
        domain: RationalProductDomain {
            denominator: root(0),
            guards: vec![],
        },
        numerator_factors: root(0),
        denominator_factors: RationalProductFactors::default(),
    }
}
fn verify(certificate: &RationalProductCertificate) -> bool {
    verify_rational_product(
        &[q(1), q(1)],
        &[q(0), q(1)],
        &[],
        &1.into(),
        &10_000.into(),
        certificate,
        &mut Unbounded,
    )
    .is_ok()
}

#[test]
fn independently_supplied_telescoper_checks_exact_value_and_reverse() {
    let certificate = witness();
    assert_eq!(
        verify_rational_product(
            &[q(1), q(1)],
            &[q(0), q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &certificate,
            &mut Unbounded
        )
        .unwrap(),
        q(10_001)
    );
    assert_eq!(
        verify_rational_product(
            &[q(1), q(1)],
            &[q(0), q(1)],
            &[],
            &10_001.into(),
            &0.into(),
            &certificate,
            &mut Unbounded
        )
        .unwrap(),
        BigRational::new(1.into(), 10_001.into())
    );
}

#[test]
fn changed_scale_or_polynomial_cannot_pass_the_identity() {
    let mut wrong_scale = witness();
    if let RationalProductCertificate::Telescoping { scale, .. } = &mut wrong_scale {
        *scale = q(2);
    }
    assert!(!verify(&wrong_scale));
    let mut wrong_polynomial = witness();
    if let RationalProductCertificate::Telescoping {
        numerator,
        numerator_factors,
        ..
    } = &mut wrong_polynomial
    {
        *numerator = vec![q(2), q(1)];
        *numerator_factors = root(-2);
    }
    assert!(!verify(&wrong_polynomial));
}

#[test]
fn original_source_coefficients_are_independent_checker_inputs() {
    assert!(
        verify_rational_product(
            &[q(2), q(1)],
            &[q(0), q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &witness(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_rational_product(
            &[q(1), q(1)],
            &[q(1), q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &witness(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn missing_and_substituted_domain_factors_are_rejected() {
    let mut candidate = witness();
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
        domain.denominator.roots.clear();
    }
    assert!(!verify(&candidate));
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
        domain.denominator = root(2);
    }
    assert!(!verify(&candidate));
}

#[test]
fn missing_and_reordered_original_guards_are_rejected() {
    let mut certificate = witness();
    let guards = vec![vec![q(-5), q(1)], vec![q(-7), q(1)]];
    assert!(
        verify_rational_product(
            &[q(1), q(1)],
            &[q(0), q(1)],
            &guards,
            &10.into(),
            &10_000.into(),
            &certificate,
            &mut Unbounded
        )
        .is_err()
    );
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut certificate {
        domain.guards = vec![root(7), root(5)];
    }
    assert!(
        verify_rational_product(
            &[q(1), q(1)],
            &[q(0), q(1)],
            &guards,
            &10.into(),
            &10_000.into(),
            &certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn complete_domain_witness_still_rejects_an_actual_input_pole() {
    let mut certificate = witness();
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut certificate {
        domain.guards = vec![root(5)];
    }
    assert!(
        verify_rational_product(
            &[q(1), q(1)],
            &[q(0), q(1)],
            &[vec![q(-5), q(1)]],
            &1.into(),
            &10_000.into(),
            &certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn irreducibility_labels_do_not_hide_rational_quadratic_zeros() {
    let forged = RationalProductFactors {
        roots: vec![],
        quadratics: vec![RationalSumQuadratic {
            constant: q(-1),
            linear: q(0),
            multiplicity: 1,
        }],
        modular: vec![],
    };
    let candidate = RationalProductCertificate::Zero {
        at: 1.into(),
        domain: RationalProductDomain {
            denominator: forged,
            guards: vec![],
        },
    };
    assert!(
        verify_rational_product(
            &[q(0)],
            &[q(-1), q(0), q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn zero_witness_requires_an_actual_forward_factor() {
    let domain = RationalProductDomain {
        denominator: RationalProductFactors::default(),
        guards: vec![],
    };
    let valid = RationalProductCertificate::Zero {
        at: 5.into(),
        domain: domain.clone(),
    };
    assert_eq!(
        verify_rational_product(
            &[q(-5), q(1)],
            &[q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &valid,
            &mut Unbounded
        )
        .unwrap(),
        BigRational::zero()
    );
    assert!(
        verify_rational_product(
            &[q(-6), q(1)],
            &[q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &valid,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_rational_product(
            &[q(-5), q(1)],
            &[q(1)],
            &[],
            &10_001.into(),
            &0.into(),
            &valid,
            &mut Unbounded
        )
        .is_err()
    );
    let outside = RationalProductCertificate::Zero {
        at: 0.into(),
        domain,
    };
    assert!(
        verify_rational_product(
            &[q(0), q(1)],
            &[q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &outside,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn direct_zero_does_not_mask_a_later_domain_failure() {
    let candidate = RationalProductCertificate::Direct {
        value: BigRational::zero(),
    };
    assert!(
        verify_rational_product(
            &[q(-1), q(1)],
            &[q(-5), q(1)],
            &[],
            &1.into(),
            &10.into(),
            &candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn empty_range_requires_the_unit_witness_and_valid_shape() {
    let one = RationalProductCertificate::Direct {
        value: BigRational::one(),
    };
    assert_eq!(
        verify_rational_product(
            &[q(0)],
            &[q(1)],
            &[],
            &i64::MIN.into(),
            &(BigInt::from(i64::MIN) - BigInt::one()),
            &one,
            &mut Unbounded
        )
        .unwrap(),
        BigRational::one()
    );
    assert!(
        verify_rational_product(
            &[q(1)],
            &[q(0)],
            &[],
            &2.into(),
            &1.into(),
            &one,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_rational_product(
            &[q(1)],
            &[q(1), q(0)],
            &[],
            &2.into(),
            &1.into(),
            &one,
            &mut Unbounded
        )
        .is_err()
    );
    let zero = RationalProductCertificate::Direct {
        value: BigRational::zero(),
    };
    assert!(
        verify_rational_product(
            &[q(1)],
            &[q(1)],
            &[],
            &2.into(),
            &1.into(),
            &zero,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn oversized_factor_multiplicity_and_direct_range_refuse_before_expansion() {
    let mut candidate = witness();
    if let RationalProductCertificate::Telescoping { domain, .. } = &mut candidate {
        domain.denominator.roots[0].multiplicity = u32::MAX;
    }
    assert!(!verify(&candidate));
    let candidate = RationalProductCertificate::Direct {
        value: BigRational::one(),
    };
    assert!(
        verify_rational_product(
            &[q(1)],
            &[q(1)],
            &[],
            &i64::MIN.into(),
            &i64::MAX.into(),
            &candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn checker_budget_refusal_never_accepts_a_stored_candidate() {
    let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        verify_rational_product(
            &[q(1), q(1)],
            &[q(0), q(1)],
            &[],
            &1.into(),
            &10_000.into(),
            &witness(),
            &mut budget
        )
        .is_err()
    );
}
