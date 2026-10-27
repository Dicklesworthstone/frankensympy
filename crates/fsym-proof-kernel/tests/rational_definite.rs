//! Definite-integral evidence tests using hand-built witnesses only.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::rational_definite::{
    MAX_DEFINITE_COEFFICIENTS, RationalDefiniteIntegralCertificate,
    RationalDefiniteIntegralVerificationError as Error, RationalIntegralEndpoint as Endpoint,
    verify_rational_definite_integral, verify_rational_integral_domain,
};
use fsym_proof_kernel::rational_integral::RationalIntegralReduction;
use fsym_proof_kernel::rational_limit::RationalLocalCertificate;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}
fn p(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|&n| q(n)).collect()
}
fn finite(n: i64) -> Endpoint {
    Endpoint::Finite(q(n))
}

fn reciprocal_limit(at: i64) -> RationalLocalCertificate {
    RationalLocalCertificate {
        numerator_order: Some(0),
        denominator_order: if at == 0 { 1 } else { 0 },
        numerator_unit: p(&[-1]),
        denominator_unit: if at == 0 { p(&[1]) } else { p(&[at, 1]) },
    }
}

fn reciprocal_integral() -> RationalDefiniteIntegralCertificate {
    // 1/x^2 = (-1/x)'; integral from 1 to 2 is 1/2.
    RationalDefiniteIntegralCertificate {
        primitive: RationalIntegralReduction {
            rational_numerator: p(&[-1]),
            rational_denominator: p(&[0, 1]),
            remainder_numerator: p(&[0]),
            remainder_denominator: p(&[1]),
            rational_pole_cofactor: p(&[0, 1]),
            remainder_pole_cofactor: p(&[0, 0, 1]),
            square_free_u: p(&[1]),
            square_free_v: p(&[0]),
        },
        lower_limit: reciprocal_limit(1),
        upper_limit: reciprocal_limit(2),
        value: BigRational::new(1.into(), 2.into()),
    }
}

#[test]
fn finite_reversed_and_improper_values_need_all_three_independent_checks() {
    let mut witness = reciprocal_integral();
    let check = |lo, hi, witness: &RationalDefiniteIntegralCertificate| {
        verify_rational_definite_integral(
            &p(&[1]),
            &p(&[0, 0, 1]),
            &lo,
            &hi,
            witness,
            &mut Unbounded,
        )
    };
    assert_eq!(
        check(finite(1), finite(2), &witness).unwrap().value(),
        &witness.value
    );
    std::mem::swap(&mut witness.lower_limit, &mut witness.upper_limit);
    witness.value = -witness.value;
    assert_eq!(
        check(finite(2), finite(1), &witness).unwrap().value(),
        &witness.value
    );

    witness.lower_limit = reciprocal_limit(1);
    witness.upper_limit = RationalLocalCertificate {
        numerator_order: Some(1),
        denominator_order: 0,
        numerator_unit: p(&[-1]),
        denominator_unit: p(&[1]),
    };
    witness.value = q(1);
    assert_eq!(
        check(finite(1), Endpoint::PositiveInfinity, &witness)
            .unwrap()
            .value(),
        &q(1)
    );
    witness.lower_limit = witness.upper_limit.clone();
    witness.upper_limit = reciprocal_limit(-1);
    assert_eq!(
        check(Endpoint::NegativeInfinity, finite(-1), &witness)
            .unwrap()
            .value(),
        &q(1)
    );
}

#[test]
fn rejects_identity_endpoint_value_and_input_rebinding_mutants() {
    for mutation in 0..8 {
        let mut witness = reciprocal_integral();
        match mutation {
            0 => witness.primitive.rational_numerator[0] = q(1),
            1 => witness.primitive.rational_pole_cofactor[0] = q(1),
            2 => witness.primitive.remainder_numerator[0] = q(1),
            3 => witness.lower_limit.denominator_unit[0] = q(2),
            4 => witness.upper_limit.numerator_order = Some(1),
            5 => witness.value = q(2),
            6 => witness.lower_limit.numerator_unit.push(q(0)),
            7 => witness.primitive.square_free_u[0] = q(0),
            _ => unreachable!(),
        }
        assert!(
            verify_rational_definite_integral(
                &p(&[1]),
                &p(&[0, 0, 1]),
                &finite(1),
                &finite(2),
                &witness,
                &mut Unbounded,
            )
            .is_err(),
            "mutation {mutation}"
        );
    }
    let witness = reciprocal_integral();
    assert!(
        verify_rational_definite_integral(
            &p(&[2]),
            &p(&[0, 0, 1]),
            &finite(1),
            &finite(2),
            &witness,
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        verify_rational_definite_integral(
            &p(&[1]),
            &p(&[0, 0, 1]),
            &finite(1),
            &finite(3),
            &witness,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn sturm_domain_detects_irrational_repeated_and_distant_poles_without_samples() {
    // (x^2-2)^2 has two double irrational roots.
    for scale in [1i64, -1, 7] {
        let d: Vec<_> = p(&[4, 0, -4, 0, 1])
            .into_iter()
            .map(|c| c * q(scale))
            .collect();
        assert!(matches!(
            verify_rational_integral_domain(&d, &finite(1), &finite(2), &mut Unbounded),
            Err(Error::InteriorSingularity)
        ));
        assert!(matches!(
            verify_rational_integral_domain(&d, &finite(-2), &finite(-1), &mut Unbounded),
            Err(Error::InteriorSingularity)
        ));
        verify_rational_integral_domain(&d, &finite(-1), &finite(1), &mut Unbounded).unwrap();
        verify_rational_integral_domain(&d, &finite(1), &finite(-1), &mut Unbounded).unwrap();
    }
    // A double pole far beyond the usual finite sampling window.
    assert!(matches!(
        verify_rational_integral_domain(
            &p(&[10_000, -200, 1]),
            &finite(0),
            &Endpoint::PositiveInfinity,
            &mut Unbounded,
        ),
        Err(Error::InteriorSingularity)
    ));
    assert!(matches!(
        verify_rational_integral_domain(
            &p(&[4, 0, -4, 0, 1]),
            &Endpoint::NegativeInfinity,
            &Endpoint::PositiveInfinity,
            &mut Unbounded,
        ),
        Err(Error::InteriorSingularity)
    ));
    // A no-real-roots denominator stays valid on the entire real line.
    verify_rational_integral_domain(
        &p(&[1, 0, 2, 0, 1]),
        &Endpoint::NegativeInfinity,
        &Endpoint::PositiveInfinity,
        &mut Unbounded,
    )
    .unwrap();
}

#[test]
fn endpoint_roots_are_distinct_from_interior_poles_and_divergence_is_not_certified() {
    // x^4 (x-1)^2: both endpoints are roots, but no root lies in (0, 1).
    let d = p(&[0, 0, 0, 0, 1, -2, 1]);
    verify_rational_integral_domain(&d, &finite(0), &finite(1), &mut Unbounded).unwrap();
    verify_rational_integral_domain(&d, &finite(1), &finite(0), &mut Unbounded).unwrap();
    assert!(matches!(
        verify_rational_integral_domain(&d, &finite(-1), &finite(1), &mut Unbounded),
        Err(Error::InteriorSingularity)
    ));
    let mut witness = reciprocal_integral();
    witness.lower_limit = reciprocal_limit(0);
    assert!(matches!(
        verify_rational_definite_integral(
            &p(&[1]),
            &p(&[0, 0, 1]),
            &finite(0),
            &finite(2),
            &witness,
            &mut Unbounded,
        ),
        Err(Error::DivergentEndpoint)
    ));
    assert!(
        verify_rational_integral_domain(&p(&[0, 1]), &finite(0), &finite(0), &mut Unbounded)
            .is_err()
    );
    assert!(
        verify_rational_integral_domain(
            &p(&[1]),
            &Endpoint::PositiveInfinity,
            &Endpoint::PositiveInfinity,
            &mut Unbounded
        )
        .is_err()
    );
    verify_rational_integral_domain(&p(&[1]), &finite(0), &finite(0), &mut Unbounded).unwrap();
}

#[test]
fn sturm_domain_agrees_with_all_known_open_interval_root_counts() {
    fn multiply(a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
        let mut product = vec![q(0); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                product[i + j] += ai * bj;
            }
        }
        product
    }
    // (x+2)^3 x^2 (x-1)^4 (x^2+1), including odd/even repeated
    // roots and a factor with no real roots. The expected domain comes
    // directly from the explicitly known real factors.
    let mut denominator = p(&[1, 0, 1]);
    for (root, multiplicity) in [(-2i64, 3), (0, 2), (1, 4)] {
        for _ in 0..multiplicity {
            denominator = multiply(&denominator, &p(&[-root, 1]));
        }
    }
    for a in -3i64..=3 {
        for b in -3i64..=3 {
            let excluded = [-2i64, 0, 1].into_iter().any(|root| {
                if a == b {
                    root == a
                } else {
                    a.min(b) < root && root < a.max(b)
                }
            });
            let accepted = verify_rational_integral_domain(
                &denominator,
                &finite(a),
                &finite(b),
                &mut Unbounded,
            )
            .is_ok();
            assert_eq!(accepted, !excluded, "interval ({a}, {b})");
        }
    }
}

#[test]
fn bounded_canonical_input_and_transport_reject_before_acceptance() {
    for d in [
        vec![],
        p(&[0]),
        p(&[1, 0]),
        vec![q(1); MAX_DEFINITE_COEFFICIENTS + 1],
    ] {
        assert!(
            verify_rational_integral_domain(&d, &finite(1), &finite(2), &mut Unbounded).is_err()
        );
    }
    let huge = BigRational::from_integer(BigInt::from(2).pow(4_097));
    assert!(
        verify_rational_integral_domain(
            std::slice::from_ref(&huge),
            &finite(1),
            &finite(2),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_rational_integral_domain(
            &p(&[1]),
            &Endpoint::Finite(huge),
            &finite(2),
            &mut Unbounded
        )
        .is_err()
    );
    let witness = reciprocal_integral();
    let mut json = serde_json::to_value(&witness).unwrap();
    json["verified"] = serde_json::Value::Bool(true);
    assert!(serde_json::from_value::<RationalDefiniteIntegralCertificate>(json).is_err());
    let decoded: RationalDefiniteIntegralCertificate =
        serde_json::from_slice(&serde_json::to_vec(&witness).unwrap()).unwrap();
    verify_rational_definite_integral(
        &p(&[1]),
        &p(&[0, 0, 1]),
        &finite(1),
        &finite(2),
        &decoded,
        &mut Unbounded,
    )
    .unwrap();
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
fn verifier_cancellation_never_returns_an_accepted_integral() {
    assert!(matches!(
        verify_rational_definite_integral(
            &p(&[1]),
            &p(&[0, 0, 1]),
            &finite(1),
            &finite(2),
            &reciprocal_integral(),
            &mut Cancelled,
        ),
        Err(Error::Meter(MeterError::Cancelled))
    ));
}
