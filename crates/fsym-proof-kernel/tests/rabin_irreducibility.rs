//! Irreducibility evidence must prove the exact polynomial/domain claim it names.

use fsym_assumptions::{Domain, ImmutableAssumptionsSnapshot, Predicate};
use fsym_budget::{Budget, BudgetLimits, Dimension, Unbounded};
use fsym_core::{BigInt, Expr};
use fsym_proof_kernel::claim::MAX_MODULAR_IRREDUCIBILITY_DEGREE;
use fsym_proof_kernel::{
    CertificatePayload, Claim, ClaimKind, DerivationStep, DerivationTree, ProofKernel, ProofRule,
    StepId, claim_verification_units, verify_derivation_independent,
};

fn irreducibility(coefficients: &[i64], prime: i64) -> Claim {
    Claim::IrreducibleModPrime {
        coeffs: coefficients.iter().copied().map(BigInt::from).collect(),
        prime: BigInt::from(prime),
    }
}

fn certificate(coefficients: &[i64], prime: i64) -> CertificatePayload {
    CertificatePayload::RabinIrreducibilityModPrime {
        coeffs: coefficients.iter().copied().map(BigInt::from).collect(),
        prime: BigInt::from(prime),
    }
}

fn rule(claim: Claim, certificate: CertificatePayload) -> ProofRule {
    ProofRule::CertificateLemma {
        family: "RabinIrreducibilityModPrime".into(),
        claim,
        certificate,
    }
}

fn reject_direct_and_replay(claim: Claim, certificate: CertificatePayload) {
    let context = ImmutableAssumptionsSnapshot::empty();
    let proof_rule = rule(claim.clone(), certificate);
    let mut kernel = ProofKernel::new((*context).clone());
    assert!(
        kernel.add_step(proof_rule.clone(), &mut Unbounded).is_err(),
        "the kernel accepted {claim:?}"
    );
    assert_eq!(kernel.step_count(), 0, "refused evidence must not publish");

    let tree = DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(0),
            rule: proof_rule,
            claim,
        }],
        root: StepId(0),
    };
    assert!(verify_derivation_independent(&tree, &context).is_err());
}

#[test]
fn irreducibility_cannot_prove_unrelated_nonzero_or_predicate_claims() {
    for claim in [
        Claim::NonZero(Expr::from_i64(0)),
        Claim::predicate(Expr::from_i64(-1), Predicate::Positive),
        Claim::predicate(Expr::from_i64(0), Predicate::NonZero),
    ] {
        // x^2 + x + 1 is irreducible over F_2; none of these claims follow.
        reject_direct_and_replay(claim, certificate(&[1, 1, 1], 2));
    }
}

#[test]
fn irreducibility_is_not_a_nonvanishing_claim_at_an_unrestricted_argument() {
    let x = Expr::symbol("x");
    let polynomial = Expr::Add(vec![
        Expr::pow(x.clone(), Expr::from_i64(2)),
        x,
        Expr::from_i64(1),
    ]);
    for claim in [
        Claim::NonZero(polynomial.clone()),
        Claim::predicate(polynomial.clone(), Predicate::Positive),
        Claim::domain_membership(polynomial, Domain::RR),
        Claim::equality(Expr::from_i64(1), Expr::from_i64(1)),
        Claim::AlgebraicIdentity {
            lhs: Expr::from_i64(1),
            rhs: Expr::from_i64(1),
        },
    ] {
        reject_direct_and_replay(claim, certificate(&[1, 1, 1], 2));
    }
}

#[test]
fn exact_irreducibility_claims_construct_serialize_and_replay() {
    for (coefficients, prime) in [
        (vec![0, 1], 2),
        (vec![1, 2], 3), // Leading coefficient need not be monic.
        (vec![1, 1, 1], 2),
        (vec![1, 0, 1], 3),
        (vec![1, 1, 0, 1], 2),
        (vec![1, 0, 1, 1], 2),
        (vec![1, 1, 0, 0, 1], 2),
        (vec![1, 1], 4_294_967_291), // Largest 32-bit prime.
    ] {
        let context = ImmutableAssumptionsSnapshot::empty();
        let claim = irreducibility(&coefficients, prime);
        assert_eq!(claim.kind(), ClaimKind::IrreducibleModPrime);
        assert!(claim.lhs().is_none());
        assert!(claim.rhs().is_none());
        let mut kernel = ProofKernel::new((*context).clone());
        let root = kernel
            .add_step(
                rule(claim.clone(), certificate(&coefficients, prime)),
                &mut Unbounded,
            )
            .expect("known irreducible polynomial should be accepted");
        assert_eq!(kernel.get_claim(root), Some(&claim));
        let tree = kernel.export_derivation(root).unwrap();
        let encoded = serde_json::to_vec(&tree).unwrap();
        let decoded: DerivationTree = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            verify_derivation_independent(&decoded, &context).unwrap(),
            claim
        );
        assert_eq!(decoded.digest(), tree.digest());
    }
}

#[test]
fn exact_polynomial_and_prime_binding_rejects_other_valid_witnesses() {
    // Each replacement witness is valid on its own, but proves another claim.
    reject_direct_and_replay(
        irreducibility(&[1, 1, 0, 1], 2),
        certificate(&[1, 0, 1, 1], 2),
    );
    reject_direct_and_replay(irreducibility(&[1, 1, 1], 2), certificate(&[1, 1, 1], 5));
    reject_direct_and_replay(irreducibility(&[1, 0, 1], 3), certificate(&[2, 0, 2], 3));
}

#[test]
fn malformed_reducible_and_noncanonical_polynomials_are_rejected() {
    for (coefficients, prime) in [
        (vec![], 2),
        (vec![0], 2),
        (vec![1], 2),
        (vec![1, 1, 1, 0], 2),
        (vec![1, 1, 3], 2),
        (vec![1, -1, 1], 2),
        (vec![3, 1, 1], 2),
        (vec![1, 0, 1], 2),       // (x + 1)^2
        (vec![1, 0, 0, 0, 1], 2), // (x + 1)^4
        (vec![1, 0, 1], 5),       // (x + 2)(x + 3)
        (vec![1, 1], 0),
        (vec![1, 1], 1),
        (vec![1, 1], -2),
        (vec![1, 1], 4),
        (vec![1, 1], 9),
        (vec![1, 1], 561),
        (vec![1, 1], 4_294_967_311),
    ] {
        reject_direct_and_replay(
            irreducibility(&coefficients, prime),
            certificate(&coefficients, prime),
        );
    }
}

#[test]
fn other_certificate_families_cannot_claim_modular_irreducibility() {
    let context = ImmutableAssumptionsSnapshot::empty();
    let claim = irreducibility(&[1, 1, 1], 2);
    for (family, payload) in [
        (
            "PolynomialNormalForm",
            CertificatePayload::PolynomialNormalForm,
        ),
        (
            "RealBall",
            CertificatePayload::RealBall(fsym_core::RealBall::from_i64(1)),
        ),
    ] {
        let mut kernel = ProofKernel::new((*context).clone());
        let rule = ProofRule::CertificateLemma {
            family: family.into(),
            claim: claim.clone(),
            certificate: payload,
        };
        assert!(kernel.add_step(rule.clone(), &mut Unbounded).is_err());
        let tree = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule,
                claim: claim.clone(),
            }],
            root: StepId(0),
        };
        assert!(verify_derivation_independent(&tree, &context).is_err());
    }
    reject_direct_and_replay(claim, CertificatePayload::PolynomialNormalForm);
}

#[test]
fn an_exported_claim_cannot_be_replaced_with_a_false_conclusion() {
    let context = ImmutableAssumptionsSnapshot::empty();
    let mut kernel = ProofKernel::new((*context).clone());
    let root = kernel
        .add_step(
            rule(irreducibility(&[1, 1, 1], 2), certificate(&[1, 1, 1], 2)),
            &mut Unbounded,
        )
        .unwrap();
    let mut tree = kernel.export_derivation(root).unwrap();
    tree.steps[0].claim = Claim::NonZero(Expr::from_i64(0));
    assert!(verify_derivation_independent(&tree, &context).is_err());
}

#[test]
fn claim_identity_includes_coefficients_prime_and_proposition_kind() {
    let claim = irreducibility(&[1, 1, 0, 1], 2);
    for other in [
        irreducibility(&[1, 0, 1, 1], 2),
        irreducibility(&[1, 1, 0, 1], 3),
        Claim::NonZero(Expr::from_i64(0)),
    ] {
        assert_ne!(claim.digest(), other.digest());
    }
}

#[test]
fn modular_claims_and_certificates_are_bounded_before_verification() {
    let coefficients = vec![1; MAX_MODULAR_IRREDUCIBILITY_DEGREE + 2];
    let claim = irreducibility(&coefficients, 2);
    let witness = certificate(&coefficients, 2);
    assert!(claim_verification_units(&claim).is_err());
    assert!(serde_json::from_slice::<Claim>(&serde_json::to_vec(&claim).unwrap()).is_err());
    assert!(
        serde_json::from_slice::<CertificatePayload>(&serde_json::to_vec(&witness).unwrap())
            .is_err()
    );
    reject_direct_and_replay(claim, witness);

    for (coefficients, prime) in [(vec![1, 1_i64 << 32], 2), (vec![1, 1], 1_i64 << 32)] {
        let claim = irreducibility(&coefficients, prime);
        let witness = certificate(&coefficients, prime);
        assert!(claim_verification_units(&claim).is_err());
        assert!(serde_json::from_slice::<Claim>(&serde_json::to_vec(&claim).unwrap()).is_err());
        assert!(
            serde_json::from_slice::<CertificatePayload>(&serde_json::to_vec(&witness).unwrap())
                .is_err()
        );
        reject_direct_and_replay(claim, witness);
    }
}

#[test]
fn a_refused_verification_budget_cannot_publish_a_step() {
    let context = ImmutableAssumptionsSnapshot::empty();
    let mut kernel = ProofKernel::new((*context).clone());
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 1;
    let mut budget = Budget::new(limits);
    assert!(
        kernel
            .add_step(
                rule(irreducibility(&[1, 1, 1], 2), certificate(&[1, 1, 1], 2)),
                &mut budget,
            )
            .is_err()
    );
    assert_eq!(kernel.step_count(), 0);

    let mut ordinary_budget = Budget::new(BudgetLimits::uniform(100, 0));
    assert!(
        kernel
            .add_step(
                rule(irreducibility(&[1, 1, 1], 2), certificate(&[1, 1, 1], 2)),
                &mut ordinary_budget,
            )
            .is_ok(),
        "small witnesses should retain a small verifier charge"
    );
}

#[test]
fn rabin_acceptance_matches_exhaustive_small_field_trial_division() {
    // This reference enumerates all possible proper monic divisors; it uses
    // neither Frobenius powers, GCDs, nor the production polynomial crates.
    fn digits(mut encoded: i64, count: usize, prime: i64) -> Vec<i64> {
        let mut result = Vec::with_capacity(count + 1);
        for _ in 0..count {
            result.push(encoded % prime);
            encoded /= prime;
        }
        result.push(1);
        result
    }

    fn has_proper_divisor(polynomial: &[i64], prime: i64) -> bool {
        let degree = polynomial.len() - 1;
        for divisor_degree in 1..=degree / 2 {
            for encoded in 0..prime.pow(divisor_degree as u32) {
                let divisor = digits(encoded, divisor_degree, prime);
                let mut remainder = polynomial.to_vec();
                for pivot in (divisor_degree..=degree).rev() {
                    let factor = remainder[pivot];
                    for (index, coefficient) in divisor.iter().enumerate() {
                        let slot = pivot - divisor_degree + index;
                        remainder[slot] =
                            (remainder[slot] - factor * coefficient).rem_euclid(prime);
                    }
                }
                if remainder.iter().all(|value| *value == 0) {
                    return true;
                }
            }
        }
        false
    }

    let context = ImmutableAssumptionsSnapshot::empty();
    for prime in [2_i64, 3, 5] {
        for degree in 1..=4 {
            for encoded in 0..prime.pow(degree) {
                let coefficients = digits(encoded, degree as usize, prime);
                let mut kernel = ProofKernel::new((*context).clone());
                let accepted = kernel
                    .add_step(
                        rule(
                            irreducibility(&coefficients, prime),
                            certificate(&coefficients, prime),
                        ),
                        &mut Unbounded,
                    )
                    .is_ok();
                assert_eq!(
                    accepted,
                    !has_proper_divisor(&coefficients, prime),
                    "polynomial {coefficients:?} over F_{prime}"
                );
            }
        }
    }
}
