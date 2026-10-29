//! Public-boundary checks for exact factorization certificates. The forged
//! witnesses below retain a matching product where possible, so acceptance
//! must check normalization and multiplicity as well as multiplication.

use fsym_core::{BigInt, BigRational, Symbol};
use fsym_polys::factorization::{
    CompleteFactorTerm, CompleteFactorization, IrreducibilityWitness, complete_factorization,
    verify_certified_irreducible_factorization, verify_complete_factorization,
};
use fsym_polys::univariate::UnivariatePoly;
use num_traits::{One, Zero};

fn poly(coefficients: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(
        Symbol::new("x"),
        coefficients
            .iter()
            .map(|value| BigRational::from_integer(BigInt::from(*value)))
            .collect(),
    )
}

fn term(coefficients: &[i64], multiplicity: usize) -> CompleteFactorTerm {
    CompleteFactorTerm {
        poly: poly(coefficients),
        multiplicity,
        irreducibility: None,
    }
}

fn certificate(factors: Vec<CompleteFactorTerm>) -> CompleteFactorization {
    CompleteFactorization {
        scale: BigRational::one(),
        factors,
    }
}

fn assert_rejected(input: &UnivariatePoly, witness: &CompleteFactorization) {
    assert!(verify_complete_factorization(input, witness).is_err());
    assert!(verify_certified_irreducible_factorization(input, witness).is_err());
}

#[test]
fn accepts_hand_checked_rational_scale_and_multiplicities() {
    // 3/5 * (x-1)^2 * (x+2)^3
    // = 3/5 * (x^5 + 4x^4 + x^3 - 10x^2 - 4x + 8).
    let scale = BigRational::new(BigInt::from(3), BigInt::from(5));
    let expected = UnivariatePoly::new(
        Symbol::new("x"),
        [8, -4, -10, 1, 4, 1]
            .into_iter()
            .map(|value| scale.clone() * BigRational::from_integer(BigInt::from(value)))
            .collect(),
    );
    let witness = CompleteFactorization {
        scale,
        factors: vec![term(&[-1, 1], 2), term(&[2, 1], 3)],
    };
    assert_eq!(witness.expand(Symbol::new("x")).unwrap(), expected);
    verify_complete_factorization(&expected, &witness).unwrap();
    verify_certified_irreducible_factorization(&expected, &witness).unwrap();
}

#[test]
fn accepts_only_canonical_zero_and_constant_decompositions() {
    for value in [0, 1, -7] {
        let input = poly(&[value]);
        let witness = CompleteFactorization {
            scale: BigRational::from_integer(BigInt::from(value)),
            factors: vec![],
        };
        verify_certified_irreducible_factorization(&input, &witness).unwrap();
        assert_eq!(witness.expand(Symbol::new("x")).unwrap(), input);
        assert_eq!(complete_factorization(&input).unwrap(), witness);
    }
    let hidden_factors = CompleteFactorization {
        scale: BigRational::zero(),
        factors: vec![term(&[-1, 1], 1)],
    };
    assert_rejected(&poly(&[0]), &hidden_factors);
    assert!(hidden_factors.expand(Symbol::new("x")).is_err());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn rejects_multiplicity_that_used_to_truncate_to_one() {
    let witness = certificate(vec![term(&[-1, 1], (1usize << 32) + 1)]);
    // The old `as u32` expansion reconstructed x-1 and both public verifiers
    // accepted this false multiplicity claim.
    assert_rejected(&poly(&[-1, 1]), &witness);
    assert!(witness.expand(Symbol::new("x")).is_err());
}

#[test]
fn rejects_weighted_degree_overflow_before_powering() {
    let witness = certificate(vec![term(&[1, 0, 1], usize::MAX)]);
    let error = verify_complete_factorization(&poly(&[1, 0, 1]), &witness).unwrap_err();
    assert!(error.to_string().contains("weighted degree overflowed"));
    assert!(witness.expand(Symbol::new("x")).is_err());
}

#[test]
fn rejects_zero_multiplicities_and_wrong_total_degree() {
    assert_rejected(&poly(&[1]), &certificate(vec![term(&[-1, 1], 0)]));
    assert_rejected(&poly(&[1, -2, 1]), &certificate(vec![term(&[-1, 1], 1)]));
    assert_rejected(&poly(&[-1, 1]), &certificate(vec![term(&[-1, 1], 2)]));
    assert!(
        certificate(vec![term(&[-1, 1], 0)])
            .expand(Symbol::new("x"))
            .is_err()
    );
}

#[test]
fn rejects_constant_and_zero_factors_without_expansion() {
    for factor in [term(&[1], 1), term(&[1], usize::MAX), term(&[0], 1)] {
        let witness = certificate(vec![factor]);
        assert_rejected(&poly(&[1]), &witness);
        assert!(witness.expand(Symbol::new("x")).is_err());
    }
    // Inserting a unit does not change a valid product, but it does violate
    // the claimed normalized factor list.
    let witness = certificate(vec![term(&[-1, 1], 2), term(&[1], 1)]);
    assert_rejected(&poly(&[1, -2, 1]), &witness);
}

#[test]
fn rejects_repeated_roots_hidden_inside_a_factor() {
    let input = poly(&[1, -2, 1]);
    let witness = certificate(vec![term(&[1, -2, 1], 1)]);
    assert_eq!(witness.expand(Symbol::new("x")).unwrap(), input);
    let error = verify_complete_factorization(&input, &witness).unwrap_err();
    assert!(error.to_string().contains("not square-free"));
    assert_rejected(&input, &witness);
}

#[test]
fn rejects_split_duplicate_factors_despite_matching_product() {
    let input = poly(&[1, -2, 1]);
    let witness = certificate(vec![term(&[-1, 1], 1), term(&[-1, 1], 1)]);
    assert_eq!(witness.expand(Symbol::new("x")).unwrap(), input);
    assert_rejected(&input, &witness);
}

#[test]
fn validates_ring_shape_scale_and_modulus_before_arithmetic() {
    let input = poly(&[-1, 1]);
    let mut wrong_ring = term(&[-1, 1], 1);
    wrong_ring.poly.gen_sym = Symbol::new("y");
    assert_rejected(&input, &certificate(vec![wrong_ring]));

    for coefficients in [vec![], vec![BigRational::one(), BigRational::zero()]] {
        let malformed = CompleteFactorTerm {
            poly: UnivariatePoly {
                gen_sym: Symbol::new("x"),
                coeffs: coefficients,
            },
            multiplicity: 1,
            irreducibility: None,
        };
        let witness = certificate(vec![malformed]);
        assert_rejected(&input, &witness);
        assert!(witness.expand(Symbol::new("x")).is_err());
    }

    let wrong_scale = CompleteFactorization {
        scale: BigRational::from_integer(BigInt::from(2)),
        factors: vec![term(&[-1, 1], 1)],
    };
    assert_rejected(&input, &wrong_scale);
    let mut oversized_modulus = term(&[-1, 1], 1);
    oversized_modulus.irreducibility = Some(IrreducibilityWitness {
        prime: BigInt::one() << 33u32,
    });
    let witness = certificate(vec![oversized_modulus]);
    assert_rejected(&input, &witness);
    assert!(witness.expand(Symbol::new("x")).is_err());
}

#[test]
fn rejects_oversized_input_factor_and_scale_components() {
    let huge = BigInt::one() << 1_048_576u32;
    for coefficient in [
        BigRational::from_integer(huge.clone()),
        BigRational::new(BigInt::one(), huge.clone()),
    ] {
        let oversized = UnivariatePoly::new(
            Symbol::new("x"),
            vec![coefficient.clone(), BigRational::one()],
        );
        let witness = certificate(vec![CompleteFactorTerm {
            poly: oversized.clone(),
            multiplicity: 1,
            irreducibility: None,
        }]);
        assert_rejected(&oversized, &witness);
        assert!(complete_factorization(&oversized).is_err());
        assert!(witness.expand(Symbol::new("x")).is_err());
        let scale_only = CompleteFactorization {
            scale: coefficient,
            factors: vec![],
        };
        assert!(scale_only.expand(Symbol::new("x")).is_err());
    }
}

#[test]
fn charges_repeated_factor_bits_before_materializing_the_power() {
    let mut coefficients = vec![BigRational::zero(); 65];
    coefficients[64] = BigRational::one();
    let input = UnivariatePoly::new(Symbol::new("x"), coefficients);
    let large_factor = UnivariatePoly::new(
        Symbol::new("x"),
        vec![
            BigRational::from_integer(BigInt::one() << 17_000u32),
            BigRational::one(),
        ],
    );
    let witness = certificate(vec![CompleteFactorTerm {
        poly: large_factor,
        multiplicity: 64,
        irreducibility: None,
    }]);
    let error = verify_complete_factorization(&input, &witness).unwrap_err();
    assert!(error.to_string().contains("product exceeds"));
    assert!(witness.expand(Symbol::new("x")).is_err());
}

#[test]
fn admits_large_common_scaling_within_the_arithmetic_envelope() {
    // Raw rational height is intentionally not confused with the generator's
    // 2048-bit primitive-part limit.
    let scale = BigRational::from_integer(BigInt::one() << 4096u32);
    let input = UnivariatePoly::new(
        Symbol::new("x"),
        vec![scale.clone(), -scale.clone() - scale.clone(), scale.clone()],
    );
    let witness = CompleteFactorization {
        scale,
        factors: vec![term(&[-1, 1], 2)],
    };
    verify_certified_irreducible_factorization(&input, &witness).unwrap();
}

#[test]
fn missing_irreducibility_is_never_promoted_to_complete_proof() {
    // x^4 + 1 is square-free, but has no irreducible reduction modulo any
    // prime. Exact decomposition is still admissible without that witness.
    let input = poly(&[1, 0, 0, 0, 1]);
    let witness = certificate(vec![term(&[1, 0, 0, 0, 1], 1)]);
    verify_complete_factorization(&input, &witness).unwrap();
    assert!(verify_certified_irreducible_factorization(&input, &witness).is_err());
}
