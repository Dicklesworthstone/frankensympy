//! Exact reference, adversarial and large-cofactor coverage for integer factoring.

use fsym_ntheory::{
    NTheoryError, carmichael, divisor_count, divisor_sum, factorint, factorint_with_work_limit,
    mobius, prime_big_omega, prime_omega, totient, verify_prime_factorization,
};
use std::collections::BTreeMap;

fn trial_factor(mut n: u64) -> BTreeMap<u64, u32> {
    let mut result = BTreeMap::new();
    let mut divisor = 2;
    while divisor <= n / divisor {
        while n.is_multiple_of(divisor) {
            *result.entry(divisor).or_insert(0) += 1;
            n /= divisor;
        }
        divisor += 1;
    }
    if n > 1 {
        *result.entry(n).or_insert(0) += 1;
    }
    result
}

#[test]
fn complete_factorizations_match_an_exhaustive_small_trial_reference() {
    for n in 1..=2_048 {
        assert_eq!(factorint(n).unwrap(), trial_factor(n), "n={n}");
    }
}

#[test]
fn semiprimes_beyond_the_old_trial_limit_and_near_u64_max() {
    for (p, q) in [
        (1_000_003u64, 1_000_033u64),
        (1_000_000_007, 1_000_000_009),
        (2_147_483_647, 4_294_967_291),
        (4_294_967_279, 4_294_967_291),
    ] {
        let n = p.checked_mul(q).unwrap();
        let expected = BTreeMap::from([(p, 1), (q, 1)]);
        assert_eq!(factorint(n).unwrap(), expected, "n={n}");
        verify_prime_factorization(n, &expected).unwrap();
    }
}

#[test]
fn prime_powers_preserve_all_multiplicities() {
    for (prime, exponent) in [
        (2u64, 63u32),
        (3, 40),
        (101, 9),
        (1_000_003, 3),
        (4_294_967_291, 2),
    ] {
        let n = prime.checked_pow(exponent).unwrap();
        assert_eq!(factorint(n).unwrap(), BTreeMap::from([(prime, exponent)]));
    }
}

#[test]
fn terminal_values_and_primality_adversaries() {
    assert_eq!(factorint(0), Err(NTheoryError::ZeroFactorization));
    assert_eq!(factorint(1).unwrap(), BTreeMap::new());
    assert_eq!(
        factorint(u64::MAX).unwrap(),
        BTreeMap::from([
            (3, 1),
            (5, 1),
            (17, 1),
            (257, 1),
            (641, 1),
            (65_537, 1),
            (6_700_417, 1)
        ])
    );
    let terminal_prime = 18_446_744_073_709_551_557;
    assert_eq!(
        factorint(terminal_prime).unwrap(),
        BTreeMap::from([(terminal_prime, 1)])
    );
    // Strong pseudoprimes to successively larger initial witness sets must
    // never be accepted merely because their product reconstructs the input.
    for composite in [
        561,
        3_215_031_751,
        341_550_071_728_321,
        3_825_123_056_546_413_051,
    ] {
        assert!(verify_prime_factorization(composite, &BTreeMap::from([(composite, 1)])).is_err());
    }
}

#[test]
fn independent_verifier_rejects_omissions_multiplicities_and_composite_factors() {
    let input = 3u64.pow(4) * 5u64.pow(2) * 101;
    let original = BTreeMap::from([(3, 4), (5, 2), (101, 1)]);
    verify_prime_factorization(input, &original).unwrap();
    for mutant in [
        BTreeMap::from([(3, 4), (5, 2)]),
        BTreeMap::from([(3, 3), (5, 2), (101, 1)]),
        BTreeMap::from([(9, 2), (5, 2), (101, 1)]),
        BTreeMap::from([(1, 1), (3, 4), (5, 2), (101, 1)]),
        BTreeMap::from([(3, 0), (5, 2), (101, 1)]),
        BTreeMap::from([(3, u32::MAX)]),
        BTreeMap::from([(18_446_744_073_709_551_557, 2)]),
    ] {
        assert!(
            verify_prime_factorization(input, &mutant).is_err(),
            "{mutant:?}"
        );
    }
    assert!(verify_prime_factorization(input + 1, &original).is_err());
    assert!(verify_prime_factorization(1, &BTreeMap::from([(2, 1)])).is_err());
    assert!(verify_prime_factorization(0, &BTreeMap::new()).is_err());
}

#[test]
fn resource_refusal_returns_no_partial_factorization_and_restarts_deterministically() {
    let n = 1_000_000_007u64 * 1_000_000_009;
    for limit in [0, 1, 25, 4_096, 10_000] {
        assert!(matches!(
            factorint_with_work_limit(n, limit),
            Err(NTheoryError::FactorizationLimitExceeded(_))
        ));
    }
    let first = factorint(n).unwrap();
    assert_eq!(factorint(n).unwrap(), first);
    assert_eq!(factorint_with_work_limit(n, u64::MAX).unwrap(), first);
    // Factoring out a small divisor still must not publish a partial answer.
    assert!(matches!(
        factorint_with_work_limit(2 * 1_000_003u64 * 1_000_033, 100),
        Err(NTheoryError::FactorizationLimitExceeded(_))
    ));
}

#[test]
fn existing_arithmetic_functions_now_support_large_composite_inputs() {
    let p = 1_000_000_007u64;
    let q = 1_000_000_009u64;
    let n = p * q;
    assert_eq!(totient(n).unwrap(), (p - 1) * (q - 1));
    assert_eq!(divisor_count(n).unwrap(), 4);
    assert_eq!(divisor_sum(n, 1).unwrap(), (p + 1) * (q + 1));
    assert_eq!(mobius(n).unwrap(), 1);
    assert_eq!(prime_omega(n).unwrap(), 2);
    assert_eq!(prime_big_omega(n).unwrap(), 2);
    assert_eq!(carmichael(n).unwrap(), (p - 1) / 2 * (q - 1));
}
