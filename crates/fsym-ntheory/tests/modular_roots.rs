use fsym_ntheory::{
    MAX_MODULAR_ROOTS, NTheoryError, factorint, is_quadratic_residue, modular_square_root,
    modular_square_roots, verify_modular_square_roots,
};
use std::collections::BTreeMap;

fn brute_roots(a: u64, modulus: u64) -> Vec<u64> {
    (0..modulus)
        .filter(|&x| u128::from(x) * u128::from(x) % u128::from(modulus) == u128::from(a % modulus))
        .collect()
}

#[test]
fn every_residue_of_small_composite_and_prime_moduli_matches_enumeration() {
    for modulus in 1..=128u64 {
        for a in 0..modulus {
            let expected = brute_roots(a, modulus);
            let actual = modular_square_roots(a, modulus).unwrap();
            assert_eq!(actual, expected, "a={a}, modulus={modulus}");
            assert_eq!(
                is_quadratic_residue(a, modulus).unwrap(),
                !expected.is_empty()
            );
            match modular_square_root(a, modulus).unwrap() {
                Some(root) => {
                    assert!(
                        expected.contains(&root),
                        "a={a}, modulus={modulus}, root={root}"
                    );
                    assert!(root <= modulus / 2);
                }
                None => assert!(expected.is_empty()),
            }
        }
    }
}

#[test]
fn prime_power_nonunits_and_crt_multiplicities_are_complete() {
    for (a, modulus, expected) in [
        (0, 9, vec![0, 3, 6]),
        (9, 27, vec![3, 6, 12, 15, 21, 24]),
        (4, 8, vec![2, 6]),
        (1, 16, vec![1, 7, 9, 15]),
        (4, 36, vec![2, 16, 20, 34]),
        (0, 36, vec![0, 6, 12, 18, 24, 30]),
        (1, 105, vec![1, 29, 34, 41, 64, 71, 76, 104]),
        (3, 27, vec![]),
        (2, 8, vec![]),
        (3, 8, vec![]),
        (5, 8, vec![]),
    ] {
        assert_eq!(modular_square_roots(a, modulus).unwrap(), expected);
        assert_eq!(
            modular_square_roots(a + 17 * modulus, modulus).unwrap(),
            expected
        );
    }
}

#[test]
fn large_odd_prime_powers_lift_and_large_semiprimes_recombine_exactly() {
    for modulus in [
        1_000_003u64.pow(3),
        3u64.pow(40),
        1_000_000_007u64 * 1_000_000_009,
        4_294_967_279u64 * 4_294_967_291,
        18_446_744_073_709_551_557,
    ] {
        let seed = 1_234_567_890u64;
        let a = (u128::from(seed) * u128::from(seed) % u128::from(modulus)) as u64;
        let roots = modular_square_roots(a, modulus).unwrap();
        assert!(roots.contains(&(seed % modulus)), "modulus={modulus}");
        assert!(roots.contains(&((modulus - seed % modulus) % modulus)));
        let factors = factorint(modulus).unwrap();
        verify_modular_square_roots(a, modulus, &factors, &roots).unwrap();
        assert!(is_quadratic_residue(a, modulus).unwrap());
    }
}

#[test]
fn two_adic_lifting_reaches_the_u64_boundary_without_overflow() {
    let modulus = 1u64 << 63;
    assert_eq!(
        modular_square_roots(1, modulus).unwrap(),
        vec![1, (1u64 << 62) - 1, (1u64 << 62) + 1, modulus - 1]
    );
    assert_eq!(modular_square_root(1, modulus).unwrap(), Some(1));
    assert!(!is_quadratic_residue(3, modulus).unwrap());
    assert_eq!(modular_square_roots(3, modulus).unwrap(), Vec::<u64>::new());
}

#[test]
fn single_root_and_existence_do_not_materialize_huge_complete_sets() {
    let modulus = 1u64 << 63;
    for (a, count, selected) in [(0, 1u64 << 31, 0), (1u64 << 60, 1u64 << 32, 1u64 << 30)] {
        assert_eq!(
            modular_square_roots(a, modulus),
            Err(NTheoryError::ModularRootCountExceeded(count))
        );
        assert_eq!(modular_square_root(a, modulus).unwrap(), Some(selected));
        assert!(is_quadratic_residue(a, modulus).unwrap());
    }
    // A later nonresidue makes the complete answer empty, even when an earlier
    // prime-power factor by itself would have too many roots to materialize.
    assert_eq!(
        modular_square_roots(1u64 << 61, (1u64 << 61) * 3).unwrap(),
        Vec::<u64>::new()
    );
}

#[test]
fn all_roots_limit_is_checked_at_the_exact_cardinality_boundary() {
    let roots = modular_square_roots(0, 1u64 << 32).unwrap();
    assert_eq!(roots.len(), MAX_MODULAR_ROOTS);
    assert_eq!(roots[0], 0);
    assert_eq!(*roots.last().unwrap(), (1u64 << 32) - (1u64 << 16));
    assert_eq!(
        modular_square_roots(0, 1u64 << 34),
        Err(NTheoryError::ModularRootCountExceeded(1u64 << 17))
    );
}

#[test]
fn verifier_rejects_missing_duplicate_invalid_and_unbound_roots() {
    let factors = BTreeMap::from([(2, 2), (3, 2)]);
    let roots = [2, 16, 20, 34];
    verify_modular_square_roots(4, 36, &factors, &roots).unwrap();
    for mutant in [
        vec![2, 16, 20],
        vec![2, 16, 20, 20],
        vec![2, 16, 20, 35],
        vec![2, 16, 20, 70],
        vec![34, 20, 16, 2],
        vec![2, 16, 20, 34, 35],
    ] {
        assert!(
            verify_modular_square_roots(4, 36, &factors, &mutant).is_err(),
            "{mutant:?}"
        );
    }
    assert!(verify_modular_square_roots(5, 36, &factors, &roots).is_err());
    assert!(verify_modular_square_roots(4, 36, &BTreeMap::from([(2, 2), (9, 1)]), &roots).is_err());
    assert!(verify_modular_square_roots(4, 36, &BTreeMap::from([(2, 2), (3, 1)]), &roots).is_err());
    assert!(verify_modular_square_roots(4, 36, &BTreeMap::from([(2, u32::MAX)]), &roots).is_err());
    assert!(verify_modular_square_roots(1, 36, &factors, &[]).is_err());
    verify_modular_square_roots(2, 36, &factors, &[]).unwrap();
}

#[test]
fn zero_and_unit_moduli_have_explicit_behavior() {
    for a in [0, 1, u64::MAX] {
        assert_eq!(
            modular_square_roots(a, 0),
            Err(NTheoryError::InvalidModulus)
        );
        assert_eq!(modular_square_root(a, 0), Err(NTheoryError::InvalidModulus));
        assert_eq!(
            is_quadratic_residue(a, 0),
            Err(NTheoryError::InvalidModulus)
        );
        assert_eq!(modular_square_roots(a, 1).unwrap(), vec![0]);
        assert_eq!(modular_square_root(a, 1).unwrap(), Some(0));
        assert!(is_quadratic_residue(a, 1).unwrap());
        verify_modular_square_roots(a, 1, &BTreeMap::new(), &[0]).unwrap();
    }
}
