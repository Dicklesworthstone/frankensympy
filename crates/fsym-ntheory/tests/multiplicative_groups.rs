use fsym_ntheory::{
    NTheoryError, is_primitive_root, multiplicative_order, multiplicative_order_with_work_limit,
    primitive_root, primitive_root_with_work_limit, verify_multiplicative_order,
};
use std::collections::BTreeMap;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn brute_order(a: u64, n: u64) -> u64 {
    let mut power = 1;
    for exponent in 1..n {
        power = power * a % n;
        if power == 1 {
            return exponent;
        }
    }
    panic!("unit has no order: {a} modulo {n}")
}

#[test]
fn every_small_unit_order_and_primitive_root_matches_complete_enumeration() {
    for n in 2..=128u64 {
        let units: Vec<_> = (1..n).filter(|&a| gcd(a, n) == 1).collect();
        let phi = units.len() as u64;
        let mut generators = Vec::new();
        for a in units {
            let expected = brute_order(a, n);
            assert_eq!(multiplicative_order(a, n).unwrap(), expected, "{a}, {n}");
            assert_eq!(multiplicative_order(a + 11 * n, n).unwrap(), expected);
            assert_eq!(is_primitive_root(a, n).unwrap(), expected == phi);
            if expected == phi {
                generators.push(a);
            }
        }
        assert_eq!(
            primitive_root(n, true).unwrap(),
            generators.first().copied()
        );
        match primitive_root(n, false).unwrap() {
            Some(root) => assert!(generators.contains(&root), "n={n}, root={root}"),
            None => assert!(generators.is_empty(), "n={n}"),
        }
    }
}

#[test]
fn composite_cyclic_groups_and_noncyclic_groups_are_distinguished() {
    for (n, expected) in [(2, 1), (3, 2), (4, 3), (9, 2), (18, 5), (25, 2), (50, 3)] {
        assert_eq!(primitive_root(n, true).unwrap(), Some(expected));
        assert!(is_primitive_root(expected, n).unwrap());
    }
    for n in [8, 12, 15, 16, 24, 45, 72, 105, 1u64 << 63] {
        assert_eq!(primitive_root(n, true).unwrap(), None);
        assert_eq!(primitive_root(n, false).unwrap(), None);
        assert!(!is_primitive_root(n - 1, n).unwrap());
    }
}

#[test]
fn large_prime_powers_and_terminal_word_moduli_are_exact() {
    let odd_power = 3u64.pow(40);
    assert_eq!(
        multiplicative_order(2, odd_power).unwrap(),
        2 * 3u64.pow(39)
    );
    assert_eq!(primitive_root(odd_power, false).unwrap(), Some(2));
    let twice_power = 2 * 3u64.pow(39);
    let lifted = primitive_root(twice_power, false).unwrap().unwrap();
    assert!(lifted < twice_power && lifted % 2 == 1);
    assert_eq!(
        multiplicative_order(lifted, twice_power).unwrap(),
        2 * 3u64.pow(38)
    );
    assert_eq!(multiplicative_order(5, 1u64 << 63).unwrap(), 1u64 << 61);
    let prime = 18_446_744_073_709_551_557;
    assert_eq!(multiplicative_order(prime - 1, prime).unwrap(), 2);
    let root = primitive_root(prime, true).unwrap().unwrap();
    assert!(is_primitive_root(root, prime).unwrap());
    assert_eq!(multiplicative_order(u64::MAX - 1, u64::MAX).unwrap(), 2);
}

#[test]
fn prime_generator_that_requires_lifting_is_not_reused_unchanged() {
    // 5 is the least primitive root modulo 40487, but is a Wieferich base
    // there; the lifted generator must be adjusted modulo the square.
    let p = 40_487u64;
    let prime_root = primitive_root(p, true).unwrap().unwrap();
    let lifted = primitive_root(p * p, false).unwrap().unwrap();
    assert_eq!(prime_root, 5);
    assert_ne!(lifted, prime_root);
    assert!(is_primitive_root(lifted, p * p).unwrap());
}

#[test]
fn order_verifier_rejects_nonminimal_unbound_and_forged_proposals() {
    let factors = BTreeMap::from([(2, 1), (3, 1)]);
    verify_multiplicative_order(3, 7, 6, &factors).unwrap();
    assert!(verify_multiplicative_order(2, 7, 6, &factors).is_err());
    assert!(verify_multiplicative_order(3, 9, 6, &factors).is_err());
    assert!(verify_multiplicative_order(3, 7, 12, &factors).is_err());
    assert!(verify_multiplicative_order(3, 7, 6, &BTreeMap::from([(6, 1)])).is_err());
    assert!(verify_multiplicative_order(3, 7, 6, &BTreeMap::from([(2, 0), (3, 1)])).is_err());
    assert!(verify_multiplicative_order(3, 7, 6, &BTreeMap::from([(2, u32::MAX)])).is_err());
    assert!(verify_multiplicative_order(1, 7, 0, &BTreeMap::new()).is_err());
    assert!(verify_multiplicative_order(1, 1, 1, &BTreeMap::new()).is_err());
    verify_multiplicative_order(1, u64::MAX, 1, &BTreeMap::new()).unwrap();
}

#[test]
fn invalid_domains_and_exhausted_search_never_become_negative_decisions() {
    for n in [0, 1] {
        assert_eq!(
            multiplicative_order(1, n),
            Err(NTheoryError::InvalidGroupModulus)
        );
        assert_eq!(
            primitive_root(n, true),
            Err(NTheoryError::InvalidGroupModulus)
        );
        assert_eq!(
            is_primitive_root(1, n),
            Err(NTheoryError::InvalidGroupModulus)
        );
    }
    for (a, n) in [(0, 2), (2, 4), (3, 9), (12, 18)] {
        assert_eq!(multiplicative_order(a, n), Err(NTheoryError::NotCoprime));
        assert_eq!(is_primitive_root(a, n), Err(NTheoryError::NotCoprime));
    }
    for smallest in [false, true] {
        assert_eq!(
            primitive_root_with_work_limit(17, smallest, 0),
            Err(NTheoryError::ModularGroupWorkLimitExceeded)
        );
    }
    assert_eq!(
        multiplicative_order_with_work_limit(3, 17, 0),
        Err(NTheoryError::ModularGroupWorkLimitExceeded)
    );
    assert_eq!(
        primitive_root_with_work_limit(17, true, 3),
        Err(NTheoryError::ModularGroupWorkLimitExceeded)
    );
}
