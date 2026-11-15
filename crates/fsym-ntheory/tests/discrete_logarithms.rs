use fsym_ntheory::{
    MAX_DISCRETE_LOG_TABLE, NTheoryError, discrete_log, discrete_log_with_work_limit,
};

fn brute_log(n: u64, a: u64, b: u64) -> Option<u64> {
    let mut value = 1 % n;
    for exponent in 0..=n {
        if value == a % n {
            return Some(exponent);
        }
        value = value * b % n;
    }
    None
}

fn power(mut base: u64, mut exponent: u64, n: u64) -> u64 {
    let mut value = 1u128;
    while exponent > 0 {
        if exponent & 1 != 0 {
            value = value * u128::from(base) % u128::from(n);
        }
        base = (u128::from(base) * u128::from(base) % u128::from(n)) as u64;
        exponent >>= 1;
    }
    value as u64
}

#[test]
fn every_small_residue_and_base_matches_least_exponent_enumeration() {
    for n in 1..=48 {
        for a in 0..n {
            for b in 0..n {
                let expected = brute_log(n, a, b);
                let actual = discrete_log(n, a, b, None);
                match expected {
                    Some(exponent) => assert_eq!(actual.unwrap(), exponent, "n={n}, a={a}, b={b}"),
                    None => assert_eq!(
                        actual,
                        Err(NTheoryError::DiscreteLogDoesNotExist),
                        "n={n}, a={a}, b={b}"
                    ),
                }
            }
        }
    }
}

#[test]
fn repeated_baby_values_cannot_overwrite_the_least_exponent() {
    for (n, a, b, expected) in [
        (17, 16, 16, 1),
        (41, 40, 40, 1),
        (41, 15, 7, 3),
        (8, 3, 3, 1),
        (9, 4, 2, 2),
        (17, 13, 4, 3),
    ] {
        assert_eq!(discrete_log(n, a, b, None).unwrap(), expected);
        assert_eq!(
            discrete_log(n, a + 17 * n, b + 11 * n, None).unwrap(),
            expected
        );
    }
}

#[test]
fn nonunit_transients_and_periodic_tails_are_both_minimal() {
    for (n, a, b, expected) in [
        (9, 0, 3, 2),
        (12, 4, 2, 2),
        (12, 8, 2, 3),
        (24, 0, 6, 3),
        (8, 0, 0, 1),
        (8, 1, 0, 0),
        (1u64 << 63, 1u64 << 61, 2, 61),
        (1u64 << 63, 0, 2, 63),
    ] {
        assert_eq!(discrete_log(n, a, b, None).unwrap(), expected);
    }
    for (n, a, b) in [(8, 3, 2), (12, 6, 2), (9, 6, 3), (8, 0, 3), (8, 5, 3)] {
        assert_eq!(
            discrete_log(n, a, b, None),
            Err(NTheoryError::DiscreteLogDoesNotExist)
        );
    }
}

#[test]
fn supplied_annihilating_orders_reduce_to_an_independently_checked_exact_order() {
    assert_eq!(discrete_log(17, 16, 16, Some(2)).unwrap(), 1);
    assert_eq!(discrete_log(41, 15, 7, Some(40)).unwrap(), 3);
    assert_eq!(discrete_log(41, 15, 7, Some(80)).unwrap(), 3);
    assert_eq!(discrete_log(17, 13, 4, Some(8)).unwrap(), 3);
    assert_eq!(discrete_log(17, 16, 16, Some(16)).unwrap(), 1);
    assert_eq!(discrete_log(15, 4, 4, Some(4)).unwrap(), 1);
    assert_eq!(discrete_log(8, 1, 2, Some(2)).unwrap(), 0);
    assert_eq!(discrete_log(9, 0, 3, Some(6)).unwrap(), 2);
    assert_eq!(
        discrete_log(8, 5, 3, Some(2)),
        Err(NTheoryError::DiscreteLogDoesNotExist)
    );
    for (n, a, b, order) in [
        (17, 16, 16, 0),
        (17, 13, 4, 1),
        (17, 13, 4, 2),
        (17, 13, 4, 3),
        (17, 13, 4, 6),
        (12, 8, 2, 3),
        (1, 0, 0, 0),
    ] {
        assert_eq!(
            discrete_log(n, a, b, Some(order)),
            Err(NTheoryError::InvalidDiscreteLogOrder)
        );
    }
}

#[test]
fn large_unit_groups_and_high_word_products_do_not_use_float_arithmetic() {
    let n = 1_000_000_007;
    let exponent = 1_234_567;
    let target = power(5, exponent, n);
    assert_eq!(discrete_log(n, target, 5, None).unwrap(), exponent);
    assert_eq!(discrete_log(n, target, 5, Some(n - 1)).unwrap(), exponent);
    let n = 1u64 << 63;
    assert_eq!(
        discrete_log(n, 1 + (1u64 << 62), 1 + (1u64 << 61), None).unwrap(),
        2
    );
}

#[test]
fn table_limit_is_admitted_before_allocation_and_never_means_nonexistence() {
    assert_eq!(MAX_DISCRETE_LOG_TABLE, 65_536);
    let n = 1u64 << 34;
    let exponent = 123_456;
    assert_eq!(
        discrete_log(n, power(5, exponent, n), 5, Some(1u64 << 32)).unwrap(),
        exponent
    );
    assert!(matches!(
        discrete_log(1u64 << 35, 25, 5, None),
        Err(NTheoryError::DiscreteLogTableLimitExceeded(92_682))
    ));
}

#[test]
fn work_exhaustion_invalid_moduli_and_trivial_rings_are_explicit() {
    assert_eq!(
        discrete_log(0, 0, 0, None),
        Err(NTheoryError::InvalidModulus)
    );
    for (a, b) in [(0, 0), (1, 1), (u64::MAX, u64::MAX)] {
        assert_eq!(discrete_log(1, a, b, None).unwrap(), 0);
        assert_eq!(discrete_log(1, a, b, Some(1)).unwrap(), 0);
    }
    for work_limit in [0, 1, 10, 50] {
        assert_eq!(
            discrete_log_with_work_limit(1019, 3, 2, Some(1018), work_limit),
            Err(NTheoryError::ModularGroupWorkLimitExceeded)
        );
    }
    assert_eq!(
        discrete_log(1019, 3, 2, Some(1018)).unwrap(),
        brute_log(1019, 3, 2).unwrap()
    );
}
