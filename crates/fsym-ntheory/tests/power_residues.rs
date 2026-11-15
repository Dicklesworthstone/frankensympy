use fsym_ntheory::{NTheoryError, is_nth_power_residue, is_nth_power_residue_with_work_limit};

fn power(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1 % modulus;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = (u128::from(result) * u128::from(base) % u128::from(modulus)) as u64;
        }
        base = (u128::from(base) * u128::from(base) % u128::from(modulus)) as u64;
        exponent >>= 1;
    }
    result
}

#[test]
fn every_small_ring_and_power_matches_complete_residue_enumeration() {
    for modulus in 1..=64 {
        for degree in 0..=10 {
            let mut residues = vec![false; modulus as usize];
            for x in 0..modulus {
                residues[power(x, degree, modulus) as usize] = true;
            }
            for a in 0..modulus {
                assert_eq!(
                    is_nth_power_residue(a, degree, modulus).unwrap(),
                    residues[a as usize],
                    "a={a}, degree={degree}, modulus={modulus}"
                );
                assert_eq!(
                    is_nth_power_residue(a + 11 * modulus, degree, modulus).unwrap(),
                    residues[a as usize]
                );
            }
        }
    }
}

#[test]
fn valuation_constraints_are_required_before_testing_the_unit_part() {
    let modulus = 3u64.pow(40);
    assert!(is_nth_power_residue(3u64.pow(5), 5, modulus).unwrap());
    assert!(!is_nth_power_residue(3u64.pow(6), 5, modulus).unwrap());
    assert!(is_nth_power_residue(8, 3, modulus).unwrap());
    assert!(!is_nth_power_residue(2, 3, modulus).unwrap());
    let mixed = 3u64.pow(12) * 5u64.pow(8);
    assert!(is_nth_power_residue(power(45, 6, mixed), 6, mixed).unwrap());
    assert!(!is_nth_power_residue(3u64.pow(5) * 5u64.pow(6), 6, mixed).unwrap());
    assert!(is_nth_power_residue(64, 3, 1u64 << 63).unwrap());
    assert!(!is_nth_power_residue(32, 3, 1u64 << 63).unwrap());
}

#[test]
fn two_adic_units_use_the_full_group_structure_at_the_word_boundary() {
    let modulus = 1u64 << 63;
    assert!(is_nth_power_residue(17, 4, modulus).unwrap());
    assert!(!is_nth_power_residue(9, 4, modulus).unwrap());
    assert!(is_nth_power_residue(33, 8, modulus).unwrap());
    assert!(!is_nth_power_residue(17, 8, modulus).unwrap());
    assert!(is_nth_power_residue(1, 1u64 << 63, modulus).unwrap());
    assert!(!is_nth_power_residue(1 + (1u64 << 62), 1u64 << 63, modulus).unwrap());
    assert!(is_nth_power_residue(u64::MAX, u64::MAX, modulus).unwrap());
    assert!(is_nth_power_residue(3, 1u64 << 63, 2).unwrap());
    assert!(!is_nth_power_residue(3, 1u64 << 63, 4).unwrap());
}

#[test]
fn huge_prime_powers_and_composites_do_not_scan_residue_intervals() {
    let modulus = 7u64.pow(22);
    assert!(is_nth_power_residue(8, 3, modulus).unwrap());
    assert!(!is_nth_power_residue(3, 3, modulus).unwrap());
    assert!(is_nth_power_residue(3, 5, modulus).unwrap());
    assert!(is_nth_power_residue(7u64.pow(3), 3, modulus).unwrap());
    assert!(!is_nth_power_residue(7u64.pow(4), 3, modulus).unwrap());
    let prime = 18_446_744_073_709_551_557;
    assert!(is_nth_power_residue(1, prime - 1, prime).unwrap());
    assert!(!is_nth_power_residue(2, prime - 1, prime).unwrap());
    for modulus in [u64::MAX, 1_000_000_007u64 * 1_000_000_009] {
        assert!(is_nth_power_residue(power(123_456_789, 5, modulus), 5, modulus).unwrap());
    }
}

#[test]
fn zero_degree_zero_residue_and_trivial_ring_conventions_are_explicit() {
    assert!(is_nth_power_residue(0, 0, 1).unwrap());
    assert!(is_nth_power_residue(1, 0, 9).unwrap());
    assert!(!is_nth_power_residue(0, 0, 9).unwrap());
    assert!(!is_nth_power_residue(2, 0, 9).unwrap());
    for modulus in [1, 2, 9, 1u64 << 63, u64::MAX] {
        assert!(is_nth_power_residue(0, 1, modulus).unwrap());
        assert!(is_nth_power_residue(0, u64::MAX, modulus).unwrap());
        assert!(is_nth_power_residue(u64::MAX, 1, modulus).unwrap());
    }
}

#[test]
fn exhausted_work_cannot_become_a_false_residue_decision() {
    assert_eq!(
        is_nth_power_residue(1, 3, 0),
        Err(NTheoryError::InvalidModulus)
    );
    for work_limit in 0..=3 {
        assert_eq!(
            is_nth_power_residue_with_work_limit(2, 3, 3u64.pow(40), work_limit),
            Err(NTheoryError::ModularGroupWorkLimitExceeded)
        );
    }
    assert!(!is_nth_power_residue(2, 3, 3u64.pow(40)).unwrap());
}
