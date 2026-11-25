//! Independent finite enumeration, cardinality forgeries and boundary checks.

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_ntheory::{
    MAX_MODULAR_ROOTS, NTheoryError, factorint, metered_modular_nth_roots,
    modular_nth_root, modular_nth_roots, verify_modular_nth_roots,
    verify_modular_nth_roots_with_meter,
};
use std::collections::BTreeMap;

fn brute_roots(a: u64, degree: u64, modulus: u64) -> Vec<u64> {
    (0..modulus)
        .filter(|&x| {
            let mut power = 1u128;
            for _ in 0..degree {
                power = power * u128::from(x) % u128::from(modulus);
            }
            power % u128::from(modulus) == u128::from(a % modulus)
        })
        .collect()
}

#[test]
fn all_small_residues_and_degrees_match_complete_finite_enumeration() {
    for modulus in 1..=80 {
        for degree in 1..=9 {
            for a in 0..modulus {
                let expected = brute_roots(a, degree, modulus);
                let actual = modular_nth_roots(a, degree, modulus).unwrap();
                assert_eq!(actual, expected, "a={a}, degree={degree}, modulus={modulus}");
                assert_eq!(
                    modular_nth_root(a, degree, modulus).unwrap(),
                    expected.first().copied(),
                    "least root: a={a}, degree={degree}, modulus={modulus}",
                );
            }
        }
    }
}

#[test]
fn prime_power_valuation_and_composite_crt_multiplicities_are_complete() {
    for (a, degree, modulus, expected) in [
        (11, 4, 19, vec![8, 11]),
        (68, 3, 109, vec![23, 32, 54]),
        (8, 3, 27, vec![2, 11, 20]),
        (8, 3, 125, vec![2]),
        (1, 6, 64, vec![1, 31, 33, 63]),
        (8, 3, 72, vec![2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68]),
        (9, 3, 27, vec![]),
        (2, 4, 19, vec![]),
    ] {
        assert_eq!(modular_nth_roots(a, degree, modulus).unwrap(), expected);
        assert_eq!(
            modular_nth_roots(a + 11 * modulus, degree, modulus).unwrap(),
            expected,
        );
        verify_modular_nth_roots(a, degree, modulus, &factorint(modulus).unwrap(), &expected)
            .unwrap();
    }
}

#[test]
fn odd_degrees_and_two_adic_bit_logs_work_at_the_u64_boundary() {
    let modulus = 1u64 << 63;
    assert_eq!(modular_nth_roots(729, 3, modulus).unwrap(), vec![9]);
    assert_eq!(
        modular_nth_roots(531_441, 6, modulus).unwrap(),
        vec![9, (1u64 << 62) - 9, (1u64 << 62) + 9, modulus - 9],
    );
    assert_eq!(
        modular_nth_roots(modulus - 1, u64::MAX, modulus).unwrap(),
        vec![modulus - 1],
    );
    // The inverse-exponent route needs neither a group generator nor a
    // sqrt(p)-sized discrete-log table for these large prime requests.
    let prime = 18_446_744_073_709_551_557;
    assert_eq!(modular_nth_roots(8, 3, prime).unwrap(), vec![2]);
    assert_eq!(modular_nth_roots(32, 5, prime).unwrap(), vec![2]);
}

#[test]
fn large_square_roots_keep_the_existing_specialized_generator() {
    let prime = 18_446_744_073_709_551_557;
    assert_eq!(modular_nth_roots(4, 2, prime).unwrap(), vec![2, prime - 2]);
    assert_eq!(modular_nth_root(4, 2, prime).unwrap(), Some(2));
}

#[test]
fn exact_cardinality_limit_and_late_nonresidue_have_distinct_outcomes() {
    let roots = modular_nth_roots(0, 3, 1u64 << 24).unwrap();
    assert_eq!(roots.len(), MAX_MODULAR_ROOTS);
    assert_eq!(roots[0], 0);
    assert_eq!(*roots.last().unwrap(), (1u64 << 24) - (1u64 << 8));
    assert_eq!(
        modular_nth_roots(0, 3, 1u64 << 26),
        Err(NTheoryError::ModularNthRootCountExceeded(1u64 << 17)),
    );
    assert_eq!(
        modular_nth_roots(1u64 << 60, 3, 1u64 << 63),
        Err(NTheoryError::ModularNthRootCountExceeded(1u64 << 40)),
    );
    assert_eq!(modular_nth_root(0, 3, 1u64 << 63).unwrap(), Some(0));
    // A huge local zero-root set at two is irrelevant when the later
    // component at seven proves there is no global cubic root.
    assert_eq!(
        modular_nth_roots(2 * (1u64 << 40), 3, 7 * (1u64 << 40)).unwrap(),
        Vec::<u64>::new(),
    );
}

#[test]
fn an_unaffordable_logarithm_is_a_refusal_and_never_an_empty_root_set() {
    // 4294967311 is prime and 8 has a cubic root (2), but the required
    // odd-group table has 65537 entries and exceeds the admitted envelope.
    assert_eq!(
        modular_nth_roots(8, 3, 4_294_967_311),
        Err(NTheoryError::DiscreteLogTableLimitExceeded(65_537)),
    );
    // A unit target of one needs no logarithm table even in the same group.
    let roots = modular_nth_roots(1, 3, 4_294_967_311).unwrap();
    assert_eq!(roots.len(), 3);
    assert_eq!(roots[0], 1);
}

#[test]
fn verifier_rejects_valid_but_incomplete_roots_and_false_nonexistence() {
    let factors = BTreeMap::from([(2, 3), (3, 2)]);
    let roots = [2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68];
    verify_modular_nth_roots(8, 3, 72, &factors, &roots).unwrap();
    for mutant in [
        roots[..11].to_vec(),
        vec![],
        vec![2; 12],
        vec![2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 67],
        vec![2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 140],
        roots.iter().rev().copied().collect(),
    ] {
        assert!(verify_modular_nth_roots(8, 3, 72, &factors, &mutant).is_err());
    }
    assert!(verify_modular_nth_roots(9, 3, 72, &factors, &roots).is_err());
    assert!(verify_modular_nth_roots(8, 4, 72, &factors, &roots).is_err());
    assert!(verify_modular_nth_roots(8, 3, 73, &factors, &roots).is_err());
    for bad in [
        BTreeMap::from([(2, 2), (3, 2)]),
        BTreeMap::from([(8, 1), (9, 1)]),
        BTreeMap::from([(2, u32::MAX)]),
        BTreeMap::new(),
    ] {
        assert!(verify_modular_nth_roots(8, 3, 72, &bad, &roots).is_err());
    }
    verify_modular_nth_roots(9, 3, 27, &BTreeMap::from([(3, 3)]), &[]).unwrap();
}

#[test]
fn malformed_degree_and_modulus_and_the_trivial_ring_are_explicit() {
    for a in [0, 1, u64::MAX] {
        assert_eq!(modular_nth_roots(a, 0, 7), Err(NTheoryError::InvalidRootDegree));
        assert_eq!(modular_nth_root(a, 0, 7), Err(NTheoryError::InvalidRootDegree));
        assert_eq!(modular_nth_roots(a, 3, 0), Err(NTheoryError::InvalidModulus));
        assert_eq!(modular_nth_root(a, 3, 0), Err(NTheoryError::InvalidModulus));
        for degree in [1, 2, 3, u64::MAX] {
            assert_eq!(modular_nth_roots(a, degree, 1).unwrap(), vec![0]);
            assert_eq!(modular_nth_root(a, degree, 1).unwrap(), Some(0));
        }
        verify_modular_nth_roots(a, 3, 1, &BTreeMap::new(), &[0]).unwrap();
    }
    assert!(verify_modular_nth_roots(1, 0, 1, &BTreeMap::new(), &[0]).is_err());
    assert!(verify_modular_nth_roots(1, 3, 0, &BTreeMap::new(), &[]).is_err());
}

#[derive(Default)]
struct Meter {
    checkpoints: u64,
    cancel_at: Option<u64>,
    charged: u64,
    steps_left: Option<u64>,
}

impl BudgetMeter for Meter {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.cancel_at.is_some_and(|limit| self.checkpoints >= limit) {
            return Err(MeterError::Cancelled);
        }
        self.checkpoints += 1;
        Ok(())
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(charges.iter().all(|(_, amount)| *amount > 0));
        let steps = charges
            .iter()
            .filter(|(dimension, _)| *dimension == Dimension::ComputeSteps)
            .map(|(_, amount)| amount)
            .sum::<u64>();
        if let Some(remaining) = self.steps_left {
            if steps > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension: Dimension::ComputeSteps,
                    requested: steps,
                    remaining,
                }));
            }
            self.steps_left = Some(remaining - steps);
        }
        self.charged += steps;
        Ok(())
    }
}

#[test]
fn cancellation_at_entry_generation_verification_and_publication_is_terminal() {
    let mut baseline_generation = Meter::default();
    let mut baseline_verification = Meter::default();
    assert_eq!(
        metered_modular_nth_roots(68, 3, 109, &mut baseline_generation, &mut baseline_verification)
            .unwrap(),
        vec![23, 32, 54],
    );
    for at in [0, 1, 2, 5, 20, baseline_generation.checkpoints - 1] {
        let mut stopped = Meter { cancel_at: Some(at), ..Meter::default() };
        assert_eq!(
            metered_modular_nth_roots(68, 3, 109, &mut stopped, &mut Unbounded),
            Err(NTheoryError::Meter(MeterError::Cancelled)),
            "generation checkpoint {at}",
        );
    }
    for at in [0, 1, 2, 5, 20, baseline_verification.checkpoints - 1] {
        let mut stopped = Meter { cancel_at: Some(at), ..Meter::default() };
        assert_eq!(
            metered_modular_nth_roots(68, 3, 109, &mut Unbounded, &mut stopped),
            Err(NTheoryError::Meter(MeterError::Cancelled)),
            "verification checkpoint {at}",
        );
    }
}

#[test]
fn budgets_are_separate_cumulative_and_checked_before_synchronous_helpers() {
    let mut generator = Meter { steps_left: Some(0), ..Meter::default() };
    let mut verifier = Meter::default();
    assert!(matches!(
        metered_modular_nth_roots(68, 3, 109, &mut generator, &mut verifier),
        Err(NTheoryError::Meter(MeterError::Budget(_))),
    ));
    assert_eq!(generator.charged, 0);
    assert_eq!(verifier.charged, 0);
    let mut generator = Meter::default();
    let mut verifier = Meter { steps_left: Some(0), ..Meter::default() };
    assert!(matches!(
        metered_modular_nth_roots(68, 3, 109, &mut generator, &mut verifier),
        Err(NTheoryError::Meter(MeterError::Budget(_))),
    ));
    assert!(generator.charged > 0);
    assert_eq!(verifier.charged, 0);
    let mut verifier = Meter { cancel_at: Some(0), ..Meter::default() };
    assert_eq!(
        verify_modular_nth_roots_with_meter(
            8, 3, 27, &BTreeMap::from([(3, 3)]), &[2, 11, 20], &mut verifier,
        ),
        Err(NTheoryError::Meter(MeterError::Cancelled)),
    );
}

#[test]
fn empty_results_use_valid_nonzero_charges_with_accounting_meters() {
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    assert!(
        metered_modular_nth_roots(9, 3, 27, &mut generator, &mut verifier)
            .unwrap()
            .is_empty(),
    );
    assert!(generator.charged > 0 && verifier.charged > 0);
}
