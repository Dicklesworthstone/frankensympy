//! Exact hand-derived GCD/LCM fixtures and adversaries for complete principal
//! intersections. A product identity without greatestness must never suffice.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldMultivariateGcdCertificate as Certificate,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    VerifiedPrimeFieldMultivariateGcd as Checked,
    checked_prime_field_ideal_intersection as intersection,
    checked_prime_field_ideal_saturation as saturation,
    checked_prime_field_multivariate_gcd as checked, verify_prime_field_multivariate_gcd as verify,
};
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn p(generators: &[Symbol], prime: u64, entries: &[(&[u32], u64)]) -> Polynomial {
    Polynomial::new(
        generators,
        prime,
        &entries
            .iter()
            .map(|(powers, c)| (powers.to_vec(), *c % prime))
            .collect(),
    )
    .unwrap()
}

fn scalar(generators: &[Symbol], prime: u64, value: u64) -> Polynomial {
    p(generators, prime, &[(&vec![0; generators.len()], value)])
}

fn solve(left: &Polynomial, right: &Polynomial) -> Checked {
    checked(left, right, &mut Unbounded, &mut Unbounded).unwrap()
}

// Independent tiny fixture construction, used only to express known products
// of pairwise distinct linear factors. The expected GCD exponents are supplied
// separately; no production arithmetic or GCD is used to derive them.
fn product(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let mut terms = Terms::new();
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            let powers: Vec<_> = a.iter().zip(b).map(|(a, b)| a + b).collect();
            *terms.entry(powers).or_default() += ca * cb;
        }
    }
    for value in terms.values_mut() {
        *value %= left.prime();
    }
    Polynomial::new(left.generators(), left.prime(), &terms).unwrap()
}

fn rejected(result: &Checked, certificate: &Certificate) {
    assert!(verify(result.left(), result.right(), certificate, &mut Unbounded).is_err());
}

#[test]
fn repeated_nonlinear_common_factors_retain_all_multiplicities() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let h = p(&generators, prime, &[(&[1, 0], 1), (&[0, 1], 1)]);
        let a = p(&generators, prime, &[(&[1, 0], 1), (&[0, 0], 1)]);
        let b = p(&generators, prime, &[(&[0, 1], 1), (&[0, 0], 1)]);
        let h2 = product(&h, &h);
        let left = product(&h2, &a);
        let right = product(&h, &b);
        let result = solve(&left, &right);
        assert_eq!(result.gcd(), &h);
        assert_eq!(result.lcm(), &product(&product(&h2, &a), &b));
        assert_eq!(result.left(), &left);
        assert_eq!(result.right(), &right);
        let replay = verify(&left, &right, result.certificate(), &mut Unbounded).unwrap();
        assert_eq!(replay.gcd(), &h);
        let (gcd, lcm) = replay.into_parts();
        assert_eq!((&gcd, &lcm), (result.gcd(), result.lcm()));
        let reversed = solve(&right, &left);
        assert_eq!(reversed.gcd(), result.gcd());
        assert_eq!(reversed.lcm(), result.lcm());
    }
}

#[test]
fn characteristic_changes_greatestness_without_adding_field_equations() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 7, 65521] {
        let left = p(&generators, prime, &[(&[2, 0], 1), (&[0, 2], 1)]);
        let right = p(&generators, prime, &[(&[1, 0], 1), (&[0, 1], 1)]);
        let result = solve(&left, &right);
        if prime == 2 {
            assert_eq!(result.gcd(), &right);
            assert_eq!(result.lcm(), &left);
        } else {
            assert_eq!(result.gcd(), &scalar(&generators, prime, 1));
            assert_eq!(result.lcm(), &product(&left, &right));
        }
    }
    // x^p-x vanishes at field-valued points but remains a nonzero polynomial.
    let prime = 3;
    let left = p(&generators, prime, &[(&[3, 0], 1), (&[1, 0], 2)]);
    let right = p(&generators, prime, &[(&[0, 1], 1)]);
    let result = solve(&left, &right);
    assert_eq!(result.gcd(), &scalar(&generators, prime, 1));
    assert_eq!(result.lcm(), &product(&left, &right));
}

#[test]
fn independent_unit_scalings_normalize_both_results() {
    let generators = ring(&["x", "y"]);
    let prime = 7;
    let h = p(&generators, prime, &[(&[1, 1], 1), (&[0, 0], 3)]);
    let a = p(&generators, prime, &[(&[1, 0], 1)]);
    let b = p(&generators, prime, &[(&[0, 1], 1)]);
    let expected_lcm = product(&product(&h, &a), &b);
    for u in [1, 2, 6] {
        for v in [1, 3, 5] {
            let left = product(&product(&h, &a), &scalar(&generators, prime, u));
            let right = product(&product(&h, &b), &scalar(&generators, prime, v));
            let result = solve(&left, &right);
            assert_eq!(result.gcd(), &h);
            assert_eq!(result.lcm(), &expected_lcm);
        }
    }
}

#[test]
fn identical_polynomials_keep_the_complete_nonlinear_factor_and_its_power() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        // x*y+1 is irreducible: as a polynomial in x its coprime
        // coefficients y,1 leave no nonunit content or possible splitting.
        let h = p(&generators, prime, &[(&[1, 1], 1), (&[0, 0], 1)]);
        let expected = product(&h, &h);
        let source = product(&expected, &scalar(&generators, prime, prime - 1));
        let result = solve(&source, &source);
        assert_eq!(result.gcd(), &expected);
        assert_eq!(result.lcm(), &expected);
        assert!(result.certificate().intersection.is_some());
    }
}

#[test]
fn monomial_corpus_uses_componentwise_minimum_and_maximum() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        for a in [[0, 0], [1, 0], [0, 2], [2, 1], [3, 2]] {
            for b in [[0, 0], [0, 1], [2, 0], [1, 3]] {
                let left = p(&generators, prime, &[(&a, prime - 1)]);
                let right = p(&generators, prime, &[(&b, 1)]);
                let result = solve(&left, &right);
                let minimum = [a[0].min(b[0]), a[1].min(b[1])];
                let maximum = [a[0].max(b[0]), a[1].max(b[1])];
                assert_eq!(result.gcd(), &p(&generators, prime, &[(&minimum, 1)]));
                assert_eq!(result.lcm(), &p(&generators, prime, &[(&maximum, 1)]));
            }
        }
    }
}

#[test]
fn zero_and_scalars_preserve_empty_and_full_declared_rings() {
    let full: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    for generators in [vec![], ring(&["x", "y"]), full] {
        for prime in [2, 5] {
            let zero = scalar(&generators, prime, 0);
            let unit = scalar(&generators, prime, prime - 1);
            let monic = if generators.is_empty() {
                scalar(&generators, prime, 1)
            } else {
                let mut powers = vec![0; generators.len()];
                powers[generators.len() - 1] = 2;
                p(
                    &generators,
                    prime,
                    &[(&powers, 1), (&vec![0; generators.len()], 1)],
                )
            };
            let scaled = product(&monic, &unit);
            let both = solve(&zero, &zero);
            assert_eq!(both.gcd(), &zero);
            assert_eq!(both.lcm(), &zero);
            for (left, right) in [(&zero, &scaled), (&scaled, &zero)] {
                let result = solve(left, right);
                assert_eq!(result.gcd(), &monic);
                assert_eq!(result.lcm(), &zero);
                assert!(result.certificate().intersection.is_none());
            }
            for (left, right) in [(&unit, &scaled), (&scaled, &unit)] {
                let result = solve(left, right);
                assert_eq!(result.gcd(), &scalar(&generators, prime, 1));
                assert_eq!(result.lcm(), &monic);
                assert!(result.certificate().intersection.is_none());
            }
        }
    }
}

#[test]
fn common_divisor_and_product_identity_do_not_establish_greatestness() {
    let generators = ring(&["x", "y"]);
    let left = p(&generators, 5, &[(&[2, 1], 1)]);
    let right = p(&generators, 5, &[(&[1, 2], 1)]);
    let result = solve(&left, &right);
    let mut candidate = result.certificate().clone();
    candidate.gcd = scalar(&generators, 5, 1);
    candidate.lcm = product(&left, &right);
    rejected(&result, &candidate); // The product still reconstructs perfectly.
    candidate.intersection = None;
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.gcd = scalar(&generators, 5, 0);
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.gcd = product(result.gcd(), &scalar(&generators, 5, 2));
    candidate.lcm = product(result.lcm(), &scalar(&generators, 5, 3));
    rejected(&result, &candidate); // Product preserved, both normalizations wrong.
}

#[test]
fn whole_original_inputs_and_operation_are_bound_to_intersection_evidence() {
    let generators = ring(&["x", "y"]);
    let left = p(&generators, 5, &[(&[2, 1], 1)]);
    let right = p(&generators, 5, &[(&[1, 2], 1)]);
    let result = solve(&left, &right);
    let unrelated = [
        intersection(
            &generators,
            5,
            std::slice::from_ref(&right),
            std::slice::from_ref(&left),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        intersection(
            &generators,
            5,
            &[product(&left, &scalar(&generators, 5, 2))],
            std::slice::from_ref(&right),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        intersection(
            &generators,
            5,
            &[left.clone(), left.clone()],
            std::slice::from_ref(&right),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        saturation(
            &generators,
            5,
            std::slice::from_ref(&left),
            &right,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    ];
    for receipt in unrelated {
        let mut candidate = result.certificate().clone();
        candidate.intersection = Some(Arc::new(receipt));
        rejected(&result, &candidate);
    }
}

#[test]
fn zero_and_scalar_mutants_cannot_erase_nonzero_inputs() {
    let generators = ring(&["x", "y"]);
    let zero = scalar(&generators, 5, 0);
    let one = scalar(&generators, 5, 1);
    let f = p(&generators, 5, &[(&[1, 0], 2), (&[0, 1], 4)]);
    let result = solve(&zero, &f);
    let mut candidate = result.certificate().clone();
    candidate.gcd = f.clone();
    rejected(&result, &candidate);
    candidate.gcd = one.clone();
    rejected(&result, &candidate);
    candidate = result.certificate().clone();
    candidate.lcm = one.clone();
    rejected(&result, &candidate);
    let result = solve(&one, &f);
    let mut candidate = result.certificate().clone();
    candidate.gcd = zero.clone();
    rejected(&result, &candidate);
    candidate = result.certificate().clone();
    candidate.lcm = f;
    rejected(&result, &candidate);
    let result = solve(&zero, &zero);
    let mut candidate = result.certificate().clone();
    candidate.gcd = one;
    rejected(&result, &candidate);
}

#[test]
fn typed_generator_order_and_characteristic_are_not_interchangeable() {
    let generators = ring(&["x", "y"]);
    let left = p(&generators, 5, &[(&[1, 1], 1)]);
    let right = p(&generators, 5, &[(&[1, 0], 1)]);
    let result = solve(&left, &right);
    let mut dummy = generators.clone();
    dummy[0] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([101; 32]));
    for altered in [
        p(&generators, 7, &[(&[1, 0], 1)]),
        p(
            &[generators[1].clone(), generators[0].clone()],
            5,
            &[(&[1, 0], 1)],
        ),
        p(&dummy, 5, &[(&[1, 0], 1)]),
    ] {
        assert!(checked(&left, &altered, &mut Unbounded, &mut Unbounded).is_err());
        assert!(verify(&left, &altered, result.certificate(), &mut Unbounded).is_err());
        let mut candidate = result.certificate().clone();
        candidate.gcd = altered;
        rejected(&result, &candidate);
    }
}

#[test]
fn auxiliary_variable_and_augmented_degree_bounds_are_explicit() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let a = p(&generators, 5, &[(&[1, 0, 0, 0, 0, 0, 0, 0], 1)]);
    let b = p(&generators, 5, &[(&[0, 0, 0, 0, 0, 0, 0, 1], 1)]);
    let error = checked(&a, &b, &mut Unbounded, &mut Unbounded).unwrap_err();
    assert!(error.to_string().contains("auxiliary"));
    let generators = &generators[..7];
    let a = p(generators, 5, &[(&[1, 0, 0, 0, 0, 0, 0], 1)]);
    let b = p(generators, 5, &[(&[0, 0, 0, 0, 0, 0, 1], 1)]);
    let result = solve(&a, &b);
    assert_eq!(result.gcd(), &scalar(generators, 5, 1));
    assert_eq!(result.lcm(), &product(&a, &b));
    let generators = ring(&["x", "y"]);
    let a = p(&generators, 5, &[(&[64, 0], 1)]);
    let b = p(&generators, 5, &[(&[0, 1], 1)]);
    assert!(checked(&a, &b, &mut Unbounded, &mut Unbounded).is_err());
}

#[derive(Default)]
struct Meter {
    steps: u64,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.steps += amount;
        }
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn independent_phase_budgets_and_cancellation_prevent_publication() {
    let generators = ring(&["x", "y"]);
    let left = p(&generators, 5, &[(&[2, 1], 1)]);
    let right = p(&generators, 5, &[(&[1, 2], 1)]);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    let accepted = checked(&left, &right, &mut generation, &mut verification).unwrap();
    assert!(generation.steps > 0 && verification.steps > 0);
    for phase in [false, true] {
        let count = if phase {
            verification.checkpoints
        } else {
            generation.checkpoints
        };
        for cancel_at in [1, count / 2, count] {
            let mut cancelled = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let error = if phase {
                checked(&left, &right, &mut Unbounded, &mut cancelled)
            } else {
                checked(&left, &right, &mut cancelled, &mut Unbounded)
            }
            .unwrap_err();
            assert!(error.to_string().to_lowercase().contains("cancel"));
            assert_eq!(cancelled.checkpoints, cancel_at);
        }
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(checked(&left, &right, &mut empty, &mut Unbounded).is_err());
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(checked(&left, &right, &mut Unbounded, &mut empty).is_err());
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(verify(&left, &right, accepted.certificate(), &mut empty).is_err());
    // A second request cannot reset the caller's consumed ledger.
    let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = generation.steps;
    let mut budget = Budget::new(limits);
    checked(&left, &right, &mut budget, &mut Unbounded).unwrap();
    assert!(checked(&left, &right, &mut budget, &mut Unbounded).is_err());
}
