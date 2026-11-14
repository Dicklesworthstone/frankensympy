//! Hand-derived geometric ideals, independent coefficient fixtures and corrupt
//! composition witnesses. Field-point sampling is deliberately insufficient.

use std::collections::BTreeMap;
use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    PrimeFieldPolynomial, PrimeFieldZeroDimensionalRadicalCertificate as Certificate, TermOrder,
    VerifiedPrimeFieldZeroDimensionalRadical as Checked, checked_prime_field_factorization,
    checked_prime_field_groebner_basis, checked_prime_field_zero_dimensional_radical as checked,
    verify_prime_field_zero_dimensional_radical as verify,
};

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

fn solve(generators: &[Symbol], prime: u64, inputs: &[Polynomial]) -> Checked {
    checked(generators, prime, inputs, &mut Unbounded, &mut Unbounded).unwrap()
}

// Tiny independent integer convolution defines fixtures; it never invokes the
// production arithmetic or radical algorithm to derive expected coefficients.
fn product(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let mut values = BTreeMap::<Vec<u32>, u64>::new();
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            let powers = a.iter().zip(b).map(|(a, b)| a + b).collect();
            *values.entry(powers).or_default() += ca * cb;
        }
    }
    for value in values.values_mut() {
        *value %= left.prime();
    }
    Polynomial::new(left.generators(), left.prime(), &values).unwrap()
}

fn power(polynomial: &Polynomial, count: usize) -> Polynomial {
    (0..count).fold(
        scalar(polynomial.generators(), polynomial.prime(), 1),
        |value, _| product(&value, polynomial),
    )
}

fn rejected(result: &Checked, certificate: &Certificate) {
    assert!(
        verify(
            result.generators(),
            result.prime(),
            result.inputs(),
            certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn nilpotent_thickened_origin_retains_every_coordinate() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let inputs = [
            p(&generators, prime, &[(&[3, 0], 1)]),
            p(&generators, prime, &[(&[0, 4], 1)]),
        ];
        let result = solve(&generators, prime, &inputs);
        assert_eq!(
            result.basis().basis(),
            &[
                p(&generators, prime, &[(&[1, 0], 1)]),
                p(&generators, prime, &[(&[0, 1], 1)])
            ]
        );
        assert_eq!(result.inputs(), &inputs);
        assert!(result.changed_ideal());
        let replay = verify(
            &generators,
            prime,
            &inputs,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(replay.basis(), result.basis());
        assert_eq!(replay.generators(), generators);
        assert_eq!(replay.prime(), prime);
    }
}

#[test]
fn characteristic_two_nested_frobenius_powers_are_removed() {
    let generators = ring(&["x", "y"]);
    let x = p(&generators, 2, &[(&[1, 0], 1), (&[0, 0], 1)]);
    let y = p(&generators, 2, &[(&[0, 2], 1), (&[0, 1], 1), (&[0, 0], 1)]);
    // Both source derivatives vanish; y's two geometric roots are outside F_2.
    let result = solve(&generators, 2, &[power(&x, 8), power(&y, 4)]);
    assert_eq!(result.basis().basis(), &[x, y]);
    assert_eq!(result.certificate().coordinates[0].nilpotence_exponent, 8);
    assert_eq!(result.certificate().coordinates[1].nilpotence_exponent, 4);
}

#[test]
fn inseparable_characteristic_three_mixed_multiplicities_are_complete() {
    let generators = ring(&["t"]);
    let t = p(&generators, 3, &[(&[1], 1)]);
    let shifted = p(&generators, 3, &[(&[1], 1), (&[0], 1)]);
    let source = product(&power(&t, 3), &power(&shifted, 6));
    let result = solve(&generators, 3, &[source]);
    assert_eq!(result.basis().basis(), &[product(&t, &shifted)]);
    assert_eq!(result.certificate().coordinates[0].nilpotence_exponent, 6);
}

#[test]
fn nonlinear_coupling_preserves_the_complete_geometric_locus() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 7] {
        let diagonal = p(&generators, prime, &[(&[1, 0], 1), (&[0, 1], prime - 1)]);
        let quadratic = p(&generators, prime, &[(&[0, 2], 1), (&[0, 1], prime - 1)]);
        let result = solve(
            &generators,
            prime,
            &[power(&diagonal, 2), power(&quadratic, 3)],
        );
        assert_eq!(result.basis().basis(), &[diagonal, quadratic]);
    }
}

#[test]
fn reduced_extension_field_points_do_not_become_an_empty_ideal() {
    let generators = ring(&["x"]);
    let irreducible = p(&generators, 2, &[(&[2], 1), (&[1], 1), (&[0], 1)]);
    let result = solve(&generators, 2, std::slice::from_ref(&irreducible));
    assert_eq!(result.basis().basis(), &[irreducible]);
    assert!(!result.changed_ideal());
    // x^2+x+1 has no F_2 point. Adding x^2-x would incorrectly give <1>.
    assert_ne!(result.basis().basis(), &[scalar(&generators, 2, 1)]);
}

#[test]
fn square_free_cartesian_grid_is_idempotent() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        p(&generators, 5, &[(&[3, 0], 1), (&[1, 0], 4)]),
        p(&generators, 5, &[(&[0, 2], 1), (&[0, 0], 3)]),
    ];
    let result = solve(&generators, 5, &inputs);
    assert_eq!(result.basis().basis(), &inputs);
    assert!(!result.changed_ideal());
    let repeated = solve(&generators, 5, result.basis().basis());
    assert_eq!(repeated.basis().basis(), result.basis().basis());
    assert!(!repeated.changed_ideal());
}

#[test]
fn unit_and_zero_variable_field_cases_have_no_coordinate_obligations() {
    for generators in [vec![], ring(&["x", "y"])] {
        let zero = scalar(&generators, 5, 0);
        let unit = scalar(&generators, 5, 3);
        let result = solve(&generators, 5, &[zero.clone(), unit, zero]);
        assert_eq!(result.basis().basis(), &[scalar(&generators, 5, 1)]);
        assert!(result.certificate().coordinates.is_empty());
        assert!(!result.changed_ideal());
    }
    let constant_field = solve(&[], 5, &[]);
    assert!(constant_field.basis().basis().is_empty());
    assert!(!constant_field.changed_ideal());
}

#[test]
fn positive_dimensional_and_zero_ideals_refuse_instead_of_claiming_completion() {
    let generators = ring(&["x", "y"]);
    for inputs in [
        vec![],
        vec![p(&generators, 5, &[(&[2, 0], 1)])],
        vec![p(&generators, 5, &[(&[1, 1], 1)])],
    ] {
        let error = checked(&generators, 5, &inputs, &mut Unbounded, &mut Unbounded).unwrap_err();
        assert!(error.to_string().contains("zero-dimensional"));
    }
}

#[test]
fn omitted_coordinates_and_forged_nilpotence_cannot_pass() {
    let generators = ring(&["x", "y"]);
    let result = solve(
        &generators,
        3,
        &[
            p(&generators, 3, &[(&[3, 0], 1)]),
            p(&generators, 3, &[(&[0, 3], 1)]),
        ],
    );
    let mut candidate = result.certificate().clone();
    candidate.coordinates.pop();
    rejected(&result, &candidate);
    for exponent in [0, 1, 2, 65, u32::MAX] {
        let mut candidate = result.certificate().clone();
        candidate.coordinates[0].nilpotence_exponent = exponent;
        rejected(&result, &candidate);
    }
    let mut candidate = result.certificate().clone();
    candidate.coordinates[0].nilpotent_quotients.clear();
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.coordinates[0].nilpotent_quotients[0] = Terms::new();
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.coordinates.swap(0, 1);
    rejected(&result, &candidate);
}

#[test]
fn square_free_proof_and_full_ideal_binding_reject_coordinated_forgery() {
    let generators = ring(&["x"]);
    let square = p(&generators, 5, &[(&[2], 1)]);
    let result = solve(&generators, 5, std::slice::from_ref(&square));
    let mut candidate = result.certificate().clone();
    // A power-membership identity alone accepts x^2, but it is not square-free.
    candidate.coordinates[0].annihilator = square.clone();
    candidate.coordinates[0].nilpotence_exponent = 1;
    candidate.coordinates[0].nilpotent_quotients = vec![[(vec![0], 1)].into_iter().collect()];
    candidate.radical_basis = Arc::new(
        checked_prime_field_groebner_basis(
            &generators,
            5,
            &[square.clone(), square],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    // Augmenting by 1 makes every point disappear, even though <1> is radical.
    let one = scalar(&generators, 5, 1);
    candidate.radical_basis = Arc::new(
        checked_prime_field_groebner_basis(
            &generators,
            5,
            &[one],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &candidate);
}

#[test]
fn omitted_geometric_root_fails_the_original_ideal_membership_identity() {
    let generators = ring(&["x"]);
    let t = p(&generators, 5, &[(&[1], 1)]);
    let shifted = p(&generators, 5, &[(&[1], 1), (&[0], 4)]);
    let source = power(&product(&t, &shifted), 2);
    let result = solve(&generators, 5, std::slice::from_ref(&source));
    let mut candidate = result.certificate().clone();
    candidate.coordinates[0].annihilator = t.clone();
    candidate.coordinates[0].square_free_factorization = Arc::new(
        checked_prime_field_factorization(
            &PrimeFieldPolynomial::new(generators[0].clone(), 5, vec![0, 1]).unwrap(),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    candidate.radical_basis = Arc::new(
        checked_prime_field_groebner_basis(
            &generators,
            5,
            &[source, t],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &candidate);
}

#[test]
fn original_multiplicities_zero_duplicates_and_ring_identity_are_bound() {
    let generators = ring(&["x", "y"]);
    let x = p(&generators, 5, &[(&[2, 0], 1)]);
    let y = p(&generators, 5, &[(&[0, 2], 1)]);
    let result = solve(
        &generators,
        5,
        &[x.clone(), scalar(&generators, 5, 0), x.clone(), y.clone()],
    );
    assert!(
        verify(
            &generators,
            5,
            &[x.clone(), y.clone()],
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify(
            &generators,
            5,
            &[y.clone(), x.clone()],
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let mut other = generators.clone();
    other[0] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([51; 32]));
    for (ring, prime) in [
        (other, 5),
        (vec![generators[1].clone(), generators[0].clone()], 5),
        (generators.clone(), 7),
    ] {
        assert!(
            verify(
                &ring,
                prime,
                result.inputs(),
                result.certificate(),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked(
                &ring,
                prime,
                result.inputs(),
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    // A unit shortcut may not hide a foreign typed source or field.
    let unit = scalar(&generators, 5, 1);
    let foreign = p(&generators, 7, &[(&[1, 0], 1)]);
    assert!(
        checked(
            &generators,
            5,
            &[unit, foreign],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn foreign_accepted_factor_evidence_and_augmented_source_are_rejected() {
    let generators = ring(&["x"]);
    let source = p(&generators, 3, &[(&[2], 1)]);
    let result = solve(&generators, 3, &[source]);
    for (symbol, prime, coefficients) in [
        (Symbol::new("y"), 3, vec![0, 1]),
        (generators[0].clone(), 5, vec![0, 1]),
        (generators[0].clone(), 3, vec![1, 1]),
    ] {
        let mut candidate = result.certificate().clone();
        candidate.coordinates[0].square_free_factorization = Arc::new(
            checked_prime_field_factorization(
                &PrimeFieldPolynomial::new(symbol, prime, coefficients).unwrap(),
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap(),
        );
        rejected(&result, &candidate);
    }
    let mut candidate = result.certificate().clone();
    candidate.source_basis = Arc::clone(&candidate.radical_basis);
    rejected(&result, &candidate);
}

#[test]
fn eight_coordinates_and_degree_sixty_four_keep_explicit_bounds() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("v{i}"))).collect();
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for coordinate in 0..generators.len() {
        let mut powers = vec![0; generators.len()];
        powers[coordinate] = 2;
        inputs.push(p(&generators, 2, &[(&powers, 1)]));
        powers[coordinate] = 1;
        expected.push(p(&generators, 2, &[(&powers, 1)]));
    }
    assert_eq!(solve(&generators, 2, &inputs).basis().basis(), expected);
    let generators = ring(&["x"]);
    let source = p(&generators, 2, &[(&[64], 1)]);
    assert_eq!(
        solve(&generators, 2, &[source]).basis().basis(),
        &[p(&generators, 2, &[(&[1], 1)])]
    );
    let many = vec![scalar(&generators, 2, 1); 33];
    assert!(checked(&generators, 2, &many, &mut Unbounded, &mut Unbounded).is_err());
    assert!(checked(&[], 4, &[], &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn powered_coordinate_envelope_refuses_without_publishing_a_partial_radical() {
    let generators = ring(&["x"]);
    // x^63*(x+1) has radical x*(x+1), but the chosen complete membership
    // certificate would require degree126, outside this bounded API.
    let source = p(&generators, 2, &[(&[64], 1), (&[63], 1)]);
    let error = checked(&generators, 2, &[source], &mut Unbounded, &mut Unbounded).unwrap_err();
    assert!(error.to_string().contains("degree64"));
}

#[derive(Default)]
struct Meter {
    amounts: [u64; fsym_budget::DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.amounts[dimension.index()] += amount;
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
fn exact_and_one_short_budgets_cover_both_phases_and_all_charged_dimensions() {
    let generators = ring(&["x"]);
    let source = p(&generators, 2, &[(&[2], 1)]);
    let inputs = [source];
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    checked(&generators, 2, &inputs, &mut generation, &mut verification).unwrap();
    for (is_verifier, measured) in [(false, generation), (true, verification)] {
        for dimension in Dimension::ALL {
            let amount = measured.amounts[dimension.index()];
            if amount == 0 {
                continue;
            }
            for (limit, succeeds) in [(amount, true), (amount - 1, false)] {
                let mut limits = BudgetLimits::uniform(u64::MAX, 0);
                limits.dimensions[dimension.index()] = limit;
                let mut bounded = Budget::new(limits);
                let actual = if is_verifier {
                    checked(&generators, 2, &inputs, &mut Unbounded, &mut bounded)
                } else {
                    checked(&generators, 2, &inputs, &mut bounded, &mut Unbounded)
                };
                assert_eq!(
                    actual.is_ok(),
                    succeeds,
                    "phase={is_verifier} dimension={dimension:?} limit={limit}"
                );
            }
        }
    }
}

#[test]
fn cancellation_at_every_checkpoint_includes_acceptance_and_output_copies() {
    let generators = ring(&["x"]);
    let inputs = [p(&generators, 2, &[(&[2], 1)])];
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    checked(&generators, 2, &inputs, &mut generation, &mut verification).unwrap();
    for (is_verifier, count) in [
        (false, generation.checkpoints),
        (true, verification.checkpoints),
    ] {
        for cancel_at in 1..=count {
            let mut cancelled = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let actual = if is_verifier {
                checked(&generators, 2, &inputs, &mut Unbounded, &mut cancelled)
            } else {
                checked(&generators, 2, &inputs, &mut cancelled, &mut Unbounded)
            };
            assert!(
                actual.is_err(),
                "phase={is_verifier} checkpoint={cancel_at}"
            );
        }
    }
}
