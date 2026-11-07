//! Independently derived finite ideal quotients over F_p. Complete expected
//! bases distinguish ideal equality from membership, saturation and agreement
//! on field-valued points. Mutants attack every composition obligation.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldIdealQuotientCertificate as Certificate,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    TermOrder, VerifiedPrimeFieldIdealQuotient as Quotient,
    checked_prime_field_groebner_basis as groebner,
    checked_prime_field_ideal_intersection as intersection,
    checked_prime_field_ideal_quotient as checked,
    checked_prime_field_ideal_saturation as saturation,
    verify_prime_field_ideal_quotient as verify,
};
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn p(generators: &[Symbol], prime: u64, entries: &[(&[u32], u64)]) -> Polynomial {
    let terms: Terms = entries
        .iter()
        .map(|(powers, c)| (powers.to_vec(), *c))
        .collect();
    Polynomial::new(generators, prime, &terms).unwrap()
}

fn monomial(generators: &[Symbol], prime: u64, powers: &[u32]) -> Polynomial {
    p(generators, prime, &[(powers, 1)])
}

fn one(generators: &[Symbol], prime: u64) -> Polynomial {
    monomial(generators, prime, &vec![0; generators.len()])
}

fn quotient(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    divisors: &[Polynomial],
) -> Quotient {
    checked(
        generators,
        prime,
        inputs,
        divisors,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn rejected(result: &Quotient, certificate: &Certificate) {
    assert!(
        verify(
            result.generators(),
            result.prime(),
            result.inputs(),
            result.divisors(),
            certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn finite_quotients_remove_only_the_supplied_multiplicity_in_every_characteristic() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let inputs = [monomial(&generators, prime, &[3, 1])];
        for (divisor, expected) in [
            (
                monomial(&generators, prime, &[1, 0]),
                monomial(&generators, prime, &[2, 1]),
            ),
            (
                monomial(&generators, prime, &[2, 0]),
                monomial(&generators, prime, &[1, 1]),
            ),
            (
                monomial(&generators, prime, &[4, 0]),
                monomial(&generators, prime, &[0, 1]),
            ),
            (
                monomial(&generators, prime, &[1, 1]),
                monomial(&generators, prime, &[2, 0]),
            ),
            (
                monomial(&generators, prime, &[0, 2]),
                monomial(&generators, prime, &[3, 0]),
            ),
            (
                p(&generators, prime, &[(&[1, 0], 1), (&[0, 0], 1)]),
                inputs[0].clone(),
            ),
            (
                p(&generators, prime, &[(&[0, 0], prime - 1)]),
                inputs[0].clone(),
            ),
        ] {
            let divisors = [divisor];
            let result = quotient(&generators, prime, &inputs, &divisors);
            assert_eq!(result.basis(), [expected]);
            assert_eq!(result.prime(), prime);
            assert_eq!(result.generators(), generators);
            assert_eq!(result.inputs(), inputs);
            assert_eq!(result.divisors(), divisors);
            let replay = verify(
                &generators,
                prime,
                &inputs,
                &divisors,
                result.certificate(),
                &mut Unbounded,
            )
            .unwrap();
            assert_eq!(replay.basis(), result.basis());
            assert_eq!(result.clone().into_basis(), result.basis());
        }
        let x = monomial(&generators, prime, &[1, 0]);
        let finite = quotient(&generators, prime, &inputs, std::slice::from_ref(&x));
        let saturated = saturation(
            &generators,
            prime,
            &inputs,
            &x,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(saturated.basis(), [monomial(&generators, prime, &[0, 1])]);
        assert_ne!(finite.basis(), saturated.basis());
    }
}

#[test]
fn characteristic_changes_common_factors_without_replacing_ideals_by_field_points() {
    let generators = ring(&["x"]);
    for prime in [2, 3, 5, 65521] {
        let source = p(&generators, prime, &[(&[2], 1), (&[0], 1)]);
        let divisor = p(&generators, prime, &[(&[1], 1), (&[0], 1)]);
        let result = quotient(
            &generators,
            prime,
            std::slice::from_ref(&source),
            std::slice::from_ref(&divisor),
        );
        let expected = if prime == 2 { divisor } else { source };
        assert_eq!(result.basis(), [expected]);
        // In F_3 there are no roots of x^2+1, yet its ideal and its quotient
        // by the coprime x+1 remain proper. No x^p-x equations may be added.
    }
}

#[test]
fn multigenerator_divisors_require_the_complete_intersection_of_principal_quotients() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 7] {
        let inputs = [
            monomial(&generators, prime, &[2, 0]),
            monomial(&generators, prime, &[0, 2]),
        ];
        let divisors = [
            monomial(&generators, prime, &[1, 0]),
            monomial(&generators, prime, &[0, 1]),
        ];
        let expected = [
            inputs[0].clone(),
            monomial(&generators, prime, &[1, 1]),
            inputs[1].clone(),
        ];
        let result = quotient(&generators, prime, &inputs, &divisors);
        assert_eq!(result.basis(), expected);
        assert_eq!(result.certificate().principal_quotients.len(), 2);
        assert_eq!(result.certificate().intersections.len(), 1);
        assert_eq!(
            quotient(
                &generators,
                prime,
                &inputs,
                &[divisors[1].clone(), divisors[0].clone()]
            )
            .basis(),
            expected
        );
        let xy = [monomial(&generators, prime, &[1, 1])];
        assert_eq!(quotient(&generators, prime, &xy, &divisors).basis(), xy);
    }
}

#[test]
fn three_divisor_chain_retains_the_socle_generator_and_each_square() {
    let generators = ring(&["x", "y", "z"]);
    for prime in [2, 5] {
        let inputs = [
            monomial(&generators, prime, &[2, 0, 0]),
            monomial(&generators, prime, &[0, 2, 0]),
            monomial(&generators, prime, &[0, 0, 2]),
        ];
        let divisors = [
            monomial(&generators, prime, &[1, 0, 0]),
            monomial(&generators, prime, &[0, 1, 0]),
            monomial(&generators, prime, &[0, 0, 1]),
        ];
        let result = quotient(&generators, prime, &inputs, &divisors);
        // A squarefree monomial m has m*x,m*y,m*z in <x^2,y^2,z^2>
        // exactly when it contains all three variables.
        assert_eq!(
            result.basis(),
            [
                inputs[0].clone(),
                monomial(&generators, prime, &[1, 1, 1]),
                inputs[1].clone(),
                inputs[2].clone()
            ]
        );
        assert_eq!(result.certificate().intersections.len(), 2);
    }
}

#[test]
fn nonmonomial_divisors_preserve_exact_coefficients_and_characteristic_two_cancellation() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 7] {
        // I=<(x-1)^2*(y+1),(x-1)*(y+1)^2>, J=<(x-1)*(y+1)>.
        let inputs = [
            p(
                &generators,
                prime,
                &[
                    (&[2, 1], 1),
                    (&[2, 0], 1),
                    (&[1, 1], prime - 2),
                    (&[1, 0], prime - 2),
                    (&[0, 1], 1),
                    (&[0, 0], 1),
                ],
            ),
            p(
                &generators,
                prime,
                &[
                    (&[1, 2], 1),
                    (&[1, 1], 2 % prime),
                    (&[1, 0], 1),
                    (&[0, 2], prime - 1),
                    (&[0, 1], prime - 2),
                    (&[0, 0], prime - 1),
                ],
            ),
        ];
        let divisor = p(
            &generators,
            prime,
            &[
                (&[1, 1], 1),
                (&[1, 0], 1),
                (&[0, 1], prime - 1),
                (&[0, 0], prime - 1),
            ],
        );
        let result = quotient(&generators, prime, &inputs, &[divisor]);
        assert_eq!(
            result.basis(),
            [
                p(&generators, prime, &[(&[1, 0], 1), (&[0, 0], prime - 1)]),
                p(&generators, prime, &[(&[0, 1], 1), (&[0, 0], 1)]),
            ]
        );
    }
    // (2*x+1)*(3*y+2) over F_5, divided by the original nonmonic 2*x+1.
    let source = p(
        &generators,
        5,
        &[(&[1, 1], 1), (&[1, 0], 4), (&[0, 1], 3), (&[0, 0], 2)],
    );
    let divisor = p(&generators, 5, &[(&[1, 0], 2), (&[0, 0], 1)]);
    assert_eq!(
        quotient(&generators, 5, &[source], &[divisor]).basis(),
        [p(&generators, 5, &[(&[0, 1], 1), (&[0, 0], 4)])]
    );
}

#[test]
fn zero_unit_and_constant_rings_obey_all_finite_colon_laws() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        let zero = p(&generators, prime, &[]);
        let unit = one(&generators, prime);
        let ideal = vec![
            monomial(&generators, prime, &[2, 0]),
            monomial(&generators, prime, &[0, 1]),
        ];
        for inputs in [vec![], ideal.clone(), vec![unit.clone()]] {
            for divisors in [vec![], vec![zero.clone()], vec![zero.clone(), zero.clone()]] {
                let result = quotient(&generators, prime, &inputs, &divisors);
                assert_eq!(result.basis(), std::slice::from_ref(&unit));
                assert!(result.certificate().principal_quotients.is_empty());
                assert!(result.certificate().intersections.is_empty());
                assert_eq!(result.inputs(), inputs);
                assert_eq!(result.divisors(), divisors);
            }
        }
        assert!(quotient(&generators, prime, &[], &ideal).basis().is_empty());
        assert_eq!(
            quotient(&generators, prime, &ideal, std::slice::from_ref(&unit)).basis(),
            ideal
        );
        assert_eq!(
            quotient(&generators, prime, std::slice::from_ref(&unit), &ideal).basis(),
            [unit]
        );
        let unit_generators = [
            monomial(&generators, prime, &[1, 0]),
            p(&generators, prime, &[(&[0, 0], 1), (&[1, 0], prime - 1)]),
        ];
        assert_eq!(
            quotient(&generators, prime, &ideal, &unit_generators).basis(),
            ideal
        );
        let constant_zero = p(&[], prime, &[]);
        let constant_one = one(&[], prime);
        assert!(
            quotient(&[], prime, &[], std::slice::from_ref(&constant_one))
                .basis()
                .is_empty()
        );
        assert_eq!(
            quotient(&[], prime, &[], &[constant_zero]).basis(),
            std::slice::from_ref(&constant_one)
        );
        assert_eq!(
            quotient(
                &[],
                prime,
                std::slice::from_ref(&constant_one),
                &[p(&[], prime, &[(&[], prime - 1)])]
            )
            .basis(),
            [constant_one]
        );
    }
}

#[test]
fn zero_duplicate_and_scaled_divisors_keep_their_original_ordered_provenance() {
    let generators = ring(&["x", "y"]);
    let zero = p(&generators, 5, &[]);
    let x = monomial(&generators, 5, &[1, 0]);
    let y = monomial(&generators, 5, &[0, 1]);
    let x2 = monomial(&generators, 5, &[2, 0]);
    let y2 = monomial(&generators, 5, &[0, 2]);
    let inputs = [x2.clone(), zero.clone(), y2.clone(), x2.clone()];
    let divisors = [zero.clone(), x, zero, y, p(&generators, 5, &[(&[1, 0], 2)])];
    let result = quotient(&generators, 5, &inputs, &divisors);
    assert_eq!(result.basis(), [x2, monomial(&generators, 5, &[1, 1]), y2]);
    assert_eq!(result.inputs(), inputs);
    assert_eq!(result.divisors(), divisors);
    assert_eq!(
        result
            .certificate()
            .principal_quotients
            .iter()
            .map(|w| w.divisor_index)
            .collect::<Vec<_>>(),
        [1, 3, 4]
    );
    assert_eq!(result.certificate().intersections.len(), 2);
}

#[test]
fn same_spelling_typed_axes_and_reserved_auxiliary_names_are_not_captured() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([101; 32]));
    let generators = vec![
        first,
        second,
        Symbol::new("_fsym_prime_field_ideal_parameter_0"),
    ];
    let inputs = [monomial(&generators, 2, &[2, 1, 0])];
    let divisors = [monomial(&generators, 2, &[1, 0, 0])];
    let result = quotient(&generators, 2, &inputs, &divisors);
    assert_eq!(result.basis(), [monomial(&generators, 2, &[1, 1, 0])]);
    assert_eq!(result.generators(), generators);
    let auxiliary = &result.certificate().principal_quotients[0]
        .intersection
        .certificate()
        .ambient_basis
        .generators()[0];
    assert!(
        generators
            .iter()
            .all(|source| source.name != auxiliary.name)
    );
    let other = [monomial(&generators, 2, &[0, 1, 0])];
    assert!(
        verify(
            &generators,
            2,
            &inputs,
            &other,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn every_nonzero_divisor_witness_is_required_once_in_original_order() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        monomial(&generators, 5, &[2, 0]),
        monomial(&generators, 5, &[0, 2]),
    ];
    let divisors = [
        monomial(&generators, 5, &[1, 0]),
        p(&generators, 5, &[]),
        monomial(&generators, 5, &[0, 1]),
    ];
    let result = quotient(&generators, 5, &inputs, &divisors);
    let mut bad = result.certificate().clone();
    bad.principal_quotients.pop();
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients.swap(0, 1);
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients[1] = bad.principal_quotients[0].clone();
    rejected(&result, &bad);
    for index in [1, usize::MAX] {
        let mut bad = result.certificate().clone();
        bad.principal_quotients[1].divisor_index = index;
        rejected(&result, &bad);
    }
    let mut bad = result.certificate().clone();
    bad.principal_quotients
        .push(bad.principal_quotients[0].clone());
    rejected(&result, &bad);
}

#[test]
fn coordinated_false_cofactor_and_accepted_basis_cannot_replace_exact_cancellation() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        let result = quotient(
            &generators,
            prime,
            &[monomial(&generators, prime, &[2, 1])],
            &[monomial(&generators, prime, &[1, 0])],
        );
        let mut bad = result.certificate().clone();
        bad.principal_quotients[0].quotients[0] =
            p(&generators, prime, &[(&[1, 1], 1), (&[0, 0], 1)]);
        bad.principal_quotients[0].quotient_basis = Arc::new(
            groebner(
                &generators,
                prime,
                &bad.principal_quotients[0].quotients,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap(),
        );
        bad.basis = bad.principal_quotients[0].quotient_basis.basis().to_vec();
        rejected(&result, &bad);
    }
}

#[test]
fn incomplete_cancellations_or_an_unrelated_normalization_cannot_prove_the_quotient() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        monomial(&generators, 5, &[2, 0]),
        monomial(&generators, 5, &[1, 1]),
    ];
    let divisors = [monomial(&generators, 5, &[1, 0])];
    let result = quotient(&generators, 5, &inputs, &divisors);
    assert_eq!(
        result.certificate().principal_quotients[0].quotients.len(),
        2
    );
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotients.pop();
    bad.principal_quotients[0].quotient_basis = Arc::new(
        groebner(
            &generators,
            5,
            &bad.principal_quotients[0].quotients,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.basis = bad.principal_quotients[0].quotient_basis.basis().to_vec();
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    let alien = [
        p(&generators, 5, &[(&[1, 0], 1), (&[0, 0], 1)]),
        monomial(&generators, 5, &[0, 1]),
    ];
    bad.principal_quotients[0].quotient_basis = Arc::new(
        groebner(
            &generators,
            5,
            &alien,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.basis = bad.principal_quotients[0].quotient_basis.basis().to_vec();
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotient_basis = Arc::new(
        groebner(
            &generators,
            5,
            &bad.principal_quotients[0].quotients,
            TermOrder::DegLex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &bad);
}

#[test]
fn immutable_intersections_remain_bound_to_the_exact_original_request_and_operation() {
    let generators = ring(&["x", "y"]);
    let inputs = [monomial(&generators, 5, &[2, 1])];
    let divisors = [monomial(&generators, 5, &[1, 0])];
    let result = quotient(&generators, 5, &inputs, &divisors);
    let scaled_inputs = [p(&generators, 5, &[(&[2, 1], 2)])];
    let scaled_divisors = [p(&generators, 5, &[(&[1, 0], 2)])];
    assert!(
        verify(
            &generators,
            5,
            &scaled_inputs,
            &divisors,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify(
            &generators,
            5,
            &inputs,
            &scaled_divisors,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].intersection = Arc::new(
        saturation(
            &generators,
            5,
            &inputs,
            &divisors[0],
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].intersection = Arc::new(
        intersection(
            &generators,
            5,
            &scaled_inputs,
            &divisors,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &bad);
}

#[test]
fn final_intersection_chain_cannot_skip_reorder_duplicate_or_replace_an_obligation() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = [
        monomial(&generators, 3, &[2, 0, 0]),
        monomial(&generators, 3, &[0, 2, 0]),
        monomial(&generators, 3, &[0, 0, 2]),
    ];
    let divisors = [
        monomial(&generators, 3, &[1, 0, 0]),
        monomial(&generators, 3, &[0, 1, 0]),
        monomial(&generators, 3, &[0, 0, 1]),
    ];
    let result = quotient(&generators, 3, &inputs, &divisors);
    let mut bad = result.certificate().clone();
    bad.intersections.pop();
    bad.basis = bad.intersections[0].basis().to_vec();
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.intersections.swap(0, 1);
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.intersections.push(bad.intersections[0].clone());
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.intersections[0] = Arc::new(
        intersection(
            &generators,
            3,
            &divisors[..1],
            &divisors[1..2],
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &bad);
}

#[test]
fn the_entire_reduced_basis_including_zero_and_unit_outcomes_is_required() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        monomial(&generators, 2, &[2, 0]),
        monomial(&generators, 2, &[1, 1]),
    ];
    let divisors = [monomial(&generators, 2, &[1, 0])];
    let result = quotient(&generators, 2, &inputs, &divisors);
    for changed in [
        vec![monomial(&generators, 2, &[1, 0])],
        vec![one(&generators, 2)],
        vec![
            monomial(&generators, 2, &[0, 1]),
            monomial(&generators, 2, &[1, 0]),
        ],
        vec![
            monomial(&generators, 2, &[2, 0]),
            monomial(&generators, 2, &[0, 2]),
        ],
        vec![],
    ] {
        let mut bad = result.certificate().clone();
        bad.basis = changed;
        rejected(&result, &bad);
    }
    let mut bad = result.certificate().clone();
    bad.basis.push(bad.basis[0].clone());
    rejected(&result, &bad);
    let zero = quotient(&generators, 2, &[], &divisors);
    let mut bad = zero.certificate().clone();
    bad.basis.push(one(&generators, 2));
    rejected(&zero, &bad);
    let unit = quotient(&generators, 2, &inputs, &[]);
    let mut bad = unit.certificate().clone();
    bad.basis.clear();
    rejected(&unit, &bad);
    let mut bad = unit.certificate().clone();
    bad.principal_quotients = result.certificate().principal_quotients.clone();
    rejected(&unit, &bad);
}

#[test]
fn wrong_typed_rings_primes_and_certificate_polynomials_cannot_hide_behind_zero() {
    let generators = ring(&["x", "y"]);
    let valid = monomial(&generators, 5, &[1, 1]);
    let zero = p(&generators, 5, &[]);
    let foreign_prime = p(&generators, 7, &[]);
    let reversed = vec![generators[1].clone(), generators[0].clone()];
    let foreign_order = p(&reversed, 5, &[]);
    let shadow = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([102; 32]));
    let foreign_identity = p(&[shadow, generators[1].clone()], 5, &[]);
    for foreign in [foreign_prime, foreign_order, foreign_identity] {
        assert!(
            checked(
                &generators,
                5,
                std::slice::from_ref(&foreign),
                &[],
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked(
                &generators,
                5,
                std::slice::from_ref(&valid),
                std::slice::from_ref(&foreign),
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked(
            &[generators[0].clone(), generators[0].clone()],
            5,
            &[],
            &[],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    for prime in [0, 1, 4, 65522, u64::MAX] {
        assert!(checked(&[], prime, &[], &[], &mut Unbounded, &mut Unbounded).is_err());
    }
    let result = quotient(
        &generators,
        5,
        &[valid],
        &[monomial(&generators, 5, &[1, 0])],
    );
    assert!(
        verify(
            &generators,
            7,
            result.inputs(),
            result.divisors(),
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotients[0] = p(&reversed, 5, &[]);
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.basis = vec![p(&generators, 7, &[])];
    rejected(&result, &bad);
    let no_constraint = quotient(&generators, 5, &[zero], &[]);
    let mut bad = no_constraint.certificate().clone();
    bad.basis = vec![one(&reversed, 5)];
    rejected(&no_constraint, &bad);
}

#[test]
fn augmented_variable_equation_and_degree_limits_refuse_without_partial_quotients() {
    let generators = ring(&["x", "y"]);
    let zero = p(&generators, 3, &[]);
    let divisor = monomial(&generators, 3, &[1, 0]);
    let boundary = monomial(&generators, 3, &[63, 0]);
    assert_eq!(
        quotient(&generators, 3, &[boundary], std::slice::from_ref(&divisor)).basis(),
        [monomial(&generators, 3, &[62, 0])]
    );
    let too_high = monomial(&generators, 3, &[64, 0]);
    assert!(
        checked(
            &generators,
            3,
            &[too_high],
            std::slice::from_ref(&divisor),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        checked(
            &generators,
            3,
            &vec![zero.clone(); 32],
            std::slice::from_ref(&divisor),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        checked(
            &generators,
            3,
            &[],
            &vec![zero.clone(); 33],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert_eq!(
        quotient(&generators, 3, &vec![zero.clone(); 32], &vec![zero; 32]).basis(),
        [one(&generators, 3)]
    );
    let wide: Vec<_> = (0..8)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    let nonzero = one(&wide, 3);
    assert!(checked(&wide, 3, &[], &[nonzero], &mut Unbounded, &mut Unbounded).is_err());
    assert_eq!(quotient(&wide, 3, &[], &[]).basis(), [one(&wide, 3)]);
    let too_wide: Vec<_> = (0..9)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    assert!(checked(&too_wide, 3, &[], &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(
        checked(
            &[Symbol::new("x".repeat(65_537))],
            3,
            &[],
            &[],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Count {
    checkpoints: usize,
    compute: u64,
    memory: u64,
    allocations: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Count {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
            Dimension::AllocationCount => self.allocations += amount,
            _ => {}
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

fn cap(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn every_nested_divisor_intersection_and_publication_shares_cumulative_phase_budgets() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        monomial(&generators, 3, &[2, 0]),
        monomial(&generators, 3, &[0, 2]),
    ];
    for divisors in [
        vec![],
        vec![
            monomial(&generators, 3, &[1, 0]),
            monomial(&generators, 3, &[0, 1]),
        ],
    ] {
        let (mut generation, mut verification) = (Count::default(), Count::default());
        checked(
            &generators,
            3,
            &inputs,
            &divisors,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        for (dimension, generated, verified) in [
            (
                Dimension::ComputeSteps,
                generation.compute,
                verification.compute,
            ),
            (
                Dimension::MemoryBytes,
                generation.memory,
                verification.memory,
            ),
            (
                Dimension::AllocationCount,
                generation.allocations,
                verification.allocations,
            ),
        ] {
            assert!(generated > 1 && verified > 1);
            checked(
                &generators,
                3,
                &inputs,
                &divisors,
                &mut cap(dimension, generated),
                &mut cap(dimension, verified),
            )
            .unwrap();
            assert!(
                checked(
                    &generators,
                    3,
                    &inputs,
                    &divisors,
                    &mut cap(dimension, generated - 1),
                    &mut Unbounded
                )
                .is_err()
            );
            assert!(
                checked(
                    &generators,
                    3,
                    &inputs,
                    &divisors,
                    &mut Unbounded,
                    &mut cap(dimension, verified - 1)
                )
                .is_err()
            );
            let mut used = cap(dimension, generated);
            used.charge(dimension, 1).unwrap();
            assert!(
                checked(
                    &generators,
                    3,
                    &inputs,
                    &divisors,
                    &mut used,
                    &mut Unbounded
                )
                .is_err()
            );
        }
    }
}

#[test]
fn cancellation_in_both_phases_and_the_last_publication_checkpoint_is_terminal() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        monomial(&generators, 2, &[2, 0]),
        monomial(&generators, 2, &[0, 2]),
    ];
    for divisors in [
        vec![],
        vec![
            monomial(&generators, 2, &[1, 0]),
            monomial(&generators, 2, &[0, 1]),
        ],
    ] {
        let (mut generation, mut verification) = (Count::default(), Count::default());
        let result = checked(
            &generators,
            2,
            &inputs,
            &divisors,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        for (in_generator, count) in [
            (true, generation.checkpoints),
            (false, verification.checkpoints),
        ] {
            for at in [1, count / 2, count - 1, count] {
                let mut stopped = Count {
                    cancel_at: Some(at),
                    ..Count::default()
                };
                let result = if in_generator {
                    checked(
                        &generators,
                        2,
                        &inputs,
                        &divisors,
                        &mut stopped,
                        &mut Unbounded,
                    )
                } else {
                    checked(
                        &generators,
                        2,
                        &inputs,
                        &divisors,
                        &mut Unbounded,
                        &mut stopped,
                    )
                };
                assert!(result.is_err(), "generator={in_generator}, checkpoint={at}");
            }
        }
        let mut standalone = Count::default();
        verify(
            &generators,
            2,
            &inputs,
            &divisors,
            result.certificate(),
            &mut standalone,
        )
        .unwrap();
        let mut stopped = Count {
            cancel_at: Some(standalone.checkpoints),
            ..Count::default()
        };
        assert!(
            verify(
                &generators,
                2,
                &inputs,
                &divisors,
                result.certificate(),
                &mut stopped
            )
            .is_err()
        );
    }
    let mut never_started = Count::default();
    assert!(
        checked(
            &generators,
            2,
            &inputs,
            &[],
            &mut Budget::new(BudgetLimits::uniform(1, 0)),
            &mut never_started
        )
        .is_err()
    );
    assert_eq!(never_started.checkpoints, 0);
}
