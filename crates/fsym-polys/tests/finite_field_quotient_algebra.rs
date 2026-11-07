//! Complete finite F_p quotient operators with exact matrix and evidence adversaries.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    PrimeFieldQuotientMultiplicationCertificate as Certificate, TermOrder,
    VerifiedPrimeFieldQuotientMultiplication as Operator,
    checked_prime_field_groebner_basis as groebner,
    checked_prime_field_quotient_multiplication as checked,
    verify_prime_field_quotient_multiplication as verify,
};

const ORDERS: [TermOrder; 3] = [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex];

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn poly(generators: &[Symbol], prime: u64, terms: &[(&[u32], u64)]) -> Polynomial {
    let terms: Terms = terms.iter().map(|(m, c)| (m.to_vec(), *c)).collect();
    Polynomial::new(generators, prime, &terms).unwrap()
}

fn coupled(generators: &[Symbol], prime: u64) -> Vec<Polynomial> {
    vec![
        poly(generators, prime, &[(&[1, 1], 1), (&[0, 0], prime - 1)]),
        poly(generators, prime, &[(&[0, 2], 1), (&[1, 0], prime - 1)]),
    ]
}

fn operator(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
) -> Operator {
    checked(
        generators,
        prime,
        inputs,
        element,
        order,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn replay(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
    certificate: &Certificate,
) -> Operator {
    verify(
        generators,
        prime,
        inputs,
        element,
        order,
        certificate,
        &mut Unbounded,
    )
    .unwrap()
}

fn reject(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
    certificate: &Certificate,
) {
    assert!(
        verify(
            generators,
            prime,
            inputs,
            element,
            order,
            certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

fn product(left: &[Vec<u64>], right: &[Vec<u64>], prime: u64) -> Vec<Vec<u64>> {
    let n = left.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| left[i][k] * right[k][j]).sum::<u64>() % prime)
                .collect()
        })
        .collect()
}

#[test]
fn all_orders_and_small_primes_have_exact_column_action_and_request_accessors() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 7] {
        let inputs = coupled(&generators, prime);
        let x = poly(&generators, prime, &[(&[1, 0], 1)]);
        let y = poly(&generators, prime, &[(&[0, 1], 1)]);
        for order in ORDERS {
            let result = operator(&generators, prime, &inputs, &x, order);
            assert_eq!(result.generators(), generators);
            assert_eq!(result.prime(), prime);
            assert_eq!(result.inputs(), inputs);
            assert_eq!(result.element(), &x);
            assert_eq!(result.order(), order);
            assert_eq!(
                result.matrix(),
                &[vec![0, 1, 0], vec![0, 0, 1], vec![1, 0, 0]]
            );
            let wanted = if order == TermOrder::Lex {
                vec![vec![0, 0], vec![0, 1], vec![0, 2]]
            } else {
                vec![vec![0, 0], vec![0, 1], vec![1, 0]]
            };
            assert_eq!(result.standard_monomials(), wanted);
            assert_eq!(result.certificate().source_basis.inputs(), inputs);
            assert_eq!(
                result
                    .certificate()
                    .source_basis
                    .standard_monomials(&mut Unbounded)
                    .unwrap(),
                result.standard_monomials(),
            );
            let again = replay(&generators, prime, &inputs, &x, order, result.certificate());
            assert_eq!(again.matrix(), result.matrix());
            assert_eq!(
                operator(&generators, prime, &inputs, &y, order).matrix(),
                &[vec![0, 0, 1], vec![1, 0, 0], vec![0, 1, 0]],
            );
        }
    }
}

#[test]
fn irreducible_no_rational_point_and_repeated_root_quotients_are_kept() {
    let generators = ring(&["x"]);
    for prime in [2, 3] {
        let inputs = [poly(&generators, prime, &[(&[2], 1), (&[0], 1)])];
        let x = poly(&generators, prime, &[(&[1], 1)]);
        let result = operator(&generators, prime, &inputs, &x, TermOrder::Lex);
        assert_eq!(result.standard_monomials(), &[vec![0], vec![1]]);
        assert_eq!(result.matrix(), &[vec![0, prime - 1], vec![1, 0]]);
        let square = product(result.matrix(), result.matrix(), prime);
        assert_eq!(square, vec![vec![prime - 1, 0], vec![0, prime - 1]]);
        if prime == 2 {
            let shifted = operator(
                &generators,
                prime,
                &inputs,
                &poly(&generators, prime, &[(&[1], 1), (&[0], 1)]),
                TermOrder::Lex,
            );
            assert_eq!(shifted.matrix(), &[vec![1, 1], vec![1, 1]]);
            assert_eq!(
                product(shifted.matrix(), shifted.matrix(), prime),
                vec![vec![0; 2]; 2]
            );
        }
    }
}

#[test]
fn general_nonmonomial_elements_have_all_exact_modular_coefficients() {
    let generators = ring(&["x", "y"]);
    for prime in [5, 7] {
        let inputs = [
            poly(&generators, prime, &[(&[2, 0], 1), (&[0, 0], prime - 2)]),
            poly(&generators, prime, &[(&[0, 2], 1), (&[0, 0], prime - 3)]),
        ];
        let element = poly(
            &generators,
            prime,
            &[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], 2)],
        );
        for order in ORDERS {
            let result = operator(&generators, prime, &inputs, &element, order);
            assert_eq!(
                result.standard_monomials(),
                &[vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
            );
            assert_eq!(
                result.matrix(),
                &[
                    vec![2, 3, 2, 0],
                    vec![1, 2, 0, 2],
                    vec![1, 0, 2, 3],
                    vec![0, 1, 1, 2]
                ],
            );
        }
    }
}

#[test]
fn nonradical_multivariate_multiplicity_and_characteristic_two_cancellation_are_exact() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        let inputs = [
            poly(&generators, prime, &[(&[2, 0], 1)]),
            poly(&generators, prime, &[(&[0, 2], 1)]),
        ];
        let element = poly(&generators, prime, &[(&[1, 0], 1), (&[0, 1], 1)]);
        let square = poly(
            &generators,
            prime,
            &[(&[2, 0], 1), (&[1, 1], 2 % prime), (&[0, 2], 1)],
        );
        for order in ORDERS {
            let result = operator(&generators, prime, &inputs, &element, order);
            assert_eq!(
                result.standard_monomials(),
                &[vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
            );
            assert_eq!(
                result.matrix(),
                &[
                    vec![0, 0, 0, 0],
                    vec![1, 0, 0, 0],
                    vec![1, 0, 0, 0],
                    vec![0, 1, 1, 0]
                ],
            );
            let expected_square =
                vec![vec![0; 4], vec![0; 4], vec![0; 4], vec![2 % prime, 0, 0, 0]];
            assert_eq!(
                product(result.matrix(), result.matrix(), prime),
                expected_square
            );
            assert_eq!(
                operator(&generators, prime, &inputs, &square, order).matrix(),
                expected_square
            );
            assert_eq!(
                product(&expected_square, result.matrix(), prime),
                vec![vec![0; 4]; 4]
            );
        }
    }
}

#[test]
fn three_variable_staircases_distinguish_graded_lex_and_reverse_lex_ordering() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = [
        poly(&generators, 5, &[(&[2, 0, 0], 1)]),
        poly(&generators, 5, &[(&[0, 3, 0], 1)]),
        poly(&generators, 5, &[(&[0, 0, 2], 1)]),
    ];
    let one = poly(&generators, 5, &[(&[0, 0, 0], 1)]);
    let mut staircases = Vec::new();
    for order in ORDERS {
        let result = operator(&generators, 5, &inputs, &one, order);
        assert_eq!(result.standard_monomials().len(), 12);
        for x in 0..2 {
            for y in 0..3 {
                for z in 0..2 {
                    assert!(result.standard_monomials().contains(&vec![x, y, z]));
                }
            }
        }
        for (i, row) in result.matrix().iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_eq!(*entry, u64::from(i == j));
            }
        }
        staircases.push(result.standard_monomials().to_vec());
    }
    let position = |basis: &[Vec<u32>], m: &[u32]| basis.iter().position(|v| v == m).unwrap();
    assert!(position(&staircases[1], &[0, 2, 0]) < position(&staircases[1], &[1, 0, 1]));
    assert!(position(&staircases[2], &[1, 0, 1]) < position(&staircases[2], &[0, 2, 0]));
}

#[test]
fn zero_and_scalar_operators_cover_proper_unit_and_constant_rings() {
    let generators = ring(&["x"]);
    let proper = [poly(&generators, 5, &[(&[2], 1)])];
    for coefficient in [0, 1, 3] {
        let element = poly(&generators, 5, &[(&[0], coefficient)]);
        let result = operator(&generators, 5, &proper, &element, TermOrder::Lex);
        assert_eq!(
            result.matrix(),
            &[vec![coefficient, 0], vec![0, coefficient]]
        );
    }
    let unit = [poly(&generators, 5, &[(&[0], 3)])];
    let element = poly(&generators, 5, &[(&[4], 1)]);
    let result = operator(&generators, 5, &unit, &element, TermOrder::Lex);
    assert!(result.standard_monomials().is_empty());
    assert!(result.matrix().is_empty());
    assert!(result.certificate().reduction_quotients.is_empty());
    replay(
        &generators,
        5,
        &unit,
        &element,
        TermOrder::Lex,
        result.certificate(),
    );

    let empty: Vec<Symbol> = vec![];
    let zero = poly(&empty, 5, &[]);
    let scalar = poly(&empty, 5, &[(&[], 3)]);
    for inputs in [vec![], vec![zero], vec![scalar.clone()]] {
        for order in ORDERS {
            let result = operator(&empty, 5, &inputs, &scalar, order);
            if inputs.iter().all(Polynomial::is_zero) {
                assert_eq!(result.standard_monomials(), &[Vec::<u32>::new()]);
                assert_eq!(result.matrix(), &[vec![3]]);
            } else {
                assert!(result.standard_monomials().is_empty());
                assert!(result.matrix().is_empty());
            }
            replay(&empty, 5, &inputs, &scalar, order, result.certificate());
        }
    }
}

#[test]
fn a_missing_staircase_tail_is_rejected_even_when_every_proposed_column_is_zero() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        poly(&generators, 5, &[(&[2, 0], 1)]),
        poly(&generators, 5, &[(&[0, 2], 1)]),
    ];
    let zero = poly(&generators, 5, &[]);
    let result = operator(&generators, 5, &inputs, &zero, TermOrder::Lex);
    let mut mutant = result.certificate().clone();
    mutant.standard_monomials.pop();
    mutant.matrix.pop();
    for row in &mut mutant.matrix {
        row.pop();
    }
    mutant.reduction_quotients.pop();
    // All three retained zero columns still have valid identities, but x*y is missing.
    reject(&generators, 5, &inputs, &zero, TermOrder::Lex, &mutant);

    mutant = result.certificate().clone();
    mutant.standard_monomials = vec![vec![1, 1]];
    mutant.matrix = vec![vec![0]];
    mutant.reduction_quotients = vec![vec![Terms::new(); mutant.source_basis.basis().len()]];
    // The singleton socle is closed under all unblocked variable multiples but omits 1.
    reject(&generators, 5, &inputs, &zero, TermOrder::Lex, &mutant);

    mutant = result.certificate().clone();
    mutant.standard_monomials.clear();
    mutant.matrix.clear();
    mutant.reduction_quotients.clear();
    reject(&generators, 5, &inputs, &zero, TermOrder::Lex, &mutant);
}

#[test]
fn duplicate_unordered_nonstandard_and_malformed_staircases_are_rejected() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    let zero = poly(&generators, 5, &[]);
    let result = operator(&generators, 5, &inputs, &zero, TermOrder::Lex);
    let certificate = result.certificate();
    let mut mutant = certificate.clone();
    mutant.standard_monomials[2] = mutant.standard_monomials[1].clone();
    reject(&generators, 5, &inputs, &zero, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.standard_monomials.swap(1, 2);
    reject(&generators, 5, &inputs, &zero, TermOrder::Lex, &mutant);
    for bad in [
        vec![1, 0],
        vec![0, 3],
        vec![0],
        vec![0, 0, 0],
        vec![65, 0],
        vec![u32::MAX, 1],
    ] {
        mutant = certificate.clone();
        mutant.standard_monomials[2] = bad;
        reject(&generators, 5, &inputs, &zero, TermOrder::Lex, &mutant);
    }
    mutant = certificate.clone();
    mutant.standard_monomials = vec![vec![0, 0]; 65];
    reject(&generators, 5, &inputs, &zero, TermOrder::Lex, &mutant);
}

#[test]
fn matrix_entries_dimensions_column_permutations_and_missing_witnesses_are_rejected() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    let element = poly(&generators, 5, &[(&[1, 0], 1)]);
    let result = operator(&generators, 5, &inputs, &element, TermOrder::Lex);
    let certificate = result.certificate();
    for value in [1, 5, u64::MAX] {
        let mut mutant = certificate.clone();
        mutant.matrix[0][0] = value;
        reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    }
    let mut mutant = certificate.clone();
    mutant.matrix.pop();
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.matrix[0].pop();
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.matrix.push(vec![0; 3]);
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    for row in &mut mutant.matrix {
        row.swap(0, 1);
    }
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.reduction_quotients.pop();
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.reduction_quotients[0].pop();
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.reduction_quotients[0].push(Terms::new());
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
}

#[test]
fn changed_and_noncanonical_column_cofactors_cannot_forge_exact_reductions() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    let element = poly(&generators, 5, &[(&[1, 0], 1)]);
    let result = operator(&generators, 5, &inputs, &element, TermOrder::Lex);
    let certificate = result.certificate();
    let (column, divisor, monomial, coefficient) = certificate
        .reduction_quotients
        .iter()
        .enumerate()
        .find_map(|(column, quotients)| {
            quotients
                .iter()
                .enumerate()
                .find_map(|(divisor, quotient)| {
                    quotient
                        .iter()
                        .next()
                        .map(|(m, c)| (column, divisor, m.clone(), *c))
                })
        })
        .unwrap();
    let mut mutant = certificate.clone();
    mutant.reduction_quotients[column][divisor].insert(monomial, (coefficient + 1) % 5);
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    for (monomial, coefficient) in [
        (vec![0], 1),
        (vec![0, 0, 0], 1),
        (vec![65, 0], 1),
        (vec![u32::MAX, 1], 1),
        (vec![0, 0], 5),
        (vec![0, 0], u64::MAX),
    ] {
        mutant = certificate.clone();
        mutant.reduction_quotients[0][0].insert(monomial, coefficient);
        reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    }
}

#[test]
fn immutable_source_receipt_is_bound_to_original_inputs_order_and_prime() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    let element = poly(&generators, 5, &[(&[1, 0], 1)]);
    let result = operator(&generators, 5, &inputs, &element, TermOrder::Lex);
    let certificate = result.certificate();
    reject(
        &generators,
        3,
        &inputs,
        &element,
        TermOrder::Lex,
        certificate,
    );
    reject(
        &generators,
        5,
        &inputs,
        &element,
        TermOrder::DegLex,
        certificate,
    );
    let changed_element = poly(&generators, 5, &[(&[1, 0], 1), (&[0, 0], 1)]);
    reject(
        &generators,
        5,
        &inputs,
        &changed_element,
        TermOrder::Lex,
        certificate,
    );

    let mut reversed = inputs.clone();
    reversed.reverse();
    reject(
        &generators,
        5,
        &reversed,
        &element,
        TermOrder::Lex,
        certificate,
    );
    let mut mutant = certificate.clone();
    mutant.source_basis = Arc::new(
        groebner(
            &generators,
            5,
            &reversed,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.source_basis = Arc::new(
        groebner(
            &generators,
            5,
            &inputs,
            TermOrder::DegRevLex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    let other_inputs = [poly(&generators, 5, &[(&[0, 0], 1)])];
    mutant = certificate.clone();
    mutant.source_basis = Arc::new(
        groebner(
            &generators,
            5,
            &other_inputs,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.source_basis = Arc::new(
        groebner(
            &generators,
            3,
            &coupled(&generators, 3),
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    reject(&generators, 5, &inputs, &element, TermOrder::Lex, &mutant);
}

#[test]
fn typed_same_spelling_axes_are_distinct_and_element_ownership_is_checked() {
    let generators = vec![
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([41; 32])),
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([42; 32])),
    ];
    let inputs = coupled(&generators, 5);
    let element = poly(&generators, 5, &[(&[1, 0], 1)]);
    let result = operator(&generators, 5, &inputs, &element, TermOrder::Lex);
    assert_eq!(result.generators(), generators);
    assert_eq!(result.element().generators(), generators);
    let reversed = vec![generators[1].clone(), generators[0].clone()];
    let foreign = ring(&["x", "y"]);
    for changed in [reversed, foreign] {
        reject(
            &changed,
            5,
            &inputs,
            &element,
            TermOrder::Lex,
            result.certificate(),
        );
        let changed_element = poly(&changed, 5, &[(&[1, 0], 1)]);
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                &changed_element,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
        reject(
            &generators,
            5,
            &inputs,
            &changed_element,
            TermOrder::Lex,
            result.certificate(),
        );
    }
    let wrong_prime = poly(&generators, 3, &[(&[1, 0], 1)]);
    assert!(
        checked(
            &generators,
            5,
            &inputs,
            &wrong_prime,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn positive_dimensional_ideals_refuse_even_for_zero_or_scalar_elements() {
    let generators = ring(&["x", "y"]);
    for inputs in [
        vec![],
        vec![poly(&generators, 5, &[(&[1, 1], 1)])],
        vec![poly(&generators, 5, &[(&[1, 0], 1)])],
    ] {
        for coefficient in [0, 1, 3] {
            let element = poly(&generators, 5, &[(&[0, 0], coefficient)]);
            for order in ORDERS {
                assert!(
                    checked(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        order,
                        &mut Unbounded,
                        &mut Unbounded
                    )
                    .is_err()
                );
            }
        }
    }
}

#[test]
fn quotient_dimension_sixty_four_is_complete_and_larger_products_refuse() {
    let generators = ring(&["x"]);
    let inputs = [poly(&generators, 2, &[(&[64], 1)])];
    let element = poly(&generators, 2, &[(&[1], 1)]);
    let result = operator(&generators, 2, &inputs, &element, TermOrder::Lex);
    assert_eq!(
        result.standard_monomials(),
        (0..64).map(|k| vec![k]).collect::<Vec<_>>()
    );
    assert_eq!(result.matrix().len(), 64);
    for (row, entries) in result.matrix().iter().enumerate() {
        assert_eq!(entries.len(), 64);
        for (column, entry) in entries.iter().enumerate() {
            assert_eq!(*entry, u64::from(row == column + 1));
        }
    }
    let generators = ring(&["x", "y"]);
    let inputs = [
        poly(&generators, 5, &[(&[8, 0], 1)]),
        poly(&generators, 5, &[(&[0, 9], 1)]),
    ];
    let zero = poly(&generators, 5, &[]);
    assert!(
        checked(
            &generators,
            5,
            &inputs,
            &zero,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn raw_input_and_ring_admission_precedes_zero_and_unit_fast_paths() {
    let generators = ring(&["x"]);
    let element = poly(&generators, 5, &[]);
    let square = poly(&generators, 5, &[(&[2], 1)]);
    let inputs = vec![square; 32];
    assert_eq!(
        operator(&generators, 5, &inputs, &element, TermOrder::Lex).standard_monomials(),
        &[vec![0], vec![1]]
    );
    let mut too_many = inputs.clone();
    too_many.push(poly(&generators, 5, &[]));
    assert!(
        checked(
            &generators,
            5,
            &too_many,
            &element,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    for prime in [0, 1, 4, 65_537] {
        assert!(
            checked(
                &generators,
                prime,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let duplicate = vec![generators[0].clone(), generators[0].clone()];
    assert!(
        checked(
            &duplicate,
            5,
            &inputs,
            &element,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let too_many_axes: Vec<_> = (0..9).map(|i| Symbol::new(format!("x{i}"))).collect();
    assert!(
        checked(
            &too_many_axes,
            5,
            &[],
            &element,
            TermOrder::Lex,
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
fn phase_budgets_span_source_staircase_every_column_and_final_publication() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    let element = poly(&generators, 5, &[(&[1, 0], 1)]);
    let (mut generation, mut verification) = (Count::default(), Count::default());
    checked(
        &generators,
        5,
        &inputs,
        &element,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for (dimension, used_g, used_v) in [
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
        assert!(used_g > 0 && used_v > 0);
        checked(
            &generators,
            5,
            &inputs,
            &element,
            TermOrder::Lex,
            &mut cap(dimension, used_g),
            &mut cap(dimension, used_v),
        )
        .unwrap();
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut cap(dimension, used_g - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut Unbounded,
                &mut cap(dimension, used_v - 1)
            )
            .is_err()
        );
        let mut spent = cap(dimension, used_g);
        BudgetMeter::charge(&mut spent, dimension, 1).unwrap();
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut spent,
                &mut Unbounded
            )
            .is_err()
        );
        let mut spent = cap(dimension, used_v);
        BudgetMeter::charge(&mut spent, dimension, 1).unwrap();
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut Unbounded,
                &mut spent
            )
            .is_err()
        );
    }
}

#[test]
fn cancellation_and_independent_verifier_budget_fail_before_any_accepted_result() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    let element = poly(&generators, 5, &[(&[1, 0], 1)]);
    let (mut generation, mut verification) = (Count::default(), Count::default());
    let result = checked(
        &generators,
        5,
        &inputs,
        &element,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for at in [
        1,
        generation.checkpoints / 2,
        generation.checkpoints - 1,
        generation.checkpoints,
    ] {
        let mut cancelled = Count {
            cancel_at: Some(at),
            ..Count::default()
        };
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut cancelled,
                &mut Unbounded
            )
            .is_err()
        );
    }
    for at in [
        1,
        verification.checkpoints / 2,
        verification.checkpoints - 1,
        verification.checkpoints,
    ] {
        let mut cancelled = Count {
            cancel_at: Some(at),
            ..Count::default()
        };
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut Unbounded,
                &mut cancelled
            )
            .is_err()
        );
    }
    let mut standalone = Count::default();
    verify(
        &generators,
        5,
        &inputs,
        &element,
        TermOrder::Lex,
        result.certificate(),
        &mut standalone,
    )
    .unwrap();
    let mut cancelled = Count {
        cancel_at: Some(standalone.checkpoints),
        ..Count::default()
    };
    assert!(
        verify(
            &generators,
            5,
            &inputs,
            &element,
            TermOrder::Lex,
            result.certificate(),
            &mut cancelled
        )
        .is_err()
    );
    for (dimension, used) in [
        (Dimension::ComputeSteps, standalone.compute),
        (Dimension::MemoryBytes, standalone.memory),
        (Dimension::AllocationCount, standalone.allocations),
    ] {
        verify(
            &generators,
            5,
            &inputs,
            &element,
            TermOrder::Lex,
            result.certificate(),
            &mut cap(dimension, used),
        )
        .unwrap();
        assert!(
            verify(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                result.certificate(),
                &mut cap(dimension, used - 1)
            )
            .is_err()
        );
    }
}
