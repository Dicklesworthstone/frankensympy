//! Complete prime-field relation modules, hand-written independent evidence,
//! and valid-but-incomplete adversaries that a dot-product-only check misses.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity, parse};
use fsym_polys::{
    PrimeFieldGroebnerCertificate, PrimeFieldGroebnerSPairCertificate,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    PrimeFieldPolynomialSyzygyCertificate as Certificate, TermOrder,
    VerifiedPrimeFieldPolynomialSyzygies as Verified,
    bounded_prime_field_syzygy_certificate as candidate,
    checked_prime_field_groebner_basis as groebner,
    checked_prime_field_multivariate_division as divide, checked_prime_field_multivariate_source,
    checked_prime_field_polynomial_syzygies as checked,
    verify_prime_field_polynomial_syzygies as verify,
};
use std::collections::BTreeMap;

const ORDERS: [TermOrder; 3] = [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex];

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn p(source: &str, generators: &[Symbol], prime: u64) -> Polynomial {
    checked_prime_field_multivariate_source(
        &parse(source).unwrap(),
        generators,
        prime,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
    .into_polynomial()
}

fn polys(sources: &[&str], generators: &[Symbol], prime: u64) -> Vec<Polynomial> {
    sources
        .iter()
        .map(|source| p(source, generators, prime))
        .collect()
}

fn run(generators: &[Symbol], prime: u64, inputs: &[Polynomial], order: TermOrder) -> Verified {
    checked(
        generators,
        prime,
        inputs,
        order,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn certificate(generators: &[Symbol], prime: u64, inputs: &[Polynomial]) -> Certificate {
    candidate(generators, prime, inputs, TermOrder::Lex, &mut Unbounded).unwrap()
}

fn reject(generators: &[Symbol], prime: u64, inputs: &[Polynomial], evidence: &Certificate) {
    assert!(
        verify(
            generators,
            prime,
            inputs,
            TermOrder::Lex,
            evidence,
            &mut Unbounded
        )
        .is_err()
    );
}

fn assert_zero_relations(result: &Verified) {
    // Direct test-side modular convolution, not the producer or its checker.
    for row in result.syzygies() {
        assert_eq!(row.len(), result.inputs().len());
        assert!(row.iter().any(|cell| !cell.is_zero()));
        let mut coefficients: BTreeMap<Vec<u32>, u64> = BTreeMap::new();
        for (cell, input) in row.iter().zip(result.inputs()) {
            assert_eq!(cell.generators(), result.generators());
            assert_eq!(cell.prime(), result.prime());
            for (a, ca) in cell.terms() {
                for (b, cb) in input.terms() {
                    let powers: Vec<_> = a.iter().zip(b).map(|(x, y)| x + y).collect();
                    let previous = coefficients.get(&powers).copied().unwrap_or(0);
                    coefficients.insert(powers, (previous + ca * cb) % result.prime());
                }
            }
        }
        assert!(coefficients.values().all(|value| *value == 0));
    }
}

#[test]
fn every_monomial_pair_is_present_in_the_complete_ordered_generating_set() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let inputs = polys(&["x**2", "x*y", "y**2"], &generators, prime);
        let expected = vec![
            polys(&["y", "-x", "0"], &generators, prime),
            polys(&["y**2", "0", "-x**2"], &generators, prime),
            polys(&["0", "y", "-x"], &generators, prime),
        ];
        for order in ORDERS {
            let result = run(&generators, prime, &inputs, order);
            assert_eq!(result.syzygies(), expected);
            assert_eq!(result.generators(), generators);
            assert_eq!(result.prime(), prime);
            assert_eq!(result.inputs(), inputs);
            assert_eq!(result.order(), order);
            assert_eq!(result.basis().inputs(), inputs);
            assert_zero_relations(&result);
            assert_eq!(result.into_syzygies(), expected);
        }
    }
}

#[test]
fn nonbasis_sources_and_nonzero_spair_reductions_keep_original_coordinates() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        let inputs = polys(&["x**2+y", "x*y"], &generators, prime);
        let result = run(&generators, prime, &inputs, TermOrder::Lex);
        assert_eq!(result.basis().basis().len(), 3);
        assert_eq!(
            result.syzygies(),
            &[
                polys(&["-x**2*y", "x**3+x*y"], &generators, prime),
                polys(&["-x*y", "x**2+y"], &generators, prime),
            ]
        );
        assert_zero_relations(&result);
    }
}

#[test]
fn change_of_generators_keeps_both_required_composition_rows() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 7] {
        let inputs = polys(&["x*y", "x*(y+1)"], &generators, prime);
        let result = run(&generators, prime, &inputs, TermOrder::Lex);
        let primitive = polys(&["y+1", "-y"], &generators, prime);
        assert_eq!(result.basis().basis(), polys(&["x"], &generators, prime));
        assert_eq!(result.syzygies(), &[primitive.clone(), primitive]);
        assert_zero_relations(&result);
    }
}

#[test]
fn characteristic_changes_zero_and_scaled_coordinate_relations() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 7] {
        let inputs = polys(&["2*x", "3*y"], &generators, prime);
        let result = run(&generators, prime, &inputs, TermOrder::Lex);
        let expected = if prime == 2 {
            polys(&["1", "0"], &generators, prime)
        } else if prime == 3 {
            polys(&["0", "1"], &generators, prime)
        } else {
            let half = (1..prime).find(|value| value * 2 % prime == 1).unwrap();
            let third = (1..prime).find(|value| value * 3 % prime == 1).unwrap();
            polys(
                &[&format!("{half}*y"), &format!("-{third}*x")],
                &generators,
                prime,
            )
        };
        assert_eq!(result.syzygies(), &[expected]);
        assert_zero_relations(&result);
        if prime > 3 {
            let evidence = certificate(&generators, prime, &inputs);
            reject(
                &generators,
                prime,
                &polys(&["x", "y"], &generators, prime),
                &evidence,
            );
        }
    }
}

#[test]
fn zero_duplicate_and_redundant_inputs_retain_every_original_position() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        let inputs = polys(&["x", "y", "x+y", "0"], &generators, prime);
        let result = run(&generators, prime, &inputs, TermOrder::Lex);
        assert_eq!(
            result.syzygies(),
            &[
                polys(&["-1", "-1", "1", "0"], &generators, prime),
                polys(&["0", "0", "0", "1"], &generators, prime),
                polys(&["y", "-x", "0", "0"], &generators, prime),
            ]
        );
        assert_zero_relations(&result);
        let inputs = polys(&["0", "x", "0", "2*x"], &generators, prime);
        let result = run(&generators, prime, &inputs, TermOrder::Lex);
        assert_eq!(
            result.syzygies(),
            &[
                polys(&["1", "0", "0", "0"], &generators, prime),
                polys(&["0", "0", "1", "0"], &generators, prime),
                polys(&["0", "-2", "0", "1"], &generators, prime),
            ]
        );
        assert_zero_relations(&result);
        let inputs = polys(&["x", "x"], &generators, prime);
        assert_eq!(
            run(&generators, prime, &inputs, TermOrder::Lex).syzygies(),
            &[polys(&["-1", "1"], &generators, prime)]
        );
    }
}

#[test]
fn empty_zero_unit_and_constant_fields_have_complete_relation_spaces() {
    for generators in [vec![], ring(&["x", "y"])] {
        for prime in [2, 5, 65521] {
            let result = run(&generators, prime, &[], TermOrder::DegRevLex);
            assert_eq!(result.generators(), generators);
            assert_eq!(result.prime(), prime);
            assert!(result.inputs().is_empty() && result.syzygies().is_empty());
            let zero = polys(&["0", "0"], &generators, prime);
            assert_eq!(
                run(&generators, prime, &zero, TermOrder::Lex).syzygies(),
                &[
                    polys(&["1", "0"], &generators, prime),
                    polys(&["0", "1"], &generators, prime),
                ]
            );
            let inputs = polys(&["0", "-1", "1"], &generators, prime);
            let result = run(&generators, prime, &inputs, TermOrder::Lex);
            assert_eq!(
                result.syzygies(),
                &[
                    polys(&["1", "0", "0"], &generators, prime),
                    polys(&["0", "1", "1"], &generators, prime),
                ]
            );
            assert_zero_relations(&result);
            assert!(
                run(
                    &generators,
                    prime,
                    &polys(&["-1"], &generators, prime),
                    TermOrder::Lex
                )
                .syzygies()
                .is_empty()
            );
        }
    }
    let generators = ring(&["x"]);
    assert_eq!(
        run(
            &generators,
            5,
            &polys(&["x", "1"], &generators, 5),
            TermOrder::Lex
        )
        .syzygies(),
        &[polys(&["1", "-x"], &generators, 5)]
    );
}

#[test]
fn every_order_spans_the_independently_known_primitive_koszul_relation() {
    let generators = ring(&["x", "y"]);
    // f=x+y^2 and g=xy+1 are coprime in every characteristic. Their complete
    // relation module is generated by (g,-f), regardless of Groebner order.
    for prime in [2, 3, 5] {
        let inputs = polys(&["x+y**2", "x*y+1"], &generators, prime);
        for order in ORDERS {
            let result = run(&generators, prime, &inputs, order);
            assert_zero_relations(&result);
            let mut factors = Vec::new();
            for row in result.syzygies() {
                let division = divide(
                    &row[0],
                    &inputs[1..],
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut Unbounded,
                )
                .unwrap();
                let (quotients, remainder) = division.into_parts();
                assert!(remainder.is_zero());
                factors.push(quotients[0].clone());
            }
            let span = groebner(
                &generators,
                prime,
                &factors,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap();
            assert_eq!(span.basis(), polys(&["1"], &generators, prime));
        }
    }
}

fn hand_certificate(generators: &[Symbol], prime: u64) -> Certificate {
    let zero = Terms::new();
    let one = p("1", generators, prime).terms().clone();
    Certificate {
        groebner: PrimeFieldGroebnerCertificate {
            basis: polys(&["x", "y"], generators, prime)
                .iter()
                .map(|value| value.terms().clone())
                .collect(),
            basis_in_input: vec![
                vec![one.clone(), zero.clone()],
                vec![zero.clone(), one.clone()],
            ],
            input_in_basis: vec![vec![one.clone(), zero.clone()], vec![zero.clone(), one]],
            s_pairs: vec![PrimeFieldGroebnerSPairCertificate {
                left: 0,
                right: 1,
                quotients: vec![zero.clone(), zero],
            }],
        },
        syzygies: vec![polys(&["y", "-x"], generators, prime)],
    }
}

#[test]
fn independent_verifier_accepts_hand_written_complete_field_evidence() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let inputs = polys(&["x", "y"], &generators, prime);
        let result = verify(
            &generators,
            prime,
            &inputs,
            TermOrder::Lex,
            &hand_certificate(&generators, prime),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(
            result.syzygies(),
            &[polys(&["y", "-x"], &generators, prime)]
        );
        assert_zero_relations(&result);
    }
}

#[test]
fn valid_but_incomplete_reordered_duplicate_or_zero_rows_are_rejected() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["x**2", "x*y", "y**2"], &generators, 5);
    let original = certificate(&generators, 5, &inputs);
    let mut changed = original.clone();
    changed.syzygies.pop();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.syzygies.clear();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.syzygies.swap(0, 2);
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.syzygies[1] = changed.syzygies[0].clone();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed
        .syzygies
        .push(polys(&["0", "0", "0"], &generators, 5));
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.syzygies[0] = polys(&["2*y", "-2*x", "0"], &generators, 5);
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original;
    changed.syzygies[0][0] = p("y+1", &generators, 5);
    reject(&generators, 5, &inputs, &changed);
}

#[test]
fn full_underlying_basis_inclusions_and_strict_pair_representations_are_rechecked() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["x", "y"], &generators, 5);
    let original = hand_certificate(&generators, 5);
    let mut changed = original.clone();
    changed.groebner.s_pairs.clear();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.groebner.s_pairs[0].right = 0;
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    // This is a valid identity 0=y*x-x*y, but its representation is not
    // strictly below xy. Treating it as Schreyer reduction erases all rows.
    changed.groebner.s_pairs[0].quotients = polys(&["y", "-x"], &generators, 5)
        .iter()
        .map(|value| value.terms().clone())
        .collect();
    changed.syzygies.clear();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.groebner.basis_in_input[0][0].clear();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.groebner.input_in_basis[1] = changed.groebner.input_in_basis[0].clone();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.groebner.basis[0] = p("x**2", &generators, 5).terms().clone();
    reject(&generators, 5, &inputs, &changed);
    reject(
        &generators,
        5,
        &polys(&["y", "x"], &generators, 5),
        &original,
    );
}

#[test]
fn zero_input_relations_require_unit_coordinates_and_not_arbitrary_multiples() {
    let generators = ring(&["x"]);
    let inputs = polys(&["0", "0"], &generators, 5);
    let mut changed = certificate(&generators, 5, &inputs);
    changed.syzygies[0] = polys(&["x", "0"], &generators, 5);
    reject(&generators, 5, &inputs, &changed);
    let mut changed = certificate(&generators, 5, &inputs);
    changed.syzygies.pop();
    reject(&generators, 5, &inputs, &changed);
    let mut empty = certificate(&generators, 5, &[]);
    empty.syzygies.push(vec![]);
    reject(&generators, 5, &[], &empty);
}

#[test]
fn polynomial_relations_are_not_pointwise_functions_on_the_prime_field() {
    let generators = ring(&["x"]);
    let inputs = polys(&["x**2-x", "0"], &generators, 2);
    let result = run(&generators, 2, &inputs, TermOrder::Lex);
    assert_eq!(result.syzygies(), &[polys(&["0", "1"], &generators, 2)]);
    let mut changed = certificate(&generators, 2, &inputs);
    changed.syzygies[0] = polys(&["1", "0"], &generators, 2);
    reject(&generators, 2, &inputs, &changed);
}

#[test]
fn typed_ordered_coordinates_and_prime_are_bound_even_for_zero_output_cells() {
    let first = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([17; 32]));
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([18; 32]));
    let generators = [first.clone(), second.clone()];
    let inputs = [
        Polynomial::new(&generators, 5, &Terms::from([(vec![1, 0], 1)])).unwrap(),
        Polynomial::new(&generators, 5, &Terms::from([(vec![0, 1], 1)])).unwrap(),
        Polynomial::new(&generators, 5, &Terms::new()).unwrap(),
    ];
    let result = run(&generators, 5, &inputs, TermOrder::Lex);
    assert_eq!(result.syzygies().len(), 2);
    assert_zero_relations(&result);
    let evidence = certificate(&generators, 5, &inputs);
    reject(&[second.clone(), first.clone()], 5, &inputs, &evidence);
    reject(&generators, 7, &inputs, &evidence);
    for (output_generators, prime) in [
        (vec![first.clone(), second.clone()], 7),
        (vec![second, first], 5),
        (ring(&["x", "y"]), 5),
    ] {
        let mut changed = evidence.clone();
        changed.syzygies[0][0] = Polynomial::new(&output_generators, prime, &Terms::new()).unwrap();
        reject(&generators, 5, &inputs, &changed);
    }
}

#[test]
fn changing_order_requires_a_basis_and_strict_witnesses_valid_for_that_order() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["x+y**2", "x*y+1"], &generators, 5);
    let evidence = certificate(&generators, 5, &inputs);
    assert!(
        verify(
            &generators,
            5,
            &inputs,
            TermOrder::DegRevLex,
            &evidence,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn malformed_sparse_witnesses_and_dimensions_are_rejected_before_composition() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["x", "y"], &generators, 5);
    let original = hand_certificate(&generators, 5);
    for payload in [
        Terms::from([(vec![0], 1)]),
        Terms::from([(vec![0, 0], 0)]),
        Terms::from([(vec![0, 0], 5)]),
        Terms::from([(vec![65, 0], 1)]),
    ] {
        for location in 0..4 {
            let mut changed = original.clone();
            let target = match location {
                0 => &mut changed.groebner.basis[0],
                1 => &mut changed.groebner.basis_in_input[0][0],
                2 => &mut changed.groebner.input_in_basis[0][0],
                _ => &mut changed.groebner.s_pairs[0].quotients[0],
            };
            *target = payload.clone();
            reject(&generators, 5, &inputs, &changed);
        }
    }
    let mut changed = original.clone();
    changed.syzygies[0].pop();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.groebner.basis_in_input[0].push(Terms::new());
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original.clone();
    changed.groebner.input_in_basis[0].clear();
    reject(&generators, 5, &inputs, &changed);
    let mut changed = original;
    changed.groebner.s_pairs[0].left = usize::MAX;
    reject(&generators, 5, &inputs, &changed);
}

#[test]
fn source_and_ring_bounds_precede_empty_or_unit_shortcuts() {
    let generators = ring(&["x", "y"]);
    for first in ["0", "1"] {
        for invalid in [p("0", &generators, 7), p("0", &ring(&["y", "x"]), 5)] {
            assert!(
                checked(
                    &generators,
                    5,
                    &[p(first, &generators, 5), invalid],
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut Unbounded
                )
                .is_err()
            );
        }
    }
    for prime in [0, 1, 4, 65537] {
        assert!(
            checked(
                &generators,
                prime,
                &[],
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    for invalid in [
        vec![Symbol::new("x".repeat(65537))],
        vec![Symbol::new("x"); 2],
        (0..9).map(|i| Symbol::new(format!("x{i}"))).collect(),
    ] {
        assert!(
            checked(
                &invalid,
                5,
                &[],
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked(
            &generators,
            5,
            &vec![p("0", &generators, 5); 33],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn eight_variables_and_thirty_two_zero_inputs_keep_the_full_free_module() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let zero = Polynomial::new(&generators, 5, &Terms::new()).unwrap();
    let one = Polynomial::new(&generators, 5, &Terms::from([(vec![0; 8], 1)])).unwrap();
    let inputs = vec![zero.clone(); 32];
    let result = run(&generators, 5, &inputs, TermOrder::Lex);
    assert_eq!(result.syzygies().len(), 32);
    for (i, row) in result.syzygies().iter().enumerate() {
        assert_eq!(row.len(), 32);
        for (j, cell) in row.iter().enumerate() {
            assert_eq!(cell, if i == j { &one } else { &zero });
        }
    }
}

#[test]
fn degree_sixty_four_relations_allow_degree_one_twenty_eight_identity_products() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 65521] {
        let inputs = polys(&["x**64", "y**64"], &generators, prime);
        let result = run(&generators, prime, &inputs, TermOrder::Lex);
        assert_eq!(
            result.syzygies(),
            &[polys(&["y**64", "-x**64"], &generators, prime)]
        );
        assert_zero_relations(&result);
    }
}

#[test]
fn output_cell_cap_is_checked_before_verifier_owned_copies() {
    let zero = Polynomial::new(&[], 5, &Terms::new()).unwrap();
    let inputs = vec![zero.clone(); 32];
    let mut pairs = Vec::new();
    for left in 0..32 {
        for right in left + 1..32 {
            pairs.push(PrimeFieldGroebnerSPairCertificate {
                left,
                right,
                quotients: vec![Terms::new(); 32],
            });
        }
    }
    // Structurally admitted lower shapes permit 528 potential rows. The
    // stricter output cap refuses 513*32 cells before any lower proof copies.
    let evidence = Certificate {
        groebner: PrimeFieldGroebnerCertificate {
            basis: vec![Terms::new(); 32],
            basis_in_input: vec![vec![Terms::new(); 32]; 32],
            input_in_basis: vec![vec![Terms::new(); 32]; 32],
            s_pairs: pairs,
        },
        syzygies: vec![vec![zero; 32]; 513],
    };
    let mut meter = Meter::default();
    let error = verify(&[], 5, &inputs, TermOrder::Lex, &evidence, &mut meter).unwrap_err();
    assert!(error.to_string().contains("output dimensions"));
    assert_eq!(meter.memory, 0);
}

#[derive(Default)]
struct Meter {
    compute: u64,
    memory: u64,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
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
fn whole_operation_budgets_do_not_restart_at_basis_or_composition_boundaries() {
    let generators = ring(&["x", "y"]);
    for inputs in [
        vec![],
        polys(&["0", "0"], &generators, 5),
        polys(&["x**2+y", "x*y"], &generators, 5),
    ] {
        let (mut generation, mut verification) = (Meter::default(), Meter::default());
        checked(
            &generators,
            5,
            &inputs,
            TermOrder::Lex,
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
        ] {
            assert!(generated > 0 && verified > 0);
            checked(
                &generators,
                5,
                &inputs,
                TermOrder::Lex,
                &mut cap(dimension, generated),
                &mut cap(dimension, verified),
            )
            .unwrap();
            for generating in [true, false] {
                let amount = if generating { generated } else { verified };
                for precharged in [false, true] {
                    let mut meter = cap(dimension, amount - u64::from(!precharged));
                    if precharged {
                        meter.charge(dimension, 1).unwrap();
                    }
                    let result = if generating {
                        checked(
                            &generators,
                            5,
                            &inputs,
                            TermOrder::Lex,
                            &mut meter,
                            &mut Unbounded,
                        )
                    } else {
                        checked(
                            &generators,
                            5,
                            &inputs,
                            TermOrder::Lex,
                            &mut Unbounded,
                            &mut meter,
                        )
                    };
                    assert!(result.is_err());
                }
            }
        }
    }
}

#[test]
fn cancellation_in_either_phase_or_final_publication_never_returns_a_partial_module() {
    let generators = ring(&["x", "y"]);
    for inputs in [
        vec![],
        polys(&["0", "0"], &generators, 5),
        polys(&["x**2+y", "x*y"], &generators, 5),
    ] {
        let (mut generation, mut verification) = (Meter::default(), Meter::default());
        checked(
            &generators,
            5,
            &inputs,
            TermOrder::Lex,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        for (generating, checkpoints) in [
            (true, generation.checkpoints),
            (false, verification.checkpoints),
        ] {
            for at in [1, checkpoints / 2, checkpoints] {
                let mut stopped = Meter {
                    cancel_at: Some(at),
                    ..Meter::default()
                };
                let result = if generating {
                    checked(
                        &generators,
                        5,
                        &inputs,
                        TermOrder::Lex,
                        &mut stopped,
                        &mut Unbounded,
                    )
                } else {
                    checked(
                        &generators,
                        5,
                        &inputs,
                        TermOrder::Lex,
                        &mut Unbounded,
                        &mut stopped,
                    )
                };
                assert!(result.is_err(), "generating={generating}, checkpoint={at}");
            }
        }
        let evidence = certificate(&generators, 5, &inputs);
        let mut replay = Meter::default();
        verify(
            &generators,
            5,
            &inputs,
            TermOrder::Lex,
            &evidence,
            &mut replay,
        )
        .unwrap();
        let mut stopped = Meter {
            cancel_at: Some(replay.checkpoints),
            ..Meter::default()
        };
        assert!(
            verify(
                &generators,
                5,
                &inputs,
                TermOrder::Lex,
                &evidence,
                &mut stopped
            )
            .is_err()
        );
        assert!(
            verify(
                &generators,
                5,
                &inputs,
                TermOrder::Lex,
                &evidence,
                &mut cap(Dimension::MemoryBytes, replay.memory - 1)
            )
            .is_err()
        );
    }
}
