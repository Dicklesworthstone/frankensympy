//! Complete polynomial-relation generators, including independent hand-written
//! evidence and adversaries that remain valid relations but omit completeness.

use std::collections::BTreeMap;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    CompleteGroebnerBasisCertificate, GroebnerBasisCertificate, GroebnerSPairCertificate,
    MultivariatePoly, PolynomialSyzygyCertificate, TermOrder, VerifiedPolynomialSyzygies,
    bounded_polynomial_from_expr, bounded_syzygy_certificate, checked_groebner_basis,
    checked_polynomial_syzygies, verify_polynomial_syzygies,
};

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn polynomial(source: &str, generators: &[Symbol]) -> MultivariatePoly {
    bounded_polynomial_from_expr(&parse(source).unwrap(), generators, &mut Unbounded).unwrap()
}

fn polynomials(sources: &[&str], generators: &[Symbol]) -> Vec<MultivariatePoly> {
    sources
        .iter()
        .map(|source| polynomial(source, generators))
        .collect()
}

fn syzygies(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    order: TermOrder,
) -> VerifiedPolynomialSyzygies {
    checked_polynomial_syzygies(inputs, generators, order, &mut Unbounded, &mut Unbounded).unwrap()
}

fn certificate(inputs: &[MultivariatePoly], generators: &[Symbol]) -> PolynomialSyzygyCertificate {
    bounded_syzygy_certificate(inputs, generators, TermOrder::Lex, &mut Unbounded).unwrap()
}

fn assert_zero_relations(result: &VerifiedPolynomialSyzygies) {
    for row in result.syzygies() {
        assert_eq!(row.len(), result.inputs().len());
        assert!(row.iter().any(|p| !p.is_zero()));
        let mut sum = MultivariatePoly::zero(result.generators().to_vec());
        for (coefficient, input) in row.iter().zip(result.inputs()) {
            sum = sum.add(&coefficient.mul(input).unwrap()).unwrap();
        }
        assert!(sum.is_zero(), "a published row is not an exact relation");
    }
}

#[test]
fn all_strict_pairs_generate_complete_relations_for_a_monomial_ideal() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "x*y", "y**2"], &generators);
    let expected = vec![
        polynomials(&["y", "-x", "0"], &generators),
        polynomials(&["y**2", "0", "-x**2"], &generators),
        polynomials(&["0", "y", "-x"], &generators),
    ];
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let result = syzygies(&inputs, &generators, order);
        assert_eq!(result.syzygies(), expected);
        assert_eq!(result.inputs(), inputs);
        assert_eq!(result.generators(), generators);
        assert_eq!(result.order(), order);
        assert_eq!(result.basis().input_basis(), inputs);
        assert_zero_relations(&result);
        assert_eq!(result.into_syzygies(), expected);
    }
}

#[test]
fn original_nonbasis_coordinates_and_nonzero_spair_reductions_are_retained() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2+y", "x*y"], &generators);
    let result = syzygies(&inputs, &generators, TermOrder::Lex);
    assert_eq!(result.basis().basis().len(), 3);
    assert_eq!(
        result.syzygies(),
        &[
            polynomials(&["-x**2*y", "x**3+x*y"], &generators),
            polynomials(&["-x*y", "x**2+y"], &generators),
        ]
    );
    assert_zero_relations(&result);
    // The primitive Koszul relation is present; the other row is its x multiple.
    let primitive = polynomials(&["-x*y", "x**2+y"], &generators);
    assert!(result.syzygies().contains(&primitive));
}

#[test]
fn nontrivial_change_of_generators_keeps_a_complete_nonminimal_set() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x*y", "x*(y+1)"], &generators);
    let result = syzygies(&inputs, &generators, TermOrder::Lex);
    assert_eq!(result.basis().basis(), polynomials(&["x"], &generators));
    let primitive = polynomials(&["y+1", "-y"], &generators);
    assert_eq!(result.syzygies(), &[primitive.clone(), primitive]);
    assert_zero_relations(&result);
}

#[test]
fn rational_scaling_is_bound_to_the_original_polynomials() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["2*x", "3*y"], &generators);
    let result = syzygies(&inputs, &generators, TermOrder::Lex);
    assert_eq!(
        result.syzygies(),
        &[polynomials(&["y/2", "-x/3"], &generators)]
    );
    assert_zero_relations(&result);
    let evidence = certificate(&inputs, &generators);
    assert!(
        verify_polynomial_syzygies(
            &polynomials(&["x", "y"], &generators),
            &generators,
            TermOrder::Lex,
            &evidence,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn zero_redundant_and_repeated_inputs_keep_their_coordinate_positions() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x", "y", "x+y", "0"], &generators);
    let result = syzygies(&inputs, &generators, TermOrder::Lex);
    assert_eq!(
        result.syzygies(),
        &[
            polynomials(&["-1", "-1", "1", "0"], &generators),
            polynomials(&["0", "0", "0", "1"], &generators),
            polynomials(&["y", "-x", "0", "0"], &generators),
        ]
    );
    assert_zero_relations(&result);
    let inputs = polynomials(&["0", "x", "0", "2*x"], &generators);
    let result = syzygies(&inputs, &generators, TermOrder::Lex);
    assert_eq!(
        result.syzygies(),
        &[
            polynomials(&["1", "0", "0", "0"], &generators),
            polynomials(&["0", "0", "1", "0"], &generators),
            polynomials(&["0", "-2", "0", "1"], &generators),
        ]
    );
    assert_zero_relations(&result);
}

#[test]
fn empty_zero_unit_and_constant_rings_have_complete_relation_spaces() {
    for generators in [Vec::new(), ring(&["x", "y"])] {
        let empty = syzygies(&[], &generators, TermOrder::DegRevLex);
        assert_eq!(empty.generators(), generators);
        assert!(empty.inputs().is_empty() && empty.syzygies().is_empty());
        let zero = polynomials(&["0", "0"], &generators);
        assert_eq!(
            syzygies(&zero, &generators, TermOrder::Lex).syzygies(),
            &[
                polynomials(&["1", "0"], &generators),
                polynomials(&["0", "1"], &generators),
            ]
        );
        let constants = polynomials(&["0", "2/3", "7/11"], &generators);
        let result = syzygies(&constants, &generators, TermOrder::Lex);
        assert_eq!(
            result.syzygies(),
            &[
                polynomials(&["1", "0", "0"], &generators),
                polynomials(&["0", "-21/22", "1"], &generators),
            ]
        );
        assert_zero_relations(&result);
        assert!(
            syzygies(
                &polynomials(&["2/3"], &generators),
                &generators,
                TermOrder::Lex
            )
            .syzygies()
            .is_empty()
        );
    }
    let generators = ring(&["x"]);
    let result = syzygies(
        &polynomials(&["x", "1"], &generators),
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(result.syzygies(), &[polynomials(&["1", "-x"], &generators)]);
    assert_zero_relations(&result);
}

#[test]
fn differing_term_orders_still_span_the_hand_known_primitive_relation() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x+y**2", "x*y+1"], &generators);
    // These two coprime polynomials have the primitive relation (g,-f).
    // Independently establish that the output multiples generate its whole
    // coefficient ideal, rather than merely checking each output is a relation.
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let result = syzygies(&inputs, &generators, order);
        assert_zero_relations(&result);
        let mut multiples = Vec::new();
        for row in result.syzygies() {
            let (quotients, remainder) = row[0].div_rem(&inputs[1..], TermOrder::Lex).unwrap();
            assert!(remainder.is_zero());
            let negative = MultivariatePoly::zero(generators.clone())
                .sub(&quotients[0].mul(&inputs[0]).unwrap())
                .unwrap();
            assert_eq!(row[1], negative);
            multiples.push(quotients[0].clone());
        }
        let generated =
            checked_groebner_basis(&multiples, TermOrder::Lex, &mut Unbounded, &mut Unbounded)
                .unwrap();
        assert_eq!(generated.basis(), polynomials(&["1"], &generators));
    }
}

fn hand_certificate(generators: &[Symbol]) -> PolynomialSyzygyCertificate {
    PolynomialSyzygyCertificate {
        groebner: CompleteGroebnerBasisCertificate {
            candidate: GroebnerBasisCertificate {
                order: TermOrder::Lex,
                basis: polynomials(&["x", "y"], generators),
                input_ideal_witnesses: vec![
                    polynomials(&["1", "0"], generators),
                    polynomials(&["0", "1"], generators),
                ],
            },
            input_basis_witnesses: vec![
                polynomials(&["1", "0"], generators),
                polynomials(&["0", "1"], generators),
            ],
            s_pairs: vec![GroebnerSPairCertificate {
                left: 0,
                right: 1,
                basis_coefficients: polynomials(&["0", "0"], generators),
            }],
        },
        syzygies: vec![polynomials(&["y", "-x"], generators)],
    }
}

#[test]
fn independent_verifier_accepts_hand_written_complete_evidence() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x", "y"], &generators);
    let result = verify_polynomial_syzygies(
        &inputs,
        &generators,
        TermOrder::Lex,
        &hand_certificate(&generators),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(result.syzygies(), &[polynomials(&["y", "-x"], &generators)]);
    assert_zero_relations(&result);
}

#[test]
fn valid_but_incomplete_reordered_duplicate_or_zero_rows_are_rejected() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "x*y", "y**2"], &generators);
    let original = certificate(&inputs, &generators);
    let rejected = |evidence: &PolynomialSyzygyCertificate| {
        assert!(
            verify_polynomial_syzygies(
                &inputs,
                &generators,
                TermOrder::Lex,
                evidence,
                &mut Unbounded
            )
            .is_err()
        );
    };
    let mut altered = original.clone();
    altered.syzygies.pop();
    rejected(&altered);
    let mut altered = original.clone();
    altered.syzygies.clear();
    rejected(&altered);
    let mut altered = original.clone();
    altered.syzygies.swap(0, 2);
    rejected(&altered);
    let mut altered = original.clone();
    altered.syzygies[1] = altered.syzygies[0].clone();
    rejected(&altered);
    let mut altered = original.clone();
    altered
        .syzygies
        .push(polynomials(&["0", "0", "0"], &generators));
    rejected(&altered);
    let mut altered = original;
    altered.syzygies[0][0] =
        polynomial("y+1/10000000000000000000000000000000000000000", &generators);
    rejected(&altered);
}

#[test]
fn underlying_basis_cofactors_and_strict_spair_completeness_are_rechecked() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x", "y"], &generators);
    let original = hand_certificate(&generators);
    let rejected = |evidence: &PolynomialSyzygyCertificate| {
        assert!(
            verify_polynomial_syzygies(
                &inputs,
                &generators,
                TermOrder::Lex,
                evidence,
                &mut Unbounded
            )
            .is_err()
        );
    };
    let mut altered = original.clone();
    altered.groebner.s_pairs.clear();
    rejected(&altered);
    let mut altered = original.clone();
    altered.groebner.s_pairs[0].right = 0;
    rejected(&altered);
    let mut altered = original.clone();
    // This is a true identity 0=y*x-x*y, but is not a strict standard
    // representation. Treating it as a Schreyer reduction would erase all
    // relations and falsely certify the empty generating set.
    altered.groebner.s_pairs[0].basis_coefficients = polynomials(&["y", "-x"], &generators);
    altered.syzygies.clear();
    rejected(&altered);
    let mut altered = original.clone();
    altered.groebner.candidate.input_ideal_witnesses[0] = polynomials(&["0", "0"], &generators);
    rejected(&altered);
    let mut altered = original.clone();
    altered.groebner.input_basis_witnesses[1] = polynomials(&["1", "0"], &generators);
    rejected(&altered);
    let mut altered = original.clone();
    altered.groebner.candidate.basis[0] = polynomial("x**2", &generators);
    rejected(&altered);
    assert!(
        verify_polynomial_syzygies(
            &inputs,
            &generators,
            TermOrder::DegLex,
            &original,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_polynomial_syzygies(
            &polynomials(&["y", "x"], &generators),
            &generators,
            TermOrder::Lex,
            &original,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn zero_module_certificates_cannot_replace_required_coordinate_generators() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["0", "0"], &generators);
    let mut evidence = certificate(&inputs, &generators);
    evidence.syzygies[0] = polynomials(&["x", "0"], &generators);
    // Every vector annihilates zero inputs, but x*e_0 does not span e_0.
    assert!(
        verify_polynomial_syzygies(
            &inputs,
            &generators,
            TermOrder::Lex,
            &evidence,
            &mut Unbounded
        )
        .is_err()
    );
    let mut empty = certificate(&[], &generators);
    empty.syzygies.push(Vec::new());
    assert!(
        verify_polynomial_syzygies(&[], &generators, TermOrder::Lex, &empty, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn same_print_symbols_remain_distinct_typed_ring_coordinates() {
    let first = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([17; 32]));
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([18; 32]));
    let generators = vec![first.clone(), second.clone()];
    let inputs = vec![
        MultivariatePoly::var(generators.clone(), &first).unwrap(),
        MultivariatePoly::var(generators.clone(), &second).unwrap(),
    ];
    let result = syzygies(&inputs, &generators, TermOrder::Lex);
    let negative_first = MultivariatePoly::zero(generators.clone())
        .sub(&inputs[0])
        .unwrap();
    assert_eq!(
        result.syzygies(),
        &[vec![inputs[1].clone(), negative_first]]
    );
    assert_zero_relations(&result);
    let evidence = certificate(&inputs, &generators);
    assert!(
        verify_polynomial_syzygies(
            &inputs,
            &[second, first],
            TermOrder::Lex,
            &evidence,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn malformed_and_oversized_sources_cannot_hide_behind_zero_or_unit_inputs() {
    let generators = ring(&["x", "y"]);
    let mut bad_width = polynomial("x", &generators);
    bad_width.terms.insert(vec![1], 1.into());
    let mut bad_coefficient = polynomial("x", &generators);
    bad_coefficient.terms.insert(vec![0, 0], 0.into());
    let degree = MultivariatePoly {
        generators: generators.clone(),
        terms: BTreeMap::from([(vec![257, 0], 1.into())]),
    };
    let height = MultivariatePoly {
        generators: generators.clone(),
        terms: BTreeMap::from([(
            vec![0, 0],
            BigRational::from_integer(BigInt::from(1_u32) << 4097_u32),
        )]),
    };
    let terms = MultivariatePoly {
        generators: generators.clone(),
        terms: (0..4097)
            .map(|i| (vec![i / 65, i % 65], 1.into()))
            .collect(),
    };
    for invalid in [bad_width, bad_coefficient, degree, height, terms] {
        for first in ["0", "1"] {
            assert!(
                checked_polynomial_syzygies(
                    &[polynomial(first, &generators), invalid.clone()],
                    &generators,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut Unbounded
                )
                .is_err()
            );
        }
    }
    assert!(
        checked_polynomial_syzygies(
            &[],
            &[Symbol::new("x".repeat(65_537))],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        checked_polynomial_syzygies(
            &[],
            &vec![Symbol::new("x"); 2],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        checked_polynomial_syzygies(
            &[],
            &(0..9)
                .map(|i| Symbol::new(format!("x{i}")))
                .collect::<Vec<_>>(),
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        checked_polynomial_syzygies(
            &vec![polynomial("0", &generators); 65],
            &generators,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn malformed_certificate_payloads_are_refused_before_composition() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x", "y"], &generators);
    let original = hand_certificate(&generators);
    for location in 0..4 {
        let mut altered = original.clone();
        let polynomial = match location {
            0 => &mut altered.syzygies[0][0],
            1 => &mut altered.groebner.candidate.input_ideal_witnesses[0][0],
            2 => &mut altered.groebner.input_basis_witnesses[0][0],
            _ => &mut altered.groebner.s_pairs[0].basis_coefficients[0],
        };
        polynomial.terms.insert(vec![0], 1.into());
        assert!(
            verify_polynomial_syzygies(
                &inputs,
                &generators,
                TermOrder::Lex,
                &altered,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let mut altered = original.clone();
    altered.syzygies[0].pop();
    assert!(
        verify_polynomial_syzygies(
            &inputs,
            &generators,
            TermOrder::Lex,
            &altered,
            &mut Unbounded
        )
        .is_err()
    );
    let mut altered = original;
    altered.groebner.candidate.input_ideal_witnesses[0].push(polynomial("0", &generators));
    assert!(
        verify_polynomial_syzygies(
            &inputs,
            &generators,
            TermOrder::Lex,
            &altered,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    compute: u64,
    memory: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.compute += amount;
        } else if dimension == Dimension::MemoryBytes {
            self.memory += amount;
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
fn cancellation_in_both_phases_and_final_publication_returns_no_accepted_prefix() {
    let generators = ring(&["x", "y"]);
    for inputs in [
        Vec::new(),
        polynomials(&["0", "0"], &generators),
        polynomials(&["x**2+y", "x*y"], &generators),
    ] {
        let mut generation = Meter::default();
        let mut verification = Meter::default();
        checked_polynomial_syzygies(
            &inputs,
            &generators,
            TermOrder::Lex,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        assert!(generation.compute > 0 && verification.compute > 0);
        assert!(generation.memory > 0 && verification.memory > 0);
        for cancel_at in [1, generation.checkpoints / 2, generation.checkpoints] {
            let mut meter = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            assert!(
                checked_polynomial_syzygies(
                    &inputs,
                    &generators,
                    TermOrder::Lex,
                    &mut meter,
                    &mut Unbounded
                )
                .is_err()
            );
        }
        for cancel_at in [1, verification.checkpoints / 2, verification.checkpoints] {
            let mut meter = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            assert!(
                checked_polynomial_syzygies(
                    &inputs,
                    &generators,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut meter
                )
                .is_err()
            );
        }
    }
}

fn budget(dimension: Dimension, cap: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[dimension.index()] = cap;
    Budget::new(limits)
}

#[test]
fn resource_budgets_are_cumulative_across_basis_composition_and_output_copy() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2+y", "x*y"], &generators);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    checked_polynomial_syzygies(
        &inputs,
        &generators,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for (dimension, generator_cap, verifier_cap) in [
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
        assert!(
            checked_polynomial_syzygies(
                &inputs,
                &generators,
                TermOrder::Lex,
                &mut budget(dimension, generator_cap - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked_polynomial_syzygies(
                &inputs,
                &generators,
                TermOrder::Lex,
                &mut Unbounded,
                &mut budget(dimension, verifier_cap - 1)
            )
            .is_err()
        );
        let mut first = budget(dimension, generator_cap);
        let mut second = budget(dimension, verifier_cap);
        checked_polynomial_syzygies(
            &inputs,
            &generators,
            TermOrder::Lex,
            &mut first,
            &mut second,
        )
        .unwrap();
        assert!(
            checked_polynomial_syzygies(
                &inputs,
                &generators,
                TermOrder::Lex,
                &mut first,
                &mut second
            )
            .is_err()
        );
    }
    let mut unused_verifier = Meter::default();
    assert!(
        checked_polynomial_syzygies(
            &inputs,
            &generators,
            TermOrder::Lex,
            &mut budget(Dimension::ComputeSteps, 1),
            &mut unused_verifier
        )
        .is_err()
    );
    assert_eq!(unused_verifier.checkpoints, 0);
}
