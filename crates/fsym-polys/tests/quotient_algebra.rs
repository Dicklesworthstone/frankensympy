//! Exact finite quotient operators and adversarial completeness evidence.

use std::collections::BTreeMap;
use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly, QuotientMultiplicationCertificate, TermOrder, VerifiedQuotientMultiplication,
    bounded_polynomial_from_expr, checked_groebner_basis, checked_quotient_multiplication,
    verify_quotient_multiplication,
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

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn integer_matrix(rows: &[&[i64]]) -> Vec<Vec<BigRational>> {
    rows.iter()
        .map(|row| row.iter().map(|&value| rational(value, 1)).collect())
        .collect()
}

fn operator(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    order: TermOrder,
) -> VerifiedQuotientMultiplication {
    checked_quotient_multiplication(
        inputs,
        element,
        generators,
        order,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn reject(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    certificate: &QuotientMultiplicationCertificate,
) {
    assert!(
        verify_quotient_multiplication(
            inputs,
            element,
            generators,
            TermOrder::Lex,
            certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn quadratic_field_matrix_has_column_action_and_exact_request_accessors() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["x**2-2"], &generators);
    let element = polynomial("x", &generators);
    let result = operator(&inputs, &element, &generators, TermOrder::Lex);
    assert_eq!(result.generators(), generators);
    assert_eq!(result.inputs(), inputs);
    assert_eq!(result.element(), &element);
    assert_eq!(result.order(), TermOrder::Lex);
    assert_eq!(result.standard_monomials(), &[vec![0], vec![1]]);
    assert_eq!(result.matrix(), integer_matrix(&[&[0, 2], &[1, 0]]));
    let replay = verify_quotient_multiplication(
        &inputs,
        &element,
        &generators,
        TermOrder::Lex,
        result.certificate(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(replay.matrix(), result.matrix());
}

#[test]
fn arbitrary_multivariate_elements_and_rational_coefficients_are_exact() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2-2", "y**2-3"], &generators);
    let result = operator(
        &inputs,
        &polynomial("x+y", &generators),
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(
        result.standard_monomials(),
        &[vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );
    assert_eq!(
        result.matrix(),
        integer_matrix(&[&[0, 3, 2, 0], &[1, 0, 0, 2], &[1, 0, 0, 3], &[0, 1, 1, 0]])
    );
    let result = operator(
        &inputs,
        &polynomial("x*y+x/2-y/3", &generators),
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(
        result.matrix(),
        &[
            vec![
                rational(0, 1),
                rational(-1, 1),
                rational(1, 1),
                rational(6, 1)
            ],
            vec![
                rational(-1, 3),
                rational(0, 1),
                rational(2, 1),
                rational(1, 1)
            ],
            vec![
                rational(1, 2),
                rational(3, 1),
                rational(0, 1),
                rational(-1, 1)
            ],
            vec![
                rational(1, 1),
                rational(1, 2),
                rational(-1, 3),
                rational(0, 1)
            ],
        ]
    );
}

#[test]
fn nonradical_quotient_keeps_nilpotent_multiplicity() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "y**2"], &generators);
    let result = operator(
        &inputs,
        &polynomial("x+y", &generators),
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(
        result.standard_monomials(),
        &[vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );
    assert_eq!(
        result.matrix(),
        integer_matrix(&[&[0, 0, 0, 0], &[1, 0, 0, 0], &[1, 0, 0, 0], &[0, 1, 1, 0]])
    );
    // (x+y)^2 = 2xy remains nonzero even though all root values are zero.
    let square = operator(
        &inputs,
        &polynomial("(x+y)**2", &generators),
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(
        square.matrix(),
        integer_matrix(&[&[0, 0, 0, 0], &[0, 0, 0, 0], &[0, 0, 0, 0], &[2, 0, 0, 0]])
    );
    let cube = operator(
        &inputs,
        &polynomial("(x+y)**3", &generators),
        &generators,
        TermOrder::Lex,
    );
    assert!(
        cube.matrix()
            .iter()
            .flatten()
            .all(|value| *value == rational(0, 1))
    );
}

#[test]
fn nonbasis_inputs_are_completed_in_each_order_without_changing_sources() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2+y", "x*y+1"], &generators);
    let element = polynomial("x", &generators);
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let result = operator(&inputs, &element, &generators, order);
        let expected = if order == TermOrder::Lex {
            vec![vec![0, 0], vec![0, 1], vec![0, 2]]
        } else {
            vec![vec![0, 0], vec![0, 1], vec![1, 0]]
        };
        assert_eq!(result.standard_monomials(), expected);
        assert_eq!(result.inputs(), inputs);
        assert_eq!(result.order(), order);
        assert_eq!(
            result.matrix(),
            integer_matrix(&[&[0, -1, 0], &[0, 0, -1], &[1, 0, 0]])
        );
    }
}

#[test]
fn zero_scalars_and_congruent_elements_have_the_correct_operators() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["x**2-2"], &generators);
    let zero = operator(
        &inputs,
        &polynomial("0", &generators),
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(zero.matrix(), integer_matrix(&[&[0, 0], &[0, 0]]));
    let scalar = operator(
        &inputs,
        &polynomial("7/11", &generators),
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(
        scalar.matrix(),
        &[
            vec![rational(7, 11), rational(0, 1)],
            vec![rational(0, 1), rational(7, 11)]
        ]
    );
    let first = operator(
        &inputs,
        &polynomial("x", &generators),
        &generators,
        TermOrder::Lex,
    );
    let shifted = polynomial("x+x**2-2", &generators);
    let second = operator(&inputs, &shifted, &generators, TermOrder::Lex);
    assert_eq!(first.matrix(), second.matrix());
    // Equal residue classes still require different exact-source witnesses.
    reject(&inputs, &shifted, &generators, first.certificate());
}

#[test]
fn zero_ring_and_constant_field_are_explicit_and_positive_dimension_refuses() {
    for generators in [Vec::new(), ring(&["x", "y"])] {
        let inputs = polynomials(&["0", "2/3"], &generators);
        let result = operator(
            &inputs,
            &polynomial("7/11", &generators),
            &generators,
            TermOrder::DegRevLex,
        );
        assert!(result.standard_monomials().is_empty());
        assert!(result.matrix().is_empty());
        assert!(result.certificate().reduction_quotients.is_empty());
    }
    for inputs in [Vec::new(), polynomials(&["0", "0"], &[])] {
        let result = operator(&inputs, &polynomial("7/11", &[]), &[], TermOrder::Lex);
        assert_eq!(result.standard_monomials(), &[Vec::<u32>::new()]);
        assert_eq!(result.matrix(), &[vec![rational(7, 11)]]);
    }
    let generators = ring(&["x", "y"]);
    for inputs in [
        Vec::new(),
        polynomials(&["0"], &generators),
        polynomials(&["x"], &generators),
    ] {
        for element in [polynomial("0", &generators), polynomial("3", &generators)] {
            assert!(
                checked_quotient_multiplication(
                    &inputs,
                    &element,
                    &generators,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut Unbounded
                )
                .is_err()
            );
        }
    }
}

#[test]
fn dimension_64_is_complete_and_dimension_65_is_not_truncated() {
    let generators = ring(&["x"]);
    let element = polynomial("x", &generators);
    let result = operator(
        &polynomials(&["x**64"], &generators),
        &element,
        &generators,
        TermOrder::Lex,
    );
    assert_eq!(result.standard_monomials().len(), 64);
    for (row, values) in result.matrix().iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            assert_eq!(*value, rational(i64::from(row == column + 1), 1));
        }
    }
    assert!(
        checked_quotient_multiplication(
            &polynomials(&["x**65"], &generators),
            &element,
            &generators,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

fn handwritten_quadratic(generators: &[Symbol]) -> QuotientMultiplicationCertificate {
    let inputs = polynomials(&["x**2-2"], generators);
    QuotientMultiplicationCertificate {
        source_basis: Arc::new(
            checked_groebner_basis(&inputs, TermOrder::Lex, &mut Unbounded, &mut Unbounded)
                .unwrap(),
        ),
        standard_monomials: vec![vec![0], vec![1]],
        reduction_quotients: vec![
            polynomials(&["0"], generators),
            polynomials(&["1"], generators),
        ],
        matrix: integer_matrix(&[&[0, 2], &[1, 0]]),
    }
}

#[test]
fn independent_handwritten_column_evidence_is_accepted() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["x**2-2"], &generators);
    let element = polynomial("x", &generators);
    let evidence = handwritten_quadratic(&generators);
    let result = verify_quotient_multiplication(
        &inputs,
        &element,
        &generators,
        TermOrder::Lex,
        &evidence,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(result.matrix(), integer_matrix(&[&[0, 2], &[1, 0]]));
}

#[test]
fn individually_valid_zero_columns_cannot_hide_an_incomplete_staircase() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["x**3"], &generators);
    let zero = polynomial("0", &generators);
    let result = operator(&inputs, &zero, &generators, TermOrder::Lex);
    let mut truncated = result.certificate().clone();
    truncated.standard_monomials.pop();
    truncated.reduction_quotients.pop();
    truncated.matrix.pop();
    for row in &mut truncated.matrix {
        row.pop();
    }
    // Every surviving 0*b = 0 identity is true. Closure must reject missing x^2.
    reject(&inputs, &zero, &generators, &truncated);
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x"], &generators);
    let evidence = QuotientMultiplicationCertificate {
        source_basis: Arc::new(
            checked_groebner_basis(&inputs, TermOrder::Lex, &mut Unbounded, &mut Unbounded)
                .unwrap(),
        ),
        standard_monomials: vec![vec![0, 0]],
        reduction_quotients: vec![polynomials(&["0"], &generators)],
        matrix: integer_matrix(&[&[0]]),
    };
    // Same attack on an infinite-dimensional quotient QQ[y].
    reject(
        &inputs,
        &polynomial("0", &generators),
        &generators,
        &evidence,
    );
}

#[test]
fn reordered_repeated_nonstandard_and_empty_staircases_are_rejected() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["x**2-2"], &generators);
    let element = polynomial("x", &generators);
    let original = handwritten_quadratic(&generators);
    for powers in [
        vec![vec![1], vec![0]],
        vec![vec![0], vec![0]],
        vec![vec![0], vec![2]],
        vec![vec![], vec![1]],
    ] {
        let mut altered = original.clone();
        altered.standard_monomials = powers;
        reject(&inputs, &element, &generators, &altered);
    }
    let mut altered = original.clone();
    altered.standard_monomials.swap(0, 1);
    altered.matrix.swap(0, 1);
    for row in &mut altered.matrix {
        row.swap(0, 1);
    }
    altered.reduction_quotients.swap(0, 1);
    reject(&inputs, &element, &generators, &altered);
    let unit = polynomials(&["1"], &generators);
    let mut altered = operator(&unit, &element, &generators, TermOrder::Lex)
        .certificate()
        .clone();
    altered.standard_monomials = vec![vec![0]];
    altered.matrix = integer_matrix(&[&[0]]);
    altered.reduction_quotients = vec![polynomials(&["x"], &generators)];
    reject(&unit, &element, &generators, &altered);
    let mut altered = operator(&[], &polynomial("0", &[]), &[], TermOrder::Lex)
        .certificate()
        .clone();
    altered.standard_monomials.clear();
    altered.matrix.clear();
    altered.reduction_quotients.clear();
    reject(&[], &polynomial("0", &[]), &[], &altered);
}

#[test]
fn every_exact_matrix_entry_and_every_column_witness_is_checked() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["x**2-2"], &generators);
    let element = polynomial("x", &generators);
    let original = handwritten_quadratic(&generators);
    for mutation in 0..10 {
        let mut altered = original.clone();
        match mutation {
            0 => altered.matrix = integer_matrix(&[&[0, 1], &[2, 0]]),
            1 => altered.matrix[0][1] = rational(2_000_001, 1_000_000),
            2 => altered.matrix = integer_matrix(&[&[0, 0], &[0, 0]]),
            3 => {
                altered.matrix[0].pop();
            }
            4 => {
                altered.reduction_quotients.pop();
            }
            5 => altered
                .reduction_quotients
                .push(polynomials(&["0"], &generators)),
            6 => {
                altered.reduction_quotients[0].clear();
            }
            7 => altered.reduction_quotients[0].push(polynomial("0", &generators)),
            8 => altered.reduction_quotients.swap(0, 1),
            _ => altered.reduction_quotients[1][0] = polynomial("2", &generators),
        }
        reject(&inputs, &element, &generators, &altered);
    }
}

#[test]
fn accepted_basis_receipts_bind_original_order_coefficients_and_term_order() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["0", "2*x**2-4", "x**2-2"], &generators);
    let element = polynomial("x", &generators);
    let result = operator(&inputs, &element, &generators, TermOrder::Lex);
    assert_eq!(result.inputs(), inputs);
    let mut reordered = inputs.clone();
    reordered.swap(0, 1);
    reject(&reordered, &element, &generators, result.certificate());
    let scaled = polynomials(&["0", "x**2-2", "x**2-2"], &generators);
    reject(&scaled, &element, &generators, result.certificate());
    let mut altered = result.certificate().clone();
    altered.source_basis = Arc::new(
        checked_groebner_basis(&scaled, TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap(),
    );
    reject(&inputs, &element, &generators, &altered);
    assert!(
        verify_quotient_multiplication(
            &inputs,
            &element,
            &generators,
            TermOrder::DegLex,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn same_print_symbols_retain_distinct_typed_coordinates() {
    let first = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([17; 32]));
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([18; 32]));
    let generators = vec![first.clone(), second.clone()];
    let inputs = vec![
        MultivariatePoly {
            generators: generators.clone(),
            terms: BTreeMap::from([(vec![2, 0], 1.into())]),
        },
        MultivariatePoly {
            generators: generators.clone(),
            terms: BTreeMap::from([(vec![0, 2], 1.into())]),
        },
    ];
    let element = MultivariatePoly::var(generators.clone(), &first).unwrap();
    let result = operator(&inputs, &element, &generators, TermOrder::Lex);
    assert_eq!(
        result.matrix(),
        integer_matrix(&[&[0, 0, 0, 0], &[0, 0, 0, 0], &[1, 0, 0, 0], &[0, 1, 0, 0]])
    );
    reject(&inputs, &element, &[second, first], result.certificate());
}

#[test]
fn malformed_sources_cannot_hide_behind_unit_or_zero_shortcuts() {
    let generators = ring(&["x", "y"]);
    let mut bad_width = polynomial("x", &generators);
    bad_width.terms.insert(vec![1], 1.into());
    let mut bad_zero = polynomial("x", &generators);
    bad_zero.terms.insert(vec![0, 0], 0.into());
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
    for invalid in [bad_width, bad_zero, degree, height, terms] {
        assert!(
            checked_quotient_multiplication(
                &polynomials(&["1"], &generators),
                &invalid,
                &generators,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked_quotient_multiplication(
                &[polynomial("1", &generators), invalid],
                &polynomial("0", &generators),
                &generators,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    for invalid_ring in [
        vec![Symbol::new("x"); 2],
        vec![Symbol::new("x".repeat(65_537))],
        (0..9).map(|i| Symbol::new(format!("x{i}"))).collect(),
    ] {
        assert!(
            checked_quotient_multiplication(
                &[],
                &MultivariatePoly::zero(invalid_ring.clone()),
                &invalid_ring,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked_quotient_multiplication(
            &vec![polynomial("1", &generators); 65],
            &polynomial("0", &generators),
            &generators,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn malformed_certificate_coefficients_shapes_and_degrees_are_refused() {
    let generators = ring(&["x"]);
    let inputs = polynomials(&["x**2-2"], &generators);
    let element = polynomial("x", &generators);
    let original = handwritten_quadratic(&generators);
    for mutation in 0..5 {
        let mut altered = original.clone();
        match mutation {
            0 => altered.matrix[0][0] = BigRational::from_integer(BigInt::from(1_u32) << 4097_u32),
            1 => altered.standard_monomials[1] = vec![257],
            2 => {
                altered.reduction_quotients[1][0]
                    .terms
                    .insert(vec![], 1.into());
            }
            3 => {
                altered.reduction_quotients[1][0]
                    .terms
                    .insert(vec![1], 0.into());
            }
            _ => altered.reduction_quotients[1][0].generators = ring(&["y"]),
        }
        reject(&inputs, &element, &generators, &altered);
    }
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
fn cancellation_in_each_phase_including_final_publication_yields_no_operator() {
    for (generators, sources, value) in [
        (ring(&["x"]), vec!["x**2-2"], "x"),
        (ring(&["x"]), vec!["1"], "x"),
        (Vec::new(), vec![], "7/11"),
    ] {
        let inputs = polynomials(&sources, &generators);
        let element = polynomial(value, &generators);
        let mut generation = Meter::default();
        let mut verification = Meter::default();
        checked_quotient_multiplication(
            &inputs,
            &element,
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
                checked_quotient_multiplication(
                    &inputs,
                    &element,
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
                checked_quotient_multiplication(
                    &inputs,
                    &element,
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
fn resource_budgets_include_basis_generation_every_column_and_publication() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2+y", "x*y+1"], &generators);
    let element = polynomial("x+y", &generators);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    checked_quotient_multiplication(
        &inputs,
        &element,
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
            checked_quotient_multiplication(
                &inputs,
                &element,
                &generators,
                TermOrder::Lex,
                &mut budget(dimension, generator_cap - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked_quotient_multiplication(
                &inputs,
                &element,
                &generators,
                TermOrder::Lex,
                &mut Unbounded,
                &mut budget(dimension, verifier_cap - 1)
            )
            .is_err()
        );
        let mut first = budget(dimension, generator_cap);
        let mut second = budget(dimension, verifier_cap);
        checked_quotient_multiplication(
            &inputs,
            &element,
            &generators,
            TermOrder::Lex,
            &mut first,
            &mut second,
        )
        .unwrap();
        assert!(
            checked_quotient_multiplication(
                &inputs,
                &element,
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
        checked_quotient_multiplication(
            &inputs,
            &element,
            &generators,
            TermOrder::Lex,
            &mut budget(Dimension::ComputeSteps, 1),
            &mut unused_verifier
        )
        .is_err()
    );
    assert_eq!(unused_verifier.checkpoints, 0);
}
