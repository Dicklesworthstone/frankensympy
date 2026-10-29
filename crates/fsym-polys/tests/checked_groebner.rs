//! Hand-computed ideals and adversaries for bounded complete basis generation.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol};
use fsym_polys::{
    MultivariatePoly, TermOrder, bounded_groebner_certificate, checked_groebner_basis,
    verify_complete_groebner_certificate,
};
use num_traits::One;
use std::collections::BTreeMap;

fn poly(generators: &[Symbol], terms: &[(Vec<u32>, i64, i64)]) -> MultivariatePoly {
    MultivariatePoly::new(
        generators.to_vec(),
        terms
            .iter()
            .map(|(powers, n, d)| (powers.clone(), BigRational::new((*n).into(), (*d).into())))
            .collect(),
    )
    .unwrap()
}

fn coupled() -> (Vec<Symbol>, Vec<MultivariatePoly>) {
    let variables = vec![Symbol::new("x"), Symbol::new("y")];
    let input = vec![
        poly(&variables, &[(vec![2, 0], 1, 1), (vec![0, 1], -1, 1)]),
        poly(&variables, &[(vec![0, 2], 1, 1), (vec![1, 0], -1, 1)]),
    ];
    (variables, input)
}

#[test]
fn coupled_ideal_has_the_hand_derived_basis_in_all_orders() {
    let (variables, input) = coupled();
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let result = checked_groebner_basis(&input, order, &mut Unbounded, &mut Unbounded).unwrap();
        let expected = if order == TermOrder::Lex {
            vec![
                poly(&variables, &[(vec![1, 0], 1, 1), (vec![0, 2], -1, 1)]),
                poly(&variables, &[(vec![0, 4], 1, 1), (vec![0, 1], -1, 1)]),
            ]
        } else {
            input.clone()
        };
        assert_eq!(result.basis(), expected);
        assert_eq!(result.input_basis(), input);
        assert_eq!(result.order(), order);
        assert_eq!(result.receipt().s_pairs_verified(), 1);
        assert_eq!(result.standard_monomials(&mut Unbounded).unwrap().len(), 4);
    }
}

#[test]
fn search_and_witnesses_handle_non_groebner_initial_generators() {
    let variables = vec![Symbol::new("x"), Symbol::new("y")];
    let input = vec![
        poly(&variables, &[(vec![1, 1], 1, 1), (vec![0, 0], -1, 1)]),
        poly(&variables, &[(vec![0, 2], 1, 1), (vec![0, 0], -1, 1)]),
    ];
    let result =
        checked_groebner_basis(&input, TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(
        result.basis(),
        [
            poly(&variables, &[(vec![1, 0], 1, 1), (vec![0, 1], -1, 1)]),
            poly(&variables, &[(vec![0, 2], 1, 1), (vec![0, 0], -1, 1)]),
        ]
    );
}

#[test]
fn rational_nonmonic_redundant_inputs_preserve_the_exact_original_ideal() {
    let variables = vec![Symbol::new("x"), Symbol::new("y")];
    let equation = poly(
        &variables,
        &[(vec![1, 0], 2, 3), (vec![0, 1], 1, 1), (vec![0, 0], -1, 3)],
    );
    let input = vec![
        equation.clone(),
        poly(&variables, &[(vec![1, 0], 4, 1), (vec![0, 1], -2, 1)]),
        MultivariatePoly::zero(variables.clone()),
        equation,
    ];
    let result =
        checked_groebner_basis(&input, TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(result.input_basis(), input);
    assert_eq!(
        result.basis(),
        [
            poly(&variables, &[(vec![1, 0], 1, 1), (vec![0, 0], -1, 8)]),
            poly(&variables, &[(vec![0, 1], 1, 1), (vec![0, 0], -1, 4)]),
        ]
    );
}

#[test]
fn inconsistent_zero_and_nonradical_ideals_keep_their_distinct_meanings() {
    let variables = vec![Symbol::new("x"), Symbol::new("y")];
    let inconsistent = vec![
        poly(
            &variables,
            &[(vec![1, 0], 1, 1), (vec![0, 1], 1, 1), (vec![0, 0], -1, 1)],
        ),
        poly(
            &variables,
            &[(vec![1, 0], 1, 1), (vec![0, 1], 1, 1), (vec![0, 0], -2, 1)],
        ),
    ];
    let result = checked_groebner_basis(
        &inconsistent,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(result.basis(), [MultivariatePoly::one(variables.clone())]);
    let zero = vec![MultivariatePoly::zero(variables.clone())];
    let result =
        checked_groebner_basis(&zero, TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap();
    assert!(result.basis().is_empty());
    assert_eq!(result.input_basis(), zero);
    assert!(
        checked_groebner_basis(&[], TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .unwrap()
            .basis()
            .is_empty()
    );
    let nonradical = vec![
        poly(&variables, &[(vec![2, 0], 1, 1)]),
        poly(&variables, &[(vec![0, 3], 1, 1)]),
    ];
    let result =
        checked_groebner_basis(&nonradical, TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(result.basis(), nonradical);
    assert_eq!(result.standard_monomials(&mut Unbounded).unwrap().len(), 6);
}

#[test]
fn independent_kernel_rejects_changed_coefficients_omitted_pairs_and_rebound_inputs() {
    let (variables, input) = coupled();
    let original = bounded_groebner_certificate(&input, TermOrder::Lex, &mut Unbounded).unwrap();
    verify_complete_groebner_certificate(&input, &original, &mut Unbounded).unwrap();

    let mut forged = original.clone();
    forged.candidate.basis[0]
        .terms
        .insert(vec![0, 0], BigRational::one());
    assert!(verify_complete_groebner_certificate(&input, &forged, &mut Unbounded).is_err());

    let mut forged = original.clone();
    forged.s_pairs.clear();
    assert!(verify_complete_groebner_certificate(&input, &forged, &mut Unbounded).is_err());

    let mut forged = original.clone();
    for coefficient in &mut forged.candidate.input_ideal_witnesses[0] {
        *coefficient = MultivariatePoly::zero(variables.clone());
    }
    assert!(verify_complete_groebner_certificate(&input, &forged, &mut Unbounded).is_err());

    let mut changed_input = input;
    changed_input[0]
        .terms
        .insert(vec![0, 0], BigRational::one());
    assert!(
        verify_complete_groebner_certificate(&changed_input, &original, &mut Unbounded).is_err()
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

#[test]
fn cancellation_in_both_phases_and_at_publication_cannot_return_acceptance() {
    let (_, input) = coupled();
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    checked_groebner_basis(&input, TermOrder::Lex, &mut generator, &mut verifier).unwrap();
    assert!(generator.compute > 0 && generator.memory > 0);
    assert!(verifier.compute > 0 && verifier.memory > 0);
    for point in [1, generator.checkpoints / 2, generator.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            checked_groebner_basis(&input, TermOrder::Lex, &mut cancelled, &mut Unbounded).is_err()
        );
    }
    for point in [1, verifier.checkpoints / 2, verifier.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            checked_groebner_basis(&input, TermOrder::Lex, &mut Unbounded, &mut cancelled).is_err()
        );
    }
}

#[test]
fn actual_budgets_enforce_separate_generation_and_verification_limits() {
    let (_, input) = coupled();
    let mut generator = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0));
    checked_groebner_basis(&input, TermOrder::Lex, &mut generator, &mut verifier).unwrap();
    let mut exhausted = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        checked_groebner_basis(&input, TermOrder::Lex, &mut exhausted, &mut untouched).is_err()
    );
    assert_eq!(untouched.checkpoints, 0);
    let mut exhausted = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        checked_groebner_basis(&input, TermOrder::Lex, &mut Unbounded, &mut exhausted).is_err()
    );
}

#[test]
fn malformed_and_oversized_sources_refuse_before_search() {
    let (variables, input) = coupled();
    let mut wrong_ring = input.clone();
    wrong_ring[1].generators.reverse();
    assert!(
        checked_groebner_basis(&wrong_ring, TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .is_err()
    );
    let malformed = MultivariatePoly {
        generators: variables.clone(),
        terms: BTreeMap::from([(vec![1], BigRational::one())]),
    };
    assert!(
        checked_groebner_basis(&[malformed], TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .is_err()
    );
    let too_large = [
        poly(&variables, &[(vec![257, 0], 1, 1)]),
        MultivariatePoly::new(
            variables.clone(),
            BTreeMap::from([(vec![1, 0], BigRational::from_integer(BigInt::one() << 4096))]),
        )
        .unwrap(),
        MultivariatePoly::one((0..9).map(|i| Symbol::new(format!("x{i}"))).collect()),
    ];
    for polynomial in too_large {
        assert!(
            checked_groebner_basis(
                &[polynomial],
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked_groebner_basis(
            &vec![input[0].clone(); 65],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}
