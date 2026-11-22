//! Strict integer distributed reduction and independent-checker adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Symbol};
use fsym_polys::{
    MultivariatePoly, TermOrder, bounded_integer_division, checked_integer_polynomial_division,
    checked_polynomial_division, verify_bounded_integer_division,
};

fn poly(gens: &[Symbol], terms: &[(Vec<u32>, i64, i64)]) -> MultivariatePoly {
    MultivariatePoly::new(
        gens.to_vec(),
        terms
            .iter()
            .map(|(powers, n, d)| (powers.clone(), BigRational::new((*n).into(), (*d).into())))
            .collect(),
    )
    .unwrap()
}

#[test]
fn integer_obstruction_tries_later_divisors_and_remaining_terms() {
    let gens = [Symbol::new("x")];
    let f = poly(&gens, &[(vec![1], 3, 1), (vec![0], 2, 1)]);
    let divisors = [
        poly(&gens, &[(vec![1], 2, 1), (vec![0], 1, 1)]),
        poly(&gens, &[(vec![1], 3, 1)]),
    ];
    let zero = MultivariatePoly::zero(gens.to_vec());
    let one = MultivariatePoly::one(gens.to_vec());
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let (q, r) = checked_integer_polynomial_division(
            &f,
            &divisors,
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(q, [zero.clone(), one.clone()]);
        assert_eq!(r, poly(&gens, &[(vec![0], 2, 1)]));
    }
    let f = poly(&gens, &[(vec![2], 1, 1), (vec![1], 2, 1)]);
    let divisor = [poly(&gens, &[(vec![1], 2, 1)])];
    let (q, r) = checked_integer_polynomial_division(
        &f,
        &divisor,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(q, [one]);
    assert_eq!(r, poly(&gens, &[(vec![2], 1, 1)]));
}

#[test]
fn rational_and_integer_coefficients_choose_different_valid_normal_forms() {
    let gens = [Symbol::new("x")];
    let f = poly(&gens, &[(vec![1], 1, 1)]);
    let divisor = [poly(&gens, &[(vec![1], 2, 1)])];
    let zero = MultivariatePoly::zero(gens.to_vec());
    let integer = checked_integer_polynomial_division(
        &f,
        &divisor,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    let rational =
        checked_polynomial_division(&f, &divisor, TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(integer, (vec![zero.clone()], f.clone()));
    assert_eq!(rational, (vec![poly(&gens, &[(vec![0], 1, 2)])], zero));
    assert!(
        verify_bounded_integer_division(
            &f,
            &divisor,
            &rational.0,
            &rational.1,
            TermOrder::Lex,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn every_supported_order_controls_the_divisor_leading_term() {
    let gens = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let f = poly(&gens, &[(vec![1, 0, 1], 2, 1)]);
    let divisors = [poly(&gens, &[(vec![1, 0, 1], 2, 1), (vec![0, 2, 0], 2, 1)])];
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let result = checked_integer_polynomial_division(
            &f,
            &divisors,
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        let expected = if order == TermOrder::DegRevLex {
            (vec![MultivariatePoly::zero(gens.to_vec())], f.clone())
        } else {
            (
                vec![MultivariatePoly::one(gens.to_vec())],
                poly(&gens, &[(vec![0, 2, 0], -2, 1)]),
            )
        };
        assert_eq!(result, expected);
    }
}

#[test]
fn independent_checker_rejects_identity_preserving_unreduced_remainder() {
    let gens = [Symbol::new("x")];
    let f = poly(&gens, &[(vec![2], 2, 1), (vec![0], 1, 1)]);
    let divisor = [poly(&gens, &[(vec![1], 2, 1)])];
    let (mut q, r) =
        bounded_integer_division(&f, &divisor, TermOrder::Lex, &mut Unbounded).unwrap();
    verify_bounded_integer_division(&f, &divisor, &q, &r, TermOrder::Lex, &mut Unbounded).unwrap();
    let zero = MultivariatePoly::zero(gens.to_vec());
    assert!(
        verify_bounded_integer_division(&f, &divisor, &[zero], &f, TermOrder::Lex, &mut Unbounded,)
            .is_err()
    );
    q[0].terms
        .insert(vec![0], BigRational::from_integer(1.into()));
    assert!(
        verify_bounded_integer_division(&f, &divisor, &q, &r, TermOrder::Lex, &mut Unbounded,)
            .is_err()
    );
    assert!(
        verify_bounded_integer_division(&f, &divisor, &[], &r, TermOrder::Lex, &mut Unbounded,)
            .is_err()
    );
}

#[test]
fn fractional_sources_and_foreign_typed_rings_cannot_enter_integer_division() {
    let gens = [Symbol::new("x")];
    let f = poly(&gens, &[(vec![1], 1, 1)]);
    let fraction = poly(&gens, &[(vec![0], 1, 2)]);
    let foreign = poly(&[Symbol::new("y")], &[(vec![0], 1, 1)]);
    for divisors in [vec![fraction.clone()], vec![foreign]] {
        assert!(
            checked_integer_polynomial_division(
                &f,
                &divisors,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
    assert!(
        checked_integer_polynomial_division(
            &fraction,
            &[],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn empty_and_zero_divisor_primitives_keep_their_existing_contract() {
    let gens = [Symbol::new("x")];
    let f = poly(&gens, &[(vec![1], 1, 1)]);
    let zero = MultivariatePoly::zero(gens.to_vec());
    for divisors in [vec![], vec![zero.clone()]] {
        let (q, r) = checked_integer_polynomial_division(
            &f,
            &divisors,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(q, vec![zero.clone(); divisors.len()]);
        assert_eq!(r, f);
    }
    assert!(
        bounded_integer_division(&f, &vec![zero; 65], TermOrder::Lex, &mut Unbounded,).is_err()
    );
}

#[derive(Default)]
struct Interruptions {
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Interruptions {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
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
fn either_phase_can_exhaust_or_cancel_without_publishing_a_result() {
    let gens = [Symbol::new("x")];
    let f = poly(&gens, &[(vec![3], 2, 1), (vec![1], 1, 1)]);
    let divisors = [poly(&gens, &[(vec![1], 2, 1), (vec![0], 2, 1)])];
    let (q, r) = bounded_integer_division(&f, &divisors, TermOrder::Lex, &mut Unbounded).unwrap();
    for verifier in [false, true] {
        for dimension in [Dimension::ComputeSteps, Dimension::MemoryBytes] {
            let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
            limits.dimensions[dimension.index()] = 0;
            let mut exhausted = Budget::new(limits);
            let result = if verifier {
                checked_integer_polynomial_division(
                    &f,
                    &divisors,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut exhausted,
                )
            } else {
                checked_integer_polynomial_division(
                    &f,
                    &divisors,
                    TermOrder::Lex,
                    &mut exhausted,
                    &mut Unbounded,
                )
            };
            assert!(result.is_err());
        }
        let mut observed = Interruptions::default();
        if verifier {
            verify_bounded_integer_division(&f, &divisors, &q, &r, TermOrder::Lex, &mut observed)
                .unwrap();
        } else {
            bounded_integer_division(&f, &divisors, TermOrder::Lex, &mut observed).unwrap();
        }
        for cancel_at in [1, observed.checkpoints / 2, observed.checkpoints] {
            let mut interrupted = Interruptions {
                checkpoints: 0,
                cancel_at: Some(cancel_at),
            };
            let result = if verifier {
                checked_integer_polynomial_division(
                    &f,
                    &divisors,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut interrupted,
                )
            } else {
                checked_integer_polynomial_division(
                    &f,
                    &divisors,
                    TermOrder::Lex,
                    &mut interrupted,
                    &mut Unbounded,
                )
            };
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .to_lowercase()
                    .contains("cancel")
            );
        }
    }
}
