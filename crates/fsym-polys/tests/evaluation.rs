//! Exact coefficient contractions, typed ring binding and evaluation adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity};
use fsym_polys::{
    MultivariatePoly as Polynomial, PolynomialEvaluationCertificate as Certificate,
    checked_polynomial_evaluation as evaluate, verify_polynomial_evaluation as verify,
};
use std::collections::BTreeMap;

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn polynomial(generators: &[Symbol], entries: &[(&[u32], i64, i64)]) -> Polynomial {
    Polynomial::new(
        generators.to_vec(),
        entries
            .iter()
            .map(|(powers, n, d)| (powers.to_vec(), rational(*n, *d)))
            .collect(),
    )
    .unwrap()
}

fn source() -> Polynomial {
    polynomial(
        &[Symbol::new("x"), Symbol::new("y"), Symbol::new("z")],
        &[
            (&[2, 1, 0], 1, 2),
            (&[1, 2, 0], -3, 2),
            (&[0, 0, 1], 3, 1),
            (&[0, 0, 0], 4, 1),
        ],
    )
}

fn run(source: &Polynomial, values: &[Option<BigRational>]) -> Polynomial {
    evaluate(source, values, &mut Unbounded, &mut Unbounded)
        .unwrap()
        .into_result()
}

#[test]
fn partial_evaluation_keeps_ordered_remaining_axes_and_exact_rational_coefficients() {
    let source = source();
    assert_eq!(
        run(&source, &[Some(rational(2, 1)), None, None]),
        polynomial(
            &[Symbol::new("y"), Symbol::new("z")],
            &[
                (&[2, 0], -3, 1),
                (&[1, 0], 2, 1),
                (&[0, 1], 3, 1),
                (&[0, 0], 4, 1),
            ]
        )
    );
    assert_eq!(
        run(&source, &[None, Some(rational(-1, 2)), None]),
        polynomial(
            &[Symbol::new("x"), Symbol::new("z")],
            &[
                (&[2, 0], -1, 4),
                (&[1, 0], -3, 8),
                (&[0, 1], 3, 1),
                (&[0, 0], 4, 1),
            ]
        )
    );
}

#[test]
fn simultaneous_and_successive_assignments_agree_as_exact_polynomial_maps() {
    let source = source();
    let simultaneous = run(&source, &[Some(rational(2, 1)), Some(rational(3, 1)), None]);
    let successive = run(
        &run(&source, &[Some(rational(2, 1)), None, None]),
        &[Some(rational(3, 1)), None],
    );
    assert_eq!(simultaneous, successive);
    assert_eq!(
        simultaneous,
        polynomial(&[Symbol::new("z")], &[(&[1], 3, 1), (&[0], -17, 1)])
    );
}

#[test]
fn full_evaluation_binds_the_source_assignments_and_explicit_constant_ring() {
    let source = source();
    let values = [
        Some(rational(2, 1)),
        Some(rational(3, 1)),
        Some(rational(4, 1)),
    ];
    let accepted = evaluate(&source, &values, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(accepted.source(), &source);
    assert_eq!(accepted.values(), &values);
    assert_eq!(accepted.result(), &polynomial(&[], &[(&[], -5, 1)]));
    assert_eq!(
        verify(
            &source,
            &values,
            &Certificate {
                result: accepted.result().clone()
            },
            &mut Unbounded
        )
        .unwrap(),
        accepted
    );
}

#[test]
fn empty_assignments_and_zero_dimensional_rings_are_supported() {
    let source = source();
    assert_eq!(run(&source, &[None, None, None]), source);
    let constant = polynomial(&[], &[(&[], 4, 3)]);
    assert_eq!(run(&constant, &[]), constant);
    let zero = Polynomial::zero(vec![]);
    assert_eq!(run(&zero, &[]), zero);
    assert!(
        evaluate(
            &constant,
            &[Some(rational(0, 1))],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn zero_and_constant_results_retain_absent_unassigned_typed_variables() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let other_x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let variables = [x, other_x.clone()];
    let zero = Polynomial::zero(variables.to_vec());
    let result = run(&zero, &[Some(rational(0, 1)), None]);
    assert_eq!(result.generators(), std::slice::from_ref(&other_x));
    assert!(result.is_zero());
    let constant = polynomial(&variables, &[(&[0, 0], 2, 3)]);
    assert_eq!(
        run(&constant, &[Some(rational(0, 1)), None]),
        polynomial(&[other_x], &[(&[0], 2, 3)])
    );
}

#[test]
fn projection_combines_colliding_coefficients_and_removes_exact_zero_columns() {
    let source = polynomial(
        &[Symbol::new("x"), Symbol::new("y")],
        &[
            (&[2, 1], 1, 1),
            (&[1, 1], -2, 1),
            (&[0, 1], 1, 1),
            (&[0, 0], 2, 1),
        ],
    );
    assert_eq!(
        run(&source, &[Some(rational(1, 1)), None]),
        polynomial(&[Symbol::new("y")], &[(&[0], 2, 1)])
    );
    assert_eq!(
        run(&source, &[Some(rational(0, 1)), None]),
        polynomial(&[Symbol::new("y")], &[(&[1], 1, 1), (&[0], 2, 1)])
    );
    assert_eq!(
        run(&source, &[Some(rational(0, 1)), Some(rational(-2, 1))]),
        Polynomial::zero(vec![])
    );
}

#[test]
fn mutations_cannot_omit_or_alter_coefficients_or_rebind_original_assignments() {
    let source = source();
    let values = [Some(rational(2, 1)), None, None];
    let result = run(&source, &values);
    let mut dropped = result.clone();
    dropped.terms.remove(&vec![2, 0]);
    let mut changed = result.clone();
    changed.terms.insert(vec![0, 0], rational(5, 1));
    let mut added = result.clone();
    added.terms.insert(vec![0, 2], rational(1, 1));
    for result in [
        Polynomial::zero(result.generators().to_vec()),
        dropped,
        changed,
        added,
    ] {
        assert!(verify(&source, &values, &Certificate { result }, &mut Unbounded).is_err());
    }
    assert!(
        verify(
            &source,
            &[Some(rational(3, 1)), None, None],
            &Certificate {
                result: result.clone()
            },
            &mut Unbounded
        )
        .is_err()
    );
    let mut other_source = source.clone();
    other_source.terms.insert(vec![0, 0, 0], rational(9, 1));
    assert!(
        verify(
            &other_source,
            &values,
            &Certificate { result },
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn zero_certificates_still_bind_generator_order_dimension_and_typed_identity() {
    let source = Polynomial::zero(source().generators().to_vec());
    let values = [Some(rational(2, 1)), None, None];
    let wrong_identity =
        Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([4; 32]));
    for variables in [
        vec![Symbol::new("z"), Symbol::new("y")],
        vec![wrong_identity, Symbol::new("z")],
        vec![],
        source.generators().to_vec(),
    ] {
        assert!(
            verify(
                &source,
                &values,
                &Certificate {
                    result: Polynomial::zero(variables)
                },
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn malformed_sources_and_candidates_refuse_before_returning_an_accepted_value() {
    let source = source();
    let values = [Some(rational(2, 1)), None, None];
    let result = run(&source, &values);
    let mut bad_width = source.clone();
    bad_width.terms.insert(vec![1], rational(1, 1));
    let mut zero_coefficient = source.clone();
    zero_coefficient.terms.insert(vec![1, 0, 0], rational(0, 1));
    let mut duplicate = source.clone();
    duplicate.generators[1] = duplicate.generators[0].clone();
    for bad in [bad_width, zero_coefficient, duplicate] {
        assert!(evaluate(&bad, &values, &mut Unbounded, &mut Unbounded).is_err());
        assert!(
            verify(
                &bad,
                &values,
                &Certificate {
                    result: result.clone()
                },
                &mut Unbounded
            )
            .is_err()
        );
    }
    let mut malformed = result.clone();
    malformed.terms.insert(vec![1], rational(1, 1));
    assert!(
        verify(
            &source,
            &values,
            &Certificate { result: malformed },
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn assignment_width_and_values_are_admitted_even_for_a_zero_source() {
    let source = Polynomial::zero(source().generators().to_vec());
    for values in [vec![], vec![None; 2], vec![None; 4]] {
        assert!(evaluate(&source, &values, &mut Unbounded, &mut Unbounded).is_err());
    }
    let too_large = BigRational::from_integer(BigInt::from(1) << 4096u32);
    assert!(
        evaluate(
            &source,
            &[Some(too_large), None, None],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn degree_coefficient_ring_and_intermediate_growth_limits_are_explicit() {
    let x = Symbol::new("x");
    let source = polynomial(std::slice::from_ref(&x), &[(&[256], 1, 1)]);
    assert_eq!(
        run(&source, &[Some(rational(-1, 1))]),
        polynomial(&[], &[(&[], 1, 1)])
    );
    let high_degree = polynomial(std::slice::from_ref(&x), &[(&[257], 1, 1)]);
    assert!(
        evaluate(
            &high_degree,
            &[Some(rational(0, 1))],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let huge = BigRational::from_integer(BigInt::from(1) << 4096u32);
    let high_coefficient = Polynomial {
        generators: vec![x.clone()],
        terms: BTreeMap::from([(vec![0], huge)]),
    };
    assert!(evaluate(&high_coefficient, &[None], &mut Unbounded, &mut Unbounded).is_err());
    let growing = polynomial(&[x], &[(&[2], 1, 1)]);
    let value = BigRational::from_integer(BigInt::from(1) << 2048u32);
    assert!(evaluate(&growing, &[Some(value)], &mut Unbounded, &mut Unbounded).is_err());
    let too_many = Polynomial::zero((0..9).map(|i| Symbol::new(format!("x{i}"))).collect());
    assert!(
        evaluate(
            &too_many,
            &[const { None }; 9],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn all_small_bilinear_sources_match_independently_derived_rational_formulas() {
    let generators = [Symbol::new("x"), Symbol::new("y")];
    for code in 0..81i64 {
        let [a, b, c, d] = [
            code % 3 - 1,
            code / 3 % 3 - 1,
            code / 9 % 3 - 1,
            code / 27 % 3 - 1,
        ];
        let source = polynomial(
            &generators,
            &[
                (&[1, 1], a, 1),
                (&[1, 0], b, 1),
                (&[0, 1], c, 1),
                (&[0, 0], d, 1),
            ],
        );
        for x in -2..=2 {
            let result = run(&source, &[Some(rational(x, 2)), None]);
            assert_eq!(
                result,
                polynomial(
                    &[generators[1].clone()],
                    &[(&[1], a * x + 2 * c, 2), (&[0], b * x + 2 * d, 2)]
                )
            );
            for y in -2..=2 {
                let result = run(&source, &[Some(rational(x, 2)), Some(rational(y, 3))]);
                assert_eq!(
                    result,
                    polynomial(&[], &[(&[], a * x * y + 3 * b * x + 2 * c * y + 6 * d, 6)])
                );
            }
        }
    }
}

#[derive(Default)]
struct Count {
    steps: u64,
    bytes: u64,
    checkpoints: usize,
    cancel: Option<usize>,
}

impl BudgetMeter for Count {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.steps += amount,
            Dimension::MemoryBytes => self.bytes += amount,
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
        if self.cancel == Some(self.checkpoints) {
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
fn cumulative_budgets_and_cancellation_cover_both_phases_and_final_publication() {
    let source = source();
    let values = [Some(rational(4, 3)), None, Some(rational(-3, 2))];
    let (mut generator, mut verifier) = (Count::default(), Count::default());
    evaluate(&source, &values, &mut generator, &mut verifier).unwrap();
    for (dimension, generated, verified) in [
        (Dimension::ComputeSteps, generator.steps, verifier.steps),
        (Dimension::MemoryBytes, generator.bytes, verifier.bytes),
    ] {
        assert!(generated > 0 && verified > 0);
        evaluate(
            &source,
            &values,
            &mut cap(dimension, generated),
            &mut cap(dimension, verified),
        )
        .unwrap();
        assert!(
            evaluate(
                &source,
                &values,
                &mut cap(dimension, generated - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            evaluate(
                &source,
                &values,
                &mut Unbounded,
                &mut cap(dimension, verified - 1)
            )
            .is_err()
        );
    }
    for (generation, checkpoints) in [(true, generator.checkpoints), (false, verifier.checkpoints)]
    {
        for at in [1, checkpoints / 2, checkpoints] {
            let mut stopped = Count {
                cancel: Some(at),
                ..Count::default()
            };
            let result = if generation {
                evaluate(&source, &values, &mut stopped, &mut Unbounded)
            } else {
                evaluate(&source, &values, &mut Unbounded, &mut stopped)
            };
            assert!(result.is_err());
        }
    }
}
