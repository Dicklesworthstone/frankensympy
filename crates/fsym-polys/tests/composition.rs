//! Exact composition formulas, typed-ring binding, forgery and resource tests.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity};
use fsym_polys::{
    MultivariatePoly as Polynomial, PolynomialCompositionCertificate as Certificate,
    checked_polynomial_composition as compose, verify_polynomial_composition as verify,
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

fn run(outer: &Polynomial, inner: &Polynomial) -> Polynomial {
    compose(outer, inner, &mut Unbounded, &mut Unbounded)
        .unwrap()
        .into_result()
}

#[test]
fn multivariate_composition_substitutes_only_the_first_declared_axis() {
    let ring = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let outer = polynomial(
        &ring,
        &[(&[2, 0, 0], 1, 1), (&[1, 1, 0], 1, 1), (&[0, 0, 1], 1, 1)],
    );
    let inner = polynomial(&ring, &[(&[1, 0, 0], 1, 1), (&[0, 1, 0], 1, 1)]);
    let expected = polynomial(
        &ring,
        &[
            (&[2, 0, 0], 1, 1),
            (&[1, 1, 0], 3, 1),
            (&[0, 2, 0], 2, 1),
            (&[0, 0, 1], 1, 1),
        ],
    );
    assert_eq!(run(&outer, &inner), expected);
    let accepted = compose(&outer, &inner, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(accepted.outer(), &outer);
    assert_eq!(accepted.inner(), &inner);
    assert_eq!(accepted.result(), &expected);
    assert_eq!(
        verify(
            &outer,
            &inner,
            &Certificate { result: expected },
            &mut Unbounded
        )
        .unwrap(),
        accepted
    );
}

#[test]
fn rational_coefficients_and_exact_cancellation_are_preserved() {
    let ring = [Symbol::new("x"), Symbol::new("y")];
    let outer = polynomial(
        &ring,
        &[
            (&[2, 0], 1, 2),
            (&[1, 1], -1, 1),
            (&[0, 2], 1, 2),
            (&[0, 0], -3, 7),
        ],
    );
    let inner = polynomial(&ring, &[(&[1, 0], 1, 3), (&[0, 1], 1, 1)]);
    assert_eq!(
        run(&outer, &inner),
        polynomial(&ring, &[(&[2, 0], 1, 18), (&[0, 0], -3, 7)])
    );
    assert_eq!(
        run(&outer, &polynomial(&ring, &[(&[0, 1], 1, 1)])),
        polynomial(&ring, &[(&[0, 0], -3, 7)])
    );
}

#[test]
fn sparse_degree_gaps_and_a_nonzero_lowest_degree_are_composed() {
    let ring = [Symbol::new("x"), Symbol::new("y")];
    let outer = polynomial(&ring, &[(&[120, 1], 2, 1), (&[3, 0], -1, 2)]);
    let inner = polynomial(&ring, &[(&[0, 2], -1, 1)]);
    assert_eq!(
        run(&outer, &inner),
        polynomial(&ring, &[(&[0, 241], 2, 1), (&[0, 6], 1, 2)])
    );
}

#[test]
fn zero_constant_and_identity_compositions_keep_the_entire_original_ring() {
    let ring = [Symbol::new("x"), Symbol::new("unused")];
    let outer = polynomial(&ring, &[(&[3, 0], 2, 1), (&[0, 0], 3, 2)]);
    let zero = Polynomial::zero(ring.to_vec());
    let constant = polynomial(&ring, &[(&[0, 0], -2, 1)]);
    let identity = polynomial(&ring, &[(&[1, 0], 1, 1)]);
    assert_eq!(run(&outer, &zero), polynomial(&ring, &[(&[0, 0], 3, 2)]));
    assert_eq!(
        run(&outer, &constant),
        polynomial(&ring, &[(&[0, 0], -29, 2)])
    );
    assert_eq!(run(&outer, &identity), outer);
    assert_eq!(run(&zero, &outer), zero);
    assert_eq!(run(&constant, &outer), constant);
    let empty = Polynomial::zero(vec![]);
    assert!(compose(&empty, &empty, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn a_binomial_corpus_matches_independent_closed_coefficients() {
    let ring = [Symbol::new("x"), Symbol::new("y"), Symbol::new("unused")];
    for degree in 0u32..=7 {
        for numerator in [-2i64, -1, 0, 1, 2] {
            for denominator in [1i64, 2, 3] {
                for offset in [-2i64, 0, 2] {
                    let outer = polynomial(&ring, &[(&[degree, 1, 0], 1, 1)]);
                    let inner = polynomial(
                        &ring,
                        &[
                            (&[1, 0, 0], numerator, denominator),
                            (&[0, 1, 0], offset, 1),
                        ],
                    );
                    let mut expected = BTreeMap::new();
                    let mut binomial = 1i64;
                    for exponent in 0..=degree {
                        let value = rational(
                            binomial * numerator.pow(exponent) * offset.pow(degree - exponent),
                            denominator.pow(exponent),
                        );
                        if value != rational(0, 1) {
                            expected.insert(vec![exponent, degree - exponent + 1, 0], value);
                        }
                        if exponent != degree {
                            binomial =
                                binomial * i64::from(degree - exponent) / i64::from(exponent + 1);
                        }
                    }
                    assert_eq!(
                        run(&outer, &inner),
                        Polynomial::new(ring.to_vec(), expected).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn changed_dropped_added_and_wrong_axis_outputs_are_rejected() {
    let ring = [Symbol::new("x"), Symbol::new("y")];
    let outer = polynomial(&ring, &[(&[2, 0], 1, 1), (&[1, 1], 2, 1)]);
    let inner = polynomial(&ring, &[(&[1, 0], 1, 1), (&[0, 0], 1, 1)]);
    let result = run(&outer, &inner);
    let mut changed = result.clone();
    changed.terms.insert(vec![1, 0], rational(-2, 1));
    let mut dropped = result.clone();
    dropped.terms.remove(&vec![1, 1]);
    let mut added = result.clone();
    added.terms.insert(vec![0, 2], rational(1, 1));
    // Replacing y instead of x would produce x^2+2*x*(x+1).
    let wrong_axis = polynomial(&ring, &[(&[2, 0], 3, 1), (&[1, 0], 2, 1)]);
    for bad in [
        changed,
        dropped,
        added,
        wrong_axis,
        Polynomial::zero(ring.to_vec()),
    ] {
        assert!(verify(&outer, &inner, &Certificate { result: bad }, &mut Unbounded).is_err());
    }
    let changed_inner = polynomial(&ring, &[(&[1, 0], 1, 1), (&[0, 0], -1, 1)]);
    assert!(
        verify(
            &outer,
            &changed_inner,
            &Certificate {
                result: result.clone()
            },
            &mut Unbounded
        )
        .is_err()
    );
    let changed_outer = polynomial(&ring, &[(&[2, 0], 1, 1)]);
    assert!(
        verify(
            &changed_outer,
            &inner,
            &Certificate { result },
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn ordered_typed_identity_is_checked_for_all_inputs_and_the_output() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let ring = [x.clone(), other.clone()];
    let outer = polynomial(&ring, &[(&[2, 0], 1, 1)]);
    let inner = polynomial(&ring, &[(&[0, 1], 1, 1)]);
    let result = run(&outer, &inner);
    assert_eq!(result, polynomial(&ring, &[(&[0, 2], 1, 1)]));
    for variables in [
        vec![other.clone(), x],
        vec![other.clone(), Symbol::new("unused")],
    ] {
        let mut changed_inner = inner.clone();
        changed_inner.generators = variables.clone();
        assert!(compose(&outer, &changed_inner, &mut Unbounded, &mut Unbounded).is_err());
        let mut changed_result = result.clone();
        changed_result.generators = variables;
        assert!(
            verify(
                &outer,
                &inner,
                &Certificate {
                    result: changed_result
                },
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn malformed_and_oversized_replacements_are_admitted_even_for_zero_sources() {
    let ring = [Symbol::new("x"), Symbol::new("y")];
    let outer = Polynomial::zero(ring.to_vec());
    let inner = polynomial(&ring, &[(&[1, 0], 1, 1)]);
    let mut wrong_width = inner.clone();
    wrong_width.terms.insert(vec![1], rational(1, 1));
    let mut zero = inner.clone();
    zero.terms.insert(vec![0, 0], rational(0, 1));
    let mut duplicate = inner.clone();
    duplicate.generators[1] = duplicate.generators[0].clone();
    let mut huge = inner.clone();
    huge.terms.insert(
        vec![0, 0],
        BigRational::from_integer(BigInt::from(1) << 4096u32),
    );
    let extreme = polynomial(&ring, &[(&[u32::MAX, 1], 1, 1)]);
    for bad in [wrong_width, zero, duplicate, huge, extreme] {
        assert!(compose(&outer, &bad, &mut Unbounded, &mut Unbounded).is_err());
        assert!(
            verify(
                &outer,
                &bad,
                &Certificate {
                    result: outer.clone()
                },
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn degree_boundary_and_intermediate_coefficient_growth_refuse_explicitly() {
    let ring = [Symbol::new("x"), Symbol::new("y")];
    let inner = polynomial(&ring, &[(&[0, 2], 1, 1)]);
    assert_eq!(
        run(&polynomial(&ring, &[(&[128, 0], 1, 1)]), &inner),
        polynomial(&ring, &[(&[0, 256], 1, 1)])
    );
    assert!(
        compose(
            &polynomial(&ring, &[(&[128, 1], 1, 1)]),
            &inner,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let huge_inner = Polynomial::new(
        ring.to_vec(),
        BTreeMap::from([(
            vec![0, 0],
            BigRational::from_integer(BigInt::from(1) << 2048u32),
        )]),
    )
    .unwrap();
    assert!(
        compose(
            &polynomial(&ring, &[(&[2, 0], 1, 1)]),
            &huge_inner,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let excessive_ring = Polynomial::zero((0..9).map(|i| Symbol::new(format!("x{i}"))).collect());
    assert!(
        compose(
            &excessive_ring,
            &excessive_ring,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn generator_and_verifier_have_separate_nonresetting_budgets() {
    let ring = [Symbol::new("x"), Symbol::new("y")];
    let outer = polynomial(&ring, &[(&[3, 0], 1, 1), (&[0, 1], 1, 1)]);
    let inner = polynomial(&ring, &[(&[1, 0], 1, 1), (&[0, 0], 1, 1)]);
    let mut limits = BudgetLimits::uniform(10_000_000, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut exhausted = Budget::new(limits);
    assert!(compose(&outer, &inner, &mut exhausted, &mut Unbounded).is_err());
    let mut exhausted = Budget::new(limits);
    assert!(compose(&outer, &inner, &mut Unbounded, &mut exhausted).is_err());
    let result = run(&outer, &inner);
    assert!(verify(&outer, &inner, &Certificate { result }, &mut exhausted).is_err());
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
fn cumulative_charges_and_cancellation_cover_generation_verification_and_publication() {
    let ring = [Symbol::new("x"), Symbol::new("y")];
    let outer = polynomial(&ring, &[(&[3, 0], 1, 1), (&[1, 1], -1, 2)]);
    let inner = polynomial(&ring, &[(&[1, 0], 2, 3), (&[0, 1], 1, 1)]);
    let (mut generator, mut verifier) = (Count::default(), Count::default());
    compose(&outer, &inner, &mut generator, &mut verifier).unwrap();
    for (dimension, generated, verified) in [
        (Dimension::ComputeSteps, generator.steps, verifier.steps),
        (Dimension::MemoryBytes, generator.bytes, verifier.bytes),
    ] {
        assert!(generated > 0 && verified > 0);
        compose(
            &outer,
            &inner,
            &mut cap(dimension, generated),
            &mut cap(dimension, verified),
        )
        .unwrap();
        assert!(
            compose(
                &outer,
                &inner,
                &mut cap(dimension, generated - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            compose(
                &outer,
                &inner,
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
                compose(&outer, &inner, &mut stopped, &mut Unbounded)
            } else {
                compose(&outer, &inner, &mut Unbounded, &mut stopped)
            };
            assert!(result.is_err());
        }
    }
}
