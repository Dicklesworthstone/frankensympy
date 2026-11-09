//! Exact partial contractions, output-ring binding and evaluation adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, Expr, Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldEvaluationCertificate as Certificate, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, checked_prime_field_evaluation as evaluate,
    checked_prime_field_multivariate_source, verify_prime_field_evaluation as verify,
};

fn terms(entries: &[(&[u32], u64)]) -> Terms {
    entries.iter().map(|(m, c)| (m.to_vec(), *c)).collect()
}

fn source() -> Polynomial {
    Polynomial::new(
        &[Symbol::new("x"), Symbol::new("y"), Symbol::new("z")],
        5,
        &terms(&[
            (&[2, 1, 0], 1),
            (&[1, 2, 0], 2),
            (&[0, 0, 1], 3),
            (&[0, 0, 0], 4),
        ]),
    )
    .unwrap()
}

fn run(source: &Polynomial, values: &[Option<u64>]) -> Polynomial {
    evaluate(source, values, &mut Unbounded, &mut Unbounded)
        .unwrap()
        .into_result()
}

#[test]
fn partial_evaluation_keeps_the_remaining_ring_and_exact_coefficients() {
    let polynomial = source();
    let result = run(&polynomial, &[Some(2), None, None]);
    assert_eq!(result.generators(), &[Symbol::new("y"), Symbol::new("z")]);
    assert_eq!(result.prime(), 5);
    assert_eq!(
        result.terms(),
        &terms(&[(&[2, 0], 4), (&[1, 0], 4), (&[0, 1], 3), (&[0, 0], 4)])
    );
    let result = run(&polynomial, &[None, Some(4), None]);
    assert_eq!(result.generators(), &[Symbol::new("x"), Symbol::new("z")]);
    assert_eq!(
        result.terms(),
        &terms(&[(&[2, 0], 4), (&[1, 0], 2), (&[0, 1], 3), (&[0, 0], 4)])
    );
}

#[test]
fn simultaneous_and_successive_assignments_are_the_same_exact_map() {
    let polynomial = source();
    let simultaneous = run(&polynomial, &[Some(2), Some(3), None]);
    let successive = run(&run(&polynomial, &[Some(2), None, None]), &[Some(3), None]);
    assert_eq!(simultaneous, successive);
    assert_eq!(simultaneous.generators(), &[Symbol::new("z")]);
    assert_eq!(simultaneous.terms(), &terms(&[(&[1], 3), (&[0], 2)]));
}

#[test]
fn full_evaluation_returns_the_explicit_constant_field() {
    let polynomial = source();
    let accepted = evaluate(
        &polynomial,
        &[Some(2), Some(3), Some(4)],
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(accepted.source(), &polynomial);
    assert_eq!(accepted.values(), &[Some(2), Some(3), Some(4)]);
    assert!(accepted.result().generators().is_empty());
    assert_eq!(accepted.result().terms(), &terms(&[(&[], 4)]));
    assert_eq!(accepted.result().prime(), 5);
    let certificate = Certificate {
        result: accepted.result().clone(),
    };
    assert_eq!(
        verify(&polynomial, accepted.values(), &certificate, &mut Unbounded).unwrap(),
        accepted
    );
}

#[test]
fn empty_assignment_and_constant_ring_are_supported_without_reinferring_variables() {
    let polynomial = source();
    assert_eq!(run(&polynomial, &[None, None, None]), polynomial);
    let constant = Polynomial::new(&[], 7, &terms(&[(&[], 4)])).unwrap();
    assert_eq!(run(&constant, &[]), constant);
    assert!(evaluate(&constant, &[Some(0)], &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn zero_and_absent_coordinates_still_keep_unassigned_typed_generators() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let other_x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let zero = Polynomial::new(&[x.clone(), other_x.clone()], 3, &Terms::new()).unwrap();
    let result = run(&zero, &[Some(0), None]);
    assert_eq!(result.generators(), &[other_x.clone()]);
    assert!(result.is_zero());
    let constant = Polynomial::new(&[x, other_x.clone()], 3, &terms(&[(&[0, 0], 2)])).unwrap();
    assert_eq!(run(&constant, &[Some(0), None]).generators(), &[other_x]);
}

#[test]
fn zero_assignment_respects_zero_exponents_and_characteristic_cancellation() {
    let source = Polynomial::new(
        &[Symbol::new("x"), Symbol::new("y")],
        5,
        &terms(&[(&[1, 1], 1), (&[0, 1], 4), (&[0, 0], 2)]),
    )
    .unwrap();
    assert_eq!(
        run(&source, &[Some(0), None]).terms(),
        &terms(&[(&[1], 4), (&[0], 2)])
    );
    assert_eq!(run(&source, &[Some(1), None]).terms(), &terms(&[(&[0], 2)]));
    assert_eq!(
        run(&source, &[Some(1), Some(0)]).terms(),
        &terms(&[(&[], 2)])
    );
}

#[test]
fn certificate_mutations_cannot_drop_terms_add_terms_or_rebind_the_source() {
    let polynomial = source();
    let values = [Some(2), None, None];
    let result = run(&polynomial, &values);
    for candidate in [
        Polynomial::new(result.generators(), 5, &Terms::new()).unwrap(),
        Polynomial::new(
            result.generators(),
            5,
            &terms(&[(&[2, 0], 4), (&[1, 0], 4), (&[0, 1], 2), (&[0, 0], 4)]),
        )
        .unwrap(),
        Polynomial::new(
            result.generators(),
            5,
            &terms(&[
                (&[2, 0], 4),
                (&[1, 0], 4),
                (&[0, 1], 3),
                (&[0, 0], 4),
                (&[0, 2], 1),
            ]),
        )
        .unwrap(),
    ] {
        assert!(
            verify(
                &polynomial,
                &values,
                &Certificate { result: candidate },
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        verify(
            &polynomial,
            &[Some(3), None, None],
            &Certificate { result },
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn zero_certificate_still_binds_characteristic_generator_order_and_identity() {
    let polynomial = Polynomial::new(source().generators(), 5, &Terms::new()).unwrap();
    let values = [Some(2), None, None];
    let wrong_identity =
        Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([4; 32]));
    for candidate in [
        Polynomial::new(&[Symbol::new("y"), Symbol::new("z")], 7, &Terms::new()).unwrap(),
        Polynomial::new(&[Symbol::new("z"), Symbol::new("y")], 5, &Terms::new()).unwrap(),
        Polynomial::new(&[wrong_identity, Symbol::new("z")], 5, &Terms::new()).unwrap(),
        Polynomial::new(&[], 5, &Terms::new()).unwrap(),
        Polynomial::new(polynomial.generators(), 5, &Terms::new()).unwrap(),
    ] {
        assert!(
            verify(
                &polynomial,
                &values,
                &Certificate { result: candidate },
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn evaluation_is_polynomial_identity_not_equality_as_a_finite_field_function() {
    // x^2+x vanishes at every point of F_2, but it is not the zero polynomial.
    let polynomial =
        Polynomial::new(&[Symbol::new("x")], 2, &terms(&[(&[2], 1), (&[1], 1)])).unwrap();
    let zero = Polynomial::new(polynomial.generators(), 2, &Terms::new()).unwrap();
    assert!(
        verify(
            &polynomial,
            &[None],
            &Certificate { result: zero },
            &mut Unbounded
        )
        .is_err()
    );
    for value in [0, 1] {
        assert!(run(&polynomial, &[Some(value)]).is_zero());
    }
}

#[test]
fn malformed_assignment_width_or_noncanonical_residue_refuses_even_for_zero() {
    let polynomial = Polynomial::new(source().generators(), 5, &Terms::new()).unwrap();
    for values in [
        vec![],
        vec![None; 2],
        vec![None; 4],
        vec![Some(5), None, None],
        vec![Some(u64::MAX), None, None],
    ] {
        assert!(evaluate(&polynomial, &values, &mut Unbounded, &mut Unbounded).is_err());
    }
}

#[test]
fn all_small_bilinear_sources_have_exact_coefficient_contractions() {
    for prime in [2u64, 3, 5] {
        for code in 0..prime.pow(4) {
            let [a, b, c, d] = [
                code % prime,
                code / prime % prime,
                code / prime.pow(2) % prime,
                code / prime.pow(3) % prime,
            ];
            let source = Polynomial::new(
                &[Symbol::new("x"), Symbol::new("y")],
                prime,
                &terms(&[(&[1, 1], a), (&[1, 0], b), (&[0, 1], c), (&[0, 0], d)]),
            )
            .unwrap();
            for x in 0..prime {
                let partial = run(&source, &[Some(x), None]);
                let expected = Polynomial::new(
                    &[Symbol::new("y")],
                    prime,
                    &terms(&[(&[1], (a * x + c) % prime), (&[0], (b * x + d) % prime)]),
                )
                .unwrap();
                assert_eq!(partial, expected);
                for y in 0..prime {
                    let result = run(&source, &[Some(x), Some(y)]);
                    let expected = (a * x * y + b * x + c * y + d) % prime;
                    assert_eq!(
                        result.terms(),
                        &if expected == 0 {
                            Terms::new()
                        } else {
                            terms(&[(&[], expected)])
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn maximal_prime_degree_and_typed_coordinate_count_are_admitted() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let source = Polynomial::new(&generators, 65521, &terms(&[(&[8; 8], 65520)])).unwrap();
    let result = run(&source, &[Some(65520); 8]);
    assert_eq!(result.terms(), &terms(&[(&[], 65520)]));
}

#[test]
fn signed_arbitrary_precision_scalar_reduction_is_independently_source_checked() {
    let value = -(BigInt::from(1) << 200u32) - BigInt::from(3);
    let source = Expr::Integer(value);
    let accepted =
        checked_prime_field_multivariate_source(&source, &[], 5, &mut Unbounded, &mut Unbounded)
            .unwrap();
    // 2^200 == 1 mod 5, so -(2^200)-3 == 1 mod 5.
    assert_eq!(accepted.polynomial().terms(), &terms(&[(&[], 1)]));
    assert_eq!(run(accepted.polynomial(), &[]), *accepted.polynomial());
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
fn cumulative_budgets_and_cancellation_cover_verification_and_final_publication() {
    let source = source();
    let values = [Some(4), None, Some(3)];
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
            assert!(result.is_err(), "generation={generation}, checkpoint={at}");
        }
    }
}
