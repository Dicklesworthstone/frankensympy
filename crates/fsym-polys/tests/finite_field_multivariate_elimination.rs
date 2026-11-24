//! Hand-derived coefficient identities, source-binding adversaries, and
//! cumulative resource boundaries for primary-axis elimination over F_p.
//! Expected formulas are independent of both production determinant paths.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse_raw};
use fsym_polys::{
    PrimeFieldMultivariateEliminationCertificate as Certificate,
    PrimeFieldMultivariateEliminationOperation as Op,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    VerifiedPrimeFieldMultivariateElimination as Checked,
    checked_prime_field_multivariate_elimination as checked,
    checked_prime_field_multivariate_source as source,
    verify_prime_field_multivariate_elimination as verify,
};

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn p(generators: &[Symbol], prime: u64, entries: &[(&[u32], i64)]) -> Polynomial {
    let terms = entries
        .iter()
        .map(|(powers, coefficient)| (powers.to_vec(), coefficient.rem_euclid(prime as i64) as u64))
        .collect();
    Polynomial::new(generators, prime, &terms).unwrap()
}

fn scalar(generators: &[Symbol], prime: u64, value: i64) -> Polynomial {
    p(generators, prime, &[(&vec![0; generators.len()], value)])
}

fn solve(operation: Op, left: &Polynomial, right: Option<&Polynomial>) -> Checked {
    checked(&operation, left, right, &mut Unbounded, &mut Unbounded).unwrap()
}

fn rejects(operation: Op, left: &Polynomial, right: Option<&Polynomial>, terms: &Terms) {
    assert!(
        verify(
            &operation,
            left,
            right,
            &Certificate {
                result: terms.clone(),
            },
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn ordered_resultants_eliminate_only_the_primary_axis_with_exact_signs() {
    let generators = ring(&["x", "y", "z", "unused"]);
    for prime in [2, 3, 5, 7, 65521] {
        let left = p(
            &generators,
            prime,
            &[(&[1, 0, 0, 0], 1), (&[0, 1, 0, 0], -1)],
        );
        let right = p(
            &generators,
            prime,
            &[(&[1, 0, 0, 0], 1), (&[0, 0, 1, 0], -1)],
        );
        let expected = p(
            &generators[1..],
            prime,
            &[(&[1, 0, 0], 1), (&[0, 1, 0], -1)],
        );
        let accepted = solve(Op::Resultant, &left, Some(&right));
        assert_eq!(accepted.result(), &expected);
        assert_eq!(accepted.operation(), Op::Resultant);
        assert_eq!(accepted.left(), &left);
        assert_eq!(accepted.right(), Some(&right));
        assert_eq!(accepted.result().generators(), &generators[1..]);

        // Res(x^3-y, x-z)=y-z^3 and swapping the ordered operands negates it.
        // This sign must hold even where the pinned upstream modular path
        // returns the same value in both directions.
        let cubic = p(
            &generators,
            prime,
            &[(&[3, 0, 0, 0], 1), (&[0, 1, 0, 0], -1)],
        );
        assert_eq!(
            solve(Op::Resultant, &cubic, Some(&right)).result(),
            &p(
                &generators[1..],
                prime,
                &[(&[0, 3, 0], -1), (&[1, 0, 0], 1)]
            )
        );
        let reverse = solve(Op::Resultant, &right, Some(&cubic));
        assert_eq!(
            reverse.result(),
            &p(
                &generators[1..],
                prime,
                &[(&[0, 3, 0], 1), (&[1, 0, 0], -1)]
            )
        );
        if prime != 2 {
            rejects(
                Op::Resultant,
                &right,
                Some(&cubic),
                solve(Op::Resultant, &cubic, Some(&right)).result().terms(),
            );
        }
    }
}

#[test]
fn repeated_factors_and_the_full_eight_row_envelope_have_exact_resultants() {
    let generators = ring(&["x", "y", "z"]);
    for prime in [2, 3, 5, 7, 65521] {
        // Res((x-y)^2,(x-z)^2)=(y-z)^4.
        let left = p(
            &generators,
            prime,
            &[(&[2, 0, 0], 1), (&[1, 1, 0], -2), (&[0, 2, 0], 1)],
        );
        let right = p(
            &generators,
            prime,
            &[(&[2, 0, 0], 1), (&[1, 0, 1], -2), (&[0, 0, 2], 1)],
        );
        let expected = p(
            &generators[1..],
            prime,
            &[
                (&[4, 0], 1),
                (&[3, 1], -4),
                (&[2, 2], 6),
                (&[1, 3], -4),
                (&[0, 4], 1),
            ],
        );
        assert_eq!(
            solve(Op::Resultant, &left, Some(&right)).result(),
            &expected
        );
        // The same explicit value follows from Res(x^4+y,x^4+z)=(z-y)^4.
        let left = p(&generators, prime, &[(&[4, 0, 0], 1), (&[0, 1, 0], 1)]);
        let right = p(&generators, prime, &[(&[4, 0, 0], 1), (&[0, 0, 1], 1)]);
        assert_eq!(
            solve(Op::Resultant, &left, Some(&right)).result(),
            &expected
        );
        assert!(solve(Op::Resultant, &left, Some(&left)).result().is_zero());
        rejects(
            Op::Resultant,
            &left,
            Some(&left),
            scalar(&generators[1..], prime, 1).terms(),
        );
        rejects(Op::Resultant, &left, Some(&right), &Terms::new());
    }
}

#[test]
fn polynomial_constants_zero_operands_and_zero_dimensional_outputs_keep_conventions() {
    for generators in [ring(&["x"]), ring(&["x", "y", "unused"])] {
        for prime in [2, 5] {
            let width = generators.len();
            let mut x3 = vec![0; width];
            x3[0] = 3;
            let f = p(&generators, prime, &[(&x3, 1), (&vec![0; width], 1)]);
            let zero = scalar(&generators, prime, 0);
            let nonzero = scalar(&generators, prime, prime as i64 - 1);
            let one = scalar(&generators[1..], prime, 1);
            for (left, right) in [(&zero, &zero), (&zero, &f), (&f, &zero)] {
                assert_eq!(
                    solve(Op::Resultant, left, Some(right)).result(),
                    &scalar(&generators[1..], prime, 0)
                );
            }
            assert_eq!(
                solve(Op::Resultant, &nonzero, Some(&nonzero)).result(),
                &one
            );
            for (left, right) in [(&nonzero, &f), (&f, &nonzero)] {
                assert_eq!(
                    solve(Op::Resultant, left, Some(right)).result(),
                    &scalar(&generators[1..], prime, -1)
                );
            }
            for source in [&zero, &nonzero] {
                assert!(solve(Op::Discriminant, source, None).result().is_zero());
            }
        }
    }
    let generators = ring(&["x", "y", "z"]);
    let y = p(&generators, 5, &[(&[0, 1, 0], 1)]);
    let z = p(&generators, 5, &[(&[0, 0, 1], 1)]);
    assert_eq!(
        solve(Op::Resultant, &y, Some(&z)).result(),
        &scalar(&generators[1..], 5, 1)
    );
    let cubic = p(&generators, 5, &[(&[3, 0, 0], 1), (&[0, 0, 1], 1)]);
    for (left, right) in [(&y, &cubic), (&cubic, &y)] {
        assert_eq!(
            solve(Op::Resultant, left, Some(right)).result(),
            &p(&generators[1..], 5, &[(&[3, 0], 1)])
        );
    }
    let linear = p(&generators, 5, &[(&[1, 1, 0], 1), (&[0, 0, 1], 1)]);
    assert_eq!(
        solve(Op::Discriminant, &linear, None).result(),
        &scalar(&generators[1..], 5, 1)
    );
}

#[test]
fn universal_nonmonic_quadratic_discriminants_keep_small_characteristic_factors() {
    let generators = ring(&["x", "a", "b", "c"]);
    for prime in [2, 3, 5, 7, 65521] {
        let f = p(
            &generators,
            prime,
            &[(&[2, 1, 0, 0], 1), (&[1, 0, 1, 0], 1), (&[0, 0, 0, 1], 1)],
        );
        let expected = p(
            &generators[1..],
            prime,
            &[(&[0, 2, 0], 1), (&[1, 0, 1], -4)],
        );
        assert_eq!(solve(Op::Discriminant, &f, None).result(), &expected);
        rejects(Op::Discriminant, &f, None, &Terms::new());
    }
    let generators = ring(&["x", "y", "z"]);
    // Formal f' has degree one even though its leading coefficient vanishes.
    // Its formal resultant is y, so dividing by the original LC y gives 1.
    let f = p(
        &generators,
        2,
        &[(&[2, 1, 0], 1), (&[1, 0, 0], 1), (&[0, 0, 1], 1)],
    );
    assert_eq!(
        solve(Op::Discriminant, &f, None).result(),
        &scalar(&generators[1..], 2, 1)
    );
    rejects(
        Op::Discriminant,
        &f,
        None,
        p(&generators[1..], 2, &[(&[1, 0], 1)]).terms(),
    );
}

#[test]
fn universal_cubic_and_nonmonic_quartic_formulas_survive_derivative_degree_drops() {
    let generators = ring(&["x", "a", "b", "c", "d"]);
    for prime in [2, 3, 5, 7, 65521] {
        let cubic = p(
            &generators,
            prime,
            &[
                (&[3, 1, 0, 0, 0], 1),
                (&[2, 0, 1, 0, 0], 1),
                (&[1, 0, 0, 1, 0], 1),
                (&[0, 0, 0, 0, 1], 1),
            ],
        );
        let expected = p(
            &generators[1..],
            prime,
            &[
                (&[0, 2, 2, 0], 1),
                (&[1, 0, 3, 0], -4),
                (&[0, 3, 0, 1], -4),
                (&[2, 0, 0, 2], -27),
                (&[1, 1, 1, 1], 18),
            ],
        );
        assert_eq!(solve(Op::Discriminant, &cubic, None).result(), &expected);
        let quartic_ring = ring(&["x", "a", "b", "c"]);
        let quartic = p(
            &quartic_ring,
            prime,
            &[(&[4, 1, 0, 0], 1), (&[1, 0, 1, 0], 1), (&[0, 0, 0, 1], 1)],
        );
        let expected = p(
            &quartic_ring[1..],
            prime,
            &[(&[3, 0, 3], 256), (&[2, 4, 0], -27)],
        );
        assert_eq!(solve(Op::Discriminant, &quartic, None).result(), &expected);
    }
    for (prime, degree) in [(2, 4), (3, 3)] {
        let generators = ring(&["x", "y"]);
        let f = p(&generators, prime, &[(&[degree, 0], 1), (&[0, 1], 1)]);
        assert!(solve(Op::Discriminant, &f, None).result().is_zero());
        rejects(
            Op::Discriminant,
            &f,
            None,
            scalar(&generators[1..], prime, 1).terms(),
        );
    }
}

#[test]
fn typed_same_spelling_axes_and_all_eight_original_coordinates_are_preserved() {
    let x = Symbol::new("x");
    let shadow = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([91; 32]));
    let mut generators = vec![x, shadow];
    generators.extend((2..8).map(|i| Symbol::new(format!("y{i}"))));
    let left = p(
        &generators,
        5,
        &[
            (&[1, 0, 0, 0, 0, 0, 0, 0], 1),
            (&[0, 1, 0, 0, 0, 0, 0, 0], -1),
        ],
    );
    let right = p(
        &generators,
        5,
        &[
            (&[1, 0, 0, 0, 0, 0, 0, 0], 1),
            (&[0, 0, 1, 0, 0, 0, 0, 0], -1),
        ],
    );
    let expected = p(
        &generators[1..],
        5,
        &[(&[1, 0, 0, 0, 0, 0, 0], 1), (&[0, 1, 0, 0, 0, 0, 0], -1)],
    );
    let result = solve(Op::Resultant, &left, Some(&right));
    assert_eq!(result.result(), &expected);
    assert_ne!(result.result().generators()[0], generators[0]);
    assert_eq!(result.clone().into_result(), expected);
    let replay = verify(
        &Op::Resultant,
        &left,
        Some(&right),
        &Certificate {
            result: expected.terms().clone(),
        },
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(replay, result);
}

#[test]
fn forged_values_false_zero_and_changed_original_sources_cannot_gain_receipts() {
    let generators = ring(&["x", "y", "z"]);
    let left = p(&generators, 5, &[(&[2, 0, 0], 1), (&[0, 1, 0], -1)]);
    let right = p(&generators, 5, &[(&[1, 0, 0], 1), (&[0, 0, 1], -1)]);
    let result = solve(Op::Resultant, &left, Some(&right));
    let original = result.result().terms();
    let mut wrong = original.clone();
    *wrong.get_mut(&vec![0, 2]).unwrap() = 2;
    rejects(Op::Resultant, &left, Some(&right), &wrong);
    rejects(Op::Resultant, &left, Some(&right), &Terms::new());
    for malformed in [
        Terms::from([(vec![0, 0, 0], 1)]),
        Terms::from([(vec![0, 0], 5)]),
        Terms::from([(vec![0, 0], 0)]),
        Terms::from([(vec![65, 0], 1)]),
    ] {
        rejects(Op::Resultant, &left, Some(&right), &malformed);
    }
    let changed = p(&generators, 5, &[(&[2, 0, 0], 1), (&[0, 1, 0], -2)]);
    rejects(Op::Resultant, &changed, Some(&right), original);
    // A discriminant candidate cannot simply be accepted as the resultant
    // with the actual derivative: the original LC division and sign matter.
    let nonmonic = p(
        &generators,
        5,
        &[(&[2, 1, 0], 1), (&[1, 0, 0], 1), (&[0, 0, 1], 1)],
    );
    let derivative = p(&generators, 5, &[(&[1, 1, 0], 2), (&[0, 0, 0], 1)]);
    let discrim = solve(Op::Discriminant, &nonmonic, None);
    rejects(
        Op::Resultant,
        &nonmonic,
        Some(&derivative),
        discrim.result().terms(),
    );
}

#[test]
fn original_ring_and_arity_admission_precedes_zero_shortcuts() {
    let generators = ring(&["x", "y"]);
    let zero = scalar(&generators, 5, 0);
    let other_prime = scalar(&generators, 7, 0);
    let reversed = scalar(&[generators[1].clone(), generators[0].clone()], 5, 0);
    let shadow = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([17; 32]));
    let other_identity = scalar(&[shadow, generators[1].clone()], 5, 0);
    for other in [&other_prime, &reversed, &other_identity] {
        assert!(
            checked(
                &Op::Resultant,
                &zero,
                Some(other),
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(checked(&Op::Resultant, &zero, None, &mut Unbounded, &mut Unbounded).is_err());
    assert!(
        checked(
            &Op::Discriminant,
            &zero,
            Some(&zero),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let constant_field = scalar(&[], 5, 0);
    assert!(
        checked(
            &Op::Discriminant,
            &constant_field,
            None,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        checked(
            &Op::Resultant,
            &constant_field,
            Some(&constant_field),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Meter {
    steps: u64,
    bytes: u64,
    allocations: u64,
    checkpoints: usize,
    cancel: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.steps += amount,
            Dimension::MemoryBytes => self.bytes += amount,
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
fn oversized_sylvester_requests_refuse_before_allocating_determinant_tables() {
    let generators = ring(&["x", "y"]);
    let f = p(&generators, 5, &[(&[5, 0], 1), (&[0, 1], 1)]);
    let g = p(&generators, 5, &[(&[4, 0], 1), (&[0, 1], -1)]);
    for (operation, right) in [(Op::Resultant, Some(&g)), (Op::Discriminant, None)] {
        let mut generation = Meter::default();
        let error = checked(&operation, &f, right, &mut generation, &mut Unbounded).unwrap_err();
        assert!(error.to_string().contains("Sylvester"));
        assert_eq!(generation.bytes, 0);
        let mut verification = Meter::default();
        assert!(
            verify(
                &operation,
                &f,
                right,
                &Certificate {
                    result: Terms::new()
                },
                &mut verification
            )
            .is_err()
        );
        assert_eq!(verification.bytes, 0);
    }
    // Coefficient growth has a separate bound even when the matrix is tiny.
    // Res(y^64,x^d+1)=y^(64*d) exceeds the output bound for d=2 and the
    // intermediate bound for d=3, although both original sources are valid.
    let huge_coefficient = p(&generators, 5, &[(&[0, 64], 1)]);
    for degree in [2, 3] {
        let polynomial = p(&generators, 5, &[(&[degree, 0], 1), (&[0, 0], 1)]);
        assert!(
            checked(
                &Op::Resultant,
                &huge_coefficient,
                Some(&polynomial),
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let zero = scalar(&generators, 5, 0);
    assert!(solve(Op::Resultant, &zero, Some(&f)).result().is_zero());
}

fn original_pipeline(
    expressions: &[Expr; 2],
    generators: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Checked, String> {
    let left = source(&expressions[0], generators, 7, generator, verifier)
        .map_err(|error| error.to_string())?;
    let right = source(&expressions[1], generators, 7, generator, verifier)
        .map_err(|error| error.to_string())?;
    checked(
        &Op::Resultant,
        left.polynomial(),
        Some(right.polynomial()),
        generator,
        verifier,
    )
    .map_err(|error| error.to_string())
}

#[test]
fn original_expression_images_feed_the_same_checked_coefficient_ring() {
    let generators = ring(&["x", "y", "z"]);
    let expressions = [parse_raw("(x-y)**3").unwrap(), parse_raw("x-z").unwrap()];
    let result =
        original_pipeline(&expressions, &generators, &mut Unbounded, &mut Unbounded).unwrap();
    let expected = p(
        &generators[1..],
        7,
        &[(&[0, 3], -1), (&[1, 2], 3), (&[2, 1], -3), (&[3, 0], 1)],
    );
    // Res((x-y)^3,x-z)=(y-z)^3; f(z) differs by (-1)^3.
    assert_eq!(result.result(), &expected);
}

#[test]
fn complete_generation_verification_and_receipt_publication_share_cumulative_ledgers() {
    let generators = ring(&["x", "y", "z"]);
    let left = p(
        &generators,
        7,
        &[(&[2, 1, 0], 1), (&[1, 0, 0], 1), (&[0, 0, 1], 1)],
    );
    let right = p(&generators, 7, &[(&[1, 0, 0], 1), (&[0, 0, 1], -1)]);
    for (operation, rhs) in [(Op::Resultant, Some(&right)), (Op::Discriminant, None)] {
        let (mut generation, mut verification) = (Meter::default(), Meter::default());
        checked(&operation, &left, rhs, &mut generation, &mut verification).unwrap();
        for (dimension, generated, verified) in [
            (
                Dimension::ComputeSteps,
                generation.steps,
                verification.steps,
            ),
            (Dimension::MemoryBytes, generation.bytes, verification.bytes),
            (
                Dimension::AllocationCount,
                generation.allocations,
                verification.allocations,
            ),
        ] {
            assert!(generated > 1 && verified > 1);
            checked(
                &operation,
                &left,
                rhs,
                &mut cap(dimension, generated),
                &mut cap(dimension, verified),
            )
            .unwrap();
            assert!(
                checked(
                    &operation,
                    &left,
                    rhs,
                    &mut cap(dimension, generated - 1),
                    &mut Unbounded
                )
                .is_err()
            );
            assert!(
                checked(
                    &operation,
                    &left,
                    rhs,
                    &mut Unbounded,
                    &mut cap(dimension, verified - 1)
                )
                .is_err()
            );
            let mut partly_spent = cap(dimension, generated);
            BudgetMeter::charge(&mut partly_spent, dimension, 1).unwrap();
            assert!(checked(&operation, &left, rhs, &mut partly_spent, &mut Unbounded).is_err());
            let mut partly_spent = cap(dimension, verified);
            BudgetMeter::charge(&mut partly_spent, dimension, 1).unwrap();
            assert!(checked(&operation, &left, rhs, &mut Unbounded, &mut partly_spent).is_err());
        }
        for (is_generator, checkpoints) in [
            (true, generation.checkpoints),
            (false, verification.checkpoints),
        ] {
            for stop in [1, checkpoints / 2, checkpoints] {
                let mut meter = Meter {
                    cancel: Some(stop),
                    ..Meter::default()
                };
                let result = if is_generator {
                    checked(&operation, &left, rhs, &mut meter, &mut Unbounded)
                } else {
                    checked(&operation, &left, rhs, &mut Unbounded, &mut meter)
                };
                assert!(
                    result.is_err(),
                    "{operation:?}, generator={is_generator}, checkpoint={stop}"
                );
            }
        }
    }
}

#[test]
fn original_source_checks_and_elimination_do_not_reset_either_callers_budget() {
    let generators = ring(&["x", "y", "z"]);
    let expressions = [parse_raw("(x-y)**2").unwrap(), parse_raw("x-z").unwrap()];
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    original_pipeline(
        &expressions,
        &generators,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for (dimension, generated, verified) in [
        (
            Dimension::ComputeSteps,
            generation.steps,
            verification.steps,
        ),
        (Dimension::MemoryBytes, generation.bytes, verification.bytes),
        (
            Dimension::AllocationCount,
            generation.allocations,
            verification.allocations,
        ),
    ] {
        original_pipeline(
            &expressions,
            &generators,
            &mut cap(dimension, generated),
            &mut cap(dimension, verified),
        )
        .unwrap();
        assert!(
            original_pipeline(
                &expressions,
                &generators,
                &mut cap(dimension, generated - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            original_pipeline(
                &expressions,
                &generators,
                &mut Unbounded,
                &mut cap(dimension, verified - 1)
            )
            .is_err()
        );
    }
}

#[test]
fn failed_allocation_charge_preserves_the_other_resource_counter() {
    let generators = ring(&["x", "y"]);
    let left = p(&generators, 5, &[(&[2, 0], 1), (&[0, 1], 1)]);
    let mut generation = cap(Dimension::AllocationCount, 0);
    let before = generation.remaining(Dimension::MemoryBytes);
    assert!(
        checked(
            &Op::Discriminant,
            &left,
            None,
            &mut generation,
            &mut Unbounded
        )
        .is_err()
    );
    assert_eq!(generation.remaining(Dimension::MemoryBytes), before);
    let mut verification = cap(Dimension::AllocationCount, 0);
    let before = verification.remaining(Dimension::MemoryBytes);
    assert!(
        checked(
            &Op::Discriminant,
            &left,
            None,
            &mut Unbounded,
            &mut verification
        )
        .is_err()
    );
    assert_eq!(verification.remaining(Dimension::MemoryBytes), before);
}
