//! Whole-ideal inversion, independently derived identities and forged witnesses.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity, parse};
use fsym_polys::{
    PrimeFieldIdealInverseCertificate as Certificate,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    TermOrder, VerifiedPrimeFieldIdealInverse as Inverse,
    checked_prime_field_groebner_basis as groebner, checked_prime_field_ideal_inverse as checked,
    checked_prime_field_multivariate_source, verify_prime_field_ideal_inverse as verify,
};
use std::sync::Arc;

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
    .polynomial()
    .clone()
}

fn run(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
) -> Inverse {
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

fn rejected(result: &Inverse, certificate: &Certificate) {
    assert!(
        verify(
            result.generators(),
            result.prime(),
            result.inputs(),
            result.element(),
            result.order(),
            certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn positive_dimensional_ideals_have_exact_field_inverses_in_every_order() {
    let generators = ring(&["x", "y", "z"]);
    for prime in [2, 3, 5, 65521] {
        let inputs = [
            p("x*y-1", &generators, prime),
            p("y*z-1", &generators, prime),
        ];
        let element = p("x", &generators, prime);
        for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
            let result = run(&generators, prime, &inputs, &element, order);
            assert_eq!(result.inverse(), Some(&p("y", &generators, prime)));
            assert!(result.is_invertible());
            assert_eq!(result.generators(), generators);
            assert_eq!(result.prime(), prime);
            assert_eq!(result.inputs(), inputs);
            assert_eq!(result.element(), &element);
            assert_eq!(result.order(), order);
            let replay = verify(
                &generators,
                prime,
                &inputs,
                &element,
                order,
                result.certificate(),
                &mut Unbounded,
            )
            .unwrap();
            assert_eq!(replay.into_inverse(), result.clone().into_inverse());
        }
    }
}

#[test]
fn nilpotent_inverse_keeps_all_multiplicities_and_characteristic_cancellations() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 7] {
        let inputs = [p("x**2", &generators, prime), p("y**2", &generators, prime)];
        let element = p("1+x+y", &generators, prime);
        let expected = p("1-x-y+2*x*y", &generators, prime);
        for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
            assert_eq!(
                run(&generators, prime, &inputs, &element, order).inverse(),
                Some(&expected)
            );
        }
    }
}

#[test]
fn characteristic_changes_invertibility_without_adding_field_point_equations() {
    let generators = ring(&["x"]);
    for prime in [2, 3, 5, 7] {
        let inputs = [p("x**2+1", &generators, prime)];
        let result = run(
            &generators,
            prime,
            &inputs,
            &p("x+1", &generators, prime),
            TermOrder::Lex,
        );
        if prime == 2 {
            assert!(result.inverse().is_none());
        } else {
            let half = prime.div_ceil(2);
            assert_eq!(
                result.inverse(),
                Some(&p(&format!("{half}*(1-x)"), &generators, prime))
            );
        }
        // The proper ideal x^2+1 over F_3 has no F_3-valued points. Its
        // quotient is still a nonzero algebra; x has inverse -x.
        assert_eq!(
            run(
                &generators,
                prime,
                &inputs,
                &p("x", &generators, prime),
                TermOrder::Lex
            )
            .inverse(),
            Some(&p("-x", &generators, prime))
        );
    }
}

#[test]
fn complete_nonunits_include_elements_that_are_not_zero_divisors() {
    let generators = ring(&["x", "y"]);
    for (sources, value) in [
        (vec![], "x"),
        (vec![], "0"),
        (vec!["x**2"], "x"),
        (vec!["x*y"], "x"),
        (vec!["x*y-1"], "x+y"),
    ] {
        let inputs: Vec<_> = sources
            .iter()
            .map(|source| p(source, &generators, 5))
            .collect();
        let result = run(
            &generators,
            5,
            &inputs,
            &p(value, &generators, 5),
            TermOrder::Lex,
        );
        assert!(!result.is_invertible());
        assert!(result.inverse().is_none());
        assert!(result.certificate().ideal_basis.is_none());
        assert!(result.certificate().bezout_coefficients.is_empty());
    }
}

#[test]
fn zero_unit_and_explicit_constant_fields_preserve_exact_conventions() {
    for generators in [Vec::new(), ring(&["x", "y"])] {
        for prime in [2, 3, 5, 65521] {
            for inputs in [Vec::new(), vec![p("0", &generators, prime)]] {
                let result = run(
                    &generators,
                    prime,
                    &inputs,
                    &p("-1", &generators, prime),
                    TermOrder::Lex,
                );
                assert_eq!(result.inverse(), Some(&p("-1", &generators, prime)));
                assert!(
                    run(
                        &generators,
                        prime,
                        &inputs,
                        &p("0", &generators, prime),
                        TermOrder::Lex
                    )
                    .inverse()
                    .is_none()
                );
            }
            for value in ["0", "1", "-2"] {
                let inputs = [p("0", &generators, prime), p("1", &generators, prime)];
                assert_eq!(
                    run(
                        &generators,
                        prime,
                        &inputs,
                        &p(value, &generators, prime),
                        TermOrder::Lex
                    )
                    .inverse(),
                    Some(&p("0", &generators, prime))
                );
            }
        }
    }
}

#[test]
fn zero_redundant_and_scaled_original_inputs_keep_exact_cofactor_positions() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        p("0", &generators, 7),
        p("2*(x*y-1)", &generators, 7),
        p("3*(x*y-1)", &generators, 7),
    ];
    let element = p("5*x", &generators, 7);
    let result = run(&generators, 7, &inputs, &element, TermOrder::Lex);
    assert_eq!(result.inverse(), Some(&p("3*y", &generators, 7)));
    assert_eq!(result.certificate().bezout_coefficients.len(), 4);
    let mut changed = inputs.clone();
    changed[1] = p("4*(x*y-1)", &generators, 7);
    assert!(
        verify(
            &generators,
            7,
            &changed,
            &element,
            TermOrder::Lex,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn exact_typed_ring_identity_does_not_use_printed_names() {
    let generators = vec![
        Symbol::new("x"),
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([31; 32])),
    ];
    let inputs = [Polynomial::new(
        &generators,
        5,
        &Terms::from([(vec![1, 1], 1), (vec![0, 0], 4)]),
    )
    .unwrap()];
    let element = Polynomial::new(&generators, 5, &Terms::from([(vec![1, 0], 1)])).unwrap();
    let expected = Polynomial::new(&generators, 5, &Terms::from([(vec![0, 1], 1)])).unwrap();
    let result = run(&generators, 5, &inputs, &element, TermOrder::Lex);
    assert_eq!(result.inverse(), Some(&expected));
    let reversed = vec![generators[1].clone(), generators[0].clone()];
    assert!(
        verify(
            &reversed,
            5,
            &inputs,
            &element,
            TermOrder::Lex,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn forged_cofactors_inverse_and_witness_dimensions_are_rejected() {
    let generators = ring(&["x", "y"]);
    let result = run(
        &generators,
        5,
        &[p("x*y-1", &generators, 5)],
        &p("x", &generators, 5),
        TermOrder::Lex,
    );
    let mut bad = result.certificate().clone();
    bad.bezout_coefficients[0].clear();
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.inverse = Some(p("y+1", &generators, 5));
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.bezout_coefficients.pop();
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.reduction_quotients.push(Terms::new());
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.bezout_coefficients[0].insert(vec![0, 0], 5);
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.inverse = Some(p("y", &generators, 7));
    rejected(&result, &bad);
}

#[test]
fn algebraically_correct_but_unreduced_inverse_is_rejected() {
    let generators = ring(&["x"]);
    let result = run(
        &generators,
        5,
        &[p("x**2-2", &generators, 5)],
        &p("x", &generators, 5),
        TermOrder::Lex,
    );
    assert_eq!(result.inverse(), Some(&p("3*x", &generators, 5)));
    let mut bad = result.certificate().clone();
    bad.inverse = Some(p("3*x+x**2-2", &generators, 5));
    let value = (bad.reduction_quotients[0]
        .get([0].as_slice())
        .copied()
        .unwrap_or(0)
        + 4)
        % 5;
    if value == 0 {
        bad.reduction_quotients[0].remove([0].as_slice());
    } else {
        bad.reduction_quotients[0].insert(vec![0], value);
    }
    rejected(&result, &bad);
}

#[test]
fn complete_unit_and_nonunit_conclusions_cannot_be_swapped() {
    let generators = ring(&["x"]);
    let unit = run(
        &generators,
        5,
        &[p("x**2-2", &generators, 5)],
        &p("x", &generators, 5),
        TermOrder::Lex,
    );
    let mut bad = unit.certificate().clone();
    bad.inverse = None;
    bad.ideal_basis = None;
    bad.bezout_coefficients.clear();
    bad.reduction_quotients.clear();
    rejected(&unit, &bad);
    let nonunit = run(&generators, 5, &[], &p("x", &generators, 5), TermOrder::Lex);
    let mut bad = nonunit.certificate().clone();
    bad.inverse = Some(p("0", &generators, 5));
    rejected(&nonunit, &bad);
}

#[test]
fn accepted_lower_receipts_are_bound_to_source_prime_and_order() {
    let generators = ring(&["x", "y"]);
    let inputs = [p("x*y-1", &generators, 5)];
    let element = p("x", &generators, 5);
    let result = run(&generators, 5, &inputs, &element, TermOrder::Lex);
    let mut bad = result.certificate().clone();
    bad.augmented_basis = Arc::new(
        groebner(
            &generators,
            5,
            &[p("2*(x*y-1)", &generators, 5), element.clone()],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.ideal_basis = Some(Arc::new(
        groebner(
            &generators,
            5,
            &inputs,
            TermOrder::DegLex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    ));
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.augmented_basis = Arc::new(
        groebner(
            &generators,
            7,
            &[p("x*y-1", &generators, 7), p("x", &generators, 7)],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &bad);
}

#[test]
fn original_admission_and_input_bounds_precede_unit_shortcuts() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let one = p("1", &generators, 65521);
    let zero = p("0", &generators, 65521);
    let mut inputs = vec![zero; 30];
    inputs.push(p("x0**64", &generators, 65521));
    let result = run(
        &generators,
        65521,
        &inputs,
        &p("1-x0", &generators, 65521),
        TermOrder::Lex,
    );
    assert_eq!(result.inverse().unwrap().terms().len(), 64);
    for exponent in 0..64 {
        let mut powers = vec![0; 8];
        powers[0] = exponent;
        assert_eq!(result.inverse().unwrap().terms().get(&powers), Some(&1));
    }
    assert!(
        checked(
            &generators,
            65521,
            &vec![one.clone(); 32],
            &one,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    for prime in [0, 1, 4, 9, 65523] {
        assert!(
            checked(
                &generators,
                prime,
                std::slice::from_ref(&one),
                &one,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let wrong = p("0", &generators, 7);
    assert!(
        checked(
            &generators,
            65521,
            std::slice::from_ref(&one),
            &wrong,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Count {
    steps: u64,
    bytes: u64,
    checkpoints: usize,
    cancel: Option<usize>,
}

#[test]
fn all_quadratic_quotients_and_linear_elements_match_independent_field_determinants() {
    let generators = ring(&["x"]);
    let mut cases = 0;
    for prime in [2, 3, 5] {
        for a in 0..prime {
            for b in 0..prime {
                let mut terms = Terms::from([(vec![2], 1)]);
                if a != 0 {
                    terms.insert(vec![1], a);
                }
                if b != 0 {
                    terms.insert(vec![0], b);
                }
                let inputs = [Polynomial::new(&generators, prime, &terms).unwrap()];
                for c in 0..prime {
                    for d in 0..prime {
                        let mut terms = Terms::new();
                        if c != 0 {
                            terms.insert(vec![1], c);
                        }
                        if d != 0 {
                            terms.insert(vec![0], d);
                        }
                        let element = Polynomial::new(&generators, prime, &terms).unwrap();
                        let result = run(&generators, prime, &inputs, &element, TermOrder::Lex);
                        // In basis (1,x), multiplication by c*x+d has matrix
                        // [[d,-b*c],[c,d-a*c]]. This independent 2x2 identity
                        // covers split, irreducible and repeated-root ideals.
                        let top = (d + prime - a * c % prime) % prime;
                        let determinant = (d * top + b * c * c) % prime;
                        let expected = if determinant == 0 {
                            None
                        } else {
                            let scale = (1..prime)
                                .find(|value| value * determinant % prime == 1)
                                .unwrap();
                            let mut terms = Terms::new();
                            let constant = top * scale % prime;
                            let linear = (prime - c) * scale % prime;
                            if constant != 0 {
                                terms.insert(vec![0], constant);
                            }
                            if linear != 0 {
                                terms.insert(vec![1], linear);
                            }
                            Some(Polynomial::new(&generators, prime, &terms).unwrap())
                        };
                        assert_eq!(
                            result.into_inverse(),
                            expected,
                            "p={prime}, q=x^2+{a}*x+{b}, f={c}*x+{d}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 722);
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
fn cumulative_phase_budgets_and_last_checkpoint_failures_never_become_nonunits() {
    let generators = ring(&["x", "y"]);
    for (inputs, element) in [
        (vec![p("x**2", &generators, 5)], p("1-x", &generators, 5)),
        (Vec::new(), p("x", &generators, 5)),
    ] {
        let (mut generator, mut verifier) = (Count::default(), Count::default());
        checked(
            &generators,
            5,
            &inputs,
            &element,
            TermOrder::Lex,
            &mut generator,
            &mut verifier,
        )
        .unwrap();
        for (dimension, generated, verified) in [
            (Dimension::ComputeSteps, generator.steps, verifier.steps),
            (Dimension::MemoryBytes, generator.bytes, verifier.bytes),
        ] {
            assert!(generated > 0 && verified > 0);
            checked(
                &generators,
                5,
                &inputs,
                &element,
                TermOrder::Lex,
                &mut cap(dimension, generated),
                &mut cap(dimension, verified),
            )
            .unwrap();
            assert!(
                checked(
                    &generators,
                    5,
                    &inputs,
                    &element,
                    TermOrder::Lex,
                    &mut cap(dimension, generated - 1),
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
                    &mut cap(dimension, verified - 1)
                )
                .is_err()
            );
        }
        for (generation, checkpoints) in
            [(true, generator.checkpoints), (false, verifier.checkpoints)]
        {
            for at in [1, checkpoints / 2, checkpoints] {
                let mut stopped = Count {
                    cancel: Some(at),
                    ..Count::default()
                };
                let result = if generation {
                    checked(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        TermOrder::Lex,
                        &mut stopped,
                        &mut Unbounded,
                    )
                } else {
                    checked(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        TermOrder::Lex,
                        &mut Unbounded,
                        &mut stopped,
                    )
                };
                assert!(result.is_err(), "generation={generation}, checkpoint={at}");
            }
        }
    }
}
