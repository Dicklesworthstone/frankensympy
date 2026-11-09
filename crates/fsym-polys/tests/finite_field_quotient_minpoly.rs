//! Complete annihilator ideals, independent coefficient expectations, exact
//! composition adversaries and terminal resource boundaries over F_p.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity, parse};
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    PrimeFieldQuotientMinimalPolynomialCertificate as Certificate,
    VerifiedPrimeFieldQuotientMinimalPolynomial as Verified,
    checked_prime_field_elimination_ideal as eliminate,
    checked_prime_field_ideal_intersection as intersect, checked_prime_field_multivariate_source,
    checked_prime_field_quotient_minimal_polynomial as calculate,
    verify_prime_field_quotient_minimal_polynomial as verify,
};
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn polynomial(generators: &[Symbol], prime: u64, entries: &[(&[u32], u64)]) -> Polynomial {
    let terms = entries.iter().map(|(m, c)| (m.to_vec(), *c)).collect();
    Polynomial::new(generators, prime, &terms).unwrap()
}

fn poly(source: &str, generators: &[Symbol], prime: u64) -> Polynomial {
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

fn run(generators: &[Symbol], prime: u64, sources: &[&str], element: &str) -> Verified {
    let inputs: Vec<_> = sources.iter().map(|s| poly(s, generators, prime)).collect();
    calculate(
        generators,
        prime,
        &inputs,
        &poly(element, generators, prime),
        &Symbol::new("t"),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn residues(values: &[i64], prime: u64) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.rem_euclid(prime as i64) as u64)
        .collect()
}

fn reject(result: &Verified, certificate: &Certificate) {
    assert!(
        verify(
            result.generators(),
            result.prime(),
            result.inputs(),
            result.element(),
            result.parameter(),
            certificate,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn finite_quotients_return_minimal_not_characteristic_polynomials() {
    let generators = ring(&["x", "y"]);
    for prime in [5, 7] {
        let sum = run(&generators, prime, &["x**2-2", "y**2-3"], "x+y");
        assert_eq!(
            sum.coefficients(),
            Some(residues(&[1, 0, -10, 0, 1], prime).as_slice())
        );
        let x = run(&generators, prime, &["x**2-2", "y**2-3"], "x");
        assert_eq!(
            x.coefficients(),
            Some(residues(&[-2, 0, 1], prime).as_slice())
        );
        assert_eq!(x.generators(), generators);
        assert_eq!(x.prime(), prime);
        assert_eq!(x.element(), &poly("x", &generators, prime));
        assert_eq!(x.parameter(), &Symbol::new("t"));
        let replay = verify(
            &generators,
            prime,
            x.inputs(),
            x.element(),
            x.parameter(),
            x.certificate(),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(
            replay.into_coefficients(),
            x.coefficients().map(<[_]>::to_vec)
        );
    }
}

#[test]
fn nilpotent_multiplicities_and_characteristic_changes_are_preserved() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        let sum = run(&generators, prime, &["x**2", "y**2"], "x+y");
        let shifted = run(&generators, prime, &["x**2", "y**2"], "1+x+y");
        if prime == 2 {
            // (x+y)^2=0 in characteristic two; in odd characteristic 2xy!=0.
            assert_eq!(sum.coefficients(), Some([0, 0, 1].as_slice()));
            assert_eq!(shifted.coefficients(), Some([1, 0, 1].as_slice()));
        } else {
            assert_eq!(sum.coefficients(), Some([0, 0, 0, 1].as_slice()));
            assert_eq!(
                shifted.coefficients(),
                Some(residues(&[-1, 3, -3, 1], prime).as_slice())
            );
        }
        assert_eq!(
            run(&generators, prime, &["x**2", "y**2"], "x*y").coefficients(),
            Some([0, 0, 1].as_slice())
        );
        assert_eq!(
            run(&generators, prime, &["x**2", "y**2"], "x**2+y**2").coefficients(),
            Some([0, 1].as_slice())
        );
    }
}

#[test]
fn positive_dimensional_quotients_prove_both_algebraicity_and_transcendence() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        assert_eq!(
            run(&generators, prime, &["x**2"], "x").coefficients(),
            Some([0, 0, 1].as_slice())
        );
        for (sources, element) in [
            (vec!["x**2"], "y"),
            (vec!["x**2"], "x+y"),
            (vec![], "x"),
            (vec!["x*y-1"], "x"),
            (vec!["x*y"], "x+y"),
        ] {
            let accepted = run(&generators, prime, &sources, element);
            assert!(accepted.coefficients().is_none());
            assert!(accepted.certificate().elimination.basis().is_empty());
        }
        assert_eq!(
            run(&generators, prime, &["x*y-1"], "x*y").coefficients(),
            Some([prime - 1, 1].as_slice())
        );
    }
}

#[test]
fn polynomial_quotients_do_not_add_finite_point_equations_or_select_a_component() {
    let generators = ring(&["x"]);
    // Every F_2 point satisfies x^2-x, but the indeterminate remains
    // transcendental in the polynomial ring F_2[x].
    assert!(run(&generators, 2, &[], "x").coefficients().is_none());
    // x^2+1 has no F_3 point but its quotient is a nonzero quadratic field.
    assert_eq!(
        run(&generators, 3, &["x**2+1"], "x").coefficients(),
        Some([1, 0, 1].as_slice())
    );
    for prime in [2, 5] {
        assert_eq!(
            run(&generators, prime, &["x**2-x"], "x").coefficients(),
            Some([0, prime - 1, 1].as_slice())
        );
    }
}

#[test]
fn characteristic_can_change_a_proper_quotient_into_the_zero_ring() {
    let generators = ring(&["x"]);
    for prime in [2, 3, 5] {
        let accepted = run(&generators, prime, &["x**2+1", "x+1"], "x");
        if prime == 2 {
            assert_eq!(accepted.coefficients(), Some([1, 1].as_slice()));
        } else {
            assert_eq!(accepted.coefficients(), Some([1].as_slice()));
        }
    }
}

#[test]
fn unit_zero_and_constant_fields_publish_explicit_complete_generators() {
    for generators in [ring(&["x"]), vec![]] {
        for prime in [2, 5, 65521] {
            for value in [0, 1, prime - 1] {
                let source = value.to_string();
                let accepted = run(&generators, prime, &[], &source);
                assert_eq!(
                    accepted.coefficients(),
                    Some([(prime - value) % prime, 1].as_slice())
                );
                assert_eq!(accepted.inputs().len(), 0);
                assert_eq!(accepted.generators(), generators);
                assert_eq!(
                    run(&generators, prime, &["1"], &source).coefficients(),
                    Some([1].as_slice())
                );
            }
        }
    }
    assert_eq!(
        run(&ring(&["x"]), 5, &["0", "0"], "0").coefficients(),
        Some([0, 1].as_slice())
    );
}

#[test]
fn original_input_order_zero_entries_and_scaling_remain_bound() {
    let generators = ring(&["y", "x"]);
    let accepted = run(&generators, 5, &["0", "2*x**2-4", "3*x**2-6", "y-x"], "x+y");
    assert_eq!(accepted.coefficients(), Some([2, 0, 1].as_slice()));
    assert_eq!(accepted.inputs().len(), 4);
    let mut swapped = accepted.inputs().to_vec();
    swapped.swap(1, 2);
    let mut rescaled = accepted.inputs().to_vec();
    rescaled[1] = poly("x**2-2", &generators, 5);
    let mut extra_zero = accepted.inputs().to_vec();
    extra_zero.push(poly("0", &generators, 5));
    for inputs in [
        swapped,
        rescaled,
        extra_zero,
        accepted.inputs()[1..].to_vec(),
    ] {
        assert!(
            verify(
                &generators,
                5,
                &inputs,
                accepted.element(),
                accepted.parameter(),
                accepted.certificate(),
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn same_printed_parameter_preserves_distinct_typed_identity() {
    let x = Symbol::new("a");
    let t = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([71; 32]));
    let generators = [x];
    let inputs = [polynomial(&generators, 5, &[(&[2], 1), (&[0], 3)])];
    let element = polynomial(&generators, 5, &[(&[1], 1)]);
    let accepted = calculate(
        &generators,
        5,
        &inputs,
        &element,
        &t,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(accepted.parameter(), &t);
    assert_eq!(accepted.coefficients(), Some([3, 0, 1].as_slice()));
    for wrong in [
        generators[0].clone(),
        Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([72; 32])),
    ] {
        assert!(
            verify(
                &generators,
                5,
                &inputs,
                &element,
                &wrong,
                accepted.certificate(),
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn wrong_annihilators_nonminimal_multiples_and_noncanonical_coefficients_are_rejected() {
    let generators = ring(&["x"]);
    let accepted = run(&generators, 5, &["x**3"], "x");
    for coefficients in [
        None,
        Some(vec![]),
        Some(vec![0, 1]),
        Some(vec![0, 0, 0, 0, 1]),
        Some(vec![0, 0, 0, 2]),
        Some(vec![0, 0, 0, 1, 0]),
        Some(vec![1, 0, 0, 1]),
        Some(vec![5, 0, 0, 1]),
        Some(vec![0, 0, 0, 6]),
        Some(vec![1; 66]),
    ] {
        let mut changed = accepted.certificate().clone();
        changed.coefficients = coefficients;
        reject(&accepted, &changed);
    }
    let transcendental = run(&generators, 5, &[], "x");
    let mut invented = transcendental.certificate().clone();
    invented.coefficients = Some(vec![0, 1]);
    reject(&transcendental, &invented);
}

#[test]
fn identical_final_coefficients_do_not_authorize_another_element_or_ideal() {
    let generators = ring(&["x", "y"]);
    let accepted = run(&generators, 7, &["x**2-2", "y**2-3"], "x+y");
    for (sources, element) in [
        (vec!["x**2-2", "y**2-3"], "x-y"),
        (vec!["x**2-3", "y**2-2"], "x+y"),
        (vec!["x**2-2", "y**2-3"], "-x-y"),
    ] {
        let replacement = run(&generators, 7, &sources, element);
        assert_eq!(accepted.coefficients(), replacement.coefficients());
        let mut changed = accepted.certificate().clone();
        changed.elimination = Arc::clone(&replacement.certificate().elimination);
        reject(&accepted, &changed);
    }
}

#[test]
fn genuine_elimination_receipts_must_contain_exactly_the_requested_t_minus_f() {
    let generators = ring(&["x"]);
    let ambient = ring(&["x", "t"]);
    let accepted = run(&generators, 5, &["x**2"], "x");
    for (relation, coefficients) in [
        ("t+x", Some(vec![0, 0, 1])),
        ("2*t-2*x", Some(vec![0, 0, 1])),
        ("t**2-x", Some(vec![0, 0, 0, 0, 1])),
        ("t-x+x**2", Some(vec![0, 0, 1])),
        ("x", None),
    ] {
        let inputs = [poly("x**2", &ambient, 5), poly(relation, &ambient, 5)];
        let elimination = eliminate(
            &ambient,
            5,
            &inputs,
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        reject(
            &accepted,
            &Certificate {
                elimination: Arc::new(elimination),
                coefficients,
            },
        );
    }
}

#[test]
fn receipt_operation_field_parameter_and_eliminated_coordinates_are_bound() {
    let generators = ring(&["x"]);
    let ambient = ring(&["x", "t"]);
    let accepted = run(&generators, 5, &["1"], "x");
    let other_field = run(&generators, 7, &["1"], "x");
    reject(&accepted, other_field.certificate());
    let augmented = [poly("1", &ambient, 5), poly("t-x", &ambient, 5)];
    for elimination in [
        eliminate(
            &ambient,
            5,
            &augmented,
            &ambient[1..],
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        intersect(
            &ambient,
            5,
            &augmented,
            &augmented,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    ] {
        reject(
            &accepted,
            &Certificate {
                elimination: Arc::new(elimination),
                coefficients: Some(vec![1]),
            },
        );
    }
    assert!(
        verify(
            &generators,
            5,
            accepted.inputs(),
            accepted.element(),
            &Symbol::new("s"),
            accepted.certificate(),
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn ring_field_and_original_sources_are_validated_before_unit_shortcuts() {
    let generators = ring(&["x", "y"]);
    let unit = poly("1", &generators, 5);
    let element = poly("x", &generators, 5);
    for invalid in [
        poly("0", &generators, 7),
        poly("0", &ring(&["y", "x"]), 5),
        polynomial(
            &[
                Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([4; 32])),
                Symbol::new("y"),
            ],
            5,
            &[],
        ),
        poly("0", &[], 5),
    ] {
        assert!(
            calculate(
                &generators,
                5,
                &[unit.clone(), invalid.clone()],
                &element,
                &Symbol::new("t"),
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
        assert!(
            calculate(
                &generators,
                5,
                std::slice::from_ref(&unit),
                &invalid,
                &Symbol::new("t"),
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
    for prime in [0, 1, 4, 65537] {
        assert!(
            calculate(
                &generators,
                prime,
                std::slice::from_ref(&unit),
                &element,
                &Symbol::new("t"),
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn reserved_variable_input_and_name_bounds_apply_before_ambient_allocation() {
    let generators = ring(&["a", "b", "c", "d", "e", "f", "g"]);
    let zero = polynomial(&generators, 5, &[]);
    let accepted = calculate(
        &generators,
        5,
        &vec![zero.clone(); 31],
        &zero,
        &Symbol::new("t"),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(accepted.coefficients(), Some([0, 1].as_slice()));
    assert_eq!(accepted.inputs().len(), 31);
    assert_eq!(accepted.generators().len(), 7);
    assert!(
        calculate(
            &generators,
            5,
            &vec![zero.clone(); 32],
            &zero,
            &Symbol::new("t"),
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let large_ring = ring(&["a", "b", "c", "d", "e", "f", "g", "h"]);
    let zero = polynomial(&large_ring, 5, &[]);
    assert!(
        calculate(
            &large_ring,
            5,
            &[],
            &zero,
            &Symbol::new("t"),
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let large_name = [Symbol::new("x".repeat(65535))];
    let zero = polynomial(&large_name, 5, &[]);
    for parameter in [Symbol::new("tt"), Symbol::new("t".repeat(65537))] {
        let mut generation = Count::default();
        assert!(
            calculate(
                &large_name,
                5,
                &[],
                &zero,
                &parameter,
                &mut generation,
                &mut Unbounded,
            )
            .is_err()
        );
        assert_eq!(generation.bytes, 0);
    }
}

#[test]
fn element_term_cap_reserves_the_exact_parameter_relation_even_in_the_zero_ring() {
    let generators = ring(&["x", "y", "z"]);
    let mut terms = Terms::new();
    for x in 0..16 {
        for y in 0..16 {
            for z in 0..16 {
                terms.insert(vec![x, y, z], 1);
            }
        }
    }
    let element = Polynomial::new(&generators, 5, &terms).unwrap();
    assert_eq!(element.terms().len(), 4096);
    let mut generation = Count::default();
    assert!(
        calculate(
            &generators,
            5,
            &[poly("1", &generators, 5)],
            &element,
            &Symbol::new("t"),
            &mut generation,
            &mut Unbounded,
        )
        .is_err()
    );
    assert_eq!(generation.bytes, 0);
}

#[test]
fn highest_admitted_degree_retains_every_nilpotent_power() {
    let accepted = run(&ring(&["x"]), 2, &["x**64"], "x");
    let mut expected = vec![0; 65];
    expected[64] = 1;
    assert_eq!(accepted.coefficients(), Some(expected.as_slice()));
}

#[test]
fn all_small_monic_quadratics_match_an_independent_affine_change_of_variable() {
    let generators = ring(&["x"]);
    // If P=x^2+c1*x+c0 and a!=0, substitution x=(t-b)/a gives the
    // exact monic minimal polynomial of a*x+b, even when P has a double root.
    for prime in [2, 3] {
        for c0 in 0..prime {
            for c1 in 0..prime {
                let input = polynomial(&generators, prime, &[(&[2], 1), (&[1], c1), (&[0], c0)]);
                for a in 1..prime {
                    for b in 0..prime {
                        let element = polynomial(&generators, prime, &[(&[1], a), (&[0], b)]);
                        let accepted = calculate(
                            &generators,
                            prime,
                            std::slice::from_ref(&input),
                            &element,
                            &Symbol::new("t"),
                            &mut Unbounded,
                            &mut Unbounded,
                        )
                        .unwrap();
                        let (a, b, c0, c1) = (a as i64, b as i64, c0 as i64, c1 as i64);
                        let expected =
                            residues(&[b * b - a * c1 * b + a * a * c0, a * c1 - 2 * b, 1], prime);
                        assert_eq!(accepted.coefficients(), Some(expected.as_slice()));
                    }
                }
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
fn budgets_accumulate_across_elimination_and_include_final_copying() {
    let generators = ring(&["x"]);
    let element = poly("x", &generators, 5);
    let parameter = Symbol::new("t");
    for inputs in [vec![poly("x**3", &generators, 5)], vec![]] {
        let (mut generation, mut verification) = (Count::default(), Count::default());
        let accepted = calculate(
            &generators,
            5,
            &inputs,
            &element,
            &parameter,
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
        ] {
            assert!(generated > 0 && verified > 0);
            calculate(
                &generators,
                5,
                &inputs,
                &element,
                &parameter,
                &mut cap(dimension, generated),
                &mut cap(dimension, verified),
            )
            .unwrap();
            for generating in [true, false] {
                let amount = if generating { generated } else { verified };
                let mut stopped = cap(dimension, amount - 1);
                let result = if generating {
                    calculate(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        &parameter,
                        &mut stopped,
                        &mut Unbounded,
                    )
                } else {
                    calculate(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        &parameter,
                        &mut Unbounded,
                        &mut stopped,
                    )
                };
                assert!(result.is_err());
                let mut precharged = cap(dimension, amount);
                precharged.charge(dimension, 1).unwrap();
                let result = if generating {
                    calculate(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        &parameter,
                        &mut precharged,
                        &mut Unbounded,
                    )
                } else {
                    calculate(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        &parameter,
                        &mut Unbounded,
                        &mut precharged,
                    )
                };
                assert!(result.is_err());
            }
        }
        let mut replay = Count::default();
        verify(
            &generators,
            5,
            &inputs,
            &element,
            &parameter,
            accepted.certificate(),
            &mut replay,
        )
        .unwrap();
        assert!(
            verify(
                &generators,
                5,
                &inputs,
                &element,
                &parameter,
                accepted.certificate(),
                &mut cap(Dimension::MemoryBytes, replay.bytes - 1)
            )
            .is_err()
        );
    }
}

#[test]
fn cancellation_cannot_become_transcendence_in_either_phase_or_at_publication() {
    let generators = ring(&["x"]);
    let element = poly("x", &generators, 5);
    let parameter = Symbol::new("t");
    for inputs in [vec![poly("x**2", &generators, 5)], vec![]] {
        let (mut generation, mut verification) = (Count::default(), Count::default());
        let accepted = calculate(
            &generators,
            5,
            &inputs,
            &element,
            &parameter,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        for (generating, checkpoints) in [
            (true, generation.checkpoints),
            (false, verification.checkpoints),
        ] {
            for at in [1, checkpoints / 2, checkpoints] {
                let mut stopped = Count {
                    cancel: Some(at),
                    ..Count::default()
                };
                let result = if generating {
                    calculate(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        &parameter,
                        &mut stopped,
                        &mut Unbounded,
                    )
                } else {
                    calculate(
                        &generators,
                        5,
                        &inputs,
                        &element,
                        &parameter,
                        &mut Unbounded,
                        &mut stopped,
                    )
                };
                assert!(result.is_err(), "generating={generating}, checkpoint={at}");
            }
        }
        let mut replay = Count::default();
        verify(
            &generators,
            5,
            &inputs,
            &element,
            &parameter,
            accepted.certificate(),
            &mut replay,
        )
        .unwrap();
        let mut stopped = Count {
            cancel: Some(replay.checkpoints),
            ..Count::default()
        };
        assert!(
            verify(
                &generators,
                5,
                &inputs,
                &element,
                &parameter,
                accepted.certificate(),
                &mut stopped
            )
            .is_err()
        );
    }
}
