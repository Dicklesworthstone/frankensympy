//! Full ideal equalities over F_p, including nonreduced schemes, original
//! ambient construction binding and terminal whole-operation resource limits.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    IdealOperation, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder, VerifiedPrimeFieldIdealOperation,
    checked_prime_field_groebner_basis as groebner,
    checked_prime_field_ideal_intersection as intersect,
    checked_prime_field_ideal_saturation as saturate,
    verify_prime_field_ideal_intersection as verify_intersection,
    verify_prime_field_ideal_saturation as verify_saturation,
};
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn polynomial(generators: &[Symbol], prime: u64, terms: &[(&[u32], u64)]) -> Polynomial {
    let terms: Terms = terms
        .iter()
        .map(|(powers, c)| (powers.to_vec(), *c))
        .collect();
    Polynomial::new(generators, prime, &terms).unwrap()
}

fn intersection(
    generators: &[Symbol],
    prime: u64,
    left: &[Polynomial],
    right: &[Polynomial],
) -> VerifiedPrimeFieldIdealOperation {
    intersect(
        generators,
        prime,
        left,
        right,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn saturation(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    factor: &Polynomial,
) -> VerifiedPrimeFieldIdealOperation {
    saturate(
        generators,
        prime,
        inputs,
        factor,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

#[test]
fn principal_intersection_preserves_multiplicity_including_characteristic_two() {
    let generators = ring(&["x"]);
    for prime in [2, 3, 7] {
        let x = polynomial(&generators, prime, &[(&[1], 1)]);
        let square = polynomial(&generators, prime, &[(&[2], 1)]);
        assert_eq!(
            intersection(&generators, prime, std::slice::from_ref(&square), &[x]).basis(),
            [square]
        );
        let linear = polynomial(&generators, prime, &[(&[1], 1), (&[0], 1)]);
        let repeated = polynomial(
            &generators,
            prime,
            &[(&[2], 1), (&[1], 2 % prime), (&[0], 1)],
        );
        let result = intersection(
            &generators,
            prime,
            std::slice::from_ref(&repeated),
            &[linear],
        );
        assert_eq!(result.basis(), [repeated]);
    }
}

#[test]
fn nonprincipal_intersection_has_all_three_required_generators() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let left = [
            polynomial(&generators, prime, &[(&[1, 0], 1)]),
            polynomial(&generators, prime, &[(&[0, 2], 1)]),
        ];
        let right = [
            polynomial(&generators, prime, &[(&[2, 0], 1)]),
            polynomial(&generators, prime, &[(&[0, 1], 1)]),
        ];
        let expected = [
            right[0].clone(),
            polynomial(&generators, prime, &[(&[1, 1], 1)]),
            left[1].clone(),
        ];
        let result = intersection(&generators, prime, &left, &right);
        assert_eq!(result.basis(), expected);
        assert_eq!(result.operation(), IdealOperation::Intersection);
        assert_eq!(result.prime(), prime);
        assert_eq!(result.generators(), generators);
        assert_eq!(result.input_generators(), generators);
        assert_eq!(result.inputs(), left);
        assert_eq!(result.other_inputs(), right);
        assert!(result.eliminated_variables().is_empty());
        assert!(result.saturating_polynomial().is_none());
        assert_eq!(
            intersection(&generators, prime, &right, &left).basis(),
            expected
        );
        verify_intersection(
            &generators,
            prime,
            &left,
            &right,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap();
    }
}

#[test]
fn characteristic_changes_intersection_from_a_product_to_a_single_linear_ideal() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        let plus = polynomial(&generators, prime, &[(&[1, 0], 1), (&[0, 1], 1)]);
        let minus = polynomial(&generators, prime, &[(&[1, 0], 1), (&[0, 1], prime - 1)]);
        let expected = if prime == 2 {
            plus.clone()
        } else {
            polynomial(&generators, prime, &[(&[2, 0], 1), (&[0, 2], prime - 1)])
        };
        assert_eq!(
            intersection(&generators, prime, &[plus], &[minus]).basis(),
            [expected]
        );
    }
}

#[test]
fn zero_unit_and_duplicate_original_inputs_obey_complete_intersection_laws() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        let zero = polynomial(&generators, prime, &[]);
        let unit = polynomial(&generators, prime, &[(&[0, 0], 1)]);
        let x = polynomial(&generators, prime, &[(&[1, 0], 1)]);
        let inputs = [zero.clone(), x.clone(), x.clone()];
        assert!(
            intersection(&generators, prime, &inputs, &[])
                .basis()
                .is_empty()
        );
        assert!(
            intersection(&generators, prime, &[zero], &inputs)
                .basis()
                .is_empty()
        );
        let result = intersection(&generators, prime, &inputs, std::slice::from_ref(&unit));
        assert_eq!(result.basis(), std::slice::from_ref(&x));
        assert_eq!(result.inputs(), inputs);
        assert_eq!(
            intersection(&generators, prime, &[unit], &inputs).basis(),
            [x]
        );
    }
}

#[test]
fn saturation_removes_only_the_inverted_factor_and_preserves_nilpotents() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        let x = polynomial(&generators, prime, &[(&[1, 0], 1)]);
        let y = polynomial(&generators, prime, &[(&[0, 1], 1)]);
        let source = polynomial(&generators, prime, &[(&[2, 1], 1)]);
        let result = saturation(&generators, prime, std::slice::from_ref(&source), &y);
        assert_eq!(
            result.basis(),
            [polynomial(&generators, prime, &[(&[2, 0], 1)])]
        );
        assert_eq!(result.operation(), IdealOperation::Saturation);
        assert_eq!(result.saturating_polynomial(), Some(&y));
        assert_eq!(result.inputs(), [source]);
        assert_eq!(result.prime(), prime);
        assert_eq!(result.generators(), generators);
        // The colon must allow arbitrary powers, not only one division by x.
        let high = polynomial(&generators, prime, &[(&[5, 2], 1)]);
        let result = saturation(&generators, prime, std::slice::from_ref(&high), &x);
        assert_eq!(
            result.basis(),
            [polynomial(&generators, prime, &[(&[0, 2], 1)])]
        );
        verify_saturation(
            &generators,
            prime,
            &[high],
            &x,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap();
    }
}

#[test]
fn nonmonomial_saturation_handles_frobenius_cancellation_without_radicalizing() {
    let generators = ring(&["x", "y", "z"]);
    for prime in [2, 3, 7] {
        let factor = polynomial(&generators, prime, &[(&[1, 0, 0], 1), (&[0, 1, 0], 1)]);
        // (x+y)^2*z^2. The mixed term vanishes only in characteristic two.
        let source = polynomial(
            &generators,
            prime,
            &[(&[2, 0, 2], 1), (&[1, 1, 2], 2 % prime), (&[0, 2, 2], 1)],
        );
        assert_eq!(
            saturation(&generators, prime, &[source], &factor).basis(),
            [polynomial(&generators, prime, &[(&[0, 0, 2], 1)])]
        );
    }
}

#[test]
fn saturation_zero_unit_factors_and_constant_fields_have_their_exact_ideal_meanings() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        let zero = polynomial(&generators, prime, &[]);
        let one = polynomial(&generators, prime, &[(&[0, 0], 1)]);
        let square = polynomial(&generators, prime, &[(&[2, 0], 1)]);
        assert_eq!(
            saturation(&generators, prime, std::slice::from_ref(&square), &zero).basis(),
            std::slice::from_ref(&one)
        );
        assert_eq!(
            saturation(&generators, prime, std::slice::from_ref(&square), &one).basis(),
            std::slice::from_ref(&square)
        );
        assert!(
            saturation(&generators, prime, &[], &square)
                .basis()
                .is_empty()
        );
        assert_eq!(
            saturation(&generators, prime, std::slice::from_ref(&one), &square).basis(),
            [one]
        );
        let constant_zero = polynomial(&[], prime, &[]);
        let constant_one = polynomial(&[], prime, &[(&[], 1)]);
        assert!(
            intersection(&[], prime, &[], std::slice::from_ref(&constant_one))
                .basis()
                .is_empty()
        );
        assert_eq!(
            intersection(
                &[],
                prime,
                std::slice::from_ref(&constant_one),
                std::slice::from_ref(&constant_one)
            )
            .basis(),
            std::slice::from_ref(&constant_one)
        );
        assert_eq!(
            saturation(&[], prime, &[], &constant_zero).basis(),
            std::slice::from_ref(&constant_one)
        );
        assert!(
            saturation(&[], prime, &[], &constant_one)
                .basis()
                .is_empty()
        );
    }
}

#[test]
fn auxiliary_names_and_same_spelling_typed_symbols_are_never_captured() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([91; 32]));
    let generators = vec![
        first,
        second,
        Symbol::new("_fsym_prime_field_ideal_parameter_0"),
        Symbol::new("_fsym_prime_field_ideal_parameter_1"),
    ];
    let input = polynomial(&generators, 2, &[(&[2, 1, 0, 0], 1)]);
    let factor = polynomial(&generators, 2, &[(&[0, 1, 0, 0], 1)]);
    let result = saturation(&generators, 2, &[input], &factor);
    assert_eq!(
        result.basis(),
        [polynomial(&generators, 2, &[(&[2, 0, 0, 0], 1)])]
    );
    let auxiliary = &result.certificate().ambient_basis.generators()[0];
    assert!(
        generators
            .iter()
            .all(|symbol| symbol.name != auxiliary.name)
    );
    assert_eq!(
        &result.certificate().ambient_basis.generators()[1..],
        generators
    );
}

#[test]
fn intersection_projection_cannot_omit_valid_members_or_replace_an_original_ideal() {
    let generators = ring(&["x", "y"]);
    let left = [
        polynomial(&generators, 5, &[(&[1, 0], 1)]),
        polynomial(&generators, 5, &[(&[0, 2], 1)]),
    ];
    let right = [
        polynomial(&generators, 5, &[(&[2, 0], 1)]),
        polynomial(&generators, 5, &[(&[0, 1], 1)]),
    ];
    let result = intersection(&generators, 5, &left, &right);
    let mut bad = result.certificate().clone();
    bad.projected_basis.remove(1); // x^2 and y^2 alone omit the required xy.
    assert!(verify_intersection(&generators, 5, &left, &right, &bad, &mut Unbounded).is_err());
    let mut changed = left.clone();
    changed[0] = polynomial(&generators, 5, &[(&[1, 0], 2)]);
    assert!(
        verify_intersection(
            &generators,
            5,
            &changed,
            &right,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_intersection(
            &generators,
            5,
            &right,
            &left,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_intersection(
            &generators,
            7,
            &left,
            &right,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn accepted_ambient_ideals_still_require_the_exact_intersection_embedding() {
    let generators = ring(&["x", "y"]);
    let left = [polynomial(&generators, 5, &[(&[1, 0], 1)])];
    let right = [polynomial(&generators, 5, &[(&[0, 1], 1)])];
    let result = intersection(&generators, 5, &left, &right);
    let ambient = &result.certificate().ambient_basis;
    let mut wrong_inputs = ambient.inputs().to_vec();
    // t*x + y has a valid complete ambient basis, but the operation required
    // t*x + (1-t)*y. A Groebner receipt alone cannot establish this binding.
    wrong_inputs[1] = polynomial(ambient.generators(), 5, &[(&[0, 0, 1], 1)]);
    let mut bad = result.certificate().clone();
    bad.ambient_basis = Arc::new(
        groebner(
            ambient.generators(),
            5,
            &wrong_inputs,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.projected_basis = right.to_vec();
    assert!(verify_intersection(&generators, 5, &left, &right, &bad, &mut Unbounded).is_err());
    let mut bad = result.certificate().clone();
    bad.ambient_basis = Arc::new(
        groebner(
            ambient.generators(),
            5,
            ambient.inputs(),
            TermOrder::DegLex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    assert!(verify_intersection(&generators, 5, &left, &right, &bad, &mut Unbounded).is_err());
}

#[test]
fn saturation_checker_requires_the_unit_relation_factor_and_every_original_equation() {
    let generators = ring(&["x", "y"]);
    let inputs = [polynomial(&generators, 5, &[(&[1, 1], 1)])];
    let x = polynomial(&generators, 5, &[(&[1, 0], 1)]);
    let y = polynomial(&generators, 5, &[(&[0, 1], 1)]);
    let result = saturation(&generators, 5, &inputs, &x);
    assert!(
        verify_saturation(
            &generators,
            5,
            &inputs,
            &y,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let ambient = &result.certificate().ambient_basis;
    let mut wrong_inputs = ambient.inputs().to_vec();
    *wrong_inputs.last_mut().unwrap() = polynomial(ambient.generators(), 5, &[(&[1, 1, 0], 4)]);
    let mut bad = result.certificate().clone();
    bad.ambient_basis = Arc::new(
        groebner(
            ambient.generators(),
            5,
            &wrong_inputs,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.projected_basis = inputs.to_vec();
    assert!(verify_saturation(&generators, 5, &inputs, &x, &bad, &mut Unbounded).is_err());
    let mut bad = result.certificate().clone();
    bad.ambient_basis = Arc::new(
        groebner(
            ambient.generators(),
            5,
            &ambient.inputs()[1..],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.projected_basis.clear();
    assert!(verify_saturation(&generators, 5, &inputs, &x, &bad, &mut Unbounded).is_err());
    let mut bad = result.certificate().clone();
    bad.projected_basis = vec![polynomial(&generators, 5, &[(&[0, 2], 1)])];
    assert!(verify_saturation(&generators, 5, &inputs, &x, &bad, &mut Unbounded).is_err());
}

#[test]
fn construction_growth_and_wrong_ring_inputs_refuse_even_for_zero_sources() {
    let generators = ring(&["x", "y"]);
    let zero = polynomial(&generators, 5, &[]);
    let factor = polynomial(&generators, 5, &[(&[1, 0], 1)]);
    let high = polynomial(&generators, 5, &[(&[64, 0], 1)]);
    assert!(
        intersect(
            &generators,
            5,
            std::slice::from_ref(&high),
            std::slice::from_ref(&zero),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(saturate(&generators, 5, &[], &high, &mut Unbounded, &mut Unbounded).is_err());
    let large: Vec<_> = (0..8)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    assert!(intersect(&large, 5, &[], &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(
        intersect(
            &generators,
            5,
            &vec![zero.clone(); 16],
            &vec![zero.clone(); 17],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        saturate(
            &generators,
            5,
            &vec![zero.clone(); 32],
            &factor,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let alien = polynomial(&ring(&["x"]), 5, &[]);
    assert!(
        intersect(
            &generators,
            5,
            std::slice::from_ref(&zero),
            std::slice::from_ref(&alien),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        saturate(
            &generators,
            5,
            std::slice::from_ref(&zero),
            &alien,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let foreign = polynomial(&generators, 7, &[]);
    assert!(
        intersect(
            &generators,
            5,
            std::slice::from_ref(&zero),
            std::slice::from_ref(&foreign),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        saturate(
            &generators,
            5,
            &[zero],
            &foreign,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Count {
    checkpoints: usize,
    steps: u64,
    bytes: u64,
    allocations: u64,
    cancel: Option<usize>,
}

impl BudgetMeter for Count {
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

fn metered_operation(
    saturation: bool,
    generators: &[Symbol],
    left: &[Polynomial],
    right: &[Polynomial],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, fsym_polys::PolyError> {
    if saturation {
        saturate(generators, 3, left, &right[0], generator, verifier)
    } else {
        intersect(generators, 3, left, right, generator, verifier)
    }
}

#[test]
fn both_composed_operations_account_for_all_work_and_final_publication() {
    let generators = ring(&["x", "y"]);
    let left = [polynomial(&generators, 3, &[(&[2, 1], 1)])];
    let right = [polynomial(&generators, 3, &[(&[0, 1], 1)])];
    for saturation in [false, true] {
        let (mut generation, mut verification) = (Count::default(), Count::default());
        metered_operation(
            saturation,
            &generators,
            &left,
            &right,
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
            assert!(generated > 1 && verified > 1);
            metered_operation(
                saturation,
                &generators,
                &left,
                &right,
                &mut cap(dimension, generated),
                &mut cap(dimension, verified),
            )
            .unwrap();
            assert!(
                metered_operation(
                    saturation,
                    &generators,
                    &left,
                    &right,
                    &mut cap(dimension, generated - 1),
                    &mut Unbounded
                )
                .is_err()
            );
            assert!(
                metered_operation(
                    saturation,
                    &generators,
                    &left,
                    &right,
                    &mut Unbounded,
                    &mut cap(dimension, verified - 1)
                )
                .is_err()
            );
            let mut used = cap(dimension, generated);
            used.charge(dimension, 1).unwrap();
            assert!(
                metered_operation(
                    saturation,
                    &generators,
                    &left,
                    &right,
                    &mut used,
                    &mut Unbounded
                )
                .is_err()
            );
        }
        for (in_generator, checkpoints) in [
            (true, generation.checkpoints),
            (false, verification.checkpoints),
        ] {
            for at in [1, checkpoints / 2, checkpoints - 1, checkpoints] {
                let mut stopped = Count {
                    cancel: Some(at),
                    ..Count::default()
                };
                let result = if in_generator {
                    metered_operation(
                        saturation,
                        &generators,
                        &left,
                        &right,
                        &mut stopped,
                        &mut Unbounded,
                    )
                } else {
                    metered_operation(
                        saturation,
                        &generators,
                        &left,
                        &right,
                        &mut Unbounded,
                        &mut stopped,
                    )
                };
                assert!(
                    result.is_err(),
                    "saturation={saturation}, generator={in_generator}, checkpoint={at}"
                );
            }
        }
    }
}
