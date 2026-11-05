//! Complete quotient-element annihilator ideals, including transcendence.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly, QuotientMinimalPolynomialCertificate, VerifiedQuotientMinimalPolynomial,
    bounded_polynomial_from_expr, checked_quotient_minimal_polynomial,
    verify_quotient_minimal_polynomial,
};
use std::collections::BTreeMap;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}
fn poly(source: &str, generators: &[Symbol]) -> MultivariatePoly {
    bounded_polynomial_from_expr(&parse(source).unwrap(), generators, &mut Unbounded).unwrap()
}
fn inputs(sources: &[&str], generators: &[Symbol]) -> Vec<MultivariatePoly> {
    sources
        .iter()
        .map(|source| poly(source, generators))
        .collect()
}
fn calculate(
    sources: &[&str],
    element: &str,
    generators: &[Symbol],
) -> VerifiedQuotientMinimalPolynomial {
    checked_quotient_minimal_polynomial(
        &inputs(sources, generators),
        &poly(element, generators),
        generators,
        &Symbol::new("t"),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}
fn coefficients(values: &[i64]) -> Vec<BigRational> {
    values
        .iter()
        .map(|&value| BigRational::from(value))
        .collect()
}
fn reject(
    result: &VerifiedQuotientMinimalPolynomial,
    candidate: &QuotientMinimalPolynomialCertificate,
) {
    assert!(
        verify_quotient_minimal_polynomial(
            result.inputs(),
            result.element(),
            result.generators(),
            result.parameter(),
            candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn reduced_field_elements_have_their_minimal_not_characteristic_polynomial() {
    let generators = ring(&["x", "y"]);
    let sum = calculate(&["x**2-2", "y**2-3"], "x+y", &generators);
    assert_eq!(
        sum.coefficients(),
        Some(coefficients(&[1, 0, -10, 0, 1]).as_slice())
    );
    let repeated = calculate(&["x**2-2", "y**2-3"], "x", &generators);
    assert_eq!(
        repeated.coefficients(),
        Some(coefficients(&[-2, 0, 1]).as_slice())
    );
    assert_eq!(repeated.generators(), generators);
    assert_eq!(repeated.element(), &poly("x", &generators));
    assert_eq!(repeated.parameter(), &Symbol::new("t"));
    let replay = verify_quotient_minimal_polynomial(
        sum.inputs(),
        sum.element(),
        &generators,
        sum.parameter(),
        sum.certificate(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        replay.into_coefficients(),
        sum.coefficients().map(<[_]>::to_vec)
    );
}

#[test]
fn nonradical_multiplicity_and_constant_shifts_are_preserved() {
    let generators = ring(&["x", "y"]);
    for (element, expected) in [
        ("x+y", vec![0, 0, 0, 1]),
        ("1+x+y", vec![-1, 3, -3, 1]),
        ("x*y", vec![0, 0, 1]),
        ("x**2+y**2", vec![0, 1]),
    ] {
        let result = calculate(&["x**2", "y**2"], element, &generators);
        assert_eq!(
            result.coefficients(),
            Some(coefficients(&expected).as_slice())
        );
    }
}

#[test]
fn positive_dimensional_quotients_separate_algebraic_and_transcendental_elements() {
    let generators = ring(&["x", "y"]);
    let algebraic = calculate(&["x**2-2"], "x", &generators);
    assert_eq!(
        algebraic.coefficients(),
        Some(coefficients(&[-2, 0, 1]).as_slice())
    );
    for (sources, element) in [
        (vec!["x**2-2"], "x+y"),
        (vec![], "x"),
        (vec!["x*y-1"], "x"),
        (vec!["x*y"], "x+y"),
    ] {
        let result = calculate(&sources, element, &generators);
        assert!(result.coefficients().is_none());
        assert!(result.certificate().elimination.basis().is_empty());
    }
    assert_eq!(
        calculate(&["x*y-1"], "x*y", &generators).coefficients(),
        Some(coefficients(&[-1, 1]).as_slice())
    );
}

#[test]
fn quotient_algebra_is_not_a_selected_real_embedding() {
    let generators = ring(&["x", "y"]);
    // The independent defining relations include all four embeddings. A
    // selected sqrt(2)+sqrt(8) would have t^2-18, which is false in this ring.
    let result = calculate(&["x**2-2", "y**2-8"], "x+y", &generators);
    assert_eq!(
        result.coefficients(),
        Some(coefficients(&[36, 0, -20, 0, 1]).as_slice())
    );
    let mut altered = result.certificate().clone();
    altered.coefficients = Some(coefficients(&[-18, 0, 1]));
    reject(&result, &altered);
}

#[test]
fn unit_zero_empty_and_constant_rings_have_explicit_generators() {
    let generators = ring(&["x"]);
    assert_eq!(
        calculate(&["1"], "x", &generators).coefficients(),
        Some(coefficients(&[1]).as_slice())
    );
    assert_eq!(
        calculate(&[], "0", &generators).coefficients(),
        Some(coefficients(&[0, 1]).as_slice())
    );
    assert_eq!(
        calculate(&["0"], "2/3", &generators).coefficients(),
        Some(
            [
                BigRational::new(BigInt::from(-2), BigInt::from(3)),
                BigRational::from(1)
            ]
            .as_slice()
        )
    );
    assert_eq!(
        calculate(&[], "-2", &[]).coefficients(),
        Some(coefficients(&[2, 1]).as_slice())
    );
    assert_eq!(
        calculate(&["7"], "0", &[]).coefficients(),
        Some(coefficients(&[1]).as_slice())
    );
}

#[test]
fn original_order_zero_scaling_and_redundancy_remain_bound() {
    let generators = ring(&["y", "x"]);
    let result = calculate(&["0", "2*x**2-4", "3*x**2-6", "y-x"], "x+y", &generators);
    assert_eq!(
        result.coefficients(),
        Some(coefficients(&[-8, 0, 1]).as_slice())
    );
    assert_eq!(result.inputs().len(), 4);
    let mut rebound = result.inputs().to_vec();
    rebound.swap(1, 2);
    assert!(
        verify_quotient_minimal_polynomial(
            &rebound,
            result.element(),
            &generators,
            result.parameter(),
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn same_printed_parameter_keeps_a_distinct_typed_identity() {
    let x = Symbol::new("a");
    let t = Symbol::with_identity("a", SymbolIdentity::from_assumptions_digest([71; 32]));
    let generators = vec![x];
    let source = MultivariatePoly::new(
        generators.clone(),
        BTreeMap::from([
            (vec![2], BigRational::from(1)),
            (vec![0], BigRational::from(-2)),
        ]),
    )
    .unwrap();
    let element = MultivariatePoly::new(
        generators.clone(),
        BTreeMap::from([(vec![1], BigRational::from(1))]),
    )
    .unwrap();
    let result = checked_quotient_minimal_polynomial(
        &[source],
        &element,
        &generators,
        &t,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(result.parameter(), &t);
    assert_eq!(
        result.coefficients(),
        Some(coefficients(&[-2, 0, 1]).as_slice())
    );
    assert!(
        verify_quotient_minimal_polynomial(
            result.inputs(),
            &element,
            &generators,
            &generators[0],
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn false_annihilators_valid_nonminimal_multiples_and_changed_normalization_are_rejected() {
    let generators = ring(&["x"]);
    let result = calculate(&["x**3"], "x", &generators);
    for value in [
        None,
        Some(vec![]),
        Some(coefficients(&[0, 1])),
        Some(coefficients(&[0, 0, 0, 0, 1])),
        Some(coefficients(&[0, 0, 0, 2])),
        Some(coefficients(&[0, 0, 0, 1, 0])),
        Some(coefficients(&[1, 0, 0, 1])),
    ] {
        let mut changed = result.certificate().clone();
        changed.coefficients = value;
        reject(&result, &changed);
    }
    let transcendental = calculate(&[], "x", &generators);
    let mut invented = transcendental.certificate().clone();
    invented.coefficients = Some(coefficients(&[0, 1]));
    reject(&transcendental, &invented);
}

#[test]
fn accepted_elimination_receipts_cannot_be_rebound_to_another_element_or_ideal() {
    let generators = ring(&["x", "y"]);
    let result = calculate(&["x**2-2", "y**2-3"], "x+y", &generators);
    for (sources, element) in [
        (vec!["x**2-2", "y**2-3"], "x-y"),
        (vec!["x**2-3", "y**2-2"], "x+y"),
        (vec!["x**2-2", "y**2-3"], "x"),
    ] {
        let replacement = calculate(&sources, element, &generators);
        let mut changed = result.certificate().clone();
        changed.elimination = replacement.certificate().elimination.clone();
        reject(&result, &changed);
    }
}

#[test]
fn original_shape_and_resource_limits_precede_unit_shortcuts() {
    let generators = ring(&["x"]);
    let one = poly("1", &generators);
    let mut malformed = poly("x", &generators);
    malformed.terms = BTreeMap::from([(vec![1, 0], BigRational::from(1))]);
    assert!(
        checked_quotient_minimal_polynomial(
            &[one.clone(), malformed],
            &one,
            &generators,
            &Symbol::new("t"),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    for element in [
        MultivariatePoly::new(
            generators.clone(),
            BTreeMap::from([(vec![257], BigRational::from(1))]),
        )
        .unwrap(),
        MultivariatePoly::new(
            generators.clone(),
            BTreeMap::from([(
                vec![0],
                BigRational::from_integer(BigInt::from(1) << 4096u32),
            )]),
        )
        .unwrap(),
    ] {
        assert!(
            checked_quotient_minimal_polynomial(
                std::slice::from_ref(&one),
                &element,
                &generators,
                &Symbol::new("t"),
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked_quotient_minimal_polynomial(
            &vec![one.clone(); 64],
            &one,
            &generators,
            &Symbol::new("t"),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        checked_quotient_minimal_polynomial(
            &[],
            &one,
            &generators,
            &Symbol::new("t".repeat(65537)),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let large_ring = ring(&["a", "b", "c", "d", "e", "f", "g", "h"]);
    let one = MultivariatePoly::one(large_ring.clone());
    assert!(
        checked_quotient_minimal_polynomial(
            &[],
            &one,
            &large_ring,
            &Symbol::new("t"),
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
fn cancellation_in_both_phases_and_final_publication_cannot_become_transcendence() {
    let generators = ring(&["x"]);
    for equations in [inputs(&["x**2-2"], &generators), vec![]] {
        let element = poly("x", &generators);
        let parameter = Symbol::new("t");
        let (mut generation, mut verification) = (Count::default(), Count::default());
        checked_quotient_minimal_polynomial(
            &equations,
            &element,
            &generators,
            &parameter,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        for (is_generator, checkpoints) in [
            (true, generation.checkpoints),
            (false, verification.checkpoints),
        ] {
            for at in [1, checkpoints / 2, checkpoints] {
                let mut stopped = Count {
                    cancel: Some(at),
                    ..Count::default()
                };
                let result = if is_generator {
                    checked_quotient_minimal_polynomial(
                        &equations,
                        &element,
                        &generators,
                        &parameter,
                        &mut stopped,
                        &mut Unbounded,
                    )
                } else {
                    checked_quotient_minimal_polynomial(
                        &equations,
                        &element,
                        &generators,
                        &parameter,
                        &mut Unbounded,
                        &mut stopped,
                    )
                };
                assert!(result.is_err());
            }
        }
    }
}

#[test]
fn whole_operation_budgets_do_not_restart_at_nested_elimination_boundaries() {
    let generators = ring(&["x"]);
    for equations in [inputs(&["x**3"], &generators), vec![]] {
        let element = poly("x", &generators);
        let parameter = Symbol::new("t");
        let (mut generation, mut verification) = (Count::default(), Count::default());
        checked_quotient_minimal_polynomial(
            &equations,
            &element,
            &generators,
            &parameter,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        for (dimension, g, v) in [
            (
                Dimension::ComputeSteps,
                generation.steps,
                verification.steps,
            ),
            (Dimension::MemoryBytes, generation.bytes, verification.bytes),
        ] {
            checked_quotient_minimal_polynomial(
                &equations,
                &element,
                &generators,
                &parameter,
                &mut cap(dimension, g),
                &mut cap(dimension, v),
            )
            .unwrap();
            assert!(
                checked_quotient_minimal_polynomial(
                    &equations,
                    &element,
                    &generators,
                    &parameter,
                    &mut cap(dimension, g - 1),
                    &mut Unbounded
                )
                .is_err()
            );
            assert!(
                checked_quotient_minimal_polynomial(
                    &equations,
                    &element,
                    &generators,
                    &parameter,
                    &mut Unbounded,
                    &mut cap(dimension, v - 1)
                )
                .is_err()
            );
        }
    }
}
