//! Exact ideal identities, original construction binding and complete
//! elimination selection. Expected ideals use closed-form algebra, not a
//! second call to the generator under test.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    IdealOperation, MultivariatePoly, TermOrder, bounded_polynomial_from_expr,
    checked_elimination_ideal, checked_groebner_basis, checked_ideal_intersection,
    checked_ideal_saturation, verify_elimination_ideal, verify_ideal_intersection,
    verify_ideal_saturation,
};
use std::collections::BTreeMap;
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn polynomial(source: &str, generators: &[Symbol]) -> MultivariatePoly {
    bounded_polynomial_from_expr(&parse(source).unwrap(), generators, &mut Unbounded).unwrap()
}

fn polynomials(sources: &[&str], generators: &[Symbol]) -> Vec<MultivariatePoly> {
    sources
        .iter()
        .map(|source| polynomial(source, generators))
        .collect()
}

fn equal_basis(actual: &[MultivariatePoly], expected: &[MultivariatePoly]) {
    assert_eq!(actual.len(), expected.len());
    for expected in expected {
        assert!(
            actual.contains(expected),
            "missing exact expected generator {expected:?}"
        );
    }
}

#[test]
fn elimination_computes_a_complete_basis_in_the_retained_ring() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = polynomials(&["x-y**2", "z-x*y"], &generators);
    let result = checked_elimination_ideal(
        &inputs,
        &generators,
        &generators[..1],
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(result.operation(), IdealOperation::Elimination);
    assert_eq!(result.generators(), &generators[1..]);
    equal_basis(result.basis(), &polynomials(&["y**3-z"], &generators[1..]));
    assert_eq!(result.input_generators(), generators);
    assert_eq!(result.inputs(), inputs);
    assert_eq!(result.eliminated_variables(), &generators[..1]);
    assert_eq!(result.certificate().ambient_basis.order(), TermOrder::Lex);
    verify_elimination_ideal(
        &inputs,
        &generators,
        &generators[..1],
        result.certificate(),
        &mut Unbounded,
    )
    .unwrap();
}

#[test]
fn arbitrary_nonleading_elimination_preserves_the_remaining_caller_order() {
    for names in [["x", "y", "z"], ["z", "y", "x"]] {
        let generators = ring(&names);
        let inputs = polynomials(&["x-y**2", "z-x*y"], &generators);
        let result = checked_elimination_ideal(
            &inputs,
            &generators,
            &generators[1..2],
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        let output = vec![generators[0].clone(), generators[2].clone()];
        assert_eq!(result.generators(), output);
        let expected = if names[0] == "x" {
            "x**3-z**2"
        } else {
            "z**2-x**3"
        };
        equal_basis(result.basis(), &[polynomial(expected, &output)]);
    }
}

#[test]
fn multiple_elimination_orders_have_the_same_complete_remaining_ideal() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = polynomials(&["x-y**2", "z-x*y", "y**2-2"], &generators);
    for eliminated in [
        vec![generators[0].clone(), generators[2].clone()],
        vec![generators[2].clone(), generators[0].clone()],
    ] {
        let result = checked_elimination_ideal(
            &inputs,
            &generators,
            &eliminated,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(result.generators(), &generators[1..2]);
        equal_basis(result.basis(), &[polynomial("y**2-2", &generators[1..2])]);
    }
}

#[test]
fn eliminating_no_or_all_variables_handles_zero_unit_and_constant_rings() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x*y-1"], &generators);
    let result =
        checked_elimination_ideal(&inputs, &generators, &[], &mut Unbounded, &mut Unbounded)
            .unwrap();
    equal_basis(result.basis(), &inputs);
    for inputs in [Vec::new(), inputs, polynomials(&["0"], &generators)] {
        let result = checked_elimination_ideal(
            &inputs,
            &generators,
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert!(result.generators().is_empty());
        assert!(result.basis().is_empty());
    }
    let unit = polynomials(&["1"], &generators);
    let result = checked_elimination_ideal(
        &unit,
        &generators,
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(result.generators().is_empty());
    assert_eq!(result.basis(), &[MultivariatePoly::one(Vec::new())]);
    let result = checked_elimination_ideal(&[], &[], &[], &mut Unbounded, &mut Unbounded).unwrap();
    assert!(result.basis().is_empty());
}

#[test]
fn principal_intersections_retain_multiplicity_and_exact_rational_coefficients() {
    let generators = ring(&["x", "y"]);
    for (left, right, expected) in [
        ("x", "y", "x*y"),
        ("x**2", "x*y", "x**2*y"),
        ("x-1", "x+1", "x**2-1"),
        ("2*x/3-1", "5*y/7+2", "x*y+14*x/5-3*y/2-21/5"),
    ] {
        let left = polynomials(&[left], &generators);
        let right = polynomials(&[right], &generators);
        let result =
            checked_ideal_intersection(&left, &right, &generators, &mut Unbounded, &mut Unbounded)
                .unwrap();
        assert_eq!(result.operation(), IdealOperation::Intersection);
        equal_basis(result.basis(), &[polynomial(expected, &generators)]);
        assert_eq!(result.inputs(), left);
        assert_eq!(result.other_inputs(), right);
        verify_ideal_intersection(
            &left,
            &right,
            &generators,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap();
    }
}

#[test]
fn nonprincipal_intersection_requires_all_three_generators() {
    let generators = ring(&["x", "y"]);
    let left = polynomials(&["x", "y**2"], &generators);
    let right = polynomials(&["x**2", "y"], &generators);
    let result =
        checked_ideal_intersection(&left, &right, &generators, &mut Unbounded, &mut Unbounded)
            .unwrap();
    equal_basis(
        result.basis(),
        &polynomials(&["x**2", "x*y", "y**2"], &generators),
    );
    let reversed =
        checked_ideal_intersection(&right, &left, &generators, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(result.basis(), reversed.basis());
    let idempotent =
        checked_ideal_intersection(&left, &left, &generators, &mut Unbounded, &mut Unbounded)
            .unwrap();
    equal_basis(idempotent.basis(), &left);
}

#[test]
fn intersections_with_zero_and_unit_ideals_obey_the_full_ideal_laws() {
    let generators = ring(&["x", "y"]);
    let ideal = polynomials(&["x**2", "y"], &generators);
    let unit = polynomials(&["1"], &generators);
    for (left, right, expected) in [
        (&[][..], ideal.as_slice(), &[][..]),
        (ideal.as_slice(), &[][..], &[][..]),
        (&[][..], &[][..], &[][..]),
        (unit.as_slice(), ideal.as_slice(), ideal.as_slice()),
        (ideal.as_slice(), unit.as_slice(), ideal.as_slice()),
    ] {
        let result =
            checked_ideal_intersection(left, right, &generators, &mut Unbounded, &mut Unbounded)
                .unwrap();
        equal_basis(result.basis(), expected);
    }
}

#[test]
fn saturation_removes_components_contained_in_the_factor_zero_locus() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = polynomials(&["x*y", "x*z"], &generators);
    let factor = polynomial("x", &generators);
    let result = checked_ideal_saturation(
        &inputs,
        &factor,
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(result.operation(), IdealOperation::Saturation);
    equal_basis(result.basis(), &polynomials(&["y", "z"], &generators));
    assert_eq!(result.saturating_polynomial(), Some(&factor));
    // Saturation is the closed ideal, so the origin remains on this component.
    for polynomial in result.basis() {
        assert_eq!(
            polynomial.eval(&[0.into(), 0.into(), 0.into()]).unwrap(),
            BigRational::from(0_i64)
        );
    }
    verify_ideal_saturation(
        &inputs,
        &factor,
        &generators,
        result.certificate(),
        &mut Unbounded,
    )
    .unwrap();
}

#[test]
fn saturation_uses_arbitrary_powers_and_handles_zero_unit_factors() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**3*y"], &generators);
    for (factor, expected) in [
        ("x", "y"),
        ("x**2", "y"),
        ("x*y", "1"),
        ("0", "1"),
        ("7/13", "x**3*y"),
        ("x+1", "x**3*y"),
    ] {
        let result = checked_ideal_saturation(
            &inputs,
            &polynomial(factor, &generators),
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        equal_basis(result.basis(), &[polynomial(expected, &generators)]);
    }
    let zero = checked_ideal_saturation(
        &[],
        &polynomial("x", &generators),
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(zero.basis().is_empty());
    let all = checked_ideal_saturation(
        &[],
        &polynomial("0", &generators),
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(all.basis(), &[MultivariatePoly::one(generators.clone())]);
}

#[test]
fn fresh_auxiliaries_and_identical_printed_names_do_not_capture_original_symbols() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([73; 32]));
    let occupied = Symbol::new("_fsym_ideal_parameter_0");
    let generators = vec![first.clone(), second.clone(), occupied];
    let product = MultivariatePoly::new(
        generators.clone(),
        BTreeMap::from([(vec![1, 1, 0], 1.into())]),
    )
    .unwrap();
    let factor = MultivariatePoly::var(generators.clone(), &first).unwrap();
    let result = checked_ideal_saturation(
        std::slice::from_ref(&product),
        &factor,
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        result.basis(),
        &[MultivariatePoly::var(generators.clone(), &second).unwrap()]
    );
    let auxiliary = &result.certificate().ambient_basis.input_basis()[0].generators()[0];
    assert!(
        generators
            .iter()
            .all(|symbol| symbol.name != auxiliary.name)
    );
    let inputs = vec![
        MultivariatePoly::var(generators.clone(), &first).unwrap(),
        MultivariatePoly::var(generators.clone(), &second).unwrap(),
    ];
    let result = checked_elimination_ideal(
        &inputs,
        &generators,
        std::slice::from_ref(&first),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(result.generators(), &generators[1..]);
    assert_eq!(
        result.basis(),
        &[MultivariatePoly::var(generators[1..].to_vec(), &second).unwrap()]
    );
}

#[test]
fn complete_projection_rejects_omitted_extra_and_altered_generators() {
    let generators = ring(&["x", "y"]);
    let left = polynomials(&["x", "y**2"], &generators);
    let right = polynomials(&["x**2", "y"], &generators);
    let result =
        checked_ideal_intersection(&left, &right, &generators, &mut Unbounded, &mut Unbounded)
            .unwrap();
    let check = |certificate| {
        assert!(
            verify_ideal_intersection(&left, &right, &generators, &certificate, &mut Unbounded)
                .is_err()
        )
    };
    let mut bad = result.certificate().clone();
    bad.projected_basis.pop();
    check(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis
        .push(MultivariatePoly::one(generators.clone()));
    check(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis[0] = polynomial(
        "x**2+1/100000000000000000000000000000000000000",
        &generators,
    );
    check(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis.swap(0, 1);
    check(bad);
}

#[test]
fn accepted_ambient_receipts_cannot_be_rebound_to_another_source_or_order() {
    let generators = ring(&["x", "y"]);
    let left = polynomials(&["x"], &generators);
    let right = polynomials(&["y"], &generators);
    let result =
        checked_ideal_intersection(&left, &right, &generators, &mut Unbounded, &mut Unbounded)
            .unwrap();
    let other = checked_ideal_intersection(
        &left,
        &polynomials(&["y**2"], &generators),
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(
        verify_ideal_intersection(
            &left,
            &right,
            &generators,
            other.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_ideal_intersection(
            &polynomials(&["2*x"], &generators),
            &right,
            &generators,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let mut bad = result.certificate().clone();
    bad.ambient_basis = Arc::new(
        checked_groebner_basis(
            result.certificate().ambient_basis.input_basis(),
            TermOrder::DegLex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    assert!(verify_ideal_intersection(&left, &right, &generators, &bad, &mut Unbounded).is_err());
    let eliminated = checked_elimination_ideal(
        &polynomials(&["x-y**2"], &generators),
        &generators,
        &generators[..1],
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(
        verify_elimination_ideal(
            &polynomials(&["x-y**2"], &generators),
            &generators,
            &generators[1..],
            eliminated.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn saturation_checker_rejects_changed_unit_relation_factor_and_original_equations() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x*y"], &generators);
    let factor = polynomial("x", &generators);
    let result = checked_ideal_saturation(
        &inputs,
        &factor,
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(
        verify_ideal_saturation(
            &inputs,
            &polynomial("y", &generators),
            &generators,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let ambient_ring = result.certificate().ambient_basis.input_basis()[0].generators();
    let wrong_relation = MultivariatePoly::new(
        ambient_ring.to_vec(),
        BTreeMap::from([(vec![1, 1, 0], (-1).into())]),
    )
    .unwrap();
    let mut wrong_inputs = result.certificate().ambient_basis.input_basis().to_vec();
    *wrong_inputs.last_mut().unwrap() = wrong_relation;
    let wrong_basis = checked_groebner_basis(
        &wrong_inputs,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    let mut bad = result.certificate().clone();
    bad.ambient_basis = Arc::new(wrong_basis);
    bad.projected_basis = inputs.clone();
    assert!(verify_ideal_saturation(&inputs, &factor, &generators, &bad, &mut Unbounded).is_err());
    let mut bad = result.certificate().clone();
    let shortened = result.certificate().ambient_basis.input_basis()[1..].to_vec();
    bad.ambient_basis = Arc::new(
        checked_groebner_basis(&shortened, TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap(),
    );
    bad.projected_basis.clear();
    assert!(verify_ideal_saturation(&inputs, &factor, &generators, &bad, &mut Unbounded).is_err());
}

#[test]
fn malformed_rings_duplicate_variables_and_auxiliary_growth_refuse() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x*y"], &generators);
    for eliminated in [
        vec![Symbol::new("unknown")],
        vec![generators[0].clone(), generators[0].clone()],
    ] {
        assert!(
            checked_elimination_ideal(
                &inputs,
                &generators,
                &eliminated,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let swapped = vec![generators[1].clone(), generators[0].clone()];
    assert!(
        checked_elimination_ideal(&inputs, &swapped, &[], &mut Unbounded, &mut Unbounded).is_err()
    );
    assert!(
        checked_ideal_saturation(
            &inputs,
            &polynomial("x", &ring(&["x"])),
            &generators,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let large_ring: Vec<_> = (0..8)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    assert!(
        checked_ideal_intersection(&[], &[], &large_ring, &mut Unbounded, &mut Unbounded).is_err()
    );
    assert!(
        checked_ideal_saturation(
            &vec![inputs[0].clone(); 64],
            &inputs[0],
            &generators,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let high = polynomial("x**256", &generators);
    assert!(
        checked_ideal_intersection(
            &[high],
            &inputs,
            &generators,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    compute: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.compute += amount;
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
fn both_phases_cancel_during_work_and_at_their_last_publication_checkpoint() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2*y", "x*y**2"], &generators);
    let factor = polynomial("x", &generators);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    checked_ideal_saturation(
        &inputs,
        &factor,
        &generators,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    assert!(generation.compute > 0 && verification.compute > 0);
    for checkpoint in [1, generation.checkpoints / 2, generation.checkpoints] {
        let mut meter = Meter {
            cancel_at: Some(checkpoint),
            ..Meter::default()
        };
        assert!(
            checked_ideal_saturation(&inputs, &factor, &generators, &mut meter, &mut Unbounded)
                .is_err()
        );
    }
    for checkpoint in [1, verification.checkpoints / 2, verification.checkpoints] {
        let mut meter = Meter {
            cancel_at: Some(checkpoint),
            ..Meter::default()
        };
        assert!(
            checked_ideal_saturation(&inputs, &factor, &generators, &mut Unbounded, &mut meter)
                .is_err()
        );
    }
}

#[test]
fn exhausted_real_budgets_do_not_publish_partial_ideals_or_restart_accounting() {
    let generators = ring(&["x", "y"]);
    let left = polynomials(&["x"], &generators);
    let right = polynomials(&["y"], &generators);
    let mut generation = Budget::new(BudgetLimits::uniform(1, 0));
    let mut verification = Meter::default();
    assert!(
        checked_ideal_intersection(
            &left,
            &right,
            &generators,
            &mut generation,
            &mut verification
        )
        .is_err()
    );
    assert_eq!(verification.checkpoints, 0);
    let mut verification = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        checked_ideal_intersection(
            &left,
            &right,
            &generators,
            &mut Unbounded,
            &mut verification
        )
        .is_err()
    );
    let result =
        checked_ideal_intersection(&left, &right, &generators, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert!(
        verify_ideal_intersection(
            &left,
            &right,
            &generators,
            result.certificate(),
            &mut Budget::new(BudgetLimits::uniform(1, 0))
        )
        .is_err()
    );
    let mut generation = Budget::new(BudgetLimits::uniform(128 * 1024 * 1024, 0));
    let mut verification = Budget::new(BudgetLimits::uniform(128 * 1024 * 1024, 0));
    let result = checked_ideal_intersection(
        &left,
        &right,
        &generators,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    equal_basis(result.basis(), &polynomials(&["x*y"], &generators));
}
