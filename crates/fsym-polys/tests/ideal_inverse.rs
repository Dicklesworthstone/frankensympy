//! Quotient-ring inverses, complete nonunits and corrupted composition claims.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    IdealInverseCertificate, MultivariatePoly, TermOrder, VerifiedIdealInverse,
    bounded_polynomial_from_expr, checked_groebner_basis, checked_ideal_inverse,
    verify_ideal_inverse,
};
use std::collections::BTreeMap;
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn poly(source: &str, generators: &[Symbol]) -> MultivariatePoly {
    bounded_polynomial_from_expr(&parse(source).unwrap(), generators, &mut Unbounded).unwrap()
}

fn polys(sources: &[&str], generators: &[Symbol]) -> Vec<MultivariatePoly> {
    sources
        .iter()
        .map(|source| poly(source, generators))
        .collect()
}

fn invert(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
) -> VerifiedIdealInverse {
    checked_ideal_inverse(
        inputs,
        element,
        generators,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn rejected(result: &VerifiedIdealInverse, certificate: &IdealInverseCertificate) {
    assert!(
        verify_ideal_inverse(
            result.inputs(),
            result.element(),
            result.generators(),
            result.order(),
            certificate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn positive_dimensional_ideals_support_exact_inverses() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = polys(&["x*y-1", "y*z-1"], &generators);
    let element = poly("x", &generators);
    let result = invert(&inputs, &element, &generators);
    assert_eq!(result.inverse(), Some(&poly("y", &generators)));
    assert!(result.is_invertible());
    assert_eq!(result.inputs(), inputs);
    assert_eq!(result.element(), &element);
    assert_eq!(result.generators(), generators);
    let replay = verify_ideal_inverse(
        &inputs,
        &element,
        &generators,
        TermOrder::Lex,
        result.certificate(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(replay.inverse(), result.inverse());
}

#[test]
fn nilpotent_quotients_keep_the_full_inverse_multiplicity() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["x**2", "y**2"], &generators);
    let element = poly("1+x+y", &generators);
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let result = checked_ideal_inverse(
            &inputs,
            &element,
            &generators,
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(result.inverse(), Some(&poly("1-x-y+2*x*y", &generators)));
        assert_eq!(result.order(), order);
    }
    let result = invert(
        &polys(&["x**3"], &generators),
        &poly("1-x", &generators),
        &generators,
    );
    assert_eq!(result.inverse(), Some(&poly("1+x+x**2", &generators)));
}

#[test]
fn inverse_is_reduced_in_the_original_ideal_and_term_order() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["x**2-2", "y-x"], &generators);
    let result = invert(&inputs, &poly("x", &generators), &generators);
    assert_eq!(result.inverse(), Some(&poly("y/2", &generators)));
    let reverse = ring(&["y", "x"]);
    let result = invert(
        &polys(&["x**2-2", "y-x"], &reverse),
        &poly("x", &reverse),
        &reverse,
    );
    assert_eq!(result.inverse(), Some(&poly("x/2", &reverse)));
}

#[test]
fn nonunits_are_complete_conclusions_even_when_not_zero_divisors() {
    let generators = ring(&["x", "y"]);
    // x in QQ[x,y] is a nonunit but is not a zero divisor.
    let domain = invert(&[], &poly("x", &generators), &generators);
    assert!(!domain.is_invertible());
    assert!(domain.inverse().is_none());
    assert!(domain.certificate().ideal_basis.is_none());
    for (inputs, element) in [
        (vec!["x**2"], "x"),
        (vec!["x*y"], "x"),
        (vec!["x**2+y**2-1"], "x"),
        (vec!["x*y-1"], "x+y"),
        (vec![], "0"),
    ] {
        let source = polys(&inputs, &generators);
        let element = poly(element, &generators);
        let result = invert(&source, &element, &generators);
        assert!(!result.is_invertible());
        verify_ideal_inverse(
            &source,
            &element,
            &generators,
            TermOrder::Lex,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap();
    }
}

#[test]
fn zero_unit_and_constant_rings_have_explicit_conventions() {
    for generators in [Vec::new(), ring(&["x", "y"])] {
        for inputs in [vec![], vec!["0"], vec!["0", "0"]] {
            let result = invert(
                &polys(&inputs, &generators),
                &poly("7/3", &generators),
                &generators,
            );
            assert_eq!(result.into_inverse(), Some(poly("3/7", &generators)));
        }
        for element in ["0", "1", "-7/13"] {
            let result = invert(
                &polys(&["0", "5/2"], &generators),
                &poly(element, &generators),
                &generators,
            );
            assert_eq!(result.inverse(), Some(&poly("0", &generators)));
        }
    }
}

#[test]
fn original_zero_redundant_and_scaled_generators_keep_their_coordinates() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["0", "2*(x*y-1)", "3*(x*y-1)"], &generators);
    let element = poly("5*x/7", &generators);
    let result = invert(&inputs, &element, &generators);
    assert_eq!(result.inverse(), Some(&poly("7*y/5", &generators)));
    assert_eq!(result.certificate().bezout_coefficients.len(), 4);
    let changed = polys(&["0", "4*(x*y-1)", "3*(x*y-1)"], &generators);
    assert!(
        verify_ideal_inverse(
            &changed,
            &element,
            &generators,
            TermOrder::Lex,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn same_printed_names_do_not_alias_typed_quotient_ring_generators() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([23; 32]));
    let generators = vec![first.clone(), second.clone()];
    let input = MultivariatePoly::new(
        generators.clone(),
        BTreeMap::from([(vec![1, 1], 1.into()), (vec![0, 0], (-1).into())]),
    )
    .unwrap();
    let element = MultivariatePoly::var(generators.clone(), &first).unwrap();
    let result = invert(&[input], &element, &generators);
    assert_eq!(
        result.inverse(),
        Some(&MultivariatePoly::var(generators.clone(), &second).unwrap())
    );
    let reversed = vec![second, first];
    assert!(
        verify_ideal_inverse(
            result.inputs(),
            &element,
            &reversed,
            TermOrder::Lex,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn corrupt_bezout_cofactors_and_inverse_values_are_rejected() {
    let generators = ring(&["x", "y"]);
    let result = invert(
        &polys(&["x*y-1"], &generators),
        &poly("x", &generators),
        &generators,
    );
    let mut bad = result.certificate().clone();
    bad.bezout_coefficients[0] = poly("0", &generators);
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.inverse = Some(poly(
        "y+1/100000000000000000000000000000000000000",
        &generators,
    ));
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.bezout_coefficients.pop();
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.reduction_quotients.push(poly("0", &generators));
    rejected(&result, &bad);
}

#[test]
fn a_correct_but_unreduced_inverse_cannot_replace_the_normal_form() {
    let generators = ring(&["x"]);
    let result = invert(
        &polys(&["x**2-2"], &generators),
        &poly("x", &generators),
        &generators,
    );
    let mut bad = result.certificate().clone();
    bad.inverse = Some(poly("x/2+x**2-2", &generators));
    // Preserve the reduction identity while changing the remainder to an
    // equivalent but non-reduced representative. The checker must still fail.
    bad.reduction_quotients[0] = bad.reduction_quotients[0]
        .sub(&poly("1", &generators))
        .unwrap();
    rejected(&result, &bad);
}

#[test]
fn unit_and_nonunit_conclusions_cannot_be_swapped() {
    let generators = ring(&["x", "y"]);
    let unit = invert(
        &polys(&["x*y-1"], &generators),
        &poly("x", &generators),
        &generators,
    );
    let mut bad = unit.certificate().clone();
    bad.inverse = None;
    bad.ideal_basis = None;
    bad.bezout_coefficients.clear();
    bad.reduction_quotients.clear();
    rejected(&unit, &bad);
    let nonunit = invert(&[], &poly("x", &generators), &generators);
    let mut bad = nonunit.certificate().clone();
    bad.inverse = Some(poly("0", &generators));
    rejected(&nonunit, &bad);
}

#[test]
fn accepted_lower_receipts_cannot_be_rebound_or_change_order() {
    let generators = ring(&["x", "y"]);
    let inputs = polys(&["x*y-1"], &generators);
    let result = invert(&inputs, &poly("x", &generators), &generators);
    let mut bad = result.certificate().clone();
    bad.augmented_basis = Arc::new(
        checked_groebner_basis(
            &polys(&["2*(x*y-1)", "x"], &generators),
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.ideal_basis = Some(Arc::new(
        checked_groebner_basis(&inputs, TermOrder::DegLex, &mut Unbounded, &mut Unbounded).unwrap(),
    ));
    rejected(&result, &bad);
    let mut bad = result.certificate().clone();
    bad.ideal_basis = Some(Arc::new(
        checked_groebner_basis(
            &polys(&["x*y-2"], &generators),
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    ));
    rejected(&result, &bad);
}

#[test]
fn source_bounds_and_malformed_storage_are_checked_before_shortcuts() {
    let generators = ring(&["x", "y"]);
    let unit = poly("1", &generators);
    let mut malformed = poly("x", &generators);
    malformed.terms.insert(vec![0], 1.into());
    let huge = MultivariatePoly::new(
        generators.clone(),
        BTreeMap::from([(vec![0, 0], BigRational::from(BigInt::from(1) << 4096u32))]),
    )
    .unwrap();
    let high_degree = MultivariatePoly::new(
        generators.clone(),
        BTreeMap::from([(vec![257, 0], 1.into())]),
    )
    .unwrap();
    for element in [malformed, huge, high_degree] {
        assert!(
            checked_ideal_inverse(
                std::slice::from_ref(&unit),
                &element,
                &generators,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked_ideal_inverse(
            &vec![unit.clone(); 64],
            &unit,
            &generators,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let large_ring: Vec<_> = (0..9).map(|i| Symbol::new(format!("v{i}"))).collect();
    assert!(
        checked_ideal_inverse(
            &[],
            &MultivariatePoly::one(large_ring.clone()),
            &large_ring,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let names = vec![Symbol::new("x".repeat(65_537))];
    assert!(
        checked_ideal_inverse(
            &[],
            &MultivariatePoly::one(names.clone()),
            &names,
            TermOrder::Lex,
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
    memory: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.compute += amount;
        }
        if dimension == Dimension::MemoryBytes {
            self.memory += amount;
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
fn cancellation_in_both_phases_and_at_publication_never_becomes_a_nonunit() {
    let generators = ring(&["x", "y"]);
    for (inputs, element) in [(vec!["x**2", "y**2"], "1+x+y"), (vec![], "x")] {
        let inputs = polys(&inputs, &generators);
        let element = poly(element, &generators);
        let mut generation = Meter::default();
        let mut verification = Meter::default();
        checked_ideal_inverse(
            &inputs,
            &element,
            &generators,
            TermOrder::Lex,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        assert!(generation.compute > 0 && verification.compute > 0);
        assert!(generation.memory > 0 && verification.memory > 0);
        for checkpoint in [1, generation.checkpoints / 2, generation.checkpoints] {
            let mut meter = Meter {
                cancel_at: Some(checkpoint),
                ..Meter::default()
            };
            assert!(
                checked_ideal_inverse(
                    &inputs,
                    &element,
                    &generators,
                    TermOrder::Lex,
                    &mut meter,
                    &mut Unbounded
                )
                .is_err()
            );
        }
        for checkpoint in [1, verification.checkpoints / 2, verification.checkpoints] {
            let mut meter = Meter {
                cancel_at: Some(checkpoint),
                ..Meter::default()
            };
            assert!(
                checked_ideal_inverse(
                    &inputs,
                    &element,
                    &generators,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut meter
                )
                .is_err()
            );
        }
    }
}

#[test]
fn cumulative_budget_exhaustion_is_terminal_for_success_and_nonunits() {
    let generators = ring(&["x", "y"]);
    let limited = |steps| {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = steps;
        Budget::new(limits)
    };
    for (inputs, element) in [(vec!["x*y-1"], "x"), (vec![], "x")] {
        let inputs = polys(&inputs, &generators);
        let element = poly(element, &generators);
        let mut generation = Meter::default();
        let mut verification = Meter::default();
        checked_ideal_inverse(
            &inputs,
            &element,
            &generators,
            TermOrder::Lex,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        assert!(
            checked_ideal_inverse(
                &inputs,
                &element,
                &generators,
                TermOrder::Lex,
                &mut limited(generation.compute - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked_ideal_inverse(
                &inputs,
                &element,
                &generators,
                TermOrder::Lex,
                &mut Unbounded,
                &mut limited(verification.compute - 1)
            )
            .is_err()
        );
        checked_ideal_inverse(
            &inputs,
            &element,
            &generators,
            TermOrder::Lex,
            &mut limited(generation.compute),
            &mut limited(verification.compute),
        )
        .unwrap();
    }
}
