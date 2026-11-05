//! Finite ideal quotient identities and adversarial composition certificates.
//! Expected generators are hand-derived ideals; neither the generator under
//! test nor a numerical oracle supplies the expected answers.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    IdealQuotientCertificate, MultivariatePoly, TermOrder, VerifiedIdealQuotient,
    bounded_polynomial_from_expr, checked_groebner_basis, checked_ideal_intersection,
    checked_ideal_quotient, checked_ideal_saturation, verify_ideal_quotient,
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

fn quotient(
    inputs: &[MultivariatePoly],
    divisors: &[MultivariatePoly],
    generators: &[Symbol],
) -> VerifiedIdealQuotient {
    checked_ideal_quotient(inputs, divisors, generators, &mut Unbounded, &mut Unbounded).unwrap()
}

fn equal_basis(actual: &[MultivariatePoly], expected: &[MultivariatePoly]) {
    assert_eq!(actual.len(), expected.len());
    for polynomial in expected {
        assert!(
            actual.contains(polynomial),
            "missing hand-derived ideal generator {polynomial:?}"
        );
    }
}

#[test]
fn principal_quotients_remove_only_the_supplied_multiplicity() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**3*y"], &generators);
    for (divisor, expected) in [
        ("x", "x**2*y"),
        ("x**2", "x*y"),
        ("x**4", "y"),
        ("x*y", "x**2"),
        ("y**2", "x**3"),
        ("x+1", "x**3*y"),
        ("7/13", "x**3*y"),
    ] {
        let divisors = polynomials(&[divisor], &generators);
        let result = quotient(&inputs, &divisors, &generators);
        equal_basis(result.basis(), &polynomials(&[expected], &generators));
        assert_eq!(result.generators(), generators);
        assert_eq!(result.inputs(), inputs);
        assert_eq!(result.divisors(), divisors);
        let replay = verify_ideal_quotient(
            &inputs,
            &divisors,
            &generators,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(result.basis(), replay.basis());
    }
    let finite = quotient(&inputs, &polynomials(&["x"], &generators), &generators);
    let saturation = checked_ideal_saturation(
        &inputs,
        &polynomial("x", &generators),
        &generators,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    equal_basis(saturation.basis(), &polynomials(&["y"], &generators));
    assert_ne!(finite.basis(), saturation.basis());
}

#[test]
fn multigenerator_divisor_is_an_intersection_of_all_principal_quotients() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "y**2"], &generators);
    let divisors = polynomials(&["x", "y"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    equal_basis(
        result.basis(),
        &polynomials(&["x**2", "x*y", "y**2"], &generators),
    );
    assert_eq!(result.certificate().principal_quotients.len(), 2);
    assert_eq!(result.certificate().intersections.len(), 1);
    let reverse = quotient(&inputs, &polynomials(&["y", "x"], &generators), &generators);
    assert_eq!(result.basis(), reverse.basis());
    let original = polynomials(&["x*y"], &generators);
    let result = quotient(&original, &divisors, &generators);
    equal_basis(result.basis(), &original);
}

#[test]
fn nonmonomial_and_rational_divisors_preserve_exact_ideal_coefficients() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["(x-1)**2*(y+1)", "(x-1)*(y+1)**2"], &generators);
    let divisors = polynomials(&["(x-1)*(y+1)"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    equal_basis(result.basis(), &polynomials(&["x-1", "y+1"], &generators));

    let inputs = polynomials(&["(2*x/3-1)*(5*y/7+2)"], &generators);
    let divisors = polynomials(&["2*x/3-1"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    equal_basis(result.basis(), &polynomials(&["y+14/5"], &generators));
}

#[test]
fn zero_empty_and_unit_ideals_obey_the_full_colon_laws() {
    let generators = ring(&["x", "y"]);
    let ideal = polynomials(&["x**2", "y"], &generators);
    let unit = polynomials(&["1"], &generators);
    for input in [Vec::new(), ideal.clone(), unit.clone()] {
        for divisors in [
            Vec::new(),
            polynomials(&["0"], &generators),
            polynomials(&["0", "0"], &generators),
        ] {
            let result = quotient(&input, &divisors, &generators);
            assert_eq!(result.basis(), unit);
            assert!(result.certificate().principal_quotients.is_empty());
            assert!(result.certificate().intersections.is_empty());
        }
    }
    let result = quotient(&[], &polynomials(&["x", "y"], &generators), &generators);
    assert!(result.basis().is_empty());
    let result = quotient(&ideal, &unit, &generators);
    equal_basis(result.basis(), &ideal);
    let result = quotient(&unit, &ideal, &generators);
    assert_eq!(result.basis(), unit);
    let result = quotient(
        &ideal,
        &polynomials(&["x", "1-x"], &generators),
        &generators,
    );
    equal_basis(result.basis(), &ideal);
}

#[test]
fn constant_rings_have_the_same_exact_zero_and_unit_contract() {
    let generators = Vec::new();
    for (inputs, divisors, expected) in [
        (vec![], vec!["1"], vec![]),
        (vec![], vec!["0"], vec!["1"]),
        (vec!["7/3"], vec!["2/5"], vec!["1"]),
        (vec!["0"], vec![], vec!["1"]),
    ] {
        let result = quotient(
            &polynomials(&inputs, &generators),
            &polynomials(&divisors, &generators),
            &generators,
        );
        equal_basis(result.basis(), &polynomials(&expected, &generators));
        assert!(result.generators().is_empty());
    }
}

#[test]
fn zero_and_repeated_divisors_retain_original_order_without_losing_obligations() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "y**2"], &generators);
    let divisors = polynomials(&["0", "x", "0", "y", "x"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    equal_basis(
        result.basis(),
        &polynomials(&["x**2", "x*y", "y**2"], &generators),
    );
    assert_eq!(
        result
            .certificate()
            .principal_quotients
            .iter()
            .map(|witness| witness.divisor_index)
            .collect::<Vec<_>>(),
        vec![1, 3, 4]
    );
    assert_eq!(result.certificate().intersections.len(), 2);
}

#[test]
fn typed_same_print_symbols_and_occupied_auxiliary_names_never_capture_each_other() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([97; 32]));
    let occupied = Symbol::new("_fsym_ideal_parameter_0");
    let generators = vec![first.clone(), second.clone(), occupied];
    let product = MultivariatePoly::new(
        generators.clone(),
        BTreeMap::from([(vec![1, 1, 0], 1.into())]),
    )
    .unwrap();
    let divisor = MultivariatePoly::var(generators.clone(), &first).unwrap();
    let result = quotient(
        std::slice::from_ref(&product),
        std::slice::from_ref(&divisor),
        &generators,
    );
    assert_eq!(
        result.basis(),
        &[MultivariatePoly::var(generators.clone(), &second).unwrap()]
    );
    let auxiliary = &result.certificate().principal_quotients[0]
        .intersection
        .certificate()
        .ambient_basis
        .input_basis()[0]
        .generators()[0];
    assert!(
        generators
            .iter()
            .all(|symbol| symbol.name != auxiliary.name)
    );
    assert!(
        verify_ideal_quotient(
            &[product],
            &[MultivariatePoly::var(generators.clone(), &second).unwrap()],
            &generators,
            result.certificate(),
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn certificates_cannot_omit_reorder_duplicate_or_add_nonzero_divisor_witnesses() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "y**2"], &generators);
    let divisors = polynomials(&["x", "0", "y"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    let rejected = |certificate: &IdealQuotientCertificate| {
        assert!(
            verify_ideal_quotient(&inputs, &divisors, &generators, certificate, &mut Unbounded)
                .is_err()
        );
    };
    let mut bad = result.certificate().clone();
    bad.principal_quotients.pop();
    rejected(&bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients.swap(0, 1);
    rejected(&bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients[1] = bad.principal_quotients[0].clone();
    rejected(&bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients[1].divisor_index = 1;
    rejected(&bad);
    let mut bad = result.certificate().clone();
    bad.principal_quotients
        .push(bad.principal_quotients[0].clone());
    rejected(&bad);
}

#[test]
fn quotient_identity_checker_rejects_a_coordinated_cofactor_and_basis_attack() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2*y"], &generators);
    let divisors = polynomials(&["x"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotients[0] =
        polynomial("x*y+1/100000000000000000000000000000000000000", &generators);
    // A valid independently accepted basis of the wrong cofactor must still
    // fail the original dividend=f*cofactor coefficient identity.
    bad.principal_quotients[0].quotient_basis = Arc::new(
        checked_groebner_basis(
            &bad.principal_quotients[0].quotients,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.basis = bad.principal_quotients[0].quotient_basis.basis().to_vec();
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());
}

#[test]
fn every_intersection_generator_and_the_entire_quotient_basis_are_required() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "x*y"], &generators);
    let divisors = polynomials(&["x"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    assert_eq!(
        result.certificate().principal_quotients[0].quotients.len(),
        2
    );
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotients.pop();
    bad.principal_quotients[0].quotient_basis = Arc::new(
        checked_groebner_basis(
            &bad.principal_quotients[0].quotients,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.basis = bad.principal_quotients[0].quotient_basis.basis().to_vec();
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());

    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotient_basis = Arc::new(
        checked_groebner_basis(
            &polynomials(&["x+1", "y"], &generators),
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.basis = bad.principal_quotients[0].quotient_basis.basis().to_vec();
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotient_basis = Arc::new(
        checked_groebner_basis(
            &bad.principal_quotients[0].quotients,
            TermOrder::DegLex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());
}

#[test]
fn accepted_intersections_cannot_be_rebound_to_scaled_inputs_or_different_operations() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2*y"], &generators);
    let divisors = polynomials(&["x"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    for (changed_inputs, changed_divisors) in [
        (polynomials(&["2*x**2*y"], &generators), divisors.clone()),
        (inputs.clone(), polynomials(&["2*x"], &generators)),
        (inputs.clone(), polynomials(&["y"], &generators)),
    ] {
        assert!(
            verify_ideal_quotient(
                &changed_inputs,
                &changed_divisors,
                &generators,
                result.certificate(),
                &mut Unbounded,
            )
            .is_err()
        );
    }
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].intersection = Arc::new(
        checked_ideal_saturation(
            &inputs,
            &divisors[0],
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());
}

#[test]
fn final_intersection_chain_cannot_omit_an_obligation_or_use_an_unrelated_receipt() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "y**2"], &generators);
    let divisors = polynomials(&["x", "y"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    let mut bad = result.certificate().clone();
    bad.intersections.clear();
    bad.basis = bad.principal_quotients[0].quotient_basis.basis().to_vec();
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());
    let mut bad = result.certificate().clone();
    bad.intersections[0] = Arc::new(
        checked_ideal_intersection(
            &polynomials(&["x"], &generators),
            &polynomials(&["y"], &generators),
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    );
    bad.basis = bad.intersections[0].basis().to_vec();
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());
    let mut bad = result.certificate().clone();
    bad.intersections.push(bad.intersections[0].clone());
    assert!(verify_ideal_quotient(&inputs, &divisors, &generators, &bad, &mut Unbounded).is_err());
}

#[test]
fn final_basis_cannot_be_shortened_extended_or_replaced_by_saturation() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "x*y"], &generators);
    let divisors = polynomials(&["x"], &generators);
    let result = quotient(&inputs, &divisors, &generators);
    let rejected = |certificate: &IdealQuotientCertificate| {
        assert!(
            verify_ideal_quotient(&inputs, &divisors, &generators, certificate, &mut Unbounded)
                .is_err()
        );
    };
    let mut bad = result.certificate().clone();
    bad.basis.pop();
    rejected(&bad);
    let mut bad = result.certificate().clone();
    bad.basis.push(MultivariatePoly::one(generators.clone()));
    rejected(&bad);
    let mut bad = result.certificate().clone();
    bad.basis = polynomials(&["1"], &generators);
    rejected(&bad);
    let zero_divisor = quotient(&inputs, &[], &generators);
    let mut bad = zero_divisor.certificate().clone();
    bad.basis.clear();
    assert!(verify_ideal_quotient(&inputs, &[], &generators, &bad, &mut Unbounded).is_err());
}

#[test]
fn malformed_sources_and_certificates_refuse_before_algebraic_shortcuts() {
    let generators = ring(&["x", "y"]);
    let valid = polynomial("x*y", &generators);
    let mut malformed_width = valid.clone();
    malformed_width.terms.insert(vec![1], 1.into());
    let mut zero_coefficient = valid.clone();
    zero_coefficient.terms.insert(vec![0, 0], 0.into());
    let oversized_degree = MultivariatePoly {
        generators: generators.clone(),
        terms: BTreeMap::from([(vec![257, 0], 1.into())]),
    };
    let oversized_coefficient = MultivariatePoly {
        generators: generators.clone(),
        terms: BTreeMap::from([(
            vec![1, 0],
            BigRational::from_integer(BigInt::from(1_u32) << 4097_u32),
        )]),
    };
    for bad in [
        malformed_width,
        zero_coefficient,
        oversized_degree,
        oversized_coefficient,
    ] {
        assert!(
            checked_ideal_quotient(
                std::slice::from_ref(&bad),
                &[],
                &generators,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
        assert!(
            checked_ideal_quotient(
                std::slice::from_ref(&valid),
                std::slice::from_ref(&bad),
                &generators,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
    let result = quotient(&[valid], &polynomials(&["x"], &generators), &generators);
    let mut bad = result.certificate().clone();
    bad.principal_quotients[0].quotients[0]
        .terms
        .insert(vec![0], 1.into());
    assert!(
        verify_ideal_quotient(
            result.inputs(),
            result.divisors(),
            &generators,
            &bad,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn mismatched_rings_and_oversized_auxiliary_constructions_refuse() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x*y"], &generators);
    let divisors = polynomials(&["x"], &generators);
    assert!(
        checked_ideal_quotient(
            &inputs,
            &polynomials(&["x"], &ring(&["x"])),
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let reversed = vec![generators[1].clone(), generators[0].clone()];
    assert!(
        checked_ideal_quotient(
            &inputs,
            &divisors,
            &reversed,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let duplicate = vec![generators[0].clone(), generators[0].clone()];
    assert!(checked_ideal_quotient(&[], &[], &duplicate, &mut Unbounded, &mut Unbounded).is_err());
    let large_ring: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let divisor = MultivariatePoly::var(large_ring.clone(), &large_ring[0]).unwrap();
    assert!(
        checked_ideal_quotient(&[], &[divisor], &large_ring, &mut Unbounded, &mut Unbounded)
            .is_err()
    );
    assert!(
        checked_ideal_quotient(
            &vec![inputs[0].clone(); 64],
            &divisors,
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        checked_ideal_quotient(
            &inputs,
            &vec![divisors[0].clone(); 65],
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let high = polynomial("x**256", &generators);
    assert!(
        checked_ideal_quotient(
            &[high],
            &divisors,
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        checked_ideal_quotient(
            &[],
            &[],
            &[Symbol::new("x".repeat(65_537))],
            &mut Unbounded,
            &mut Unbounded,
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
        } else if dimension == Dimension::MemoryBytes {
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
fn generator_and_verifier_cancellation_including_final_publication_never_returns_a_basis() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "y**2"], &generators);
    let divisors = polynomials(&["x", "y"], &generators);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    checked_ideal_quotient(
        &inputs,
        &divisors,
        &generators,
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
            checked_ideal_quotient(&inputs, &divisors, &generators, &mut meter, &mut Unbounded)
                .is_err()
        );
    }
    for checkpoint in [1, verification.checkpoints / 2, verification.checkpoints] {
        let mut meter = Meter {
            cancel_at: Some(checkpoint),
            ..Meter::default()
        };
        assert!(
            checked_ideal_quotient(&inputs, &divisors, &generators, &mut Unbounded, &mut meter)
                .is_err()
        );
    }
}

#[test]
fn exhausted_budgets_span_all_divisors_intersections_and_publication() {
    let generators = ring(&["x", "y"]);
    let inputs = polynomials(&["x**2", "y**2"], &generators);
    let divisors = polynomials(&["x", "y"], &generators);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    let result = checked_ideal_quotient(
        &inputs,
        &divisors,
        &generators,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    let compute_budget = |steps| {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = steps;
        Budget::new(limits)
    };
    assert!(
        checked_ideal_quotient(
            &inputs,
            &divisors,
            &generators,
            &mut compute_budget(generation.compute - 1),
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        checked_ideal_quotient(
            &inputs,
            &divisors,
            &generators,
            &mut Unbounded,
            &mut compute_budget(verification.compute - 1),
        )
        .is_err()
    );
    let mut generation_budget = compute_budget(generation.compute);
    let mut verification_budget = compute_budget(verification.compute);
    checked_ideal_quotient(
        &inputs,
        &divisors,
        &generators,
        &mut generation_budget,
        &mut verification_budget,
    )
    .unwrap();
    assert!(
        checked_ideal_quotient(
            &inputs,
            &divisors,
            &generators,
            &mut generation_budget,
            &mut verification_budget,
        )
        .is_err()
    );
    assert!(
        verify_ideal_quotient(
            &inputs,
            &divisors,
            &generators,
            result.certificate(),
            &mut Budget::new(BudgetLimits::uniform(1, 0)),
        )
        .is_err()
    );
    let mut unused_verifier = Meter::default();
    assert!(
        checked_ideal_quotient(
            &inputs,
            &divisors,
            &generators,
            &mut Budget::new(BudgetLimits::uniform(1, 0)),
            &mut unused_verifier,
        )
        .is_err()
    );
    assert_eq!(unused_verifier.checkpoints, 0);
}
