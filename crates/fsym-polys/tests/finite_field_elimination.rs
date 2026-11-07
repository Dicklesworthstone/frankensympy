//! Independently derived finite-field elimination ideals and adversarial
//! composition receipts. Exact expected bases test equality, not just that
//! each published polynomial happens to belong to the requested ideal.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    IdealOperation, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder,
    bounded_prime_field_groebner_certificate as groebner_candidate,
    checked_prime_field_elimination_ideal as eliminate,
    checked_prime_field_groebner_basis as groebner,
    verify_prime_field_elimination_ideal as verify,
    verify_prime_field_groebner_certificate as verify_groebner,
};
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn polynomial(generators: &[Symbol], prime: u64, terms: &[(&[u32], u64)]) -> Polynomial {
    let terms: Terms = terms.iter().map(|(powers, c)| (powers.to_vec(), *c)).collect();
    Polynomial::new(generators, prime, &terms).unwrap()
}

#[test]
fn complete_elimination_retains_a_nonlinear_relation_in_every_characteristic() {
    let generators = ring(&["x", "y", "z"]);
    for prime in [2, 3, 5, 65521] {
        // x=y^2 and x=z imply precisely y^2-z=0 in F_p[y,z].
        let inputs = [
            polynomial(&generators, prime, &[(&[1, 0, 0], 1), (&[0, 2, 0], prime - 1)]),
            polynomial(&generators, prime, &[(&[1, 0, 0], 1), (&[0, 0, 1], prime - 1)]),
        ];
        let result = eliminate(
            &generators, prime, &inputs, &generators[..1], &mut Unbounded, &mut Unbounded,
        ).unwrap();
        assert_eq!(result.operation(), IdealOperation::Elimination);
        assert_eq!(result.prime(), prime);
        assert_eq!(result.input_generators(), generators);
        assert_eq!(result.inputs(), inputs);
        assert_eq!(result.generators(), &generators[1..]);
        assert_eq!(result.eliminated_variables(), &generators[..1]);
        assert!(result.other_inputs().is_empty());
        assert!(result.saturating_polynomial().is_none());
        assert_eq!(result.basis(), [polynomial(
            &generators[1..], prime, &[(&[2, 0], 1), (&[0, 1], prime - 1)],
        )]);
        verify(&generators, prime, &inputs, &generators[..1], result.certificate(), &mut Unbounded).unwrap();
    }
}

#[test]
fn nonleading_elimination_reorders_the_ambient_ring_before_selecting() {
    let generators = ring(&["x", "y", "z"]);
    let retained = vec![generators[0].clone(), generators[2].clone()];
    for prime in [2, 7] {
        let inputs = [
            polynomial(&generators, prime, &[(&[1, 0, 0], 1), (&[0, 2, 0], prime - 1)]),
            polynomial(&generators, prime, &[(&[0, 1, 0], 1), (&[0, 0, 1], prime - 1)]),
        ];
        let result = eliminate(
            &generators, prime, &inputs, &generators[1..2], &mut Unbounded, &mut Unbounded,
        ).unwrap();
        assert_eq!(result.generators(), retained);
        assert_eq!(result.basis(), [polynomial(
            &retained, prime, &[(&[1, 0], 1), (&[0, 2], prime - 1)],
        )]);
    }
}

#[test]
fn noncontiguous_elimination_preserves_the_retained_order_and_all_components() {
    let generators = ring(&["x", "y", "z", "w"]);
    let retained = vec![generators[1].clone(), generators[3].clone()];
    for prime in [2, 3, 11] {
        // x=y^2, z=y*w, x=z leaves y*(y-w), including both components.
        let inputs = [
            polynomial(&generators, prime, &[(&[1, 0, 0, 0], 1), (&[0, 2, 0, 0], prime - 1)]),
            polynomial(&generators, prime, &[(&[0, 0, 1, 0], 1), (&[0, 1, 0, 1], prime - 1)]),
            polynomial(&generators, prime, &[(&[1, 0, 0, 0], 1), (&[0, 0, 1, 0], prime - 1)]),
        ];
        for eliminated in [
            vec![generators[0].clone(), generators[2].clone()],
            vec![generators[2].clone(), generators[0].clone()],
        ] {
            let result = eliminate(
                &generators, prime, &inputs, &eliminated, &mut Unbounded, &mut Unbounded,
            ).unwrap();
            assert_eq!(result.generators(), retained);
            assert_eq!(result.eliminated_variables(), eliminated);
            assert_eq!(result.basis(), [polynomial(
                &retained, prime, &[(&[2, 0], 1), (&[1, 1], prime - 1)],
            )]);
        }
    }
}

#[test]
fn characteristic_two_retains_a_nonradical_ideal_that_is_unit_in_odd_characteristic() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let inputs = [
            polynomial(&generators, prime, &[(&[2, 0], 1), (&[0, 0], 1)]),
            polynomial(&generators, prime, &[(&[1, 0], 1), (&[0, 0], 1)]),
            polynomial(&generators, prime, &[(&[0, 2], 1)]),
        ];
        let result = eliminate(
            &generators, prime, &inputs, &generators[..1], &mut Unbounded, &mut Unbounded,
        ).unwrap();
        let expected = if prime == 2 {
            polynomial(&generators[1..], prime, &[(&[2], 1)])
        } else {
            polynomial(&generators[1..], prime, &[(&[0], 1)])
        };
        assert_eq!(result.basis(), [expected], "characteristic {prime}");
    }
}

#[test]
fn no_variables_all_variables_and_constant_rings_keep_zero_and_unit_distinct() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        let inputs = [
            polynomial(&generators, prime, &[(&[2, 0], 1)]),
            polynomial(&generators, prime, &[(&[0, 1], 1)]),
        ];
        let unchanged = eliminate(&generators, prime, &inputs, &[], &mut Unbounded, &mut Unbounded).unwrap();
        assert_eq!(unchanged.basis(), inputs);
        assert_eq!(unchanged.generators(), generators);
        for ideal in [vec![], inputs.to_vec()] {
            let result = eliminate(&generators, prime, &ideal, &generators, &mut Unbounded, &mut Unbounded).unwrap();
            assert!(result.generators().is_empty());
            assert!(result.basis().is_empty());
        }
        let unit = polynomial(&generators, prime, &[(&[0, 0], 1)]);
        let result = eliminate(&generators, prime, &[unit], &generators, &mut Unbounded, &mut Unbounded).unwrap();
        let constant_one = polynomial(&[], prime, &[(&[], 1)]);
        assert_eq!(result.basis(), std::slice::from_ref(&constant_one));
        assert!(result.generators().is_empty());
        let constant = eliminate(&[], prime, std::slice::from_ref(&constant_one), &[], &mut Unbounded, &mut Unbounded).unwrap();
        assert_eq!(constant.basis(), [constant_one]);
        assert!(eliminate(&[], prime, &[], &[], &mut Unbounded, &mut Unbounded).unwrap().basis().is_empty());
    }
    // x^2+1 has no F_3-rational root, but its polynomial ideal is proper.
    // Point enumeration or adding x^3-x would falsely produce the unit ideal.
    let variable = ring(&["x"]);
    let irreducible = polynomial(&variable, 3, &[(&[2], 1), (&[0], 1)]);
    let result = eliminate(&variable, 3, &[irreducible], &variable, &mut Unbounded, &mut Unbounded).unwrap();
    assert!(result.generators().is_empty());
    assert!(result.basis().is_empty());
}

#[test]
fn typed_same_spelling_variables_and_original_zero_scaled_duplicates_are_retained() {
    let first = Symbol::new("x");
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([83; 32]));
    let generators = vec![first.clone(), second.clone()];
    let zero = polynomial(&generators, 5, &[]);
    let source = polynomial(&generators, 5, &[(&[1, 0], 2), (&[0, 2], 3)]);
    let retained = polynomial(&generators, 5, &[(&[0, 3], 4)]);
    let inputs = [source.clone(), zero, retained, source];
    let result = eliminate(&generators, 5, &inputs, &[first], &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(result.inputs(), inputs);
    assert_eq!(result.generators(), std::slice::from_ref(&second));
    assert_eq!(result.basis(), [polynomial(&[second], 5, &[(&[3], 1)])]);
    let alien = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([84; 32]));
    assert!(eliminate(&generators, 5, &inputs, &[alien], &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn the_entire_projected_basis_and_its_order_are_required() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = [
        polynomial(&generators, 5, &[(&[1, 0, 0], 1)]),
        polynomial(&generators, 5, &[(&[0, 2, 0], 1)]),
        polynomial(&generators, 5, &[(&[0, 0, 2], 1)]),
    ];
    let result = eliminate(&generators, 5, &inputs, &generators[..1], &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(result.basis(), [
        polynomial(&generators[1..], 5, &[(&[2, 0], 1)]),
        polynomial(&generators[1..], 5, &[(&[0, 2], 1)]),
    ]);
    let reject = |certificate| {
        assert!(verify(&generators, 5, &inputs, &generators[..1], &certificate, &mut Unbounded).is_err());
    };
    let mut bad = result.certificate().clone();
    bad.projected_basis.pop(); // Every remaining member is valid, but incomplete.
    reject(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis.clear();
    reject(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis.push(polynomial(&generators[1..], 5, &[(&[0, 0], 1)]));
    reject(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis[0] = polynomial(&generators[1..], 5, &[(&[2, 0], 2)]);
    reject(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis.swap(0, 1);
    reject(bad);
    let mut bad = result.certificate().clone();
    bad.projected_basis.push(bad.projected_basis[0].clone());
    reject(bad);
}

#[test]
fn accepted_ambient_receipts_cannot_change_order_original_coefficients_or_prime() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        polynomial(&generators, 5, &[(&[1, 0], 1), (&[0, 2], 4)]),
        polynomial(&generators, 5, &[(&[0, 3], 1)]),
    ];
    let result = eliminate(&generators, 5, &inputs, &generators[..1], &mut Unbounded, &mut Unbounded).unwrap();
    let ambient = &result.certificate().ambient_basis;
    let mut bad = result.certificate().clone();
    bad.ambient_basis = Arc::new(groebner(ambient.generators(), 5, ambient.inputs(), TermOrder::DegRevLex, &mut Unbounded, &mut Unbounded).unwrap());
    assert!(verify(&generators, 5, &inputs, &generators[..1], &bad, &mut Unbounded).is_err());
    let mut altered = inputs.clone();
    altered[0] = polynomial(&generators, 5, &[(&[1, 0], 2), (&[0, 2], 3)]);
    // Scaling an input preserves its ideal but not the source-bound claim.
    assert!(verify(&generators, 5, &altered, &generators[..1], result.certificate(), &mut Unbounded).is_err());
    assert!(verify(&generators, 7, &inputs, &generators[..1], result.certificate(), &mut Unbounded).is_err());
    assert!(verify(&generators, 5, &inputs, &generators[1..], result.certificate(), &mut Unbounded).is_err());
    let other_inputs = [
        polynomial(&generators, 5, &[(&[1, 0], 1), (&[0, 2], 4)]),
        polynomial(&generators, 5, &[(&[0, 2], 1)]),
    ];
    let other = eliminate(&generators, 5, &other_inputs, &generators[..1], &mut Unbounded, &mut Unbounded).unwrap();
    assert!(verify(&generators, 5, &inputs, &generators[..1], other.certificate(), &mut Unbounded).is_err());
}

#[test]
fn a_forged_cofactor_or_omitted_spair_cannot_obtain_an_ambient_receipt() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        polynomial(&generators, 3, &[(&[1, 0], 1), (&[0, 2], 2)]),
        polynomial(&generators, 3, &[(&[0, 3], 1)]),
    ];
    let candidate = groebner_candidate(&generators, 3, &inputs, TermOrder::Lex, &mut Unbounded).unwrap();
    let mut bad = candidate.clone();
    bad.basis_in_input.iter_mut().flatten().find(|terms| !terms.is_empty()).unwrap().clear();
    assert!(verify_groebner(&generators, 3, &inputs, TermOrder::Lex, &bad, &mut Unbounded).is_err());
    let mut bad = candidate.clone();
    bad.input_in_basis.pop();
    assert!(verify_groebner(&generators, 3, &inputs, TermOrder::Lex, &bad, &mut Unbounded).is_err());
    let mut bad = candidate;
    assert!(bad.s_pairs.pop().is_some());
    assert!(verify_groebner(&generators, 3, &inputs, TermOrder::Lex, &bad, &mut Unbounded).is_err());
}

#[test]
fn malformed_typed_rings_and_elimination_lists_refuse_before_publication() {
    let generators = ring(&["x", "y"]);
    let inputs = [polynomial(&generators, 5, &[(&[1, 1], 1)])];
    for eliminated in [vec![Symbol::new("absent")], vec![generators[0].clone(); 2]] {
        assert!(eliminate(&generators, 5, &inputs, &eliminated, &mut Unbounded, &mut Unbounded).is_err());
    }
    let reversed = vec![generators[1].clone(), generators[0].clone()];
    assert!(eliminate(&reversed, 5, &inputs, &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(eliminate(&generators, 7, &inputs, &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(eliminate(&generators, 15, &[], &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(eliminate(&[], 15, &[], &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(eliminate(&[generators[0].clone(), generators[0].clone()], 5, &[], &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(eliminate(&generators, 5, &vec![inputs[0].clone(); 33], &[], &mut Unbounded, &mut Unbounded).is_err());
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
        if self.cancel == Some(self.checkpoints) { Err(MeterError::Cancelled) } else { Ok(()) }
    }
}

fn cap(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn elimination_construction_nested_groebner_and_publication_share_phase_ledgers() {
    let generators = ring(&["x", "y"]);
    let inputs = [
        polynomial(&generators, 5, &[(&[1, 0], 1), (&[0, 2], 4)]),
        polynomial(&generators, 5, &[(&[0, 3], 1)]),
    ];
    let (mut generation, mut verification) = (Count::default(), Count::default());
    eliminate(&generators, 5, &inputs, &generators[..1], &mut generation, &mut verification).unwrap();
    for (dimension, generated, verified) in [
        (Dimension::ComputeSteps, generation.steps, verification.steps),
        (Dimension::MemoryBytes, generation.bytes, verification.bytes),
        (Dimension::AllocationCount, generation.allocations, verification.allocations),
    ] {
        assert!(generated > 1 && verified > 1);
        eliminate(&generators, 5, &inputs, &generators[..1], &mut cap(dimension, generated), &mut cap(dimension, verified)).unwrap();
        assert!(eliminate(&generators, 5, &inputs, &generators[..1], &mut cap(dimension, generated - 1), &mut Unbounded).is_err());
        assert!(eliminate(&generators, 5, &inputs, &generators[..1], &mut Unbounded, &mut cap(dimension, verified - 1)).is_err());
        let mut used = cap(dimension, generated);
        used.charge(dimension, 1).unwrap();
        assert!(eliminate(&generators, 5, &inputs, &generators[..1], &mut used, &mut Unbounded).is_err());
    }
}

#[test]
fn generator_verifier_and_final_publication_cancellation_are_terminal() {
    let generators = ring(&["x", "y"]);
    let inputs = [polynomial(&generators, 2, &[(&[1, 0], 1), (&[0, 1], 1)])];
    let (mut generation, mut verification) = (Count::default(), Count::default());
    let result = eliminate(&generators, 2, &inputs, &generators[..1], &mut generation, &mut verification).unwrap();
    for (in_generator, checkpoints) in [(true, generation.checkpoints), (false, verification.checkpoints)] {
        for at in [1, checkpoints / 2, checkpoints - 1, checkpoints] {
            let mut stopped = Count { cancel: Some(at), ..Count::default() };
            let result = if in_generator {
                eliminate(&generators, 2, &inputs, &generators[..1], &mut stopped, &mut Unbounded)
            } else {
                eliminate(&generators, 2, &inputs, &generators[..1], &mut Unbounded, &mut stopped)
            };
            assert!(result.is_err(), "generator={in_generator}, checkpoint={at}");
        }
    }
    let mut never_started = Count::default();
    assert!(eliminate(&generators, 2, &inputs, &generators[..1], &mut Budget::new(BudgetLimits::uniform(1, 0)), &mut never_started).is_err());
    assert_eq!(never_started.checkpoints, 0);
    let mut standalone = Count::default();
    verify(&generators, 2, &inputs, &generators[..1], result.certificate(), &mut standalone).unwrap();
    let mut stopped = Count { cancel: Some(standalone.checkpoints), ..Count::default() };
    assert!(verify(&generators, 2, &inputs, &generators[..1], result.certificate(), &mut stopped).is_err());
}
