//! Radical equalities with independently identified point sets and hostile
//! inclusion, square-free and nilpotence witnesses.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly, TermOrder, VerifiedZeroDimensionalRadical, bounded_polynomial_from_expr,
    checked_groebner_basis, checked_zero_dimensional_radical, verify_zero_dimensional_radical,
};

fn variables() -> Vec<Symbol> {
    vec![Symbol::new("x"), Symbol::new("y")]
}

fn polynomial(source: &str, variables: &[Symbol]) -> MultivariatePoly {
    bounded_polynomial_from_expr(&parse(source).unwrap(), variables, &mut Unbounded).unwrap()
}

fn polynomials(sources: &[&str], variables: &[Symbol]) -> Vec<MultivariatePoly> {
    sources
        .iter()
        .map(|source| polynomial(source, variables))
        .collect()
}

fn radical(sources: &[&str], variables: &[Symbol]) -> VerifiedZeroDimensionalRadical {
    checked_zero_dimensional_radical(
        &polynomials(sources, variables),
        variables,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn assert_basis(result: &VerifiedZeroDimensionalRadical, expected: &[&str]) {
    let expected = polynomials(expected, result.generators());
    assert_eq!(result.basis().basis().len(), expected.len());
    for polynomial in expected {
        assert!(
            result.basis().basis().contains(&polynomial),
            "missing {polynomial:?}"
        );
    }
    verify_zero_dimensional_radical(
        result.inputs(),
        result.generators(),
        result.certificate(),
        &mut Unbounded,
    )
    .unwrap();
}

#[test]
fn coupled_nilpotents_have_the_hand_identified_radical() {
    let variables = variables();
    let result = radical(&["x**2", "x*y", "y**2"], &variables);
    assert_basis(&result, &["x", "y"]);
    assert!(result.changed_ideal());
    assert_eq!(
        result.inputs(),
        polynomials(&["x**2", "x*y", "y**2"], &variables)
    );
    assert!(
        result
            .certificate()
            .coordinates
            .iter()
            .all(|coordinate| coordinate.nilpotence_exponent == 2)
    );

    // q has five distinct real roots (sign changes at -2,-3/2,-1,0,1,2).
    // q^2=q*y=y^2=0 has exactly those points with y=0, despite a
    // 15-dimensional quotient with two independent nilpotent directions.
    let q = "x**5+x**4-4*x**3-3*x**2+3*x+1";
    let sources = [format!("({q})**2"), format!("({q})*y"), "y**2".into()];
    let result = radical(
        &sources.iter().map(String::as_str).collect::<Vec<_>>(),
        &variables,
    );
    assert_basis(&result, &[q, "y"]);
}

#[test]
fn translations_unions_rational_coefficients_and_caller_order_are_exact() {
    let variables = variables();
    assert_basis(
        &radical(&["(x-1)**2", "(x-1)*(y+2)", "(y+2)**2"], &variables),
        &["x-1", "y+2"],
    );
    assert_basis(
        &radical(&["(2*x-1)**2", "y-x"], &variables),
        &["x-1/2", "y-1/2"],
    );
    let result = radical(&["x**2*(x-1)**2", "y-x"], &variables);
    assert_basis(&result, &["x-y", "y**2-y"]);
    let reversed = vec![variables[1].clone(), variables[0].clone()];
    assert_basis(
        &radical(&["x**2*(x-1)**2", "y-x"], &reversed),
        &["y-x", "x**2-x"],
    );
}

#[test]
fn reduced_and_unit_ideals_are_supported_but_infinite_varieties_refuse() {
    let variables = variables();
    let result = radical(&["x**2-x", "y**2-2"], &variables);
    assert_basis(&result, &["x**2-x", "y**2-2"]);
    assert!(!result.changed_ideal());
    assert!(
        result
            .certificate()
            .coordinates
            .iter()
            .all(|coordinate| coordinate.nilpotence_exponent == 1)
    );
    let unit = radical(&["x", "x-1"], &variables);
    assert_basis(&unit, &["1"]);
    assert!(unit.certificate().coordinates.is_empty());
    for sources in [vec![], vec!["0"], vec!["x*y"], vec!["x"]] {
        assert!(
            checked_zero_dimensional_radical(
                &polynomials(&sources, &variables),
                &variables,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn original_request_coordinate_and_nilpotence_mutations_refuse() {
    let variables = variables();
    let result = radical(&["x**2", "x*y", "y**2"], &variables);
    let reject = |certificate| {
        assert!(
            verify_zero_dimensional_radical(
                result.inputs(),
                &variables,
                &certificate,
                &mut Unbounded
            )
            .is_err()
        );
    };
    let mut changed = result.certificate().clone();
    changed.coordinates[0].nilpotent_quotients =
        vec![polynomial("0", &variables); result.source_basis().basis().len()];
    reject(changed);
    for exponent in [0, 1, 65, u32::MAX] {
        let mut changed = result.certificate().clone();
        changed.coordinates[0].nilpotence_exponent = exponent;
        reject(changed);
    }
    let mut changed = result.certificate().clone();
    changed.coordinates.pop();
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.coordinates.swap(0, 1);
    reject(changed);
    let scaled = radical(&["2*x**2", "x*y", "y**2"], &variables);
    let mut changed = result.certificate().clone();
    changed.source_basis = Arc::clone(&scaled.certificate().source_basis);
    reject(changed);
    let mut typed = variables.clone();
    typed[0] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([71; 32]));
    assert!(
        verify_zero_dimensional_radical(
            result.inputs(),
            &typed,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

fn rebound_augmented_basis(
    result: &VerifiedZeroDimensionalRadical,
    certificate: &mut fsym_polys::ZeroDimensionalRadicalCertificate,
) {
    let mut inputs = result.source_basis().basis().to_vec();
    inputs.extend(
        certificate
            .coordinates
            .iter()
            .map(|coordinate| coordinate.annihilator.clone()),
    );
    certificate.radical_basis = Arc::new(
        checked_groebner_basis(&inputs, TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap(),
    );
}

#[test]
fn nilpotence_membership_does_not_replace_square_free_proof() {
    let variables = variables();
    let result = radical(&["x**2", "y**2"], &variables);
    let mut changed = result.certificate().clone();
    let coordinate = &mut changed.coordinates[0];
    coordinate.annihilator = polynomial("x**2", &variables);
    coordinate.nilpotence_exponent = 1;
    coordinate.nilpotent_quotients = result
        .source_basis()
        .basis()
        .iter()
        .map(|input| {
            polynomial(
                if *input == polynomial("x**2", &variables) {
                    "1"
                } else {
                    "0"
                },
                &variables,
            )
        })
        .collect();
    coordinate.square_free_u = polynomial("0", &variables);
    coordinate.square_free_v = polynomial("0", &variables);
    rebound_augmented_basis(&result, &mut changed);
    let error =
        verify_zero_dimensional_radical(result.inputs(), &variables, &changed, &mut Unbounded)
            .unwrap_err();
    assert!(error.to_string().contains("square-free Bezout"), "{error}");
}

#[test]
fn square_free_generators_cannot_delete_original_points_without_nilpotence() {
    let variables = variables();
    let result = radical(&["x**2", "y**2"], &variables);
    let mut changed = result.certificate().clone();
    let coordinate = &mut changed.coordinates[0];
    coordinate.annihilator = polynomial("x+1", &variables);
    coordinate.nilpotence_exponent = 1;
    coordinate.nilpotent_quotients =
        vec![polynomial("0", &variables); result.source_basis().basis().len()];
    coordinate.square_free_u = polynomial("0", &variables);
    coordinate.square_free_v = polynomial("1", &variables);
    rebound_augmented_basis(&result, &mut changed);
    assert!(changed.radical_basis.basis()[0].is_one());
    let error =
        verify_zero_dimensional_radical(result.inputs(), &variables, &changed, &mut Unbounded)
            .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("quotient/remainder coefficient identity"),
        "{error}"
    );
}

#[test]
fn coordinate_annihilators_do_not_replace_original_ideal_inclusion() {
    let variables = variables();
    let result = radical(&["x**2-x", "y-x"], &variables);
    let mut changed = result.certificate().clone();
    // These coordinate annihilators describe a four-point grid. The original
    // diagonal contains only (0,0),(1,1); all its equations must be retained.
    let mut wrong = vec![polynomial("0", &variables); result.source_basis().basis().len()];
    wrong.extend(
        changed
            .coordinates
            .iter()
            .map(|coordinate| coordinate.annihilator.clone()),
    );
    changed.radical_basis = Arc::new(
        checked_groebner_basis(&wrong, TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap(),
    );
    assert!(
        verify_zero_dimensional_radical(result.inputs(), &variables, &changed, &mut Unbounded)
            .is_err()
    );
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
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
fn cancellation_and_exhaustion_refuse_in_both_phases_and_before_publication() {
    let variables = variables();
    let inputs = polynomials(&["x**2", "x*y", "y**2"], &variables);
    let (mut generator, mut verifier) = (Meter::default(), Meter::default());
    checked_zero_dimensional_radical(&inputs, &variables, &mut generator, &mut verifier).unwrap();
    for (phase, count) in [(0, generator.checkpoints), (1, verifier.checkpoints)] {
        for cancel_at in [1, count / 2, count] {
            let mut meter = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let result = if phase == 0 {
                checked_zero_dimensional_radical(&inputs, &variables, &mut meter, &mut Unbounded)
            } else {
                checked_zero_dimensional_radical(&inputs, &variables, &mut Unbounded, &mut meter)
            };
            assert!(result.is_err(), "phase {phase}, checkpoint {cancel_at}");
        }
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        let result = if phase == 0 {
            checked_zero_dimensional_radical(&inputs, &variables, &mut budget, &mut Unbounded)
        } else {
            checked_zero_dimensional_radical(&inputs, &variables, &mut Unbounded, &mut budget)
        };
        assert!(result.is_err());
    }
}
