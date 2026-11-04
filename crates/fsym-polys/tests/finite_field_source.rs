//! Exact source-content binding around complete modular component factorizations.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    PrimeFieldPolynomial, PrimeFieldSourceFactorizationCertificate, PrimeFieldSourceRequest,
    VerifiedPrimeFieldSourceFactorization, admit_prime_field_source,
    checked_prime_field_factorization, checked_prime_field_source_factorization,
    verify_prime_field_source_factorization,
};

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn solve(
    source: &str,
    prime: u64,
    content: BigRational,
    components: &[(&str, u32)],
) -> VerifiedPrimeFieldSourceFactorization {
    let source = parse(source).unwrap();
    let variable = Symbol::new("x");
    let expressions: Vec<_> = components
        .iter()
        .map(|&(value, _)| parse(value).unwrap())
        .collect();
    let parts: Vec<_> = expressions
        .iter()
        .zip(components)
        .map(|(expression, &(_, exponent))| (expression, exponent))
        .collect();
    let request = PrimeFieldSourceRequest {
        source: &source,
        variable: &variable,
        prime,
        content: &content,
        components: &parts,
    };
    checked_prime_field_source_factorization(&request, &mut Unbounded, &mut Unbounded).unwrap()
}

#[test]
fn rational_prefactors_and_symmetric_modular_units_remain_distinct() {
    let half = solve("x/2+1", 3, q(1, 2), &[("x+2", 1)]);
    assert_eq!(half.unit(), &q(1, 2));
    assert_eq!(half.factors()[0].coefficients, [2, 1]);
    // This is an exterior rational coefficient, not an embedding of QQ into F_3.
    let third = solve("x/3+1", 3, q(1, 3), &[("x+3", 1)]);
    assert_eq!(third.unit(), &q(1, 3));
    assert_eq!(third.factors()[0].coefficients, [0, 1]);
    let mixed = solve("x**2/2+x/2+1/3", 3, q(1, 6), &[("3*x**2+3*x+2", 1)]);
    assert_eq!(mixed.unit(), &q(-1, 6));
    assert!(mixed.factors().is_empty());
    let exterior = solve("2*x**3+2*x", 3, q(2, 1), &[("x", 1), ("x**2+1", 1)]);
    assert_eq!(exterior.unit(), &q(2, 1));
    let ordinary = solve("2*x+2", 3, q(1, 1), &[("2*x+2", 1)]);
    assert_eq!(ordinary.unit(), &q(-1, 1));
    let repeated = solve("4*x**2+8*x+4", 5, q(1, 1), &[("4*x**2+8*x+4", 1)]);
    assert_eq!(repeated.unit(), &q(-1, 1));
    assert_eq!(repeated.factors()[0].coefficients, [1, 1]);
    assert_eq!(repeated.factors()[0].multiplicity, 2);
    assert_eq!(solve("7", 5, q(7, 1), &[]).unit(), &q(7, 1));
}

#[test]
fn inseparable_multiplicities_merge_across_all_original_components() {
    let result = solve(
        "(x**2+1)**3*(x+1)**2",
        2,
        q(1, 1),
        &[("x**2+1", 3), ("x+1", 2)],
    );
    assert_eq!(result.factors().len(), 1);
    assert_eq!(result.factors()[0].coefficients, [1, 1]);
    assert_eq!(result.factors()[0].multiplicity, 8);
    let vanished = solve("3*x+3", 3, q(1, 1), &[("3*x+3", 1)]);
    assert_eq!(vanished.unit(), &q(0, 1));
    assert!(vanished.factors().is_empty());
    assert_eq!(solve("0", 3, q(0, 1), &[]).unit(), &q(0, 1));
    let degree_drop = solve("3*x**2+x+1", 3, q(1, 1), &[("3*x**2+x+1", 1)]);
    assert_eq!(degree_drop.factors()[0].coefficients, [1, 1]);
}

#[test]
fn zero_modular_components_require_canonical_zero_without_discarding_receipts() {
    let source = parse("(3*x+3)*(x**2+1)").unwrap();
    let zero_component = parse("3*x+3").unwrap();
    let other_component = parse("x**2+1").unwrap();
    let variable = Symbol::new("x");
    let content = q(1, 1);
    let components = [(&zero_component, 1), (&other_component, 1)];
    let request = PrimeFieldSourceRequest {
        source: &source,
        variable: &variable,
        prime: 3,
        content: &content,
        components: &components,
    };
    let accepted =
        checked_prime_field_source_factorization(&request, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(accepted.unit(), &q(0, 1));
    assert!(accepted.factors().is_empty());
    assert_eq!(accepted.certificate().components.len(), 2);
    let mut noncanonical = accepted.certificate().clone();
    noncanonical.factors = noncanonical.components[1].factors().to_vec();
    assert!(
        verify_prime_field_source_factorization(&request, &noncanonical, &mut Unbounded).is_err()
    );
    noncanonical.factors.clear();
    noncanonical.components.pop();
    assert!(
        verify_prime_field_source_factorization(&request, &noncanonical, &mut Unbounded).is_err()
    );
}

#[test]
fn content_source_component_and_output_mutations_cannot_rebind_a_receipt() {
    let source = parse("(x**2+1)**3*(x+1)**2/2").unwrap();
    let a = parse("x**2+1").unwrap();
    let b = parse("x+1").unwrap();
    let parts = [(&a, 3), (&b, 2)];
    let variable = Symbol::new("x");
    let content = q(1, 2);
    let request = PrimeFieldSourceRequest {
        source: &source,
        variable: &variable,
        prime: 2,
        content: &content,
        components: &parts,
    };
    let result =
        checked_prime_field_source_factorization(&request, &mut Unbounded, &mut Unbounded).unwrap();
    let reject = |candidate| {
        assert!(
            verify_prime_field_source_factorization(&request, &candidate, &mut Unbounded).is_err()
        )
    };
    let mut changed = result.certificate().clone();
    changed.unit = q(1, 1);
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.factors[0].multiplicity = 7;
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.factors.clear();
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.components.swap(0, 1);
    reject(changed);
    let other = PrimeFieldPolynomial::new(variable.clone(), 3, vec![1, 0, 1]).unwrap();
    let accepted =
        checked_prime_field_factorization(&other, &mut Unbounded, &mut Unbounded).unwrap();
    let mut changed = result.certificate().clone();
    changed.components[0] = Arc::new(accepted);
    reject(changed);
    let wrong_content = q(1, 3);
    let changed_request = PrimeFieldSourceRequest {
        content: &wrong_content,
        ..request
    };
    assert!(
        verify_prime_field_source_factorization(
            &changed_request,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let wrong_source = parse("x").unwrap();
    let changed_request = PrimeFieldSourceRequest {
        source: &wrong_source,
        ..request
    };
    assert!(
        verify_prime_field_source_factorization(
            &changed_request,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let changed_request = PrimeFieldSourceRequest {
        components: &parts[..1],
        ..request
    };
    assert!(
        verify_prime_field_source_factorization(
            &changed_request,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn coordinated_content_and_component_mutations_cannot_change_characteristic_meaning() {
    let variable = Symbol::new("x");
    // Each proposed content * component is exactly the original QQ polynomial.
    // The first two invent a characteristic denominator, the third extracts a
    // characteristic numerator, and the last repeats denominator factors which
    // the original LCD contains only once (LCD 6, proposed denominator 12).
    for (source, content, component, residues) in [
        ("x", q(1, 3), "3*x", vec![]),
        ("x/2", q(1, 6), "3*x", vec![]),
        ("3*x+3", q(3, 1), "x+1", vec![1, 1]),
        ("x**2/2+x/2+1/3", q(1, 12), "6*x**2+6*x+4", vec![1]),
    ] {
        let source = parse(source).unwrap();
        let component = parse(component).unwrap();
        let parts = [(&component, 1)];
        let request = PrimeFieldSourceRequest {
            source: &source,
            variable: &variable,
            prime: 3,
            content: &content,
            components: &parts,
        };
        // Supply a genuinely accepted factorization for the proposed modular
        // component: a valid component receipt alone cannot bless this rebind.
        let modular = PrimeFieldPolynomial::new(variable.clone(), 3, residues).unwrap();
        let accepted =
            checked_prime_field_factorization(&modular, &mut Unbounded, &mut Unbounded).unwrap();
        let certificate = PrimeFieldSourceFactorizationCertificate {
            unit: &content * &q(accepted.unit() as i64, 1),
            factors: accepted.factors().to_vec(),
            components: vec![Arc::new(accepted)],
        };
        let direct =
            verify_prime_field_source_factorization(&request, &certificate, &mut Unbounded)
                .unwrap_err();
        assert!(direct.to_string().contains("exterior"), "{direct}");
        let generated =
            checked_prime_field_source_factorization(&request, &mut Unbounded, &mut Unbounded)
                .unwrap_err();
        assert!(generated.to_string().contains("exterior"), "{generated}");
    }
}

#[test]
fn characteristic_divisible_literal_exterior_content_explicitly_refuses() {
    let variable = Symbol::new("x");
    for (source, content) in [
        ("3", q(3, 1)),
        ("-3", q(-3, 1)),
        ("3/2", q(3, 2)),
        ("-3/2", q(-3, 2)),
    ] {
        let source = parse(source).unwrap();
        let request = PrimeFieldSourceRequest {
            source: &source,
            variable: &variable,
            prime: 3,
            content: &content,
            components: &[],
        };
        let certificate = PrimeFieldSourceFactorizationCertificate {
            unit: content.clone(),
            factors: vec![],
            components: vec![],
        };
        assert!(
            verify_prime_field_source_factorization(&request, &certificate, &mut Unbounded)
                .is_err()
        );
        assert!(
            checked_prime_field_source_factorization(&request, &mut Unbounded, &mut Unbounded)
                .is_err()
        );
    }
}

#[test]
fn held_undefined_children_and_foreign_typed_symbols_refuse_before_zero_shortcuts() {
    let variable = Symbol::new("x");
    for exponent in [-1, 0] {
        let invalid = Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(
                Arc::new(Expr::from_i64(0)),
                Arc::new(Expr::from_i64(exponent)),
            ),
        ]);
        assert!(admit_prime_field_source(&invalid, &variable, 2, &mut Unbounded).is_err());
        let content = q(0, 1);
        let request = PrimeFieldSourceRequest {
            source: &invalid,
            variable: &variable,
            prime: 2,
            content: &content,
            components: &[],
        };
        assert!(
            checked_prime_field_source_factorization(&request, &mut Unbounded, &mut Unbounded)
                .is_err()
        );
    }
    let foreign = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([24; 32]));
    assert!(admit_prime_field_source(&Expr::Sym(foreign), &variable, 3, &mut Unbounded).is_err());
    let ordinary = parse("x").unwrap();
    let bad = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("opaque".into(), vec![ordinary.clone()]),
    ]);
    let components = [(&bad, 1)];
    let content = q(1, 1);
    let request = PrimeFieldSourceRequest {
        source: &ordinary,
        variable: &variable,
        prime: 3,
        content: &content,
        components: &components,
    };
    assert!(
        checked_prime_field_source_factorization(&request, &mut Unbounded, &mut Unbounded).is_err()
    );
    assert!(
        admit_prime_field_source(&parse("x**65").unwrap(), &variable, 2, &mut Unbounded).is_err()
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
fn source_composition_has_cumulative_cancellation_and_resource_refusal() {
    let source = parse("(x**2+1)**3*(x+1)**2").unwrap();
    let a = parse("x**2+1").unwrap();
    let b = parse("x+1").unwrap();
    let parts = [(&a, 3), (&b, 2)];
    let variable = Symbol::new("x");
    let content = q(1, 1);
    let request = PrimeFieldSourceRequest {
        source: &source,
        variable: &variable,
        prime: 2,
        content: &content,
        components: &parts,
    };
    let (mut generator, mut verifier) = (Meter::default(), Meter::default());
    checked_prime_field_source_factorization(&request, &mut generator, &mut verifier).unwrap();
    for (phase, count) in [(0, generator.checkpoints), (1, verifier.checkpoints)] {
        for cancel_at in [1, count / 2, count] {
            let mut meter = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let result = if phase == 0 {
                checked_prime_field_source_factorization(&request, &mut meter, &mut Unbounded)
            } else {
                checked_prime_field_source_factorization(&request, &mut Unbounded, &mut meter)
            };
            assert!(result.is_err(), "phase {phase}, checkpoint {cancel_at}");
        }
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        let result = if phase == 0 {
            checked_prime_field_source_factorization(&request, &mut budget, &mut Unbounded)
        } else {
            checked_prime_field_source_factorization(&request, &mut Unbounded, &mut budget)
        };
        assert!(result.is_err());
    }
}
