//! Source-domain conservation, independent construction mutations and metering.
//! Closed-form expectations deliberately do not rerun the proposal algorithm.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly, RationalLocalizationCertificate, RationalSystemAdmission,
    VerifiedRationalLocalization, bounded_polynomial_from_expr, metered_localize_rational_system,
    metered_rational_source, verify_rational_localization, verify_rational_source,
};
use std::sync::Arc;

fn vars(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn ratio(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, power(denominator, -1)])
}

fn expression(source: &str) -> Expr {
    parse(source).unwrap()
}

fn polynomial(source: &str, variables: &[Symbol]) -> MultivariatePoly {
    bounded_polynomial_from_expr(&expression(source), variables, &mut Unbounded).unwrap()
}

fn localize(input: &[Expr], variables: &[Symbol]) -> VerifiedRationalLocalization {
    match metered_localize_rational_system(input, variables, &mut Unbounded, &mut Unbounded)
        .unwrap()
    {
        RationalSystemAdmission::Localized(result) => *result,
        RationalSystemAdmission::Polynomial => panic!("expected a checked rational localization"),
    }
}

fn cancellation_fixture() -> (Vec<Expr>, Vec<Symbol>, VerifiedRationalLocalization) {
    let variables = vars(&["x", "y"]);
    let input = vec![
        ratio(expression("x**2-1"), expression("x-1")),
        expression("y-x"),
    ];
    let result = localize(&input, &variables);
    (input, variables, result)
}

#[test]
fn cancelled_factor_retains_its_original_pole_and_every_source_numerator() {
    let (input, variables, result) = cancellation_fixture();
    assert_eq!(result.original_width(), 2);
    assert_eq!(result.original_variables(), variables);
    assert_eq!(result.sources(), input);
    assert_eq!(result.variables()[..2], variables);
    assert_eq!(result.variables().len(), 3);
    assert_eq!(result.guards(), &[polynomial("x-1", &variables)]);
    let fractions = &result.certificate().fractions;
    assert_eq!(fractions[0].numerator, polynomial("x**2-1", &variables));
    assert_eq!(fractions[0].denominator, polynomial("x-1", &variables));
    assert_eq!(fractions[1].numerator, polynomial("y-x", &variables));
    assert_eq!(fractions[1].denominator, polynomial("1", &variables));
    let auxiliary = &result.certificate().auxiliary.name;
    let expected = [
        polynomial("x**2-1", result.variables()),
        polynomial("y-x", result.variables()),
        polynomial(&format!("{auxiliary}*(x-1)-1"), result.variables()),
    ];
    assert_eq!(result.polynomials(), expected);
    for (source, expected) in result.equations().iter().zip(&expected) {
        assert_eq!(
            bounded_polynomial_from_expr(source, result.variables(), &mut Unbounded).unwrap(),
            *expected
        );
    }
    verify_rational_localization(&input, &variables, result.certificate(), &mut Unbounded).unwrap();
}

#[test]
fn nested_reciprocal_retains_the_inner_pole_after_its_denominator_disappears() {
    let variables = vars(&["x"]);
    let source = power(power(expression("x"), -1), -1);
    let result = localize(&[source], &variables);
    let fraction = &result.certificate().fractions[0];
    assert_eq!(fraction.numerator, polynomial("x", &variables));
    assert_eq!(fraction.denominator, polynomial("1", &variables));
    assert_eq!(
        result.guards(),
        [polynomial("x", &variables), polynomial("1", &variables),]
    );
    let auxiliary = &result.certificate().auxiliary.name;
    assert_eq!(
        result.polynomials().last().unwrap(),
        &polynomial(&format!("{auxiliary}*x-1"), result.variables())
    );
}

#[test]
fn zero_products_and_zeroth_powers_keep_all_child_domain_obligations() {
    let variables = vars(&["x"]);
    for (source, numerator, denominator) in [
        (
            Expr::Mul(vec![Expr::from_i64(0), power(expression("x"), -1)]),
            "0",
            "x",
        ),
        (power(power(expression("x"), -1), 0), "1", "1"),
    ] {
        let result = localize(&[source], &variables);
        assert_eq!(
            result.certificate().fractions[0].numerator,
            polynomial(numerator, &variables)
        );
        assert_eq!(
            result.certificate().fractions[0].denominator,
            polynomial(denominator, &variables)
        );
        assert_eq!(result.guards(), &[polynomial("x", &variables)]);
    }
}

#[test]
fn sums_and_nested_negative_powers_match_independent_closed_form_fractions() {
    let variables = vars(&["x", "y"]);
    let sum = Expr::Add(vec![
        power(expression("x"), -1),
        power(expression("y"), -1),
        Expr::from_i64(-1),
    ]);
    let nested = power(ratio(expression("x"), expression("y-1")), -2);
    let result = localize(&[sum, nested], &variables);
    assert_eq!(
        result.certificate().fractions[0].numerator,
        polynomial("x+y-x*y", &variables)
    );
    assert_eq!(
        result.certificate().fractions[0].denominator,
        polynomial("x*y", &variables)
    );
    assert_eq!(
        result.certificate().fractions[1].numerator,
        polynomial("(y-1)**2", &variables)
    );
    assert_eq!(
        result.certificate().fractions[1].denominator,
        polynomial("x**2", &variables)
    );
    assert_eq!(
        result.guards(),
        [
            polynomial("x", &variables),
            polynomial("y", &variables),
            polynomial("y-1", &variables),
            polynomial("x", &variables),
        ]
    );
    let auxiliary = &result.certificate().auxiliary.name;
    assert_eq!(
        result.polynomials().last().unwrap(),
        &polynomial(&format!("{auxiliary}*x**2*y*(y-1)-1"), result.variables())
    );
}

#[test]
fn routing_keeps_polynomials_and_nonzero_constant_inverses_on_the_existing_path() {
    let variables = vars(&["x", "y"]);
    for source in [
        expression("x**2+y-1"),
        ratio(expression("x+y"), Expr::from_i64(2)),
        ratio(expression("x+y"), expression("x-x+2")),
        Expr::from_i64(0),
    ] {
        assert!(matches!(
            metered_localize_rational_system(&[source], &variables, &mut Unbounded, &mut Unbounded)
                .unwrap(),
            RationalSystemAdmission::Polynomial
        ));
    }
    // A routing hint never claims to have admitted an unsupported original.
    let opaque = Expr::Function("opaque".into(), vec![expression("x")]);
    assert!(matches!(
        metered_localize_rational_system(
            std::slice::from_ref(&opaque),
            &variables,
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap(),
        RationalSystemAdmission::Polynomial
    ));
    assert!(bounded_polynomial_from_expr(&opaque, &variables, &mut Unbounded).is_err());
}

#[test]
fn held_invalid_children_are_admitted_before_zero_or_inconsistent_equations() {
    let variables = vars(&["x", "y"]);
    for invalid in [
        power(Expr::from_i64(0), -1),
        power(Expr::from_i64(0), 0),
        power(expression("x-x"), -1),
        power(power(expression("x-x"), -1), 0),
        Expr::Function("opaque".into(), vec![expression("x")]),
    ] {
        let hidden = Expr::Mul(vec![Expr::from_i64(0), invalid, power(expression("x"), -1)]);
        for input in [vec![hidden.clone()], vec![Expr::from_i64(1), hidden]] {
            assert!(
                metered_localize_rational_system(
                    &input,
                    &variables,
                    &mut Unbounded,
                    &mut Unbounded
                )
                .is_err()
            );
        }
    }
    let undeclared = ratio(expression("x"), expression("z"));
    assert!(
        metered_localize_rational_system(
            &[Expr::from_i64(1), undeclared],
            &variables,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn typed_variables_and_auxiliary_freshness_survive_same_printed_names() {
    let plain = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([71; 32]));
    let reserved = Symbol::new("_fsym_rational_inverse_0");
    let variables = vec![plain.clone(), other.clone(), reserved.clone()];
    let input = vec![ratio(Expr::Sym(plain.clone()), Expr::Sym(other.clone()))];
    let result = localize(&input, &variables);
    assert_eq!(result.original_variables(), variables);
    assert_eq!(
        result.certificate().auxiliary.name,
        "_fsym_rational_inverse_1"
    );
    let expected = MultivariatePoly::var(variables.clone(), &other).unwrap();
    assert_eq!(result.guards(), &[expected]);
    assert!(
        metered_localize_rational_system(
            &input,
            &[plain.clone(), reserved],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        metered_localize_rational_system(
            &input,
            &[plain.clone(), plain],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

fn reject(input: &[Expr], variables: &[Symbol], certificate: &RationalLocalizationCertificate) {
    assert!(verify_rational_localization(input, variables, certificate, &mut Unbounded).is_err());
}

#[test]
fn independent_checker_rejects_fraction_guard_and_augmented_equation_mutations() {
    let (input, variables, result) = cancellation_fixture();
    let valid = result.certificate();

    let mut changed = valid.clone();
    changed.fractions[0].numerator = polynomial("x+1", &variables);
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.fractions[0].denominator = polynomial("1", &variables);
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.guards.clear();
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.guards.push(polynomial("y", &variables));
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.guards[0] = polynomial("x+1", &variables);
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.guards[0].terms.insert(
        vec![0, 0],
        BigRational::new((-1).into(), BigInt::from(10).pow(80)),
    );
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.equations[0] = polynomial("x+1", result.variables());
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed
        .equations
        .last_mut()
        .unwrap()
        .terms
        .insert(vec![0, 0, 0], BigRational::from(-2));
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.equations.pop();
    reject(&input, &variables, &changed);

    let mut changed = valid.clone();
    changed.fractions.pop();
    reject(&input, &variables, &changed);

    // Mutating every proposal consistently still cannot authorize a new source.
    let replacement = localize(
        &[
            ratio(expression("x**2-1"), expression("x+1")),
            expression("y-x"),
        ],
        &variables,
    );
    reject(&input, &variables, replacement.certificate());
}

#[test]
fn checker_binds_held_source_domains_even_when_the_reduced_expression_matches() {
    let variables = vars(&["x"]);
    let source = ratio(expression("x"), expression("x"));
    let result = localize(std::slice::from_ref(&source), &variables);
    reject(&[Expr::from_i64(1)], &variables, result.certificate());

    let nested = power(power(expression("x"), -1), -1);
    let nested_result = localize(&[nested], &variables);
    reject(&[expression("x")], &variables, nested_result.certificate());

    let changed = ratio(expression("x+1"), expression("x+1"));
    reject(&[changed], &variables, result.certificate());
}

#[test]
fn checker_rejects_capture_reordered_rings_and_malformed_polynomial_shapes() {
    let (input, variables, result) = cancellation_fixture();
    let mut changed = result.certificate().clone();
    changed.auxiliary = variables[0].clone();
    reject(&input, &variables, &changed);

    let mut changed = result.certificate().clone();
    changed.guards[0].generators.swap(0, 1);
    reject(&input, &variables, &changed);

    let mut changed = result.certificate().clone();
    changed.fractions[0]
        .denominator
        .terms
        .insert(vec![1], BigRational::from(1));
    reject(&input, &variables, &changed);

    let mut changed = result.certificate().clone();
    changed.equations[0]
        .terms
        .insert(vec![1, 0, 0], BigRational::from(0));
    reject(&input, &variables, &changed);

    let mut changed = result.certificate().clone();
    changed.guards[0]
        .terms
        .insert(vec![257, 0], BigRational::from(1));
    reject(&input, &variables, &changed);

    let mut changed = result.certificate().clone();
    changed.fractions[0].numerator.terms.insert(
        vec![0, 0],
        BigRational::from_integer(BigInt::from(1) << 4097),
    );
    reject(&input, &variables, &changed);

    let mut reversed = variables.clone();
    reversed.reverse();
    reject(&input, &reversed, result.certificate());
}

#[test]
fn dimensions_source_size_powers_and_guard_product_growth_refuse_explicitly() {
    let variables = vars(&["x"]);
    let inverse = power(expression("x"), -1);
    let many_variables: Vec<_> = (0..8).map(|i| Symbol::new(format!("v{i}"))).collect();
    assert!(
        metered_localize_rational_system(
            &[power(Expr::Sym(many_variables[0].clone()), -1)],
            &many_variables,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        metered_localize_rational_system(
            &vec![inverse.clone(); 64],
            &variables,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );

    let mut deeply_nested = inverse.clone();
    for _ in 0..65 {
        deeply_nested = Expr::Add(vec![deeply_nested]);
    }
    for source in [
        Expr::Add(vec![inverse.clone(); 4097]),
        deeply_nested,
        power(expression("x"), -257),
        ratio(Expr::Integer(BigInt::from(1) << 4097), expression("x")),
        Expr::Mul(vec![inverse; 129]),
        // Each guard fits individually, but the joint localization does not.
        Expr::Add(vec![
            power(expression("x**200+1"), -1),
            power(expression("x**100+2"), -1),
        ]),
    ] {
        assert!(
            metered_localize_rational_system(&[source], &variables, &mut Unbounded, &mut Unbounded)
                .is_err()
        );
    }

    let long = Symbol::new("x".repeat(65_537));
    assert!(
        metered_localize_rational_system(
            &[power(Expr::Sym(long.clone()), -1)],
            &[long],
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
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
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
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn separate_cumulative_lanes_cancel_before_during_and_at_receipt_publication() {
    let (input, variables, _) = cancellation_fixture();
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_localize_rational_system(&input, &variables, &mut generator, &mut verifier).unwrap();
    assert!(generator.compute > 0 && generator.memory > 0);
    assert!(verifier.compute > 0 && verifier.memory > 0);
    for point in [1, generator.checkpoints / 2, generator.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_localize_rational_system(&input, &variables, &mut stopped, &mut Unbounded)
                .is_err()
        );
    }
    for point in [1, verifier.checkpoints / 2, verifier.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_localize_rational_system(&input, &variables, &mut Unbounded, &mut stopped)
                .is_err()
        );
    }
}

#[test]
fn public_checker_and_real_resource_budgets_cannot_publish_unpaid_receipts() {
    let (input, variables, result) = cancellation_fixture();
    let mut measured = Meter::default();
    verify_rational_localization(&input, &variables, result.certificate(), &mut measured).unwrap();
    for point in [1, measured.checkpoints / 2, measured.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            verify_rational_localization(&input, &variables, result.certificate(), &mut stopped)
                .is_err()
        );
    }
    let mut generator = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0));
    assert!(matches!(
        metered_localize_rational_system(&input, &variables, &mut generator, &mut verifier)
            .unwrap(),
        RationalSystemAdmission::Localized(_)
    ));
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        metered_localize_rational_system(&input, &variables, &mut tiny, &mut untouched).is_err()
    );
    assert_eq!(untouched.checkpoints, 0);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_localize_rational_system(&input, &variables, &mut Unbounded, &mut tiny).is_err()
    );
}

#[test]
fn direct_source_admission_retains_exact_fractions_without_an_auxiliary_ring() {
    let variables = vars(&["x"]);
    let source = ratio(expression("x**2-1"), expression("x-1"));
    let checked =
        metered_rational_source(&source, &variables, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(checked.source(), &source);
    assert_eq!(checked.variables(), variables);
    assert_eq!(
        checked.fraction().numerator,
        polynomial("x**2-1", &variables)
    );
    assert_eq!(
        checked.fraction().denominator,
        polynomial("x-1", &variables)
    );
    assert_eq!(checked.guards(), [polynomial("x-1", &variables)]);
    verify_rational_source(&source, &variables, checked.certificate(), &mut Unbounded).unwrap();
}

#[test]
fn direct_source_checks_polynomial_constants_and_all_eight_original_variables() {
    let variables = vars(&["a", "b", "c", "d", "e", "f", "g", "h"]);
    for source in [expression("a+b*c+h**3"), expression("7/3"), expression("0")] {
        let checked =
            metered_rational_source(&source, &variables, &mut Unbounded, &mut Unbounded).unwrap();
        assert_eq!(
            checked.fraction().numerator,
            bounded_polynomial_from_expr(&source, &variables, &mut Unbounded).unwrap()
        );
        assert_eq!(checked.fraction().denominator, polynomial("1", &variables));
        assert!(checked.guards().is_empty());
        assert_eq!(checked.variables().len(), 8);
    }
}

#[test]
fn direct_source_preserves_nested_guards_even_when_the_value_has_no_denominator() {
    let variables = vars(&["x"]);
    let reciprocal = power(expression("x"), -1);
    for (source, expected, guard_count) in [
        (power(reciprocal.clone(), -1), "x", 2),
        (power(reciprocal.clone(), 0), "1", 1),
        (Expr::Mul(vec![Expr::from_i64(0), reciprocal]), "0", 1),
    ] {
        let checked =
            metered_rational_source(&source, &variables, &mut Unbounded, &mut Unbounded).unwrap();
        assert_eq!(
            checked.fraction().numerator,
            polynomial(expected, &variables)
        );
        assert_eq!(checked.guards()[0], polynomial("x", &variables));
        assert_eq!(checked.guards().len(), guard_count);
    }
}

#[test]
fn direct_source_verifier_rejects_changed_values_dropped_guards_and_rebinding() {
    let variables = vars(&["x"]);
    let source = power(power(expression("x"), -1), -1);
    let checked =
        metered_rational_source(&source, &variables, &mut Unbounded, &mut Unbounded).unwrap();
    let mut omitted = checked.certificate().clone();
    omitted.guards.remove(0);
    assert!(verify_rational_source(&source, &variables, &omitted, &mut Unbounded).is_err());
    let mut changed = checked.certificate().clone();
    changed.fraction.numerator = polynomial("x+1", &variables);
    assert!(verify_rational_source(&source, &variables, &changed, &mut Unbounded).is_err());
    let mut zero = checked.certificate().clone();
    zero.fraction.denominator = polynomial("0", &variables);
    assert!(verify_rational_source(&source, &variables, &zero, &mut Unbounded).is_err());
    assert!(
        verify_rational_source(
            &expression("x"),
            &variables,
            checked.certificate(),
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        verify_rational_source(
            &source,
            &vars(&["y"]),
            checked.certificate(),
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn direct_source_admits_every_original_before_zero_and_power_shortcuts() {
    let variables = vars(&["x"]);
    let unsupported = Expr::Function("sin".into(), vec![expression("x")]);
    for bad in [
        Expr::Mul(vec![Expr::from_i64(0), unsupported.clone()]),
        power(unsupported, 0),
        power(Expr::from_i64(0), 0),
        power(Expr::from_i64(0), -1),
        Expr::Mul(vec![Expr::from_i64(0), expression("y")]),
        power(expression("x"), 257),
    ] {
        assert!(metered_rational_source(&bad, &variables, &mut Unbounded, &mut Unbounded).is_err());
    }
}

#[test]
fn direct_source_keeps_both_resource_lanes_cumulative_until_publication() {
    let variables = vars(&["x"]);
    let source = ratio(expression("x**2-1"), expression("x-1"));
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    let checked =
        metered_rational_source(&source, &variables, &mut generation, &mut verification).unwrap();
    assert!(generation.compute > 0 && generation.memory > 0);
    assert!(verification.compute > 0 && verification.memory > 0);
    for point in [1, generation.checkpoints / 2, generation.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_rational_source(&source, &variables, &mut cancelled, &mut Unbounded).is_err()
        );
    }
    for point in [1, verification.checkpoints / 2, verification.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_rational_source(&source, &variables, &mut Unbounded, &mut cancelled).is_err()
        );
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(verify_rational_source(&source, &variables, checked.certificate(), &mut tiny).is_err());
}
