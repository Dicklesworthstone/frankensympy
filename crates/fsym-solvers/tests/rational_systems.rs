//! Exact finite rational systems assembled from checked source localization
//! and the established complete finite / real RUR solver engines.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    RationalSystemAdmission, VerifiedRationalLocalization, metered_localize_rational_system,
};
use fsym_solvers::SolverError;
use fsym_solvers::polysys::metered_solve_polynomial_system;
use fsym_solvers::real_rur::metered_solve_real_rur_system;
use std::collections::HashSet;
use std::sync::Arc;

fn variables(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn expression(source: &str) -> Expr {
    parse(source).unwrap()
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn ratio(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, power(denominator, -1)])
}

fn localize(
    input: &[Expr],
    variables: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> VerifiedRationalLocalization {
    match metered_localize_rational_system(input, variables, generator, verifier).unwrap() {
        RationalSystemAdmission::Localized(result) => *result,
        RationalSystemAdmission::Polynomial => panic!("expected original rational poles"),
    }
}

fn finite(
    input: &[Expr],
    variables: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Vec<Vec<Expr>>, SolverError> {
    let receipt = localize(input, variables, generator, verifier);
    let mut solutions = metered_solve_polynomial_system(
        receipt.equations(),
        receipt.variables(),
        generator,
        verifier,
    )?;
    for point in &mut solutions {
        assert_eq!(point.len(), receipt.variables().len());
        point.truncate(receipt.original_width());
    }
    Ok(solutions)
}

#[test]
fn cancelled_factors_remove_original_poles_from_the_complete_finite_set() {
    let input = [
        ratio(expression("x**2-1"), expression("x-1")),
        expression("y-x"),
    ];
    assert_eq!(
        finite(
            &input,
            &variables(&["x", "y"]),
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap(),
        [vec![Expr::from_i64(-1), Expr::from_i64(-1)]]
    );
}

#[test]
fn multiple_original_denominators_exclude_different_points_without_losing_valid_branches() {
    let input = [
        ratio(expression("(x-1)*(x+1)*(x-2)"), expression("x-1")),
        ratio(expression("y-x"), expression("x+1")),
    ];
    assert_eq!(
        finite(
            &input,
            &variables(&["x", "y"]),
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap(),
        [vec![Expr::from_i64(2), Expr::from_i64(2)]]
    );
}

#[test]
fn nested_reciprocal_and_held_zero_terms_preserve_hidden_domain_restrictions() {
    let x = expression("x");
    // 1/(1/x)=x where x!=0, hence its zero set is empty.
    let nested = power(power(x.clone(), -1), -1);
    assert!(
        finite(
            &[nested],
            &variables(&["x"]),
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap()
        .is_empty()
    );

    // x + 0*(1/y) = 0, y*(y-1)=0 admits only (0,1).
    let input = [
        Expr::Add(vec![
            x,
            Expr::Mul(vec![Expr::from_i64(0), power(expression("y"), -1)]),
        ]),
        expression("y*(y-1)"),
    ];
    assert_eq!(
        finite(
            &input,
            &variables(&["x", "y"]),
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap(),
        [vec![Expr::from_i64(0), Expr::from_i64(1)]]
    );

    // The zero-valued held expression (1/x)^0-1 still excludes x=0.
    let held_zero = Expr::Add(vec![
        power(power(expression("x"), -1), 0),
        Expr::from_i64(-1),
    ]);
    assert_eq!(
        finite(
            &[held_zero, expression("x*(x-1)")],
            &variables(&["x"]),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        [vec![Expr::from_i64(1)]]
    );
}

#[test]
fn rational_coordinate_sums_and_caller_variable_order_remain_exact() {
    let input = [
        Expr::Add(vec![
            power(expression("x"), -1),
            power(expression("y"), -1),
            Expr::from_i64(-3),
        ]),
        expression("y-1"),
    ];
    let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
    assert_eq!(
        finite(
            &input,
            &variables(&["x", "y"]),
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap(),
        [vec![half.clone(), Expr::from_i64(1)]]
    );
    assert_eq!(
        finite(
            &input,
            &variables(&["y", "x"]),
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap(),
        [vec![Expr::from_i64(1), half]]
    );
}

#[test]
fn radical_and_complex_finite_coordinates_survive_exact_localization() {
    let input = [
        ratio(expression("x**2-2"), expression("x-1")),
        expression("y-x"),
    ];
    let roots = finite(
        &input,
        &variables(&["x", "y"]),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(roots.len(), 2);
    assert_ne!(roots[0], roots[1]);
    for root in roots {
        assert_eq!(root[0], root[1]);
        let residual = root[0].clone() * root[0].clone() - Expr::from_i64(2);
        assert!(fsym_simplify::try_simplify(&residual).unwrap().is_zero());
    }

    let complex = [
        ratio(expression("x**2+1"), expression("x")),
        expression("y-x"),
    ];
    let actual = finite(
        &complex,
        &variables(&["x", "y"]),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    let i = Expr::Const(Constant::I);
    let expected = HashSet::from([vec![i.clone(), i.clone()], vec![-i.clone(), -i]]);
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.into_iter().collect::<HashSet<_>>(), expected);
}

#[test]
fn distinct_typed_symbols_with_the_same_name_remain_distinct_coordinates() {
    let plain = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([81; 32]));
    let input = [
        ratio(
            Expr::Sym(plain.clone()) - Expr::from_i64(1),
            Expr::Sym(other.clone()),
        ),
        Expr::Sym(other.clone()) - Expr::from_i64(2),
    ];
    assert_eq!(
        finite(&input, &[other, plain], &mut Unbounded, &mut Unbounded).unwrap(),
        [vec![Expr::from_i64(2), Expr::from_i64(1)]]
    );
}

#[test]
fn punctured_positive_dimensional_families_remain_explicitly_unsupported() {
    let vars = variables(&["x", "y"]);
    for source in [
        ratio(expression("x"), expression("y")),
        Expr::Mul(vec![Expr::from_i64(0), power(expression("x"), -1)]),
        Expr::Add(vec![
            ratio(expression("x"), expression("x")),
            Expr::from_i64(-1),
        ]),
    ] {
        let receipt = localize(&[source], &vars, &mut Unbounded, &mut Unbounded);
        assert_eq!(
            metered_solve_polynomial_system(
                receipt.equations(),
                receipt.variables(),
                &mut Unbounded,
                &mut Unbounded
            ),
            Err(SolverError::InfiniteSolutions)
        );
    }
}

#[test]
fn empty_rational_loci_are_exact_and_invalid_originals_cannot_hide_behind_inconsistency() {
    let vars = variables(&["x", "y"]);
    let pole_only = [ratio(expression("x"), expression("x")), expression("y")];
    assert!(
        finite(&pole_only, &vars, &mut Unbounded, &mut Unbounded)
            .unwrap()
            .is_empty()
    );

    let inconsistent = [
        ratio(expression("x-1"), expression("y")),
        expression("x-2"),
        expression("y-1"),
    ];
    assert!(
        finite(&inconsistent, &vars, &mut Unbounded, &mut Unbounded)
            .unwrap()
            .is_empty()
    );

    let held_zero = Expr::Add(vec![
        Expr::Sym(vars[0].clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(vars[0].clone())]),
    ]);
    let invalid = Expr::Mul(vec![Expr::from_i64(0), power(held_zero, -1)]);
    assert!(
        metered_localize_rational_system(
            &[Expr::from_i64(1), invalid],
            &vars,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn a_polynomial_routing_hint_still_requires_full_original_admission() {
    let vars = variables(&["x", "y"]);
    let invalid = Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]);
    let input = [Expr::from_i64(1), invalid.clone()];
    assert!(matches!(
        metered_localize_rational_system(&input, &vars, &mut Unbounded, &mut Unbounded).unwrap(),
        RationalSystemAdmission::Polynomial,
    ));
    assert!(fsym_polys::bounded_polynomial_from_expr(&invalid, &vars, &mut Unbounded).is_err());
    assert!(
        metered_solve_polynomial_system(&input, &vars, &mut Unbounded, &mut Unbounded).is_err()
    );
}

#[test]
fn checked_real_rur_keeps_all_five_quintic_solutions_and_the_auxiliary_last() {
    let input = [
        ratio(expression("x**5+x**4-4*x**3-3*x**2+3*x+1"), expression("x")),
        expression("y-x"),
    ];
    let vars = variables(&["x", "y"]);
    let mut generator = Unbounded;
    let mut verifier = Unbounded;
    let receipt = localize(&input, &vars, &mut generator, &mut verifier);
    assert!(
        metered_solve_polynomial_system(
            receipt.equations(),
            receipt.variables(),
            &mut generator,
            &mut verifier
        )
        .is_err()
    );
    let result = metered_solve_real_rur_system(
        receipt.equations(),
        receipt.variables(),
        &mut generator,
        &mut verifier,
    )
    .unwrap()
    .unwrap();
    // The original quintic has five distinct real roots and no zero root.
    // Localization is bijective, so all five must remain represented.
    assert!(result.all_complex_roots_real());
    assert_eq!(result.roots().intervals().len(), 5);
    assert_eq!(result.coordinates().len(), 3);
    assert_eq!(result.variables()[..2], vars);
    assert_eq!(result.coordinates()[0], result.coordinates()[1]);
    assert_eq!(result.parameter(), &receipt.certificate().auxiliary);
    let reciprocal_eliminant: Vec<_> = [1, 1, -4, -3, 3, 1]
        .into_iter()
        .map(BigRational::from)
        .collect();
    assert_eq!(result.eliminant(), reciprocal_eliminant);
}

#[test]
fn a_real_rur_subset_never_claims_a_complete_complex_quintic_locus() {
    let input = [
        ratio(expression("x**5-x+1"), expression("x")),
        expression("y-x"),
    ];
    let receipt = localize(
        &input,
        &variables(&["x", "y"]),
        &mut Unbounded,
        &mut Unbounded,
    );
    let result = metered_solve_real_rur_system(
        receipt.equations(),
        receipt.variables(),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
    .unwrap();
    assert!(!result.all_complex_roots_real());
    assert_eq!(result.roots().intervals().len(), 1);
}

#[derive(Default)]
struct Meter {
    compute: u64,
    memory: u64,
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
        Ok(())
    }
}

#[test]
fn original_admission_and_finite_solving_spend_the_same_cumulative_lane_budgets() {
    let input = [
        ratio(expression("x**2-1"), expression("x-1")),
        expression("y-x"),
    ];
    let vars = variables(&["x", "y"]);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    let receipt = localize(&input, &vars, &mut generation, &mut verification);
    let generation_before = generation.compute;
    let verification_before = verification.compute;
    metered_solve_polynomial_system(
        receipt.equations(),
        receipt.variables(),
        &mut generation,
        &mut verification,
    )
    .unwrap();
    assert!(generation.compute > generation_before);
    assert!(verification.compute > verification_before);
    assert!(generation.memory > 0 && verification.memory > 0);

    // Pay exactly for localization's compute budget. Solving cannot reset it.
    let mut limit = BudgetLimits::uniform(256 * 1024 * 1024, 0);
    limit.dimensions[Dimension::ComputeSteps.index()] = generation_before;
    let mut exhausted = Budget::new(limit);
    let receipt = localize(&input, &vars, &mut exhausted, &mut Unbounded);
    assert!(
        metered_solve_polynomial_system(
            receipt.equations(),
            receipt.variables(),
            &mut exhausted,
            &mut Unbounded,
        )
        .is_err()
    );
}
