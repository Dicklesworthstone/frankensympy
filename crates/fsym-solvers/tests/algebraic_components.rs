//! Complete finite algebraic loci crossed with original free coordinates.

use std::collections::HashSet;
use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::polynomial_components::{
    PolynomialComponentsOutcome, VerifiedPolynomialComponents, metered_rational_components_system,
    rational_components_system,
};

fn variables(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn sources(values: &[&str]) -> Vec<Expr> {
    values.iter().map(|source| parse(source).unwrap()).collect()
}

fn families(values: &[&str], names: &[&str]) -> VerifiedPolynomialComponents {
    let result = rational_components_system(&sources(values), &variables(names)).unwrap();
    match result {
        PolynomialComponentsOutcome::Families(result) => result,
        other => panic!("expected complete algebraic components, received {other:?}"),
    }
}

fn zero(value: Expr) {
    assert!(fsym_simplify::try_simplify(&value).unwrap().is_zero());
}

#[test]
fn both_quadratic_conjugates_keep_the_original_free_coordinate() {
    let result = families(&["x*x-2"], &["x", "y"]);
    assert_eq!(result.families().len(), 2);
    assert_ne!(result.families()[0].values(), result.families()[1].values());
    for family in result.families() {
        assert!(family.is_algebraic());
        assert!(!family.is_rational());
        assert!(family.basis().is_none());
        assert_eq!(family.free_variables(), &[Symbol::new("y")]);
        assert_eq!(family.values()[1], Expr::symbol("y"));
        assert!(family.values()[0].free_symbols().is_empty());
        zero(family.values()[0].clone() * family.values()[0].clone() - Expr::from_i64(2));
    }
}

#[test]
fn complex_roots_and_multiple_free_coordinates_follow_caller_order() {
    let result = families(&["x*x+1"], &["y", "x", "z"]);
    let imaginary = Expr::Const(Constant::I);
    let expected = HashSet::from([
        vec![Expr::symbol("y"), imaginary.clone(), Expr::symbol("z")],
        vec![Expr::symbol("y"), -imaginary, Expr::symbol("z")],
    ]);
    assert_eq!(
        result.into_values().into_iter().collect::<HashSet<_>>(),
        expected
    );
}

#[test]
fn coupled_active_subsystems_preserve_all_equations_and_free_positions() {
    let result = families(&["x*x-2", "z-x", "2*z-2*x", "0"], &["z", "y", "x", "w"]);
    assert_eq!(result.families().len(), 2);
    for family in result.families() {
        let row = family.values();
        assert_eq!(row[0], row[2]);
        assert_eq!(row[1], Expr::symbol("y"));
        assert_eq!(row[3], Expr::symbol("w"));
        assert_eq!(family.free_variables(), variables(&["y", "w"]));
        zero(row[0].clone() * row[0].clone() - Expr::from_i64(2));
    }
}

#[test]
fn independent_active_roots_form_every_cartesian_combination() {
    let result = families(&["x*x-2", "z*z-3"], &["z", "y", "x"]);
    assert_eq!(result.families().len(), 4);
    let mut first = HashSet::new();
    let mut second = HashSet::new();
    let mut pairs = HashSet::new();
    for family in result.families() {
        let row = family.values();
        assert_eq!(row[1], Expr::symbol("y"));
        zero(row[0].clone() * row[0].clone() - Expr::from_i64(3));
        zero(row[2].clone() * row[2].clone() - Expr::from_i64(2));
        first.insert(row[0].clone());
        second.insert(row[2].clone());
        pairs.insert((row[0].clone(), row[2].clone()));
    }
    assert_eq!((first.len(), second.len(), pairs.len()), (2, 2, 4));
}

#[test]
fn split_unions_keep_polynomial_fibers_and_every_algebraic_conjugate() {
    for source in ["y*(x*x-2)", "y*y*(x*x-2)**3"] {
        let result = families(&[source], &["x", "y"]);
        assert_eq!(result.families().len(), 3);
        assert!(
            result
                .families()
                .iter()
                .any(|family| family.values() == [Expr::symbol("x"), Expr::from_i64(0)])
        );
        assert_eq!(
            result
                .families()
                .iter()
                .filter(|family| family.is_algebraic())
                .count(),
            2
        );
        for family in result
            .families()
            .iter()
            .filter(|family| family.is_algebraic())
        {
            assert_eq!(family.values()[1], Expr::symbol("y"));
            zero(family.values()[0].clone() * family.values()[0].clone() - Expr::from_i64(2));
        }
    }
}

#[test]
fn rational_curves_and_algebraic_lines_keep_their_own_parameter_domains() {
    let result = families(&["(x*x-2)*(x*y-1)"], &["x", "y"]);
    assert_eq!(result.families().len(), 3);
    assert_eq!(
        result
            .families()
            .iter()
            .filter(|family| family.is_algebraic())
            .count(),
        2
    );
    let rational = result
        .families()
        .iter()
        .find(|family| family.is_rational())
        .unwrap();
    assert_eq!(rational.nonzero_conditions(), &[Expr::symbol("y")]);
    for family in result
        .families()
        .iter()
        .filter(|family| family.is_algebraic())
    {
        assert!(family.nonzero_conditions().is_empty());
        assert_eq!(family.values()[1], Expr::symbol("y"));
        zero(family.values()[0].clone() * family.values()[0].clone() - Expr::from_i64(2));
    }
}

#[test]
fn isolated_algebraic_points_are_not_erased_by_an_unrelated_free_family() {
    // (0,y) and (+-sqrt(2),0) are all required; checking the generic x=0
    // component alone would omit two valid isolated points.
    let result = families(&["x*(x*x-2)", "x*y"], &["x", "y"]);
    assert_eq!(result.families().len(), 3);
    assert!(
        result
            .families()
            .iter()
            .any(|family| family.values() == [Expr::from_i64(0), Expr::symbol("y")])
    );
    for family in result
        .families()
        .iter()
        .filter(|family| family.is_algebraic())
    {
        assert!(family.free_variables().is_empty());
        assert_eq!(family.values()[1], Expr::from_i64(0));
        zero(family.values()[0].clone() * family.values()[0].clone() - Expr::from_i64(2));
    }
}

#[test]
fn same_printing_active_and_free_symbols_keep_distinct_typed_identities() {
    let plain = Symbol::new("coordinate");
    let assumed = Symbol::with_identity(
        "coordinate",
        SymbolIdentity::from_assumptions_digest([89; 32]),
    );
    let active = Expr::Sym(assumed.clone());
    let source = active.clone() * active - Expr::from_i64(2);
    let PolynomialComponentsOutcome::Families(result) =
        rational_components_system(&[source], &[plain.clone(), assumed]).unwrap()
    else {
        panic!("the two typed coordinate roles must remain distinct")
    };
    assert_eq!(result.families().len(), 2);
    for family in result.families() {
        assert_eq!(family.free_variables(), std::slice::from_ref(&plain));
        assert_eq!(family.values()[0], Expr::Sym(plain.clone()));
        assert!(family.values()[1].free_symbols().is_empty());
        zero(family.values()[1].clone() * family.values()[1].clone() - Expr::from_i64(2));
    }
}

#[test]
fn unsupported_or_still_constrained_components_never_publish_a_partial_union() {
    for source in ["y*(x**5-x+1)", "(x*x-2)*(x*x+y*y-1)", "x*x+y*y-1"] {
        assert!(matches!(
            rational_components_system(&sources(&[source]), &variables(&["x", "y"])).unwrap(),
            PolynomialComponentsOutcome::NotApplicable,
        ));
    }
}

#[test]
fn exact_inconsistency_and_original_undefined_nodes_cannot_be_hidden() {
    let vars = variables(&["x", "y"]);
    let equation = parse("x*x-2").unwrap();
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(80)));
    assert!(matches!(
        rational_components_system(&[equation.clone(), equation.clone() - epsilon], &vars).unwrap(),
        PolynomialComponentsOutcome::Empty,
    ));
    for exponent in [-1, 0] {
        let bad = Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(
                Arc::new(Expr::from_i64(0)),
                Arc::new(Expr::from_i64(exponent)),
            ),
        ]);
        for prefix in [equation.clone(), Expr::from_i64(1)] {
            assert!(rational_components_system(&[prefix, bad.clone()], &vars).is_err());
        }
    }
}

#[test]
fn finite_root_expansion_is_included_in_the_component_output_limit() {
    let vars = variables(&["a", "b", "c", "d", "e", "f", "free"]);
    let input = sources(&["a*a-2", "b*b-2", "c*c-2", "d*d-2", "e*e-2", "f*f-2"]);
    // 64 distinct algebraic rows, each crossed with the free coordinate,
    // exceed the existing 32-component contract. No subset may escape.
    assert!(rational_components_system(&input, &vars).is_err());
}

#[derive(Default)]
struct Meter {
    compute: u64,
    checkpoints: usize,
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
fn root_search_projection_and_union_share_budgets_and_terminal_cancellation() {
    let input = sources(&["y*(x*x-2)"]);
    let vars = variables(&["x", "y"]);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    metered_rational_components_system(&input, &vars, &mut generation, &mut verification).unwrap();
    assert!(generation.compute > 0 && verification.compute > 0);
    for (generator_lane, measured) in [(true, &generation), (false, &verification)] {
        for point in [1, measured.checkpoints / 2, measured.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            let result = if generator_lane {
                metered_rational_components_system(&input, &vars, &mut stopped, &mut Unbounded)
            } else {
                metered_rational_components_system(&input, &vars, &mut Unbounded, &mut stopped)
            };
            assert!(result.is_err());
        }
        let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = measured.compute - 1;
        let mut exhausted = Budget::new(limits);
        let result = if generator_lane {
            metered_rational_components_system(&input, &vars, &mut exhausted, &mut Unbounded)
        } else {
            metered_rational_components_system(&input, &vars, &mut Unbounded, &mut exhausted)
        };
        assert!(result.is_err());
    }
}
