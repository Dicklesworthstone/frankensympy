//! Complete unions, source domains, typed identities and bounded failure.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::MultivariatePoly;
use fsym_solvers::polynomial_components::{
    PolynomialComponentsOutcome, VerifiedPolynomialComponents,
    metered_polynomial_components_system, polynomial_components_system,
};

fn variables(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn equations(sources: &[&str]) -> Vec<Expr> {
    sources
        .iter()
        .map(|source| parse(source).unwrap())
        .collect()
}

fn families(sources: &[&str], names: &[&str]) -> VerifiedPolynomialComponents {
    match polynomial_components_system(&equations(sources), &variables(names)).unwrap() {
        PolynomialComponentsOutcome::Families(answer) => answer,
        other => panic!("expected complete components, received {other:?}"),
    }
}

fn assert_rows(answer: &VerifiedPolynomialComponents, expected: &[&[&str]]) {
    let mut actual: Vec<_> = answer
        .families()
        .iter()
        .map(|family| {
            family
                .values()
                .iter()
                .map(|value| MultivariatePoly::from_expr(value, answer.variables()).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    for row in expected {
        let row: Vec<_> = row
            .iter()
            .map(|source| {
                MultivariatePoly::from_expr(&parse(source).unwrap(), answer.variables()).unwrap()
            })
            .collect();
        let position = actual
            .iter()
            .position(|candidate| candidate == &row)
            .unwrap_or_else(|| panic!("missing expected row {row:?} from {actual:?}"));
        actual.remove(position);
    }
    assert!(
        actual.is_empty(),
        "unexpected extra component rows: {actual:?}"
    );
}

#[test]
fn coordinate_axes_are_a_complete_union_not_a_single_arbitrary_branch() {
    let answer = families(&["x*y"], &["x", "y"]);
    assert_rows(&answer, &[&["0", "y"], &["x", "0"]]);
    assert_eq!(answer.equations().len(), 1);
    assert_eq!(answer.variables(), variables(&["x", "y"]));
    assert_eq!(answer.into_values().len(), 2);
}

#[test]
fn source_factors_handle_general_polynomial_graph_unions() {
    let answer = families(&["(x-y*y-y-1)*(x+y*y-y-1)"], &["x", "y"]);
    assert_rows(&answer, &[&["y*y+y+1", "y"], &["-y*y+y+1", "y"]]);
}

#[test]
fn expanded_difference_of_squares_is_recognized_without_source_factor_hints() {
    let answer = families(&["x*x-y**4"], &["x", "y"]);
    assert_rows(&answer, &[&["y*y", "y"], &["-y*y", "y"]]);
    let scaled = families(&["4*x*x-9*y**4"], &["x", "y"]);
    assert_rows(&scaled, &[&["3*y*y/2", "y"], &["-3*y*y/2", "y"]]);
}

#[test]
fn repeated_factors_do_not_duplicate_components_or_reset_their_domains() {
    let answer = families(&["(x-y*y)**7*(x+y*y)**3"], &["x", "y"]);
    assert_rows(&answer, &[&["y*y", "y"], &["-y*y", "y"]]);
    let thick_axis = families(&["x**256"], &["x", "y"]);
    assert_rows(&thick_axis, &[&["0", "y"]]);
}

#[test]
fn intersections_remove_only_components_with_checked_containment() {
    // The x=0 branch covers the y=0,x=0 sub-branch. The x-axis is not
    // contained in it, and must survive the same redundancy check.
    let answer = families(&["x*y", "x*z"], &["x", "y", "z"]);
    assert_rows(&answer, &[&["0", "y", "z"], &["x", "0", "0"]]);
    let reordered = families(&["x*z", "x*y", "2*x*y"], &["x", "y", "z"]);
    assert_rows(&reordered, &[&["0", "y", "z"], &["x", "0", "0"]]);
}

#[test]
fn polynomial_families_and_isolated_rational_points_can_share_one_union() {
    let answer = families(&["x*(x-1)", "x*y"], &["x", "y"]);
    assert_rows(&answer, &[&["0", "y"], &["1", "0"]]);
    assert_eq!(
        answer
            .families()
            .iter()
            .filter(|family| family.free_variables().is_empty())
            .count(),
        1
    );
}

#[test]
fn branch_constraints_are_all_retained_through_nonlinear_chains() {
    let answer = families(&["(x-y*y)*(x+y*y)", "z-x*y"], &["x", "y", "z"]);
    assert_rows(&answer, &[&["y*y", "y", "y**3"], &["-y*y", "y", "-y**3"]]);
}

#[test]
fn finite_rational_cartesian_components_are_complete() {
    let answer = families(&["x*x-1", "y*y-4"], &["x", "y"]);
    assert_rows(
        &answer,
        &[&["-1", "-2"], &["-1", "2"], &["1", "-2"], &["1", "2"]],
    );
    assert!(
        answer
            .families()
            .iter()
            .all(|family| family.free_variables().is_empty())
    );
}

#[test]
fn caller_coordinate_order_and_unconstrained_symbols_are_preserved() {
    let answer = families(&["x*y"], &["z", "y", "x"]);
    assert_rows(&answer, &[&["z", "y", "0"], &["z", "0", "x"]]);
}

#[test]
fn same_spelling_typed_coordinates_remain_distinct_in_every_component() {
    let plain = Symbol::new("coordinate");
    let assumed = Symbol::with_identity(
        "coordinate",
        SymbolIdentity::from_assumptions_digest([41; 32]),
    );
    let vars = [plain.clone(), assumed.clone()];
    let product = Expr::Mul(vec![Expr::Sym(plain.clone()), Expr::Sym(assumed.clone())]);
    let PolynomialComponentsOutcome::Families(answer) =
        polynomial_components_system(&[product], &vars).unwrap()
    else {
        panic!("typed coordinate axes must be recognized");
    };
    let rows = answer.into_values();
    assert!(rows.contains(&vec![Expr::from_i64(0), Expr::Sym(assumed)]));
    assert!(rows.contains(&vec![Expr::Sym(plain), Expr::from_i64(0)]));
}

#[test]
fn an_unsupported_component_prevents_publication_of_a_partial_union() {
    for sources in [
        vec!["x*(x*x+y*y-1)"],
        vec!["x*y-1"],
        vec!["x**5-x+1"],
        vec!["x*x+y*y-1"],
    ] {
        assert!(matches!(
            polynomial_components_system(&equations(&sources), &variables(&["x", "y"])).unwrap(),
            PolynomialComponentsOutcome::NotApplicable
        ));
    }
}

#[test]
fn zero_ideals_and_proved_empty_systems_keep_their_distinct_outcomes() {
    let vars = variables(&["x", "y"]);
    for sources in [vec![], vec!["0", "0"]] {
        let PolynomialComponentsOutcome::Families(answer) =
            polynomial_components_system(&equations(&sources), &vars).unwrap()
        else {
            panic!("zero ideal is the complete ambient family");
        };
        assert_rows(&answer, &[&["x", "y"]]);
    }
    for sources in [
        vec!["1"],
        vec!["x*y", "x*y-1"],
        vec!["(x-y)*(x-y-1)", "x-y-2"],
    ] {
        assert!(matches!(
            polynomial_components_system(&equations(&sources), &vars).unwrap(),
            PolynomialComponentsOutcome::Empty
        ));
    }
}

#[test]
fn original_undefined_sources_are_checked_before_every_branch_or_empty_decision() {
    let vars = variables(&["x", "y"]);
    for hidden in [
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(0))),
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
        Expr::Function("opaque".into(), vec![Expr::Sym(vars[0].clone())]),
    ] {
        for prefix in [Expr::from_i64(0), Expr::from_i64(1), parse("x*y").unwrap()] {
            let source = Expr::Mul(vec![Expr::from_i64(0), hidden.clone()]);
            assert!(polynomial_components_system(&[prefix, source], &vars).is_err());
        }
    }
}

#[test]
fn input_and_branch_explosion_bounds_refuse_without_partial_results() {
    let vars = variables(&["x", "y"]);
    assert!(polynomial_components_system(&[], &[]).is_err());
    assert!(polynomial_components_system(&[], &[vars[0].clone(), vars[0].clone()]).is_err());
    assert!(polynomial_components_system(&vec![Expr::from_i64(0); 65], &vars).is_err());
    assert!(polynomial_components_system(&[Expr::Integer(BigInt::one() << 4097)], &vars).is_err());
    assert!(polynomial_components_system(&[], &[Symbol::new("a".repeat(65_537))]).is_err());
    let vars = variables(&["a", "b", "c", "d", "e", "f"]);
    let inputs = equations(&["a*a-1", "b*b-1", "c*c-1", "d*d-1", "e*e-1", "f*f-1"]);
    assert!(polynomial_components_system(&inputs, &vars).is_err());
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    compute: u64,
    memory: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
            _ => {}
        }
        Ok(())
    }
    fn charge_batch(&mut self, values: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        for &(dimension, amount) in values {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn both_meters_can_cancel_mid_tree_or_at_the_final_publication() {
    let vars = variables(&["x", "y"]);
    for source in ["x*y", "1", "x*(x*x+y*y-1)"] {
        let inputs = equations(&[source]);
        let mut generator = Meter::default();
        let mut verifier = Meter::default();
        metered_polynomial_components_system(&inputs, &vars, &mut generator, &mut verifier)
            .unwrap();
        assert!(generator.compute > 0 && generator.memory > 0);
        assert!(verifier.compute > 0 && verifier.memory > 0);
        for point in [1, generator.checkpoints / 2, generator.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            assert!(
                metered_polynomial_components_system(&inputs, &vars, &mut stopped, &mut Unbounded)
                    .is_err()
            );
        }
        for point in [1, verifier.checkpoints / 2, verifier.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            assert!(
                metered_polynomial_components_system(&inputs, &vars, &mut Unbounded, &mut stopped)
                    .is_err()
            );
        }
    }
}

#[test]
fn exhausted_actual_budgets_never_publish_the_solved_subset() {
    let inputs = equations(&["x*y"]);
    let vars = variables(&["x", "y"]);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        metered_polynomial_components_system(&inputs, &vars, &mut tiny, &mut untouched).is_err()
    );
    assert_eq!(untouched.checkpoints, 0);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_polynomial_components_system(&inputs, &vars, &mut Unbounded, &mut tiny).is_err()
    );
}
