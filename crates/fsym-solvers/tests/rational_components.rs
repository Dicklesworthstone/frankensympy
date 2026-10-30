//! Complete mixed component unions preserve every branch and its own poles.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::MultivariatePoly;
use fsym_solvers::polynomial_components::{
    PolynomialComponentsOutcome, VerifiedPolynomialComponents, metered_rational_components_system,
    polynomial_components_system, rational_components_system,
};

fn variables(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn sources(expressions: &[&str]) -> Vec<Expr> {
    expressions
        .iter()
        .map(|source| parse(source).unwrap())
        .collect()
}

fn families(expressions: &[&str], names: &[&str]) -> VerifiedPolynomialComponents {
    match rational_components_system(&sources(expressions), &variables(names)).unwrap() {
        PolynomialComponentsOutcome::Families(result) => result,
        other => panic!("expected a complete union, received {other:?}"),
    }
}

type Parts = (
    Vec<MultivariatePoly>,
    Vec<MultivariatePoly>,
    Vec<MultivariatePoly>,
);

fn polynomial_parts(parts: (Vec<Expr>, Vec<Expr>, Vec<Expr>), variables: &[Symbol]) -> Parts {
    let convert = |values: Vec<Expr>| {
        values
            .iter()
            .map(|value| MultivariatePoly::from_expr(value, variables).unwrap())
            .collect()
    };
    (convert(parts.0), convert(parts.1), convert(parts.2))
}

fn expected_parts(
    numerators: &[&str],
    denominators: &[&str],
    guards: &[&str],
    variables: &[Symbol],
) -> Parts {
    polynomial_parts(
        (sources(numerators), sources(denominators), sources(guards)),
        variables,
    )
}

fn assert_parts(result: VerifiedPolynomialComponents, expected: &[(&[&str], &[&str], &[&str])]) {
    let variables = result.variables().to_vec();
    let mut parts: Vec<_> = result
        .into_parts()
        .into_iter()
        .map(|row| polynomial_parts(row, &variables))
        .collect();
    for (numerators, denominators, guards) in expected {
        let wanted = expected_parts(numerators, denominators, guards, &variables);
        let index = parts
            .iter()
            .position(|row| row == &wanted)
            .unwrap_or_else(|| panic!("missing exact row {wanted:?} from {parts:?}"));
        parts.remove(index);
    }
    assert!(parts.is_empty(), "unexpected components: {parts:?}");
}

#[test]
fn coordinate_plane_and_hyperbola_both_survive_the_complete_union() {
    let result = families(&["x*(x*y-1)"], &["x", "y"]);
    assert_eq!(result.families().len(), 2);
    assert_eq!(
        result
            .families()
            .iter()
            .filter(|family| family.is_rational())
            .count(),
        1
    );
    assert_eq!(result.equations().len(), 1);
    assert_parts(
        result,
        &[
            (&["0", "y"], &["1", "1"], &[]),
            (&["1", "y"], &["y", "1"], &["y"]),
        ],
    );
}

#[test]
fn expanded_and_factored_hyperbola_unions_have_both_components() {
    for source in ["(x*y-1)*(x*y+1)", "x*x*y*y-1"] {
        assert_parts(
            families(&[source], &["x", "y"]),
            &[
                (&["1", "y"], &["y", "1"], &["y"]),
                (&["-1", "y"], &["y", "1"], &["y"]),
            ],
        );
    }
}

#[test]
fn polynomial_and_rational_curves_keep_distinct_parameter_domains() {
    assert_parts(
        families(&["(x-y*y)*(x*y-1)"], &["x", "y"]),
        &[
            (&["y*y", "y"], &["1", "1"], &[]),
            (&["1", "y"], &["y", "1"], &["y"]),
        ],
    );
}

#[test]
fn repeated_factors_keep_one_copy_of_each_complete_component() {
    assert_parts(
        families(&["x*x*(x*y-1)**3"], &["x", "y"]),
        &[
            (&["0", "y"], &["1", "1"], &[]),
            (&["1", "y"], &["y", "1"], &["y"]),
        ],
    );
}

#[test]
fn complex_parameter_poles_belong_only_to_the_rational_component() {
    assert_parts(
        families(&["x*(x*(y*y+1)-1)"], &["x", "y"]),
        &[
            (&["0", "y"], &["1", "1"], &[]),
            (&["1", "y"], &["y*y+1", "1"], &["y*y+1"]),
        ],
    );
}

#[test]
fn two_rational_components_do_not_share_each_others_excluded_parameters() {
    assert_parts(
        families(&["(x*y-1)*(x*(y-1)-1)"], &["x", "y"]),
        &[
            (&["1", "y"], &["y", "1"], &["y"]),
            (&["1", "y"], &["y-1", "1"], &["y-1"]),
        ],
    );
}

#[test]
fn nonlinear_chains_retain_every_original_coordinate_and_guard() {
    assert_parts(
        families(&["x*(x*y-1)", "z-x-y"], &["x", "z", "y"]),
        &[
            (&["0", "y", "y"], &["1", "1", "1"], &[]),
            (&["1", "y*y+1", "y"], &["y", "y", "1"], &["y"]),
        ],
    );
}

#[test]
fn checked_containment_can_remove_a_point_covered_by_a_rational_curve() {
    for source in [
        ["(x*y-1)*(x-1)", "(x*y-1)*(y-1)"],
        ["(x*y-1)*(y-1)", "(x*y-1)*(x-1)"],
    ] {
        assert_parts(
            families(&source, &["x", "y"]),
            &[(&["1", "y"], &["y", "1"], &["y"])],
        );
    }
}

#[test]
fn independent_component_choices_form_the_complete_cartesian_union() {
    assert_parts(
        families(&["x*(x*y-1)", "z*(z*w-1)"], &["x", "y", "z", "w"]),
        &[
            (&["0", "y", "0", "w"], &["1", "1", "1", "1"], &[]),
            (&["1", "y", "0", "w"], &["y", "1", "1", "1"], &["y"]),
            (&["0", "y", "1", "w"], &["1", "1", "w", "1"], &["w"]),
            (&["1", "y", "1", "w"], &["y", "1", "w", "1"], &["y", "w"]),
        ],
    );
}

#[test]
fn checked_whole_node_can_cover_unsupported_intermediate_factor_leaves() {
    // The original ideal is <xy-1>. A source split also creates the
    // constrained z^2+1 leaf, but that leaf is already covered by this whole
    // rational graph. No unsupported subset is published on its own.
    assert_parts(
        families(&["(x*y-1)*(z*z+1)", "(x*y-1)*(z*z+2)"], &["x", "y", "z"]),
        &[(&["1", "y", "z"], &["y", "1", "1"], &["y"])],
    );
}

#[test]
fn original_caller_order_and_unconstrained_coordinates_are_retained() {
    assert_parts(
        families(&["x*(x*y-1)"], &["z", "y", "x"]),
        &[
            (&["z", "y", "0"], &["1", "1", "1"], &[]),
            (&["z", "1", "x"], &["1", "x", "1"], &["x"]),
        ],
    );
}

#[test]
fn equally_printed_typed_symbols_do_not_alias_in_branch_domains() {
    let plain = Symbol::new("coordinate");
    let assumed = Symbol::with_identity(
        "coordinate",
        SymbolIdentity::from_assumptions_digest([37; 32]),
    );
    let product = Expr::Mul(vec![Expr::Sym(plain.clone()), Expr::Sym(assumed.clone())]);
    let source = Expr::Mul(vec![Expr::Sym(plain.clone()), product - Expr::from_i64(1)]);
    let PolynomialComponentsOutcome::Families(result) =
        rational_components_system(&[source], &[plain.clone(), assumed.clone()]).unwrap()
    else {
        panic!("both typed components must be recognized");
    };
    assert_eq!(result.families().len(), 2);
    let rational = result
        .families()
        .iter()
        .find(|family| family.is_rational())
        .unwrap();
    assert_eq!(rational.free_variables(), std::slice::from_ref(&assumed));
    assert_eq!(rational.nonzero_conditions(), &[Expr::Sym(assumed.clone())]);
    let rows = result.into_parts();
    assert!(rows.iter().any(|(_, denominators, conditions)| denominators
        == &[Expr::Sym(assumed.clone()), Expr::from_i64(1)]
        && conditions == &[Expr::Sym(assumed.clone())]));
    assert_ne!(plain, assumed);
}

#[test]
fn one_unsolved_component_still_blocks_every_partial_union() {
    for expression in ["x*(x*y-1)*(x*x+y*y-1)", "x*(x**5-x+1)", "x*x+y*y-1"] {
        assert!(matches!(
            rational_components_system(&sources(&[expression]), &variables(&["x", "y"])).unwrap(),
            PolynomialComponentsOutcome::NotApplicable
        ));
    }
    // The opt-in API does not silently widen older polynomial-only callers.
    assert!(matches!(
        polynomial_components_system(&sources(&["x*(x*y-1)"]), &variables(&["x", "y"])).unwrap(),
        PolynomialComponentsOutcome::NotApplicable
    ));
}

#[test]
fn source_domains_are_checked_before_any_mixed_empty_or_family_result() {
    let vars = variables(&["x", "y"]);
    for exponent in [-1, 0] {
        let bad = Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(
                Arc::new(Expr::from_i64(0)),
                Arc::new(Expr::from_i64(exponent)),
            ),
        ]);
        for prefix in [parse("x*(x*y-1)").unwrap(), Expr::from_i64(1)] {
            assert!(rational_components_system(&[prefix, bad.clone()], &vars).is_err());
        }
    }
    assert!(matches!(
        rational_components_system(&sources(&["x*(x*y-1)", "x*x-2", "x*x-3"]), &vars).unwrap(),
        PolynomialComponentsOutcome::Empty
    ));
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    compute: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.compute += amount;
        }
        Ok(())
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        for &(dimension, amount) in charges {
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
fn both_lanes_cancel_during_mixed_search_and_before_publication() {
    let vars = variables(&["x", "y"]);
    let source = sources(&["x*(x*y-1)"]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    assert!(matches!(
        metered_rational_components_system(&source, &vars, &mut generator, &mut verifier).unwrap(),
        PolynomialComponentsOutcome::Families(_)
    ));
    for point in [1, generator.checkpoints / 2, generator.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_rational_components_system(&source, &vars, &mut stopped, &mut Unbounded)
                .is_err()
        );
    }
    for point in [1, verifier.checkpoints / 2, verifier.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_rational_components_system(&source, &vars, &mut Unbounded, &mut stopped)
                .is_err()
        );
    }
}

#[test]
fn actual_budget_exhaustion_never_returns_the_already_solved_branch() {
    let vars = variables(&["x", "y"]);
    let source = sources(&["x*(x*y-1)"]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_rational_components_system(&source, &vars, &mut generator, &mut verifier).unwrap();
    for (lane, amount) in [(true, generator.compute), (false, verifier.compute)] {
        let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = amount - 1;
        let mut stopped = Budget::new(limits);
        let result = if lane {
            metered_rational_components_system(&source, &vars, &mut stopped, &mut Unbounded)
        } else {
            metered_rational_components_system(&source, &vars, &mut Unbounded, &mut stopped)
        };
        assert!(result.is_err());
    }
}
