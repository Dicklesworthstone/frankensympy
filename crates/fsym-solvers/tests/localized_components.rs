//! Complete positive-dimensional rational sources retain all original guards.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    RationalSystemAdmission, VerifiedRationalLocalization, metered_localize_rational_system,
};
use fsym_solvers::localized_components::{
    LocalizedComponentsOutcome, localized_rational_components,
    metered_localized_rational_components,
};

fn source(text: &str) -> Expr {
    parse(text).unwrap()
}

fn variables(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn localize(input: &[Expr], variables: &[Symbol]) -> VerifiedRationalLocalization {
    match metered_localize_rational_system(input, variables, &mut Unbounded, &mut Unbounded)
        .unwrap()
    {
        RationalSystemAdmission::Localized(value) => *value,
        _ => panic!("expected an original rational source"),
    }
}

#[test]
fn reciprocal_equation_keeps_the_complete_hyperbola_and_original_parameter() {
    let vars = variables(&["x", "y"]);
    let localization = localize(&[source("1/x-y")], &vars);
    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        panic!("expected rational family")
    };
    assert_eq!(result.variables(), vars);
    assert_eq!(result.localization().sources(), localization.sources());
    assert!(!result.is_finite());
    assert_eq!(result.families().len(), 1);
    let family = result.families().next().unwrap();
    assert_eq!(family.free_variables(), &[vars[1].clone()]);
    assert_eq!(family.values()[1], Expr::Sym(vars[1].clone()));
    assert_eq!(family.nonzero_conditions(), &[Expr::Sym(vars[1].clone())]);
    let parts = result.into_parts();
    assert_eq!(
        parts,
        vec![(
            vec![Expr::from_i64(1), source("y")],
            vec![source("y"), Expr::from_i64(1)],
            vec![source("y")],
        )]
    );
}

#[test]
fn caller_order_and_independent_free_coordinates_remain_original() {
    let vars = variables(&["y", "z", "x"]);
    let localization = localize(&[source("1/x-y")], &vars);
    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        panic!("expected reordered family")
    };
    let family = result.families().next().unwrap();
    assert_eq!(family.values()[1..], [source("z"), source("x")]);
    assert_eq!(family.free_variables(), &vars[1..]);
    assert_eq!(family.nonzero_conditions(), &[source("x")]);
    for value in family.values().iter().chain(family.nonzero_conditions()) {
        assert!(
            !value
                .free_symbols()
                .contains(&localization.certificate().auxiliary)
        );
    }
}

#[test]
fn constraints_used_only_by_auxiliary_coordinate_survive_native_projection() {
    let vars = variables(&["x", "y"]);
    let localization = localize(&[source("(x-y)/(x-1)")], &vars);
    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        panic!("expected a punctured diagonal")
    };
    let family = result.families().next().unwrap();
    assert_eq!(family.values(), [source("y"), source("y")]);
    assert_eq!(family.nonzero_conditions(), &[source("y-1")]);
    assert!(result.into_finite_values().is_err());

    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        unreachable!()
    };
    let parts = result.into_parts();
    assert_eq!(parts[0].1, [Expr::from_i64(1), Expr::from_i64(1)]);
    assert_eq!(parts[0].2, [source("y-1")]);
}

#[test]
fn held_zero_and_identity_sources_keep_punctured_parameter_domains() {
    let vars = variables(&["x", "y"]);
    let inverse = power(source("x"), -1);
    let sources = [
        Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
        Expr::Add(vec![
            Expr::Mul(vec![source("x"), inverse.clone()]),
            Expr::from_i64(-1),
        ]),
        Expr::Add(vec![power(inverse, 0), Expr::from_i64(-1)]),
    ];
    for input in sources {
        let localization = localize(&[input], &vars);
        let LocalizedComponentsOutcome::Families(result) =
            localized_rational_components(&localization).unwrap()
        else {
            panic!("expected original punctured domain")
        };
        let family = result.families().next().unwrap();
        assert_eq!(family.values(), [source("x"), source("y")]);
        assert_eq!(family.nonzero_conditions(), &[source("x")]);
    }
}

#[test]
fn original_poles_remove_an_entire_spurious_component_without_losing_the_valid_one() {
    let vars = variables(&["x", "y"]);
    let localization = localize(&[source("x*(1/x-y)")], &vars);
    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        panic!("expected hyperbola after original-domain exclusion")
    };
    assert_eq!(result.families().len(), 1);
    let parts = result.into_parts();
    assert_eq!(parts[0].0, [Expr::from_i64(1), source("y")]);
    assert_eq!(parts[0].1, [source("y"), Expr::from_i64(1)]);
    assert_eq!(parts[0].2, [source("y")]);
}

#[test]
fn factored_rational_union_keeps_every_branch_and_separate_conditions() {
    let vars = variables(&["x", "y"]);
    let localization = localize(&[source("(1/x-y)*(1/x+y)")], &vars);
    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        panic!("expected both hyperbolas")
    };
    let parts = result.into_parts();
    assert_eq!(parts.len(), 2);
    for (numerators, denominators, guards) in &parts {
        assert_eq!(numerators[1], source("y"));
        assert_eq!(denominators, &[source("y"), Expr::from_i64(1)]);
        assert_eq!(guards, &[source("y")]);
        assert!(numerators[0] == Expr::from_i64(1) || numerators[0] == Expr::from_i64(-1));
    }
    assert_ne!(parts[0].0[0], parts[1].0[0]);
}

#[test]
fn same_printing_typed_variables_and_auxiliary_name_collisions_stay_distinct() {
    let plain = Symbol::new("_fsym_rational_inverse_0");
    let assumed = Symbol::with_identity(
        "_fsym_rational_inverse_0",
        SymbolIdentity::from_assumptions_digest([105; 32]),
    );
    let input = power(Expr::Sym(plain.clone()), -1) - Expr::Sym(assumed.clone());
    let vars = [plain.clone(), assumed.clone()];
    let localization = localize(&[input], &vars);
    assert_ne!(localization.certificate().auxiliary.name, plain.name);
    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        panic!("typed original parameters must remain distinct")
    };
    let family = result.families().next().unwrap();
    assert_eq!(family.free_variables(), std::slice::from_ref(&assumed));
    assert_eq!(family.values()[1], Expr::Sym(assumed.clone()));
    assert_eq!(family.nonzero_conditions(), &[Expr::Sym(assumed)]);
}

#[test]
fn unsupported_original_component_prevents_any_partial_union() {
    let vars = variables(&["x", "y"]);
    let localization = localize(&[source("(1/x-y)*(x*x+y*y-1)")], &vars);
    assert!(matches!(
        localized_rational_components(&localization).unwrap(),
        LocalizedComponentsOutcome::NotApplicable,
    ));
}

#[test]
fn empty_loci_and_finite_points_keep_their_existing_meaning() {
    let vars = variables(&["x", "y"]);
    let localization = localize(&[power(power(source("x"), -1), -1)], &vars);
    assert!(matches!(
        localized_rational_components(&localization).unwrap(),
        LocalizedComponentsOutcome::Empty,
    ));
    let localization = localize(&[source("(x*x-1)/(x-1)"), source("y-x")], &vars);
    let LocalizedComponentsOutcome::Families(result) =
        localized_rational_components(&localization).unwrap()
    else {
        panic!("expected the complete finite point set")
    };
    assert!(result.is_finite());
    assert_eq!(
        result.into_finite_values().unwrap(),
        [vec![Expr::from_i64(-1); 2]]
    );
}

#[derive(Default)]
struct Meter {
    compute: u64,
    checkpoints: usize,
    stop: Option<usize>,
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
        if self.stop == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn projection_search_and_reference_share_cumulative_budgets_and_cancellation() {
    let vars = variables(&["x", "y"]);
    let localization = localize(&[source("1/x-y")], &vars);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    metered_localized_rational_components(&localization, &mut generation, &mut verification)
        .unwrap();
    assert!(generation.compute > 0 && verification.compute > 0);
    for (generator_lane, measured) in [(true, &generation), (false, &verification)] {
        for point in [1, measured.checkpoints / 2, measured.checkpoints] {
            let mut stopped = Meter {
                stop: Some(point),
                ..Meter::default()
            };
            let result = if generator_lane {
                metered_localized_rational_components(&localization, &mut stopped, &mut Unbounded)
            } else {
                metered_localized_rational_components(&localization, &mut Unbounded, &mut stopped)
            };
            assert!(result.is_err());
        }
        let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = measured.compute - 1;
        let mut stopped = Budget::new(limits);
        let result = if generator_lane {
            metered_localized_rational_components(&localization, &mut stopped, &mut Unbounded)
        } else {
            metered_localized_rational_components(&localization, &mut Unbounded, &mut stopped)
        };
        assert!(result.is_err());
    }
}
