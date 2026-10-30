//! Complete rational families, including their exceptional-fiber obligations.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::MultivariatePoly;
use fsym_solvers::rational_graph::{
    RationalGraphOutcome, VerifiedRationalGraph, metered_rational_graph_system,
    rational_graph_system,
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

fn family(sources: &[&str], names: &[&str]) -> VerifiedRationalGraph {
    match rational_graph_system(&equations(sources), &variables(names)).unwrap() {
        RationalGraphOutcome::Family(result) => *result,
        other => panic!("expected a complete rational graph, received {other:?}"),
    }
}

fn assert_coordinate_parts(
    family: &VerifiedRationalGraph,
    numerators: &[&str],
    denominators: &[&str],
) {
    for ((actual_numerator, actual_denominator), (numerator, denominator)) in family
        .numerators()
        .iter()
        .zip(family.denominators())
        .zip(numerators.iter().zip(denominators))
    {
        assert_eq!(
            *actual_numerator,
            MultivariatePoly::from_expr(&parse(numerator).unwrap(), family.variables()).unwrap()
        );
        assert_eq!(
            *actual_denominator,
            MultivariatePoly::from_expr(&parse(denominator).unwrap(), family.variables()).unwrap()
        );
    }
}

#[test]
fn hyperbola_is_a_complete_family_with_a_retained_pole() {
    let graph = family(&["x*y-1"], &["x", "y"]);
    assert_coordinate_parts(&graph, &["1", "y"], &["y", "1"]);
    assert_eq!(graph.free_variables(), variables(&["y"]));
    assert_eq!(graph.nonzero_conditions(), equations(&["y"]));
    assert_eq!(graph.nonzero_bases().len(), 1);
    assert!(
        graph.nonzero_bases()[0]
            .basis()
            .iter()
            .any(MultivariatePoly::is_one)
    );
    match &graph.values()[0] {
        Expr::Mul(factors) => assert!(
            matches!(&factors[1], Expr::Pow(_, exponent) if **exponent == Expr::from_i64(-1))
        ),
        other => panic!("a held reciprocal domain must remain, received {other:?}"),
    }
}

#[test]
fn caller_coordinate_order_and_rational_scaling_are_preserved() {
    let graph = family(&["2*x*y-3"], &["y", "x"]);
    // The checked monic relation removes the harmless nonzero scalar.
    assert_coordinate_parts(&graph, &["3/2", "x"], &["x", "1"]);
    assert_eq!(graph.variables(), variables(&["y", "x"]));
    assert_eq!(graph.free_variables(), variables(&["x"]));
}

#[test]
fn several_reciprocal_coordinates_share_one_parameter_and_one_guard() {
    let graph = family(&["x*y-1", "z*y-2"], &["x", "z", "y"]);
    assert_coordinate_parts(&graph, &["1", "2", "y"], &["y", "y", "1"]);
    assert_eq!(graph.nonzero_conditions(), equations(&["y"]));
    assert_eq!(graph.nonzero_bases().len(), 1);
    assert_eq!(graph.free_variables(), variables(&["y"]));
}

#[test]
fn independent_reciprocal_components_retain_both_domains() {
    let graph = family(&["x*y-1", "z*w-1"], &["x", "y", "z", "w"]);
    assert_coordinate_parts(&graph, &["1", "y", "1", "w"], &["y", "1", "w", "1"]);
    assert_eq!(graph.nonzero_conditions(), equations(&["y", "w"]));
    assert_eq!(graph.free_variables(), variables(&["y", "w"]));
}

#[test]
fn nonlinear_denominators_retain_their_complex_poles() {
    let graph = family(&["x*(y*y+1)-1"], &["x", "y"]);
    assert_coordinate_parts(&graph, &["1", "y"], &["y*y+1", "1"]);
    assert_eq!(graph.nonzero_conditions().len(), 1);
    let guard =
        MultivariatePoly::from_expr(&graph.nonzero_conditions()[0], graph.variables()).unwrap();
    assert_eq!(
        guard,
        MultivariatePoly::from_expr(&parse("y*y+1").unwrap(), graph.variables()).unwrap()
    );
    // The domain is complex as well as real; the guard cannot disappear merely
    // because this particular denominator has no real zero.
    assert_eq!(graph.nonzero_bases()[0].input_basis().last(), Some(&guard));
}

#[test]
fn ideal_reduction_recovers_a_graph_hidden_in_nonlinear_generators() {
    let graph = family(&["(x*y-1)*(z*z+1)", "(x*y-1)*(z*z+2)"], &["x", "y", "z"]);
    assert_coordinate_parts(&graph, &["1", "y", "z"], &["y", "1", "1"]);
    assert_eq!(graph.free_variables(), variables(&["y", "z"]));
    assert_eq!(graph.basis().input_basis().len(), 2);
}

#[test]
fn exceptional_fibers_cannot_be_silently_discarded() {
    for sources in [vec!["x*y"], vec!["x*(y-1)"], vec!["x*y-y*y"]] {
        assert!(matches!(
            rational_graph_system(&equations(&sources), &variables(&["x", "y"])).unwrap(),
            RationalGraphOutcome::NotApplicable
        ));
    }
    // x=z/y omits the y=z=0 fiber. The complete polynomial chart z=x*y
    // accounts for that fiber and is the only accepted chart here.
    let graph = family(&["x*y-z"], &["x", "y", "z"]);
    assert_coordinate_parts(&graph, &["x", "y", "x*y"], &["1", "1", "1"]);
    assert!(graph.nonzero_conditions().is_empty());
}

#[test]
fn hidden_free_parameter_equations_and_finite_algebraic_sets_are_not_graphs() {
    for sources in [
        vec!["x*y-1", "y*y-2"],
        vec!["x*y-1", "z*z-2"],
        vec!["x*x+y*y-1"],
    ] {
        assert!(matches!(
            rational_graph_system(&equations(&sources), &variables(&["x", "y", "z"])).unwrap(),
            RationalGraphOutcome::NotApplicable
        ));
    }
}

#[test]
fn exact_small_residuals_are_never_rounded_away() {
    let vars = variables(&["x", "y", "z"]);
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(80)));
    let input = vec![
        parse("x*y-1").unwrap(),
        Expr::Mul(vec![epsilon, parse("z*z-2").unwrap()]),
    ];
    assert!(matches!(
        rational_graph_system(&input, &vars).unwrap(),
        RationalGraphOutcome::NotApplicable
    ));
}

#[test]
fn zero_and_unit_ideals_have_distinct_complete_outcomes() {
    let vars = variables(&["x", "y"]);
    for input in [vec![], vec![Expr::from_i64(0); 2]] {
        let RationalGraphOutcome::Family(graph) = rational_graph_system(&input, &vars).unwrap()
        else {
            panic!("the zero ideal must keep every coordinate free");
        };
        assert_eq!(graph.free_variables(), vars);
        assert_eq!(graph.values(), equations(&["x", "y"]));
        assert!(graph.nonzero_conditions().is_empty());
    }
    for sources in [vec!["1"], vec!["x*y-1", "x*y-2"]] {
        assert!(matches!(
            rational_graph_system(&equations(&sources), &vars).unwrap(),
            RationalGraphOutcome::Empty
        ));
    }
}

#[test]
fn same_printed_symbols_keep_independent_parameter_identity() {
    let plain = Symbol::new("coordinate");
    let assumed = Symbol::with_identity(
        "coordinate",
        SymbolIdentity::from_assumptions_digest([93; 32]),
    );
    let input =
        Expr::Mul(vec![Expr::Sym(plain.clone()), Expr::Sym(assumed.clone())]) - Expr::from_i64(1);
    let vars = [plain.clone(), assumed.clone()];
    let RationalGraphOutcome::Family(graph) = rational_graph_system(&[input], &vars).unwrap()
    else {
        panic!("typed hyperbola must be recognized");
    };
    assert_eq!(graph.free_variables(), std::slice::from_ref(&assumed));
    assert_eq!(graph.nonzero_conditions(), &[Expr::Sym(assumed)]);
    assert_eq!(graph.variables(), &vars);
    assert!(rational_graph_system(&[], &[plain.clone(), plain]).is_err());
}

#[test]
fn original_undefined_children_are_admitted_before_empty_or_chart_results() {
    let vars = variables(&["x", "y"]);
    for hidden in [
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(0))),
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
        Expr::Function("opaque".into(), vec![Expr::Sym(vars[0].clone())]),
    ] {
        let bad = Expr::Mul(vec![Expr::from_i64(0), hidden]);
        for prefix in [
            Expr::from_i64(0),
            Expr::from_i64(1),
            parse("x*y-1").unwrap(),
        ] {
            assert!(rational_graph_system(&[prefix, bad.clone()], &vars).is_err());
        }
    }
    let inverse = Expr::Pow(
        Arc::new(Expr::Sym(vars[0].clone())),
        Arc::new(Expr::from_i64(-1)),
    );
    assert!(
        rational_graph_system(
            &[Expr::Mul(vec![Expr::Sym(vars[0].clone()), inverse])],
            &vars
        )
        .is_err()
    );
}

#[test]
fn source_dimensions_coefficient_height_and_exponent_limits_are_enforced() {
    let x = Symbol::new("x");
    for source in [
        Expr::Pow(
            Arc::new(Expr::Sym(x.clone())),
            Arc::new(Expr::from_i64(257)),
        ),
        Expr::Pow(
            Arc::new(Expr::from_i64(2)),
            Arc::new(Expr::from_i64(i64::MAX)),
        ),
        Expr::Integer(BigInt::from(1) << 4097),
    ] {
        assert!(rational_graph_system(&[source], std::slice::from_ref(&x)).is_err());
    }
    assert!(rational_graph_system(&vec![Expr::from_i64(0); 65], std::slice::from_ref(&x)).is_err());
    assert!(rational_graph_system(&[], &[]).is_err());
    let too_many: Vec<_> = (0..9)
        .map(|index| Symbol::new(format!("v{index}")))
        .collect();
    assert!(rational_graph_system(&[], &too_many).is_err());
    assert!(rational_graph_system(&[], &[Symbol::new("a".repeat(65_537))]).is_err());
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
fn both_lanes_cancel_during_search_verification_and_publication() {
    let vars = variables(&["x", "y"]);
    for sources in [vec!["x*y-1"], vec!["1"], vec!["x*y"]] {
        let input = equations(&sources);
        let mut generator = Meter::default();
        let mut verifier = Meter::default();
        metered_rational_graph_system(&input, &vars, &mut generator, &mut verifier).unwrap();
        assert!(generator.compute > 0 && generator.memory > 0);
        assert!(verifier.compute > 0 && verifier.memory > 0);
        for point in [1, generator.checkpoints / 2, generator.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            assert!(
                metered_rational_graph_system(&input, &vars, &mut stopped, &mut Unbounded).is_err()
            );
        }
        for point in [1, verifier.checkpoints / 2, verifier.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            assert!(
                metered_rational_graph_system(&input, &vars, &mut Unbounded, &mut stopped).is_err()
            );
        }
    }
}

#[test]
fn real_budget_exhaustion_cannot_publish_a_chart() {
    let input = equations(&["x*y-1"]);
    let vars = variables(&["x", "y"]);
    let mut generator = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(metered_rational_graph_system(&input, &vars, &mut generator, &mut untouched).is_err());
    assert_eq!(untouched.checkpoints, 0);
    let mut verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(metered_rational_graph_system(&input, &vars, &mut Unbounded, &mut verifier).is_err());
}
