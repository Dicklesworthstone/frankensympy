//! Complete positive-dimensional polynomial graphs and explicit recognition misses.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::MultivariatePoly;
use fsym_solvers::polynomial_graph::{
    PolynomialGraphOutcome, VerifiedPolynomialGraph, metered_polynomial_graph_system,
    polynomial_graph_system,
};
use std::sync::Arc;

fn variables(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn equations(sources: &[&str]) -> Vec<Expr> {
    sources
        .iter()
        .map(|source| parse(source).unwrap())
        .collect()
}

fn family(sources: &[&str], names: &[&str]) -> VerifiedPolynomialGraph {
    match polynomial_graph_system(&equations(sources), &variables(names)).unwrap() {
        PolynomialGraphOutcome::Family(result) => result,
        other => panic!("expected a complete graph, received {other:?}"),
    }
}

fn assert_coordinates(family: &VerifiedPolynomialGraph, expected: &[&str]) {
    assert_eq!(family.values().len(), expected.len());
    for (value, expected) in family.values().iter().zip(expected) {
        assert_eq!(
            MultivariatePoly::from_expr(value, family.variables()).unwrap(),
            MultivariatePoly::from_expr(&parse(expected).unwrap(), family.variables()).unwrap(),
        );
    }
}

#[test]
fn parabola_orientation_does_not_depend_on_requested_coordinate_order() {
    let first = family(&["x-y**2"], &["x", "y"]);
    assert_coordinates(&first, &["y**2", "y"]);
    assert_eq!(first.free_variables(), variables(&["y"]));
    let second = family(&["y-x**2"], &["x", "y"]);
    assert_coordinates(&second, &["x", "x**2"]);
    assert_eq!(second.free_variables(), variables(&["x"]));
    let reversed = family(&["x-y**2"], &["y", "x"]);
    assert_coordinates(&reversed, &["y", "y**2"]);
    assert_eq!(reversed.variables(), variables(&["y", "x"]));
}

#[test]
fn nonlinear_graphs_retain_every_free_coordinate() {
    let surface = family(&["3*z-2*x*y-1"], &["x", "y", "z", "w"]);
    assert_coordinates(&surface, &["x", "y", "2*x*y/3+1/3", "w"]);
    assert_eq!(surface.free_variables(), variables(&["x", "y", "w"]));
    assert!(!surface.basis().is_zero_dimensional());
}

#[test]
fn chains_eliminate_intermediate_coordinates_and_redundant_constraints() {
    let graph = family(
        &["x-y**2", "3*z-2*x*y-1", "(x-y**2)*(z+1)"],
        &["x", "y", "z"],
    );
    assert_coordinates(&graph, &["y**2", "y", "2*y**3/3+1/3"]);
    assert_eq!(graph.free_variables(), variables(&["y"]));
    assert_eq!(graph.basis().input_basis().len(), 3);
}

#[test]
fn a_graph_hidden_in_nonlinear_generators_is_recovered_by_ideal_equality() {
    // Subtracting the two original equations recovers x-y^2; neither
    // individual input states the complete family on its own.
    let graph = family(
        &["(x-y**2)*(z**2+1)", "(x-y**2)*(z**2+2)"],
        &["x", "y", "z"],
    );
    assert_coordinates(&graph, &["y**2", "y", "z"]);
    assert_eq!(graph.free_variables(), variables(&["y", "z"]));
}

#[test]
fn all_caller_permutations_map_the_internal_basis_back_to_the_requested_tuple() {
    for order in [
        ["x", "y", "z"],
        ["x", "z", "y"],
        ["y", "x", "z"],
        ["y", "z", "x"],
        ["z", "x", "y"],
        ["z", "y", "x"],
    ] {
        let graph = family(&["x-y**2", "z-x**3"], &order);
        let expected: Vec<_> = order
            .iter()
            .map(|name| match *name {
                "x" => "y**2",
                "y" => "y",
                _ => "y**6",
            })
            .collect();
        assert_coordinates(&graph, &expected);
        assert_eq!(graph.free_variables(), variables(&["y"]));
    }
}

#[test]
fn affine_and_unique_rational_solutions_are_also_complete_graphs() {
    assert_coordinates(&family(&["x+2*y-3"], &["x", "y"]), &["3-2*y", "y"]);
    let unique = family(&["3*x-1", "5*y+2"], &["x", "y"]);
    assert_coordinates(&unique, &["1/3", "-2/5"]);
    assert!(unique.free_variables().is_empty());
    assert!(unique.basis().is_zero_dimensional());
}

#[test]
fn zero_ideals_bind_the_full_ambient_ring_and_unit_ideals_are_empty() {
    let vars = variables(&["x", "y", "z"]);
    for input in [vec![], vec![Expr::from_i64(0); 3]] {
        let PolynomialGraphOutcome::Family(graph) = polynomial_graph_system(&input, &vars).unwrap()
        else {
            panic!("zero ideal must retain all free coordinates");
        };
        assert_coordinates(&graph, &["x", "y", "z"]);
        assert_eq!(graph.free_variables(), vars);
        assert_eq!(graph.basis().input_basis()[0].generators(), vars);
        assert!(graph.basis().basis().is_empty());
    }
    for sources in [vec!["1"], vec!["x*x+y*y", "x*x+y*y-1"]] {
        assert!(matches!(
            polynomial_graph_system(&equations(&sources), &vars).unwrap(),
            PolynomialGraphOutcome::Empty
        ));
    }
}

#[test]
fn nonlinear_non_graphs_never_masquerade_as_empty_or_complete_single_families() {
    for sources in [
        vec!["x*y"],
        vec!["x*y-1"],
        vec!["x*x+y*y-1"],
        vec!["x*x-2", "y-x"],
        vec!["x*x", "y-x"],
    ] {
        assert!(matches!(
            polynomial_graph_system(&equations(&sources), &variables(&["x", "y"])).unwrap(),
            PolynomialGraphOutcome::NotApplicable
        ));
    }
}

#[test]
fn same_spelling_free_coordinates_keep_their_typed_identity() {
    let plain = Symbol::new("coordinate");
    let assumed = Symbol::with_identity(
        "coordinate",
        SymbolIdentity::from_assumptions_digest([73; 32]),
    );
    let square = Expr::Pow(
        Arc::new(Expr::Sym(assumed.clone())),
        Arc::new(Expr::from_i64(2)),
    );
    let input = vec![Expr::Sym(plain.clone()) - square.clone()];
    let PolynomialGraphOutcome::Family(graph) =
        polynomial_graph_system(&input, &[plain.clone(), assumed.clone()]).unwrap()
    else {
        panic!("typed parabola must be recognized");
    };
    assert_eq!(graph.values(), &[square, Expr::Sym(assumed.clone())]);
    assert_eq!(graph.free_variables(), &[assumed]);
    assert!(polynomial_graph_system(&input, &[plain.clone(), plain]).is_err());
}

#[test]
fn original_undefined_or_opaque_children_cannot_be_hidden_by_zero_or_unit_equations() {
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
            parse("x-y**2").unwrap(),
        ] {
            assert!(polynomial_graph_system(&[prefix, bad.clone()], &vars).is_err());
        }
    }
    let inverse = Expr::Pow(
        Arc::new(Expr::Sym(vars[0].clone())),
        Arc::new(Expr::from_i64(-1)),
    );
    assert!(
        polynomial_graph_system(
            &[Expr::Mul(vec![Expr::Sym(vars[0].clone()), inverse])],
            &vars
        )
        .is_err()
    );
}

#[test]
fn source_dimensions_coefficient_height_and_power_limits_are_enforced() {
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
        assert!(polynomial_graph_system(&[source], std::slice::from_ref(&x)).is_err());
    }
    assert!(
        polynomial_graph_system(&vec![Expr::from_i64(0); 65], std::slice::from_ref(&x)).is_err()
    );
    assert!(polynomial_graph_system(&[], &[]).is_err());
    let too_many: Vec<_> = (0..9)
        .map(|index| Symbol::new(format!("v{index}")))
        .collect();
    assert!(polynomial_graph_system(&[], &too_many).is_err());
    assert!(polynomial_graph_system(&[], &[x.clone(), x]).is_err());
    assert!(polynomial_graph_system(&[], &[Symbol::new("a".repeat(65_537))]).is_err());
}

#[test]
fn exact_tiny_inconsistency_is_proved_empty() {
    let vars = variables(&["x", "y"]);
    let relation = parse("x-y**2").unwrap();
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(80)));
    assert!(matches!(
        polynomial_graph_system(&[relation.clone(), relation - epsilon], &vars).unwrap(),
        PolynomialGraphOutcome::Empty
    ));
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
fn both_meters_cancel_during_work_and_at_each_kind_of_publication() {
    let vars = variables(&["x", "y"]);
    for sources in [vec!["x-y*y"], vec!["1"], vec!["x*y-1"]] {
        let input = equations(&sources);
        let mut generator = Meter::default();
        let mut verifier = Meter::default();
        metered_polynomial_graph_system(&input, &vars, &mut generator, &mut verifier).unwrap();
        assert!(generator.compute > 0 && generator.memory > 0);
        assert!(verifier.compute > 0 && verifier.memory > 0);
        for point in [1, generator.checkpoints / 2, generator.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            assert!(
                metered_polynomial_graph_system(&input, &vars, &mut stopped, &mut Unbounded)
                    .is_err()
            );
        }
        for point in [1, verifier.checkpoints / 2, verifier.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            assert!(
                metered_polynomial_graph_system(&input, &vars, &mut Unbounded, &mut stopped)
                    .is_err()
            );
        }
    }
}

#[test]
fn actual_budget_exhaustion_cannot_publish_a_graph() {
    let input = equations(&["x-y*y"]);
    let vars = variables(&["x", "y"]);
    let mut generator = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        metered_polynomial_graph_system(&input, &vars, &mut generator, &mut untouched).is_err()
    );
    assert_eq!(untouched.checkpoints, 0);
    let mut verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(metered_polynomial_graph_system(&input, &vars, &mut Unbounded, &mut verifier).is_err());
}
