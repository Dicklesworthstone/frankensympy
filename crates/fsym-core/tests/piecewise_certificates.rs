//! Analytic and adversarial checks for source-bound Piecewise interval claims.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_NODES, RealBall, RootBoxError,
    RootBoxWitness, StationaryPointKind, Symbol, SymbolIdentity, certify_root_box,
    certify_stationary_point, enclose_objective, enclose_residual_jacobian,
    metered_enclose_residual_jacobian,
};
use std::{slice, sync::Arc};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn interval(lo: i64, hi: i64) -> RealBall {
    RealBall::new(q(lo + hi, 2), q(hi - lo, 2)).unwrap()
}
fn function(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Function(name.into(), args)
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn truth(value: bool) -> Expr {
    Expr::symbol(if value { "True" } else { "False" })
}
fn relation(name: &str, left: Expr, right: Expr) -> Expr {
    function(name, vec![left, right])
}
fn piecewise(branches: Vec<(Expr, Expr)>) -> Expr {
    function(
        "Piecewise",
        branches
            .into_iter()
            .map(|(value, condition)| function("Tuple", vec![value, condition]))
            .collect(),
    )
}
fn conditional(condition: Expr) -> Expr {
    piecewise(vec![
        (Expr::from_i64(7), condition),
        (Expr::from_i64(11), truth(true)),
    ])
}

#[test]
fn selected_logarithm_and_square_root_keep_original_source_and_exact_slopes() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let source = piecewise(vec![
        (
            function("log", vec![x.clone()]),
            relation("StrictGreaterThan", x.clone(), Expr::from_i64(0)),
        ),
        (
            function("sqrt", vec![-x.clone()]),
            relation("StrictLessThan", x, Expr::from_i64(0)),
        ),
        (Expr::from_i64(0), truth(true)),
    ]);
    let positive =
        enclose_residual_jacobian(slice::from_ref(&source), &vars, &[interval(1, 2)], 24).unwrap();
    assert_eq!(positive.residuals(), slice::from_ref(&source));
    assert_eq!(positive.variables(), vars);
    assert_eq!(positive.domain(), [interval(1, 2)]);
    assert!(positive.residual_values()[0].contains(&q(0, 1)));
    for slope in [q(1, 2), q(1, 1)] {
        assert!(positive.jacobian()[0][0].contains(&slope));
    }
    let negative = enclose_residual_jacobian(&[source], &vars, &[interval(-4, -1)], 24).unwrap();
    for value in [q(1, 1), q(2, 1)] {
        assert!(negative.residual_values()[0].contains(&value));
    }
    for slope in [q(-1, 2), q(-1, 4)] {
        assert!(negative.jacobian()[0][0].contains(&slope));
    }
}

#[test]
fn ordered_overlapping_and_nested_branches_are_selected_before_arithmetic() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let nested = piecewise(vec![
        (function("log", vec![Expr::from_i64(0)]), truth(false)),
        (power(x.clone(), 2), truth(true)),
    ]);
    let source = Expr::from_i64(3)
        + piecewise(vec![
            (
                nested,
                relation("StrictGreaterThan", x.clone(), Expr::from_i64(0)),
            ),
            (
                Expr::from_i64(999),
                relation("StrictGreaterThan", x, Expr::from_i64(1)),
            ),
        ]);
    let accepted = enclose_objective(&source, &vars, &[interval(2, 3)], 24).unwrap();
    for value in [7, 12] {
        assert!(accepted.value().contains(&q(value, 1)));
    }
    for slope in [4, 6] {
        assert!(accepted.gradient()[0].contains(&q(slope, 1)));
    }
    assert_eq!(accepted.hessian(), [vec![RealBall::from_i64(2)]]);
}

#[test]
fn all_six_relations_use_strict_separation_for_both_outcomes() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    for (name, negative_truth, positive_truth) in [
        ("Equality", false, false),
        ("Unequality", true, true),
        ("StrictLessThan", true, false),
        ("LessThan", true, false),
        ("StrictGreaterThan", false, true),
        ("GreaterThan", false, true),
    ] {
        let source = conditional(relation(name, x.clone(), Expr::from_i64(0)));
        for (domain, expected) in [
            (interval(-2, -1), negative_truth),
            (interval(1, 2), positive_truth),
        ] {
            let accepted =
                enclose_residual_jacobian(slice::from_ref(&source), &vars, &[domain], 24).unwrap();
            assert_eq!(
                accepted.residual_values(),
                [RealBall::from_i64(if expected { 7 } else { 11 })],
                "{name}"
            );
            assert_eq!(accepted.jacobian(), [vec![RealBall::from_i64(0)]]);
        }
    }
}

#[test]
fn point_equalities_and_touching_or_crossing_seams_never_prove_smoothness() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    for name in [
        "Equality",
        "Unequality",
        "StrictLessThan",
        "LessThan",
        "StrictGreaterThan",
        "GreaterThan",
    ] {
        let source = conditional(relation(name, x.clone(), Expr::from_i64(0)));
        for domain in [
            interval(-1, 1),
            interval(0, 0),
            interval(0, 1),
            interval(-1, 0),
        ] {
            assert!(matches!(
                enclose_residual_jacobian(
                    slice::from_ref(&source),
                    &vars,
                    slice::from_ref(&domain),
                    24
                ),
                Err(RootBoxError::Domain(_))
            ));
            assert!(matches!(
                enclose_objective(&source, &vars, &[domain], 24),
                Err(RootBoxError::Domain(_))
            ));
        }
    }
}

#[test]
fn identical_defined_sources_are_stable_even_when_intervals_overlap() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let operand = Expr::Add(vec![power(x, 2), Expr::from_i64(1)]);
    for (name, expected) in [
        ("Equality", true),
        ("Unequality", false),
        ("StrictLessThan", false),
        ("LessThan", true),
        ("StrictGreaterThan", false),
        ("GreaterThan", true),
    ] {
        let source = conditional(relation(name, operand.clone(), operand.clone()));
        let accepted = enclose_objective(&source, &vars, &[interval(-1, 1)], 24).unwrap();
        assert_eq!(
            accepted.value(),
            &RealBall::from_i64(if expected { 7 } else { 11 })
        );
    }
    let undefined = function("log", vec![Expr::from_i64(0)]);
    let source = conditional(relation("Equality", undefined.clone(), undefined));
    assert!(matches!(
        enclose_objective(&source, &vars, &[interval(-1, 1)], 24),
        Err(RootBoxError::Domain(_))
    ));
}

#[test]
fn same_printed_names_with_different_typed_identities_are_not_equal_guards() {
    let vars = [
        Symbol::new("x"),
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([17; 32])),
    ];
    let source = conditional(relation(
        "Equality",
        Expr::Sym(vars[0].clone()),
        Expr::Sym(vars[1].clone()),
    ));
    assert!(matches!(
        enclose_objective(&source, &vars, &[interval(0, 0), interval(0, 0)], 24),
        Err(RootBoxError::Domain(_))
    ));
    let accepted =
        enclose_objective(&source, &vars, &[interval(-2, -1), interval(1, 2)], 24).unwrap();
    assert_eq!(accepted.value(), &RealBall::from_i64(11));
}

#[test]
fn boolean_short_circuit_skips_only_unvisited_domain_obligations() {
    let vars = [Symbol::new("x")];
    let bad = relation(
        "StrictGreaterThan",
        function("log", vec![Expr::from_i64(0)]),
        Expr::from_i64(0),
    );
    for (guard, expected) in [
        (function("And", vec![truth(false), bad.clone()]), false),
        (function("Or", vec![truth(true), bad.clone()]), true),
        (
            function(
                "Not",
                vec![function("And", vec![truth(false), bad.clone()])],
            ),
            true,
        ),
        (function("And", vec![]), true),
        (function("Or", vec![]), false),
    ] {
        let accepted =
            enclose_objective(&conditional(guard), &vars, &[interval(-1, 1)], 24).unwrap();
        assert_eq!(
            accepted.value(),
            &RealBall::from_i64(if expected { 7 } else { 11 })
        );
    }
    for guard in [
        function("And", vec![truth(true), bad.clone()]),
        function("Or", vec![truth(false), bad]),
    ] {
        assert!(matches!(
            enclose_objective(&conditional(guard), &vars, &[interval(-1, 1)], 24),
            Err(RootBoxError::Domain(_))
        ));
    }
}

#[test]
fn symbolic_boolean_names_no_match_and_malformed_branches_refuse() {
    let vars = [Symbol::new("x")];
    for guard in [
        Expr::Sym(Symbol::with_identity(
            "True",
            SymbolIdentity::from_assumptions_digest([3; 32]),
        )),
        function("_FsymBooleanSymbol", vec![Expr::symbol("False")]),
        Expr::Sym(vars[0].clone()),
    ] {
        assert!(matches!(
            enclose_objective(&conditional(guard), &vars, &[interval(1, 2)], 24),
            Err(RootBoxError::Unsupported(_))
        ));
    }
    assert!(matches!(
        enclose_objective(
            &piecewise(vec![(Expr::from_i64(1), truth(false))]),
            &vars,
            &[interval(1, 2)],
            24,
        ),
        Err(RootBoxError::Domain(_))
    ));
    for source in [
        piecewise(vec![]),
        function("Piecewise", vec![Expr::from_i64(1)]),
        function(
            "Piecewise",
            vec![
                function("Tuple", vec![Expr::from_i64(1), truth(true)]),
                function("Tuple", vec![Expr::from_i64(2)]),
            ],
        ),
    ] {
        assert!(matches!(
            enclose_objective(&source, &vars, &[interval(1, 2)], 24),
            Err(RootBoxError::Unsupported(_))
        ));
    }
}

#[test]
fn root_and_stationary_certificates_bind_the_original_piecewise_source() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let guard = relation("StrictGreaterThan", x.clone(), Expr::from_i64(0));
    let residue = piecewise(vec![
        (x.clone() - Expr::from_i64(2), guard.clone()),
        (function("log", vec![-x.clone()]), truth(true)),
    ]);
    let mut witness = RootBoxWitness {
        domain: vec![interval(1, 3)],
        center: vec![q(2, 1)],
        preconditioner: vec![vec![q(1, 1)]],
        precision_digits: 24,
    };
    let accepted = certify_root_box(slice::from_ref(&residue), &vars, &witness).unwrap();
    assert_eq!(accepted.residuals(), [residue]);
    assert_eq!(accepted.root_enclosure(), [RealBall::from_i64(2)]);
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    let objective = piecewise(vec![
        (power(x.clone() - Expr::from_i64(2), 2), guard),
        (function("log", vec![-x]), truth(true)),
    ]);
    witness.preconditioner[0][0] = q(1, 2);
    let accepted = certify_stationary_point(&objective, &vars, &witness).unwrap();
    assert_eq!(accepted.objective(), &objective);
    assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
    assert_eq!(accepted.stationary_enclosure(), [RealBall::from_i64(2)]);
    witness.domain[0] = interval(-1, 3);
    assert!(matches!(
        certify_stationary_point(&objective, &vars, &witness),
        Err(RootBoxError::Domain(_))
    ));
}

#[test]
fn selected_held_zero_cannot_erase_a_pole_and_dead_sources_remain_bounded() {
    let vars = [Symbol::new("x")];
    let undefined = Expr::Mul(vec![
        Expr::from_i64(0),
        function("log", vec![Expr::from_i64(0)]),
    ]);
    assert!(matches!(
        enclose_objective(
            &piecewise(vec![(undefined, truth(true))]),
            &vars,
            &[interval(1, 2)],
            24
        ),
        Err(RootBoxError::Domain(_))
    ));
    let mut deep = truth(false);
    for _ in 0..MAX_ROOT_BOX_DEPTH {
        deep = function("Not", vec![deep]);
    }
    for source in [
        piecewise(vec![
            (Expr::from_i64(1), truth(true)),
            (Expr::from_i64(0), deep),
        ]),
        piecewise(vec![
            (Expr::from_i64(1), truth(true)),
            (
                Expr::Add(vec![Expr::from_i64(0); MAX_ROOT_BOX_NODES]),
                truth(false),
            ),
        ]),
    ] {
        assert!(matches!(
            enclose_objective(&source, &vars, &[interval(1, 2)], 24),
            Err(RootBoxError::ResourceLimit(_))
        ));
    }
}

#[derive(Default)]
struct Cancel {
    visits: usize,
    stop: Option<usize>,
}
impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let here = self.visits;
        self.visits += 1;
        if self.stop == Some(here) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn guard_work_and_final_publication_share_one_cancellable_ledger() {
    let vars = [Symbol::new("x")];
    let source = conditional(relation(
        "StrictGreaterThan",
        Expr::Sym(vars[0].clone()),
        Expr::from_i64(0),
    ));
    let domain = [interval(1, 2)];
    let mut count = Cancel::default();
    metered_enclose_residual_jacobian(slice::from_ref(&source), &vars, &domain, 24, &mut count)
        .unwrap();
    for stop in 0..count.visits {
        let mut cancel = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            metered_enclose_residual_jacobian(
                slice::from_ref(&source),
                &vars,
                &domain,
                24,
                &mut cancel
            ),
            Err(RootBoxError::Meter(MeterError::Cancelled)),
            "checkpoint {stop}",
        );
    }
    let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 10;
    let mut budget = Budget::new(limits);
    assert!(matches!(
        metered_enclose_residual_jacobian(&[source], &vars, &domain, 24, &mut budget),
        Err(RootBoxError::Meter(_))
    ));
}
