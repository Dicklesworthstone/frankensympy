//! Independent complete-set examples, original-domain cases and budget adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::real_roots::RealRootError;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_solvers::{
    MAX_REAL_CONSTRAINTS, MAX_REAL_FORMULA_DEPTH, MAX_REAL_FORMULA_FANOUT, MAX_REAL_FORMULA_NODES,
    RealConstraint, RealEquationError, RealFormula, RealRelation, VerifiedRealConstraints,
    metered_solve_real_constraints, solve_real_constraints,
};
use std::sync::Arc;

fn pow(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn shift(x: &Symbol, constant: i64) -> Expr {
    Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(constant)])
}

fn quadratic(x: &Symbol, constant: i64) -> Expr {
    Expr::Add(vec![pow(Expr::Sym(x.clone()), 2), Expr::from_i64(constant)])
}

fn atom(expression: Expr, relation: RealRelation) -> RealConstraint {
    RealConstraint {
        expression,
        relation,
    }
}

fn all(indices: &[usize]) -> RealFormula {
    RealFormula::All(indices.iter().copied().map(RealFormula::Atom).collect())
}

fn any(indices: &[usize]) -> RealFormula {
    RealFormula::Any(indices.iter().copied().map(RealFormula::Atom).collect())
}

fn solve(
    constraints: &[RealConstraint],
    x: &Symbol,
    formula: &RealFormula,
) -> VerifiedRealConstraints {
    solve_real_constraints(constraints, x, formula)
        .expect("bounded admitted exact problem")
        .expect("supported rational sources")
}

fn chart(result: &VerifiedRealConstraints, coefficients: &[i64], gaps: &[bool], points: &[bool]) {
    assert_eq!(
        result.polynomial().coeffs,
        coefficients
            .iter()
            .copied()
            .map(BigRational::from)
            .collect::<Vec<_>>()
    );
    assert_eq!(result.roots().coefficients(), result.polynomial().coeffs);
    assert_eq!(result.gap_inclusions(), gaps);
    assert_eq!(result.point_inclusions(), points);
    assert_eq!(result.roots().intervals().len(), points.len());
    assert_eq!(gaps.len(), points.len() + 1);
    assert!(
        result
            .roots()
            .intervals()
            .iter()
            .all(|root| root.multiplicity == 1)
    );
}

#[test]
fn irrational_annulus_has_complete_exact_closed_boundaries() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let constraints = vec![
        atom(quadratic(&x, -2), RealRelation::Ge),
        atom(quadratic(&x, -3), RealRelation::Le),
    ];
    let formula = all(&[0, 1]);
    let result = solve(&constraints, &x, &formula);
    // [-sqrt(3), -sqrt(2)] union [sqrt(2), sqrt(3)].
    chart(
        &result,
        &[6, 0, -5, 0, 1],
        &[false, true, false, true, false],
        &[true, true, true, true],
    );
    assert_eq!(result.constraints(), constraints);
    assert_eq!(result.variable(), &x);
    assert_eq!(result.polynomial().gen_sym, x);
    assert_eq!(result.formula(), &formula);
}

#[test]
fn a_foreign_algebraic_boundary_inherits_the_correct_local_gap() {
    let x = Symbol::new("x");
    let constraints = [
        atom(quadratic(&x, -2), RealRelation::Lt),
        atom(shift(&x, -1), RealRelation::Ge),
    ];
    // The rational point 1 is inside the first atom's open positive-root gap.
    // The complete answer is [1, sqrt(2)), not three disconnected boundaries.
    chart(
        &solve(&constraints, &x, &all(&[0, 1])),
        &[2, -2, -1, 1],
        &[false, false, true, false],
        &[false, true, false],
    );
}

#[test]
fn unions_mix_isolated_equation_roots_with_open_inequality_intervals() {
    let x = Symbol::new("x");
    let constraints = [
        atom(quadratic(&x, -2), RealRelation::Lt),
        atom(shift(&x, -3), RealRelation::Eq),
    ];
    // (-sqrt(2), sqrt(2)) union {3}.
    chart(
        &solve(&constraints, &x, &any(&[0, 1])),
        &[6, -2, -3, 1],
        &[false, true, false, false],
        &[false, false, true],
    );
    let constraints = [
        atom(quadratic(&x, -2), RealRelation::Eq),
        atom(Expr::Sym(x.clone()), RealRelation::Gt),
    ];
    // {-sqrt(2)} union (0, infinity); +sqrt(2) is already in the latter.
    chart(
        &solve(&constraints, &x, &any(&[0, 1])),
        &[0, -2, 0, 1],
        &[false, false, true, true],
        &[true, false, true],
    );
}

#[test]
fn shared_irrational_roots_and_multiplicities_are_deduplicated() {
    let x = Symbol::new("x");
    let q = quadratic(&x, -2);
    let constraints = [
        atom(q.clone(), RealRelation::Le),
        atom(pow(q.clone(), 4), RealRelation::Ge),
        atom(
            Expr::Mul(vec![Expr::from_i64(-7), pow(q, 3)]),
            RealRelation::Ge,
        ),
    ];
    chart(
        &solve(&constraints, &x, &all(&[0, 1, 2])),
        &[-2, 0, 1],
        &[false, true, false],
        &[true, true],
    );
}

#[test]
fn equality_and_disequality_partition_only_the_original_defined_domain() {
    let x = Symbol::new("x");
    let source = Expr::Mul(vec![quadratic(&x, -2), pow(shift(&x, -1), -1)]);
    let constraints = [
        atom(source.clone(), RealRelation::Eq),
        atom(source.clone(), RealRelation::Ne),
    ];
    let equal = solve(&constraints, &x, &RealFormula::Atom(0));
    let unequal = solve(&constraints, &x, &RealFormula::Atom(1));
    chart(
        &equal,
        &[2, -2, -1, 1],
        &[false, false, false, false],
        &[true, false, true],
    );
    chart(
        &unequal,
        &[2, -2, -1, 1],
        &[true, true, true, true],
        &[false, false, false],
    );
    chart(
        &solve(&constraints, &x, &any(&[0, 1])),
        &[2, -2, -1, 1],
        &[true, true, true, true],
        &[true, false, true],
    );
    let impossible = solve(&constraints, &x, &all(&[0, 1]));
    assert!(impossible.gap_inclusions().iter().all(|included| !included));
    assert!(
        impossible
            .point_inclusions()
            .iter()
            .all(|included| !included)
    );
    assert_eq!(equal.constraints()[0].expression, source);
}

#[test]
fn a_defined_disjunct_can_supply_another_atoms_excluded_point() {
    let x = Symbol::new("x");
    let source = Expr::Mul(vec![Expr::from_i64(0), pow(Expr::Sym(x.clone()), -1)]);
    let constraints = [
        atom(source, RealRelation::Eq),
        atom(Expr::Sym(x.clone()), RealRelation::Eq),
    ];
    // Every atom denotes its own partial relation's solution set. Their union
    // is R even though the first expression is undefined at zero.
    chart(
        &solve(&constraints, &x, &any(&[0, 1])),
        &[0, 1],
        &[true, true],
        &[true],
    );
    chart(
        &solve(&constraints, &x, &all(&[0, 1])),
        &[0, 1],
        &[false, false],
        &[false],
    );
}

#[test]
fn held_cancellation_and_nested_zero_powers_keep_irrational_domain_holes() {
    let x = Symbol::new("x");
    let q = quadratic(&x, -2);
    let source = Expr::Mul(vec![pow(q.clone(), 2), pow(q.clone(), -1)]);
    let held_one = pow(pow(q, -1), 0);
    let constraints = [
        atom(source, RealRelation::Ge),
        atom(held_one, RealRelation::Ne),
    ];
    chart(
        &solve(&constraints, &x, &all(&[0, 1])),
        &[-2, 0, 1],
        &[true, false, true],
        &[false, false],
    );
}

#[test]
fn rational_disjunction_excludes_both_poles_exactly() {
    let x = Symbol::new("x");
    let constraints = [
        atom(pow(shift(&x, -1), -1), RealRelation::Gt),
        atom(pow(shift(&x, 1), -1), RealRelation::Lt),
    ];
    chart(
        &solve(&constraints, &x, &any(&[0, 1])),
        &[-1, 0, 1],
        &[true, false, true],
        &[false, false],
    );
}

#[test]
fn equality_intersects_an_inequality_without_adding_spurious_gap_solutions() {
    let x = Symbol::new("x");
    let constraints = [
        atom(quadratic(&x, -2), RealRelation::Eq),
        atom(Expr::Sym(x.clone()), RealRelation::Ge),
    ];
    chart(
        &solve(&constraints, &x, &all(&[0, 1])),
        &[0, -2, 0, 1],
        &[false, false, false, false],
        &[false, false, true],
    );
}

#[test]
fn irreducible_quintic_boundaries_need_no_radical_or_floating_approximation() {
    let x = Symbol::new("x");
    let quintic = Expr::Add(vec![
        pow(Expr::Sym(x.clone()), 5),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(x.clone())]),
        Expr::from_i64(1),
    ]);
    let constraints = [
        atom(quintic, RealRelation::Ge),
        atom(shift(&x, -1), RealRelation::Lt),
    ];
    // The quintic has one real root alpha in (-2,-1); the answer is [alpha,1).
    let result = solve(&constraints, &x, &all(&[0, 1]));
    chart(
        &result,
        &[-1, 2, -1, 0, 0, -1, 1],
        &[false, true, false],
        &[true, false],
    );
    let alpha = &result.roots().intervals()[0];
    assert!(alpha.lower < BigRational::from(-1));
    assert!(alpha.upper > BigRational::from(-2));
}

#[test]
fn close_rational_boundaries_remain_distinct_beyond_binary64_precision() {
    let x = Symbol::new("x");
    let epsilon = BigRational::new(1.into(), BigInt::from(10).pow(40));
    let constraints = [
        atom(Expr::Sym(x.clone()), RealRelation::Gt),
        atom(
            Expr::Add(vec![Expr::Sym(x.clone()), Expr::Rational(-epsilon.clone())]),
            RealRelation::Le,
        ),
    ];
    let result = solve(&constraints, &x, &all(&[0, 1]));
    assert_eq!(
        result.polynomial().coeffs,
        vec![BigRational::from(0), -epsilon, BigRational::from(1)]
    );
    assert_eq!(result.roots().intervals().len(), 2);
    assert_eq!(result.gap_inclusions(), [false, true, false]);
    assert_eq!(result.point_inclusions(), [false, true]);
}

#[test]
fn constant_relations_and_nowhere_defined_atoms_are_distinct() {
    let x = Symbol::new("x");
    for (value, relation, expected) in [
        (0, RealRelation::Eq, true),
        (0, RealRelation::Ne, false),
        (0, RealRelation::Ge, true),
        (0, RealRelation::Le, true),
        (0, RealRelation::Gt, false),
        (0, RealRelation::Lt, false),
        (1, RealRelation::Eq, false),
        (1, RealRelation::Ne, true),
        (1, RealRelation::Gt, true),
        (1, RealRelation::Lt, false),
        (-1, RealRelation::Ge, false),
        (-1, RealRelation::Le, true),
    ] {
        let constraints = [atom(Expr::from_i64(value), relation)];
        chart(
            &solve(&constraints, &x, &RealFormula::Atom(0)),
            &[1],
            &[expected],
            &[],
        );
    }
    for relation in [
        RealRelation::Eq,
        RealRelation::Ne,
        RealRelation::Lt,
        RealRelation::Le,
        RealRelation::Gt,
        RealRelation::Ge,
    ] {
        let undefined = Expr::Mul(vec![Expr::from_i64(0), pow(Expr::from_i64(0), -1)]);
        chart(
            &solve(&[atom(undefined, relation)], &x, &RealFormula::Atom(0)),
            &[1],
            &[false],
            &[],
        );
    }
}

#[test]
fn complex_only_zeros_do_not_create_real_boundaries() {
    let x = Symbol::new("x");
    let constraints = [
        atom(quadratic(&x, 1), RealRelation::Eq),
        atom(quadratic(&x, 1), RealRelation::Ne),
    ];
    let equation = solve(&constraints, &x, &RealFormula::Atom(0));
    let disequation = solve(&constraints, &x, &RealFormula::Atom(1));
    assert!(equation.roots().intervals().is_empty());
    assert!(disequation.roots().intervals().is_empty());
    assert_eq!(equation.gap_inclusions(), [false]);
    assert_eq!(disequation.gap_inclusions(), [true]);
    let constraints = [
        atom(quadratic(&x, 1), RealRelation::Gt),
        atom(shift(&x, -1), RealRelation::Lt),
    ];
    // A nonconstant local critical polynomial may have no real roots while
    // the common polynomial has a boundary contributed by another atom.
    chart(
        &solve(&constraints, &x, &all(&[0, 1])),
        &[-1, 1, -1, 1],
        &[true, false],
        &[false],
    );
}

#[test]
fn empty_boolean_connectives_have_their_mathematical_identity_sets() {
    let x = Symbol::new("x");
    chart(&solve(&[], &x, &all(&[])), &[1], &[true], &[]);
    chart(&solve(&[], &x, &any(&[])), &[1], &[false], &[]);
    assert!(matches!(
        solve_real_constraints(&[], &x, &RealFormula::Atom(0)),
        Err(RealEquationError::InvalidConstraints(_))
    ));
}

#[test]
fn boolean_laws_hold_on_the_same_complete_algebraic_decomposition() {
    let x = Symbol::new("x");
    let constraints = [
        atom(shift(&x, 1), RealRelation::Lt),
        atom(shift(&x, -2), RealRelation::Le),
        atom(Expr::Sym(x.clone()), RealRelation::Ge),
    ];
    let left = RealFormula::All(vec![RealFormula::Atom(0), any(&[1, 2])]);
    let distributed = RealFormula::Any(vec![all(&[0, 1]), all(&[0, 2])]);
    let duplicate = RealFormula::All(vec![left.clone(), left.clone()]);
    let permuted = RealFormula::All(vec![any(&[2, 1]), RealFormula::Atom(0)]);
    for formula in [left, distributed, duplicate, permuted] {
        chart(
            &solve(&constraints, &x, &formula),
            &[0, -2, -1, 1],
            &[true, false, false, false],
            &[false, false, false],
        );
    }
}

#[test]
fn atom_permutation_preserves_the_solution_when_formula_indices_follow() {
    let x = Symbol::new("x");
    let left = atom(quadratic(&x, -2), RealRelation::Ge);
    let right = atom(shift(&x, -1), RealRelation::Lt);
    let first = solve(&[left.clone(), right.clone()], &x, &all(&[0, 1]));
    let second = solve(&[right, left], &x, &all(&[1, 0]));
    assert_eq!(first.polynomial(), second.polynomial());
    assert_eq!(first.gap_inclusions(), second.gap_inclusions());
    assert_eq!(first.point_inclusions(), second.point_inclusions());
}

#[test]
fn every_source_is_admitted_even_if_the_formula_would_short_circuit() {
    let x = Symbol::new("x");
    let foreign = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32]));
    for unsupported in [
        Expr::Sym(foreign),
        Expr::Function("sin".to_owned(), vec![Expr::Sym(x.clone())]),
    ] {
        let constraints = [atom(unsupported, RealRelation::Eq)];
        for formula in [all(&[]), any(&[])] {
            assert_eq!(
                solve_real_constraints(&constraints, &x, &formula).unwrap(),
                None
            );
        }
    }
    let unsupported = Expr::Function("sin".to_owned(), vec![Expr::Sym(x.clone())]);
    let hidden_growth = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Integer(BigInt::one() << 4096u32),
    ]);
    let constraints = [
        atom(unsupported, RealRelation::Eq),
        atom(hidden_growth, RealRelation::Gt),
    ];
    assert!(matches!(
        solve_real_constraints(&constraints, &x, &all(&[])),
        Err(RealEquationError::ResourceLimit(_))
    ));
}

#[test]
fn formula_indices_and_independent_structural_limits_fail_explicitly() {
    let x = Symbol::new("x");
    let constraints = [atom(Expr::Sym(x.clone()), RealRelation::Ge)];
    assert!(matches!(
        solve_real_constraints(&constraints, &x, &RealFormula::Atom(1)),
        Err(RealEquationError::InvalidConstraints(_))
    ));
    assert!(matches!(
        solve_real_constraints(&constraints, &x, &RealFormula::Atom(usize::MAX)),
        Err(RealEquationError::InvalidConstraints(_))
    ));
    let oversized_atoms = vec![constraints[0].clone(); MAX_REAL_CONSTRAINTS + 1];
    assert!(matches!(
        solve_real_constraints(&oversized_atoms, &x, &all(&[])),
        Err(RealEquationError::InvalidConstraints(_))
    ));
    let mut deep = RealFormula::Atom(0);
    for _ in 0..=MAX_REAL_FORMULA_DEPTH {
        deep = RealFormula::All(vec![deep]);
    }
    let broad = RealFormula::All(vec![RealFormula::Atom(0); MAX_REAL_FORMULA_FANOUT + 1]);
    let groups = MAX_REAL_FORMULA_NODES / MAX_REAL_FORMULA_FANOUT + 1;
    let many_nodes = RealFormula::All(vec![
        RealFormula::All(vec![
            RealFormula::Atom(0);
            MAX_REAL_FORMULA_FANOUT
        ]);
        groups
    ]);
    for formula in [deep, broad, many_nodes] {
        assert!(matches!(
            solve_real_constraints(&constraints, &x, &formula),
            Err(RealEquationError::ResourceLimit(_))
        ));
    }
}

#[test]
fn the_combined_critical_degree_is_bounded_even_when_each_atom_fits() {
    let x = Symbol::new("x");
    let constraints = [33, 32].map(|degree| {
        atom(
            Expr::Add(vec![pow(Expr::Sym(x.clone()), degree), Expr::from_i64(1)]),
            RealRelation::Gt,
        )
    });
    // x^33+1 and x^32+1 are coprime, so the common basis has degree 65.
    assert!(matches!(
        solve_real_constraints(&constraints, &x, &all(&[0, 1])),
        Err(RealEquationError::ResourceLimit(_))
    ));
}

fn interrupted(error: &RealEquationError) -> bool {
    matches!(
        error,
        RealEquationError::Meter(MeterError::Cancelled)
            | RealEquationError::Roots(RealRootError::Meter(MeterError::Cancelled))
    )
}

fn exhausted(error: &RealEquationError) -> bool {
    matches!(
        error,
        RealEquationError::Meter(MeterError::Budget(_))
            | RealEquationError::Roots(RealRootError::Meter(MeterError::Budget(_)))
    )
}

#[test]
fn successful_and_failed_calls_preserve_cumulative_caller_accounting() {
    let x = Symbol::new("x");
    let constraints = [
        atom(quadratic(&x, -2), RealRelation::Ge),
        atom(quadratic(&x, -3), RealRelation::Le),
    ];
    let formula = all(&[0, 1]);
    let allowance = 512 * 1024 * 1024;
    let mut generator = Budget::new(BudgetLimits::uniform(allowance, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(allowance, 0));
    for _ in 0..2 {
        let old_generator = generator.remaining(Dimension::ComputeSteps);
        let old_verifier = verifier.remaining(Dimension::ComputeSteps);
        let result = metered_solve_real_constraints(
            &constraints,
            &x,
            &formula,
            &mut generator,
            &mut verifier,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.point_inclusions(), [true; 4]);
        assert!(generator.remaining(Dimension::ComputeSteps) < old_generator);
        assert!(verifier.remaining(Dimension::ComputeSteps) < old_verifier);
    }
    for tiny_generator in [true, false] {
        let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
        let error = if tiny_generator {
            metered_solve_real_constraints(&constraints, &x, &formula, &mut tiny, &mut Unbounded)
        } else {
            metered_solve_real_constraints(&constraints, &x, &formula, &mut Unbounded, &mut tiny)
        }
        .unwrap_err();
        assert!(exhausted(&error), "unexpected error: {error}");
        let snapshot = tiny.snapshot();
        let retry = if tiny_generator {
            metered_solve_real_constraints(&constraints, &x, &formula, &mut tiny, &mut Unbounded)
        } else {
            metered_solve_real_constraints(&constraints, &x, &formula, &mut Unbounded, &mut tiny)
        };
        assert!(retry.is_err());
        assert!(
            tiny.snapshot()
                .dimensions
                .iter()
                .zip(snapshot.dimensions)
                .all(|(&now, before)| now <= before)
        );
    }
}

#[derive(Default)]
struct Interrupt {
    checkpoints: usize,
    cancel_at: usize,
}

impl BudgetMeter for Interrupt {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints == self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_stops_both_lanes_before_during_and_at_publication() {
    let x = Symbol::new("x");
    let constraints = [
        atom(quadratic(&x, -2), RealRelation::Ge),
        atom(shift(&x, -1), RealRelation::Lt),
    ];
    let formula = all(&[0, 1]);
    let mut generator = Interrupt::default();
    let mut verifier = Interrupt::default();
    metered_solve_real_constraints(&constraints, &x, &formula, &mut generator, &mut verifier)
        .unwrap()
        .unwrap();
    for (generator_lane, count) in [(true, generator.checkpoints), (false, verifier.checkpoints)] {
        assert!(count > 2);
        for cancel_at in [1, count / 2, count] {
            let mut meter = Interrupt {
                checkpoints: 0,
                cancel_at,
            };
            let error = if generator_lane {
                metered_solve_real_constraints(
                    &constraints,
                    &x,
                    &formula,
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_solve_real_constraints(
                    &constraints,
                    &x,
                    &formula,
                    &mut Unbounded,
                    &mut meter,
                )
            }
            .unwrap_err();
            assert!(interrupted(&error), "unexpected error: {error}");
        }
    }
}
