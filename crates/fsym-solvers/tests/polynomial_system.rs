//! Hand-derived complete finite systems and original-source adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::SolverError;
use fsym_solvers::polysys::{metered_solve_polynomial_system, solve_polynomial_system};
use std::collections::HashSet;
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
fn rational(numerator: i64, denominator: i64) -> Expr {
    Expr::Rational(BigRational::new(numerator.into(), denominator.into()))
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn solve(sources: &[&str], names: &[&str]) -> Vec<Vec<Expr>> {
    solve_polynomial_system(&equations(sources), &variables(names)).unwrap()
}

#[test]
fn rational_overdetermined_three_variable_system_preserves_all_constraints() {
    let result = solve(
        &["2*x/3 + y - 1/3", "4*x - 2*y", "z - x - y", "8*z - 3"],
        &["x", "y", "z"],
    );
    assert_eq!(
        result,
        [vec![rational(1, 8), rational(1, 4), rational(3, 8)]]
    );
}

#[test]
fn cartesian_quadratic_system_returns_every_combination() {
    let result = solve(&["x**2 - 4", "y**2 - 9"], &["x", "y"]);
    let expected: HashSet<_> = [-2, 2]
        .into_iter()
        .flat_map(|x| {
            [-3, 3]
                .into_iter()
                .map(move |y| vec![Expr::from_i64(x), Expr::from_i64(y)])
        })
        .collect();
    assert_eq!(result.len(), expected.len());
    assert_eq!(result.into_iter().collect::<HashSet<_>>(), expected);
}

#[test]
fn specialization_keeps_branches_where_a_leading_coefficient_vanishes() {
    // y=0 permits both x=-1 and x=0; y=1 permits only x=0.
    let result = solve(&["x*y", "x**2 + x", "y**2 - y"], &["x", "y"]);
    let expected: HashSet<_> = [(-1, 0), (0, 0), (0, 1)]
        .into_iter()
        .map(|(x, y)| vec![Expr::from_i64(x), Expr::from_i64(y)])
        .collect();
    assert_eq!(result.len(), 3);
    assert_eq!(result.into_iter().collect::<HashSet<_>>(), expected);
}

#[test]
fn real_radical_and_complex_roots_remain_complete() {
    let diagonal = solve(&["x-y", "y**2 - 2"], &["x", "y"]);
    assert_eq!(diagonal.len(), 2);
    assert_ne!(diagonal[0], diagonal[1]);
    for tuple in diagonal {
        assert_eq!(tuple[0], tuple[1]);
        let squared = tuple[0].clone() * tuple[0].clone() - Expr::from_i64(2);
        assert!(fsym_simplify::try_simplify(&squared).unwrap().is_zero());
    }
    let complex = solve(&["x**2 + 1", "y-x"], &["x", "y"]);
    let i = Expr::Const(Constant::I);
    let expected = HashSet::from([vec![i.clone(), i.clone()], vec![-i.clone(), -i]]);
    assert_eq!(complex.len(), 2);
    assert_eq!(complex.into_iter().collect::<HashSet<_>>(), expected);
}

#[test]
fn univariate_constraints_intersect_algebraically_and_repeated_roots_are_distinct() {
    let roots = solve(&["x**2-2", "(x**2-2)*(x+1)"], &["x"]);
    assert_eq!(roots.len(), 2);
    for root in roots {
        assert_eq!(root.len(), 1);
        let residual = root[0].clone() * root[0].clone() - Expr::from_i64(2);
        assert!(fsym_simplify::try_simplify(&residual).unwrap().is_zero());
    }
    assert_eq!(
        solve(&["(x-1)**2", "(y+2)**3"], &["x", "y"]),
        [vec![Expr::from_i64(1), Expr::from_i64(-2)]]
    );
}

#[test]
fn exact_tiny_inconsistency_does_not_become_a_numerical_zero() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let epsilon = Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(80)));
    let input = vec![
        xe.clone() - Expr::from_i64(1),
        xe - Expr::from_i64(1) - epsilon,
    ];
    assert!(solve_polynomial_system(&input, &[x]).unwrap().is_empty());
}

#[test]
fn unsupported_or_infinite_systems_never_publish_an_empty_solution_set() {
    let vars = variables(&["x", "y"]);
    for input in [equations(&["x*y"]), vec![Expr::from_i64(0)], vec![]] {
        assert_eq!(
            solve_polynomial_system(&input, &vars),
            Err(SolverError::InfiniteSolutions)
        );
    }
    assert!(solve_polynomial_system(&equations(&["x**5-x+1", "y-x"]), &vars).is_err());
    assert!(
        solve_polynomial_system(&equations(&["x+y-1", "x+y-2"]), &vars)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn held_undefined_and_opaque_children_cannot_disappear_through_zero_products() {
    let vars = variables(&["x", "y"]);
    for hidden in [
        power(Expr::from_i64(0), 0),
        power(Expr::from_i64(0), -1),
        Expr::Function("opaque".into(), vec![Expr::from_i64(1)]),
    ] {
        let original = Expr::Add(vec![
            Expr::Sym(vars[0].clone()),
            Expr::Mul(vec![Expr::from_i64(0), hidden]),
        ]);
        assert!(solve_polynomial_system(&[original.clone()], &vars[..1]).is_err());
        assert!(solve_polynomial_system(&[original, Expr::Sym(vars[1].clone())], &vars).is_err());
    }
    let cancelled_inverse = Expr::Mul(vec![
        Expr::Sym(vars[0].clone()),
        power(Expr::Sym(vars[0].clone()), -1),
    ]);
    assert!(solve_polynomial_system(&[cancelled_inverse], &vars[..1]).is_err());
}

#[test]
fn same_spelling_variables_are_distinguished_by_typed_identity() {
    let plain = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([37; 32]));
    let input = [
        Expr::Sym(plain.clone()) - Expr::from_i64(1),
        Expr::Sym(other.clone()) - Expr::from_i64(2),
    ];
    assert_eq!(
        solve_polynomial_system(&input, &[plain.clone(), other]).unwrap(),
        [vec![Expr::from_i64(1), Expr::from_i64(2)]]
    );
    assert!(solve_polynomial_system(&input, &[plain.clone()]).is_err());
    assert!(solve_polynomial_system(&input, &[plain.clone(), plain]).is_err());
}

#[test]
fn oversized_sources_and_dimensions_refuse_before_symbolic_expansion() {
    let x = Symbol::new("x");
    for source in [
        power(Expr::Sym(x.clone()), 257),
        power(Expr::from_i64(2), i64::MAX),
        Expr::Integer(BigInt::from(1) << 4097),
    ] {
        assert!(solve_polynomial_system(&[source], &[x.clone()]).is_err());
    }
    assert!(solve_polynomial_system(&vec![Expr::Sym(x.clone()); 65], &[x.clone()]).is_err());
    let too_many_vars: Vec<_> = (0..9).map(|i| Symbol::new(format!("x{i}"))).collect();
    assert!(solve_polynomial_system(&[Expr::from_i64(1)], &too_many_vars).is_err());
    assert!(solve_polynomial_system(&[Expr::from_i64(1)], &[]).is_err());
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
fn separate_cumulative_meters_cancel_before_during_and_at_final_publication() {
    let input = equations(&["x**2-4", "y-x"]);
    let vars = variables(&["x", "y"]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    let result =
        metered_solve_polynomial_system(&input, &vars, &mut generator, &mut verifier).unwrap();
    assert_eq!(result.len(), 2);
    assert!(generator.compute > 0 && generator.memory > 0);
    assert!(verifier.compute > 0 && verifier.memory > 0);
    for point in [1, generator.checkpoints / 2, generator.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_solve_polynomial_system(&input, &vars, &mut stopped, &mut Unbounded).is_err()
        );
    }
    for point in [1, verifier.checkpoints / 2, verifier.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_solve_polynomial_system(&input, &vars, &mut Unbounded, &mut stopped).is_err()
        );
    }
}

#[test]
fn actual_budget_exhaustion_in_either_phase_is_explicit() {
    let input = equations(&["x-1", "y-2"]);
    let vars = variables(&["x", "y"]);
    let mut generation = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0));
    let mut verification = Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0));
    assert_eq!(
        metered_solve_polynomial_system(&input, &vars, &mut generation, &mut verification).unwrap(),
        [vec![Expr::from_i64(1), Expr::from_i64(2)]]
    );
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(metered_solve_polynomial_system(&input, &vars, &mut tiny, &mut untouched).is_err());
    assert_eq!(untouched.checkpoints, 0);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(metered_solve_polynomial_system(&input, &vars, &mut Unbounded, &mut tiny).is_err());
}
