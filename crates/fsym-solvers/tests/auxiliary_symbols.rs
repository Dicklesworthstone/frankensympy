//! Internal elimination atoms must not alias user coordinates or parameters.

use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::{SolverError, solve, univariate::solve_univariate};
use std::collections::HashMap;

fn renamed(source: &str, coordinate: &Symbol) -> Expr {
    parse(source).unwrap().subs(&HashMap::from([(
        Symbol::new("x"),
        Expr::Sym(coordinate.clone()),
    )]))
}

#[test]
fn transcendental_generator_solving_is_invariant_under_coordinate_renaming() {
    for name in ["x", "_fsym_solve_u", "_fsym_solve_u_0", "_fsym_rad0"] {
        let coordinate = Symbol::new(name);
        for source in ["exp(2*x)-3*exp(x)+2", "log(x)**2-log(x)"] {
            let ordinary = parse(source).unwrap();
            let expected = solve_univariate(&ordinary, &Symbol::new("x")).unwrap();
            assert!(!expected.is_empty());
            let equation = renamed(source, &coordinate);
            assert_eq!(
                solve_univariate(&equation, &coordinate),
                Ok(expected.clone()),
                "{name}: {source}"
            );
            assert_eq!(
                solve(&equation, &coordinate),
                Ok(expected),
                "public {name}: {source}"
            );
        }
    }
}

#[test]
fn radical_elimination_preserves_original_branch_checks_for_reserved_names() {
    for name in ["x", "_fsym_rad0", "_fsym_rad1", "_fsym_rad0_0"] {
        let coordinate = Symbol::new(name);
        let equation = renamed("sqrt(x+1)-x+1", &coordinate);
        let roots = solve_univariate(&equation, &coordinate).unwrap();
        assert_eq!(roots, vec![Expr::from_i64(3)], "coordinate {name}");
        assert_eq!(solve(&equation, &coordinate), Ok(roots.clone()));
        for root in roots {
            let residual = equation.subs(&HashMap::from([(coordinate.clone(), root)]));
            assert!(fsym_simplify::try_simplify(&residual).unwrap().is_zero());
        }
    }
}

#[test]
fn same_printed_name_with_typed_identity_keeps_the_same_solutions() {
    for name in ["_fsym_solve_u", "_fsym_rad0"] {
        let coordinate =
            Symbol::with_identity(name, SymbolIdentity::from_assumptions_digest([73; 32]));
        for source in ["exp(2*x)-3*exp(x)+2", "sqrt(x+1)-x+1"] {
            let expected = solve_univariate(&parse(source).unwrap(), &Symbol::new("x")).unwrap();
            assert_eq!(
                solve_univariate(&renamed(source, &coordinate), &coordinate),
                Ok(expected)
            );
        }
    }
}

#[test]
fn a_user_parameter_cannot_cancel_the_transcendental_generator() {
    let coordinate = Symbol::new("x");
    // With a colliding auxiliary, log(x)^2 - a*log(x) incorrectly becomes
    // u^2-u^2 and claims infinitely many solutions. Symbolic coefficients
    // may remain unsupported, but this is not the identically zero equation.
    for source in [
        "log(x)**2-_fsym_solve_u*log(x)",
        "log(x)**2-(_fsym_solve_u+_fsym_solve_u_0)*log(x)",
    ] {
        let expression = parse(source).unwrap();
        for result in [
            solve_univariate(&expression, &coordinate),
            solve(&expression, &coordinate),
        ] {
            assert_ne!(result, Err(SolverError::InfiniteSolutions), "{source}");
            if let Ok(roots) = result {
                for root in roots {
                    assert!(!root.free_symbols().contains(&coordinate));
                }
            }
        }
    }
}
