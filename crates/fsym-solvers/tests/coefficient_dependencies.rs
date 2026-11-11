//! Polynomial coefficient extraction must retain the complete typed variable dependency.

use fsym_core::{Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::{SolverError, quadratic_coeffs, solve, solve_linear, solve_quadratic};
use std::collections::HashMap;
use std::sync::Arc;

fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}

#[test]
fn variable_exponents_are_not_polynomial_constants() {
    let x = Symbol::new("x");
    for source in ["2**x", "2**(x+1)", "2**sin(x)", "3**(2**x)", "a**(x+b)"] {
        let expression = parse(source).unwrap();
        assert_eq!(
            solve_linear(&expression, &x),
            Err(SolverError::NonLinear),
            "linear source {source}",
        );
        assert_eq!(
            quadratic_coeffs(&expression, &x),
            Err(SolverError::NonLinear),
            "quadratic source {source}",
        );
        assert_eq!(
            solve_quadratic(&expression, &x),
            Err(SolverError::NonLinear),
            "quadratic solver source {source}",
        );
    }
}

#[test]
fn solver_fallback_does_not_publish_target_dependent_roots() {
    let x = Symbol::new("x");
    for source in [
        "x+2**x",
        "x**2+2**x",
        "x+3**(2**x)",
        "x**2+2**sin(x)",
        "x+2**x+1",
    ] {
        let expression = parse(source).unwrap();
        assert_eq!(
            solve(&expression, &x),
            Err(SolverError::NonLinear),
            "source {source}",
        );
    }
}

#[test]
fn supported_exponential_inversion_still_uses_the_nonpolynomial_lane() {
    let x = Symbol::new("x");
    for (source, root) in [
        ("2**x-8", 3),
        ("2**(2*x+1)-8", 1),
        ("3**(x+1)-9", 1),
        ("2**x-1/4", -2),
    ] {
        assert_eq!(
            solve(&parse(source).unwrap(), &x),
            Ok(vec![Expr::from_i64(root)]),
            "source {source}",
        );
    }
}

#[test]
fn powers_inside_products_cannot_smuggle_variable_coefficients() {
    let x = Symbol::new("x");
    for source in ["2**x*x+1", "(2**x+1)*x+3", "x*(x+2**x)", "2**x*x**2+1"] {
        let expression = parse(source).unwrap();
        assert_eq!(
            quadratic_coeffs(&expression, &x),
            Err(SolverError::NonLinear),
            "source {source}",
        );
        assert_eq!(
            solve_linear(&expression, &x),
            Err(SolverError::NonLinear),
            "linear source {source}",
        );
    }
}

#[test]
fn every_function_argument_in_an_exponent_retains_its_dependency() {
    let x = Symbol::new("x");
    let parameter = Expr::symbol("a");
    for arguments in [
        vec![Expr::Sym(x.clone()), parameter.clone()],
        vec![parameter.clone(), Expr::Sym(x.clone())],
        vec![
            parameter.clone(),
            power(Expr::from_i64(2), Expr::Sym(x.clone())),
        ],
    ] {
        let expression = power(
            Expr::from_i64(2),
            Expr::Function("custom".into(), arguments.clone()),
        );
        assert_eq!(solve_linear(&expression, &x), Err(SolverError::NonLinear));
        assert_eq!(
            quadratic_coeffs(&expression, &x),
            Err(SolverError::NonLinear)
        );
        // The direct function lane must agree with the enclosing-power lane.
        let function = Expr::Function("custom".into(), arguments);
        assert_eq!(solve_linear(&function, &x), Err(SolverError::NonLinear));
        assert_eq!(quadratic_coeffs(&function, &x), Err(SolverError::NonLinear));
    }
}

#[test]
fn held_zero_factors_do_not_erase_an_unsupported_exponent() {
    let x = Symbol::new("x");
    let dependent = power(Expr::from_i64(0), Expr::Sym(x.clone()));
    let held = Expr::Mul(vec![Expr::from_i64(0), dependent]);
    let expression = Expr::Add(vec![Expr::Sym(x.clone()), held]);
    assert_eq!(solve_linear(&expression, &x), Err(SolverError::NonLinear));
    assert_eq!(
        quadratic_coeffs(&expression, &x),
        Err(SolverError::NonLinear)
    );
}

#[test]
fn parameter_only_powers_remain_valid_affine_coefficients() {
    let x = Symbol::new("x");
    let expression = parse("2**a*x+3").unwrap();
    let root = solve_linear(&expression, &x).unwrap();
    assert!(!root.free_symbols().contains(&x));
    assert!(root.free_symbols().contains(&Symbol::new("a")));
    let specialized = root.subs(&HashMap::from([(Symbol::new("a"), Expr::from_i64(3))]));
    let residual = Expr::from_i64(8) * specialized + Expr::from_i64(3);
    assert!(fsym_simplify::try_simplify(&residual).unwrap().is_zero());
    assert_eq!(
        quadratic_coeffs(&expression, &x).unwrap(),
        (Expr::from_i64(0), parse("2**a").unwrap(), Expr::from_i64(3)),
    );
}

#[test]
fn parameter_only_powers_remain_valid_quadratic_coefficients() {
    let x = Symbol::new("x");
    let expression = parse("x**2-2**a").unwrap();
    let roots = solve_quadratic(&expression, &x).unwrap();
    assert_eq!(roots.len(), 2);
    for root in roots {
        assert!(!root.free_symbols().contains(&x));
        let specialized = root.subs(&HashMap::from([(Symbol::new("a"), Expr::from_i64(2))]));
        let residual = specialized.clone() * specialized - Expr::from_i64(4);
        assert!(fsym_simplify::try_simplify(&residual).unwrap().is_zero());
    }
}

#[test]
fn same_name_parameter_and_target_have_distinct_typed_dependencies() {
    let target = Symbol::new("x");
    let parameter = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([19; 32]));
    let coefficient = power(Expr::from_i64(2), Expr::Sym(parameter.clone()));
    let expression = Expr::Add(vec![Expr::Sym(target.clone()), coefficient.clone()]);
    let root = solve_linear(&expression, &target).unwrap();
    assert!(!root.free_symbols().contains(&target));
    assert!(root.free_symbols().contains(&parameter));
    assert_eq!(
        quadratic_coeffs(&expression, &target).unwrap(),
        (Expr::from_i64(0), Expr::from_i64(1), coefficient),
    );
    assert_eq!(
        solve_linear(&expression, &parameter),
        Err(SolverError::NonLinear)
    );
    assert_eq!(
        quadratic_coeffs(&expression, &parameter),
        Err(SolverError::NonLinear)
    );
}
