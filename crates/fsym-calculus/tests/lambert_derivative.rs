use fsym_budget::Unbounded;
use fsym_calculus::diff;
use fsym_core::{Constant, Expr, Symbol};
use fsym_polys::metered_rational_source;
use fsym_simplify::simplify;
use std::collections::HashMap;

#[test]
fn both_real_branches_obey_the_differentiated_defining_identity() {
    let x = Symbol::new("lambert_derivative_x");
    let sx = Expr::Sym(x.clone());
    let argument = Expr::from_i64(3) * sx.clone() + Expr::from_i64(2);
    for branch in [None, Some(-1), Some(2)] {
        let mut args = vec![argument.clone()];
        if let Some(branch) = branch {
            args.push(Expr::from_i64(branch));
        }
        let w = Expr::Function("LambertW".into(), args);
        let derivative = diff(&w, &x);
        // Implicit differentiation of w*exp(w)=3*x+2, after eliminating
        // exp(w), requires u*(w+1)*w' - 3*w = 0 on its smooth domain.
        let formal_w = Symbol::new("formal_lambert_value");
        for (candidate, expected_zero) in [
            (derivative.clone(), true),
            (derivative + Expr::rational(1, 1_000_000).unwrap(), false),
        ] {
            let residual = argument.clone() * (w.clone() + Expr::from_i64(1)) * candidate
                - Expr::from_i64(3) * w.clone();
            // Treat the complete fixed-branch W application as a fresh
            // indeterminate. Independent rational source replay then checks
            // the exact numerator; no simplify heuristic decides acceptance.
            let formal = residual.subs_expr(&w, &Expr::Sym(formal_w.clone()));
            let checked = metered_rational_source(
                &formal,
                &[x.clone(), formal_w.clone()],
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap();
            assert_eq!(checked.fraction().numerator.is_zero(), expected_zero);
        }
    }
}

#[test]
fn principal_derivative_at_e_has_the_exact_inverse_function_value() {
    let x = Symbol::new("lambert_e_x");
    let w = Expr::Function("LambertW".into(), vec![Expr::Sym(x.clone())]);
    let derivative = diff(&w, &x).subs(&HashMap::from([(x, Expr::Const(Constant::E))]));
    let normalized = derivative * Expr::from_i64(2) * Expr::Const(Constant::E);
    assert_eq!(simplify(&normalized), Expr::from_i64(1));
}

#[test]
fn fixed_branch_does_not_create_a_dependency_on_an_unrelated_variable() {
    let x = Symbol::new("lambert_free_x");
    let y = Symbol::new("lambert_free_y");
    let w = Expr::Function("LambertW".into(), vec![Expr::Sym(x), Expr::from_i64(-1)]);
    assert_eq!(diff(&w, &y), Expr::from_i64(0));
}

#[test]
fn symbolic_branch_parameter_is_not_silently_frozen() {
    let x = Symbol::new("lambert_variable_branch_x");
    let sx = Expr::Sym(x.clone());
    let w = Expr::Function("LambertW".into(), vec![sx.clone(), sx]);
    let derivative = diff(&w, &x);
    // Such a source is outside the fixed-integer-branch rule; retaining a
    // formal derivative must not reuse the single-argument formula.
    assert!(matches!(derivative,Expr::Function(ref name,_) if name=="diff"));
}
