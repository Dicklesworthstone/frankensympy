//! Polynomial system solvers and solution certificate verification (WS19).

#![forbid(unsafe_code)]

use crate::SolverError;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::groebner::groebner_basis;
use fsym_polys::multivariate::{MultivariatePoly, TermOrder};
use fsym_polys::univariate::UnivariatePoly;
use fsym_simplify::simplify;
use std::collections::HashMap;

/// Solves a triangular or zero-dimensional 2-variable polynomial system using Lexicographical Groebner elimination.
pub fn solve_2var_poly_system(
    eqs: &[MultivariatePoly],
    x: &Symbol,
    y: &Symbol,
) -> Result<Vec<HashMap<Symbol, Expr>>, SolverError> {
    if eqs.is_empty() {
        return Ok(Vec::new());
    }
    // Compute Groebner basis under Lex (x > y)
    let gb = groebner_basis(eqs, TermOrder::Lex).map_err(|_e| SolverError::NonLinear)?;

    // Find univariate polynomial in y (degree in x == 0)
    let y_polys: Vec<&MultivariatePoly> = gb.iter().filter(|p| p.degree_in(0) == 0).collect();
    if y_polys.is_empty() {
        return Err(SolverError::UnsupportedDegree(gb.len()));
    }

    // Convert univariate polynomial in y to UnivariatePoly
    let y_poly_mv = y_polys[0];
    let max_deg_y = y_poly_mv.degree_in(1) as usize;
    let mut y_coeffs = vec![BigRational::from_integer(0.into()); max_deg_y + 1];
    for (exp, coeff) in &y_poly_mv.terms {
        let deg = exp[1] as usize;
        y_coeffs[deg] = coeff.clone();
    }

    let uni_y = UnivariatePoly::new(y.clone(), y_coeffs);
    let y_roots = crate::solve_poly(&uni_y)?;

    let mut solutions = Vec::new();
    // For each y root, back-substitute to find x roots
    let x_polys: Vec<&MultivariatePoly> = gb.iter().filter(|p| p.degree_in(0) > 0).collect();
    if x_polys.is_empty() {
        return Err(SolverError::InfiniteSolutions);
    }
    let x_poly_mv = x_polys[0];

    for y_root in y_roots {
        // Evaluate x_poly_mv at y = y_root
        // x_poly is linear in x: c1(y) * x + c0(y) = 0 => x = -c0(y) / c1(y)
        let mut x_uni_coeffs = vec![Expr::from_i64(0); (x_poly_mv.degree_in(0) as usize) + 1];
        for (exp, coeff) in &x_poly_mv.terms {
            let deg_x = exp[0] as usize;
            let deg_y = exp[1] as usize;

            let coeff_expr = Expr::Rational(coeff.clone());
            let y_term = if deg_y == 0 {
                coeff_expr
            } else {
                Expr::Mul(vec![
                    coeff_expr,
                    Expr::Pow(
                        std::sync::Arc::new(y_root.clone()),
                        std::sync::Arc::new(Expr::from_i64(deg_y as i64)),
                    ),
                ])
            };
            x_uni_coeffs[deg_x] = simplify(&Expr::Add(vec![x_uni_coeffs[deg_x].clone(), y_term]));
        }

        if x_uni_coeffs.len() == 2 {
            let c0 = &x_uni_coeffs[0];
            let c1 = &x_uni_coeffs[1];
            let x_root = if c1 == &Expr::from_i64(1) {
                simplify(&Expr::Mul(vec![Expr::from_i64(-1), c0.clone()]))
            } else {
                simplify(&Expr::Mul(vec![
                    Expr::from_i64(-1),
                    c0.clone(),
                    Expr::Pow(
                        std::sync::Arc::new(c1.clone()),
                        std::sync::Arc::new(Expr::from_i64(-1)),
                    ),
                ]))
            };

            let mut sol = HashMap::new();
            sol.insert(x.clone(), x_root);
            sol.insert(y.clone(), y_root);
            solutions.push(sol);
        }
    }

    Ok(solutions)
}

/// Independent verifier checking that candidate solution satisfies all equations in the polynomial system.
pub fn verify_poly_system_solution(eqs: &[Expr], solution: &HashMap<Symbol, Expr>) -> bool {
    for eq in eqs {
        let evaluated = eq.subs(solution);
        let simplified = simplify(&evaluated);
        if !simplified.is_zero() {
            return false;
        }
    }
    true
}
