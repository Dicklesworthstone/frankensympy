//! Polynomial system solvers and solution certificate verification (WS19).

#![forbid(unsafe_code)]

use crate::SolverError;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::groebner::groebner_basis;
use fsym_polys::multivariate::{MultivariatePoly, TermOrder};
use fsym_polys::univariate::UnivariatePoly;
use fsym_simplify::simplify;
use std::collections::HashMap;

fn evaluate_as_univariate_in_x(poly: &MultivariatePoly, y_root: &Expr) -> Vec<Expr> {
    let mut coefficients = vec![Expr::from_i64(0); (poly.degree_in(0) as usize) + 1];
    for (exponents, coefficient) in &poly.terms {
        let degree_x = exponents[0] as usize;
        let degree_y = exponents[1];
        let coefficient_expr = Expr::Rational(coefficient.clone());
        let term = if degree_y == 0 {
            coefficient_expr
        } else {
            Expr::Mul(vec![
                coefficient_expr,
                Expr::Pow(
                    std::sync::Arc::new(y_root.clone()),
                    std::sync::Arc::new(Expr::from_i64(i64::from(degree_y))),
                ),
            ])
        };
        coefficients[degree_x] = simplify(&Expr::Add(vec![coefficients[degree_x].clone(), term]));
    }
    coefficients
}

/// Solves a triangular or zero-dimensional 2-variable polynomial system using Lexicographical Groebner elimination.
pub fn solve_2var_poly_system(
    eqs: &[MultivariatePoly],
    x: &Symbol,
    y: &Symbol,
) -> Result<Vec<HashMap<Symbol, Expr>>, SolverError> {
    if eqs.is_empty() {
        return Ok(Vec::new());
    }
    if x == y {
        return Err(SolverError::InvalidSystem(
            "the two solver variables must be distinct".to_string(),
        ));
    }
    let expected_generators = [x.clone(), y.clone()];
    for equation in eqs {
        if equation.generators.as_slice() != expected_generators.as_slice() {
            return Err(SolverError::InvalidSystem(format!(
                "every equation must use generators [{}, {}] in that exact order",
                x.name, y.name
            )));
        }
        if equation
            .terms
            .keys()
            .any(|exponents| exponents.len() != expected_generators.len())
        {
            return Err(SolverError::InvalidSystem(
                "an exponent-vector width does not match the two-variable ring".to_string(),
            ));
        }
    }
    // Compute Groebner basis under Lex (x > y)
    let gb = groebner_basis(eqs, TermOrder::Lex)
        .map_err(|error| SolverError::InvalidSystem(error.to_string()))?;

    // Find univariate polynomial in y (degree in x == 0)
    let Some(y_poly_mv) = gb
        .iter()
        .find(|poly| poly.degree_in(0) == 0 && poly.degree_in(1) > 0)
    else {
        return Err(SolverError::UnsupportedDegree(gb.len()));
    };

    // Convert univariate polynomial in y to UnivariatePoly
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
    let x_polys: Vec<&MultivariatePoly> = gb.iter().filter(|p| p.degree_in(0) == 1).collect();
    if x_polys.is_empty() {
        let maximum_x_degree = gb
            .iter()
            .map(|poly| poly.degree_in(0) as usize)
            .max()
            .unwrap_or(0);
        return Err(if maximum_x_degree == 0 {
            SolverError::InfiniteSolutions
        } else {
            SolverError::UnsupportedDegree(maximum_x_degree)
        });
    }

    for y_root in y_roots {
        let mut candidate = None;
        let mut root_is_impossible = false;
        for x_poly in &x_polys {
            let x_uni_coeffs = evaluate_as_univariate_in_x(x_poly, &y_root);
            let c0 = &x_uni_coeffs[0];
            let c1 = &x_uni_coeffs[1];
            if c1.is_zero() {
                if c0.is_zero() {
                    continue;
                }
                root_is_impossible = true;
                break;
            }
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
            candidate = Some(x_root);
            break;
        }

        if root_is_impossible {
            continue;
        }

        let Some(x_root) = candidate else {
            return Err(SolverError::IncompleteSolutionSet(format!(
                "no linear back-substitution relation isolates {} at y = {}",
                x.name, y_root
            )));
        };
        let mut solution = HashMap::new();
        solution.insert(x.clone(), x_root);
        solution.insert(y.clone(), y_root);

        let mut verified = true;
        for equation in eqs {
            let expression = equation
                .to_expr()
                .map_err(|error| SolverError::InvalidSystem(error.to_string()))?;
            if !simplify(&expression.subs(&solution)).is_zero() {
                verified = false;
                break;
            }
        }
        if !verified {
            return Err(SolverError::IncompleteSolutionSet(
                "a generated candidate did not satisfy every input equation".to_string(),
            ));
        }
        solutions.push(solution);
    }

    if solutions.is_empty() {
        Err(SolverError::NoSolution)
    } else {
        Ok(solutions)
    }
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
