//! # fsym-solvers
//!
//! Algebraic equation solvers (`solve`, `solveset`), linear systems, polynomial systems,
//! and differential equations (`dsolve`).

pub mod ode;
pub mod system;

pub use ode::*;
pub use system::*;

use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use num_traits::identities::{One, Zero};
use std::ops::{Add, Mul};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SolverError {
    #[error("No solution found for equation")]
    NoSolution,
    #[error("Infinite solutions for underdetermined system")]
    InfiniteSolutions,
    #[error("Non-linear equation degree {0} not supported by exact solver")]
    UnsupportedDegree(usize),
    #[error("Expression is non-linear in the target variable")]
    NonLinear,
    #[error("Invalid polynomial system: {0}")]
    InvalidSystem(String),
    #[error("The available exact solver cannot establish a complete solution set: {0}")]
    IncompleteSolutionSet(String),
}

/// Solve a univariate polynomial equation `poly(x) = 0`.
pub fn solve_poly(poly: &UnivariatePoly) -> Result<Vec<Expr>, SolverError> {
    if poly.is_zero() {
        return Err(SolverError::InfiniteSolutions);
    }
    match poly.degree() {
        None => Err(SolverError::InfiniteSolutions),
        Some(0) => Err(SolverError::NoSolution), // c = 0 with c != 0
        Some(1) => {
            // c0 + c1 * x = 0 => x = -c0 / c1
            let c0 = &poly.coeffs[0];
            let c1 = &poly.coeffs[1];
            let root = -c0 / c1;
            let expr = if root.is_integer() {
                Expr::Integer(root.to_integer())
            } else {
                Expr::Rational(root)
            };
            Ok(vec![expr])
        }
        Some(2) => {
            // Quadratic equation: c0 + c1*x + c2*x^2 = 0
            // x = (-c1 ± sqrt(c1^2 - 4*c0*c2)) / (2*c2)
            let c0 = &poly.coeffs[0];
            let c1 = &poly.coeffs[1];
            let c2 = &poly.coeffs[2];
            let disc = c1 * c1 - BigRational::from_integer(4.into()) * c0 * c2;
            let neg_b = Expr::Rational(-c1.clone());
            let two_a = Expr::Rational(c2 * BigRational::from_integer(2.into()));
            let disc_expr = Expr::Rational(disc);
            let sqrt_disc = Expr::Pow(
                std::sync::Arc::new(disc_expr),
                std::sync::Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
            );
            let r1 = Expr::Mul(vec![
                Expr::Add(vec![neg_b.clone(), sqrt_disc.clone()]),
                Expr::Pow(
                    std::sync::Arc::new(two_a.clone()),
                    std::sync::Arc::new(Expr::from_i64(-1)),
                ),
            ]);
            let r2 = Expr::Mul(vec![
                Expr::Add(vec![neg_b, Expr::Mul(vec![Expr::from_i64(-1), sqrt_disc])]),
                Expr::Pow(
                    std::sync::Arc::new(two_a),
                    std::sync::Arc::new(Expr::from_i64(-1)),
                ),
            ]);
            Ok(vec![r1, r2])
        }
        Some(d) => {
            if let Ok(factorization) = fsym_polys::factor_polynomial(poly) {
                if factorization.factors.len() > 1
                    || factorization
                        .factors
                        .iter()
                        .any(|f| f.poly.degree().unwrap_or(0) < d)
                {
                    let mut all_roots = Vec::new();
                    for factor in &factorization.factors {
                        let factor_roots = solve_poly(&factor.poly)?;
                        for r in factor_roots {
                            if !all_roots.contains(&r) {
                                all_roots.push(r);
                            }
                        }
                    }
                    if !all_roots.is_empty() {
                        return Ok(all_roots);
                    }
                }
            }
            Err(SolverError::UnsupportedDegree(d))
        }
    }
}

/// Interpret `expr` as a polynomial of degree <= 1 in `var`, returning `(a, b)`
/// with `expr = a*var + b`. Numeric leaves are folded eagerly so common cases
/// stay canonical.
fn linear_coeffs(expr: &Expr, var: &Symbol) -> Result<(Expr, Expr), SolverError> {
    fn as_rational(e: &Expr) -> Option<BigRational> {
        match e {
            Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
            Expr::Rational(r) => Some(r.clone()),
            _ => None,
        }
    }

    fn fold(e: Expr) -> Expr {
        if let Some(r) = as_rational(&e) {
            if r.is_integer() {
                return Expr::Integer(r.to_integer());
            }
            return Expr::Rational(r);
        }
        e
    }

    /// Add two expressions, folding numeric pairs and dropping zero identities.
    fn add(x: Expr, y: Expr) -> Expr {
        match (as_rational(&x), as_rational(&y)) {
            (Some(a), Some(b)) => fold(Expr::Rational(a + b)),
            (Some(a), None) => {
                if a.is_zero() {
                    fold(y)
                } else {
                    x.add(y)
                }
            }
            (None, Some(b)) => {
                if b.is_zero() {
                    fold(x)
                } else {
                    x.add(y)
                }
            }
            (None, None) => x.add(y),
        }
    }

    /// Multiply two expressions, folding numeric pairs and absorbing zero/one identities.
    fn mul(x: Expr, y: Expr) -> Expr {
        match (as_rational(&x), as_rational(&y)) {
            (Some(a), Some(b)) => fold(Expr::Rational(a * b)),
            (Some(a), None) => {
                if a.is_zero() {
                    Expr::from_i64(0)
                } else if a.is_one() {
                    fold(y)
                } else {
                    x.mul(y)
                }
            }
            (None, Some(b)) => {
                if b.is_zero() {
                    Expr::from_i64(0)
                } else if b.is_one() {
                    fold(x)
                } else {
                    x.mul(y)
                }
            }
            (None, None) => x.mul(y),
        }
    }

    let contains_var = |e: &Expr| e.free_symbols().iter().any(|s| s == var);

    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {
            Ok((Expr::from_i64(0), expr.clone()))
        }
        Expr::Sym(s) if s == var => Ok((Expr::from_i64(1), Expr::from_i64(0))),
        Expr::Sym(_) => Ok((Expr::from_i64(0), expr.clone())),
        Expr::Add(terms) => {
            let mut acc = (Expr::from_i64(0), Expr::from_i64(0));
            for t in terms {
                let (a, b) = linear_coeffs(t, var)?;
                acc.0 = add(acc.0, a);
                acc.1 = add(acc.1, b);
            }
            Ok(acc)
        }
        Expr::Mul(factors) => {
            // At most one factor may contain `var`; the rest form a constant multiplier.
            let mut var_part: Option<(Expr, Expr)> = None;
            let mut constants: Vec<Expr> = Vec::new();
            for f in factors {
                if !contains_var(f) {
                    constants.push(f.clone());
                    continue;
                }
                if var_part.is_some() {
                    return Err(SolverError::NonLinear);
                }
                var_part = Some(linear_coeffs(f, var)?);
            }
            let k = constants
                .into_iter()
                .reduce(mul)
                .unwrap_or_else(|| Expr::from_i64(1));
            match var_part {
                None => Ok((Expr::from_i64(0), expr.clone())),
                Some((a, b)) => Ok((mul(k.clone(), a), mul(k, b))),
            }
        }
        Expr::Pow(base, exp) => {
            if contains_var(base) {
                // Linear only through an exponent-1 passthrough: (expr)^1.
                if exp.as_ref() == &Expr::from_i64(1) {
                    return linear_coeffs(base, var);
                }
                return Err(SolverError::NonLinear);
            }
            Ok((Expr::from_i64(0), expr.clone()))
        }
        Expr::Function(_, args) => {
            if args.iter().any(contains_var) {
                return Err(SolverError::NonLinear);
            }
            Ok((Expr::from_i64(0), expr.clone()))
        }
    }
}

/// Solve `expr = 0` for `var`, where `expr` must be linear (`a*var + b`) in `var`.
///
/// Returns `x = -b/a`: folded to an exact `Integer`/`Rational` when both
/// coefficients are numeric, otherwise left structural (`(-1) * b * a^-1`).
pub fn solve_linear(expr: &Expr, var: &Symbol) -> Result<Expr, SolverError> {
    let free = expr.free_symbols();
    if !free.contains(var) {
        if expr.is_zero() {
            return Err(SolverError::InfiniteSolutions);
        } else {
            return Err(SolverError::NoSolution);
        }
    }
    let (a, b) = linear_coeffs(expr, var)?;
    if a.is_zero() {
        return if b.is_zero() {
            Err(SolverError::InfiniteSolutions)
        } else {
            Err(SolverError::NoSolution)
        };
    }
    // Zero constant term: root is exactly 0 regardless of the coefficient's shape.
    if b.is_zero() {
        return Ok(Expr::from_i64(0));
    }
    // Fold when both coefficients are numeric.
    fn as_rational(e: &Expr) -> Option<BigRational> {
        match e {
            Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
            Expr::Rational(r) => Some(r.clone()),
            _ => None,
        }
    }
    if let (Some(ra), Some(rb)) = (as_rational(&a), as_rational(&b)) {
        let root = -rb / ra;
        return Ok(if root.is_integer() {
            Expr::Integer(root.to_integer())
        } else {
            Expr::Rational(root)
        });
    }
    Ok(Expr::Mul(vec![
        Expr::from_i64(-1),
        b,
        a.pow(Expr::from_i64(-1)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::BigInt;

    #[test]
    fn test_solve_linear_poly() {
        let x = Symbol::new("x");
        // 2x - 6 = 0 => x = 3
        let p = UnivariatePoly::new(
            x,
            vec![
                BigRational::from_integer(BigInt::from(-6)),
                BigRational::from_integer(BigInt::from(2)),
            ],
        );
        let roots = solve_poly(&p).unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], Expr::from_i64(3));
    }

    #[test]
    fn test_solve_linear_numeric() {
        let x = Symbol::new("x");
        // 3*x - 12 = 0 => x = 4
        let e = Expr::Add(vec![
            Expr::Mul(vec![Expr::from_i64(3), Expr::Sym(x.clone())]),
            Expr::from_i64(-12),
        ]);
        assert_eq!(solve_linear(&e, &x).unwrap(), Expr::from_i64(4));
    }

    #[test]
    fn test_solve_linear_rational_coefficient() {
        let x = Symbol::new("x");
        // x/2 + 1/2 = 0 => x = -1
        let half_x = Expr::Mul(vec![
            Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2))),
            Expr::Sym(x.clone()),
        ]);
        let e = Expr::Add(vec![
            half_x,
            Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2))),
        ]);
        assert_eq!(solve_linear(&e, &x).unwrap(), Expr::from_i64(-1));
    }

    #[test]
    fn test_solve_linear_symbolic_constant() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        // 2*x + y = 0 => x = -y * 2^-1 (structural)
        let e = Expr::Add(vec![
            Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(x.clone())]),
            Expr::Sym(y.clone()),
        ]);
        assert_eq!(
            solve_linear(&e, &x).unwrap(),
            Expr::Mul(vec![
                Expr::from_i64(-1),
                Expr::Sym(y),
                Expr::Pow(
                    std::sync::Arc::new(Expr::from_i64(2)),
                    std::sync::Arc::new(Expr::from_i64(-1))
                ),
            ])
        );
    }

    #[test]
    fn test_solve_linear_rejects_nonlinear() {
        let x = Symbol::new("x");
        // x^2 - 1: quadratic in the linear solver
        let e = Expr::Add(vec![
            Expr::Pow(
                std::sync::Arc::new(Expr::Sym(x.clone())),
                std::sync::Arc::new(Expr::from_i64(2)),
            ),
            Expr::from_i64(-1),
        ]);
        assert_eq!(solve_linear(&e, &x), Err(SolverError::NonLinear));
        // sin(x): transcendental, not linear
        let s = Expr::Function("sin".to_string(), vec![Expr::Sym(x.clone())]);
        assert_eq!(solve_linear(&s, &x), Err(SolverError::NonLinear));
        // x*y: linear in x with symbolic coefficient y (sympy: solve(x*y, x) == [0])
        let y2 = Symbol::new("y");
        let p = Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y2)]);
        assert_eq!(solve_linear(&p, &x), Ok(Expr::from_i64(0)));
    }

    #[test]
    fn test_dsolve_linear_first_order_and_verification() {
        // y'(x) + 2*y(x) = 4
        // P(x) = 2, Q(x) = 4
        // Solution: y(x) = 2 + C1 * exp(-2*x)
        let x = Symbol::new("x");
        let c1 = Symbol::new("C1");
        let p_expr = Expr::from_i64(2);
        let q_expr = Expr::from_i64(4);

        let sol = dsolve_linear_first_order(&p_expr, &q_expr, &x, &c1).unwrap();
        // Check ODE satisfaction: y' + 2*y - 4 = 0
        let dy = fsym_calculus::diff(&sol, &x);
        let ode_eval = Expr::Add(vec![
            dy,
            Expr::Mul(vec![Expr::from_i64(2), sol]),
            Expr::from_i64(-4),
        ]);
        let mut map = std::collections::HashMap::new();
        map.insert(x.clone(), Expr::from_i64(1));
        map.insert(c1.clone(), Expr::from_i64(3));
        let num_val = ode_eval.subs(&map).evalf().unwrap();
        assert!(num_val.abs() < 1e-10, "ODE residual must be zero");

        let unsupported = Expr::Function("log".to_string(), vec![Expr::Sym(x.clone())]);
        assert!(matches!(
            dsolve_linear_first_order(&unsupported, &q_expr, &x, &c1),
            Err(SolverError::IncompleteSolutionSet(message))
                if message.contains("coefficient")
        ));
    }

    #[test]
    fn test_dsolve_const_coeff_second_order_and_verification() {
        // y'' - 5*y' + 6*y = 0
        // Characteristic roots: r1 = 2, r2 = 3
        // y(x) = C1 * exp(2*x) + C2 * exp(3*x)
        let x = Symbol::new("x");
        let c1 = Symbol::new("C1");
        let c2 = Symbol::new("C2");

        let sol = dsolve_const_coeff_second_order(1, -5, 6, &x, &c1, &c2).unwrap();
        assert!(verify_const_coeff_second_order_solution(&sol, 1, -5, 6, &x));

        // Harmonic oscillator: y'' + 4*y = 0
        // y(x) = C1 * cos(2*x) + C2 * sin(2*x)
        let sol_osc = dsolve_const_coeff_second_order(1, 0, 4, &x, &c1, &c2).unwrap();
        assert!(verify_const_coeff_second_order_solution(
            &sol_osc, 1, 0, 4, &x
        ));
    }

    #[test]
    fn ode_solver_and_residual_checker_refuse_overflows_and_false_positives() {
        let x = Symbol::new("x");
        let c1 = Symbol::new("C1");
        let c2 = Symbol::new("C2");

        assert!(matches!(
            dsolve_const_coeff_second_order(0, 1, 1, &x, &c1, &c2),
            Err(SolverError::InvalidSystem(_))
        ));
        assert!(dsolve_const_coeff_second_order(i64::MIN, 0, 0, &x, &c1, &c2).is_ok());
        assert!(dsolve_const_coeff_second_order(1, i64::MIN, 0, &x, &c1, &c2).is_ok());
        assert!(dsolve_const_coeff_second_order(1, 0, i64::MAX, &x, &c1, &c2).is_ok());

        // The old sampled fallback substituted x=1 and accepted this nonzero
        // residual for the equation y=0.
        let sampled_false_positive = Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(-1)]);
        assert!(!verify_const_coeff_second_order_solution(
            &sampled_false_positive,
            0,
            0,
            1,
            &x,
        ));

        let left = Expr::Add(
            (0..65)
                .map(|index| Expr::symbol(format!("u{index}")))
                .collect(),
        );
        let right = Expr::Add(
            (0..65)
                .map(|index| Expr::symbol(format!("v{index}")))
                .collect(),
        );
        let oversized_residual = Expr::Mul(vec![left, right]);
        assert!(!verify_const_coeff_second_order_solution(
            &oversized_residual,
            0,
            0,
            1,
            &x,
        ));
    }

    #[test]
    fn test_solve_2var_poly_system() {
        // System:
        // x + y - 5 = 0
        // x - y - 1 = 0
        // Solution: x = 3, y = 2
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let gens = vec![x.clone(), y.clone()];

        let mut t1 = std::collections::BTreeMap::new();
        t1.insert(vec![1, 0], BigRational::one());
        t1.insert(vec![0, 1], BigRational::one());
        t1.insert(vec![0, 0], BigRational::from_integer((-5).into()));
        let p1 = fsym_polys::multivariate::MultivariatePoly::new(gens.clone(), t1);

        let mut t2 = std::collections::BTreeMap::new();
        t2.insert(vec![1, 0], BigRational::one());
        t2.insert(vec![0, 1], BigRational::from_integer((-1).into()));
        t2.insert(vec![0, 0], BigRational::from_integer((-1).into()));
        let p2 = fsym_polys::multivariate::MultivariatePoly::new(gens.clone(), t2);

        let sols = solve_2var_poly_system(&[p1, p2], &x, &y).unwrap();
        assert_eq!(sols.len(), 1);
        let sol = &sols[0];
        assert_eq!(sol.get(&x).unwrap(), &Expr::from_i64(3));
        assert_eq!(sol.get(&y).unwrap(), &Expr::from_i64(2));
    }

    #[test]
    fn two_variable_solver_refuses_wrong_ring_shape_and_order() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let one_variable = fsym_polys::multivariate::MultivariatePoly::one(vec![x.clone()]);
        assert!(matches!(
            solve_2var_poly_system(&[one_variable], &x, &y),
            Err(SolverError::InvalidSystem(_))
        ));

        let reversed = fsym_polys::multivariate::MultivariatePoly::one(vec![y.clone(), x.clone()]);
        assert!(matches!(
            solve_2var_poly_system(&[reversed], &x, &y),
            Err(SolverError::InvalidSystem(_))
        ));
        assert!(matches!(
            solve_2var_poly_system(
                &[fsym_polys::multivariate::MultivariatePoly::one(vec![
                    x.clone(),
                    y.clone(),
                ])],
                &x,
                &x,
            ),
            Err(SolverError::InvalidSystem(_))
        ));
    }

    #[test]
    fn two_variable_solver_refuses_nonlinear_back_substitution() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let generators = vec![x.clone(), y.clone()];

        let mut x_squared_terms = std::collections::BTreeMap::new();
        x_squared_terms.insert(vec![2, 0], BigRational::one());
        let x_squared =
            fsym_polys::multivariate::MultivariatePoly::new(generators.clone(), x_squared_terms);

        let mut y_terms = std::collections::BTreeMap::new();
        y_terms.insert(vec![0, 1], BigRational::one());
        let y_equation = fsym_polys::multivariate::MultivariatePoly::new(generators, y_terms);

        assert_eq!(
            solve_2var_poly_system(&[x_squared, y_equation], &x, &y),
            Err(SolverError::UnsupportedDegree(2))
        );
    }

    #[test]
    fn two_variable_solver_preflights_elimination_degree_before_allocation() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let generators = vec![x.clone(), y.clone()];

        let mut x_terms = std::collections::BTreeMap::new();
        x_terms.insert(vec![1, 0], BigRational::one());
        let x_equation =
            fsym_polys::multivariate::MultivariatePoly::new(generators.clone(), x_terms);

        let mut huge_y_terms = std::collections::BTreeMap::new();
        huge_y_terms.insert(vec![0, u32::MAX], BigRational::one());
        let huge_y_equation =
            fsym_polys::multivariate::MultivariatePoly::new(generators, huge_y_terms);

        assert_eq!(
            solve_2var_poly_system(&[x_equation, huge_y_equation], &x, &y),
            Err(SolverError::UnsupportedDegree(u32::MAX as usize))
        );
    }

    #[test]
    fn two_variable_solver_rejects_noncanonical_zero_coefficients() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let mut terms = std::collections::BTreeMap::new();
        terms.insert(vec![1, 0], BigRational::zero());
        let malformed = fsym_polys::multivariate::MultivariatePoly {
            generators: vec![x.clone(), y.clone()],
            terms,
        };

        assert!(matches!(
            solve_2var_poly_system(&[malformed], &x, &y),
            Err(SolverError::InvalidSystem(_))
        ));
    }

    #[test]
    fn test_solve_higher_degree_poly_via_factorization() {
        let x = Symbol::new("x");

        // x^3 - 6x^2 + 11x - 6 = (x - 1)(x - 2)(x - 3) = 0
        let p_cubic = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(-6)),
                BigRational::from_integer(BigInt::from(11)),
                BigRational::from_integer(BigInt::from(-6)),
                BigRational::one(),
            ],
        );

        let roots = solve_poly(&p_cubic).unwrap();
        assert_eq!(roots.len(), 3);
        assert!(roots.contains(&Expr::Integer(BigInt::from(1))));
        assert!(roots.contains(&Expr::Integer(BigInt::from(2))));
        assert!(roots.contains(&Expr::Integer(BigInt::from(3))));
    }

    #[test]
    fn test_first_order_ode_solving_and_verification() {
        let x = Symbol::new("x");
        let c1 = Symbol::new("C1");

        // y'(x) = x * y(x) -> y(x) = C1 * exp(x^2 / 2)
        let f_x = Expr::Sym(x.clone());
        let sol = dsolve_separable_linear(&f_x, &x, &c1).unwrap();
        assert!(matches!(sol, Expr::Mul(_)));

        // y'(x) + y(x) = 1 -> y(x) = 1 + C1 * exp(-x)
        let p = Expr::from_i64(1);
        let q = Expr::from_i64(1);
        let lin_sol = dsolve_linear_first_order(&p, &q, &x, &c1).unwrap();
        let dy = fsym_calculus::diff(&lin_sol, &x);
        let py = Expr::Mul(vec![p.clone(), lin_sol.clone()]);
        let residual = Expr::Add(vec![dy.clone(), py.clone(), Expr::Mul(vec![Expr::from_i64(-1), q.clone()])]);
        println!("lin_sol = {lin_sol:?}");
        println!("dy = {dy:?}");
        println!("py = {py:?}");
        println!("residual = {residual:?}");
        println!("simplified = {:?}", fsym_simplify::simplify(&residual));
        assert!(verify_first_order_linear_solution(&lin_sol, &p, &q, &x));
    }

    #[test]
    fn test_nonhomogeneous_second_order_ode_solving_and_verification() {
        let x = Symbol::new("x");
        let c1 = Symbol::new("C1");
        let c2 = Symbol::new("C2");

        // 1. Constant forcing: y'' - 3*y' + 2*y = 4
        let f_const = Expr::from_i64(4);
        let sol_const =
            dsolve_const_coeff_second_order_nonhomogeneous(1, -3, 2, &f_const, &x, &c1, &c2)
                .unwrap();
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &sol_const, 1, -3, 2, &f_const, &x
        ));

        // 2. Exponential forcing: y'' - y = exp(2*x)
        let f_exp = Expr::Function(
            "exp".into(),
            vec![Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(x.clone())])],
        );
        let sol_exp =
            dsolve_const_coeff_second_order_nonhomogeneous(1, 0, -1, &f_exp, &x, &c1, &c2).unwrap();
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &sol_exp, 1, 0, -1, &f_exp, &x
        ));

        // 3. Trigonometric forcing: y'' + 4*y = cos(x)
        let f_cos = Expr::Function("cos".into(), vec![Expr::Sym(x.clone())]);
        let sol_cos =
            dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 4, &f_cos, &x, &c1, &c2).unwrap();
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &sol_cos, 1, 0, 4, &f_cos, &x
        ));

        // 4. Resonant trigonometric forcing: y'' + y = cos(x)
        let f_res_cos = Expr::Function("cos".into(), vec![Expr::Sym(x.clone())]);
        let sol_res_cos =
            dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 1, &f_res_cos, &x, &c1, &c2)
                .unwrap();
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &sol_res_cos,
            1,
            0,
            1,
            &f_res_cos,
            &x
        ));

        // 5. Resonant exponential forcing: y'' - y = exp(x)
        let f_res_exp = Expr::Function("exp".into(), vec![Expr::Sym(x.clone())]);
        let sol_res_exp =
            dsolve_const_coeff_second_order_nonhomogeneous(1, 0, -1, &f_res_exp, &x, &c1, &c2)
                .unwrap();
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &sol_res_exp,
            1,
            0,
            -1,
            &f_res_exp,
            &x
        ));
    }
}
