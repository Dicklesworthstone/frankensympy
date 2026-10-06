//! Zero-dimensional polynomial systems over Q in any number of variables.
//!
//! The reduced lexicographic Groebner basis `G` (x1 > ... > xn) is
//! triangular by elimination: `G_k = G ∩ Q[x_k, ..., x_n]` generates the
//! k-th elimination ideal. Solutions are built from the last variable
//! upwards: for each partial solution of `(x_{k+1}, ..., x_n)`, the values of
//! `x_k` are the common roots of the basis elements whose leading variable is
//! `x_k`, specialized at that partial solution. The first nonvanishing
//! specialized polynomial is solved exactly and its roots are kept only where
//! every other specialized element vanishes. This yields every solution
//! (complete) and only solutions (each candidate satisfies all of `G`, hence
//! the input ideal).
//!
//! Zero tests use exact simplification first and a numerical enclosure only
//! to *reject* clearly nonzero values; anything undecided refuses with a
//! typed error instead of guessing. Positive-dimensional systems (some
//! specialized polynomial set vanishes identically) refuse as well.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use fsym_core::{Expr, Symbol};
use fsym_polys::groebner::groebner_basis;
use fsym_polys::multivariate::{MultivariatePoly, TermOrder};

use crate::SolverError;
use crate::univariate::{solve_univariate, sort_roots};

const MAX_SOLUTIONS: usize = 256;

fn refuse(msg: impl Into<String>) -> SolverError {
    SolverError::IncompleteSolutionSet(msg.into())
}

/// `Some(true)` exactly zero, `Some(false)` provably (or clearly) nonzero,
/// `None` undecided.
fn zero_status(e: &Expr) -> Option<bool> {
    if e.is_zero() {
        return Some(true);
    }
    let s = fsym_simplify::simplify(e);
    if s.is_zero() {
        return Some(true);
    }
    if let Ok(ball) = s.evalf_ball(30) {
        if !ball.contains_zero() {
            return Some(false);
        }
        return None;
    }
    let v = fsym_calculus::gruntz::complex_value(&s)?;
    (v.norm().is_finite() && v.norm() > 1e-6).then_some(false)
}

/// Index of the leading (lowest-index) variable of `p`, if any.
fn leading_var(p: &MultivariatePoly) -> Option<usize> {
    (0..p.generators().len()).find(|&i| p.degree_in(i) > 0)
}

/// Specializes `p` at the assigned trailing variables and returns it as an
/// expression in the remaining variables.
fn specialize(
    p: &MultivariatePoly,
    assignment: &HashMap<Symbol, Expr>,
) -> Result<Expr, SolverError> {
    let e = p
        .to_expr()
        .map_err(|err| SolverError::InvalidSystem(err.to_string()))?;
    Ok(fsym_simplify::expand(&e.subs(assignment)))
}

/// All solutions of `eqs = 0` in `vars`, each a vector aligned with `vars`.
pub fn solve_polynomial_system(
    eqs: &[Expr],
    vars: &[Symbol],
) -> Result<Vec<Vec<Expr>>, SolverError> {
    if vars.is_empty() {
        return Err(SolverError::InvalidSystem("no solve variables".into()));
    }
    let mut polys = Vec::new();
    for e in eqs {
        let p = MultivariatePoly::from_expr(e, vars)
            .map_err(|err| SolverError::InvalidSystem(err.to_string()))?;
        if !p.is_zero() {
            polys.push(p);
        }
    }
    if polys.is_empty() {
        return Err(SolverError::InfiniteSolutions);
    }
    let gb = groebner_basis(&polys, TermOrder::Lex)
        .map_err(|err| refuse(format!("Groebner basis computation failed: {err}")))?;
    if gb.iter().any(|p| !p.is_zero() && leading_var(p).is_none()) {
        return Ok(Vec::new());
    }
    let n = vars.len();
    let mut partials: Vec<HashMap<Symbol, Expr>> = vec![HashMap::new()];
    for k in (0..n).rev() {
        let level: Vec<&MultivariatePoly> =
            gb.iter().filter(|p| leading_var(p) == Some(k)).collect();
        if level.is_empty() {
            return Err(SolverError::InfiniteSolutions);
        }
        let xk = &vars[k];
        let mut next = Vec::new();
        for partial in &partials {
            let specialized: Vec<Expr> = level
                .iter()
                .map(|p| specialize(p, partial))
                .collect::<Result<_, _>>()?;
            let mut pivot = None;
            for (i, s) in specialized.iter().enumerate() {
                match zero_status(s) {
                    Some(true) => {}
                    Some(false) => {
                        pivot = Some(i);
                        break;
                    }
                    None => {
                        // A nonconstant polynomial in x_k is nonzero.
                        if s.free_symbols().iter().any(|v| v == xk) {
                            pivot = Some(i);
                            break;
                        }
                        return Err(refuse("undecided zero status during back-substitution"));
                    }
                }
            }
            let Some(pivot) = pivot else {
                return Err(SolverError::InfiniteSolutions);
            };
            let main = &specialized[pivot];
            if !main.free_symbols().iter().any(|v| v == xk) {
                // A nonzero constant: this partial solution does not extend.
                continue;
            }
            let roots = match solve_univariate(main, xk) {
                Ok(r) => r,
                Err(SolverError::NoSolution) => Vec::new(),
                Err(e) => return Err(e),
            };
            for r in roots {
                let mut keep = true;
                for (i, s) in specialized.iter().enumerate() {
                    if i == pivot {
                        continue;
                    }
                    let at = s.subs(&HashMap::from([(xk.clone(), r.clone())]));
                    match zero_status(&at) {
                        Some(true) => {}
                        Some(false) => {
                            keep = false;
                            break;
                        }
                        None => return Err(refuse("undecided root filter")),
                    }
                }
                if keep {
                    let mut extended = partial.clone();
                    extended.insert(xk.clone(), r);
                    next.push(extended);
                    if next.len() > MAX_SOLUTIONS {
                        return Err(refuse("too many solutions"));
                    }
                }
            }
        }
        partials = next;
        if partials.is_empty() {
            return Ok(Vec::new());
        }
    }
    let mut out: Vec<Vec<Expr>> = partials
        .into_iter()
        .map(|sol| vars.iter().map(|v| sol[v].clone()).collect())
        .collect();
    out.dedup();
    Ok(out)
}

/// Sorts solution tuples lexicographically by the upstream root order of
/// each coordinate.
pub fn sort_solutions(mut sols: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    let rank = |col: usize, sols: &[Vec<Expr>]| -> HashMap<String, usize> {
        let values: Vec<Expr> = sols.iter().map(|s| s[col].clone()).collect();
        sort_roots(values)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v.to_string(), i))
            .collect()
    };
    if sols.is_empty() {
        return sols;
    }
    let ranks: Vec<HashMap<String, usize>> = (0..sols[0].len()).map(|c| rank(c, &sols)).collect();
    sols.sort_by_key(|s| {
        s.iter()
            .enumerate()
            .map(|(c, v)| ranks[c].get(&v.to_string()).copied().unwrap_or(usize::MAX))
            .collect::<Vec<_>>()
    });
    sols
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::parse;

    fn solve(eqs: &[&str], vars: &[&str]) -> Vec<Vec<String>> {
        let es: Vec<Expr> = eqs.iter().map(|s| parse(s).unwrap()).collect();
        let vs: Vec<Symbol> = vars.iter().map(|v| Symbol::new(*v)).collect();
        sort_solutions(solve_polynomial_system(&es, &vs).unwrap())
            .into_iter()
            .map(|s| s.into_iter().map(|e| e.to_string()).collect())
            .collect()
    }

    #[test]
    fn circle_and_line() {
        let s = solve(&["x**2 + y**2 - 1", "x - y"], &["x", "y"]);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0][0], s[0][1]);
    }

    #[test]
    fn non_shape_position_system() {
        let s = solve(&["x**2 - y", "y - 4"], &["x", "y"]);
        assert_eq!(s, vec![vec!["-2", "4"], vec!["2", "4"]]);
    }

    #[test]
    fn inconsistent_and_three_variable_systems() {
        assert!(solve(&["x + y - 1", "x + y - 2"], &["x", "y"]).is_empty());
        let s = solve(&["x*y - z", "x - 1", "y - 2"], &["x", "y", "z"]);
        assert_eq!(s, vec![vec!["1", "2", "2"]]);
    }

    #[test]
    fn positive_dimensional_refuses() {
        let es = vec![parse("x*y").unwrap()];
        let vs = vec![Symbol::new("x"), Symbol::new("y")];
        assert!(solve_polynomial_system(&es, &vs).is_err());
    }
}
