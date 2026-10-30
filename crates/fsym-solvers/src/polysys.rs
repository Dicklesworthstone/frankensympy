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
//! Original expressions are admitted before simplification and the basis is
//! checked by the independent proof kernel. Zero tests use exact identities
//! or directed-rounding complex enclosures, never a floating-point tolerance.
//! Every retained root satisfies the pivot as well as the other constraints;
//! every final tuple is checked against all original equations. An undecided
//! obligation refuses the complete result rather than dropping that branch.
//!
//! This synchronous lane has bounded source/basis work and safe points around
//! symbolic root generation and residual checks. It does not promise runtime
//! supervision or cancellation inside every legacy radical-construction step.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension};
use fsym_core::{Constant, Expr, Symbol};
use fsym_polys::multivariate::{MultivariatePoly, TermOrder};
use fsym_polys::{bounded_polynomial_from_expr, checked_groebner_basis};

use crate::SolverError;
use crate::univariate::{solve_univariate, sort_roots, zero_status};

const MAX_SOLUTIONS: usize = 256;
const MAX_EQUATIONS: usize = 64;
const MAX_VARIABLES: usize = 8;
const MAX_EXPRESSION_NODES: usize = 4096;
const MAX_EXPRESSION_DEPTH: usize = 64;

/// Source-bound receipt for the complete finite tuple set. Construction stays
/// inside this module, after the checked basis, complete root generation and
/// every original residual have passed. Component solvers may bind the receipt
/// to an exact coordinate projection, but cannot create an accepted root list.
#[derive(Debug, Clone)]
pub(crate) struct VerifiedFinitePolynomialSystem {
    variables: Vec<Symbol>,
    equations: Vec<MultivariatePoly>,
    solutions: Vec<Vec<Expr>>,
}

impl VerifiedFinitePolynomialSystem {
    pub(crate) fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    /// Original admitted nonzero polynomial equations, in caller order.
    pub(crate) fn equations(&self) -> &[MultivariatePoly] {
        &self.equations
    }

    pub(crate) fn solutions(&self) -> &[Vec<Expr>] {
        &self.solutions
    }
}

fn refuse(msg: impl Into<String>) -> SolverError {
    SolverError::IncompleteSolutionSet(msg.into())
}

fn charge(meter: &mut impl BudgetMeter, nodes: usize) -> Result<(), SolverError> {
    meter.checkpoint().map_err(|e| refuse(e.to_string()))?;
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, nodes.max(1) as u64),
            (Dimension::MemoryBytes, nodes.max(1) as u64 * 256),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(|e| refuse(e.to_string()))
}

/// Preflight recursive operations and their substituted output before cloning.
/// Source admission separately rejects every non-polynomial original node.
pub(crate) fn expression_size(e: &Expr) -> Result<usize, SolverError> {
    let mut stack = vec![(e, 0usize)];
    let mut nodes = 0;
    while let Some((term, depth)) = stack.pop() {
        nodes += 1;
        if nodes > MAX_EXPRESSION_NODES || depth > MAX_EXPRESSION_DEPTH {
            return Err(refuse("symbolic polynomial-system expression limit"));
        }
        match term {
            Expr::Integer(n) if n.bits() > 4096 => {
                return Err(refuse("symbolic polynomial-system coefficient limit"));
            }
            Expr::Rational(q) if q.numer().bits().max(q.denom().bits()) > 4096 => {
                return Err(refuse("symbolic polynomial-system coefficient limit"));
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                if args.len() > MAX_EXPRESSION_NODES.saturating_sub(nodes + stack.len()) {
                    return Err(refuse("symbolic polynomial-system fanout limit"));
                }
                stack.extend(args.iter().map(|arg| (arg, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                stack.push((base, depth + 1));
                stack.push((exponent, depth + 1));
            }
            _ => {}
        }
    }
    Ok(nodes)
}

fn substitute(
    source: &Expr,
    assignment: &HashMap<Symbol, Expr>,
    meter: &mut impl BudgetMeter,
) -> Result<Expr, SolverError> {
    let source_size = expression_size(source)?;
    let largest = assignment
        .values()
        .map(expression_size)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(1);
    let projected = source_size
        .checked_mul(largest)
        .filter(|&size| size <= MAX_EXPRESSION_NODES)
        .ok_or_else(|| refuse("polynomial-system substitution growth limit"))?;
    charge(meter, projected)?;
    let value = source.subs(assignment);
    expression_size(&value)?;
    meter.checkpoint().map_err(|e| refuse(e.to_string()))?;
    Ok(value)
}

fn checked_zero(e: &Expr, meter: &mut impl BudgetMeter) -> Result<Option<bool>, SolverError> {
    charge(meter, expression_size(e)?)?;
    let result = zero_status(e);
    meter.checkpoint().map_err(|e| refuse(e.to_string()))?;
    Ok(result)
}

/// A zero residual cannot justify using an undefined candidate whose poles
/// were erased during substitution. Polynomial root formulas must be finite
/// ground values in this deliberately small exact expression vocabulary.
fn validate_root_value(root: &Expr, meter: &mut impl BudgetMeter) -> Result<(), SolverError> {
    charge(meter, expression_size(root)?)?;
    let mut stack = vec![root];
    while let Some(term) = stack.pop() {
        match term {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Const(Constant::I | Constant::Pi) => {}
            Expr::Add(args) | Expr::Mul(args) => stack.extend(args),
            Expr::Function(name, args)
                if matches!(name.as_str(), "sin" | "cos") && args.len() == 1 =>
            {
                stack.extend(args);
            }
            Expr::Pow(base, exponent) => {
                let needs_nonzero = match exponent.as_ref() {
                    Expr::Integer(value) => value <= &0_i64,
                    Expr::Rational(value) => value.numer() <= &0_i64,
                    _ => return Err(refuse("a polynomial root has a nonrational exponent")),
                };
                if needs_nonzero && checked_zero(base, meter)? != Some(false) {
                    return Err(refuse(
                        "a polynomial root has an undefined or undecided power base",
                    ));
                }
                stack.push(base);
            }
            _ => {
                return Err(refuse(
                    "a polynomial root is not an admitted finite ground value",
                ));
            }
        }
    }
    Ok(())
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
    meter: &mut impl BudgetMeter,
) -> Result<Expr, SolverError> {
    let projected = p.terms().len().saturating_mul(p.generators().len() * 4 + 2);
    if projected > MAX_EXPRESSION_NODES {
        return Err(refuse("polynomial-system lifting growth limit"));
    }
    charge(meter, projected)?;
    let e = p
        .to_expr()
        .map_err(|err| SolverError::InvalidSystem(err.to_string()))?;
    let value = substitute(&e, assignment, meter)?;
    let value = fsym_simplify::expand_with(&value, meter).map_err(|e| refuse(e.to_string()))?;
    expression_size(&value)?;
    Ok(value)
}

/// All solutions of `eqs = 0` in `vars`, each a vector aligned with `vars`.
pub fn solve_polynomial_system(
    eqs: &[Expr],
    vars: &[Symbol],
) -> Result<Vec<Vec<Expr>>, SolverError> {
    metered_solve_polynomial_system(
        eqs,
        vars,
        &mut Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0)),
        &mut Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0)),
    )
}

/// Solve an admitted exact polynomial system with distinct cumulative meters
/// for generation and independent verification. Meters are never reset during
/// back-substitution. Unknown residuals cannot become an empty or partial set.
pub fn metered_solve_polynomial_system(
    eqs: &[Expr],
    vars: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Vec<Vec<Expr>>, SolverError> {
    metered_checked_polynomial_system(eqs, vars, generator_meter, verifier_meter)
        .map(|checked| checked.solutions)
}

fn finish_system(
    equations: Vec<MultivariatePoly>,
    variables: &[Symbol],
    solutions: Vec<Vec<Expr>>,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedFinitePolynomialSystem, SolverError> {
    if solutions.iter().any(|row| row.len() != variables.len()) {
        return Err(refuse("finite system coordinate count changed"));
    }
    charge(generator, variables.len())?;
    generator
        .charge(
            Dimension::MemoryBytes,
            variables.iter().map(|v| v.name.len() as u64).sum(),
        )
        .map_err(|error| refuse(error.to_string()))?;
    let result = VerifiedFinitePolynomialSystem {
        variables: variables.to_vec(),
        equations,
        solutions,
    };
    generator
        .checkpoint()
        .map_err(|error| refuse(error.to_string()))?;
    verifier
        .checkpoint()
        .map_err(|error| refuse(error.to_string()))?;
    Ok(result)
}

pub(crate) fn metered_checked_polynomial_system(
    eqs: &[Expr],
    vars: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedFinitePolynomialSystem, SolverError> {
    charge(generator_meter, 1)?;
    if vars.is_empty() || vars.len() > MAX_VARIABLES || eqs.len() > MAX_EQUATIONS {
        return Err(SolverError::InvalidSystem(
            "polynomial systems require 1..8 variables and at most 64 equations".into(),
        ));
    }
    if vars.iter().enumerate().any(|(i, v)| vars[..i].contains(v)) {
        return Err(SolverError::InvalidSystem(
            "duplicate solve variables".into(),
        ));
    }
    let mut polys = Vec::new();
    for e in eqs {
        let p = bounded_polynomial_from_expr(e, vars, generator_meter)
            .map_err(|err| SolverError::InvalidSystem(err.to_string()))?;
        if !p.is_zero() {
            polys.push(p);
        }
    }
    if polys.is_empty() {
        return Err(SolverError::InfiniteSolutions);
    }
    let checked = checked_groebner_basis(&polys, TermOrder::Lex, generator_meter, verifier_meter)
        .map_err(|err| refuse(format!("Groebner basis computation failed: {err}")))?;
    let gb = checked.basis();
    if gb.iter().any(|p| !p.is_zero() && leading_var(p).is_none()) {
        generator_meter
            .checkpoint()
            .map_err(|e| refuse(e.to_string()))?;
        verifier_meter
            .checkpoint()
            .map_err(|e| refuse(e.to_string()))?;
        return finish_system(polys, vars, Vec::new(), generator_meter, verifier_meter);
    }
    if !checked.is_zero_dimensional() {
        return Err(SolverError::InfiniteSolutions);
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
                .map(|p| specialize(p, partial, generator_meter))
                .collect::<Result<_, _>>()?;
            let mut pivot = None;
            for (i, s) in specialized.iter().enumerate() {
                match checked_zero(s, verifier_meter)? {
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
            charge(generator_meter, expression_size(main)?)?;
            let roots = match solve_univariate(main, xk) {
                Ok(r) => r,
                Err(SolverError::NoSolution) => Vec::new(),
                Err(e) => return Err(e),
            };
            charge(generator_meter, roots.len())?;
            if roots.len() > MAX_SOLUTIONS {
                return Err(refuse("too many candidate roots"));
            }
            for r in roots {
                validate_root_value(&r, verifier_meter)?;
                let mut keep = true;
                for s in &specialized {
                    let at =
                        substitute(s, &HashMap::from([(xk.clone(), r.clone())]), verifier_meter)?;
                    match checked_zero(&at, verifier_meter)? {
                        Some(true) => {}
                        Some(false) => {
                            keep = false;
                            break;
                        }
                        None => return Err(refuse("undecided root filter")),
                    }
                }
                if keep {
                    charge(generator_meter, partial.len() + expression_size(&r)?)?;
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
            generator_meter
                .checkpoint()
                .map_err(|e| refuse(e.to_string()))?;
            verifier_meter
                .checkpoint()
                .map_err(|e| refuse(e.to_string()))?;
            return finish_system(polys, vars, Vec::new(), generator_meter, verifier_meter);
        }
    }
    let mut out = Vec::new();
    for solution in partials {
        for original in eqs {
            let residual = substitute(original, &solution, verifier_meter)?;
            if checked_zero(&residual, verifier_meter)? != Some(true) {
                return Err(refuse(
                    "a final tuple does not verify against every original equation",
                ));
            }
        }
        let tuple: Vec<_> = vars.iter().map(|v| solution[v].clone()).collect();
        if !out.contains(&tuple) {
            charge(generator_meter, tuple.len())?;
            out.push(tuple);
        }
    }
    generator_meter
        .checkpoint()
        .map_err(|e| refuse(e.to_string()))?;
    verifier_meter
        .checkpoint()
        .map_err(|e| refuse(e.to_string()))?;
    finish_system(polys, vars, out, generator_meter, verifier_meter)
}

/// Sorts solution tuples lexicographically by the upstream root order of
/// each coordinate.
pub fn sort_solutions(mut sols: Vec<Vec<Expr>>) -> Vec<Vec<Expr>> {
    let rank = |col: usize, sols: &[Vec<Expr>]| -> HashMap<Expr, usize> {
        let values: Vec<Expr> = sols.iter().map(|s| s[col].clone()).collect();
        sort_roots(values)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect()
    };
    if sols.is_empty() {
        return sols;
    }
    let ranks: Vec<HashMap<Expr, usize>> = (0..sols[0].len()).map(|c| rank(c, &sols)).collect();
    sols.sort_by_key(|s| {
        s.iter()
            .enumerate()
            .map(|(c, v)| ranks[c].get(v).copied().unwrap_or(usize::MAX))
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
