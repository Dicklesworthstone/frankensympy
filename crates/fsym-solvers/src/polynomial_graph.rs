//! Complete polynomial graph families from an independently checked ideal.
//!
//! A reduced lexicographic basis whose leading monomials are distinct variables
//! has the form `x_i - p_i(free_variables)`. The pivot coordinates are therefore
//! uniquely determined by freely chosen remaining coordinates. Both inclusions
//! of the original ideal are checked by the independent Groebner proof kernel;
//! this module separately checks the graph shape and the actual lifted values.
//! Failure to recognize this form is `NotApplicable`, never an empty set.
//!
//! Original sources are admitted before any result. Up to four deterministic
//! variable orders share the caller's cumulative generator/verifier meters.
//! Source and basis bounds are those of the checked polynomial lane: eight
//! variables, 64 equations, degree 256, 4096 terms and 4096-bit coefficients.
//! The returned family additionally has at most 4096 projected expression nodes.
//! This is a synchronous bounded lane, not a runtime-supervision claim.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension};
use fsym_core::{Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, TermOrder, VerifiedGroebnerBasis, bounded_polynomial_from_expr,
    checked_groebner_basis,
};
use num_traits::One;

use crate::SolverError;

const MAX_VARIABLES: usize = 8;
const MAX_EQUATIONS: usize = 64;
const MAX_OUTPUT_NODES: usize = 4096;

/// The complete recognized solution set, or an explicit recognition miss.
#[derive(Debug, Clone)]
pub enum PolynomialGraphOutcome {
    Empty,
    Family(VerifiedPolynomialGraph),
    NotApplicable,
}

/// Immutable coordinates and the independently checked ideal that binds them.
/// The free coordinates are the caller's original typed symbols.
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialGraph {
    variables: Vec<Symbol>,
    values: Vec<Expr>,
    free_variables: Vec<Symbol>,
    basis: VerifiedGroebnerBasis,
}

impl VerifiedPolynomialGraph {
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    pub fn values(&self) -> &[Expr] {
        &self.values
    }

    /// Transfer already checked coordinate ownership to a result bridge
    /// without cloning potentially large exact values or symbol names.
    pub fn into_values(self) -> Vec<Expr> {
        self.values
    }

    pub fn free_variables(&self) -> &[Symbol] {
        &self.free_variables
    }

    pub fn basis(&self) -> &VerifiedGroebnerBasis {
        &self.basis
    }
}

fn refuse(message: impl std::fmt::Display) -> SolverError {
    SolverError::IncompleteSolutionSet(format!("polynomial graph refused: {message}"))
}

fn charge(meter: &mut impl BudgetMeter, compute: usize, memory: usize) -> Result<(), SolverError> {
    meter.checkpoint().map_err(refuse)?;
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, compute.max(1) as u64),
            (Dimension::MemoryBytes, memory.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(refuse)
}

fn ring_bytes(variables: &[Symbol]) -> usize {
    variables
        .iter()
        .map(|variable| variable.name.len() + 128)
        .sum()
}

fn orders(
    inputs: &[MultivariatePoly],
    width: usize,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<Vec<usize>>, SolverError> {
    let term_count: usize = inputs.iter().map(|poly| poly.terms().len()).sum();
    charge(meter, term_count * width * width, 1024)?;
    let mut constant_linear = vec![false; width];
    let mut nonlinear = vec![0usize; width];
    for poly in inputs {
        for index in 0..width {
            charge(meter, 1, width * 4 + 64)?;
            let mut unit = vec![0; width];
            unit[index] = 1;
            if poly.degree_in(index) == 1 && poly.terms().contains_key(&unit) {
                // A standalone term and degree one are insufficient when an
                // additional mixed term makes the coefficient nonconstant.
                constant_linear[index] |= poly
                    .terms()
                    .keys()
                    .all(|powers| powers[index] == 0 || powers == &unit);
            }
            nonlinear[index] += poly
                .terms()
                .keys()
                .filter(|powers| powers[index] != 0 && *powers != &unit)
                .count();
        }
    }
    let caller: Vec<_> = (0..width).collect();
    let mut heuristic = caller.clone();
    heuristic.sort_by_key(|&index| (!constant_linear[index], nonlinear[index], index));
    let reverse_heuristic: Vec<_> = heuristic.iter().copied().rev().collect();
    let reverse_caller: Vec<_> = caller.iter().copied().rev().collect();
    let mut result = Vec::new();
    for order in [heuristic, caller, reverse_heuristic, reverse_caller] {
        if !result.contains(&order) {
            result.push(order);
        }
    }
    Ok(result)
}

fn permute(
    inputs: &[MultivariatePoly],
    variables: &[Symbol],
    order: &[usize],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<MultivariatePoly>, SolverError> {
    let storage = ring_bytes(variables);
    charge(meter, order.len(), storage + inputs.len() * 128)?;
    let ordered: Vec<_> = order
        .iter()
        .map(|&index| variables[index].clone())
        .collect();
    inputs
        .iter()
        .map(|poly| {
            charge(
                meter,
                poly.terms().len() * order.len(),
                poly.terms().len() * 2048 + storage,
            )?;
            let terms = poly
                .terms()
                .iter()
                .map(|(powers, coefficient)| {
                    (
                        order.iter().map(|&index| powers[index]).collect(),
                        coefficient.clone(),
                    )
                })
                .collect();
            MultivariatePoly::new(ordered.clone(), terms).map_err(refuse)
        })
        .collect()
}

fn unit_variable(powers: &[u32]) -> Option<usize> {
    let mut result = None;
    for (index, &power) in powers.iter().enumerate() {
        match power {
            0 => {}
            1 if result.is_none() => result = Some(index),
            _ => return None,
        }
    }
    result
}

/// Independently inspect all graph obligations even though basis reducedness
/// already implies the absence of pivot variables from the tails.
fn graph_pivots(
    basis: &VerifiedGroebnerBasis,
    width: usize,
    meter: &mut impl BudgetMeter,
) -> Result<Option<Vec<usize>>, SolverError> {
    if basis.order() != TermOrder::Lex {
        return Ok(None);
    }
    let terms: usize = basis.basis().iter().map(|poly| poly.terms().len()).sum();
    charge(meter, terms * width * 3, width * 128)?;
    let mut pivots = Vec::new();
    for poly in basis.basis() {
        let Some((leading, coefficient)) = poly.leading_term(TermOrder::Lex) else {
            return Ok(None);
        };
        let Some(pivot) = unit_variable(leading) else {
            return Ok(None);
        };
        if !coefficient.is_one() || pivots.contains(&pivot) {
            return Ok(None);
        }
        pivots.push(pivot);
    }
    for (poly, &pivot) in basis.basis().iter().zip(&pivots) {
        let mut unit = vec![0; width];
        unit[pivot] = 1;
        for powers in poly.terms().keys() {
            if powers != &unit && pivots.iter().any(|&index| powers[index] != 0) {
                return Ok(None);
            }
        }
    }
    Ok(Some(pivots))
}

/// Reference check of the published coordinates, independent of tail lifting.
/// Each reconstructed relation must equal the corresponding checked basis
/// equation; each free coordinate must be its original typed variable.
fn verify_values(
    basis: &VerifiedGroebnerBasis,
    variables: &[Symbol],
    values: &[Expr],
    meter: &mut impl BudgetMeter,
) -> Result<(), SolverError> {
    let ambient = basis
        .input_basis()
        .first()
        .ok_or_else(|| refuse("the checked ambient polynomial ring is absent"))?
        .generators();
    if variables.len() != ambient.len()
        || values.len() != ambient.len()
        || variables.iter().enumerate().any(|(index, variable)| {
            !ambient.contains(variable) || variables[..index].contains(variable)
        })
    {
        return Err(refuse(
            "graph coordinates do not bind the original variables",
        ));
    }
    let pivots = graph_pivots(basis, ambient.len(), meter)?
        .ok_or_else(|| refuse("the checked basis does not define a polynomial graph"))?;
    for (variable, value) in variables.iter().zip(values) {
        let index = ambient
            .iter()
            .position(|candidate| candidate == variable)
            .ok_or_else(|| refuse("graph variable is outside the checked ring"))?;
        let coordinate = bounded_polynomial_from_expr(value, ambient, meter).map_err(refuse)?;
        charge(
            meter,
            coordinate.terms().len() + 1,
            coordinate.terms().len() * 4096 + ring_bytes(ambient) * 2 + 2048,
        )?;
        let unit = MultivariatePoly::var(ambient.to_vec(), variable).map_err(refuse)?;
        if let Some(row) = pivots.iter().position(|&pivot| pivot == index) {
            if unit.sub(&coordinate).map_err(refuse)? != basis.basis()[row] {
                return Err(refuse(
                    "a graph coordinate does not reproduce its checked relation",
                ));
            }
        } else if coordinate != unit {
            return Err(refuse("a free coordinate was changed or captured"));
        }
    }
    meter.checkpoint().map_err(refuse)
}

/// Recognize a complete polynomial graph with separate cumulative meters.
pub fn metered_polynomial_graph_system(
    equations: &[Expr],
    variables: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<PolynomialGraphOutcome, SolverError> {
    charge(generator_meter, 1, 1)?;
    if variables.is_empty() || variables.len() > MAX_VARIABLES || equations.len() > MAX_EQUATIONS {
        return Err(SolverError::InvalidSystem(
            "polynomial graphs require 1..8 variables and at most 64 equations".into(),
        ));
    }
    // This also validates symbol names and duplicate typed generators before
    // any source copies, including the otherwise empty-equation case.
    let zero = bounded_polynomial_from_expr(&Expr::from_i64(0), variables, generator_meter)
        .map_err(|error| SolverError::InvalidSystem(error.to_string()))?;
    charge(
        generator_meter,
        equations.len(),
        equations.len().max(1) * 128,
    )?;
    let mut inputs = Vec::with_capacity(equations.len().max(1));
    for equation in equations {
        inputs.push(
            bounded_polynomial_from_expr(equation, variables, generator_meter)
                .map_err(|error| SolverError::InvalidSystem(error.to_string()))?,
        );
    }
    // A zero polynomial binds the ambient ring in the empty ideal's receipt.
    if inputs.is_empty() {
        inputs.push(zero);
    }
    for order in orders(&inputs, variables.len(), generator_meter)? {
        let reordered = permute(&inputs, variables, &order, generator_meter)?;
        let basis =
            checked_groebner_basis(&reordered, TermOrder::Lex, generator_meter, verifier_meter)
                .map_err(refuse)?;
        if basis.basis().iter().any(MultivariatePoly::is_one) {
            generator_meter.checkpoint().map_err(refuse)?;
            verifier_meter.checkpoint().map_err(refuse)?;
            return Ok(PolynomialGraphOutcome::Empty);
        }
        let Some(pivots) = graph_pivots(&basis, variables.len(), verifier_meter)? else {
            continue;
        };
        let ambient = reordered[0].generators();
        let storage = ring_bytes(variables);
        charge(
            generator_meter,
            variables.len(),
            storage * 3 + variables.len() * 128,
        )?;
        let mut values: Vec<_> = variables.iter().cloned().map(Expr::Sym).collect();
        let mut output_nodes = variables.len();
        for (poly, &pivot) in basis.basis().iter().zip(&pivots) {
            let projected = 1 + poly
                .terms()
                .keys()
                .map(|powers| 2 + powers.iter().filter(|&&power| power != 0).count() * 4)
                .sum::<usize>();
            output_nodes = output_nodes
                .checked_add(projected)
                .filter(|&nodes| nodes <= MAX_OUTPUT_NODES)
                .ok_or_else(|| refuse("graph output exceeds 4096 projected expression nodes"))?;
            charge(
                generator_meter,
                projected,
                projected * (2048 + storage) + storage,
            )?;
            let mut unit = vec![0; variables.len()];
            unit[pivot] = 1;
            let tail: BTreeMap<_, _> = poly
                .terms()
                .iter()
                .filter(|(powers, _)| *powers != &unit)
                .map(|(powers, coefficient)| (powers.clone(), -coefficient.clone()))
                .collect();
            values[order[pivot]] = MultivariatePoly::new(ambient.to_vec(), tail)
                .and_then(|poly| poly.to_expr())
                .map_err(refuse)?;
        }
        verify_values(&basis, variables, &values, verifier_meter)?;
        let free_variables = variables
            .iter()
            .enumerate()
            .filter(|(index, _)| !pivots.iter().any(|&pivot| order[pivot] == *index))
            .map(|(_, variable)| variable.clone())
            .collect();
        let family = VerifiedPolynomialGraph {
            variables: variables.to_vec(),
            values,
            free_variables,
            basis,
        };
        generator_meter.checkpoint().map_err(refuse)?;
        verifier_meter.checkpoint().map_err(refuse)?;
        return Ok(PolynomialGraphOutcome::Family(family));
    }
    generator_meter.checkpoint().map_err(refuse)?;
    verifier_meter.checkpoint().map_err(refuse)?;
    Ok(PolynomialGraphOutcome::NotApplicable)
}

/// Convenience entry point with finite cumulative budgets for both phases.
pub fn polynomial_graph_system(
    equations: &[Expr],
    variables: &[Symbol],
) -> Result<PolynomialGraphOutcome, SolverError> {
    metered_polynomial_graph_system(
        equations,
        variables,
        &mut Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0)),
        &mut Budget::new(BudgetLimits::uniform(256 * 1024 * 1024, 0)),
    )
}

#[cfg(test)]
mod reference_checks {
    use super::*;
    use fsym_budget::Unbounded;
    use fsym_core::{SymbolIdentity, parse};

    #[test]
    fn lifted_coordinate_mutations_do_not_inherit_the_basis_verification() {
        let variables = vec![Symbol::new("x"), Symbol::new("y")];
        let PolynomialGraphOutcome::Family(family) =
            polynomial_graph_system(&[parse("x-y**2").unwrap()], &variables).unwrap()
        else {
            panic!("parabola must be recognized");
        };
        verify_values(family.basis(), &variables, family.values(), &mut Unbounded).unwrap();
        for bad in [
            vec![parse("2*y**2").unwrap(), parse("y").unwrap()],
            vec![parse("y**2").unwrap(), parse("y+1").unwrap()],
            vec![parse("y**2").unwrap()],
            vec![parse("x").unwrap(), parse("y").unwrap()],
        ] {
            assert!(verify_values(family.basis(), &variables, &bad, &mut Unbounded).is_err());
        }
        let captured =
            Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([91; 32]));
        let mut bad = family.values().to_vec();
        bad[1] = Expr::Sym(captured.clone());
        assert!(verify_values(family.basis(), &variables, &bad, &mut Unbounded).is_err());
        assert!(
            verify_values(
                family.basis(),
                &[variables[0].clone(), captured],
                family.values(),
                &mut Unbounded
            )
            .is_err()
        );
    }
}
