//! Complete Boolean combinations of bounded rational constraints on the real line.
//!
//! Every atom retains its original source domain. Undefined atoms are false for
//! all six relations, including disequality; conjunction and disjunction then
//! have ordinary set intersection/union semantics. There is deliberately no
//! negation operator that could accidentally turn a pole into a solution.
//!
//! Independently checked atom charts are refined onto one complete square-free
//! critical polynomial. Exact factor ownership establishes the increasing map
//! of each local root set into the common partition. A separate stack evaluator
//! checks every proposed Boolean cell decision. All stages consume the same two
//! cumulative caller ledgers and intrinsic caps. These are immutable in-process
//! receipts, not portable proof-kernel certificates or a general CAD solver.

use super::real_inequality::{CappedMeter, VerifiedRealChart, metered_real_chart};
use super::*;
use fsym_core::real_roots::identify_real_root_factors;
use serde::{Deserialize, Serialize};

pub const MAX_REAL_CONSTRAINTS: usize = 32;
pub const MAX_REAL_FORMULA_NODES: usize = 256;
pub const MAX_REAL_FORMULA_DEPTH: usize = 32;
pub const MAX_REAL_FORMULA_FANOUT: usize = 64;

/// Relation between an original rational expression and zero, on its domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealRelation {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealConstraint {
    pub expression: Expr,
    pub relation: RealRelation,
}

/// A bounded formula over indices into the original ordered constraint list.
/// Empty conjunction is true; empty disjunction is false. All supplied atoms
/// are admitted even if unused or behind a logically decisive branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RealFormula {
    Atom(usize),
    All(Vec<RealFormula>),
    Any(Vec<RealFormula>),
}

/// Complete real set on a common ordered algebraic partition. With `n` roots,
/// `gap_inclusions` has `n+1` open gaps and `point_inclusions` has `n` points.
/// Private fields and absence of a deserializer prevent unchecked acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRealConstraints {
    constraints: Vec<RealConstraint>,
    variable: Symbol,
    formula: RealFormula,
    polynomial: UnivariatePoly,
    roots: VerifiedRealRoots,
    gap_inclusions: Vec<bool>,
    point_inclusions: Vec<bool>,
}

impl VerifiedRealConstraints {
    pub fn constraints(&self) -> &[RealConstraint] {
        &self.constraints
    }
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn formula(&self) -> &RealFormula {
        &self.formula
    }
    pub fn polynomial(&self) -> &UnivariatePoly {
        &self.polynomial
    }
    pub fn roots(&self) -> &VerifiedRealRoots {
        &self.roots
    }
    pub fn gap_inclusions(&self) -> &[bool] {
        &self.gap_inclusions
    }
    pub fn point_inclusions(&self) -> &[bool] {
        &self.point_inclusions
    }
}

fn malformed(reason: &'static str) -> RealEquationError {
    RealEquationError::InvalidConstraints(reason)
}

/// Iterative preflight precedes recursive formula traversal or source cloning.
fn admit_formula(
    formula: &RealFormula,
    atoms: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    if atoms > MAX_REAL_CONSTRAINTS {
        return Err(malformed("at most 32 real constraints are supported"));
    }
    work.charge(1, 64)?;
    let mut pending = vec![(formula, 0usize)];
    let mut count = 0usize;
    while let Some((node, depth)) = pending.pop() {
        count += 1;
        if count > MAX_REAL_FORMULA_NODES || depth > MAX_REAL_FORMULA_DEPTH {
            return Err(limit("Boolean formula nodes/depth"));
        }
        work.charge(1, 96)?;
        match node {
            RealFormula::Atom(index) => {
                if *index >= atoms {
                    return Err(malformed(
                        "formula atom index is outside the constraint list",
                    ));
                }
            }
            RealFormula::All(children) | RealFormula::Any(children) => {
                if children.len() > MAX_REAL_FORMULA_FANOUT
                    || pending.len() + children.len() > MAX_REAL_FORMULA_NODES
                {
                    return Err(limit("Boolean formula fanout"));
                }
                work.charge(1, children.len() as u64 * 32)?;
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
        }
    }
    Ok(count)
}

fn critical_bounds(polynomial: &[BigRational]) -> Result<()> {
    if polynomial.is_empty()
        || polynomial.len() > MAX_REAL_ROOT_DEGREE + 1
        || polynomial
            .iter()
            .any(|q| bits(q) > MAX_REAL_ROOT_INPUT_BITS)
    {
        return Err(limit("common critical degree/coefficient bits"));
    }
    Ok(())
}

fn same_polynomial(
    left: &[BigRational],
    right: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    critical_bounds(left)?;
    critical_bounds(right)?;
    work.charge(1, 32)?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.iter().zip(right) {
        work.scalar(left)?;
        work.scalar(right)?;
        work.charge(bits(left).max(bits(right)).div_ceil(32), 1)?;
        if left != right {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Exact ordinal association. The complete common polynomial is split into
/// the atom's square-free polynomial and its complementary factor. Reference
/// Sturm counts locate every root; overlapping rational isolators or equal
/// printed approximations never determine equality or order.
fn root_map(
    common: &VerifiedRealRoots,
    atom: &VerifiedRealChart,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<usize>> {
    let (complement, remainder) = work.division(common.coefficients(), &atom.polynomial.coeffs)?;
    let recovered = work.product(&complement, &atom.polynomial.coeffs)?;
    if !zero(&remainder) || !same_polynomial(&recovered, common.coefficients(), work)? {
        return Err(invalid(
            "atom critical polynomial does not divide common partition",
        ));
    }
    work.charge(1, 128)?;
    let mut factors = Vec::with_capacity(2);
    let atom_factor = if atom.polynomial.coeffs.len() > 1 {
        factors.push(work.copy(&atom.polynomial.coeffs)?);
        Some(0)
    } else {
        None
    };
    if complement.len() > 1 {
        factors.push(complement);
    }
    let multiplicities = [1usize; 2];
    let owners = identify_real_root_factors(
        common,
        &factors,
        &multiplicities[..factors.len()],
        work.meter,
    )?;
    work.charge(1, atom.roots.intervals().len() as u64 * 16 + 32)?;
    let mut mapping = Vec::with_capacity(atom.roots.intervals().len());
    for (global, (owner, local)) in owners.into_iter().enumerate() {
        work.charge(1, 1)?;
        if Some(owner) == atom_factor {
            if local != mapping.len() {
                return Err(invalid("atom root ordinal changed"));
            }
            mapping.push(global);
        }
    }
    if mapping.len() != atom.roots.intervals().len() {
        return Err(invalid("common partition omitted an atom root"));
    }
    Ok(mapping)
}

struct ConstraintCandidate {
    maps: Vec<Vec<usize>>,
    gaps: Vec<bool>,
    points: Vec<bool>,
}

struct ConstraintStatement<'a> {
    constraints: &'a [RealConstraint],
    variable: &'a Symbol,
    formula: &'a RealFormula,
    formula_nodes: usize,
    atoms: &'a [VerifiedRealChart],
    polynomial: &'a [BigRational],
    roots: &'a VerifiedRealRoots,
}

fn propose_formula(
    formula: &RealFormula,
    values: &[bool],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    work.charge(1, 1)?;
    match formula {
        RealFormula::Atom(index) => values
            .get(*index)
            .copied()
            .ok_or(invalid("candidate formula atom index")),
        RealFormula::All(children) => {
            let mut value = true;
            for child in children {
                value &= propose_formula(child, values, work)?;
            }
            Ok(value)
        }
        RealFormula::Any(children) => {
            let mut value = false;
            for child in children {
                value |= propose_formula(child, values, work)?;
            }
            Ok(value)
        }
    }
}

fn propose_constraints(
    statement: &ConstraintStatement<'_>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ConstraintCandidate> {
    let count = statement.roots.intervals().len();
    work.charge(
        1,
        statement.atoms.len() as u64 * 64 + count as u64 * 4 + 128,
    )?;
    let mut maps = Vec::with_capacity(statement.atoms.len());
    for atom in statement.atoms {
        maps.push(root_map(statement.roots, atom, work)?);
    }
    let mut gaps = Vec::with_capacity(count + 1);
    let mut points = Vec::with_capacity(count);
    let mut values = vec![false; statement.atoms.len()];
    for global in 0..=count {
        for (index, (atom, mapping)) in statement.atoms.iter().zip(&maps).enumerate() {
            work.charge(1, 1)?;
            let local_gap = mapping.partition_point(|root| *root < global);
            values[index] = atom.gaps[local_gap];
        }
        gaps.push(propose_formula(statement.formula, &values, work)?);
        if global < count {
            for (index, (atom, mapping)) in statement.atoms.iter().zip(&maps).enumerate() {
                work.charge(1, 1)?;
                values[index] = match mapping.binary_search(&global) {
                    Ok(local) => atom.points[local],
                    Err(local_gap) => atom.gaps[local_gap],
                };
            }
            points.push(propose_formula(statement.formula, &values, work)?);
        }
    }
    Ok(ConstraintCandidate { maps, gaps, points })
}

/// Independent postfix evaluator, separate from the recursive proposal lane.
/// Every branch is evaluated; there is no Boolean short-circuit admission.
fn reference_formula(
    formula: &RealFormula,
    nodes: usize,
    atoms: &[bool],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    enum Frame<'a> {
        Visit(&'a RealFormula),
        All(usize),
        Any(usize),
    }
    work.charge(1, nodes as u64 * 80 + 64)?;
    let mut pending = Vec::with_capacity(nodes);
    let mut values = Vec::with_capacity(nodes);
    pending.push(Frame::Visit(formula));
    while let Some(frame) = pending.pop() {
        work.charge(1, 1)?;
        match frame {
            Frame::Visit(RealFormula::Atom(index)) => {
                values.push(*atoms.get(*index).ok_or(invalid("reference atom index"))?);
            }
            Frame::Visit(RealFormula::All(children)) => {
                pending.push(Frame::All(children.len()));
                pending.extend(children.iter().rev().map(Frame::Visit));
            }
            Frame::Visit(RealFormula::Any(children)) => {
                pending.push(Frame::Any(children.len()));
                pending.extend(children.iter().rev().map(Frame::Visit));
            }
            Frame::All(count) | Frame::Any(count) => {
                let start = values
                    .len()
                    .checked_sub(count)
                    .ok_or(invalid("formula stack"))?;
                let value = if matches!(frame, Frame::All(_)) {
                    values[start..].iter().all(|value| *value)
                } else {
                    values[start..].iter().any(|value| *value)
                };
                values.truncate(start);
                values.push(value);
            }
        }
    }
    if values.len() != 1 {
        return Err(invalid("formula final stack"));
    }
    Ok(values[0])
}

/// Independently check source binding, complete increasing root maps, then
/// replay every atom at every common cell using advancing local-root cursors.
/// At a foreign atom's endpoint continuity retains that atom's open-gap value.
fn check_constraints(
    statement: &ConstraintStatement<'_>,
    candidate: &ConstraintCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let count = statement.roots.intervals().len();
    if statement.constraints.len() != statement.atoms.len()
        || candidate.maps.len() != statement.atoms.len()
        || candidate.gaps.len() != count + 1
        || candidate.points.len() != count
        || !same_polynomial(statement.roots.coefficients(), statement.polynomial, work)?
        || statement
            .roots
            .intervals()
            .iter()
            .any(|root| root.multiplicity != 1)
    {
        return Err(invalid("constraint partition shape or source"));
    }
    for ((constraint, atom), mapping) in statement
        .constraints
        .iter()
        .zip(statement.atoms)
        .zip(&candidate.maps)
    {
        work.charge(1, 32)?;
        // Structural equality is itself bounded verifier work, including all
        // original held children and large exact coefficients. Do not charge
        // only the final boolean comparison after generator-side admission.
        if !admit(&constraint.expression, statement.variable, work)?
            || !admit(&atom.expression, statement.variable, work)?
        {
            return Err(invalid("unadmitted original constraint at verification"));
        }
        if atom.expression != constraint.expression
            || atom.variable != *statement.variable
            || atom.relation != constraint.relation
            || atom.polynomial.gen_sym != *statement.variable
            || !same_polynomial(&atom.polynomial.coeffs, atom.roots.coefficients(), work)?
            || atom.points.len() != atom.roots.intervals().len()
            || atom.gaps.len() != atom.points.len() + 1
            || mapping.len() != atom.points.len()
            || mapping.iter().any(|index| *index >= count)
            || mapping.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid("constraint source or local-root map"));
        }
        if *mapping != root_map(statement.roots, atom, work)? {
            return Err(invalid(
                "constraint root map differs from exact factor ownership",
            ));
        }
    }
    work.charge(1, statement.atoms.len() as u64 * 32 + 64)?;
    let mut cursors = vec![0usize; statement.atoms.len()];
    let mut values = vec![false; statement.atoms.len()];
    for global in 0..=count {
        for (index, atom) in statement.atoms.iter().enumerate() {
            work.charge(1, 1)?;
            values[index] = atom.gaps[cursors[index]];
        }
        let expected =
            reference_formula(statement.formula, statement.formula_nodes, &values, work)?;
        if candidate.gaps[global] != expected {
            return Err(invalid(
                "Boolean gap decision differs from checked atom charts",
            ));
        }
        if global < count {
            for (index, atom) in statement.atoms.iter().enumerate() {
                work.charge(1, 1)?;
                if candidate.maps[index].get(cursors[index]) == Some(&global) {
                    values[index] = atom.points[cursors[index]];
                    cursors[index] += 1;
                } else {
                    values[index] = atom.gaps[cursors[index]];
                }
            }
            let expected =
                reference_formula(statement.formula, statement.formula_nodes, &values, work)?;
            if candidate.points[global] != expected {
                return Err(invalid(
                    "Boolean endpoint decision differs from checked atom charts",
                ));
            }
        }
    }
    if cursors
        .iter()
        .zip(statement.atoms)
        .any(|(cursor, atom)| *cursor != atom.points.len())
    {
        return Err(invalid("incomplete atom chart replay"));
    }
    work.meter.checkpoint()?;
    Ok(())
}

/// Solve a bounded Boolean formula of exact rational atoms over the entire
/// real line. Unsupported original expression grammar returns `Ok(None)` only
/// after every source has been preflighted. Malformed formulas, cancellation,
/// resource exhaustion and failed verification remain terminal errors.
pub fn metered_solve_real_constraints(
    constraints: &[RealConstraint],
    variable: &Symbol,
    formula: &RealFormula,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealConstraints>> {
    let mut generator_meter = CappedMeter::new(generator_meter);
    let mut verifier_meter = CappedMeter::new(verifier_meter);
    let mut generator = Work::new(&mut generator_meter)?;
    let mut verifier = Work::new(&mut verifier_meter)?;
    let formula_nodes = admit_formula(formula, constraints.len(), &mut generator)?;
    if variable.name.len() > 4096 {
        return Err(limit("symbol bytes"));
    }
    let mut supported = true;
    for constraint in constraints {
        // Deliberately do not return early on unsupported or unused atoms.
        supported &= admit(&constraint.expression, variable, &mut generator)?;
    }
    if !supported {
        return Ok(None);
    }
    generator.charge(1, constraints.len() as u64 * 256 + 64)?;
    let mut atoms = Vec::with_capacity(constraints.len());
    let mut polynomial = generator.constant(1)?;
    for constraint in constraints {
        let atom = metered_real_chart(
            &constraint.expression,
            variable,
            constraint.relation,
            generator.meter,
            verifier.meter,
        )?
        .ok_or(invalid("admitted atom grammar changed"))?;
        let gcd = checked_gcd(
            &polynomial,
            &atom.polynomial.coeffs,
            variable,
            &mut generator,
            &mut verifier,
        )?;
        let extra = checked_quotient(&atom.polynomial.coeffs, &gcd, &mut generator, &mut verifier)?;
        if polynomial.len() + extra.len() - 2 > MAX_REAL_ROOT_DEGREE {
            return Err(limit("common critical degree"));
        }
        polynomial = verifier.product(&polynomial, &extra)?;
        critical_bounds(&polynomial)?;
        atoms.push(atom);
    }
    let roots = isolate_real_roots(&polynomial, generator.meter, verifier.meter)?;
    let statement = ConstraintStatement {
        constraints,
        variable,
        formula,
        formula_nodes,
        atoms: &atoms,
        polynomial: &polynomial,
        roots: &roots,
    };
    let candidate = propose_constraints(&statement, &mut generator)?;
    check_constraints(&statement, &candidate, &mut verifier)?;
    generator.charge(
        1,
        formula_nodes as u64 * 64 + variable.name.len() as u64 * 2 + 256,
    )?;
    generator.meter.checkpoint()?;
    verifier.meter.checkpoint()?;
    Ok(Some(VerifiedRealConstraints {
        constraints: atoms
            .into_iter()
            .map(|atom| RealConstraint {
                expression: atom.expression,
                relation: atom.relation,
            })
            .collect(),
        variable: variable.clone(),
        formula: formula.clone(),
        polynomial: UnivariatePoly::new(variable.clone(), polynomial),
        roots,
        gap_inclusions: candidate.gaps,
        point_inclusions: candidate.points,
    }))
}

/// Convenience entry point retaining all intrinsic caps without a caller meter.
pub fn solve_real_constraints(
    constraints: &[RealConstraint],
    variable: &Symbol,
    formula: &RealFormula,
) -> Result<Option<VerifiedRealConstraints>> {
    metered_solve_real_constraints(
        constraints,
        variable,
        formula,
        &mut Unbounded,
        &mut Unbounded,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_composition_rejects_maps_masks_sources_and_formula_mutants() {
        let x = Symbol::new("x");
        let constraints = vec![
            RealConstraint {
                expression: Expr::Sym(x.clone()),
                relation: RealRelation::Ge,
            },
            RealConstraint {
                expression: Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(-1)]),
                relation: RealRelation::Le,
            },
            RealConstraint {
                expression: Expr::Mul(vec![
                    Expr::Sym(x.clone()),
                    Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(-1)]),
                ]),
                relation: RealRelation::Ne,
            },
        ];
        let formula = RealFormula::All(vec![RealFormula::Atom(0), RealFormula::Atom(1)]);
        let mut atoms = Vec::new();
        for constraint in &constraints {
            atoms.push(
                metered_real_chart(
                    &constraint.expression,
                    &x,
                    constraint.relation,
                    &mut Unbounded,
                    &mut Unbounded,
                )
                .unwrap()
                .unwrap(),
            );
        }
        let polynomial = vec![0.into(), (-1).into(), 1.into()];
        let roots = isolate_real_roots(&polynomial, &mut Unbounded, &mut Unbounded).unwrap();
        let statement = ConstraintStatement {
            constraints: &constraints,
            variable: &x,
            formula: &formula,
            formula_nodes: 3,
            atoms: &atoms,
            polynomial: &polynomial,
            roots: &roots,
        };
        let mut candidate =
            propose_constraints(&statement, &mut Work::new(&mut Unbounded).unwrap()).unwrap();
        let check = |candidate: &ConstraintCandidate| {
            check_constraints(
                &statement,
                candidate,
                &mut Work::new(&mut Unbounded).unwrap(),
            )
        };
        assert!(check(&candidate).is_ok());
        candidate.gaps[0] = true;
        assert!(check(&candidate).is_err());
        candidate.gaps[0] = false;
        candidate.points[0] = false;
        assert!(check(&candidate).is_err());
        candidate.points[0] = true;
        candidate.maps[0][0] = 1;
        assert!(check(&candidate).is_err());
        candidate.maps[0][0] = 0;
        candidate.maps[0].push(0);
        assert!(check(&candidate).is_err());
        candidate.maps[0].pop();
        candidate.maps[2].swap(0, 1);
        assert!(check(&candidate).is_err());
        candidate.maps[2].swap(0, 1);
        candidate.maps[2][1] = 0;
        assert!(check(&candidate).is_err());
        candidate.maps[2][1] = 1;
        assert!(check(&candidate).is_ok());
        let mut changed_constraints = constraints.clone();
        changed_constraints[0].relation = RealRelation::Gt;
        let changed_statement = ConstraintStatement {
            constraints: &changed_constraints,
            ..statement
        };
        assert!(
            check_constraints(
                &changed_statement,
                &candidate,
                &mut Work::new(&mut Unbounded).unwrap()
            )
            .is_err()
        );
        let changed_formula = RealFormula::Any(vec![RealFormula::Atom(0), RealFormula::Atom(1)]);
        let changed_statement = ConstraintStatement {
            formula: &changed_formula,
            ..statement
        };
        assert!(
            check_constraints(
                &changed_statement,
                &candidate,
                &mut Work::new(&mut Unbounded).unwrap()
            )
            .is_err()
        );
    }
}
