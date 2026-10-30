//! Complete bounded unions of polynomial, rational and algebraic components.
//!
//! A split replaces one equation `p = c product(f_i^m_i)` by one branch for
//! each distinct factor. A reference coefficient convolution checks that
//! identity, and the tree checker requires every branch. Each leaf is bound
//! to an independently checked graph, finite cylinder or unit ideal. A finite
//! cylinder crosses a complete finite solution set with exactly the original
//! coordinates absent from every equation. Rational leaves retain their exact
//! nonzero parameter domains. No partial union escapes when even one leaf
//! cannot be recognized.
//!
//! Factor discovery is deliberately incomplete: typed source factors,
//! monomial factors and rational differences of two squared monomials. A
//! successful union is complete regardless of factor irreducibility. This is
//! not general multivariate factorization, regular-chain decomposition or a
//! complex algebraic-field solver. All attempts share cumulative meters.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, TermOrder, VerifiedGroebnerBasis, bounded_division,
    bounded_polynomial_from_expr, checked_groebner_basis, checked_polynomial_division,
};
use num_traits::{One, Signed, Zero};

use crate::SolverError;
use crate::polynomial_graph::{
    PolynomialGraphOutcome, VerifiedPolynomialGraph, metered_polynomial_graph_system,
};
use crate::polysys::{VerifiedFinitePolynomialSystem, metered_checked_polynomial_system};
use crate::rational_graph::{
    RationalGraphOutcome, VerifiedRationalGraph, metered_rational_graph_system,
};

const MAX_VARIABLES: usize = 8;
const MAX_EQUATIONS: usize = 64;
const MAX_CANDIDATES: usize = 128;
const MAX_TREE_NODES: usize = 63;
const MAX_COMPONENTS: usize = 32;
const MAX_DEPTH: usize = 16;
const MAX_DEGREE: usize = 256;
const MAX_TERMS: usize = 4096;
const MAX_BITS: u64 = 4096;
const MAX_OUTPUT_NODES: usize = 4096;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

type Result<T> = std::result::Result<T, SolverError>;

#[derive(Debug, Clone)]
pub enum PolynomialComponentsOutcome {
    Empty,
    Families(VerifiedPolynomialComponents),
    NotApplicable,
}

/// A complete union, bound to the original admitted polynomial equations.
/// Private fields and the absence of deserialization prevent unchecked
/// component lists from being presented as accepted solution sets.
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialComponents {
    variables: Vec<Symbol>,
    equations: Vec<MultivariatePoly>,
    families: Vec<VerifiedComponentFamily>,
}

#[derive(Debug, Clone)]
enum ComponentGraph {
    Polynomial(VerifiedPolynomialGraph),
    Rational(Box<VerifiedRationalGraph>),
    Algebraic(AlgebraicCylinderPoint),
}

/// One selected finite tuple times its original unconstrained coordinates.
/// The complete finite receipt is shared by all rows of this cylinder; the
/// tree verifier checks the whole list before any row may be published.
#[derive(Debug, Clone)]
struct AlgebraicCylinderPoint {
    variables: Vec<Symbol>,
    values: Vec<Expr>,
    free_variables: Vec<Symbol>,
    finite_system: Arc<VerifiedFinitePolynomialSystem>,
    solution_index: usize,
}

/// One complete family in an accepted union. Graph receipts prove equality
/// with their full branch varieties, not only generic subsets. Algebraic
/// cylinders retain the selected tuple and its complete finite source receipt.
/// Rational exclusions remain attached to the component that needs them.
#[derive(Debug, Clone)]
pub struct VerifiedComponentFamily {
    graph: ComponentGraph,
    unit_denominators: Vec<Expr>,
}

impl VerifiedComponentFamily {
    fn polynomial(graph: VerifiedPolynomialGraph, meter: &mut impl BudgetMeter) -> Result<Self> {
        let width = graph.variables().len();
        charge(meter, width, width * (std::mem::size_of::<Expr>() + 64))?;
        Ok(Self {
            graph: ComponentGraph::Polynomial(graph),
            unit_denominators: vec![Expr::from_i64(1); width],
        })
    }

    fn rational(graph: Box<VerifiedRationalGraph>) -> Self {
        Self {
            graph: ComponentGraph::Rational(graph),
            unit_denominators: Vec::new(),
        }
    }

    pub fn variables(&self) -> &[Symbol] {
        match &self.graph {
            ComponentGraph::Polynomial(graph) => graph.variables(),
            ComponentGraph::Rational(graph) => graph.variables(),
            ComponentGraph::Algebraic(graph) => &graph.variables,
        }
    }

    pub fn values(&self) -> &[Expr] {
        match &self.graph {
            ComponentGraph::Polynomial(graph) => graph.values(),
            ComponentGraph::Rational(graph) => graph.values(),
            ComponentGraph::Algebraic(graph) => &graph.values,
        }
    }

    pub fn free_variables(&self) -> &[Symbol] {
        match &self.graph {
            ComponentGraph::Polynomial(graph) => graph.free_variables(),
            ComponentGraph::Rational(graph) => graph.free_variables(),
            ComponentGraph::Algebraic(graph) => &graph.free_variables,
        }
    }

    /// A graph's exact defining ideal over QQ. An individual algebraic row
    /// cannot use its whole finite system's QQ ideal: that ideal may also
    /// contain conjugate points. Such rows instead retain a selected-root
    /// receipt and deliberately return `None` here.
    pub fn basis(&self) -> Option<&VerifiedGroebnerBasis> {
        match &self.graph {
            ComponentGraph::Polynomial(graph) => Some(graph.basis()),
            ComponentGraph::Rational(graph) => Some(graph.basis()),
            ComponentGraph::Algebraic(_) => None,
        }
    }

    pub fn is_rational(&self) -> bool {
        matches!(&self.graph, ComponentGraph::Rational(_))
    }

    pub fn is_algebraic(&self) -> bool {
        matches!(&self.graph, ComponentGraph::Algebraic(_))
    }

    pub fn nonzero_conditions(&self) -> &[Expr] {
        match &self.graph {
            ComponentGraph::Polynomial(_) | ComponentGraph::Algebraic(_) => &[],
            ComponentGraph::Rational(graph) => graph.nonzero_conditions(),
        }
    }

    pub fn into_values(self) -> Vec<Expr> {
        match self.graph {
            ComponentGraph::Polynomial(graph) => graph.into_values(),
            ComponentGraph::Rational(graph) => graph.into_values(),
            ComponentGraph::Algebraic(graph) => graph.values,
        }
    }

    /// Transfer checked, precharged numerator/denominator/condition parts.
    /// A bridge must preserve nonconstant denominators as held reciprocals.
    pub fn into_parts(self) -> (Vec<Expr>, Vec<Expr>, Vec<Expr>) {
        match self.graph {
            ComponentGraph::Polynomial(graph) => {
                (graph.into_values(), self.unit_denominators, Vec::new())
            }
            ComponentGraph::Rational(graph) => graph.into_parts(),
            ComponentGraph::Algebraic(graph) => (graph.values, self.unit_denominators, Vec::new()),
        }
    }
}

impl VerifiedPolynomialComponents {
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    pub fn equations(&self) -> &[MultivariatePoly] {
        &self.equations
    }

    pub fn families(&self) -> &[VerifiedComponentFamily] {
        &self.families
    }

    /// Transfer checked coordinates without copying exact values or names.
    pub fn into_values(self) -> Vec<Vec<Expr>> {
        self.families
            .into_iter()
            .map(VerifiedComponentFamily::into_values)
            .collect()
    }

    /// Each row keeps its own exact parameter exclusions. The outer result
    /// allocation and polynomial unit denominators were charged before the
    /// accepted union crossed its publication checkpoint.
    pub fn into_parts(self) -> Vec<(Vec<Expr>, Vec<Expr>, Vec<Expr>)> {
        self.families
            .into_iter()
            .map(VerifiedComponentFamily::into_parts)
            .collect()
    }
}

fn refuse(message: impl std::fmt::Display) -> SolverError {
    SolverError::IncompleteSolutionSet(format!("polynomial components refused: {message}"))
}

fn charge(meter: &mut impl BudgetMeter, steps: usize, bytes: usize) -> Result<()> {
    meter.checkpoint().map_err(refuse)?;
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(refuse)
}

/// Intrinsic caps remain cumulative even when the caller supplies Unbounded.
struct Capped<'a, M> {
    inner: &'a mut M,
    compute: u64,
    memory: u64,
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let (mut compute, mut memory) = (self.compute, self.memory);
        for &(dimension, amount) in charges {
            let (used, cap) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_COMPUTE),
                Dimension::MemoryBytes => (&mut memory, MAX_MEMORY),
                _ => continue,
            };
            let remaining = cap.saturating_sub(*used);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *used += amount;
        }
        self.inner.charge_batch(charges)?;
        self.compute = compute;
        self.memory = memory;
        Ok(())
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.inner.checkpoint()
    }
}

fn ring_bytes(variables: &[Symbol]) -> usize {
    variables.iter().map(|symbol| symbol.name.len() + 128).sum()
}

fn storage(poly: &MultivariatePoly) -> usize {
    ring_bytes(poly.generators()) + poly.terms().len() * (2048 + 4 * poly.generators().len()) + 128
}

fn copy_poly(poly: &MultivariatePoly, meter: &mut impl BudgetMeter) -> Result<MultivariatePoly> {
    charge(meter, poly.terms().len(), storage(poly))?;
    Ok(poly.clone())
}

fn copy_inputs(
    inputs: &[MultivariatePoly],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<MultivariatePoly>> {
    charge(meter, inputs.len(), inputs.len() * 128)?;
    inputs.iter().map(|poly| copy_poly(poly, meter)).collect()
}

fn validate_poly(
    poly: &MultivariatePoly,
    variables: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    charge(meter, poly.terms().len() * (variables.len() + 1), 1)?;
    let name_bytes = variables.iter().try_fold(0usize, |total, variable| {
        total.checked_add(variable.name.len())
    });
    if variables.is_empty()
        || variables.len() > MAX_VARIABLES
        || name_bytes.is_none_or(|bytes| bytes > 65_536)
        || variables
            .iter()
            .enumerate()
            .any(|(index, variable)| variables[..index].contains(variable))
        || poly.generators() != variables
        || poly.terms().len() > MAX_TERMS
    {
        return Err(refuse("polynomial ring or term bound mismatch"));
    }
    for (powers, value) in poly.terms() {
        if powers.len() != variables.len()
            || value.is_zero()
            || powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE as u64
            || value.numer().bits().max(value.denom().bits()) > MAX_BITS
        {
            return Err(refuse("polynomial degree or coefficient bound exceeded"));
        }
    }
    Ok(())
}

fn lift(inputs: &[MultivariatePoly], meter: &mut impl BudgetMeter) -> Result<Vec<Expr>> {
    inputs
        .iter()
        .map(|poly| {
            let nodes = 1 + poly
                .terms()
                .keys()
                .map(|powers| 2 + 4 * powers.iter().filter(|&&power| power != 0).count())
                .sum::<usize>();
            if nodes > MAX_OUTPUT_NODES {
                return Err(refuse("branch expression exceeds 4096 projected nodes"));
            }
            charge(meter, nodes, nodes * (2048 + ring_bytes(poly.generators())))?;
            poly.to_expr().map_err(refuse)
        })
        .collect()
}

fn rebind(
    poly: &MultivariatePoly,
    variables: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<MultivariatePoly> {
    if variables.len() != poly.generators().len()
        || variables.iter().enumerate().any(|(index, variable)| {
            !poly.generators().contains(variable) || variables[..index].contains(variable)
        })
    {
        return Err(refuse("component variables changed their typed identity"));
    }
    charge(meter, poly.terms().len() * variables.len(), storage(poly))?;
    let positions: Vec<_> = variables
        .iter()
        .map(|variable| {
            poly.generators()
                .iter()
                .position(|old| old == variable)
                .unwrap()
        })
        .collect();
    let terms = poly
        .terms()
        .iter()
        .map(|(powers, value)| {
            (
                positions.iter().map(|&index| powers[index]).collect(),
                value.clone(),
            )
        })
        .collect();
    MultivariatePoly::new(variables.to_vec(), terms).map_err(refuse)
}

fn bind_leaf(
    basis: &VerifiedGroebnerBasis,
    inputs: &[MultivariatePoly],
    variables: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let ambient = basis
        .input_basis()
        .first()
        .ok_or_else(|| refuse("a component leaf lost its ambient ring"))?
        .generators();
    if basis.order() != TermOrder::Lex || basis.input_basis().len() != inputs.len() {
        return Err(refuse("a component leaf lost an original branch equation"));
    }
    for (original, checked) in inputs.iter().zip(basis.input_basis()) {
        validate_poly(original, variables, meter)?;
        if rebind(original, ambient, meter)? != *checked {
            return Err(refuse("a component leaf is bound to different equations"));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Split {
    equation: usize,
    scale: BigRational,
    factors: Vec<(MultivariatePoly, usize)>,
}

#[derive(Debug, Clone)]
enum Tree {
    Family(VerifiedComponentFamily),
    Cylinder {
        finite_system: Arc<VerifiedFinitePolynomialSystem>,
        families: Vec<VerifiedComponentFamily>,
    },
    Empty(VerifiedGroebnerBasis),
    Split {
        claim: Split,
        children: Vec<Tree>,
    },
}

/// Direct reference convolution. It does not invoke generator multiplication,
/// polynomial division, factor search or expression simplification.
fn reference_product(
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<MultivariatePoly> {
    let count = left.terms().len().saturating_mul(right.terms().len());
    if count > 1_000_000 {
        return Err(refuse("split product convolution exceeds its bound"));
    }
    charge(meter, count, ring_bytes(left.generators()) + 128)?;
    let mut terms = BTreeMap::new();
    for (a, x) in left.terms() {
        for (b, y) in right.terms() {
            charge(meter, a.len(), a.len() * 4 + 2048)?;
            let powers: Vec<_> = a.iter().zip(b).map(|(a, b)| a + b).collect();
            if powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE as u64 {
                return Err(refuse("split product degree exceeds its bound"));
            }
            let product = x.metered_mul(y, meter).map_err(refuse)?;
            let value = match terms.get(&powers) {
                Some(previous) => product.metered_add(previous, meter).map_err(refuse)?,
                None => product,
            };
            if value.numer().bits().max(value.denom().bits()) > MAX_BITS {
                return Err(refuse("split product coefficient exceeds its bound"));
            }
            if value.is_zero() {
                terms.remove(&powers);
            } else {
                if !terms.contains_key(&powers) && terms.len() == MAX_TERMS {
                    return Err(refuse("split product term count exceeds its bound"));
                }
                terms.insert(powers, value);
            }
        }
    }
    MultivariatePoly::new(left.generators().to_vec(), terms).map_err(refuse)
}

fn verify_split(
    input: &MultivariatePoly,
    claim: &Split,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    validate_poly(input, input.generators(), meter)?;
    if claim.factors.is_empty()
        || claim.factors.len() > 2
        || claim.scale.is_zero()
        || claim.scale.numer().bits().max(claim.scale.denom().bits()) > MAX_BITS
    {
        return Err(refuse("invalid split factor count or nonzero scale"));
    }
    let degree = input
        .total_degree()
        .ok_or_else(|| refuse("cannot split the zero polynomial"))? as usize;
    let mut weighted_degree = 0usize;
    for (factor, multiplicity) in &claim.factors {
        validate_poly(factor, input.generators(), meter)?;
        let factor_degree = factor.total_degree().unwrap_or(0) as usize;
        if factor_degree == 0
            || factor_degree >= degree
            || *multiplicity == 0
            || *multiplicity > degree
        {
            return Err(refuse(
                "split factor or multiplicity makes no bounded progress",
            ));
        }
        weighted_degree = weighted_degree
            .checked_add(
                factor_degree
                    .checked_mul(*multiplicity)
                    .ok_or_else(|| refuse("split degree overflow"))?,
            )
            .filter(|&total| total <= degree)
            .ok_or_else(|| refuse("split weighted degree exceeds the original"))?;
    }
    if weighted_degree != degree {
        return Err(refuse(
            "split weighted degree does not reproduce the original",
        ));
    }
    charge(meter, 1, ring_bytes(input.generators()) + 2048)?;
    let mut terms = BTreeMap::new();
    terms.insert(vec![0; input.generators().len()], claim.scale.clone());
    let mut product = MultivariatePoly::new(input.generators().to_vec(), terms).map_err(refuse)?;
    for (factor, multiplicity) in &claim.factors {
        for _ in 0..*multiplicity {
            product = reference_product(&product, factor, meter)?;
        }
    }
    if product != *input {
        return Err(refuse(
            "split product does not reproduce the original equation",
        ));
    }
    meter.checkpoint().map_err(refuse)
}

/// Reference check for a cylinder projection. Inspect the original coefficient
/// maps directly: every dropped coordinate must have exponent zero in every
/// monomial, and every remaining coefficient must appear unchanged in the
/// exact source bound to the finite solver's receipt. This does not use the
/// generator's projection or expression-lifting implementation.
fn verify_cylinder(
    finite_system: &VerifiedFinitePolynomialSystem,
    families: &[VerifiedComponentFamily],
    inputs: &[MultivariatePoly],
    variables: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let active = finite_system.variables();
    if active.is_empty()
        || active.len() > variables.len()
        || families.len() != finite_system.solutions().len()
        || families.len() > MAX_COMPONENTS
    {
        return Err(refuse(
            "finite cylinder lost its complete coordinate or root set",
        ));
    }
    charge(meter, variables.len() * active.len(), variables.len() * 16)?;
    let positions = active
        .iter()
        .map(|variable| {
            variables
                .iter()
                .position(|original| original == variable)
                .ok_or_else(|| refuse("finite cylinder changed a typed active variable"))
        })
        .collect::<Result<Vec<_>>>()?;
    if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(refuse(
            "finite cylinder reordered or duplicated active coordinates",
        ));
    }
    for input in inputs {
        validate_poly(input, variables, meter)?;
    }
    if inputs.iter().filter(|input| !input.is_zero()).count() != finite_system.equations().len() {
        return Err(refuse("finite cylinder lost an original equation"));
    }
    for (original, projected) in inputs
        .iter()
        .filter(|input| !input.is_zero())
        .zip(finite_system.equations())
    {
        validate_poly(projected, active, meter)?;
        if original.terms().len() != projected.terms().len() {
            return Err(refuse("finite cylinder lost an original monomial"));
        }
        for (powers, coefficient) in original.terms() {
            charge(
                meter,
                variables.len() * (active.len() + 1),
                active.len() * 4 + 32,
            )?;
            if powers
                .iter()
                .enumerate()
                .any(|(column, &power)| power != 0 && !positions.contains(&column))
            {
                return Err(refuse("finite cylinder dropped a constrained coordinate"));
            }
            let powers: Vec<_> = positions.iter().map(|&column| powers[column]).collect();
            if projected.terms().get(&powers) != Some(coefficient) {
                return Err(refuse("finite cylinder changed an original coefficient"));
            }
        }
    }
    let mut receipt_work = finite_system
        .equations()
        .iter()
        .map(|equation| equation.terms().len() * (active.len() + 1))
        .sum::<usize>();
    for value in finite_system.solutions().iter().flatten() {
        let nodes = crate::polysys::expression_size(value)?;
        charge(meter, nodes, 1)?;
        receipt_work += nodes;
    }
    for (index, family) in families.iter().enumerate() {
        let ComponentGraph::Algebraic(point) = &family.graph else {
            return Err(refuse("finite cylinder substituted an unrelated component"));
        };
        charge(meter, receipt_work + variables.len() + active.len(), 1)?;
        for value in &point.values {
            charge(meter, crate::polysys::expression_size(value)?, 1)?;
        }
        if point.variables != variables
            || point.solution_index != index
            || point.finite_system.variables() != active
            || point.finite_system.equations() != finite_system.equations()
            || point.finite_system.solutions() != finite_system.solutions()
            || point.values.len() != variables.len()
            || family.unit_denominators.len() != variables.len()
            || family.unit_denominators.iter().any(|value| !value.is_one())
            || finite_system.solutions()[index].len() != active.len()
            || point.free_variables.len() != variables.len() - active.len()
        {
            return Err(refuse(
                "finite cylinder row lost its source-bound root receipt",
            ));
        }
        let mut free = point.free_variables.iter();
        for (column, variable) in variables.iter().enumerate() {
            if let Some(active_column) = positions.iter().position(|&position| position == column) {
                if point.values[column] != finite_system.solutions()[index][active_column] {
                    return Err(refuse("finite cylinder replaced or omitted an exact root"));
                }
            } else if free.next() != Some(variable)
                || !matches!(&point.values[column], Expr::Sym(symbol) if symbol == variable)
            {
                return Err(refuse(
                    "finite cylinder captured an original free coordinate",
                ));
            }
        }
    }
    meter.checkpoint().map_err(refuse)
}

fn verify_tree(
    tree: &Tree,
    inputs: &[MultivariatePoly],
    variables: &[Symbol],
    depth: usize,
    nodes: &mut usize,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    charge(meter, 1, 1)?;
    *nodes += 1;
    if depth > MAX_DEPTH || *nodes > MAX_TREE_NODES {
        return Err(refuse("component verification tree exceeds its bound"));
    }
    match tree {
        Tree::Family(family) => {
            if family.variables() != variables {
                return Err(refuse("component coordinate order or identity changed"));
            }
            match &family.graph {
                ComponentGraph::Polynomial(_) => {
                    if family.unit_denominators.len() != variables.len()
                        || family
                            .unit_denominators
                            .iter()
                            .any(|value| *value != Expr::from_i64(1))
                    {
                        return Err(refuse("polynomial component lost its unit denominators"));
                    }
                }
                ComponentGraph::Rational(_) if !family.unit_denominators.is_empty() => {
                    return Err(refuse("rational component has unrelated denominator parts"));
                }
                ComponentGraph::Rational(_) => {}
                ComponentGraph::Algebraic(_) => {
                    return Err(refuse(
                        "an algebraic row escaped its complete cylinder leaf",
                    ));
                }
            }
            bind_leaf(
                family
                    .basis()
                    .ok_or_else(|| refuse("graph has no defining ideal"))?,
                inputs,
                variables,
                meter,
            )
        }
        Tree::Cylinder {
            finite_system,
            families,
        } => verify_cylinder(finite_system, families, inputs, variables, meter),
        Tree::Empty(basis) => {
            bind_leaf(basis, inputs, variables, meter)?;
            if !basis.basis().iter().any(MultivariatePoly::is_one) {
                return Err(refuse("an empty component has no checked unit ideal"));
            }
            Ok(())
        }
        Tree::Split { claim, children } => {
            let input = inputs
                .get(claim.equation)
                .ok_or_else(|| refuse("split equation index is invalid"))?;
            verify_split(input, claim, meter)?;
            if children.len() != claim.factors.len() {
                return Err(refuse("component split omitted or added a branch"));
            }
            for ((factor, _), child) in claim.factors.iter().zip(children) {
                let mut branch = copy_inputs(inputs, meter)?;
                branch[claim.equation] = copy_poly(factor, meter)?;
                verify_tree(child, &branch, variables, depth + 1, nodes, meter)?;
            }
            Ok(())
        }
    }
}

fn add_candidate(
    candidates: &mut Vec<MultivariatePoly>,
    candidate: MultivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    if candidate.total_degree().unwrap_or(0) == 0 {
        return Ok(());
    }
    charge(
        meter,
        candidates.len() * candidate.terms().len().max(1),
        128,
    )?;
    if !candidates.contains(&candidate) {
        if candidates.len() == MAX_CANDIDATES {
            return Err(refuse("source factor candidate count exceeds 128"));
        }
        candidates.push(candidate);
    }
    Ok(())
}

fn source_candidates(
    sources: &[Expr],
    variables: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<MultivariatePoly>> {
    charge(
        meter,
        variables.len(),
        variables.len() * (ring_bytes(variables) + 2048),
    )?;
    let mut candidates = variables
        .iter()
        .map(|variable| MultivariatePoly::var(variables.to_vec(), variable).map_err(refuse))
        .collect::<Result<Vec<_>>>()?;
    charge(
        meter,
        sources.len(),
        sources.len() * std::mem::size_of::<&Expr>(),
    )?;
    let mut stack: Vec<_> = sources.iter().collect();
    let mut visited = 0;
    while let Some(node) = stack.pop() {
        charge(meter, 1, 64)?;
        visited += 1;
        if visited > MAX_EQUATIONS * 4096 {
            return Err(refuse("source factor traversal exceeds its bound"));
        }
        match node {
            Expr::Mul(children) => {
                for child in children {
                    let candidate =
                        bounded_polynomial_from_expr(child, variables, meter).map_err(refuse)?;
                    add_candidate(&mut candidates, candidate, meter)?;
                }
                charge(
                    meter,
                    children.len(),
                    children.len() * std::mem::size_of::<&Expr>(),
                )?;
                stack.extend(children);
            }
            Expr::Add(children) => {
                charge(
                    meter,
                    children.len(),
                    children.len() * std::mem::size_of::<&Expr>(),
                )?;
                stack.extend(children);
            }
            Expr::Pow(base, _) => {
                let candidate =
                    bounded_polynomial_from_expr(base, variables, meter).map_err(refuse)?;
                add_candidate(&mut candidates, candidate, meter)?;
                stack.push(base);
            }
            _ => {}
        }
    }
    Ok(candidates)
}

fn integer_square_root(value: &BigInt, meter: &mut impl BudgetMeter) -> Result<Option<BigInt>> {
    if value.is_negative() {
        return Ok(None);
    }
    if value.is_zero() {
        return Ok(Some(BigInt::zero()));
    }
    charge(meter, 1, (value.bits() as usize).div_ceil(8) + 64)?;
    let shift = u32::try_from(value.bits().div_ceil(2))
        .map_err(|_| refuse("square-root candidate bit width"))?;
    let mut current = BigInt::one() << shift;
    let two = BigInt::from(2);
    loop {
        charge(
            meter,
            value.bits() as usize,
            (value.bits() as usize + 7) / 4 + 64,
        )?;
        let (quotient, _) = fsym_core::arith::metered_div_rem(value, &current, meter)
            .map_err(refuse)?
            .ok_or_else(|| refuse("square-root candidate divided by zero"))?;
        let sum = &current + quotient;
        let (next, _) = fsym_core::arith::metered_div_rem(&sum, &two, meter)
            .map_err(refuse)?
            .ok_or_else(|| refuse("invalid square-root divisor"))?;
        if next >= current {
            let squared =
                fsym_core::arith::metered_mul(&current, &current, meter).map_err(refuse)?;
            return Ok((squared == *value).then_some(current));
        }
        current = next;
    }
}

fn binomial_candidate(
    poly: &MultivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<Option<MultivariatePoly>> {
    if poly.terms().len() != 2 {
        return Ok(None);
    }
    let mut terms = poly.terms().iter();
    let (left, a) = terms.next().unwrap();
    let (right, b) = terms.next().unwrap();
    if left.iter().chain(right).any(|power| power % 2 != 0) || a.is_negative() == b.is_negative() {
        return Ok(None);
    }
    charge(meter, left.len(), storage(poly))?;
    let ratio = (-b.clone()).metered_div(a, meter).map_err(refuse)?;
    if ratio.numer().bits().max(ratio.denom().bits()) > MAX_BITS {
        return Err(refuse("square-factor ratio exceeds the coefficient bound"));
    }
    let Some(numerator) = integer_square_root(ratio.numer(), meter)? else {
        return Ok(None);
    };
    let Some(denominator) = integer_square_root(ratio.denom(), meter)? else {
        return Ok(None);
    };
    charge(meter, 1, storage(poly))?;
    let mut factor = BTreeMap::new();
    factor.insert(
        left.iter().map(|power| power / 2).collect(),
        BigRational::one(),
    );
    let coefficient = BigRational::from_integer(numerator)
        .metered_div(&BigRational::from_integer(denominator), meter)
        .map_err(refuse)?;
    factor.insert(right.iter().map(|power| power / 2).collect(), coefficient);
    Ok(Some(
        MultivariatePoly::new(poly.generators().to_vec(), factor).map_err(refuse)?,
    ))
}

fn try_factor(
    input: &MultivariatePoly,
    factor: &MultivariatePoly,
    equation: usize,
    meter: &mut impl BudgetMeter,
) -> Result<Option<Split>> {
    let degree = input.total_degree().unwrap_or(0);
    let factor_degree = factor.total_degree().unwrap_or(0);
    if factor_degree == 0 || factor_degree >= degree {
        return Ok(None);
    }
    let mut remaining = copy_poly(input, meter)?;
    let mut multiplicity = 0;
    while remaining.total_degree().unwrap_or(0) >= factor_degree {
        let (mut quotients, remainder) = bounded_division(
            &remaining,
            std::slice::from_ref(factor),
            TermOrder::Lex,
            meter,
        )
        .map_err(refuse)?;
        if !remainder.is_zero() {
            break;
        }
        let quotient = quotients
            .pop()
            .ok_or_else(|| refuse("missing factor quotient"))?;
        if quotient.total_degree().unwrap_or(0) >= remaining.total_degree().unwrap_or(0) {
            return Err(refuse("factor search did not decrease degree"));
        }
        multiplicity += 1;
        remaining = quotient;
    }
    if multiplicity == 0 {
        return Ok(None);
    }
    let mut factors = vec![(copy_poly(factor, meter)?, multiplicity)];
    let scale = if remaining.total_degree() == Some(0) {
        charge(meter, 1, 2048)?;
        remaining.terms().values().next().unwrap().clone()
    } else {
        factors.push((remaining, 1));
        BigRational::one()
    };
    Ok(Some(Split {
        equation,
        scale,
        factors,
    }))
}

struct Generate<'a, G, V> {
    sources: &'a [Expr],
    variables: &'a [Symbol],
    candidates: Option<Vec<MultivariatePoly>>,
    nodes: usize,
    rational_leaves: bool,
    generator: &'a mut G,
    verifier: &'a mut V,
}

impl<G: BudgetMeter, V: BudgetMeter> Generate<'_, G, V> {
    fn finite_cylinder(&mut self, inputs: &[MultivariatePoly]) -> Result<Option<Tree>> {
        charge(
            self.generator,
            self.variables.len(),
            self.variables.len() * 16 + ring_bytes(self.variables),
        )?;
        let mut used = vec![false; self.variables.len()];
        for input in inputs {
            charge(
                self.generator,
                input.terms().len() * self.variables.len(),
                1,
            )?;
            for powers in input.terms().keys() {
                for (seen, &power) in used.iter_mut().zip(powers) {
                    *seen |= power != 0;
                }
            }
        }
        let positions: Vec<_> = used
            .iter()
            .enumerate()
            .filter_map(|(column, &present)| present.then_some(column))
            .collect();
        if positions.is_empty() {
            // Constant and zero ideals are already handled by the graph lane.
            return Ok(None);
        }
        let active: Vec<_> = positions
            .iter()
            .map(|&column| self.variables[column].clone())
            .collect();
        charge(self.generator, inputs.len(), inputs.len() * 128)?;
        let mut projected = Vec::with_capacity(inputs.len());
        for input in inputs {
            charge(
                self.generator,
                input.terms().len() * positions.len(),
                storage(input),
            )?;
            let terms = input
                .terms()
                .iter()
                .map(|(powers, coefficient)| {
                    (
                        positions.iter().map(|&column| powers[column]).collect(),
                        coefficient.clone(),
                    )
                })
                .collect();
            projected.push(MultivariatePoly::new(active.clone(), terms).map_err(refuse)?);
        }
        let expressions = lift(&projected, self.generator)?;
        let finite = match metered_checked_polynomial_system(
            &expressions,
            &active,
            self.generator,
            self.verifier,
        ) {
            Ok(finite) => finite,
            Err(SolverError::InfiniteSolutions | SolverError::UnsupportedDegree(_)) => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if finite.solutions().len() > MAX_COMPONENTS {
            return Err(refuse("finite cylinder expands beyond 32 components"));
        }
        // Bound and reserve all exact root copies, typed names and owned
        // coordinate vectors before constructing any published row.
        let mut nodes = finite.solutions().len() * self.variables.len() * 2;
        for row in finite.solutions() {
            if row.len() != active.len() {
                return Err(refuse("finite cylinder root width changed"));
            }
            for value in row {
                nodes = nodes
                    .checked_add(crate::polysys::expression_size(value)?)
                    .filter(|&count| count <= MAX_OUTPUT_NODES)
                    .ok_or_else(|| refuse("finite cylinder exceeds 4096 output nodes"))?;
            }
        }
        charge(
            self.generator,
            nodes,
            nodes * 2048 + finite.solutions().len() * (3 * ring_bytes(self.variables) + 512),
        )?;
        let finite_system = Arc::new(finite);
        let mut families = Vec::with_capacity(finite_system.solutions().len());
        for (solution_index, solution) in finite_system.solutions().iter().enumerate() {
            self.generator.checkpoint().map_err(refuse)?;
            let values = self
                .variables
                .iter()
                .enumerate()
                .map(|(column, variable)| {
                    match positions.iter().position(|&position| position == column) {
                        Some(index) => solution[index].clone(),
                        None => Expr::Sym(variable.clone()),
                    }
                })
                .collect();
            let free_variables = self
                .variables
                .iter()
                .enumerate()
                .filter(|(column, _)| !positions.contains(column))
                .map(|(_, variable)| variable.clone())
                .collect();
            families.push(VerifiedComponentFamily {
                graph: ComponentGraph::Algebraic(AlgebraicCylinderPoint {
                    variables: self.variables.to_vec(),
                    values,
                    free_variables,
                    finite_system: Arc::clone(&finite_system),
                    solution_index,
                }),
                unit_denominators: vec![Expr::from_i64(1); self.variables.len()],
            });
        }
        Ok(Some(Tree::Cylinder {
            finite_system,
            families,
        }))
    }

    fn remaining_leaf(
        &mut self,
        inputs: &[MultivariatePoly],
        sources: Option<&[Expr]>,
    ) -> Result<Option<Tree>> {
        if let Some(cylinder) = self.finite_cylinder(inputs)? {
            return Ok(Some(cylinder));
        }
        self.rational_leaf(inputs, sources)
    }

    fn rational_leaf(
        &mut self,
        inputs: &[MultivariatePoly],
        sources: Option<&[Expr]>,
    ) -> Result<Option<Tree>> {
        if !self.rational_leaves {
            return Ok(None);
        }
        let lifted;
        let expressions = match sources {
            Some(expressions) => expressions,
            None => {
                lifted = lift(inputs, self.generator)?;
                &lifted
            }
        };
        match metered_rational_graph_system(
            expressions,
            self.variables,
            self.generator,
            self.verifier,
        )? {
            RationalGraphOutcome::Family(family) => Ok(Some(Tree::Family(
                VerifiedComponentFamily::rational(family),
            ))),
            RationalGraphOutcome::Empty => {
                let basis =
                    checked_groebner_basis(inputs, TermOrder::Lex, self.generator, self.verifier)
                        .map_err(refuse)?;
                Ok(Some(Tree::Empty(basis)))
            }
            RationalGraphOutcome::NotApplicable => Ok(None),
        }
    }

    fn node(
        &mut self,
        inputs: &[MultivariatePoly],
        sources: Option<&[Expr]>,
        depth: usize,
    ) -> Result<Option<Tree>> {
        charge(self.generator, 1, 256)?;
        self.nodes += 1;
        if depth > MAX_DEPTH || self.nodes > MAX_TREE_NODES {
            return Err(refuse("component generation tree exceeds its bound"));
        }
        // Split monomial powers before requesting a Groebner receipt. Their
        // zero set depends on the distinct variable factors, and need not
        // force the graph proof normalizer to expand a large multiplicity.
        // The reference split checker still verifies the full original power.
        let mut found = None;
        for (equation, input) in inputs.iter().enumerate() {
            if input.terms().len() != 1 || input.total_degree().unwrap_or(0) <= 1 {
                continue;
            }
            let powers = input.terms().keys().next().unwrap();
            if let Some(index) = powers.iter().position(|&power| power != 0) {
                charge(
                    self.generator,
                    self.variables.len(),
                    ring_bytes(self.variables) + 2048,
                )?;
                let factor = MultivariatePoly::var(self.variables.to_vec(), &self.variables[index])
                    .map_err(refuse)?;
                found = try_factor(input, &factor, equation, self.generator)?;
                if found.is_some() {
                    break;
                }
            }
        }
        if found.is_none() {
            let lifted;
            let expressions = match sources {
                Some(expressions) => expressions,
                None => {
                    lifted = lift(inputs, self.generator)?;
                    &lifted
                }
            };
            match metered_polynomial_graph_system(
                expressions,
                self.variables,
                self.generator,
                self.verifier,
            )? {
                PolynomialGraphOutcome::Family(family) => {
                    return Ok(Some(Tree::Family(VerifiedComponentFamily::polynomial(
                        family,
                        self.generator,
                    )?)));
                }
                PolynomialGraphOutcome::Empty => {
                    // Retain a checked unit-ideal receipt so the tree checker can
                    // bind this empty leaf to its exact branch equations.
                    let basis = checked_groebner_basis(
                        inputs,
                        TermOrder::Lex,
                        self.generator,
                        self.verifier,
                    )
                    .map_err(refuse)?;
                    return Ok(Some(Tree::Empty(basis)));
                }
                PolynomialGraphOutcome::NotApplicable => {}
            }
            if self.candidates.is_none() {
                self.candidates = Some(source_candidates(
                    self.sources,
                    self.variables,
                    self.generator,
                )?);
            }
            for (equation, input) in inputs.iter().enumerate() {
                for candidate in self.candidates.as_ref().unwrap() {
                    if let Some(split) = try_factor(input, candidate, equation, self.generator)? {
                        found = Some(split);
                        break;
                    }
                }
                if found.is_none()
                    && let Some(candidate) = binomial_candidate(input, self.generator)?
                {
                    found = try_factor(input, &candidate, equation, self.generator)?;
                }
                if found.is_some() {
                    break;
                }
            }
        }
        let Some(claim) = found else {
            // Every failed chart shares the same outer meters; operational
            // failures propagate and cannot restart work with a fresh cap.
            return self.remaining_leaf(inputs, sources);
        };
        charge(
            self.generator,
            claim.factors.len(),
            claim.factors.len() * 256,
        )?;
        let mut children = Vec::with_capacity(claim.factors.len());
        for (factor, _) in &claim.factors {
            let mut branch = copy_inputs(inputs, self.generator)?;
            branch[claim.equation] = copy_poly(factor, self.generator)?;
            let Some(child) = self.node(&branch, None, depth + 1)? else {
                // Factoring can introduce unsupported constrained subleaves
                // even when the complete original node is one rational
                // graph. Recheck that entire original ideal; never publish
                // the successfully solved portion of the split.
                return self.remaining_leaf(inputs, sources);
            };
            children.push(child);
        }
        Ok(Some(Tree::Split { claim, children }))
    }
}

fn collect(
    tree: Tree,
    families: &mut Vec<VerifiedComponentFamily>,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    match tree {
        Tree::Family(family) => {
            if families.len() == MAX_COMPONENTS {
                return Err(refuse("component count exceeds 32"));
            }
            charge(meter, 1, 1)?;
            families.push(family);
        }
        Tree::Cylinder {
            families: points, ..
        } => {
            if points.len() > MAX_COMPONENTS.saturating_sub(families.len()) {
                return Err(refuse("finite cylinder union exceeds 32 components"));
            }
            charge(meter, points.len(), 1)?;
            families.extend(points);
        }
        Tree::Empty(_) => {}
        Tree::Split { children, .. } => {
            for child in children {
                collect(child, families, meter)?;
            }
        }
    }
    Ok(())
}

/// Every defining equation of the covering graph must be in the contained
/// graph's ideal. Each accepted rational graph equals its full branch variety,
/// including the independently proved denominator exclusions; thus the same
/// ideal inclusion is sufficient for polynomial and rational graphs. Checked
/// zero remainders prove containment; a nonzero remainder only declines this
/// optional redundancy removal.
fn contained_in(
    contained: &VerifiedComponentFamily,
    covering: &VerifiedComponentFamily,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<bool> {
    // A finite source ideal can describe several conjugate rows. It must
    // never be treated as one row's defining ideal: doing so can delete all
    // but one valid root. Exact identical rows may be deduplicated, while
    // other algebraic containment remains a conservative recognition miss.
    if contained.is_algebraic() || covering.is_algebraic() {
        charge(generator, MAX_OUTPUT_NODES, 1)?;
        return Ok(contained.variables() == covering.variables()
            && contained.values() == covering.values()
            && contained.free_variables() == covering.free_variables()
            && contained.nonzero_conditions() == covering.nonzero_conditions());
    }
    let contained_basis = contained
        .basis()
        .ok_or_else(|| refuse("missing contained graph ideal"))?;
    let covering_basis = covering
        .basis()
        .ok_or_else(|| refuse("missing covering graph ideal"))?;
    let ambient = contained_basis.input_basis()[0].generators();
    for equation in covering_basis.basis() {
        let rebound = rebind(equation, ambient, generator)?;
        let (_, remainder) = checked_polynomial_division(
            &rebound,
            contained_basis.basis(),
            TermOrder::Lex,
            generator,
            verifier,
        )
        .map_err(refuse)?;
        if !remainder.is_zero() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Recognize at most 32 polynomial graphs or explicit algebraic cylinders.
/// Cancellation or resource exhaustion always propagates; a recognition miss
/// never publishes the successfully solved portion of a split tree.
pub fn metered_polynomial_components_system(
    equations: &[Expr],
    variables: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<PolynomialComponentsOutcome> {
    metered_components_system(equations, variables, false, generator_meter, verifier_meter)
}

/// Recognize complete polynomial, rational and finite-algebraic component
/// unions. Rational denominators are explicit per component; an unsupported
/// leaf prevents the entire union from being published. The polynomial entry
/// point also admits finite cylinders but does not perform rational searches.
pub fn metered_rational_components_system(
    equations: &[Expr],
    variables: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<PolynomialComponentsOutcome> {
    metered_components_system(equations, variables, true, generator_meter, verifier_meter)
}

fn metered_components_system(
    equations: &[Expr],
    variables: &[Symbol],
    rational_leaves: bool,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<PolynomialComponentsOutcome> {
    let mut generator = Capped {
        inner: generator_meter,
        compute: 0,
        memory: 0,
    };
    let mut verifier = Capped {
        inner: verifier_meter,
        compute: 0,
        memory: 0,
    };
    charge(&mut generator, 1, 1)?;
    if variables.is_empty() || variables.len() > MAX_VARIABLES || equations.len() > MAX_EQUATIONS {
        return Err(SolverError::InvalidSystem(
            "polynomial components require 1..8 variables and at most 64 equations".into(),
        ));
    }
    let zero = bounded_polynomial_from_expr(&Expr::from_i64(0), variables, &mut generator)
        .map_err(refuse)?;
    charge(
        &mut generator,
        equations.len(),
        equations.len().max(1) * 128,
    )?;
    let mut inputs = equations
        .iter()
        .map(|source| {
            bounded_polynomial_from_expr(source, variables, &mut generator).map_err(refuse)
        })
        .collect::<Result<Vec<_>>>()?;
    if inputs.is_empty() {
        inputs.push(zero);
    }
    let tree = Generate {
        sources: equations,
        variables,
        candidates: None,
        nodes: 0,
        rational_leaves,
        generator: &mut generator,
        verifier: &mut verifier,
    }
    .node(&inputs, Some(equations), 0)?;
    let Some(tree) = tree else {
        generator.checkpoint().map_err(refuse)?;
        verifier.checkpoint().map_err(refuse)?;
        return Ok(PolynomialComponentsOutcome::NotApplicable);
    };
    let mut verified_nodes = 0;
    verify_tree(
        &tree,
        &inputs,
        variables,
        0,
        &mut verified_nodes,
        &mut verifier,
    )?;
    charge(&mut generator, MAX_COMPONENTS, MAX_COMPONENTS * 256)?;
    let mut candidates = Vec::with_capacity(MAX_COMPONENTS);
    collect(tree, &mut candidates, &mut generator)?;
    if candidates.len() > MAX_COMPONENTS {
        return Err(refuse("component count exceeds 32"));
    }
    let mut families: Vec<VerifiedComponentFamily> = Vec::new();
    for candidate in candidates {
        let mut redundant = false;
        for existing in &families {
            if contained_in(&candidate, existing, &mut generator, &mut verifier)? {
                redundant = true;
                break;
            }
        }
        if redundant {
            continue;
        }
        charge(
            &mut generator,
            families.len() + 1,
            (families.len() + 1) * 256,
        )?;
        let mut retained = Vec::with_capacity(families.len() + 1);
        for existing in families {
            if !contained_in(&existing, &candidate, &mut generator, &mut verifier)? {
                retained.push(existing);
            }
        }
        retained.push(candidate);
        families = retained;
    }
    charge(
        &mut generator,
        variables.len() + families.len(),
        ring_bytes(variables)
            + families.len() * std::mem::size_of::<(Vec<Expr>, Vec<Expr>, Vec<Expr>)>(),
    )?;
    let result = if families.is_empty() {
        PolynomialComponentsOutcome::Empty
    } else {
        PolynomialComponentsOutcome::Families(VerifiedPolynomialComponents {
            variables: variables.to_vec(),
            equations: inputs,
            families,
        })
    };
    generator.checkpoint().map_err(refuse)?;
    verifier.checkpoint().map_err(refuse)?;
    Ok(result)
}

pub fn polynomial_components_system(
    equations: &[Expr],
    variables: &[Symbol],
) -> Result<PolynomialComponentsOutcome> {
    metered_polynomial_components_system(
        equations,
        variables,
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
    )
}

pub fn rational_components_system(
    equations: &[Expr],
    variables: &[Symbol],
) -> Result<PolynomialComponentsOutcome> {
    metered_rational_components_system(
        equations,
        variables,
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
    )
}

#[cfg(test)]
mod reference_checks {
    use super::*;
    use fsym_budget::Unbounded;
    use fsym_core::{SymbolIdentity, parse};

    fn fixture() -> (Vec<Symbol>, Vec<MultivariatePoly>, Tree) {
        let variables = vec![Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
        let sources = vec![parse("x*y").unwrap(), parse("z-x-y").unwrap()];
        let inputs = sources
            .iter()
            .map(|source| bounded_polynomial_from_expr(source, &variables, &mut Unbounded).unwrap())
            .collect::<Vec<_>>();
        let tree = Generate {
            sources: &sources,
            variables: &variables,
            candidates: None,
            nodes: 0,
            rational_leaves: false,
            generator: &mut Unbounded,
            verifier: &mut Unbounded,
        }
        .node(&inputs, Some(&sources), 0)
        .unwrap()
        .unwrap();
        verify_tree(&tree, &inputs, &variables, 0, &mut 0, &mut Unbounded).unwrap();
        (variables, inputs, tree)
    }

    #[test]
    fn split_checker_rejects_missing_swapped_and_rebound_branches() {
        let (variables, inputs, tree) = fixture();
        let mut missing = tree.clone();
        let Tree::Split { children, .. } = &mut missing else {
            panic!("fixture must split");
        };
        children.pop();
        assert!(verify_tree(&missing, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        let mut swapped = tree.clone();
        let Tree::Split { children, .. } = &mut swapped else {
            unreachable!();
        };
        children.swap(0, 1);
        assert!(verify_tree(&swapped, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        let mut rebound = inputs.clone();
        rebound[0] =
            bounded_polynomial_from_expr(&parse("x*y-1").unwrap(), &variables, &mut Unbounded)
                .unwrap();
        assert!(verify_tree(&tree, &rebound, &variables, 0, &mut 0, &mut Unbounded).is_err());
        let mut typed = variables.clone();
        typed[0] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([59; 32]));
        assert!(verify_tree(&tree, &inputs, &typed, 0, &mut 0, &mut Unbounded).is_err());
    }

    #[test]
    fn split_checker_rejects_product_and_multiplicity_mutations_before_expansion() {
        let (variables, inputs, tree) = fixture();
        for multiplicity in [0, 2, usize::MAX] {
            let mut bad = tree.clone();
            let Tree::Split { claim, .. } = &mut bad else {
                unreachable!();
            };
            claim.factors[0].1 = multiplicity;
            assert!(verify_tree(&bad, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        }
        for scale in [BigRational::zero(), BigRational::from(2_i64)] {
            let mut bad = tree.clone();
            let Tree::Split { claim, .. } = &mut bad else {
                unreachable!();
            };
            claim.scale = scale;
            assert!(verify_tree(&bad, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        }
        let mut bad = tree.clone();
        let Tree::Split { claim, .. } = &mut bad else {
            unreachable!();
        };
        claim.factors[0]
            .0
            .terms
            .values_mut()
            .for_each(|coefficient| *coefficient = BigRational::from(2_i64));
        assert!(verify_tree(&bad, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        let mut bad = tree;
        let Tree::Split { claim, .. } = &mut bad else {
            unreachable!();
        };
        claim.equation = inputs.len();
        assert!(verify_tree(&bad, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
    }

    #[test]
    fn unrelated_valid_leaf_receipts_do_not_prove_the_original_branch() {
        let (variables, inputs, tree) = fixture();
        let PolynomialGraphOutcome::Family(unrelated) = metered_polynomial_graph_system(
            &[parse("x").unwrap()],
            &variables,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap() else {
            panic!("single coordinate equation is a graph");
        };
        let mut omitted = tree.clone();
        let Tree::Split { children, .. } = &mut omitted else {
            unreachable!();
        };
        children[0] =
            Tree::Family(VerifiedComponentFamily::polynomial(unrelated, &mut Unbounded).unwrap());
        assert!(verify_tree(&omitted, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        let unit =
            bounded_polynomial_from_expr(&Expr::from_i64(1), &variables, &mut Unbounded).unwrap();
        let receipt =
            checked_groebner_basis(&[unit], TermOrder::Lex, &mut Unbounded, &mut Unbounded)
                .unwrap();
        let mut empty = tree;
        let Tree::Split { children, .. } = &mut empty else {
            unreachable!();
        };
        children[0] = Tree::Empty(receipt);
        assert!(verify_tree(&empty, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
    }

    fn mixed_fixture() -> (Vec<Symbol>, Vec<MultivariatePoly>, Tree) {
        let variables = vec![Symbol::new("x"), Symbol::new("y")];
        let sources = vec![parse("x*(x*y-1)").unwrap()];
        let inputs = sources
            .iter()
            .map(|source| bounded_polynomial_from_expr(source, &variables, &mut Unbounded).unwrap())
            .collect::<Vec<_>>();
        let tree = Generate {
            sources: &sources,
            variables: &variables,
            candidates: None,
            nodes: 0,
            rational_leaves: true,
            generator: &mut Unbounded,
            verifier: &mut Unbounded,
        }
        .node(&inputs, Some(&sources), 0)
        .unwrap()
        .unwrap();
        verify_tree(&tree, &inputs, &variables, 0, &mut 0, &mut Unbounded).unwrap();
        (variables, inputs, tree)
    }

    #[test]
    fn mixed_split_requires_both_original_branches_and_their_exact_receipts() {
        let (variables, inputs, tree) = mixed_fixture();
        let mut missing = tree.clone();
        let Tree::Split { children, .. } = &mut missing else {
            panic!("the mixed fixture must preserve its factor split");
        };
        let index = children
            .iter()
            .position(|child| matches!(child, Tree::Family(family) if family.is_rational()))
            .unwrap();
        children.remove(index);
        assert!(verify_tree(&missing, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());

        let mut swapped = tree.clone();
        let Tree::Split { children, .. } = &mut swapped else {
            unreachable!()
        };
        children.swap(0, 1);
        assert!(verify_tree(&swapped, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());

        let RationalGraphOutcome::Family(unrelated) = metered_rational_graph_system(
            &[parse("x*(y-1)-1").unwrap()],
            &variables,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap() else {
            panic!("the unrelated equation must have its own valid rational receipt");
        };
        let mut replaced = tree.clone();
        let Tree::Split { children, .. } = &mut replaced else {
            unreachable!()
        };
        children[index] = Tree::Family(VerifiedComponentFamily::rational(unrelated));
        assert!(verify_tree(&replaced, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());

        let mut typed = variables.clone();
        typed[1] = Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([71; 32]));
        assert!(verify_tree(&tree, &inputs, &typed, 0, &mut 0, &mut Unbounded).is_err());
    }

    #[test]
    fn mixed_component_serialization_cannot_attach_unchecked_denominator_parts() {
        let (variables, inputs, tree) = mixed_fixture();
        for missing in [false, true] {
            let mut bad = tree.clone();
            let Tree::Split { children, .. } = &mut bad else {
                unreachable!()
            };
            let family = children
                .iter_mut()
                .find_map(|child| match child {
                    Tree::Family(family) if !family.is_rational() => Some(family),
                    _ => None,
                })
                .unwrap();
            if missing {
                family.unit_denominators.pop();
            } else {
                family.unit_denominators[0] = Expr::from_i64(0);
            }
            assert!(verify_tree(&bad, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        }
        let mut bad = tree;
        let Tree::Split { children, .. } = &mut bad else {
            unreachable!()
        };
        let family = children
            .iter_mut()
            .find_map(|child| match child {
                Tree::Family(family) if family.is_rational() => Some(family),
                _ => None,
            })
            .unwrap();
        family.unit_denominators = vec![Expr::from_i64(1); variables.len()];
        assert!(verify_tree(&bad, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
    }

    fn cylinder_fixture() -> (Vec<Symbol>, Vec<MultivariatePoly>, Tree) {
        let variables = vec![Symbol::new("x"), Symbol::new("y")];
        let sources = vec![parse("x*x-2").unwrap()];
        let inputs = sources
            .iter()
            .map(|source| bounded_polynomial_from_expr(source, &variables, &mut Unbounded).unwrap())
            .collect::<Vec<_>>();
        let tree = Generate {
            sources: &sources,
            variables: &variables,
            candidates: None,
            nodes: 0,
            rational_leaves: true,
            generator: &mut Unbounded,
            verifier: &mut Unbounded,
        }
        .node(&inputs, Some(&sources), 0)
        .unwrap()
        .unwrap();
        assert!(matches!(&tree, Tree::Cylinder { families, .. } if families.len() == 2));
        verify_tree(&tree, &inputs, &variables, 0, &mut 0, &mut Unbounded).unwrap();
        (variables, inputs, tree)
    }

    #[test]
    fn cylinder_checker_rejects_missing_duplicate_or_detached_conjugates() {
        let (variables, inputs, tree) = cylinder_fixture();
        let mut missing = tree.clone();
        let Tree::Cylinder { families, .. } = &mut missing else {
            unreachable!()
        };
        families.pop();
        assert!(verify_tree(&missing, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());

        let mut duplicated = tree.clone();
        let Tree::Cylinder { families, .. } = &mut duplicated else {
            unreachable!()
        };
        families[1] = families[0].clone();
        let ComponentGraph::Algebraic(point) = &mut families[1].graph else {
            unreachable!()
        };
        point.solution_index = 1;
        assert!(verify_tree(&duplicated, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());

        let Tree::Cylinder { mut families, .. } = tree else {
            unreachable!()
        };
        let detached = Tree::Family(families.remove(0));
        assert!(verify_tree(&detached, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
    }

    #[test]
    fn cylinder_checker_rejects_changed_roots_captured_parameters_and_domains() {
        let (variables, inputs, tree) = cylinder_fixture();
        for mutation in 0..5 {
            let mut bad = tree.clone();
            let Tree::Cylinder { families, .. } = &mut bad else {
                unreachable!()
            };
            let family = &mut families[0];
            let ComponentGraph::Algebraic(point) = &mut family.graph else {
                unreachable!()
            };
            match mutation {
                0 => point.values[0] = Expr::from_i64(2),
                1 => point.values[1] = Expr::from_i64(0),
                2 => {
                    point.free_variables[0] = Symbol::with_identity(
                        "y",
                        SymbolIdentity::from_assumptions_digest([91; 32]),
                    )
                }
                3 => family.unit_denominators[0] = Expr::from_i64(0),
                4 => point.solution_index = 1,
                _ => unreachable!(),
            }
            assert!(verify_tree(&bad, &inputs, &variables, 0, &mut 0, &mut Unbounded).is_err());
        }
    }

    #[test]
    fn cylinder_projection_is_bound_to_every_original_coefficient_and_typed_variable() {
        let (variables, inputs, tree) = cylinder_fixture();
        for source in ["x*x-3", "x*x-2+y"] {
            let changed = vec![
                bounded_polynomial_from_expr(&parse(source).unwrap(), &variables, &mut Unbounded)
                    .unwrap(),
            ];
            assert!(verify_tree(&tree, &changed, &variables, 0, &mut 0, &mut Unbounded).is_err());
        }
        let mut typed = variables.clone();
        typed[0] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([92; 32]));
        assert!(verify_tree(&tree, &inputs, &typed, 0, &mut 0, &mut Unbounded).is_err());
        let mut omitted = inputs.clone();
        omitted.push(
            bounded_polynomial_from_expr(&parse("y").unwrap(), &variables, &mut Unbounded).unwrap(),
        );
        assert!(verify_tree(&tree, &omitted, &variables, 0, &mut 0, &mut Unbounded).is_err());
    }
}
