//! Exact rational expressions in several selected real algebraic embeddings.
//!
//! A checked rational univariate representation of the Cartesian defining
//! system supplies one primitive parameter. The selected primitive root is
//! bound to every requested embedding by exact, independently checked
//! coordinate inequalities against that input's uniquely isolating interval.
//! No approximate root matching, irreducibility claim, or decimal recognition
//! participates in acceptance.
//!
//! The final source is substituted structurally, retaining all held products
//! and powers. The existing one-root reference verifier therefore checks every
//! original pole and zero-power guard before cancellation. This is an in-process
//! composition of immutable checked receipts, not a portable RUR certificate.
//! Both lanes retain cumulative caps across isolation, separation, rejected
//! embedding candidates, source conversion, and final publication.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::real_roots::{
    MAX_REAL_ROOT_DEGREE, MAX_REAL_ROOT_DIGITS, MAX_REAL_ROOT_INPUT_BITS, RealRootError,
    RealRootIsolationCertificate, VerifiedRealRoots, isolate_real_roots,
    verify_real_root_isolation,
};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::MultivariatePoly;
use num_traits::Zero;
use thiserror::Error;

use crate::SolverError;
use crate::real_equation::{
    AlgebraicValueCertificate, RealEquationError, VerifiedAlgebraicValue,
    metered_evaluate_real_algebraic, verify_real_algebraic_value,
};
use crate::real_rur::{VerifiedRealRurSystem, metered_solve_real_rur_system};

const MAX_BINDINGS: usize = 8;
const MAX_NODES: usize = 2048;
const MAX_DEPTH: usize = 64;
const MAX_FANOUT: usize = 256;
const MAX_WORK_BITS: u64 = 32_768;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

type Result<T> = std::result::Result<T, AlgebraicExpressionError>;

/// A typed symbol bound to a distinct-real-root index of an exact polynomial.
/// Coefficients are ascending; the original scaling and trailing zeros remain
/// part of the retained input. Equal printed names do not alias typed symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealAlgebraicBinding {
    pub variable: Symbol,
    pub coefficients: Vec<BigRational>,
    pub index: usize,
}

#[derive(Debug, Error)]
pub enum AlgebraicExpressionError {
    #[error("algebraic expression resource limit exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("invalid algebraic expression certificate: {0}")]
    InvalidCertificate(&'static str),
    #[error(transparent)]
    Roots(#[from] RealRootError),
    #[error(transparent)]
    Value(#[from] RealEquationError),
    #[error(transparent)]
    System(#[from] SolverError),
    #[error("algebraic expression work stopped: {0}")]
    Meter(#[from] MeterError),
}

/// An untrusted composition proposal. The RUR itself must already have crossed
/// its independent acceptance boundary; other fields can be freely mutated
/// and confer no acceptance until checked against the explicit input source.
#[derive(Debug, Clone)]
pub struct AlgebraicExpressionCertificate {
    pub representation: Arc<VerifiedRealRurSystem>,
    pub input_roots: Vec<RealRootIsolationCertificate>,
    pub primitive_index: usize,
    /// For input i: coordinate[i] - lower[i], upper[i] - coordinate[i].
    /// Each checked value must be nonnegative at the selected primitive root.
    pub embedding_checks: Vec<(AlgebraicValueCertificate, AlgebraicValueCertificate)>,
    pub value: AlgebraicValueCertificate,
}

/// A source-bound exact enclosure and correctly rounded decimal for the
/// requested tuple of real algebraic embeddings.
#[derive(Debug, Clone)]
pub struct VerifiedAlgebraicExpression {
    expression: Expr,
    bindings: Vec<RealAlgebraicBinding>,
    precision: u32,
    certificate: AlgebraicExpressionCertificate,
}

impl VerifiedAlgebraicExpression {
    pub fn expression(&self) -> &Expr {
        &self.expression
    }
    pub fn bindings(&self) -> &[RealAlgebraicBinding] {
        &self.bindings
    }
    pub fn precision(&self) -> u32 {
        self.precision
    }
    pub fn lower(&self) -> &BigRational {
        &self.certificate.value.lower
    }
    pub fn upper(&self) -> &BigRational {
        &self.certificate.value.upper
    }
    pub fn decimal(&self) -> &str {
        &self.certificate.value.decimal
    }
    pub fn certificate(&self) -> &AlgebraicExpressionCertificate {
        &self.certificate
    }
}

fn limit(reason: &'static str) -> AlgebraicExpressionError {
    AlgebraicExpressionError::ResourceLimit(reason)
}
fn invalid(reason: &'static str) -> AlgebraicExpressionError {
    AlgebraicExpressionError::InvalidCertificate(reason)
}
fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

struct Capped<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let mut next = self.used;
        for &(dimension, amount) in charges {
            let (index, cap) = match dimension {
                Dimension::ComputeSteps => (0, MAX_COMPUTE),
                Dimension::MemoryBytes => (1, MAX_MEMORY),
                Dimension::AllocationCount => (2, MAX_COMPUTE),
                _ => continue,
            };
            let remaining = cap.saturating_sub(next[index]);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            next[index] += amount;
        }
        self.inner.charge_batch(charges)?;
        self.used = next;
        Ok(())
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.inner.checkpoint()
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
}

impl<M: BudgetMeter> Work<'_, M> {
    fn charge(&mut self, steps: usize, bytes: usize) -> Result<()> {
        self.meter.checkpoint()?;
        self.meter.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }

    fn scalar(&mut self, value: &BigRational) -> Result<()> {
        if bits(value) > MAX_WORK_BITS {
            return Err(limit("intermediate rational bits"));
        }
        self.charge(
            1,
            96 + (value.numer().bits() + value.denom().bits()).div_ceil(8) as usize,
        )
    }

    fn symbol(&mut self, symbol: &Symbol) -> Result<()> {
        if symbol.name.len() > 4096 {
            return Err(limit("symbol bytes"));
        }
        self.charge(1, symbol.name.len() + 128)
    }
}

/// Preflight every original node, including unsupported children hidden by a
/// held zero product/power. No RUR generation occurs before this complete walk.
fn admit<M: BudgetMeter>(
    expression: &Expr,
    bindings: &[RealAlgebraicBinding],
    digits: u32,
    work: &mut Work<'_, M>,
) -> Result<bool> {
    if !(1..=MAX_REAL_ROOT_DIGITS).contains(&digits) {
        return Err(RealRootError::InvalidPrecision.into());
    }
    if bindings.is_empty() || bindings.len() > MAX_BINDINGS {
        return Err(limit("binding count"));
    }
    work.charge(bindings.len(), bindings.len() * 128)?;
    let mut degree_product = 1usize;
    for (i, binding) in bindings.iter().enumerate() {
        work.symbol(&binding.variable)?;
        if bindings[..i]
            .iter()
            .any(|other| other.variable == binding.variable)
        {
            return Err(invalid("duplicate typed binding variable"));
        }
        if binding.coefficients.is_empty() || binding.coefficients.len() > MAX_REAL_ROOT_DEGREE + 1
        {
            return Err(limit("defining coefficient count"));
        }
        work.charge(binding.coefficients.len(), binding.coefficients.len() * 64)?;
        for coefficient in &binding.coefficients {
            if bits(coefficient) > MAX_REAL_ROOT_INPUT_BITS {
                return Err(limit("defining coefficient bits"));
            }
            work.scalar(coefficient)?;
        }
        let degree = binding
            .coefficients
            .iter()
            .rposition(|coefficient| !coefficient.is_zero())
            .ok_or(RealRootError::ZeroPolynomial)?;
        if degree == 0 {
            return Err(RealRootError::InvalidIndex.into());
        }
        degree_product = degree_product
            .checked_mul(degree)
            .filter(|product| *product <= MAX_REAL_ROOT_DEGREE)
            .ok_or_else(|| limit("Cartesian defining degree"))?;
    }
    work.charge(1, 32)?;
    let mut pending = vec![(expression, 0usize)];
    let mut count = 0;
    let mut supported = true;
    while let Some((node, depth)) = pending.pop() {
        count += 1;
        if count > MAX_NODES || depth > MAX_DEPTH {
            return Err(limit("source nodes/depth"));
        }
        work.charge(1, 128)?; // Includes the retained source clone.
        let children: &[Expr] = match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_REAL_ROOT_INPUT_BITS {
                    return Err(limit("source integer bits"));
                }
                work.charge(1, 32 + value.bits().div_ceil(8) as usize)?;
                &[]
            }
            Expr::Rational(value) => {
                if bits(value) > MAX_REAL_ROOT_INPUT_BITS {
                    return Err(limit("source rational bits"));
                }
                work.scalar(value)?;
                &[]
            }
            Expr::Sym(symbol) => {
                work.symbol(symbol)?;
                supported &= bindings.iter().any(|binding| &binding.variable == symbol);
                &[]
            }
            Expr::Add(children) | Expr::Mul(children) => children,
            Expr::Pow(base, exponent) => {
                match exponent.as_ref() {
                    Expr::Integer(value) => {
                        if value
                            .to_i64()
                            .is_none_or(|n| n.unsigned_abs() > MAX_REAL_ROOT_DEGREE as u64)
                        {
                            return Err(limit("source integer exponent"));
                        }
                    }
                    _ => supported = false,
                }
                if pending.len() > MAX_NODES - 2 {
                    return Err(limit("source pending nodes"));
                }
                work.charge(1, 64)?;
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
                continue;
            }
            Expr::Function(name, children) => {
                if name.len() > 4096 {
                    return Err(limit("source function bytes"));
                }
                supported = false;
                children
            }
            Expr::Const(_) => {
                supported = false;
                &[]
            }
        };
        if children.len() > MAX_FANOUT || pending.len() + children.len() > MAX_NODES {
            return Err(limit("source fanout"));
        }
        work.charge(1, children.len() * 32)?;
        pending.extend(children.iter().map(|child| (child, depth + 1)));
    }
    Ok(supported)
}

fn polynomial_expression<M: BudgetMeter>(
    coefficients: &[BigRational],
    variable: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    if coefficients.is_empty() || coefficients.len() > MAX_REAL_ROOT_DEGREE + 1 {
        return Err(limit("coordinate degree"));
    }
    work.charge(coefficients.len(), coefficients.len() * 320)?;
    let mut terms = Vec::with_capacity(coefficients.len());
    for (degree, coefficient) in coefficients.iter().enumerate() {
        work.scalar(coefficient)?;
        if coefficient.is_zero() {
            continue;
        }
        let constant = Expr::Rational(coefficient.clone());
        if degree == 0 {
            terms.push(constant);
        } else {
            work.symbol(variable)?;
            let power = if degree == 1 {
                Expr::Sym(variable.clone())
            } else {
                Expr::Pow(
                    Arc::new(Expr::Sym(variable.clone())),
                    Arc::new(Expr::from_i64(degree as i64)),
                )
            };
            terms.push(Expr::Mul(vec![constant, power]));
        }
    }
    if terms.is_empty() {
        Ok(Expr::from_i64(0))
    } else if terms.len() == 1 {
        Ok(terms.pop().unwrap())
    } else {
        Ok(Expr::Add(terms))
    }
}

fn source_polynomials<M: BudgetMeter>(
    bindings: &[RealAlgebraicBinding],
    work: &mut Work<'_, M>,
) -> Result<Vec<MultivariatePoly>> {
    work.charge(bindings.len(), bindings.len() * 128)?;
    let mut result = Vec::with_capacity(bindings.len());
    for (index, binding) in bindings.iter().enumerate() {
        work.charge(binding.coefficients.len(), binding.coefficients.len() * 160)?;
        let mut terms = BTreeMap::new();
        for (degree, coefficient) in binding.coefficients.iter().enumerate() {
            work.scalar(coefficient)?;
            if !coefficient.is_zero() {
                work.charge(bindings.len(), bindings.len() * 4)?;
                let mut powers = vec![0; bindings.len()];
                powers[index] = degree as u32;
                terms.insert(powers, coefficient.clone());
            }
        }
        for source in bindings {
            work.symbol(&source.variable)?;
        }
        result.push(MultivariatePoly {
            generators: bindings
                .iter()
                .map(|source| source.variable.clone())
                .collect(),
            terms,
        });
    }
    Ok(result)
}

fn substitute<M: BudgetMeter>(
    node: &Expr,
    bindings: &[RealAlgebraicBinding],
    coordinates: &[Expr],
    work: &mut Work<'_, M>,
    remaining: &mut usize,
    depth: usize,
) -> Result<Expr> {
    if *remaining == 0 || depth > MAX_DEPTH {
        return Err(limit("substituted source nodes/depth"));
    }
    *remaining -= 1;
    work.charge(1, 128)?;
    match node {
        Expr::Sym(symbol) => {
            let index = bindings
                .iter()
                .position(|binding| &binding.variable == symbol)
                .ok_or_else(|| invalid("unbound source symbol"))?;
            copy_coordinate(&coordinates[index], work, remaining, depth)
        }
        Expr::Add(children) | Expr::Mul(children) => {
            work.charge(children.len(), children.len() * 64)?;
            let children = children
                .iter()
                .map(|child| substitute(child, bindings, coordinates, work, remaining, depth + 1))
                .collect::<Result<Vec<_>>>()?;
            Ok(if matches!(node, Expr::Add(_)) {
                Expr::Add(children)
            } else {
                Expr::Mul(children)
            })
        }
        Expr::Pow(base, exponent) => Ok(Expr::Pow(
            Arc::new(substitute(
                base,
                bindings,
                coordinates,
                work,
                remaining,
                depth + 1,
            )?),
            Arc::new(substitute(
                exponent,
                bindings,
                coordinates,
                work,
                remaining,
                depth + 1,
            )?),
        )),
        Expr::Integer(value) => {
            work.charge(1, 32 + value.bits().div_ceil(8) as usize)?;
            Ok(node.clone())
        }
        Expr::Rational(value) => {
            work.scalar(value)?;
            Ok(node.clone())
        }
        _ => Err(invalid("unsupported original source")),
    }
}

/// Coordinates are newly built bounded polynomial trees. Walk before cloning
/// so repeated source occurrences cannot bypass the output/retained-tree cap.
fn copy_coordinate<M: BudgetMeter>(
    expression: &Expr,
    work: &mut Work<'_, M>,
    remaining: &mut usize,
    depth: usize,
) -> Result<Expr> {
    work.charge(1, 32)?;
    let mut pending = vec![(expression, depth)];
    while let Some((node, depth)) = pending.pop() {
        if *remaining == 0 || depth > MAX_DEPTH {
            return Err(limit("substituted source nodes/depth"));
        }
        *remaining -= 1;
        work.charge(1, 128)?;
        match node {
            Expr::Integer(value) => work.charge(1, 32 + value.bits().div_ceil(8) as usize)?,
            Expr::Rational(value) => work.scalar(value)?,
            Expr::Sym(symbol) => work.symbol(symbol)?,
            Expr::Add(children) | Expr::Mul(children) => {
                work.charge(children.len(), children.len() * 32)?;
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                work.charge(1, 64)?;
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            _ => return Err(invalid("nonpolynomial RUR coordinate")),
        }
    }
    Ok(expression.clone())
}

fn difference<M: BudgetMeter>(left: &Expr, right: &Expr, work: &mut Work<'_, M>) -> Result<Expr> {
    work.charge(1, 384)?;
    let mut remaining = MAX_NODES - 3;
    let left = copy_coordinate(left, work, &mut remaining, 1)?;
    let right = copy_coordinate(right, work, &mut remaining, 2)?;
    Ok(Expr::Add(vec![
        left,
        Expr::Mul(vec![Expr::from_i64(-1), right]),
    ]))
}

fn coordinate_sources<M: BudgetMeter>(
    representation: &VerifiedRealRurSystem,
    work: &mut Work<'_, M>,
) -> Result<Vec<Expr>> {
    work.charge(
        representation.coordinates().len(),
        representation.coordinates().len() * 64,
    )?;
    representation
        .coordinates()
        .iter()
        .map(|coordinate| polynomial_expression(coordinate, representation.parameter(), work))
        .collect()
}

fn binding_differences<M: BudgetMeter>(
    coordinates: &[Expr],
    bindings: &[RealAlgebraicBinding],
    roots: &[VerifiedRealRoots],
    work: &mut Work<'_, M>,
) -> Result<Vec<(Expr, Expr)>> {
    work.charge(bindings.len(), bindings.len() * 128)?;
    bindings
        .iter()
        .zip(roots)
        .zip(coordinates)
        .map(|((binding, roots), coordinate)| {
            let interval = roots
                .intervals()
                .get(binding.index)
                .ok_or(RealRootError::InvalidIndex)?;
            work.scalar(&interval.lower)?;
            work.scalar(&interval.upper)?;
            Ok((
                difference(coordinate, &Expr::Rational(interval.lower.clone()), work)?,
                difference(&Expr::Rational(interval.upper.clone()), coordinate, work)?,
            ))
        })
        .collect()
}

fn checked_value<G: BudgetMeter, V: BudgetMeter>(
    source: &Expr,
    representation: &VerifiedRealRurSystem,
    index: usize,
    digits: u32,
    generator: &mut G,
    verifier: &mut V,
) -> Result<VerifiedAlgebraicValue> {
    metered_evaluate_real_algebraic(
        source,
        representation.parameter(),
        representation.eliminant(),
        index,
        digits,
        generator,
        verifier,
    )?
    .ok_or_else(|| invalid("substituted source escaped the exact rational grammar"))
}

fn preflight_root_certificate<M: BudgetMeter>(
    certificate: &RealRootIsolationCertificate,
    work: &mut Work<'_, M>,
) -> Result<()> {
    if certificate.intervals.len() > MAX_REAL_ROOT_DEGREE {
        return Err(limit("root certificate count"));
    }
    work.charge(
        certificate.intervals.len(),
        certificate.intervals.len() * 96,
    )?;
    for interval in &certificate.intervals {
        work.scalar(&interval.lower)?;
        work.scalar(&interval.upper)?;
    }
    Ok(())
}

fn preflight_value_certificate<M: BudgetMeter>(
    certificate: &AlgebraicValueCertificate,
    digits: u32,
    work: &mut Work<'_, M>,
) -> Result<()> {
    if certificate.decimal.len() > digits as usize + 64 {
        return Err(limit("decimal certificate bytes"));
    }
    preflight_root_certificate(&certificate.roots, work)?;
    work.scalar(&certificate.lower)?;
    work.scalar(&certificate.upper)?;
    if let Some(exact) = &certificate.exact_value {
        work.scalar(exact)?;
    }
    work.charge(1, certificate.decimal.len() + 128)
}

/// Independently check the original source, exact Cartesian ideal binding,
/// selected embeddings, original guards, enclosure and final decimal. This
/// invokes no RUR search and accepts no printed name or numeric proximity.
pub fn verify_real_algebraic_expression(
    expression: &Expr,
    bindings: &[RealAlgebraicBinding],
    digits: u32,
    certificate: &AlgebraicExpressionCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedAlgebraicExpression> {
    let mut capped = Capped {
        inner: meter,
        used: [0; 3],
    };
    let mut work = Work { meter: &mut capped };
    if !admit(expression, bindings, digits, &mut work)? {
        return Err(invalid("unsupported original source"));
    }
    if certificate.input_roots.len() != bindings.len()
        || certificate.embedding_checks.len() != bindings.len()
    {
        return Err(invalid("embedding witness cardinality"));
    }
    let representation = certificate.representation.as_ref();
    if representation.variables().len() != bindings.len()
        || representation.coordinates().len() != bindings.len()
        || representation
            .variables()
            .iter()
            .zip(bindings)
            .any(|(a, b)| a != &b.variable)
        || representation.equations() != source_polynomials(bindings, &mut work)?
    {
        return Err(invalid("RUR is not the original Cartesian defining system"));
    }
    if certificate.primitive_index >= representation.roots().intervals().len() {
        return Err(RealRootError::InvalidIndex.into());
    }
    work.charge(bindings.len(), bindings.len() * 128)?;
    let mut roots = Vec::with_capacity(bindings.len());
    for (binding, source_roots) in bindings.iter().zip(&certificate.input_roots) {
        preflight_root_certificate(source_roots, &mut work)?;
        roots.push(verify_real_root_isolation(
            &binding.coefficients,
            source_roots,
            work.meter,
        )?);
    }
    let coordinates = coordinate_sources(representation, &mut work)?;
    let differences = binding_differences(&coordinates, bindings, &roots, &mut work)?;
    for ((lower_source, upper_source), (lower_certificate, upper_certificate)) in
        differences.iter().zip(&certificate.embedding_checks)
    {
        for (source, proposal) in [
            (lower_source, lower_certificate),
            (upper_source, upper_certificate),
        ] {
            preflight_value_certificate(proposal, 1, &mut work)?;
            let value = verify_real_algebraic_value(
                source,
                representation.parameter(),
                representation.eliminant(),
                certificate.primitive_index,
                1,
                proposal,
                work.meter,
            )?;
            if value.lower() < &BigRational::zero() {
                return Err(invalid("coordinate is outside the requested root interval"));
            }
        }
    }
    let mut remaining = MAX_NODES;
    let source = substitute(
        expression,
        bindings,
        &coordinates,
        &mut work,
        &mut remaining,
        0,
    )?;
    preflight_value_certificate(&certificate.value, digits, &mut work)?;
    verify_real_algebraic_value(
        &source,
        representation.parameter(),
        representation.eliminant(),
        certificate.primitive_index,
        digits,
        &certificate.value,
        work.meter,
    )?;
    // The checked RUR is retained through Arc, never cloned without accounting
    // for its private proof storage. Every other retained field was preflighted.
    work.charge(1, 256)?;
    let accepted = VerifiedAlgebraicExpression {
        expression: expression.clone(),
        bindings: bindings.to_vec(),
        precision: digits,
        certificate: certificate.clone(),
    };
    work.meter.checkpoint()?;
    Ok(accepted)
}

/// Evaluate exact rational arithmetic at several independently selected real
/// algebraic roots. `None` means unsupported source or no checked separating
/// representation in the bounded search. Domain, resource, verification and
/// cancellation errors propagate; they never become a candidate or a retry
/// with fresh accounting.
pub fn metered_evaluate_real_algebraic_expression(
    expression: &Expr,
    bindings: &[RealAlgebraicBinding],
    digits: u32,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedAlgebraicExpression>> {
    let mut generator = Capped {
        inner: generator_meter,
        used: [0; 3],
    };
    let mut verifier = Capped {
        inner: verifier_meter,
        used: [0; 3],
    };
    if !admit(
        expression,
        bindings,
        digits,
        &mut Work {
            meter: &mut generator,
        },
    )? {
        generator.checkpoint()?;
        verifier.checkpoint()?;
        return Ok(None);
    }
    Work {
        meter: &mut generator,
    }
    .charge(bindings.len(), bindings.len() * 128)?;
    let mut roots = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let isolated = isolate_real_roots(&binding.coefficients, &mut generator, &mut verifier)?;
        if binding.index >= isolated.intervals().len() {
            return Err(RealRootError::InvalidIndex.into());
        }
        roots.push(isolated);
    }
    Work {
        meter: &mut generator,
    }
    .charge(bindings.len(), bindings.len() * 128)?;
    let equations = bindings
        .iter()
        .map(|binding| {
            polynomial_expression(
                &binding.coefficients,
                &binding.variable,
                &mut Work {
                    meter: &mut generator,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for binding in bindings {
        Work {
            meter: &mut generator,
        }
        .symbol(&binding.variable)?;
    }
    let variables = bindings
        .iter()
        .map(|binding| binding.variable.clone())
        .collect::<Vec<_>>();
    let Some(representation) =
        metered_solve_real_rur_system(&equations, &variables, &mut generator, &mut verifier)?
    else {
        generator.checkpoint()?;
        verifier.checkpoint()?;
        return Ok(None);
    };
    let coordinates = coordinate_sources(
        &representation,
        &mut Work {
            meter: &mut generator,
        },
    )?;
    let differences = binding_differences(
        &coordinates,
        bindings,
        &roots,
        &mut Work {
            meter: &mut generator,
        },
    )?;
    let mut selected = None;
    for index in 0..representation.roots().intervals().len() {
        Work {
            meter: &mut generator,
        }
        .charge(bindings.len(), bindings.len() * 128)?;
        let mut checks = Vec::with_capacity(bindings.len());
        for (lower_source, upper_source) in &differences {
            let lower = checked_value(
                lower_source,
                &representation,
                index,
                1,
                &mut generator,
                &mut verifier,
            )?;
            if lower.lower() < &BigRational::zero() {
                break;
            }
            let upper = checked_value(
                upper_source,
                &representation,
                index,
                1,
                &mut generator,
                &mut verifier,
            )?;
            if upper.lower() < &BigRational::zero() {
                break;
            }
            preflight_value_certificate(
                lower.certificate(),
                1,
                &mut Work {
                    meter: &mut generator,
                },
            )?;
            preflight_value_certificate(
                upper.certificate(),
                1,
                &mut Work {
                    meter: &mut generator,
                },
            )?;
            checks.push((lower.certificate().clone(), upper.certificate().clone()));
        }
        if checks.len() == bindings.len() {
            selected = Some((index, checks));
            break;
        }
    }
    let (primitive_index, embedding_checks) =
        selected.ok_or_else(|| invalid("requested real embedding missing from complete RUR"))?;
    let mut remaining = MAX_NODES;
    let source = substitute(
        expression,
        bindings,
        &coordinates,
        &mut Work {
            meter: &mut generator,
        },
        &mut remaining,
        0,
    )?;
    let value = checked_value(
        &source,
        &representation,
        primitive_index,
        digits,
        &mut generator,
        &mut verifier,
    )?;
    preflight_value_certificate(
        value.certificate(),
        digits,
        &mut Work {
            meter: &mut generator,
        },
    )?;
    for root in &roots {
        preflight_root_certificate(
            root.certificate(),
            &mut Work {
                meter: &mut generator,
            },
        )?;
    }
    Work {
        meter: &mut generator,
    }
    .charge(1, 256)?;
    let certificate = AlgebraicExpressionCertificate {
        representation: Arc::new(representation),
        input_roots: roots
            .iter()
            .map(|root| root.certificate().clone())
            .collect(),
        primitive_index,
        embedding_checks,
        value: value.certificate().clone(),
    };
    let accepted = verify_real_algebraic_expression(
        expression,
        bindings,
        digits,
        &certificate,
        &mut verifier,
    )?;
    generator.checkpoint()?;
    verifier.checkpoint()?;
    Ok(Some(accepted))
}

pub fn evaluate_real_algebraic_expression(
    expression: &Expr,
    bindings: &[RealAlgebraicBinding],
    digits: u32,
) -> Result<Option<VerifiedAlgebraicExpression>> {
    metered_evaluate_real_algebraic_expression(
        expression,
        bindings,
        digits,
        &mut Unbounded,
        &mut Unbounded,
    )
}
