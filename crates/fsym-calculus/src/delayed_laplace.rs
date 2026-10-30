//! Checked finite causal sums of delayed proper rational Laplace transforms.
//!
//! Each accepted part means H(t-delay) times the shifted ordinary function.
//! Delays are exact nonnegative rationals. The reference lane independently
//! expands every original sum/product, derives each delay from polynomial
//! coefficients, and checks every complete rational inverse certificate.
//! Original rational children are retained before zero or cancellation rules.
//! The common abscissa is sufficient, not an optimal convergence boundary.
//! These bounded in-process receipts are not portable proof artifacts.

#![forbid(unsafe_code)]

use crate::rational_laplace::{
    RationalLaplaceCertificate, RationalLaplaceError, admit_rational_laplace_source,
    generate_rational_laplace_certificate, verify_rational_laplace_certificate,
};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_polys::{PolyError, bounded_polynomial_from_expr};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;

const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_TEXT: usize = 65_536;
const MAX_TERMS: usize = 32;
const MAX_DEGREE: usize = 32;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_BITS: u64 = 65_536;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;
type Result<T> = std::result::Result<T, RationalLaplaceError>;

/// Untrusted ordered coverage of every delayed term in the original source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedLaplaceTermCertificate {
    pub delay: BigRational,
    pub rational: RationalLaplaceCertificate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedLaplaceCertificate {
    pub terms: Vec<DelayedLaplaceTermCertificate>,
}

/// A shifted ordinary function; the causal surface must attach H(t-delay).
#[derive(Debug, Clone)]
pub struct VerifiedDelayedLaplacePart {
    delay: BigRational,
    shifted_function: Expr,
}

impl VerifiedDelayedLaplacePart {
    pub fn delay(&self) -> &BigRational {
        &self.delay
    }

    pub fn shifted_function(&self) -> &Expr {
        &self.shifted_function
    }

    pub fn into_parts(self) -> (BigRational, Expr) {
        (self.delay, self.shifted_function)
    }
}

/// Complete causal inverse of the original source on a sufficient half-plane.
#[derive(Debug, Clone)]
pub struct VerifiedDelayedLaplace {
    source: Expr,
    transform_variable: Symbol,
    time_variable: Symbol,
    certificate: DelayedLaplaceCertificate,
    parts: Vec<VerifiedDelayedLaplacePart>,
    sufficient_abscissa: BigRational,
}

impl VerifiedDelayedLaplace {
    pub fn source(&self) -> &Expr {
        &self.source
    }
    pub fn transform_variable(&self) -> &Symbol {
        &self.transform_variable
    }
    pub fn time_variable(&self) -> &Symbol {
        &self.time_variable
    }
    pub fn certificate(&self) -> &DelayedLaplaceCertificate {
        &self.certificate
    }
    pub fn parts(&self) -> &[VerifiedDelayedLaplacePart] {
        &self.parts
    }
    pub fn sufficient_abscissa(&self) -> &BigRational {
        &self.sufficient_abscissa
    }
    pub fn into_parts(self) -> Vec<VerifiedDelayedLaplacePart> {
        self.parts
    }
}

struct Capped<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
}

impl<'a, M: BudgetMeter> Capped<'a, M> {
    fn new(inner: &'a mut M) -> Self {
        Self {
            inner,
            used: [0; 3],
            stopped: None,
        }
    }
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = (|| {
            let mut next = self.used;
            for &(dimension, amount) in charges {
                let (index, limit) = match dimension {
                    Dimension::ComputeSteps => (0, MAX_COMPUTE),
                    Dimension::MemoryBytes => (1, MAX_MEMORY),
                    Dimension::AllocationCount => (2, MAX_COMPUTE),
                    _ => continue,
                };
                let remaining = limit.saturating_sub(next[index]);
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
        })();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

fn charge(meter: &mut impl BudgetMeter, steps: usize, bytes: usize) -> Result<()> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, steps.max(1) as u64),
        (Dimension::MemoryBytes, bytes.max(1) as u64),
        (Dimension::AllocationCount, 1),
    ])?;
    Ok(())
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn number(value: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    if value.denom().is_zero() || value.denom().is_negative() || bits(value) > MAX_BITS {
        return Err(RationalLaplaceError::ResourceLimit(
            "delayed coefficient height",
        ));
    }
    charge(meter, 1, 96 + 2 * bits(value).div_ceil(8) as usize)?;
    Ok(value.clone())
}

fn add(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    let value = a.metered_add(b, meter)?;
    number(&value, meter)
}

fn mul(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    let value = a.metered_mul(b, meter)?;
    number(&value, meter)
}

fn integer_power(value: &Expr) -> Option<i64> {
    match value {
        Expr::Integer(value) => value.to_i64(),
        Expr::Rational(value) if value.is_integer() => value.numer().to_i64(),
        _ => None,
    }
}

/// Inspect the whole borrowed original before recursion, cloning or routing.
fn admit(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    charge(meter, 1, MAX_NODES * 24)?;
    let mut text = variable.name.len().saturating_add(time.name.len());
    if text > MAX_TEXT {
        return Err(RationalLaplaceError::ResourceLimit("typed variable names"));
    }
    charge(meter, text + 128, 1)?;
    if variable == time {
        return Err(RationalLaplaceError::InvalidSource(
            "transform and time variables must be distinct",
        ));
    }
    let mut pending = vec![(source, 0usize)];
    let mut nodes = 0usize;
    while let Some((node, depth)) = pending.pop() {
        nodes += 1;
        charge(meter, 1, 1)?;
        if nodes > MAX_NODES || depth > MAX_DEPTH {
            return Err(RationalLaplaceError::ResourceLimit("delayed source shape"));
        }
        match node {
            Expr::Integer(value) if value.bits() > MAX_INPUT_BITS => {
                return Err(RationalLaplaceError::ResourceLimit("source integer height"));
            }
            Expr::Rational(value) if bits(value) > MAX_INPUT_BITS => {
                return Err(RationalLaplaceError::ResourceLimit(
                    "source rational height",
                ));
            }
            Expr::Sym(symbol) => {
                text = text.saturating_add(symbol.name.len());
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if matches!(node, Expr::Add(_) | Expr::Mul(_)) && children.is_empty() {
                    return Err(RationalLaplaceError::InvalidSource("empty arithmetic node"));
                }
                if let Expr::Function(name, _) = node {
                    text = text.saturating_add(name.len());
                }
                if children.len() > MAX_NODES.saturating_sub(nodes + pending.len()) {
                    return Err(RationalLaplaceError::ResourceLimit("delayed source fanout"));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if let Some(power) = integer_power(exponent) {
                    if power.unsigned_abs() > 4096 {
                        return Err(RationalLaplaceError::ResourceLimit("source integer power"));
                    }
                    if power <= 0 && base.is_zero() {
                        return Err(RationalLaplaceError::InvalidSource(
                            "zero base of a nonpositive power",
                        ));
                    }
                } else if matches!(exponent.as_ref(), Expr::Integer(_)) {
                    return Err(RationalLaplaceError::ResourceLimit("source integer power"));
                }
                if 2 > MAX_NODES.saturating_sub(nodes + pending.len()) {
                    return Err(RationalLaplaceError::ResourceLimit("delayed source fanout"));
                }
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Const(
                Constant::NaN
                | Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity,
            ) => {
                return Err(RationalLaplaceError::InvalidSource(
                    "nonfinite transform source",
                ));
            }
            _ => {}
        }
        if text > MAX_TEXT {
            return Err(RationalLaplaceError::ResourceLimit("delayed source text"));
        }
    }
    Ok(())
}

/// Cheap original numerator/denominator degree envelope. No cancellation or
/// polynomial factorization is performed; every original child is visited.
fn rational_degree(
    source: &Expr,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<Option<(usize, usize)>> {
    charge(meter, 1, 1)?;
    let cap = MAX_DEGREE + 1;
    let plus = |a: usize, b: usize| a.saturating_add(b).min(cap);
    let value = match source {
        Expr::Integer(_) | Expr::Rational(_) => (0, 0),
        Expr::Sym(symbol) if symbol == variable => (1, 0),
        Expr::Add(children) => {
            let mut result = (0, 0);
            for child in children {
                let Some(part) = rational_degree(child, variable, meter)? else {
                    return Ok(None);
                };
                result = (
                    plus(result.0, part.1).max(plus(part.0, result.1)),
                    plus(result.1, part.1),
                );
            }
            result
        }
        Expr::Mul(children) => {
            let mut result = (0, 0);
            for child in children {
                let Some(part) = rational_degree(child, variable, meter)? else {
                    return Ok(None);
                };
                result = (plus(result.0, part.0), plus(result.1, part.1));
            }
            result
        }
        Expr::Pow(base, exponent) => {
            let Some(power) = integer_power(exponent) else {
                return Ok(None);
            };
            if power.unsigned_abs() > MAX_DEGREE as u64 {
                return Ok(None);
            }
            let Some(base) = rational_degree(base, variable, meter)? else {
                return Ok(None);
            };
            let n = power.unsigned_abs() as usize;
            let value = (
                base.0.saturating_mul(n).min(cap),
                base.1.saturating_mul(n).min(cap),
            );
            if power < 0 { (value.1, value.0) } else { value }
        }
        _ => return Ok(None),
    };
    if value.0 > MAX_DEGREE || value.1 > MAX_DEGREE {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

/// Generator affine recurrence; the reference lane independently lowers the
/// original exponential argument to an exact polynomial coefficient map.
fn affine(
    source: &Expr,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<Option<(BigRational, BigRational)>> {
    charge(meter, 1, 128)?;
    let zero = BigRational::zero;
    let one = BigRational::one;
    let result = match source {
        Expr::Integer(value) => {
            charge(meter, 1, 96 + value.bits().div_ceil(8) as usize)?;
            (zero(), BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value) => (zero(), number(value, meter)?),
        Expr::Sym(symbol) if symbol == variable => (one(), zero()),
        Expr::Add(children) => {
            let mut out = (zero(), zero());
            for child in children {
                let Some(part) = affine(child, variable, meter)? else {
                    return Ok(None);
                };
                out = (add(&out.0, &part.0, meter)?, add(&out.1, &part.1, meter)?);
            }
            out
        }
        Expr::Mul(children) => {
            let mut out = (zero(), one());
            for child in children {
                let Some(part) = affine(child, variable, meter)? else {
                    return Ok(None);
                };
                if !out.0.is_zero() && !part.0.is_zero() {
                    return Ok(None);
                }
                let left = mul(&out.0, &part.1, meter)?;
                let right = mul(&out.1, &part.0, meter)?;
                out = (add(&left, &right, meter)?, mul(&out.1, &part.1, meter)?);
            }
            out
        }
        Expr::Pow(base, exponent) => {
            let Some(power) = integer_power(exponent) else {
                return Ok(None);
            };
            // Exponential arguments must be entire affine polynomials.
            // A held zeroth/negative power can carry a removed source hole.
            if !(1..=MAX_DEGREE as i64).contains(&power) {
                return Ok(None);
            }
            let Some(base) = affine(base, variable, meter)? else {
                return Ok(None);
            };
            if power == 1 {
                base
            } else {
                if !base.0.is_zero() {
                    return Ok(None);
                }
                let mut constant = one();
                for _ in 0..power {
                    constant = mul(&constant, &base.1, meter)?;
                }
                (zero(), constant)
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}

struct Plan<'a> {
    delay: BigRational,
    factors: Vec<&'a Expr>,
}

fn clone_plan<'a>(plan: &Plan<'a>, meter: &mut impl BudgetMeter) -> Result<Plan<'a>> {
    charge(meter, plan.factors.len(), plan.factors.len() * 16)?;
    Ok(Plan {
        delay: number(&plan.delay, meter)?,
        factors: plan.factors.clone(),
    })
}

fn generate_plans<'a>(
    source: &'a Expr,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<Option<Vec<Plan<'a>>>> {
    charge(meter, 1, 128)?;
    if rational_degree(source, variable, meter)?.is_some() {
        return Ok(Some(vec![Plan {
            delay: BigRational::zero(),
            factors: vec![source],
        }]));
    }
    let plans = match source {
        Expr::Function(name, args) if name == "exp" && args.len() == 1 => {
            let Some((slope, constant)) = affine(&args[0], variable, meter)? else {
                return Ok(None);
            };
            if !constant.is_zero() {
                return Ok(None);
            }
            let delay = BigRational::zero().metered_sub(&slope, meter)?;
            vec![Plan {
                delay,
                factors: Vec::new(),
            }]
        }
        Expr::Add(children) => {
            let mut out = Vec::new();
            for child in children {
                let Some(part) = generate_plans(child, variable, meter)? else {
                    return Ok(None);
                };
                if part.len() > MAX_TERMS.saturating_sub(out.len()) {
                    return Err(RationalLaplaceError::ResourceLimit(
                        "delayed sum term count",
                    ));
                }
                charge(meter, part.len(), part.len() * 128)?;
                out.extend(part);
            }
            out
        }
        Expr::Mul(children) => {
            let mut out = vec![Plan {
                delay: BigRational::zero(),
                factors: Vec::new(),
            }];
            for child in children {
                let Some(part) = generate_plans(child, variable, meter)? else {
                    return Ok(None);
                };
                let count = out.len().saturating_mul(part.len());
                if count > MAX_TERMS {
                    return Err(RationalLaplaceError::ResourceLimit(
                        "delayed product term count",
                    ));
                }
                charge(meter, count, count * 128)?;
                let mut next = Vec::with_capacity(count);
                for left in &out {
                    for right in &part {
                        let mut joined = clone_plan(left, meter)?;
                        joined.delay = add(&left.delay, &right.delay, meter)?;
                        charge(meter, right.factors.len(), right.factors.len() * 16)?;
                        joined.factors.extend(&right.factors);
                        next.push(joined);
                    }
                }
                out = next;
            }
            out
        }
        _ => return Ok(None),
    };
    Ok(Some(plans))
}

struct Group {
    delay: BigRational,
    source: Expr,
}

fn expression_cost(
    source: &Expr,
    meter: &mut impl BudgetMeter,
) -> Result<(usize, usize, usize, u64)> {
    charge(meter, 1, MAX_NODES * 24)?;
    let mut pending = vec![(source, 0usize)];
    let (mut nodes, mut bytes, mut depth_max, mut height) = (0usize, 0usize, 0usize, 1u64);
    while let Some((node, depth)) = pending.pop() {
        nodes += 1;
        if nodes > MAX_NODES || depth > MAX_DEPTH {
            return Err(RationalLaplaceError::ResourceLimit(
                "delayed expression growth",
            ));
        }
        depth_max = depth_max.max(depth);
        bytes = bytes.saturating_add(128);
        charge(meter, 1, 1)?;
        match node {
            Expr::Integer(n) => {
                height = height.max(n.bits());
                bytes = bytes.saturating_add(n.bits().div_ceil(8) as usize);
            }
            Expr::Rational(q) => {
                height = height.max(bits(q));
                bytes = bytes.saturating_add(2 * bits(q).div_ceil(8) as usize);
            }
            Expr::Sym(s) => {
                bytes = bytes.saturating_add(s.name.len());
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if children.len() > MAX_NODES.saturating_sub(nodes + pending.len()) {
                    return Err(RationalLaplaceError::ResourceLimit(
                        "delayed expression fanout",
                    ));
                }
                if let Expr::Function(name, _) = node {
                    bytes = bytes.saturating_add(name.len());
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Const(_) => {}
        }
    }
    Ok((nodes, bytes, depth_max, height))
}

fn copy_expression(source: &Expr, meter: &mut impl BudgetMeter) -> Result<Expr> {
    let (nodes, bytes, _, _) = expression_cost(source, meter)?;
    charge(meter, nodes, bytes)?;
    Ok(source.clone())
}

/// Equal delays are combined before inversion, retaining every original
/// rational summand. This permits polynomial-part cancellation at one delay.
fn generate_groups(
    plans: Vec<Plan<'_>>,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<Option<Vec<Group>>> {
    let mut groups: Vec<(BigRational, Vec<Expr>)> = Vec::new();
    for plan in plans {
        if plan.delay.is_negative() {
            return Ok(None);
        }
        charge(meter, plan.factors.len(), plan.factors.len() * 128)?;
        let factors = plan
            .factors
            .iter()
            .map(|source| copy_expression(source, meter))
            .collect::<Result<Vec<_>>>()?;
        let source = match factors.len() {
            0 => Expr::from_i64(1),
            1 => factors.into_iter().next().expect("one factor"),
            _ => Expr::Mul(factors),
        };
        let mut selected = None;
        for (index, (delay, _)) in groups.iter().enumerate() {
            charge(
                meter,
                bits(delay).max(bits(&plan.delay)).div_ceil(64) as usize,
                1,
            )?;
            if delay == &plan.delay {
                selected = Some(index);
                break;
            }
        }
        if let Some(index) = selected {
            charge(meter, 1, 128)?;
            groups[index].1.push(source);
        } else {
            charge(meter, 1, 256)?;
            groups.push((plan.delay, vec![source]));
        }
    }
    charge(meter, groups.len(), groups.len() * 256)?;
    let mut out = Vec::with_capacity(groups.len());
    for (delay, mut terms) in groups {
        let source = if terms.len() == 1 {
            terms.pop().expect("one term")
        } else {
            Expr::Add(terms)
        };
        if rational_degree(&source, variable, meter)?.is_none() {
            return Ok(None);
        }
        expression_cost(&source, meter)?;
        out.push(Group { delay, source });
    }
    Ok(Some(out))
}

/// The reference independently rejects every held nonpositive power in
/// exponential arguments, before polynomial lowering can erase its domain.
fn reference_entire_argument(source: &Expr, meter: &mut impl BudgetMeter) -> Result<bool> {
    charge(meter, 1, MAX_NODES * 16)?;
    let mut pending = vec![source];
    while let Some(node) = pending.pop() {
        charge(meter, 1, 1)?;
        match node {
            Expr::Pow(base, exponent) => {
                let Some(power) = integer_power(exponent) else {
                    return Ok(false);
                };
                if !(1..=MAX_DEGREE as i64).contains(&power) {
                    return Ok(false);
                }
                pending.push(base);
                pending.push(exponent);
            }
            Expr::Add(children) | Expr::Mul(children) => pending.extend(children),
            Expr::Function(_, _) | Expr::Const(_) => return Ok(false),
            _ => {}
        }
    }
    Ok(true)
}

/// Independent explicit-frontier expansion. This does not call the generator
/// recurrence, its Cartesian-product routine, or its rational source builder.
fn reference_groups(
    source: &Expr,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<Option<Vec<Group>>> {
    struct Frame<'a> {
        pending: Vec<&'a Expr>,
        factors: Vec<&'a Expr>,
        delay: BigRational,
    }
    charge(meter, 1, 256)?;
    let mut pending = vec![Frame {
        pending: vec![source],
        factors: Vec::new(),
        delay: BigRational::zero(),
    }];
    let mut groups: Vec<(BigRational, Vec<Expr>)> = Vec::new();
    let mut completed = 0usize;
    while let Some(mut frame) = pending.pop() {
        let mut split = false;
        while let Some(node) = frame.pending.pop() {
            charge(meter, 1, 16)?;
            if rational_degree(node, variable, meter)?.is_some() {
                frame.factors.push(node);
                continue;
            }
            match node {
                Expr::Add(children) => {
                    if completed + pending.len() + children.len() > MAX_TERMS {
                        return Err(RationalLaplaceError::ResourceLimit(
                            "reference delayed term count",
                        ));
                    }
                    for child in children.iter().rev() {
                        charge(
                            meter,
                            frame.pending.len() + frame.factors.len() + 1,
                            (frame.pending.len() + frame.factors.len() + 1) * 16 + 256,
                        )?;
                        let mut tail = frame.pending.clone();
                        tail.push(child);
                        pending.push(Frame {
                            pending: tail,
                            factors: frame.factors.clone(),
                            delay: number(&frame.delay, meter)?,
                        });
                    }
                    split = true;
                    break;
                }
                Expr::Mul(children) => {
                    charge(meter, children.len(), children.len() * 16)?;
                    frame.pending.extend(children.iter().rev());
                }
                Expr::Function(name, args) if name == "exp" && args.len() == 1 => {
                    if !reference_entire_argument(&args[0], meter)? {
                        return Ok(None);
                    }
                    let polynomial = match bounded_polynomial_from_expr(
                        &args[0],
                        std::slice::from_ref(variable),
                        meter,
                    ) {
                        Ok(polynomial) => polynomial,
                        Err(PolyError::NonPolynomialExpression(_)) => return Ok(None),
                        Err(error) => return Err(error.into()),
                    };
                    let mut slope = BigRational::zero();
                    for (powers, coefficient) in polynomial.terms() {
                        charge(meter, 1, 1)?;
                        if powers.len() != 1 {
                            return Err(RationalLaplaceError::Verification(
                                "delay polynomial ring",
                            ));
                        }
                        if powers[0] == 1 {
                            slope = number(coefficient, meter)?;
                        } else if !coefficient.is_zero() {
                            return Ok(None);
                        }
                    }
                    let delay = BigRational::zero().metered_sub(&slope, meter)?;
                    frame.delay = add(&frame.delay, &delay, meter)?;
                }
                _ => return Ok(None),
            }
        }
        if split {
            continue;
        }
        completed += 1;
        if completed > MAX_TERMS {
            return Err(RationalLaplaceError::ResourceLimit(
                "reference delayed term count",
            ));
        }
        if frame.delay.is_negative() {
            return Ok(None);
        }
        charge(meter, frame.factors.len(), frame.factors.len() * 128)?;
        let mut product = Vec::with_capacity(frame.factors.len());
        for factor in frame.factors {
            product.push(copy_expression(factor, meter)?);
        }
        let term = if product.is_empty() {
            Expr::from_i64(1)
        } else {
            Expr::Mul(product)
        };
        let mut found = None;
        for (index, (delay, _)) in groups.iter().enumerate() {
            if delay.metered_cmp(&frame.delay, meter)?.is_eq() {
                found = Some(index);
                break;
            }
        }
        if let Some(index) = found {
            charge(meter, 1, 128)?;
            groups[index].1.push(term);
        } else {
            charge(meter, 1, 256)?;
            groups.push((frame.delay, vec![term]));
        }
    }
    charge(meter, groups.len(), groups.len() * 256)?;
    let mut result = Vec::with_capacity(groups.len());
    for (delay, terms) in groups {
        // The reference always retains the additive and multiplicative
        // identities. Its inner rational checker derives their coefficients.
        let source = Expr::Add(terms);
        if rational_degree(&source, variable, meter)?.is_none() {
            return Ok(None);
        }
        expression_cost(&source, meter)?;
        result.push(Group { delay, source });
    }
    Ok(Some(result))
}

fn prepay_shift(
    source: &Expr,
    time: &Symbol,
    delay: &BigRational,
    output_nodes: &mut usize,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let (nodes, bytes, depth, height) = expression_cost(source, meter)?;
    let added = if delay.is_zero() {
        nodes
    } else {
        nodes.saturating_mul(3)
    };
    *output_nodes = output_nodes
        .checked_add(added + 8)
        .filter(|&value| value <= MAX_NODES)
        .ok_or(RationalLaplaceError::ResourceLimit(
            "complete delayed output nodes",
        ))?;
    if !delay.is_zero() && depth + 1 > MAX_DEPTH {
        return Err(RationalLaplaceError::ResourceLimit(
            "shifted expression depth",
        ));
    }
    // Inverse blocks use polynomial powers through degree32. Prepay both
    // direct typed substitution and possible public polynomial expansion.
    // Reject coefficient growth before it can amplify a large exact delay.
    let growth_bits = if delay.is_zero() {
        height
    } else {
        height
            .saturating_add(bits(delay).saturating_mul((MAX_DEGREE + 1) as u64))
            .saturating_add(MAX_DEGREE as u64)
    };
    if growth_bits > MAX_BITS {
        return Err(RationalLaplaceError::ResourceLimit(
            "time-shift coefficient growth",
        ));
    }
    let expansion = if delay.is_zero() { 1 } else { MAX_DEGREE + 1 };
    let copied = bytes.saturating_add(
        nodes.saturating_mul(256 + time.name.len() + 2 * growth_bits.div_ceil(8) as usize),
    );
    charge(
        meter,
        nodes.saturating_mul(expansion).saturating_mul(expansion),
        copied.saturating_mul(expansion),
    )?;
    Ok(())
}

/// Exact typed substitution only: no printer, parser, generic simplifier or
/// expansion search participates in the checked time-shift construction.
fn shift_expression(source: &Expr, time: &Symbol, displacement: &Expr) -> Expr {
    match source {
        Expr::Sym(symbol) if symbol == time => displacement.clone(),
        Expr::Add(children) => Expr::Add(
            children
                .iter()
                .map(|child| shift_expression(child, time, displacement))
                .collect(),
        ),
        Expr::Mul(children) => Expr::Mul(
            children
                .iter()
                .map(|child| shift_expression(child, time, displacement))
                .collect(),
        ),
        Expr::Pow(base, exponent) => Expr::Pow(
            Arc::new(shift_expression(base, time, displacement)),
            Arc::new(shift_expression(exponent, time, displacement)),
        ),
        Expr::Function(name, children) => Expr::Function(
            name.clone(),
            children
                .iter()
                .map(|child| shift_expression(child, time, displacement))
                .collect(),
        ),
        _ => source.clone(),
    }
}

fn verify_inner(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    certificate: DelayedLaplaceCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedDelayedLaplace> {
    admit(source, variable, time, meter)?;
    if certificate.terms.len() > MAX_TERMS {
        return Err(RationalLaplaceError::ResourceLimit(
            "delayed certificate term count",
        ));
    }
    for term in &certificate.terms {
        if term.delay.is_negative()
            || bits(&term.delay) > MAX_BITS
            || term.delay.denom().is_zero()
            || term.delay.denom().is_negative()
        {
            return Err(RationalLaplaceError::Verification(
                "invalid claimed time delay",
            ));
        }
    }
    let groups = reference_groups(source, variable, meter)?.ok_or(
        RationalLaplaceError::Verification("source is outside the delayed rational grammar"),
    )?;
    if groups.len() != certificate.terms.len() {
        return Err(RationalLaplaceError::Verification(
            "missing or duplicated delayed term",
        ));
    }
    charge(meter, groups.len(), groups.len() * 256)?;
    let mut parts = Vec::with_capacity(groups.len());
    let mut abscissa = None;
    let mut output_nodes = 0usize;
    for (group, candidate) in groups.into_iter().zip(&certificate.terms) {
        if !group.delay.metered_cmp(&candidate.delay, meter)?.is_eq() {
            return Err(RationalLaplaceError::Verification(
                "changed or reordered time delay",
            ));
        }
        let checked = verify_rational_laplace_certificate(
            &group.source,
            variable,
            time,
            &candidate.rational,
            meter,
        )?;
        let replace_bound = match &abscissa {
            None => true,
            Some(previous) => checked
                .sufficient_abscissa()
                .metered_cmp(previous, meter)?
                .is_gt(),
        };
        if replace_bound {
            abscissa = Some(number(checked.sufficient_abscissa(), meter)?);
        }
        prepay_shift(
            checked.function(),
            time,
            &group.delay,
            &mut output_nodes,
            meter,
        )?;
        let shifted_function = if group.delay.is_zero() {
            checked.into_function()
        } else {
            let negative = BigRational::zero().metered_sub(&group.delay, meter)?;
            let displacement = Expr::Add(vec![Expr::Sym(time.clone()), rational_expr(negative)]);
            shift_expression(checked.function(), time, &displacement)
        };
        parts.push(VerifiedDelayedLaplacePart {
            delay: group.delay,
            shifted_function,
        });
    }
    let source = copy_expression(source, meter)?;
    charge(meter, 2, variable.name.len() + time.name.len() + 256)?;
    let result = VerifiedDelayedLaplace {
        source,
        transform_variable: variable.clone(),
        time_variable: time.clone(),
        certificate,
        parts,
        sufficient_abscissa: abscissa.unwrap_or_else(BigRational::zero),
    };
    meter.checkpoint()?;
    Ok(result)
}

/// Take an untrusted certificate by value so retaining it cannot hide an
/// unmetered deep copy. No candidate field supplies the accepted time function.
pub fn verify_delayed_laplace_certificate(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    certificate: DelayedLaplaceCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedDelayedLaplace> {
    let mut meter = Capped::new(verifier);
    let result = verify_inner(
        source,
        transform_variable,
        time_variable,
        certificate,
        &mut meter,
    );
    if let Some(error) = meter.stopped {
        return Err(error.into());
    }
    result
}

/// None is a declared grammar/degree miss before rational factorization.
/// Every actual source, cancellation or budget failure stays terminal.
pub fn metered_inverse_delayed_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Option<VerifiedDelayedLaplace>> {
    let mut generation = Capped::new(generator);
    let mut verification = Capped::new(verifier);
    let result = (|| {
        admit(source, transform_variable, time_variable, &mut generation)?;
        let Some(plans) = generate_plans(source, transform_variable, &mut generation)? else {
            return Ok(None);
        };
        let Some(groups) = generate_groups(plans, transform_variable, &mut generation)? else {
            return Ok(None);
        };
        // Admit the complete original rational coverage before even the
        // first denominator is factored or an improper group is declined.
        for group in &groups {
            admit_rational_laplace_source(
                &group.source,
                transform_variable,
                time_variable,
                &mut generation,
            )?;
        }
        charge(&mut generation, groups.len(), groups.len() * 256)?;
        let mut terms = Vec::with_capacity(groups.len());
        for group in groups {
            let rational = generate_rational_laplace_certificate(
                &group.source,
                transform_variable,
                time_variable,
                &mut generation,
            )?;
            terms.push(DelayedLaplaceTermCertificate {
                delay: group.delay,
                rational,
            });
        }
        let result = verify_inner(
            source,
            transform_variable,
            time_variable,
            DelayedLaplaceCertificate { terms },
            &mut verification,
        )?;
        generation.checkpoint()?;
        verification.checkpoint()?;
        Ok(Some(result))
    })();
    if let Some(error) = generation.stopped.or(verification.stopped) {
        return Err(error.into());
    }
    result
}

pub fn inverse_delayed_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
) -> Result<Option<VerifiedDelayedLaplace>> {
    metered_inverse_delayed_laplace(
        source,
        transform_variable,
        time_variable,
        &mut Unbounded,
        &mut Unbounded,
    )
}
