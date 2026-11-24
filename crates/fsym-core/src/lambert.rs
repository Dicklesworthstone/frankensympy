//! Independent complete real solving for a bounded affine exponential form.
//!
//! A candidate supplies only six rational coefficients. The verifier replays
//! the complete original tree into a dense formal polynomial in x and t,
//! checking that every t denotes the same exp(e*x+f), and accepts precisely
//! a*x+b+(c*x+d)*t with e != 0 and exactly one of a,c nonzero. Every admitted
//! source is defined on the whole real line; original reciprocal expressions
//! involving x are refused before a zero multiplier can erase their domain.
//!
//! An invertible affine change of variable reduces these equations to
//! w*exp(w)=q*exp(r). On the real line w*exp(w) decreases on (-infinity,-1),
//! increases on (-1,infinity), has minimum -1/e, and tends to zero from below
//! at negative infinity. These facts establish the complete branch count.
//! Negative arguments are compared to the minimum with certified enclosures,
//! never floating-point approximations. No arbitrary complex branch claim is
//! made. The optimizing generator is not a dependency of this module.

use crate::{BallError, BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use thiserror::Error;

pub const MAX_REAL_LAMBERT_NODES: usize = 1024;
pub const MAX_REAL_LAMBERT_DEPTH: usize = 32;
pub const MAX_REAL_LAMBERT_DEGREE: usize = 16;
pub const MAX_REAL_LAMBERT_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 8192;
const MAX_WORK: u64 = 4_000_000;
const MAX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RealLambertError {
    #[error("unsupported real Lambert equation: {0}")]
    Unsupported(&'static str),
    #[error("invalid original real Lambert source: {0}")]
    InvalidSource(&'static str),
    #[error("invalid real Lambert reduction: {0}")]
    InvalidWitness(&'static str),
    #[error("real Lambert equation resource limit exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("real Lambert branch comparison remains inconclusive")]
    Inconclusive,
    #[error("real Lambert work stopped: {0}")]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Enclosure(#[from] BallError),
}

type Result<T> = std::result::Result<T, RealLambertError>;

/// Untrusted coefficients of a*x+b+(c*x+d)*exp(e*x+f).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealLambertWitness {
    pub linear_slope: BigRational,
    pub linear_offset: BigRational,
    pub exponential_slope: BigRational,
    pub exponential_offset: BigRational,
    pub argument_slope: BigRational,
    pub argument_offset: BigRational,
}

/// Complete finite real roots bound to the exact original typed expression.
/// There is no unchecked constructor or deserialization path. LambertW(z)
/// denotes the principal branch; LambertW(z,-1) denotes the lower real
/// branch. A double root at the branch point occurs once in this distinct set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRealLambert {
    expression: Expr,
    variable: Symbol,
    normal_form: RealLambertWitness,
    argument: Expr,
    roots: Vec<Expr>,
}

impl VerifiedRealLambert {
    pub fn expression(&self) -> &Expr {
        &self.expression
    }
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn normal_form(&self) -> &RealLambertWitness {
        &self.normal_form
    }
    pub fn argument(&self) -> &Expr {
        &self.argument
    }
    pub fn roots(&self) -> &[Expr] {
        &self.roots
    }
}

fn limit(reason: &'static str) -> RealLambertError {
    RealLambertError::ResourceLimit(reason)
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

struct Work<'a, M> {
    meter: &'a mut M,
    steps: u64,
    bytes: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Result<Self> {
        meter.checkpoint()?;
        Ok(Self {
            meter,
            steps: 0,
            bytes: 0,
        })
    }

    fn charge(&mut self, steps: u64, bytes: u64) -> Result<()> {
        self.charge_all(steps.max(1), bytes.max(1), 1)
    }

    fn charge_all(&mut self, steps: u64, bytes: u64, allocations: u64) -> Result<()> {
        self.meter.checkpoint()?;
        let total_steps = self.steps.checked_add(steps).ok_or(limit("work"))?;
        let total_bytes = self.bytes.checked_add(bytes).ok_or(limit("allocation"))?;
        if total_steps > MAX_WORK || total_bytes > MAX_BYTES {
            return Err(limit("cumulative work/allocation"));
        }
        let mut charges = [(Dimension::ComputeSteps, 0); 3];
        let mut count = 0;
        for (dimension, amount) in [
            (Dimension::ComputeSteps, steps),
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, allocations),
        ] {
            if amount != 0 {
                charges[count] = (dimension, amount);
                count += 1;
            }
        }
        if count != 0 {
            self.meter.charge_batch(&charges[..count])?;
        }
        self.steps = total_steps;
        self.bytes = total_bytes;
        Ok(())
    }

    fn scalar(&mut self, value: &BigRational, maximum: u64) -> Result<()> {
        if !value.denom().is_positive() || bits(value) > maximum {
            return Err(limit("rational coefficient height"));
        }
        self.charge(1, 64 + 2 * bits(value).div_ceil(8))
    }

    fn binary(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        let bound = bits(a).saturating_add(bits(b)).saturating_add(1);
        if bound > MAX_WORK_BITS {
            return Err(limit("rational arithmetic growth"));
        }
        self.charge(bound.div_ceil(32).pow(2), 64 + bound.div_ceil(8) * 8)
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.binary(a, b)?;
        Ok(a + b)
    }

    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.binary(a, b)?;
        Ok(a - b)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.binary(a, b)?;
        Ok(a * b)
    }

    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        if b.is_zero() {
            return Err(RealLambertError::InvalidWitness("zero coefficient divisor"));
        }
        self.binary(a, b)?;
        Ok(a / b)
    }

    fn clone_scalar(&mut self, value: &BigRational) -> Result<BigRational> {
        self.scalar(value, MAX_WORK_BITS)?;
        Ok(value.clone())
    }
}

/// Dense coefficient rectangle, indexed by x exponent then t exponent.
/// This deliberately differs from the generator's sparse representation.
struct Polynomial {
    nx: usize,
    nt: usize,
    values: Vec<BigRational>,
}

impl Polynomial {
    fn empty(nx: usize, nt: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        if nx == 0
            || nt == 0
            || nx > MAX_REAL_LAMBERT_DEGREE + 1
            || nt > MAX_REAL_LAMBERT_DEGREE + 1
        {
            return Err(limit("formal polynomial degree"));
        }
        work.charge((nx * nt) as u64, (nx * nt * 64) as u64)?;
        Ok(Self {
            nx,
            nt,
            values: vec![BigRational::zero(); nx * nt],
        })
    }

    fn constant(value: BigRational, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let mut result = Self::empty(1, 1, work)?;
        result.values[0] = value;
        Ok(result)
    }

    fn get(&self, x: usize, t: usize) -> Option<&BigRational> {
        if x < self.nx && t < self.nt {
            Some(&self.values[t * self.nx + x])
        } else {
            None
        }
    }

    fn compact(self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        work.charge(self.values.len() as u64, 1)?;
        let mut nx = 1;
        let mut nt = 1;
        for t in 0..self.nt {
            for x in 0..self.nx {
                if !self.values[t * self.nx + x].is_zero() {
                    nx = nx.max(x + 1);
                    nt = nt.max(t + 1);
                }
            }
        }
        if nx == self.nx && nt == self.nt {
            return Ok(self);
        }
        let mut result = Self::empty(nx, nt, work)?;
        for t in 0..nt {
            for x in 0..nx {
                result.values[t * nx + x] = work.clone_scalar(&self.values[t * self.nx + x])?;
            }
        }
        Ok(result)
    }

    fn add(&self, other: &Self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let mut result = Self::empty(self.nx.max(other.nx), self.nt.max(other.nt), work)?;
        let zero = BigRational::zero();
        for t in 0..result.nt {
            for x in 0..result.nx {
                result.values[t * result.nx + x] = work.add(
                    self.get(x, t).unwrap_or(&zero),
                    other.get(x, t).unwrap_or(&zero),
                )?;
            }
        }
        result.compact(work)
    }

    fn mul(&self, other: &Self, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        let mut result = Self::empty(self.nx + other.nx - 1, self.nt + other.nt - 1, work)?;
        work.charge((self.values.len() * other.values.len()) as u64, 1)?;
        for t1 in 0..self.nt {
            for x1 in 0..self.nx {
                let a = &self.values[t1 * self.nx + x1];
                if a.is_zero() {
                    continue;
                }
                for t2 in 0..other.nt {
                    for x2 in 0..other.nx {
                        let b = &other.values[t2 * other.nx + x2];
                        if b.is_zero() {
                            continue;
                        }
                        let index = (t1 + t2) * result.nx + x1 + x2;
                        let product = work.mul(a, b)?;
                        result.values[index] = work.add(&result.values[index], &product)?;
                    }
                }
            }
        }
        result.compact(work)
    }

    fn power(mut self, exponent: i64, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        if exponent <= 0 {
            // Refusing nonconstant nonpositive powers also keeps original
            // reciprocal domains even when their bases later cancel.
            if self.nx != 1 || self.nt != 1 {
                return Err(RealLambertError::Unsupported(
                    "nonconstant nonpositive power",
                ));
            }
            if self.values[0].is_zero() {
                return Err(RealLambertError::InvalidSource(
                    "zero to a nonpositive power",
                ));
            }
            if exponent < 0 {
                self.values[0] = work.div(&BigRational::one(), &self.values[0])?;
            }
        }
        let mut result = Self::constant(BigRational::one(), work)?;
        // Reference repeated multiplication intentionally does not share the
        // generator's binary exponentiation implementation.
        for _ in 0..exponent.unsigned_abs() {
            result = result.mul(&self, work)?;
        }
        Ok(result)
    }
}

fn preflight(
    source: &Expr,
    variable: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<u64> {
    if variable.name.len() > 4096 {
        return Err(limit("coordinate identifier"));
    }
    work.charge(1, 64)?;
    let mut stack = vec![(source, 0usize)];
    let mut nodes = 0usize;
    let mut bytes = 128 + variable.name.len() as u64;
    while let Some((node, depth)) = stack.pop() {
        work.charge(1, 64)?;
        nodes += 1;
        if nodes > MAX_REAL_LAMBERT_NODES || depth > MAX_REAL_LAMBERT_DEPTH {
            return Err(limit("original expression size/depth"));
        }
        let children = match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_REAL_LAMBERT_INPUT_BITS {
                    return Err(limit("input integer bits"));
                }
                bytes += 64 + value.bits().div_ceil(8);
                continue;
            }
            Expr::Rational(value) => {
                work.scalar(value, MAX_REAL_LAMBERT_INPUT_BITS)?;
                bytes += 64 + 2 * bits(value).div_ceil(8);
                continue;
            }
            Expr::Sym(symbol) => {
                if symbol != variable {
                    return Err(RealLambertError::Unsupported(
                        "foreign typed coordinate or parameter",
                    ));
                }
                bytes += 128 + symbol.name.len() as u64;
                continue;
            }
            Expr::Add(values) | Expr::Mul(values) => values,
            Expr::Function(name, values) if name == "exp" && values.len() == 1 => values,
            Expr::Pow(base, exponent) => {
                let Expr::Integer(power) = exponent.as_ref() else {
                    return Err(RealLambertError::Unsupported("noninteger literal power"));
                };
                let power = power.to_i64().ok_or(limit("literal exponent"))?;
                if power.unsigned_abs() > MAX_REAL_LAMBERT_DEGREE as u64 {
                    return Err(limit("literal exponent"));
                }
                work.charge(1, 128)?;
                stack.push((base, depth + 1));
                stack.push((exponent, depth + 1));
                bytes += 128;
                continue;
            }
            _ => return Err(RealLambertError::Unsupported("original expression grammar")),
        };
        if children.len() > 128 || stack.len() + children.len() > MAX_REAL_LAMBERT_NODES {
            return Err(limit("original fanout/stack"));
        }
        work.charge(1, (children.len() * 32 + 128) as u64)?;
        bytes += 128;
        stack.extend(children.iter().map(|child| (child, depth + 1)));
    }
    Ok(bytes)
}

fn replay(
    source: &Expr,
    variable: &Symbol,
    witness: &RealLambertWitness,
    allow_exp: bool,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Polynomial> {
    work.charge(1, 1)?;
    match source {
        Expr::Integer(value) => {
            work.charge(1, 64 + value.bits().div_ceil(8))?;
            Polynomial::constant(BigRational::from_integer(value.clone()), work)
        }
        Expr::Rational(value) => Polynomial::constant(work.clone_scalar(value)?, work),
        Expr::Sym(symbol) if symbol == variable => {
            let mut p = Polynomial::empty(2, 1, work)?;
            p.values[1] = BigRational::one();
            Ok(p)
        }
        Expr::Add(values) | Expr::Mul(values) => {
            let adding = matches!(source, Expr::Add(_));
            let mut result = Polynomial::constant(
                if adding {
                    BigRational::zero()
                } else {
                    BigRational::one()
                },
                work,
            )?;
            // Do not skip original children after a zero product.
            for child in values {
                let part = replay(child, variable, witness, allow_exp, work)?;
                result = if adding {
                    result.add(&part, work)?
                } else {
                    result.mul(&part, work)?
                };
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            let Expr::Integer(power) = exponent.as_ref() else {
                return Err(RealLambertError::Unsupported("power"));
            };
            let power = power.to_i64().ok_or(limit("power"))?;
            replay(base, variable, witness, allow_exp, work)?.power(power, work)
        }
        Expr::Function(name, values) if allow_exp && name == "exp" && values.len() == 1 => {
            let argument = replay(&values[0], variable, witness, false, work)?;
            let zero = BigRational::zero();
            if argument.nt != 1
                || argument.nx > 2
                || argument.get(0, 0).unwrap_or(&zero) != &witness.argument_offset
                || argument.get(1, 0).unwrap_or(&zero) != &witness.argument_slope
            {
                return Err(RealLambertError::InvalidWitness(
                    "exponential argument identity",
                ));
            }
            let mut p = Polynomial::empty(1, 2, work)?;
            p.values[1] = BigRational::one();
            Ok(p)
        }
        _ => Err(RealLambertError::Unsupported(
            "original source or nested exponential",
        )),
    }
}

fn rational_expr(value: BigRational) -> Expr {
    if value.denom().is_one() {
        Expr::Integer(value.numer().clone())
    } else {
        Expr::Rational(value)
    }
}

fn scaled_exponential(
    q: &BigRational,
    r: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    work.charge(1, 256)?;
    if q.is_zero() {
        return Ok(Expr::from_i64(0));
    }
    if r.is_zero() {
        return Ok(rational_expr(work.clone_scalar(q)?));
    }
    let exponential = Expr::Function("exp".into(), vec![rational_expr(work.clone_scalar(r)?)]);
    if q.is_one() {
        return Ok(exponential);
    }
    Ok(Expr::Mul(vec![
        rational_expr(work.clone_scalar(q)?),
        exponential,
    ]))
}

fn affine_root(
    value: Expr,
    scale: &BigRational,
    offset: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    work.charge(1, 256)?;
    let literal = match &value {
        Expr::Integer(integer) => Some(BigRational::from_integer(integer.clone())),
        Expr::Rational(rational) => Some(work.clone_scalar(rational)?),
        _ => None,
    };
    if let Some(literal) = literal {
        let product = work.mul(scale, &literal)?;
        return Ok(rational_expr(work.add(offset, &product)?));
    }
    let value = if scale.is_one() {
        value
    } else {
        Expr::Mul(vec![rational_expr(work.clone_scalar(scale)?), value])
    };
    if offset.is_zero() {
        Ok(value)
    } else {
        Ok(Expr::Add(vec![
            rational_expr(work.clone_scalar(offset)?),
            value,
        ]))
    }
}

fn negative_branch_order(
    q: &BigRational,
    r: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Ordering> {
    let minus_one = -BigRational::one();
    if q == &minus_one && r == &minus_one {
        return Ok(Ordering::Equal);
    }
    let exponent = work.add(r, &BigRational::one())?;
    let positive = work.sub(&BigRational::zero(), q)?;
    // Cheap exact comparisons avoid exponentiation in the common distant
    // cases. The remaining comparisons use independent certified intervals.
    if exponent.is_zero() {
        return Ok(positive.cmp(&BigRational::one()));
    }
    if exponent.is_positive() && positive >= BigRational::one() {
        return Ok(Ordering::Greater);
    }
    if exponent.is_negative() && positive <= BigRational::one() {
        return Ok(Ordering::Less);
    }
    // Compare one positive exponential directly. For s>0, p*exp(s)<1
    // iff exp(s)<1/p; for s<0, p*exp(s)<1 iff exp(-s)>p.
    // The existing relative-exp kernel range-reduces and outward-squares
    // certified intervals, with an explicit charge/checkpoint for each stage.
    // It does not allocate the unrelated root-box/Jacobian workspace.
    let reverse = exponent.is_negative();
    let magnitude = if reverse {
        work.sub(&BigRational::zero(), &exponent)?
    } else {
        work.clone_scalar(&exponent)?
    };
    let target = if reverse {
        positive
    } else {
        work.div(&BigRational::one(), &positive)?
    };
    work.charge(1, 128 + 2 * bits(&magnitude).div_ceil(8))?;
    let point = RealBall::exact(magnitude);
    for precision in [8, 16, 32, 64, 128] {
        let value = point.positive_exp_with_work::<RealLambertError>(
            precision,
            &mut |steps, bytes, allocations| work.charge_all(steps, bytes, allocations),
        )?;
        work.binary(value.midpoint(), value.radius())?;
        let lower = value.lower();
        work.binary(value.midpoint(), value.radius())?;
        let upper = value.upper();
        work.binary(&upper, &target)?;
        if upper < target {
            return Ok(if reverse {
                Ordering::Greater
            } else {
                Ordering::Less
            });
        }
        work.binary(&lower, &target)?;
        if lower > target {
            return Ok(if reverse {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
    }
    Err(RealLambertError::Inconclusive)
}

/// Independently check the exact original equation and construct every real
/// root. The witness cannot supply a branch list, root count or accepted flag.
/// Caller-owned accounting is preserved through replay, branch comparison and
/// publication. Unsupported or inconclusive work publishes no partial set.
pub fn verify_real_lambert_equation(
    source: &Expr,
    variable: &Symbol,
    witness: &RealLambertWitness,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealLambert> {
    let mut work = Work::new(meter)?;
    let source_bytes = preflight(source, variable, &mut work)?;
    for coefficient in [
        &witness.linear_slope,
        &witness.linear_offset,
        &witness.exponential_slope,
        &witness.exponential_offset,
        &witness.argument_slope,
        &witness.argument_offset,
    ] {
        work.scalar(coefficient, MAX_REAL_LAMBERT_INPUT_BITS)?;
    }
    if witness.argument_slope.is_zero()
        || witness.linear_slope.is_zero() == witness.exponential_slope.is_zero()
    {
        return Err(RealLambertError::InvalidWitness(
            "normal form is not a supported invertible Lambert reduction",
        ));
    }
    let polynomial = replay(source, variable, witness, true, &mut work)?;
    let zero = BigRational::zero();
    if polynomial.nx > 2
        || polynomial.nt > 2
        || polynomial.get(0, 0).unwrap_or(&zero) != &witness.linear_offset
        || polynomial.get(1, 0).unwrap_or(&zero) != &witness.linear_slope
        || polynomial.get(0, 1).unwrap_or(&zero) != &witness.exponential_offset
        || polynomial.get(1, 1).unwrap_or(&zero) != &witness.exponential_slope
    {
        return Err(RealLambertError::InvalidWitness(
            "original formal coefficient identity",
        ));
    }
    let e = &witness.argument_slope;
    let (q, r, scale, offset) = if witness.linear_slope.is_zero() {
        let shift = work.div(&witness.exponential_offset, &witness.exponential_slope)?;
        let scaled = work.mul(e, &witness.linear_offset)?;
        let neg_scaled = work.sub(&zero, &scaled)?;
        let q = work.div(&neg_scaled, &witness.exponential_slope)?;
        let exp_shift = work.mul(e, &shift)?;
        let r = work.sub(&exp_shift, &witness.argument_offset)?;
        let scale = work.div(&BigRational::one(), e)?;
        let offset = work.sub(&zero, &shift)?;
        (q, r, scale, offset)
    } else {
        let shift = work.div(&witness.linear_offset, &witness.linear_slope)?;
        let scaled = work.mul(e, &witness.exponential_offset)?;
        let q = work.div(&scaled, &witness.linear_slope)?;
        let exp_shift = work.mul(e, &shift)?;
        let r = work.sub(&witness.argument_offset, &exp_shift)?;
        let scale = work.div(&-BigRational::one(), e)?;
        let offset = work.sub(&zero, &shift)?;
        (q, r, scale, offset)
    };
    let argument = scaled_exponential(&q, &r, &mut work)?;
    work.charge(1, 512)?;
    let mut roots = Vec::with_capacity(2);
    if q.is_zero() {
        roots.push(rational_expr(work.clone_scalar(&offset)?));
    } else {
        let order = if q.is_positive() {
            Ordering::Less
        } else {
            negative_branch_order(&q, &r, &mut work)?
        };
        match order {
            Ordering::Greater => {}
            Ordering::Equal => {
                roots.push(rational_expr(work.sub(&offset, &scale)?));
            }
            Ordering::Less => {
                if q.is_negative() {
                    work.charge(1, 256 + 4 * bits(&q).div_ceil(8) + 4 * bits(&r).div_ceil(8))?;
                    let lower = Expr::Function(
                        "LambertW".into(),
                        vec![argument.clone(), Expr::from_i64(-1)],
                    );
                    roots.push(affine_root(lower, &scale, &offset, &mut work)?);
                }
                work.charge(1, 256 + 4 * bits(&q).div_ceil(8) + 4 * bits(&r).div_ceil(8))?;
                let principal = if q.is_one() && r.is_one() {
                    Expr::from_i64(1)
                } else {
                    Expr::Function("LambertW".into(), vec![argument.clone()])
                };
                roots.push(affine_root(principal, &scale, &offset, &mut work)?);
                if scale.is_negative() {
                    roots.reverse();
                }
            }
        }
    }
    work.charge(
        1,
        source_bytes + 512 + 16 * MAX_REAL_LAMBERT_INPUT_BITS.div_ceil(8),
    )?;
    let receipt = VerifiedRealLambert {
        expression: source.clone(),
        variable: variable.clone(),
        normal_form: witness.clone(),
        argument,
        roots,
    };
    work.meter.checkpoint()?;
    Ok(receipt)
}
