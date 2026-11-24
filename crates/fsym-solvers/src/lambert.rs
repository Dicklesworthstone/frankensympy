//! Candidate recognition for complete real affine-exponential equations.
//!
//! The sparse two-generator algebra below proposes a rational normal form.
//! Acceptance lives in fsym-core's independent dense reference replay and
//! real-branch checker. No numeric sampling or candidate residual is used to
//! claim completeness. Grammar nonmatches return None; admitted resource,
//! original-domain, verification and cancellation failures remain errors.

use fsym_budget::{BudgetMeter, Dimension, Unbounded};
use fsym_core::{
    BigRational, Expr, MAX_REAL_LAMBERT_DEGREE, MAX_REAL_LAMBERT_DEPTH,
    MAX_REAL_LAMBERT_INPUT_BITS, MAX_REAL_LAMBERT_NODES, RealLambertError, RealLambertWitness,
    Symbol, VerifiedRealLambert, verify_real_lambert_equation,
};
use num_traits::{One, Zero};
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, RealLambertError>;
type Polynomial = BTreeMap<(usize, usize), BigRational>;
const MAX_BITS: u64 = 8192;
const MAX_STEPS: u64 = 4_000_000;
const MAX_BYTES: u64 = 64 * 1024 * 1024;

fn limit(reason: &'static str) -> RealLambertError {
    RealLambertError::ResourceLimit(reason)
}
fn unsupported(reason: &'static str) -> RealLambertError {
    RealLambertError::Unsupported(reason)
}
fn bits(q: &BigRational) -> u64 {
    q.numer().bits().max(q.denom().bits()).max(1)
}

struct Generator<'a, M> {
    meter: &'a mut M,
    steps: u64,
    bytes: u64,
    argument: Option<(BigRational, BigRational)>,
}

impl<'a, M: BudgetMeter> Generator<'a, M> {
    fn charge(&mut self, steps: u64, bytes: u64) -> Result<()> {
        self.meter.checkpoint()?;
        self.steps = self
            .steps
            .checked_add(steps.max(1))
            .ok_or(limit("generator work"))?;
        self.bytes = self
            .bytes
            .checked_add(bytes.max(1))
            .ok_or(limit("generator allocation"))?;
        if self.steps > MAX_STEPS || self.bytes > MAX_BYTES {
            return Err(limit("generator cumulative work/allocation"));
        }
        self.meter.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1)),
            (Dimension::MemoryBytes, bytes.max(1)),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }

    fn scalar(&mut self, q: &BigRational, maximum: u64) -> Result<()> {
        if !q.denom().is_positive() || bits(q) > maximum {
            return Err(limit("generator rational height"));
        }
        self.charge(1, 64 + 2 * bits(q).div_ceil(8))
    }

    fn copy(&mut self, q: &BigRational) -> Result<BigRational> {
        self.scalar(q, MAX_BITS)?;
        Ok(q.clone())
    }

    fn arithmetic(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        let bound = bits(a).saturating_add(bits(b)).saturating_add(1);
        if bound > MAX_BITS {
            return Err(limit("generator coefficient growth"));
        }
        self.charge(bound.div_ceil(32).pow(2), 64 + 8 * bound.div_ceil(8))
    }

    fn constant(&mut self, q: BigRational) -> Result<Polynomial> {
        self.charge(1, 128)?;
        let mut result = Polynomial::new();
        if !q.is_zero() {
            result.insert((0, 0), q);
        }
        Ok(result)
    }

    fn insert(
        &mut self,
        p: &mut Polynomial,
        key: (usize, usize),
        value: BigRational,
    ) -> Result<()> {
        if key.0 > MAX_REAL_LAMBERT_DEGREE || key.1 > MAX_REAL_LAMBERT_DEGREE {
            return Err(limit("generator polynomial degree"));
        }
        self.charge(1, 128 + 2 * bits(&value).div_ceil(8))?;
        let combined = if let Some(old) = p.get(&key) {
            self.arithmetic(old, &value)?;
            old + value
        } else {
            value
        };
        if combined.is_zero() {
            p.remove(&key);
        } else {
            p.insert(key, combined);
        }
        Ok(())
    }

    fn add(&mut self, mut a: Polynomial, b: Polynomial) -> Result<Polynomial> {
        self.charge((a.len() + b.len() + 1) as u64, 1)?;
        for (key, value) in b {
            self.insert(&mut a, key, value)?;
        }
        Ok(a)
    }

    fn mul(&mut self, a: &Polynomial, b: &Polynomial) -> Result<Polynomial> {
        self.charge((a.len() * b.len() + 1) as u64, 128)?;
        let mut result = Polynomial::new();
        for (&(xa, ta), va) in a {
            for (&(xb, tb), vb) in b {
                self.arithmetic(va, vb)?;
                self.insert(&mut result, (xa + xb, ta + tb), va * vb)?;
            }
        }
        Ok(result)
    }

    fn power(&mut self, mut p: Polynomial, n: i64) -> Result<Polynomial> {
        if n <= 0 {
            if p.keys().any(|key| *key != (0, 0)) {
                return Err(unsupported("nonconstant nonpositive power"));
            }
            let Some(value) = p.get(&(0, 0)) else {
                return Err(RealLambertError::InvalidSource(
                    "zero to a nonpositive power",
                ));
            };
            if n < 0 {
                self.arithmetic(&BigRational::one(), value)?;
                p = self.constant(BigRational::one() / value)?;
            }
        }
        let mut result = self.constant(BigRational::one())?;
        let mut exponent = n.unsigned_abs();
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = self.mul(&result, &p)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                p = self.mul(&p, &p)?;
            }
        }
        Ok(result)
    }

    fn preflight(&mut self, source: &Expr, variable: &Symbol) -> Result<()> {
        if variable.name.len() > 4096 {
            return Err(limit("coordinate name"));
        }
        self.charge(1, 64)?;
        let mut pending = vec![(source, 0usize)];
        let mut seen = 0usize;
        while let Some((node, depth)) = pending.pop() {
            self.charge(1, 64)?;
            seen += 1;
            if seen > MAX_REAL_LAMBERT_NODES || depth > MAX_REAL_LAMBERT_DEPTH {
                return Err(limit("generator original source size/depth"));
            }
            let children = match node {
                Expr::Integer(value) => {
                    if value.bits() > MAX_REAL_LAMBERT_INPUT_BITS {
                        return Err(limit("input integer bits"));
                    }
                    continue;
                }
                Expr::Rational(value) => {
                    self.scalar(value, MAX_REAL_LAMBERT_INPUT_BITS)?;
                    continue;
                }
                Expr::Sym(symbol) => {
                    if symbol != variable {
                        return Err(unsupported("foreign typed coordinate or parameter"));
                    }
                    continue;
                }
                Expr::Add(values) | Expr::Mul(values) => values,
                Expr::Function(name, values) if name == "exp" && values.len() == 1 => values,
                Expr::Pow(base, exponent) => {
                    let Expr::Integer(value) = exponent.as_ref() else {
                        return Err(unsupported("noninteger literal power"));
                    };
                    let value = value.to_i64().ok_or(limit("literal exponent"))?;
                    if value.unsigned_abs() > MAX_REAL_LAMBERT_DEGREE as u64 {
                        return Err(limit("literal exponent"));
                    }
                    self.charge(1, 128)?;
                    pending.push((base, depth + 1));
                    pending.push((exponent, depth + 1));
                    continue;
                }
                _ => return Err(unsupported("original expression grammar")),
            };
            if children.len() > 128 || pending.len() + children.len() > MAX_REAL_LAMBERT_NODES {
                return Err(limit("original fanout/stack"));
            }
            self.charge(1, (32 * children.len() + 1) as u64)?;
            pending.extend(children.iter().map(|child| (child, depth + 1)));
        }
        Ok(())
    }

    fn read(&mut self, source: &Expr, variable: &Symbol, allow_exp: bool) -> Result<Polynomial> {
        self.charge(1, 1)?;
        match source {
            Expr::Integer(value) => {
                self.charge(1, 64 + value.bits().div_ceil(8))?;
                self.constant(BigRational::from_integer(value.clone()))
            }
            Expr::Rational(value) => {
                let value = self.copy(value)?;
                self.constant(value)
            }
            Expr::Sym(symbol) if symbol == variable => {
                let mut result = Polynomial::new();
                self.insert(&mut result, (1, 0), BigRational::one())?;
                Ok(result)
            }
            Expr::Add(values) | Expr::Mul(values) => {
                let adding = matches!(source, Expr::Add(_));
                let mut result = self.constant(if adding {
                    BigRational::zero()
                } else {
                    BigRational::one()
                })?;
                for child in values {
                    // Visit all original operands, including those whose
                    // enclosing product already has zero coefficients.
                    let part = self.read(child, variable, allow_exp)?;
                    result = if adding {
                        self.add(result, part)?
                    } else {
                        self.mul(&result, &part)?
                    };
                }
                Ok(result)
            }
            Expr::Pow(base, exponent) => {
                let Expr::Integer(value) = exponent.as_ref() else {
                    return Err(unsupported("literal power"));
                };
                let value = value.to_i64().ok_or(limit("literal exponent"))?;
                let base = self.read(base, variable, allow_exp)?;
                self.power(base, value)
            }
            Expr::Function(name, values) if allow_exp && name == "exp" && values.len() == 1 => {
                let argument = self.read(&values[0], variable, false)?;
                if argument.keys().any(|key| *key != (0, 0) && *key != (1, 0)) {
                    return Err(unsupported("non-affine exponential argument"));
                }
                let zero = BigRational::zero();
                let slope = self.copy(argument.get(&(1, 0)).unwrap_or(&zero))?;
                let offset = self.copy(argument.get(&(0, 0)).unwrap_or(&zero))?;
                if slope.is_zero() {
                    return Err(unsupported("constant exponential argument"));
                }
                if let Some((old_slope, old_offset)) = &self.argument {
                    if old_slope != &slope || old_offset != &offset {
                        return Err(unsupported("multiple exponential generators"));
                    }
                } else {
                    self.argument = Some((slope, offset));
                }
                let mut result = Polynomial::new();
                self.insert(&mut result, (0, 1), BigRational::one())?;
                Ok(result)
            }
            _ => Err(unsupported("original source or nested exponential")),
        }
    }

    fn propose(&mut self, source: &Expr, variable: &Symbol) -> Result<Option<RealLambertWitness>> {
        self.preflight(source, variable)?;
        let coefficients = self.read(source, variable, true)?;
        if coefficients.keys().any(|&(x, t)| x > 1 || t > 1) {
            return Ok(None);
        }
        let Some((argument_slope, argument_offset)) = self.argument.take() else {
            return Ok(None);
        };
        let zero = BigRational::zero();
        let linear_slope = self.copy(coefficients.get(&(1, 0)).unwrap_or(&zero))?;
        let exponential_slope = self.copy(coefficients.get(&(1, 1)).unwrap_or(&zero))?;
        if linear_slope.is_zero() == exponential_slope.is_zero() {
            return Ok(None);
        }
        let witness = RealLambertWitness {
            linear_slope,
            linear_offset: self.copy(coefficients.get(&(0, 0)).unwrap_or(&zero))?,
            exponential_slope,
            exponential_offset: self.copy(coefficients.get(&(0, 1)).unwrap_or(&zero))?,
            argument_slope,
            argument_offset,
        };
        self.meter.checkpoint()?;
        Ok(Some(witness))
    }
}

/// Propose only rational normal-form coefficients; this value is untrusted.
pub fn propose_real_lambert(
    source: &Expr,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<Option<RealLambertWitness>> {
    meter.checkpoint()?;
    let mut generator = Generator {
        meter,
        steps: 0,
        bytes: 0,
        argument: None,
    };
    match generator.propose(source, variable) {
        Err(RealLambertError::Unsupported(_)) => Ok(None),
        result => result,
    }
}

/// Complete distinct real roots, or a grammar nonmatch. Generation and
/// verification receive separate caller-owned cumulative ledgers. No retry
/// resets either ledger, and a terminal failure publishes no partial roots.
pub fn metered_solve_real_lambert(
    source: &Expr,
    variable: &Symbol,
    generation_meter: &mut impl BudgetMeter,
    verification_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealLambert>> {
    let Some(witness) = propose_real_lambert(source, variable, generation_meter)? else {
        return Ok(None);
    };
    let checked = verify_real_lambert_equation(source, variable, &witness, verification_meter)?;
    generation_meter.checkpoint()?;
    verification_meter.checkpoint()?;
    Ok(Some(checked))
}

pub fn solve_real_lambert(source: &Expr, variable: &Symbol) -> Result<Option<VerifiedRealLambert>> {
    metered_solve_real_lambert(source, variable, &mut Unbounded, &mut Unbounded)
}
