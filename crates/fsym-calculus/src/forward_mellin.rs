//! Bounded, checked forward Mellin transforms on the positive real axis.
//!
//! Two ordinary absolutely convergent integral identities are admitted:
//! `c*x^a*exp(-b*x^p)` and `c*x^a*(A+B*x^p)^(-q)`, with exact rational
//! parameters, b,A,B,q positive and p a nonzero bounded integer. The returned
//! open strip is maximal for each nonzero kernel. No sum rule is used: source
//! cancellation can enlarge a fundamental strip.
//!
//! Recognition is not verification. The checker independently differentiates
//! the original source logarithmically as a rational function and checks its
//! value at x=1. On the admitted smooth, nonvanishing positive-axis domain,
//! these two identities bind the kernel to the source. It separately checks
//! the gamma/beta formula and both convergence endpoints. Original powers and
//! denominators are checked before any cancellation, even in zero products.
//! These are in-process identity receipts, not portable proof-kernel proofs.

#![forbid(unsafe_code)]

use crate::rational_mellin::{MellinStrip, MellinStripBound};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use num_traits::{One, Signed, Zero};
use std::sync::Arc;
use thiserror::Error;

const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_NAMES: usize = 65_536;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 65_536;
const MAX_POWER: i32 = 16;
const MAX_TERMS: usize = 128;
const MAX_DEGREE: i32 = 1024;
const MAX_COMPUTE: u64 = 40_000_000;
const MAX_MEMORY: u64 = 256 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ForwardMellinError {
    #[error("forward Mellin catalog does not support {0}")]
    Unsupported(&'static str),
    #[error("forward Mellin transform refused: {0}")]
    Refused(&'static str),
    #[error("forward Mellin verification failed: {0}")]
    Verification(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for ForwardMellinError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, ForwardMellinError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MellinKernel {
    Zero,
    Gamma {
        coefficient: BigRational,
        shift: BigRational,
        power: i32,
        rate: BigRational,
    },
    Beta {
        coefficient: BigRational,
        shift: BigRational,
        power: i32,
        constant: BigRational,
        scale: BigRational,
        order: BigRational,
    },
}

/// Untrusted candidate. Its fields are public to permit independent transport
/// and mutation testing; only `verify_mellin_candidate` admits a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MellinCandidate {
    pub kernel: MellinKernel,
    pub transform: Expr,
    pub strip: MellinStrip,
}

#[derive(Debug)]
pub struct VerifiedMellin {
    transform: Expr,
    strip: MellinStrip,
}

impl VerifiedMellin {
    pub fn transform(&self) -> &Expr {
        &self.transform
    }

    pub fn strip(&self) -> &MellinStrip {
        &self.strip
    }

    pub fn into_parts(self) -> (Expr, MellinStrip) {
        (self.transform, self.strip)
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    compute: u64,
    memory: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            compute: 0,
            memory: 0,
        }
    }

    fn allocate(&mut self, count: usize, bytes: u64) -> Result<()> {
        self.checkpoint()?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, count.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1)),
            (Dimension::AllocationCount, count.max(1) as u64),
        ])?;
        Ok(())
    }

    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        self.allocate(1, 64 + 2 * bits(value).div_ceil(8))?;
        Ok(value.clone())
    }

    fn preflight(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        self.checkpoint()?;
        if bits(a).saturating_add(bits(b)).saturating_add(2) > MAX_WORK_BITS {
            return Err(ForwardMellinError::Refused(
                "intermediate scalar growth bound",
            ));
        }
        Ok(())
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_add(b, self)?)
    }

    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_sub(b, self)?)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_mul(b, self)?)
    }

    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        if b.is_zero() {
            return Err(ForwardMellinError::Refused("undefined exact division"));
        }
        Ok(a.metered_div(b, self)?)
    }

    fn neg(&mut self, a: &BigRational) -> Result<BigRational> {
        self.sub(&BigRational::zero(), a)
    }

    fn pow(&mut self, base: &BigRational, power: i32) -> Result<BigRational> {
        self.checkpoint()?;
        if base.is_zero() && power <= 0 {
            return Err(ForwardMellinError::Refused("undefined original zero power"));
        }
        if power.unsigned_abs() > MAX_POWER as u32
            || bits(base).saturating_mul(u64::from(power.unsigned_abs()).max(1)) > MAX_WORK_BITS / 2
        {
            return Err(ForwardMellinError::Refused("rational power growth bound"));
        }
        Ok(base.metered_pow(power, self)?)
    }

    fn equal(&mut self, a: &BigRational, b: &BigRational) -> Result<bool> {
        self.charge(Dimension::ComputeSteps, bits(a).max(bits(b)).div_ceil(64))?;
        Ok(a == b)
    }

    fn scalar(&mut self, value: &BigRational) -> Result<Expr> {
        self.allocate(1, 128)?;
        Ok(rational_expr(self.copy(value)?))
    }

    fn sum(&mut self, mut terms: Vec<Expr>) -> Result<Expr> {
        self.allocate(terms.len() + 1, 128 * (terms.len() + 1) as u64)?;
        terms.retain(|term| !term.is_zero());
        Ok(match terms.len() {
            0 => rational_expr(BigRational::zero()),
            1 => terms.pop().expect("one term"),
            _ => Expr::Add(terms),
        })
    }

    fn product(&mut self, mut terms: Vec<Expr>) -> Result<Expr> {
        self.allocate(terms.len() + 1, 128 * (terms.len() + 1) as u64)?;
        if terms.iter().any(Expr::is_zero) {
            return self.scalar(&BigRational::zero());
        }
        terms.retain(|term| !term.is_one());
        Ok(match terms.len() {
            0 => rational_expr(BigRational::one()),
            1 => terms.pop().expect("one factor"),
            _ => Expr::Mul(terms),
        })
    }

    fn power(&mut self, base: Expr, exponent: Expr) -> Result<Expr> {
        self.allocate(3, 384)?;
        if base.is_one() || exponent.is_zero() {
            return self.scalar(&BigRational::one());
        }
        if exponent.is_one() {
            return Ok(base);
        }
        Ok(Expr::Pow(Arc::new(base), Arc::new(exponent)))
    }

    fn gamma(&mut self, argument: Expr) -> Result<Expr> {
        self.allocate(2, 256)?;
        Ok(Expr::Function("gamma".into(), vec![argument]))
    }

    fn affine(
        &mut self,
        variable: &Symbol,
        slope: &BigRational,
        offset: &BigRational,
    ) -> Result<Expr> {
        self.allocate(1, 128 + variable.name.len() as u64)?;
        let coefficient = self.scalar(slope)?;
        let term = self.product(vec![coefficient, Expr::Sym(variable.clone())])?;
        let constant = self.scalar(offset)?;
        self.sum(vec![term, constant])
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let (mut compute, mut memory) = (self.compute, self.memory);
        for &(dimension, amount) in charges {
            let (used, limit) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_COMPUTE),
                Dimension::MemoryBytes => (&mut memory, MAX_MEMORY),
                _ => continue,
            };
            let remaining = limit.saturating_sub(*used);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *used += amount;
        }
        self.meter.charge_batch(charges)?;
        (self.compute, self.memory) = (compute, memory);
        Ok(())
    }
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn literal<M: BudgetMeter>(source: &Expr, work: &mut Work<'_, M>) -> Result<Option<BigRational>> {
    work.charge(Dimension::ComputeSteps, 1)?;
    match source {
        Expr::Integer(value) => {
            work.allocate(1, 64 + value.bits().div_ceil(8))?;
            Ok(Some(BigRational::from_integer(value.clone())))
        }
        Expr::Rational(value) => Ok(Some(work.copy(value)?)),
        _ => Ok(None),
    }
}

/// Borrow every native node before cloning or recursively inspecting it.
fn envelope<M: BudgetMeter>(source: &Expr, input: bool, work: &mut Work<'_, M>) -> Result<()> {
    work.allocate(MAX_NODES, MAX_NODES as u64 * 32)?;
    let mut pending = Vec::with_capacity(MAX_NODES);
    pending.push((source, 0));
    let (mut count, mut names) = (0, 0usize);
    let bit_limit = if input { MAX_INPUT_BITS } else { MAX_WORK_BITS };
    while let Some((node, depth)) = pending.pop() {
        count += 1;
        work.charge(Dimension::ComputeSteps, 1)?;
        if count > MAX_NODES || depth > MAX_DEPTH {
            return Err(ForwardMellinError::Refused("original node/depth bound"));
        }
        match node {
            Expr::Integer(value) => {
                if value.bits() > bit_limit {
                    return Err(ForwardMellinError::Refused("coefficient bit bound"));
                }
                work.charge(Dimension::ComputeSteps, value.bits().div_ceil(64).max(1))?;
            }
            Expr::Rational(value) => {
                if bits(value) > bit_limit {
                    return Err(ForwardMellinError::Refused("coefficient bit bound"));
                }
                work.charge(Dimension::ComputeSteps, bits(value).div_ceil(64))?;
            }
            Expr::Sym(symbol) => names = names.saturating_add(symbol.name.len()),
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = node {
                    names = names.saturating_add(name.len());
                }
                if children.len() > MAX_NODES.saturating_sub(count + pending.len()) {
                    return Err(ForwardMellinError::Refused("original node bound"));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if 2 > MAX_NODES.saturating_sub(count + pending.len()) {
                    return Err(ForwardMellinError::Refused("original node bound"));
                }
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Const(
                Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity
                | Constant::NaN,
            ) => {
                return Err(ForwardMellinError::Refused("nonfinite original source"));
            }
            Expr::Const(_) => {}
        }
        if names > MAX_NAMES {
            return Err(ForwardMellinError::Refused("source name bound"));
        }
    }
    work.charge(Dimension::ComputeSteps, names.max(1) as u64)?;
    Ok(())
}

fn variables<M: BudgetMeter>(x: &Symbol, s: &Symbol, work: &mut Work<'_, M>) -> Result<()> {
    let names = x.name.len().saturating_add(s.name.len());
    if names > MAX_NAMES {
        return Err(ForwardMellinError::Refused("variable name bound"));
    }
    work.charge(Dimension::ComputeSteps, names.max(1) as u64)?;
    if x == s {
        return Err(ForwardMellinError::Refused(
            "integration and transform variables must be distinct",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sign {
    Negative,
    Zero,
    Positive,
    Unknown,
}

struct Domain {
    sign: Sign,
    scalar: Option<BigRational>,
}

fn sign(value: &BigRational) -> Sign {
    if value.is_zero() {
        Sign::Zero
    } else if value.is_positive() {
        Sign::Positive
    } else {
        Sign::Negative
    }
}

/// Visit all operands, including those annihilated by a later zero. Unknown
/// signs are never used to cancel a denominator or choose a real power branch.
fn domain<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    work: &mut Work<'_, M>,
    supported: &mut bool,
) -> Result<Domain> {
    if let Some(value) = literal(source, work)? {
        return Ok(Domain {
            sign: sign(&value),
            scalar: Some(value),
        });
    }
    let unknown = || Domain {
        sign: Sign::Unknown,
        scalar: None,
    };
    match source {
        Expr::Sym(symbol) if symbol == x => Ok(Domain {
            sign: Sign::Positive,
            scalar: None,
        }),
        Expr::Add(children) | Expr::Mul(children) => {
            let addition = matches!(source, Expr::Add(_));
            let mut value = if addition {
                BigRational::zero()
            } else {
                BigRational::one()
            };
            let mut scalar = true;
            let mut state = if addition { Sign::Zero } else { Sign::Positive };
            for child in children {
                let child = domain(child, x, work, supported)?;
                if let Some(number) = child.scalar {
                    value = if addition {
                        work.add(&value, &number)?
                    } else {
                        work.mul(&value, &number)?
                    };
                } else {
                    scalar = false;
                }
                state = if addition {
                    match (state, child.sign) {
                        (Sign::Zero, b) => b,
                        (a, Sign::Zero) => a,
                        (Sign::Positive, Sign::Positive) => Sign::Positive,
                        (Sign::Negative, Sign::Negative) => Sign::Negative,
                        _ => Sign::Unknown,
                    }
                } else {
                    match (state, child.sign) {
                        (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
                        (Sign::Positive, b) => b,
                        (a, Sign::Positive) => a,
                        (Sign::Negative, Sign::Negative) => Sign::Positive,
                        _ => Sign::Unknown,
                    }
                };
            }
            Ok(if scalar {
                Domain {
                    sign: sign(&value),
                    scalar: Some(value),
                }
            } else {
                Domain {
                    sign: state,
                    scalar: None,
                }
            })
        }
        Expr::Pow(base, exponent) => {
            let base = domain(base, x, work, supported)?;
            // Check the original exponent's complete tree too.
            let _ = domain(exponent, x, work, supported)?;
            let Some(power) = literal(exponent, work)? else {
                *supported = false;
                return Ok(unknown());
            };
            let magnitude = power.abs();
            work.charge(Dimension::ComputeSteps, bits(&power).div_ceil(64))?;
            if magnitude > BigRational::from(i64::from(MAX_POWER)) {
                return Err(ForwardMellinError::Refused("original rational power bound"));
            }
            if base.sign == Sign::Zero && !power.is_positive() {
                return Err(ForwardMellinError::Refused("undefined original zero power"));
            }
            if base.sign == Sign::Unknown || (base.sign == Sign::Negative && !power.is_integer()) {
                *supported = false;
                return Ok(unknown());
            }
            if let Some(value) = base.scalar.as_ref()
                && power.is_integer()
            {
                let n = power
                    .numer()
                    .to_i64()
                    .ok_or(ForwardMellinError::Refused("integer power bound"))?
                    as i32;
                let value = work.pow(value, n)?;
                return Ok(Domain {
                    sign: sign(&value),
                    scalar: Some(value),
                });
            }
            let state = if power.is_zero() || base.sign == Sign::Positive {
                Sign::Positive
            } else if base.sign == Sign::Zero {
                Sign::Zero
            } else if power.numer().to_i64().is_some_and(|n| n % 2 == 0) {
                Sign::Positive
            } else {
                Sign::Negative
            };
            Ok(Domain {
                sign: state,
                scalar: None,
            })
        }
        Expr::Function(name, children) => {
            for child in children {
                let _ = domain(child, x, work, supported)?;
            }
            if name == "exp" && children.len() == 1 {
                Ok(Domain {
                    sign: Sign::Positive,
                    scalar: None,
                })
            } else {
                *supported = false;
                Ok(unknown())
            }
        }
        _ => {
            *supported = false;
            Ok(unknown())
        }
    }
}

fn admit<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    s: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<()> {
    variables(x, s, work)?;
    envelope(source, true, work)?;
    let mut supported = true;
    let _ = domain(source, x, work, &mut supported)?;
    if !supported {
        return Err(ForwardMellinError::Unsupported(
            "original positive-axis domain or source class",
        ));
    }
    Ok(())
}

// Generator: structural monomial/binomial factor recognition.
// Rational shifts are allowed outside kernels. Inside the exponential or
// binomial, the independent rational-function checker admits integer powers
// only; recognition must not promise a wider algebraic-function lane.
fn integer_arithmetic<M: BudgetMeter>(source: &Expr, work: &mut Work<'_, M>) -> Result<bool> {
    work.charge(Dimension::ComputeSteps, 1)?;
    Ok(match source {
        Expr::Pow(base, exponent) => {
            matches!(exponent.as_ref(), Expr::Integer(_)) && integer_arithmetic(base, work)?
        }
        Expr::Add(children) | Expr::Mul(children) => {
            for child in children {
                if !integer_arithmetic(child, work)? {
                    return Ok(false);
                }
            }
            true
        }
        Expr::Integer(_) | Expr::Rational(_) | Expr::Sym(_) => true,
        _ => false,
    })
}

fn monomial<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Option<(BigRational, BigRational)>> {
    if let Some(value) = literal(source, work)? {
        return Ok(Some((value, BigRational::zero())));
    }
    match source {
        Expr::Sym(symbol) if symbol == x => Ok(Some((BigRational::one(), BigRational::one()))),
        Expr::Mul(children) => {
            let (mut coefficient, mut power) = (BigRational::one(), BigRational::zero());
            for child in children {
                let Some((c, p)) = monomial(child, x, work)? else {
                    return Ok(None);
                };
                coefficient = work.mul(&coefficient, &c)?;
                power = work.add(&power, &p)?;
            }
            Ok(Some((coefficient, power)))
        }
        Expr::Add(children) => {
            if !integer_arithmetic(source, work)? {
                return Ok(None);
            }
            let mut answer: Option<(BigRational, BigRational)> = None;
            for child in children {
                let Some((c, p)) = monomial(child, x, work)? else {
                    return Ok(None);
                };
                if let Some((old, degree)) = answer.as_mut() {
                    if !work.equal(degree, &p)? {
                        return Ok(None);
                    }
                    *old = work.add(old, &c)?;
                } else {
                    answer = Some((c, p));
                }
            }
            Ok(answer.or_else(|| Some((BigRational::zero(), BigRational::zero()))))
        }
        Expr::Pow(base, exponent) => {
            let Some((coefficient, degree)) = monomial(base, x, work)? else {
                return Ok(None);
            };
            let Some(exponent) = literal(exponent, work)? else {
                return Ok(None);
            };
            let coefficient = if coefficient.is_one() {
                coefficient
            } else if exponent.is_integer() {
                let n = exponent
                    .numer()
                    .to_i64()
                    .ok_or(ForwardMellinError::Refused("integer power bound"))?
                    as i32;
                work.pow(&coefficient, n)?
            } else {
                return Ok(None);
            };
            Ok(Some((coefficient, work.mul(&degree, &exponent)?)))
        }
        _ => Ok(None),
    }
}

fn integral_power(value: &BigRational) -> Result<i32> {
    let power = if value.is_integer() {
        value.numer().to_i64()
    } else {
        None
    };
    match power {
        Some(p) if p != 0 && p.unsigned_abs() <= MAX_POWER as u64 => Ok(p as i32),
        _ => Err(ForwardMellinError::Unsupported(
            "nonzero integer kernel power within 16",
        )),
    }
}

fn recognize<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<MellinKernel> {
    if source.is_zero() {
        return Ok(MellinKernel::Zero);
    }
    work.allocate(MAX_NODES, MAX_NODES as u64 * 8)?;
    let mut pending = vec![source];
    let (mut coefficient, mut shift) = (BigRational::one(), BigRational::zero());
    let mut kernel = None;
    while let Some(factor) = pending.pop() {
        if let Expr::Mul(children) = factor {
            pending.extend(children);
            continue;
        }
        if let Some((c, p)) = monomial(factor, x, work)? {
            coefficient = work.mul(&coefficient, &c)?;
            shift = work.add(&shift, &p)?;
            continue;
        }
        if kernel.is_some() {
            return Err(ForwardMellinError::Unsupported(
                "products of distinct kernels",
            ));
        }
        kernel = match factor {
            Expr::Function(name, arguments) if name == "exp" && arguments.len() == 1 => {
                if !integer_arithmetic(&arguments[0], work)? {
                    return Err(ForwardMellinError::Unsupported(
                        "noninteger powers inside exponential kernel",
                    ));
                }
                let (rate, power) = monomial(&arguments[0], x, work)?.ok_or(
                    ForwardMellinError::Unsupported("nonmonomial exponential argument"),
                )?;
                if !rate.is_negative() {
                    return Err(ForwardMellinError::Unsupported("nondecaying exponential"));
                }
                Some(MellinKernel::Gamma {
                    coefficient: BigRational::one(),
                    shift: BigRational::zero(),
                    power: integral_power(&power)?,
                    rate: work.neg(&rate)?,
                })
            }
            Expr::Pow(base, exponent) => {
                let order = literal(exponent, work)?
                    .ok_or(ForwardMellinError::Unsupported("nonrational beta order"))?;
                if !order.is_negative() {
                    return Err(ForwardMellinError::Unsupported("nondecaying beta power"));
                }
                let Expr::Add(terms) = base.as_ref() else {
                    return Err(ForwardMellinError::Unsupported("nonbinomial beta base"));
                };
                if !integer_arithmetic(base, work)? {
                    return Err(ForwardMellinError::Unsupported(
                        "noninteger powers inside beta base",
                    ));
                }
                let (mut constant, mut scale) = (BigRational::zero(), BigRational::zero());
                let mut power = None;
                for term in terms {
                    let (c, degree) = monomial(term, x, work)?
                        .ok_or(ForwardMellinError::Unsupported("nonbinomial beta base"))?;
                    if degree.is_zero() {
                        constant = work.add(&constant, &c)?;
                    } else {
                        let p = integral_power(&degree)?;
                        if power.is_some_and(|old| old != p) {
                            return Err(ForwardMellinError::Unsupported(
                                "beta base with multiple nonconstant powers",
                            ));
                        }
                        power = Some(p);
                        scale = work.add(&scale, &c)?;
                    }
                }
                if !constant.is_positive() || !scale.is_positive() {
                    return Err(ForwardMellinError::Unsupported(
                        "beta coefficients must be positive",
                    ));
                }
                Some(MellinKernel::Beta {
                    coefficient: BigRational::one(),
                    shift: BigRational::zero(),
                    power: power.ok_or(ForwardMellinError::Unsupported("constant beta base"))?,
                    constant,
                    scale,
                    order: work.neg(&order)?,
                })
            }
            _ => {
                return Err(ForwardMellinError::Unsupported(
                    "source outside the gamma/beta kernel catalog",
                ));
            }
        };
    }
    if coefficient.is_zero() {
        return Err(ForwardMellinError::Unsupported(
            "nonliteral held zero source",
        ));
    }
    match kernel.as_mut() {
        Some(
            MellinKernel::Gamma {
                coefficient: c,
                shift: a,
                ..
            }
            | MellinKernel::Beta {
                coefficient: c,
                shift: a,
                ..
            },
        ) => {
            *c = coefficient;
            *a = shift;
        }
        _ => {
            return Err(ForwardMellinError::Unsupported(
                "pure power has no nonempty absolute convergence strip",
            ));
        }
    }
    Ok(kernel.expect("kernel checked"))
}

fn generate_formula<M: BudgetMeter>(
    kernel: &MellinKernel,
    s: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<(Expr, MellinStrip)> {
    let (coefficient, shift, power) = match kernel {
        MellinKernel::Zero => {
            return Ok((
                work.scalar(&BigRational::zero())?,
                MellinStrip {
                    lower: MellinStripBound::NegativeInfinity,
                    upper: MellinStripBound::PositiveInfinity,
                },
            ));
        }
        MellinKernel::Gamma {
            coefficient,
            shift,
            power,
            ..
        }
        | MellinKernel::Beta {
            coefficient,
            shift,
            power,
            ..
        } => (coefficient, shift, *power),
    };
    let denominator = BigRational::from(i64::from(power));
    let slope = work.div(&BigRational::one(), &denominator)?;
    let offset = work.div(shift, &denominator)?;
    let negative_slope = work.neg(&slope)?;
    let negative_offset = work.neg(&offset)?;
    let amplitude = work.div(coefficient, &BigRational::from(i64::from(power.abs())))?;
    let mut factors = vec![work.scalar(&amplitude)?];
    let boundary = work.neg(shift)?;
    let strip = match kernel {
        MellinKernel::Gamma { rate, .. } => {
            let base = work.scalar(rate)?;
            let exponent = work.affine(s, &negative_slope, &negative_offset)?;
            factors.push(work.power(base, exponent)?);
            let argument = work.affine(s, &slope, &offset)?;
            factors.push(work.gamma(argument)?);
            if power > 0 {
                MellinStrip {
                    lower: MellinStripBound::Finite(boundary),
                    upper: MellinStripBound::PositiveInfinity,
                }
            } else {
                MellinStrip {
                    lower: MellinStripBound::NegativeInfinity,
                    upper: MellinStripBound::Finite(boundary),
                }
            }
        }
        MellinKernel::Beta {
            constant,
            scale,
            order,
            ..
        } => {
            let base = work.scalar(constant)?;
            let offset_minus_order = work.sub(&offset, order)?;
            let exponent = work.affine(s, &slope, &offset_minus_order)?;
            factors.push(work.power(base, exponent)?);
            let base = work.scalar(scale)?;
            let exponent = work.affine(s, &negative_slope, &negative_offset)?;
            factors.push(work.power(base, exponent)?);
            let argument = work.affine(s, &slope, &offset)?;
            factors.push(work.gamma(argument)?);
            let order_minus_offset = work.sub(order, &offset)?;
            let argument = work.affine(s, &negative_slope, &order_minus_offset)?;
            factors.push(work.gamma(argument)?);
            let argument = work.scalar(order)?;
            let divisor = work.gamma(argument)?;
            let inverse = work.scalar(&BigRational::from(-1))?;
            factors.push(work.power(divisor, inverse)?);
            let far = work.mul(&denominator, order)?;
            let far = work.sub(&far, shift)?;
            if power > 0 {
                MellinStrip {
                    lower: MellinStripBound::Finite(boundary),
                    upper: MellinStripBound::Finite(far),
                }
            } else {
                MellinStrip {
                    lower: MellinStripBound::Finite(far),
                    upper: MellinStripBound::Finite(boundary),
                }
            }
        }
        MellinKernel::Zero => unreachable!(),
    };
    Ok((work.product(factors)?, strip))
}

/// Generate only; callers must verify before publishing a mathematical result.
pub fn generate_mellin_candidate<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    s: &Symbol,
    meter: &mut M,
) -> Result<MellinCandidate> {
    let mut work = Work::new(meter);
    admit(source, x, s, &mut work)?;
    let kernel = recognize(source, x, &mut work)?;
    let (transform, strip) = generate_formula(&kernel, s, &mut work)?;
    work.checkpoint()?;
    Ok(MellinCandidate {
        kernel,
        transform,
        strip,
    })
}

// Verifier: exact sparse Laurent rational functions. No recognizer calls.
// A vector avoids unmetered rational-key comparisons; exponents are bounded
// machine integers and all coefficient arithmetic goes through Work.
type Polynomial = Vec<(i32, BigRational)>;

fn insert<M: BudgetMeter>(
    polynomial: &mut Polynomial,
    degree: i32,
    value: BigRational,
    work: &mut Work<'_, M>,
) -> Result<()> {
    if degree.unsigned_abs() > MAX_DEGREE as u32 {
        return Err(ForwardMellinError::Refused("verifier Laurent degree bound"));
    }
    work.charge(Dimension::ComputeSteps, polynomial.len().max(1) as u64)?;
    if let Some(index) = polynomial.iter().position(|(old, _)| *old == degree) {
        let sum = work.add(&polynomial[index].1, &value)?;
        if sum.is_zero() {
            polynomial.remove(index);
        } else {
            polynomial[index].1 = sum;
        }
    } else if !value.is_zero() {
        if polynomial.len() == MAX_TERMS {
            return Err(ForwardMellinError::Refused(
                "verifier polynomial term bound",
            ));
        }
        work.allocate(1, 96 + 2 * bits(&value).div_ceil(8))?;
        polynomial.push((degree, value));
        polynomial.sort_unstable_by_key(|(degree, _)| *degree);
    }
    Ok(())
}

fn poly_sum<M: BudgetMeter>(
    left: &Polynomial,
    right: &Polynomial,
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    let mut result = vec![];
    for (degree, coefficient) in left.iter().chain(right) {
        insert(&mut result, *degree, work.copy(coefficient)?, work)?;
    }
    Ok(result)
}

fn poly_product<M: BudgetMeter>(
    left: &Polynomial,
    right: &Polynomial,
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    work.charge(
        Dimension::ComputeSteps,
        left.len().saturating_mul(right.len()).max(1) as u64,
    )?;
    let mut result = vec![];
    for (a, c) in left {
        for (b, d) in right {
            insert(&mut result, a + b, work.mul(c, d)?, work)?;
        }
    }
    Ok(result)
}

fn poly_derivative<M: BudgetMeter>(
    source: &Polynomial,
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    let mut result = vec![];
    for (power, coefficient) in source {
        if *power != 0 {
            insert(
                &mut result,
                power - 1,
                work.mul(coefficient, &BigRational::from(i64::from(*power)))?,
                work,
            )?;
        }
    }
    Ok(result)
}

fn poly_scale<M: BudgetMeter>(
    source: &Polynomial,
    coefficient: &BigRational,
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    let mut result = vec![];
    for (power, value) in source {
        insert(&mut result, *power, work.mul(value, coefficient)?, work)?;
    }
    Ok(result)
}

fn polynomial<M: BudgetMeter>(
    degree: i32,
    value: BigRational,
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    let mut result = vec![];
    insert(&mut result, degree, value, work)?;
    Ok(result)
}

struct Quotient {
    numerator: Polynomial,
    denominator: Polynomial,
}

fn ratio<M: BudgetMeter>(
    degree: i32,
    value: BigRational,
    work: &mut Work<'_, M>,
) -> Result<Quotient> {
    Ok(Quotient {
        numerator: polynomial(degree, value, work)?,
        denominator: polynomial(0, BigRational::one(), work)?,
    })
}

fn ratio_add<M: BudgetMeter>(
    a: &Quotient,
    b: &Quotient,
    work: &mut Work<'_, M>,
) -> Result<Quotient> {
    let left = poly_product(&a.numerator, &b.denominator, work)?;
    let right = poly_product(&b.numerator, &a.denominator, work)?;
    Ok(Quotient {
        numerator: poly_sum(&left, &right, work)?,
        denominator: poly_product(&a.denominator, &b.denominator, work)?,
    })
}

fn ratio_mul<M: BudgetMeter>(
    a: &Quotient,
    b: &Quotient,
    work: &mut Work<'_, M>,
) -> Result<Quotient> {
    Ok(Quotient {
        numerator: poly_product(&a.numerator, &b.numerator, work)?,
        denominator: poly_product(&a.denominator, &b.denominator, work)?,
    })
}

fn derivative<M: BudgetMeter>(source: &Quotient, work: &mut Work<'_, M>) -> Result<Quotient> {
    let numerator_prime = poly_derivative(&source.numerator, work)?;
    let denominator_prime = poly_derivative(&source.denominator, work)?;
    let first = poly_product(&numerator_prime, &source.denominator, work)?;
    let second = poly_product(&source.numerator, &denominator_prime, work)?;
    let second = poly_scale(&second, &BigRational::from(-1), work)?;
    Ok(Quotient {
        numerator: poly_sum(&first, &second, work)?,
        denominator: poly_product(&source.denominator, &source.denominator, work)?,
    })
}

fn quotient<M: BudgetMeter>(source: &Expr, x: &Symbol, work: &mut Work<'_, M>) -> Result<Quotient> {
    if let Some(value) = literal(source, work)? {
        return ratio(0, value, work);
    }
    match source {
        Expr::Sym(symbol) if symbol == x => ratio(1, BigRational::one(), work),
        Expr::Add(children) | Expr::Mul(children) => {
            let addition = matches!(source, Expr::Add(_));
            let mut result = ratio(
                0,
                if addition {
                    BigRational::zero()
                } else {
                    BigRational::one()
                },
                work,
            )?;
            for child in children {
                let child = quotient(child, x, work)?;
                result = if addition {
                    ratio_add(&result, &child, work)?
                } else {
                    ratio_mul(&result, &child, work)?
                };
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            let value = literal(exponent, work)?.ok_or(ForwardMellinError::Verification(
                "nonrational verifier exponent",
            ))?;
            if !value.is_integer() {
                return Err(ForwardMellinError::Verification(
                    "noninteger rational-function exponent",
                ));
            }
            let power = value
                .numer()
                .to_i64()
                .ok_or(ForwardMellinError::Refused("verifier exponent bound"))?;
            if power.unsigned_abs() > MAX_POWER as u64 {
                return Err(ForwardMellinError::Refused("verifier exponent bound"));
            }
            let mut base = quotient(base, x, work)?;
            if base.numerator.is_empty() && power <= 0 {
                return Err(ForwardMellinError::Verification(
                    "undefined zero power in original candidate",
                ));
            }
            if power < 0 {
                std::mem::swap(&mut base.numerator, &mut base.denominator);
            }
            if base.denominator.is_empty() {
                return Err(ForwardMellinError::Verification(
                    "zero original denominator",
                ));
            }
            let mut result = ratio(0, BigRational::one(), work)?;
            for _ in 0..power.unsigned_abs() {
                result = ratio_mul(&result, &base, work)?;
            }
            Ok(result)
        }
        _ => Err(ForwardMellinError::Verification(
            "nonrational source subexpression",
        )),
    }
}

fn logarithmic_derivative<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Quotient> {
    if let Some(value) = literal(source, work)? {
        if value.is_zero() {
            return Err(ForwardMellinError::Verification(
                "zero source has no logarithmic derivative",
            ));
        }
        return ratio(0, BigRational::zero(), work);
    }
    match source {
        Expr::Mul(children) => {
            let mut result = ratio(0, BigRational::zero(), work)?;
            for child in children {
                let child = logarithmic_derivative(child, x, work)?;
                result = ratio_add(&result, &child, work)?;
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            let coefficient = literal(exponent, work)?.ok_or(ForwardMellinError::Verification(
                "nonrational logarithmic exponent",
            ))?;
            let result = logarithmic_derivative(base, x, work)?;
            Ok(Quotient {
                numerator: poly_scale(&result.numerator, &coefficient, work)?,
                denominator: result.denominator,
            })
        }
        Expr::Function(name, arguments) if name == "exp" && arguments.len() == 1 => {
            let argument = quotient(&arguments[0], x, work)?;
            derivative(&argument, work)
        }
        _ => {
            let value = quotient(source, x, work)?;
            if value.numerator.is_empty() {
                return Err(ForwardMellinError::Verification("zero source factor"));
            }
            let derivative = derivative(&value, work)?;
            Ok(Quotient {
                numerator: poly_product(&derivative.numerator, &value.denominator, work)?,
                denominator: poly_product(&derivative.denominator, &value.numerator, work)?,
            })
        }
    }
}

fn same_ratio<M: BudgetMeter>(
    left: &Quotient,
    right: &Quotient,
    work: &mut Work<'_, M>,
) -> Result<bool> {
    let a = poly_product(&left.numerator, &right.denominator, work)?;
    let b = poly_product(&right.numerator, &left.denominator, work)?;
    if a.len() != b.len() {
        return Ok(false);
    }
    for ((ad, ac), (bd, bc)) in a.iter().zip(&b) {
        if ad != bd || !work.equal(ac, bc)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Debug)]
struct Anchor {
    coefficient: BigRational,
    exponential: BigRational,
    powers: Vec<(BigRational, BigRational)>,
}

impl Anchor {
    fn rational(value: BigRational) -> Self {
        Self {
            coefficient: value,
            exponential: BigRational::zero(),
            powers: vec![],
        }
    }
}

fn anchor_power<M: BudgetMeter>(
    base: BigRational,
    exponent: BigRational,
    work: &mut Work<'_, M>,
) -> Result<Anchor> {
    if exponent.is_integer() {
        let power = exponent
            .numer()
            .to_i64()
            .ok_or(ForwardMellinError::Refused("anchor power bound"))?;
        if power.unsigned_abs() > MAX_POWER as u64 {
            return Err(ForwardMellinError::Refused("anchor power bound"));
        }
        return Ok(Anchor::rational(work.pow(&base, power as i32)?));
    }
    if !base.is_positive() {
        return Err(ForwardMellinError::Verification(
            "anchor is not positive for rational power",
        ));
    }
    if base.is_one() {
        return Ok(Anchor::rational(BigRational::one()));
    }
    work.allocate(1, 128 + 2 * (bits(&base) + bits(&exponent)).div_ceil(8))?;
    Ok(Anchor {
        coefficient: BigRational::one(),
        exponential: BigRational::zero(),
        powers: vec![(base, exponent)],
    })
}

fn anchor<M: BudgetMeter>(source: &Expr, x: &Symbol, work: &mut Work<'_, M>) -> Result<Anchor> {
    if let Some(value) = literal(source, work)? {
        return Ok(Anchor::rational(value));
    }
    match source {
        Expr::Sym(symbol) if symbol == x => Ok(Anchor::rational(BigRational::one())),
        Expr::Mul(children) => {
            let mut result = Anchor::rational(BigRational::one());
            for child in children {
                let child = anchor(child, x, work)?;
                result.coefficient = work.mul(&result.coefficient, &child.coefficient)?;
                result.exponential = work.add(&result.exponential, &child.exponential)?;
                for (base, exponent) in child.powers {
                    let mut found = None;
                    for (index, (old, _)) in result.powers.iter().enumerate() {
                        if work.equal(&base, old)? {
                            found = Some(index);
                            break;
                        }
                    }
                    if let Some(index) = found {
                        result.powers[index].1 = work.add(&result.powers[index].1, &exponent)?;
                    } else {
                        if result.powers.len() == MAX_TERMS {
                            return Err(ForwardMellinError::Refused("anchor factor bound"));
                        }
                        work.allocate(1, 128)?;
                        result.powers.push((base, exponent));
                    }
                }
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            let base = anchor(base, x, work)?;
            if !base.exponential.is_zero() || !base.powers.is_empty() {
                return Err(ForwardMellinError::Verification(
                    "nested nonrational anchor",
                ));
            }
            let exponent = literal(exponent, work)?.ok_or(ForwardMellinError::Verification(
                "nonrational anchor exponent",
            ))?;
            anchor_power(base.coefficient, exponent, work)
        }
        Expr::Function(name, arguments) if name == "exp" && arguments.len() == 1 => {
            let value = at_one(&quotient(&arguments[0], x, work)?, work)?;
            Ok(Anchor {
                coefficient: BigRational::one(),
                exponential: value,
                powers: vec![],
            })
        }
        Expr::Add(_) => Ok(Anchor::rational(at_one(&quotient(source, x, work)?, work)?)),
        _ => Err(ForwardMellinError::Verification(
            "unsupported anchor source",
        )),
    }
}

fn at_one<M: BudgetMeter>(source: &Quotient, work: &mut Work<'_, M>) -> Result<BigRational> {
    let mut numerator = BigRational::zero();
    let mut denominator = BigRational::zero();
    for (_, coefficient) in &source.numerator {
        numerator = work.add(&numerator, coefficient)?;
    }
    for (_, coefficient) in &source.denominator {
        denominator = work.add(&denominator, coefficient)?;
    }
    work.div(&numerator, &denominator)
}

fn same_anchor<M: BudgetMeter>(
    left: &Anchor,
    right: &Anchor,
    work: &mut Work<'_, M>,
) -> Result<bool> {
    if !work.equal(&left.coefficient, &right.coefficient)?
        || !work.equal(&left.exponential, &right.exponential)?
        || left.powers.len() != right.powers.len()
    {
        return Ok(false);
    }
    for (base, exponent) in &left.powers {
        let mut found = false;
        for (other_base, other_exponent) in &right.powers {
            if work.equal(base, other_base)? && work.equal(exponent, other_exponent)? {
                found = true;
                break;
            }
        }
        if !found {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Independently replay the source through differentiation and an exact
/// anchor. The generator's monomial/binomial recognizer is never called.
fn verify_source<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    kernel: &MellinKernel,
    work: &mut Work<'_, M>,
) -> Result<()> {
    let (coefficient, shift, power) = match kernel {
        MellinKernel::Zero => {
            return if source.is_zero() {
                Ok(())
            } else {
                Err(ForwardMellinError::Verification("zero source binding"))
            };
        }
        MellinKernel::Gamma {
            coefficient,
            shift,
            power,
            ..
        }
        | MellinKernel::Beta {
            coefficient,
            shift,
            power,
            ..
        } => (coefficient, shift, *power),
    };
    if coefficient.is_zero() || power == 0 || power.unsigned_abs() > MAX_POWER as u32 {
        return Err(ForwardMellinError::Verification(
            "invalid nonzero kernel parameters",
        ));
    }
    let mut expected_anchor = Anchor::rational(work.copy(coefficient)?);
    let expected_derivative = match kernel {
        MellinKernel::Gamma { rate, .. } => {
            if !rate.is_positive() {
                return Err(ForwardMellinError::Verification("nonpositive gamma scale"));
            }
            expected_anchor.exponential = work.neg(rate)?;
            let mut numerator = polynomial(-1, work.copy(shift)?, work)?;
            let derivative = work.mul(rate, &BigRational::from(-i64::from(power)))?;
            insert(&mut numerator, power - 1, derivative, work)?;
            Quotient {
                numerator,
                denominator: polynomial(0, BigRational::one(), work)?,
            }
        }
        MellinKernel::Beta {
            constant,
            scale,
            order,
            ..
        } => {
            if !constant.is_positive() || !scale.is_positive() || !order.is_positive() {
                return Err(ForwardMellinError::Verification(
                    "nonpositive beta parameter",
                ));
            }
            let base = work.add(constant, scale)?;
            let exponent = work.neg(order)?;
            let value = anchor_power(base, exponent, work)?;
            expected_anchor.coefficient =
                work.mul(&expected_anchor.coefficient, &value.coefficient)?;
            expected_anchor.powers = value.powers;
            let first = work.mul(shift, constant)?;
            let pq = work.mul(&BigRational::from(i64::from(power)), order)?;
            let remaining = work.sub(shift, &pq)?;
            let second = work.mul(scale, &remaining)?;
            let mut numerator = polynomial(-1, first, work)?;
            insert(&mut numerator, power - 1, second, work)?;
            let mut denominator = polynomial(0, work.copy(constant)?, work)?;
            insert(&mut denominator, power, work.copy(scale)?, work)?;
            Quotient {
                numerator,
                denominator,
            }
        }
        MellinKernel::Zero => unreachable!(),
    };
    let actual_derivative = logarithmic_derivative(source, x, work)?;
    if !same_ratio(&actual_derivative, &expected_derivative, work)? {
        return Err(ForwardMellinError::Verification(
            "original logarithmic derivative",
        ));
    }
    let actual_anchor = anchor(source, x, work)?;
    if !same_anchor(&actual_anchor, &expected_anchor, work)? {
        return Err(ForwardMellinError::Verification("original value at one"));
    }
    Ok(())
}

// Formula checker reads a multiplicative signature rather than calling the
// formula generator. Gamma arguments and rational-base exponents are checked
// as affine polynomials in the original typed transform variable.
#[derive(Debug)]
enum FormulaFactor {
    Power(BigRational, BigRational, BigRational),
    Gamma(BigRational, BigRational, i32),
}

fn affine_coefficients<M: BudgetMeter>(
    expression: &Expr,
    s: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<(BigRational, BigRational)> {
    // Generated affine arguments contain only entire polynomial arithmetic.
    // Refuse original powers before reduction: (s-1)^0 or a cancelled pole
    // must not become an innocuous one on a strip that contains s=1.
    // This candidate-only rule does not restrict positive-axis source powers.
    fn entire_arithmetic<M: BudgetMeter>(source: &Expr, work: &mut Work<'_, M>) -> Result<()> {
        work.charge(Dimension::ComputeSteps, 1)?;
        match source {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Sym(_) => Ok(()),
            Expr::Add(children) | Expr::Mul(children) => {
                for child in children {
                    entire_arithmetic(child, work)?;
                }
                Ok(())
            }
            _ => Err(ForwardMellinError::Verification(
                "original affine formula domain",
            )),
        }
    }
    entire_arithmetic(expression, work)?;
    let value = quotient(expression, s, work)?;
    if value.denominator.len() != 1 || value.denominator[0].0 != 0 {
        return Err(ForwardMellinError::Verification(
            "nonaffine transform argument",
        ));
    }
    let divisor = &value.denominator[0].1;
    let (mut slope, mut offset) = (BigRational::zero(), BigRational::zero());
    for (degree, coefficient) in &value.numerator {
        match degree {
            0 => offset = work.div(coefficient, divisor)?,
            1 => slope = work.div(coefficient, divisor)?,
            _ => {
                return Err(ForwardMellinError::Verification(
                    "nonaffine transform argument",
                ));
            }
        }
    }
    Ok((slope, offset))
}

fn formula_factors<M: BudgetMeter>(
    expression: &Expr,
    s: &Symbol,
    work: &mut Work<'_, M>,
    scalar: &mut BigRational,
    factors: &mut Vec<FormulaFactor>,
) -> Result<()> {
    if let Some(value) = literal(expression, work)? {
        *scalar = work.mul(scalar, &value)?;
        return Ok(());
    }
    match expression {
        Expr::Mul(children) => {
            for child in children {
                formula_factors(child, s, work, scalar, factors)?;
            }
        }
        Expr::Function(name, arguments) if name == "gamma" && arguments.len() == 1 => {
            let (a, b) = affine_coefficients(&arguments[0], s, work)?;
            work.allocate(1, 192)?;
            factors.push(FormulaFactor::Gamma(a, b, 1));
        }
        Expr::Pow(base, exponent) => {
            if let Some(base) = literal(base, work)? {
                if !base.is_positive() {
                    return Err(ForwardMellinError::Verification(
                        "nonpositive formula power base",
                    ));
                }
                let (a, b) = affine_coefficients(exponent, s, work)?;
                work.allocate(1, 192)?;
                factors.push(FormulaFactor::Power(base, a, b));
            } else if let Expr::Function(name, arguments) = base.as_ref()
                && name == "gamma"
                && arguments.len() == 1
                && literal(exponent, work)?.is_some_and(|value| value == BigRational::from(-1))
            {
                let (a, b) = affine_coefficients(&arguments[0], s, work)?;
                work.allocate(1, 192)?;
                factors.push(FormulaFactor::Gamma(a, b, -1));
            } else {
                return Err(ForwardMellinError::Verification(
                    "unexpected formula factor",
                ));
            }
        }
        _ => {
            return Err(ForwardMellinError::Verification(
                "unexpected formula expression",
            ));
        }
    }
    if factors.len() > 8 {
        return Err(ForwardMellinError::Verification("formula factor count"));
    }
    Ok(())
}

fn verify_formula_and_strip<M: BudgetMeter>(
    candidate: &MellinCandidate,
    s: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<()> {
    let (coefficient, shift, power) = match &candidate.kernel {
        MellinKernel::Zero => {
            return if candidate.transform.is_zero()
                && candidate.strip.lower == MellinStripBound::NegativeInfinity
                && candidate.strip.upper == MellinStripBound::PositiveInfinity
            {
                Ok(())
            } else {
                Err(ForwardMellinError::Verification("zero formula or strip"))
            };
        }
        MellinKernel::Gamma {
            coefficient,
            shift,
            power,
            ..
        }
        | MellinKernel::Beta {
            coefficient,
            shift,
            power,
            ..
        } => (coefficient, shift, *power),
    };
    let mut scalar = BigRational::one();
    let mut factors = vec![];
    formula_factors(&candidate.transform, s, work, &mut scalar, &mut factors)?;
    let expected_scalar = work.div(coefficient, &BigRational::from(i64::from(power.abs())))?;
    if !work.equal(&scalar, &expected_scalar)? {
        return Err(ForwardMellinError::Verification("formula amplitude"));
    }
    let p = BigRational::from(i64::from(power));
    let slope = work.div(&BigRational::one(), &p)?;
    let offset = work.mul(shift, &slope)?;
    let opposite_slope = work.neg(&slope)?;
    let opposite_offset = work.neg(&offset)?;
    let mut expected = vec![FormulaFactor::Gamma(
        work.copy(&slope)?,
        work.copy(&offset)?,
        1,
    )];
    let boundary = work.neg(shift)?;
    let (lower, upper) = match &candidate.kernel {
        MellinKernel::Gamma { rate, .. } => {
            if !rate.is_one() {
                expected.push(FormulaFactor::Power(
                    work.copy(rate)?,
                    opposite_slope,
                    opposite_offset,
                ));
            }
            if power > 0 {
                (
                    MellinStripBound::Finite(boundary),
                    MellinStripBound::PositiveInfinity,
                )
            } else {
                (
                    MellinStripBound::NegativeInfinity,
                    MellinStripBound::Finite(boundary),
                )
            }
        }
        MellinKernel::Beta {
            constant,
            scale,
            order,
            ..
        } => {
            if !constant.is_one() {
                expected.push(FormulaFactor::Power(
                    work.copy(constant)?,
                    work.copy(&slope)?,
                    work.sub(&offset, order)?,
                ));
            }
            if !scale.is_one() {
                expected.push(FormulaFactor::Power(
                    work.copy(scale)?,
                    work.copy(&opposite_slope)?,
                    opposite_offset,
                ));
            }
            expected.push(FormulaFactor::Gamma(
                opposite_slope,
                work.sub(order, &offset)?,
                1,
            ));
            expected.push(FormulaFactor::Gamma(
                BigRational::zero(),
                work.copy(order)?,
                -1,
            ));
            // Near zero/infinity the beta source has exponents a and a-p*q;
            // exchanging them for p<0 gives this maximal open interval.
            let tail = work.mul(&p, order)?;
            let tail = work.sub(&tail, shift)?;
            if power > 0 {
                (
                    MellinStripBound::Finite(boundary),
                    MellinStripBound::Finite(tail),
                )
            } else {
                (
                    MellinStripBound::Finite(tail),
                    MellinStripBound::Finite(boundary),
                )
            }
        }
        MellinKernel::Zero => unreachable!(),
    };
    if candidate.strip.lower != lower || candidate.strip.upper != upper {
        return Err(ForwardMellinError::Verification(
            "fundamental strip endpoints",
        ));
    }
    if factors.len() != expected.len() {
        return Err(ForwardMellinError::Verification(
            "formula factor multiplicity",
        ));
    }
    for factor in factors {
        let mut found = None;
        for (index, other) in expected.iter().enumerate() {
            let equal = match (&factor, other) {
                (FormulaFactor::Power(a, b, c), FormulaFactor::Power(d, e, f)) => {
                    work.equal(a, d)? && work.equal(b, e)? && work.equal(c, f)?
                }
                (FormulaFactor::Gamma(a, b, c), FormulaFactor::Gamma(d, e, f)) => {
                    c == f && work.equal(a, d)? && work.equal(b, e)?
                }
                _ => false,
            };
            if equal {
                found = Some(index);
                break;
            }
        }
        let index = found.ok_or(ForwardMellinError::Verification(
            "gamma/beta substitution formula",
        ))?;
        expected.remove(index);
    }
    Ok(())
}

/// Check original source, formula, and strip without invoking recognition.
pub fn verify_mellin_candidate<M: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    s: &Symbol,
    candidate: &MellinCandidate,
    meter: &mut M,
) -> Result<()> {
    let mut work = Work::new(meter);
    admit(source, x, s, &mut work)?;
    envelope(&candidate.transform, false, &mut work)?;
    let parameters: Vec<&BigRational> = match &candidate.kernel {
        MellinKernel::Zero => vec![],
        MellinKernel::Gamma {
            coefficient,
            shift,
            rate,
            ..
        } => vec![coefficient, shift, rate],
        MellinKernel::Beta {
            coefficient,
            shift,
            constant,
            scale,
            order,
            ..
        } => vec![coefficient, shift, constant, scale, order],
    };
    for parameter in parameters {
        if bits(parameter) > MAX_WORK_BITS / 2 {
            return Err(ForwardMellinError::Refused(
                "candidate parameter growth bound",
            ));
        }
        work.charge(Dimension::ComputeSteps, bits(parameter).div_ceil(64))?;
    }
    for bound in [&candidate.strip.lower, &candidate.strip.upper] {
        if let MellinStripBound::Finite(value) = bound {
            if bits(value) > MAX_WORK_BITS / 2 {
                return Err(ForwardMellinError::Refused("candidate strip growth bound"));
            }
            work.charge(Dimension::ComputeSteps, bits(value).div_ceil(64))?;
        }
    }
    verify_source(source, x, &candidate.kernel, &mut work)?;
    verify_formula_and_strip(candidate, s, &mut work)?;
    work.checkpoint()?;
    Ok(())
}

pub fn metered_mellin_transform<G: BudgetMeter, V: BudgetMeter>(
    source: &Expr,
    x: &Symbol,
    s: &Symbol,
    generator: &mut G,
    verifier: &mut V,
) -> Result<VerifiedMellin> {
    let candidate = generate_mellin_candidate(source, x, s, generator)?;
    verify_mellin_candidate(source, x, s, &candidate, verifier)?;
    generator.checkpoint()?;
    verifier.checkpoint()?;
    Ok(VerifiedMellin {
        transform: candidate.transform,
        strip: candidate.strip,
    })
}

pub fn mellin_transform(source: &Expr, x: &Symbol, s: &Symbol) -> Result<VerifiedMellin> {
    metered_mellin_transform(source, x, s, &mut Unbounded, &mut Unbounded)
}
