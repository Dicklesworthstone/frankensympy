//! Bernoulli equations y' + p*y = q(x)*y^n with rational constant p.
//!
//! The input q is its original canonical ascending coefficient vector. For
//! integer n != 0,1, the substitution u=y^(1-n), on y != 0, gives the linear
//! equation u' + (1-n)*p*u = (1-n)*q. Generation solves a triangular polynomial
//! system; the reference checker independently reconstructs every coefficient
//! of its differential identity. The homogeneous coefficient is free and
//! nonvanishing, so the resulting implicit relation describes the nonzero
//! local family. No choice of algebraic branch, global interval, or value
//! through a zero of u is certified. The separate zero solution is valid
//! exactly when n>1. These are in-process receipts, not portable proofs.

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use std::sync::Arc;
use thiserror::Error;

const MAX_DEGREE: usize = 16;
const MAX_INPUT_BITS: u64 = 1024;
const MAX_BITS: u64 = 16_384;
const MAX_TEXT: usize = 16_384;
const MAX_COMPUTE: u64 = 2_000_000;
const MAX_MEMORY: u64 = 32 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum BernoulliError {
    #[error("invalid Bernoulli equation: {0}")]
    InvalidSource(&'static str),
    #[error("Bernoulli resource limit: {0}")]
    ResourceLimit(&'static str),
    #[error("Bernoulli verification failed: {0}")]
    Verification(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for BernoulliError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, BernoulliError>;

/// An untrusted polynomial particular for the transformed linear equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BernoulliCertificate {
    pub particular: Vec<BigRational>,
}

/// Accepted source-bound implicit nonzero family, with no chosen root branch.
#[derive(Clone, Debug)]
pub struct VerifiedBernoulli {
    p: BigRational,
    q: Vec<BigRational>,
    exponent: i32,
    variable: Symbol,
    constant: Symbol,
    certificate: BernoulliCertificate,
    transformed: Expr,
}

impl VerifiedBernoulli {
    pub fn p(&self) -> &BigRational {
        &self.p
    }
    pub fn q(&self) -> &[BigRational] {
        &self.q
    }
    pub fn exponent(&self) -> i32 {
        self.exponent
    }
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn constant(&self) -> &Symbol {
        &self.constant
    }
    pub fn certificate(&self) -> &BernoulliCertificate {
        &self.certificate
    }
    /// Right side of y^(1-n)=u(x), valid on any nonzero chosen local branch.
    pub fn transformed(&self) -> &Expr {
        &self.transformed
    }
    pub fn has_zero_solution(&self) -> bool {
        self.exponent > 1
    }
    pub fn into_transformed(self) -> Expr {
        self.transformed
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    used: [u64; 3],
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
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
        self.meter.charge_batch(charges)?;
        self.used = next;
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; 3],
        }
    }
    fn reserve(&mut self, count: usize, bytes: usize) -> Result<()> {
        self.checkpoint()?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, count as u64),
            (Dimension::MemoryBytes, bytes as u64),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }
    fn scalar(&mut self, value: &BigRational) -> Result<BigRational> {
        validate_scalar(value, MAX_BITS)?;
        self.reserve(1, 128 + bits(value).div_ceil(8) as usize * 2)?;
        Ok(value.clone())
    }
    fn preflight(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        self.checkpoint()?;
        if bits(a).saturating_add(bits(b)).saturating_add(2) > MAX_BITS {
            return Err(BernoulliError::ResourceLimit(
                "intermediate coefficient height",
            ));
        }
        Ok(())
    }
    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_mul(b, self)?)
    }
    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_div(b, self)?)
    }
    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_add(b, self)?)
    }
    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_sub(b, self)?)
    }
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits())
}
fn validate_scalar(value: &BigRational, cap: u64) -> Result<()> {
    if bits(value) > cap {
        return Err(BernoulliError::ResourceLimit("coefficient height"));
    }
    Ok(())
}
fn admit_coefficients(values: &[BigRational], max_len: usize, cap: u64) -> Result<()> {
    if values.is_empty()
        || values.len() > max_len
        || (values.len() > 1 && values.last().is_some_and(Zero::is_zero))
    {
        return Err(BernoulliError::InvalidSource(
            "noncanonical or oversized coefficient vector",
        ));
    }
    for value in values {
        validate_scalar(value, cap)?;
    }
    Ok(())
}
fn admit(p: &BigRational, q: &[BigRational], n: i32, x: &Symbol, c: &Symbol) -> Result<()> {
    if !(-16..=-1).contains(&n) && !(2..=16).contains(&n) {
        return Err(BernoulliError::InvalidSource(
            "exponent must be in [-16,-1] or [2,16]",
        ));
    }
    if x == c {
        return Err(BernoulliError::InvalidSource(
            "integration constant aliases the variable",
        ));
    }
    if x.name.len().saturating_add(c.name.len()) > MAX_TEXT {
        return Err(BernoulliError::ResourceLimit("symbol text"));
    }
    validate_scalar(p, MAX_INPUT_BITS)?;
    admit_coefficients(q, MAX_DEGREE + 1, MAX_INPUT_BITS)
}
fn trim(values: &mut Vec<BigRational>) {
    while values.len() > 1 && values.last().is_some_and(Zero::is_zero) {
        values.pop();
    }
}

pub fn generate_bernoulli_certificate<M: BudgetMeter>(
    p: &BigRational,
    q: &[BigRational],
    n: i32,
    x: &Symbol,
    c: &Symbol,
    meter: &mut M,
) -> Result<BernoulliCertificate> {
    meter.checkpoint()?;
    admit(p, q, n, x, c)?;
    let mut work = Work::new(meter);
    let m = BigRational::from(i64::from(1 - n));
    let alpha = work.mul(&m, p)?;
    work.reserve(q.len() + 1, 128 * (q.len() + 1))?;
    let mut result = vec![BigRational::zero(); q.len() + usize::from(alpha.is_zero())];
    if alpha.is_zero() {
        for (degree, value) in q.iter().enumerate() {
            let rhs = work.mul(&m, value)?;
            result[degree + 1] = work.div(&rhs, &BigRational::from((degree + 1) as i64))?;
        }
    } else {
        for degree in (0..q.len()).rev() {
            let rhs = work.mul(&m, &q[degree])?;
            let residual = if let Some(next) = result.get(degree + 1) {
                let derivative = work.mul(&BigRational::from((degree + 1) as i64), next)?;
                work.sub(&rhs, &derivative)?
            } else {
                rhs
            };
            result[degree] = work.div(&residual, &alpha)?;
        }
    }
    trim(&mut result);
    work.checkpoint()?;
    Ok(BernoulliCertificate { particular: result })
}

fn lift<M: BudgetMeter>(
    p: &BigRational,
    particular: &[BigRational],
    n: i32,
    x: &Symbol,
    c: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    // Charge every expression, symbol-name copy and rational copy before
    // constructing the bounded output. No general simplifier is invoked.
    let nodes = 16 + particular.len() * 6;
    let bytes = nodes * 192
        + (particular.len() + 1) * x.name.len()
        + c.name.len()
        + particular
            .iter()
            .map(|v| 2 * bits(v).div_ceil(8) as usize)
            .sum::<usize>();
    work.reserve(nodes, bytes)?;
    let rate = work.mul(&BigRational::from(i64::from(n - 1)), p)?;
    let homogeneous = if rate.is_zero() {
        Expr::Sym(c.clone())
    } else {
        let argument = if rate.is_one() {
            Expr::Sym(x.clone())
        } else {
            Expr::Mul(vec![rational_expr(rate), Expr::Sym(x.clone())])
        };
        Expr::Mul(vec![
            Expr::Sym(c.clone()),
            Expr::Function("exp".into(), vec![argument]),
        ])
    };
    let mut terms = vec![homogeneous];
    for (degree, coefficient) in particular.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let value = rational_expr(work.scalar(coefficient)?);
        let power = match degree {
            0 => {
                terms.push(value);
                continue;
            }
            1 => Expr::Sym(x.clone()),
            degree => Expr::Pow(
                Arc::new(Expr::Sym(x.clone())),
                Arc::new(Expr::from_i64(degree as i64)),
            ),
        };
        terms.push(if coefficient.is_one() {
            power
        } else {
            Expr::Mul(vec![value, power])
        });
    }
    Ok(if terms.len() == 1 {
        terms.remove(0)
    } else {
        Expr::Add(terms)
    })
}

pub fn verify_bernoulli_certificate<M: BudgetMeter>(
    p: &BigRational,
    q: &[BigRational],
    n: i32,
    x: &Symbol,
    c: &Symbol,
    certificate: BernoulliCertificate,
    meter: &mut M,
) -> Result<VerifiedBernoulli> {
    meter.checkpoint()?;
    admit(p, q, n, x, c)?;
    admit_coefficients(&certificate.particular, MAX_DEGREE + 2, MAX_BITS)?;
    let mut work = Work::new(meter);
    let m = BigRational::from(i64::from(1 - n));
    let alpha = work.mul(&m, p)?;
    let zero = BigRational::zero();
    // The reference lane forms D(r)+alpha*r from the complete candidate,
    // including its highest coefficient. It never repeats the backward solve.
    for degree in 0..q.len().max(certificate.particular.len()) {
        let derivative = work.mul(
            &BigRational::from((degree + 1) as i64),
            certificate.particular.get(degree + 1).unwrap_or(&zero),
        )?;
        let linear = work.mul(&alpha, certificate.particular.get(degree).unwrap_or(&zero))?;
        let actual = work.add(&derivative, &linear)?;
        let expected = work.mul(&m, q.get(degree).unwrap_or(&zero))?;
        if actual != expected {
            return Err(BernoulliError::Verification(
                "transformed differential coefficient identity",
            ));
        }
    }
    let transformed = lift(p, &certificate.particular, n, x, c, &mut work)?;
    work.reserve(
        q.len() + 4,
        256 * (q.len() + 4) + x.name.len() + c.name.len(),
    )?;
    let p = work.scalar(p)?;
    let q = q
        .iter()
        .map(|v| work.scalar(v))
        .collect::<Result<Vec<_>>>()?;
    work.checkpoint()?;
    Ok(VerifiedBernoulli {
        p,
        q,
        exponent: n,
        variable: x.clone(),
        constant: c.clone(),
        certificate,
        transformed,
    })
}

pub fn metered_bernoulli<G: BudgetMeter, V: BudgetMeter>(
    p: &BigRational,
    q: &[BigRational],
    n: i32,
    x: &Symbol,
    c: &Symbol,
    generator: &mut G,
    verifier: &mut V,
) -> Result<VerifiedBernoulli> {
    let candidate = generate_bernoulli_certificate(p, q, n, x, c, generator)?;
    verify_bernoulli_certificate(p, q, n, x, c, candidate, verifier)
}

pub fn bernoulli(
    p: &BigRational,
    q: &[BigRational],
    n: i32,
    x: &Symbol,
    c: &Symbol,
) -> Result<VerifiedBernoulli> {
    metered_bernoulli(p, q, n, x, c, &mut Unbounded, &mut Unbounded)
}
