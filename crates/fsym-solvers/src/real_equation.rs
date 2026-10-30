//! Complete bounded real solution sets for exact rational expressions.
//!
//! Admit the original typed tree before cancellation. Every inverse records
//! its original base numerator, including inverses in held products or zeroth
//! powers. Independently checked Bezout and quotient identities remove domain
//! exclusions; the core real-root checker establishes complete real counting.
//! A zero numerator describes the real line minus those original exclusions.
//!
//! This synchronous coefficient-level lane is not a general solver certificate
//! registry or runtime reservation. Generator and reference-checker meters are
//! separate; fixed intrinsic arithmetic and allocation bounds remain in force.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::real_roots::{
    MAX_REAL_ROOT_DEGREE, MAX_REAL_ROOT_INPUT_BITS, RealRootError, VerifiedRealRoots,
    isolate_real_roots,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_polys::gcd::BezoutCertificate;
use num_traits::{One, Zero};
use thiserror::Error;

mod algebraic_value;
mod real_inequality;
pub use algebraic_value::{
    AlgebraicValueCertificate, VerifiedAlgebraicValue, evaluate_real_algebraic,
    metered_evaluate_real_algebraic, verify_real_algebraic_value,
};
pub use real_inequality::{
    VerifiedRealInequality, metered_solve_real_inequality, solve_real_inequality,
};

const MAX_NODES: usize = 2_048;
const MAX_DEPTH: usize = 64;
const MAX_FANOUT: usize = 256;
const MAX_GUARDS: usize = 128;
const MAX_WORK_DEGREE: usize = 2 * MAX_REAL_ROOT_DEGREE;
const MAX_WORK_BITS: u64 = 32_768;
const MAX_WORK: u64 = 100_000_000;
const MAX_ALLOCATION: u64 = 256 * 1024 * 1024;

type Poly = Vec<BigRational>;
type Result<T> = std::result::Result<T, RealEquationError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RealEquationError {
    #[error("exact real equation resource limit exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("invalid real equation reduction: {0}")]
    InvalidReduction(&'static str),
    #[error(transparent)]
    Roots(#[from] RealRootError),
    #[error("exact real equation work stopped: {0}")]
    Meter(#[from] MeterError),
}

/// Complete finite/cofinite real set bound to the original typed source.
/// If is_complement() is false, polynomial() has exactly the solution roots;
/// otherwise its roots are exactly the excluded real points. The polynomial
/// is nonzero, monic and square-free; constant one represents no finite roots.
/// No unchecked deserializer or public mutable field can construct acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRealEquation {
    expression: Expr,
    variable: Symbol,
    polynomial: UnivariatePoly,
    roots: VerifiedRealRoots,
    complement: bool,
}

impl VerifiedRealEquation {
    pub fn expression(&self) -> &Expr {
        &self.expression
    }
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn polynomial(&self) -> &UnivariatePoly {
        &self.polynomial
    }
    pub fn roots(&self) -> &VerifiedRealRoots {
        &self.roots
    }
    pub fn is_complement(&self) -> bool {
        self.complement
    }
}

fn bits(q: &BigRational) -> u64 {
    q.numer().bits().max(q.denom().bits()).max(1)
}
fn trim(p: &mut Poly) {
    while p.len() > 1 && p.last().is_some_and(Zero::is_zero) {
        p.pop();
    }
}
fn zero(p: &[BigRational]) -> bool {
    p.iter().all(Zero::is_zero)
}
fn limit(reason: &'static str) -> RealEquationError {
    RealEquationError::ResourceLimit(reason)
}
fn invalid(reason: &'static str) -> RealEquationError {
    RealEquationError::InvalidReduction(reason)
}

struct Work<'a, M> {
    meter: &'a mut M,
    compute: u64,
    allocated: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Result<Self> {
        meter.checkpoint()?;
        Ok(Self {
            meter,
            compute: 0,
            allocated: 0,
        })
    }
    fn charge(&mut self, steps: u64, bytes: u64) -> Result<()> {
        self.meter.checkpoint()?;
        self.compute = self
            .compute
            .checked_add(steps.max(1))
            .ok_or(limit("work"))?;
        self.allocated = self
            .allocated
            .checked_add(bytes.max(1))
            .ok_or(limit("allocation"))?;
        if self.compute > MAX_WORK || self.allocated > MAX_ALLOCATION {
            return Err(limit("cumulative work/allocation"));
        }
        self.meter.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1)),
            (Dimension::MemoryBytes, bytes.max(1)),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }
    fn scalar(&mut self, q: &BigRational) -> Result<()> {
        if bits(q) > MAX_WORK_BITS {
            return Err(limit("intermediate rational bits"));
        }
        self.charge(1, 32 + (q.numer().bits() + q.denom().bits()).div_ceil(8))
    }
    fn binary(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        let growth = bits(a).saturating_add(bits(b)).saturating_add(1);
        if growth > MAX_WORK_BITS {
            return Err(limit("rational operation growth"));
        }
        let limbs = growth.div_ceil(32).max(1);
        self.charge(limbs.saturating_mul(limbs), 64 + growth.div_ceil(8) * 8)
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
            return Err(invalid("zero scalar divisor"));
        }
        self.binary(a, b)?;
        Ok(a / b)
    }
    fn zeros(&mut self, len: usize) -> Result<Poly> {
        if len == 0 || len > MAX_WORK_DEGREE + 1 {
            return Err(limit("intermediate degree"));
        }
        self.charge(1, len as u64 * 32)?;
        Ok(vec![BigRational::zero(); len])
    }
    fn constant(&mut self, n: i64) -> Result<Poly> {
        self.charge(1, 64)?;
        Ok(vec![BigRational::from_integer(n.into())])
    }
    fn copy(&mut self, p: &[BigRational]) -> Result<Poly> {
        if p.is_empty() || p.len() > MAX_WORK_DEGREE + 1 {
            return Err(limit("intermediate degree"));
        }
        self.charge(1, p.len() as u64 * 32)?;
        for q in p {
            self.scalar(q)?;
        }
        Ok(p.to_vec())
    }
    fn sum(&mut self, a: &[BigRational], b: &[BigRational], subtract: bool) -> Result<Poly> {
        let mut p = self.zeros(a.len().max(b.len()))?;
        let z = BigRational::zero();
        for (i, value) in p.iter_mut().enumerate() {
            let left = a.get(i).unwrap_or(&z);
            let right = b.get(i).unwrap_or(&z);
            *value = if subtract {
                self.sub(left, right)?
            } else {
                self.add(left, right)?
            };
        }
        trim(&mut p);
        Ok(p)
    }
    fn product(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<Poly> {
        if zero(a) || zero(b) {
            return self.constant(0);
        }
        let mut p = self.zeros(a.len() + b.len() - 1)?;
        for (i, left) in a.iter().enumerate() {
            for (j, right) in b.iter().enumerate() {
                let product = self.mul(left, right)?;
                p[i + j] = self.add(&p[i + j], &product)?;
            }
        }
        trim(&mut p);
        Ok(p)
    }
    fn scale(&mut self, p: &[BigRational], scale: &BigRational) -> Result<Poly> {
        let mut result = self.zeros(p.len())?;
        for (target, value) in result.iter_mut().zip(p) {
            *target = self.mul(value, scale)?;
        }
        trim(&mut result);
        Ok(result)
    }
    fn monic(&mut self, p: &[BigRational]) -> Result<Poly> {
        if zero(p) {
            return self.constant(0);
        }
        let scale = self.div(
            &BigRational::one(),
            p.last().ok_or(invalid("empty polynomial"))?,
        )?;
        self.scale(p, &scale)
    }
    fn derivative(&mut self, p: &[BigRational]) -> Result<Poly> {
        let mut out = self.zeros(p.len().saturating_sub(1).max(1))?;
        for (i, q) in p.iter().enumerate().skip(1) {
            out[i - 1] = self.mul(q, &BigRational::from_integer(BigInt::from(i as u64)))?;
        }
        trim(&mut out);
        Ok(out)
    }
    fn division(&mut self, a: &[BigRational], b: &[BigRational]) -> Result<(Poly, Poly)> {
        if zero(b) {
            return Err(invalid("zero polynomial divisor"));
        }
        let mut r = self.copy(a)?;
        let mut q = self.zeros(a.len().saturating_sub(b.len()) + 1)?;
        while !zero(&r) && r.len() >= b.len() {
            let old_len = r.len();
            let shift = old_len - b.len();
            let c = self.div(r.last().unwrap(), b.last().unwrap())?;
            self.scalar(&c)?;
            q[shift] = c.clone();
            for (j, bj) in b.iter().enumerate() {
                let product = self.mul(&c, bj)?;
                r[shift + j] = self.sub(&r[shift + j], &product)?;
            }
            trim(&mut r);
            if !zero(&r) && r.len() >= old_len {
                return Err(invalid("division did not decrease degree"));
            }
        }
        trim(&mut q);
        Ok((q, r))
    }
    fn power(&mut self, p: &[BigRational], exponent: u32) -> Result<Poly> {
        if p.len().saturating_sub(1).saturating_mul(exponent as usize) > MAX_REAL_ROOT_DEGREE {
            return Err(limit("expanded rational degree"));
        }
        let mut result = self.constant(1)?;
        let mut base = self.copy(p)?;
        let mut exponent = exponent;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = self.product(&result, &base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = self.product(&base, &base)?;
            }
        }
        Ok(result)
    }
}

/// Visit every held child before recursive conversion or source cloning.
fn admit<M: BudgetMeter>(
    expression: &Expr,
    variable: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<bool> {
    if variable.name.len() > 4096 {
        return Err(limit("symbol bytes"));
    }
    work.charge(1, 64)?;
    let mut stack = vec![(expression, 0usize)];
    let mut count = 0usize;
    let mut supported = true;
    while let Some((node, depth)) = stack.pop() {
        count += 1;
        if count > MAX_NODES || depth > MAX_DEPTH {
            return Err(limit("expression nodes/depth"));
        }
        work.charge(1, 128)?; // Includes the bounded retained source clone.
        let children: &[Expr] = match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_REAL_ROOT_INPUT_BITS {
                    return Err(limit("input integer bits"));
                }
                work.charge(1, 32 + value.bits().div_ceil(8))?;
                &[]
            }
            Expr::Rational(value) => {
                if bits(value) > MAX_REAL_ROOT_INPUT_BITS {
                    return Err(limit("input rational bits"));
                }
                work.scalar(value)?;
                &[]
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > 4096 {
                    return Err(limit("symbol bytes"));
                }
                work.charge(1, symbol.name.len() as u64 + 64)?;
                supported &= symbol == variable;
                &[]
            }
            Expr::Add(terms) | Expr::Mul(terms) => terms,
            Expr::Pow(base, exponent) => {
                match exponent.as_ref() {
                    Expr::Integer(n) => {
                        if n.to_i64()
                            .is_none_or(|n| n.unsigned_abs() > MAX_REAL_ROOT_DEGREE as u64)
                        {
                            return Err(limit("integer exponent"));
                        }
                    }
                    _ => supported = false,
                }
                work.charge(1, 64)?;
                stack.push((base, depth + 1));
                stack.push((exponent, depth + 1));
                continue;
            }
            Expr::Function(name, terms) => {
                if name.len() > 4096 {
                    return Err(limit("function bytes"));
                }
                supported = false;
                terms
            }
            Expr::Const(_) => {
                supported = false;
                &[]
            }
        };
        if children.len() > MAX_FANOUT || stack.len() + children.len() > MAX_NODES {
            return Err(limit("expression fanout"));
        }
        work.charge(1, children.len() as u64 * 32)?;
        stack.extend(children.iter().map(|child| (child, depth + 1)));
    }
    Ok(supported)
}

struct Fraction {
    numerator: Poly,
    denominator: Poly,
}

fn convert<M: BudgetMeter>(
    e: &Expr,
    variable: &Symbol,
    guards: &mut Vec<Poly>,
    work: &mut Work<'_, M>,
) -> Result<Fraction> {
    work.charge(1, 64)?;
    let (numerator, denominator) = match e {
        Expr::Integer(n) => {
            work.charge(1, 64 + n.bits().div_ceil(8))?;
            (
                vec![BigRational::from_integer(n.clone())],
                work.constant(1)?,
            )
        }
        Expr::Rational(q) => (work.copy(std::slice::from_ref(q))?, work.constant(1)?),
        Expr::Sym(symbol) if symbol == variable => (
            vec![BigRational::zero(), BigRational::one()],
            work.constant(1)?,
        ),
        Expr::Add(terms) | Expr::Mul(terms) => {
            let adding = matches!(e, Expr::Add(_));
            let mut numerator = work.constant(if adding { 0 } else { 1 })?;
            let mut denominator = work.constant(1)?;
            for term in terms {
                let part = convert(term, variable, guards, work)?;
                numerator = if adding {
                    let left = work.product(&numerator, &part.denominator)?;
                    let right = work.product(&part.numerator, &denominator)?;
                    work.sum(&left, &right, false)?
                } else {
                    work.product(&numerator, &part.numerator)?
                };
                denominator = work.product(&denominator, &part.denominator)?;
                if numerator.len() > MAX_REAL_ROOT_DEGREE + 1
                    || denominator.len() > MAX_REAL_ROOT_DEGREE + 1
                {
                    return Err(limit("expanded rational degree"));
                }
            }
            (numerator, denominator)
        }
        Expr::Pow(base, exponent) => {
            let Expr::Integer(exponent) = exponent.as_ref() else {
                return Err(invalid("unadmitted exponent"));
            };
            let exponent = exponent.to_i64().ok_or(limit("integer exponent"))?;
            let part = convert(base, variable, guards, work)?;
            // A held zeroth power still inherits its original base's domain.
            if exponent < 0 {
                if guards.len() == MAX_GUARDS {
                    return Err(limit("original domain guards"));
                }
                work.charge(1, 32)?;
                guards.push(work.copy(&part.numerator)?);
            }
            let n = work.power(&part.numerator, exponent.unsigned_abs() as u32)?;
            let d = work.power(&part.denominator, exponent.unsigned_abs() as u32)?;
            if exponent < 0 { (d, n) } else { (n, d) }
        }
        _ => return Err(invalid("unadmitted expression")),
    };
    Ok(Fraction {
        numerator,
        denominator,
    })
}

/// Candidate search; its Euclidean trace is not used as acceptance evidence.
fn propose_gcd<M: BudgetMeter>(
    a: &[BigRational],
    b: &[BigRational],
    symbol: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<BezoutCertificate> {
    let (mut old_r, mut r) = (work.copy(a)?, work.copy(b)?);
    let (mut old_s, mut s) = (work.constant(1)?, work.constant(0)?);
    let (mut old_t, mut t) = (work.constant(0)?, work.constant(1)?);
    while !zero(&r) {
        let (q, next_r) = work.division(&old_r, &r)?;
        let qs = work.product(&q, &s)?;
        let qt = work.product(&q, &t)?;
        let next_s = work.sum(&old_s, &qs, true)?;
        let next_t = work.sum(&old_t, &qt, true)?;
        (old_r, r) = (r, next_r);
        (old_s, s) = (s, next_s);
        (old_t, t) = (t, next_t);
    }
    if zero(&old_r) {
        return Err(invalid("both gcd inputs are zero"));
    }
    let scale = work.div(&BigRational::one(), old_r.last().unwrap())?;
    work.charge(1, symbol.name.len() as u64 * 3 + 256)?;
    Ok(BezoutCertificate {
        gcd: UnivariatePoly::new(symbol.clone(), work.scale(&old_r, &scale)?),
        u: UnivariatePoly::new(symbol.clone(), work.scale(&old_s, &scale)?),
        v: UnivariatePoly::new(symbol.clone(), work.scale(&old_t, &scale)?),
    })
}

/// Independent reference check of monicity, divisibility and Bezout identity.
fn check_gcd<M: BudgetMeter>(
    a: &[BigRational],
    b: &[BigRational],
    symbol: &Symbol,
    cert: &BezoutCertificate,
    work: &mut Work<'_, M>,
) -> Result<()> {
    for p in [&cert.gcd, &cert.u, &cert.v] {
        p.validate_shape()
            .map_err(|_| invalid("gcd polynomial shape"))?;
        if p.gen_sym != *symbol {
            return Err(invalid("gcd variable identity"));
        }
        work.copy(&p.coeffs)?;
    }
    if cert.gcd.is_zero() || !cert.gcd.leading_coeff().is_one() {
        return Err(invalid("nonmonic gcd"));
    }
    let ua = work.product(&cert.u.coeffs, a)?;
    let vb = work.product(&cert.v.coeffs, b)?;
    if work.sum(&ua, &vb, false)? != cert.gcd.coeffs {
        return Err(invalid("Bezout identity"));
    }
    for p in [a, b] {
        let (_, remainder) = work.division(p, &cert.gcd.coeffs)?;
        if !zero(&remainder) {
            return Err(invalid("gcd divisibility"));
        }
    }
    Ok(())
}

fn checked_gcd<G: BudgetMeter, V: BudgetMeter>(
    a: &[BigRational],
    b: &[BigRational],
    symbol: &Symbol,
    generator: &mut Work<'_, G>,
    verifier: &mut Work<'_, V>,
) -> Result<Poly> {
    let certificate = propose_gcd(a, b, symbol, generator)?;
    check_gcd(a, b, symbol, &certificate, verifier)?;
    Ok(certificate.gcd.coeffs)
}

fn checked_quotient<G: BudgetMeter, V: BudgetMeter>(
    a: &[BigRational],
    b: &[BigRational],
    generator: &mut Work<'_, G>,
    verifier: &mut Work<'_, V>,
) -> Result<Poly> {
    let (q, remainder) = generator.division(a, b)?;
    if !zero(&remainder) || verifier.product(&q, b)? != a {
        return Err(invalid("exact quotient identity"));
    }
    Ok(q)
}

fn square_free<G: BudgetMeter, V: BudgetMeter>(
    p: &[BigRational],
    symbol: &Symbol,
    generator: &mut Work<'_, G>,
    verifier: &mut Work<'_, V>,
) -> Result<Poly> {
    if zero(p) {
        return Err(invalid("zero square-free input"));
    }
    let derivative = verifier.derivative(p)?;
    let gcd = checked_gcd(p, &derivative, symbol, generator, verifier)?;
    let quotient = checked_quotient(p, &gcd, generator, verifier)?;
    verifier.monic(&quotient)
}

/// Solve expression = 0 over all real values of the typed variable.
/// Ok(None) denotes unsupported original grammar. Recognized resource,
/// cancellation and reduction failures are errors, never an empty solution set.
pub fn metered_solve_real_equation(
    expression: &Expr,
    variable: &Symbol,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealEquation>> {
    let mut generator = Work::new(generator_meter)?;
    if !admit(expression, variable, &mut generator)? {
        return Ok(None);
    }
    let mut verifier = Work::new(verifier_meter)?;
    let mut guards = Vec::new();
    let fraction = convert(expression, variable, &mut guards, &mut generator)?;
    generator.charge(1, 32)?;
    guards.push(fraction.denominator);
    let nowhere_defined = guards.iter().any(|p| zero(p));
    let complement = zero(&fraction.numerator) && !nowhere_defined;
    let mut polynomial = if nowhere_defined || complement {
        generator.constant(1)?
    } else {
        square_free(&fraction.numerator, variable, &mut generator, &mut verifier)?
    };
    if !nowhere_defined {
        for guard in guards {
            if guard.len() == 1 {
                continue;
            }
            if complement {
                let guard = square_free(&guard, variable, &mut generator, &mut verifier)?;
                let gcd =
                    checked_gcd(&polynomial, &guard, variable, &mut generator, &mut verifier)?;
                let remaining = checked_quotient(&guard, &gcd, &mut generator, &mut verifier)?;
                polynomial = verifier.product(&polynomial, &remaining)?;
            } else {
                let gcd =
                    checked_gcd(&polynomial, &guard, variable, &mut generator, &mut verifier)?;
                polynomial = checked_quotient(&polynomial, &gcd, &mut generator, &mut verifier)?;
            }
            if polynomial.len() > MAX_REAL_ROOT_DEGREE + 1 {
                return Err(limit("solution/exclusion degree"));
            }
        }
    }
    if polynomial
        .iter()
        .any(|q| bits(q) > MAX_REAL_ROOT_INPUT_BITS)
    {
        return Err(limit("solution coefficient bits"));
    }
    let roots = isolate_real_roots(&polynomial, generator.meter, verifier.meter)?;
    if roots.intervals().iter().any(|root| root.multiplicity != 1) {
        return Err(invalid("solution polynomial is not square-free"));
    }
    generator.charge(1, 256 + variable.name.len() as u64 * 2)?;
    Ok(Some(VerifiedRealEquation {
        expression: expression.clone(),
        variable: variable.clone(),
        polynomial: UnivariatePoly::new(variable.clone(), polynomial),
        roots,
        complement,
    }))
}

/// Convenience entry point retaining intrinsic bounds without a caller budget.
pub fn solve_real_equation(
    expression: &Expr,
    variable: &Symbol,
) -> Result<Option<VerifiedRealEquation>> {
    metered_solve_real_equation(expression, variable, &mut Unbounded, &mut Unbounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_gcd_check_rejects_mutations_and_same_name_variable_alias() {
        let x = Symbol::new("x");
        let a = vec![(-1).into(), 0.into(), 1.into()];
        let b = vec![(-1).into(), 1.into()];
        let mut meter = Unbounded;
        let mut generator = Work::new(&mut meter).unwrap();
        let certificate = propose_gcd(&a, &b, &x, &mut generator).unwrap();
        let check = |cert: &BezoutCertificate| {
            check_gcd(&a, &b, &x, cert, &mut Work::new(&mut Unbounded).unwrap())
        };
        assert!(check(&certificate).is_ok());
        let mut forged = certificate.clone();
        forged.gcd = UnivariatePoly::one(x.clone());
        assert!(check(&forged).is_err());
        let mut forged = certificate.clone();
        forged.u = UnivariatePoly::one(x.clone());
        assert!(check(&forged).is_err());
        let mut forged = certificate.clone();
        forged.gcd = UnivariatePoly::new(x.clone(), vec![(-2).into(), 2.into()]);
        assert!(check(&forged).is_err());
        let mut forged = certificate;
        forged.v.gen_sym = Symbol::with_identity(
            "x",
            fsym_core::SymbolIdentity::from_assumptions_digest([1; 32]),
        );
        assert!(check(&forged).is_err());
    }
}
