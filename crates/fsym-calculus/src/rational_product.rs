//! Bounded exact finite products over Q(k), preserving original source poles.
//!
//! Short ranges use direct rational evaluation. Longer ranges group rational
//! polynomial factors into integer-shift orbits to find a rational
//! multiplicative telescoper. Higher-degree factors must have a bounded modular
//! witness excluding every integer zero. A dependency-separated checker proves the full
//! coefficient identity, every original inverse obligation and the endpoint
//! value. Unsupported products remain unevaluated; invalid sources, exhausted
//! resources and cancellation are terminal. No symbolic-bound or infinite
//! product theorem is claimed.

mod modular;

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, PolyError, UnivariatePoly, VerifiedRationalSource,
    metered_complete_factorization, metered_rational_source,
};
use fsym_proof_kernel::rational_product::{
    MAX_DIRECT_PRODUCT_TERMS, RationalProductCertificate, RationalProductDomain,
    RationalProductFactors, RationalProductVerificationError, verify_rational_product,
};
use fsym_proof_kernel::rational_sum::{RationalSumPole, RationalSumQuadratic};
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use thiserror::Error;

const MAX_INPUT_DEGREE: usize = 64;
const MAX_OUTPUT_DEGREE: usize = 128;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 16_384;
const MAX_SHIFT_SPAN: i64 = 64;
const MAX_SOURCE_NODES: usize = 4096;
const MAX_SOURCE_DEPTH: usize = 64;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum RationalProductError {
    #[error("rational product resource limit: {0}")]
    ResourceLimit(&'static str),
    #[error("rational product is undefined: {0}")]
    Undefined(&'static str),
    #[error(transparent)]
    Polynomial(#[from] PolyError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Verification(#[from] RationalProductVerificationError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalProductError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalProductError>;
type Factor = (UnivariatePoly, u32);
type DomainFactors = (RationalProductDomain, Vec<Factor>, bool);
type DomainFactorCandidates = (Vec<Factor>, bool);

/// An immutable in-process product receipt, bound to the original expression,
/// its exact typed index, original integer bounds, and retained inverse guards.
#[derive(Debug, Clone)]
pub struct VerifiedRationalProduct {
    source: VerifiedRationalSource,
    lower: BigInt,
    upper: BigInt,
    value: BigRational,
    certificate: RationalProductCertificate,
}

impl VerifiedRationalProduct {
    pub fn source(&self) -> &VerifiedRationalSource {
        &self.source
    }
    pub fn lower(&self) -> &BigInt {
        &self.lower
    }
    pub fn upper(&self) -> &BigInt {
        &self.upper
    }
    pub fn value(&self) -> &BigRational {
        &self.value
    }
    pub fn certificate(&self) -> &RationalProductCertificate {
        &self.certificate
    }
    pub fn into_value(self) -> BigRational {
        self.value
    }
}

/// One cumulative cap for each phase, including lower-level source checking,
/// factorization, scalar arithmetic, output materialization and verification.
struct Capped<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
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
                let error = MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                });
                self.stopped = Some(error);
                return Err(error);
            }
            next[index] += amount;
        }
        let result = self.inner.charge_batch(charges);
        if let Err(error) = result {
            self.stopped = Some(error);
        } else {
            self.used = next;
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

impl<'a, M: BudgetMeter> Capped<'a, M> {
    fn new(inner: &'a mut M) -> Self {
        Self {
            inner,
            used: [0; 3],
            stopped: None,
        }
    }
    fn polynomial<T>(&self, result: std::result::Result<T, PolyError>) -> Result<T> {
        if let Some(error) = self.stopped {
            Err(error.into())
        } else {
            result.map_err(Into::into)
        }
    }
}

fn reserve(meter: &mut impl BudgetMeter, count: usize) -> Result<()> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, count.max(1) as u64),
        (Dimension::AllocationCount, count.max(1) as u64),
        (
            Dimension::MemoryBytes,
            count.max(1) as u64 * (64 + 2 * MAX_WORK_BITS.div_ceil(8)),
        ),
    ])?;
    Ok(())
}

fn checked(value: BigRational) -> Result<BigRational> {
    if value.numer().bits().max(value.denom().bits()) > MAX_WORK_BITS {
        Err(RationalProductError::ResourceLimit(
            "coefficient or output bit height",
        ))
    } else {
        Ok(value)
    }
}

fn add(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    checked(a.metered_add(b, meter)?)
}
fn mul(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    checked(a.metered_mul(b, meter)?)
}
fn div(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    if b.is_zero() {
        return Err(RationalProductError::Undefined(
            "zero original factor or denominator",
        ));
    }
    checked(a.metered_div(b, meter)?)
}

/// Bounded borrowed grammar recognition. A nonrational source is not normalized
/// into this lane. Inspect every child before deciding unsupported, including
/// source hidden under zero products and powers.
pub(crate) fn is_rational_source(
    source: &Expr,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<bool> {
    reserve(meter, 1)?;
    let mut stack = vec![(source, 0usize)];
    let mut visited = 0usize;
    let mut supported = true;
    while let Some((node, depth)) = stack.pop() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        visited += 1;
        if visited > MAX_SOURCE_NODES || depth > MAX_SOURCE_DEPTH {
            return Err(RationalProductError::ResourceLimit(
                "original node/depth envelope",
            ));
        }
        let children = match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_INPUT_BITS {
                    return Err(RationalProductError::ResourceLimit(
                        "original integer height",
                    ));
                }
                &[][..]
            }
            Expr::Rational(value) => {
                if value.numer().bits().max(value.denom().bits()) > MAX_INPUT_BITS {
                    return Err(RationalProductError::ResourceLimit(
                        "original rational height",
                    ));
                }
                &[][..]
            }
            Expr::Sym(symbol) => {
                supported &= symbol == variable;
                &[][..]
            }
            Expr::Add(children) | Expr::Mul(children) => children.as_slice(),
            Expr::Function(_, children) => {
                supported = false;
                children.as_slice()
            }
            Expr::Pow(base, exponent) => {
                if !matches!(exponent.as_ref(), Expr::Integer(_)) {
                    supported = false;
                } else if !matches!(exponent.as_ref(), Expr::Integer(value)
                    if value.to_i64().is_some_and(|n| (-256..=256).contains(&n)))
                {
                    return Err(RationalProductError::ResourceLimit(
                        "original exponent envelope",
                    ));
                }
                if visited + stack.len() + 2 > MAX_SOURCE_NODES {
                    return Err(RationalProductError::ResourceLimit(
                        "original node envelope",
                    ));
                }
                reserve(meter, 2)?;
                stack.push((base, depth + 1));
                stack.push((exponent, depth + 1));
                continue;
            }
            _ => {
                supported = false;
                &[][..]
            }
        };
        if children.len() > MAX_SOURCE_NODES.saturating_sub(visited + stack.len()) {
            return Err(RationalProductError::ResourceLimit(
                "original node envelope",
            ));
        }
        if !children.is_empty() {
            reserve(meter, children.len())?;
        }
        stack.extend(children.iter().map(|child| (child, depth + 1)));
    }
    Ok(supported)
}

fn dense(
    poly: &MultivariatePoly,
    variable: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<UnivariatePoly> {
    if poly.generators() != std::slice::from_ref(variable) {
        return Err(PolyError::General("product source has another typed index".into()).into());
    }
    let degree = poly.degree_in(0) as usize;
    if degree > MAX_INPUT_DEGREE {
        return Err(RationalProductError::ResourceLimit(
            "source polynomial degree",
        ));
    }
    reserve(meter, degree + 1)?;
    let mut coefficients = vec![BigRational::zero(); degree + 1];
    for (powers, coefficient) in &poly.terms {
        coefficients[powers[0] as usize] = coefficient.clone();
    }
    Ok(UnivariatePoly::new(variable.clone(), coefficients))
}

/// Deliberately different from the verifier's Horner recurrence: scalar powers
/// are accumulated upwards and combined with the original dense coefficients.
fn evaluate(
    poly: &UnivariatePoly,
    point: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<BigRational> {
    let mut power = BigRational::one();
    let mut value = BigRational::zero();
    for (index, coefficient) in poly.coeffs.iter().enumerate() {
        let term = mul(coefficient, &power, meter)?;
        value = add(&value, &term, meter)?;
        if index + 1 != poly.coeffs.len() {
            power = mul(&power, point, meter)?;
        }
    }
    Ok(value)
}

fn direct(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    guards: &[UnivariatePoly],
    lower: &BigInt,
    upper: &BigInt,
    meter: &mut impl BudgetMeter,
) -> Result<Option<RationalProductCertificate>> {
    reserve(meter, 4)?;
    let reversed = upper < lower;
    let (lo, hi) = if reversed {
        (upper + BigInt::one(), lower - BigInt::one())
    } else {
        (lower.clone(), upper.clone())
    };
    if hi < lo {
        return Ok(Some(RationalProductCertificate::Direct {
            value: BigRational::one(),
        }));
    }
    let count = &hi - &lo + BigInt::one();
    let Some(length) = count.to_u64().filter(|n| *n <= MAX_DIRECT_PRODUCT_TERMS) else {
        return Ok(None);
    };
    let mut point = BigRational::from_integer(lo);
    let mut value = BigRational::one();
    for _ in 0..length {
        for guard in guards {
            if evaluate(guard, &point, meter)?.is_zero() {
                return Err(RationalProductError::Undefined(
                    "original inverse guard vanishes in the range",
                ));
            }
        }
        let top = evaluate(numerator, &point, meter)?;
        let bottom = evaluate(denominator, &point, meter)?;
        let factor = div(&top, &bottom, meter)?;
        value = mul(&value, &factor, meter)?;
        // Do not return early on zero: later original factors may be undefined.
        point = add(&point, &BigRational::one(), meter)?;
    }
    if reversed {
        value = div(&BigRational::one(), &value, meter)?;
    }
    Ok(Some(RationalProductCertificate::Direct { value }))
}

fn factors(
    poly: &UnivariatePoly,
    meter: &mut Capped<'_, impl BudgetMeter>,
) -> Result<Option<Vec<Factor>>> {
    let result = metered_complete_factorization(poly, meter);
    let factorization = meter.polynomial(result)?;
    reserve(meter, factorization.factors.len())?;
    let mut result = Vec::with_capacity(factorization.factors.len());
    for factor in factorization.factors {
        if factor
            .poly
            .degree()
            .is_none_or(|degree| degree == 0 || degree > MAX_INPUT_DEGREE)
        {
            return Ok(None);
        }
        reserve(meter, factor.poly.coeffs.len())?;
        let leading = factor.poly.leading_coeff();
        let coefficients = factor
            .poly
            .coeffs
            .iter()
            .map(|coefficient| div(coefficient, leading, meter))
            .collect::<Result<Vec<_>>>()?;
        let multiplicity = u32::try_from(factor.multiplicity)
            .map_err(|_| RationalProductError::ResourceLimit("factor multiplicity"))?;
        result.push((
            UnivariatePoly::new(poly.gen_sym.clone(), coefficients),
            multiplicity,
        ));
    }
    Ok(Some(result))
}

fn monic(poly: &UnivariatePoly, meter: &mut impl BudgetMeter) -> Result<UnivariatePoly> {
    reserve(meter, poly.coeffs.len())?;
    let leading = poly.leading_coeff();
    let coefficients = poly
        .coeffs
        .iter()
        .map(|coefficient| div(coefficient, leading, meter))
        .collect::<Result<Vec<_>>>()?;
    meter.charge(Dimension::MemoryBytes, poly.gen_sym.name.len() as u64)?;
    Ok(UnivariatePoly::new(poly.gen_sym.clone(), coefficients))
}

/// A whole polynomial with no integer zeros already suffices for a domain
/// witness, including repeated or reducible factors. Avoid full QQ factor
/// search when this inexpensive, independently checkable proof is available.
fn domain_factors(
    poly: &UnivariatePoly,
    meter: &mut Capped<'_, impl BudgetMeter>,
) -> Result<Option<DomainFactorCandidates>> {
    if poly.degree().is_some_and(|degree| degree >= 3) {
        let normalized = monic(poly, meter)?;
        if modular::nonzero_factor(&normalized.coeffs, 1, meter)?.is_some() {
            reserve(meter, 1)?;
            return Ok(Some((vec![(normalized, 1)], true)));
        }
    }
    // Reaching this alternative means only that no modular witness was
    // found. An actual error from the attempt above propagates immediately.
    factors(poly, meter).map(|candidate| candidate.map(|factors| (factors, false)))
}

fn witnesses(
    factors: &[Factor],
    meter: &mut impl BudgetMeter,
) -> Result<Option<RationalProductFactors>> {
    reserve(meter, factors.len())?;
    let mut result = RationalProductFactors::default();
    for (poly, multiplicity) in factors {
        match poly.degree() {
            Some(1) => result.roots.push(RationalSumPole {
                root: -poly.coeffs[0].clone(),
                multiplicity: *multiplicity,
            }),
            Some(2) => result.quadratics.push(RationalSumQuadratic {
                constant: poly.coeffs[0].clone(),
                linear: poly.coeffs[1].clone(),
                multiplicity: *multiplicity,
            }),
            Some(degree) if degree >= 3 => {
                let Some(factor) = modular::nonzero_factor(&poly.coeffs, *multiplicity, meter)?
                else {
                    return Ok(None);
                };
                result.modular.push(factor);
            }
            _ => return Err(PolyError::General("invalid product factor degree".into()).into()),
        }
    }
    Ok(Some(result))
}

fn domain(
    denominator: &UnivariatePoly,
    guards: &[UnivariatePoly],
    meter: &mut Capped<'_, impl BudgetMeter>,
) -> Result<Option<DomainFactors>> {
    let Some((denominator_factors, whole_denominator)) = domain_factors(denominator, meter)? else {
        return Ok(None);
    };
    let Some(denominator_witness) = witnesses(&denominator_factors, meter)? else {
        return Ok(None);
    };
    reserve(meter, guards.len())?;
    let mut guard_witnesses = Vec::with_capacity(guards.len());
    for guard in guards {
        meter.charge(Dimension::ComputeSteps, guard.coeffs.len() as u64)?;
        // All these polynomials were admitted in the same exact typed ring.
        // Reuse only byte-exact coefficient matches; every original guard
        // still gets a separate full witness and independent source check.
        let witness = if guard.coeffs == denominator.coeffs {
            witnesses(&denominator_factors, meter)?
        } else {
            let Some((guard_factors, _)) = domain_factors(guard, meter)? else {
                return Ok(None);
            };
            witnesses(&guard_factors, meter)?
        };
        let Some(witness) = witness else {
            return Ok(None);
        };
        guard_witnesses.push(witness);
    }
    Ok(Some((
        RationalProductDomain {
            denominator: denominator_witness,
            guards: guard_witnesses,
        },
        denominator_factors,
        whole_denominator,
    )))
}

fn polynomial_product(
    a: &UnivariatePoly,
    b: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<UnivariatePoly> {
    let length = a.coeffs.len() + b.coeffs.len() - 1;
    if length > MAX_OUTPUT_DEGREE + 1 {
        return Err(RationalProductError::ResourceLimit(
            "telescoper polynomial degree",
        ));
    }
    reserve(meter, length)?;
    let mut result = vec![BigRational::zero(); length];
    for (i, left) in a.coeffs.iter().enumerate() {
        for (j, right) in b.coeffs.iter().enumerate() {
            let value = mul(left, right, meter)?;
            result[i + j] = add(&result[i + j], &value, meter)?;
        }
    }
    Ok(UnivariatePoly::new(a.gen_sym.clone(), result))
}

fn shifted(
    factor: &UnivariatePoly,
    shift: i64,
    meter: &mut impl BudgetMeter,
) -> Result<UnivariatePoly> {
    reserve(meter, factor.coeffs.len())?;
    let offset = BigRational::from_integer(shift.into());
    // Binomial expansion of each original monomial, separate from the
    // independent verifier's Horner composition by k+1. All exact scalar
    // operations remain on the original cumulative generator meter.
    let mut coefficients = vec![BigRational::zero(); factor.coeffs.len()];
    for (degree, coefficient) in factor.coeffs.iter().enumerate() {
        let mut term = coefficient.clone();
        for exponent in (0..=degree).rev() {
            coefficients[exponent] = add(&coefficients[exponent], &term, meter)?;
            if exponent != 0 {
                term = mul(&term, &offset, meter)?;
                let ratio = BigRational::new(
                    (exponent as u64).into(),
                    ((degree - exponent + 1) as u64).into(),
                );
                term = mul(&term, &ratio, meter)?;
            }
        }
    }
    Ok(UnivariatePoly::new(factor.gen_sym.clone(), coefficients))
}

/// Recognize N(k)/D(k) = c*D_monic(k+s)/D_monic(k) by exact
/// coefficient comparison, without asking a factorizer to rediscover a shift.
fn whole_shift(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<Option<(i64, UnivariatePoly)>> {
    if numerator.degree() != denominator.degree() {
        return Ok(None);
    }
    let degree = denominator.degree().expect("source denominator is nonzero");
    let base = monic(denominator, meter)?;
    if degree == 0 {
        return Ok(Some((0, base)));
    }
    let target = monic(numerator, meter)?;
    let difference = add(
        &target.coeffs[degree - 1],
        &(-base.coeffs[degree - 1].clone()),
        meter,
    )?;
    let offset = div(
        &difference,
        &BigRational::from_integer((degree as u64).into()),
        meter,
    )?;
    if !offset.is_integer() {
        return Ok(None);
    }
    let Some(offset) = offset.to_integer().to_i64() else {
        return Ok(None);
    };
    if offset.unsigned_abs() > MAX_SHIFT_SPAN as u64 || shifted(&base, offset, meter)? != target {
        return Ok(None);
    }
    Ok(Some((offset, base)))
}

fn merge_factor(
    factors: &mut Vec<Factor>,
    polynomial: UnivariatePoly,
    multiplicity: u32,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    for (existing, power) in factors.iter_mut() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, existing.coeffs.len() as u64)?;
        if existing.coeffs == polynomial.coeffs {
            *power = power
                .checked_add(multiplicity)
                .ok_or(RationalProductError::ResourceLimit(
                    "product factor multiplicity",
                ))?;
            return Ok(());
        }
    }
    reserve(meter, 1)?;
    factors.push((polynomial, multiplicity));
    Ok(())
}

fn whole_shift_certificate(
    base: &UnivariatePoly,
    offset: i64,
    factors: &[Factor],
    domain: RationalProductDomain,
    scale: BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<Option<RationalProductCertificate>> {
    let degree = base.degree().expect("monic polynomial is nonzero");
    if (degree as u64) * offset.unsigned_abs() > MAX_OUTPUT_DEGREE as u64 {
        return Err(RationalProductError::ResourceLimit(
            "telescoper polynomial degree",
        ));
    }
    let (first, end) = if offset >= 0 {
        (0, offset)
    } else {
        (offset, 0)
    };
    let mut polynomial = UnivariatePoly::one(base.gen_sym.clone());
    let mut output_factors = Vec::new();
    for shift in first..end {
        let factor = shifted(base, shift, meter)?;
        polynomial = polynomial_product(&polynomial, &factor, meter)?;
        for (factor, multiplicity) in factors {
            let factor = shifted(factor, shift, meter)?;
            merge_factor(&mut output_factors, factor, *multiplicity, meter)?;
        }
    }
    let Some(witness) = witnesses(&output_factors, meter)? else {
        return Ok(None);
    };
    let (numerator, denominator, numerator_factors, denominator_factors) = if offset >= 0 {
        (
            polynomial.coeffs,
            vec![BigRational::one()],
            witness,
            RationalProductFactors::default(),
        )
    } else {
        (
            vec![BigRational::one()],
            polynomial.coeffs,
            RationalProductFactors::default(),
            witness,
        )
    };
    reserve(meter, 1)?;
    Ok(Some(RationalProductCertificate::Telescoping {
        scale,
        numerator,
        denominator,
        domain: Box::new(domain),
        numerator_factors,
        denominator_factors,
    }))
}

struct Orbit {
    base: UnivariatePoly,
    exponents: BTreeMap<i64, i32>,
}

fn telescoper(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    guards: &[UnivariatePoly],
    lower: &BigInt,
    upper: &BigInt,
    meter: &mut Capped<'_, impl BudgetMeter>,
) -> Result<Option<RationalProductCertificate>> {
    let Some((domain, denominator_factors, whole_denominator)) =
        domain(denominator, guards, meter)?
    else {
        return Ok(None);
    };
    let (lo, hi) = if upper < lower {
        (upper + BigInt::one(), lower - BigInt::one())
    } else {
        (lower.clone(), upper.clone())
    };
    if numerator.is_zero() {
        return Ok(Some(RationalProductCertificate::Zero { at: lo, domain }));
    }
    if let Some((offset, base)) = whole_shift(numerator, denominator, meter)? {
        // A whole shift can move an original linear root into the numerator
        // range. Keep the zero witness before constructing a nonzero quotient.
        for (factor, _) in &denominator_factors {
            if factor.degree() == Some(1) {
                let translated = shifted(factor, offset, meter)?;
                let root = -translated.coeffs[0].clone();
                if root.is_integer() && root.numer() >= &lo && root.numer() <= &hi {
                    return Ok(Some(RationalProductCertificate::Zero {
                        at: root.to_integer(),
                        domain,
                    }));
                }
            }
        }
        let scale = div(
            numerator.leading_coeff(),
            denominator.leading_coeff(),
            meter,
        )?;
        return whole_shift_certificate(&base, offset, &denominator_factors, domain, scale, meter);
    }
    let Some(numerator_factors) = factors(numerator, meter)? else {
        return Ok(None);
    };
    // Retain a zero-factor witness before attempting the nonzero quotient
    // identity, whose endpoint expression cannot represent a zero crossing.
    for (factor, _) in &numerator_factors {
        if factor.degree() == Some(1) {
            let root = -factor.coeffs[0].clone();
            if root.is_integer() && root.numer() >= &lo && root.numer() <= &hi {
                return Ok(Some(RationalProductCertificate::Zero {
                    at: root.to_integer(),
                    domain,
                }));
            }
        }
    }
    if numerator.degree() != denominator.degree() {
        return Ok(None);
    }
    // A coarse whole-polynomial domain witness need not be a single shift
    // orbit. Refine it for the general orbit search when the exact whole-shift
    // identity did not match; otherwise mixed existing families would be lost.
    let denominator_factors = if whole_denominator {
        let Some(factors) = factors(denominator, meter)? else {
            return Ok(None);
        };
        factors
    } else {
        denominator_factors
    };
    let mut orbits: Vec<Orbit> = Vec::new();
    for (factor, multiplicity, sign) in numerator_factors
        .into_iter()
        .map(|(factor, power)| (factor, power, 1i32))
        .chain(
            denominator_factors
                .into_iter()
                .map(|(factor, power)| (factor, power, -1i32)),
        )
    {
        let degree = factor.degree().expect("factor degree was admitted");
        let mut existing = None;
        for (index, orbit) in orbits.iter().enumerate() {
            if orbit.base.degree() != Some(degree) {
                continue;
            }
            let delta = add(
                &factor.coeffs[degree - 1],
                &(-orbit.base.coeffs[degree - 1].clone()),
                meter,
            )?;
            let offset = div(
                &delta,
                &BigRational::from_integer((degree as u64).into()),
                meter,
            )?;
            if !offset.is_integer() {
                continue;
            }
            let Some(offset) = offset.to_integer().to_i64() else {
                continue;
            };
            if offset.unsigned_abs() > MAX_SHIFT_SPAN as u64 {
                continue;
            }
            if shifted(&orbit.base, offset, meter)? == factor {
                existing = Some((index, offset));
                break;
            }
        }
        let signed = sign * multiplicity as i32;
        reserve(meter, 1)?;
        if let Some((index, offset)) = existing {
            *orbits[index].exponents.entry(offset).or_default() += signed;
        } else {
            orbits.push(Orbit {
                base: factor,
                exponents: BTreeMap::from([(0, signed)]),
            });
        }
    }
    let mut top = UnivariatePoly::one(numerator.gen_sym.clone());
    let mut bottom = UnivariatePoly::one(numerator.gen_sym.clone());
    let mut top_factors = Vec::new();
    let mut bottom_factors = Vec::new();
    for orbit in orbits {
        if orbit.exponents.values().copied().sum::<i32>() != 0 {
            return Ok(None);
        }
        let minimum = *orbit.exponents.keys().next().expect("orbit is nonempty");
        let maximum = *orbit
            .exponents
            .keys()
            .next_back()
            .expect("orbit is nonempty");
        if maximum - minimum > MAX_SHIFT_SPAN {
            return Ok(None);
        }
        let mut exponent = 0i32;
        for offset in minimum..maximum {
            exponent -= orbit.exponents.get(&offset).copied().unwrap_or(0);
            if exponent == 0 {
                continue;
            }
            let factor = shifted(&orbit.base, offset, meter)?;
            let multiplicity = exponent.unsigned_abs();
            let (polynomial, factors) = if exponent > 0 {
                (&mut top, &mut top_factors)
            } else {
                (&mut bottom, &mut bottom_factors)
            };
            for _ in 0..multiplicity {
                *polynomial = polynomial_product(polynomial, &factor, meter)?;
            }
            reserve(meter, factor.coeffs.len())?;
            factors.push((factor, multiplicity));
        }
    }
    let scale = div(
        numerator.leading_coeff(),
        denominator.leading_coeff(),
        meter,
    )?;
    let Some(numerator_factors) = witnesses(&top_factors, meter)? else {
        return Ok(None);
    };
    let Some(denominator_factors) = witnesses(&bottom_factors, meter)? else {
        return Ok(None);
    };
    reserve(meter, 1)?;
    Ok(Some(RationalProductCertificate::Telescoping {
        scale,
        numerator: top.coeffs,
        denominator: bottom.coeffs,
        domain: Box::new(domain),
        numerator_factors,
        denominator_factors,
    }))
}

/// Construct and independently check a finite rational product. `None` means
/// unsupported source grammar, a non-telescoping long product, or factor/shift
/// structure outside this exact family. Operational and domain errors remain
/// errors, with no retry or accounting reset. Bounds may exceed machine words.
pub fn metered_rational_product_source(
    source: &Expr,
    variable: &Symbol,
    lower: &BigInt,
    upper: &BigInt,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRationalProduct>> {
    let mut generator = Capped::new(generator_meter);
    let mut verifier = Capped::new(verifier_meter);
    generator.checkpoint()?;
    verifier.checkpoint()?;
    if lower.bits() > MAX_INPUT_BITS || upper.bits() > MAX_INPUT_BITS {
        return Err(RationalProductError::ResourceLimit(
            "integer endpoint height",
        ));
    }
    if !is_rational_source(source, variable, &mut generator)? {
        return Ok(None);
    }
    let source_result = metered_rational_source(
        source,
        std::slice::from_ref(variable),
        &mut generator,
        &mut verifier,
    );
    if let Some(error) = verifier.stopped {
        return Err(error.into());
    }
    let source = generator.polynomial(source_result)?;
    let numerator = dense(&source.fraction().numerator, variable, &mut generator)?;
    let denominator = dense(&source.fraction().denominator, variable, &mut generator)?;
    reserve(&mut generator, source.guards().len())?;
    let guards = source
        .guards()
        .iter()
        .map(|guard| dense(guard, variable, &mut generator))
        .collect::<Result<Vec<_>>>()?;
    let candidate = direct(
        &numerator,
        &denominator,
        &guards,
        lower,
        upper,
        &mut generator,
    )?;
    let candidate = match candidate {
        Some(candidate) => Some(candidate),
        None => telescoper(
            &numerator,
            &denominator,
            &guards,
            lower,
            upper,
            &mut generator,
        )?,
    };
    let Some(certificate) = candidate else {
        return Ok(None);
    };
    reserve(
        &mut verifier,
        guards.iter().map(|guard| guard.coeffs.len()).sum(),
    )?;
    let guard_coefficients: Vec<_> = guards.iter().map(|guard| guard.coeffs.clone()).collect();
    let value = verify_rational_product(
        &numerator.coeffs,
        &denominator.coeffs,
        &guard_coefficients,
        lower,
        upper,
        &certificate,
        &mut verifier,
    )?;
    reserve(&mut verifier, 3)?;
    generator.checkpoint()?;
    verifier.checkpoint()?;
    Ok(Some(VerifiedRationalProduct {
        source,
        lower: lower.clone(),
        upper: upper.clone(),
        value,
        certificate,
    }))
}

pub fn rational_product_source(
    source: &Expr,
    variable: &Symbol,
    lower: &BigInt,
    upper: &BigInt,
) -> Result<Option<VerifiedRationalProduct>> {
    metered_rational_product_source(
        source,
        variable,
        lower,
        upper,
        &mut Unbounded,
        &mut Unbounded,
    )
}
