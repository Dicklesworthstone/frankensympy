//! Checked source composition for the public modular factorization convention.
//!
//! The shell proposes an exact rational exterior coefficient and integer
//! polynomial components, as its ordinary numerator/denominator conventions
//! require. Before reducing anything modulo p, the checker reconstructs every
//! rational coefficient of the original held source. Each component's modular
//! image is checked separately by byte-wise Horner reduction, independently of
//! generation's governed integer division. Accepted complete field
//! factorizations are then composed without dropping or rebinding factors.
//!
//! This convention preserves exact exterior coefficients (including rational
//! denominators) outside the modular polynomial calculation. It does not assert
//! an embedding of QQ into a positive-characteristic field. All original nodes
//! are admitted before zero shortcuts, and both phases keep cumulative caps.
//! The exterior denominator must be the original canonical least common
//! denominator, and every nonzero exterior numerator must be coprime to p.
//! Thus normalization cannot manufacture a characteristic denominator or move
//! a characteristic factor out of an integer component. Pinned shell forms
//! retaining a p-divisible exterior coefficient, including nonzero constants,
//! explicitly refuse in this bounded lane; ordinary component reduction to
//! the zero polynomial remains valid.

use std::collections::BTreeMap;
use std::sync::Arc;

use fsym_bigint::{
    NonZeroBigInt, metered_add, metered_div_rem_nonzero, metered_gcd, metered_multiply,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use num_traits::{One, Zero};

use crate::division::{PolynomialWork, refused};
use crate::{
    MAX_PRIME_FIELD_DEGREE, PolyError, PrimeFieldFactor, PrimeFieldPolynomial,
    VerifiedPrimeFieldFactorization, bounded_polynomial_from_expr,
    checked_prime_field_factorization,
};

type RationalCoefficients = Vec<BigRational>;

/// Exact source and an untrusted proposed exterior-content decomposition.
pub struct PrimeFieldSourceRequest<'a> {
    pub source: &'a Expr,
    pub variable: &'a Symbol,
    pub prime: u64,
    pub content: &'a BigRational,
    pub components: &'a [(&'a Expr, u32)],
}

#[derive(Debug, Clone)]
pub struct PrimeFieldSourceFactorizationCertificate {
    pub unit: BigRational,
    pub factors: Vec<PrimeFieldFactor>,
    pub components: Vec<Arc<VerifiedPrimeFieldFactorization>>,
}

#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldSourceFactorization {
    variable: Symbol,
    prime: u64,
    source: RationalCoefficients,
    certificate: PrimeFieldSourceFactorizationCertificate,
}

impl VerifiedPrimeFieldSourceFactorization {
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn prime(&self) -> u64 {
        self.prime
    }
    pub fn source(&self) -> &[BigRational] {
        &self.source
    }
    pub fn unit(&self) -> &BigRational {
        &self.certificate.unit
    }
    pub fn factors(&self) -> &[PrimeFieldFactor] {
        &self.certificate.factors
    }
    pub fn certificate(&self) -> &PrimeFieldSourceFactorizationCertificate {
        &self.certificate
    }
}

fn lower(
    source: &Expr,
    variable: &Symbol,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RationalCoefficients, PolyError> {
    let polynomial = bounded_polynomial_from_expr(source, std::slice::from_ref(variable), work)?;
    let degree = polynomial.degree_in(0) as usize;
    if degree > MAX_PRIME_FIELD_DEGREE {
        return Err(refused("modular source exceeds degree 64"));
    }
    work.allocate((degree + 1) * std::mem::size_of::<BigRational>())?;
    let mut coefficients = vec![BigRational::zero(); degree + 1];
    for (powers, value) in polynomial.terms() {
        coefficients[powers[0] as usize] = work.clone_value(value)?;
    }
    trim(&mut coefficients);
    Ok(coefficients)
}

fn trim(coefficients: &mut RationalCoefficients) {
    while coefficients.last().is_some_and(Zero::is_zero) {
        coefficients.pop();
    }
}

/// Native admission before the shell's numerator/denominator normalization.
pub fn admit_prime_field_source(
    source: &Expr,
    variable: &Symbol,
    prime: u64,
    meter: &mut impl BudgetMeter,
) -> Result<(), PolyError> {
    let mut work = PolynomialWork::new(meter);
    PrimeFieldPolynomial::metered_new(variable, prime, &[], &mut work)?;
    lower(source, variable, &mut work)?;
    work.tick(1)
}

struct LoweredRequest {
    source: RationalCoefficients,
    components: Vec<RationalCoefficients>,
}

/// Content cannot be chosen freely in characteristic p, even when the QQ
/// reconstruction is exact: x=(1/p)*(p*x) would otherwise acquire a zero
/// modular image. Check denominator coverage without constructing a possibly
/// enormous LCM: lcm(gcd(target,d_i)) always divides the bounded target.
fn admit_content(
    source: &[BigRational],
    content: &BigRational,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if !content.is_zero() && reference_residue(content.numer(), prime, work)? == 0 {
        return Err(refused(
            "nonzero exterior content divisible by the characteristic is outside the bounded normalization lane",
        ));
    }
    work.allocate(16)?;
    let mut covered = BigInt::one();
    let target = content.denom();
    for coefficient in source {
        let overlap = metered_gcd(target, coefficient.denom(), work).map_err(refused)?;
        let shared = metered_gcd(&covered, &overlap, work).map_err(refused)?;
        let (extra, remainder) =
            metered_div_rem_nonzero(&overlap, NonZeroBigInt::new(&shared).unwrap(), work)
                .map_err(refused)?;
        if !remainder.is_zero() {
            return Err(refused("source denominator coverage was not exact"));
        }
        covered = metered_multiply(&covered, &extra, work).map_err(refused)?;
    }
    if &covered != target {
        return Err(refused(
            "exterior denominator is not covered by the original canonical source denominators",
        ));
    }
    // Exact reconstruction with integer components establishes the converse
    // divisibility, so together these checks force the denominator to be the LCD.
    Ok(())
}

fn lower_request(
    request: &PrimeFieldSourceRequest<'_>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<LoweredRequest, PolyError> {
    work.tick(1)?;
    if request.components.len() > MAX_PRIME_FIELD_DEGREE {
        return Err(refused("modular source has more than 64 components"));
    }
    PrimeFieldPolynomial::metered_new(request.variable, request.prime, &[], work)?;
    work.clone_value(request.content)?;
    let source = lower(request.source, request.variable, work)?;
    admit_content(&source, request.content, request.prime, work)?;
    work.allocate(request.components.len() * std::mem::size_of::<RationalCoefficients>())?;
    let mut components = Vec::with_capacity(request.components.len());
    let mut total_degree = 0u64;
    for &(expression, multiplicity) in request.components {
        if multiplicity == 0 || multiplicity > MAX_PRIME_FIELD_DEGREE as u32 {
            return Err(refused("modular source component power is outside 1..64"));
        }
        let polynomial = lower(expression, request.variable, work)?;
        if polynomial
            .iter()
            .any(|coefficient| !coefficient.is_integer())
        {
            return Err(refused(
                "modular source components must have integer coefficients",
            ));
        }
        total_degree += polynomial.len().saturating_sub(1) as u64 * u64::from(multiplicity);
        if total_degree > MAX_PRIME_FIELD_DEGREE as u64 {
            return Err(refused("modular source component degrees exceed 64"));
        }
        components.push(polynomial);
    }
    Ok(LoweredRequest { source, components })
}

fn exact_product(
    left: &[BigRational],
    right: &[BigRational],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RationalCoefficients, PolyError> {
    if left.is_empty() || right.is_empty() {
        work.tick(1)?;
        return Ok(Vec::new());
    }
    let length = left.len() + right.len() - 1;
    if length > MAX_PRIME_FIELD_DEGREE + 1 {
        return Err(refused("source reconstruction exceeds degree 64"));
    }
    work.allocate(length * std::mem::size_of::<BigRational>())?;
    let mut result = vec![BigRational::zero(); length];
    for (power, coefficient) in result.iter_mut().enumerate() {
        work.tick(left.len())?;
        for (i, a) in left.iter().enumerate() {
            if let Some(j) = power.checked_sub(i).filter(|&j| j < right.len()) {
                let product = work.multiply(a, &right[j])?;
                *coefficient = work.add(coefficient, &product)?;
            }
        }
    }
    trim(&mut result);
    Ok(result)
}

fn generation_residue(
    value: &BigInt,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    work.allocate(16)?;
    let modulus = BigInt::from(prime);
    let (_, mut remainder) =
        metered_div_rem_nonzero(value, NonZeroBigInt::new(&modulus).unwrap(), work)
            .map_err(refused)?;
    if remainder.is_negative() {
        remainder = metered_add(&remainder, &modulus, work).map_err(refused)?;
    }
    remainder
        .to_u64()
        .ok_or_else(|| refused("integer reduction did not produce a canonical residue"))
}

fn reference_residue(
    value: &BigInt,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    let bytes = value.bits().div_ceil(8) as usize;
    work.allocate(bytes.max(1))?;
    work.tick(bytes)?;
    let magnitude = value.to_bytes_le();
    let mut result = 0u64;
    for &byte in magnitude.iter().rev() {
        result = (result * 256 + u64::from(byte)) % prime;
    }
    if value.is_negative() && result != 0 {
        result = prime - result;
    }
    Ok(result)
}

fn signed_unit(value: u64, prime: u64) -> BigRational {
    BigRational::from(if value > prime / 2 {
        value as i64 - prime as i64
    } else {
        value as i64
    })
}

fn unit_power(
    value: &BigRational,
    mut exponent: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<BigRational, PolyError> {
    let mut result = work.clone_value(&BigRational::one())?;
    let mut base = work.clone_value(value)?;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = work.multiply(&result, &base)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = work.multiply(&base, &base)?;
        }
    }
    Ok(result)
}

fn order(left: &PrimeFieldFactor, right: &PrimeFieldFactor) -> std::cmp::Ordering {
    left.coefficients
        .len()
        .cmp(&right.coefficients.len())
        .then(left.multiplicity.cmp(&right.multiplicity))
        .then_with(|| {
            left.coefficients
                .iter()
                .rev()
                .cmp(right.coefficients.iter().rev())
        })
}

/// Check original-source reconstruction, modular images and complete composition.
pub fn verify_prime_field_source_factorization(
    request: &PrimeFieldSourceRequest<'_>,
    certificate: &PrimeFieldSourceFactorizationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldSourceFactorization, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    let lowered = lower_request(request, &mut work)?;
    if certificate.components.len() != request.components.len()
        || certificate.factors.len() > MAX_PRIME_FIELD_DEGREE
    {
        return Err(refused(
            "modular source certificate component count changed",
        ));
    }
    work.allocate(std::mem::size_of::<BigRational>())?;
    let mut reconstructed = vec![work.clone_value(request.content)?];
    trim(&mut reconstructed);
    let mut unit = work.clone_value(request.content)?;
    for ((polynomial, &(_, multiplicity)), accepted) in lowered
        .components
        .iter()
        .zip(request.components)
        .zip(&certificate.components)
    {
        if accepted.variable() != request.variable || accepted.prime() != request.prime {
            return Err(refused("modular source component changed its typed field"));
        }
        work.allocate(polynomial.len() * std::mem::size_of::<u64>())?;
        let mut residues = polynomial
            .iter()
            .map(|coefficient| reference_residue(coefficient.numer(), request.prime, &mut work))
            .collect::<Result<Vec<_>, _>>()?;
        while residues.last() == Some(&0) {
            residues.pop();
        }
        if residues != accepted.source() {
            return Err(refused(
                "modular image differs from its exact integer source",
            ));
        }
        work.allocate(16)?;
        let component_unit = signed_unit(accepted.unit(), request.prime);
        for _ in 0..multiplicity {
            reconstructed = exact_product(&reconstructed, polynomial, &mut work)?;
            unit = work.multiply(&unit, &component_unit)?;
        }
    }
    if reconstructed != lowered.source {
        return Err(PolyError::IdentityCheckFailed("exterior content and integer components do not reconstruct the original rational source".into()));
    }
    work.clone_value(&certificate.unit)?;
    if unit != certificate.unit {
        return Err(refused("modular source output unit changed"));
    }
    if unit.is_zero() && !certificate.factors.is_empty() {
        return Err(refused(
            "zero modular source requires the canonical empty factor list",
        ));
    }
    for (index, factor) in certificate.factors.iter().enumerate() {
        work.tick(MAX_PRIME_FIELD_DEGREE * (MAX_PRIME_FIELD_DEGREE + 1))?;
        if factor.coefficients.len() > MAX_PRIME_FIELD_DEGREE + 1
            || factor.coefficients.len() < 2
            || factor.coefficients.last() != Some(&1)
            || factor
                .coefficients
                .iter()
                .any(|&coefficient| coefficient >= request.prime)
            || factor.multiplicity == 0
            || factor.multiplicity > MAX_PRIME_FIELD_DEGREE as u32
            || certificate.factors[..index]
                .iter()
                .any(|other| other.coefficients == factor.coefficients)
            || (index > 0 && !order(&certificate.factors[index - 1], factor).is_lt())
        {
            return Err(refused(
                "modular source factors are not canonical distinct sorted factors",
            ));
        }
        let mut expected = 0u64;
        for (&(_, exponent), component) in request.components.iter().zip(&certificate.components) {
            for term in component.factors() {
                if term.coefficients == factor.coefficients {
                    expected += u64::from(exponent) * u64::from(term.multiplicity);
                }
            }
        }
        if expected != u64::from(factor.multiplicity) {
            return Err(refused("modular source factor multiplicity changed"));
        }
    }
    for component in &certificate.components {
        for term in component.factors() {
            work.tick(certificate.factors.len() * (MAX_PRIME_FIELD_DEGREE + 1))?;
            if !unit.is_zero()
                && !certificate
                    .factors
                    .iter()
                    .any(|factor| factor.coefficients == term.coefficients)
            {
                return Err(refused(
                    "modular source composition omitted an irreducible factor",
                ));
            }
        }
    }
    work.allocate(
        request.variable.name.len()
            + std::mem::size_of::<Symbol>()
            + certificate.factors.len() * std::mem::size_of::<PrimeFieldFactor>()
            + certificate.components.len()
                * std::mem::size_of::<Arc<VerifiedPrimeFieldFactorization>>(),
    )?;
    let factors = certificate
        .factors
        .iter()
        .map(|factor| {
            work.allocate(factor.coefficients.len() * std::mem::size_of::<u64>())?;
            Ok(PrimeFieldFactor {
                coefficients: factor.coefficients.clone(),
                multiplicity: factor.multiplicity,
            })
        })
        .collect::<Result<Vec<_>, PolyError>>()?;
    let result = VerifiedPrimeFieldSourceFactorization {
        variable: request.variable.clone(),
        prime: request.prime,
        source: lowered.source,
        certificate: PrimeFieldSourceFactorizationCertificate {
            unit,
            factors,
            components: certificate.components.clone(),
        },
    };
    work.tick(1)?;
    Ok(result)
}

pub fn checked_prime_field_source_factorization(
    request: &PrimeFieldSourceRequest<'_>,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldSourceFactorization, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let lowered = lower_request(request, &mut generation)?;
    generation.allocate(
        request.components.len() * std::mem::size_of::<Arc<VerifiedPrimeFieldFactorization>>(),
    )?;
    let mut components = Vec::with_capacity(request.components.len());
    let mut unit = generation.clone_value(request.content)?;
    let mut grouped = BTreeMap::<Vec<u64>, u32>::new();
    for (polynomial, &(_, exponent)) in lowered.components.iter().zip(request.components) {
        generation.allocate(polynomial.len() * std::mem::size_of::<u64>())?;
        let mut residues = polynomial
            .iter()
            .map(|coefficient| {
                generation_residue(coefficient.numer(), request.prime, &mut generation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        while residues.last() == Some(&0) {
            residues.pop();
        }
        let input = PrimeFieldPolynomial::metered_new(
            request.variable,
            request.prime,
            &residues,
            &mut generation,
        )?;
        let accepted =
            checked_prime_field_factorization(&input, &mut generation, &mut verification)?;
        generation.allocate(16)?;
        let component_unit = signed_unit(accepted.unit(), request.prime);
        let power = unit_power(&component_unit, exponent, &mut generation)?;
        unit = generation.multiply(&unit, &power)?;
        for factor in accepted.factors() {
            generation.tick(grouped.len() * (MAX_PRIME_FIELD_DEGREE + 1))?;
            generation.allocate(factor.coefficients.len() * std::mem::size_of::<u64>() + 256)?;
            *grouped.entry(factor.coefficients.clone()).or_default() +=
                factor.multiplicity * exponent;
        }
        generation.allocate(std::mem::size_of::<VerifiedPrimeFieldFactorization>() + 64)?;
        components.push(Arc::new(accepted));
    }
    // Keep every checked component receipt even when one modular image is zero,
    // but publish the canonical zero factorization only after all source nodes
    // and component factorizations have been admitted.
    if unit.is_zero() {
        grouped.clear();
    }
    generation.allocate(grouped.len() * std::mem::size_of::<PrimeFieldFactor>())?;
    let mut factors: Vec<_> = grouped
        .into_iter()
        .map(|(coefficients, multiplicity)| PrimeFieldFactor {
            coefficients,
            multiplicity,
        })
        .collect();
    generation.tick(factors.len() * factors.len() * (MAX_PRIME_FIELD_DEGREE + 1))?;
    factors.sort_by(order);
    let certificate = PrimeFieldSourceFactorizationCertificate {
        unit,
        factors,
        components,
    };
    let accepted =
        verify_prime_field_source_factorization(request, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
