//! Checked recursive pseudo-division in ZZ[x_2, ..., x_n][x_1] and QQ.
//!
//! For nonzero g, b = LC_x1(g) and e = max(0, deg_x1(f)-deg_x1(g)+1),
//! the accepted pair is uniquely characterized by b^e*f = q*g+r and
//! deg_x1(r) < deg_x1(g). A zero dividend has e=0. No coefficient-ring
//! division or promotion from ZZ to QQ is used by generation.
//!
//! The separate reference lane recomputes the original leading coefficient
//! and full scaling exponent, then checks the identity by direct signed
//! coefficient convolution. It never invokes pseudo-division, ordinary
//! division, the generator's multiplication, or symbolic simplification.
//! These are bounded in-process cross-check receipts, not portable kernel proofs.
//!
//! Both complete phases have separate cumulative 20M-compute/128MiB
//! logical-allocation caps and caller cancellation ledgers. Inputs, output,
//! and intermediate polynomials have one to eight ordered typed variables,
//! total degree at most 256, at most 4096 terms and 4096-bit coefficients.
//! Public expression callers must admit the entire original held source
//! before entering this polynomial API. Failure publishes no partial receipt.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_DEGREE, MAX_TERMS, MAX_VARIABLES, PolynomialWork, Terms, accumulate, copy_poly,
    multiply_polynomials, refused, ring_storage, same_ring, validate_poly,
};
use crate::{MultivariatePoly as Polynomial, PolyError, RecursiveDivisionDomain};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::Zero;
use std::mem::size_of;

/// Untrusted pseudo-quotient, pseudo-remainder and required scaling exponent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PseudoDivisionCertificate {
    pub quotient: Polynomial,
    pub remainder: Polynomial,
    pub scale_exponent: u32,
}

/// A uniquely determined pseudo-division of the exact original operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPseudoDivision {
    domain: RecursiveDivisionDomain,
    dividend: Polynomial,
    divisor: Polynomial,
    quotient: Polynomial,
    remainder: Polynomial,
    scale_exponent: u32,
}

impl VerifiedPseudoDivision {
    pub fn domain(&self) -> RecursiveDivisionDomain {
        self.domain
    }

    pub fn dividend(&self) -> &Polynomial {
        &self.dividend
    }

    pub fn divisor(&self) -> &Polynomial {
        &self.divisor
    }

    pub fn quotient(&self) -> &Polynomial {
        &self.quotient
    }

    pub fn remainder(&self) -> &Polynomial {
        &self.remainder
    }

    pub fn scale_exponent(&self) -> u32 {
        self.scale_exponent
    }

    pub fn into_parts(self) -> (Polynomial, Polynomial) {
        (self.quotient, self.remainder)
    }
}

fn mismatch(message: &'static str) -> PolyError {
    PolyError::IdentityCheckFailed(message.into())
}

fn admit_ring(
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if variables.is_empty() || variables.len() > MAX_VARIABLES {
        return Err(refused("pseudo-division requires one to eight variables"));
    }
    let mut bytes = 0usize;
    for (index, variable) in variables.iter().enumerate() {
        bytes = bytes
            .checked_add(variable.name.len())
            .filter(|&value| value <= 65_536)
            .ok_or_else(|| refused("pseudo-division generator names exceed 65536 bytes"))?;
        work.tick((index + 1) * (variable.name.len() + 1))?;
        if variables[..index].contains(variable) {
            return Err(refused("duplicate typed pseudo-division variable"));
        }
    }
    Ok(())
}

fn admit_polynomial(
    polynomial: &Polynomial,
    variables: &[Symbol],
    domain: RecursiveDivisionDomain,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    // Bound untrusted ring payloads before equality and diagnostic formatting.
    admit_ring(polynomial.generators(), work)?;
    same_ring(polynomial.generators(), variables)?;
    validate_poly(polynomial, work)?;
    work.tick(polynomial.terms().len() + 1)?;
    if domain == RecursiveDivisionDomain::ZZ
        && polynomial.terms().values().any(|value| !value.is_integer())
    {
        return Err(refused("ZZ pseudo-division requires integer coefficients"));
    }
    Ok(())
}

fn admit_request(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_polynomial(dividend, dividend.generators(), domain, work)?;
    admit_polynomial(divisor, dividend.generators(), domain, work)?;
    if divisor.is_zero() {
        return Err(PolyError::DivisionByZero);
    }
    Ok(())
}

fn zero(
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    work.allocate(size_of::<Polynomial>() + ring_storage(variables))?;
    Ok(Polynomial::zero(variables.to_vec()))
}

fn generated_degree(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<u32>, PolyError> {
    work.tick(1)?;
    Ok(polynomial
        .terms()
        .last_key_value()
        .map(|(powers, _)| powers[0]))
}

fn generated_coefficient(
    polynomial: &Polynomial,
    degree: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut result = zero(polynomial.generators(), work)?;
    for (powers, coefficient) in polynomial.terms() {
        work.tick(powers.len() + 1)?;
        if powers[0] == degree {
            work.allocate(powers.len() * 4)?;
            let mut projected = powers.clone();
            projected[0] = 0;
            let coefficient = work.clone_value(coefficient)?;
            accumulate(&mut result.terms, projected, coefficient, work)?;
        }
    }
    Ok(result)
}

fn generated_shift(
    coefficient: &Polynomial,
    shift: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut result = zero(coefficient.generators(), work)?;
    for (powers, value) in coefficient.terms() {
        work.tick(powers.len() + 1)?;
        if powers[0] != 0 {
            return Err(refused(
                "pseudo-division coefficient contains its main variable",
            ));
        }
        work.allocate(powers.len() * 4)?;
        let mut moved = powers.clone();
        moved[0] = shift;
        let value = work.clone_value(value)?;
        accumulate(&mut result.terms, moved, value, work)?;
    }
    Ok(result)
}

fn generated_sum(
    left: &Polynomial,
    right: &Polynomial,
    subtract: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut result = copy_poly(left, work)?;
    for (powers, value) in right.terms() {
        work.allocate(powers.len() * 4)?;
        let coefficient = if subtract {
            work.subtract(&BigRational::zero(), value)?
        } else {
            work.clone_value(value)?
        };
        accumulate(&mut result.terms, powers.clone(), coefficient, work)?;
    }
    Ok(result)
}

/// Generate an untrusted recursive pseudo-division without coefficient division.
pub fn bounded_pseudo_division_certificate(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    generator: &mut impl BudgetMeter,
) -> Result<PseudoDivisionCertificate, PolyError> {
    let mut work = PolynomialWork::new(generator);
    admit_request(dividend, divisor, domain, &mut work)?;
    let divisor_degree = generated_degree(divisor, &mut work)?.ok_or(PolyError::DivisionByZero)?;
    let leading = generated_coefficient(divisor, divisor_degree, &mut work)?;
    let scale_exponent = generated_degree(dividend, &mut work)?
        .and_then(|degree| degree.checked_sub(divisor_degree))
        .map_or(0, |difference| difference + 1);
    let mut remaining = scale_exponent;
    let mut quotient = zero(dividend.generators(), &mut work)?;
    let mut remainder = copy_poly(dividend, &mut work)?;
    while let Some(degree) = generated_degree(&remainder, &mut work)? {
        if degree < divisor_degree {
            break;
        }
        remaining = remaining
            .checked_sub(1)
            .ok_or_else(|| refused("pseudo-division exceeded its degree descent bound"))?;
        let coefficient = generated_coefficient(&remainder, degree, &mut work)?;
        let step = generated_shift(&coefficient, degree - divisor_degree, &mut work)?;
        let scaled_quotient = multiply_polynomials(&leading, &quotient, &mut work)?;
        quotient = generated_sum(&scaled_quotient, &step, false, &mut work)?;
        let scaled_remainder = multiply_polynomials(&leading, &remainder, &mut work)?;
        let removed = multiply_polynomials(&step, divisor, &mut work)?;
        remainder = generated_sum(&scaled_remainder, &removed, true, &mut work)?;
        if generated_degree(&remainder, &mut work)?.is_some_and(|next| next >= degree) {
            return Err(refused(
                "pseudo-division did not decrease its primary degree",
            ));
        }
    }
    // Degree drops can skip cancellations, including exact division after one
    // step. The profile still requires the full original degree-difference power.
    for _ in 0..remaining {
        quotient = multiply_polynomials(&leading, &quotient, &mut work)?;
        remainder = multiply_polynomials(&leading, &remainder, &mut work)?;
    }
    work.tick(1)?;
    Ok(PseudoDivisionCertificate {
        quotient,
        remainder,
        scale_exponent,
    })
}

fn reference_degree(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<u32>, PolyError> {
    let mut degree: Option<u32> = None;
    for powers in polynomial.terms().keys() {
        work.tick(powers.len() + 1)?;
        degree = Some(degree.map_or(powers[0], |previous| previous.max(powers[0])));
    }
    Ok(degree)
}

fn reference_accumulate(
    terms: &mut Terms,
    powers: Vec<u32>,
    coefficient: BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(powers.len() + 1)?;
    if powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE {
        return Err(refused("pseudo-division reference term exceeds degree 256"));
    }
    let value = if let Some(previous) = terms.remove(&powers) {
        work.add(&previous, &coefficient)?
    } else {
        coefficient
    };
    if !value.is_zero() {
        if terms.len() >= MAX_TERMS {
            return Err(refused("pseudo-division reference term count exceeds 4096"));
        }
        work.allocate(256)?;
        terms.insert(powers, value);
    }
    Ok(())
}

/// Direct coefficient convolution; independent of the generator's multiplier.
fn reference_product(
    left: &Polynomial,
    right: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut result = zero(left.generators(), work)?;
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            work.tick(a.len() + 1)?;
            work.allocate(a.len() * 4)?;
            let mut powers = Vec::with_capacity(a.len());
            for (&a, &b) in a.iter().zip(b) {
                powers.push(
                    a.checked_add(b)
                        .ok_or_else(|| refused("pseudo-division reference exponent overflow"))?,
                );
            }
            let coefficient = work.multiply(ca, cb)?;
            reference_accumulate(&mut result.terms, powers, coefficient, work)?;
        }
    }
    Ok(result)
}

/// Independently accept the unique scaled division in the original typed ring.
/// The scale and strict primary-degree remainder condition are both mandatory.
pub fn verify_pseudo_division(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    certificate: &PseudoDivisionCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPseudoDivision, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit_request(dividend, divisor, domain, &mut work)?;
    for polynomial in [&certificate.quotient, &certificate.remainder] {
        admit_polynomial(polynomial, dividend.generators(), domain, &mut work)?;
    }
    let divisor_degree = reference_degree(divisor, &mut work)?.ok_or(PolyError::DivisionByZero)?;
    let scale_exponent = reference_degree(dividend, &mut work)?
        .and_then(|degree| degree.checked_sub(divisor_degree))
        .map_or(0, |difference| difference + 1);
    if certificate.scale_exponent != scale_exponent {
        return Err(mismatch(
            "pseudo-division changed the original scaling exponent",
        ));
    }
    if reference_degree(&certificate.remainder, &mut work)?
        .is_some_and(|degree| degree >= divisor_degree)
    {
        return Err(mismatch(
            "pseudo-remainder has not reached its strict primary-degree bound",
        ));
    }
    let mut leading = zero(dividend.generators(), &mut work)?;
    for (powers, value) in divisor.terms() {
        work.tick(powers.len() + 1)?;
        if powers[0] == divisor_degree {
            work.allocate(256 + powers.len() * 4)?;
            let projected = std::iter::once(0)
                .chain(powers.iter().skip(1).copied())
                .collect();
            let coefficient = work.clone_value(value)?;
            leading.terms.insert(projected, coefficient);
        }
    }
    let mut residual = copy_poly(dividend, &mut work)?;
    for _ in 0..scale_exponent {
        residual = reference_product(&residual, &leading, &mut work)?;
    }
    let product = reference_product(&certificate.quotient, divisor, &mut work)?;
    for polynomial in [&product, &certificate.remainder] {
        for (powers, value) in polynomial.terms() {
            work.allocate(powers.len() * 4)?;
            let coefficient = work.subtract(&BigRational::zero(), value)?;
            reference_accumulate(&mut residual.terms, powers.clone(), coefficient, &mut work)?;
        }
    }
    if !residual.is_zero() {
        return Err(mismatch(
            "pseudo-division original-source coefficient identity failed",
        ));
    }
    work.allocate(size_of::<VerifiedPseudoDivision>())?;
    let accepted = VerifiedPseudoDivision {
        domain,
        dividend: copy_poly(dividend, &mut work)?,
        divisor: copy_poly(divisor, &mut work)?,
        quotient: copy_poly(&certificate.quotient, &mut work)?,
        remainder: copy_poly(&certificate.remainder, &mut work)?,
        scale_exponent,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently verify under separate cumulative phase budgets.
pub fn checked_pseudo_division(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPseudoDivision, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let certificate =
        bounded_pseudo_division_certificate(dividend, divisor, domain, &mut generation)?;
    let accepted =
        verify_pseudo_division(dividend, divisor, domain, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
