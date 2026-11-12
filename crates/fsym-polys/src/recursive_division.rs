//! Checked recursive division in QQ[x_2, ..., x_n][x_1] and its strict ZZ subring.
//!
//! The first declared variable is primary. Each step must divide the whole
//! leading coefficient in the remaining polynomial ring. At the first
//! obstruction the operation stops, even if individual monomials could still
//! be reduced. This is the Poly.div operation, not a distributed normal form.
//! Strict ZZ division also stops when the unique rational coefficient quotient
//! has a fractional coefficient; it never silently extends the declared domain.
//!
//! Generation uses bounded sparse division for coefficient divisibility. An
//! independent checker replays every forced primary-degree cancellation from
//! the proposed quotient and verifies the nonzero coefficient remainder that
//! certifies a terminal obstruction. Reconstruction alone is insufficient.
//!
//! Both inputs and all proposed values retain one to eight ordered typed
//! variables, total degree256, 4096 terms and 4096-bit exact coefficients.
//! Each complete phase shares a cumulative 20M-compute/128MiB allocation cap
//! and the caller's cancellation ledger. The immutable receipt is an
//! in-process composition result, not portable proof-kernel evidence. Public
//! expression consumers must first admit their original held sources through
//! bounded_polynomial_from_expr on these same phase ledgers.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_DEGREE, MAX_TERMS, MAX_VARIABLES, PolynomialWork, Terms, accumulate, copy_poly,
    divide_with_work, multiply_polynomials, refused, ring_storage, same_ring, validate_poly,
};
use crate::{MultivariatePoly as Polynomial, PolyError, TermOrder, verify_bounded_division};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::Zero;
use std::mem::size_of;

/// Exact coefficient domain; strict integer division never promotes to QQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursiveDivisionDomain {
    QQ,
    ZZ,
}

/// A nonzero reduced coefficient remainder proves that the next mandatory
/// primary-variable cancellation is impossible in the coefficient ring.
/// For strict ZZ, a zero remainder with a nonintegral quotient is also a
/// stopping witness: its unique rational quotient is outside the integer ring.
/// Both polynomials use the complete original ring with first exponent zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveCoefficientObstruction {
    pub quotient: Polynomial,
    pub remainder: Polynomial,
}

/// Untrusted recursive quotient/remainder and its optional stopping evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveDivisionCertificate {
    pub quotient: Polynomial,
    pub remainder: Polynomial,
    pub terminal: Option<RecursiveCoefficientObstruction>,
}

/// The uniquely determined recursive division for the exact original inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRecursiveDivision {
    domain: RecursiveDivisionDomain,
    dividend: Polynomial,
    divisor: Polynomial,
    quotient: Polynomial,
    remainder: Polynomial,
}

impl VerifiedRecursiveDivision {
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

    pub fn into_parts(self) -> (Polynomial, Polynomial) {
        (self.quotient, self.remainder)
    }
}

fn mismatch(message: &'static str) -> PolyError {
    PolyError::IdentityCheckFailed(message.into())
}

fn check_ring(
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if variables.is_empty() || variables.len() > MAX_VARIABLES {
        return Err(refused(
            "recursive division requires one to eight variables",
        ));
    }
    let mut bytes = 0usize;
    for (index, variable) in variables.iter().enumerate() {
        bytes = bytes
            .checked_add(variable.name.len())
            .filter(|&value| value <= 65_536)
            .ok_or_else(|| refused("recursive generator names exceed 65536 bytes"))?;
        work.tick((index + 1) * (variable.name.len() + 1))?;
        if variables[..index].contains(variable) {
            return Err(refused("duplicate typed recursive-division variable"));
        }
    }
    Ok(())
}

fn check_polynomial(
    polynomial: &Polynomial,
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    // Untrusted ring payloads are bounded before equality/diagnostic formatting.
    check_ring(polynomial.generators(), work)?;
    same_ring(polynomial.generators(), variables)?;
    validate_poly(polynomial, work)
}

fn admit_request(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_polynomial(dividend, dividend.generators(), work)?;
    check_polynomial(divisor, dividend.generators(), work)?;
    if domain == RecursiveDivisionDomain::ZZ
        && (!integral(dividend, work)? || !integral(divisor, work)?)
    {
        return Err(refused(
            "strict ZZ recursive division requires integer coefficients",
        ));
    }
    if divisor.is_zero() {
        return Err(PolyError::DivisionByZero);
    }
    Ok(())
}

fn admit_certificate(
    certificate: &RecursiveDivisionCertificate,
    variables: &[Symbol],
    domain: RecursiveDivisionDomain,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_polynomial(&certificate.quotient, variables, work)?;
    check_polynomial(&certificate.remainder, variables, work)?;
    if domain == RecursiveDivisionDomain::ZZ
        && (!integral(&certificate.quotient, work)? || !integral(&certificate.remainder, work)?)
    {
        return Err(mismatch(
            "strict ZZ recursive output contains a fractional coefficient",
        ));
    }
    if let Some(terminal) = &certificate.terminal {
        for polynomial in [&terminal.quotient, &terminal.remainder] {
            check_polynomial(polynomial, variables, work)?;
            work.tick(polynomial.terms().len())?;
            if polynomial.terms().keys().any(|powers| powers[0] != 0) {
                return Err(refused(
                    "recursive coefficient witness contains its primary variable",
                ));
            }
        }
    }
    Ok(())
}

fn integral(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    work.tick(polynomial.terms().len() + 1)?;
    Ok(polynomial.terms().values().all(BigRational::is_integer))
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
    // Canonical sparse storage is ordered lexicographically by exponent vector.
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
    let mut coefficient = zero(polynomial.generators(), work)?;
    for (powers, value) in polynomial.terms() {
        work.tick(powers.len() + 1)?;
        if powers[0] == degree {
            work.allocate(powers.len() * 4)?;
            let mut projected = powers.clone();
            projected[0] = 0;
            let value = work.clone_value(value)?;
            accumulate(&mut coefficient.terms, projected, value, work)?;
        }
    }
    Ok(coefficient)
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
                "generated coefficient contains its primary variable",
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
        let value = if subtract {
            work.subtract(&BigRational::zero(), value)?
        } else {
            work.clone_value(value)?
        };
        accumulate(&mut result.terms, powers.clone(), value, work)?;
    }
    Ok(result)
}

/// Generate an untrusted recursive-division candidate under a cumulative meter.
pub fn bounded_recursive_division_certificate(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    generator: &mut impl BudgetMeter,
) -> Result<RecursiveDivisionCertificate, PolyError> {
    let mut work = PolynomialWork::new(generator);
    admit_request(dividend, divisor, domain, &mut work)?;
    let divisor_degree = generated_degree(divisor, &mut work)?.ok_or(PolyError::DivisionByZero)?;
    let divisor_coefficient = generated_coefficient(divisor, divisor_degree, &mut work)?;
    let mut current = copy_poly(dividend, &mut work)?;
    let mut quotient = zero(dividend.generators(), &mut work)?;
    let mut terminal = None;
    while let Some(degree) = generated_degree(&current, &mut work)? {
        if degree < divisor_degree {
            break;
        }
        let coefficient = generated_coefficient(&current, degree, &mut work)?;
        let (mut quotients, remainder) = divide_with_work(
            &coefficient,
            std::slice::from_ref(&divisor_coefficient),
            TermOrder::Lex,
            &mut work,
        )?;
        if quotients.len() != 1 {
            return Err(refused(
                "recursive coefficient division changed its quotient count",
            ));
        }
        let part = quotients
            .pop()
            .ok_or_else(|| refused("missing coefficient quotient"))?;
        if !remainder.is_zero()
            || (domain == RecursiveDivisionDomain::ZZ && !integral(&part, &mut work)?)
        {
            work.allocate(size_of::<RecursiveCoefficientObstruction>())?;
            terminal = Some(RecursiveCoefficientObstruction {
                quotient: part,
                remainder,
            });
            break;
        }
        let step = generated_shift(&part, degree - divisor_degree, &mut work)?;
        quotient = generated_sum(&quotient, &step, false, &mut work)?;
        let multiple = multiply_polynomials(&step, divisor, &mut work)?;
        current = generated_sum(&current, &multiple, true, &mut work)?;
        if generated_degree(&current, &mut work)?.is_some_and(|next| next >= degree) {
            return Err(refused(
                "recursive division failed to decrease its primary degree",
            ));
        }
    }
    work.allocate(size_of::<RecursiveDivisionCertificate>())?;
    let certificate = RecursiveDivisionCertificate {
        quotient,
        remainder: current,
        terminal,
    };
    admit_certificate(&certificate, dividend.generators(), domain, &mut work)?;
    work.tick(1)?;
    Ok(certificate)
}

// Reference arithmetic scans degrees and accumulates exact sparse columns. It
// never calls generated_degree/coefficient/shift/sum, polynomial multiplication,
// or a division generator. Only exact scalar arithmetic and storage are shared.
fn reference_degree(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<u32>, PolyError> {
    let mut degree = None;
    for powers in polynomial.terms().keys() {
        work.tick(powers.len() + 1)?;
        degree = Some(degree.map_or(powers[0], |previous: u32| previous.max(powers[0])));
    }
    Ok(degree)
}

fn reference_coefficient(
    polynomial: &Polynomial,
    degree: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut result = zero(polynomial.generators(), work)?;
    for (powers, value) in polynomial.terms() {
        work.tick(powers.len() + 1)?;
        if powers[0] == degree {
            work.allocate(256 + powers.len() * 4)?;
            let projected = std::iter::once(0)
                .chain(powers.iter().skip(1).copied())
                .collect();
            let value = work.clone_value(value)?;
            result.terms.insert(projected, value);
        }
    }
    Ok(result)
}

fn reference_accumulate(
    result: &mut Terms,
    powers: Vec<u32>,
    coefficient: BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(powers.len() + 1)?;
    if powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE {
        return Err(refused("recursive reference term exceeds degree256"));
    }
    let value = if let Some(previous) = result.get(&powers) {
        work.add(previous, &coefficient)?
    } else {
        coefficient
    };
    if value.is_zero() {
        result.remove(&powers);
    } else {
        if !result.contains_key(&powers) {
            if result.len() == MAX_TERMS {
                return Err(refused("recursive reference sparse term bound"));
            }
            work.allocate(256)?;
        }
        result.insert(powers, value);
    }
    Ok(())
}

/// Subtract x_1^shift * left * right by a direct coefficient convolution.
fn reference_subtract_product(
    current: &Polynomial,
    left: &Polynomial,
    right: &Polynomial,
    shift: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut result = copy_poly(current, work)?;
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            work.tick(a.len() + 1)?;
            work.allocate(a.len() * 4)?;
            let mut powers = Vec::with_capacity(a.len());
            for (axis, (&a, &b)) in a.iter().zip(b).enumerate() {
                powers.push(
                    a.checked_add(b)
                        .and_then(|value| value.checked_add(if axis == 0 { shift } else { 0 }))
                        .ok_or_else(|| refused("recursive reference exponent overflow"))?,
                );
            }
            let product = work.multiply(ca, cb)?;
            let negative = work.subtract(&BigRational::zero(), &product)?;
            reference_accumulate(&mut result.terms, powers, negative, work)?;
        }
    }
    Ok(result)
}

fn same_polynomial(
    actual: &Polynomial,
    expected: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if actual.terms().len() != expected.terms().len() {
        return Err(mismatch("recursive remainder changed exact term coverage"));
    }
    for ((a, ca), (b, cb)) in actual.terms().iter().zip(expected.terms()) {
        let limbs = ca.numer().limb_count()
            + ca.denom().limb_count()
            + cb.numer().limb_count()
            + cb.denom().limb_count();
        work.tick(a.len() + limbs as usize + 1)?;
        if a != b || ca != cb {
            return Err(mismatch("recursive remainder changed an exact coefficient"));
        }
    }
    Ok(())
}

/// Independently replay all uniquely forced cancellations and certify stopping.
/// This does not claim a distributed monomial normal form or ideal nonmembership.
pub fn verify_recursive_division(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    certificate: &RecursiveDivisionCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRecursiveDivision, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit_request(dividend, divisor, domain, &mut work)?;
    admit_certificate(certificate, dividend.generators(), domain, &mut work)?;
    let divisor_degree = reference_degree(divisor, &mut work)?.ok_or(PolyError::DivisionByZero)?;
    let divisor_coefficient = reference_coefficient(divisor, divisor_degree, &mut work)?;
    let mut current = copy_poly(dividend, &mut work)?;
    if let Some(quotient_degree) = reference_degree(&certificate.quotient, &mut work)? {
        for shift in (0..=quotient_degree).rev() {
            work.tick(1)?;
            let coefficient = reference_coefficient(&certificate.quotient, shift, &mut work)?;
            if coefficient.is_zero() {
                continue;
            }
            let degree = reference_degree(&current, &mut work)?
                .ok_or_else(|| mismatch("recursive quotient continues after a zero remainder"))?;
            if degree.checked_sub(divisor_degree) != Some(shift) {
                return Err(mismatch(
                    "recursive quotient skipped or added a primary-degree step",
                ));
            }
            let leading = reference_coefficient(&current, degree, &mut work)?;
            let cancellation = reference_subtract_product(
                &leading,
                &coefficient,
                &divisor_coefficient,
                0,
                &mut work,
            )?;
            if !cancellation.is_zero() {
                return Err(mismatch(
                    "recursive step does not cancel the entire leading coefficient",
                ));
            }
            current =
                reference_subtract_product(&current, &coefficient, divisor, shift, &mut work)?;
            if reference_degree(&current, &mut work)?.is_some_and(|next| next >= degree) {
                return Err(mismatch(
                    "recursive reference primary degree did not decrease",
                ));
            }
        }
    }
    same_polynomial(&current, &certificate.remainder, &mut work)?;
    let degree = reference_degree(&current, &mut work)?;
    let blocked = degree.is_some_and(|degree| degree >= divisor_degree);
    match (blocked, &certificate.terminal) {
        (false, None) => {}
        (true, Some(terminal)) => {
            if terminal.remainder.is_zero()
                && (domain == RecursiveDivisionDomain::QQ
                    || integral(&terminal.quotient, &mut work)?)
            {
                return Err(mismatch(
                    "recursive terminal coefficient is still exactly divisible in its domain",
                ));
            }
            let degree = degree.ok_or_else(|| mismatch("missing recursive terminal degree"))?;
            let coefficient = reference_coefficient(&current, degree, &mut work)?;
            verify_bounded_division(
                &coefficient,
                std::slice::from_ref(&divisor_coefficient),
                std::slice::from_ref(&terminal.quotient),
                &terminal.remainder,
                TermOrder::Lex,
                &mut work,
            )?;
        }
        _ => {
            return Err(mismatch(
                "recursive division has an incorrect terminal witness",
            ));
        }
    }
    work.allocate(size_of::<VerifiedRecursiveDivision>())?;
    let accepted = VerifiedRecursiveDivision {
        domain,
        dividend: copy_poly(dividend, &mut work)?,
        divisor: copy_poly(divisor, &mut work)?,
        quotient: copy_poly(&certificate.quotient, &mut work)?,
        remainder: copy_poly(&certificate.remainder, &mut work)?,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently accept recursive QQ or strict ZZ division under separate
/// cumulative ledgers. Failure never returns an accepted partial quotient.
pub fn checked_recursive_division(
    dividend: &Polynomial,
    divisor: &Polynomial,
    domain: RecursiveDivisionDomain,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRecursiveDivision, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let certificate =
        bounded_recursive_division_certificate(dividend, divisor, domain, &mut generation)?;
    let accepted =
        verify_recursive_division(dividend, divisor, domain, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
