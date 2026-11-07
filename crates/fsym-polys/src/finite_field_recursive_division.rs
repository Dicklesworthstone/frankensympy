//! Checked recursive polynomial division in F_p[x_2, ..., x_n][x_1].
//!
//! Public Poly.div treats the first declared variable as the main variable.
//! A step is permitted only when the entire leading coefficient is exactly
//! divisible in the remaining polynomial ring. At the first obstruction it
//! stops, even when individual remainder monomials could still be reduced.
//! Sparse reduced() and Groebner normal forms have different semantics.
//!
//! The generator uses sparse single-divisor division to decide exact leading
//! coefficient divisibility. The independent verifier never calls division:
//! it replays each forced main-degree cancellation using signed coefficient
//! columns, then checks a nonzero fully reduced coefficient remainder when
//! division stopped at an obstruction. Reconstruction and a terminal
//! obstruction alone would not determine the unique recursive quotient.
//!
//! Inputs and witnesses retain one to eight ordered typed variables, a prime
//! up to 65521, degree at most 64 and at most 4096 terms per polynomial.
//! Temporary products may have degree 128. Each complete phase shares one
//! cumulative 20M-step / 128MiB logical-allocation envelope. Refusal or
//! cancellation publishes no accepted partial result. These in-process
//! receipts are not portable proof-kernel certificates.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    Columns, MAX_DEGREE, MAX_TERMS, accumulate, admit_ring, admit_terms, copied_polynomial,
    copy_terms, divide, empty, identity_zero, monomial_copy, product, reference_add,
    reference_product, same_ring, sum, verify_division_identity,
};
use crate::{
    PolyError, PrimeFieldMultivariateDivisionCertificate,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    TermOrder,
};
use fsym_budget::BudgetMeter;

/// Untrusted recursive quotient and remainder, in the original ordered ring.
///
/// When the final remainder still has sufficient main-variable degree, the
/// terminal witness divides its leading coefficient by the divisor's leading
/// coefficient. It has exactly one quotient and a nonzero fully reduced
/// remainder, proving why no further recursive step is possible. Every term
/// in that witness has zero first coordinate. Otherwise terminal is absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldRecursiveDivisionCertificate {
    pub quotient: Terms,
    pub remainder: Terms,
    pub terminal: Option<PrimeFieldMultivariateDivisionCertificate>,
}

/// The unique recursive Poly.div result for the supplied typed operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPrimeFieldRecursiveDivision {
    dividend: Polynomial,
    divisor: Polynomial,
    quotient: Polynomial,
    remainder: Polynomial,
}

impl VerifiedPrimeFieldRecursiveDivision {
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

fn mismatch(message: &str) -> PolyError {
    PolyError::IdentityCheckFailed(message.into())
}

fn admit_request(
    dividend: &Polynomial,
    divisor: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if dividend.generators().is_empty() {
        return Err(refused(
            "recursive polynomial division requires a declared main variable",
        ));
    }
    admit_ring(dividend.generators(), dividend.prime(), work)?;
    same_ring(dividend, dividend.generators(), dividend.prime(), work)?;
    same_ring(divisor, dividend.generators(), dividend.prime(), work)?;
    if divisor.is_zero() {
        return Err(PolyError::DivisionByZero);
    }
    Ok(())
}

fn admit_certificate(
    certificate: &PrimeFieldRecursiveDivisionCertificate,
    width: usize,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_terms(&certificate.quotient, width, prime, MAX_DEGREE, false, work)?;
    admit_terms(
        &certificate.remainder,
        width,
        prime,
        MAX_DEGREE,
        false,
        work,
    )?;
    if let Some(terminal) = &certificate.terminal {
        if terminal.quotients.len() != 1 {
            return Err(refused("recursive terminal coefficient quotient count"));
        }
        for terms in terminal
            .quotients
            .iter()
            .chain(std::iter::once(&terminal.remainder))
        {
            admit_terms(terms, width, prime, MAX_DEGREE, false, work)?;
            work.tick(terms.len())?;
            if terms.keys().any(|monomial| monomial[0] != 0) {
                return Err(refused(
                    "recursive coefficient witness contains the main variable",
                ));
            }
        }
    }
    Ok(())
}

fn generated_degree(
    source: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<u32>, PolyError> {
    work.tick(1)?;
    // Canonical BTreeMap keys use lexicographic vector order.
    Ok(source.last_key_value().map(|(monomial, _)| monomial[0]))
}

fn generated_coefficient(
    source: &Terms,
    degree: u32,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut coefficient = empty(work)?;
    for (monomial, value) in source {
        work.tick(monomial.len() + 1)?;
        if monomial[0] == degree {
            let mut projected = monomial_copy(monomial, work)?;
            projected[0] = 0;
            accumulate(&mut coefficient, projected, *value, prime, work)?;
        }
    }
    Ok(coefficient)
}

fn generated_shift(
    coefficient: &Terms,
    shift: u32,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut shifted = empty(work)?;
    for (monomial, value) in coefficient {
        if monomial[0] != 0 {
            return Err(refused("generated coefficient contains the main variable"));
        }
        let mut moved = monomial_copy(monomial, work)?;
        moved[0] = shift;
        accumulate(&mut shifted, moved, *value, prime, work)?;
    }
    Ok(shifted)
}

fn generate(
    dividend: &Polynomial,
    divisor: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldRecursiveDivisionCertificate, PolyError> {
    let prime = dividend.prime();
    let divisor_degree =
        generated_degree(divisor.terms(), work)?.ok_or(PolyError::DivisionByZero)?;
    let divisor_coefficient = generated_coefficient(divisor.terms(), divisor_degree, prime, work)?;
    let mut current = copy_terms(dividend.terms(), work)?;
    let mut quotient = empty(work)?;
    let mut terminal = None;
    while let Some(degree) = generated_degree(&current, work)? {
        if degree < divisor_degree {
            break;
        }
        let coefficient = generated_coefficient(&current, degree, prime, work)?;
        let division = divide(
            &coefficient,
            &[&divisor_coefficient],
            TermOrder::Lex,
            prime,
            work,
        )?;
        if !division.remainder.is_empty() {
            terminal = Some(division);
            break;
        }
        let step = generated_shift(&division.quotients[0], degree - divisor_degree, prime, work)?;
        quotient = sum(&quotient, &step, false, prime, work)?;
        let multiple = product(&step, divisor.terms(), prime, work)?;
        current = sum(&current, &multiple, true, prime, work)?;
        if generated_degree(&current, work)?.is_some_and(|next| next >= degree) {
            return Err(refused(
                "recursive division did not decrease its main degree",
            ));
        }
    }
    let certificate = PrimeFieldRecursiveDivisionCertificate {
        quotient,
        remainder: current,
        terminal,
    };
    admit_certificate(&certificate, dividend.generators().len(), prime, work)?;
    work.tick(1)?;
    Ok(certificate)
}

/// Generate an untrusted recursive-division candidate with a whole-phase meter.
pub fn bounded_prime_field_recursive_division_certificate(
    dividend: &Polynomial,
    divisor: &Polynomial,
    generator: &mut impl BudgetMeter,
) -> Result<PrimeFieldRecursiveDivisionCertificate, PolyError> {
    let mut work = PolynomialWork::new(generator);
    admit_request(dividend, divisor, &mut work)?;
    generate(dividend, divisor, &mut work)
}

// Reference work scans main-variable degrees and assembles signed integer
// columns. It does not use generated_degree/coefficient/shift, sparse modular
// sum/product, or either recursive or sparse division generation.
fn reference_degree(
    source: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<u32>, PolyError> {
    let mut degree = None;
    for monomial in source.keys() {
        work.tick(monomial.len() + 1)?;
        degree = Some(degree.map_or(monomial[0], |previous: u32| previous.max(monomial[0])));
    }
    Ok(degree)
}

fn reference_coefficient(
    source: &Terms,
    degree: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (monomial, value) in source {
        work.tick(monomial.len() + 1)?;
        if monomial[0] != degree {
            continue;
        }
        work.allocate(128 + std::mem::size_of_val(monomial.as_slice()))?;
        let projected = std::iter::once(0)
            .chain(monomial.iter().skip(1).copied())
            .collect();
        result.insert(projected, *value);
    }
    Ok(result)
}

fn reference_shift(
    coefficient: &Terms,
    shift: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (monomial, value) in coefficient {
        work.tick(monomial.len() + 1)?;
        work.allocate(128 + std::mem::size_of_val(monomial.as_slice()))?;
        let moved = std::iter::once(shift)
            .chain(monomial.iter().skip(1).copied())
            .collect();
        result.insert(moved, *value);
    }
    Ok(result)
}

fn columns(work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Columns, PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    Ok(Columns::new())
}

fn normalized_columns(
    columns: Columns,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (monomial, value) in columns {
        work.tick(monomial.len() + 1)?;
        let value = value.rem_euclid(i128::from(prime)) as u64;
        if value != 0 {
            if result.len() == MAX_TERMS {
                return Err(refused("recursive reference sparse term bound"));
            }
            work.allocate(128)?;
            result.insert(monomial, value);
        }
    }
    Ok(result)
}

fn verify_steps(
    dividend: &Polynomial,
    divisor: &Polynomial,
    certificate: &PrimeFieldRecursiveDivisionCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let prime = dividend.prime();
    let divisor_degree =
        reference_degree(divisor.terms(), work)?.ok_or(PolyError::DivisionByZero)?;
    let divisor_coefficient = reference_coefficient(divisor.terms(), divisor_degree, work)?;
    let mut current = copy_terms(dividend.terms(), work)?;
    if let Some(quotient_degree) = reference_degree(&certificate.quotient, work)? {
        for shift in (0..=quotient_degree).rev() {
            work.tick(1)?;
            let coefficient = reference_coefficient(&certificate.quotient, shift, work)?;
            if coefficient.is_empty() {
                continue;
            }
            let degree = reference_degree(&current, work)?
                .ok_or_else(|| mismatch("recursive quotient continues after a zero remainder"))?;
            if degree.checked_sub(divisor_degree) != Some(shift) {
                return Err(mismatch(
                    "recursive quotient skipped or added a main-degree step",
                ));
            }
            let leading_coefficient = reference_coefficient(&current, degree, work)?;
            let mut cancellation = columns(work)?;
            reference_add(&mut cancellation, &leading_coefficient, 1, work)?;
            reference_product(
                &mut cancellation,
                &coefficient,
                &divisor_coefficient,
                -1,
                work,
            )?;
            identity_zero(&cancellation, prime, work)?;

            let step = reference_shift(&coefficient, shift, work)?;
            let mut next = columns(work)?;
            reference_add(&mut next, &current, 1, work)?;
            reference_product(&mut next, &step, divisor.terms(), -1, work)?;
            current = normalized_columns(next, prime, work)?;
            admit_terms(
                &current,
                dividend.generators().len(),
                prime,
                2 * MAX_DEGREE,
                false,
                work,
            )?;
            if reference_degree(&current, work)?.is_some_and(|next| next >= degree) {
                return Err(mismatch("recursive reference main degree did not decrease"));
            }
        }
    }

    let mut remainder_identity = columns(work)?;
    reference_add(&mut remainder_identity, &current, 1, work)?;
    reference_add(&mut remainder_identity, &certificate.remainder, -1, work)?;
    identity_zero(&remainder_identity, prime, work)?;

    let blocked = reference_degree(&current, work)?.is_some_and(|degree| degree >= divisor_degree);
    match (blocked, &certificate.terminal) {
        (false, None) => {}
        (true, Some(terminal)) => {
            if terminal.remainder.is_empty() {
                return Err(mismatch(
                    "recursive terminal coefficient is still exactly divisible",
                ));
            }
            let degree = reference_degree(&current, work)?
                .ok_or_else(|| mismatch("missing recursive terminal main degree"))?;
            let coefficient = reference_coefficient(&current, degree, work)?;
            verify_division_identity(
                &coefficient,
                &[&divisor_coefficient],
                TermOrder::Lex,
                prime,
                terminal,
                work,
            )?;
        }
        _ => {
            return Err(mismatch(
                "recursive division has an incorrect terminal witness",
            ));
        }
    }
    work.tick(1)
}

/// Independently replay every uniquely forced cancellation and certify stopping.
///
/// This proves the recursive Poly.div result, not a complete monomial normal
/// form or ideal nonmembership. Original ordered typed rings are checked even
/// for zero and constant operands, before any arithmetic shortcut.
pub fn verify_prime_field_recursive_division(
    dividend: &Polynomial,
    divisor: &Polynomial,
    certificate: &PrimeFieldRecursiveDivisionCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldRecursiveDivision, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit_request(dividend, divisor, &mut work)?;
    admit_certificate(
        certificate,
        dividend.generators().len(),
        dividend.prime(),
        &mut work,
    )?;
    verify_steps(dividend, divisor, certificate, &mut work)?;
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldRecursiveDivision>())?;
    let accepted = VerifiedPrimeFieldRecursiveDivision {
        dividend: copied_polynomial(
            dividend.generators(),
            dividend.prime(),
            dividend.terms(),
            &mut work,
        )?,
        divisor: copied_polynomial(
            divisor.generators(),
            divisor.prime(),
            divisor.terms(),
            &mut work,
        )?,
        quotient: copied_polynomial(
            dividend.generators(),
            dividend.prime(),
            &certificate.quotient,
            &mut work,
        )?,
        remainder: copied_polynomial(
            dividend.generators(),
            dividend.prime(),
            &certificate.remainder,
            &mut work,
        )?,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently accept the recursive result under separate ledgers.
pub fn checked_prime_field_recursive_division(
    dividend: &Polynomial,
    divisor: &Polynomial,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldRecursiveDivision, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let certificate =
        bounded_prime_field_recursive_division_certificate(dividend, divisor, &mut generation)?;
    let accepted =
        verify_prime_field_recursive_division(dividend, divisor, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
