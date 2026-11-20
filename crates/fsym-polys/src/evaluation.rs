//! Checked partial evaluation in an ordered rational polynomial ring.
//!
//! Each original coordinate has either an exact rational assignment or remains
//! an ordered typed generator, even when it is absent from every monomial.
//! Assigning every coordinate produces the explicit constant ring. Integer
//! coefficient/assignment policy belongs to callers; this exact map is over QQ.
//!
//! Generation contracts sparse monomials with binary powering. The independent
//! verifier instead repeatedly multiplies the original coefficients and adds
//! them to signed coefficient columns initialized from the proposed result.
//! It checks the complete output ring and every coefficient, never samples.
//! Separate generation and verification ledgers retain cumulative 20M-step /
//! 128MiB logical-allocation limits through immutable publication. The lane
//! admits eight variables, degree256, 4096 terms and 4096-bit coefficients and
//! assignments. These are in-process composition receipts, not portable proofs.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_TERMS, MAX_VARIABLES, PolynomialWork, Terms, accumulate, copy_poly, refused, ring_storage,
    validate_poly,
};
use crate::{MultivariatePoly as Polynomial, PolyError};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::mem::size_of;

/// Untrusted proposed value, including the exact remaining typed generators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialEvaluationCertificate {
    pub result: Polynomial,
}

/// An immutable exact source, assignment and independently checked value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPolynomialEvaluation {
    source: Polynomial,
    values: Vec<Option<BigRational>>,
    result: Polynomial,
}

impl VerifiedPolynomialEvaluation {
    pub fn source(&self) -> &Polynomial {
        &self.source
    }

    pub fn values(&self) -> &[Option<BigRational>] {
        &self.values
    }

    pub fn result(&self) -> &Polynomial {
        &self.result
    }

    pub fn into_result(self) -> Polynomial {
        self.result
    }
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused("evaluation supports at most eight generators"));
    }
    let mut names = 0usize;
    for (index, generator) in generators.iter().enumerate() {
        names = names
            .checked_add(generator.name.len())
            .filter(|&bytes| bytes <= 65_536)
            .ok_or_else(|| refused("evaluation generator names exceed 65536 bytes"))?;
        work.tick((index + 1) * (generator.name.len() + 1))?;
        if generators[..index].contains(generator) {
            return Err(refused("duplicate typed evaluation generator"));
        }
    }
    Ok(())
}

fn check_value(value: &BigRational) -> Result<(), PolyError> {
    if value.numer().bits() > 4096 || value.denom().bits() > 4096 {
        return Err(refused("evaluation value exceeds 4096 bits"));
    }
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(refused(
            "evaluation requires a positive rational denominator",
        ));
    }
    Ok(())
}

fn admit(
    source: &Polynomial,
    values: &[Option<BigRational>],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_ring(source.generators(), work)?;
    validate_poly(source, work)?;
    if values.len() != source.generators().len() {
        return Err(refused(
            "evaluation requires one assignment per original generator",
        ));
    }
    for value in values {
        work.tick(1)?;
        if let Some(value) = value {
            check_value(value)?;
        }
    }
    Ok(())
}

fn binary_power(
    value: &BigRational,
    mut exponent: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<BigRational, PolyError> {
    work.allocate(64)?;
    let mut power = BigRational::one();
    if exponent == 0 {
        return Ok(power);
    }
    let mut base = work.clone_value(value)?;
    loop {
        work.tick(1)?;
        if exponent & 1 != 0 {
            power = work.multiply(&power, &base)?;
        }
        exponent >>= 1;
        if exponent == 0 {
            return Ok(power);
        }
        base = work.multiply(&base, &base)?;
    }
}

fn generated_value(
    source: &Polynomial,
    values: &[Option<BigRational>],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    work.allocate(ring_storage(source.generators()) + size_of::<Terms>())?;
    let remaining: Vec<_> = source
        .generators()
        .iter()
        .zip(values)
        .filter(|(_, value)| value.is_none())
        .map(|(generator, _)| generator.clone())
        .collect();
    let mut terms = Terms::new();
    for (monomial, coefficient) in source.terms() {
        work.tick(monomial.len() + 1)?;
        work.allocate(remaining.len() * size_of::<u32>())?;
        let mut projected = Vec::with_capacity(remaining.len());
        let mut coefficient = work.clone_value(coefficient)?;
        for (&exponent, value) in monomial.iter().zip(values) {
            if let Some(value) = value {
                let power = binary_power(value, exponent, work)?;
                coefficient = work.multiply(&coefficient, &power)?;
            } else {
                projected.push(exponent);
            }
        }
        accumulate(&mut terms, projected, coefficient, work)?;
    }
    work.tick(1)?;
    Ok(Polynomial {
        generators: remaining,
        terms,
    })
}

fn reference_identity(
    source: &Polynomial,
    values: &[Option<BigRational>],
    candidate: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_ring(candidate.generators(), work)?;
    validate_poly(candidate, work)?;
    let width = values.iter().filter(|value| value.is_none()).count();
    if candidate.generators().len() != width {
        return Err(PolyError::IdentityCheckFailed(
            "evaluation output ring dimension".into(),
        ));
    }
    let mut output_index = 0;
    for (generator, value) in source.generators().iter().zip(values) {
        work.tick(generator.name.len() + 1)?;
        if value.is_none() {
            if &candidate.generators()[output_index] != generator {
                return Err(PolyError::IdentityCheckFailed(
                    "evaluation changed a remaining typed generator".into(),
                ));
            }
            output_index += 1;
        }
    }

    work.allocate(size_of::<BTreeMap<Vec<u32>, BigRational>>() + 64)?;
    let mut columns: BTreeMap<Vec<u32>, BigRational> = BTreeMap::new();
    let zero = BigRational::zero();
    for (monomial, coefficient) in candidate.terms() {
        work.tick(monomial.len() + 1)?;
        work.allocate(256 + monomial.len() * size_of::<u32>())?;
        columns.insert(monomial.clone(), work.subtract(&zero, coefficient)?);
    }
    for (monomial, coefficient) in source.terms() {
        work.tick(monomial.len() + 1)?;
        work.allocate(width * size_of::<u32>())?;
        let mut projected = Vec::with_capacity(width);
        let mut value = work.clone_value(coefficient)?;
        for (index, &exponent) in monomial.iter().enumerate() {
            match &values[index] {
                None => projected.push(exponent),
                Some(assigned) => {
                    // Reference recurrence does not call binary_power or the
                    // generator's polynomial accumulation helper.
                    for _ in 0..exponent {
                        work.tick(1)?;
                        value = work.multiply(&value, assigned)?;
                    }
                }
            }
        }
        if let Some(previous) = columns.get_mut(&projected) {
            *previous = work.add(previous, &value)?;
        } else {
            if columns.len() >= 2 * MAX_TERMS {
                return Err(refused("evaluation verifier coefficient-column bound"));
            }
            work.allocate(256)?;
            columns.insert(projected, value);
        }
    }
    for value in columns.values() {
        work.tick(1)?;
        if !value.is_zero() {
            return Err(PolyError::IdentityCheckFailed(
                "rational evaluation coefficient identity".into(),
            ));
        }
    }
    Ok(())
}

/// Independently verify the exact partial-evaluation map and bind its output.
pub fn verify_polynomial_evaluation(
    source: &Polynomial,
    values: &[Option<BigRational>],
    certificate: &PolynomialEvaluationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialEvaluation, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(source, values, &mut work)?;
    reference_identity(source, values, &certificate.result, &mut work)?;
    work.allocate(size_of::<VerifiedPolynomialEvaluation>() + std::mem::size_of_val(values))?;
    let mut retained = Vec::with_capacity(values.len());
    for value in values {
        work.tick(1)?;
        retained.push(
            value
                .as_ref()
                .map(|value| work.clone_value(value))
                .transpose()?,
        );
    }
    let accepted = VerifiedPolynomialEvaluation {
        source: copy_poly(source, &mut work)?,
        values: retained,
        result: copy_poly(&certificate.result, &mut work)?,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently check a partial or complete rational evaluation.
pub fn checked_polynomial_evaluation(
    source: &Polynomial,
    values: &[Option<BigRational>],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialEvaluation, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(source, values, &mut generation)?;
    let certificate = PolynomialEvaluationCertificate {
        result: generated_value(source, values, &mut generation)?,
    };
    let accepted = verify_polynomial_evaluation(source, values, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
