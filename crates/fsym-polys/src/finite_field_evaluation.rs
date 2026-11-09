//! Checked partial evaluation of sparse prime-field polynomials.
//!
//! An assignment contains one optional canonical field element per original
//! typed generator. Assigned coordinates disappear; the remaining generators
//! keep their original order, including coordinates absent from every term.
//! Assigning every coordinate returns a polynomial in the explicit constant
//! field. No sampling, field extension, or reinterpretation over QQ is used.
//!
//! The generator contracts sparse terms using binary modular powers. The
//! independent verifier multiplies each assigned coordinate once per original
//! exponent and accumulates complete signed coefficient columns. It checks the
//! output characteristic and exact ordered typed ring before any identity.
//! Both phases retain the polynomial work limits and cancellation checkpoints
//! through immutable publication. Receipts are in-process, not portable proofs.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field::scalar_power;
use crate::finite_field_multivariate::{
    MAX_DEGREE, MAX_TERMS, accumulate, admit_ring, admit_terms, copied_polynomial, empty, same_ring,
};
use crate::{PolyError, PrimeFieldMultivariatePolynomial as Polynomial};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::collections::BTreeMap;

/// Untrusted proposed value, including its output characteristic and typed ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldEvaluationCertificate {
    pub result: Polynomial,
}

/// An exact original polynomial, assignment and independently checked value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPrimeFieldEvaluation {
    source: Polynomial,
    values: Vec<Option<u64>>,
    result: Polynomial,
}

impl VerifiedPrimeFieldEvaluation {
    pub fn source(&self) -> &Polynomial {
        &self.source
    }

    pub fn values(&self) -> &[Option<u64>] {
        &self.values
    }

    pub fn result(&self) -> &Polynomial {
        &self.result
    }

    pub fn into_result(self) -> Polynomial {
        self.result
    }
}

fn admit(
    source: &Polynomial,
    values: &[Option<u64>],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(source.generators(), source.prime(), work)?;
    same_ring(source, source.generators(), source.prime(), work)?;
    if values.len() != source.generators().len() {
        return Err(refused(
            "evaluation requires one assignment per original generator",
        ));
    }
    for value in values {
        work.tick(1)?;
        if value.is_some_and(|value| value >= source.prime()) {
            return Err(refused(
                "evaluation requires canonical prime-field residues",
            ));
        }
    }
    Ok(())
}

fn generated_value(
    source: &Polynomial,
    values: &[Option<u64>],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let bytes: usize = source.generators().iter().map(|g| g.name.len()).sum();
    work.tick(bytes + values.len())?;
    work.allocate(bytes + source.generators().len() * std::mem::size_of::<Symbol>())?;
    let remaining: Vec<_> = source
        .generators()
        .iter()
        .zip(values)
        .filter(|(_, value)| value.is_none())
        .map(|(generator, _)| generator.clone())
        .collect();
    let mut terms = empty(work)?;
    for (monomial, coefficient) in source.terms() {
        work.tick(monomial.len() + 1)?;
        work.allocate(remaining.len() * std::mem::size_of::<u32>())?;
        let mut projected = Vec::with_capacity(remaining.len());
        let mut coefficient = *coefficient;
        for (&exponent, value) in monomial.iter().zip(values) {
            if let Some(value) = value {
                coefficient = coefficient
                    * scalar_power(*value, u64::from(exponent), source.prime(), work)?
                    % source.prime();
            } else {
                projected.push(exponent);
            }
        }
        accumulate(&mut terms, projected, coefficient, source.prime(), work)?;
    }
    copied_polynomial(&remaining, source.prime(), &terms, work)
}

fn reference_identity(
    source: &Polynomial,
    values: &[Option<u64>],
    candidate: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let width = values.iter().filter(|value| value.is_none()).count();
    if candidate.prime() != source.prime() || candidate.generators().len() != width {
        return Err(PolyError::IdentityCheckFailed(
            "evaluation output ring".into(),
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
    admit_terms(
        candidate.terms(),
        width,
        source.prime(),
        MAX_DEGREE,
        false,
        work,
    )?;
    work.allocate(std::mem::size_of::<BTreeMap<Vec<u32>, i128>>())?;
    let mut columns: BTreeMap<Vec<u32>, i128> = BTreeMap::new();
    for (monomial, coefficient) in candidate.terms() {
        work.tick(monomial.len() + 1)?;
        work.allocate(128 + monomial.len() * std::mem::size_of::<u32>())?;
        columns.insert(monomial.clone(), -i128::from(*coefficient));
    }
    for (monomial, coefficient) in source.terms() {
        work.tick(monomial.len() + 1)?;
        work.allocate(width * std::mem::size_of::<u32>())?;
        let mut projected = Vec::with_capacity(width);
        let mut value = i128::from(*coefficient);
        for (index, &exponent) in monomial.iter().enumerate() {
            match values[index] {
                None => projected.push(exponent),
                Some(assigned) => {
                    // Deliberately independent of the generator's binary
                    // exponentiation and modular accumulation helpers.
                    for _ in 0..exponent {
                        work.tick(1)?;
                        value = value * i128::from(assigned) % i128::from(source.prime());
                    }
                }
            }
        }
        if !columns.contains_key(&projected) {
            if columns.len() >= 2 * MAX_TERMS {
                return Err(refused("evaluation verifier coefficient-column bound"));
            }
            work.allocate(128)?;
        }
        // At most 4096 canonical source coefficients enter any column, so
        // even the unreduced signed sum is safely bounded within i128.
        *columns.entry(projected).or_default() += value;
    }
    for value in columns.values() {
        work.tick(1)?;
        if value.rem_euclid(i128::from(source.prime())) != 0 {
            return Err(PolyError::IdentityCheckFailed(
                "prime-field evaluation coefficient identity".into(),
            ));
        }
    }
    Ok(())
}

/// Verify the exact evaluation map, not merely agreement at sampled points.
pub fn verify_prime_field_evaluation(
    source: &Polynomial,
    values: &[Option<u64>],
    certificate: &PrimeFieldEvaluationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldEvaluation, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(source, values, &mut work)?;
    reference_identity(source, values, &certificate.result, &mut work)?;
    work.allocate(
        std::mem::size_of::<VerifiedPrimeFieldEvaluation>() + std::mem::size_of_val(values),
    )?;
    let accepted = VerifiedPrimeFieldEvaluation {
        source: copied_polynomial(
            source.generators(),
            source.prime(),
            source.terms(),
            &mut work,
        )?,
        values: values.to_vec(),
        result: copied_polynomial(
            certificate.result.generators(),
            certificate.result.prime(),
            certificate.result.terms(),
            &mut work,
        )?,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently verify a partial or complete exact evaluation.
pub fn checked_prime_field_evaluation(
    source: &Polynomial,
    values: &[Option<u64>],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldEvaluation, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(source, values, &mut generation)?;
    let certificate = PrimeFieldEvaluationCertificate {
        result: generated_value(source, values, &mut generation)?,
    };
    let accepted = verify_prime_field_evaluation(source, values, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
