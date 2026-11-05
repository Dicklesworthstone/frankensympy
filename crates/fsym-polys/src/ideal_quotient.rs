//! Checked finite ideal quotients over QQ.
//!
//! For a nonzero divisor f, complete generators of `I intersect <f>` are divided
//! exactly by f. Their quotients generate `I:<f>`: multiplication by f identifies
//! the quotient ideal with that intersection in the polynomial integral domain.
//! The quotient by `J=<f_1,...,f_n>` is the intersection of all principal quotients.
//! In particular this removes only one factor, unlike principal saturation.
//!
//! A separate composition checker binds every immutable accepted intersection
//! and Groebner receipt to the exact original request, checks all quotient
//! identities with the independent coefficient-convolution verifier, and checks
//! every link and the entire final basis. These are in-process composition
//! receipts, not new portable proof-kernel certificates.
//!
//! Both phases keep cumulative 20M-compute/128MiB logical allocation envelopes
//! through all nested calls. Nonzero divisor ideals require one auxiliary
//! variable, so at most seven original variables and 63 original equations are
//! admitted. Divisor count is at most 64; intermediate intersections retain the
//! existing 64-generator bound. Degrees, coefficient heights and term counts
//! retain the bounded polynomial limits. Exhaustion never publishes a prefix.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_BASIS, MAX_VARIABLES, PolynomialWork, copy_poly, divide_with_work, refused, ring_storage,
    same_ring, validate_poly,
};
use crate::{
    IdealOperation, MultivariatePoly, PolyError, TermOrder, VerifiedGroebnerBasis,
    VerifiedIdealOperation, checked_groebner_basis, checked_ideal_intersection,
    verify_bounded_division,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::sync::Arc;

/// Untrusted composition links for one nonzero original divisor.
/// Quotients cover every intersection basis element, in its original order.
#[derive(Debug, Clone)]
pub struct PrincipalIdealQuotientCertificate {
    pub divisor_index: usize,
    pub intersection: Arc<VerifiedIdealOperation>,
    pub quotients: Vec<MultivariatePoly>,
    pub quotient_basis: Arc<VerifiedGroebnerBasis>,
}

/// Nonzero-divisor witnesses retain the original divisor order. Intersections
/// combine the first principal quotient with each following one, in order.
/// With no nonzero divisors the only valid final basis is the unit ideal.
#[derive(Debug, Clone)]
pub struct IdealQuotientCertificate {
    pub principal_quotients: Vec<PrincipalIdealQuotientCertificate>,
    pub intersections: Vec<Arc<VerifiedIdealOperation>>,
    pub basis: Vec<MultivariatePoly>,
}

/// The complete reduced Lex basis of I:J in the exact supplied QQ ring.
/// This is an ideal operation, not a pointwise rational-domain exclusion.
#[derive(Debug, Clone)]
pub struct VerifiedIdealQuotient {
    generators: Vec<Symbol>,
    inputs: Vec<MultivariatePoly>,
    divisors: Vec<MultivariatePoly>,
    certificate: IdealQuotientCertificate,
}

impl VerifiedIdealQuotient {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }

    pub fn inputs(&self) -> &[MultivariatePoly] {
        &self.inputs
    }

    pub fn divisors(&self) -> &[MultivariatePoly] {
        &self.divisors
    }

    pub fn basis(&self) -> &[MultivariatePoly] {
        &self.certificate.basis
    }

    pub fn into_basis(self) -> Vec<MultivariatePoly> {
        self.certificate.basis
    }

    pub fn certificate(&self) -> &IdealQuotientCertificate {
        &self.certificate
    }
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused("ideal quotient ring exceeds eight variables"));
    }
    let mut name_bytes = 0usize;
    for (index, generator) in generators.iter().enumerate() {
        work.tick(index + 1)?;
        name_bytes = name_bytes
            .checked_add(generator.name.len())
            .filter(|&length| length <= 65_536)
            .ok_or_else(|| refused("ideal quotient generator names exceed 65536 bytes"))?;
        if generators[..index].contains(generator) {
            return Err(refused("duplicate typed ideal quotient generator"));
        }
    }
    Ok(())
}

fn check_polynomials(
    polynomials: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if polynomials.len() > MAX_BASIS {
        return Err(refused(
            "ideal quotient polynomial list exceeds 64 elements",
        ));
    }
    for polynomial in polynomials {
        // Bound supplied names before shape diagnostics or ring comparisons.
        check_ring(polynomial.generators(), work)?;
        same_ring(polynomial.generators(), generators)?;
        validate_poly(polynomial, work)?;
    }
    Ok(())
}

fn admit(
    inputs: &[MultivariatePoly],
    divisors: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<usize, PolyError> {
    check_ring(generators, work)?;
    check_polynomials(inputs, generators, work)?;
    check_polynomials(divisors, generators, work)?;
    let count = divisors
        .iter()
        .filter(|polynomial| !polynomial.is_zero())
        .count();
    if count != 0 && (generators.len() == MAX_VARIABLES || inputs.len() == MAX_BASIS) {
        return Err(refused(
            "nonzero ideal quotient requires an auxiliary variable and one extra input",
        ));
    }
    Ok(count)
}

/// Compare full typed coefficients after shape preflight and charge their
/// complete comparison. Equal ideals with differently scaled inputs do not
/// substitute for exact original-request binding.
fn same_polynomials(
    actual: &[MultivariatePoly],
    expected: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if actual.len() != expected.len() {
        return Err(refused(
            "ideal quotient witness changed polynomial coverage",
        ));
    }
    check_polynomials(actual, generators, work)?;
    check_polynomials(expected, generators, work)?;
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.terms().len() != expected.terms().len() {
            return Err(refused("ideal quotient witness changed term coverage"));
        }
        for ((powers, value), (expected_powers, expected_value)) in
            actual.terms().iter().zip(expected.terms())
        {
            let limbs = value.numer().limb_count()
                + value.denom().limb_count()
                + expected_value.numer().limb_count()
                + expected_value.denom().limb_count();
            work.tick(powers.len() + limbs as usize + 1)?;
            if powers != expected_powers || value != expected_value {
                return Err(refused("ideal quotient witness changed exact coefficients"));
            }
        }
    }
    Ok(())
}

fn constant(
    generators: &[Symbol],
    one: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    work.allocate(
        std::mem::size_of::<MultivariatePoly>()
            + ring_storage(generators)
            + 256
            + generators.len() * 4,
    )?;
    Ok(if one {
        MultivariatePoly::one(generators.to_vec())
    } else {
        MultivariatePoly::zero(generators.to_vec())
    })
}

fn copy_polynomials(
    polynomials: &[MultivariatePoly],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    work.allocate(std::mem::size_of_val(polynomials))?;
    polynomials.iter().map(|p| copy_poly(p, work)).collect()
}

/// The receipt type is immutable and can only be constructed after the
/// independent lower-layer checker accepts it. The composition checker still
/// binds its complete ordered request and original typed ring here.
fn bind_intersection(
    receipt: &VerifiedIdealOperation,
    left: &[MultivariatePoly],
    right: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if receipt.operation() != IdealOperation::Intersection
        || receipt.input_generators() != generators
        || receipt.generators() != generators
        || receipt.saturating_polynomial().is_some()
        || !receipt.eliminated_variables().is_empty()
    {
        return Err(refused(
            "ideal quotient has an unrelated intersection receipt",
        ));
    }
    same_polynomials(receipt.inputs(), left, generators, work)?;
    same_polynomials(receipt.other_inputs(), right, generators, work)?;
    check_polynomials(receipt.basis(), generators, work)
}

fn bind_quotient_basis(
    witness: &PrincipalIdealQuotientCertificate,
    generators: &[Symbol],
    zero: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if witness.quotient_basis.order() != TermOrder::Lex {
        return Err(refused("ideal quotient requires an accepted Lex basis"));
    }
    let expected = if witness.quotients.is_empty() {
        // The typed zero sentinel preserves even a zero-dimensional ring.
        std::slice::from_ref(zero)
    } else {
        witness.quotients.as_slice()
    };
    same_polynomials(
        witness.quotient_basis.input_basis(),
        expected,
        generators,
        work,
    )?;
    check_polynomials(witness.quotient_basis.basis(), generators, work)
}

/// Independently verify every exact composition link and all coefficient
/// identities. No division, multiplication, Groebner search or intersection
/// generator is invoked by this checker.
pub fn verify_ideal_quotient(
    inputs: &[MultivariatePoly],
    divisors: &[MultivariatePoly],
    generators: &[Symbol],
    certificate: &IdealQuotientCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealQuotient, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    let count = admit(inputs, divisors, generators, &mut work)?;
    if certificate.principal_quotients.len() != count
        || certificate.intersections.len() != count.saturating_sub(1)
    {
        return Err(refused(
            "ideal quotient lost or added a divisor or chain link",
        ));
    }
    let zero = constant(generators, false, &mut work)?;
    for (witness, (index, divisor)) in certificate.principal_quotients.iter().zip(
        divisors
            .iter()
            .enumerate()
            .filter(|(_, polynomial)| !polynomial.is_zero()),
    ) {
        work.tick(1)?;
        if witness.divisor_index != index {
            return Err(refused("ideal quotient changed the original divisor order"));
        }
        bind_intersection(
            &witness.intersection,
            inputs,
            std::slice::from_ref(divisor),
            generators,
            &mut work,
        )?;
        if witness.quotients.len() != witness.intersection.basis().len() {
            return Err(refused("ideal quotient omitted an intersection generator"));
        }
        check_polynomials(&witness.quotients, generators, &mut work)?;
        for (dividend, quotient) in witness.intersection.basis().iter().zip(&witness.quotients) {
            verify_bounded_division(
                dividend,
                std::slice::from_ref(divisor),
                std::slice::from_ref(quotient),
                &zero,
                TermOrder::Lex,
                &mut work,
            )?;
        }
        bind_quotient_basis(witness, generators, &zero, &mut work)?;
    }
    for (index, intersection) in certificate.intersections.iter().enumerate() {
        let left = if index == 0 {
            certificate.principal_quotients[0].quotient_basis.basis()
        } else {
            certificate.intersections[index - 1].basis()
        };
        let right = certificate.principal_quotients[index + 1]
            .quotient_basis
            .basis();
        bind_intersection(intersection, left, right, generators, &mut work)?;
    }
    let unit;
    let expected = if let Some(intersection) = certificate.intersections.last() {
        intersection.basis()
    } else if let Some(principal) = certificate.principal_quotients.first() {
        principal.quotient_basis.basis()
    } else {
        unit = constant(generators, true, &mut work)?;
        std::slice::from_ref(&unit)
    };
    same_polynomials(&certificate.basis, expected, generators, &mut work)?;

    work.allocate(
        std::mem::size_of::<VerifiedIdealQuotient>()
            + ring_storage(generators)
            + count * std::mem::size_of::<PrincipalIdealQuotientCertificate>()
            + std::mem::size_of_val(certificate.intersections.as_slice()),
    )?;
    let mut principal_quotients = Vec::with_capacity(count);
    for witness in &certificate.principal_quotients {
        principal_quotients.push(PrincipalIdealQuotientCertificate {
            divisor_index: witness.divisor_index,
            intersection: Arc::clone(&witness.intersection),
            quotients: copy_polynomials(&witness.quotients, &mut work)?,
            quotient_basis: Arc::clone(&witness.quotient_basis),
        });
    }
    let accepted = VerifiedIdealQuotient {
        generators: generators.to_vec(),
        inputs: copy_polynomials(inputs, &mut work)?,
        divisors: copy_polynomials(divisors, &mut work)?,
        certificate: IdealQuotientCertificate {
            principal_quotients,
            intersections: certificate.intersections.clone(),
            basis: copy_polynomials(&certificate.basis, &mut work)?,
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Compute the finite ideal quotient I:J={h : h*f belongs to I for every f
/// in J}. Empty and zero divisor ideals return the unit ideal. All original
/// inputs are admitted before these algebraic shortcuts can take effect.
pub fn checked_ideal_quotient(
    inputs: &[MultivariatePoly],
    divisors: &[MultivariatePoly],
    generators: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealQuotient, PolyError> {
    // Nested operation envelopes charge these outer meters as well. Neither
    // the count of divisors nor the number of intersections resets a phase.
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let count = admit(inputs, divisors, generators, &mut generation)?;
    generation.allocate(count * std::mem::size_of::<PrincipalIdealQuotientCertificate>())?;
    let mut principal_quotients = Vec::with_capacity(count);
    for (index, divisor) in divisors.iter().enumerate() {
        generation.tick(1)?;
        if divisor.is_zero() {
            continue;
        }
        let intersection = checked_ideal_intersection(
            inputs,
            std::slice::from_ref(divisor),
            generators,
            &mut generation,
            &mut verification,
        )?;
        generation.allocate(std::mem::size_of_val(intersection.basis()))?;
        let mut quotients = Vec::with_capacity(intersection.basis().len());
        for dividend in intersection.basis() {
            let (mut quotient, remainder) = divide_with_work(
                dividend,
                std::slice::from_ref(divisor),
                TermOrder::Lex,
                &mut generation,
            )?;
            if !remainder.is_zero() || quotient.len() != 1 {
                return Err(refused("intersection generator is not exactly divisible"));
            }
            quotients.push(
                quotient
                    .pop()
                    .ok_or_else(|| refused("missing principal quotient"))?,
            );
        }
        let quotient_basis = if quotients.is_empty() {
            let zero = constant(generators, false, &mut generation)?;
            checked_groebner_basis(
                std::slice::from_ref(&zero),
                TermOrder::Lex,
                &mut generation,
                &mut verification,
            )?
        } else {
            checked_groebner_basis(
                &quotients,
                TermOrder::Lex,
                &mut generation,
                &mut verification,
            )?
        };
        generation.allocate(
            std::mem::size_of::<VerifiedIdealOperation>()
                + std::mem::size_of::<VerifiedGroebnerBasis>()
                + 64,
        )?;
        principal_quotients.push(PrincipalIdealQuotientCertificate {
            divisor_index: index,
            intersection: Arc::new(intersection),
            quotients,
            quotient_basis: Arc::new(quotient_basis),
        });
    }
    generation
        .allocate(count.saturating_sub(1) * std::mem::size_of::<Arc<VerifiedIdealOperation>>())?;
    let mut intersections: Vec<Arc<VerifiedIdealOperation>> =
        Vec::with_capacity(count.saturating_sub(1));
    for principal in principal_quotients.iter().skip(1) {
        let left = intersections.last().map_or_else(
            || principal_quotients[0].quotient_basis.basis(),
            |previous| previous.basis(),
        );
        let intersection = checked_ideal_intersection(
            left,
            principal.quotient_basis.basis(),
            generators,
            &mut generation,
            &mut verification,
        )?;
        generation.allocate(std::mem::size_of::<VerifiedIdealOperation>() + 32)?;
        intersections.push(Arc::new(intersection));
    }
    let basis = if let Some(intersection) = intersections.last() {
        copy_polynomials(intersection.basis(), &mut generation)?
    } else if let Some(principal) = principal_quotients.first() {
        copy_polynomials(principal.quotient_basis.basis(), &mut generation)?
    } else {
        generation.allocate(std::mem::size_of::<MultivariatePoly>())?;
        vec![constant(generators, true, &mut generation)?]
    };
    generation.allocate(std::mem::size_of::<IdealQuotientCertificate>())?;
    let certificate = IdealQuotientCertificate {
        principal_quotients,
        intersections,
        basis,
    };
    let accepted = verify_ideal_quotient(
        inputs,
        divisors,
        generators,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
