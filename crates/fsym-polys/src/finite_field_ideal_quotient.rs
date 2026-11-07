//! Complete bounded finite ideal quotients over F_p.
//!
//! For nonzero g, exact division of every generator of I intersect <g> by g
//! generates I:<g>. Cancellation is valid in the polynomial integral domain;
//! an arbitrary list of elements h with h*g in I would not prove completeness.
//! For J=<g_1,...,g_k>, I:J is the intersection of all principal quotients.
//! Zero polynomial divisors impose no condition, and I:<0> is the whole ring.
//! Only the finite multiplicity in J is removed; this is not saturation or a
//! set operation on F_p-rational points.
//!
//! The separate composition checker binds immutable accepted intersection and
//! Groebner receipts to every original typed coefficient and the exact prime.
//! Signed integer coefficient convolution checks each proposed cancellation.
//! Every principal quotient, every intersection link and the entire final
//! reduced Lex basis are required. The verifier does not generate intersections,
//! divide polynomials, multiply through generator helpers or search for bases.
//! These are immutable in-process composition receipts, not portable proofs.
//!
//! Existing bounds remain: eight variables, 32 numerator inputs and 32 divisor
//! generators, degree 64 and 4096 terms. Any nonzero divisor needs one auxiliary
//! variable and one extra intersection input, leaving at most seven original
//! variables and 31 numerator inputs. Augmented degrees and intermediate
//! intersections retain the existing bounds. Both complete phases keep separate
//! cumulative 20M-step / 128MiB logical-allocation caps through publication.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    Columns, MAX_BASIS, MAX_DEGREE, MAX_VARIABLES, admit_ring, admit_terms, constant,
    copied_polynomial, copy_generators, divide, identity_zero, reference_add, reference_product,
    same_ring,
};
use crate::{
    IdealOperation, PolyError, PrimeFieldMultivariatePolynomial as Polynomial, TermOrder,
    VerifiedPrimeFieldGroebnerBasis, VerifiedPrimeFieldIdealOperation,
    checked_prime_field_groebner_basis, checked_prime_field_ideal_intersection,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::sync::Arc;

/// Untrusted composition links for one nonzero original divisor. Every
/// intersection basis polynomial has exactly one quotient in the same order.
#[derive(Debug, Clone)]
pub struct PrimeFieldPrincipalIdealQuotientCertificate {
    pub divisor_index: usize,
    pub intersection: Arc<VerifiedPrimeFieldIdealOperation>,
    pub quotients: Vec<Polynomial>,
    pub quotient_basis: Arc<VerifiedPrimeFieldGroebnerBasis>,
}

/// Nonzero principal witnesses occur in original divisor order. The ith link
/// intersects the first i+1 principal quotients with the next one. With no
/// nonzero divisors there are no witnesses or links and the final basis is [1].
#[derive(Debug, Clone)]
pub struct PrimeFieldIdealQuotientCertificate {
    pub principal_quotients: Vec<PrimeFieldPrincipalIdealQuotientCertificate>,
    pub intersections: Vec<Arc<VerifiedPrimeFieldIdealOperation>>,
    pub basis: Vec<Polynomial>,
}

/// Complete reduced Lex basis of I:J in the exact ordered typed F_p ring.
/// Original generators and divisors are retained, including zero entries.
#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldIdealQuotient {
    generators: Vec<Symbol>,
    prime: u64,
    inputs: Vec<Polynomial>,
    divisors: Vec<Polynomial>,
    certificate: PrimeFieldIdealQuotientCertificate,
}

impl VerifiedPrimeFieldIdealQuotient {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }
    pub fn prime(&self) -> u64 {
        self.prime
    }
    pub fn inputs(&self) -> &[Polynomial] {
        &self.inputs
    }
    pub fn divisors(&self) -> &[Polynomial] {
        &self.divisors
    }
    pub fn basis(&self) -> &[Polynomial] {
        &self.certificate.basis
    }
    pub fn into_basis(self) -> Vec<Polynomial> {
        self.certificate.basis
    }
    pub fn certificate(&self) -> &PrimeFieldIdealQuotientCertificate {
        &self.certificate
    }
}

fn check_polynomials(
    polynomials: &[Polynomial],
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if polynomials.len() > MAX_BASIS {
        return Err(refused(
            "prime-field ideal quotient list exceeds 32 polynomials",
        ));
    }
    for polynomial in polynomials {
        same_ring(polynomial, generators, prime, work)?;
    }
    Ok(())
}

fn admit(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    divisors: &[Polynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<usize, PolyError> {
    admit_ring(generators, prime, work)?;
    // Even a unit numerator or an all-zero divisor list cannot discard an
    // unadmitted original polynomial or its exact ordered ring and prime.
    check_polynomials(inputs, generators, prime, work)?;
    check_polynomials(divisors, generators, prime, work)?;
    work.tick(divisors.len())?;
    let count = divisors
        .iter()
        .filter(|polynomial| !polynomial.is_zero())
        .count();
    if count != 0 && (generators.len() == MAX_VARIABLES || inputs.len() == MAX_BASIS) {
        return Err(refused(
            "nonzero prime-field quotient requires an auxiliary variable and one extra input",
        ));
    }
    Ok(count)
}

/// Exact coefficient binding, not ideal equality: a scaled or reordered
/// original input cannot silently replace the requested source polynomial.
fn same_polynomials(
    actual: &[Polynomial],
    expected: &[Polynomial],
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if actual.len() != expected.len() {
        return Err(refused(
            "prime-field quotient witness changed polynomial coverage",
        ));
    }
    check_polynomials(actual, generators, prime, work)?;
    check_polynomials(expected, generators, prime, work)?;
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.terms().len() != expected.terms().len() {
            return Err(refused(
                "prime-field quotient witness changed term coverage",
            ));
        }
        for ((powers, value), (expected_powers, expected_value)) in
            actual.terms().iter().zip(expected.terms())
        {
            work.tick(powers.len() + 1)?;
            if powers != expected_powers || value != expected_value {
                return Err(refused(
                    "prime-field quotient witness changed original coefficients",
                ));
            }
        }
    }
    Ok(())
}

fn unit(
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let terms = constant(generators.len(), 1, work)?;
    copied_polynomial(generators, prime, &terms, work)
}

fn copy_polynomials(
    polynomials: &[Polynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Polynomial>, PolyError> {
    work.allocate(std::mem::size_of_val(polynomials))?;
    let mut copied = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        copied.push(copied_polynomial(
            polynomial.generators(),
            polynomial.prime(),
            polynomial.terms(),
            work,
        )?);
    }
    Ok(copied)
}

/// Only the independent lower layer constructs this immutable receipt. Bind
/// its whole exact request again before using its complete intersection claim.
fn bind_intersection(
    receipt: &VerifiedPrimeFieldIdealOperation,
    left: &[Polynomial],
    right: &[Polynomial],
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if receipt.operation() != IdealOperation::Intersection
        || receipt.prime() != prime
        || receipt.input_generators() != generators
        || receipt.generators() != generators
        || receipt.saturating_polynomial().is_some()
        || !receipt.eliminated_variables().is_empty()
    {
        return Err(refused(
            "prime-field quotient has an unrelated intersection receipt",
        ));
    }
    same_polynomials(receipt.inputs(), left, generators, prime, work)?;
    same_polynomials(receipt.other_inputs(), right, generators, prime, work)?;
    check_polynomials(receipt.basis(), generators, prime, work)
}

fn bind_quotient_basis(
    witness: &PrimeFieldPrincipalIdealQuotientCertificate,
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    let basis = witness.quotient_basis.as_ref();
    if basis.order() != TermOrder::Lex || basis.prime() != prime || basis.generators() != generators
    {
        return Err(refused(
            "prime-field quotient requires the exact typed Lex basis ring",
        ));
    }
    // Explicit ring metadata retains the zero ideal even with no quotients.
    same_polynomials(basis.inputs(), &witness.quotients, generators, prime, work)?;
    check_polynomials(basis.basis(), generators, prime, work)
}

/// Reference signed coefficient convolution; no generator product, reduction
/// or division routine participates in this exact cancellation check.
fn verify_cancellation(
    dividend: &Polynomial,
    divisor: &Polynomial,
    quotient: &Polynomial,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    let mut columns = Columns::new();
    reference_product(&mut columns, divisor.terms(), quotient.terms(), 1, work)?;
    reference_add(&mut columns, dividend.terms(), -1, work)?;
    identity_zero(&columns, prime, work)
}

/// Independently check every principal cancellation and complete intersection
/// link. A list of correct members, or a saturation receipt, is insufficient.
pub fn verify_prime_field_ideal_quotient(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    divisors: &[Polynomial],
    certificate: &PrimeFieldIdealQuotientCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealQuotient, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    let count = admit(generators, prime, inputs, divisors, &mut work)?;
    if certificate.principal_quotients.len() != count
        || certificate.intersections.len() != count.saturating_sub(1)
    {
        return Err(refused(
            "prime-field quotient lost or added a divisor or chain link",
        ));
    }
    check_polynomials(&certificate.basis, generators, prime, &mut work)?;
    for (witness, (index, divisor)) in certificate.principal_quotients.iter().zip(
        divisors
            .iter()
            .enumerate()
            .filter(|(_, polynomial)| !polynomial.is_zero()),
    ) {
        work.tick(1)?;
        if witness.divisor_index != index {
            return Err(refused(
                "prime-field quotient changed the original divisor order",
            ));
        }
        bind_intersection(
            &witness.intersection,
            inputs,
            std::slice::from_ref(divisor),
            generators,
            prime,
            &mut work,
        )?;
        if witness.quotients.len() != witness.intersection.basis().len() {
            return Err(refused(
                "prime-field quotient omitted an intersection generator",
            ));
        }
        check_polynomials(&witness.quotients, generators, prime, &mut work)?;
        for (dividend, quotient) in witness.intersection.basis().iter().zip(&witness.quotients) {
            verify_cancellation(dividend, divisor, quotient, prime, &mut work)?;
        }
        bind_quotient_basis(witness, generators, prime, &mut work)?;
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
        bind_intersection(intersection, left, right, generators, prime, &mut work)?;
    }
    let one;
    let expected = if let Some(intersection) = certificate.intersections.last() {
        intersection.basis()
    } else if let Some(principal) = certificate.principal_quotients.first() {
        principal.quotient_basis.basis()
    } else {
        one = unit(generators, prime, &mut work)?;
        std::slice::from_ref(&one)
    };
    same_polynomials(&certificate.basis, expected, generators, prime, &mut work)?;

    // Retained large ambient evidence is shared, while every untrusted
    // polynomial and original source list is bounded and copied on acceptance.
    work.allocate(
        std::mem::size_of::<VerifiedPrimeFieldIdealQuotient>()
            + count * std::mem::size_of::<PrimeFieldPrincipalIdealQuotientCertificate>()
            + std::mem::size_of_val(certificate.intersections.as_slice()),
    )?;
    let mut principal_quotients = Vec::with_capacity(count);
    for witness in &certificate.principal_quotients {
        work.tick(1)?;
        principal_quotients.push(PrimeFieldPrincipalIdealQuotientCertificate {
            divisor_index: witness.divisor_index,
            intersection: Arc::clone(&witness.intersection),
            quotients: copy_polynomials(&witness.quotients, &mut work)?,
            quotient_basis: Arc::clone(&witness.quotient_basis),
        });
    }
    work.tick(certificate.intersections.len())?;
    let accepted = VerifiedPrimeFieldIdealQuotient {
        generators: copy_generators(generators, &mut work)?,
        prime,
        inputs: copy_polynomials(inputs, &mut work)?,
        divisors: copy_polynomials(divisors, &mut work)?,
        certificate: PrimeFieldIdealQuotientCertificate {
            principal_quotients,
            intersections: certificate.intersections.clone(),
            basis: copy_polynomials(&certificate.basis, &mut work)?,
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Compute the complete finite ideal quotient I:J. Empty or zero J returns
/// the unit ideal, after all original typed sources have passed admission.
pub fn checked_prime_field_ideal_quotient(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    divisors: &[Polynomial],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealQuotient, PolyError> {
    // Every nested helper charges these same outer meters, so no phase cap is
    // reset between divisors, cancellation, normalization or final publication.
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let count = admit(generators, prime, inputs, divisors, &mut generation)?;
    generation
        .allocate(count * std::mem::size_of::<PrimeFieldPrincipalIdealQuotientCertificate>())?;
    let mut principal_quotients = Vec::with_capacity(count);
    for (index, divisor) in divisors.iter().enumerate() {
        generation.tick(1)?;
        if divisor.is_zero() {
            continue;
        }
        let intersection = checked_prime_field_ideal_intersection(
            generators,
            prime,
            inputs,
            std::slice::from_ref(divisor),
            &mut generation,
            &mut verification,
        )?;
        generation.allocate(std::mem::size_of_val(intersection.basis()))?;
        let mut quotients = Vec::with_capacity(intersection.basis().len());
        for dividend in intersection.basis() {
            let mut division = divide(
                dividend.terms(),
                &[divisor.terms()],
                TermOrder::Lex,
                prime,
                &mut generation,
            )?;
            if !division.remainder.is_empty() || division.quotients.len() != 1 {
                return Err(refused(
                    "prime-field intersection generator is not exactly divisible",
                ));
            }
            let quotient = division
                .quotients
                .pop()
                .ok_or_else(|| refused("missing principal quotient"))?;
            admit_terms(
                &quotient,
                generators.len(),
                prime,
                MAX_DEGREE,
                false,
                &mut generation,
            )?;
            quotients.push(copied_polynomial(
                generators,
                prime,
                &quotient,
                &mut generation,
            )?);
        }
        let quotient_basis = checked_prime_field_groebner_basis(
            generators,
            prime,
            &quotients,
            TermOrder::Lex,
            &mut generation,
            &mut verification,
        )?;
        generation.allocate(
            std::mem::size_of::<VerifiedPrimeFieldIdealOperation>()
                + std::mem::size_of::<VerifiedPrimeFieldGroebnerBasis>()
                + 64,
        )?;
        principal_quotients.push(PrimeFieldPrincipalIdealQuotientCertificate {
            divisor_index: index,
            intersection: Arc::new(intersection),
            quotients,
            quotient_basis: Arc::new(quotient_basis),
        });
    }
    generation.allocate(
        count.saturating_sub(1) * std::mem::size_of::<Arc<VerifiedPrimeFieldIdealOperation>>(),
    )?;
    let mut intersections: Vec<Arc<VerifiedPrimeFieldIdealOperation>> =
        Vec::with_capacity(count.saturating_sub(1));
    for principal in principal_quotients.iter().skip(1) {
        let left = intersections.last().map_or_else(
            || principal_quotients[0].quotient_basis.basis(),
            |previous| previous.basis(),
        );
        let intersection = checked_prime_field_ideal_intersection(
            generators,
            prime,
            left,
            principal.quotient_basis.basis(),
            &mut generation,
            &mut verification,
        )?;
        generation.allocate(std::mem::size_of::<VerifiedPrimeFieldIdealOperation>() + 32)?;
        intersections.push(Arc::new(intersection));
    }
    let basis = if let Some(intersection) = intersections.last() {
        copy_polynomials(intersection.basis(), &mut generation)?
    } else if let Some(principal) = principal_quotients.first() {
        copy_polynomials(principal.quotient_basis.basis(), &mut generation)?
    } else {
        generation.allocate(std::mem::size_of::<Polynomial>())?;
        vec![unit(generators, prime, &mut generation)?]
    };
    generation.allocate(std::mem::size_of::<PrimeFieldIdealQuotientCertificate>())?;
    let certificate = PrimeFieldIdealQuotientCertificate {
        principal_quotients,
        intersections,
        basis,
    };
    let accepted = verify_prime_field_ideal_quotient(
        generators,
        prime,
        inputs,
        divisors,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
