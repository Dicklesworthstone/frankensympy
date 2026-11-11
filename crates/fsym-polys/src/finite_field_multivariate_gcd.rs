//! Complete bounded GCD and LCM in an ordered polynomial ring over F_p.
//!
//! For nonzero nonconstant inputs, the complete principal-ideal intersection
//! <f> intersect <g> is principal and its monic generator is lcm(f,g). Exact
//! division then proposes gcd(f,g). The independent composition checker binds
//! the accepted intersection to BOTH original polynomials and checks every
//! coefficient of lc(f) lc(g) gcd lcm = f g, together with monic normalization.
//! A common divisor or a product identity alone would not prove greatestness.
//!
//! Zero and scalar inputs have direct reference checks; gcd(0,0)=lcm(0,0)=0.
//! Nontrivial intersections need one auxiliary variable, so they admit up to
//! seven original variables. Zero/scalar cases retain all eight. Degree 64,
//! 4096-term, prime 65521 and separate cumulative 20M-step / 128MiB phase caps
//! remain unchanged. Incomplete or exhausted work never publishes a result.
//! These are immutable in-process composition receipts, not portable proofs.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field::scalar_power;
use crate::finite_field_multivariate::{
    Columns, MAX_DEGREE, MAX_VARIABLES, admit_ring, admit_terms, constant, copied_polynomial,
    divide, empty, identity_zero, leading, reference_add, reference_head, reference_product,
    same_ring, scaled,
};
use crate::{
    IdealOperation, PolyError, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder, VerifiedPrimeFieldIdealOperation,
    checked_prime_field_ideal_intersection,
};
use fsym_budget::BudgetMeter;
use std::sync::Arc;

/// Untrusted proposed monic values and an immutable complete-intersection
/// receipt. The latter is required exactly when both inputs are nonconstant.
#[derive(Debug, Clone)]
pub struct PrimeFieldMultivariateGcdCertificate {
    pub gcd: Polynomial,
    pub lcm: Polynomial,
    pub intersection: Option<Arc<VerifiedPrimeFieldIdealOperation>>,
}

/// Complete monic GCD/LCM bound to both original ordered typed field inputs.
#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldMultivariateGcd {
    left: Polynomial,
    right: Polynomial,
    certificate: PrimeFieldMultivariateGcdCertificate,
}

impl VerifiedPrimeFieldMultivariateGcd {
    pub fn left(&self) -> &Polynomial {
        &self.left
    }
    pub fn right(&self) -> &Polynomial {
        &self.right
    }
    pub fn gcd(&self) -> &Polynomial {
        &self.certificate.gcd
    }
    pub fn lcm(&self) -> &Polynomial {
        &self.certificate.lcm
    }
    pub fn certificate(&self) -> &PrimeFieldMultivariateGcdCertificate {
        &self.certificate
    }
    pub fn into_parts(self) -> (Polynomial, Polynomial) {
        (self.certificate.gcd, self.certificate.lcm)
    }
}

fn is_scalar(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    work.tick(polynomial.terms().len() * (polynomial.generators().len() + 1))?;
    Ok(polynomial
        .terms()
        .keys()
        .all(|powers| powers.iter().all(|exponent| *exponent == 0)))
}

fn admit(
    left: &Polynomial,
    right: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    admit_ring(left.generators(), left.prime(), work)?;
    same_ring(left, left.generators(), left.prime(), work)?;
    same_ring(right, left.generators(), left.prime(), work)?;
    let nontrivial = !is_scalar(left, work)? && !is_scalar(right, work)?;
    if nontrivial && left.generators().len() == MAX_VARIABLES {
        return Err(refused(
            "nonconstant prime-field GCD/LCM needs one auxiliary variable",
        ));
    }
    Ok(nontrivial)
}

fn copy(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    copied_polynomial(
        polynomial.generators(),
        polynomial.prime(),
        polynomial.terms(),
        work,
    )
}

fn monic(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let prime = polynomial.prime();
    let coefficient = leading(polynomial.terms(), TermOrder::Lex, work)?.map_or(1, |(_, c)| c);
    let inverse = scalar_power(coefficient, prime - 2, prime, work)?;
    let terms = scaled(polynomial.terms(), inverse, prime, work)?;
    copied_polynomial(polynomial.generators(), prime, &terms, work)
}

fn columns(work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Columns, PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    Ok(Columns::new())
}

/// Exact source binding, not equality of ideals or equality up to a unit.
fn same_polynomial(
    actual: &Polynomial,
    expected: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    same_ring(actual, expected.generators(), expected.prime(), work)?;
    work.tick((actual.terms().len() + expected.terms().len()) * (expected.generators().len() + 1))?;
    if actual.terms() != expected.terms() {
        return Err(refused(
            "prime-field GCD/LCM witness changed original coefficients",
        ));
    }
    Ok(())
}

fn verify_monic(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if reference_head(polynomial.terms(), TermOrder::Lex, work)?.is_none_or(|(_, c)| c != 1) {
        return Err(refused("nonzero prime-field GCD/LCM must be monic"));
    }
    Ok(())
}

fn verify_normalization(
    original: &Polynomial,
    candidate: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let Some((_, coefficient)) = reference_head(original.terms(), TermOrder::Lex, work)? else {
        if !candidate.is_zero() {
            return Err(refused("zero prime-field GCD/LCM normalization changed"));
        }
        return Ok(());
    };
    verify_monic(candidate, work)?;
    let mut identity = columns(work)?;
    reference_add(
        &mut identity,
        candidate.terms(),
        i128::from(coefficient),
        work,
    )?;
    reference_add(&mut identity, original.terms(), -1, work)?;
    identity_zero(&identity, original.prime(), work)
}

/// Check a complete GCD/LCM proposal without generating an intersection,
/// performing polynomial division or calling the generator's modular product.
pub fn verify_prime_field_multivariate_gcd(
    left: &Polynomial,
    right: &Polynomial,
    certificate: &PrimeFieldMultivariateGcdCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateGcd, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    let nontrivial = admit(left, right, &mut work)?;
    let prime = left.prime();
    for candidate in [&certificate.gcd, &certificate.lcm] {
        same_ring(candidate, left.generators(), prime, &mut work)?;
    }
    if certificate.intersection.is_some() != nontrivial {
        return Err(refused(
            "prime-field GCD/LCM intersection evidence is missing or extraneous",
        ));
    }
    if left.is_zero() || right.is_zero() {
        if !certificate.lcm.is_zero() {
            return Err(refused("LCM with a zero input must be zero"));
        }
        let original = if left.is_zero() { right } else { left };
        verify_normalization(original, &certificate.gcd, &mut work)?;
    } else if !nontrivial {
        let original = if is_scalar(left, &mut work)? {
            right
        } else {
            left
        };
        if certificate.gcd.terms().len() != 1
            || !is_scalar(&certificate.gcd, &mut work)?
            || certificate.gcd.terms().values().next() != Some(&1)
        {
            return Err(refused("GCD with a nonzero scalar must be one"));
        }
        verify_normalization(original, &certificate.lcm, &mut work)?;
    } else {
        let intersection = certificate
            .intersection
            .as_ref()
            .ok_or_else(|| refused("missing intersection"))?;
        if intersection.operation() != IdealOperation::Intersection
            || intersection.prime() != prime
            || intersection.input_generators() != left.generators()
            || intersection.generators() != left.generators()
            || intersection.saturating_polynomial().is_some()
            || !intersection.eliminated_variables().is_empty()
            || intersection.inputs().len() != 1
            || intersection.other_inputs().len() != 1
            || intersection.basis().len() != 1
        {
            return Err(refused(
                "GCD/LCM needs the complete original principal intersection",
            ));
        }
        same_polynomial(&intersection.inputs()[0], left, &mut work)?;
        same_polynomial(&intersection.other_inputs()[0], right, &mut work)?;
        same_polynomial(&certificate.lcm, &intersection.basis()[0], &mut work)?;
        verify_monic(&certificate.lcm, &mut work)?;
        verify_monic(&certificate.gcd, &mut work)?;
        let (_, left_lead) = reference_head(left.terms(), TermOrder::Lex, &mut work)?
            .ok_or_else(|| refused("missing nonzero left input"))?;
        let (_, right_lead) = reference_head(right.terms(), TermOrder::Lex, &mut work)?
            .ok_or_else(|| refused("missing nonzero right input"))?;
        let mut identity = columns(&mut work)?;
        reference_product(&mut identity, left.terms(), right.terms(), -1, &mut work)?;
        reference_product(
            &mut identity,
            certificate.gcd.terms(),
            certificate.lcm.terms(),
            i128::from(left_lead) * i128::from(right_lead),
            &mut work,
        )?;
        identity_zero(&identity, prime, &mut work)?;
    }
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldMultivariateGcd>())?;
    let accepted = VerifiedPrimeFieldMultivariateGcd {
        left: copy(left, &mut work)?,
        right: copy(right, &mut work)?,
        certificate: PrimeFieldMultivariateGcdCertificate {
            gcd: copy(&certificate.gcd, &mut work)?,
            lcm: copy(&certificate.lcm, &mut work)?,
            intersection: certificate.intersection.as_ref().map(Arc::clone),
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

fn exact_quotient(
    numerator: &Terms,
    denominator: &Terms,
    prime: u64,
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut divided = divide(numerator, &[denominator], TermOrder::Lex, prime, work)?;
    if !divided.remainder.is_empty() || divided.quotients.len() != 1 {
        return Err(refused("prime-field GCD/LCM quotient is not exact"));
    }
    let quotient = divided
        .quotients
        .pop()
        .ok_or_else(|| refused("missing quotient"))?;
    admit_terms(&quotient, width, prime, MAX_DEGREE, false, work)?;
    Ok(quotient)
}

/// Compute a complete monic GCD and LCM with separate cumulative generation
/// and reference-verification meters, retained through every nested operation.
pub fn checked_prime_field_multivariate_gcd(
    left: &Polynomial,
    right: &Polynomial,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateGcd, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let nontrivial = admit(left, right, &mut generation)?;
    let prime = left.prime();
    let generators = left.generators();
    let (gcd, lcm, intersection) = if left.is_zero() || right.is_zero() {
        let original = if left.is_zero() { right } else { left };
        let zero = empty(&mut generation)?;
        (
            monic(original, &mut generation)?,
            copied_polynomial(generators, prime, &zero, &mut generation)?,
            None,
        )
    } else if !nontrivial {
        let original = if is_scalar(left, &mut generation)? {
            right
        } else {
            left
        };
        let one = constant(generators.len(), 1, &mut generation)?;
        (
            copied_polynomial(generators, prime, &one, &mut generation)?,
            monic(original, &mut generation)?,
            None,
        )
    } else {
        let receipt = checked_prime_field_ideal_intersection(
            generators,
            prime,
            std::slice::from_ref(left),
            std::slice::from_ref(right),
            &mut generation,
            &mut verification,
        )?;
        let [lcm] = receipt.basis() else {
            return Err(refused(
                "nonzero principal intersection did not produce one generator",
            ));
        };
        // Divide in two stages, avoiding the potentially twice-degree product
        // f*g as a generated polynomial. All intermediate results stay bounded.
        let complement = exact_quotient(
            lcm.terms(),
            right.terms(),
            prime,
            generators.len(),
            &mut generation,
        )?;
        let gcd = exact_quotient(
            left.terms(),
            &complement,
            prime,
            generators.len(),
            &mut generation,
        )?;
        let gcd = copied_polynomial(generators, prime, &gcd, &mut generation)?;
        let gcd = monic(&gcd, &mut generation)?;
        let lcm = copy(lcm, &mut generation)?;
        generation.allocate(std::mem::size_of::<VerifiedPrimeFieldIdealOperation>() + 32)?;
        (gcd, lcm, Some(Arc::new(receipt)))
    };
    generation.allocate(std::mem::size_of::<PrimeFieldMultivariateGcdCertificate>())?;
    let certificate = PrimeFieldMultivariateGcdCertificate {
        gcd,
        lcm,
        intersection,
    };
    let accepted =
        verify_prime_field_multivariate_gcd(left, right, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
