//! Complete bounded GCD and LCM in an ordered polynomial ring over QQ.
//!
//! For nonconstant inputs, the complete principal-ideal intersection
//! `<f> intersect <g>` has one monic generator, the monic LCM. The composition
//! checker binds that immutable independently accepted intersection to both
//! exact original inputs, then checks `lc(f) lc(g) gcd lcm = f g` by separate
//! coefficient convolution. Divisibility or a product identity alone cannot
//! establish greatestness. No factorization of either input is required.
//!
//! Zero and scalar cases have direct reference checks. The generator and
//! verifier retain separate cumulative 20M-compute/128MiB logical allocation
//! limits through nested operations and publication. Polynomial degree 256,
//! 4096 terms, 4096-bit rational components and eight variables are admitted;
//! nonconstant pairs need one auxiliary variable, leaving seven original
//! variables and the ambient ideal's degree/term bounds. Reference product
//! columns may have degree 512 and at most 16384 distinct monomials, with the
//! same rational arithmetic bounds. Exhaustion never publishes a result.
//! These are immutable in-process composition receipts, not portable proofs.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_DEGREE, MAX_VARIABLES, PolynomialWork, Terms, copy_poly, divide_with_work, refused,
    ring_storage, same_ring, validate_poly,
};
use crate::{
    IdealOperation, MultivariatePoly, PolyError, TermOrder, VerifiedIdealOperation,
    checked_ideal_intersection,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};
use std::sync::Arc;

const MAX_REFERENCE_COLUMNS: usize = 16_384;

/// Untrusted monic values and complete-intersection evidence. The latter is
/// required exactly when both original polynomials are nonconstant.
#[derive(Debug, Clone)]
pub struct MultivariateGcdCertificate {
    pub gcd: MultivariatePoly,
    pub lcm: MultivariatePoly,
    pub intersection: Option<Arc<VerifiedIdealOperation>>,
}

/// Complete monic GCD and LCM for the exact ordered typed QQ inputs.
#[derive(Debug, Clone)]
pub struct VerifiedMultivariateGcd {
    left: MultivariatePoly,
    right: MultivariatePoly,
    certificate: MultivariateGcdCertificate,
}

impl VerifiedMultivariateGcd {
    pub fn left(&self) -> &MultivariatePoly {
        &self.left
    }

    pub fn right(&self) -> &MultivariatePoly {
        &self.right
    }

    pub fn gcd(&self) -> &MultivariatePoly {
        &self.certificate.gcd
    }

    pub fn lcm(&self) -> &MultivariatePoly {
        &self.certificate.lcm
    }

    pub fn certificate(&self) -> &MultivariateGcdCertificate {
        &self.certificate
    }

    pub fn into_parts(self) -> (MultivariatePoly, MultivariatePoly) {
        (self.certificate.gcd, self.certificate.lcm)
    }
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused("multivariate GCD/LCM ring exceeds eight variables"));
    }
    let mut names = 0usize;
    for (index, symbol) in generators.iter().enumerate() {
        names = names
            .checked_add(symbol.name.len())
            .filter(|&size| size <= 65_536)
            .ok_or_else(|| refused("multivariate GCD/LCM generator names exceed 65536 bytes"))?;
        work.tick(symbol.name.len() + index + 1)?;
        if generators[..index].contains(symbol) {
            return Err(refused("duplicate typed multivariate GCD/LCM generator"));
        }
    }
    Ok(())
}

fn check_polynomial(
    polynomial: &MultivariatePoly,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    // Bound names before diagnostics, comparisons, or any owned copy.
    check_ring(polynomial.generators(), work)?;
    same_ring(polynomial.generators(), generators)?;
    validate_poly(polynomial, work)
}

fn is_scalar(
    polynomial: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    work.tick(polynomial.terms().len() * (polynomial.generators().len() + 1))?;
    Ok(polynomial
        .terms()
        .keys()
        .all(|powers| powers.iter().all(|power| *power == 0)))
}

fn admit(
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    check_polynomial(left, left.generators(), work)?;
    check_polynomial(right, left.generators(), work)?;
    let nontrivial = !is_scalar(left, work)? && !is_scalar(right, work)?;
    if nontrivial && left.generators().len() == MAX_VARIABLES {
        return Err(refused(
            "nonconstant multivariate GCD/LCM needs one auxiliary variable",
        ));
    }
    Ok(nontrivial)
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

fn monic(
    polynomial: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let mut result = copy_poly(polynomial, work)?;
    work.tick(polynomial.terms().len() * polynomial.generators().len().max(1))?;
    if let Some(leading) = polynomial.leading_coeff(TermOrder::Lex) {
        for coefficient in result.terms.values_mut() {
            *coefficient = work.divide(coefficient, leading)?;
        }
    }
    Ok(result)
}

/// Reference Lex head selection, separate from the generator's term-order API.
fn reference_leading<'a>(
    polynomial: &'a MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<&'a BigRational>, PolyError> {
    work.tick(polynomial.terms().len() * polynomial.generators().len().max(1))?;
    Ok(polynomial
        .terms()
        .iter()
        .max_by(|(left, _), (right, _)| left.cmp(right))
        .map(|(_, coefficient)| coefficient))
}

fn verify_monic(
    polynomial: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if reference_leading(polynomial, work)?.is_none_or(|coefficient| !coefficient.is_one()) {
        return Err(refused("nonzero multivariate GCD/LCM must be monic"));
    }
    Ok(())
}

/// Exact coefficient binding, not equality of ideals or equality up to a unit.
fn same_polynomial(
    actual: &MultivariatePoly,
    expected: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_polynomial(actual, expected.generators(), work)?;
    if actual.terms().len() != expected.terms().len() {
        return Err(refused(
            "multivariate GCD/LCM witness changed term coverage",
        ));
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
            return Err(refused(
                "multivariate GCD/LCM witness changed exact original coefficients",
            ));
        }
    }
    Ok(())
}

fn verify_normalization(
    original: &MultivariatePoly,
    candidate: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let Some(leading) = reference_leading(original, work)? else {
        return if candidate.is_zero() {
            Ok(())
        } else {
            Err(refused("zero multivariate GCD/LCM normalization changed"))
        };
    };
    verify_monic(candidate, work)?;
    if original.terms().len() != candidate.terms().len() {
        return Err(refused(
            "multivariate GCD/LCM normalization changed term coverage",
        ));
    }
    for ((powers, value), (candidate_powers, candidate_value)) in
        original.terms().iter().zip(candidate.terms())
    {
        work.tick(powers.len() + 1)?;
        if powers != candidate_powers || work.multiply(leading, candidate_value)? != *value {
            return Err(refused(
                "multivariate GCD/LCM normalization changed coefficients",
            ));
        }
    }
    Ok(())
}

fn reference_accumulate(
    columns: &mut Terms,
    powers: Vec<u32>,
    coefficient: BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(powers.len() + 1)?;
    let value = if let Some(previous) = columns.get(&powers) {
        work.add(previous, &coefficient)?
    } else {
        coefficient
    };
    if value.is_zero() {
        columns.remove(&powers);
    } else {
        if !columns.contains_key(&powers) {
            if columns.len() >= MAX_REFERENCE_COLUMNS {
                return Err(refused("multivariate GCD/LCM reference column bound"));
            }
            work.allocate(256)?;
        }
        columns.insert(powers, value);
    }
    Ok(())
}

/// Separate coefficient convolution; never invokes polynomial multiplication,
/// division, simplification or either candidate generator.
fn reference_product(
    columns: &mut Terms,
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    scale: &BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    for (a, coefficient_a) in left.terms() {
        for (b, coefficient_b) in right.terms() {
            work.tick(a.len() + 1)?;
            work.allocate(a.len() * 4)?;
            let mut powers = Vec::with_capacity(a.len());
            let mut degree = 0u64;
            for (&a, &b) in a.iter().zip(b) {
                let exponent = a
                    .checked_add(b)
                    .ok_or_else(|| refused("multivariate GCD/LCM reference exponent bound"))?;
                degree += u64::from(exponent);
                powers.push(exponent);
            }
            if degree > 2 * MAX_DEGREE {
                return Err(refused("multivariate GCD/LCM reference degree bound"));
            }
            let product = work.multiply(coefficient_a, coefficient_b)?;
            let value = work.multiply(&product, scale)?;
            reference_accumulate(columns, powers, value, work)?;
        }
    }
    Ok(())
}

/// Verify complete monic GCD and LCM without repeating an ideal computation,
/// polynomial division, or the candidate generator's product operations.
pub fn verify_multivariate_gcd(
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    certificate: &MultivariateGcdCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedMultivariateGcd, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    let nontrivial = admit(left, right, &mut work)?;
    for candidate in [&certificate.gcd, &certificate.lcm] {
        check_polynomial(candidate, left.generators(), &mut work)?;
    }
    if certificate.intersection.is_some() != nontrivial {
        return Err(refused(
            "multivariate GCD/LCM intersection evidence is missing or extraneous",
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
        work.tick(left.generators().len() + 1)?;
        if !certificate.gcd.is_one() {
            return Err(refused("GCD with a nonzero scalar must be one"));
        }
        verify_normalization(original, &certificate.lcm, &mut work)?;
    } else {
        let intersection = certificate
            .intersection
            .as_ref()
            .ok_or_else(|| refused("missing intersection"))?;
        work.tick(1)?;
        if intersection.operation() != IdealOperation::Intersection
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
        verify_monic(&certificate.gcd, &mut work)?;
        verify_monic(&certificate.lcm, &mut work)?;
        let left_lead = reference_leading(left, &mut work)?
            .ok_or_else(|| refused("missing nonzero left input"))?;
        let right_lead = reference_leading(right, &mut work)?
            .ok_or_else(|| refused("missing nonzero right input"))?;
        let scale = work.multiply(left_lead, right_lead)?;
        work.allocate(std::mem::size_of::<Terms>())?;
        let mut columns = Terms::new();
        reference_product(&mut columns, left, right, &-BigRational::one(), &mut work)?;
        reference_product(
            &mut columns,
            &certificate.gcd,
            &certificate.lcm,
            &scale,
            &mut work,
        )?;
        if !columns.is_empty() {
            return Err(refused("multivariate GCD/LCM coefficient identity failed"));
        }
    }
    work.allocate(std::mem::size_of::<VerifiedMultivariateGcd>())?;
    let accepted = VerifiedMultivariateGcd {
        left: copy_poly(left, &mut work)?,
        right: copy_poly(right, &mut work)?,
        certificate: MultivariateGcdCertificate {
            gcd: copy_poly(&certificate.gcd, &mut work)?,
            lcm: copy_poly(&certificate.lcm, &mut work)?,
            intersection: certificate.intersection.as_ref().map(Arc::clone),
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

fn exact_quotient(
    numerator: &MultivariatePoly,
    denominator: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let (mut quotients, remainder) = divide_with_work(
        numerator,
        std::slice::from_ref(denominator),
        TermOrder::Lex,
        work,
    )?;
    if !remainder.is_zero() || quotients.len() != 1 {
        return Err(refused("multivariate GCD/LCM quotient is not exact"));
    }
    quotients.pop().ok_or_else(|| refused("missing quotient"))
}

/// Compute complete monic GCD/LCM with separate cumulative generation and
/// reference-verification meters. `gcd(0, 0) = lcm(0, 0) = 0`.
pub fn checked_multivariate_gcd(
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedMultivariateGcd, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let nontrivial = admit(left, right, &mut generation)?;
    let generators = left.generators();
    let (gcd, lcm, intersection) = if left.is_zero() || right.is_zero() {
        let original = if left.is_zero() { right } else { left };
        (
            monic(original, &mut generation)?,
            constant(generators, false, &mut generation)?,
            None,
        )
    } else if !nontrivial {
        let original = if is_scalar(left, &mut generation)? {
            right
        } else {
            left
        };
        (
            constant(generators, true, &mut generation)?,
            monic(original, &mut generation)?,
            None,
        )
    } else {
        let receipt = checked_ideal_intersection(
            std::slice::from_ref(left),
            std::slice::from_ref(right),
            generators,
            &mut generation,
            &mut verification,
        )?;
        let [lcm] = receipt.basis() else {
            return Err(refused(
                "nonzero principal intersection did not produce one generator",
            ));
        };
        // Two exact divisions avoid materializing the potentially twice-degree
        // product f*g as a generated polynomial; all outputs remain bounded.
        let complement = exact_quotient(lcm, right, &mut generation)?;
        let raw_gcd = exact_quotient(left, &complement, &mut generation)?;
        let gcd = monic(&raw_gcd, &mut generation)?;
        let lcm = copy_poly(lcm, &mut generation)?;
        generation.allocate(std::mem::size_of::<VerifiedIdealOperation>() + 32)?;
        (gcd, lcm, Some(Arc::new(receipt)))
    };
    generation.allocate(std::mem::size_of::<MultivariateGcdCertificate>())?;
    let certificate = MultivariateGcdCertificate {
        gcd,
        lcm,
        intersection,
    };
    let accepted = verify_multivariate_gcd(left, right, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
