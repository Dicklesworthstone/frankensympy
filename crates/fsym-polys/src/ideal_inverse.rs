//! Checked inversion in a rational polynomial quotient ring.
//!
//! An element f is invertible modulo I exactly when the complete ideal
//! I + <f> contains 1. A unit-ideal cofactor supplies a Bezout identity; its
//! coefficient of f is reduced modulo a separately checked basis of I.
//! A proper complete augmented basis certifies noninvertibility, even in a
//! positive-dimensional quotient. A nonunit need not be a zero divisor.
//!
//! The independent composition checker binds immutable lower receipts to the
//! original ordered source and checks both identities by reference coefficient
//! convolution. It performs no Groebner search, division or multiplication.
//! These are in-process composition receipts, not new portable kernel claims.
//!
//! Both phases retain cumulative 20M-compute/128MiB logical allocation limits.
//! At most eight variables and 63 ideal generators are admitted, leaving one
//! generator for f. Other polynomial bounds are inherited from bounded division.
//! In the zero ring (I=<1>), the unique reduced inverse is zero, including f=0.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_BASIS, MAX_VARIABLES, PolynomialWork, copy_poly, divide_with_work, refused, ring_storage,
    same_ring, validate_poly,
};
use crate::{
    MultivariatePoly, PolyError, TermOrder, VerifiedGroebnerBasis, bounded_groebner_certificate,
    checked_groebner_basis, verify_bounded_division, verify_complete_groebner_certificate,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::sync::Arc;

/// Untrusted composition claim. The augmented receipt must be for the exact
/// original ideal generators followed by the element being inverted.
///
/// For a unit augmented ideal, `bezout_coefficients` expresses 1 in those
/// inputs, and `reduction_quotients` reduces its last coefficient to `inverse`
/// in `ideal_basis`. A proper augmented ideal requires all optional witnesses
/// absent and both coefficient lists empty.
#[derive(Debug, Clone)]
pub struct IdealInverseCertificate {
    pub augmented_basis: Arc<VerifiedGroebnerBasis>,
    pub ideal_basis: Option<Arc<VerifiedGroebnerBasis>>,
    pub bezout_coefficients: Vec<MultivariatePoly>,
    pub reduction_quotients: Vec<MultivariatePoly>,
    pub inverse: Option<MultivariatePoly>,
}

/// An accepted inverse, or an accepted complete noninvertibility conclusion.
/// Operational and resource failures are errors, never nonunit conclusions.
#[derive(Debug, Clone)]
pub struct VerifiedIdealInverse {
    generators: Vec<Symbol>,
    inputs: Vec<MultivariatePoly>,
    element: MultivariatePoly,
    order: TermOrder,
    certificate: IdealInverseCertificate,
}

impl VerifiedIdealInverse {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }

    pub fn inputs(&self) -> &[MultivariatePoly] {
        &self.inputs
    }

    pub fn element(&self) -> &MultivariatePoly {
        &self.element
    }

    pub fn order(&self) -> TermOrder {
        self.order
    }

    pub fn inverse(&self) -> Option<&MultivariatePoly> {
        self.certificate.inverse.as_ref()
    }

    pub fn into_inverse(self) -> Option<MultivariatePoly> {
        self.certificate.inverse
    }

    pub fn is_invertible(&self) -> bool {
        self.certificate.inverse.is_some()
    }

    pub fn certificate(&self) -> &IdealInverseCertificate {
        &self.certificate
    }
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused("ideal inverse ring exceeds eight variables"));
    }
    let mut names = 0usize;
    for (index, symbol) in generators.iter().enumerate() {
        names = names
            .checked_add(symbol.name.len())
            .filter(|&size| size <= 65_536)
            .ok_or_else(|| refused("ideal inverse generator names exceed 65536 bytes"))?;
        work.tick(symbol.name.len() + index + 1)?;
        if generators[..index].contains(symbol) {
            return Err(refused("duplicate typed ideal inverse generator"));
        }
    }
    Ok(())
}

fn check_polynomials(
    values: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if values.len() > MAX_BASIS {
        return Err(refused("ideal inverse witness exceeds 64 polynomials"));
    }
    for value in values {
        check_ring(value.generators(), work)?;
        same_ring(value.generators(), generators)?;
        validate_poly(value, work)?;
    }
    Ok(())
}

fn admit(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_ring(generators, work)?;
    if inputs.len() >= MAX_BASIS {
        return Err(refused(
            "ideal inverse supports at most 63 ideal generators",
        ));
    }
    check_polynomials(inputs, generators, work)?;
    check_polynomials(std::slice::from_ref(element), generators, work)
}

fn copy_polynomials(
    values: &[MultivariatePoly],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    work.allocate(std::mem::size_of_val(values))?;
    values.iter().map(|value| copy_poly(value, work)).collect()
}

fn augmented_inputs(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    work.allocate((inputs.len() + 1) * std::mem::size_of::<MultivariatePoly>())?;
    let mut augmented = Vec::with_capacity(inputs.len() + 1);
    for value in inputs.iter().chain(std::iter::once(element)) {
        augmented.push(copy_poly(value, work)?);
    }
    Ok(augmented)
}

fn constant(
    generators: &[Symbol],
    one: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    work.allocate(std::mem::size_of::<MultivariatePoly>() + ring_storage(generators) + 256)?;
    Ok(if one {
        MultivariatePoly::one(generators.to_vec())
    } else {
        MultivariatePoly::zero(generators.to_vec())
    })
}

fn bind_basis(
    basis: &VerifiedGroebnerBasis,
    expected: &[MultivariatePoly],
    generators: &[Symbol],
    order: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if basis.order() != order || basis.input_basis().len() != expected.len() {
        return Err(refused(
            "ideal inverse receipt changed its order or source coverage",
        ));
    }
    check_polynomials(basis.input_basis(), generators, work)?;
    check_polynomials(basis.basis(), generators, work)?;
    for (actual, expected) in basis.input_basis().iter().zip(expected) {
        if actual.terms().len() != expected.terms().len() {
            return Err(refused("ideal inverse receipt changed its original source"));
        }
        for ((powers, coefficient), (expected_powers, expected_coefficient)) in
            actual.terms().iter().zip(expected.terms())
        {
            work.tick(
                powers.len()
                    + coefficient.numer().limb_count() as usize
                    + coefficient.denom().limb_count() as usize
                    + expected_coefficient.numer().limb_count() as usize
                    + expected_coefficient.denom().limb_count() as usize
                    + 1,
            )?;
            if powers != expected_powers || coefficient != expected_coefficient {
                return Err(refused(
                    "ideal inverse receipt changed its exact coefficients",
                ));
            }
        }
    }
    work.tick(1)
}

/// Independently check an inverse or complete nonunit conclusion. The only
/// accepted lower receipts have private fields and were independently checked
/// when constructed; the full original source is bound again here.
pub fn verify_ideal_inverse(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    order: TermOrder,
    certificate: &IdealInverseCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealInverse, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(inputs, element, generators, &mut work)?;
    let augmented = augmented_inputs(inputs, element, &mut work)?;
    bind_basis(
        &certificate.augmented_basis,
        &augmented,
        generators,
        order,
        &mut work,
    )?;
    let is_unit = certificate.augmented_basis.basis().len() == 1
        && certificate.augmented_basis.basis()[0].is_one();
    if is_unit {
        let ideal_basis = certificate
            .ideal_basis
            .as_ref()
            .ok_or_else(|| refused("ideal inverse is missing its original ideal receipt"))?;
        let inverse = certificate
            .inverse
            .as_ref()
            .ok_or_else(|| refused("a unit augmented ideal cannot certify noninvertibility"))?;
        let zero = constant(generators, false, &mut work)?;
        let original = if inputs.is_empty() {
            std::slice::from_ref(&zero)
        } else {
            inputs
        };
        bind_basis(ideal_basis, original, generators, order, &mut work)?;
        check_polynomials(&certificate.bezout_coefficients, generators, &mut work)?;
        check_polynomials(&certificate.reduction_quotients, generators, &mut work)?;
        check_polynomials(std::slice::from_ref(inverse), generators, &mut work)?;
        if certificate.bezout_coefficients.len() != augmented.len() {
            return Err(refused(
                "ideal inverse Bezout witness changed source coverage",
            ));
        }
        let one = constant(generators, true, &mut work)?;
        verify_bounded_division(
            &one,
            &augmented,
            &certificate.bezout_coefficients,
            &zero,
            order,
            &mut work,
        )?;
        let raw_inverse = certificate
            .bezout_coefficients
            .last()
            .ok_or_else(|| refused("ideal inverse has no element cofactor"))?;
        verify_bounded_division(
            raw_inverse,
            ideal_basis.basis(),
            &certificate.reduction_quotients,
            inverse,
            order,
            &mut work,
        )?;
    } else if certificate.ideal_basis.is_some()
        || certificate.inverse.is_some()
        || !certificate.bezout_coefficients.is_empty()
        || !certificate.reduction_quotients.is_empty()
    {
        return Err(refused(
            "a proper augmented ideal cannot certify an inverse",
        ));
    }

    work.allocate(std::mem::size_of::<VerifiedIdealInverse>() + ring_storage(generators))?;
    let accepted = VerifiedIdealInverse {
        generators: generators.to_vec(),
        inputs: copy_polynomials(inputs, &mut work)?,
        element: copy_poly(element, &mut work)?,
        order,
        certificate: IdealInverseCertificate {
            augmented_basis: Arc::clone(&certificate.augmented_basis),
            ideal_basis: certificate.ideal_basis.as_ref().map(Arc::clone),
            bezout_coefficients: copy_polynomials(&certificate.bezout_coefficients, &mut work)?,
            reduction_quotients: copy_polynomials(&certificate.reduction_quotients, &mut work)?,
            inverse: certificate
                .inverse
                .as_ref()
                .map(|value| copy_poly(value, &mut work))
                .transpose()?,
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Invert `element` modulo the ideal generated by `inputs`. A returned result
/// with no inverse proves noninvertibility using a complete proper augmented
/// ideal. Every original input is admitted before zero/unit shortcuts.
pub fn checked_ideal_inverse(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealInverse, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(inputs, element, generators, &mut generation)?;
    let augmented = augmented_inputs(inputs, element, &mut generation)?;
    let candidate = bounded_groebner_certificate(&augmented, order, &mut generation)?;
    let augmented_basis =
        verify_complete_groebner_certificate(&augmented, &candidate, &mut verification)?;
    generation.allocate(std::mem::size_of::<VerifiedGroebnerBasis>() + 32)?;
    let augmented_basis = Arc::new(augmented_basis);
    let (ideal_basis, bezout_coefficients, reduction_quotients, inverse) =
        if augmented_basis.basis().len() == 1 && augmented_basis.basis()[0].is_one() {
            let zero = constant(generators, false, &mut generation)?;
            let original = if inputs.is_empty() {
                std::slice::from_ref(&zero)
            } else {
                inputs
            };
            let ideal_basis =
                checked_groebner_basis(original, order, &mut generation, &mut verification)?;
            let row = candidate
                .candidate
                .input_ideal_witnesses
                .first()
                .ok_or_else(|| refused("unit augmented ideal has no Bezout witness"))?;
            let bezout = copy_polynomials(row, &mut generation)?;
            let raw_inverse = bezout
                .last()
                .ok_or_else(|| refused("unit augmented ideal has no element cofactor"))?;
            let (quotients, inverse) =
                divide_with_work(raw_inverse, ideal_basis.basis(), order, &mut generation)?;
            generation.allocate(std::mem::size_of::<VerifiedGroebnerBasis>() + 32)?;
            (
                Some(Arc::new(ideal_basis)),
                bezout,
                quotients,
                Some(inverse),
            )
        } else {
            (None, Vec::new(), Vec::new(), None)
        };
    generation.allocate(std::mem::size_of::<IdealInverseCertificate>())?;
    let certificate = IdealInverseCertificate {
        augmented_basis,
        ideal_basis,
        bezout_coefficients,
        reduction_quotients,
        inverse,
    };
    let accepted = verify_ideal_inverse(
        inputs,
        element,
        generators,
        order,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
