//! Exact inverses and complete nonunit conclusions in polynomial quotients over F_p.
//!
//! The complete augmented ideal I + <f> is the unit ideal exactly when f is
//! invertible modulo I. Its tracked cofactors give a Bezout identity, and the
//! coefficient of f is reduced against a separately checked basis of I.
//! A proper augmented ideal proves noninvertibility without assuming finite
//! dimension or confusing nonunits with zero divisors.
//!
//! The independent composition verifier binds both immutable lower receipts to
//! the original typed field and ordered sources. Signed coefficient convolution
//! checks the Bezout and reduction identities, and an independent leading-term
//! scan checks that the published inverse is reduced. It invokes no basis
//! search, polynomial division or generator multiplication helper.
//!
//! At most eight variables and 31 original ideal generators leave room for f.
//! Other sparse-field bounds and separate cumulative 20M-step / 128MiB phase
//! budgets remain in force through publication. In the zero ring the unique
//! reduced inverse is zero, including for f=0. Receipts are in-process only.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    Columns, MAX_BASIS, MAX_DEGREE, admit_ring, admit_terms, constant, copied_polynomial,
    copy_generators, copy_terms, divide, identity_zero, reference_add, reference_head,
    reference_product, same_ring,
};
use crate::{
    PolyError, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder, VerifiedPrimeFieldGroebnerBasis,
    bounded_prime_field_groebner_certificate, checked_prime_field_groebner_basis,
    verify_prime_field_groebner_certificate,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::sync::Arc;

/// Untrusted composition witnesses around independently accepted complete bases.
/// Cofactor i multiplies original input i, with the element appended last.
#[derive(Clone, Debug)]
pub struct PrimeFieldIdealInverseCertificate {
    pub augmented_basis: Arc<VerifiedPrimeFieldGroebnerBasis>,
    pub ideal_basis: Option<Arc<VerifiedPrimeFieldGroebnerBasis>>,
    pub bezout_coefficients: Vec<Terms>,
    pub reduction_quotients: Vec<Terms>,
    pub inverse: Option<Polynomial>,
}

/// An immutable accepted inverse or a complete algebraic nonunit conclusion.
#[derive(Clone, Debug)]
pub struct VerifiedPrimeFieldIdealInverse {
    generators: Vec<Symbol>,
    prime: u64,
    inputs: Vec<Polynomial>,
    element: Polynomial,
    order: TermOrder,
    certificate: PrimeFieldIdealInverseCertificate,
}

impl VerifiedPrimeFieldIdealInverse {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }
    pub fn prime(&self) -> u64 {
        self.prime
    }
    pub fn inputs(&self) -> &[Polynomial] {
        &self.inputs
    }
    pub fn element(&self) -> &Polynomial {
        &self.element
    }
    pub fn order(&self) -> TermOrder {
        self.order
    }
    pub fn inverse(&self) -> Option<&Polynomial> {
        self.certificate.inverse.as_ref()
    }
    pub fn into_inverse(self) -> Option<Polynomial> {
        self.certificate.inverse
    }
    pub fn is_invertible(&self) -> bool {
        self.certificate.inverse.is_some()
    }
    pub fn certificate(&self) -> &PrimeFieldIdealInverseCertificate {
        &self.certificate
    }
}

fn admit(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(generators, prime, work)?;
    if inputs.len() >= MAX_BASIS {
        return Err(refused(
            "prime-field ideal inverse supports at most 31 ideal generators",
        ));
    }
    for value in inputs.iter().chain(std::iter::once(element)) {
        same_ring(value, generators, prime, work)?;
    }
    Ok(())
}

fn copy_polynomials(
    values: &[Polynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Polynomial>, PolyError> {
    work.allocate(std::mem::size_of_val(values))?;
    values
        .iter()
        .map(|value| copied_polynomial(value.generators(), value.prime(), value.terms(), work))
        .collect()
}

fn augmented_inputs(
    inputs: &[Polynomial],
    element: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Polynomial>, PolyError> {
    work.allocate((inputs.len() + 1) * std::mem::size_of::<Polynomial>())?;
    inputs
        .iter()
        .chain(std::iter::once(element))
        .map(|value| copied_polynomial(value.generators(), value.prime(), value.terms(), work))
        .collect()
}

fn copy_rows(
    values: &[Terms],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Terms>, PolyError> {
    work.allocate(std::mem::size_of_val(values))?;
    values.iter().map(|value| copy_terms(value, work)).collect()
}

fn bind_basis(
    basis: &VerifiedPrimeFieldGroebnerBasis,
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    order: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(generators.iter().map(|g| g.name.len() + 1).sum())?;
    if basis.generators() != generators
        || basis.prime() != prime
        || basis.order() != order
        || basis.inputs().len() != inputs.len()
    {
        return Err(refused(
            "prime-field inverse receipt changed the exact source ring, order or coverage",
        ));
    }
    for (actual, expected) in basis.inputs().iter().zip(inputs) {
        same_ring(actual, generators, prime, work)?;
        if actual.terms().len() != expected.terms().len() {
            return Err(refused(
                "prime-field inverse receipt changed original term coverage",
            ));
        }
        for ((powers, value), (expected_powers, expected_value)) in
            actual.terms().iter().zip(expected.terms())
        {
            work.tick(powers.len() + 1)?;
            if powers != expected_powers || value != expected_value {
                return Err(refused(
                    "prime-field inverse receipt changed original coefficients",
                ));
            }
        }
    }
    Ok(())
}

fn is_unit(basis: &VerifiedPrimeFieldGroebnerBasis) -> bool {
    basis.basis().len() == 1
        && basis.basis()[0].terms().len() == 1
        && basis.basis()[0]
            .terms()
            .iter()
            .any(|(powers, value)| *value == 1 && powers.iter().all(|power| *power == 0))
}

fn verify_reduction(
    basis: &VerifiedPrimeFieldGroebnerBasis,
    source: &Terms,
    quotients: &[Terms],
    inverse: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if quotients.len() != basis.basis().len() {
        return Err(refused("prime-field inverse reduction witness width"));
    }
    work.allocate(std::mem::size_of::<Columns>())?;
    let mut columns = Columns::new();
    reference_add(&mut columns, source, 1, work)?;
    reference_add(&mut columns, inverse.terms(), -1, work)?;
    for (quotient, divisor) in quotients.iter().zip(basis.basis()) {
        let (head, _) = reference_head(divisor.terms(), basis.order(), work)?
            .ok_or_else(|| refused("checked prime-field inverse basis has a zero divisor"))?;
        for powers in inverse.terms().keys() {
            work.tick(powers.len())?;
            if head.iter().zip(powers).all(|(a, b)| a <= b) {
                return Err(PolyError::IdentityCheckFailed(
                    "prime-field inverse is not reduced".into(),
                ));
            }
        }
        reference_product(&mut columns, quotient, divisor.terms(), -1, work)?;
    }
    identity_zero(&columns, basis.prime(), work)
}

/// Independently verify an inverse or a complete proper-ideal nonunit witness.
/// Operational failures remain errors, never conclusions of noninvertibility.
pub fn verify_prime_field_ideal_inverse(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
    certificate: &PrimeFieldIdealInverseCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealInverse, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, element, &mut work)?;
    let augmented = augmented_inputs(inputs, element, &mut work)?;
    bind_basis(
        &certificate.augmented_basis,
        generators,
        prime,
        &augmented,
        order,
        &mut work,
    )?;
    work.tick(generators.len() + 1)?;
    if is_unit(&certificate.augmented_basis) {
        let basis = certificate
            .ideal_basis
            .as_ref()
            .ok_or_else(|| refused("prime-field inverse is missing its original ideal receipt"))?;
        let inverse = certificate
            .inverse
            .as_ref()
            .ok_or_else(|| refused("a unit augmented ideal cannot certify a nonunit"))?;
        bind_basis(basis, generators, prime, inputs, order, &mut work)?;
        same_ring(inverse, generators, prime, &mut work)?;
        if certificate.bezout_coefficients.len() != augmented.len()
            || certificate.reduction_quotients.len() != basis.basis().len()
        {
            return Err(refused(
                "prime-field inverse composition witness dimensions",
            ));
        }
        for value in certificate
            .bezout_coefficients
            .iter()
            .chain(&certificate.reduction_quotients)
        {
            admit_terms(value, generators.len(), prime, MAX_DEGREE, false, &mut work)?;
        }
        let one = constant(generators.len(), 1, &mut work)?;
        work.allocate(std::mem::size_of::<Columns>())?;
        let mut columns = Columns::new();
        reference_add(&mut columns, &one, 1, &mut work)?;
        // Original zero generators keep their cofactor positions. Unlike a
        // division request, this linear-combination identity allows zeros.
        for (cofactor, source) in certificate.bezout_coefficients.iter().zip(&augmented) {
            reference_product(&mut columns, cofactor, source.terms(), -1, &mut work)?;
        }
        identity_zero(&columns, prime, &mut work)?;
        let raw_inverse = certificate
            .bezout_coefficients
            .last()
            .ok_or_else(|| refused("prime-field inverse is missing its element cofactor"))?;
        verify_reduction(
            basis,
            raw_inverse,
            &certificate.reduction_quotients,
            inverse,
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
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldIdealInverse>())?;
    let accepted = VerifiedPrimeFieldIdealInverse {
        generators: copy_generators(generators, &mut work)?,
        prime,
        inputs: copy_polynomials(inputs, &mut work)?,
        element: copied_polynomial(generators, prime, element.terms(), &mut work)?,
        order,
        certificate: PrimeFieldIdealInverseCertificate {
            augmented_basis: Arc::clone(&certificate.augmented_basis),
            ideal_basis: certificate.ideal_basis.as_ref().map(Arc::clone),
            bezout_coefficients: copy_rows(&certificate.bezout_coefficients, &mut work)?,
            reduction_quotients: copy_rows(&certificate.reduction_quotients, &mut work)?,
            inverse: certificate
                .inverse
                .as_ref()
                .map(|value| copied_polynomial(generators, prime, value.terms(), &mut work))
                .transpose()?,
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently check the reduced inverse or complete nonunit.
pub fn checked_prime_field_ideal_inverse(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealInverse, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, element, &mut generation)?;
    let augmented = augmented_inputs(inputs, element, &mut generation)?;
    let candidate = bounded_prime_field_groebner_certificate(
        generators,
        prime,
        &augmented,
        order,
        &mut generation,
    )?;
    let augmented_basis = verify_prime_field_groebner_certificate(
        generators,
        prime,
        &augmented,
        order,
        &candidate,
        &mut verification,
    )?;
    generation.allocate(std::mem::size_of::<VerifiedPrimeFieldGroebnerBasis>() + 32)?;
    let augmented_basis = Arc::new(augmented_basis);
    generation.tick(generators.len() + 1)?;
    let (ideal_basis, bezout_coefficients, reduction_quotients, inverse) =
        if is_unit(&augmented_basis) {
            let basis = checked_prime_field_groebner_basis(
                generators,
                prime,
                inputs,
                order,
                &mut generation,
                &mut verification,
            )?;
            let row = candidate
                .basis_in_input
                .first()
                .ok_or_else(|| refused("unit prime-field augmented ideal has no Bezout witness"))?;
            let bezout = copy_rows(row, &mut generation)?;
            let raw_inverse = bezout.last().ok_or_else(|| {
                refused("unit prime-field augmented ideal has no element cofactor")
            })?;
            generation.allocate(basis.basis().len() * std::mem::size_of::<&Terms>())?;
            let divisors: Vec<_> = basis.basis().iter().map(Polynomial::terms).collect();
            let reduced = divide(raw_inverse, &divisors, order, prime, &mut generation)?;
            let inverse =
                copied_polynomial(generators, prime, &reduced.remainder, &mut generation)?;
            generation.allocate(std::mem::size_of::<VerifiedPrimeFieldGroebnerBasis>() + 32)?;
            (
                Some(Arc::new(basis)),
                bezout,
                reduced.quotients,
                Some(inverse),
            )
        } else {
            (None, Vec::new(), Vec::new(), None)
        };
    generation.allocate(std::mem::size_of::<PrimeFieldIdealInverseCertificate>())?;
    let certificate = PrimeFieldIdealInverseCertificate {
        augmented_basis,
        ideal_basis,
        bezout_coefficients,
        reduction_quotients,
        inverse,
    };
    let accepted = verify_prime_field_ideal_inverse(
        generators,
        prime,
        inputs,
        element,
        order,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
