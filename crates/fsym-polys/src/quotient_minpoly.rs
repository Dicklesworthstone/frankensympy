//! Minimal polynomials of elements in arbitrary rational polynomial quotients.
//!
//! The kernel of QQ[t] -> QQ[x]/I, t |-> f, is the complete elimination ideal
//! <I,t-f> intersect QQ[t]. Its reduced basis is a single monic minimal
//! polynomial (1 in the zero ring), or empty, proving that no nonzero
//! polynomial annihilates f. The quotient need not be finite dimensional.
//!
//! An independent composition checker binds the immutable accepted elimination
//! to every original coefficient and the exact relation t-f, then checks the
//! entire output. Nilpotent multiplicities are retained: no square-free or
//! radical shortcut is valid. These are in-process composition receipts,
//! not new portable proof-kernel claims. Each cumulative phase is bounded by
//! 20M compute steps and 128MiB logical allocations, including publication.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_BASIS, MAX_VARIABLES, PolynomialWork, copy_poly, refused, ring_storage, same_ring,
    validate_poly,
};
use crate::{
    IdealOperation, MultivariatePoly, PolyError, VerifiedIdealOperation, checked_elimination_ideal,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Ascending untrusted coefficients around an immutable checked elimination.
/// None asserts that the complete annihilator ideal is zero.
#[derive(Debug, Clone)]
pub struct QuotientMinimalPolynomialCertificate {
    pub elimination: Arc<VerifiedIdealOperation>,
    pub coefficients: Option<Vec<BigRational>>,
}

#[derive(Debug, Clone)]
pub struct VerifiedQuotientMinimalPolynomial {
    generators: Vec<Symbol>,
    inputs: Vec<MultivariatePoly>,
    element: MultivariatePoly,
    parameter: Symbol,
    certificate: QuotientMinimalPolynomialCertificate,
}

impl VerifiedQuotientMinimalPolynomial {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }
    pub fn inputs(&self) -> &[MultivariatePoly] {
        &self.inputs
    }
    pub fn element(&self) -> &MultivariatePoly {
        &self.element
    }
    pub fn parameter(&self) -> &Symbol {
        &self.parameter
    }
    pub fn coefficients(&self) -> Option<&[BigRational]> {
        self.certificate.coefficients.as_deref()
    }
    pub fn into_coefficients(self) -> Option<Vec<BigRational>> {
        self.certificate.coefficients
    }
    pub fn certificate(&self) -> &QuotientMinimalPolynomialCertificate {
        &self.certificate
    }
}

fn admit(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    parameter: &Symbol,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() >= MAX_VARIABLES || inputs.len() >= MAX_BASIS {
        return Err(refused(
            "quotient minimal polynomials support seven variables and 63 ideal generators",
        ));
    }
    let mut bytes = parameter.name.len();
    for (index, symbol) in generators.iter().enumerate() {
        work.tick(index + symbol.name.len() + 1)?;
        bytes = bytes
            .checked_add(symbol.name.len())
            .ok_or_else(|| refused("quotient minimal polynomial ring size overflow"))?;
        if generators[..index].contains(symbol) || symbol == parameter {
            return Err(refused(
                "quotient minimal polynomial requires distinct typed variables",
            ));
        }
    }
    if bytes > 65_536 {
        return Err(refused(
            "quotient minimal polynomial variable names exceed 65536 bytes",
        ));
    }
    for value in inputs.iter().chain(std::iter::once(element)) {
        same_ring(value.generators(), generators)?;
        validate_poly(value, work)?;
    }
    Ok(())
}

/// Generator-only lifting; the reference lane reads coefficients directly.
fn lift(
    source: &MultivariatePoly,
    ambient: &[Symbol],
    relation: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    work.allocate(ring_storage(ambient) + source.terms().len() * (ambient.len() * 4 + 64) + 512)?;
    let mut terms = BTreeMap::new();
    for (powers, coefficient) in source.terms() {
        work.tick(powers.len() + 1)?;
        let mut powers = powers.clone();
        powers.push(0);
        let coefficient = work.clone_value(coefficient)?;
        terms.insert(powers, if relation { -coefficient } else { coefficient });
    }
    if relation {
        let mut powers = vec![0; ambient.len()];
        *powers
            .last_mut()
            .ok_or_else(|| refused("missing quotient parameter"))? = 1;
        terms.insert(powers, BigRational::one());
    }
    let value = MultivariatePoly::new(ambient.to_vec(), terms)?;
    validate_poly(&value, work)?;
    Ok(value)
}

fn bind_lift(
    source: &MultivariatePoly,
    actual: &MultivariatePoly,
    generators: &[Symbol],
    parameter: &Symbol,
    relation: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if actual.generators().len() != generators.len() + 1
        || actual.generators()[..generators.len()] != *generators
        || actual.generators().last() != Some(parameter)
    {
        return Err(refused(
            "minimal polynomial elimination rebound the ambient ring",
        ));
    }
    validate_poly(actual, work)?;
    if actual.terms().len() != source.terms().len() + usize::from(relation) {
        return Err(refused(
            "minimal polynomial elimination changed original term coverage",
        ));
    }
    let mut parameter_terms = 0;
    for (powers, coefficient) in actual.terms() {
        work.tick(
            powers.len()
                + coefficient.numer().limb_count() as usize
                + coefficient.denom().limb_count() as usize
                + 1,
        )?;
        let original_powers = &powers[..generators.len()];
        if powers[generators.len()] != 0 {
            if !relation
                || powers[generators.len()] != 1
                || original_powers.iter().any(|&power| power != 0)
                || !coefficient.is_one()
            {
                return Err(refused("minimal polynomial relation is not exactly t-f"));
            }
            parameter_terms += 1;
            continue;
        }
        let expected = source
            .terms()
            .get(original_powers)
            .ok_or_else(|| refused("minimal polynomial elimination inserted an original term"))?;
        let agrees = if relation {
            *coefficient == -work.clone_value(expected)?
        } else {
            coefficient == expected
        };
        if !agrees {
            return Err(refused(
                "minimal polynomial elimination changed an original coefficient",
            ));
        }
    }
    if parameter_terms != usize::from(relation) {
        return Err(refused(
            "minimal polynomial elimination omitted the parameter relation",
        ));
    }
    Ok(())
}

/// Check exact composition and the complete generator, including its absence.
/// No candidate lifting, Groebner search, division or polynomial multiplication
/// is called by this checker. The accepted lower receipt has private fields.
pub fn verify_quotient_minimal_polynomial(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    parameter: &Symbol,
    certificate: &QuotientMinimalPolynomialCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedQuotientMinimalPolynomial, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(inputs, element, generators, parameter, &mut work)?;
    let elimination = &certificate.elimination;
    if elimination.operation() != IdealOperation::Elimination
        || !elimination.other_inputs().is_empty()
        || elimination.saturating_polynomial().is_some()
        || elimination.eliminated_variables() != generators
        || elimination.generators() != std::slice::from_ref(parameter)
        || elimination.inputs().len() != inputs.len() + 1
        || elimination.input_generators().len() != generators.len() + 1
        || elimination.input_generators()[..generators.len()] != *generators
        || elimination.input_generators().last() != Some(parameter)
    {
        return Err(refused(
            "minimal polynomial receipt changed the exact elimination request",
        ));
    }
    for (index, source) in inputs.iter().chain(std::iter::once(element)).enumerate() {
        bind_lift(
            source,
            &elimination.inputs()[index],
            generators,
            parameter,
            index == inputs.len(),
            &mut work,
        )?;
    }
    match (elimination.basis(), certificate.coefficients.as_deref()) {
        ([], None) => {}
        ([polynomial], Some(coefficients)) => {
            validate_poly(polynomial, &mut work)?;
            same_ring(polynomial.generators(), std::slice::from_ref(parameter))?;
            if coefficients.is_empty()
                || coefficients.len() > 257
                || coefficients.last() != Some(&BigRational::one())
                || polynomial.degree_in(0) as usize + 1 != coefficients.len()
            {
                return Err(refused(
                    "minimal polynomial output is not the complete monic generator",
                ));
            }
            let mut nonzero = 0;
            for (degree, coefficient) in coefficients.iter().enumerate() {
                work.tick(1)?;
                work.clone_value(coefficient)?;
                let key = [degree as u32];
                if coefficient.is_zero() {
                    if polynomial.terms().contains_key(key.as_slice()) {
                        return Err(refused("minimal polynomial output omitted a coefficient"));
                    }
                } else {
                    nonzero += 1;
                    if polynomial.terms().get(key.as_slice()) != Some(coefficient) {
                        return Err(refused("minimal polynomial output changed a coefficient"));
                    }
                }
            }
            if nonzero != polynomial.terms().len() {
                return Err(refused("minimal polynomial output changed term coverage"));
            }
        }
        _ => {
            return Err(refused(
                "minimal polynomial conclusion disagrees with complete elimination",
            ));
        }
    }
    work.allocate(
        std::mem::size_of::<VerifiedQuotientMinimalPolynomial>()
            + ring_storage(generators)
            + parameter.name.len()
            + std::mem::size_of_val(inputs),
    )?;
    let inputs = inputs
        .iter()
        .map(|value| copy_poly(value, &mut work))
        .collect::<Result<Vec<_>, _>>()?;
    let element = copy_poly(element, &mut work)?;
    let coefficients = certificate
        .coefficients
        .as_ref()
        .map(|values| {
            work.allocate(values.len() * std::mem::size_of::<BigRational>())?;
            values
                .iter()
                .map(|value| work.clone_value(value))
                .collect::<Result<Vec<_>, PolyError>>()
        })
        .transpose()?;
    let result = VerifiedQuotientMinimalPolynomial {
        inputs,
        element,
        generators: generators.to_vec(),
        parameter: parameter.clone(),
        certificate: QuotientMinimalPolynomialCertificate {
            elimination: Arc::clone(elimination),
            coefficients,
        },
    };
    work.tick(1)?;
    Ok(result)
}

pub fn checked_quotient_minimal_polynomial(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    parameter: &Symbol,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedQuotientMinimalPolynomial, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(inputs, element, generators, parameter, &mut generation)?;
    generation.allocate(
        ring_storage(generators)
            + parameter.name.len()
            + 512
            + (inputs.len() + 1) * std::mem::size_of::<MultivariatePoly>(),
    )?;
    let ambient: Vec<_> = generators
        .iter()
        .chain(std::iter::once(parameter))
        .cloned()
        .collect();
    let mut augmented = Vec::with_capacity(inputs.len() + 1);
    for value in inputs {
        augmented.push(lift(value, &ambient, false, &mut generation)?);
    }
    augmented.push(lift(element, &ambient, true, &mut generation)?);
    let elimination = checked_elimination_ideal(
        &augmented,
        &ambient,
        generators,
        &mut generation,
        &mut verification,
    )?;
    let coefficients = match elimination.basis() {
        [] => None,
        [polynomial] => {
            let size = polynomial.degree_in(0) as usize + 1;
            generation.allocate(size * std::mem::size_of::<BigRational>())?;
            let mut values = vec![BigRational::zero(); size];
            for (powers, value) in polynomial.terms() {
                generation.tick(1)?;
                values[powers[0] as usize] = generation.clone_value(value)?;
            }
            Some(values)
        }
        _ => {
            return Err(refused(
                "univariate elimination returned more than one reduced generator",
            ));
        }
    };
    generation.allocate(std::mem::size_of::<VerifiedIdealOperation>())?;
    let certificate = QuotientMinimalPolynomialCertificate {
        elimination: Arc::new(elimination),
        coefficients,
    };
    let result = verify_quotient_minimal_polynomial(
        inputs,
        element,
        generators,
        parameter,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(result)
}
