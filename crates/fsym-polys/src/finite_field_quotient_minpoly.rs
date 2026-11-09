//! Complete annihilator ideals of elements in bounded prime-field quotients.
//!
//! The kernel of F_p[t] -> F_p[x]/I, t |-> f, is the elimination ideal
//! <I,t-f> intersect F_p[t]. Its complete reduced basis is either empty,
//! proving that f is transcendental, or one monic polynomial (1 for the zero
//! ring). No finite-dimensionality, radical or finite-point assumption is
//! made, and nilpotent multiplicities are retained.
//!
//! An independent composition checker binds every original coefficient and
//! the exact t-f relation to an immutable accepted elimination receipt. It
//! does not invoke candidate lifting, polynomial arithmetic or Groebner
//! search. These are in-process composition receipts, not portable kernel
//! proofs. Both phases retain cumulative 20M-step/128MiB limits through
//! nested elimination, output copying and final publication.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    MAX_BASIS, MAX_DEGREE, MAX_TERMS, MAX_VARIABLES, admit_ring, admit_terms, copied_polynomial,
    copy_generators, empty, same_ring,
};
use crate::{
    IdealOperation, PolyError, PrimeFieldMultivariatePolynomial as Polynomial,
    VerifiedPrimeFieldIdealOperation, checked_prime_field_elimination_ideal,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::sync::Arc;

/// Untrusted ascending canonical residues around an immutable elimination.
/// None asserts that the complete annihilator ideal is zero.
#[derive(Debug, Clone)]
pub struct PrimeFieldQuotientMinimalPolynomialCertificate {
    pub elimination: Arc<VerifiedPrimeFieldIdealOperation>,
    pub coefficients: Option<Vec<u64>>,
}

#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldQuotientMinimalPolynomial {
    generators: Vec<Symbol>,
    prime: u64,
    inputs: Vec<Polynomial>,
    element: Polynomial,
    parameter: Symbol,
    certificate: PrimeFieldQuotientMinimalPolynomialCertificate,
}

impl VerifiedPrimeFieldQuotientMinimalPolynomial {
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
    pub fn parameter(&self) -> &Symbol {
        &self.parameter
    }
    pub fn coefficients(&self) -> Option<&[u64]> {
        self.certificate.coefficients.as_deref()
    }
    pub fn into_coefficients(self) -> Option<Vec<u64>> {
        self.certificate.coefficients
    }
    pub fn certificate(&self) -> &PrimeFieldQuotientMinimalPolynomialCertificate {
        &self.certificate
    }
}

fn admit(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    parameter: &Symbol,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() >= MAX_VARIABLES || inputs.len() >= MAX_BASIS {
        return Err(refused(
            "prime-field quotient minimal polynomials support seven variables and 31 ideal generators",
        ));
    }
    // The new parameter is part of the same bounded ambient ring. Check all
    // name bytes before copying a Symbol or allocating the ambient ring.
    let mut name_bytes = parameter.name.len();
    for symbol in generators {
        name_bytes = name_bytes
            .checked_add(symbol.name.len())
            .ok_or_else(|| refused("prime-field minimal polynomial ring size overflow"))?;
    }
    if name_bytes > 65_536 {
        return Err(refused(
            "prime-field minimal polynomial variable names exceed 65536 bytes",
        ));
    }
    admit_ring(generators, prime, work)?;
    work.tick((name_bytes + 1) * (generators.len() + 1))?;
    if generators.contains(parameter) {
        return Err(refused(
            "prime-field minimal polynomial requires a fresh typed parameter",
        ));
    }
    // t-f has exactly one more term than f, even for f=0 or a constant.
    if element.terms().len() >= MAX_TERMS {
        return Err(refused(
            "prime-field minimal polynomial element must leave one term for t-f",
        ));
    }
    for value in inputs.iter().chain(std::iter::once(element)) {
        same_ring(value, generators, prime, work)?;
    }
    Ok(())
}

/// Generator-only lifting; the independent checker reads original terms.
fn lift(
    source: &Polynomial,
    ambient: &[Symbol],
    relation: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut terms = empty(work)?;
    for (original, coefficient) in source.terms() {
        work.tick(ambient.len() + 1)?;
        work.allocate(128 + ambient.len() * std::mem::size_of::<u32>())?;
        let mut powers = Vec::with_capacity(ambient.len());
        powers.extend_from_slice(original);
        powers.push(0);
        terms.insert(
            powers,
            if relation {
                source.prime() - *coefficient
            } else {
                *coefficient
            },
        );
    }
    if relation {
        work.tick(ambient.len() + 1)?;
        work.allocate(128 + ambient.len() * std::mem::size_of::<u32>())?;
        let mut powers = vec![0; ambient.len()];
        *powers
            .last_mut()
            .ok_or_else(|| refused("missing prime-field quotient parameter"))? = 1;
        terms.insert(powers, 1);
    }
    admit_terms(
        &terms,
        ambient.len(),
        source.prime(),
        MAX_DEGREE,
        false,
        work,
    )?;
    copied_polynomial(ambient, source.prime(), &terms, work)
}

fn bind_lift(
    source: &Polynomial,
    actual: &Polynomial,
    generators: &[Symbol],
    parameter: &Symbol,
    relation: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(
        actual
            .generators()
            .iter()
            .map(|symbol| symbol.name.len() + 1)
            .sum(),
    )?;
    if actual.prime() != source.prime()
        || actual.generators().len() != generators.len() + 1
        || actual.generators()[..generators.len()] != *generators
        || actual.generators().last() != Some(parameter)
    {
        return Err(refused(
            "prime-field minimal polynomial elimination rebound the ambient ring",
        ));
    }
    admit_terms(
        actual.terms(),
        generators.len() + 1,
        source.prime(),
        MAX_DEGREE,
        false,
        work,
    )?;
    if actual.terms().len() != source.terms().len() + usize::from(relation) {
        return Err(refused(
            "prime-field minimal polynomial elimination changed original term coverage",
        ));
    }
    let mut parameter_terms = 0;
    for (powers, coefficient) in actual.terms() {
        work.tick(powers.len() + 1)?;
        let original_powers = &powers[..generators.len()];
        if powers[generators.len()] != 0 {
            if !relation
                || powers[generators.len()] != 1
                || original_powers.iter().any(|&power| power != 0)
                || *coefficient != 1
            {
                return Err(refused(
                    "prime-field minimal polynomial relation is not exactly t-f",
                ));
            }
            parameter_terms += 1;
            continue;
        }
        let expected = source.terms().get(original_powers).ok_or_else(|| {
            refused("prime-field minimal polynomial elimination inserted an original term")
        })?;
        // Deliberately use an independent integer identity rather than the
        // generator's modular negation. Both residues are canonical nonzero.
        let agrees = if relation {
            i128::from(*coefficient) + i128::from(*expected) == i128::from(source.prime())
        } else {
            coefficient == expected
        };
        if !agrees {
            return Err(refused(
                "prime-field minimal polynomial elimination changed an original coefficient",
            ));
        }
    }
    if parameter_terms != usize::from(relation) {
        return Err(refused(
            "prime-field minimal polynomial elimination omitted the parameter relation",
        ));
    }
    Ok(())
}

/// Independently check exact composition with the complete elimination ideal.
/// The immutable lower receipt already establishes all ideal containments and
/// Buchberger pairs. This checker binds the original ring, field, ordered
/// inputs, element, parameter relation and complete monic conclusion.
pub fn verify_prime_field_quotient_minimal_polynomial(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    parameter: &Symbol,
    certificate: &PrimeFieldQuotientMinimalPolynomialCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldQuotientMinimalPolynomial, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, element, parameter, &mut work)?;
    let elimination = &certificate.elimination;
    if elimination.operation() != IdealOperation::Elimination
        || elimination.prime() != prime
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
            "prime-field minimal polynomial receipt changed the exact elimination request",
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
            same_ring(
                polynomial,
                std::slice::from_ref(parameter),
                prime,
                &mut work,
            )?;
            let degree = polynomial
                .terms()
                .keys()
                .map(|powers| powers[0])
                .max()
                .unwrap_or(0) as usize;
            if coefficients.is_empty()
                || coefficients.len() > MAX_DEGREE as usize + 1
                || coefficients.last() != Some(&1)
                || degree + 1 != coefficients.len()
            {
                return Err(refused(
                    "prime-field minimal polynomial output is not the complete monic generator",
                ));
            }
            let mut nonzero = 0;
            for (degree, coefficient) in coefficients.iter().enumerate() {
                work.tick(1)?;
                if *coefficient >= prime {
                    return Err(refused(
                        "prime-field minimal polynomial output has a noncanonical coefficient",
                    ));
                }
                let key = [degree as u32];
                if *coefficient == 0 {
                    if polynomial.terms().contains_key(key.as_slice()) {
                        return Err(refused(
                            "prime-field minimal polynomial output omitted a coefficient",
                        ));
                    }
                } else {
                    nonzero += 1;
                    if polynomial.terms().get(key.as_slice()) != Some(coefficient) {
                        return Err(refused(
                            "prime-field minimal polynomial output changed a coefficient",
                        ));
                    }
                }
            }
            if nonzero != polynomial.terms().len() {
                return Err(refused(
                    "prime-field minimal polynomial output changed term coverage",
                ));
            }
        }
        _ => {
            return Err(refused(
                "prime-field minimal polynomial conclusion disagrees with complete elimination",
            ));
        }
    }
    work.allocate(
        std::mem::size_of::<VerifiedPrimeFieldQuotientMinimalPolynomial>()
            + std::mem::size_of_val(inputs)
            + parameter.name.len(),
    )?;
    let inputs = inputs
        .iter()
        .map(|value| copied_polynomial(generators, prime, value.terms(), &mut work))
        .collect::<Result<Vec<_>, _>>()?;
    let element = copied_polynomial(generators, prime, element.terms(), &mut work)?;
    let coefficients = certificate
        .coefficients
        .as_ref()
        .map(|values| {
            work.tick(values.len())?;
            work.allocate(values.len() * std::mem::size_of::<u64>())?;
            Ok::<_, PolyError>(values.clone())
        })
        .transpose()?;
    let generators = copy_generators(generators, &mut work)?;
    work.tick(parameter.name.len() + 1)?;
    let accepted = VerifiedPrimeFieldQuotientMinimalPolynomial {
        generators,
        prime,
        inputs,
        element,
        parameter: parameter.clone(),
        certificate: PrimeFieldQuotientMinimalPolynomialCertificate {
            elimination: Arc::clone(elimination),
            coefficients,
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Compute the complete monic annihilator, or prove its absence, in F_p[x]/I.
/// The original ring may have at most seven variables and 31 ideal inputs;
/// t-f occupies one additional variable/input and at most 4096 sparse terms.
/// Every source and result has total degree at most 64, and p is a checked
/// prime in 2..=65521. Exhaustion/refusal never becomes a transcendence result.
pub fn checked_prime_field_quotient_minimal_polynomial(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    parameter: &Symbol,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldQuotientMinimalPolynomial, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(
        generators,
        prime,
        inputs,
        element,
        parameter,
        &mut generation,
    )?;
    let ambient_storage = (generators.len() + 1) * std::mem::size_of::<Symbol>()
        + generators
            .iter()
            .map(|symbol| symbol.name.len())
            .sum::<usize>()
        + parameter.name.len();
    generation.tick(ambient_storage)?;
    generation
        .allocate(ambient_storage + (inputs.len() + 1) * std::mem::size_of::<Polynomial>())?;
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
    let elimination = checked_prime_field_elimination_ideal(
        &ambient,
        prime,
        &augmented,
        generators,
        &mut generation,
        &mut verification,
    )?;
    let coefficients = match elimination.basis() {
        [] => None,
        [polynomial] => {
            let size = polynomial
                .terms()
                .keys()
                .map(|powers| powers[0])
                .max()
                .unwrap_or(0) as usize
                + 1;
            generation.tick(size)?;
            generation.allocate(size * std::mem::size_of::<u64>())?;
            let mut values = vec![0; size];
            for (powers, value) in polynomial.terms() {
                generation.tick(1)?;
                values[powers[0] as usize] = *value;
            }
            Some(values)
        }
        _ => {
            return Err(refused(
                "univariate prime-field elimination returned more than one reduced generator",
            ));
        }
    };
    generation.allocate(std::mem::size_of::<VerifiedPrimeFieldIdealOperation>() + 32)?;
    let certificate = PrimeFieldQuotientMinimalPolynomialCertificate {
        elimination: Arc::new(elimination),
        coefficients,
    };
    let accepted = verify_prime_field_quotient_minimal_polynomial(
        generators,
        prime,
        inputs,
        element,
        parameter,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
