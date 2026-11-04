//! Checked radicals of bounded zero-dimensional ideals over QQ.
//!
//! For every coordinate, generation obtains a univariate annihilator in the
//! original ideal and proposes its square-free part h_i. The independent
//! composition checker requires h_i^k_i in I and a Bezout identity proving
//! gcd(h_i,h_i')=1. An accepted Groebner basis is bound to exactly
//! I + <h_1,...,h_n>. This gives I <= J <= radical(I); Seidenberg's lemma
//! makes J radical because it contains a square-free univariate polynomial
//! in every coordinate. Thus J = radical(I), preserving every complex point.
//!
//! These are immutable in-process composition receipts over independently
//! checked Groebner and division identities, not new portable kernel claims.
//! Separate cumulative generator/verifier meters cover all nested operations.
//! Coordinate degree and nilpotence exponent are at most 64; powered witness
//! degree remains at most 256. Positive-dimensional ideals refuse explicitly.

#![forbid(unsafe_code)]

use std::sync::Arc;

use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};

use crate::division::{
    MAX_BASIS, MAX_DEGREE, MAX_VARIABLES, PolynomialWork, Terms, accumulate, copy_poly,
    divide_with_work, multiply_polynomials, refused, ring_storage, same_ring, validate_poly,
};
use crate::{
    MultivariatePoly, PolyError, TermOrder, VerifiedGroebnerBasis, checked_elimination_ideal,
    checked_groebner_basis, verify_bounded_division,
};

const MAX_COORDINATE_DEGREE: u32 = 64;

/// Untrusted witnesses for one coordinate, in the original typed ambient ring.
#[derive(Debug, Clone)]
pub struct RadicalCoordinateWitness {
    pub annihilator: MultivariatePoly,
    pub nilpotence_exponent: u32,
    /// h^k = sum(nilpotent_quotients[j] * source_basis[j]).
    pub nilpotent_quotients: Vec<MultivariatePoly>,
    /// square_free_u*h + square_free_v*dh/dx = 1.
    pub square_free_u: MultivariatePoly,
    pub square_free_v: MultivariatePoly,
}

/// Exact original-ideal and augmented-ideal binding around accepted bases.
#[derive(Debug, Clone)]
pub struct ZeroDimensionalRadicalCertificate {
    pub source_basis: Arc<VerifiedGroebnerBasis>,
    pub radical_basis: Arc<VerifiedGroebnerBasis>,
    pub coordinates: Vec<RadicalCoordinateWitness>,
}

#[derive(Debug, Clone)]
pub struct VerifiedZeroDimensionalRadical {
    generators: Vec<Symbol>,
    certificate: ZeroDimensionalRadicalCertificate,
}

impl VerifiedZeroDimensionalRadical {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }

    pub fn inputs(&self) -> &[MultivariatePoly] {
        self.certificate.source_basis.input_basis()
    }

    pub fn source_basis(&self) -> &VerifiedGroebnerBasis {
        &self.certificate.source_basis
    }

    pub fn basis(&self) -> &VerifiedGroebnerBasis {
        &self.certificate.radical_basis
    }

    pub fn certificate(&self) -> &ZeroDimensionalRadicalCertificate {
        &self.certificate
    }

    /// Reduced Lex bases are unique, so this exact comparison detects whether
    /// removing nilpotents changed the original ideal, rather than its variety.
    pub fn changed_ideal(&self) -> bool {
        self.source_basis().basis() != self.basis().basis()
    }
}

fn unit_basis(basis: &VerifiedGroebnerBasis) -> bool {
    basis.basis().len() == 1 && basis.basis()[0].is_one()
}

fn admit(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if generators.is_empty() || generators.len() > MAX_VARIABLES || inputs.len() > MAX_BASIS {
        return Err(refused(
            "radical requires 1..8 variables and at most 64 inputs",
        ));
    }
    // Validate even an empty input's typed ring before dimension decisions.
    work.allocate(ring_storage(generators))?;
    validate_poly(&MultivariatePoly::zero(generators.to_vec()), work)?;
    for input in inputs {
        validate_poly(input, work)?;
        same_ring(input.generators(), generators)?;
    }
    Ok(())
}

fn constant(
    generators: &[Symbol],
    value: i64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    work.allocate(ring_storage(generators) + 256 + generators.len() * 4)?;
    let mut result = MultivariatePoly::zero(generators.to_vec());
    if value != 0 {
        result
            .terms
            .insert(vec![0; generators.len()], BigRational::from(value));
    }
    Ok(result)
}

fn subtract(
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    same_ring(left.generators(), right.generators())?;
    let mut result = copy_poly(left, work)?;
    for (powers, coefficient) in right.terms() {
        work.allocate(powers.len() * 4)?;
        let value = work.subtract(&BigRational::zero(), coefficient)?;
        accumulate(&mut result.terms, powers.clone(), value, work)?;
    }
    Ok(result)
}

fn derivative(
    polynomial: &MultivariatePoly,
    coordinate: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let mut result = constant(polynomial.generators(), 0, work)?;
    for (powers, coefficient) in polynomial.terms() {
        if powers[coordinate] != 0 {
            work.allocate(powers.len() * 4)?;
            let mut exponent = powers.clone();
            exponent[coordinate] -= 1;
            let value = work.multiply(
                coefficient,
                &BigRational::from(i64::from(powers[coordinate])),
            )?;
            accumulate(&mut result.terms, exponent, value, work)?;
        }
    }
    Ok(result)
}

/// Bounded generator Euclid, using the sparse division lane for every step.
fn bezout(
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(MultivariatePoly, MultivariatePoly, MultivariatePoly), PolyError> {
    let generators = left.generators();
    let (mut a, mut b) = (copy_poly(left, work)?, copy_poly(right, work)?);
    let (mut u, mut next_u) = (
        constant(generators, 1, work)?,
        constant(generators, 0, work)?,
    );
    let (mut v, mut next_v) = (
        constant(generators, 0, work)?,
        constant(generators, 1, work)?,
    );
    while !b.is_zero() {
        let (quotients, remainder) =
            divide_with_work(&a, std::slice::from_ref(&b), TermOrder::Lex, work)?;
        let product_u = multiply_polynomials(&quotients[0], &next_u, work)?;
        let product_v = multiply_polynomials(&quotients[0], &next_v, work)?;
        let new_u = subtract(&u, &product_u, work)?;
        let new_v = subtract(&v, &product_v, work)?;
        (a, b) = (b, remainder);
        (u, next_u) = (next_u, new_u);
        (v, next_v) = (next_v, new_v);
    }
    let leading = work.clone_value(
        a.leading_term(TermOrder::Lex)
            .ok_or_else(|| refused("zero Bezout gcd"))?
            .1,
    )?;
    for polynomial in [&mut a, &mut u, &mut v] {
        for value in polynomial.terms.values_mut() {
            *value = work.divide(value, &leading)?;
        }
    }
    Ok((a, u, v))
}

fn coordinate_degree(
    polynomial: &MultivariatePoly,
    generators: &[Symbol],
    coordinate: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u32, PolyError> {
    validate_poly(polynomial, work)?;
    same_ring(polynomial.generators(), generators)?;
    if polynomial.terms().keys().any(|powers| {
        powers
            .iter()
            .enumerate()
            .any(|(index, &power)| index != coordinate && power != 0)
    }) {
        return Err(refused("annihilator involves another coordinate"));
    }
    let degree = polynomial.degree_in(coordinate);
    if degree == 0
        || degree > MAX_COORDINATE_DEGREE
        || !polynomial
            .leading_term(TermOrder::Lex)
            .is_some_and(|(_, value)| value.is_one())
    {
        return Err(refused(
            "coordinate annihilator must be monic of degree 1..64",
        ));
    }
    Ok(degree)
}

/// Independent dense repeated convolution; generation uses sparse products.
fn reference_power(
    polynomial: &MultivariatePoly,
    coordinate: usize,
    exponent: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let degree = polynomial.degree_in(coordinate) as usize;
    if exponent == 0
        || exponent > MAX_COORDINATE_DEGREE
        || degree as u64 * u64::from(exponent) > MAX_DEGREE
    {
        return Err(refused(
            "nilpotence exponent or powered degree exceeds its bound",
        ));
    }
    work.allocate((degree + 1) * 128)?;
    let mut coefficients = vec![BigRational::zero(); degree + 1];
    for (powers, value) in polynomial.terms() {
        coefficients[powers[coordinate] as usize] = work.clone_value(value)?;
    }
    work.allocate(128)?;
    let mut result = vec![BigRational::one()];
    for _ in 0..exponent {
        work.tick(result.len() * coefficients.len())?;
        work.allocate((result.len() + degree) * 128)?;
        let mut next = vec![BigRational::zero(); result.len() + degree];
        for (i, left) in result.iter().enumerate() {
            for (j, right) in coefficients.iter().enumerate() {
                let product = work.multiply(left, right)?;
                next[i + j] = work.add(&next[i + j], &product)?;
            }
        }
        result = next;
    }
    let mut output = constant(polynomial.generators(), 0, work)?;
    for (power, value) in result.into_iter().enumerate() {
        if !value.is_zero() {
            work.allocate(polynomial.generators().len() * 4)?;
            let mut powers = vec![0; polynomial.generators().len()];
            powers[coordinate] = power as u32;
            accumulate(&mut output.terms, powers, value, work)?;
        }
    }
    Ok(output)
}

/// Check U*h + V*h' = 1 directly from coefficient indices. This neither
/// calls the generator derivative/Euclid nor trusts a reported gcd.
fn verify_square_free(
    witness: &RadicalCoordinateWitness,
    coordinate: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let mut terms = Terms::new();
    for (differentiate, coefficient_poly) in [
        (false, &witness.square_free_u),
        (true, &witness.square_free_v),
    ] {
        validate_poly(coefficient_poly, work)?;
        same_ring(
            coefficient_poly.generators(),
            witness.annihilator.generators(),
        )?;
        for (left, a) in coefficient_poly.terms() {
            for (right, b) in witness.annihilator.terms() {
                work.tick(left.len() + 1)?;
                if differentiate && right[coordinate] == 0 {
                    continue;
                }
                work.allocate(left.len() * 4)?;
                let mut powers: Vec<_> = left.iter().zip(right).map(|(x, y)| x + y).collect();
                let mut coefficient = work.multiply(a, b)?;
                if differentiate {
                    powers[coordinate] -= 1;
                    coefficient = work.multiply(
                        &coefficient,
                        &BigRational::from(i64::from(right[coordinate])),
                    )?;
                }
                accumulate(&mut terms, powers, coefficient, work)?;
            }
        }
    }
    if terms != constant(witness.annihilator.generators(), 1, work)?.terms {
        return Err(PolyError::IdentityCheckFailed(
            "coordinate square-free Bezout identity failed".into(),
        ));
    }
    Ok(())
}

fn copy_witness(
    witness: &RadicalCoordinateWitness,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RadicalCoordinateWitness, PolyError> {
    work.allocate(witness.nilpotent_quotients.len() * std::mem::size_of::<MultivariatePoly>())?;
    Ok(RadicalCoordinateWitness {
        annihilator: copy_poly(&witness.annihilator, work)?,
        nilpotence_exponent: witness.nilpotence_exponent,
        nilpotent_quotients: witness
            .nilpotent_quotients
            .iter()
            .map(|p| copy_poly(p, work))
            .collect::<Result<_, _>>()?,
        square_free_u: copy_poly(&witness.square_free_u, work)?,
        square_free_v: copy_poly(&witness.square_free_v, work)?,
    })
}

/// Independently establish equality with the radical of the exact supplied
/// ideal. Accepted basis receipts are immutable; all composition links and
/// untrusted polynomial identities are checked again against this request.
pub fn verify_zero_dimensional_radical(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    certificate: &ZeroDimensionalRadicalCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedZeroDimensionalRadical, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(inputs, generators, &mut work)?;
    let source = &certificate.source_basis;
    let radical = &certificate.radical_basis;
    if source.order() != TermOrder::Lex
        || radical.order() != TermOrder::Lex
        || source.input_basis().len() != inputs.len()
        || source.basis().len() > MAX_BASIS
        || radical.basis().len() > MAX_BASIS
        || source.input_basis() != inputs
    {
        return Err(refused(
            "radical source is not the exact requested Lex ideal",
        ));
    }
    for polynomial in source
        .basis()
        .iter()
        .chain(radical.basis())
        .chain(radical.input_basis())
    {
        validate_poly(polynomial, &mut work)?;
        same_ring(polynomial.generators(), generators)?;
    }
    let coordinate_count = if unit_basis(source) {
        0
    } else {
        generators.len()
    };
    if !unit_basis(source) && (source.basis().is_empty() || !source.is_zero_dimensional()) {
        return Err(refused(
            "radical currently requires a zero-dimensional ideal",
        ));
    }
    if certificate.coordinates.len() != coordinate_count
        || source.basis().len() + coordinate_count > MAX_BASIS
        || radical.input_basis().len() != source.basis().len() + coordinate_count
        || radical.input_basis()[..source.basis().len()] != *source.basis()
    {
        return Err(refused(
            "radical augmented ideal lost an original generator or coordinate",
        ));
    }
    let zero = constant(generators, 0, &mut work)?;
    for (coordinate, witness) in certificate.coordinates.iter().enumerate() {
        coordinate_degree(&witness.annihilator, generators, coordinate, &mut work)?;
        if radical.input_basis()[source.basis().len() + coordinate] != witness.annihilator
            || witness.nilpotent_quotients.len() != source.basis().len()
        {
            return Err(refused(
                "radical coordinate or nilpotence witnesses are not bound to their ideals",
            ));
        }
        verify_square_free(witness, coordinate, &mut work)?;
        let power = reference_power(
            &witness.annihilator,
            coordinate,
            witness.nilpotence_exponent,
            &mut work,
        )?;
        verify_bounded_division(
            &power,
            source.basis(),
            &witness.nilpotent_quotients,
            &zero,
            TermOrder::Lex,
            &mut work,
        )?;
    }
    work.allocate(
        ring_storage(generators)
            + certificate.coordinates.len() * std::mem::size_of::<RadicalCoordinateWitness>(),
    )?;
    let coordinates = certificate
        .coordinates
        .iter()
        .map(|witness| copy_witness(witness, &mut work))
        .collect::<Result<_, _>>()?;
    let result = VerifiedZeroDimensionalRadical {
        generators: generators.to_vec(),
        certificate: ZeroDimensionalRadicalCertificate {
            source_basis: Arc::clone(source),
            radical_basis: Arc::clone(radical),
            coordinates,
        },
    };
    work.tick(1)?;
    Ok(result)
}

fn coordinate_candidate(
    source: &VerifiedGroebnerBasis,
    generators: &[Symbol],
    coordinate: usize,
    generation: &mut PolynomialWork<'_, impl BudgetMeter>,
    verification: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RadicalCoordinateWitness, PolyError> {
    generation.allocate(ring_storage(generators))?;
    let eliminated: Vec<_> = generators
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != coordinate)
        .map(|(_, variable)| variable.clone())
        .collect();
    let projection = checked_elimination_ideal(
        source.basis(),
        generators,
        &eliminated,
        generation,
        verification,
    )?;
    if projection.basis().len() != 1 {
        return Err(refused(
            "finite coordinate elimination did not produce one annihilator",
        ));
    }
    let univariate = &projection.basis()[0];
    if univariate.generators() != &generators[coordinate..coordinate + 1] {
        return Err(refused("coordinate elimination changed its typed variable"));
    }
    let mut polynomial = constant(generators, 0, generation)?;
    for (power, coefficient) in univariate.terms() {
        generation.allocate(generators.len() * 4)?;
        let mut powers = vec![0; generators.len()];
        powers[coordinate] = power[0];
        let value = generation.clone_value(coefficient)?;
        accumulate(&mut polynomial.terms, powers, value, generation)?;
    }
    let degree = coordinate_degree(&polynomial, generators, coordinate, generation)?;
    let differentiated = derivative(&polynomial, coordinate, generation)?;
    let (gcd, _, _) = bezout(&polynomial, &differentiated, generation)?;
    let (mut quotients, remainder) =
        divide_with_work(&polynomial, &[gcd], TermOrder::Lex, generation)?;
    if !remainder.is_zero() {
        return Err(refused("square-free candidate is not an exact quotient"));
    }
    let annihilator = quotients.remove(0);
    let differentiated = derivative(&annihilator, coordinate, generation)?;
    let (gcd, square_free_u, square_free_v) = bezout(&annihilator, &differentiated, generation)?;
    if !gcd.is_one() {
        return Err(refused("candidate coordinate is not square-free"));
    }
    let mut power = constant(generators, 1, generation)?;
    for nilpotence_exponent in 1..=degree {
        power = multiply_polynomials(&power, &annihilator, generation)?;
        let (nilpotent_quotients, remainder) =
            divide_with_work(&power, source.basis(), TermOrder::Lex, generation)?;
        if remainder.is_zero() {
            return Ok(RadicalCoordinateWitness {
                annihilator,
                nilpotence_exponent,
                nilpotent_quotients,
                square_free_u,
                square_free_v,
            });
        }
    }
    Err(refused(
        "coordinate nilpotence witness search exhausted its degree bound",
    ))
}

/// Reuse a complete checked Lex source basis; no generator search is trusted as
/// proof of radicality. Nested elimination and basis work share the same caps.
pub fn checked_zero_dimensional_radical_from_basis(
    source: Arc<VerifiedGroebnerBasis>,
    generators: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedZeroDimensionalRadical, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(source.input_basis(), generators, &mut generation)?;
    if source.order() != TermOrder::Lex
        || (!unit_basis(&source) && (source.basis().is_empty() || !source.is_zero_dimensional()))
    {
        return Err(refused(
            "radical requires a complete zero-dimensional Lex basis",
        ));
    }
    let count = if unit_basis(&source) {
        0
    } else {
        generators.len()
    };
    if source.basis().len() + count > MAX_BASIS {
        return Err(refused("radical augmented basis exceeds 64 generators"));
    }
    generation.allocate(count * std::mem::size_of::<RadicalCoordinateWitness>())?;
    let mut coordinates = Vec::with_capacity(count);
    for coordinate in 0..count {
        coordinates.push(coordinate_candidate(
            &source,
            generators,
            coordinate,
            &mut generation,
            &mut verification,
        )?);
    }
    generation
        .allocate((source.basis().len() + count) * std::mem::size_of::<MultivariatePoly>())?;
    let mut augmented = source
        .basis()
        .iter()
        .map(|p| copy_poly(p, &mut generation))
        .collect::<Result<Vec<_>, _>>()?;
    for coordinate in &coordinates {
        augmented.push(copy_poly(&coordinate.annihilator, &mut generation)?);
    }
    let radical_basis = checked_groebner_basis(
        &augmented,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )?;
    generation.allocate(std::mem::size_of::<VerifiedGroebnerBasis>() + 64)?;
    let certificate = ZeroDimensionalRadicalCertificate {
        source_basis: source,
        radical_basis: Arc::new(radical_basis),
        coordinates,
    };
    let result = verify_zero_dimensional_radical(
        certificate.source_basis.input_basis(),
        generators,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(result)
}

pub fn checked_zero_dimensional_radical(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedZeroDimensionalRadical, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(inputs, generators, &mut generation)?;
    let source =
        checked_groebner_basis(inputs, TermOrder::Lex, &mut generation, &mut verification)?;
    generation.allocate(std::mem::size_of::<VerifiedGroebnerBasis>() + 64)?;
    checked_zero_dimensional_radical_from_basis(
        Arc::new(source),
        generators,
        &mut generation,
        &mut verification,
    )
}
