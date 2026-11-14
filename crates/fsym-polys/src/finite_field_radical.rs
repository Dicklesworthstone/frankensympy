//! Complete checked radicals of bounded zero-dimensional ideals over F_p.
//!
//! Coordinate elimination gives univariate annihilators. Complete, independently
//! Rabin-checked factorization removes every multiplicity, including inseparable
//! p-th powers. The composition verifier reconstructs each square-free coordinate
//! polynomial h from the distinct accepted factors and checks h^k in the original
//! ideal using separate signed coefficient convolutions. An independently checked
//! reduced basis must have exactly the original source basis and every h as input.
//!
//! Thus I <= J <= radical(I), and J contains a separable univariate polynomial in
//! each coordinate. The quotient by those polynomials is a finite product of
//! separable fields; its quotients are reduced, so J = radical(I). No equations
//! x^p-x are added: geometric points over the algebraic closure are preserved.
//!
//! Inputs retain eight variables, 32 generators, degree64 and 4096 terms. The
//! augmented basis also has at most32 generators, and h^k has degree at most64.
//! All nested work and output copies share separate cumulative 20M-step/128MiB
//! generation and verification caps. These are immutable in-process receipts,
//! not a new portable proof family. Positive-dimensional nonconstant ideals refuse.

#![forbid(unsafe_code)]

use std::sync::Arc;

use crate::division::{PolynomialWork, refused};
use crate::finite_field::{multiply, zeros};
use crate::finite_field_multivariate::{
    Columns, MAX_BASIS, MAX_DEGREE, MAX_TERMS, admit_ring, admit_terms, constant,
    copied_polynomial, copy_terms, divide, empty, identity_zero, product, reference_add,
    reference_head, reference_product, same_ring,
};
use crate::{
    PolyError, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, PrimeFieldPolynomial, TermOrder,
    VerifiedPrimeFieldFactorization, VerifiedPrimeFieldGroebnerBasis,
    checked_prime_field_elimination_ideal, checked_prime_field_factorization,
    checked_prime_field_groebner_basis,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;

/// Untrusted coordinate and membership witnesses. The immutable factorization
/// supplies distinct monic irreducible factors; the verifier still binds their
/// exact product to `annihilator` and checks every original-ideal identity.
#[derive(Clone, Debug)]
pub struct PrimeFieldRadicalCoordinateWitness {
    pub annihilator: Polynomial,
    pub nilpotence_exponent: u32,
    /// annihilator^nilpotence_exponent = sum(q_j * source_basis[j]).
    pub nilpotent_quotients: Vec<Terms>,
    pub square_free_factorization: Arc<VerifiedPrimeFieldFactorization>,
}

/// Both accepted bases are immutable. Their exact source binding, coordinate
/// completeness and nilpotence remain obligations of this composition checker.
#[derive(Clone, Debug)]
pub struct PrimeFieldZeroDimensionalRadicalCertificate {
    pub source_basis: Arc<VerifiedPrimeFieldGroebnerBasis>,
    pub radical_basis: Arc<VerifiedPrimeFieldGroebnerBasis>,
    pub coordinates: Vec<PrimeFieldRadicalCoordinateWitness>,
}

#[derive(Clone, Debug)]
pub struct VerifiedPrimeFieldZeroDimensionalRadical {
    certificate: PrimeFieldZeroDimensionalRadicalCertificate,
}

impl VerifiedPrimeFieldZeroDimensionalRadical {
    pub fn generators(&self) -> &[Symbol] {
        self.certificate.source_basis.generators()
    }
    pub fn prime(&self) -> u64 {
        self.certificate.source_basis.prime()
    }
    pub fn inputs(&self) -> &[Polynomial] {
        self.certificate.source_basis.inputs()
    }
    pub fn source_basis(&self) -> &VerifiedPrimeFieldGroebnerBasis {
        &self.certificate.source_basis
    }
    pub fn basis(&self) -> &VerifiedPrimeFieldGroebnerBasis {
        &self.certificate.radical_basis
    }
    pub fn certificate(&self) -> &PrimeFieldZeroDimensionalRadicalCertificate {
        &self.certificate
    }
    /// Canonical reduced Lex bases distinguish the original ideal from its
    /// radical, even though their algebraic-closure point sets always agree.
    pub fn changed_ideal(&self) -> bool {
        self.source_basis().basis() != self.basis().basis()
    }
}

fn admit(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(generators, prime, work)?;
    if inputs.len() > MAX_BASIS {
        return Err(refused("prime-field radical exceeds 32 source polynomials"));
    }
    for input in inputs {
        same_ring(input, generators, prime, work)?;
    }
    Ok(())
}

fn unit_basis(
    source: &VerifiedPrimeFieldGroebnerBasis,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    work.tick(source.generators().len() + 1)?;
    Ok(source.basis().len() == 1
        && source.basis()[0].terms().len() == 1
        && source.basis()[0]
            .terms()
            .iter()
            .all(|(powers, coefficient)| {
                *coefficient == 1 && powers.iter().all(|power| *power == 0)
            }))
}

fn finite_coordinate_count(
    source: &VerifiedPrimeFieldGroebnerBasis,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<usize, PolyError> {
    if unit_basis(source, work)? {
        return Ok(0);
    }
    // A complete monomial ideal is zero-dimensional exactly when it contains
    // a pure power in every coordinate. The empty ring is the constant field.
    for coordinate in 0..source.generators().len() {
        let mut bounded = false;
        for polynomial in source.basis() {
            if let Some((powers, _)) = reference_head(polynomial.terms(), TermOrder::Lex, work)? {
                work.tick(powers.len())?;
                if powers[coordinate] != 0
                    && powers
                        .iter()
                        .enumerate()
                        .all(|(i, power)| i == coordinate || *power == 0)
                {
                    bounded = true;
                    break;
                }
            }
        }
        if !bounded {
            return Err(refused(
                "prime-field radical requires a zero-dimensional ideal",
            ));
        }
    }
    Ok(source.generators().len())
}

fn coordinate_degree(
    polynomial: &Polynomial,
    generators: &[Symbol],
    prime: u64,
    coordinate: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u32, PolyError> {
    same_ring(polynomial, generators, prime, work)?;
    let mut degree = 0;
    for powers in polynomial.terms().keys() {
        work.tick(powers.len())?;
        if powers
            .iter()
            .enumerate()
            .any(|(i, power)| i != coordinate && *power != 0)
        {
            return Err(refused("radical annihilator contains another coordinate"));
        }
        degree = degree.max(powers[coordinate]);
    }
    if degree == 0 {
        return Err(refused(
            "radical coordinate requires a nonconstant annihilator",
        ));
    }
    Ok(degree)
}

fn lift_coefficients(
    coefficients: &[u64],
    width: usize,
    coordinate: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (power, coefficient) in coefficients.iter().enumerate() {
        work.tick(1)?;
        if *coefficient != 0 {
            work.allocate(width * 4 + 128)?;
            let mut powers = vec![0; width];
            powers[coordinate] = u32::try_from(power).map_err(refused)?;
            result.insert(powers, *coefficient);
        }
    }
    Ok(result)
}

fn columns(work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Columns, PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    Ok(Columns::new())
}

/// Independent reference convolution, never the generator's modular product.
fn reference_multiply(
    left: &Terms,
    right: &Terms,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut values = columns(work)?;
    reference_product(&mut values, left, right, 1, work)?;
    let mut result = empty(work)?;
    for (powers, value) in values {
        work.tick(powers.len() + 1)?;
        if powers.iter().map(|power| u64::from(*power)).sum::<u64>() > MAX_DEGREE {
            return Err(refused("radical reference product exceeds degree64"));
        }
        let residue = value.rem_euclid(i128::from(prime)) as u64;
        if residue != 0 {
            if result.len() == MAX_TERMS {
                return Err(refused("radical reference product exceeds 4096 terms"));
            }
            work.allocate(128)?;
            result.insert(powers, residue);
        }
    }
    Ok(result)
}

fn verify_coordinate(
    witness: &PrimeFieldRadicalCoordinateWitness,
    source: &VerifiedPrimeFieldGroebnerBasis,
    coordinate: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let (generators, prime) = (source.generators(), source.prime());
    let degree = coordinate_degree(&witness.annihilator, generators, prime, coordinate, work)?;
    if witness.nilpotence_exponent == 0
        || u64::from(degree) * u64::from(witness.nilpotence_exponent) > MAX_DEGREE
        || witness.nilpotent_quotients.len() != source.basis().len()
    {
        return Err(refused(
            "radical nilpotence witness shape or powered degree",
        ));
    }
    for quotient in &witness.nilpotent_quotients {
        admit_terms(quotient, generators.len(), prime, MAX_DEGREE, false, work)?;
    }
    let factorization = &witness.square_free_factorization;
    work.tick(generators[coordinate].name.len() + 1)?;
    if factorization.variable() != &generators[coordinate]
        || factorization.prime() != prime
        || factorization.factors().is_empty()
    {
        return Err(refused(
            "radical square-free evidence changed typed coordinate or field",
        ));
    }
    let mut square_free = constant(generators.len(), 1, work)?;
    for factor in factorization.factors() {
        let lifted = lift_coefficients(&factor.coefficients, generators.len(), coordinate, work)?;
        square_free = reference_multiply(&square_free, &lifted, prime, work)?;
    }
    work.tick((square_free.len() + witness.annihilator.terms().len()) * (generators.len() + 1))?;
    if square_free != *witness.annihilator.terms() {
        return Err(refused(
            "radical coordinate is not the complete square-free factor product",
        ));
    }
    let mut powered = constant(generators.len(), 1, work)?;
    for _ in 0..witness.nilpotence_exponent {
        powered = reference_multiply(&powered, &square_free, prime, work)?;
    }
    let mut identity = columns(work)?;
    reference_add(&mut identity, &powered, 1, work)?;
    for (quotient, original) in witness.nilpotent_quotients.iter().zip(source.basis()) {
        reference_product(&mut identity, quotient, original.terms(), -1, work)?;
    }
    identity_zero(&identity, prime, work)
}

fn copy_witness(
    witness: &PrimeFieldRadicalCoordinateWitness,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldRadicalCoordinateWitness, PolyError> {
    let polynomial = &witness.annihilator;
    let annihilator = copied_polynomial(
        polynomial.generators(),
        polynomial.prime(),
        polynomial.terms(),
        work,
    )?;
    work.allocate(witness.nilpotent_quotients.len() * std::mem::size_of::<Terms>())?;
    let nilpotent_quotients = witness
        .nilpotent_quotients
        .iter()
        .map(|terms| copy_terms(terms, work))
        .collect::<Result<_, _>>()?;
    Ok(PrimeFieldRadicalCoordinateWitness {
        annihilator,
        nilpotence_exponent: witness.nilpotence_exponent,
        nilpotent_quotients,
        square_free_factorization: Arc::clone(&witness.square_free_factorization),
    })
}

/// Bind exact original sources, complete coordinate radicality and nilpotent
/// membership before returning an immutable accepted radical ideal.
pub fn verify_prime_field_zero_dimensional_radical(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    certificate: &PrimeFieldZeroDimensionalRadicalCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldZeroDimensionalRadical, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, &mut work)?;
    let (source, radical) = (&certificate.source_basis, &certificate.radical_basis);
    for basis in [source, radical] {
        work.tick(generators.len() + 1)?;
        if basis.generators() != generators
            || basis.prime() != prime
            || basis.order() != TermOrder::Lex
        {
            return Err(refused(
                "radical evidence changed its exact Lex polynomial ring",
            ));
        }
        admit(generators, prime, basis.inputs(), &mut work)?;
        admit(generators, prime, basis.basis(), &mut work)?;
    }
    work.tick(
        inputs
            .iter()
            .map(|p| p.terms().len() * (generators.len() + 1))
            .sum::<usize>()
            + 1,
    )?;
    if source.inputs() != inputs {
        return Err(refused(
            "radical evidence changed an original source polynomial",
        ));
    }
    let count = finite_coordinate_count(source, &mut work)?;
    if certificate.coordinates.len() != count
        || source.basis().len() + count > MAX_BASIS
        || radical.inputs().len() != source.basis().len() + count
        || radical.inputs()[..source.basis().len()] != *source.basis()
    {
        return Err(refused(
            "radical augmented ideal lost an original basis element or coordinate",
        ));
    }
    for (coordinate, witness) in certificate.coordinates.iter().enumerate() {
        if radical.inputs()[source.basis().len() + coordinate] != witness.annihilator {
            return Err(refused(
                "radical coordinate is not bound to the augmented ideal",
            ));
        }
        verify_coordinate(witness, source, coordinate, &mut work)?;
    }
    work.allocate(count * std::mem::size_of::<PrimeFieldRadicalCoordinateWitness>())?;
    let coordinates = certificate
        .coordinates
        .iter()
        .map(|witness| copy_witness(witness, &mut work))
        .collect::<Result<_, _>>()?;
    let result = VerifiedPrimeFieldZeroDimensionalRadical {
        certificate: PrimeFieldZeroDimensionalRadicalCertificate {
            source_basis: Arc::clone(source),
            radical_basis: Arc::clone(radical),
            coordinates,
        },
    };
    work.tick(1)?;
    Ok(result)
}

fn coordinate_candidate(
    source: &VerifiedPrimeFieldGroebnerBasis,
    coordinate: usize,
    generation: &mut PolynomialWork<'_, impl BudgetMeter>,
    verification: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldRadicalCoordinateWitness, PolyError> {
    let (generators, prime) = (source.generators(), source.prime());
    generation.allocate(
        generators
            .iter()
            .map(|s| std::mem::size_of::<Symbol>() + s.name.len())
            .sum(),
    )?;
    let eliminated: Vec<_> = generators
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != coordinate)
        .map(|(_, symbol)| symbol.clone())
        .collect();
    let projection = checked_prime_field_elimination_ideal(
        generators,
        prime,
        source.basis(),
        &eliminated,
        generation,
        verification,
    )?;
    if projection.basis().len() != 1
        || projection.generators() != &generators[coordinate..coordinate + 1]
    {
        return Err(refused(
            "finite coordinate elimination did not return one univariate annihilator",
        ));
    }
    let univariate = &projection.basis()[0];
    let degree = univariate
        .terms()
        .keys()
        .map(|powers| powers[0])
        .max()
        .unwrap_or(0);
    if degree == 0 {
        return Err(refused("finite coordinate annihilator has degree zero"));
    }
    let mut coefficients = zeros(degree as usize + 1, generation)?;
    for (powers, coefficient) in univariate.terms() {
        generation.tick(1)?;
        coefficients[powers[0] as usize] = *coefficient;
    }
    let polynomial = PrimeFieldPolynomial::metered_new(
        &generators[coordinate],
        prime,
        &coefficients,
        generation,
    )?;
    let factorization = checked_prime_field_factorization(&polynomial, generation, verification)?;
    let mut square_free = zeros(1, generation)?;
    square_free[0] = 1;
    let mut exponent = 1;
    for factor in factorization.factors() {
        square_free = multiply(&square_free, &factor.coefficients, prime, generation)?;
        exponent = exponent.max(factor.multiplicity);
    }
    let terms = lift_coefficients(&square_free, generators.len(), coordinate, generation)?;
    let annihilator = copied_polynomial(generators, prime, &terms, generation)?;
    if (square_free.len() - 1) as u64 * u64::from(exponent) > MAX_DEGREE {
        return Err(refused("radical powered coordinate exceeds degree64"));
    }
    let mut power = constant(generators.len(), 1, generation)?;
    for _ in 0..exponent {
        power = product(&power, &terms, prime, generation)?;
    }
    generation.allocate(source.basis().len() * std::mem::size_of::<&Terms>())?;
    let divisors: Vec<_> = source.basis().iter().map(Polynomial::terms).collect();
    let membership = divide(&power, &divisors, TermOrder::Lex, prime, generation)?;
    if !membership.remainder.is_empty() {
        return Err(refused(
            "coordinate radical power is not in the source ideal",
        ));
    }
    generation.allocate(std::mem::size_of::<VerifiedPrimeFieldFactorization>() + 32)?;
    Ok(PrimeFieldRadicalCoordinateWitness {
        annihilator,
        nilpotence_exponent: exponent,
        nilpotent_quotients: membership.quotients,
        square_free_factorization: Arc::new(factorization),
    })
}

/// Compute the full radical over the algebraic closure, retaining the exact
/// typed F_p ring and all source-domain admission performed by the caller.
pub fn checked_prime_field_zero_dimensional_radical(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldZeroDimensionalRadical, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, &mut generation)?;
    let source = checked_prime_field_groebner_basis(
        generators,
        prime,
        inputs,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )?;
    let count = finite_coordinate_count(&source, &mut generation)?;
    if source.basis().len() + count > MAX_BASIS {
        return Err(refused("radical augmented ideal exceeds 32 generators"));
    }
    generation.allocate(count * std::mem::size_of::<PrimeFieldRadicalCoordinateWitness>())?;
    let mut coordinates = Vec::with_capacity(count);
    for coordinate in 0..count {
        coordinates.push(coordinate_candidate(
            &source,
            coordinate,
            &mut generation,
            &mut verification,
        )?);
    }
    generation.allocate((source.basis().len() + count) * std::mem::size_of::<Polynomial>())?;
    let mut augmented = Vec::with_capacity(source.basis().len() + count);
    for polynomial in source
        .basis()
        .iter()
        .chain(coordinates.iter().map(|w| &w.annihilator))
    {
        augmented.push(copied_polynomial(
            generators,
            prime,
            polynomial.terms(),
            &mut generation,
        )?);
    }
    let radical = checked_prime_field_groebner_basis(
        generators,
        prime,
        &augmented,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )?;
    generation.allocate(2 * (std::mem::size_of::<VerifiedPrimeFieldGroebnerBasis>() + 32))?;
    let certificate = PrimeFieldZeroDimensionalRadicalCertificate {
        source_basis: Arc::new(source),
        radical_basis: Arc::new(radical),
        coordinates,
    };
    let result = verify_prime_field_zero_dimensional_radical(
        generators,
        prime,
        inputs,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(result)
}
