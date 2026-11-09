//! Complete bounded polynomial syzygies over the original prime-field ring.
//!
//! For the ordered input column F and a complete monic Groebner basis G, the
//! lower certificate supplies G=A*F, F=B*G and strict standard representations
//! for all S-pairs. The resulting Schreyer rows S generate all relations on G.
//! Hence I-B*A and S*A generate all relations on F: u*F=0 implies that u*B is a
//! relation on G, and u=u*(I-B*A)+(u*B)*A. Only identically zero rows are omitted.
//!
//! The independent checker rechecks the full untrusted Groebner certificate,
//! then forms every composition with signed integer coefficient convolution
//! and a separate leading-term comparator. It checks the exact ordered rows
//! modulo p as well as their zero dot products with the original inputs.
//! Merely returning some valid relations cannot establish completeness.
//!
//! Inputs retain zero, duplicate and scaled coordinates. The envelope is eight
//! typed variables, 32 inputs, degree 64, 4096 terms per output polynomial and
//! 16384 output polynomial cells. Temporary products may have degree 128. Both
//! phases retain cumulative 20M-step/128MiB budgets through publication. These
//! in-process receipts establish a complete generating set, not a minimal or
//! reduced module basis or a portable proof-kernel object.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    Columns, MAX_BASIS, MAX_DEGREE, accumulate, admit_ring, admit_terms, constant,
    copied_polynomial, identity_zero, leading, product, reference_add, reference_head,
    reference_product, same_ring,
};
use crate::{
    PolyError, PrimeFieldGroebnerCertificate, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder, VerifiedPrimeFieldGroebnerBasis,
    bounded_prime_field_groebner_certificate, verify_prime_field_groebner_certificate,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;

const MAX_OUTPUT_POLYNOMIALS: usize = 16_384;

/// Untrusted complete relation evidence on the exact original coordinates.
/// Nonzero I-B*A rows precede nonzero S*A rows in increasing S-pair order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldPolynomialSyzygyCertificate {
    pub groebner: PrimeFieldGroebnerCertificate,
    pub syzygies: Vec<Vec<Polynomial>>,
}

/// Every relation on the original input list is a polynomial combination of
/// these rows. The accepted lower basis retains the exact source ring/order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPrimeFieldPolynomialSyzygies {
    basis: VerifiedPrimeFieldGroebnerBasis,
    syzygies: Vec<Vec<Polynomial>>,
}

impl VerifiedPrimeFieldPolynomialSyzygies {
    pub fn generators(&self) -> &[Symbol] {
        self.basis.generators()
    }
    pub fn prime(&self) -> u64 {
        self.basis.prime()
    }
    pub fn inputs(&self) -> &[Polynomial] {
        self.basis.inputs()
    }
    pub fn order(&self) -> TermOrder {
        self.basis.order()
    }
    pub fn basis(&self) -> &VerifiedPrimeFieldGroebnerBasis {
        &self.basis
    }
    pub fn syzygies(&self) -> &[Vec<Polynomial>] {
        &self.syzygies
    }
    pub fn into_syzygies(self) -> Vec<Vec<Polynomial>> {
        self.syzygies
    }
}

fn reserve<T>(
    length: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<T>, PolyError> {
    let bytes = length
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| refused("prime-field syzygy allocation size overflow"))?;
    work.allocate(bytes)?;
    let mut result = Vec::new();
    result.try_reserve_exact(length).map_err(refused)?;
    Ok(result)
}

fn admit(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if inputs.len() > MAX_BASIS {
        return Err(refused("prime-field syzygies support at most 32 inputs"));
    }
    admit_ring(generators, prime, work)?;
    for input in inputs {
        same_ring(input, generators, prime, work)?;
    }
    Ok(())
}

/// Structural admission is not mathematical acceptance. The independent full
/// Groebner verifier must still establish monicity, both ideal inclusions and
/// a strict representation for every S-pair before composition is accepted.
fn preflight_certificate(
    input_count: usize,
    width: usize,
    prime: u64,
    certificate: &PrimeFieldGroebnerCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<usize, PolyError> {
    work.tick(1)?;
    let count = certificate.basis.len();
    if count > MAX_BASIS {
        return Err(refused("prime-field syzygy basis exceeds 32 polynomials"));
    }
    let pairs = count * count.saturating_sub(1) / 2;
    if certificate.basis_in_input.len() != count
        || certificate.input_in_basis.len() != input_count
        || certificate.s_pairs.len() != pairs
    {
        return Err(refused("prime-field syzygy Groebner witness dimensions"));
    }
    // Check all dimensions before traversing caller-owned coefficient maps.
    for row in &certificate.basis_in_input {
        work.tick(1)?;
        if row.len() != input_count {
            return Err(refused("prime-field syzygy basis-in-input width"));
        }
    }
    for row in &certificate.input_in_basis {
        work.tick(1)?;
        if row.len() != count {
            return Err(refused("prime-field syzygy input-in-basis width"));
        }
    }
    let mut index = 0;
    for left in 0..count {
        for right in left + 1..count {
            work.tick(1)?;
            let pair = &certificate.s_pairs[index];
            if (pair.left, pair.right) != (left, right) || pair.quotients.len() != count {
                return Err(refused(
                    "prime-field syzygy S-pair coverage, order or width",
                ));
            }
            index += 1;
        }
    }
    for polynomial in certificate
        .basis
        .iter()
        .chain(certificate.basis_in_input.iter().flatten())
        .chain(certificate.input_in_basis.iter().flatten())
        .chain(certificate.s_pairs.iter().flat_map(|pair| &pair.quotients))
    {
        admit_terms(polynomial, width, prime, MAX_DEGREE, false, work)?;
    }
    Ok(input_count + pairs)
}

fn generator_subtract_product(
    target: &mut Terms,
    left: &Terms,
    right: &Terms,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if !left.is_empty() && !right.is_empty() {
        let multiple = product(left, right, prime, work)?;
        for (powers, coefficient) in multiple {
            accumulate(target, powers, prime - coefficient, prime, work)?;
        }
    }
    Ok(())
}

fn generator_cell(
    terms: &Terms,
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    admit_terms(terms, generators.len(), prime, MAX_DEGREE, false, work)?;
    copied_polynomial(generators, prime, terms, work)
}

fn generator_keep_row(
    rows: &mut Vec<Vec<Polynomial>>,
    row: Vec<Polynomial>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(row.len())?;
    if row.iter().all(Polynomial::is_zero) {
        return Ok(());
    }
    if (rows.len() + 1) * row.len() > MAX_OUTPUT_POLYNOMIALS {
        return Err(refused(
            "prime-field syzygies exceed 16384 polynomial cells",
        ));
    }
    rows.push(row);
    Ok(())
}

fn generate_relations(
    generators: &[Symbol],
    prime: u64,
    input_count: usize,
    order: TermOrder,
    certificate: &PrimeFieldGroebnerCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Vec<Polynomial>>, PolyError> {
    let a = &certificate.basis_in_input;
    let capacity =
        (input_count + certificate.s_pairs.len()).min(MAX_OUTPUT_POLYNOMIALS / input_count.max(1));
    let mut result = reserve(capacity, work)?;
    for (index, coefficients) in certificate.input_in_basis.iter().enumerate() {
        let mut row = reserve(input_count, work)?;
        for coordinate in 0..input_count {
            let mut value = constant(generators.len(), u64::from(coordinate == index), work)?;
            for (coefficient, basis_row) in coefficients.iter().zip(a) {
                generator_subtract_product(
                    &mut value,
                    coefficient,
                    &basis_row[coordinate],
                    prime,
                    work,
                )?;
            }
            row.push(generator_cell(&value, generators, prime, work)?);
        }
        generator_keep_row(&mut result, row, work)?;
    }
    for pair in &certificate.s_pairs {
        let (left, lc) = leading(&certificate.basis[pair.left], order, work)?
            .ok_or_else(|| refused("zero prime-field syzygy S-pair generator"))?;
        let (right, rc) = leading(&certificate.basis[pair.right], order, work)?
            .ok_or_else(|| refused("zero prime-field syzygy S-pair generator"))?;
        if lc != 1 || rc != 1 {
            return Err(refused("prime-field syzygy basis must be monic"));
        }
        let mut left_shift = reserve(generators.len(), work)?;
        let mut right_shift = reserve(generators.len(), work)?;
        for (&l, &r) in left.iter().zip(right) {
            work.tick(1)?;
            left_shift.push(l.max(r) - l);
            right_shift.push(l.max(r) - r);
        }
        work.allocate(2 * (std::mem::size_of::<Terms>() + 128))?;
        let left_multiplier = Terms::from([(left_shift, 1)]);
        let right_multiplier = Terms::from([(right_shift, 1)]);
        let mut row = reserve(input_count, work)?;
        for coordinate in 0..input_count {
            let mut value = product(&left_multiplier, &a[pair.left][coordinate], prime, work)?;
            generator_subtract_product(
                &mut value,
                &right_multiplier,
                &a[pair.right][coordinate],
                prime,
                work,
            )?;
            for (coefficient, basis_row) in pair.quotients.iter().zip(a) {
                generator_subtract_product(
                    &mut value,
                    coefficient,
                    &basis_row[coordinate],
                    prime,
                    work,
                )?;
            }
            row.push(generator_cell(&value, generators, prime, work)?);
        }
        generator_keep_row(&mut result, row, work)?;
    }
    Ok(result)
}

/// Produce untrusted complete-relation evidence without invoking its verifier.
pub fn bounded_prime_field_syzygy_certificate(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    order: TermOrder,
    generator: &mut impl BudgetMeter,
) -> Result<PrimeFieldPolynomialSyzygyCertificate, PolyError> {
    let mut work = PolynomialWork::new(generator);
    admit(generators, prime, inputs, &mut work)?;
    let groebner =
        bounded_prime_field_groebner_certificate(generators, prime, inputs, order, &mut work)?;
    preflight_certificate(inputs.len(), generators.len(), prime, &groebner, &mut work)?;
    let syzygies =
        generate_relations(generators, prime, inputs.len(), order, &groebner, &mut work)?;
    work.tick(1)?;
    Ok(PrimeFieldPolynomialSyzygyCertificate { groebner, syzygies })
}

fn reference_columns(
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Columns, PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    Ok(Columns::new())
}

fn reference_monomial(
    powers: Vec<u32>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    work.allocate(std::mem::size_of::<Terms>() + 128)?;
    Ok(Terms::from([(powers, 1)]))
}

fn reference_row(
    expected: &[Columns],
    inputs: &[Polynomial],
    prime: u64,
    proposed: &[Vec<Polynomial>],
    next: &mut usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let mut nonzero = false;
    for column in expected {
        for value in column.values() {
            work.tick(1)?;
            nonzero |= value.rem_euclid(i128::from(prime)) != 0;
        }
    }
    if !nonzero {
        return Ok(());
    }
    let row = proposed
        .get(*next)
        .ok_or_else(|| refused("prime-field syzygy certificate omitted a complete relation"))?;
    for (columns, actual) in expected.iter().zip(row) {
        let mut count = 0;
        for (powers, signed) in columns {
            work.tick(powers.len() + 1)?;
            let residue = signed.rem_euclid(i128::from(prime)) as u64;
            if residue != 0 {
                count += 1;
                if actual.terms().get(powers) != Some(&residue) {
                    return Err(refused(
                        "prime-field syzygy certificate changed exact relation coefficients",
                    ));
                }
            }
        }
        if count != actual.terms().len() {
            return Err(refused(
                "prime-field syzygy certificate changed relation term coverage",
            ));
        }
    }
    // Independently check each published relation itself, in addition to its
    // exact complete-generation formula. Original zero generators are valid.
    let mut residual = reference_columns(work)?;
    for (coefficient, source) in row.iter().zip(inputs) {
        reference_product(&mut residual, coefficient.terms(), source.terms(), 1, work)?;
    }
    identity_zero(&residual, prime, work)?;
    *next += 1;
    Ok(())
}

fn verify_compositions(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    order: TermOrder,
    certificate: &PrimeFieldPolynomialSyzygyCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let groebner = &certificate.groebner;
    let a = &groebner.basis_in_input;
    let mut next = 0;
    let mut zero_powers = reserve(generators.len(), work)?;
    zero_powers.resize(generators.len(), 0);
    let one = reference_monomial(zero_powers, work)?;
    for (index, coefficients) in groebner.input_in_basis.iter().enumerate() {
        let mut expected = reserve(inputs.len(), work)?;
        for (coordinate, _) in inputs.iter().enumerate() {
            let mut value = reference_columns(work)?;
            if coordinate == index {
                reference_add(&mut value, &one, 1, work)?;
            }
            for (basis_index, coefficient) in coefficients.iter().enumerate() {
                reference_product(
                    &mut value,
                    coefficient,
                    &a[basis_index][coordinate],
                    -1,
                    work,
                )?;
            }
            expected.push(value);
        }
        reference_row(
            &expected,
            inputs,
            prime,
            &certificate.syzygies,
            &mut next,
            work,
        )?;
    }
    for pair in &groebner.s_pairs {
        let (left, _) = reference_head(&groebner.basis[pair.left], order, work)?
            .ok_or_else(|| refused("zero polynomial in checked prime-field syzygy basis"))?;
        let (right, _) = reference_head(&groebner.basis[pair.right], order, work)?
            .ok_or_else(|| refused("zero polynomial in checked prime-field syzygy basis"))?;
        let mut left_shift = reserve(generators.len(), work)?;
        let mut right_shift = reserve(generators.len(), work)?;
        for (&l, &r) in left.iter().zip(right) {
            work.tick(1)?;
            // Independent of the generator's LCM-minus-head construction.
            left_shift.push(r.saturating_sub(l));
            right_shift.push(l.saturating_sub(r));
        }
        let left_multiplier = reference_monomial(left_shift, work)?;
        let right_multiplier = reference_monomial(right_shift, work)?;
        let mut expected = reserve(inputs.len(), work)?;
        for (coordinate, _) in inputs.iter().enumerate() {
            let mut value = reference_columns(work)?;
            reference_product(
                &mut value,
                &left_multiplier,
                &a[pair.left][coordinate],
                1,
                work,
            )?;
            reference_product(
                &mut value,
                &right_multiplier,
                &a[pair.right][coordinate],
                -1,
                work,
            )?;
            for (basis_index, coefficient) in pair.quotients.iter().enumerate() {
                reference_product(
                    &mut value,
                    coefficient,
                    &a[basis_index][coordinate],
                    -1,
                    work,
                )?;
            }
            expected.push(value);
        }
        reference_row(
            &expected,
            inputs,
            prime,
            &certificate.syzygies,
            &mut next,
            work,
        )?;
    }
    if next != certificate.syzygies.len() {
        return Err(refused(
            "prime-field syzygy certificate has extra or zero relation rows",
        ));
    }
    work.tick(1)
}

/// Independently verify the full generating-set claim on original coordinates.
/// The full lower Groebner certificate is rechecked; valid output dot products
/// alone do not prove that every relation is generated.
pub fn verify_prime_field_polynomial_syzygies(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    order: TermOrder,
    certificate: &PrimeFieldPolynomialSyzygyCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldPolynomialSyzygies, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, &mut work)?;
    let possible_rows = preflight_certificate(
        inputs.len(),
        generators.len(),
        prime,
        &certificate.groebner,
        &mut work,
    )?;
    if certificate.syzygies.len() > possible_rows
        || certificate.syzygies.len() * inputs.len() > MAX_OUTPUT_POLYNOMIALS
        || certificate
            .syzygies
            .iter()
            .any(|row| row.len() != inputs.len())
    {
        return Err(refused(
            "prime-field syzygy output dimensions exceed or change original coordinates",
        ));
    }
    for polynomial in certificate.syzygies.iter().flatten() {
        same_ring(polynomial, generators, prime, &mut work)?;
    }
    let basis = verify_prime_field_groebner_certificate(
        generators,
        prime,
        inputs,
        order,
        &certificate.groebner,
        &mut work,
    )?;
    verify_compositions(generators, prime, inputs, order, certificate, &mut work)?;
    let mut syzygies = reserve(certificate.syzygies.len(), &mut work)?;
    for row in &certificate.syzygies {
        let mut copied = reserve(inputs.len(), &mut work)?;
        for polynomial in row {
            copied.push(copied_polynomial(
                generators,
                prime,
                polynomial.terms(),
                &mut work,
            )?);
        }
        syzygies.push(copied);
    }
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldPolynomialSyzygies>())?;
    let accepted = VerifiedPrimeFieldPolynomialSyzygies { basis, syzygies };
    work.tick(1)?;
    Ok(accepted)
}

/// Compute and independently verify every polynomial relation on the ordered
/// original inputs. Exhaustion and cancellation never publish a partial set.
pub fn checked_prime_field_polynomial_syzygies(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldPolynomialSyzygies, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let certificate =
        bounded_prime_field_syzygy_certificate(generators, prime, inputs, order, &mut generation)?;
    let accepted = verify_prime_field_polynomial_syzygies(
        generators,
        prime,
        inputs,
        order,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
