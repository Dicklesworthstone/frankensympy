//! Checked multiplication operators in finite rational quotient algebras.
//!
//! The original ordered polynomials first receive a complete independently
//! checked Groebner basis. Its standard monomials form the vector basis of
//! QQ[x]/I, including nilpotents: no radicalization or root sampling occurs.
//! Column j of the matrix is the normal form of h*b_j in that ordered basis.
//!
//! The composition verifier binds the accepted basis to the full request and
//! checks the staircase independently. It must contain 1 for a proper ideal,
//! every entry must be standard, and every variable multiple not blocked by a
//! leading monomial must also occur. This proves completeness, not just linear
//! independence or the validity of some selected normal forms. Each column is
//! then checked with the independent coefficient-convolution division checker.
//! The source shift and column remainder are reconstructed directly without
//! invoking generator polynomial arithmetic or standard-monomial enumeration.
//!
//! Matrices are row-major and act on coefficient column vectors. Monomials
//! are ascending in the requested order. Dimension is at most 64, with at most
//! eight variables and 64 original generators. Existing degree, coefficient
//! and term bounds apply, and both entire phases retain cumulative 20M-work /
//! 128MiB logical-allocation budgets. The zero ring has an empty matrix; QQ
//! itself has basis [1]. These are in-process checked composition receipts.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::sync::Arc;

use crate::division::{
    MAX_BASIS, MAX_VARIABLES, PolynomialWork, Terms, check_monomial, copy_poly, divide_with_work,
    monomial_poly, multiply_polynomials, refused, ring_storage, same_ring, validate_poly,
};
use crate::{
    MultivariatePoly, PolyError, TermOrder, VerifiedGroebnerBasis, checked_groebner_basis,
    verify_bounded_division,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};

const MAX_QUOTIENT_DIMENSION: usize = 64;
const MAX_COEFFICIENT_BITS: u64 = 4096;

/// Untrusted full matrix and column identities around an immutable checked
/// source basis. `reduction_quotients[column][basis_index]` expresses the
/// difference between h*b_column and that column's proposed normal form.
#[derive(Debug, Clone)]
pub struct QuotientMultiplicationCertificate {
    pub source_basis: Arc<VerifiedGroebnerBasis>,
    pub standard_monomials: Vec<Vec<u32>>,
    pub reduction_quotients: Vec<Vec<MultivariatePoly>>,
    pub matrix: Vec<Vec<BigRational>>,
}

/// The exact linear operator v -> element*v on the complete finite quotient.
/// Construction is private and follows independent verification of all columns.
#[derive(Debug, Clone)]
pub struct VerifiedQuotientMultiplication {
    generators: Vec<Symbol>,
    element: MultivariatePoly,
    certificate: QuotientMultiplicationCertificate,
}

impl VerifiedQuotientMultiplication {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }

    pub fn inputs(&self) -> &[MultivariatePoly] {
        self.certificate.source_basis.input_basis()
    }

    pub fn element(&self) -> &MultivariatePoly {
        &self.element
    }

    pub fn order(&self) -> TermOrder {
        self.certificate.source_basis.order()
    }

    pub fn standard_monomials(&self) -> &[Vec<u32>] {
        &self.certificate.standard_monomials
    }

    pub fn matrix(&self) -> &[Vec<BigRational>] {
        &self.certificate.matrix
    }

    pub fn certificate(&self) -> &QuotientMultiplicationCertificate {
        &self.certificate
    }
}

fn reserve<T>(
    count: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<T>, PolyError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| refused("quotient-operator allocation size overflow"))?;
    work.allocate(bytes)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(refused)?;
    Ok(values)
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused(
            "quotient operator supports at most eight variables",
        ));
    }
    let mut names = 0usize;
    for (index, generator) in generators.iter().enumerate() {
        names = names
            .checked_add(generator.name.len())
            .filter(|&bytes| bytes <= 65_536)
            .ok_or_else(|| refused("quotient-operator names exceed 65536 bytes"))?;
        work.tick(generator.name.len() + index + 1)?;
        if generators[..index].contains(generator) {
            return Err(refused("duplicate typed quotient-operator generator"));
        }
    }
    Ok(())
}

fn check_polynomial(
    polynomial: &MultivariatePoly,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_ring(polynomial.generators(), work)?;
    same_ring(polynomial.generators(), generators)?;
    validate_poly(polynomial, work)
}

fn check_polynomials(
    polynomials: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if polynomials.len() > MAX_BASIS {
        return Err(refused(
            "quotient-operator polynomial list exceeds 64 entries",
        ));
    }
    for polynomial in polynomials {
        check_polynomial(polynomial, generators, work)?;
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
    check_polynomials(inputs, generators, work)?;
    check_polynomial(element, generators, work)
}

fn bind_basis(
    source: &VerifiedGroebnerBasis,
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    order: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if source.order() != order || source.input_basis().len() != inputs.len() {
        return Err(refused(
            "quotient operator changed source order or input coverage",
        ));
    }
    check_polynomials(source.input_basis(), generators, work)?;
    check_polynomials(source.basis(), generators, work)?;
    for (actual, original) in source.input_basis().iter().zip(inputs) {
        if actual.terms.len() != original.terms.len() {
            return Err(refused(
                "quotient operator changed its original ideal generators",
            ));
        }
        for ((powers, value), (original_powers, original_value)) in
            actual.terms.iter().zip(&original.terms)
        {
            let limbs = value.numer().limb_count()
                + value.denom().limb_count()
                + original_value.numer().limb_count()
                + original_value.denom().limb_count();
            work.tick(powers.len() + limbs as usize + 1)?;
            if powers != original_powers || value != original_value {
                return Err(refused(
                    "quotient operator changed exact original coefficients",
                ));
            }
        }
    }
    Ok(())
}

// A separate reference ordering avoids relying on the generator's leading
// monomial selection or on its standard-monomial enumeration for completeness.
fn reference_compare(left: &[u32], right: &[u32], order: TermOrder) -> Ordering {
    if order != TermOrder::Lex {
        let left_degree: u64 = left.iter().map(|&value| u64::from(value)).sum();
        let right_degree: u64 = right.iter().map(|&value| u64::from(value)).sum();
        match left_degree.cmp(&right_degree) {
            Ordering::Equal => {}
            unequal => return unequal,
        }
    }
    if order == TermOrder::DegRevLex {
        for (&a, &b) in left.iter().rev().zip(right.iter().rev()) {
            match b.cmp(&a) {
                Ordering::Equal => {}
                unequal => return unequal,
            }
        }
        Ordering::Equal
    } else {
        left.iter().cmp(right.iter())
    }
}

fn reference_head<'a>(
    polynomial: &'a MultivariatePoly,
    order: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<&'a [u32], PolyError> {
    let mut result: Option<&[u32]> = None;
    for powers in polynomial.terms.keys() {
        work.tick(3 * powers.len() + 1)?;
        if result.is_none_or(|previous| reference_compare(powers, previous, order).is_gt()) {
            result = Some(powers);
        }
    }
    result.ok_or_else(|| refused("zero polynomial in accepted quotient basis"))
}

fn reference_divides(divisor: &[u32], monomial: &[u32]) -> bool {
    divisor.len() == monomial.len() && divisor.iter().zip(monomial).all(|(a, b)| a <= b)
}

fn verify_staircase(
    basis: &VerifiedGroebnerBasis,
    monomials: &[Vec<u32>],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if monomials.len() > MAX_QUOTIENT_DIMENSION {
        return Err(refused("quotient dimension exceeds 64"));
    }
    for powers in monomials {
        work.tick(powers.len().min(MAX_VARIABLES) + 1)?;
        if powers.len() != generators.len() {
            return Err(refused("quotient monomial has the wrong coordinate width"));
        }
        check_monomial(powers)?;
    }
    let unit = basis.basis().len() == 1
        && basis.basis()[0].terms.len() == 1
        && basis.basis()[0]
            .terms
            .iter()
            .next()
            .is_some_and(|(powers, value)| {
                powers.iter().all(|&power| power == 0) && value.is_one()
            });
    if unit {
        return if monomials.is_empty() {
            work.tick(1)
        } else {
            Err(refused(
                "the zero quotient ring must have an empty monomial basis",
            ))
        };
    }
    if !monomials
        .first()
        .is_some_and(|powers| powers.iter().all(|&power| power == 0))
    {
        return Err(refused(
            "a proper quotient monomial basis must start with one",
        ));
    }
    for pair in monomials.windows(2) {
        work.tick(3 * generators.len() + 1)?;
        if !reference_compare(&pair[0], &pair[1], basis.order()).is_lt() {
            return Err(refused("quotient monomials are repeated or out of order"));
        }
    }
    let mut heads = reserve(basis.basis().len(), work)?;
    for polynomial in basis.basis() {
        heads.push(reference_head(polynomial, basis.order(), work)?);
    }
    for monomial in monomials {
        work.tick(heads.len().max(1) * generators.len().max(1))?;
        if heads.iter().any(|head| reference_divides(head, monomial)) {
            return Err(refused("quotient basis contains a nonstandard monomial"));
        }
        for (variable, _) in generators.iter().enumerate() {
            work.allocate(generators.len() * 4)?;
            let mut child = monomial.clone();
            child[variable] = child[variable]
                .checked_add(1)
                .ok_or_else(|| refused("quotient border exponent overflow"))?;
            check_monomial(&child)?;
            work.tick(heads.len().max(1) * generators.len().max(1))?;
            if !heads.iter().any(|head| reference_divides(head, &child)) {
                work.tick(monomials.len() * generators.len().max(1))?;
                if !monomials.iter().any(|monomial| monomial == &child) {
                    return Err(refused(
                        "quotient monomial basis omitted an unblocked border",
                    ));
                }
            }
        }
    }
    work.tick(1)
}

fn preflight_columns(
    certificate: &QuotientMultiplicationCertificate,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    let dimension = certificate.standard_monomials.len();
    if dimension > MAX_QUOTIENT_DIMENSION
        || certificate.matrix.len() != dimension
        || certificate.matrix.iter().any(|row| row.len() != dimension)
        || certificate.reduction_quotients.len() != dimension
        || certificate
            .reduction_quotients
            .iter()
            .any(|column| column.len() != certificate.source_basis.basis().len())
    {
        return Err(refused(
            "quotient matrix or column witness dimensions changed",
        ));
    }
    for value in certificate.matrix.iter().flatten() {
        work.tick(1)?;
        if value.numer().bits() > MAX_COEFFICIENT_BITS
            || value.denom().bits() > MAX_COEFFICIENT_BITS
        {
            return Err(refused("quotient matrix coefficient exceeds 4096 bits"));
        }
    }
    for column in &certificate.reduction_quotients {
        check_polynomials(column, generators, work)?;
    }
    Ok(())
}

/// Reference construction of h*b from the original coefficients. Translation
/// of exponent vectors is injective, so no generator normalization is needed.
fn reference_shift(
    element: &MultivariatePoly,
    monomial: &[u32],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let mut terms = Terms::new();
    for (powers, coefficient) in &element.terms {
        work.tick(powers.len() + 1)?;
        let mut shifted = reserve(powers.len(), work)?;
        for (&power, &offset) in powers.iter().zip(monomial) {
            shifted.push(
                power
                    .checked_add(offset)
                    .ok_or_else(|| refused("quotient source degree overflow"))?,
            );
        }
        check_monomial(&shifted)?;
        let value = work.clone_value(coefficient)?;
        work.allocate(256 + shifted.len() * 4)?;
        terms.insert(shifted, value);
    }
    work.allocate(std::mem::size_of::<MultivariatePoly>() + ring_storage(&element.generators))?;
    Ok(MultivariatePoly {
        generators: element.generators.clone(),
        terms,
    })
}

fn reference_column(
    certificate: &QuotientMultiplicationCertificate,
    column: usize,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let mut terms = Terms::new();
    for (powers, row) in certificate
        .standard_monomials
        .iter()
        .zip(&certificate.matrix)
    {
        work.tick(1)?;
        let value = &row[column];
        if !value.is_zero() {
            work.allocate(256 + powers.len() * 4)?;
            terms.insert(powers.clone(), work.clone_value(value)?);
        }
    }
    work.allocate(std::mem::size_of::<MultivariatePoly>() + ring_storage(generators))?;
    Ok(MultivariatePoly {
        generators: generators.to_vec(),
        terms,
    })
}

fn copy_certificate(
    certificate: &QuotientMultiplicationCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<QuotientMultiplicationCertificate, PolyError> {
    let dimension = certificate.standard_monomials.len();
    let mut standard_monomials = reserve(dimension, work)?;
    for powers in &certificate.standard_monomials {
        work.allocate(powers.len() * 4)?;
        standard_monomials.push(powers.clone());
    }
    let mut matrix = reserve(dimension, work)?;
    for row in &certificate.matrix {
        let mut values = reserve(dimension, work)?;
        for value in row {
            values.push(work.clone_value(value)?);
        }
        matrix.push(values);
    }
    let mut reduction_quotients = reserve(dimension, work)?;
    for column in &certificate.reduction_quotients {
        let mut values = reserve(column.len(), work)?;
        for polynomial in column {
            values.push(copy_poly(polynomial, work)?);
        }
        reduction_quotients.push(values);
    }
    work.allocate(std::mem::size_of::<QuotientMultiplicationCertificate>())?;
    Ok(QuotientMultiplicationCertificate {
        source_basis: Arc::clone(&certificate.source_basis),
        standard_monomials,
        reduction_quotients,
        matrix,
    })
}

/// Independently check a complete quotient basis and every multiplication
/// column. Commutativity, a characteristic polynomial, or correct values on a
/// strict subset of the staircase never substitutes for these obligations.
pub fn verify_quotient_multiplication(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    order: TermOrder,
    certificate: &QuotientMultiplicationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedQuotientMultiplication, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(inputs, element, generators, &mut work)?;
    bind_basis(
        &certificate.source_basis,
        inputs,
        generators,
        order,
        &mut work,
    )?;
    preflight_columns(certificate, generators, &mut work)?;
    verify_staircase(
        &certificate.source_basis,
        &certificate.standard_monomials,
        generators,
        &mut work,
    )?;
    for (column, monomial) in certificate.standard_monomials.iter().enumerate() {
        let dividend = reference_shift(element, monomial, &mut work)?;
        let remainder = reference_column(certificate, column, generators, &mut work)?;
        verify_bounded_division(
            &dividend,
            certificate.source_basis.basis(),
            &certificate.reduction_quotients[column],
            &remainder,
            order,
            &mut work,
        )?;
    }
    work.allocate(
        std::mem::size_of::<VerifiedQuotientMultiplication>() + ring_storage(generators),
    )?;
    let result = VerifiedQuotientMultiplication {
        generators: generators.to_vec(),
        element: copy_poly(element, &mut work)?,
        certificate: copy_certificate(certificate, &mut work)?,
    };
    work.tick(1)?;
    Ok(result)
}

/// Construct the exact matrix for multiplication by `element` in QQ[x]/I.
/// Inputs need not already be a Groebner basis. A positive-dimensional ideal
/// or oversized finite quotient is refused, never replaced by a truncated
/// monomial set. Every source is admitted before zero or unit shortcuts.
pub fn checked_quotient_multiplication(
    inputs: &[MultivariatePoly],
    element: &MultivariatePoly,
    generators: &[Symbol],
    order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedQuotientMultiplication, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(inputs, element, generators, &mut generation)?;
    if inputs.is_empty() && !generators.is_empty() {
        return Err(refused(
            "quotient operator requires a finite-dimensional quotient",
        ));
    }
    let source_basis = checked_groebner_basis(inputs, order, &mut generation, &mut verification)?;
    let standard_monomials = source_basis.standard_monomials(&mut generation)?;
    let dimension = standard_monomials.len();
    if dimension > MAX_QUOTIENT_DIMENSION {
        return Err(refused("quotient dimension exceeds 64"));
    }
    let mut matrix = reserve(dimension, &mut generation)?;
    for _ in &standard_monomials {
        let mut row = reserve(dimension, &mut generation)?;
        generation.allocate(dimension * 16)?;
        row.resize(dimension, BigRational::zero());
        matrix.push(row);
    }
    let mut reduction_quotients = reserve(dimension, &mut generation)?;
    for (column, monomial) in standard_monomials.iter().enumerate() {
        generation.allocate(monomial.len() * 4)?;
        let basis_element = monomial_poly(generators, monomial.clone(), &mut generation)?;
        let product = multiply_polynomials(element, &basis_element, &mut generation)?;
        let (quotients, remainder) =
            divide_with_work(&product, source_basis.basis(), order, &mut generation)?;
        for (powers, value) in remainder.terms {
            generation.tick(dimension.max(1) * generators.len().max(1))?;
            let row = standard_monomials
                .iter()
                .position(|monomial| monomial == &powers)
                .ok_or_else(|| refused("generated quotient remainder is outside its staircase"))?;
            matrix[row][column] = value;
        }
        reduction_quotients.push(quotients);
    }
    generation.allocate(
        std::mem::size_of::<VerifiedGroebnerBasis>()
            + std::mem::size_of::<QuotientMultiplicationCertificate>()
            + 32,
    )?;
    let certificate = QuotientMultiplicationCertificate {
        source_basis: Arc::new(source_basis),
        standard_monomials,
        reduction_quotients,
        matrix,
    };
    let accepted = verify_quotient_multiplication(
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
