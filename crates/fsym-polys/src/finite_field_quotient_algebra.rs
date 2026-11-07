//! Complete finite quotient algebras and checked multiplication over F_p.
//!
//! An immutable independently checked source Groebner basis supplies the exact
//! original ideal, typed ring, characteristic and monomial order. Its complete
//! staircase retains nilpotents and multiplicities; no field-point equations,
//! radicalization or numerical rank tests are introduced. Standard monomials
//! are ascending in the requested order. Matrices are row-major, with column j
//! equal to the normal form of element times standard monomial j.
//!
//! The composition verifier checks standardness, the presence of one and every
//! unblocked coordinate multiple independently of enumeration. It reconstructs
//! every source column by injective exponent translation and verifies exact
//! identities through signed coefficient convolution, without invoking the
//! generator's modular arithmetic, normal-form search or monomial comparator.
//!
//! The existing bounds of eight variables, 32 original generators, degree 64
//! and 4096 terms remain in force. Quotient dimension is at most 64; temporary
//! products may have degree 128. Each entire generation and verification phase
//! retains separate cumulative 20M-work / 128MiB logical-allocation limits.
//! The unit ideal has an empty staircase and matrix; the constant field has
//! staircase [1]. These are immutable in-process composition receipts.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    Columns, MAX_BASIS, MAX_DEGREE, MAX_VARIABLES, admit_division, admit_ring, admit_terms,
    copied_polynomial, copy_terms, divide, divides, empty, identity_zero, leading, monomial_copy,
    product, reference_add, reference_head, reference_order, reference_product, same_ring,
};
use crate::{
    PolyError, PrimeFieldMultivariatePolynomial, PrimeFieldMultivariateTerms, TermOrder,
    VerifiedPrimeFieldGroebnerBasis, checked_prime_field_groebner_basis,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;

type Polynomial = PrimeFieldMultivariatePolynomial;
type Terms = PrimeFieldMultivariateTerms;
const MAX_QUOTIENT_DIMENSION: usize = 64;
const MAX_MONOMIAL_STATES: usize = 1 + MAX_QUOTIENT_DIMENSION * MAX_VARIABLES;

/// An untrusted complete staircase, matrix and column identities around an
/// immutable accepted source basis. `reduction_quotients[j][k]` multiplies source
/// basis polynomial k in the identity for element times standard monomial j.
#[derive(Clone, Debug)]
pub struct PrimeFieldQuotientMultiplicationCertificate {
    pub source_basis: Arc<VerifiedPrimeFieldGroebnerBasis>,
    pub standard_monomials: Vec<Vec<u32>>,
    pub reduction_quotients: Vec<Vec<Terms>>,
    pub matrix: Vec<Vec<u64>>,
}

/// The exact multiplication operator on the complete finite quotient F_p[x]/I.
/// Only independent composition verification constructs this accepted owner.
#[derive(Clone, Debug)]
pub struct VerifiedPrimeFieldQuotientMultiplication {
    element: Polynomial,
    certificate: PrimeFieldQuotientMultiplicationCertificate,
}

impl VerifiedPrimeFieldQuotientMultiplication {
    pub fn generators(&self) -> &[Symbol] {
        self.certificate.source_basis.generators()
    }
    pub fn prime(&self) -> u64 {
        self.certificate.source_basis.prime()
    }
    pub fn inputs(&self) -> &[Polynomial] {
        self.certificate.source_basis.inputs()
    }
    pub fn element(&self) -> &Polynomial {
        &self.element
    }
    pub fn order(&self) -> TermOrder {
        self.certificate.source_basis.order()
    }
    pub fn standard_monomials(&self) -> &[Vec<u32>] {
        &self.certificate.standard_monomials
    }
    pub fn matrix(&self) -> &[Vec<u64>] {
        &self.certificate.matrix
    }
    pub fn certificate(&self) -> &PrimeFieldQuotientMultiplicationCertificate {
        &self.certificate
    }
}

fn reserve<T>(
    count: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<T>, PolyError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| refused("prime-field quotient allocation size overflow"))?;
    work.allocate(bytes)?;
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(refused)?;
    Ok(result)
}

fn check_polynomials(
    polynomials: &[Polynomial],
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if polynomials.len() > MAX_BASIS {
        return Err(refused("prime-field quotient input exceeds 32 polynomials"));
    }
    for polynomial in polynomials {
        same_ring(polynomial, generators, prime, work)?;
    }
    Ok(())
}

fn admit(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(generators, prime, work)?;
    check_polynomials(inputs, generators, prime, work)?;
    same_ring(element, generators, prime, work)
}

fn check_source(
    source: &VerifiedPrimeFieldGroebnerBasis,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(source.generators(), source.prime(), work)?;
    check_polynomials(source.inputs(), source.generators(), source.prime(), work)?;
    check_polynomials(source.basis(), source.generators(), source.prime(), work)
}

fn bind_source(
    source: &VerifiedPrimeFieldGroebnerBasis,
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    order: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if source.generators() != generators
        || source.prime() != prime
        || source.order() != order
        || source.inputs().len() != inputs.len()
    {
        return Err(refused(
            "prime-field quotient changed its source ring, order or coverage",
        ));
    }
    check_source(source, work)?;
    for (actual, original) in source.inputs().iter().zip(inputs) {
        if actual.terms().len() != original.terms().len() {
            return Err(refused(
                "prime-field quotient changed an original ideal generator",
            ));
        }
        for ((powers, coefficient), (original_powers, original_coefficient)) in
            actual.terms().iter().zip(original.terms())
        {
            work.tick(powers.len() + 1)?;
            if powers != original_powers || coefficient != original_coefficient {
                return Err(refused(
                    "prime-field quotient changed original coefficients",
                ));
            }
        }
    }
    Ok(())
}

fn check_monomial(
    monomial: &[u32],
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(width + 1)?;
    if monomial.len() != width
        || monomial.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE
    {
        return Err(refused(
            "prime-field quotient monomial width or degree bound",
        ));
    }
    Ok(())
}

fn unit_ideal(source: &VerifiedPrimeFieldGroebnerBasis) -> bool {
    source.basis().len() == 1
        && source.basis()[0].terms().len() == 1
        && source.basis()[0]
            .terms()
            .iter()
            .next()
            .is_some_and(|(powers, value)| *value == 1 && powers.iter().all(|&power| power == 0))
}

// This reference criterion does not enumerate monomials or invoke generator
// divisibility/leading-term selection. Starting from one, complete unblocked
// coordinate closure reaches every standard monomial by induction on degree.
fn verify_staircase(
    source: &VerifiedPrimeFieldGroebnerBasis,
    monomials: &[Vec<u32>],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if monomials.len() > MAX_QUOTIENT_DIMENSION {
        return Err(refused("prime-field quotient dimension exceeds 64"));
    }
    let width = source.generators().len();
    for monomial in monomials {
        check_monomial(monomial, width, work)?;
    }
    if unit_ideal(source) {
        return if monomials.is_empty() {
            work.tick(1)
        } else {
            Err(refused(
                "the zero quotient ring must have an empty staircase",
            ))
        };
    }
    if !monomials
        .first()
        .is_some_and(|powers| powers.iter().all(|&power| power == 0))
    {
        return Err(refused(
            "a proper prime-field quotient staircase must start with one",
        ));
    }
    for pair in monomials.windows(2) {
        work.tick(3 * width + 1)?;
        if !reference_order(&pair[0], &pair[1], source.order()).is_lt() {
            return Err(refused(
                "prime-field quotient monomials are repeated or out of order",
            ));
        }
    }
    let mut heads = reserve(source.basis().len(), work)?;
    for polynomial in source.basis() {
        let (head, _) = reference_head(polynomial.terms(), source.order(), work)?
            .ok_or_else(|| refused("zero polynomial in an accepted quotient basis"))?;
        heads.push(head);
    }
    for monomial in monomials {
        work.tick(heads.len().max(1) * width.max(1))?;
        if heads
            .iter()
            .any(|head| head.iter().zip(monomial).all(|(a, b)| a <= b))
        {
            return Err(refused(
                "prime-field quotient contains a nonstandard monomial",
            ));
        }
        for variable in 0..width {
            let mut child = monomial_copy(monomial, work)?;
            child[variable] = child[variable]
                .checked_add(1)
                .ok_or_else(|| refused("prime-field quotient border exponent overflow"))?;
            check_monomial(&child, width, work)?;
            work.tick(heads.len().max(1) * width.max(1))?;
            if !heads
                .iter()
                .any(|head| head.iter().zip(&child).all(|(a, b)| a <= b))
            {
                work.tick(monomials.len() * width.max(1))?;
                if !monomials.iter().any(|entry| entry == &child) {
                    return Err(refused("prime-field quotient omitted an unblocked border"));
                }
            }
        }
    }
    work.tick(1)
}

fn enumerate_staircase(
    source: &VerifiedPrimeFieldGroebnerBasis,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Vec<u32>>, PolyError> {
    check_source(source, work)?;
    if unit_ideal(source) {
        work.tick(1)?;
        return Ok(Vec::new());
    }
    let width = source.generators().len();
    let mut heads = reserve(source.basis().len(), work)?;
    let mut bounded = reserve(width, work)?;
    bounded.resize(width, false);
    for polynomial in source.basis() {
        let (head, _) = leading(polynomial.terms(), source.order(), work)?
            .ok_or_else(|| refused("zero polynomial in an accepted quotient basis"))?;
        work.tick(width + 1)?;
        let mut nonzero = head.iter().enumerate().filter(|(_, power)| **power != 0);
        if let Some((variable, _)) = nonzero.next()
            && nonzero.next().is_none()
        {
            bounded[variable] = true;
        }
        heads.push(head);
    }
    if !bounded.into_iter().all(|value| value) {
        return Err(refused(
            "prime-field quotient requires a finite-dimensional source ideal",
        ));
    }
    work.allocate(2 * std::mem::size_of::<BTreeSet<Vec<u32>>>())?;
    let mut pending = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut one = reserve(width, work)?;
    one.resize(width, 0);
    work.allocate(128)?;
    pending.insert(one);
    let mut result = reserve(MAX_QUOTIENT_DIMENSION, work)?;
    while let Some(monomial) = pending.pop_first() {
        work.tick(MAX_MONOMIAL_STATES * width.max(1))?;
        if visited.contains(&monomial) {
            continue;
        }
        if result.len() == MAX_QUOTIENT_DIMENSION {
            return Err(refused("prime-field quotient dimension exceeds 64"));
        }
        work.allocate(128)?;
        visited.insert(monomial_copy(&monomial, work)?);
        for variable in 0..width {
            let mut child = monomial_copy(&monomial, work)?;
            child[variable] = child[variable]
                .checked_add(1)
                .ok_or_else(|| refused("prime-field quotient border exponent overflow"))?;
            check_monomial(&child, width, work)?;
            work.tick((heads.len() + 2 * MAX_MONOMIAL_STATES).max(1) * width.max(1))?;
            if !heads.iter().any(|head| divides(head, &child))
                && !visited.contains(&child)
                && !pending.contains(&child)
            {
                if pending.len() >= MAX_MONOMIAL_STATES {
                    return Err(refused(
                        "prime-field quotient frontier exceeds its state bound",
                    ));
                }
                work.allocate(128)?;
                pending.insert(child);
            }
        }
        result.push(monomial);
    }
    work.tick(result.len().max(1) * result.len().max(1) * width.max(1))?;
    result.sort_unstable_by(|a, b| source.order().compare_monomials(a, b));
    verify_staircase(source, &result, work)?;
    work.tick(1)?;
    Ok(result)
}

impl VerifiedPrimeFieldGroebnerBasis {
    /// Complete ordered standard monomials of F_p[x]/I. Positive dimension or
    /// dimension above 64 refuses explicitly; no truncated basis is returned.
    /// The unit ideal yields [], and the constant field's zero ideal yields [1].
    pub fn standard_monomials(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<Vec<Vec<u32>>, PolyError> {
        enumerate_staircase(self, &mut PolynomialWork::new(meter))
    }
}

fn preflight_columns(
    certificate: &PrimeFieldQuotientMultiplicationCertificate,
    width: usize,
    prime: u64,
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
            "prime-field quotient matrix or witness dimensions changed",
        ));
    }
    for value in certificate.matrix.iter().flatten() {
        work.tick(1)?;
        if *value >= prime {
            return Err(refused(
                "prime-field quotient matrix residue is not canonical",
            ));
        }
    }
    for column in &certificate.reduction_quotients {
        for quotient in column {
            admit_terms(quotient, width, prime, MAX_DEGREE, false, work)?;
        }
    }
    Ok(())
}

// Translation is injective on the admitted original support. Construct the
// reference product directly, without modular sum/product or normalization.
fn reference_shift(
    element: &Polynomial,
    monomial: &[u32],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (powers, coefficient) in element.terms() {
        work.tick(powers.len() + 1)?;
        let mut shifted = reserve(powers.len(), work)?;
        for (&power, &offset) in powers.iter().zip(monomial) {
            shifted.push(
                power
                    .checked_add(offset)
                    .ok_or_else(|| refused("prime-field quotient source exponent overflow"))?,
            );
        }
        if shifted.iter().map(|&power| u64::from(power)).sum::<u64>() > 2 * MAX_DEGREE {
            return Err(refused("prime-field quotient intermediate degree bound"));
        }
        work.allocate(128)?;
        result.insert(shifted, *coefficient);
    }
    Ok(result)
}

fn reference_column(
    certificate: &PrimeFieldQuotientMultiplicationCertificate,
    column: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (monomial, row) in certificate
        .standard_monomials
        .iter()
        .zip(&certificate.matrix)
    {
        work.tick(1)?;
        let value = row[column];
        if value != 0 {
            work.allocate(128)?;
            result.insert(monomial_copy(monomial, work)?, value);
        }
    }
    Ok(result)
}

fn copy_certificate(
    certificate: &PrimeFieldQuotientMultiplicationCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldQuotientMultiplicationCertificate, PolyError> {
    let dimension = certificate.standard_monomials.len();
    let mut standard_monomials = reserve(dimension, work)?;
    for monomial in &certificate.standard_monomials {
        standard_monomials.push(monomial_copy(monomial, work)?);
    }
    let mut matrix = reserve(dimension, work)?;
    for row in &certificate.matrix {
        let mut values = reserve(dimension, work)?;
        work.tick(dimension)?;
        values.extend_from_slice(row);
        matrix.push(values);
    }
    let mut reduction_quotients = reserve(dimension, work)?;
    for column in &certificate.reduction_quotients {
        let mut values = reserve(column.len(), work)?;
        for quotient in column {
            values.push(copy_terms(quotient, work)?);
        }
        reduction_quotients.push(values);
    }
    work.allocate(std::mem::size_of::<
        PrimeFieldQuotientMultiplicationCertificate,
    >())?;
    Ok(PrimeFieldQuotientMultiplicationCertificate {
        source_basis: Arc::clone(&certificate.source_basis),
        standard_monomials,
        reduction_quotients,
        matrix,
    })
}

/// Independently verify complete staircase coverage and every exact matrix
/// column against the original ideal, element, typed field and declared order.
pub fn verify_prime_field_quotient_multiplication(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
    certificate: &PrimeFieldQuotientMultiplicationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldQuotientMultiplication, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, element, &mut work)?;
    bind_source(
        &certificate.source_basis,
        generators,
        prime,
        inputs,
        order,
        &mut work,
    )?;
    preflight_columns(certificate, generators.len(), prime, &mut work)?;
    verify_staircase(
        &certificate.source_basis,
        &certificate.standard_monomials,
        &mut work,
    )?;
    for (column, monomial) in certificate.standard_monomials.iter().enumerate() {
        let source = reference_shift(element, monomial, &mut work)?;
        let remainder = reference_column(certificate, column, &mut work)?;
        work.allocate(std::mem::size_of::<Columns>())?;
        let mut columns = Columns::new();
        reference_add(&mut columns, &source, 1, &mut work)?;
        reference_add(&mut columns, &remainder, -1, &mut work)?;
        for (quotient, divisor) in certificate.reduction_quotients[column]
            .iter()
            .zip(certificate.source_basis.basis())
        {
            reference_product(&mut columns, quotient, divisor.terms(), -1, &mut work)?;
        }
        identity_zero(&columns, prime, &mut work)?;
    }
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldQuotientMultiplication>())?;
    let accepted = VerifiedPrimeFieldQuotientMultiplication {
        element: copied_polynomial(generators, prime, element.terms(), &mut work)?,
        certificate: copy_certificate(certificate, &mut work)?,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Construct the exact multiplication matrix in the complete finite quotient.
/// Every original input and the element are admitted before zero/unit shortcuts.
/// Oversized or positive-dimensional quotients refuse without partial output.
pub fn checked_prime_field_quotient_multiplication(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    element: &Polynomial,
    order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldQuotientMultiplication, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(generators, prime, inputs, element, &mut generation)?;
    let source_basis = checked_prime_field_groebner_basis(
        generators,
        prime,
        inputs,
        order,
        &mut generation,
        &mut verification,
    )?;
    let standard_monomials = source_basis.standard_monomials(&mut generation)?;
    let dimension = standard_monomials.len();
    let mut matrix = reserve(dimension, &mut generation)?;
    for _ in 0..dimension {
        let mut row = reserve(dimension, &mut generation)?;
        row.resize(dimension, 0);
        matrix.push(row);
    }
    let mut divisors = reserve(source_basis.basis().len(), &mut generation)?;
    for polynomial in source_basis.basis() {
        divisors.push(polynomial.terms());
    }
    let mut reduction_quotients = reserve(dimension, &mut generation)?;
    for (column, monomial) in standard_monomials.iter().enumerate() {
        let mut basis_element = empty(&mut generation)?;
        generation.allocate(128)?;
        basis_element.insert(monomial_copy(monomial, &mut generation)?, 1);
        let multiplied = product(element.terms(), &basis_element, prime, &mut generation)?;
        let reduction = divide(&multiplied, &divisors, order, prime, &mut generation)?;
        admit_division(
            &reduction,
            generators.len(),
            divisors.len(),
            prime,
            &mut generation,
        )?;
        for (powers, coefficient) in reduction.remainder {
            generation.tick(dimension.max(1) * generators.len().max(1))?;
            let row = standard_monomials
                .iter()
                .position(|entry| entry == &powers)
                .ok_or_else(|| refused("prime-field remainder escaped its complete staircase"))?;
            matrix[row][column] = coefficient;
        }
        reduction_quotients.push(reduction.quotients);
    }
    generation.allocate(
        std::mem::size_of::<VerifiedPrimeFieldGroebnerBasis>()
            + std::mem::size_of::<PrimeFieldQuotientMultiplicationCertificate>()
            + 32,
    )?;
    let certificate = PrimeFieldQuotientMultiplicationCertificate {
        source_basis: Arc::new(source_basis),
        standard_monomials,
        reduction_quotients,
        matrix,
    };
    let accepted = verify_prime_field_quotient_multiplication(
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
