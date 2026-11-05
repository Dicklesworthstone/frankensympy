//! Complete bounded generators of polynomial syzygies over QQ.
//!
//! Let F be the ordered original generators and G their checked reduced
//! Groebner basis. Its complete certificate supplies G=A*F and F=B*G, together
//! with a strict standard representation for every S-pair of G. The resulting
//! Schreyer relations S generate all relations on G. Consequently the rows of
//! I-B*A and S*A generate all relations on F: if u*F=0, then u*B is a relation
//! on G and u=u*(I-B*A)+(u*B)*A. Only identically zero output rows are removed.
//!
//! The independent checker first verifies the entire underlying Groebner
//! certificate in the dependency-separated proof kernel. A distinct reference
//! convolution then checks every ordered composition row and its zero dot
//! product with F. It uses its own leading-monomial comparison, no generator
//! multiplication, division, S-polynomial, simplification or basis search.
//! This is a checked complete generating set, not a minimal generating set or
//! a reduced module Groebner basis, and not a new portable kernel claim.
//!
//! Input coordinates, including zero and repeated generators, are retained.
//! The envelope is eight variables, 64 inputs, 16,384 output polynomial cells,
//! and the existing degree/term/coefficient limits. Each whole phase has a
//! cumulative 20M-compute/128MiB logical-allocation cap through nested calls.

#![forbid(unsafe_code)]

use std::cmp::Ordering;

use crate::division::{
    MAX_BASIS, MAX_TERMS, MAX_VARIABLES, PolynomialWork, Terms, accumulate, check_monomial,
    copy_poly, monomial_poly, multiply_polynomials, refused, ring_storage, same_ring,
    validate_poly,
};
use crate::groebner::complete_certificate_dimensions;
use crate::{
    CompleteGroebnerBasisCertificate, MultivariatePoly, PolyError, TermOrder,
    VerifiedGroebnerBasis, bounded_groebner_certificate, verify_complete_groebner_certificate,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};

const MAX_OUTPUT_POLYNOMIALS: usize = 16_384;

/// Untrusted evidence, bound by the verifier to the exact original input list.
/// Nonzero rows of I-B*A precede nonzero S*A rows in complete ordered S-pair
/// order. Deleting a valid nonzero row is not accepted as a complete result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialSyzygyCertificate {
    pub groebner: CompleteGroebnerBasisCertificate,
    pub syzygies: Vec<Vec<MultivariatePoly>>,
}

/// All polynomial relations on the original ordered input generators are
/// polynomial combinations of these rows. The rows need not be independent.
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialSyzygies {
    generators: Vec<Symbol>,
    basis: VerifiedGroebnerBasis,
    syzygies: Vec<Vec<MultivariatePoly>>,
}

impl VerifiedPolynomialSyzygies {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }

    pub fn inputs(&self) -> &[MultivariatePoly] {
        self.basis.input_basis()
    }

    pub fn order(&self) -> TermOrder {
        self.basis.order()
    }

    pub fn basis(&self) -> &VerifiedGroebnerBasis {
        &self.basis
    }

    pub fn syzygies(&self) -> &[Vec<MultivariatePoly>] {
        &self.syzygies
    }

    pub fn into_syzygies(self) -> Vec<Vec<MultivariatePoly>> {
        self.syzygies
    }
}

fn reserve<T>(
    length: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<T>, PolyError> {
    let bytes = length
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| refused("syzygy allocation size overflow"))?;
    work.allocate(bytes)?;
    let mut result = Vec::new();
    result.try_reserve_exact(length).map_err(refused)?;
    Ok(result)
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused("syzygy ring exceeds eight variables"));
    }
    let mut name_bytes = 0usize;
    for (index, generator) in generators.iter().enumerate() {
        name_bytes = name_bytes
            .checked_add(generator.name.len())
            .filter(|&bytes| bytes <= 65_536)
            .ok_or_else(|| refused("syzygy generator names exceed 65536 bytes"))?;
        work.tick(index + generator.name.len() + 1)?;
        if generators[..index].contains(generator) {
            return Err(refused("duplicate typed syzygy generator"));
        }
    }
    Ok(())
}

fn check_polynomial(
    polynomial: &MultivariatePoly,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    // Bound names and dimensions before ring diagnostics or polynomial copies.
    check_ring(polynomial.generators(), work)?;
    same_ring(polynomial.generators(), generators)?;
    validate_poly(polynomial, work)
}

fn admit(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_ring(generators, work)?;
    if inputs.len() > MAX_BASIS {
        return Err(refused("syzygy input list exceeds 64 polynomials"));
    }
    for polynomial in inputs {
        check_polynomial(polynomial, generators, work)?;
    }
    Ok(())
}

/// Structural admission only. Mathematical acceptance is exclusively the
/// complete Groebner checker followed by the reference composition checker.
fn preflight_certificate(
    input_count: usize,
    generators: &[Symbol],
    order: TermOrder,
    certificate: &CompleteGroebnerBasisCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<usize, PolyError> {
    work.tick(1)?;
    let count = certificate.candidate.basis.len();
    let pairs = complete_certificate_dimensions(input_count, count)?;
    if certificate.candidate.order != order
        || certificate.candidate.input_ideal_witnesses.len() != count
        || certificate
            .candidate
            .input_ideal_witnesses
            .iter()
            .any(|row| row.len() != input_count)
        || certificate.input_basis_witnesses.len() != input_count
        || certificate
            .input_basis_witnesses
            .iter()
            .any(|row| row.len() != count)
        || certificate.s_pairs.len() != pairs
        || certificate
            .s_pairs
            .iter()
            .any(|pair| pair.basis_coefficients.len() != count)
    {
        return Err(refused(
            "syzygy Groebner witness dimensions or order changed",
        ));
    }
    let mut index = 0;
    for left in 0..count {
        for right in left + 1..count {
            work.tick(1)?;
            let pair = &certificate.s_pairs[index];
            if (pair.left, pair.right) != (left, right) {
                return Err(refused("syzygy S-pair coverage or order changed"));
            }
            index += 1;
        }
    }
    for polynomial in certificate
        .candidate
        .basis
        .iter()
        .chain(certificate.candidate.input_ideal_witnesses.iter().flatten())
        .chain(certificate.input_basis_witnesses.iter().flatten())
        .chain(
            certificate
                .s_pairs
                .iter()
                .flat_map(|pair| &pair.basis_coefficients),
        )
    {
        check_polynomial(polynomial, generators, work)?;
    }
    Ok(input_count + pairs)
}

fn generator_constant(
    generators: &[Symbol],
    one: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    if one {
        work.allocate(generators.len() * 4)?;
        monomial_poly(generators, vec![0; generators.len()], work)
    } else {
        work.allocate(std::mem::size_of::<MultivariatePoly>() + ring_storage(generators))?;
        Ok(MultivariatePoly::zero(generators.to_vec()))
    }
}

fn generator_subtract_product(
    target: &mut MultivariatePoly,
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if left.is_zero() || right.is_zero() {
        return Ok(());
    }
    let product = multiply_polynomials(left, right, work)?;
    for (powers, value) in product.terms {
        let negative = work.subtract(&BigRational::zero(), &value)?;
        accumulate(&mut target.terms, powers, negative, work)?;
    }
    Ok(())
}

fn generator_keep_row(
    rows: &mut Vec<Vec<MultivariatePoly>>,
    row: Vec<MultivariatePoly>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(row.len())?;
    if row.iter().all(MultivariatePoly::is_zero) {
        return Ok(());
    }
    if (rows.len() + 1) * row.len() > MAX_OUTPUT_POLYNOMIALS {
        return Err(refused("syzygy output exceeds 16384 polynomial cells"));
    }
    rows.push(row);
    Ok(())
}

fn generate_relations(
    input_count: usize,
    generators: &[Symbol],
    certificate: &CompleteGroebnerBasisCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Vec<MultivariatePoly>>, PolyError> {
    let a = &certificate.candidate.input_ideal_witnesses;
    let b = &certificate.input_basis_witnesses;
    let capacity =
        (input_count + certificate.s_pairs.len()).min(MAX_OUTPUT_POLYNOMIALS / input_count.max(1));
    let mut result = reserve(capacity, work)?;
    for (index, coefficients) in b.iter().enumerate() {
        let mut row = reserve(input_count, work)?;
        for coordinate in 0..input_count {
            let mut value = generator_constant(generators, index == coordinate, work)?;
            for (coefficient, basis_row) in coefficients.iter().zip(a) {
                generator_subtract_product(&mut value, coefficient, &basis_row[coordinate], work)?;
            }
            row.push(value);
        }
        generator_keep_row(&mut result, row, work)?;
    }
    for pair in &certificate.s_pairs {
        let basis = &certificate.candidate.basis;
        work.tick(
            (basis[pair.left].terms.len() + basis[pair.right].terms.len())
                * generators.len().max(1),
        )?;
        let left = basis[pair.left]
            .leading_term(certificate.candidate.order)
            .ok_or_else(|| refused("zero syzygy S-pair generator"))?
            .0;
        let right = basis[pair.right]
            .leading_term(certificate.candidate.order)
            .ok_or_else(|| refused("zero syzygy S-pair generator"))?
            .0;
        work.allocate(generators.len() * 8)?;
        let left_shift = left
            .iter()
            .zip(right)
            .map(|(&l, &r)| l.max(r) - l)
            .collect();
        let right_shift = left
            .iter()
            .zip(right)
            .map(|(&l, &r)| l.max(r) - r)
            .collect();
        let left_multiplier = monomial_poly(generators, left_shift, work)?;
        let right_multiplier = monomial_poly(generators, right_shift, work)?;
        let mut row = reserve(input_count, work)?;
        for coordinate in 0..input_count {
            let mut value =
                multiply_polynomials(&left_multiplier, &a[pair.left][coordinate], work)?;
            generator_subtract_product(
                &mut value,
                &right_multiplier,
                &a[pair.right][coordinate],
                work,
            )?;
            for (coefficient, basis_row) in pair.basis_coefficients.iter().zip(a) {
                generator_subtract_product(&mut value, coefficient, &basis_row[coordinate], work)?;
            }
            row.push(value);
        }
        generator_keep_row(&mut result, row, work)?;
    }
    Ok(result)
}

/// Produce untrusted complete-relation evidence. This generator never invokes
/// the verifier and grants no accepted result on its own.
pub fn bounded_syzygy_certificate(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    order: TermOrder,
    generator_meter: &mut impl BudgetMeter,
) -> Result<PolynomialSyzygyCertificate, PolyError> {
    let mut work = PolynomialWork::new(generator_meter);
    admit(inputs, generators, &mut work)?;
    let groebner = bounded_groebner_certificate(inputs, order, &mut work)?;
    preflight_certificate(inputs.len(), generators, order, &groebner, &mut work)?;
    let syzygies = generate_relations(inputs.len(), generators, &groebner, &mut work)?;
    work.tick(1)?;
    Ok(PolynomialSyzygyCertificate { groebner, syzygies })
}

// The following reference lane deliberately does not use generator polynomial
// arithmetic, term comparison, S-pair construction, or matrix composition.
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
    let mut head: Option<&[u32]> = None;
    for powers in polynomial.terms.keys() {
        work.tick(3 * powers.len() + 1)?;
        if head.is_none_or(|previous| reference_compare(powers, previous, order).is_gt()) {
            head = Some(powers);
        }
    }
    head.ok_or_else(|| refused("zero polynomial in checked syzygy basis"))
}

fn reference_accumulate(
    sum: &mut Terms,
    powers: Vec<u32>,
    coefficient: BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(powers.len() + 1)?;
    check_monomial(&powers)?;
    if coefficient.is_zero() {
        return Ok(());
    }
    let value = match sum.get(&powers) {
        Some(previous) => work.add(previous, &coefficient)?,
        None => coefficient,
    };
    if value.is_zero() {
        sum.remove(&powers);
    } else {
        if !sum.contains_key(&powers) {
            if sum.len() >= MAX_TERMS {
                return Err(refused("reference syzygy term limit exceeded"));
            }
            work.allocate(256 + powers.len() * 4)?;
        }
        sum.insert(powers, value);
    }
    Ok(())
}

fn reference_product(
    sum: &mut Terms,
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    negative: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    for (left_powers, left_value) in &left.terms {
        for (right_powers, right_value) in &right.terms {
            work.tick(left_powers.len() + 1)?;
            let mut powers = reserve(left_powers.len(), work)?;
            for (&a, &b) in left_powers.iter().zip(right_powers) {
                powers.push(
                    a.checked_add(b)
                        .ok_or_else(|| refused("syzygy degree overflow"))?,
                );
            }
            check_monomial(&powers)?;
            let value = work.multiply(left_value, right_value)?;
            let value = if negative {
                work.subtract(&BigRational::zero(), &value)?
            } else {
                value
            };
            reference_accumulate(sum, powers, value, work)?;
        }
    }
    Ok(())
}

fn reference_shift(
    sum: &mut Terms,
    source: &MultivariatePoly,
    shift: &[u32],
    negative: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    for (original, value) in &source.terms {
        work.tick(original.len() + 1)?;
        let mut powers = reserve(original.len(), work)?;
        for (&a, &b) in original.iter().zip(shift) {
            powers.push(
                a.checked_add(b)
                    .ok_or_else(|| refused("syzygy shift overflow"))?,
            );
        }
        check_monomial(&powers)?;
        let coefficient = if negative {
            work.subtract(&BigRational::zero(), value)?
        } else {
            work.clone_value(value)?
        };
        reference_accumulate(sum, powers, coefficient, work)?;
    }
    Ok(())
}

fn reference_row(
    expected: &[Terms],
    inputs: &[MultivariatePoly],
    proposed: &[Vec<MultivariatePoly>],
    next: &mut usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(expected.len())?;
    if expected.iter().all(Terms::is_empty) {
        return Ok(());
    }
    let row = proposed
        .get(*next)
        .ok_or_else(|| refused("syzygy certificate omitted a complete relation"))?;
    for (terms, actual) in expected.iter().zip(row) {
        if terms.len() != actual.terms.len() {
            return Err(refused("syzygy certificate changed a composed relation"));
        }
        for ((powers, value), (actual_powers, actual_value)) in terms.iter().zip(&actual.terms) {
            let limbs = value.numer().limb_count()
                + value.denom().limb_count()
                + actual_value.numer().limb_count()
                + actual_value.denom().limb_count();
            work.tick(powers.len() + limbs as usize + 1)?;
            if powers != actual_powers || value != actual_value {
                return Err(refused(
                    "syzygy certificate changed exact relation coefficients",
                ));
            }
        }
    }
    // Independently check the published mathematical identity itself as well
    // as its complete-generation formula; a valid subset is still insufficient.
    let mut residual = Terms::new();
    for (coefficient, input) in row.iter().zip(inputs) {
        reference_product(&mut residual, coefficient, input, false, work)?;
    }
    if !residual.is_empty() {
        return Err(PolyError::IdentityCheckFailed(
            "proposed syzygy does not annihilate the original inputs".into(),
        ));
    }
    *next += 1;
    Ok(())
}

fn verify_compositions(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    certificate: &PolynomialSyzygyCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let groebner = &certificate.groebner;
    let a = &groebner.candidate.input_ideal_witnesses;
    let mut next = 0;
    for (index, coefficients) in groebner.input_basis_witnesses.iter().enumerate() {
        let mut expected = reserve(inputs.len(), work)?;
        for coordinate in 0..inputs.len() {
            let mut value = Terms::new();
            if coordinate == index {
                work.allocate(generators.len() * 4)?;
                reference_accumulate(
                    &mut value,
                    vec![0; generators.len()],
                    BigRational::one(),
                    work,
                )?;
            }
            for (basis_index, coefficient) in coefficients.iter().enumerate() {
                reference_product(
                    &mut value,
                    coefficient,
                    &a[basis_index][coordinate],
                    true,
                    work,
                )?;
            }
            expected.push(value);
        }
        reference_row(&expected, inputs, &certificate.syzygies, &mut next, work)?;
    }
    for pair in &groebner.s_pairs {
        let left = reference_head(
            &groebner.candidate.basis[pair.left],
            groebner.candidate.order,
            work,
        )?;
        let right = reference_head(
            &groebner.candidate.basis[pair.right],
            groebner.candidate.order,
            work,
        )?;
        let mut left_shift = reserve(generators.len(), work)?;
        let mut right_shift = reserve(generators.len(), work)?;
        for (&a, &b) in left.iter().zip(right) {
            work.tick(1)?;
            left_shift.push(b.saturating_sub(a));
            right_shift.push(a.saturating_sub(b));
        }
        let mut expected = reserve(inputs.len(), work)?;
        for coordinate in 0..inputs.len() {
            let mut value = Terms::new();
            reference_shift(
                &mut value,
                &a[pair.left][coordinate],
                &left_shift,
                false,
                work,
            )?;
            reference_shift(
                &mut value,
                &a[pair.right][coordinate],
                &right_shift,
                true,
                work,
            )?;
            for (basis_index, coefficient) in pair.basis_coefficients.iter().enumerate() {
                reference_product(
                    &mut value,
                    coefficient,
                    &a[basis_index][coordinate],
                    true,
                    work,
                )?;
            }
            expected.push(value);
        }
        reference_row(&expected, inputs, &certificate.syzygies, &mut next, work)?;
    }
    if next != certificate.syzygies.len() {
        return Err(refused(
            "syzygy certificate has extra or zero relation rows",
        ));
    }
    work.tick(1)
}

/// Independently check the full original-coordinate generating-set claim.
/// The complete underlying Groebner certificate is rechecked, never replaced
/// by a stored acceptance flag or merely by checking that each row sums to zero.
pub fn verify_polynomial_syzygies(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    order: TermOrder,
    certificate: &PolynomialSyzygyCertificate,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialSyzygies, PolyError> {
    let mut work = PolynomialWork::new(verifier_meter);
    admit(inputs, generators, &mut work)?;
    let possible_rows = preflight_certificate(
        inputs.len(),
        generators,
        order,
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
            "syzygy output dimensions exceed or change the original coordinates",
        ));
    }
    for polynomial in certificate.syzygies.iter().flatten() {
        check_polynomial(polynomial, generators, &mut work)?;
    }
    let basis = verify_complete_groebner_certificate(inputs, &certificate.groebner, &mut work)?;
    verify_compositions(inputs, generators, certificate, &mut work)?;
    let mut syzygies = reserve(certificate.syzygies.len(), &mut work)?;
    for row in &certificate.syzygies {
        let mut copied = reserve(inputs.len(), &mut work)?;
        for polynomial in row {
            copied.push(copy_poly(polynomial, &mut work)?);
        }
        syzygies.push(copied);
    }
    work.allocate(std::mem::size_of::<VerifiedPolynomialSyzygies>() + ring_storage(generators))?;
    let generators = generators.to_vec();
    work.tick(1)?;
    Ok(VerifiedPolynomialSyzygies {
        generators,
        basis,
        syzygies,
    })
}

/// Compute and independently verify all polynomial relations on the exact
/// ordered inputs. Cancellation or resource refusal in either phase prevents
/// publication, including empty and all-zero input cases.
pub fn checked_polynomial_syzygies(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    order: TermOrder,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialSyzygies, PolyError> {
    let mut generator = PolynomialWork::new(generator_meter);
    let mut verifier = PolynomialWork::new(verifier_meter);
    let certificate = bounded_syzygy_certificate(inputs, generators, order, &mut generator)?;
    let accepted =
        verify_polynomial_syzygies(inputs, generators, order, &certificate, &mut verifier)?;
    generator.tick(1)?;
    verifier.tick(1)?;
    Ok(accepted)
}
