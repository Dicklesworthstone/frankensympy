//! Bounded complete reduced Gröbner bases in an exact typed prime-field ring.
//!
//! Buchberger generation retains every basis element's original-input
//! cofactors. The independent checker uses signed integer coefficient columns
//! and a separate monomial comparator to establish both ideal inclusions,
//! monicity, reducedness, and every strict S-pair representation. In particular,
//! an arbitrary ideal-membership representation of an S-polynomial is not a
//! Gröbner-basis proof: each product must lead strictly below the pair LCM.
//!
//! Inputs and witnesses have at most eight ordered typed variables, 32 input
//! or intermediate basis elements, degree 64 and 4096 terms per polynomial.
//! Temporary products may have degree 128. Each whole generation/verification
//! phase has cumulative 20M-step / 128MiB logical-allocation caps. Refusal and
//! cancellation publish no accepted partial basis or membership conclusion.
//! These immutable in-process receipts are not portable proof-kernel objects.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field::scalar_power;
use crate::finite_field_multivariate::{
    Columns, MAX_BASIS, MAX_DEGREE, accumulate, admit_ring, admit_terms, constant,
    copied_polynomial, copy_generators, copy_terms, divide, divides, empty, identity_zero, leading,
    monomial_copy, monomial_difference, monomial_sum, product, reference_add, reference_head,
    reference_order, reference_product, same_ring, sum,
};
use crate::{
    PolyError, PrimeFieldMultivariatePolynomial, PrimeFieldMultivariateTerms, TermOrder,
    VerifiedPrimeFieldMultivariateDivision, checked_prime_field_multivariate_division,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::cmp::Ordering;

type Terms = PrimeFieldMultivariateTerms;
type Polynomial = PrimeFieldMultivariatePolynomial;
const MAX_PAIRS: usize = MAX_BASIS * (MAX_BASIS - 1) / 2;

/// Untrusted strict representation of one S-pair in the candidate basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldGroebnerSPairCertificate {
    pub left: usize,
    pub right: usize,
    pub quotients: Vec<Terms>,
}

/// Untrusted complete-basis witness for the ring and order supplied to verification.
///
/// `basis_in_input[j][i]` expresses candidate j in original input i;
/// `input_in_basis[i][j]` expresses original input i in candidate j.
/// Pairs occur exactly once in nested increasing `(left, right)` order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldGroebnerCertificate {
    pub basis: Vec<Terms>,
    pub basis_in_input: Vec<Vec<Terms>>,
    pub input_in_basis: Vec<Vec<Terms>>,
    pub s_pairs: Vec<PrimeFieldGroebnerSPairCertificate>,
}

/// Complete reduced basis of the original input ideal over the retained F_p ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPrimeFieldGroebnerBasis {
    generators: Vec<Symbol>,
    prime: u64,
    order: TermOrder,
    inputs: Vec<Polynomial>,
    basis: Vec<Polynomial>,
}

impl VerifiedPrimeFieldGroebnerBasis {
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }
    pub fn prime(&self) -> u64 {
        self.prime
    }
    pub fn order(&self) -> TermOrder {
        self.order
    }
    pub fn inputs(&self) -> &[Polynomial] {
        &self.inputs
    }
    pub fn basis(&self) -> &[Polynomial] {
        &self.basis
    }
    pub fn into_basis(self) -> Vec<Polynomial> {
        self.basis
    }
}

fn admit_request(
    generators: &[Symbol],
    prime: u64,
    initial: &[Polynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if initial.len() > MAX_BASIS {
        return Err(refused("prime-field Groebner input exceeds 32 polynomials"));
    }
    admit_ring(generators, prime, work)?;
    // Even zero polynomials retain their exact ordered typed ring and prime.
    for polynomial in initial {
        same_ring(polynomial, generators, prime, work)?;
    }
    Ok(())
}

struct TrackedPolynomial {
    polynomial: Terms,
    input_coefficients: Vec<Terms>,
}

fn seed(
    polynomial: &Polynomial,
    index: usize,
    count: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<TrackedPolynomial, PolyError> {
    work.allocate(count * std::mem::size_of::<Terms>())?;
    let mut input_coefficients = Vec::with_capacity(count);
    for i in 0..count {
        input_coefficients.push(constant(
            polynomial.generators().len(),
            u64::from(i == index),
            work,
        )?);
    }
    Ok(TrackedPolynomial {
        polynomial: copy_terms(polynomial.terms(), work)?,
        input_coefficients,
    })
}

fn copy_tracked(
    polynomial: &TrackedPolynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<TrackedPolynomial, PolyError> {
    work.allocate(polynomial.input_coefficients.len() * std::mem::size_of::<Terms>())?;
    let mut input_coefficients = Vec::with_capacity(polynomial.input_coefficients.len());
    for coefficient in &polynomial.input_coefficients {
        input_coefficients.push(copy_terms(coefficient, work)?);
    }
    Ok(TrackedPolynomial {
        polynomial: copy_terms(&polynomial.polynomial, work)?,
        input_coefficients,
    })
}

fn monic(
    mut polynomial: TrackedPolynomial,
    order: TermOrder,
    prime: u64,
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<TrackedPolynomial, PolyError> {
    let (_, coefficient) = leading(&polynomial.polynomial, order, work)?
        .ok_or_else(|| refused("zero Groebner polynomial cannot be made monic"))?;
    if coefficient != 1 {
        let inverse = scalar_power(coefficient, prime - 2, prime, work)?;
        for source in std::iter::once(&mut polynomial.polynomial)
            .chain(polynomial.input_coefficients.iter_mut())
        {
            for coefficient in source.values_mut() {
                work.tick(1)?;
                *coefficient = *coefficient * inverse % prime;
            }
        }
    }
    // Basis entries and retained witnesses must return to the published bound.
    admit_terms(
        &polynomial.polynomial,
        width,
        prime,
        MAX_DEGREE,
        false,
        work,
    )?;
    for coefficient in &polynomial.input_coefficients {
        admit_terms(coefficient, width, prime, MAX_DEGREE, false, work)?;
    }
    work.tick(1)?;
    Ok(polynomial)
}

fn reduce_tracked(
    mut dividend: TrackedPolynomial,
    divisors: &[TrackedPolynomial],
    skipped: Option<usize>,
    order: TermOrder,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<TrackedPolynomial, PolyError> {
    work.allocate(divisors.len() * std::mem::size_of::<&Terms>())?;
    let views: Vec<_> = divisors
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != skipped)
        .map(|(_, item)| &item.polynomial)
        .collect();
    let division = divide(&dividend.polynomial, &views, order, prime, work)?;
    for (quotient, (_, divisor)) in division.quotients.iter().zip(
        divisors
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != skipped),
    ) {
        work.tick(1)?;
        if quotient.is_empty() {
            continue;
        }
        for (target, coefficient) in dividend
            .input_coefficients
            .iter_mut()
            .zip(&divisor.input_coefficients)
        {
            work.tick(1)?;
            if !coefficient.is_empty() {
                let multiple = product(quotient, coefficient, prime, work)?;
                *target = sum(target, &multiple, true, prime, work)?;
            }
        }
    }
    dividend.polynomial = division.remainder;
    work.tick(1)?;
    Ok(dividend)
}

fn pair_shifts(
    left: &Terms,
    right: &Terms,
    order: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(Vec<u32>, Vec<u32>), PolyError> {
    let (a, ca) =
        leading(left, order, work)?.ok_or_else(|| refused("zero left Groebner S-pair input"))?;
    let (b, cb) =
        leading(right, order, work)?.ok_or_else(|| refused("zero right Groebner S-pair input"))?;
    if ca != 1 || cb != 1 {
        return Err(refused("prime-field Groebner basis must be monic"));
    }
    work.tick(a.len())?;
    work.allocate(std::mem::size_of_val(a.as_slice()))?;
    let common: Vec<_> = a.iter().zip(b).map(|(x, y)| *x.max(y)).collect();
    Ok((
        monomial_difference(&common, a, work)?,
        monomial_difference(&common, b, work)?,
    ))
}

fn shifted_difference(
    left: &Terms,
    left_shift: &[u32],
    right: &Terms,
    right_shift: &[u32],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (source, shift, subtract) in [(left, left_shift, false), (right, right_shift, true)] {
        for (monomial, coefficient) in source {
            let shifted = monomial_sum(monomial, shift, work)?;
            let value = if subtract {
                prime - coefficient
            } else {
                *coefficient
            };
            accumulate(&mut result, shifted, value, prime, work)?;
        }
    }
    Ok(result)
}

fn s_pair(
    left: &TrackedPolynomial,
    right: &TrackedPolynomial,
    order: TermOrder,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<TrackedPolynomial, PolyError> {
    let (a, b) = pair_shifts(&left.polynomial, &right.polynomial, order, work)?;
    let polynomial = shifted_difference(&left.polynomial, &a, &right.polynomial, &b, prime, work)?;
    work.allocate(left.input_coefficients.len() * std::mem::size_of::<Terms>())?;
    let mut input_coefficients = Vec::with_capacity(left.input_coefficients.len());
    for (left, right) in left
        .input_coefficients
        .iter()
        .zip(&right.input_coefficients)
    {
        input_coefficients.push(shifted_difference(left, &a, right, &b, prime, work)?);
    }
    Ok(TrackedPolynomial {
        polynomial,
        input_coefficients,
    })
}

fn is_unit(polynomial: &Terms) -> bool {
    polynomial.len() == 1
        && polynomial
            .keys()
            .next()
            .is_some_and(|m| m.iter().all(|e| *e == 0))
}

fn reduced_basis(
    basis: Vec<TrackedPolynomial>,
    order: TermOrder,
    prime: u64,
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<TrackedPolynomial>, PolyError> {
    work.allocate(basis.len() * std::mem::size_of::<Vec<u32>>())?;
    let mut heads = Vec::with_capacity(basis.len());
    for item in &basis {
        let (head, _) = leading(&item.polynomial, order, work)?
            .ok_or_else(|| refused("zero polynomial in Groebner basis"))?;
        heads.push(monomial_copy(head, work)?);
    }
    work.allocate(basis.len())?;
    let mut keep = Vec::with_capacity(basis.len());
    for (i, head) in heads.iter().enumerate() {
        work.tick(heads.len() * width)?;
        keep.push(!heads.iter().enumerate().any(|(j, other)| {
            i != j
                && if head == other {
                    j < i
                } else {
                    divides(other, head)
                }
        }));
    }
    work.allocate(basis.len() * std::mem::size_of::<TrackedPolynomial>())?;
    let minimal: Vec<_> = basis
        .into_iter()
        .zip(keep)
        .filter_map(|(item, keep)| keep.then_some(item))
        .collect();
    work.allocate(minimal.len() * std::mem::size_of::<(Vec<u32>, TrackedPolynomial)>())?;
    let mut sorted = Vec::with_capacity(minimal.len());
    for i in 0..minimal.len() {
        let source = copy_tracked(&minimal[i], work)?;
        let remainder = reduce_tracked(source, &minimal, Some(i), order, prime, work)?;
        let item = monic(remainder, order, prime, width, work)?;
        let (head, _) = leading(&item.polynomial, order, work)?
            .ok_or_else(|| refused("zero reduced Groebner generator"))?;
        sorted.push((monomial_copy(head, work)?, item));
    }
    work.tick(sorted.len() * sorted.len() * width)?;
    sorted.sort_unstable_by(|(a, _), (b, _)| order.compare_monomials(b, a));
    work.allocate(sorted.len() * std::mem::size_of::<TrackedPolynomial>())?;
    Ok(sorted.into_iter().map(|(_, item)| item).collect())
}

pub(crate) fn complete_candidate(
    initial: &[Polynomial],
    basis: Vec<Terms>,
    basis_in_input: Vec<Vec<Terms>>,
    order: TermOrder,
    prime: u64,
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldGroebnerCertificate, PolyError> {
    work.allocate(basis.len() * std::mem::size_of::<&Terms>())?;
    let views: Vec<_> = basis.iter().collect();
    work.allocate(initial.len() * std::mem::size_of::<Vec<Terms>>())?;
    let mut input_in_basis = Vec::with_capacity(initial.len());
    for input in initial {
        let division = divide(input.terms(), &views, order, prime, work)?;
        if !division.remainder.is_empty() {
            return Err(refused(
                "generated Groebner basis omitted an input generator",
            ));
        }
        input_in_basis.push(division.quotients);
    }
    let pairs = basis.len() * basis.len().saturating_sub(1) / 2;
    work.allocate(pairs * std::mem::size_of::<PrimeFieldGroebnerSPairCertificate>())?;
    let mut s_pairs = Vec::with_capacity(pairs);
    for left in 0..basis.len() {
        for right in (left + 1)..basis.len() {
            let (a, b) = pair_shifts(&basis[left], &basis[right], order, work)?;
            let source = shifted_difference(&basis[left], &a, &basis[right], &b, prime, work)?;
            let division = divide(&source, &views, order, prime, work)?;
            if !division.remainder.is_empty() {
                return Err(refused(
                    "prime-field candidate is not a complete Groebner basis",
                ));
            }
            s_pairs.push(PrimeFieldGroebnerSPairCertificate {
                left,
                right,
                quotients: division.quotients,
            });
        }
    }
    let certificate = PrimeFieldGroebnerCertificate {
        basis,
        basis_in_input,
        input_in_basis,
        s_pairs,
    };
    admit_certificate(&certificate, initial.len(), width, prime, work)?;
    Ok(certificate)
}

/// Generate a bounded untrusted complete-basis witness; no acceptance is implied.
/// The explicit ring also binds the zero ideal when `initial` is empty.
pub fn bounded_prime_field_groebner_certificate(
    generators: &[Symbol],
    prime: u64,
    initial: &[Polynomial],
    order: TermOrder,
    generator: &mut impl BudgetMeter,
) -> Result<PrimeFieldGroebnerCertificate, PolyError> {
    let mut work = PolynomialWork::new(generator);
    admit_request(generators, prime, initial, &mut work)?;
    work.allocate(MAX_BASIS * std::mem::size_of::<TrackedPolynomial>())?;
    let mut basis = Vec::with_capacity(MAX_BASIS);
    for (index, input) in initial.iter().enumerate() {
        work.tick(1)?;
        if input.is_zero() {
            continue;
        }
        let source = seed(input, index, initial.len(), &mut work)?;
        let remainder = reduce_tracked(source, &basis, None, order, prime, &mut work)?;
        if !remainder.polynomial.is_empty() {
            let item = monic(remainder, order, prime, generators.len(), &mut work)?;
            if is_unit(&item.polynomial) {
                basis.clear();
                basis.push(item);
                break;
            }
            basis.push(item);
        }
    }
    work.allocate(MAX_PAIRS * std::mem::size_of::<(usize, usize)>())?;
    let mut pending = Vec::with_capacity(MAX_PAIRS);
    for i in 0..basis.len() {
        for j in (i + 1)..basis.len() {
            work.tick(1)?;
            pending.push((i, j));
        }
    }
    while let Some((i, j)) = pending.pop() {
        let source = s_pair(&basis[i], &basis[j], order, prime, &mut work)?;
        let remainder = reduce_tracked(source, &basis, None, order, prime, &mut work)?;
        if remainder.polynomial.is_empty() {
            continue;
        }
        let item = monic(remainder, order, prime, generators.len(), &mut work)?;
        if is_unit(&item.polynomial) {
            basis.clear();
            basis.push(item);
            break;
        }
        if basis.len() == MAX_BASIS {
            return Err(refused(
                "prime-field Groebner intermediate basis exceeds 32 polynomials",
            ));
        }
        for old in 0..basis.len() {
            work.tick(1)?;
            pending.push((old, basis.len()));
        }
        basis.push(item);
    }
    let reduced = reduced_basis(basis, order, prime, generators.len(), &mut work)?;
    work.allocate(
        reduced.len() * (std::mem::size_of::<Terms>() + std::mem::size_of::<Vec<Terms>>()),
    )?;
    let mut candidate = Vec::with_capacity(reduced.len());
    let mut inclusion = Vec::with_capacity(reduced.len());
    for item in reduced {
        candidate.push(item.polynomial);
        inclusion.push(item.input_coefficients);
    }
    let certificate = complete_candidate(
        initial,
        candidate,
        inclusion,
        order,
        prime,
        generators.len(),
        &mut work,
    )?;
    work.tick(1)?;
    Ok(certificate)
}

fn admit_certificate(
    certificate: &PrimeFieldGroebnerCertificate,
    input_count: usize,
    width: usize,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    let count = certificate.basis.len();
    if count > MAX_BASIS {
        return Err(refused(
            "prime-field Groebner witness exceeds 32 basis elements",
        ));
    }
    let pairs = count * count.saturating_sub(1) / 2;
    if certificate.basis_in_input.len() != count
        || certificate.input_in_basis.len() != input_count
        || certificate.s_pairs.len() != pairs
    {
        return Err(refused("prime-field Groebner witness dimensions"));
    }
    // Check all outer dimensions before traversing caller-owned polynomial maps.
    for row in &certificate.basis_in_input {
        work.tick(1)?;
        if row.len() != input_count {
            return Err(refused("prime-field basis-in-input witness width"));
        }
    }
    for row in &certificate.input_in_basis {
        work.tick(1)?;
        if row.len() != count {
            return Err(refused("prime-field input-in-basis witness width"));
        }
    }
    for pair in &certificate.s_pairs {
        work.tick(1)?;
        if pair.quotients.len() != count {
            return Err(refused("prime-field S-pair witness width"));
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
    Ok(())
}

// Independent reference verification: none of these routines call sparse
// modular sum/product/division, tracked generation, or its monomial comparator.
fn reference_columns(
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Columns, PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    Ok(Columns::new())
}

fn reference_combination(
    target: &Terms,
    coefficients: &[Terms],
    sources: &[&Terms],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let mut columns = reference_columns(work)?;
    reference_add(&mut columns, target, 1, work)?;
    for (coefficient, source) in coefficients.iter().zip(sources) {
        work.tick(1)?;
        reference_product(&mut columns, coefficient, source, -1, work)?;
    }
    identity_zero(&columns, prime, work)
}

fn reference_reduced_heads<'a>(
    basis: &'a [Terms],
    order: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<&'a Vec<u32>>, PolyError> {
    work.allocate(basis.len() * std::mem::size_of::<&Vec<u32>>())?;
    let mut heads = Vec::with_capacity(basis.len());
    for polynomial in basis {
        let (head, coefficient) = reference_head(polynomial, order, work)?
            .ok_or_else(|| refused("zero polynomial in prime-field Groebner witness"))?;
        if coefficient != 1 {
            return Err(PolyError::IdentityCheckFailed(
                "prime-field Groebner basis is not monic".into(),
            ));
        }
        heads.push(head);
    }
    for (i, polynomial) in basis.iter().enumerate() {
        for (j, head) in heads.iter().enumerate() {
            if i == j {
                continue;
            }
            for monomial in polynomial.keys() {
                work.tick(monomial.len())?;
                if head.iter().zip(monomial).all(|(a, b)| a <= b) {
                    return Err(PolyError::IdentityCheckFailed(
                        "prime-field Groebner basis is not reduced".into(),
                    ));
                }
            }
        }
    }
    Ok(heads)
}

fn reference_shift(
    common: &[u32],
    head: &[u32],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    work.tick(common.len())?;
    work.allocate(std::mem::size_of::<Terms>() + 128 + std::mem::size_of_val(common))?;
    let mut shift = Vec::with_capacity(common.len());
    for (a, b) in common.iter().zip(head) {
        shift.push(
            a.checked_sub(*b)
                .ok_or_else(|| refused("invalid reference S-pair shift"))?,
        );
    }
    let mut result = Terms::new();
    result.insert(shift, 1);
    Ok(result)
}

fn reference_s_pairs(
    certificate: &PrimeFieldGroebnerCertificate,
    heads: &[&Vec<u32>],
    order: TermOrder,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let mut index = 0;
    for left in 0..heads.len() {
        for right in (left + 1)..heads.len() {
            work.tick(1)?;
            let pair = &certificate.s_pairs[index];
            index += 1;
            if pair.left != left || pair.right != right {
                return Err(refused("missing, repeated or reordered prime-field S-pair"));
            }
            work.tick(heads[left].len())?;
            work.allocate(std::mem::size_of_val(heads[left].as_slice()))?;
            let common: Vec<_> = heads[left]
                .iter()
                .zip(heads[right])
                .map(|(a, b)| if a >= b { *a } else { *b })
                .collect();
            let a = reference_shift(&common, heads[left], work)?;
            let b = reference_shift(&common, heads[right], work)?;
            let mut columns = reference_columns(work)?;
            reference_product(&mut columns, &a, &certificate.basis[left], 1, work)?;
            reference_product(&mut columns, &b, &certificate.basis[right], -1, work)?;
            for ((quotient, polynomial), head) in
                pair.quotients.iter().zip(&certificate.basis).zip(heads)
            {
                // The identity alone is vacuous: the defining two S-pair
                // multiples always reconstruct it. Strictly smaller products
                // are essential for Buchberger's criterion.
                for monomial in quotient.keys() {
                    work.tick(monomial.len())?;
                    work.allocate(std::mem::size_of_val(monomial.as_slice()))?;
                    let mut product_head = Vec::with_capacity(monomial.len());
                    for (a, b) in monomial.iter().zip(head.iter()) {
                        product_head.push(
                            a.checked_add(*b)
                                .ok_or_else(|| refused("reference S-pair exponent bound"))?,
                        );
                    }
                    if reference_order(&product_head, &common, order) != Ordering::Less {
                        return Err(PolyError::IdentityCheckFailed(
                            "S-pair representation is not strictly below its LCM".into(),
                        ));
                    }
                }
                reference_product(&mut columns, quotient, polynomial, -1, work)?;
            }
            identity_zero(&columns, prime, work)?;
        }
    }
    Ok(())
}

fn verify_candidate(
    generators: &[Symbol],
    prime: u64,
    initial: &[Polynomial],
    order: TermOrder,
    certificate: &PrimeFieldGroebnerCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_request(generators, prime, initial, work)?;
    admit_certificate(certificate, initial.len(), generators.len(), prime, work)?;
    let heads = reference_reduced_heads(&certificate.basis, order, work)?;
    work.allocate((initial.len() + certificate.basis.len()) * std::mem::size_of::<&Terms>())?;
    let inputs: Vec<_> = initial.iter().map(Polynomial::terms).collect();
    let basis: Vec<_> = certificate.basis.iter().collect();
    for (polynomial, row) in certificate.basis.iter().zip(&certificate.basis_in_input) {
        reference_combination(polynomial, row, &inputs, prime, work)?;
    }
    for (polynomial, row) in inputs.iter().zip(&certificate.input_in_basis) {
        reference_combination(polynomial, row, &basis, prime, work)?;
    }
    reference_s_pairs(certificate, &heads, order, prime, work)?;
    work.tick(1)
}

/// Independently establish exact ideal equality and the complete reduced basis.
/// The untrusted certificate cannot change the requested typed ring or order.
pub fn verify_prime_field_groebner_certificate(
    generators: &[Symbol],
    prime: u64,
    initial: &[Polynomial],
    order: TermOrder,
    certificate: &PrimeFieldGroebnerCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldGroebnerBasis, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    verify_candidate(generators, prime, initial, order, certificate, &mut work)?;
    work.allocate(
        std::mem::size_of::<VerifiedPrimeFieldGroebnerBasis>()
            + (initial.len() + certificate.basis.len()) * std::mem::size_of::<Polynomial>(),
    )?;
    let mut inputs = Vec::with_capacity(initial.len());
    let mut basis = Vec::with_capacity(certificate.basis.len());
    for polynomial in initial {
        inputs.push(copied_polynomial(
            generators,
            prime,
            polynomial.terms(),
            &mut work,
        )?);
    }
    for polynomial in &certificate.basis {
        basis.push(copied_polynomial(generators, prime, polynomial, &mut work)?);
    }
    let accepted = VerifiedPrimeFieldGroebnerBasis {
        generators: copy_generators(generators, &mut work)?,
        prime,
        order,
        inputs,
        basis,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently verify a complete reduced basis over F_p.
/// Separate cumulative meters cover generation and verification; both are
/// checkpointed immediately before returning an accepted result.
pub fn checked_prime_field_groebner_basis(
    generators: &[Symbol],
    prime: u64,
    initial: &[Polynomial],
    order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldGroebnerBasis, PolyError> {
    let certificate =
        bounded_prime_field_groebner_certificate(generators, prime, initial, order, generator)?;
    let accepted = verify_prime_field_groebner_certificate(
        generators,
        prime,
        initial,
        order,
        &certificate,
        verifier,
    )?;
    generator.checkpoint().map_err(refused)?;
    verifier.checkpoint().map_err(refused)?;
    Ok(accepted)
}

/// Validate the supplied current divisor list as a complete reduced basis,
/// then verify its ordered division identity and irreducible remainder.
///
/// Quotients retain the caller's divisor order. A zero remainder establishes
/// membership in this exact current ideal; a nonzero remainder establishes
/// nonmembership only because the full basis was checked first. Empty divisors
/// describe the zero ideal. Zero, nonmonic, redundant or incomplete divisor
/// lists are refused instead of being silently replaced by a different basis.
pub fn checked_prime_field_groebner_reduction(
    dividend: &Polynomial,
    divisors: &[Polynomial],
    order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateDivision, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let generators = dividend.generators();
    let prime = dividend.prime();
    admit_request(generators, prime, divisors, &mut generation)?;
    same_ring(dividend, generators, prime, &mut generation)?;
    generation.allocate(
        divisors.len() * (std::mem::size_of::<Terms>() + std::mem::size_of::<Vec<Terms>>()),
    )?;
    let mut candidate = Vec::with_capacity(divisors.len());
    let mut inclusion = Vec::with_capacity(divisors.len());
    for (i, divisor) in divisors.iter().enumerate() {
        let (_, coefficient) =
            leading(divisor.terms(), order, &mut generation)?.ok_or(PolyError::DivisionByZero)?;
        if coefficient != 1 {
            return Err(refused(
                "prime-field Groebner reduction requires a monic basis",
            ));
        }
        candidate.push(copy_terms(divisor.terms(), &mut generation)?);
        generation.allocate(divisors.len() * std::mem::size_of::<Terms>())?;
        let mut row = Vec::with_capacity(divisors.len());
        for j in 0..divisors.len() {
            row.push(constant(
                generators.len(),
                u64::from(i == j),
                &mut generation,
            )?);
        }
        inclusion.push(row);
    }
    let certificate = complete_candidate(
        divisors,
        candidate,
        inclusion,
        order,
        prime,
        generators.len(),
        &mut generation,
    )?;
    verify_candidate(
        generators,
        prime,
        divisors,
        order,
        &certificate,
        &mut verification,
    )?;
    // Nested work wrappers feed these same outer phase ledgers: validating the
    // basis and dividing the dividend cannot each obtain a fresh envelope.
    let accepted = checked_prime_field_multivariate_division(
        dividend,
        divisors,
        order,
        &mut generation,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
