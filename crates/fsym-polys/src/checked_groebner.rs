//! Bounded, metered Buchberger generation with complete independent checking.
//!
//! Every generated polynomial retains its exact cofactors in the original
//! input ideal. Search, autoreduction, and certificate construction share one
//! cumulative work meter. The existing dependency-separated proof kernel
//! checks both ideal inclusions, reducedness, and every strict S-pair before
//! a `VerifiedGroebnerBasis` is published.
//!
//! This synchronous lane admits at most eight variables, 64 input/intermediate
//! basis elements, 4096 terms per polynomial, total degree 256, and 4096-bit
//! rational coefficients. The division work envelope additionally limits each
//! phase to twenty million compute units and 128 MiB of cumulative logical
//! allocation. Certificate dimensions have their existing independent bounds.
//! Separate meters do not themselves establish a protected runtime reservation.

use crate::division::{
    MAX_BASIS, PolynomialWork, Terms, accumulate, check_monomial, copy_poly, divide_with_work,
    divides, multiply_polynomials, refused, ring_storage, same_ring, validate_poly,
};
use crate::fglm::certificate_with_inclusion;
use crate::groebner::{
    CompleteGroebnerBasisCertificate, GroebnerSPairCertificate, VerifiedGroebnerBasis,
    complete_certificate_dimensions, verify_complete_groebner_certificate,
};
use crate::{MultivariatePoly, PolyError, TermOrder};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};

struct TrackedPolynomial {
    polynomial: MultivariatePoly,
    // `polynomial = sum(input_coefficients[i] * original_input[i])`.
    input_coefficients: Vec<MultivariatePoly>,
}

fn zero_polynomial<M: BudgetMeter>(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    work.allocate(std::mem::size_of::<MultivariatePoly>() + ring_storage(generators))?;
    Ok(MultivariatePoly::zero(generators.to_vec()))
}

fn seed<M: BudgetMeter>(
    polynomial: &MultivariatePoly,
    index: usize,
    input_count: usize,
    work: &mut PolynomialWork<'_, M>,
) -> Result<TrackedPolynomial, PolyError> {
    work.allocate(input_count * std::mem::size_of::<MultivariatePoly>())?;
    let mut coefficients = Vec::with_capacity(input_count);
    for i in 0..input_count {
        let mut coefficient = zero_polynomial(&polynomial.generators, work)?;
        if i == index {
            work.allocate(256 + polynomial.generators.len() * 4)?;
            coefficient
                .terms
                .insert(vec![0; polynomial.generators.len()], BigRational::one());
        }
        coefficients.push(coefficient);
    }
    Ok(TrackedPolynomial {
        polynomial: copy_poly(polynomial, work)?,
        input_coefficients: coefficients,
    })
}

fn copy_tracked<M: BudgetMeter>(
    polynomial: &TrackedPolynomial,
    work: &mut PolynomialWork<'_, M>,
) -> Result<TrackedPolynomial, PolyError> {
    work.allocate(polynomial.input_coefficients.len() * std::mem::size_of::<MultivariatePoly>())?;
    Ok(TrackedPolynomial {
        polynomial: copy_poly(&polynomial.polynomial, work)?,
        input_coefficients: polynomial
            .input_coefficients
            .iter()
            .map(|coefficient| copy_poly(coefficient, work))
            .collect::<Result<_, _>>()?,
    })
}

fn subtract<M: BudgetMeter>(
    a: &MultivariatePoly,
    b: &MultivariatePoly,
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    same_ring(&a.generators, &b.generators)?;
    let mut result = copy_poly(a, work)?;
    for (monomial, coefficient) in &b.terms {
        work.allocate(monomial.len() * 4)?;
        let negative = work.subtract(&BigRational::zero(), coefficient)?;
        accumulate(&mut result.terms, monomial.clone(), negative, work)?;
    }
    Ok(result)
}

fn monic<M: BudgetMeter>(
    mut polynomial: TrackedPolynomial,
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<TrackedPolynomial, PolyError> {
    work.tick(polynomial.polynomial.terms.len() * polynomial.polynomial.generators.len().max(1))?;
    let (_, leading) = polynomial
        .polynomial
        .leading_term(order)
        .ok_or_else(|| refused("zero tracked polynomial cannot be made monic"))?;
    let leading = work.clone_value(leading)?;
    if !leading.is_one() {
        for value in polynomial.polynomial.terms.values_mut() {
            *value = work.divide(value, &leading)?;
        }
        for coefficient in &mut polynomial.input_coefficients {
            for value in coefficient.terms.values_mut() {
                *value = work.divide(value, &leading)?;
            }
        }
    }
    work.tick(1)?;
    Ok(polynomial)
}

fn reduce<M: BudgetMeter>(
    mut dividend: TrackedPolynomial,
    divisors: &[TrackedPolynomial],
    skipped: Option<usize>,
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<TrackedPolynomial, PolyError> {
    work.allocate(divisors.len() * std::mem::size_of::<MultivariatePoly>())?;
    let polynomials = divisors
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != skipped)
        .map(|(_, divisor)| copy_poly(&divisor.polynomial, work))
        .collect::<Result<Vec<_>, _>>()?;
    let (quotients, remainder) = divide_with_work(&dividend.polynomial, &polynomials, order, work)?;
    for (quotient, (_, divisor)) in quotients.iter().zip(
        divisors
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != skipped),
    ) {
        work.tick(1)?;
        if quotient.is_zero() {
            continue;
        }
        for (target, coefficient) in dividend
            .input_coefficients
            .iter_mut()
            .zip(&divisor.input_coefficients)
        {
            work.tick(1)?;
            if coefficient.is_zero() {
                continue;
            }
            let product = multiply_polynomials(quotient, coefficient, work)?;
            *target = subtract(target, &product, work)?;
        }
    }
    dividend.polynomial = remainder;
    work.tick(1)?;
    Ok(dividend)
}

fn shifted_combination<M: BudgetMeter>(
    a: &MultivariatePoly,
    a_shift: &[u32],
    a_scale: &BigRational,
    b: &MultivariatePoly,
    b_shift: &[u32],
    b_scale: &BigRational,
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    same_ring(&a.generators, &b.generators)?;
    let mut terms = Terms::new();
    for (polynomial, shift, scale) in [(a, a_shift, a_scale), (b, b_shift, b_scale)] {
        for (monomial, coefficient) in &polynomial.terms {
            work.allocate(monomial.len() * 4)?;
            let shifted: Vec<_> = monomial.iter().zip(shift).map(|(a, b)| a + b).collect();
            check_monomial(&shifted)?;
            let value = work.multiply(coefficient, scale)?;
            accumulate(&mut terms, shifted, value, work)?;
        }
    }
    work.allocate(ring_storage(&a.generators))?;
    Ok(MultivariatePoly {
        generators: a.generators.clone(),
        terms,
    })
}

fn s_pair<M: BudgetMeter>(
    a: &TrackedPolynomial,
    b: &TrackedPolynomial,
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<TrackedPolynomial, PolyError> {
    let width = a.polynomial.generators.len();
    work.tick((a.polynomial.terms.len() + b.polynomial.terms.len()) * width.max(1))?;
    let (a_head, a_value) = a
        .polynomial
        .leading_term(order)
        .ok_or_else(|| refused("zero left S-pair polynomial"))?;
    let (b_head, b_value) = b
        .polynomial
        .leading_term(order)
        .ok_or_else(|| refused("zero right S-pair polynomial"))?;
    work.allocate(width * 12)?;
    let common: Vec<_> = a_head.iter().zip(b_head).map(|(a, b)| *a.max(b)).collect();
    check_monomial(&common)?;
    let a_shift: Vec<_> = common.iter().zip(a_head).map(|(a, b)| a - b).collect();
    let b_shift: Vec<_> = common.iter().zip(b_head).map(|(a, b)| a - b).collect();
    let a_scale = work.divide(&BigRational::one(), a_value)?;
    let inverse_b = work.divide(&BigRational::one(), b_value)?;
    let b_scale = work.subtract(&BigRational::zero(), &inverse_b)?;
    let polynomial = shifted_combination(
        &a.polynomial,
        &a_shift,
        &a_scale,
        &b.polynomial,
        &b_shift,
        &b_scale,
        work,
    )?;
    work.allocate(a.input_coefficients.len() * std::mem::size_of::<MultivariatePoly>())?;
    let input_coefficients = a
        .input_coefficients
        .iter()
        .zip(&b.input_coefficients)
        .map(|(a, b)| shifted_combination(a, &a_shift, &a_scale, b, &b_shift, &b_scale, work))
        .collect::<Result<_, _>>()?;
    Ok(TrackedPolynomial {
        polynomial,
        input_coefficients,
    })
}

fn reduced_basis<M: BudgetMeter>(
    basis: Vec<TrackedPolynomial>,
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<Vec<TrackedPolynomial>, PolyError> {
    work.allocate(basis.len() * (std::mem::size_of::<Vec<u32>>() + 1))?;
    let mut heads = Vec::with_capacity(basis.len());
    for item in &basis {
        work.tick(item.polynomial.terms.len() * item.polynomial.generators.len().max(1))?;
        work.allocate(item.polynomial.generators.len() * 4)?;
        heads.push(
            item.polynomial
                .leading_monomial(order)
                .ok_or_else(|| refused("zero polynomial in Buchberger basis"))?,
        );
    }
    let mut keep = Vec::with_capacity(basis.len());
    for (i, head) in heads.iter().enumerate() {
        work.tick(heads.len() * head.len().max(1))?;
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
    work.allocate(minimal.len() * std::mem::size_of::<TrackedPolynomial>())?;
    let mut result = Vec::with_capacity(minimal.len());
    for i in 0..minimal.len() {
        let item = copy_tracked(&minimal[i], work)?;
        let remainder = reduce(item, &minimal, Some(i), order, work)?;
        if remainder.polynomial.is_zero() {
            return Err(refused("minimal Groebner generator reduced to zero"));
        }
        result.push(monic(remainder, order, work)?);
    }
    let max_terms = result
        .iter()
        .map(|p| p.polynomial.terms.len())
        .max()
        .unwrap_or(1);
    let width = result
        .first()
        .map_or(1, |p| p.polynomial.generators.len().max(1));
    work.tick(result.len() * result.len() * max_terms * width)?;
    result.sort_unstable_by(|a, b| {
        let (a, _) = a
            .polynomial
            .leading_term(order)
            .expect("nonzero reduced polynomial");
        let (b, _) = b
            .polynomial
            .leading_term(order)
            .expect("nonzero reduced polynomial");
        order.compare_monomials(b, a)
    });
    Ok(result)
}

/// Generate an untrusted complete-basis candidate with bounded, cumulative work.
/// No acceptance is granted until the independent kernel verifies this witness.
pub fn bounded_groebner_certificate(
    initial: &[MultivariatePoly],
    order: TermOrder,
    generator_meter: &mut impl BudgetMeter,
) -> Result<CompleteGroebnerBasisCertificate, PolyError> {
    let mut work = PolynomialWork::new(generator_meter);
    work.tick(1)?;
    if initial.len() > MAX_BASIS {
        return Err(refused("Groebner input basis exceeds 64 elements"));
    }
    let ring = initial.first().map_or(&[][..], |p| p.generators.as_slice());
    for polynomial in initial {
        validate_poly(polynomial, &mut work)?;
        same_ring(ring, &polynomial.generators)?;
    }
    work.allocate(MAX_BASIS * std::mem::size_of::<TrackedPolynomial>())?;
    let mut basis = Vec::with_capacity(MAX_BASIS);
    for (index, polynomial) in initial.iter().enumerate() {
        work.tick(1)?;
        if polynomial.is_zero() {
            continue;
        }
        let item = seed(polynomial, index, initial.len(), &mut work)?;
        let remainder = reduce(item, &basis, None, order, &mut work)?;
        if !remainder.polynomial.is_zero() {
            let item = monic(remainder, order, &mut work)?;
            if item.polynomial.is_one() {
                basis.clear();
                basis.push(item);
                break;
            }
            basis.push(item);
        }
    }
    // Every pair is inserted once. The static frontier is at most 64*63/2.
    const MAX_PAIRS: usize = MAX_BASIS * (MAX_BASIS - 1) / 2;
    work.allocate(MAX_PAIRS * std::mem::size_of::<(usize, usize)>())?;
    let mut pending = Vec::with_capacity(MAX_PAIRS);
    for i in 0..basis.len() {
        for j in (i + 1)..basis.len() {
            work.tick(1)?;
            pending.push((i, j));
        }
    }
    while let Some((i, j)) = pending.pop() {
        let candidate = s_pair(&basis[i], &basis[j], order, &mut work)?;
        let remainder = reduce(candidate, &basis, None, order, &mut work)?;
        if remainder.polynomial.is_zero() {
            continue;
        }
        let item = monic(remainder, order, &mut work)?;
        if item.polynomial.is_one() {
            basis.clear();
            basis.push(item);
            break;
        }
        if basis.len() >= MAX_BASIS {
            return Err(refused("Groebner intermediate basis exceeds 64 elements"));
        }
        let index = basis.len();
        for old in 0..index {
            work.tick(1)?;
            pending.push((old, index));
        }
        basis.push(item);
    }
    let reduced = reduced_basis(basis, order, &mut work)?;
    let pairs = complete_certificate_dimensions(initial.len(), reduced.len())?;
    let objects =
        2 * initial.len() * reduced.len() + pairs * reduced.len() + initial.len() + reduced.len();
    work.allocate(
        objects.max(1) * std::mem::size_of::<MultivariatePoly>()
            + pairs * std::mem::size_of::<GroebnerSPairCertificate>(),
    )?;
    let mut candidate = Vec::with_capacity(reduced.len());
    let mut inclusion = Vec::with_capacity(reduced.len());
    for item in reduced {
        candidate.push(item.polynomial);
        inclusion.push(item.input_coefficients);
    }
    let certificate = certificate_with_inclusion(initial, &candidate, order, inclusion, &mut work)?;
    work.tick(1)?;
    Ok(certificate)
}

/// Compute and independently verify a complete reduced Groebner basis.
/// The caller supplies separate cumulative generator and verifier meters;
/// either cancellation or resource refusal prevents accepted publication.
pub fn checked_groebner_basis(
    initial: &[MultivariatePoly],
    order: TermOrder,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedGroebnerBasis, PolyError> {
    let certificate = bounded_groebner_certificate(initial, order, generator_meter)?;
    let accepted = verify_complete_groebner_certificate(initial, &certificate, verifier_meter)?;
    generator_meter.checkpoint().map_err(refused)?;
    verifier_meter.checkpoint().map_err(refused)?;
    Ok(accepted)
}
