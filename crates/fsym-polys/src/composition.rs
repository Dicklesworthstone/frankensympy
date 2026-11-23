//! Checked functional composition in an ordered rational polynomial ring.
//!
//! `f.compose(g)` substitutes `g` for the first declared generator of `f`;
//! every other axis remains a coefficient variable. Generation uses sparse
//! Horner evaluation with binary powering across absent degrees. A separate
//! checker expands ascending powers of the original replacement and checks
//! every signed coefficient of the claimed result, without using Horner or
//! the generator's multiplication/accumulation routines.
//!
//! Both source rings and all original coefficients are admitted even when the
//! outer polynomial is constant. Each phase retains cumulative 20M compute /
//! 128MiB logical allocation caps, eight variables, degree 256, 4096 terms and
//! 4096-bit coefficients. The accepted immutable receipt is in-process
//! evidence, not a portable proof-kernel object.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_TERMS, PolynomialWork, Terms, accumulate, check_monomial, copy_poly, refused, ring_storage,
    same_ring, validate_poly,
};
use crate::{MultivariatePoly as Polynomial, PolyError};
use fsym_budget::BudgetMeter;
use fsym_core::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::mem::size_of;

/// Untrusted proposed polynomial, including its complete ordered typed ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialCompositionCertificate {
    pub result: Polynomial,
}

/// Independently accepted composition bound to both complete original inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPolynomialComposition {
    outer: Polynomial,
    inner: Polynomial,
    result: Polynomial,
}

impl VerifiedPolynomialComposition {
    pub fn outer(&self) -> &Polynomial {
        &self.outer
    }

    pub fn inner(&self) -> &Polynomial {
        &self.inner
    }

    pub fn result(&self) -> &Polynomial {
        &self.result
    }

    pub fn into_result(self) -> Polynomial {
        self.result
    }
}

fn admit(
    outer: &Polynomial,
    inner: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    validate_poly(outer, work)?;
    validate_poly(inner, work)?;
    if outer.generators().is_empty() {
        return Err(refused("composition requires a primary generator"));
    }
    let bytes = ring_storage(outer.generators()) + ring_storage(inner.generators());
    work.tick(bytes)?;
    same_ring(outer.generators(), inner.generators())
}

fn empty(work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Terms, PolyError> {
    work.allocate(size_of::<Terms>())?;
    Ok(Terms::new())
}

fn one(width: usize, work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Terms, PolyError> {
    let mut terms = empty(work)?;
    work.allocate(width * size_of::<u32>() + 64)?;
    accumulate(&mut terms, vec![0; width], BigRational::one(), work)?;
    Ok(terms)
}

fn copy_terms(
    terms: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (powers, coefficient) in terms {
        work.tick(powers.len() + 1)?;
        work.allocate(powers.len() * size_of::<u32>())?;
        let coefficient = work.clone_value(coefficient)?;
        accumulate(&mut result, powers.clone(), coefficient, work)?;
    }
    Ok(result)
}

fn multiply(
    left: &Terms,
    right: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (left_powers, left_coefficient) in left {
        for (right_powers, right_coefficient) in right {
            work.tick(left_powers.len() + 1)?;
            work.allocate(left_powers.len() * size_of::<u32>())?;
            let powers = left_powers
                .iter()
                .zip(right_powers)
                .map(|(left, right)| {
                    left.checked_add(*right)
                        .ok_or_else(|| refused("composition exponent overflow"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            check_monomial(&powers)?;
            let coefficient = work.multiply(left_coefficient, right_coefficient)?;
            accumulate(&mut result, powers, coefficient, work)?;
        }
    }
    Ok(result)
}

fn power(
    source: &Terms,
    mut exponent: u32,
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = one(width, work)?;
    if exponent == 0 {
        return Ok(result);
    }
    let mut base = copy_terms(source, work)?;
    loop {
        work.tick(1)?;
        if exponent & 1 != 0 {
            result = multiply(&result, &base, work)?;
        }
        exponent >>= 1;
        if exponent == 0 {
            return Ok(result);
        }
        base = multiply(&base, &base, work)?;
    }
}

fn generate(
    outer: &Polynomial,
    inner: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    work.allocate(size_of::<BTreeMap<u32, Terms>>())?;
    let mut coefficients = BTreeMap::<u32, Terms>::new();
    for (monomial, coefficient) in outer.terms() {
        work.tick(monomial.len() + 1)?;
        work.allocate(monomial.len() * size_of::<u32>())?;
        let mut powers = monomial.clone();
        let degree = powers[0];
        powers[0] = 0;
        if !coefficients.contains_key(&degree) {
            work.allocate(128 + size_of::<Terms>())?;
        }
        let value = work.clone_value(coefficient)?;
        accumulate(coefficients.entry(degree).or_default(), powers, value, work)?;
    }
    let width = outer.generators().len();
    let mut terms = empty(work)?;
    let mut previous = coefficients
        .last_key_value()
        .map_or(0, |(degree, _)| *degree);
    for (degree, coefficient) in coefficients.iter().rev() {
        work.tick(1)?;
        let gap = previous - degree;
        if gap != 0 {
            let replacement = power(inner.terms(), gap, width, work)?;
            terms = multiply(&terms, &replacement, work)?;
        }
        for (powers, value) in coefficient {
            work.allocate(powers.len() * size_of::<u32>())?;
            let value = work.clone_value(value)?;
            accumulate(&mut terms, powers.clone(), value, work)?;
        }
        previous = *degree;
    }
    if previous != 0 {
        let replacement = power(inner.terms(), previous, width, work)?;
        terms = multiply(&terms, &replacement, work)?;
    }
    work.allocate(ring_storage(outer.generators()) + size_of::<Polynomial>())?;
    Ok(Polynomial {
        generators: outer.generators().to_vec(),
        terms,
    })
}

// These reference coefficient columns deliberately do not call multiply,
// power, copy_terms or accumulate from the generating path.
fn add_column(
    columns: &mut Terms,
    monomial: Vec<u32>,
    value: BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(monomial.len() + 1)?;
    check_monomial(&monomial)?;
    if let Some(previous) = columns.get_mut(&monomial) {
        *previous = work.add(previous, &value)?;
    } else {
        if columns.len() >= 2 * MAX_TERMS {
            return Err(refused("composition verifier coefficient-column bound"));
        }
        work.allocate(256 + monomial.len() * size_of::<u32>())?;
        columns.insert(monomial, value);
    }
    Ok(())
}

fn reference_product(
    left: &Terms,
    right: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    work.allocate(size_of::<Terms>())?;
    let mut columns = Terms::new();
    for (right_monomial, right_coefficient) in right {
        for (left_monomial, left_coefficient) in left {
            work.tick(right_monomial.len() + 1)?;
            work.allocate(right_monomial.len() * size_of::<u32>())?;
            let mut monomial = Vec::with_capacity(right_monomial.len());
            for index in 0..right_monomial.len() {
                monomial.push(
                    right_monomial[index]
                        .checked_add(left_monomial[index])
                        .ok_or_else(|| refused("composition verifier exponent overflow"))?,
                );
            }
            let value = work.multiply(right_coefficient, left_coefficient)?;
            add_column(&mut columns, monomial, value, work)?;
        }
    }
    work.tick(columns.len())?;
    columns.retain(|_, value| !value.is_zero());
    Ok(columns)
}

fn reference_identity(
    outer: &Polynomial,
    inner: &Polynomial,
    result: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    validate_poly(result, work)?;
    work.tick(ring_storage(result.generators()) + ring_storage(outer.generators()))?;
    same_ring(outer.generators(), result.generators())?;
    work.allocate(2 * size_of::<Terms>())?;
    let mut columns = Terms::new();
    for (powers, coefficient) in result.terms() {
        work.allocate(powers.len() * size_of::<u32>())?;
        let negative = work.subtract(&BigRational::zero(), coefficient)?;
        add_column(&mut columns, powers.clone(), negative, work)?;
    }
    let width = outer.generators().len();
    work.allocate(256 + width * size_of::<u32>())?;
    let mut replacement_power = Terms::from([(vec![0; width], BigRational::one())]);
    work.tick(outer.terms().len())?;
    let maximum_degree = outer
        .terms()
        .keys()
        .map(|powers| powers[0])
        .max()
        .unwrap_or(0);
    for degree in 0..=maximum_degree {
        work.tick(outer.terms().len())?;
        for (source_powers, source_coefficient) in outer.terms() {
            if source_powers[0] != degree {
                continue;
            }
            for (powers, coefficient) in &replacement_power {
                work.tick(width + 1)?;
                work.allocate(width * size_of::<u32>())?;
                let mut shifted = powers.clone();
                for index in 1..width {
                    shifted[index] = shifted[index]
                        .checked_add(source_powers[index])
                        .ok_or_else(|| refused("composition verifier exponent overflow"))?;
                }
                let value = work.multiply(coefficient, source_coefficient)?;
                add_column(&mut columns, shifted, value, work)?;
            }
        }
        if degree != maximum_degree {
            replacement_power = reference_product(&replacement_power, inner.terms(), work)?;
        }
    }
    for value in columns.values() {
        work.tick(1)?;
        if !value.is_zero() {
            return Err(PolyError::IdentityCheckFailed(
                "rational polynomial composition coefficient identity".into(),
            ));
        }
    }
    Ok(())
}

/// Check complete coefficient equality for the requested first-axis composition.
pub fn verify_polynomial_composition(
    outer: &Polynomial,
    inner: &Polynomial,
    certificate: &PolynomialCompositionCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialComposition, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(outer, inner, &mut work)?;
    reference_identity(outer, inner, &certificate.result, &mut work)?;
    work.allocate(size_of::<VerifiedPolynomialComposition>())?;
    let accepted = VerifiedPolynomialComposition {
        outer: copy_poly(outer, &mut work)?,
        inner: copy_poly(inner, &mut work)?,
        result: copy_poly(&certificate.result, &mut work)?,
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Generate and independently verify a complete bounded rational composition.
pub fn checked_polynomial_composition(
    outer: &Polynomial,
    inner: &Polynomial,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialComposition, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    admit(outer, inner, &mut generation)?;
    let certificate = PolynomialCompositionCertificate {
        result: generate(outer, inner, &mut generation)?,
    };
    let accepted = verify_polynomial_composition(outer, inner, &certificate, verifier)?;
    generation.tick(1)?;
    Ok(accepted)
}
