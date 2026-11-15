//! Reference integer-domain checks for higher-degree product factors.
//!
//! The witness supplies a small prime, never a trusted modular image or a
//! factorization label. We reduce each original rational coefficient ourselves
//! and check every residue. If no residue is a zero, no integer can be a zero.
//! This is not an irreducibility claim over either Q or the finite field.

use super::{RationalProductFactors, Result, invalid};
use crate::rational_sum::{MAX_SUM_COEFFICIENTS, Reference, validate};
use fsym_budget::BudgetMeter;
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};

pub const MAX_PRODUCT_DOMAIN_PRIME: u32 = 97;

/// A complete monic factor, with a witness that it has no integer roots.
/// Coefficients are ascending exact rationals; the verifier derives their
/// residues and checks every value modulo `prime`. Coefficient denominators
/// must be units modulo that prime. No irreducibility is asserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalProductModularFactor {
    pub coefficients: Vec<BigRational>,
    pub multiplicity: u32,
    pub prime: u32,
}

/// Byte-wise reference reduction, separate from the generator's metered
/// arbitrary-precision division. Signs are applied only after the complete
/// unsigned magnitude has been reduced.
fn residue(
    value: &BigInt,
    prime: u64,
    reference: &mut Reference<'_, impl BudgetMeter>,
) -> Result<u64> {
    reference.reserve(1)?;
    reference.work(value.bits().div_ceil(8).max(1))?;
    let bytes = value.to_bytes_le();
    let mut result = 0;
    for byte in bytes.iter().rev() {
        reference.work(1)?;
        result = (result * 256 + u64::from(*byte)) % prime;
    }
    Ok(if value.is_negative() && result != 0 {
        prime - result
    } else {
        result
    })
}

fn check_no_integer_root(
    factor: &RationalProductModularFactor,
    reference: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    reference.work(1)?;
    if !(2..=MAX_PRODUCT_DOMAIN_PRIME).contains(&factor.prime) {
        return Err(invalid(
            "product domain prime is outside the declared bound",
        ));
    }
    let prime = u64::from(factor.prime);
    for divisor in 2..prime {
        reference.work(1)?;
        if divisor * divisor > prime {
            break;
        }
        if prime % divisor == 0 {
            return Err(invalid("product domain modulus is not prime"));
        }
    }
    reference.reserve(factor.coefficients.len())?;
    let mut coefficients = Vec::with_capacity(factor.coefficients.len());
    for coefficient in &factor.coefficients {
        let numerator = residue(coefficient.numer(), prime, reference)?;
        let denominator = residue(coefficient.denom(), prime, reference)?;
        // Exhaustive small-field inversion is deliberately separate from the
        // generator's binary exponentiation. A zero denominator is refused,
        // rather than cancelling a characteristic factor out of the source.
        let mut inverse = None;
        for candidate in 1..prime {
            reference.work(1)?;
            if (denominator * candidate) % prime == 1 {
                inverse = Some(candidate);
                break;
            }
        }
        let inverse = inverse.ok_or_else(|| {
            invalid("original product factor coefficient denominator is not a modular unit")
        })?;
        coefficients.push((numerator * inverse) % prime);
    }
    // Every integer maps to one of these residues. This establishes a global
    // integer-domain fact, not sampled nonvanishing on the requested range.
    for point in 0..prime {
        let mut value = 0;
        for coefficient in coefficients.iter().rev() {
            reference.work(1)?;
            value = (value * point + coefficient) % prime;
        }
        if value == 0 {
            return Err(invalid(
                "product factor has a zero modulo the witness prime",
            ));
        }
    }
    reference.work(1)?;
    Ok(())
}

fn add_degree(total: &mut usize, degree: usize, multiplicity: u32, limit: usize) -> Result<()> {
    if multiplicity == 0 {
        return Err(invalid("zero product factor multiplicity"));
    }
    let multiplicity = usize::try_from(multiplicity)
        .map_err(|_| invalid("product factor multiplicity overflow"))?;
    *total = total
        .checked_add(
            degree
                .checked_mul(multiplicity)
                .ok_or_else(|| invalid("product factor degree overflow"))?,
        )
        .filter(|total| *total <= limit)
        .ok_or_else(|| invalid("product factor degree exceeds the original polynomial"))?;
    Ok(())
}

pub(super) fn check_factors(
    polynomial: &[BigRational],
    factors: &RationalProductFactors,
    reference: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    if factors.modular.is_empty() {
        reference.check_denominator(polynomial, &factors.roots, &factors.quadratics)?;
        return Ok(());
    }
    validate(polynomial)?;
    if polynomial.last().is_none_or(Zero::is_zero) {
        return Err(invalid("zero product domain polynomial"));
    }
    if [
        factors.roots.len(),
        factors.quadratics.len(),
        factors.modular.len(),
    ]
    .into_iter()
    .any(|count| count >= MAX_SUM_COEFFICIENTS)
        || factors.roots.len() + factors.quadratics.len() + factors.modular.len()
            >= MAX_SUM_COEFFICIENTS
    {
        return Err(invalid("product domain factor count limit"));
    }
    let mut degree = 0;
    let limit = polynomial.len() - 1;
    // Bound all untrusted degrees, heights and multiplicities before any
    // product expansion or coefficient copy, including the low-degree part.
    for pole in &factors.roots {
        reference.work(1)?;
        validate(std::slice::from_ref(&pole.root))?;
        add_degree(&mut degree, 1, pole.multiplicity, limit)?;
    }
    for factor in &factors.quadratics {
        reference.work(1)?;
        validate(std::slice::from_ref(&factor.constant))?;
        validate(std::slice::from_ref(&factor.linear))?;
        add_degree(&mut degree, 2, factor.multiplicity, limit)?;
    }
    for (index, factor) in factors.modular.iter().enumerate() {
        reference.work((index + 1) as u64)?;
        validate(&factor.coefficients)?;
        if factor.coefficients.len() < 4 || !factor.coefficients.last().is_some_and(One::is_one) {
            return Err(invalid(
                "modular product factors must be monic of degree at least three",
            ));
        }
        add_degree(
            &mut degree,
            factor.coefficients.len() - 1,
            factor.multiplicity,
            limit,
        )?;
        if factors.modular[..index]
            .iter()
            .any(|earlier| earlier.coefficients == factor.coefficients)
        {
            return Err(invalid("duplicate modular product factor"));
        }
    }
    if degree != limit {
        return Err(invalid("incomplete product domain factor list"));
    }

    reference.reserve(1)?;
    let mut low_degree = vec![BigRational::one()];
    for pole in &factors.roots {
        reference.reserve(2)?;
        let factor = [-pole.root.clone(), BigRational::one()];
        for _ in 0..pole.multiplicity {
            low_degree = reference.product(&low_degree, &factor)?;
        }
    }
    for factor in &factors.quadratics {
        reference.reserve(3)?;
        let coefficients = [
            factor.constant.clone(),
            factor.linear.clone(),
            BigRational::one(),
        ];
        for _ in 0..factor.multiplicity {
            low_degree = reference.product(&low_degree, &coefficients)?;
        }
    }
    // Reuse the existing independent rational-root and quadratic checks. The
    // complete source binding is established separately by the final product,
    // which includes every higher-degree factor and the original leading scale.
    reference.check_denominator(&low_degree, &factors.roots, &factors.quadratics)?;
    reference.reserve(1)?;
    let leading = polynomial.last().expect("validated nonempty").clone();
    let mut reconstructed = reference.product(&low_degree, &[leading])?;
    for factor in &factors.modular {
        check_no_integer_root(factor, reference)?;
        for _ in 0..factor.multiplicity {
            reconstructed = reference.product(&reconstructed, &factor.coefficients)?;
        }
    }
    if reconstructed != polynomial {
        return Err(invalid(
            "product domain factors do not reconstruct the original polynomial",
        ));
    }
    reference.work(1)?;
    Ok(())
}
