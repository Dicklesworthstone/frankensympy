//! Complete bounded square-free decomposition in an ordered QQ polynomial ring.
//!
//! In characteristic zero, `g = gcd(f, df/dx_1, ..., df/dx_n)` removes exactly
//! one occurrence of every irreducible factor of a nonzero polynomial. The
//! generator uses this fact and Yun's successive GCD layers. It never requests
//! irreducible factorization.
//!
//! Acceptance binds a complete checked-GCD chain to every original partial
//! derivative, using a separate coefficient check for differentiation. Separate
//! reference convolutions check `f = scale * product(f_i^i)`,
//! `part = product(f_i)` and `f = scale * g * part`. The last identity establishes
//! that `part` is the monic radical of f; consequently every f_i is square-free,
//! the factors are pairwise coprime, and the increasing multiplicities are
//! complete. A product identity by itself grants none of these stronger claims.
//!
//! These are immutable in-process composition receipts, not portable proofs or
//! irreducibility certificates. Both phases have cumulative 20M-compute/128MiB
//! logical-allocation limits through nested GCD calls and publication. Inputs
//! retain the bounded polynomial degree256/4096term/4096bit/eight-variable
//! envelope; general nonmonomial GCD pairs additionally need an auxiliary
//! variable and its existing ambient degree/term envelope. Reference products permit 16384
//! columns while retaining degree256 and the same coefficient/work limits.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_DEGREE, MAX_VARIABLES, PolynomialWork, Terms, accumulate, copy_poly, divide_with_work,
    refused, ring_storage, same_ring, validate_poly,
};
use crate::{
    MultivariatePoly, PolyError, TermOrder, VerifiedMultivariateGcd, checked_multivariate_gcd,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};
use std::mem::size_of;
use std::sync::Arc;

const MAX_REFERENCE_COLUMNS: usize = 16_384;

/// One monic, nonconstant square-free component, grouped by its exact positive
/// multiplicity. Distinct components have distinct increasing multiplicities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultivariateSquareFreeFactor {
    pub polynomial: MultivariatePoly,
    pub multiplicity: usize,
}

/// Untrusted decomposition and immutable lower-level evidence. The first GCD
/// has left input f; each subsequent left input is the preceding accepted GCD.
/// Right inputs cover every original partial derivative in generator order.
#[derive(Debug, Clone)]
pub struct MultivariateSquareFreeCertificate {
    pub scale: BigRational,
    pub factors: Vec<MultivariateSquareFreeFactor>,
    pub square_free_part: MultivariatePoly,
    pub derivative_gcds: Vec<Arc<VerifiedMultivariateGcd>>,
}

/// Complete QQ decomposition with monic factors and monic square-free part.
/// Zero has scale0, no factors and part0; a nonzero constant c has scale c,
/// no factors and part1. The scale is never included in the square-free part.
#[derive(Debug, Clone)]
pub struct VerifiedMultivariateSquareFree {
    input: MultivariatePoly,
    certificate: MultivariateSquareFreeCertificate,
}

impl VerifiedMultivariateSquareFree {
    pub fn input(&self) -> &MultivariatePoly {
        &self.input
    }

    pub fn scale(&self) -> &BigRational {
        &self.certificate.scale
    }

    pub fn factors(&self) -> &[MultivariateSquareFreeFactor] {
        &self.certificate.factors
    }

    pub fn square_free_part(&self) -> &MultivariatePoly {
        &self.certificate.square_free_part
    }

    pub fn certificate(&self) -> &MultivariateSquareFreeCertificate {
        &self.certificate
    }

    pub fn into_parts(
        self,
    ) -> (
        BigRational,
        Vec<MultivariateSquareFreeFactor>,
        MultivariatePoly,
    ) {
        (
            self.certificate.scale,
            self.certificate.factors,
            self.certificate.square_free_part,
        )
    }
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused("square-free ring exceeds eight variables"));
    }
    let mut names = 0usize;
    for (index, symbol) in generators.iter().enumerate() {
        names = names
            .checked_add(symbol.name.len())
            .filter(|&bytes| bytes <= 65_536)
            .ok_or_else(|| refused("square-free generator names exceed 65536 bytes"))?;
        work.tick(symbol.name.len() + index + 1)?;
        if generators[..index].contains(symbol) {
            return Err(refused("duplicate typed square-free generator"));
        }
    }
    Ok(())
}

fn check_polynomial(
    polynomial: &MultivariatePoly,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    // Names are bounded before ring diagnostics, comparisons or owned copies.
    check_ring(polynomial.generators(), work)?;
    same_ring(polynomial.generators(), generators)?;
    validate_poly(polynomial, work)
}

fn degree(
    polynomial: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    let mut degree = 0;
    for powers in polynomial.terms().keys() {
        work.tick(powers.len() + 1)?;
        degree = degree.max(powers.iter().map(|&power| u64::from(power)).sum());
    }
    Ok(degree)
}

fn is_one(polynomial: &MultivariatePoly) -> bool {
    polynomial.terms().len() == 1
        && polynomial.terms().iter().any(|(powers, coefficient)| {
            coefficient.is_one() && powers.iter().all(|&power| power == 0)
        })
}

fn constant(
    generators: &[Symbol],
    one: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    work.allocate(
        size_of::<MultivariatePoly>() + ring_storage(generators) + 256 + generators.len() * 4,
    )?;
    Ok(if one {
        MultivariatePoly::one(generators.to_vec())
    } else {
        MultivariatePoly::zero(generators.to_vec())
    })
}

fn exact_quotient(
    numerator: &MultivariatePoly,
    denominator: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let (mut quotients, remainder) = divide_with_work(
        numerator,
        std::slice::from_ref(denominator),
        TermOrder::Lex,
        work,
    )?;
    if !remainder.is_zero() || quotients.len() != 1 {
        return Err(refused("square-free quotient is not exact"));
    }
    quotients
        .pop()
        .ok_or_else(|| refused("missing square-free quotient"))
}

fn derivative(
    input: &MultivariatePoly,
    axis: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let mut result = constant(input.generators(), false, work)?;
    for (powers, coefficient) in input.terms() {
        work.tick(powers.len() + 1)?;
        if powers[axis] != 0 {
            work.allocate(powers.len() * 4)?;
            let mut differentiated = powers.clone();
            differentiated[axis] -= 1;
            let scalar = BigRational::from(i64::from(powers[axis]));
            let value = work.multiply(coefficient, &scalar)?;
            accumulate(&mut result.terms, differentiated, value, work)?;
        }
    }
    Ok(result)
}

fn same_polynomial(
    actual: &MultivariatePoly,
    expected: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_polynomial(actual, expected.generators(), work)?;
    if actual.terms().len() != expected.terms().len() {
        return Err(refused(
            "square-free witness changed original term coverage",
        ));
    }
    for ((powers, value), (expected_powers, expected_value)) in
        actual.terms().iter().zip(expected.terms())
    {
        let limbs = value.numer().limb_count()
            + value.denom().limb_count()
            + expected_value.numer().limb_count()
            + expected_value.denom().limb_count();
        work.tick(powers.len() + limbs as usize + 1)?;
        if powers != expected_powers || value != expected_value {
            return Err(refused(
                "square-free witness changed exact original coefficients",
            ));
        }
    }
    Ok(())
}

/// Check derivative coefficients and their complete support without calling
/// the generator's derivative constructor or any polynomial differentiation.
fn verify_derivative(
    input: &MultivariatePoly,
    axis: usize,
    candidate: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_polynomial(candidate, input.generators(), work)?;
    let mut covered = 0usize;
    for (powers, coefficient) in input.terms() {
        work.tick(powers.len() + 1)?;
        let power = powers[axis];
        if power == 0 {
            continue;
        }
        covered += 1;
        work.allocate(powers.len() * 4)?;
        let mut target = powers.clone();
        target[axis] -= 1;
        let actual = candidate
            .terms()
            .get(&target)
            .ok_or_else(|| refused("square-free derivative omitted an original term"))?;
        let expected = work.multiply(coefficient, &BigRational::from_integer(power.into()))?;
        if *actual != expected {
            return Err(refused("square-free derivative coefficient changed"));
        }
    }
    if covered != candidate.terms().len() {
        return Err(refused("square-free derivative has extraneous terms"));
    }
    Ok(())
}

fn require_monic(polynomial: &MultivariatePoly) -> Result<(), PolyError> {
    if polynomial
        .terms()
        .last_key_value()
        .is_none_or(|(_, value)| !value.is_one())
    {
        return Err(refused("square-free components must be nonzero and monic"));
    }
    Ok(())
}

fn reference_one(
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    work.allocate(size_of::<Terms>() + 256 + width * 4)?;
    Ok(Terms::from([(vec![0; width], BigRational::one())]))
}

/// Simple sparse reference convolution. It never calls generator polynomial
/// multiplication, division, GCD search or expression normalization.
fn reference_product(
    left: &Terms,
    right: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    work.allocate(size_of::<Terms>())?;
    let mut result = Terms::new();
    for (a, ca) in left {
        for (b, cb) in right {
            work.tick(a.len() + 1)?;
            work.allocate(a.len() * 4)?;
            let powers: Vec<_> = a.iter().zip(b).map(|(a, b)| a + b).collect();
            if powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE {
                return Err(refused("square-free reference product exceeds degree256"));
            }
            let product = work.multiply(ca, cb)?;
            let coefficient = if let Some(previous) = result.get(&powers) {
                work.add(previous, &product)?
            } else {
                product
            };
            if coefficient.is_zero() {
                result.remove(&powers);
            } else {
                if !result.contains_key(&powers) {
                    if result.len() >= MAX_REFERENCE_COLUMNS {
                        return Err(refused("square-free reference column bound"));
                    }
                    work.allocate(256 + powers.len() * 4)?;
                }
                result.insert(powers, coefficient);
            }
        }
    }
    Ok(result)
}

fn same_scaled_terms(
    actual: &Terms,
    expected: &Terms,
    scale: &BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if actual.len() != expected.len() {
        return Err(refused("square-free reconstruction changed term coverage"));
    }
    for ((powers, value), (expected_powers, expected_value)) in actual.iter().zip(expected) {
        work.tick(powers.len() + 1)?;
        if powers != expected_powers || work.multiply(value, scale)? != *expected_value {
            return Err(refused(
                "square-free reconstruction coefficient identity failed",
            ));
        }
    }
    Ok(())
}

fn publish(
    input: &MultivariatePoly,
    certificate: &MultivariateSquareFreeCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<VerifiedMultivariateSquareFree, PolyError> {
    work.allocate(
        size_of::<VerifiedMultivariateSquareFree>()
            + certificate.factors.len() * size_of::<MultivariateSquareFreeFactor>()
            + certificate.derivative_gcds.len() * size_of::<Arc<VerifiedMultivariateGcd>>(),
    )?;
    let mut factors = Vec::with_capacity(certificate.factors.len());
    for factor in &certificate.factors {
        factors.push(MultivariateSquareFreeFactor {
            polynomial: copy_poly(&factor.polynomial, work)?,
            multiplicity: factor.multiplicity,
        });
    }
    let accepted = VerifiedMultivariateSquareFree {
        input: copy_poly(input, work)?,
        certificate: MultivariateSquareFreeCertificate {
            scale: work.clone_value(&certificate.scale)?,
            factors,
            square_free_part: copy_poly(&certificate.square_free_part, work)?,
            derivative_gcds: certificate.derivative_gcds.iter().map(Arc::clone).collect(),
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Verify complete square-free multiplicities in the exact original QQ ring.
/// Immutable complete GCD receipts establish the gradient gcd. Independent
/// derivative and product checks bind those receipts to this claim.
pub fn verify_multivariate_square_free(
    input: &MultivariatePoly,
    certificate: &MultivariateSquareFreeCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedMultivariateSquareFree, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    check_ring(input.generators(), &mut work)?;
    validate_poly(input, &mut work)?;
    if certificate.factors.len() > MAX_DEGREE as usize
        || certificate.derivative_gcds.len() > MAX_VARIABLES
        || certificate.scale.numer().bits() > 4096
        || certificate.scale.denom().bits() > 4096
    {
        return Err(refused(
            "square-free certificate exceeds its bounded envelope",
        ));
    }
    check_polynomial(&certificate.square_free_part, input.generators(), &mut work)?;
    if input.is_zero() {
        if !certificate.scale.is_zero()
            || !certificate.factors.is_empty()
            || !certificate.derivative_gcds.is_empty()
            || !certificate.square_free_part.is_zero()
        {
            return Err(refused("zero square-free decomposition changed"));
        }
        return publish(input, certificate, &mut work);
    }
    let leading = input.terms().last_key_value().expect("nonzero input").1;
    work.tick((leading.numer().limb_count() + leading.denom().limb_count()) as usize + 1)?;
    if certificate.scale != *leading {
        return Err(refused(
            "square-free scale must equal the original Lex leading coefficient",
        ));
    }
    require_monic(&certificate.square_free_part)?;
    let input_degree = degree(input, &mut work)?;
    let expected_steps = if input_degree == 0 {
        0
    } else {
        input.generators().len()
    };
    if certificate.derivative_gcds.len() != expected_steps {
        return Err(refused(
            "square-free evidence must cover every original partial derivative",
        ));
    }
    let mut gradient_gcd = input;
    for (axis, receipt) in certificate.derivative_gcds.iter().enumerate() {
        same_polynomial(receipt.left(), gradient_gcd, &mut work)?;
        verify_derivative(input, axis, receipt.right(), &mut work)?;
        gradient_gcd = receipt.gcd();
    }
    let mut weighted_degree = 0u64;
    let mut previous_multiplicity = 0;
    for factor in &certificate.factors {
        check_polynomial(&factor.polynomial, input.generators(), &mut work)?;
        require_monic(&factor.polynomial)?;
        let factor_degree = degree(&factor.polynomial, &mut work)?;
        if factor_degree == 0
            || factor.multiplicity <= previous_multiplicity
            || factor.multiplicity > MAX_DEGREE as usize
        {
            return Err(refused(
                "square-free factors require increasing positive multiplicities and positive degree",
            ));
        }
        weighted_degree += factor_degree * factor.multiplicity as u64;
        if weighted_degree > input_degree {
            return Err(refused(
                "square-free multiplicity degree exceeds the original degree",
            ));
        }
        previous_multiplicity = factor.multiplicity;
    }
    if weighted_degree != input_degree {
        return Err(refused("square-free multiplicities omit original degree"));
    }
    let mut product = reference_one(input.generators().len(), &mut work)?;
    let mut part = reference_one(input.generators().len(), &mut work)?;
    for factor in &certificate.factors {
        part = reference_product(&part, factor.polynomial.terms(), &mut work)?;
        for _ in 0..factor.multiplicity {
            product = reference_product(&product, factor.polynomial.terms(), &mut work)?;
        }
    }
    same_scaled_terms(&product, input.terms(), &certificate.scale, &mut work)?;
    same_scaled_terms(
        &part,
        certificate.square_free_part.terms(),
        &BigRational::one(),
        &mut work,
    )?;
    if input_degree != 0 {
        let reconstructed = reference_product(gradient_gcd.terms(), &part, &mut work)?;
        same_scaled_terms(&reconstructed, input.terms(), &certificate.scale, &mut work)?;
    }
    publish(input, certificate, &mut work)
}

/// Construct complete monic square-free components without irreducible
/// factorization. The caller's separate phase meters remain cumulative through
/// all partial derivatives, nested checked GCDs, multiplicity layers and output.
pub fn checked_multivariate_square_free(
    input: &MultivariatePoly,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedMultivariateSquareFree, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    check_ring(input.generators(), &mut generation)?;
    validate_poly(input, &mut generation)?;
    let input_degree = degree(input, &mut generation)?;
    let scale = match input.terms().last_key_value() {
        Some((_, value)) => generation.clone_value(value)?,
        None => BigRational::zero(),
    };
    generation.allocate(
        size_of::<MultivariateSquareFreeCertificate>()
            + input.generators().len() * size_of::<Arc<VerifiedMultivariateGcd>>()
            + input_degree as usize * size_of::<MultivariateSquareFreeFactor>(),
    )?;
    let mut derivative_gcds = Vec::with_capacity(input.generators().len());
    let mut factors = Vec::with_capacity(input_degree as usize);
    let square_free_part = if input_degree == 0 {
        constant(input.generators(), !input.is_zero(), &mut generation)?
    } else {
        let mut common = copy_poly(input, &mut generation)?;
        for axis in 0..input.generators().len() {
            let differentiated = derivative(input, axis, &mut generation)?;
            let receipt = checked_multivariate_gcd(
                &common,
                &differentiated,
                &mut generation,
                &mut verification,
            )?;
            common = copy_poly(receipt.gcd(), &mut generation)?;
            generation.allocate(size_of::<VerifiedMultivariateGcd>() + 32)?;
            derivative_gcds.push(Arc::new(receipt));
        }
        let mut remaining = exact_quotient(input, &common, &mut generation)?;
        for coefficient in remaining.terms.values_mut() {
            *coefficient = generation.divide(coefficient, &scale)?;
        }
        let part = copy_poly(&remaining, &mut generation)?;
        let mut multiplicity = 1usize;
        while !is_one(&remaining) {
            generation.tick(1)?;
            if multiplicity > input_degree as usize {
                return Err(refused(
                    "square-free multiplicity layers exceed the original degree",
                ));
            }
            let layer =
                checked_multivariate_gcd(&remaining, &common, &mut generation, &mut verification)?;
            let factor = exact_quotient(&remaining, layer.gcd(), &mut generation)?;
            if !is_one(&factor) {
                factors.push(MultivariateSquareFreeFactor {
                    polynomial: factor,
                    multiplicity,
                });
            }
            remaining = copy_poly(layer.gcd(), &mut generation)?;
            common = exact_quotient(&common, &remaining, &mut generation)?;
            multiplicity += 1;
        }
        if !is_one(&common) {
            return Err(refused(
                "characteristic-zero square-free layers did not terminate",
            ));
        }
        part
    };
    let certificate = MultivariateSquareFreeCertificate {
        scale,
        factors,
        square_free_part,
        derivative_gcds,
    };
    let accepted = verify_multivariate_square_free(input, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
