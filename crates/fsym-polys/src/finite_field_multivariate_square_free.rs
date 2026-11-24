//! Complete bounded square-free decomposition over an ordered prime field.
//!
//! Generation uses gradient GCDs and Yun layers. An inseparable remainder is
//! a pth power over F_p: its exponents are divided by p, its coefficients stay
//! fixed, and recursive multiplicities are multiplied by p. Degree 64 bounds
//! Frobenius descent to six steps. No irreducible factorization is requested.
//!
//! Acceptance is independent of that search. Signed coefficient convolutions
//! check input = unit * product(f_i^m_i) and part = product(f_i). Immutable
//! complete GCD receipts are bound to independently reconstructed partial
//! derivatives of part, and their complete chain must end at one. Over the
//! perfect field F_p this proves part square-free, hence every f_i square-free,
//! pairwise coprime, and present with its complete original multiplicity.
//! Product reconstruction alone never grants those stronger claims.
//!
//! Sources and all results retain the original typed ordered ring. Existing
//! degree 64/4096-term/prime 65521 limits and separate cumulative 20M-step/128MiB
//! phase envelopes apply through nested GCDs and publication. General GCDs
//! retain their auxiliary-axis and augmented-degree refusals: eight axes are
//! available only when the required GCD operations admit them. These are
//! immutable in-process composition receipts, not portable proofs or claims
//! of irreducibility.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field::scalar_power;
use crate::finite_field_multivariate::{
    Columns, MAX_DEGREE, MAX_TERMS, MAX_VARIABLES, admit_ring, admit_terms, constant,
    copied_polynomial, divide, empty, identity_zero, leading, monomial_copy, product,
    reference_add, reference_head, reference_product, same_ring, scaled,
};
use crate::{
    PolyError, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder, VerifiedPrimeFieldMultivariateGcd,
    checked_prime_field_multivariate_gcd,
};
use fsym_budget::BudgetMeter;
use std::mem::size_of;
use std::sync::Arc;

const MAX_FROBENIUS_DEPTH: usize = 6;

/// One monic nonconstant square-free component at its exact positive
/// multiplicity. Components are grouped by increasing distinct multiplicity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeFieldMultivariateSquareFreeFactor {
    pub polynomial: Polynomial,
    pub multiplicity: usize,
}

/// Untrusted decomposition plus immutable complete lower-level GCD evidence.
/// The first GCD starts at square_free_part. Each successive left input is
/// the preceding GCD, and right inputs cover every partial derivative of
/// square_free_part in the original generator order.
#[derive(Debug, Clone)]
pub struct PrimeFieldMultivariateSquareFreeCertificate {
    pub unit: u64,
    pub factors: Vec<PrimeFieldMultivariateSquareFreeFactor>,
    pub square_free_part: Polynomial,
    pub radical_derivative_gcds: Vec<Arc<VerifiedPrimeFieldMultivariateGcd>>,
}

/// Complete monic square-free multiplicity groups in the original prime ring.
/// Zero has unit 0, no groups and part 0; a nonzero constant c has unit c, no
/// groups and part 1. The unit is never included in the square-free part.
#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldMultivariateSquareFree {
    input: Polynomial,
    certificate: PrimeFieldMultivariateSquareFreeCertificate,
}

impl VerifiedPrimeFieldMultivariateSquareFree {
    pub fn input(&self) -> &Polynomial {
        &self.input
    }

    pub fn unit(&self) -> u64 {
        self.certificate.unit
    }

    pub fn factors(&self) -> &[PrimeFieldMultivariateSquareFreeFactor] {
        &self.certificate.factors
    }

    pub fn square_free_part(&self) -> &Polynomial {
        &self.certificate.square_free_part
    }

    pub fn certificate(&self) -> &PrimeFieldMultivariateSquareFreeCertificate {
        &self.certificate
    }

    pub fn into_parts(self) -> (u64, Vec<PrimeFieldMultivariateSquareFreeFactor>, Polynomial) {
        (
            self.certificate.unit,
            self.certificate.factors,
            self.certificate.square_free_part,
        )
    }
}

fn admit(
    input: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(input.generators(), input.prime(), work)?;
    same_ring(input, input.generators(), input.prime(), work)
}

fn copy(
    input: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    copied_polynomial(input.generators(), input.prime(), input.terms(), work)
}

fn from_terms(
    original: &Polynomial,
    terms: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    admit_terms(
        terms,
        original.generators().len(),
        original.prime(),
        MAX_DEGREE,
        false,
        work,
    )?;
    copied_polynomial(original.generators(), original.prime(), terms, work)
}

fn constant_polynomial(
    original: &Polynomial,
    value: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let terms = constant(original.generators().len(), value, work)?;
    from_terms(original, &terms, work)
}

fn degree(
    input: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    let mut result = 0u64;
    for powers in input.terms().keys() {
        work.tick(powers.len() + 1)?;
        result = result.max(powers.iter().map(|&power| u64::from(power)).sum());
    }
    Ok(result)
}

fn is_one(input: &Polynomial) -> bool {
    input.terms().len() == 1
        && input.terms().iter().any(|(powers, coefficient)| {
            *coefficient == 1 && powers.iter().all(|&power| power == 0)
        })
}

fn derivative(
    input: &Polynomial,
    axis: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut terms = empty(work)?;
    for (powers, coefficient) in input.terms() {
        work.tick(powers.len() + 1)?;
        let value = coefficient * (u64::from(powers[axis]) % input.prime()) % input.prime();
        if value == 0 {
            continue;
        }
        let mut differentiated = monomial_copy(powers, work)?;
        differentiated[axis] -= 1;
        work.allocate(128)?;
        terms.insert(differentiated, value);
    }
    from_terms(input, &terms, work)
}

fn exact_quotient(
    numerator: &Polynomial,
    denominator: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let mut division = divide(
        numerator.terms(),
        &[denominator.terms()],
        TermOrder::Lex,
        numerator.prime(),
        work,
    )?;
    if !division.remainder.is_empty() || division.quotients.len() != 1 {
        return Err(refused("prime-field square-free quotient is not exact"));
    }
    let quotient = division
        .quotients
        .pop()
        .ok_or_else(|| refused("missing prime-field square-free quotient"))?;
    from_terms(numerator, &quotient, work)
}

fn frobenius_root(
    input: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let prime = input.prime();
    let mut terms = empty(work)?;
    for (powers, coefficient) in input.terms() {
        work.tick(powers.len() + 1)?;
        if powers.iter().any(|&power| u64::from(power) % prime != 0) {
            return Err(refused(
                "inseparable residual has a non-p-divisible exponent",
            ));
        }
        let mut root = monomial_copy(powers, work)?;
        for power in &mut root {
            *power = u32::try_from(u64::from(*power) / prime)
                .map_err(|_| refused("Frobenius root exponent bound"))?;
        }
        // The admitted coefficient field is exactly F_p, whose Frobenius
        // automorphism fixes every coefficient. This is not an extension field.
        work.allocate(128)?;
        terms.insert(root, *coefficient);
    }
    from_terms(input, &terms, work)
}

fn generate_layers(
    input: &Polynomial,
    multiplicity_scale: usize,
    depth: usize,
    factors: &mut Vec<PrimeFieldMultivariateSquareFreeFactor>,
    generation: &mut PolynomialWork<'_, impl BudgetMeter>,
    verification: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    generation.tick(1)?;
    if depth > MAX_FROBENIUS_DEPTH {
        return Err(refused("prime-field square-free Frobenius depth bound"));
    }
    let input_degree = degree(input, generation)?;
    if input_degree == 0 {
        return if is_one(input) {
            Ok(())
        } else {
            Err(refused("nonmonic constant in square-free descent"))
        };
    }
    let mut common = copy(input, generation)?;
    for axis in 0..input.generators().len() {
        let differentiated = derivative(input, axis, generation)?;
        let receipt = checked_prime_field_multivariate_gcd(
            &common,
            &differentiated,
            generation,
            verification,
        )?;
        common = copy(receipt.gcd(), generation)?;
    }
    let mut remaining = exact_quotient(input, &common, generation)?;
    let mut multiplicity = 1usize;
    while !is_one(&remaining) {
        generation.tick(1)?;
        if multiplicity > input_degree as usize {
            return Err(refused("prime-field square-free multiplicity layer bound"));
        }
        let layer =
            checked_prime_field_multivariate_gcd(&remaining, &common, generation, verification)?;
        let factor = exact_quotient(&remaining, layer.gcd(), generation)?;
        if !is_one(&factor) {
            let exponent = multiplicity_scale
                .checked_mul(multiplicity)
                .filter(|&value| value <= MAX_DEGREE as usize)
                .ok_or_else(|| refused("prime-field square-free multiplicity overflow"))?;
            if factors.len() == MAX_DEGREE as usize {
                return Err(refused("prime-field square-free factor count bound"));
            }
            factors.push(PrimeFieldMultivariateSquareFreeFactor {
                polynomial: factor,
                multiplicity: exponent,
            });
        }
        remaining = copy(layer.gcd(), generation)?;
        common = exact_quotient(&common, &remaining, generation)?;
        multiplicity += 1;
    }
    if !is_one(&common) {
        let prime = usize::try_from(input.prime())
            .map_err(|_| refused("prime-field characteristic exceeds platform size"))?;
        let scale = multiplicity_scale
            .checked_mul(prime)
            .filter(|&value| value <= MAX_DEGREE as usize)
            .ok_or_else(|| refused("Frobenius multiplicity exceeds degree64"))?;
        let root = frobenius_root(&common, generation)?;
        if degree(&root, generation)? >= degree(&common, generation)? {
            return Err(refused("Frobenius descent did not decrease degree"));
        }
        generate_layers(&root, scale, depth + 1, factors, generation, verification)?;
    }
    generation.tick(1)
}

fn same_polynomial(
    actual: &Polynomial,
    expected: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    same_ring(actual, expected.generators(), expected.prime(), work)?;
    work.tick((actual.terms().len() + expected.terms().len()) * (expected.generators().len() + 1))?;
    if actual.terms() != expected.terms() {
        return Err(refused(
            "square-free GCD evidence changed its original coefficients",
        ));
    }
    Ok(())
}

/// Reconstruct every nonzero formal derivative coefficient and complete
/// support without calling the generator's derivative or modular arithmetic.
fn verify_derivative(
    input: &Polynomial,
    axis: usize,
    candidate: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    same_ring(candidate, input.generators(), input.prime(), work)?;
    let mut covered = 0;
    for (powers, coefficient) in input.terms() {
        work.tick(powers.len() + 1)?;
        let expected =
            (u128::from(*coefficient) * u128::from(powers[axis])) % u128::from(input.prime());
        if expected == 0 {
            continue;
        }
        covered += 1;
        work.allocate(size_of::<u32>() * powers.len())?;
        let mut differentiated = powers.clone();
        differentiated[axis] -= 1;
        if candidate.terms().get(&differentiated).copied() != Some(expected as u64) {
            return Err(refused(
                "square-free derivative coefficient or support changed",
            ));
        }
    }
    if candidate.terms().len() != covered {
        return Err(refused("square-free derivative has extraneous terms"));
    }
    Ok(())
}

fn require_monic(
    input: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if reference_head(input.terms(), TermOrder::Lex, work)?
        .is_none_or(|(_, coefficient)| coefficient != 1)
    {
        return Err(refused("square-free components must be nonzero and monic"));
    }
    Ok(())
}

fn reference_one(
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    work.allocate(size_of::<Terms>() + 128 + width * size_of::<u32>())?;
    Ok(Terms::from([(vec![0; width], 1)]))
}

/// Each convolution first assembles complete signed integer coefficient
/// columns, independently of the generator's sparse modular product. Only
/// completed columns are reduced into F_p for the next reference multiplication.
fn reference_multiply(
    left: &Terms,
    right: &Terms,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    work.allocate(size_of::<Columns>())?;
    let mut columns = Columns::new();
    reference_product(&mut columns, left, right, 1, work)?;
    let mut result = empty(work)?;
    for (powers, value) in columns {
        work.tick(powers.len() + 1)?;
        if powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE {
            return Err(refused("square-free reference product exceeds degree64"));
        }
        let value = value.rem_euclid(i128::from(prime)) as u64;
        if value != 0 {
            if result.len() == MAX_TERMS {
                return Err(refused("square-free reference product term bound"));
            }
            work.allocate(128)?;
            result.insert(powers, value);
        }
    }
    Ok(result)
}

fn same_scaled_terms(
    actual: &Terms,
    expected: &Terms,
    unit: u64,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.allocate(size_of::<Columns>())?;
    let mut columns = Columns::new();
    reference_add(&mut columns, actual, i128::from(unit), work)?;
    reference_add(&mut columns, expected, -1, work)?;
    identity_zero(&columns, prime, work)
}

fn publish(
    input: &Polynomial,
    certificate: &PrimeFieldMultivariateSquareFreeCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<VerifiedPrimeFieldMultivariateSquareFree, PolyError> {
    work.allocate(
        size_of::<VerifiedPrimeFieldMultivariateSquareFree>()
            + certificate.factors.len() * size_of::<PrimeFieldMultivariateSquareFreeFactor>()
            + certificate.radical_derivative_gcds.len()
                * size_of::<Arc<VerifiedPrimeFieldMultivariateGcd>>(),
    )?;
    let mut factors = Vec::with_capacity(certificate.factors.len());
    for factor in &certificate.factors {
        factors.push(PrimeFieldMultivariateSquareFreeFactor {
            polynomial: copy(&factor.polynomial, work)?,
            multiplicity: factor.multiplicity,
        });
    }
    work.tick(certificate.radical_derivative_gcds.len())?;
    let result = VerifiedPrimeFieldMultivariateSquareFree {
        input: copy(input, work)?,
        certificate: PrimeFieldMultivariateSquareFreeCertificate {
            unit: certificate.unit,
            factors,
            square_free_part: copy(&certificate.square_free_part, work)?,
            radical_derivative_gcds: certificate
                .radical_derivative_gcds
                .iter()
                .map(Arc::clone)
                .collect(),
        },
    };
    work.tick(1)?;
    Ok(result)
}

/// Verify exact original reconstruction, complete multiplicity groups and
/// square-freeness of their product, including positive-characteristic cases.
/// Immutable complete GCD receipts cannot substitute another ring or source.
pub fn verify_prime_field_multivariate_square_free(
    input: &Polynomial,
    certificate: &PrimeFieldMultivariateSquareFreeCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateSquareFree, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(input, &mut work)?;
    if certificate.factors.len() > MAX_DEGREE as usize
        || certificate.radical_derivative_gcds.len() > MAX_VARIABLES
        || certificate.unit >= input.prime()
    {
        return Err(refused("prime-field square-free certificate envelope"));
    }
    same_ring(
        &certificate.square_free_part,
        input.generators(),
        input.prime(),
        &mut work,
    )?;
    if input.is_zero() {
        if certificate.unit != 0
            || !certificate.factors.is_empty()
            || !certificate.radical_derivative_gcds.is_empty()
            || !certificate.square_free_part.is_zero()
        {
            return Err(refused("zero prime-field square-free convention changed"));
        }
        return publish(input, certificate, &mut work);
    }
    let unit = reference_head(input.terms(), TermOrder::Lex, &mut work)?
        .ok_or_else(|| refused("nonzero square-free source lost its leading term"))?
        .1;
    if certificate.unit != unit {
        return Err(refused(
            "square-free unit is not the original Lex leading coefficient",
        ));
    }
    require_monic(&certificate.square_free_part, &mut work)?;
    let input_degree = degree(input, &mut work)?;
    let expected_derivatives = if input_degree == 0 {
        0
    } else {
        input.generators().len()
    };
    if certificate.radical_derivative_gcds.len() != expected_derivatives {
        return Err(refused(
            "square-free evidence must cover every partial derivative",
        ));
    }

    let mut weighted_degree = 0u64;
    let mut previous = 0;
    for factor in &certificate.factors {
        same_ring(
            &factor.polynomial,
            input.generators(),
            input.prime(),
            &mut work,
        )?;
        require_monic(&factor.polynomial, &mut work)?;
        let factor_degree = degree(&factor.polynomial, &mut work)?;
        if factor_degree == 0
            || factor.multiplicity <= previous
            || factor.multiplicity > MAX_DEGREE as usize
        {
            return Err(refused(
                "square-free factors need positive degree and increasing multiplicity",
            ));
        }
        weighted_degree = weighted_degree
            .checked_add(factor_degree * factor.multiplicity as u64)
            .filter(|&value| value <= input_degree)
            .ok_or_else(|| refused("square-free weighted degree exceeds its original source"))?;
        previous = factor.multiplicity;
    }
    if weighted_degree != input_degree {
        return Err(refused("square-free multiplicities omit original degree"));
    }

    let mut gradient_gcd = &certificate.square_free_part;
    for (axis, receipt) in certificate.radical_derivative_gcds.iter().enumerate() {
        same_polynomial(receipt.left(), gradient_gcd, &mut work)?;
        verify_derivative(
            &certificate.square_free_part,
            axis,
            receipt.right(),
            &mut work,
        )?;
        gradient_gcd = receipt.gcd();
    }
    if !is_one(gradient_gcd) {
        return Err(refused(
            "proposed square-free part has a nonunit complete gradient GCD",
        ));
    }

    let mut reconstructed = reference_one(input.generators().len(), &mut work)?;
    let mut part = reference_one(input.generators().len(), &mut work)?;
    for factor in &certificate.factors {
        part = reference_multiply(&part, factor.polynomial.terms(), input.prime(), &mut work)?;
        for _ in 0..factor.multiplicity {
            reconstructed = reference_multiply(
                &reconstructed,
                factor.polynomial.terms(),
                input.prime(),
                &mut work,
            )?;
        }
    }
    same_scaled_terms(
        &reconstructed,
        input.terms(),
        certificate.unit,
        input.prime(),
        &mut work,
    )?;
    same_scaled_terms(
        &part,
        certificate.square_free_part.terms(),
        1,
        input.prime(),
        &mut work,
    )?;
    publish(input, certificate, &mut work)
}

/// Generate complete monic multiplicity groups without irreducible
/// factorization. Nested checked GCDs and every Frobenius descent share the
/// original caller's separate cumulative generation and verification meters.
pub fn checked_prime_field_multivariate_square_free(
    input: &Polynomial,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateSquareFree, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(input, &mut generation)?;
    let input_degree = degree(input, &mut generation)?;
    let unit =
        leading(input.terms(), TermOrder::Lex, &mut generation)?.map_or(0, |(_, value)| value);
    generation.allocate(
        size_of::<PrimeFieldMultivariateSquareFreeCertificate>()
            + input_degree as usize * size_of::<PrimeFieldMultivariateSquareFreeFactor>()
            + input.generators().len() * size_of::<Arc<VerifiedPrimeFieldMultivariateGcd>>(),
    )?;
    let mut factors = Vec::with_capacity(input_degree as usize);
    let mut radical_derivative_gcds = Vec::with_capacity(input.generators().len());
    let square_free_part = if input_degree == 0 {
        constant_polynomial(input, u64::from(!input.is_zero()), &mut generation)?
    } else {
        let inverse = scalar_power(unit, input.prime() - 2, input.prime(), &mut generation)?;
        let monic_terms = scaled(input.terms(), inverse, input.prime(), &mut generation)?;
        let monic = from_terms(input, &monic_terms, &mut generation)?;
        generate_layers(
            &monic,
            1,
            0,
            &mut factors,
            &mut generation,
            &mut verification,
        )?;
        generation.tick(factors.len() * factors.len() + 1)?;
        factors.sort_unstable_by_key(|factor| factor.multiplicity);
        let mut part = constant_polynomial(input, 1, &mut generation)?;
        for factor in &factors {
            let terms = product(
                part.terms(),
                factor.polynomial.terms(),
                input.prime(),
                &mut generation,
            )?;
            part = from_terms(input, &terms, &mut generation)?;
        }
        let mut common = copy(&part, &mut generation)?;
        for axis in 0..input.generators().len() {
            let differentiated = derivative(&part, axis, &mut generation)?;
            let receipt = checked_prime_field_multivariate_gcd(
                &common,
                &differentiated,
                &mut generation,
                &mut verification,
            )?;
            common = copy(receipt.gcd(), &mut generation)?;
            generation.allocate(size_of::<VerifiedPrimeFieldMultivariateGcd>() + 32)?;
            radical_derivative_gcds.push(Arc::new(receipt));
        }
        part
    };
    let certificate = PrimeFieldMultivariateSquareFreeCertificate {
        unit,
        factors,
        square_free_part,
        radical_derivative_gcds,
    };
    let accepted =
        verify_prime_field_multivariate_square_free(input, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;
    use fsym_core::Symbol;

    #[test]
    fn frobenius_root_rejects_one_nondivisible_coordinate() {
        let generators = [Symbol::new("x"), Symbol::new("y")];
        let input = Polynomial::new(
            &generators,
            3,
            &Terms::from([(vec![3, 3], 1), (vec![6, 1], 2)]),
        )
        .unwrap();
        let mut meter = Unbounded;
        let error = frobenius_root(&input, &mut PolynomialWork::new(&mut meter)).unwrap_err();
        assert!(error.to_string().contains("non-p-divisible"));
    }

    #[test]
    fn frobenius_root_preserves_prime_field_coefficients() {
        let generators = [Symbol::new("x"), Symbol::new("y")];
        let input = Polynomial::new(
            &generators,
            5,
            &Terms::from([(vec![5, 10], 3), (vec![0, 5], 4)]),
        )
        .unwrap();
        let expected = Polynomial::new(
            &generators,
            5,
            &Terms::from([(vec![1, 2], 3), (vec![0, 1], 4)]),
        )
        .unwrap();
        let mut meter = Unbounded;
        let result = frobenius_root(&input, &mut PolynomialWork::new(&mut meter)).unwrap();
        assert_eq!(result, expected);
    }
}
