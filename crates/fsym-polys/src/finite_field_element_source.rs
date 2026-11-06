//! Independently checked integer-polynomial reduction into a prime field.
//!
//! This is the coefficient-domain boundary used by `Poly(modulus=p)`, distinct
//! from modular factorization's exterior rational-content convention. The
//! complete original expression must expand to a polynomial over ZZ before
//! reduction. Rational intermediate expressions may cancel exactly; a final
//! nonintegral coefficient is never interpreted as an embedding of QQ in F_p.
//!
//! Generation uses bounded sparse polynomial conversion and governed integer
//! division. Verification separately evaluates the original syntax into dense
//! rational coefficient vectors using column-wise convolution and linear
//! repeated powers, then reduces integer magnitudes by byte-wise Horner.
//! It never invokes the sparse converter or generator's integer division.
//! Every original child is admitted, including children hidden by zero factors
//! or powers. Typed generators, original coefficients and the characteristic
//! remain bound through immutable accepted publication.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::{
    MAX_PRIME_FIELD_DEGREE, PolyError, PrimeFieldPolynomial, bounded_polynomial_from_expr,
};
use fsym_bigint::{NonZeroBigInt, metered_add, metered_div_rem_nonzero};
use fsym_budget::BudgetMeter;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use num_traits::{One, Zero};

const MAX_SOURCE_DEGREE: usize = 256;
const MAX_SOURCE_NODES: usize = 4096;
const MAX_SOURCE_DEPTH: usize = 64;
const MAX_BITS: u64 = 4096;

/// Keep a domain-coercion refusal distinct from resource or verifier failures.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PrimeFieldElementSourceError {
    #[error("prime-field Poly requires integer polynomial coefficients")]
    NonIntegralCoefficient,
    #[error(transparent)]
    Polynomial(#[from] PolyError),
}

/// Exact original source and its independently checked modular image.
#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldElementSource {
    source: Expr,
    polynomial: PrimeFieldPolynomial,
}

impl VerifiedPrimeFieldElementSource {
    pub fn source(&self) -> &Expr {
        &self.source
    }

    pub fn polynomial(&self) -> &PrimeFieldPolynomial {
        &self.polynomial
    }

    pub fn into_polynomial(self) -> PrimeFieldPolynomial {
        self.polynomial
    }
}

#[derive(Default)]
struct SourceSize {
    nodes: usize,
    bytes: usize,
}

fn preflight(
    expression: &Expr,
    variable: &Symbol,
    depth: usize,
    size: &mut SourceSize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    size.nodes += 1;
    if size.nodes > MAX_SOURCE_NODES || depth > MAX_SOURCE_DEPTH {
        return Err(refused("prime-field source exceeds 4096 nodes or depth 64"));
    }
    size.bytes += 128;
    match expression {
        Expr::Integer(value) => {
            if value.bits() > MAX_BITS {
                return Err(refused("prime-field input integer exceeds 4096 bits"));
            }
            size.bytes += value.bits().div_ceil(8) as usize;
        }
        Expr::Rational(value) => {
            if value.numer().bits().max(value.denom().bits()) > MAX_BITS {
                return Err(refused("prime-field input rational exceeds 4096 bits"));
            }
            size.bytes += (value.numer().bits() + value.denom().bits()).div_ceil(8) as usize;
        }
        Expr::Sym(symbol) => {
            if symbol.name.len() > 65_536 {
                return Err(refused("prime-field source symbol exceeds 65536 bytes"));
            }
            work.tick(symbol.name.len() + 1)?;
            if symbol != variable {
                return Err(refused(
                    "prime-field source contains an undeclared typed generator",
                ));
            }
            size.bytes += symbol.name.len() + 128;
        }
        Expr::Add(children) | Expr::Mul(children) => {
            if children.len() > MAX_SOURCE_NODES - size.nodes {
                return Err(refused(
                    "prime-field source arity exceeds its node envelope",
                ));
            }
            for child in children {
                preflight(child, variable, depth + 1, size, work)?;
            }
        }
        Expr::Pow(base, exponent) => {
            preflight(base, variable, depth + 1, size, work)?;
            preflight(exponent, variable, depth + 1, size, work)?;
            if !matches!(exponent.as_ref(), Expr::Integer(value)
                if value.to_i64().is_some_and(|power| (-256..=256).contains(&power)))
            {
                return Err(refused(
                    "prime-field source requires integer powers in -256..256",
                ));
            }
        }
        _ => {
            return Err(refused(
                "prime-field source contains unsupported original syntax",
            ));
        }
    }
    Ok(())
}

type Dense = Vec<BigRational>;

fn zeros(
    length: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Dense, PolyError> {
    if length > MAX_SOURCE_DEGREE + 1 {
        return Err(refused("prime-field source expansion exceeds degree 256"));
    }
    work.tick(length)?;
    work.allocate(length * (std::mem::size_of::<BigRational>() + 32))?;
    Ok(vec![BigRational::zero(); length])
}

fn trim(coefficients: &mut Dense) {
    while coefficients.last().is_some_and(Zero::is_zero) {
        coefficients.pop();
    }
}

fn scalar(
    value: &BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Dense, PolyError> {
    let mut values = zeros(1, work)?;
    values[0] = work.clone_value(value)?;
    trim(&mut values);
    Ok(values)
}

fn reference_sum(
    left: &[BigRational],
    right: &[BigRational],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Dense, PolyError> {
    let mut output = zeros(left.len().max(right.len()), work)?;
    for (index, value) in output.iter_mut().enumerate() {
        *value = match (left.get(index), right.get(index)) {
            (Some(a), Some(b)) => work.add(a, b)?,
            (Some(value), None) | (None, Some(value)) => work.clone_value(value)?,
            (None, None) => return Err(refused("invalid reference coefficient width")),
        };
    }
    trim(&mut output);
    Ok(output)
}

fn reference_product(
    left: &[BigRational],
    right: &[BigRational],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Dense, PolyError> {
    work.tick(1)?;
    if left.is_empty() || right.is_empty() {
        return Ok(Vec::new());
    }
    let mut output = zeros(left.len() + right.len() - 1, work)?;
    // Independent dense convolution by output coefficient, not the sparse
    // generator's pair accumulation or monomial arithmetic.
    for (power, value) in output.iter_mut().enumerate() {
        for (index, coefficient) in left.iter().enumerate() {
            work.tick(1)?;
            if let Some(other) = power.checked_sub(index).and_then(|j| right.get(j)) {
                let product = work.multiply(coefficient, other)?;
                *value = work.add(value, &product)?;
            }
        }
    }
    trim(&mut output);
    Ok(output)
}

fn reference_coefficients(
    expression: &Expr,
    variable: &Symbol,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Dense, PolyError> {
    work.tick(1)?;
    match expression {
        Expr::Integer(value) => {
            work.allocate(64 + 8 * value.limb_count() as usize)?;
            scalar(&BigRational::from_integer(value.clone()), work)
        }
        Expr::Rational(value) => scalar(value, work),
        Expr::Sym(symbol) if symbol == variable => {
            let mut coefficients = zeros(2, work)?;
            work.allocate(32)?;
            coefficients[1] = BigRational::one();
            Ok(coefficients)
        }
        Expr::Add(children) | Expr::Mul(children) => {
            let product = matches!(expression, Expr::Mul(_));
            let mut coefficients = if product {
                scalar(&BigRational::one(), work)?
            } else {
                Vec::new()
            };
            for child in children {
                // Do not skip the right-hand child after a zero product.
                let next = reference_coefficients(child, variable, work)?;
                coefficients = if product {
                    reference_product(&coefficients, &next, work)?
                } else {
                    reference_sum(&coefficients, &next, work)?
                };
            }
            Ok(coefficients)
        }
        Expr::Pow(base, exponent) => {
            let mut factor = reference_coefficients(base, variable, work)?;
            let Expr::Integer(power) = exponent.as_ref() else {
                return Err(refused("noninteger original prime-field exponent"));
            };
            let power = power
                .to_i64()
                .ok_or_else(|| refused("prime-field exponent is out of bounds"))?;
            if factor.is_empty() && power <= 0 {
                return Err(refused(
                    "zero base with a nonpositive original exponent is undefined",
                ));
            }
            if power < 0 {
                if factor.len() != 1 {
                    return Err(refused(
                        "prime-field polynomial source contains a nonconstant reciprocal",
                    ));
                }
                factor[0] = work.divide(&BigRational::one(), &factor[0])?;
            }
            let mut coefficients = scalar(&BigRational::one(), work)?;
            // The generator exponentiates by squaring. This reference route
            // uses the independently admitted, at-most-256 repeated products.
            for _ in 0..power.unsigned_abs() {
                coefficients = reference_product(&coefficients, &factor, work)?;
            }
            Ok(coefficients)
        }
        _ => Err(refused("unsupported reference prime-field source node")),
    }
}

fn reference_residue(
    value: &BigInt,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    let length = value.bits().div_ceil(8) as usize;
    work.allocate(length.max(1))?;
    work.tick(length)?;
    let bytes = value.to_bytes_le();
    let mut residue = 0u64;
    for &byte in bytes.iter().rev() {
        work.tick(1)?;
        residue = (residue * 256 + u64::from(byte)) % prime;
    }
    Ok(if value.is_negative() && residue != 0 {
        prime - residue
    } else {
        residue
    })
}

/// Check a proposed modular image against the complete original expression.
pub fn verify_prime_field_element_source(
    source: &Expr,
    variable: &Symbol,
    prime: u64,
    candidate: &PrimeFieldPolynomial,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldElementSource, PrimeFieldElementSourceError> {
    let mut work = PolynomialWork::new(verifier);
    PrimeFieldPolynomial::metered_new(variable, prime, &[], &mut work)?;
    if candidate.variable() != variable || candidate.prime() != prime {
        return Err(refused("prime-field source candidate changed its exact typed field").into());
    }
    let mut size = SourceSize::default();
    preflight(source, variable, 0, &mut size, &mut work)?;
    let coefficients = reference_coefficients(source, variable, &mut work)?;
    if coefficients.len() > MAX_PRIME_FIELD_DEGREE + 1 {
        return Err(refused("prime-field original polynomial exceeds degree 64").into());
    }
    work.allocate(coefficients.len() * 8)?;
    let mut residues = Vec::with_capacity(coefficients.len());
    for coefficient in &coefficients {
        if !coefficient.is_integer() {
            return Err(PrimeFieldElementSourceError::NonIntegralCoefficient);
        }
        residues.push(reference_residue(coefficient.numer(), prime, &mut work)?);
    }
    while residues.last() == Some(&0) {
        residues.pop();
    }
    work.tick(residues.len() + 1)?;
    if residues != candidate.coefficients() {
        return Err(PolyError::IdentityCheckFailed(
            "prime-field coefficients do not match the exact original integer source".into(),
        )
        .into());
    }
    work.allocate(size.bytes + std::mem::size_of::<VerifiedPrimeFieldElementSource>())?;
    work.tick(size.nodes)?;
    let result = VerifiedPrimeFieldElementSource {
        source: source.clone(),
        polynomial: PrimeFieldPolynomial::metered_new(variable, prime, &residues, &mut work)?,
    };
    work.tick(1)?;
    Ok(result)
}

/// Lower an integer polynomial into F_p[x] and independently verify its image.
pub fn checked_prime_field_element_source(
    source: &Expr,
    variable: &Symbol,
    prime: u64,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldElementSource, PrimeFieldElementSourceError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    PrimeFieldPolynomial::metered_new(variable, prime, &[], &mut generation)?;
    let mut size = SourceSize::default();
    preflight(source, variable, 0, &mut size, &mut generation)?;
    let polynomial =
        bounded_polynomial_from_expr(source, std::slice::from_ref(variable), &mut generation)?;
    let degree = polynomial.degree_in(0) as usize;
    if degree > MAX_PRIME_FIELD_DEGREE {
        return Err(refused("prime-field original polynomial exceeds degree 64").into());
    }
    generation.allocate((degree + 1) * 8 + 32)?;
    let mut residues = vec![0; degree + 1];
    let divisor = BigInt::from(prime);
    for (powers, coefficient) in polynomial.terms() {
        if !coefficient.is_integer() {
            return Err(PrimeFieldElementSourceError::NonIntegralCoefficient);
        }
        let (_, mut remainder) = metered_div_rem_nonzero(
            coefficient.numer(),
            NonZeroBigInt::new(&divisor).ok_or(PolyError::DivisionByZero)?,
            &mut generation,
        )
        .map_err(refused)?;
        if remainder.is_negative() {
            remainder = metered_add(&remainder, &divisor, &mut generation).map_err(refused)?;
        }
        residues[powers[0] as usize] = remainder
            .to_u64()
            .ok_or_else(|| refused("prime-field residue is not representable"))?;
    }
    let candidate = PrimeFieldPolynomial::metered_new(variable, prime, &residues, &mut generation)?;
    let accepted =
        verify_prime_field_element_source(source, variable, prime, &candidate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
