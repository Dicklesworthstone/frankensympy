//! Checked original integer-polynomial sources for multivariate prime fields.
//!
//! The coefficient-domain boundary is ZZ[x_1, ..., x_n] -> F_p[x_1, ..., x_n].
//! Rational intermediates may cancel, but every final rational coefficient
//! must be integral before modular reduction. The final original polynomial
//! has total degree at most 64, including terms that vanish modulo p.
//!
//! Generation uses the bounded QQ polynomial converter and governed integer
//! division. Verification separately interprets the complete original syntax
//! into ordered sparse rational maps: ordered column merges for sums, complete
//! sorted product rows for multiplication, and linear repeated powers. It
//! neither invokes the generator parser nor its accumulation/power routines.
//! Integer residues are checked by byte-wise Horner rather than division.
//!
//! All original children are admitted before arithmetic, including children
//! hidden by zero products and zeroth powers. Each phase shares the caller's
//! meter and an intrinsic 20-million-unit / 128-MiB cumulative work ledger.
//! Sources have at most 4096 nodes, depth 64 and 4096-bit coefficients; rational
//! intermediates have total degree at most 256 and at most 4096 nonzero terms.
//! Accepted receipts retain the original owned expression and exact ordered
//! typed field. This is an in-process checked composition, not a new portable
//! proof-kernel certificate.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::{
    PolyError, PrimeFieldElementSourceError, PrimeFieldMultivariatePolynomial,
    PrimeFieldMultivariateTerms, bounded_polynomial_from_expr,
};
use fsym_bigint::{NonZeroBigInt, metered_add, metered_div_rem_nonzero};
use fsym_budget::BudgetMeter;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const MAX_SOURCE_DEGREE: u64 = 256;
const MAX_FINAL_DEGREE: u64 = 64;
const MAX_SOURCE_TERMS: usize = 4096;
const MAX_SOURCE_NODES: usize = 4096;
const MAX_SOURCE_DEPTH: usize = 64;
const MAX_BITS: u64 = 4096;

type Sparse = BTreeMap<Vec<u32>, BigRational>;

/// Immutable original syntax and its independently verified prime-field image.
#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldMultivariateSource {
    source: Expr,
    polynomial: PrimeFieldMultivariatePolynomial,
}

impl VerifiedPrimeFieldMultivariateSource {
    pub fn source(&self) -> &Expr {
        &self.source
    }

    pub fn polynomial(&self) -> &PrimeFieldMultivariatePolynomial {
        &self.polynomial
    }

    pub fn into_polynomial(self) -> PrimeFieldMultivariatePolynomial {
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
    generators: &[Symbol],
    depth: usize,
    size: &mut SourceSize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    size.nodes += 1;
    if size.nodes > MAX_SOURCE_NODES || depth > MAX_SOURCE_DEPTH {
        return Err(refused(
            "multivariate prime-field source exceeds 4096 nodes or depth 64",
        ));
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
            work.tick(
                symbol.name.len() + generators.iter().map(|g| g.name.len() + 1).sum::<usize>() + 1,
            )?;
            if !generators.contains(symbol) {
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
                preflight(child, generators, depth + 1, size, work)?;
            }
        }
        Expr::Pow(base, exponent) => {
            preflight(base, generators, depth + 1, size, work)?;
            preflight(exponent, generators, depth + 1, size, work)?;
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

fn degree(powers: &[u32]) -> u64 {
    powers.iter().map(|&power| u64::from(power)).sum()
}

fn owned_term(
    output: &mut Sparse,
    powers: Vec<u32>,
    coefficient: BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(powers.len() + 1)?;
    if degree(&powers) > MAX_SOURCE_DEGREE {
        return Err(refused("prime-field source expansion exceeds degree 256"));
    }
    if !coefficient.is_zero() {
        if output.len() >= MAX_SOURCE_TERMS {
            return Err(refused("prime-field source expansion exceeds 4096 terms"));
        }
        work.allocate(256 + std::mem::size_of::<BigRational>())?;
        if output.insert(powers, coefficient).is_some() {
            return Err(refused("duplicate reference polynomial output column"));
        }
    }
    Ok(())
}

fn copied_powers(
    powers: &[u32],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u32>, PolyError> {
    work.tick(powers.len())?;
    work.allocate(std::mem::size_of_val(powers))?;
    Ok(powers.to_vec())
}

fn scalar(
    value: &BigRational,
    width: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Sparse, PolyError> {
    work.tick(1)?;
    let mut output = Sparse::new();
    work.allocate(width * std::mem::size_of::<u32>())?;
    let powers = vec![0; width];
    let coefficient = work.clone_value(value)?;
    owned_term(&mut output, powers, coefficient, work)?;
    Ok(output)
}

fn reference_sum(
    left: &Sparse,
    right: &Sparse,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Sparse, PolyError> {
    let (mut a, mut b) = (left.iter().peekable(), right.iter().peekable());
    let mut output = Sparse::new();
    // Each ordered output column is visited once. No generator accumulator
    // or mutable copy of one operand participates in this interpretation.
    while a.peek().is_some() || b.peek().is_some() {
        work.tick(9)?;
        let ordering = match (a.peek(), b.peek()) {
            (Some((p, _)), Some((q, _))) => p.cmp(q),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => break,
        };
        let (powers, coefficient) = match ordering {
            Ordering::Less => {
                let (powers, value) = a.next().ok_or_else(|| refused("missing left column"))?;
                (copied_powers(powers, work)?, work.clone_value(value)?)
            }
            Ordering::Greater => {
                let (powers, value) = b.next().ok_or_else(|| refused("missing right column"))?;
                (copied_powers(powers, work)?, work.clone_value(value)?)
            }
            Ordering::Equal => {
                let (powers, value) = a.next().ok_or_else(|| refused("missing left column"))?;
                let (_, other) = b.next().ok_or_else(|| refused("missing right column"))?;
                (copied_powers(powers, work)?, work.add(value, other)?)
            }
        };
        owned_term(&mut output, powers, coefficient, work)?;
    }
    Ok(output)
}

fn reference_product(
    left: &Sparse,
    right: &Sparse,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Sparse, PolyError> {
    work.tick(left.len() + right.len() + 1)?;
    if left.is_empty() || right.is_empty() {
        return Ok(Sparse::new());
    }
    let left_degree = left.keys().map(|powers| degree(powers)).max().unwrap_or(0);
    let right_degree = right.keys().map(|powers| degree(powers)).max().unwrap_or(0);
    if left_degree + right_degree > MAX_SOURCE_DEGREE {
        return Err(refused("prime-field source expansion exceeds degree 256"));
    }
    let count = left
        .len()
        .checked_mul(right.len())
        .ok_or_else(|| refused("prime-field reference product count overflowed"))?;
    let bytes = count
        .checked_mul(std::mem::size_of::<(Vec<u32>, BigRational)>())
        .ok_or_else(|| refused("prime-field reference product storage overflowed"))?;
    work.allocate(bytes)?;
    let mut products = Vec::with_capacity(count);
    for (left_powers, a) in left {
        for (right_powers, b) in right {
            work.tick(left_powers.len() + 1)?;
            work.allocate(left_powers.len() * std::mem::size_of::<u32>())?;
            let mut powers = Vec::with_capacity(left_powers.len());
            for (&p, &q) in left_powers.iter().zip(right_powers) {
                powers.push(
                    p.checked_add(q)
                        .ok_or_else(|| refused("prime-field reference exponent overflowed"))?,
                );
            }
            let coefficient = work.multiply(a, b)?;
            products.push((powers, coefficient));
        }
    }
    // Unlike generator pair accumulation, first retain the complete product
    // rows, then group them by an independent lexicographic sort. Charge a
    // conservative comparison bound before the non-allocating sort runs.
    let sorting = count
        .checked_mul((usize::BITS - count.leading_zeros()) as usize + 1)
        .and_then(|value| value.checked_mul(32))
        .ok_or_else(|| refused("prime-field reference sorting bound overflowed"))?;
    work.tick(sorting)?;
    products.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut output = Sparse::new();
    let mut rows = products.into_iter().peekable();
    while let Some((powers, mut coefficient)) = rows.next() {
        while rows.peek().is_some_and(|(next, _)| next == &powers) {
            work.tick(powers.len() + 1)?;
            let (_, other) = rows.next().ok_or_else(|| refused("missing product row"))?;
            coefficient = work.add(&coefficient, &other)?;
        }
        owned_term(&mut output, powers, coefficient, work)?;
    }
    Ok(output)
}

fn reference_coefficients(
    expression: &Expr,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Sparse, PolyError> {
    work.tick(1)?;
    match expression {
        Expr::Integer(value) => {
            work.allocate(64 + 8 * value.limb_count() as usize)?;
            scalar(
                &BigRational::from_integer(value.clone()),
                generators.len(),
                work,
            )
        }
        Expr::Rational(value) => scalar(value, generators.len(), work),
        Expr::Sym(symbol) => {
            work.tick(generators.iter().map(|g| g.name.len() + 1).sum::<usize>())?;
            let position = generators
                .iter()
                .position(|generator| generator == symbol)
                .ok_or_else(|| refused("undeclared reference prime-field generator"))?;
            let mut output = Sparse::new();
            work.allocate(generators.len() * 4 + 32)?;
            let mut powers = vec![0; generators.len()];
            powers[position] = 1;
            owned_term(&mut output, powers, BigRational::one(), work)?;
            Ok(output)
        }
        Expr::Add(children) | Expr::Mul(children) => {
            let product = matches!(expression, Expr::Mul(_));
            let mut coefficients = if product {
                scalar(&BigRational::one(), generators.len(), work)?
            } else {
                Sparse::new()
            };
            for child in children {
                // Evaluate every child even when a prior factor is zero.
                let next = reference_coefficients(child, generators, work)?;
                coefficients = if product {
                    reference_product(&coefficients, &next, work)?
                } else {
                    reference_sum(&coefficients, &next, work)?
                };
            }
            Ok(coefficients)
        }
        Expr::Pow(base, exponent) => {
            let mut factor = reference_coefficients(base, generators, work)?;
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
                if factor.len() != 1 || factor.keys().any(|powers| powers.iter().any(|&p| p != 0)) {
                    return Err(refused(
                        "prime-field polynomial source contains a nonconstant reciprocal",
                    ));
                }
                let value = factor
                    .values_mut()
                    .next()
                    .ok_or(PolyError::DivisionByZero)?;
                *value = work.divide(&BigRational::one(), value)?;
            }
            let mut coefficients = scalar(&BigRational::one(), generators.len(), work)?;
            // Linear repetition is separate from the generator's squaring.
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

/// Independently check a proposed multivariate image of an original source.
pub fn verify_prime_field_multivariate_source(
    source: &Expr,
    generators: &[Symbol],
    prime: u64,
    candidate: &PrimeFieldMultivariatePolynomial,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateSource, PrimeFieldElementSourceError> {
    let mut work = PolynomialWork::new(verifier);
    PrimeFieldMultivariatePolynomial::metered_new(
        generators,
        prime,
        &PrimeFieldMultivariateTerms::new(),
        &mut work,
    )?;
    if candidate.generators() != generators || candidate.prime() != prime {
        return Err(refused("prime-field source candidate changed its exact typed field").into());
    }
    let mut size = SourceSize::default();
    preflight(source, generators, 0, &mut size, &mut work)?;
    let coefficients = reference_coefficients(source, generators, &mut work)?;
    let mut residues = PrimeFieldMultivariateTerms::new();
    for (powers, coefficient) in coefficients {
        work.tick(powers.len() + 1)?;
        if degree(&powers) > MAX_FINAL_DEGREE {
            return Err(refused("prime-field original polynomial exceeds total degree 64").into());
        }
        if !coefficient.is_integer() {
            return Err(PrimeFieldElementSourceError::NonIntegralCoefficient);
        }
        let residue = reference_residue(coefficient.numer(), prime, &mut work)?;
        if residue != 0 {
            work.allocate(256 + std::mem::size_of::<u64>())?;
            residues.insert(powers, residue);
        }
    }
    work.tick(residues.len() * (generators.len() + 1) + 1)?;
    if &residues != candidate.terms() {
        return Err(PolyError::IdentityCheckFailed(
            "prime-field monomials or coefficients do not match the exact original integer source"
                .into(),
        )
        .into());
    }
    work.allocate(size.bytes + std::mem::size_of::<VerifiedPrimeFieldMultivariateSource>())?;
    work.tick(size.nodes)?;
    let result = VerifiedPrimeFieldMultivariateSource {
        source: source.clone(),
        polynomial: PrimeFieldMultivariatePolynomial::metered_new(
            generators, prime, &residues, &mut work,
        )?,
    };
    work.tick(1)?;
    Ok(result)
}

/// Lower an original integer polynomial and independently verify its F_p image.
pub fn checked_prime_field_multivariate_source(
    source: &Expr,
    generators: &[Symbol],
    prime: u64,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateSource, PrimeFieldElementSourceError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    PrimeFieldMultivariatePolynomial::metered_new(
        generators,
        prime,
        &PrimeFieldMultivariateTerms::new(),
        &mut generation,
    )?;
    let mut size = SourceSize::default();
    preflight(source, generators, 0, &mut size, &mut generation)?;
    let polynomial = bounded_polynomial_from_expr(source, generators, &mut generation)?;
    let mut residues = PrimeFieldMultivariateTerms::new();
    let divisor = BigInt::from(prime);
    generation.allocate(32)?;
    for (powers, coefficient) in polynomial.terms() {
        generation.tick(powers.len() + 1)?;
        if degree(powers) > MAX_FINAL_DEGREE {
            return Err(refused("prime-field original polynomial exceeds total degree 64").into());
        }
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
        let residue = remainder
            .to_u64()
            .ok_or_else(|| refused("prime-field residue is not representable"))?;
        if residue != 0 {
            generation.allocate(256 + powers.len() * 4)?;
            residues.insert(powers.clone(), residue);
        }
    }
    let candidate = PrimeFieldMultivariatePolynomial::metered_new(
        generators,
        prime,
        &residues,
        &mut generation,
    )?;
    let accepted = verify_prime_field_multivariate_source(
        source,
        generators,
        prime,
        &candidate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
