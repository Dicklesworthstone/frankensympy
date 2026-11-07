//! Complete bounded factorization in prime polynomial fields.
//!
//! Generation uses positive-characteristic square-free decomposition and
//! Berlekamp's fixed algebra, including characteristic two. Acceptance uses
//! a separate integer-convolution product check and the proof kernel's exact
//! Rabin irreducibility rule for every distinct monic factor. Neither a
//! successful split nor a reported nullity is accepted as irreducibility.
//!
//! The field, typed generator, original coefficients, unit and multiplicities
//! are bound explicitly. Degree is at most 64 and the prime at most 65521;
//! both phases additionally retain the standard polynomial work envelopes.
//! An exhausted splitting search refuses rather than publishing a partial
//! factorization. The composition receipt is immutable and in-process.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{BigInt, Symbol};
use fsym_proof_kernel::{CertificatePayload, Claim, DerivationTree, ProofKernel};

use crate::PolyError;
use crate::division::{PolynomialWork, refused};

pub const MAX_PRIME_FIELD_DEGREE: usize = 64;
pub const MAX_PRIME_FIELD_CHARACTERISTIC: u64 = 65_521;

type Coefficients = Vec<u64>;

/// A polynomial in the exact field F_p with a typed indeterminate.
/// Coefficients are ascending canonical residues; the empty vector is zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeFieldPolynomial {
    variable: Symbol,
    prime: u64,
    coefficients: Coefficients,
}

impl PrimeFieldPolynomial {
    pub fn new(
        variable: Symbol,
        prime: u64,
        coefficients: Coefficients,
    ) -> Result<Self, PolyError> {
        Self::metered_new(&variable, prime, &coefficients, &mut Unbounded)
    }

    pub fn metered_new(
        variable: &Symbol,
        prime: u64,
        coefficients: &[u64],
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, PolyError> {
        let mut work = PolynomialWork::new(meter);
        admit_field(variable, prime, &mut work)?;
        admit_coefficients(coefficients, prime, false, &mut work)?;
        let mut coefficients = copy(coefficients, &mut work)?;
        trim(&mut coefficients);
        work.allocate(variable.name.len() + std::mem::size_of::<Symbol>())?;
        let result = Self {
            variable: variable.clone(),
            prime,
            coefficients,
        };
        work.tick(1)?;
        Ok(result)
    }

    pub fn variable(&self) -> &Symbol {
        &self.variable
    }

    pub fn prime(&self) -> u64 {
        self.prime
    }

    pub fn coefficients(&self) -> &[u64] {
        &self.coefficients
    }
}

/// Untrusted monic factor and its positive multiplicity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeFieldFactor {
    pub coefficients: Coefficients,
    pub multiplicity: u32,
}

/// Untrusted proposal. A product identity alone is insufficient for acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeFieldFactorizationCertificate {
    pub variable: Symbol,
    pub prime: u64,
    pub source: Coefficients,
    pub unit: u64,
    pub factors: Vec<PrimeFieldFactor>,
}

/// Complete factorization accepted by exact independent checks.
#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldFactorization {
    certificate: PrimeFieldFactorizationCertificate,
    irreducibility: Vec<DerivationTree>,
}

impl VerifiedPrimeFieldFactorization {
    pub fn certificate(&self) -> &PrimeFieldFactorizationCertificate {
        &self.certificate
    }

    pub fn variable(&self) -> &Symbol {
        &self.certificate.variable
    }

    pub fn prime(&self) -> u64 {
        self.certificate.prime
    }

    pub fn source(&self) -> &[u64] {
        &self.certificate.source
    }

    pub fn unit(&self) -> u64 {
        self.certificate.unit
    }

    pub fn factors(&self) -> &[PrimeFieldFactor] {
        &self.certificate.factors
    }

    /// Each derivation proves the corresponding exact factor irreducible in
    /// this field. It is not an equality claim about its integer lift.
    pub fn irreducibility(&self) -> &[DerivationTree] {
        &self.irreducibility
    }
}

fn admit_field(
    variable: &Symbol,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if variable.name.len() > 65_536 {
        return Err(refused("prime-field generator name bound"));
    }
    admit_characteristic(prime, work)
}

/// Shared field admission also covers explicit zero-variable constant rings.
pub(super) fn admit_characteristic(
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if !(2..=MAX_PRIME_FIELD_CHARACTERISTIC).contains(&prime) {
        return Err(refused(
            "prime field requires a prime in 2..65521 and a bounded symbol",
        ));
    }
    let mut divisor = 2;
    while divisor * divisor <= prime {
        work.tick(1)?;
        if prime.is_multiple_of(divisor) {
            return Err(refused("polynomial field characteristic is composite"));
        }
        divisor += 1;
    }
    Ok(())
}

fn admit_coefficients(
    coefficients: &[u64],
    prime: u64,
    canonical: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if coefficients.len() > MAX_PRIME_FIELD_DEGREE + 1 {
        return Err(refused("prime-field polynomial exceeds degree 64"));
    }
    work.tick(coefficients.len())?;
    if coefficients.iter().any(|&value| value >= prime)
        || (canonical && coefficients.last() == Some(&0))
    {
        return Err(refused(
            "prime-field coefficients are not canonical residues",
        ));
    }
    Ok(())
}

pub(super) fn zeros(
    length: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    work.allocate(length * std::mem::size_of::<u64>())?;
    Ok(vec![0; length])
}

pub(super) fn copy(
    values: &[u64],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    work.tick(values.len())?;
    work.allocate(std::mem::size_of_val(values))?;
    Ok(values.to_vec())
}

pub(super) fn trim(values: &mut Coefficients) {
    while values.last() == Some(&0) {
        values.pop();
    }
}

pub(super) fn scalar_power(
    mut base: u64,
    mut exponent: u64,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<u64, PolyError> {
    let mut result = 1;
    while exponent != 0 {
        work.tick(1)?;
        if exponent & 1 != 0 {
            result = result * base % prime;
        }
        base = base * base % prime;
        exponent >>= 1;
    }
    Ok(result)
}

fn monic(
    values: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let Some(&leading) = values.last() else {
        return Ok(Vec::new());
    };
    let inverse = scalar_power(leading, prime - 2, prime, work)?;
    let mut result = copy(values, work)?;
    work.tick(result.len())?;
    for value in &mut result {
        *value = *value * inverse % prime;
    }
    Ok(result)
}

pub(super) fn multiply(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    if left.is_empty() || right.is_empty() {
        work.tick(1)?;
        return Ok(Vec::new());
    }
    let mut result = zeros(left.len() + right.len() - 1, work)?;
    for (i, &a) in left.iter().enumerate() {
        work.tick(right.len())?;
        for (j, &b) in right.iter().enumerate() {
            result[i + j] = (result[i + j] + a * b) % prime;
        }
    }
    trim(&mut result);
    Ok(result)
}

pub(super) fn divide(
    dividend: &[u64],
    divisor: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(Coefficients, Coefficients), PolyError> {
    let Some(&leading) = divisor.last() else {
        return Err(PolyError::DivisionByZero);
    };
    let mut remainder = copy(dividend, work)?;
    let length = dividend
        .len()
        .checked_sub(divisor.len())
        .map_or(0, |length| length + 1);
    let mut quotient = zeros(length, work)?;
    let inverse = scalar_power(leading, prime - 2, prime, work)?;
    while remainder.len() >= divisor.len() {
        work.tick(divisor.len())?;
        let offset = remainder.len() - divisor.len();
        let coefficient = remainder[remainder.len() - 1] * inverse % prime;
        quotient[offset] = coefficient;
        for (index, &value) in divisor.iter().enumerate() {
            remainder[offset + index] =
                (remainder[offset + index] + prime - coefficient * value % prime) % prime;
        }
        trim(&mut remainder);
    }
    trim(&mut quotient);
    Ok((quotient, remainder))
}

fn exact_quotient(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let (quotient, remainder) = divide(left, right, prime, work)?;
    if !remainder.is_empty() {
        return Err(refused("prime-field factor division was not exact"));
    }
    Ok(quotient)
}

fn gcd(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let (mut a, mut b) = (copy(left, work)?, copy(right, work)?);
    while !b.is_empty() {
        let (_, remainder) = divide(&a, &b, prime, work)?;
        (a, b) = (b, remainder);
    }
    monic(&a, prime, work)
}

fn power_mod(
    base: &[u64],
    mut exponent: u64,
    modulus: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let mut result = copy(&[1], work)?;
    let mut base = divide(base, modulus, prime, work)?.1;
    while exponent != 0 {
        work.tick(1)?;
        if exponent & 1 != 0 {
            result = divide(
                &multiply(&result, &base, prime, work)?,
                modulus,
                prime,
                work,
            )?
            .1;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = divide(&multiply(&base, &base, prime, work)?, modulus, prime, work)?.1;
        }
    }
    Ok(result)
}

pub(super) fn derivative(
    values: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let mut result = zeros(values.len().saturating_sub(1), work)?;
    work.tick(values.len())?;
    for (power, &value) in values.iter().enumerate().skip(1) {
        result[power - 1] = value * power as u64 % prime;
    }
    trim(&mut result);
    Ok(result)
}

/// Characteristic-p square-free decomposition, including inseparable powers.
fn square_free(
    values: &[u64],
    prime: u64,
    multiplier: u32,
    output: &mut Vec<PrimeFieldFactor>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if values.len() <= 1 {
        return Ok(());
    }
    let differentiated = derivative(values, prime, work)?;
    let mut repeated = gcd(values, &differentiated, prime, work)?;
    let mut remaining = exact_quotient(values, &repeated, prime, work)?;
    let mut multiplicity = 1u32;
    while remaining != [1] {
        work.tick(1)?;
        if multiplicity > MAX_PRIME_FIELD_DEGREE as u32 {
            return Err(refused(
                "positive-characteristic square-free loop exceeded degree",
            ));
        }
        let shared = gcd(&remaining, &repeated, prime, work)?;
        let distinct = exact_quotient(&remaining, &shared, prime, work)?;
        if distinct.len() > 1 {
            work.allocate(std::mem::size_of::<PrimeFieldFactor>())?;
            output.push(PrimeFieldFactor {
                coefficients: distinct,
                multiplicity: multiplicity * multiplier,
            });
        }
        remaining = shared;
        repeated = exact_quotient(&repeated, &remaining, prime, work)?;
        multiplicity += 1;
    }
    if repeated != [1] {
        let mut root = zeros((repeated.len() - 1) / prime as usize + 1, work)?;
        for (power, &coefficient) in repeated.iter().enumerate() {
            work.tick(1)?;
            if !(power as u64).is_multiple_of(prime) && coefficient != 0 {
                return Err(refused("inseparable remainder is not a pth power"));
            }
            if (power as u64).is_multiple_of(prime) {
                root[power / prime as usize] = coefficient;
            }
        }
        let next = u64::from(multiplier) * prime;
        if next > MAX_PRIME_FIELD_DEGREE as u64 {
            return Err(refused("prime-field multiplicity exceeds degree 64"));
        }
        square_free(&root, prime, next as u32, output, work)?;
    }
    Ok(())
}

/// Nullspace of Frobenius minus identity, by modular Gauss-Jordan elimination.
fn fixed_algebra(
    values: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Coefficients>, PolyError> {
    let degree = values.len() - 1;
    work.allocate(degree * std::mem::size_of::<Coefficients>())?;
    let mut matrix = (0..degree)
        .map(|_| zeros(degree, work))
        .collect::<Result<Vec<_>, _>>()?;
    let x_to_p = power_mod(&[0, 1], prime, values, prime, work)?;
    let mut column = copy(&[1], work)?;
    for col in 0..degree {
        work.tick(degree)?;
        for (row, entries) in matrix.iter_mut().enumerate() {
            entries[col] = column.get(row).copied().unwrap_or(0);
            if row == col {
                entries[col] = (entries[col] + prime - 1) % prime;
            }
        }
        if col + 1 < degree {
            column = divide(
                &multiply(&column, &x_to_p, prime, work)?,
                values,
                prime,
                work,
            )?
            .1;
        }
    }
    work.allocate(degree * std::mem::size_of::<usize>())?;
    let mut pivots = Vec::with_capacity(degree);
    for col in 0..degree {
        work.tick(degree)?;
        let Some(row) = (pivots.len()..degree).find(|&row| matrix[row][col] != 0) else {
            continue;
        };
        let pivot = pivots.len();
        matrix.swap(pivot, row);
        let inverse = scalar_power(matrix[pivot][col], prime - 2, prime, work)?;
        work.tick(degree)?;
        for value in &mut matrix[pivot] {
            *value = *value * inverse % prime;
        }
        let pivot_values = copy(&matrix[pivot], work)?;
        for (row, values) in matrix.iter_mut().enumerate() {
            if row != pivot {
                let coefficient = values[col];
                work.tick(degree)?;
                for (value, &pivot_value) in values.iter_mut().zip(&pivot_values) {
                    *value = (*value + prime - coefficient * pivot_value % prime) % prime;
                }
            }
        }
        pivots.push(col);
    }
    work.allocate((degree - pivots.len()) * std::mem::size_of::<Coefficients>())?;
    let mut basis = Vec::with_capacity(degree - pivots.len());
    for free in 0..degree {
        work.tick(degree)?;
        if !pivots.contains(&free) {
            let mut vector = zeros(degree, work)?;
            vector[free] = 1;
            for (row, &pivot) in pivots.iter().enumerate() {
                vector[pivot] = (prime - matrix[row][free]) % prime;
            }
            trim(&mut vector);
            basis.push(vector);
        }
    }
    Ok(basis)
}

fn split_square_free(
    values: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Coefficients>, PolyError> {
    let basis = fixed_algebra(values, prime, work)?;
    if basis.is_empty() {
        return Err(refused(
            "Berlekamp fixed algebra omitted the constant polynomial",
        ));
    }
    work.allocate(basis.len() * std::mem::size_of::<Coefficients>())?;
    let mut factors = Vec::with_capacity(basis.len());
    factors.push(copy(values, work)?);
    for vector in &basis {
        if factors.len() == basis.len() {
            break;
        }
        if vector.len() <= 1 {
            continue;
        }
        for scalar in 0..prime {
            if factors.len() == basis.len() {
                break;
            }
            let mut shifted = copy(vector, work)?;
            shifted[0] = (shifted[0] + prime - scalar) % prime;
            trim(&mut shifted);
            let length = factors.len();
            for index in 0..length {
                work.tick(1)?;
                if factors[index].len() <= 2 {
                    continue;
                }
                let common = gcd(&factors[index], &shifted, prime, work)?;
                if common.len() > 1 && common.len() < factors[index].len() {
                    let other = exact_quotient(&factors[index], &common, prime, work)?;
                    factors[index] = common;
                    factors.push(other);
                }
            }
        }
    }
    if factors.len() != basis.len() {
        return Err(refused(
            "bounded Berlekamp splitting did not recover every factor",
        ));
    }
    Ok(factors)
}

fn factor_order(left: &PrimeFieldFactor, right: &PrimeFieldFactor) -> std::cmp::Ordering {
    left.coefficients
        .len()
        .cmp(&right.coefficients.len())
        .then(left.multiplicity.cmp(&right.multiplicity))
        .then_with(|| {
            left.coefficients
                .iter()
                .rev()
                .cmp(right.coefficients.iter().rev())
        })
}

/// Independent coefficient columns: accumulate whole integer convolution sums
/// in u128, then reduce once per column. Generation reduces each u64 term.
fn reference_product(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let mut result = zeros(left.len() + right.len() - 1, work)?;
    for (power, coefficient) in result.iter_mut().enumerate() {
        work.tick(left.len())?;
        let mut sum = 0u128;
        for (i, &a) in left.iter().enumerate() {
            if let Some(j) = power.checked_sub(i).filter(|&j| j < right.len()) {
                sum += u128::from(a) * u128::from(right[j]);
            }
        }
        *coefficient = (sum % u128::from(prime)) as u64;
    }
    trim(&mut result);
    Ok(result)
}

/// Check the complete exact factorization independently of the splitting code.
pub fn verify_prime_field_factorization(
    input: &PrimeFieldPolynomial,
    certificate: &PrimeFieldFactorizationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldFactorization, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit_field(input.variable(), input.prime(), &mut work)?;
    admit_coefficients(input.coefficients(), input.prime(), true, &mut work)?;
    if certificate.variable != *input.variable()
        || certificate.prime != input.prime()
        || certificate.source != input.coefficients()
        || certificate.unit >= input.prime()
        || certificate.factors.len() > MAX_PRIME_FIELD_DEGREE
    {
        return Err(refused(
            "prime-field factorization is not bound to the original ring and polynomial",
        ));
    }
    let unit = input.coefficients().last().copied().unwrap_or(0);
    if certificate.unit != unit
        || (input.coefficients().len() <= 1 && !certificate.factors.is_empty())
    {
        return Err(refused(
            "prime-field unit or zero/constant factorization is incorrect",
        ));
    }
    let mut total_degree = 0u64;
    for (index, factor) in certificate.factors.iter().enumerate() {
        admit_coefficients(&factor.coefficients, input.prime(), true, &mut work)?;
        if factor.coefficients.len() < 2
            || factor.coefficients.last() != Some(&1)
            || factor.multiplicity == 0
            || factor.multiplicity > MAX_PRIME_FIELD_DEGREE as u32
            || certificate.factors[..index]
                .iter()
                .any(|previous| previous.coefficients == factor.coefficients)
            || (index > 0 && !factor_order(&certificate.factors[index - 1], factor).is_lt())
        {
            return Err(refused(
                "prime-field factors must be distinct sorted monic nonconstants with positive multiplicities",
            ));
        }
        total_degree += (factor.coefficients.len() - 1) as u64 * u64::from(factor.multiplicity);
    }
    if total_degree != input.coefficients().len().saturating_sub(1) as u64 {
        return Err(refused(
            "prime-field factor degrees do not cover the original polynomial",
        ));
    }
    let mut product = copy(&[unit], &mut work)?;
    for factor in &certificate.factors {
        for _ in 0..factor.multiplicity {
            product = reference_product(&product, &factor.coefficients, input.prime(), &mut work)?;
        }
    }
    trim(&mut product);
    if product != input.coefficients() {
        return Err(PolyError::IdentityCheckFailed(
            "prime-field factor product coefficients differ from the original polynomial".into(),
        ));
    }
    work.allocate(certificate.factors.len() * std::mem::size_of::<DerivationTree>() + 256)?;
    let context = ImmutableAssumptionsSnapshot::empty();
    let mut irreducibility = Vec::with_capacity(certificate.factors.len());
    for factor in &certificate.factors {
        // Prepay coefficient/claim/rule copies and the small fixed kernel maps.
        work.allocate(factor.coefficients.len() * 512 + 4096)?;
        let coeffs: Vec<_> = factor
            .coefficients
            .iter()
            .copied()
            .map(BigInt::from)
            .collect();
        let prime = BigInt::from(input.prime());
        let claim = Claim::IrreducibleModPrime {
            coeffs: coeffs.clone(),
            prime: prime.clone(),
        };
        let payload = CertificatePayload::RabinIrreducibilityModPrime { coeffs, prime };
        let mut kernel = ProofKernel::new((*context).clone());
        let root = kernel
            .prove_certificate_lemma("RabinIrreducibilityModPrime", claim, payload, &mut work)
            .map_err(refused)?;
        irreducibility.push(kernel.export_derivation(root).map_err(refused)?);
    }
    work.allocate(
        certificate.variable.name.len()
            + std::mem::size_of::<PrimeFieldFactorizationCertificate>()
            + certificate.factors.len() * std::mem::size_of::<PrimeFieldFactor>(),
    )?;
    let factors = certificate
        .factors
        .iter()
        .map(|factor| {
            Ok(PrimeFieldFactor {
                coefficients: copy(&factor.coefficients, &mut work)?,
                multiplicity: factor.multiplicity,
            })
        })
        .collect::<Result<_, PolyError>>()?;
    let certificate = PrimeFieldFactorizationCertificate {
        variable: certificate.variable.clone(),
        prime: certificate.prime,
        source: copy(&certificate.source, &mut work)?,
        unit: certificate.unit,
        factors,
    };
    work.tick(1)?;
    Ok(VerifiedPrimeFieldFactorization {
        certificate,
        irreducibility,
    })
}

pub fn checked_prime_field_factorization(
    input: &PrimeFieldPolynomial,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldFactorization, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    admit_field(input.variable(), input.prime(), &mut generation)?;
    admit_coefficients(input.coefficients(), input.prime(), true, &mut generation)?;
    let unit = input.coefficients().last().copied().unwrap_or(0);
    generation.allocate(MAX_PRIME_FIELD_DEGREE * std::mem::size_of::<PrimeFieldFactor>())?;
    let mut factors = Vec::with_capacity(MAX_PRIME_FIELD_DEGREE);
    if input.coefficients().len() > 1 {
        let normalized = monic(input.coefficients(), input.prime(), &mut generation)?;
        generation.allocate(MAX_PRIME_FIELD_DEGREE * std::mem::size_of::<PrimeFieldFactor>())?;
        let mut square_free_factors = Vec::with_capacity(MAX_PRIME_FIELD_DEGREE);
        square_free(
            &normalized,
            input.prime(),
            1,
            &mut square_free_factors,
            &mut generation,
        )?;
        for component in square_free_factors {
            for coefficients in
                split_square_free(&component.coefficients, input.prime(), &mut generation)?
            {
                factors.push(PrimeFieldFactor {
                    coefficients,
                    multiplicity: component.multiplicity,
                });
            }
        }
    }
    generation.tick(factors.len() * factors.len() * (MAX_PRIME_FIELD_DEGREE + 1))?;
    factors.sort_by(factor_order);
    generation.allocate(
        input.variable().name.len() + std::mem::size_of::<PrimeFieldFactorizationCertificate>(),
    )?;
    let certificate = PrimeFieldFactorizationCertificate {
        variable: input.variable().clone(),
        prime: input.prime(),
        source: copy(input.coefficients(), &mut generation)?,
        unit,
        factors,
    };
    let accepted = verify_prime_field_factorization(input, &certificate, verifier)?;
    generation.tick(1)?;
    Ok(accepted)
}
