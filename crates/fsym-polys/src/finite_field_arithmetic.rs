//! Checked arithmetic in F_p[x], preserving the exact characteristic.
//!
//! Generation reuses the bounded modular arithmetic used by factorization.
//! Acceptance uses separate signed-sum and whole-column u128 convolution
//! routines. Division proves reconstruction and the remainder degree bound.
//! GCD evidence proves common divisibility and a Bezout identity; neither
//! condition alone establishes greatestness. The same evidence supports LCM,
//! extended GCD, and a reduced inverse or a proved nonunit.
//!
//! Sources are immutable canonical `PrimeFieldPolynomial` values, whose
//! constructor verifies primality, typed variable, and coefficient bounds.
//! Each accepted result retains both complete sources and the operation.
//! Degrees of sources, outputs, and witnesses are at most 64, and p <= 65521.
//! Intermediate products have degree at most 128. Each whole phase retains
//! cumulative 20M-step / 128MiB logical-allocation bounds and cancellation
//! through publication. These receipts are in-process, not portable proofs.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field::{copy, derivative, divide, multiply, scalar_power, trim, zeros};
use crate::{MAX_PRIME_FIELD_DEGREE, PolyError, PrimeFieldPolynomial};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;

type Coefficients = Vec<u64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimeFieldArithmeticOperation {
    Normalize,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Monic,
    Gcd,
    ExtendedGcd,
    Lcm,
    Invert,
    Derivative(u32),
    Power(u64),
    Compose,
}

impl PrimeFieldArithmeticOperation {
    fn binary(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Subtract
                | Self::Multiply
                | Self::Divide
                | Self::Gcd
                | Self::ExtendedGcd
                | Self::Lcm
                | Self::Invert
                | Self::Compose
        )
    }

    fn requires_gcd(self) -> bool {
        matches!(
            self,
            Self::Gcd | Self::ExtendedGcd | Self::Lcm | Self::Invert
        )
    }
}

/// Untrusted identities a=h*left_quotient, b=h*right_quotient, s*a+t*b=h.
/// The nonzero h must be monic. All five vectors are ascending canonical
/// residues in the same typed field as the original operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeFieldGcdWitness {
    pub gcd: Coefficients,
    pub left_quotient: Coefficients,
    pub right_quotient: Coefficients,
    pub bezout_left: Coefficients,
    pub bezout_right: Coefficients,
}

/// Results: divide -> [q,r], extended GCD -> [s,t,h], inverse -> [inverse]
/// or [] for a proved nonunit, and every other operation -> [result].
/// A zero polynomial is an empty inner vector, not an absent result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeFieldArithmeticCertificate {
    pub results: Vec<Coefficients>,
    pub gcd: Option<PrimeFieldGcdWitness>,
    /// For an invertible residue class, a*inverse = 1 + inverse_quotient*b.
    pub inverse_quotient: Option<Coefficients>,
}

#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldArithmetic {
    operation: PrimeFieldArithmeticOperation,
    left: PrimeFieldPolynomial,
    right: Option<PrimeFieldPolynomial>,
    certificate: PrimeFieldArithmeticCertificate,
}

impl VerifiedPrimeFieldArithmetic {
    pub fn operation(&self) -> PrimeFieldArithmeticOperation {
        self.operation
    }
    pub fn left(&self) -> &PrimeFieldPolynomial {
        &self.left
    }
    pub fn right(&self) -> Option<&PrimeFieldPolynomial> {
        self.right.as_ref()
    }
    pub fn variable(&self) -> &Symbol {
        self.left.variable()
    }
    pub fn prime(&self) -> u64 {
        self.left.prime()
    }
    pub fn results(&self) -> &[Coefficients] {
        &self.certificate.results
    }
    pub fn into_results(self) -> Vec<Coefficients> {
        self.certificate.results
    }
    pub fn certificate(&self) -> &PrimeFieldArithmeticCertificate {
        &self.certificate
    }
}

fn canonical(
    values: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if values.len() > MAX_PRIME_FIELD_DEGREE + 1 {
        return Err(refused("prime-field arithmetic exceeds degree 64"));
    }
    work.tick(values.len())?;
    if values.last() == Some(&0) || values.iter().any(|&value| value >= prime) {
        return Err(refused(
            "prime-field arithmetic requires canonical residues",
        ));
    }
    Ok(())
}

fn admit(
    operation: PrimeFieldArithmeticOperation,
    left: &PrimeFieldPolynomial,
    right: Option<&PrimeFieldPolynomial>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if operation.binary() != right.is_some() {
        return Err(refused(
            "prime-field arithmetic has the wrong operand count",
        ));
    }
    canonical(left.coefficients(), left.prime(), work)?;
    if let Some(right) = right {
        work.tick(left.variable().name.len() + right.variable().name.len() + 1)?;
        if left.prime() != right.prime() || left.variable() != right.variable() {
            return Err(refused(
                "prime-field arithmetic changed the exact typed polynomial ring",
            ));
        }
        canonical(right.coefficients(), right.prime(), work)?;
        if matches!(
            operation,
            PrimeFieldArithmeticOperation::Divide | PrimeFieldArithmeticOperation::Invert
        ) && right.coefficients().is_empty()
        {
            return Err(PolyError::DivisionByZero);
        }
        if operation == PrimeFieldArithmeticOperation::Compose {
            let degree = left.coefficients().len().saturating_sub(1)
                * right.coefficients().len().saturating_sub(1);
            if degree > MAX_PRIME_FIELD_DEGREE {
                return Err(refused("prime-field composition exceeds degree 64"));
            }
        }
    }
    if let PrimeFieldArithmeticOperation::Power(exponent) = operation {
        let degree = left.coefficients().len().saturating_sub(1) as u64;
        if degree
            .checked_mul(exponent)
            .is_none_or(|value| value > MAX_PRIME_FIELD_DEGREE as u64)
        {
            return Err(refused("prime-field power exceeds degree 64"));
        }
    }
    Ok(())
}

fn generated_sum(
    left: &[u64],
    right: &[u64],
    subtract: bool,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let mut result = zeros(left.len().max(right.len()), work)?;
    for (index, value) in result.iter_mut().enumerate() {
        work.tick(1)?;
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        *value = if subtract {
            (a + prime - b) % prime
        } else {
            (a + b) % prime
        };
    }
    trim(&mut result);
    Ok(result)
}

fn scale(
    values: &[u64],
    scalar: u64,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let mut result = copy(values, work)?;
    for value in &mut result {
        work.tick(1)?;
        *value = *value * scalar % prime;
    }
    trim(&mut result);
    Ok(result)
}

fn generated_monic(
    values: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let Some(&leading) = values.last() else {
        work.tick(1)?;
        return Ok(Vec::new());
    };
    let inverse = scalar_power(leading, prime - 2, prime, work)?;
    scale(values, inverse, prime, work)
}

fn generated_gcd(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldGcdWitness, PolyError> {
    let (mut a, mut b) = (copy(left, work)?, copy(right, work)?);
    let (mut s, mut next_s) = (copy(&[1], work)?, Vec::new());
    let (mut t, mut next_t) = (Vec::new(), copy(&[1], work)?);
    while !b.is_empty() {
        let (quotient, remainder) = divide(&a, &b, prime, work)?;
        let product_s = multiply(&quotient, &next_s, prime, work)?;
        let product_t = multiply(&quotient, &next_t, prime, work)?;
        let remainder_s = generated_sum(&s, &product_s, true, prime, work)?;
        let remainder_t = generated_sum(&t, &product_t, true, prime, work)?;
        (a, b) = (b, remainder);
        (s, next_s) = (next_s, remainder_s);
        (t, next_t) = (next_t, remainder_t);
    }
    let Some(&leading) = a.last() else {
        return Ok(PrimeFieldGcdWitness {
            gcd: Vec::new(),
            left_quotient: Vec::new(),
            right_quotient: Vec::new(),
            bezout_left: Vec::new(),
            bezout_right: Vec::new(),
        });
    };
    let inverse = scalar_power(leading, prime - 2, prime, work)?;
    let gcd = scale(&a, inverse, prime, work)?;
    let bezout_left = scale(&s, inverse, prime, work)?;
    let bezout_right = scale(&t, inverse, prime, work)?;
    let (left_quotient, left_remainder) = divide(left, &gcd, prime, work)?;
    let (right_quotient, right_remainder) = divide(right, &gcd, prime, work)?;
    if !left_remainder.is_empty() || !right_remainder.is_empty() {
        return Err(refused(
            "generated prime-field gcd did not divide its sources",
        ));
    }
    Ok(PrimeFieldGcdWitness {
        gcd,
        left_quotient,
        right_quotient,
        bezout_left,
        bezout_right,
    })
}

fn generate(
    operation: PrimeFieldArithmeticOperation,
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldArithmeticCertificate, PolyError> {
    use PrimeFieldArithmeticOperation as Op;
    work.allocate(
        std::mem::size_of::<PrimeFieldArithmeticCertificate>()
            + 3 * std::mem::size_of::<Coefficients>(),
    )?;
    let mut certificate = PrimeFieldArithmeticCertificate {
        results: Vec::new(),
        gcd: None,
        inverse_quotient: None,
    };
    if operation.requires_gcd() {
        let witness = generated_gcd(left, right, prime, work)?;
        certificate.results = match operation {
            Op::Gcd => vec![copy(&witness.gcd, work)?],
            Op::ExtendedGcd => vec![
                copy(&witness.bezout_left, work)?,
                copy(&witness.bezout_right, work)?,
                copy(&witness.gcd, work)?,
            ],
            Op::Lcm if left.is_empty() || right.is_empty() => vec![Vec::new()],
            Op::Lcm => {
                let product = multiply(left, right, prime, work)?;
                let (quotient, remainder) = divide(&product, &witness.gcd, prime, work)?;
                if !remainder.is_empty() {
                    return Err(refused("generated prime-field lcm was not exact"));
                }
                vec![generated_monic(&quotient, prime, work)?]
            }
            Op::Invert if witness.gcd == [1] => {
                let inverse = divide(&witness.bezout_left, right, prime, work)?.1;
                let product = multiply(left, &inverse, prime, work)?;
                let numerator = generated_sum(&product, &[1], true, prime, work)?;
                let (quotient, remainder) = divide(&numerator, right, prime, work)?;
                if !remainder.is_empty() {
                    return Err(refused("generated prime-field inverse identity failed"));
                }
                certificate.inverse_quotient = Some(quotient);
                vec![inverse]
            }
            Op::Invert => Vec::new(),
            _ => unreachable!(),
        };
        certificate.gcd = Some(witness);
        return Ok(certificate);
    }
    certificate.results = match operation {
        Op::Normalize => vec![copy(left, work)?],
        Op::Negate => vec![scale(left, prime - 1, prime, work)?],
        Op::Add => vec![generated_sum(left, right, false, prime, work)?],
        Op::Subtract => vec![generated_sum(left, right, true, prime, work)?],
        Op::Multiply => vec![multiply(left, right, prime, work)?],
        Op::Divide => {
            let (quotient, remainder) = divide(left, right, prime, work)?;
            vec![quotient, remainder]
        }
        Op::Monic => vec![generated_monic(left, prime, work)?],
        Op::Derivative(order) => {
            let mut result = copy(left, work)?;
            for _ in 0..order.min(left.len() as u32) {
                result = derivative(&result, prime, work)?;
                if result.is_empty() {
                    break;
                }
            }
            vec![result]
        }
        Op::Power(mut exponent) => {
            let mut result = copy(&[1], work)?;
            let mut base = copy(left, work)?;
            while exponent != 0 {
                work.tick(1)?;
                if exponent & 1 != 0 {
                    result = multiply(&result, &base, prime, work)?;
                }
                exponent >>= 1;
                if exponent != 0 {
                    base = multiply(&base, &base, prime, work)?;
                }
            }
            vec![result]
        }
        Op::Compose => {
            let mut result = Vec::new();
            for &coefficient in left.iter().rev() {
                result = multiply(&result, right, prime, work)?;
                result = generated_sum(&result, &[coefficient], false, prime, work)?;
            }
            vec![result]
        }
        _ => unreachable!(),
    };
    Ok(certificate)
}

// The verifier uses signed sums and whole integer convolution columns; it
// never calls the generator's modular add, multiply, divide or Euclidean loop.
fn reference_sum(
    left: &[u64],
    right: &[u64],
    subtract: bool,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let mut result = zeros(left.len().max(right.len()), work)?;
    for (index, value) in result.iter_mut().enumerate() {
        work.tick(1)?;
        let a = i128::from(left.get(index).copied().unwrap_or(0));
        let b = i128::from(right.get(index).copied().unwrap_or(0));
        *value = (if subtract { a - b } else { a + b }).rem_euclid(i128::from(prime)) as u64;
    }
    while result.last() == Some(&0) {
        result.pop();
    }
    Ok(result)
}

fn reference_product(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    if left.is_empty() || right.is_empty() {
        work.tick(1)?;
        return Ok(Vec::new());
    }
    let length = left.len() + right.len() - 1;
    if length > 2 * MAX_PRIME_FIELD_DEGREE + 1 {
        return Err(refused("prime-field reference product exceeds degree 128"));
    }
    let mut result = zeros(length, work)?;
    for (power, value) in result.iter_mut().enumerate() {
        let mut sum = 0u128;
        work.tick(left.len())?;
        for (index, &coefficient) in left.iter().enumerate() {
            if let Some(other) = power
                .checked_sub(index)
                .filter(|&other| other < right.len())
            {
                sum += u128::from(coefficient) * u128::from(right[other]);
            }
        }
        *value = (sum % u128::from(prime)) as u64;
    }
    while result.last() == Some(&0) {
        result.pop();
    }
    Ok(result)
}

fn reference_power(
    source: &[u64],
    exponent: u64,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    if exponent == 0 {
        return copy(&[1], work);
    }
    let Some(&constant) = source.first() else {
        work.tick(1)?;
        return Ok(Vec::new());
    };
    if source.len() == 1 {
        // The input constructor proved that p is prime. Fermat bounds this
        // independent repeated-scalar reference even for a full u64 exponent.
        let mut value = 1u64;
        for _ in 0..exponent % (prime - 1) {
            work.tick(1)?;
            value = (u128::from(value) * u128::from(constant) % u128::from(prime)) as u64;
        }
        return copy(&[value], work);
    }
    let mut value = copy(&[1], work)?;
    for _ in 0..exponent {
        value = reference_product(&value, source, prime, work)?;
    }
    Ok(value)
}

fn reference_derivative(
    source: &[u64],
    order: u32,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let order = order as usize;
    if order >= source.len() {
        work.tick(1)?;
        return Ok(Vec::new());
    }
    let mut result = zeros(source.len() - order, work)?;
    for (power, &coefficient) in source.iter().enumerate().skip(order) {
        let mut value = u128::from(coefficient);
        work.tick(order + 1)?;
        for shift in 0..order {
            value = value * (power - shift) as u128 % u128::from(prime);
        }
        result[power - order] = value as u64;
    }
    while result.last() == Some(&0) {
        result.pop();
    }
    Ok(result)
}

fn reference_compose(
    left: &[u64],
    right: &[u64],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Coefficients, PolyError> {
    let mut power = copy(&[1], work)?;
    let mut result = Vec::new();
    for (index, &coefficient) in left.iter().enumerate() {
        let term = reference_product(&[coefficient], &power, prime, work)?;
        result = reference_sum(&result, &term, false, prime, work)?;
        if index + 1 < left.len() {
            power = reference_product(&power, right, prime, work)?;
        }
    }
    Ok(result)
}

fn equal(
    actual: &[u64],
    expected: &[u64],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(actual.len().max(expected.len()))?;
    if actual != expected {
        return Err(refused(
            "prime-field arithmetic coefficient identity failed",
        ));
    }
    Ok(())
}

fn preflight(
    certificate: &PrimeFieldArithmeticCertificate,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if certificate.results.len() > 3 {
        return Err(refused("prime-field arithmetic has too many results"));
    }
    for values in &certificate.results {
        canonical(values, prime, work)?;
    }
    if let Some(witness) = &certificate.gcd {
        for values in [
            &witness.gcd,
            &witness.left_quotient,
            &witness.right_quotient,
            &witness.bezout_left,
            &witness.bezout_right,
        ] {
            canonical(values, prime, work)?;
        }
    }
    if let Some(quotient) = &certificate.inverse_quotient {
        canonical(quotient, prime, work)?;
    }
    Ok(())
}

fn verify_gcd(
    left: &[u64],
    right: &[u64],
    witness: &PrimeFieldGcdWitness,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if !witness.gcd.is_empty() && witness.gcd.last() != Some(&1) {
        return Err(refused("prime-field gcd must be monic"));
    }
    let left_product = reference_product(&witness.gcd, &witness.left_quotient, prime, work)?;
    let right_product = reference_product(&witness.gcd, &witness.right_quotient, prime, work)?;
    equal(left, &left_product, work)?;
    equal(right, &right_product, work)?;
    let s_left = reference_product(&witness.bezout_left, left, prime, work)?;
    let t_right = reference_product(&witness.bezout_right, right, prime, work)?;
    let sum = reference_sum(&s_left, &t_right, false, prime, work)?;
    equal(&sum, &witness.gcd, work)
}

fn verify_outputs(
    operation: PrimeFieldArithmeticOperation,
    left: &[u64],
    right: &[u64],
    prime: u64,
    certificate: &PrimeFieldArithmeticCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    use PrimeFieldArithmeticOperation as Op;
    if operation.requires_gcd() != certificate.gcd.is_some()
        || (operation != Op::Invert && certificate.inverse_quotient.is_some())
    {
        return Err(refused(
            "prime-field arithmetic has missing or irrelevant witnesses",
        ));
    }
    let outputs = &certificate.results;
    let expected_count = match operation {
        Op::Divide => 2,
        Op::ExtendedGcd => 3,
        Op::Invert => outputs.len(),
        _ => 1,
    };
    if outputs.len() != expected_count || (operation == Op::Invert && outputs.len() > 1) {
        return Err(refused("prime-field arithmetic result count changed"));
    }
    if let Some(witness) = &certificate.gcd {
        verify_gcd(left, right, witness, prime, work)?;
        return match operation {
            Op::Gcd => equal(&outputs[0], &witness.gcd, work),
            Op::ExtendedGcd => {
                equal(&outputs[0], &witness.bezout_left, work)?;
                equal(&outputs[1], &witness.bezout_right, work)?;
                equal(&outputs[2], &witness.gcd, work)
            }
            Op::Lcm if left.is_empty() || right.is_empty() => equal(&outputs[0], &[], work),
            Op::Lcm => {
                if outputs[0].last() != Some(&1) {
                    return Err(refused("prime-field lcm must be monic"));
                }
                let product = reference_product(left, right, prime, work)?;
                let common = reference_product(&witness.gcd, &outputs[0], prime, work)?;
                let unit = left[left.len() - 1] * right[right.len() - 1] % prime;
                let reconstructed = reference_product(&[unit], &common, prime, work)?;
                equal(&product, &reconstructed, work)
            }
            Op::Invert if witness.gcd == [1] => {
                if outputs.len() != 1 || (!outputs[0].is_empty() && outputs[0].len() >= right.len())
                {
                    return Err(refused("prime-field inverse is missing or not reduced"));
                }
                let quotient = certificate
                    .inverse_quotient
                    .as_ref()
                    .ok_or_else(|| refused("prime-field inverse quotient is missing"))?;
                let product = reference_product(left, &outputs[0], prime, work)?;
                let multiple = reference_product(quotient, right, prime, work)?;
                let reconstructed = reference_sum(&[1], &multiple, false, prime, work)?;
                equal(&product, &reconstructed, work)
            }
            Op::Invert => {
                if !outputs.is_empty() || certificate.inverse_quotient.is_some() {
                    return Err(refused("a prime-field nonunit cannot publish an inverse"));
                }
                work.tick(1)
            }
            _ => unreachable!(),
        };
    }
    let output = &outputs[0];
    match operation {
        Op::Normalize => equal(output, left, work),
        Op::Negate => {
            let sum = reference_sum(output, left, false, prime, work)?;
            equal(&sum, &[], work)
        }
        Op::Add | Op::Subtract => {
            let expected = reference_sum(left, right, operation == Op::Subtract, prime, work)?;
            equal(output, &expected, work)
        }
        Op::Multiply => {
            let expected = reference_product(left, right, prime, work)?;
            equal(output, &expected, work)
        }
        Op::Divide => {
            if !outputs[1].is_empty() && outputs[1].len() >= right.len() {
                return Err(refused("prime-field division remainder is not reduced"));
            }
            let product = reference_product(output, right, prime, work)?;
            let reconstructed = reference_sum(&product, &outputs[1], false, prime, work)?;
            equal(&reconstructed, left, work)
        }
        Op::Monic => {
            if let Some(&leading) = left.last() {
                if output.last() != Some(&1) {
                    return Err(refused("prime-field monic result is not monic"));
                }
                let reconstructed = reference_product(output, &[leading], prime, work)?;
                equal(&reconstructed, left, work)
            } else {
                equal(output, &[], work)
            }
        }
        Op::Derivative(order) => {
            let expected = reference_derivative(left, order, prime, work)?;
            equal(output, &expected, work)
        }
        Op::Power(exponent) => {
            let expected = reference_power(left, exponent, prime, work)?;
            equal(output, &expected, work)
        }
        Op::Compose => {
            let expected = reference_compose(left, right, prime, work)?;
            equal(output, &expected, work)
        }
        _ => unreachable!(),
    }
}

fn copy_source(
    source: &PrimeFieldPolynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldPolynomial, PolyError> {
    work.tick(source.coefficients().len() + source.variable().name.len() + 1)?;
    work.allocate(
        std::mem::size_of::<PrimeFieldPolynomial>()
            + source.variable().name.len()
            + std::mem::size_of_val(source.coefficients()),
    )?;
    Ok(source.clone())
}

fn copy_certificate(
    certificate: &PrimeFieldArithmeticCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldArithmeticCertificate, PolyError> {
    work.allocate(
        std::mem::size_of::<PrimeFieldArithmeticCertificate>()
            + certificate.results.len() * std::mem::size_of::<Coefficients>(),
    )?;
    let results = certificate
        .results
        .iter()
        .map(|values| copy(values, work))
        .collect::<Result<_, _>>()?;
    let gcd = certificate
        .gcd
        .as_ref()
        .map(|witness| {
            Ok::<_, PolyError>(PrimeFieldGcdWitness {
                gcd: copy(&witness.gcd, work)?,
                left_quotient: copy(&witness.left_quotient, work)?,
                right_quotient: copy(&witness.right_quotient, work)?,
                bezout_left: copy(&witness.bezout_left, work)?,
                bezout_right: copy(&witness.bezout_right, work)?,
            })
        })
        .transpose()?;
    let inverse_quotient = certificate
        .inverse_quotient
        .as_ref()
        .map(|values| copy(values, work))
        .transpose()?;
    Ok(PrimeFieldArithmeticCertificate {
        results,
        gcd,
        inverse_quotient,
    })
}

/// Independently accept exactly the requested operation over its typed field.
pub fn verify_prime_field_arithmetic(
    operation: PrimeFieldArithmeticOperation,
    left: &PrimeFieldPolynomial,
    right: Option<&PrimeFieldPolynomial>,
    certificate: &PrimeFieldArithmeticCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldArithmetic, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(operation, left, right, &mut work)?;
    preflight(certificate, left.prime(), &mut work)?;
    verify_outputs(
        operation,
        left.coefficients(),
        right.map_or(&[], PrimeFieldPolynomial::coefficients),
        left.prime(),
        certificate,
        &mut work,
    )?;
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldArithmetic>())?;
    let result = VerifiedPrimeFieldArithmetic {
        operation,
        left: copy_source(left, &mut work)?,
        right: right
            .map(|source| copy_source(source, &mut work))
            .transpose()?,
        certificate: copy_certificate(certificate, &mut work)?,
    };
    work.tick(1)?;
    Ok(result)
}

/// Generate and independently verify an operation, with cumulative separate
/// budgets from operand admission to final accepted publication. None of the
/// arithmetic operations silently changes the characteristic or variable.
pub fn checked_prime_field_arithmetic(
    operation: PrimeFieldArithmeticOperation,
    left: &PrimeFieldPolynomial,
    right: Option<&PrimeFieldPolynomial>,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldArithmetic, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(operation, left, right, &mut generation)?;
    let certificate = generate(
        operation,
        left.coefficients(),
        right.map_or(&[], PrimeFieldPolynomial::coefficients),
        left.prime(),
        &mut generation,
    )?;
    let accepted =
        verify_prime_field_arithmetic(operation, left, right, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
