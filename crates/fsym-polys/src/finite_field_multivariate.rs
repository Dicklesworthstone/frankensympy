//! Bounded sparse arithmetic in F_p[x_1, ..., x_n].
//!
//! Sources retain typed, ordered generators and a validated prime field.
//! Generation uses sparse modular updates; verification assembles independent
//! signed integer coefficient columns and reduces only the completed identity.
//! Division proves reconstruction and a fully reduced remainder. It does not
//! claim uniqueness of quotients for an arbitrary ordered list of divisors.
//! Every whole phase has cumulative 20M-step / 128MiB logical allocation caps.
//! These immutable in-process receipts are not portable proof-kernel objects.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field::scalar_power;
use crate::{PolyError, PrimeFieldPolynomial, TermOrder};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::Symbol;
use std::cmp::Ordering;
use std::collections::BTreeMap;

pub type PrimeFieldMultivariateTerms = BTreeMap<Vec<u32>, u64>;
type Terms = PrimeFieldMultivariateTerms;
pub(super) const MAX_VARIABLES: usize = 8;
pub(super) const MAX_DEGREE: u64 = 64;
pub(super) const MAX_TERMS: usize = 4096;
pub(super) const MAX_BASIS: usize = 32;
const MAX_COLUMNS: usize = 16384;

/// Canonical, immutable sparse polynomial in an ordered, typed prime-field ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldMultivariatePolynomial {
    generators: Vec<Symbol>,
    prime: u64,
    terms: Terms,
}

impl PrimeFieldMultivariatePolynomial {
    pub fn new(generators: &[Symbol], prime: u64, terms: &Terms) -> Result<Self, PolyError> {
        Self::metered_new(generators, prime, terms, &mut Unbounded)
    }

    pub fn metered_new(
        generators: &[Symbol],
        prime: u64,
        terms: &Terms,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, PolyError> {
        let mut work = PolynomialWork::new(meter);
        admit_ring(generators, prime, &mut work)?;
        // Even zero coefficients must have valid original monomials.
        admit_terms(terms, generators.len(), prime, MAX_DEGREE, true, &mut work)?;
        let mut canonical = copy_terms(terms, &mut work)?;
        canonical.retain(|_, value| *value != 0);
        let result = Self {
            generators: copy_generators(generators, &mut work)?,
            prime,
            terms: canonical,
        };
        work.tick(1)?;
        Ok(result)
    }

    pub fn generators(&self) -> &[Symbol] { &self.generators }
    pub fn prime(&self) -> u64 { self.prime }
    pub fn terms(&self) -> &Terms { &self.terms }
    pub fn is_zero(&self) -> bool { self.terms.is_empty() }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimeFieldMultivariateOperation {
    Normalize,
    Negate,
    Add,
    Subtract,
    Multiply,
    Monic(TermOrder),
    Power(u64),
    Derivative(Vec<u32>),
}

/// Untrusted candidate. Only a successful independent verifier creates a receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldMultivariateArithmeticCertificate {
    pub result: Terms,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPrimeFieldMultivariateArithmetic {
    operation: PrimeFieldMultivariateOperation,
    left: PrimeFieldMultivariatePolynomial,
    right: Option<PrimeFieldMultivariatePolynomial>,
    result: PrimeFieldMultivariatePolynomial,
}
impl VerifiedPrimeFieldMultivariateArithmetic {
    pub fn operation(&self) -> &PrimeFieldMultivariateOperation { &self.operation }
    pub fn left(&self) -> &PrimeFieldMultivariatePolynomial { &self.left }
    pub fn right(&self) -> Option<&PrimeFieldMultivariatePolynomial> { self.right.as_ref() }
    pub fn result(&self) -> &PrimeFieldMultivariatePolynomial { &self.result }
    pub fn into_result(self) -> PrimeFieldMultivariatePolynomial { self.result }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldMultivariateDivisionCertificate {
    pub quotients: Vec<Terms>,
    pub remainder: Terms,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPrimeFieldMultivariateDivision {
    dividend: PrimeFieldMultivariatePolynomial,
    divisors: Vec<PrimeFieldMultivariatePolynomial>,
    order: TermOrder,
    quotients: Vec<PrimeFieldMultivariatePolynomial>,
    remainder: PrimeFieldMultivariatePolynomial,
}
impl VerifiedPrimeFieldMultivariateDivision {
    pub fn dividend(&self) -> &PrimeFieldMultivariatePolynomial { &self.dividend }
    pub fn divisors(&self) -> &[PrimeFieldMultivariatePolynomial] { &self.divisors }
    pub fn order(&self) -> TermOrder { self.order }
    pub fn quotients(&self) -> &[PrimeFieldMultivariatePolynomial] { &self.quotients }
    pub fn remainder(&self) -> &PrimeFieldMultivariatePolynomial { &self.remainder }
    pub fn into_parts(self) -> (Vec<PrimeFieldMultivariatePolynomial>, PrimeFieldMultivariatePolynomial) {
        (self.quotients, self.remainder)
    }
}

pub(super) fn admit_ring(
    generators: &[Symbol], prime: u64, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if generators.is_empty() || generators.len() > MAX_VARIABLES {
        return Err(refused("prime-field polynomial requires 1..=8 generators"));
    }
    let bytes = generators.iter().try_fold(0usize, |total, generator| {
        total.checked_add(generator.name.len()).ok_or_else(|| refused("generator name bound"))
    })?;
    if bytes > 65536 { return Err(refused("generator name bound")); }
    work.tick((bytes + 1) * generators.len())?;
    for (index, generator) in generators.iter().enumerate() {
        if generators[..index].contains(generator) {
            return Err(refused("duplicate typed generator"));
        }
    }
    PrimeFieldPolynomial::metered_new(&generators[0], prime, &[], work)?;
    Ok(())
}

pub(super) fn admit_terms(
    terms: &Terms, width: usize, prime: u64, degree: u64, zeros: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if terms.len() > MAX_TERMS { return Err(refused("sparse term bound")); }
    for (monomial, coefficient) in terms {
        work.tick(width + 1)?;
        if monomial.len() != width
            || monomial.iter().map(|value| u64::from(*value)).sum::<u64>() > degree
            || *coefficient >= prime || (!zeros && *coefficient == 0)
        {
            return Err(refused("noncanonical or oversized prime-field term"));
        }
    }
    Ok(())
}

pub(super) fn same_ring(
    polynomial: &PrimeFieldMultivariatePolynomial, generators: &[Symbol], prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(generators.iter().map(|g| g.name.len() + 1).sum())?;
    if polynomial.generators() != generators || polynomial.prime() != prime {
        return Err(refused("incompatible typed prime-field polynomial rings"));
    }
    admit_terms(polynomial.terms(), generators.len(), prime, MAX_DEGREE, false, work)
}

pub(super) fn empty(work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Terms, PolyError> {
    work.allocate(std::mem::size_of::<Terms>())?;
    Ok(Terms::new())
}
pub(super) fn monomial_copy(
    monomial: &[u32], work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u32>, PolyError> {
    work.tick(monomial.len())?;
    work.allocate(std::mem::size_of_val(monomial))?;
    Ok(monomial.to_vec())
}
pub(super) fn copy_terms(
    terms: &Terms, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (monomial, coefficient) in terms {
        work.allocate(128)?;
        result.insert(monomial_copy(monomial, work)?, *coefficient);
    }
    Ok(result)
}
pub(super) fn copy_generators(
    generators: &[Symbol], work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Symbol>, PolyError> {
    let bytes = generators.iter().map(|g| g.name.len() + std::mem::size_of::<Symbol>()).sum();
    work.tick(bytes)?;
    work.allocate(bytes)?;
    Ok(generators.to_vec())
}
pub(super) fn copied_polynomial(
    generators: &[Symbol], prime: u64, terms: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldMultivariatePolynomial, PolyError> {
    work.allocate(std::mem::size_of::<PrimeFieldMultivariatePolynomial>())?;
    Ok(PrimeFieldMultivariatePolynomial {
        generators: copy_generators(generators, work)?,
        prime,
        terms: copy_terms(terms, work)?,
    })
}
pub(super) fn constant(
    width: usize, value: u64, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    if value != 0 {
        work.allocate(128 + width * 4)?;
        result.insert(vec![0; width], value);
    }
    Ok(result)
}
pub(super) fn accumulate(
    output: &mut Terms, monomial: Vec<u32>, coefficient: u64, prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(monomial.len() + 1)?;
    if monomial.iter().map(|v| u64::from(*v)).sum::<u64>() > 2 * MAX_DEGREE {
        return Err(refused("intermediate degree bound"));
    }
    if coefficient == 0 { return Ok(()); }
    let value = (output.get(&monomial).copied().unwrap_or(0) + coefficient) % prime;
    if value == 0 {
        output.remove(&monomial);
    } else {
        if !output.contains_key(&monomial) {
            if output.len() == MAX_TERMS { return Err(refused("intermediate sparse term bound")); }
            work.allocate(128)?;
        }
        output.insert(monomial, value);
    }
    Ok(())
}
pub(super) fn sum(
    left: &Terms, right: &Terms, subtract: bool, prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = copy_terms(left, work)?;
    for (monomial, coefficient) in right {
        let value = if subtract { (prime - coefficient) % prime } else { *coefficient };
        accumulate(&mut result, monomial_copy(monomial, work)?, value, prime, work)?;
    }
    Ok(result)
}
pub(super) fn monomial_sum(
    left: &[u32], right: &[u32], work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u32>, PolyError> {
    if left.len() != right.len() { return Err(refused("monomial width mismatch")); }
    work.tick(left.len())?;
    work.allocate(std::mem::size_of_val(left))?;
    let result: Vec<u32> = left.iter().zip(right).map(|(a, b)| a + b).collect();
    if result.iter().map(|v| u64::from(*v)).sum::<u64>() > 2 * MAX_DEGREE {
        return Err(refused("intermediate degree bound"));
    }
    Ok(result)
}
pub(super) fn product(
    left: &Terms, right: &Terms, prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (a, ca) in left {
        for (b, cb) in right {
            let monomial = monomial_sum(a, b, work)?;
            accumulate(&mut result, monomial, ca * cb % prime, prime, work)?;
        }
    }
    Ok(result)
}
pub(super) fn scaled(
    source: &Terms, scalar: u64, prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut result = empty(work)?;
    for (monomial, coefficient) in source {
        accumulate(&mut result, monomial_copy(monomial, work)?, coefficient * scalar % prime, prime, work)?;
    }
    Ok(result)
}
pub(super) fn leading<'a>(
    terms: &'a Terms, order: TermOrder, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<(&'a Vec<u32>, u64)>, PolyError> {
    work.tick(terms.len() * (terms.keys().next().map_or(0, Vec::len) + 1))?;
    Ok(terms.iter().max_by(|(a, _), (b, _)| order.compare_monomials(a, b)).map(|(m, c)| (m, *c)))
}
pub(super) fn divides(left: &[u32], right: &[u32]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a <= b)
}
pub(super) fn monomial_difference(
    numerator: &[u32], denominator: &[u32], work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u32>, PolyError> {
    work.tick(numerator.len())?;
    if !divides(denominator, numerator) { return Err(refused("nondivisible monomial")); }
    work.allocate(std::mem::size_of_val(numerator))?;
    Ok(numerator.iter().zip(denominator).map(|(a, b)| a - b).collect())
}
pub(super) fn divide(
    source: &Terms, divisors: &[&Terms], order: TermOrder, prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<PrimeFieldMultivariateDivisionCertificate, PolyError> {
    work.allocate(divisors.len() * (std::mem::size_of::<Terms>() + 32))?;
    let mut quotients = Vec::with_capacity(divisors.len());
    let mut heads = Vec::with_capacity(divisors.len());
    for divisor in divisors {
        let (monomial, coefficient) = leading(divisor, order, work)?.ok_or(PolyError::DivisionByZero)?;
        heads.push((monomial_copy(monomial, work)?, scalar_power(coefficient, prime - 2, prime, work)?));
        quotients.push(empty(work)?);
    }
    let mut current = copy_terms(source, work)?;
    let mut remainder = empty(work)?;
    while let Some((monomial, coefficient)) = leading(&current, order, work)? {
        let previous = monomial_copy(monomial, work)?;
        let mut reduced = false;
        for (index, (head, inverse)) in heads.iter().enumerate() {
            work.tick(previous.len())?;
            if divides(head, &previous) {
                let shift = monomial_difference(&previous, head, work)?;
                let value = coefficient * inverse % prime;
                // Update only the affected sparse row. Rebuilding the complete
                // current polynomial at every reduction is quadratic in the
                // evolving support and dominates Gröbner reductions.
                for (monomial, coefficient) in divisors[index] {
                    let translated = monomial_sum(monomial, &shift, work)?;
                    let subtraction = (prime - coefficient * value % prime) % prime;
                    accumulate(&mut current, translated, subtraction, prime, work)?;
                }
                accumulate(&mut quotients[index], shift, value, prime, work)?;
                reduced = true;
                break;
            }
        }
        if !reduced {
            current.remove(&previous);
            accumulate(&mut remainder, monomial_copy(&previous, work)?, coefficient, prime, work)?;
        }
        if let Some((next, _)) = leading(&current, order, work)?
            && order.compare_monomials(next, &previous) != Ordering::Less
        {
            return Err(refused("division did not decrease its leading monomial"));
        }
    }
    Ok(PrimeFieldMultivariateDivisionCertificate { quotients, remainder })
}

// The verifier never calls sparse modular sum/product/divide or the generator's
// monomial comparator. It accumulates unreduced signed integer columns instead.
pub(super) type Columns = BTreeMap<Vec<u32>, i128>;
fn new_columns(work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Columns, PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    Ok(Columns::new())
}
fn column(
    output: &mut Columns, monomial: Vec<u32>, value: i128,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(monomial.len() + 1)?;
    if value == 0 { return Ok(()); }
    if let Some(previous) = output.get_mut(&monomial) {
        *previous = previous.checked_add(value).ok_or_else(|| refused("reference coefficient bound"))?;
    } else {
        if output.len() == MAX_COLUMNS { return Err(refused("reference column bound")); }
        work.allocate(144)?;
        output.insert(monomial, value);
    }
    Ok(())
}
pub(super) fn reference_add(
    output: &mut Columns, source: &Terms, scale: i128,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    for (monomial, coefficient) in source {
        let value = i128::from(*coefficient).checked_mul(scale).ok_or_else(|| refused("reference coefficient bound"))?;
        column(output, monomial_copy(monomial, work)?, value, work)?;
    }
    Ok(())
}
pub(super) fn reference_product(
    output: &mut Columns, left: &Terms, right: &Terms, scale: i128,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    for (a, ca) in left {
        for (b, cb) in right {
            work.tick(a.len() + 1)?;
            if a.len() != b.len() { return Err(refused("reference monomial width")); }
            work.allocate(std::mem::size_of_val(a.as_slice()))?;
            let mut monomial = Vec::with_capacity(a.len());
            let mut degree = 0u64;
            for (x, y) in a.iter().zip(b) {
                let exponent = x.checked_add(*y).ok_or_else(|| refused("reference exponent bound"))?;
                degree += u64::from(exponent);
                monomial.push(exponent);
            }
            if degree > 2 * MAX_DEGREE { return Err(refused("reference degree bound")); }
            let value = (i128::from(*ca) * i128::from(*cb)).checked_mul(scale)
                .ok_or_else(|| refused("reference coefficient bound"))?;
            column(output, monomial, value, work)?;
        }
    }
    Ok(())
}
pub(super) fn identity_zero(
    columns: &Columns, prime: u64, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    for value in columns.values() {
        work.tick(1)?;
        if value.rem_euclid(i128::from(prime)) != 0 {
            return Err(PolyError::IdentityCheckFailed("prime-field coefficient identity".into()));
        }
    }
    Ok(())
}
fn reference_value(
    left: &Terms, right: &Terms, prime: u64, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut columns = new_columns(work)?;
    reference_product(&mut columns, left, right, 1, work)?;
    let mut result = empty(work)?;
    for (monomial, value) in columns {
        work.tick(1)?;
        let value = value.rem_euclid(i128::from(prime)) as u64;
        if value != 0 {
            if result.len() == MAX_TERMS { return Err(refused("reference sparse term bound")); }
            work.allocate(128)?;
            result.insert(monomial, value);
        }
    }
    Ok(result)
}
pub(super) fn reference_order(left: &[u32], right: &[u32], order: TermOrder) -> Ordering {
    if order != TermOrder::Lex {
        let a: u64 = left.iter().map(|x| u64::from(*x)).sum();
        let b: u64 = right.iter().map(|x| u64::from(*x)).sum();
        if a != b { return a.cmp(&b); }
        if order == TermOrder::DegRevLex {
            for (a, b) in left.iter().rev().zip(right.iter().rev()) {
                if a != b { return b.cmp(a); }
            }
            return Ordering::Equal;
        }
    }
    for (a, b) in left.iter().zip(right) {
        if a != b { return a.cmp(b); }
    }
    left.len().cmp(&right.len())
}
pub(super) fn reference_head<'a>(
    terms: &'a Terms, order: TermOrder, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Option<(&'a Vec<u32>, u64)>, PolyError> {
    let mut result: Option<(&Vec<u32>, u64)> = None;
    for (monomial, coefficient) in terms {
        work.tick(monomial.len() + 1)?;
        if result.is_none_or(|(head, _)| reference_order(monomial, head, order) == Ordering::Greater) {
            result = Some((monomial, *coefficient));
        }
    }
    Ok(result)
}
pub(super) fn verify_division_identity(
    source: &Terms, divisors: &[&Terms], order: TermOrder, prime: u64,
    certificate: &PrimeFieldMultivariateDivisionCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if certificate.quotients.len() != divisors.len() { return Err(refused("division quotient width")); }
    let mut columns = new_columns(work)?;
    reference_add(&mut columns, source, 1, work)?;
    reference_add(&mut columns, &certificate.remainder, -1, work)?;
    for (divisor, quotient) in divisors.iter().zip(&certificate.quotients) {
        let (head, _) = reference_head(divisor, order, work)?.ok_or(PolyError::DivisionByZero)?;
        for monomial in certificate.remainder.keys() {
            work.tick(monomial.len())?;
            if head.iter().zip(monomial).all(|(a, b)| a <= b) {
                return Err(PolyError::IdentityCheckFailed("division remainder is reducible".into()));
            }
        }
        reference_product(&mut columns, quotient, divisor, -1, work)?;
    }
    identity_zero(&columns, prime, work)
}

fn arithmetic_admission(
    operation: &PrimeFieldMultivariateOperation, left: &PrimeFieldMultivariatePolynomial,
    right: Option<&PrimeFieldMultivariatePolynomial>, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    same_ring(left, left.generators(), left.prime(), work)?;
    if let Some(right) = right { same_ring(right, left.generators(), left.prime(), work)?; }
    let binary = matches!(operation, PrimeFieldMultivariateOperation::Add
        | PrimeFieldMultivariateOperation::Subtract | PrimeFieldMultivariateOperation::Multiply);
    if binary != right.is_some() { return Err(refused("arithmetic operand count")); }
    if let PrimeFieldMultivariateOperation::Derivative(orders) = operation {
        if orders.len() != left.generators().len() { return Err(refused("derivative order vector width")); }
        work.tick(orders.len())?;
    }
    if let PrimeFieldMultivariateOperation::Power(exponent) = operation {
        let degree = left.terms().keys().map(|m| m.iter().map(|v| u64::from(*v)).sum::<u64>()).max().unwrap_or(0);
        work.tick(left.terms().len() * (left.generators().len() + 1))?;
        if degree.checked_mul(*exponent).is_none_or(|value| value > MAX_DEGREE) {
            return Err(refused("power degree bound"));
        }
    }
    Ok(())
}
fn generated_arithmetic(
    operation: &PrimeFieldMultivariateOperation, left: &PrimeFieldMultivariatePolynomial,
    right: Option<&PrimeFieldMultivariatePolynomial>, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    use PrimeFieldMultivariateOperation as Op;
    let p = left.prime();
    let source = left.terms();
    match operation {
        Op::Normalize => copy_terms(source, work),
        Op::Negate => scaled(source, p - 1, p, work),
        Op::Add | Op::Subtract | Op::Multiply => {
            let right = right.ok_or_else(|| refused("missing right operand"))?.terms();
            match operation {
                Op::Multiply => product(source, right, p, work),
                _ => sum(source, right, *operation == Op::Subtract, p, work),
            }
        }
        Op::Monic(order) => {
            let coefficient = leading(source, *order, work)?.map_or(1, |(_, c)| c);
            let inverse = scalar_power(coefficient, p - 2, p, work)?;
            scaled(source, inverse, p, work)
        }
        Op::Power(exponent) => {
            let mut exponent = *exponent;
            let mut current = copy_terms(source, work)?;
            let mut result = constant(left.generators().len(), 1, work)?;
            while exponent != 0 {
                work.tick(1)?;
                if exponent & 1 != 0 { result = product(&result, &current, p, work)?; }
                exponent >>= 1;
                if exponent != 0 { current = product(&current, &current, p, work)?; }
            }
            Ok(result)
        }
        Op::Derivative(orders) => {
            let mut current = copy_terms(source, work)?;
            for (variable, order) in orders.iter().enumerate() {
                // No admitted source survives more than 64 differentiations.
                for _ in 0..(*order).min(65) {
                    if current.is_empty() { break; }
                    let mut next = empty(work)?;
                    for (monomial, coefficient) in &current {
                        work.tick(monomial.len() + 1)?;
                        if monomial[variable] == 0 { continue; }
                        let value = coefficient * u64::from(monomial[variable]) % p;
                        let mut differentiated = monomial_copy(monomial, work)?;
                        differentiated[variable] -= 1;
                        accumulate(&mut next, differentiated, value, p, work)?;
                    }
                    current = next;
                }
            }
            Ok(current)
        }
    }
}
fn reference_arithmetic(
    operation: &PrimeFieldMultivariateOperation, left: &PrimeFieldMultivariatePolynomial,
    right: Option<&PrimeFieldMultivariatePolynomial>, candidate: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    use PrimeFieldMultivariateOperation as Op;
    let p = left.prime();
    let source = left.terms();
    let mut columns = new_columns(work)?;
    reference_add(&mut columns, candidate, 1, work)?;
    match operation {
        Op::Normalize => reference_add(&mut columns, source, -1, work)?,
        Op::Negate => reference_add(&mut columns, source, 1, work)?,
        Op::Add | Op::Subtract | Op::Multiply => {
            let right = right.ok_or_else(|| refused("missing right operand"))?.terms();
            if *operation == Op::Multiply {
                reference_product(&mut columns, source, right, -1, work)?;
            } else {
                reference_add(&mut columns, source, -1, work)?;
                reference_add(&mut columns, right, if *operation == Op::Subtract { 1 } else { -1 }, work)?;
            }
        }
        Op::Monic(order) => {
            if let Some((_, coefficient)) = reference_head(source, *order, work)? {
                if reference_head(candidate, *order, work)?.is_none_or(|(_, c)| c != 1) {
                    return Err(PolyError::IdentityCheckFailed("monic leading coefficient".into()));
                }
                columns = new_columns(work)?;
                reference_add(&mut columns, candidate, i128::from(coefficient), work)?;
                reference_add(&mut columns, source, -1, work)?;
            }
        }
        Op::Power(exponent) => {
            let mut expected = constant(left.generators().len(), 1, work)?;
            let is_constant = source.keys().all(|m| m.iter().all(|e| *e == 0));
            let count = if source.is_empty() && *exponent > 0 {
                expected = empty(work)?;
                0
            } else if is_constant { exponent % (p - 1) } else { *exponent };
            for _ in 0..count { expected = reference_value(&expected, source, p, work)?; }
            reference_add(&mut columns, &expected, -1, work)?;
        }
        Op::Derivative(orders) => {
            for (monomial, coefficient) in source {
                work.tick(monomial.len() + 1)?;
                if monomial.iter().zip(orders).any(|(power, order)| power < order) { continue; }
                let mut value = i128::from(*coefficient);
                let mut shifted = monomial_copy(monomial, work)?;
                for (variable, order) in orders.iter().enumerate() {
                    for offset in 0..*order {
                        work.tick(1)?;
                        value = value * i128::from(monomial[variable] - offset) % i128::from(p);
                    }
                    shifted[variable] -= order;
                }
                column(&mut columns, shifted, -value, work)?;
            }
        }
    }
    identity_zero(&columns, p, work)
}

pub fn verify_prime_field_multivariate_arithmetic(
    operation: &PrimeFieldMultivariateOperation, left: &PrimeFieldMultivariatePolynomial,
    right: Option<&PrimeFieldMultivariatePolynomial>,
    certificate: &PrimeFieldMultivariateArithmeticCertificate, verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateArithmetic, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    arithmetic_admission(operation, left, right, &mut work)?;
    admit_terms(&certificate.result, left.generators().len(), left.prime(), MAX_DEGREE, false, &mut work)?;
    reference_arithmetic(operation, left, right, &certificate.result, &mut work)?;
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldMultivariateArithmetic>() + MAX_VARIABLES * 4)?;
    let result = VerifiedPrimeFieldMultivariateArithmetic {
        operation: operation.clone(),
        left: copied_polynomial(left.generators(), left.prime(), left.terms(), &mut work)?,
        right: right.map(|r| copied_polynomial(r.generators(), r.prime(), r.terms(), &mut work)).transpose()?,
        result: copied_polynomial(left.generators(), left.prime(), &certificate.result, &mut work)?,
    };
    work.tick(1)?;
    Ok(result)
}

pub fn checked_prime_field_multivariate_arithmetic(
    operation: &PrimeFieldMultivariateOperation, left: &PrimeFieldMultivariatePolynomial,
    right: Option<&PrimeFieldMultivariatePolynomial>, generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateArithmetic, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    arithmetic_admission(operation, left, right, &mut generation)?;
    let certificate = PrimeFieldMultivariateArithmeticCertificate {
        result: generated_arithmetic(operation, left, right, &mut generation)?,
    };
    let accepted = verify_prime_field_multivariate_arithmetic(operation, left, right, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}

fn division_admission(
    dividend: &PrimeFieldMultivariatePolynomial, divisors: &[PrimeFieldMultivariatePolynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if divisors.len() > MAX_BASIS { return Err(refused("division divisor bound")); }
    same_ring(dividend, dividend.generators(), dividend.prime(), work)?;
    for divisor in divisors {
        same_ring(divisor, dividend.generators(), dividend.prime(), work)?;
    }
    if divisors.iter().any(PrimeFieldMultivariatePolynomial::is_zero) { return Err(PolyError::DivisionByZero); }
    Ok(())
}
pub(super) fn admit_division(
    certificate: &PrimeFieldMultivariateDivisionCertificate, width: usize,
    divisor_count: usize, prime: u64, work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if certificate.quotients.len() != divisor_count { return Err(refused("division quotient width")); }
    for quotient in &certificate.quotients {
        admit_terms(quotient, width, prime, MAX_DEGREE, false, work)?;
    }
    admit_terms(&certificate.remainder, width, prime, MAX_DEGREE, false, work)
}
fn term_views<'a>(
    divisors: &'a [PrimeFieldMultivariatePolynomial], work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<&'a Terms>, PolyError> {
    work.tick(divisors.len())?;
    work.allocate(std::mem::size_of_val(divisors) + divisors.len() * std::mem::size_of::<&Terms>())?;
    Ok(divisors.iter().map(PrimeFieldMultivariatePolynomial::terms).collect())
}

pub fn verify_prime_field_multivariate_division(
    dividend: &PrimeFieldMultivariatePolynomial, divisors: &[PrimeFieldMultivariatePolynomial],
    order: TermOrder, certificate: &PrimeFieldMultivariateDivisionCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateDivision, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    division_admission(dividend, divisors, &mut work)?;
    admit_division(certificate, dividend.generators().len(), divisors.len(), dividend.prime(), &mut work)?;
    let views = term_views(divisors, &mut work)?;
    verify_division_identity(dividend.terms(), &views, order, dividend.prime(), certificate, &mut work)?;
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldMultivariateDivision>()
        + 2 * divisors.len() * std::mem::size_of::<PrimeFieldMultivariatePolynomial>())?;
    let mut accepted_divisors = Vec::with_capacity(divisors.len());
    let mut quotients = Vec::with_capacity(divisors.len());
    for divisor in divisors {
        accepted_divisors.push(copied_polynomial(divisor.generators(), divisor.prime(), divisor.terms(), &mut work)?);
    }
    for quotient in &certificate.quotients {
        quotients.push(copied_polynomial(dividend.generators(), dividend.prime(), quotient, &mut work)?);
    }
    let result = VerifiedPrimeFieldMultivariateDivision {
        dividend: copied_polynomial(dividend.generators(), dividend.prime(), dividend.terms(), &mut work)?,
        divisors: accepted_divisors,
        order,
        quotients,
        remainder: copied_polynomial(dividend.generators(), dividend.prime(), &certificate.remainder, &mut work)?,
    };
    work.tick(1)?;
    Ok(result)
}

pub fn checked_prime_field_multivariate_division(
    dividend: &PrimeFieldMultivariatePolynomial, divisors: &[PrimeFieldMultivariatePolynomial],
    order: TermOrder, generator: &mut impl BudgetMeter, verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateDivision, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    division_admission(dividend, divisors, &mut generation)?;
    let views = term_views(divisors, &mut generation)?;
    let certificate = divide(dividend.terms(), &views, order, dividend.prime(), &mut generation)?;
    let accepted = verify_prime_field_multivariate_division(dividend, divisors, order, &certificate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
