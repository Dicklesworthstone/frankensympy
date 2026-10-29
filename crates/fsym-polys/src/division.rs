//! Bounded sparse division over Q and a separate coefficient-identity checker.
//!
//! This is a deterministic synchronous lane. Generation and verification take
//! separate meters. Each phase has additional limits of eight variables, 64
//! divisors, total degree 256, 4096 terms per polynomial, 4096 coefficient bits,
//! twenty million compute units and 128 MiB of cumulative logical allocation
//! charges. Unsupported or exhausted cases never publish a checked result.

use crate::{MultivariatePoly, PolyError, TermOrder};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError};
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use std::collections::BTreeMap;

pub(crate) const MAX_VARIABLES: usize = 8;
pub(crate) const MAX_BASIS: usize = 64;
pub(crate) const MAX_TERMS: usize = 4096;
pub(crate) const MAX_DEGREE: u64 = 256;
const MAX_BITS: u64 = 4096;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

pub(crate) type Terms = BTreeMap<Vec<u32>, BigRational>;

pub(crate) fn refused(message: impl std::fmt::Display) -> PolyError {
    PolyError::General(format!("bounded polynomial operation refused: {message}"))
}

pub(crate) struct PolynomialWork<'a, M> {
    meter: &'a mut M,
    compute: u64,
    memory: u64,
}

impl<'a, M: BudgetMeter> PolynomialWork<'a, M> {
    pub(crate) fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            compute: 0,
            memory: 0,
        }
    }

    pub(crate) fn tick(&mut self, units: usize) -> Result<(), PolyError> {
        self.checkpoint().map_err(refused)?;
        self.charge(Dimension::ComputeSteps, units.max(1) as u64)
            .map_err(refused)
    }

    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<(), PolyError> {
        self.checkpoint().map_err(refused)?;
        self.charge_batch(&[
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(refused)
    }

    pub(crate) fn clone_value(&mut self, value: &BigRational) -> Result<BigRational, PolyError> {
        check_coefficient(value)?;
        self.allocate(
            ((value.numer().limb_count() + value.denom().limb_count()).max(1) * 8) as usize,
        )?;
        Ok(value.clone())
    }

    pub(crate) fn add(
        &mut self,
        a: &BigRational,
        b: &BigRational,
    ) -> Result<BigRational, PolyError> {
        arithmetic_preflight(a, b)?;
        let value = a.metered_add(b, self).map_err(refused)?;
        check_coefficient(&value)?;
        Ok(value)
    }

    pub(crate) fn subtract(
        &mut self,
        a: &BigRational,
        b: &BigRational,
    ) -> Result<BigRational, PolyError> {
        arithmetic_preflight(a, b)?;
        let value = a.metered_sub(b, self).map_err(refused)?;
        check_coefficient(&value)?;
        Ok(value)
    }

    pub(crate) fn multiply(
        &mut self,
        a: &BigRational,
        b: &BigRational,
    ) -> Result<BigRational, PolyError> {
        arithmetic_preflight(a, b)?;
        let value = a.metered_mul(b, self).map_err(refused)?;
        check_coefficient(&value)?;
        Ok(value)
    }

    pub(crate) fn divide(
        &mut self,
        a: &BigRational,
        b: &BigRational,
    ) -> Result<BigRational, PolyError> {
        arithmetic_preflight(a, b)?;
        if b.is_zero() {
            return Err(PolyError::DivisionByZero);
        }
        let value = a.metered_div(b, self).map_err(refused)?;
        check_coefficient(&value)?;
        Ok(value)
    }
}

impl<M: BudgetMeter> BudgetMeter for PolynomialWork<'_, M> {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.meter.checkpoint()
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let mut compute = self.compute;
        let mut memory = self.memory;
        for &(dimension, amount) in charges {
            let (counter, cap) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_COMPUTE),
                Dimension::MemoryBytes => (&mut memory, MAX_MEMORY),
                _ => continue,
            };
            let remaining = cap.saturating_sub(*counter);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *counter += amount;
        }
        self.meter.charge_batch(charges)?;
        self.compute = compute;
        self.memory = memory;
        Ok(())
    }
}

fn check_coefficient(value: &BigRational) -> Result<(), PolyError> {
    if value.numer().bits() > MAX_BITS || value.denom().bits() > MAX_BITS {
        return Err(refused("coefficient exceeds 4096 bits"));
    }
    Ok(())
}

fn arithmetic_preflight(a: &BigRational, b: &BigRational) -> Result<(), PolyError> {
    check_coefficient(a)?;
    check_coefficient(b)?;
    // All two-operand cross products have at most 8192 bits; a sum can
    // require one additional bit. The result must return to the 4096-bit
    // envelope, and the primitive meter owns all intermediate buffers.
    Ok(())
}

pub(crate) fn validate_poly<M: BudgetMeter>(
    poly: &MultivariatePoly,
    work: &mut PolynomialWork<'_, M>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if poly.generators.len() > MAX_VARIABLES || poly.terms.len() > MAX_TERMS {
        return Err(refused(
            "variable or term count exceeds the bounded envelope",
        ));
    }
    poly.validate_shape()?;
    work.tick(poly.generators.len())?;
    for (exponents, coefficient) in &poly.terms {
        work.tick(exponents.len() + 1)?;
        check_monomial(exponents)?;
        check_coefficient(coefficient)?;
    }
    Ok(())
}

pub(crate) fn check_monomial(exponents: &[u32]) -> Result<(), PolyError> {
    if exponents.iter().map(|&e| u64::from(e)).sum::<u64>() > MAX_DEGREE {
        return Err(refused("total monomial degree exceeds 256"));
    }
    Ok(())
}

pub(crate) fn divides(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a <= b)
}

pub(crate) fn same_ring(left: &[Symbol], right: &[Symbol]) -> Result<(), PolyError> {
    if left != right {
        return Err(PolyError::IncompatibleGenerators(
            format!("{left:?}"),
            format!("{right:?}"),
        ));
    }
    Ok(())
}

pub(crate) fn ring_storage(generators: &[Symbol]) -> usize {
    generators
        .iter()
        .map(|g| g.name.len() + std::mem::size_of::<Symbol>())
        .sum()
}

pub(crate) fn charge_copy<M: BudgetMeter>(
    poly: &MultivariatePoly,
    work: &mut PolynomialWork<'_, M>,
) -> Result<(), PolyError> {
    validate_poly(poly, work)?;
    work.allocate(ring_storage(&poly.generators))?;
    for (monomial, value) in &poly.terms {
        let coefficient =
            ((value.numer().limb_count() + value.denom().limb_count()).max(1) * 8) as usize;
        work.allocate(256 + monomial.len() * 4 + coefficient)?;
    }
    Ok(())
}

pub(crate) fn copy_poly<M: BudgetMeter>(
    poly: &MultivariatePoly,
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    charge_copy(poly, work)?;
    Ok(poly.clone())
}

pub(crate) fn accumulate<M: BudgetMeter>(
    terms: &mut Terms,
    monomial: Vec<u32>,
    value: BigRational,
    work: &mut PolynomialWork<'_, M>,
) -> Result<(), PolyError> {
    work.tick(monomial.len() + 1)?;
    check_monomial(&monomial)?;
    if value.is_zero() {
        return Ok(());
    }
    let value = if let Some(old) = terms.get(&monomial) {
        work.add(old, &value)?
    } else {
        value
    };
    if value.is_zero() {
        terms.remove(&monomial);
    } else {
        if !terms.contains_key(&monomial) {
            if terms.len() >= MAX_TERMS {
                return Err(refused("polynomial term growth limit exceeded"));
            }
            work.allocate(256 + monomial.len() * 4)?;
        }
        terms.insert(monomial, value);
    }
    Ok(())
}

pub(crate) fn monomial_poly<M: BudgetMeter>(
    generators: &[Symbol],
    exponents: Vec<u32>,
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    check_monomial(&exponents)?;
    work.allocate(256 + ring_storage(generators) + exponents.len() * 4)?;
    Ok(MultivariatePoly {
        generators: generators.to_vec(),
        terms: BTreeMap::from([(exponents, BigRational::one())]),
    })
}

pub(crate) fn divide_with_work<M: BudgetMeter>(
    f: &MultivariatePoly,
    divisors: &[MultivariatePoly],
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<(Vec<MultivariatePoly>, MultivariatePoly), PolyError> {
    validate_poly(f, work)?;
    if divisors.len() > MAX_BASIS {
        return Err(refused("divisor count exceeds 64"));
    }
    for divisor in divisors {
        validate_poly(divisor, work)?;
        same_ring(&f.generators, &divisor.generators)?;
    }
    let mut pending = copy_poly(f, work)?.terms;
    work.allocate(
        (divisors.len() + 1)
            * (std::mem::size_of::<MultivariatePoly>() + ring_storage(&f.generators)),
    )?;
    let mut quotients = vec![MultivariatePoly::zero(f.generators.clone()); divisors.len()];
    let mut remainder = MultivariatePoly::zero(f.generators.clone());
    work.allocate(divisors.len() * 64)?;
    let leading: Vec<_> = divisors.iter().map(|g| g.leading_term(order)).collect();
    while !pending.is_empty() {
        work.tick(pending.len() * f.generators.len().max(1))?;
        let (monomial, coefficient) = pending
            .iter()
            .max_by(|(a, _), (b, _)| order.compare_monomials(a, b))
            .expect("nonempty map");
        work.allocate(monomial.len() * 4)?;
        let monomial = monomial.clone();
        let coefficient = work.clone_value(coefficient)?;
        let mut reduced = false;
        for (index, head) in leading.iter().enumerate() {
            work.tick(f.generators.len() + 1)?;
            let Some((divisor_monomial, divisor_coefficient)) = head else {
                continue;
            };
            if !divides(divisor_monomial, &monomial) {
                continue;
            }
            work.allocate(monomial.len() * 4)?;
            let shift: Vec<u32> = monomial
                .iter()
                .zip(*divisor_monomial)
                .map(|(a, b)| a - b)
                .collect();
            let quotient = work.divide(&coefficient, divisor_coefficient)?;
            accumulate(
                &mut quotients[index].terms,
                shift.clone(),
                work.clone_value(&quotient)?,
                work,
            )?;
            for (exponents, value) in &divisors[index].terms {
                work.allocate(exponents.len() * 4)?;
                let shifted: Vec<u32> = exponents.iter().zip(&shift).map(|(a, b)| a + b).collect();
                check_monomial(&shifted)?;
                let product = work.multiply(&quotient, value)?;
                let negative = work.subtract(&BigRational::zero(), &product)?;
                accumulate(&mut pending, shifted, negative, work)?;
            }
            reduced = true;
            break;
        }
        if !reduced {
            pending.remove(&monomial);
            accumulate(&mut remainder.terms, monomial, coefficient, work)?;
        }
    }
    work.tick(1)?;
    Ok((quotients, remainder))
}

/// Generate an untrusted exact quotient/remainder candidate. Zero divisors in
/// the list are skipped with a zero quotient; an empty list returns `([], f)`.
pub fn bounded_division(
    f: &MultivariatePoly,
    divisors: &[MultivariatePoly],
    order: TermOrder,
    meter: &mut impl BudgetMeter,
) -> Result<(Vec<MultivariatePoly>, MultivariatePoly), PolyError> {
    divide_with_work(f, divisors, order, &mut PolynomialWork::new(meter))
}

/// Independently check `f = sum(q_i*g_i) + r` and that no remainder monomial
/// is divisible by any divisor's leading monomial. This reference convolution
/// never invokes polynomial division, multiplication, simplification or search.
/// For arbitrary divisors this proves division, not ideal non-membership.
pub fn verify_bounded_division(
    f: &MultivariatePoly,
    divisors: &[MultivariatePoly],
    quotients: &[MultivariatePoly],
    remainder: &MultivariatePoly,
    order: TermOrder,
    meter: &mut impl BudgetMeter,
) -> Result<(), PolyError> {
    let mut work = PolynomialWork::new(meter);
    validate_poly(f, &mut work)?;
    if divisors.len() > MAX_BASIS || divisors.len() != quotients.len() {
        return Err(refused("quotient/divisor count mismatch"));
    }
    for poly in divisors
        .iter()
        .chain(quotients)
        .chain(std::iter::once(remainder))
    {
        validate_poly(poly, &mut work)?;
        same_ring(&f.generators, &poly.generators)?;
    }
    for monomial in remainder.terms.keys() {
        for divisor in divisors {
            work.tick(divisor.terms.len().max(1) * f.generators.len().max(1))?;
            if divisor
                .leading_term(order)
                .is_some_and(|(leading, _)| divides(leading, monomial))
            {
                return Err(PolyError::IdentityCheckFailed(
                    "remainder is not completely reduced".into(),
                ));
            }
        }
    }
    let mut reconstructed = copy_poly(remainder, &mut work)?.terms;
    for (quotient, divisor) in quotients.iter().zip(divisors) {
        for (left, a) in &quotient.terms {
            for (right, b) in &divisor.terms {
                work.allocate(left.len() * 4)?;
                let exponents: Vec<u32> = left.iter().zip(right).map(|(a, b)| a + b).collect();
                check_monomial(&exponents)?;
                let coefficient = work.multiply(a, b)?;
                accumulate(&mut reconstructed, exponents, coefficient, &mut work)?;
            }
        }
    }
    if reconstructed != f.terms {
        return Err(PolyError::IdentityCheckFailed(
            "quotient/remainder coefficient identity failed".into(),
        ));
    }
    work.tick(1)
}

/// Generate and independently check a bounded division using distinct meters.
pub fn checked_polynomial_division(
    f: &MultivariatePoly,
    divisors: &[MultivariatePoly],
    order: TermOrder,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<(Vec<MultivariatePoly>, MultivariatePoly), PolyError> {
    let (quotients, remainder) = bounded_division(f, divisors, order, generator_meter)?;
    verify_bounded_division(f, divisors, &quotients, &remainder, order, verifier_meter)?;
    Ok((quotients, remainder))
}

const MAX_INPUT_DEPTH: usize = 64;
const MAX_INPUT_NODES: usize = 4096;

fn preflight_expression<M: BudgetMeter>(
    expression: &Expr,
    generators: &[Symbol],
    depth: usize,
    nodes: &mut usize,
    work: &mut PolynomialWork<'_, M>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    *nodes += 1;
    if depth > MAX_INPUT_DEPTH || *nodes > MAX_INPUT_NODES {
        return Err(refused(
            "expression depth or node count exceeds the conversion bound",
        ));
    }
    match expression {
        Expr::Integer(value) => {
            if value.bits() > MAX_BITS {
                return Err(refused("input integer exceeds 4096 bits"));
            }
        }
        Expr::Rational(value) => check_coefficient(value)?,
        Expr::Sym(symbol) => {
            if !generators.contains(symbol) {
                return Err(refused(
                    "expression contains an undeclared polynomial generator",
                ));
            }
        }
        Expr::Add(children) | Expr::Mul(children) => {
            if children.len() > MAX_INPUT_NODES {
                return Err(refused("expression arity exceeds the conversion bound"));
            }
            for child in children {
                preflight_expression(child, generators, depth + 1, nodes, work)?;
            }
        }
        Expr::Pow(base, exponent) => {
            preflight_expression(base, generators, depth + 1, nodes, work)?;
            preflight_expression(exponent, generators, depth + 1, nodes, work)?;
            if !matches!(exponent.as_ref(), Expr::Integer(value) if value.to_i64().is_some_and(|n| (-256..=256).contains(&n)))
            {
                return Err(refused(
                    "polynomial conversion requires literal integer powers between -256 and 256",
                ));
            }
        }
        _ => {
            return Err(PolyError::NonPolynomialExpression(
                "unsupported original syntax in bounded polynomial conversion".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn multiply_polynomials<M: BudgetMeter>(
    a: &MultivariatePoly,
    b: &MultivariatePoly,
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    same_ring(&a.generators, &b.generators)?;
    work.allocate(ring_storage(&a.generators))?;
    if a.is_zero() || b.is_zero() {
        return Ok(MultivariatePoly::zero(a.generators.clone()));
    }
    let degree_a = a
        .terms
        .keys()
        .map(|m| m.iter().map(|&e| u64::from(e)).sum::<u64>())
        .max()
        .unwrap_or(0);
    let degree_b = b
        .terms
        .keys()
        .map(|m| m.iter().map(|&e| u64::from(e)).sum::<u64>())
        .max()
        .unwrap_or(0);
    if degree_a + degree_b > MAX_DEGREE {
        return Err(refused("expanded polynomial degree exceeds 256"));
    }
    work.tick(a.terms.len() * b.terms.len())?;
    let mut terms = Terms::new();
    for (left, x) in &a.terms {
        for (right, y) in &b.terms {
            work.allocate(left.len() * 4)?;
            let powers = left.iter().zip(right).map(|(x, y)| x + y).collect();
            let value = work.multiply(x, y)?;
            accumulate(&mut terms, powers, value, work)?;
        }
    }
    Ok(MultivariatePoly {
        generators: a.generators.clone(),
        terms,
    })
}

fn convert_expression<M: BudgetMeter>(
    expression: &Expr,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    work.tick(1)?;
    work.allocate(ring_storage(generators) + 256 + generators.len() * 4)?;
    let mut output = MultivariatePoly::zero(generators.to_vec());
    match expression {
        Expr::Integer(value) => {
            work.allocate((value.limb_count().max(1) * 8) as usize)?;
            if !value.is_zero() {
                output.terms.insert(
                    vec![0; generators.len()],
                    BigRational::from_integer(value.clone()),
                );
            }
        }
        Expr::Rational(value) => {
            if !value.is_zero() {
                output
                    .terms
                    .insert(vec![0; generators.len()], work.clone_value(value)?);
            }
        }
        Expr::Sym(symbol) => {
            let index = generators
                .iter()
                .position(|g| g == symbol)
                .ok_or_else(|| refused("undeclared generator"))?;
            let mut powers = vec![0; generators.len()];
            powers[index] = 1;
            output.terms.insert(powers, BigRational::one());
        }
        Expr::Add(children) => {
            for child in children {
                let polynomial = convert_expression(child, generators, work)?;
                for (powers, value) in polynomial.terms {
                    accumulate(&mut output.terms, powers, value, work)?;
                }
            }
        }
        Expr::Mul(children) => {
            output
                .terms
                .insert(vec![0; generators.len()], BigRational::one());
            for child in children {
                // Every child is converted, including after a zero factor;
                // undefined reciprocals cannot disappear through cancellation.
                let polynomial = convert_expression(child, generators, work)?;
                output = multiply_polynomials(&output, &polynomial, work)?;
            }
        }
        Expr::Pow(base, exponent) => {
            let mut factor = convert_expression(base, generators, work)?;
            let Expr::Integer(power) = exponent.as_ref() else {
                return Err(refused("nonliteral polynomial exponent"));
            };
            let power = power
                .to_i64()
                .ok_or_else(|| refused("polynomial exponent out of range"))?;
            if factor.is_zero() && power <= 0 {
                return Err(PolyError::NonPolynomialExpression(
                    "zero base with a nonpositive exponent is undefined".into(),
                ));
            }
            if power < 0 {
                if factor.terms.len() != 1 || factor.terms.keys().any(|m| m.iter().any(|&e| e != 0))
                {
                    return Err(PolyError::NonPolynomialExpression(
                        "a negative power requires a nonzero rational constant".into(),
                    ));
                }
                let value = factor
                    .terms
                    .values_mut()
                    .next()
                    .ok_or(PolyError::DivisionByZero)?;
                *value = work.divide(&BigRational::one(), value)?;
            }
            let mut remaining = power.unsigned_abs();
            output
                .terms
                .insert(vec![0; generators.len()], BigRational::one());
            while remaining != 0 {
                work.tick(1)?;
                if remaining & 1 != 0 {
                    output = multiply_polynomials(&output, &factor, work)?;
                }
                remaining >>= 1;
                if remaining != 0 {
                    factor = multiply_polynomials(&factor, &factor, work)?;
                }
            }
        }
        _ => {
            return Err(PolyError::NonPolynomialExpression(
                "unsupported polynomial syntax".into(),
            ));
        }
    }
    Ok(output)
}

/// Lower an exact polynomial expression without unmetered expansion. The whole
/// original syntax is validated before arithmetic, including children of zero
/// products and zero powers. Literal constant reciprocals are admitted; poles,
/// foreign symbols, unsupported nodes and excessive growth remain refusals.
pub fn bounded_polynomial_from_expr(
    expression: &Expr,
    generators: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<MultivariatePoly, PolyError> {
    let mut work = PolynomialWork::new(meter);
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused("polynomial generator boundary exceeded"));
    }
    let name_bytes = generators.iter().try_fold(0_usize, |bytes, generator| {
        bytes.checked_add(generator.name.len())
    });
    if name_bytes.is_none_or(|bytes| bytes > 65_536) {
        return Err(refused("polynomial generator boundary exceeded"));
    }
    work.allocate(ring_storage(generators))?;
    MultivariatePoly::zero(generators.to_vec()).validate_shape()?;
    let mut nodes = 0;
    preflight_expression(expression, generators, 0, &mut nodes, &mut work)?;
    let result = convert_expression(expression, generators, &mut work)?;
    validate_poly(&result, &mut work)?;
    work.tick(1)?;
    Ok(result)
}
