//! Independent complete Groebner-basis certificates over a declared rational polynomial ring.
//!
//! This module checks canonical sparse data with direct reference coefficient
//! convolution, never with `fsym-polys` or its search, multiplication, division,
//! or term-order routines. Sparse exponents are added with overflow checks;
//! they are not expanded through the generic expression-power normalizer.

use crate::KernelError;
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::mem::size_of;
use thiserror::Error;

const MAX_BASIS: usize = 1_024;
const MAX_POLYNOMIALS: usize = 16_384;
const MAX_TOTAL_NODES: usize = 262_144;
const MAX_IDENTITY_NODES: usize = 16_384;
const MAX_TERMS: usize = 4_096;
const MAX_VARIABLES: usize = 256;
const MAX_NAME_BYTES: usize = 65_536;
const MAX_COEFFICIENT_LIMBS: u64 = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroebnerTermOrder {
    Lex,
    DegLex,
    DegRevLex,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseRationalTerm {
    pub exponents: Vec<u32>,
    pub coefficient: BigRational,
}

/// Canonical sparse polynomial. Terms are strictly increasing in lexicographic
/// exponent-vector order, independently of the order claimed for the basis.
/// Zero has no terms; stored zero coefficients and duplicate terms are invalid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseRationalPolynomial {
    pub terms: Vec<SparseRationalTerm>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroebnerSPairWitness {
    pub left: usize,
    pub right: usize,
    pub coefficients: Vec<SparseRationalPolynomial>,
}

/// Untrusted cofactor evidence. Both ideal inclusions and every S-pair are
/// explicit. S-pair rows must occur in increasing `(left, right)` index order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroebnerBasisWitness {
    pub order: GroebnerTermOrder,
    pub basis: Vec<SparseRationalPolynomial>,
    pub basis_in_input: Vec<Vec<SparseRationalPolynomial>>,
    pub input_in_basis: Vec<Vec<SparseRationalPolynomial>>,
    pub s_pairs: Vec<GroebnerSPairWitness>,
}

/// Summary of an independently checked call. There is no unchecked decoder or
/// public constructor; callers retaining a verified artifact must also retain the
/// exact immutable inputs supplied to that call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroebnerVerificationReceipt {
    inputs: usize,
    basis: usize,
    s_pairs: usize,
    order: GroebnerTermOrder,
}

impl GroebnerVerificationReceipt {
    pub fn input_polynomials(&self) -> usize {
        self.inputs
    }
    pub fn basis_polynomials(&self) -> usize {
        self.basis
    }
    pub fn s_pairs_verified(&self) -> usize {
        self.s_pairs
    }
    pub fn order(&self) -> GroebnerTermOrder {
        self.order
    }
}

#[derive(Debug, Error)]
pub enum GroebnerVerificationError {
    #[error("invalid complete Groebner certificate: {0}")]
    Invalid(String),
    #[error(transparent)]
    Budget(#[from] MeterError),
    #[error(transparent)]
    Identity(#[from] KernelError),
}

type Result<T> = std::result::Result<T, GroebnerVerificationError>;
fn invalid(reason: impl std::fmt::Display) -> GroebnerVerificationError {
    GroebnerVerificationError::Invalid(reason.to_string())
}
fn work(meter: &mut impl BudgetMeter, steps: usize) -> Result<()> {
    meter.checkpoint()?;
    meter.charge(Dimension::ComputeSteps, steps.max(1) as u64)?;
    Ok(())
}

fn dimensions(inputs: usize, basis: usize) -> Result<usize> {
    if inputs > MAX_BASIS || basis > MAX_BASIS {
        return Err(invalid("basis dimension bound exceeded"));
    }
    let pairs = basis
        .checked_mul(basis.saturating_sub(1))
        .map(|n| n / 2)
        .ok_or_else(|| invalid("S-pair count overflow"))?;
    let objects = inputs
        .checked_mul(basis)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| pairs.checked_mul(basis).and_then(|p| n.checked_add(p)))
        .and_then(|n| n.checked_add(inputs))
        .and_then(|n| n.checked_add(basis))
        .ok_or_else(|| invalid("certificate dimension overflow"))?;
    if objects > MAX_POLYNOMIALS {
        return Err(invalid("certificate polynomial bound exceeded"));
    }
    Ok(pairs)
}

fn polynomial_cost(poly: &SparseRationalPolynomial, generators: &[Symbol]) -> (usize, u64) {
    let mut nodes = 1usize;
    let mut payload = 0u64;
    for term in &poly.terms {
        // Term Mul, rational coefficient, and the monomial expression root.
        nodes += 3;
        payload += term.coefficient.numer().bits().div_ceil(8)
            + term.coefficient.denom().bits().div_ceil(8);
        for (index, &power) in term.exponents.iter().enumerate() {
            if power != 0 {
                nodes += if power == 1 { 1 } else { 3 };
                payload += generators[index].name.len() as u64;
            }
        }
    }
    (
        nodes,
        payload + (nodes * (std::mem::size_of::<Expr>() + 2 * std::mem::size_of::<usize>())) as u64,
    )
}

fn preflight(
    generators: &[Symbol],
    inputs: &[SparseRationalPolynomial],
    witness: &GroebnerBasisWitness,
    meter: &mut impl BudgetMeter,
) -> Result<usize> {
    work(meter, 1)?;
    let pairs = dimensions(inputs.len(), witness.basis.len())?;
    if generators.len() > MAX_VARIABLES {
        return Err(invalid("variable count bound exceeded"));
    }
    let mut name_bytes = 0usize;
    for (index, symbol) in generators.iter().enumerate() {
        name_bytes = name_bytes
            .checked_add(symbol.name.len())
            .ok_or_else(|| invalid("name size overflow"))?;
        if name_bytes > MAX_NAME_BYTES {
            return Err(invalid("variable name bound exceeded"));
        }
        work(meter, index + symbol.name.len() + 1)?;
        if generators[..index].contains(symbol) {
            return Err(invalid("duplicate ring variable"));
        }
    }
    let m = witness.basis.len();
    if witness.basis_in_input.len() != m
        || witness
            .basis_in_input
            .iter()
            .any(|row| row.len() != inputs.len())
        || witness.input_in_basis.len() != inputs.len()
        || witness.input_in_basis.iter().any(|row| row.len() != m)
        || witness.s_pairs.len() != pairs
        || witness
            .s_pairs
            .iter()
            .any(|pair| pair.coefficients.len() != m)
    {
        return Err(invalid("cofactor matrix or S-pair dimensions are invalid"));
    }
    let all = inputs
        .iter()
        .chain(&witness.basis)
        .chain(witness.basis_in_input.iter().flatten())
        .chain(witness.input_in_basis.iter().flatten())
        .chain(witness.s_pairs.iter().flat_map(|pair| &pair.coefficients));
    let mut total_nodes = 0usize;
    for poly in all {
        work(meter, 1)?;
        if poly.terms.len() > MAX_TERMS {
            return Err(invalid("polynomial term bound exceeded"));
        }
        let mut previous: Option<&[u32]> = None;
        for term in &poly.terms {
            work(meter, generators.len() + 1)?;
            if term.exponents.len() != generators.len() {
                return Err(invalid("exponent width does not match ring"));
            }
            if term.coefficient.is_zero() {
                return Err(invalid("stored zero coefficient"));
            }
            if term.coefficient.numer().limb_count() > MAX_COEFFICIENT_LIMBS
                || term.coefficient.denom().limb_count() > MAX_COEFFICIENT_LIMBS
            {
                return Err(invalid("coefficient limb bound exceeded"));
            }
            if previous.is_some_and(|prior| prior >= term.exponents.as_slice()) {
                return Err(invalid("noncanonical or duplicate polynomial terms"));
            }
            previous = Some(&term.exponents);
        }
        let (nodes, _) = polynomial_cost(poly, generators);
        if nodes > MAX_IDENTITY_NODES {
            return Err(invalid("polynomial expression bound exceeded"));
        }
        total_nodes = total_nodes
            .checked_add(nodes)
            .ok_or_else(|| invalid("node count overflow"))?;
        if total_nodes > MAX_TOTAL_NODES {
            return Err(invalid("certificate expression bound exceeded"));
        }
    }
    Ok(pairs)
}

fn monomial_cmp(a: &[u32], b: &[u32], order: GroebnerTermOrder) -> Ordering {
    if order != GroebnerTermOrder::Lex {
        let left: u128 = a.iter().map(|&power| u128::from(power)).sum();
        let right: u128 = b.iter().map(|&power| u128::from(power)).sum();
        if left != right {
            return left.cmp(&right);
        }
    }
    if order == GroebnerTermOrder::DegRevLex {
        for (&left, &right) in a.iter().rev().zip(b.iter().rev()) {
            if left != right {
                return right.cmp(&left);
            }
        }
    } else {
        for (&left, &right) in a.iter().zip(b) {
            if left != right {
                return left.cmp(&right);
            }
        }
    }
    Ordering::Equal
}

fn leading_monomials<'a>(
    basis: &'a [SparseRationalPolynomial],
    order: GroebnerTermOrder,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<&'a [u32]>> {
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            (basis.len().max(1) * std::mem::size_of::<&[u32]>()) as u64,
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut leading = Vec::new();
    leading.try_reserve_exact(basis.len()).map_err(invalid)?;
    for polynomial in basis {
        let mut best: Option<&SparseRationalTerm> = None;
        for term in &polynomial.terms {
            work(meter, 2 * term.exponents.len() + 1)?;
            if best
                .is_none_or(|prior| monomial_cmp(&term.exponents, &prior.exponents, order).is_gt())
            {
                best = Some(term);
            }
        }
        let term = best.ok_or_else(|| invalid("basis contains zero"))?;
        if !term.coefficient.is_one() {
            return Err(invalid("basis is not monic under the declared order"));
        }
        leading.push(term.exponents.as_slice());
    }
    for (index, polynomial) in basis.iter().enumerate() {
        for (other_index, monomial) in leading.iter().enumerate() {
            if index == other_index {
                continue;
            }
            for term in &polynomial.terms {
                work(meter, term.exponents.len() + 1)?;
                if monomial
                    .iter()
                    .zip(&term.exponents)
                    .all(|(&divisor, &power)| divisor <= power)
                {
                    return Err(invalid("basis is not reduced"));
                }
            }
        }
    }
    Ok(leading)
}

fn reserve_identity<'a>(
    generators: &[Symbol],
    polynomials: impl IntoIterator<Item = &'a SparseRationalPolynomial>,
    wrappers: usize,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let mut nodes = wrappers;
    let mut bytes =
        (wrappers * (std::mem::size_of::<Expr>() + 2 * std::mem::size_of::<usize>())) as u64;
    for polynomial in polynomials {
        work(meter, polynomial.terms.len() + 1)?;
        let (poly_nodes, poly_bytes) = polynomial_cost(polynomial, generators);
        nodes = nodes
            .checked_add(poly_nodes)
            .ok_or_else(|| invalid("identity node count overflow"))?;
        bytes = bytes
            .checked_add(poly_bytes)
            .ok_or_else(|| invalid("identity payload overflow"))?;
        if nodes > MAX_IDENTITY_NODES {
            return Err(invalid("identity expression bound exceeded"));
        }
    }
    meter.charge_batch(&[
        (Dimension::MemoryBytes, bytes.max(1)),
        (Dimension::AllocationCount, nodes as u64),
    ])?;
    meter.checkpoint()?;
    Ok(())
}

type IdentityTerms = BTreeMap<Vec<u32>, BigRational>;

fn reserve_reference(bytes: usize, meter: &mut impl BudgetMeter) -> Result<()> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::MemoryBytes, bytes.max(1) as u64),
        (Dimension::AllocationCount, 1),
    ])?;
    Ok(())
}

fn bounded_coefficient(value: &BigRational) -> Result<()> {
    if value.numer().limb_count() > MAX_COEFFICIENT_LIMBS
        || value.denom().limb_count() > MAX_COEFFICIENT_LIMBS
    {
        return Err(invalid("reference coefficient limb bound exceeded"));
    }
    Ok(())
}

fn checked_arithmetic(
    result: std::result::Result<BigRational, RationalArithmeticError>,
) -> Result<BigRational> {
    let value = result.map_err(|error| match error {
        RationalArithmeticError::Meter(error) => GroebnerVerificationError::Budget(error),
        error => invalid(error),
    })?;
    bounded_coefficient(&value)?;
    Ok(value)
}

fn copy_coefficient(value: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    bounded_coefficient(value)?;
    let limbs = value.numer().limb_count() + value.denom().limb_count();
    work(meter, limbs as usize + 1)?;
    reserve_reference(size_of::<BigRational>() + limbs as usize * 8, meter)?;
    Ok(value.clone())
}

/// Add one signed term. Both operands are bounded before scalar arithmetic;
/// hence cross products use at most twice the admitted limb count, plus one
/// carry limb for a sum. The exact-arithmetic layer meters those temporary
/// buffers and normalization before the result returns to the same height cap.
fn accumulate_reference(
    result: &mut IdentityTerms,
    exponents: Vec<u32>,
    coefficient: BigRational,
    subtract: bool,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    bounded_coefficient(&coefficient)?;
    work(
        meter,
        (exponents.len() + 1) * (result.len().max(1).ilog2().min(12) as usize + 1),
    )?;
    let combined = if let Some(previous) = result.get(&exponents) {
        bounded_coefficient(previous)?;
        checked_arithmetic(if subtract {
            previous.metered_sub(&coefficient, meter)
        } else {
            previous.metered_add(&coefficient, meter)
        })?
    } else if subtract {
        checked_arithmetic(BigRational::zero().metered_sub(&coefficient, meter))?
    } else {
        coefficient
    };
    if combined.is_zero() {
        result.remove(&exponents);
    } else {
        if !result.contains_key(&exponents) {
            if result.len() == MAX_TERMS {
                return Err(invalid("reference identity term bound exceeded"));
            }
            reserve_reference(256 + size_of::<BigRational>(), meter)?;
        }
        result.insert(exponents, combined);
    }
    Ok(())
}

fn add_exponents(left: &[u32], right: &[u32], meter: &mut impl BudgetMeter) -> Result<Vec<u32>> {
    if left.len() != right.len() {
        return Err(invalid("identity exponent widths differ"));
    }
    work(meter, left.len() + 1)?;
    reserve_reference(left.len() * size_of::<u32>(), meter)?;
    let mut result = Vec::with_capacity(left.len());
    for (&a, &b) in left.iter().zip(right) {
        result.push(
            a.checked_add(b)
                .ok_or_else(|| invalid("identity exponent overflow"))?,
        );
    }
    Ok(result)
}

fn combination_terms(
    inputs: &[SparseRationalPolynomial],
    coefficients: &[SparseRationalPolynomial],
    meter: &mut impl BudgetMeter,
) -> Result<IdentityTerms> {
    reserve_reference(size_of::<IdentityTerms>(), meter)?;
    let mut result = IdentityTerms::new();
    for (input, coefficient) in inputs.iter().zip(coefficients) {
        work(meter, 1)?;
        // Preserve the generic reference normalizer's product-column bound.
        if input.terms.len().saturating_mul(coefficient.terms.len()) > MAX_TERMS {
            return Err(invalid("reference product term bound exceeded"));
        }
        for a in &input.terms {
            for b in &coefficient.terms {
                let exponents = add_exponents(&a.exponents, &b.exponents, meter)?;
                bounded_coefficient(&a.coefficient)?;
                bounded_coefficient(&b.coefficient)?;
                let value = checked_arithmetic(a.coefficient.metered_mul(&b.coefficient, meter))?;
                accumulate_reference(&mut result, exponents, value, false, meter)?;
            }
        }
    }
    Ok(result)
}

fn same_coefficient(
    left_exponents: &[u32],
    left: &BigRational,
    right_exponents: &[u32],
    right: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    work(
        meter,
        left_exponents.len()
            + (left.numer().limb_count()
                + left.denom().limb_count()
                + right.numer().limb_count()
                + right.denom().limb_count()) as usize
            + 1,
    )?;
    if left_exponents != right_exponents || left != right {
        return Err(invalid("reference coefficient identity failed"));
    }
    Ok(())
}

fn verify_combination(
    generators: &[Symbol],
    target: &SparseRationalPolynomial,
    inputs: &[SparseRationalPolynomial],
    coefficients: &[SparseRationalPolynomial],
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    reserve_identity(
        generators,
        std::iter::once(target).chain(inputs).chain(coefficients),
        inputs.len() + 1,
        meter,
    )?;
    let combined = combination_terms(inputs, coefficients, meter)?;
    if target.terms.len() != combined.len() {
        return Err(invalid("reference identity term coverage differs"));
    }
    for (expected, (exponents, coefficient)) in target.terms.iter().zip(&combined) {
        same_coefficient(
            &expected.exponents,
            &expected.coefficient,
            exponents,
            coefficient,
            meter,
        )?;
    }
    meter.checkpoint()?;
    Ok(())
}

fn verify_s_pair(
    generators: &[Symbol],
    witness: &GroebnerBasisWitness,
    leading: &[&[u32]],
    pair: &GroebnerSPairWitness,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let width = generators.len();
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            (4 * width.max(1) * std::mem::size_of::<u32>()) as u64,
        ),
        (Dimension::AllocationCount, 4),
    ])?;
    let lcm = leading[pair.left]
        .iter()
        .zip(leading[pair.right])
        .map(|(&a, &b)| a.max(b))
        .collect::<Vec<_>>();
    let mut shifted = vec![0u32; width];
    // Multiplicativity of all admitted orders makes term(q)+LM(g) dominate
    // every term of that product, without invoking polynomial multiplication.
    for (coefficient, leading_basis) in pair.coefficients.iter().zip(leading) {
        for term in &coefficient.terms {
            work(meter, 3 * width + 1)?;
            for ((result, &power), &basis_power) in
                shifted.iter_mut().zip(&term.exponents).zip(*leading_basis)
            {
                *result = power
                    .checked_add(basis_power)
                    .ok_or_else(|| invalid("S-pair exponent overflow"))?;
            }
            if !monomial_cmp(&shifted, &lcm, witness.order).is_lt() {
                return Err(invalid(
                    "S-pair representation is not strictly below its LCM",
                ));
            }
        }
    }
    let left_shift = lcm
        .iter()
        .zip(leading[pair.left])
        .map(|(&a, &b)| a - b)
        .collect::<Vec<_>>();
    let right_shift = lcm
        .iter()
        .zip(leading[pair.right])
        .map(|(&a, &b)| a - b)
        .collect::<Vec<_>>();
    let basis = &witness.basis;
    reserve_identity(
        generators,
        [&basis[pair.left], &basis[pair.right]]
            .into_iter()
            .chain(&pair.coefficients)
            .chain(basis),
        basis.len() + 8 + 6 * width,
        meter,
    )?;
    reserve_reference(size_of::<IdentityTerms>(), meter)?;
    let mut lhs = IdentityTerms::new();
    for (index, shift, subtract) in [
        (pair.left, &left_shift, false),
        (pair.right, &right_shift, true),
    ] {
        for term in &basis[index].terms {
            let exponents = add_exponents(&term.exponents, shift, meter)?;
            let coefficient = copy_coefficient(&term.coefficient, meter)?;
            accumulate_reference(&mut lhs, exponents, coefficient, subtract, meter)?;
        }
    }
    let rhs = combination_terms(basis, &pair.coefficients, meter)?;
    if lhs.len() != rhs.len() {
        return Err(invalid("reference S-pair term coverage differs"));
    }
    for ((a, ca), (b, cb)) in lhs.iter().zip(&rhs) {
        same_coefficient(a, ca, b, cb, meter)?;
    }
    meter.checkpoint()?;
    Ok(())
}

/// Checks both ideal inclusions, monicity, reducedness, and Buchberger's criterion
/// without any optimizing generator. Each S-pair identity must have a strict standard
/// representation: every monomial of every product `q_k*g_k` lies below the pair LCM.
/// This additional inequality makes the identities a completeness certificate rather
/// than the tautology that S-polynomials belong to the ideal generated by the basis.
///
/// The caller supplies the exact ring and input ideal; certificate fields never replace
/// them. Work and allocations are metered, shapes and normalization have static bounds,
/// and cancellation or refusal never returns an accepted receipt.
pub fn verify_groebner_basis_witness(
    generators: &[Symbol],
    inputs: &[SparseRationalPolynomial],
    witness: &GroebnerBasisWitness,
    meter: &mut impl BudgetMeter,
) -> Result<GroebnerVerificationReceipt> {
    let pairs = preflight(generators, inputs, witness, meter)?;
    let leading = leading_monomials(&witness.basis, witness.order, meter)?;
    for (target, coefficients) in witness.basis.iter().zip(&witness.basis_in_input) {
        verify_combination(generators, target, inputs, coefficients, meter)?;
    }
    for (target, coefficients) in inputs.iter().zip(&witness.input_in_basis) {
        verify_combination(generators, target, &witness.basis, coefficients, meter)?;
    }
    let mut rows = witness.s_pairs.iter();
    for left in 0..witness.basis.len() {
        for right in (left + 1)..witness.basis.len() {
            work(meter, 1)?;
            let pair = rows.next().ok_or_else(|| invalid("missing S-pair"))?;
            if pair.left != left || pair.right != right {
                return Err(invalid(
                    "S-pair indices are missing, duplicated, or out of order",
                ));
            }
            verify_s_pair(generators, witness, &leading, pair, meter)?;
        }
    }
    meter.checkpoint()?;
    Ok(GroebnerVerificationReceipt {
        inputs: inputs.len(),
        basis: witness.basis.len(),
        s_pairs: pairs,
        order: witness.order,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, Unbounded};
    use fsym_core::BigInt;

    fn poly(terms: &[([u32; 2], i64)]) -> SparseRationalPolynomial {
        let mut terms = terms
            .iter()
            .map(|(powers, coefficient)| SparseRationalTerm {
                exponents: powers.to_vec(),
                coefficient: BigRational::from_integer((*coefficient).into()),
            })
            .collect::<Vec<_>>();
        terms.sort_by(|a, b| a.exponents.cmp(&b.exponents));
        SparseRationalPolynomial { terms }
    }

    /// Hand-derived evidence for <xy-1,y²-1> = <x-y,y²-1>.
    /// No optimizing polynomial crate is available in this crate's dependencies.
    fn hand_certificate() -> (
        Vec<Symbol>,
        Vec<SparseRationalPolynomial>,
        GroebnerBasisWitness,
    ) {
        let generators = vec![Symbol::new("x"), Symbol::new("y")];
        let f = poly(&[([1, 1], 1), ([0, 0], -1)]);
        let h = poly(&[([0, 2], 1), ([0, 0], -1)]);
        let g = poly(&[([1, 0], 1), ([0, 1], -1)]);
        let zero = poly(&[]);
        let one = poly(&[([0, 0], 1)]);
        let y = poly(&[([0, 1], 1)]);
        let minus_x = poly(&[([1, 0], -1)]);
        let minus_y = poly(&[([0, 1], -1)]);
        let witness = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: vec![g, h.clone()],
            // x-y = y*(xy-1)-x*(y²-1).
            basis_in_input: vec![vec![y.clone(), minus_x], vec![zero.clone(), one.clone()]],
            // xy-1 = y*(x-y)+(y²-1).
            input_in_basis: vec![vec![y, one.clone()], vec![zero, one.clone()]],
            // y²*(x-y)-x*(y²-1) = (x-y)-y*(y²-1).
            s_pairs: vec![GroebnerSPairWitness {
                left: 0,
                right: 1,
                coefficients: vec![one, minus_y],
            }],
        };
        (generators, vec![f, h], witness)
    }

    #[test]
    fn hand_derived_complete_basis_checks_in_the_independent_kernel() {
        let (generators, input, mut witness) = hand_certificate();
        for order in [
            GroebnerTermOrder::Lex,
            GroebnerTermOrder::DegLex,
            GroebnerTermOrder::DegRevLex,
        ] {
            witness.order = order;
            let receipt =
                verify_groebner_basis_witness(&generators, &input, &witness, &mut Unbounded)
                    .unwrap();
            assert_eq!(receipt.input_polynomials(), 2);
            assert_eq!(receipt.basis_polynomials(), 2);
            assert_eq!(receipt.s_pairs_verified(), 1);
            assert_eq!(receipt.order(), order);
        }
        let bytes = serde_json::to_vec(&witness).unwrap();
        let decoded: GroebnerBasisWitness = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, witness);
        verify_groebner_basis_witness(&generators, &input, &decoded, &mut Unbounded).unwrap();
    }

    #[test]
    fn sparse_high_exponents_preserve_both_inclusions_and_strict_s_pairs() {
        let (generators, mut input, mut witness) = hand_certificate();
        // Substitute y^80 for y in the hand-derived Lex certificate. The
        // S-pair identity reaches y^240, while all sparse rows stay tiny.
        // Its proof must not expand an expression power or factor a source.
        for polynomial in input
            .iter_mut()
            .chain(&mut witness.basis)
            .chain(witness.basis_in_input.iter_mut().flatten())
            .chain(witness.input_in_basis.iter_mut().flatten())
            .chain(
                witness
                    .s_pairs
                    .iter_mut()
                    .flat_map(|pair| &mut pair.coefficients),
            )
        {
            for term in &mut polynomial.terms {
                term.exponents[1] *= 80;
            }
        }
        let accepted =
            verify_groebner_basis_witness(&generators, &input, &witness, &mut Unbounded).unwrap();
        assert_eq!(accepted.s_pairs_verified(), 1);
        for family in 0..3 {
            let mut corrupted = witness.clone();
            let coefficient = match family {
                0 => &mut corrupted.basis_in_input[0][0],
                1 => &mut corrupted.input_in_basis[0][0],
                _ => &mut corrupted.s_pairs[0].coefficients[0],
            };
            coefficient.terms[0].coefficient += BigRational::new(1.into(), 1_000_003.into());
            assert!(
                verify_groebner_basis_witness(&generators, &input, &corrupted, &mut Unbounded)
                    .is_err()
            );
        }
        let mut changed_input = input.clone();
        changed_input[0].terms[0].coefficient = BigRational::from_integer(2.into());
        assert!(
            verify_groebner_basis_witness(&generators, &changed_input, &witness, &mut Unbounded)
                .is_err()
        );
    }

    #[test]
    fn compact_monomial_proofs_support_representable_exponents_in_every_order() {
        let generators = vec![Symbol::new("x"), Symbol::new("y")];
        let input = vec![poly(&[([u32::MAX, 0], 1)]), poly(&[([0, 100], 1)])];
        let zero = poly(&[]);
        let one = poly(&[([0, 0], 1)]);
        let identity = vec![vec![one.clone(), zero.clone()], vec![zero.clone(), one]];
        for order in [
            GroebnerTermOrder::Lex,
            GroebnerTermOrder::DegLex,
            GroebnerTermOrder::DegRevLex,
        ] {
            let witness = GroebnerBasisWitness {
                order,
                basis: input.clone(),
                basis_in_input: identity.clone(),
                input_in_basis: identity.clone(),
                s_pairs: vec![GroebnerSPairWitness {
                    left: 0,
                    right: 1,
                    coefficients: vec![zero.clone(), zero.clone()],
                }],
            };
            verify_groebner_basis_witness(&generators, &input, &witness, &mut Unbounded).unwrap();
        }
        let input = vec![input[0].clone()];
        let overflow = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: input.clone(),
            basis_in_input: vec![vec![poly(&[([1, 0], 1)])]],
            input_in_basis: vec![vec![poly(&[([0, 0], 1)])]],
            s_pairs: vec![],
        };
        let error = verify_groebner_basis_witness(&generators, &input, &overflow, &mut Unbounded)
            .unwrap_err();
        assert!(error.to_string().contains("exponent overflow"));
    }

    #[test]
    fn reference_coefficient_height_is_bounded_without_rejecting_exact_cancellation() {
        let generators = vec![Symbol::new("x"), Symbol::new("y")];
        let large =
            BigRational::from_integer(BigInt::one() << ((MAX_COEFFICIENT_LIMBS * 64 - 1) as u32));
        let constant = |coefficient| SparseRationalPolynomial {
            terms: vec![SparseRationalTerm {
                exponents: vec![0, 0],
                coefficient,
            }],
        };
        let one = constant(BigRational::one());
        verify_combination(
            &generators,
            &one,
            &[constant(large.clone())],
            &[constant(BigRational::one() / large.clone())],
            &mut Unbounded,
        )
        .unwrap();
        let error = verify_combination(
            &generators,
            &one,
            &[constant(large.clone())],
            &[constant(BigRational::from_integer(2.into()))],
            &mut Unbounded,
        )
        .unwrap_err();
        assert!(error.to_string().contains("coefficient limb bound"));
        let error = verify_combination(
            &generators,
            &one,
            &[constant(large.clone()), constant(large)],
            &[one.clone(), one.clone()],
            &mut Unbounded,
        )
        .unwrap_err();
        assert!(error.to_string().contains("coefficient limb bound"));
    }

    #[test]
    fn sparse_reference_retains_product_and_identity_size_refusals() {
        let generators = vec![Symbol::new("x"), Symbol::new("y")];
        let many = SparseRationalPolynomial {
            terms: (0..65)
                .map(|power| SparseRationalTerm {
                    exponents: vec![power, 0],
                    coefficient: BigRational::one(),
                })
                .collect(),
        };
        let error = verify_combination(
            &generators,
            &poly(&[]),
            std::slice::from_ref(&many),
            std::slice::from_ref(&many),
            &mut Unbounded,
        )
        .unwrap_err();
        assert!(error.to_string().contains("product term bound"));
        let oversized = SparseRationalPolynomial {
            terms: (0..4096)
                .map(|power| SparseRationalTerm {
                    exponents: vec![power, 0],
                    coefficient: BigRational::one(),
                })
                .collect(),
        };
        let error = verify_combination(
            &generators,
            &oversized,
            std::slice::from_ref(&oversized),
            &[poly(&[([0, 0], 1)])],
            &mut Unbounded,
        )
        .unwrap_err();
        assert!(error.to_string().contains("identity expression bound"));
    }

    #[test]
    fn tautological_s_pair_membership_does_not_establish_buchberger() {
        let (generators, input, good) = hand_certificate();
        let zero = poly(&[]);
        let one = poly(&[([0, 0], 1)]);
        let y = poly(&[([0, 1], 1)]);
        let minus_x = poly(&[([1, 0], -1)]);
        // The identity itself is true: S(xy-1,y²-1)=x-y. This is NOT a
        // standard representation, because both coefficient products reach xy².
        verify_combination(
            &generators,
            &good.basis[0],
            &input,
            &[y.clone(), minus_x.clone()],
            &mut Unbounded,
        )
        .unwrap();
        let identity = vec![vec![one.clone(), zero.clone()], vec![zero, one]];
        let bad = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: input.clone(),
            basis_in_input: identity.clone(),
            input_in_basis: identity,
            s_pairs: vec![GroebnerSPairWitness {
                left: 0,
                right: 1,
                coefficients: vec![y, minus_x],
            }],
        };
        let error =
            verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded).unwrap_err();
        assert!(error.to_string().contains("strictly below"));
    }

    #[test]
    fn both_ideal_inclusions_and_the_exact_requested_input_are_required() {
        let (generators, input, good) = hand_certificate();
        let zero = poly(&[]);
        let one = poly(&[([0, 0], 1)]);
        let mut bad = good.clone();
        bad.basis_in_input[0] = vec![zero.clone(), zero.clone()];
        assert!(verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded).is_err());
        bad = good.clone();
        bad.input_in_basis[0] = vec![zero.clone(), zero.clone()];
        assert!(verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded).is_err());
        let forged_unit = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: vec![one.clone()],
            basis_in_input: vec![vec![zero.clone(), zero.clone()]],
            input_in_basis: input
                .iter()
                .map(|polynomial| vec![polynomial.clone()])
                .collect(),
            s_pairs: vec![],
        };
        assert!(
            verify_groebner_basis_witness(&generators, &input, &forged_unit, &mut Unbounded)
                .is_err()
        );
        let incomplete = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: vec![input[1].clone()],
            basis_in_input: vec![vec![zero.clone(), one.clone()]],
            input_in_basis: vec![vec![zero], vec![one]],
            s_pairs: vec![],
        };
        assert!(
            verify_groebner_basis_witness(&generators, &input, &incomplete, &mut Unbounded)
                .is_err()
        );
        let mut changed_input = input.clone();
        changed_input[0].terms[0].coefficient = BigRational::from_integer(1.into());
        assert!(
            verify_groebner_basis_witness(&generators, &changed_input, &good, &mut Unbounded)
                .is_err()
        );
    }

    #[test]
    fn missing_mutated_and_malformed_s_pairs_are_refused() {
        let (generators, input, good) = hand_certificate();
        let mut bad = good.clone();
        bad.s_pairs.clear();
        assert!(verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded).is_err());
        for (left, right) in [(0, 0), (1, 0), (usize::MAX, 1)] {
            bad = good.clone();
            bad.s_pairs[0].left = left;
            bad.s_pairs[0].right = right;
            assert!(
                verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded).is_err()
            );
        }
        bad = good.clone();
        bad.s_pairs[0].coefficients = vec![poly(&[]), poly(&[])];
        assert!(verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded).is_err());
        bad = good.clone();
        bad.s_pairs[0].coefficients[0] = poly(&[([u32::MAX, 0], 1)]);
        assert!(
            verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded)
                .unwrap_err()
                .to_string()
                .contains("exponent overflow")
        );
    }

    #[test]
    fn canonical_shape_monicity_ring_and_coefficient_bounds_are_checked() {
        let (generators, input, good) = hand_certificate();
        let mut cases = Vec::new();
        let mut bad = good.clone();
        bad.basis[0].terms[1].coefficient = BigRational::from_integer(2.into());
        cases.push(bad);
        let mut bad = good.clone();
        bad.basis[0].terms.clear();
        cases.push(bad);
        let mut bad = good.clone();
        bad.basis[0].terms[0].exponents.pop();
        cases.push(bad);
        let mut bad = good.clone();
        bad.basis[0].terms[0].coefficient = BigRational::zero();
        cases.push(bad);
        let mut bad = good.clone();
        bad.basis[0].terms.swap(0, 1);
        cases.push(bad);
        let mut bad = good.clone();
        let duplicate = bad.basis[0].terms[1].clone();
        bad.basis[0].terms.push(duplicate);
        cases.push(bad);
        let mut bad = good.clone();
        bad.basis_in_input[0].pop();
        cases.push(bad);
        let mut bad = good.clone();
        bad.basis_in_input[0][0].terms[0].coefficient =
            BigRational::from_integer(BigInt::one() << ((MAX_COEFFICIENT_LIMBS * 64) as u32));
        cases.push(bad);
        for bad in cases {
            assert!(
                verify_groebner_basis_witness(&generators, &input, &bad, &mut Unbounded).is_err()
            );
        }
        assert!(
            verify_groebner_basis_witness(
                &[generators[0].clone(), generators[0].clone()],
                &input,
                &good,
                &mut Unbounded
            )
            .is_err()
        );
    }

    #[test]
    fn term_order_changes_cannot_reuse_a_lexicographic_leading_term_claim() {
        assert!(monomial_cmp(&[1, 0, 1], &[0, 2, 0], GroebnerTermOrder::DegLex).is_gt());
        assert!(monomial_cmp(&[1, 0, 1], &[0, 2, 0], GroebnerTermOrder::DegRevLex).is_lt());
        let generators = vec![Symbol::new("x"), Symbol::new("y")];
        let input = vec![poly(&[([1, 0], 1), ([0, 2], -1)])];
        let one = poly(&[([0, 0], 1)]);
        let mut witness = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: input.clone(),
            basis_in_input: vec![vec![one.clone()]],
            input_in_basis: vec![vec![one]],
            s_pairs: vec![],
        };
        verify_groebner_basis_witness(&generators, &input, &witness, &mut Unbounded).unwrap();
        witness.order = GroebnerTermOrder::DegLex;
        assert!(
            verify_groebner_basis_witness(&generators, &input, &witness, &mut Unbounded)
                .unwrap_err()
                .to_string()
                .contains("not monic")
        );
    }

    #[derive(Default)]
    struct Checkpoints {
        seen: usize,
        cancel_at: Option<usize>,
    }
    impl BudgetMeter for Checkpoints {
        fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
            self.seen += 1;
            if self.cancel_at == Some(self.seen) {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn cancellation_and_resource_refusal_never_return_a_complete_receipt() {
        let (generators, input, witness) = hand_certificate();
        let mut count = Checkpoints::default();
        verify_groebner_basis_witness(&generators, &input, &witness, &mut count).unwrap();
        assert!(count.seen > 10);
        for checkpoint in 1..=count.seen {
            let mut meter = Checkpoints {
                seen: 0,
                cancel_at: Some(checkpoint),
            };
            assert!(
                verify_groebner_basis_witness(&generators, &input, &witness, &mut meter).is_err()
            );
        }
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut limits = BudgetLimits::uniform(1_000_000, 0);
            limits.dimensions[dimension.index()] = 0;
            assert!(
                verify_groebner_basis_witness(
                    &generators,
                    &input,
                    &witness,
                    &mut Budget::new(limits)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn finite_budgets_accept_empty_zero_and_constant_ideals_and_empty_names() {
        let budget = || Budget::new(BudgetLimits::uniform(1_000_000, 0));
        let empty = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: vec![],
            basis_in_input: vec![],
            input_in_basis: vec![],
            s_pairs: vec![],
        };
        verify_groebner_basis_witness(&[], &[], &empty, &mut budget()).unwrap();
        let mut zero = empty.clone();
        zero.input_in_basis.push(vec![]);
        verify_groebner_basis_witness(
            &[Symbol::new("x"), Symbol::new("y")],
            &[poly(&[])],
            &zero,
            &mut budget(),
        )
        .unwrap();
        let scalar = |n: i64, d: i64| SparseRationalPolynomial {
            terms: vec![SparseRationalTerm {
                exponents: vec![],
                coefficient: BigRational::new(n.into(), d.into()),
            }],
        };
        let unit = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: vec![scalar(1, 1)],
            basis_in_input: vec![vec![scalar(1, 2)]],
            input_in_basis: vec![vec![scalar(2, 1)]],
            s_pairs: vec![],
        };
        verify_groebner_basis_witness(&[], &[scalar(2, 1)], &unit, &mut budget()).unwrap();
        let (_, input, witness) = hand_certificate();
        let generators = [
            Symbol::with_identity(
                "",
                fsym_core::SymbolIdentity::from_assumptions_digest([1; 32]),
            ),
            Symbol::with_identity(
                "",
                fsym_core::SymbolIdentity::from_assumptions_digest([2; 32]),
            ),
        ];
        verify_groebner_basis_witness(&generators, &input, &witness, &mut budget()).unwrap();
        let false_zero = GroebnerBasisWitness {
            order: GroebnerTermOrder::Lex,
            basis: vec![poly(&[([1, 0], 1), ([0, 1], -1)])],
            basis_in_input: vec![vec![]],
            input_in_basis: vec![],
            s_pairs: vec![],
        };
        assert!(
            verify_groebner_basis_witness(&generators, &[], &false_zero, &mut budget()).is_err()
        );
    }
}
