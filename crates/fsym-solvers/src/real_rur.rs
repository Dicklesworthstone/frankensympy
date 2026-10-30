//! Complete bounded real solution sets through a separating linear form.
//!
//! Ordinary shape position is tried first. A finite nonshape ideal is then
//! subjected to a bounded deterministic sequence of invertible shears
//! `x_last = t - sum(c_i*x_i)`, keeping the polynomial ring at its original
//! width. Finite shear attempts use a checked degree-reverse-lexicographic
//! basis followed by checked FGLM conversion, avoiding a direct lexicographic
//! search with large intermediate cofactors. Both accepted ideals remain in
//! the receipt and their exact link is checked. A shape basis then proves every
//! solution is represented by exactly one root of its eliminant. No separate
//! heuristic assertion that the linear form separates the roots is trusted.
//!
//! The reference boundary independently expands the inverse substitution on
//! every transformed input and recovers every original polynomial exactly.
//! It also checks coordinate recovery and every original residual modulo the
//! eliminant. Thus the accepted representation proves both solution directions,
//! rather than merely verifying a collection of candidate points.
//!
//! All attempts share separate cumulative generator/verifier caps. Exhausted
//! work, failed verification and cancellation propagate. A recognition miss
//! never means an empty set; nonmonogenic nonradical quotients, positive
//! dimensional systems and forms outside this search remain unsupported.
//! The receipt represents all real solutions. A complex-domain consumer must
//! additionally require `all_complex_roots_real()` before publishing it as a
//! complete complex solution set.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::real_roots::{MAX_REAL_ROOT_DEGREE, VerifiedRealRoots};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, TermOrder, VerifiedGroebnerBasis, bounded_groebner_certificate,
    bounded_polynomial_from_expr, verify_complete_groebner_certificate,
};
use num_traits::{One, Zero};

use crate::SolverError;
use crate::real_polysys::{VerifiedRealShapeSystem, metered_real_shape_from_basis};

const MAX_VARIABLES: usize = 8;
const MAX_EQUATIONS: usize = 64;
const MAX_TERMS: usize = 4096;
const MAX_DEGREE: u64 = 256;
const MAX_BITS: u64 = 4096;
const MAX_SOURCE_NODES: usize = 4096;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;
const SHEAR_VALUES: [i64; 7] = [1, -1, 2, -2, 3, -3, 4];

type Result<T> = std::result::Result<T, SolverError>;
type Coefficients = Vec<BigRational>;
type Terms = BTreeMap<Vec<u32>, BigRational>;

#[derive(Debug, Clone)]
struct CheckedShape {
    shape: VerifiedRealShapeSystem,
    // FGLM's input ideal is the previous reduced basis. Retain the independently
    // checked previous receipt so no original transformed equation is lost.
    intermediate: Option<VerifiedGroebnerBasis>,
}

impl CheckedShape {
    fn transformed_basis(&self) -> &VerifiedGroebnerBasis {
        self.intermediate
            .as_ref()
            .unwrap_or_else(|| self.shape.basis())
    }
}

/// An immutable complete real representation bound to the original ideal.
///
/// For each distinct indexed root `t` of `eliminant`, evaluate each ascending
/// coordinate vector at `t`. Coordinates follow the original caller order;
/// the last coordinate need not equal `t`. Repeated roots retain the accepted
/// quotient multiplicity without duplicating the corresponding tuple.
#[derive(Debug, Clone)]
pub struct VerifiedRealRurSystem {
    variables: Vec<Symbol>,
    equations: Vec<MultivariatePoly>,
    parameter: Symbol,
    shear: Coefficients,
    coordinates: Vec<Coefficients>,
    representation: CheckedShape,
}

impl VerifiedRealRurSystem {
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    pub fn equations(&self) -> &[MultivariatePoly] {
        &self.equations
    }

    /// Original final variable for ordinary shape, otherwise a fresh internal
    /// symbol for `t = x_last + sum(shear[i] * x_i)`.
    pub fn parameter(&self) -> &Symbol {
        &self.parameter
    }

    pub fn shear(&self) -> &[BigRational] {
        &self.shear
    }

    pub fn eliminant(&self) -> &[BigRational] {
        self.representation.shape.eliminant()
    }

    pub fn coordinates(&self) -> &[Vec<BigRational>] {
        &self.coordinates
    }

    pub fn roots(&self) -> &VerifiedRealRoots {
        self.representation.shape.roots()
    }

    pub fn basis(&self) -> &VerifiedGroebnerBasis {
        self.representation.shape.basis()
    }

    /// First accepted ideal in transformed coordinates, whose input equations
    /// independently recover every original equation under the inverse shear.
    pub fn transformed_basis(&self) -> &VerifiedGroebnerBasis {
        self.representation.transformed_basis()
    }

    pub fn all_complex_roots_real(&self) -> bool {
        self.representation.shape.all_complex_roots_real()
    }
}

fn refuse(message: impl std::fmt::Display) -> SolverError {
    SolverError::IncompleteSolutionSet(format!("real univariate representation refused: {message}"))
}

/// Nested kernels consume the same allowance even for an Unbounded caller.
struct Capped<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let mut next = self.used;
        for &(dimension, amount) in charges {
            let (index, limit) = match dimension {
                Dimension::ComputeSteps => (0, MAX_COMPUTE),
                Dimension::MemoryBytes => (1, MAX_MEMORY),
                Dimension::AllocationCount => (2, MAX_COMPUTE),
                _ => continue,
            };
            let remaining = limit.saturating_sub(next[index]);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            next[index] += amount;
        }
        self.inner.charge_batch(charges)?;
        self.used = next;
        Ok(())
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.inner.checkpoint()
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
}

impl<M: BudgetMeter> Work<'_, M> {
    fn charge(&mut self, steps: usize, bytes: usize) -> Result<()> {
        self.meter.checkpoint().map_err(refuse)?;
        self.meter
            .charge_batch(&[
                (Dimension::ComputeSteps, steps.max(1) as u64),
                (Dimension::MemoryBytes, bytes.max(1) as u64),
                (Dimension::AllocationCount, 1),
            ])
            .map_err(refuse)
    }

    fn scalar(&mut self, value: &BigRational) -> Result<()> {
        check_scalar(value)?;
        self.charge(
            1,
            128 + (value.numer().bits() + value.denom().bits()).div_ceil(8) as usize,
        )
    }

    fn copy_scalar(&mut self, value: &BigRational) -> Result<BigRational> {
        self.scalar(value)?;
        Ok(value.clone())
    }

    fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        check_scalar(left)?;
        check_scalar(right)?;
        let value = left.metered_add(right, self.meter).map_err(refuse)?;
        check_scalar(&value)?;
        Ok(value)
    }

    fn subtract(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        check_scalar(left)?;
        check_scalar(right)?;
        let value = left.metered_sub(right, self.meter).map_err(refuse)?;
        check_scalar(&value)?;
        Ok(value)
    }

    fn multiply(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        check_scalar(left)?;
        check_scalar(right)?;
        let value = left.metered_mul(right, self.meter).map_err(refuse)?;
        check_scalar(&value)?;
        Ok(value)
    }

    fn copy_vector(&mut self, values: &[BigRational]) -> Result<Coefficients> {
        check_vector(values)?;
        self.charge(values.len(), values.len() * 64)?;
        values.iter().map(|value| self.copy_scalar(value)).collect()
    }

    fn zeros(&mut self, count: usize) -> Result<Coefficients> {
        if count == 0 || count > 2 * MAX_REAL_ROOT_DEGREE + 1 {
            return Err(refuse("coefficient allocation exceeds its degree bound"));
        }
        self.charge(count, count * 128)?;
        Ok(vec![BigRational::zero(); count])
    }
}

fn check_scalar(value: &BigRational) -> Result<()> {
    if value.numer().bits().max(value.denom().bits()) > MAX_BITS {
        return Err(refuse("coefficient exceeds 4096 bits"));
    }
    Ok(())
}

fn check_vector(values: &[BigRational]) -> Result<()> {
    if values.is_empty()
        || values.len() > MAX_REAL_ROOT_DEGREE + 1
        || values.len() > 1 && values.last().is_some_and(Zero::is_zero)
    {
        return Err(refuse("noncanonical or oversized coefficient vector"));
    }
    for value in values {
        check_scalar(value)?;
    }
    Ok(())
}

fn ring_bytes(variables: &[Symbol]) -> usize {
    variables.iter().map(|symbol| symbol.name.len() + 128).sum()
}

fn validate_poly(
    polynomial: &MultivariatePoly,
    variables: &[Symbol],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.charge(1, 1)?;
    if variables.is_empty()
        || variables.len() > MAX_VARIABLES
        || variables
            .iter()
            .try_fold(0usize, |total, variable| {
                total.checked_add(variable.name.len())
            })
            .is_none_or(|bytes| bytes > 65_536)
        || variables
            .iter()
            .enumerate()
            .any(|(index, variable)| variables[..index].contains(variable))
        || polynomial.generators() != variables
        || polynomial.terms().len() > MAX_TERMS
    {
        return Err(refuse("polynomial dimensions or typed ring mismatch"));
    }
    for (powers, coefficient) in polynomial.terms() {
        work.charge(powers.len() + 1, 1)?;
        if powers.len() != variables.len()
            || powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE
            || coefficient.is_zero()
        {
            return Err(refuse("noncanonical or oversized polynomial monomial"));
        }
        check_scalar(coefficient)?;
    }
    Ok(())
}

fn fresh_parameter(variables: &[Symbol], work: &mut Work<'_, impl BudgetMeter>) -> Result<Symbol> {
    for index in 0..=MAX_VARIABLES {
        work.charge(variables.len() + 1, 256)?;
        let name = format!("_fsym_rur_parameter_{index}");
        if variables.iter().all(|variable| variable.name != name) {
            return Ok(Symbol::new(name));
        }
    }
    Err(refuse("could not choose a fresh separating parameter"))
}

fn coefficients(
    value: i64,
    width: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    work.charge(width, width * 128)?;
    (1..width)
        .map(|power| {
            let coefficient = value
                .checked_pow(power as u32)
                .ok_or_else(|| refuse("separating coefficient overflow"))?;
            Ok(BigRational::from(coefficient))
        })
        .collect()
}

/// Generator lane: build an unexpanded, held expression for the forward
/// substitution and let the existing bounded converter compute its polynomial.
/// Reference checking below never calls this routine or that converter.
fn transformed_inputs(
    inputs: &[MultivariatePoly],
    variables: &[Symbol],
    parameter: &Symbol,
    shear: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>> {
    let width = variables.len();
    let last = width - 1;
    work.charge(
        width + inputs.len(),
        2 * ring_bytes(variables) + inputs.len() * 128 + 512,
    )?;
    let mut ring = variables[..last].to_vec();
    ring.push(parameter.clone());
    let mut linear = Vec::with_capacity(width);
    linear.push(Expr::Sym(parameter.clone()));
    let mut linear_nodes = 2;
    for (variable, coefficient) in variables.iter().zip(shear) {
        if !coefficient.is_zero() {
            work.scalar(coefficient)?;
            linear.push(Expr::Mul(vec![
                Expr::Rational(-coefficient.clone()),
                Expr::Sym(variable.clone()),
            ]));
            linear_nodes += 3;
        }
    }
    let linear = Expr::Add(linear);
    let mut result = Vec::with_capacity(inputs.len());
    for polynomial in inputs {
        let mut nodes = 1usize;
        for powers in polynomial.terms().keys() {
            nodes += 2;
            for (index, &power) in powers.iter().enumerate() {
                if power != 0 {
                    nodes += if index == last { linear_nodes } else { 1 };
                    if power != 1 {
                        nodes += 2;
                    }
                }
            }
            if nodes > MAX_SOURCE_NODES {
                return Err(refuse("transformed source exceeds 4096 projected nodes"));
            }
        }
        work.charge(nodes, nodes * (1024 + ring_bytes(&ring)))?;
        let mut terms = Vec::with_capacity(polynomial.terms().len());
        for (powers, coefficient) in polynomial.terms() {
            let mut factors = Vec::with_capacity(width + 1);
            factors.push(Expr::Rational(coefficient.clone()));
            for (index, &power) in powers.iter().enumerate() {
                if power == 0 {
                    continue;
                }
                let base = if index == last {
                    linear.clone()
                } else {
                    Expr::Sym(variables[index].clone())
                };
                factors.push(if power == 1 {
                    base
                } else {
                    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(i64::from(power))))
                });
            }
            terms.push(Expr::Mul(factors));
        }
        result.push(
            bounded_polynomial_from_expr(&Expr::Add(terms), &ring, work.meter).map_err(refuse)?,
        );
    }
    Ok(result)
}

fn accumulate(
    terms: &mut Terms,
    powers: Vec<u32>,
    value: BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.charge(powers.len() + 1, 1)?;
    if powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE {
        return Err(refuse("inverse substitution exceeds degree 256"));
    }
    let value = if let Some(previous) = terms.get(&powers) {
        work.add(previous, &value)?
    } else {
        value
    };
    if value.is_zero() {
        terms.remove(&powers);
    } else {
        if !terms.contains_key(&powers) {
            if terms.len() == MAX_TERMS {
                return Err(refuse("inverse substitution exceeds 4096 terms"));
            }
            work.charge(1, 256 + powers.len() * 4)?;
        }
        terms.insert(powers, value);
    }
    Ok(())
}

/// Reference lane: direct repeated coefficient convolution with
/// `t = x_last + sum(c_i*x_i)`. This uses neither expression substitution,
/// source conversion, generator polynomial multiplication nor Groebner search.
fn inverse_substitution(
    transformed: &MultivariatePoly,
    variables: &[Symbol],
    shear: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly> {
    let width = variables.len();
    let last = width - 1;
    work.charge(1, ring_bytes(variables) + 256)?;
    let mut result = Terms::new();
    for (powers, coefficient) in transformed.terms() {
        work.charge(width, width * 4 + 256)?;
        let mut initial = powers.clone();
        initial[last] = 0;
        let mut term = Terms::from([(initial, work.copy_scalar(coefficient)?)]);
        for _ in 0..powers[last] {
            let mut next = Terms::new();
            for (powers, value) in &term {
                for index in 0..width {
                    let factor = if index == last {
                        BigRational::one()
                    } else {
                        work.copy_scalar(&shear[index])?
                    };
                    if factor.is_zero() {
                        continue;
                    }
                    work.charge(width, width * 4 + 256)?;
                    let mut shifted = powers.clone();
                    shifted[index] = shifted[index]
                        .checked_add(1)
                        .ok_or_else(|| refuse("inverse substitution exponent overflow"))?;
                    let value = work.multiply(value, &factor)?;
                    accumulate(&mut next, shifted, value, work)?;
                }
            }
            term = next;
        }
        for (powers, value) in term {
            accumulate(&mut result, powers, value, work)?;
        }
    }
    // The convolution already checked canonical nonzero coefficients, term
    // count, exponent width and degree. Move its charged tree directly;
    // calling the normalizing constructor would allocate another map.
    Ok(MultivariatePoly {
        generators: variables.to_vec(),
        terms: result,
    })
}

fn trim(values: &mut Coefficients) {
    while values.len() > 1 && values.last().is_some_and(Zero::is_zero) {
        values.pop();
    }
}

fn is_zero(values: &[BigRational]) -> bool {
    values.len() == 1 && values[0].is_zero()
}

fn remainder(
    mut values: Coefficients,
    modulus: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    if !modulus.last().is_some_and(One::is_one) {
        return Err(refuse("reference reduction requires a monic eliminant"));
    }
    if modulus.len() == 1 {
        return work.zeros(1);
    }
    trim(&mut values);
    while values.len() >= modulus.len() && !is_zero(&values) {
        let shift = values.len() - modulus.len();
        let coefficient = work.copy_scalar(values.last().unwrap())?;
        for (index, divisor) in modulus.iter().enumerate() {
            let product = work.multiply(&coefficient, divisor)?;
            values[index + shift] = work.subtract(&values[index + shift], &product)?;
        }
        trim(&mut values);
    }
    Ok(values)
}

fn product_modulo(
    left: &[BigRational],
    right: &[BigRational],
    modulus: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    let mut values = work.zeros(left.len() + right.len() - 1)?;
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            let product = work.multiply(a, b)?;
            values[i + j] = work.add(&values[i + j], &product)?;
        }
    }
    remainder(values, modulus, work)
}

fn power_modulo(
    value: &[BigRational],
    mut exponent: u32,
    modulus: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    let mut result = work.zeros(1)?;
    result[0] = BigRational::one();
    let mut base = remainder(work.copy_vector(value)?, modulus, work)?;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = product_modulo(&result, &base, modulus, work)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = product_modulo(&base, &base, modulus, work)?;
        }
    }
    Ok(result)
}

fn check_residuals(
    inputs: &[MultivariatePoly],
    coordinates: &[Coefficients],
    eliminant: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    for input in inputs {
        let mut residual = work.zeros(eliminant.len())?;
        for (powers, coefficient) in input.terms() {
            work.charge(1, 128)?;
            let mut term = vec![work.copy_scalar(coefficient)?];
            for (index, &power) in powers.iter().enumerate() {
                if power != 0 {
                    let factor = power_modulo(&coordinates[index], power, eliminant, work)?;
                    term = product_modulo(&term, &factor, eliminant, work)?;
                }
            }
            for (index, coefficient) in term.iter().enumerate() {
                residual[index] = work.add(&residual[index], coefficient)?;
            }
        }
        let residual = remainder(residual, eliminant, work)?;
        if !is_zero(&residual) {
            return Err(refuse(
                "an original equation has a nonzero coordinate residual",
            ));
        }
    }
    Ok(())
}

fn recover_coordinates(
    shape: &VerifiedRealShapeSystem,
    shear: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<Coefficients>> {
    let width = shape.variables().len();
    work.charge(width, width * 64)?;
    let mut coordinates = Vec::with_capacity(width);
    for coordinate in &shape.coordinates()[..width - 1] {
        coordinates.push(work.copy_vector(coordinate)?);
    }
    let count = shape
        .coordinates()
        .iter()
        .map(Vec::len)
        .max()
        .unwrap()
        .max(2);
    let mut last = work.zeros(count)?;
    last[1] = BigRational::one();
    for (coefficient, coordinate) in shear.iter().zip(&coordinates) {
        for (index, value) in coordinate.iter().enumerate() {
            let product = work.multiply(coefficient, value)?;
            last[index] = work.subtract(&last[index], &product)?;
        }
    }
    trim(&mut last);
    coordinates.push(last);
    Ok(coordinates)
}

/// Rebind the accepted transformed ideal to the exact original source and
/// verify the proposed original-coordinate vectors independently.
fn verify_binding(
    inputs: &[MultivariatePoly],
    variables: &[Symbol],
    parameter: &Symbol,
    shear: &[BigRational],
    coordinates: &[Coefficients],
    representation: &CheckedShape,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.charge(1, 1)?;
    let shape = &representation.shape;
    let binding_inputs = representation.transformed_basis().input_basis();
    if variables.is_empty()
        || variables.len() > MAX_VARIABLES
        || inputs.is_empty()
        || inputs.len() > MAX_EQUATIONS
        || shear.len() + 1 != variables.len()
        || coordinates.len() != variables.len()
        || shape.variables().len() != variables.len()
        || binding_inputs.len() != inputs.len()
    {
        return Err(refuse(
            "representation dimensions differ from the original system",
        ));
    }
    if let Some(intermediate) = &representation.intermediate {
        let terms: usize = intermediate
            .basis()
            .iter()
            .map(|poly| poly.terms().len())
            .sum();
        work.charge(terms * (variables.len() + 128) + ring_bytes(variables), 1)?;
        if intermediate.order() != TermOrder::DegRevLex
            || shape.basis().input_basis() != intermediate.basis()
        {
            return Err(refuse(
                "shape order-conversion receipt lost its source ideal",
            ));
        }
    }
    let last = variables.len() - 1;
    if shape.variables()[..last] != variables[..last]
        || shape.variables()[last] != *parameter
        || variables.contains(parameter)
            && (parameter != &variables[last] || shear.iter().any(|value| !value.is_zero()))
        || shear
            .iter()
            .any(|value| !value.denom().is_one() || value.numer().bits() > 16)
    {
        return Err(refuse("invalid or captured separating transform"));
    }
    for (input, transformed) in inputs.iter().zip(binding_inputs) {
        validate_poly(input, variables, work)?;
        validate_poly(transformed, shape.variables(), work)?;
        if inverse_substitution(transformed, variables, shear, work)? != *input {
            return Err(refuse(
                "inverse transform does not recover an original equation",
            ));
        }
    }
    check_vector(shape.eliminant())?;
    for coordinate in coordinates {
        check_vector(coordinate)?;
    }
    if coordinates[..last] != shape.coordinates()[..last] {
        return Err(refuse(
            "a transformed pivot coordinate was changed or reordered",
        ));
    }
    // The forward linear form evaluated at the recovered coordinates must
    // equal its parameter in the exact quotient. This differs from the
    // generator's subtraction recurrence and also covers degree-one ideals.
    let count = coordinates.iter().map(Vec::len).max().unwrap().max(2);
    let mut relation = work.zeros(count)?;
    for (index, value) in coordinates[last].iter().enumerate() {
        relation[index] = work.copy_scalar(value)?;
    }
    for (coefficient, coordinate) in shear.iter().zip(&coordinates[..last]) {
        for (index, value) in coordinate.iter().enumerate() {
            let product = work.multiply(coefficient, value)?;
            relation[index] = work.add(&relation[index], &product)?;
        }
    }
    relation[1] = work.subtract(&relation[1], &BigRational::one())?;
    if !is_zero(&remainder(relation, shape.eliminant(), work)?) {
        return Err(refuse(
            "original coordinates do not recover the separating form",
        ));
    }
    check_residuals(inputs, coordinates, shape.eliminant(), work)?;
    work.meter.checkpoint().map_err(refuse)
}

fn accept(
    inputs: Vec<MultivariatePoly>,
    variables: &[Symbol],
    parameter: Symbol,
    shear: Coefficients,
    representation: CheckedShape,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRealRurSystem> {
    let coordinates = recover_coordinates(
        &representation.shape,
        &shear,
        &mut Work { meter: generator },
    )?;
    verify_binding(
        &inputs,
        variables,
        &parameter,
        &shear,
        &coordinates,
        &representation,
        &mut Work { meter: verifier },
    )?;
    Work { meter: generator }.charge(variables.len(), ring_bytes(variables))?;
    let result = VerifiedRealRurSystem {
        variables: variables.to_vec(),
        equations: inputs,
        parameter,
        shear,
        coordinates,
        representation,
    };
    generator.checkpoint().map_err(refuse)?;
    verifier.checkpoint().map_err(refuse)?;
    Ok(result)
}

fn checked_basis<G: BudgetMeter, V: BudgetMeter>(
    inputs: &[MultivariatePoly],
    order: TermOrder,
    phase: &str,
    generator: &mut Capped<'_, G>,
    verifier: &mut Capped<'_, V>,
) -> Result<VerifiedGroebnerBasis> {
    let start = generator.used[0];
    let candidate = bounded_groebner_certificate(inputs, order, generator).map_err(|error| {
        refuse(format!(
            "{phase} generation after {} compute units ({} cumulative): {error}",
            generator.used[0] - start,
            generator.used[0],
        ))
    })?;
    let start = verifier.used[0];
    let accepted =
        verify_complete_groebner_certificate(inputs, &candidate, verifier).map_err(|error| {
            refuse(format!(
                "{phase} verification after {} compute units ({} cumulative): {error}",
                verifier.used[0] - start,
                verifier.used[0],
            ))
        })?;
    generator.checkpoint().map_err(refuse)?;
    verifier.checkpoint().map_err(refuse)?;
    Ok(accepted)
}

/// Recognize a complete real rational univariate representation using the
/// original shape order followed by seven small deterministic shears.
///
/// Every original expression is admitted before any unit-ideal, dimension or
/// shape decision. Unsupported recognition returns `None`; resource and
/// verification errors propagate and never restart either caller meter.
pub fn metered_solve_real_rur_system(
    equations: &[Expr],
    variables: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealRurSystem>> {
    let mut generator = Capped {
        inner: generator_meter,
        used: [0; 3],
    };
    let mut verifier = Capped {
        inner: verifier_meter,
        used: [0; 3],
    };
    generator.checkpoint().map_err(refuse)?;
    if variables.is_empty() || variables.len() > MAX_VARIABLES || equations.len() > MAX_EQUATIONS {
        return Err(SolverError::InvalidSystem(
            "real univariate systems require 1..8 variables and at most 64 equations".into(),
        ));
    }
    // This bounds names and validates duplicate typed generators even when
    // there are no equations. Every original is converted before dispatch.
    bounded_polynomial_from_expr(&Expr::from_i64(0), variables, &mut generator).map_err(refuse)?;
    Work {
        meter: &mut generator,
    }
    .charge(equations.len(), equations.len() * 128)?;
    let inputs = equations
        .iter()
        .map(|equation| {
            bounded_polynomial_from_expr(equation, variables, &mut generator).map_err(refuse)
        })
        .collect::<Result<Vec<_>>>()?;
    if inputs.is_empty() {
        generator.checkpoint().map_err(refuse)?;
        verifier.checkpoint().map_err(refuse)?;
        return Ok(None);
    }
    let basis = checked_basis(
        &inputs,
        TermOrder::Lex,
        "original lex",
        &mut generator,
        &mut verifier,
    )?;
    let finite = basis.is_zero_dimensional();
    if let Some(shape) = metered_real_shape_from_basis(basis, &mut generator, &mut verifier)? {
        let shear = coefficients(
            0,
            variables.len(),
            &mut Work {
                meter: &mut generator,
            },
        )?;
        Work {
            meter: &mut generator,
        }
        .charge(1, variables.last().unwrap().name.len() + 128)?;
        return accept(
            inputs,
            variables,
            variables.last().unwrap().clone(),
            shear,
            CheckedShape {
                shape,
                intermediate: None,
            },
            &mut generator,
            &mut verifier,
        )
        .map(Some);
    }
    if !finite {
        generator.checkpoint().map_err(refuse)?;
        verifier.checkpoint().map_err(refuse)?;
        return Ok(None);
    }
    let parameter = fresh_parameter(
        variables,
        &mut Work {
            meter: &mut generator,
        },
    )?;
    for value in SHEAR_VALUES {
        let shear = coefficients(
            value,
            variables.len(),
            &mut Work {
                meter: &mut generator,
            },
        )?;
        let transformed = transformed_inputs(
            &inputs,
            variables,
            &parameter,
            &shear,
            &mut Work {
                meter: &mut generator,
            },
        )?;
        let intermediate = checked_basis(
            &transformed,
            TermOrder::DegRevLex,
            &format!("shear {value} degree-reverse-lex"),
            &mut generator,
            &mut verifier,
        )?;
        let (generator_start, verifier_start) = (generator.used[0], verifier.used[0]);
        let basis = intermediate
            .fglm(TermOrder::Lex, &mut generator, &mut verifier)
            .map_err(|error| {
                refuse(format!(
                    "shear {value} FGLM after {} generator / {} verifier compute units: {error}",
                    generator.used[0] - generator_start,
                    verifier.used[0] - verifier_start,
                ))
            })?;
        if let Some(shape) = metered_real_shape_from_basis(basis, &mut generator, &mut verifier)? {
            return accept(
                inputs,
                variables,
                parameter,
                shear,
                CheckedShape {
                    shape,
                    intermediate: Some(intermediate),
                },
                &mut generator,
                &mut verifier,
            )
            .map(Some);
        }
    }
    generator.checkpoint().map_err(refuse)?;
    verifier.checkpoint().map_err(refuse)?;
    Ok(None)
}

pub fn solve_real_rur_system(
    equations: &[Expr],
    variables: &[Symbol],
) -> Result<Option<VerifiedRealRurSystem>> {
    metered_solve_real_rur_system(
        equations,
        variables,
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
    )
}

#[cfg(test)]
mod reference_checks {
    use super::*;
    use fsym_budget::Unbounded;
    use fsym_core::{BigInt, SymbolIdentity, parse};

    fn fixture(sources: &[&str]) -> VerifiedRealRurSystem {
        solve_real_rur_system(
            &sources
                .iter()
                .map(|source| parse(source).unwrap())
                .collect::<Vec<_>>(),
            &[Symbol::new("x"), Symbol::new("y")],
        )
        .unwrap()
        .unwrap()
    }

    fn check(
        result: &VerifiedRealRurSystem,
        inputs: &[MultivariatePoly],
        variables: &[Symbol],
        parameter: &Symbol,
        shear: &[BigRational],
        coordinates: &[Coefficients],
    ) -> Result<()> {
        verify_binding(
            inputs,
            variables,
            parameter,
            shear,
            coordinates,
            &result.representation,
            &mut Work {
                meter: &mut Unbounded,
            },
        )
    }

    #[test]
    fn inverse_reference_rejects_changed_missing_or_rebound_original_inputs() {
        let result = fixture(&["x**2-2", "y**2-3"]);
        check(
            &result,
            result.equations(),
            result.variables(),
            result.parameter(),
            result.shear(),
            result.coordinates(),
        )
        .unwrap();
        let mut changed = result.equations().to_vec();
        *changed[0].terms.get_mut(&vec![0, 0]).unwrap() += BigRational::one();
        assert!(
            check(
                &result,
                &changed,
                result.variables(),
                result.parameter(),
                result.shear(),
                result.coordinates(),
            )
            .is_err()
        );
        assert!(
            check(
                &result,
                &result.equations()[..1],
                result.variables(),
                result.parameter(),
                result.shear(),
                result.coordinates(),
            )
            .is_err()
        );
        let mut changed_variables = result.variables().to_vec();
        changed_variables[0] =
            Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([57; 32]));
        assert!(
            check(
                &result,
                result.equations(),
                &changed_variables,
                result.parameter(),
                result.shear(),
                result.coordinates(),
            )
            .is_err()
        );
        let unrelated = fixture(&["x**2-5", "y**2-3"]);
        assert!(
            check(
                &unrelated,
                result.equations(),
                result.variables(),
                unrelated.parameter(),
                unrelated.shear(),
                unrelated.coordinates(),
            )
            .is_err()
        );
        let mut missing_link = result.clone();
        missing_link.representation.intermediate = None;
        assert!(
            check(
                &missing_link,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                result.coordinates(),
            )
            .is_err()
        );
        let mut changed_link = result.clone();
        changed_link.representation.intermediate = unrelated.representation.intermediate.clone();
        assert!(
            check(
                &changed_link,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                result.coordinates(),
            )
            .is_err()
        );
        let mut circular_link = result.clone();
        circular_link.representation.intermediate = Some(result.basis().clone());
        assert!(
            check(
                &circular_link,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                result.coordinates(),
            )
            .is_err()
        );
        // Equal reduced intermediate ideals do not establish literal source
        // binding. Replacing just the receipt with a differently scaled source
        // must still fail inverse recovery of the original input equations.
        let scaled = fixture(&["2*x**2-4", "y**2-3"]);
        assert_eq!(
            scaled.transformed_basis().basis(),
            result.transformed_basis().basis()
        );
        let mut rebound_source = result.clone();
        rebound_source.representation.intermediate = scaled.representation.intermediate;
        assert!(
            check(
                &rebound_source,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                result.coordinates(),
            )
            .is_err()
        );
    }

    #[test]
    fn reference_rejects_forged_shear_capture_and_coordinate_recovery() {
        let result = fixture(&["x**2-2", "y**2-3"]);
        for shear in [
            vec![BigRational::from(-1_i64)],
            vec![],
            vec![BigRational::new(1.into(), 2.into())],
            vec![BigRational::from_integer(BigInt::one() << 20)],
        ] {
            assert!(
                check(
                    &result,
                    result.equations(),
                    result.variables(),
                    result.parameter(),
                    &shear,
                    result.coordinates(),
                )
                .is_err()
            );
        }
        for parameter in [&result.variables()[0], &Symbol::new("unrelated_parameter")] {
            assert!(
                check(
                    &result,
                    result.equations(),
                    result.variables(),
                    parameter,
                    result.shear(),
                    result.coordinates(),
                )
                .is_err()
            );
        }
        let tiny = BigRational::new(1.into(), BigInt::from(10).pow(80));
        for coordinate in 0..2 {
            let mut changed = result.coordinates().to_vec();
            changed[coordinate][0] += &tiny;
            assert!(
                check(
                    &result,
                    result.equations(),
                    result.variables(),
                    result.parameter(),
                    result.shear(),
                    &changed,
                )
                .is_err()
            );
        }
        let mut swapped = result.coordinates().to_vec();
        swapped.swap(0, 1);
        assert!(
            check(
                &result,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                &swapped,
            )
            .is_err()
        );
        let mut padded = result.coordinates().to_vec();
        padded[1].push(BigRational::zero());
        assert!(
            check(
                &result,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                &padded,
            )
            .is_err()
        );
    }

    #[test]
    fn a_valid_point_is_not_a_complete_solution_representation() {
        let result = fixture(&["x**2-x", "y**2-y"]);
        let one_point = vec![vec![BigRational::zero()]; 2];
        // (0,0) satisfies every original equation. It still cannot replace
        // the four-point representation because ideal/coordinate binding is
        // checked independently of these residual identities.
        check_residuals(
            result.equations(),
            &one_point,
            result.eliminant(),
            &mut Work {
                meter: &mut Unbounded,
            },
        )
        .unwrap();
        assert!(
            check(
                &result,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                &one_point,
            )
            .is_err()
        );
        assert!(
            check(
                &result,
                result.equations(),
                result.variables(),
                result.parameter(),
                result.shear(),
                &result.coordinates()[..1],
            )
            .is_err()
        );
    }
}
