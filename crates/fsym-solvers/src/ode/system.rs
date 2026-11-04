//! Complete homogeneous two-component ODE systems over the rationals.
//!
//! The generator uses exact spectral projectors for nonnilpotent rank-one
//! matrices and the trace/discriminant formula otherwise for exp(A*(t-t0)).
//! Acceptance does not repeat that formula: it differentiates every proposed
//! matrix entry, checks Phi' = A*Phi, and separately checks Phi(t0) = I.
//! Constant rational coefficients are smooth on the real line. The checked
//! normalized fundamental matrix therefore gives every solution, and a unique
//! solution for each exact rational initial vector. This is an in-process
//! receipt, not a portable proof-kernel certificate.
//!
//! Recursive differentiation and simplification are bounded synchronous calls.
//! Work is prepaid and cancellation is checked between entries and before
//! publication; no forced interruption inside a simplifier call is claimed.

use super::{ode_expression_is_defined, residual_is_exact_zero};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::diff;
use fsym_core::arith::RationalArithmeticError;
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_simplify::try_simplify;
use num_traits::{One, Signed, Zero};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Independently checked polynomial forcing on the complete system family.
pub mod polynomial;

pub type RationalMatrix2 = [[BigRational; 2]; 2];
pub type ExpressionMatrix2 = [[Expr; 2]; 2];

const MAX_INPUT_BITS: u64 = 1024;
const MAX_DERIVED_BITS: u64 = 16_384;
const MAX_NODES: usize = 256;
const MAX_DEPTH: usize = 32;
const MAX_TEXT: usize = 16_384;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LinearOdeSystemError {
    #[error("invalid linear ODE system: {0}")]
    InvalidSource(&'static str),
    #[error("linear ODE system resource limit: {0}")]
    ResourceLimit(&'static str),
    #[error("linear ODE system verification failed: {0}")]
    Verification(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for LinearOdeSystemError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, LinearOdeSystemError>;

/// An untrusted matrix whose columns are proposed independent solutions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearOdeSystemCertificate {
    pub fundamental: ExpressionMatrix2,
}

/// A normalized fundamental matrix on the complete real time axis.
#[derive(Debug, Clone)]
pub struct VerifiedLinearOdeFundamental {
    coefficients: RationalMatrix2,
    time: Symbol,
    initial_point: BigRational,
    fundamental: ExpressionMatrix2,
}

impl VerifiedLinearOdeFundamental {
    pub fn coefficients(&self) -> &RationalMatrix2 {
        &self.coefficients
    }
    pub fn time(&self) -> &Symbol {
        &self.time
    }
    pub fn initial_point(&self) -> &BigRational {
        &self.initial_point
    }
    pub fn fundamental(&self) -> &ExpressionMatrix2 {
        &self.fundamental
    }
}

/// A complete parameterized family, or the unique solution of rational data.
/// Initial entries are exact rationals or distinct fresh constant symbols.
#[derive(Debug, Clone)]
pub struct VerifiedLinearOdeSystem {
    fundamental: VerifiedLinearOdeFundamental,
    initial_values: [Expr; 2],
    solutions: [Expr; 2],
}

impl VerifiedLinearOdeSystem {
    pub fn fundamental(&self) -> &VerifiedLinearOdeFundamental {
        &self.fundamental
    }
    pub fn initial_values(&self) -> &[Expr; 2] {
        &self.initial_values
    }
    pub fn solutions(&self) -> &[Expr; 2] {
        &self.solutions
    }
    pub fn into_solutions(self) -> [Expr; 2] {
        self.solutions
    }
}

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

fn charge(meter: &mut impl BudgetMeter, steps: usize, bytes: usize) -> Result<()> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, steps.max(1) as u64),
        (Dimension::MemoryBytes, bytes.max(1) as u64),
        (Dimension::AllocationCount, steps.max(1) as u64),
    ])?;
    Ok(())
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate_number(value: &BigRational, maximum: u64) -> Result<()> {
    if !value.denom().is_positive() || bits(value) > maximum {
        return Err(LinearOdeSystemError::ResourceLimit(
            "rational coefficient height or denominator",
        ));
    }
    Ok(())
}

fn scalar(value: BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    validate_number(&value, MAX_DERIVED_BITS)?;
    charge(meter, 1, 128 + 2 * bits(&value).div_ceil(8) as usize)?;
    Ok(value)
}

fn admit_inputs(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    charge(meter, 1, 256)?;
    if time.name.len() > MAX_TEXT {
        return Err(LinearOdeSystemError::ResourceLimit("time symbol text"));
    }
    validate_number(point, MAX_INPUT_BITS)?;
    for value in matrix.iter().flatten() {
        validate_number(value, MAX_INPUT_BITS)?;
    }
    charge(
        meter,
        6,
        1024 + time.name.len()
            + matrix
                .iter()
                .flatten()
                .map(|q| 2 * bits(q).div_ceil(8) as usize)
                .sum::<usize>(),
    )
}

/// Borrowed admission precedes differentiation, substitution and expansion.
fn prepay_expression(
    expression: &Expr,
    allowed: &[&Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let mut pending = vec![(expression, 0usize)];
    let (mut nodes, mut text, mut height) = (0usize, 0usize, 1u64);
    while let Some((node, depth)) = pending.pop() {
        nodes += 1;
        if nodes > MAX_NODES || depth > MAX_DEPTH {
            return Err(LinearOdeSystemError::ResourceLimit(
                "candidate expression shape",
            ));
        }
        match node {
            Expr::Integer(value) => height = height.max(value.bits()),
            Expr::Rational(value) => {
                validate_number(value, MAX_DERIVED_BITS)?;
                height = height.max(bits(value));
            }
            Expr::Sym(symbol) => {
                if !allowed.contains(&symbol) {
                    return Err(LinearOdeSystemError::Verification(
                        "foreign or rebound symbol",
                    ));
                }
                text = text.saturating_add(symbol.name.len());
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if children.is_empty()
                    || children.len() > 32
                    || nodes + pending.len() + children.len() > MAX_NODES
                {
                    return Err(LinearOdeSystemError::ResourceLimit("candidate fanout"));
                }
                if let Expr::Function(name, _) = node {
                    if children.len() != 1
                        || !matches!(name.as_str(), "exp" | "sin" | "cos" | "sinh" | "cosh")
                    {
                        return Err(LinearOdeSystemError::Verification(
                            "unsupported candidate function",
                        ));
                    }
                    text = text.saturating_add(name.len());
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                let rational = match exponent.as_ref() {
                    Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => {
                        BigRational::from_integer(value.clone())
                    }
                    Expr::Rational(value) if bits(value) <= MAX_INPUT_BITS => value.clone(),
                    _ => {
                        return Err(LinearOdeSystemError::Verification(
                            "nonliteral candidate power",
                        ));
                    }
                };
                validate_number(&rational, MAX_INPUT_BITS)?;
                // Generated powers are positive literal square roots. Permit
                // small integer powers too, without an unbounded expansion.
                if rational != BigRational::new(1.into(), 2.into())
                    && rational != BigRational::one()
                    && rational != BigRational::from(2)
                {
                    return Err(LinearOdeSystemError::Verification(
                        "unsupported candidate power",
                    ));
                }
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Const(_) => {
                return Err(LinearOdeSystemError::Verification(
                    "unsupported candidate constant",
                ));
            }
        }
        if text > MAX_TEXT || height > MAX_DERIVED_BITS {
            return Err(LinearOdeSystemError::ResourceLimit(
                "candidate scalar or text growth",
            ));
        }
    }
    if !ode_expression_is_defined(expression, None) {
        return Err(LinearOdeSystemError::Verification(
            "candidate is not defined on the real time axis",
        ));
    }
    charge(
        meter,
        (nodes + 1) * (nodes + 1) * 64,
        (nodes + 32) * (height as usize + 128) * 64,
    )
}

fn sum(terms: Vec<Expr>) -> Expr {
    let terms: Vec<_> = terms.into_iter().filter(|term| !term.is_zero()).collect();
    match terms.len() {
        0 => Expr::from_i64(0),
        1 => terms.into_iter().next().expect("one term"),
        _ => Expr::Add(terms),
    }
}

fn product(factors: Vec<Expr>) -> Expr {
    if factors.iter().any(Expr::is_zero) {
        return Expr::from_i64(0);
    }
    let factors: Vec<_> = factors
        .into_iter()
        .filter(|factor| *factor != Expr::from_i64(1))
        .collect();
    match factors.len() {
        0 => Expr::from_i64(1),
        1 => factors.into_iter().next().expect("one factor"),
        _ => Expr::Mul(factors),
    }
}

fn normalize(expression: Expr, allowed: &[&Symbol], meter: &mut impl BudgetMeter) -> Result<Expr> {
    prepay_expression(&expression, allowed, meter)?;
    let result = try_simplify(&expression)
        .map_err(|_| LinearOdeSystemError::ResourceLimit("expression simplification"))?;
    prepay_expression(&result, allowed, meter)?;
    Ok(result)
}

fn generate(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<LinearOdeSystemCertificate> {
    admit_inputs(matrix, time, point, meter)?;
    // Prepay the fixed-width expression assembly and its typed-symbol copies.
    // Recursive radical/function normalization is prepaid separately below.
    charge(meter, 128, 65_536 + 32 * time.name.len())?;
    let trace = scalar(matrix[0][0].metered_add(&matrix[1][1], meter)?, meter)?;
    let diagonal = matrix[0][0].metered_mul(&matrix[1][1], meter)?;
    let cross = matrix[0][1].metered_mul(&matrix[1][0], meter)?;
    let determinant = scalar(diagonal.metered_sub(&cross, meter)?, meter)?;
    let offset = scalar(BigRational::zero().metered_sub(point, meter)?, meter)?;
    if determinant.is_zero() && !trace.is_zero() {
        // Cayley-Hamilton gives A^2 = trace(A)*A. Thus A/trace(A) and
        // I-A/trace(A) project onto the two distinct eigenspaces. Keeping the
        // zero mode exact avoids exp(u/2)*(cosh(u/2)-sinh(u/2)) in the output,
        // which otherwise obscures constant coordinates and forced IVPs.
        // The independent checker still differentiates every entry and
        // checks the full normalized initial matrix; this is only a proposal.
        let scaled_offset = scalar(trace.metered_mul(&offset, meter)?, meter)?;
        let exponential = Expr::Function(
            "exp".into(),
            vec![sum(vec![
                product(vec![rational_expr(trace.clone()), Expr::Sym(time.clone())]),
                rational_expr(scaled_offset),
            ])],
        );
        let increment = sum(vec![exponential, Expr::from_i64(-1)]);
        let mut fundamental = std::array::from_fn(|_| std::array::from_fn(|_| Expr::from_i64(0)));
        for row in 0..2 {
            for column in 0..2 {
                let coefficient = scalar(matrix[row][column].metered_div(&trace, meter)?, meter)?;
                fundamental[row][column] = normalize(
                    sum(vec![
                        Expr::from_i64(i64::from(row == column)),
                        product(vec![rational_expr(coefficient), increment.clone()]),
                    ]),
                    &[time],
                    meter,
                )?;
            }
        }
        meter.checkpoint()?;
        return Ok(LinearOdeSystemCertificate { fundamental });
    }
    let u = sum(vec![Expr::Sym(time.clone()), rational_expr(offset)]);
    let two = BigRational::from(2);
    let center = scalar(trace.metered_div(&two, meter)?, meter)?;
    let difference = scalar(matrix[0][0].metered_sub(&center, meter)?, meter)?;
    let square = difference.metered_mul(&difference, meter)?;
    let discriminant = scalar(square.metered_add(&cross, meter)?, meter)?;
    let exponential = if center.is_zero() {
        Expr::from_i64(1)
    } else {
        Expr::Function(
            "exp".into(),
            vec![product(vec![rational_expr(center.clone()), u.clone()])],
        )
    };
    let (c, s) = if discriminant.is_zero() {
        (Expr::from_i64(1), u)
    } else {
        let magnitude = scalar(discriminant.abs(), meter)?;
        let frequency = Expr::Pow(
            Arc::new(rational_expr(magnitude.clone())),
            Arc::new(rational_expr(BigRational::new(1.into(), 2.into()))),
        );
        let reciprocal = scalar(BigRational::one().metered_div(&magnitude, meter)?, meter)?;
        let argument = product(vec![frequency.clone(), u]);
        let (cosine, sine) = if discriminant.is_positive() {
            ("cosh", "sinh")
        } else {
            ("cos", "sin")
        };
        (
            Expr::Function(cosine.into(), vec![argument.clone()]),
            product(vec![
                rational_expr(reciprocal),
                frequency,
                Expr::Function(sine.into(), vec![argument]),
            ]),
        )
    };
    let mut fundamental = std::array::from_fn(|_| std::array::from_fn(|_| Expr::from_i64(0)));
    for row in 0..2 {
        for column in 0..2 {
            let coefficient = if row == column {
                scalar(matrix[row][column].metered_sub(&center, meter)?, meter)?
            } else {
                scalar(matrix[row][column].clone(), meter)?
            };
            let inner = sum(vec![
                if row == column {
                    c.clone()
                } else {
                    Expr::from_i64(0)
                },
                product(vec![rational_expr(coefficient), s.clone()]),
            ]);
            fundamental[row][column] =
                normalize(product(vec![exponential.clone(), inner]), &[time], meter)?;
        }
    }
    meter.checkpoint()?;
    Ok(LinearOdeSystemCertificate { fundamental })
}

fn verify(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    candidate: LinearOdeSystemCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedLinearOdeFundamental> {
    admit_inputs(matrix, time, point, meter)?;
    for entry in candidate.fundamental.iter().flatten() {
        prepay_expression(entry, &[time], meter)?;
    }
    for (row, coefficients) in matrix.iter().enumerate() {
        for column in 0..2 {
            let entry = &candidate.fundamental[row][column];
            prepay_expression(entry, &[time], meter)?;
            let derivative = diff(entry, time);
            let residual = sum(vec![
                derivative,
                product(vec![
                    Expr::from_i64(-1),
                    sum((0..2)
                        .map(|index| {
                            product(vec![
                                rational_expr(coefficients[index].clone()),
                                candidate.fundamental[index][column].clone(),
                            ])
                        })
                        .collect()),
                ]),
            ]);
            prepay_expression(&residual, &[time], meter)?;
            if !residual_is_exact_zero(&residual) {
                return Err(LinearOdeSystemError::Verification(
                    "fundamental derivative identity",
                ));
            }
            let initial = entry.subs(&HashMap::from([(
                time.clone(),
                rational_expr(point.clone()),
            )]));
            let expected = Expr::from_i64(i64::from(row == column));
            let residual = sum(vec![initial, product(vec![Expr::from_i64(-1), expected])]);
            prepay_expression(&residual, &[], meter)?;
            if !residual_is_exact_zero(&residual) {
                return Err(LinearOdeSystemError::Verification(
                    "fundamental initial identity",
                ));
            }
            meter.checkpoint()?;
        }
    }
    charge(meter, 8, 4096 + 2 * time.name.len())?;
    let result = VerifiedLinearOdeFundamental {
        coefficients: matrix.clone(),
        time: time.clone(),
        initial_point: point.clone(),
        fundamental: candidate.fundamental,
    };
    meter.checkpoint()?;
    Ok(result)
}

/// Independently verify an arbitrary proposed normalized fundamental matrix.
pub fn verify_linear_ode_fundamental(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    candidate: LinearOdeSystemCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedLinearOdeFundamental> {
    let mut meter = Capped {
        inner: verifier,
        used: [0; 3],
    };
    // The claim is on the real time axis, not on a caller's discrete or
    // contradictory assumption subset. Shadow ambient name fallback facts.
    with_symbol_facts(HashMap::from([(time.clone(), Facts::default())]), || {
        verify(matrix, time, point, candidate, &mut meter)
    })
}

pub fn metered_linear_ode_fundamental(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedLinearOdeFundamental> {
    let mut generation = Capped {
        inner: generator,
        used: [0; 3],
    };
    let mut verification = Capped {
        inner: verifier,
        used: [0; 3],
    };
    with_symbol_facts(HashMap::from([(time.clone(), Facts::default())]), || {
        let candidate = generate(matrix, time, point, &mut generation)?;
        let result = verify(matrix, time, point, candidate, &mut verification)?;
        generation.checkpoint()?;
        verification.checkpoint()?;
        Ok(result)
    })
}

pub fn linear_ode_fundamental(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
) -> Result<VerifiedLinearOdeFundamental> {
    metered_linear_ode_fundamental(matrix, time, point, &mut Unbounded, &mut Unbounded)
}

/// Solve with exact rational values and/or fresh arbitrary integration constants.
pub fn metered_linear_ode_system(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    initial: [&Expr; 2],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedLinearOdeSystem> {
    let mut generation = Capped {
        inner: generator,
        used: [0; 3],
    };
    let mut verification = Capped {
        inner: verifier,
        used: [0; 3],
    };
    admit_inputs(matrix, time, point, &mut generation)?;
    let mut parameters = Vec::new();
    for value in initial {
        match value {
            Expr::Integer(n) if n.bits() <= MAX_INPUT_BITS => {}
            Expr::Rational(q) => validate_number(q, MAX_INPUT_BITS)?,
            Expr::Sym(symbol)
                if symbol != time
                    && !parameters.contains(&symbol)
                    && symbol.name.len() <= MAX_TEXT =>
            {
                parameters.push(symbol)
            }
            _ => {
                return Err(LinearOdeSystemError::InvalidSource(
                    "initial values require rationals or distinct fresh constant symbols",
                ));
            }
        }
    }
    charge(
        &mut generation,
        4,
        4096 + parameters
            .iter()
            .map(|symbol| symbol.name.len())
            .sum::<usize>(),
    )?;
    let facts: HashMap<_, _> = std::iter::once(time)
        .chain(parameters.iter().copied())
        .map(|symbol| (symbol.clone(), Facts::default()))
        .collect();
    with_symbol_facts(facts, || {
        let candidate = generate(matrix, time, point, &mut generation)?;
        let fundamental = verify(matrix, time, point, candidate, &mut verification)?;
        let mut allowed = vec![time];
        allowed.extend(parameters.iter().copied());
        let mut solutions = [Expr::from_i64(0), Expr::from_i64(0)];
        for (row, result) in solutions.iter_mut().enumerate() {
            // Construct from the independently accepted matrix. No candidate
            // solution field can change the initial data or coordinate order.
            charge(&mut verification, 8, 4096)?;
            let raw = sum((0..2)
                .map(|column| {
                    product(vec![
                        fundamental.fundamental[row][column].clone(),
                        initial[column].clone(),
                    ])
                })
                .collect());
            prepay_expression(&raw, &allowed, &mut verification)?;
            *result = normalize(raw.clone(), &allowed, &mut verification)?;
            let identity = sum(vec![raw, product(vec![Expr::from_i64(-1), result.clone()])]);
            prepay_expression(&identity, &allowed, &mut verification)?;
            if !residual_is_exact_zero(&identity) {
                return Err(LinearOdeSystemError::Verification(
                    "solution assembly identity",
                ));
            }
        }
        charge(
            &mut verification,
            2,
            4096 + parameters
                .iter()
                .map(|symbol| symbol.name.len())
                .sum::<usize>(),
        )?;
        let result = VerifiedLinearOdeSystem {
            fundamental,
            initial_values: [initial[0].clone(), initial[1].clone()],
            solutions,
        };
        generation.checkpoint()?;
        verification.checkpoint()?;
        Ok(result)
    })
}

pub fn linear_ode_system(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    initial: [&Expr; 2],
) -> Result<VerifiedLinearOdeSystem> {
    metered_linear_ode_system(matrix, time, point, initial, &mut Unbounded, &mut Unbounded)
}
