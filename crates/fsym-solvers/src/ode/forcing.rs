//! Source-bound scalar ODE forcing through the independently checked system lane.
//!
//! The source reader proposes an exact polynomial/exponential/trigonometric
//! decomposition. The existing system generator and coefficient verifier solve
//! the companion system. A separate scalar checker then differentiates the
//! actual returned expression against the original, unnormalized forcing.
//! Consequently a source-conversion mistake cannot certify its own answer.
//!
//! Generator and verifier have separate cumulative meters. Recursive symbolic
//! checks are bounded and prepaid, with cancellation safe points before and
//! after each synchronous check; this does not claim forced cancellation inside
//! the simplifier. Results are checked expressions, not portable certificates.

use super::system::general::GeneralOdeSystemError;
use super::system::general::forcing::{ElementaryForcing, metered_forced_ode_system};
use super::{
    ode_inputs_are_total, require_fresh_integration_constants,
    verify_const_coeff_second_order_nonhomogeneous_solution,
};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::{One, Signed, Zero};
use std::collections::BTreeMap;

const MAX_DEGREE: usize = 16;
const MAX_MODES: usize = 8;
const MAX_INPUT_BITS: u64 = 1024;
const MAX_DERIVED_BITS: u64 = 4096;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

type Result<T> = std::result::Result<T, GeneralOdeSystemError>;
type Polynomial = Vec<BigRational>;
type Modes = BTreeMap<Mode, Polynomial>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Mode {
    rate: BigRational,
    frequency: BigRational,
    sine: bool,
}

impl Mode {
    fn polynomial() -> Self {
        Self {
            rate: BigRational::zero(),
            frequency: BigRational::zero(),
            sine: false,
        }
    }

    fn is_polynomial(&self) -> bool {
        self.rate.is_zero() && self.frequency.is_zero() && !self.sine
    }
}

struct Work<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
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

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn unsupported(reason: &'static str) -> GeneralOdeSystemError {
    GeneralOdeSystemError::Unsupported(reason)
}

impl<M: BudgetMeter> Work<'_, M> {
    fn pay(&mut self, steps: u64, bytes: u64) -> Result<()> {
        self.checkpoint()?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1)),
            (Dimension::MemoryBytes, bytes.max(1)),
            (Dimension::AllocationCount, steps.max(1)),
        ])?;
        Ok(())
    }

    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        if bits(value) > MAX_DERIVED_BITS || !value.denom().is_positive() {
            return Err(GeneralOdeSystemError::ResourceLimit(
                "scalar ODE forcing coefficient height",
            ));
        }
        self.pay(1, 128 + 2 * bits(value).div_ceil(8))?;
        Ok(value.clone())
    }

    fn preflight(&mut self, left: &BigRational, right: &BigRational) -> Result<()> {
        self.checkpoint()?;
        if bits(left).saturating_add(bits(right)).saturating_add(2) > MAX_DERIVED_BITS {
            return Err(GeneralOdeSystemError::ResourceLimit(
                "scalar ODE forcing intermediate coefficient growth",
            ));
        }
        Ok(())
    }

    fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.preflight(left, right)?;
        Ok(left.metered_add(right, self)?)
    }

    fn mul(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.preflight(left, right)?;
        Ok(left.metered_mul(right, self)?)
    }

    fn div(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.preflight(left, right)?;
        Ok(left.metered_div(right, self)?)
    }

    fn zeroes(&mut self, length: usize) -> Result<Polynomial> {
        if length > MAX_DEGREE + 1 {
            return Err(unsupported(
                "scalar ODE polynomial forcing degree exceeds sixteen",
            ));
        }
        self.pay(length as u64, 128 * length as u64)?;
        Ok(vec![BigRational::zero(); length])
    }

    fn insert(&mut self, output: &mut Modes, mode: Mode, values: Polynomial) -> Result<()> {
        self.pay(1, 256 + 128 * values.len() as u64)?;
        if values.iter().all(Zero::is_zero) {
            return Ok(());
        }
        if !output.contains_key(&mode) && output.len() == MAX_MODES {
            return Err(unsupported(
                "scalar ODE forcing exceeds eight elementary modes",
            ));
        }
        let polynomial = output.entry(mode).or_default();
        if values.len() > polynomial.len() {
            polynomial.resize(values.len(), BigRational::zero());
        }
        for (index, value) in values.iter().enumerate() {
            polynomial[index] = self.add(&polynomial[index], value)?;
        }
        while polynomial.len() > 1 && polynomial.last().is_some_and(Zero::is_zero) {
            polynomial.pop();
        }
        Ok(())
    }

    fn constant(&mut self, value: BigRational) -> Result<Modes> {
        let mut output = BTreeMap::new();
        self.insert(&mut output, Mode::polynomial(), vec![value])?;
        Ok(output)
    }

    fn polynomial_product(
        &mut self,
        left: &[BigRational],
        right: &[BigRational],
    ) -> Result<Polynomial> {
        let mut result = self.zeroes(left.len() + right.len() - 1)?;
        for (i, a) in left.iter().enumerate() {
            for (j, b) in right.iter().enumerate() {
                let product = self.mul(a, b)?;
                result[i + j] = self.add(&result[i + j], &product)?;
            }
        }
        Ok(result)
    }

    fn product(&mut self, left: &Modes, right: &Modes) -> Result<Modes> {
        let mut result = BTreeMap::new();
        for (a, p) in left {
            for (b, q) in right {
                if !a.frequency.is_zero() && !b.frequency.is_zero() {
                    return Err(unsupported("products of trigonometric forcing factors"));
                }
                let oscillation = if a.frequency.is_zero() { b } else { a };
                let mode = Mode {
                    rate: self.add(&a.rate, &b.rate)?,
                    frequency: self.copy(&oscillation.frequency)?,
                    sine: oscillation.sine,
                };
                let values = self.polynomial_product(p, q)?;
                self.insert(&mut result, mode, values)?;
            }
        }
        Ok(result)
    }

    fn linear_argument(&mut self, expression: &Expr, variable: &Symbol) -> Result<BigRational> {
        let modes = self.read(expression, variable)?;
        if modes.is_empty() {
            return Ok(BigRational::zero());
        }
        let mut entries = modes.into_iter();
        let (mode, polynomial) = entries.next().expect("nonempty source modes");
        if entries.next().is_some()
            || !mode.is_polynomial()
            || polynomial.len() > 2
            || !polynomial[0].is_zero()
        {
            return Err(unsupported(
                "scalar ODE elementary arguments must be rational multiples of time",
            ));
        }
        polynomial
            .get(1)
            .map_or(Ok(BigRational::zero()), |value| self.copy(value))
    }

    fn read(&mut self, expression: &Expr, variable: &Symbol) -> Result<Modes> {
        self.pay(1, 256)?;
        match expression {
            Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => {
                self.pay(1, 128 + value.bits().div_ceil(8))?;
                self.constant(BigRational::from_integer(value.clone()))
            }
            Expr::Rational(value) if bits(value) <= MAX_INPUT_BITS => {
                let value = self.copy(value)?;
                self.constant(value)
            }
            Expr::Integer(_) | Expr::Rational(_) => Err(GeneralOdeSystemError::ResourceLimit(
                "scalar ODE forcing input coefficient exceeds 1024 bits",
            )),
            Expr::Sym(symbol) if symbol == variable => {
                let mut result = BTreeMap::new();
                self.insert(
                    &mut result,
                    Mode::polynomial(),
                    vec![BigRational::zero(), BigRational::one()],
                )?;
                Ok(result)
            }
            Expr::Add(terms) => {
                let mut result = BTreeMap::new();
                for term in terms {
                    for (mode, polynomial) in self.read(term, variable)? {
                        self.insert(&mut result, mode, polynomial)?;
                    }
                }
                Ok(result)
            }
            Expr::Mul(factors) => {
                let mut result = self.constant(BigRational::one())?;
                for factor in factors {
                    // Read every original factor even after an exact zero.
                    let next = self.read(factor, variable)?;
                    result = self.product(&result, &next)?;
                }
                Ok(result)
            }
            Expr::Pow(base, exponent) => {
                let Expr::Integer(exponent) = exponent.as_ref() else {
                    return Err(unsupported(
                        "scalar ODE forcing needs nonnegative integer polynomial powers",
                    ));
                };
                let exponent = exponent
                    .to_u64()
                    .filter(|&power| power <= MAX_DEGREE as u64)
                    .ok_or_else(|| unsupported("scalar ODE polynomial power exceeds sixteen"))?;
                let base = self.read(base, variable)?;
                if base.keys().any(|mode| !mode.is_polynomial()) {
                    return Err(unsupported("powers of nonpolynomial scalar ODE forcing"));
                }
                let mut result = self.constant(BigRational::one())?;
                for _ in 0..exponent {
                    result = self.product(&result, &base)?;
                }
                Ok(result)
            }
            Expr::Function(name, arguments)
                if arguments.len() == 1 && matches!(name.as_str(), "exp" | "sin" | "cos") =>
            {
                let argument = self.linear_argument(&arguments[0], variable)?;
                let mut mode = Mode::polynomial();
                let mut value = BigRational::one();
                if name == "exp" {
                    mode.rate = argument;
                } else {
                    mode.sine = name == "sin";
                    if argument.is_zero() {
                        return self.constant(if mode.sine {
                            BigRational::zero()
                        } else {
                            value
                        });
                    }
                    if argument.is_negative() {
                        mode.frequency = self.mul(&BigRational::from(-1), &argument)?;
                        if mode.sine {
                            value = BigRational::from(-1);
                        }
                    } else {
                        mode.frequency = argument;
                    }
                }
                let mut result = BTreeMap::new();
                self.insert(&mut result, mode, vec![value])?;
                Ok(result)
            }
            _ => Err(unsupported(
                "scalar ODE forcing contains an unsupported original source",
            )),
        }
    }
}

/// Solve `a*y'' + b*y' + c*y = source` with `y(0)=C1`, `y'(0)=C2`.
///
/// The two constants must be distinct fresh symbols. Sources are finite sums
/// and products of rational polynomials (degree at most sixteen), `exp(r*x)`,
/// `sin(w*x)` and `cos(w*x)` with rational rates/frequencies. Products with two
/// oscillatory factors, other parameters and non-entire original syntax refuse.
/// The forcing derivative closure plus the two physical coordinates must fit
/// the existing eight-dimensional system limit: a lone exponential supports
/// degree five, and a lone oscillatory mode supports degree two. Multiple modes
/// share that bound. Polynomial forcing does not add auxiliary coordinates.
///
/// Input/output expression bounds, 1024-bit input coefficients, cumulative
/// phase caps and the independent residual check can refuse sooner. No sampled
/// numerical value establishes the returned identity or its source domain.
pub fn metered_second_order(
    coefficients: [i64; 3],
    source: &Expr,
    variable: &Symbol,
    constants: [&Symbol; 2],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Expr> {
    let mut generator = Work {
        inner: generator,
        used: [0; 3],
    };
    let mut verifier = Work {
        inner: verifier,
        used: [0; 3],
    };
    generator.pay(256, 256 * 1024)?;
    verifier.checkpoint()?;
    let [a, b, c] = coefficients;
    if a == 0 {
        return Err(GeneralOdeSystemError::InvalidSource(
            "second-order leading coefficient is zero",
        ));
    }
    require_fresh_integration_constants(variable, &constants, &[source]).map_err(|_| {
        GeneralOdeSystemError::InvalidSource(
            "scalar ODE source bounds or nonfresh integration constants",
        )
    })?;
    if !ode_inputs_are_total(&[source]) {
        return Err(GeneralOdeSystemError::InvalidSource(
            "scalar ODE original forcing is not total on real time",
        ));
    }
    let modes = generator.read(source, variable)?;
    let a = BigRational::from(a);
    let mut polynomial = vec![BigRational::zero()];
    let mut elementary = Vec::new();
    generator.pay(64, 16 * 1024)?;
    for (mode, values) in modes {
        if mode.is_polynomial() {
            polynomial = values
                .iter()
                .map(|value| generator.div(value, &a))
                .collect::<Result<_>>()?;
            continue;
        }
        for (power, value) in values.iter().enumerate() {
            if value.is_zero() {
                continue;
            }
            if elementary.len() == MAX_MODES {
                return Err(unsupported(
                    "scalar ODE forcing exceeds eight nonzero elementary terms",
                ));
            }
            let value = generator.div(value, &a)?;
            let mut cosine = vec![BigRational::zero(); 2];
            let mut sine = vec![BigRational::zero(); 2];
            if mode.sine {
                sine[1] = value;
            } else {
                cosine[1] = value;
            }
            elementary.push(ElementaryForcing {
                power,
                rate: generator.copy(&mode.rate)?,
                frequency: generator.copy(&mode.frequency)?,
                cosine,
                sine,
            });
        }
    }
    let matrix = vec![
        vec![BigRational::zero(), BigRational::one()],
        vec![
            generator.div(&(-BigRational::from(c)), &a)?,
            generator.div(&(-BigRational::from(b)), &a)?,
        ],
    ];
    let initial = constants.map(|constant| Expr::Sym(constant.clone()));
    let result = metered_forced_ode_system(
        &matrix,
        variable,
        &BigRational::zero(),
        &[vec![BigRational::zero()], polynomial],
        &elementary,
        &initial,
        &mut generator,
        &mut verifier,
    )?
    .into_solutions()
    .into_iter()
    .next()
    .expect("admitted two-dimensional system");
    if !crate::verifier_inputs_within_bounds([&result, source]) {
        return Err(GeneralOdeSystemError::ResourceLimit(
            "scalar ODE final expression verification envelope",
        ));
    }
    // The scalar checker is independent of source decomposition and system
    // realization. Prepay its bounded recursive differentiation/normalization.
    verifier.pay(20_000_000, 64 * 1024 * 1024)?;
    if !verify_const_coeff_second_order_nonhomogeneous_solution(
        &result,
        coefficients[0],
        b,
        c,
        source,
        variable,
    ) {
        return Err(GeneralOdeSystemError::Verification(
            "original scalar ODE residual",
        ));
    }
    verifier.checkpoint()?;
    Ok(result)
}

/// Explicit unbounded-meter convenience wrapper with the same internal caps.
pub fn second_order(
    coefficients: [i64; 3],
    source: &Expr,
    variable: &Symbol,
    constants: [&Symbol; 2],
) -> Result<Expr> {
    metered_second_order(
        coefficients,
        source,
        variable,
        constants,
        &mut Unbounded,
        &mut Unbounded,
    )
}
