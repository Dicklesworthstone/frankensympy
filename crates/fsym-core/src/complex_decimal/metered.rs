//! Significant decimal rounding of original complex expressions.
//!
//! The existing rectangular enclosure evaluator and this metered interface
//! share one dispatcher. Acceptance compares the exact rational endpoints of
//! each component; neither a midpoint nor a binary64 value chooses digits.

use crate::arith::RationalArithmeticError;
use crate::{BallError, BigInt, BigRational, ComplexBall, Constant, Expr, LambertWError, RealBall};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError};
use num_traits::{ToPrimitive, Zero};
use thiserror::Error;

const MAX_NODES: usize = 1024;
const MAX_DEPTH: usize = 32;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 65_536;
const MAX_WORKING_DIGITS: u32 = 1792;
const MAX_STEPS: u64 = 1_000_000_000;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ALLOCATIONS: u64 = 4_000_000;

/// A bounded complex decimal evaluation stopped without publishing either
/// component. Operational failures remain distinct from mathematical refusals.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MeteredComplexDecimalError {
    #[error("complex decimal ball evaluation failed: {0}")]
    Ball(#[from] BallError),
    #[error("complex decimal evaluation stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("complex decimal exact arithmetic failed: {0}")]
    Arithmetic(RationalArithmeticError),
    #[error("complex decimal resource limit: {0}")]
    ResourceLimit(&'static str),
    #[error("unsupported complex decimal source: {0}")]
    Unsupported(&'static str),
    #[error("complex enclosure could not separate the requested decimal rounding")]
    RoundingInconclusive,
}

impl Expr {
    /// Maximum requested significant decimal digits for complex evaluation.
    pub const MAX_COMPLEX_DECIMAL_DIGITS: u32 = 1000;

    /// Correctly round a ground complex expression to `digits` significant
    /// decimal digits in each component, using round-to-nearest/half-even.
    /// The result is `(real, imaginary)` in scientific notation, or exactly
    /// `"0"` for a component independently enclosed by the singleton [0,0].
    ///
    /// Principal logarithm/root/power domains and every original source child
    /// are retained. An uncertain component containing zero is never rounded
    /// to zero: bounded refinement must separate it or refuse. The strings
    /// are rounded values, not interval certificates or exact recognition.
    ///
    /// One cumulative caller ledger covers source admission, every precision
    /// retry, arithmetic/elementary reservations, rounding and publication.
    /// The intrinsic bounds are 1024 source nodes, depth 32, 4096-bit input
    /// coefficients, 65536-bit component rationals, one billion logical work
    /// units, 512 MiB cumulative allocation reservations and four million
    /// allocations. Requested precision is 1..=1000; working precision is at
    /// most 1792, leaving room for all recursive/kernel guard digits.
    ///
    /// Cancellation checkpoints occur at source nodes, algebraic operations,
    /// bounded elementary-kernel boundaries and publication. The existing
    /// kernels retain their fixed internal envelopes; this API does not claim
    /// preemption inside a bigint operation or transcendental series loop.
    pub fn evalf_complex_decimal_metered<M: BudgetMeter>(
        &self,
        digits: u32,
        meter: &mut M,
    ) -> Result<(String, String), MeteredComplexDecimalError> {
        let mut work = Work::new(meter);
        work.charge(1, 0, 0)?;
        if digits == 0 || digits > Self::MAX_COMPLEX_DECIMAL_DIGITS {
            return Err(MeteredComplexDecimalError::Unsupported(
                "requested digits must be in 1..=1000",
            ));
        }
        let mut nodes = 0;
        work.admit(self, 0, &mut nodes)?;
        let mut precision = digits + 16;
        loop {
            let value = evaluate(self, precision, &mut work)?;
            let real = work.round_component(value.real(), digits)?;
            let imag = work.round_component(value.imag(), digits)?;
            if let (Some(real), Some(imag)) = (real, imag) {
                work.charge(1, (real.len() + imag.len()) as u64, 2)?;
                work.meter.checkpoint()?;
                return Ok((real, imag));
            }
            if precision == MAX_WORKING_DIGITS {
                work.meter.checkpoint()?;
                return Err(MeteredComplexDecimalError::RoundingInconclusive);
            }
            precision = precision.saturating_mul(2).min(MAX_WORKING_DIGITS);
        }
    }
}

/// Mathematical dispatch is shared, while the historical convenience API
/// retains its original intrinsic-only accounting and admission behavior.
pub(crate) fn unmetered_ball(expr: &Expr, digits: u32) -> Result<ComplexBall, BallError> {
    evaluate(expr, digits, &mut Unmetered).map_err(|error| match error {
        MeteredComplexDecimalError::Ball(error) => error,
        // Unmetered never produces an operational/refinement error.
        other => BallError::ArgumentOutsideDeclaredEnvelope(other.to_string()),
    })
}

trait BallWork {
    fn node(&mut self, source: &Expr) -> Result<(), MeteredComplexDecimalError>;
    fn arithmetic(&mut self, values: &[&ComplexBall]) -> Result<(), MeteredComplexDecimalError>;
    fn elementary(
        &mut self,
        values: &[&ComplexBall],
        digits: u32,
        count: u64,
    ) -> Result<(), MeteredComplexDecimalError>;
    fn power(
        &mut self,
        value: &ComplexBall,
        exponent: i32,
    ) -> Result<(), MeteredComplexDecimalError>;
    fn real_source(
        &mut self,
        source: &Expr,
        digits: u32,
    ) -> Result<ComplexBall, MeteredComplexDecimalError>;
    fn finish(&mut self, value: &ComplexBall) -> Result<(), MeteredComplexDecimalError>;
}

struct Unmetered;

impl BallWork for Unmetered {
    fn node(&mut self, _: &Expr) -> Result<(), MeteredComplexDecimalError> {
        Ok(())
    }

    fn arithmetic(&mut self, _: &[&ComplexBall]) -> Result<(), MeteredComplexDecimalError> {
        Ok(())
    }

    fn elementary(
        &mut self,
        _: &[&ComplexBall],
        _: u32,
        _: u64,
    ) -> Result<(), MeteredComplexDecimalError> {
        Ok(())
    }

    fn power(&mut self, _: &ComplexBall, _: i32) -> Result<(), MeteredComplexDecimalError> {
        Ok(())
    }

    fn real_source(
        &mut self,
        source: &Expr,
        digits: u32,
    ) -> Result<ComplexBall, MeteredComplexDecimalError> {
        Ok(ComplexBall::from_real(source.eval_real_ball(digits)?))
    }

    fn finish(&mut self, _: &ComplexBall) -> Result<(), MeteredComplexDecimalError> {
        Ok(())
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    steps: u64,
    bytes: u64,
    allocations: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            steps: 0,
            bytes: 0,
            allocations: 0,
        }
    }

    fn charge(
        &mut self,
        steps: u64,
        bytes: u64,
        allocations: u64,
    ) -> Result<(), MeteredComplexDecimalError> {
        self.meter.checkpoint()?;
        let next_steps = self
            .steps
            .checked_add(steps)
            .filter(|n| *n <= MAX_STEPS)
            .ok_or(MeteredComplexDecimalError::ResourceLimit(
                "cumulative arithmetic work",
            ))?;
        let next_bytes = self
            .bytes
            .checked_add(bytes)
            .filter(|n| *n <= MAX_BYTES)
            .ok_or(MeteredComplexDecimalError::ResourceLimit(
                "cumulative allocation bytes",
            ))?;
        let next_allocations = self
            .allocations
            .checked_add(allocations)
            .filter(|n| *n <= MAX_ALLOCATIONS)
            .ok_or(MeteredComplexDecimalError::ResourceLimit(
                "cumulative allocations",
            ))?;
        let mut charges = [(Dimension::ComputeSteps, 0); 3];
        let mut count = 0;
        for (dimension, amount) in [
            (Dimension::ComputeSteps, steps),
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, allocations),
        ] {
            if amount != 0 {
                charges[count] = (dimension, amount);
                count += 1;
            }
        }
        if count != 0 {
            self.meter.charge_batch(&charges[..count])?;
        }
        self.steps = next_steps;
        self.bytes = next_bytes;
        self.allocations = next_allocations;
        Ok(())
    }

    fn admit(
        &mut self,
        expr: &Expr,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(), MeteredComplexDecimalError> {
        self.charge(1, 0, 0)?;
        *nodes += 1;
        if depth > MAX_DEPTH || *nodes > MAX_NODES {
            return Err(MeteredComplexDecimalError::ResourceLimit(
                "source depth or node count",
            ));
        }
        match expr {
            Expr::Integer(n) if n.bits() > MAX_INPUT_BITS => {
                return Err(MeteredComplexDecimalError::ResourceLimit(
                    "integer source height",
                ));
            }
            Expr::Rational(q) if q.height().max_bits() > MAX_INPUT_BITS => {
                return Err(MeteredComplexDecimalError::ResourceLimit(
                    "rational source height",
                ));
            }
            Expr::Sym(_) => return Err(MeteredComplexDecimalError::Unsupported("free symbol")),
            Expr::Const(
                Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity
                | Constant::NaN,
            ) => return Err(MeteredComplexDecimalError::Unsupported("nonfinite source")),
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                if let Expr::Function(name, args) = expr {
                    let supported = match name.as_str() {
                        "LambertW" => matches!(args.as_slice(), [_] | [_, Expr::Integer(_)]),
                        "atan2" => args.len() == 2,
                        "exp" | "log" | "ln" | "sqrt" | "sin" | "cos" | "tan" | "sinh" | "cosh"
                        | "tanh" | "abs" | "Abs" | "re" | "im" | "conjugate" | "asin" | "acos"
                        | "atan" | "asinh" | "acosh" | "atanh" | "erf" | "erfc" | "sinc"
                        | "acot" | "acoth" | "asec" | "acsc" | "asech" | "acsch" | "cot"
                        | "sec" | "csc" | "coth" | "sech" | "csch" => args.len() == 1,
                        _ => false,
                    };
                    if !supported {
                        return Err(MeteredComplexDecimalError::Unsupported(
                            "function tag or arity",
                        ));
                    }
                }
                if args.len() > MAX_NODES - *nodes {
                    return Err(MeteredComplexDecimalError::ResourceLimit(
                        "source node count",
                    ));
                }
                for arg in args {
                    self.admit(arg, depth + 1, nodes)?;
                }
            }
            Expr::Pow(base, exponent) => {
                self.admit(base, depth + 1, nodes)?;
                self.admit(exponent, depth + 1, nodes)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn round_component(
        &mut self,
        value: &RealBall,
        digits: u32,
    ) -> Result<Option<String>, MeteredComplexDecimalError> {
        let height = value
            .midpoint()
            .height()
            .max_bits()
            .saturating_add(value.radius().height().max_bits())
            .saturating_add(2);
        self.preflight(height)?;
        let width = height.div_ceil(64) + u64::from(digits) + 1;
        self.charge(64 * width.pow(2), 512 * width, 32)?;
        let lower = value.lower();
        let upper = value.upper();
        if lower.is_zero() && upper.is_zero() {
            return Ok(Some("0".into()));
        }
        if lower <= BigRational::zero() && upper >= BigRational::zero() {
            return Ok(None);
        }
        let first = crate::round_significant(&lower, digits);
        self.meter.checkpoint()?;
        let last = crate::round_significant(&upper, digits);
        if first != last {
            return Ok(None);
        }
        Ok(Some(format!("{}e{}", first.0, first.1)))
    }

    fn preflight(&mut self, bits: u64) -> Result<(), MeteredComplexDecimalError> {
        self.meter.checkpoint()?;
        if bits > MAX_WORK_BITS {
            return Err(MeteredComplexDecimalError::ResourceLimit(
                "rational work height",
            ));
        }
        Ok(())
    }
}

fn height(value: &ComplexBall) -> u64 {
    [
        value.real().midpoint(),
        value.real().radius(),
        value.imag().midpoint(),
        value.imag().radius(),
    ]
    .into_iter()
    .map(|q| q.height().max_bits())
    .max()
    .unwrap_or(0)
}

impl<M: BudgetMeter> BallWork for Work<'_, M> {
    fn node(&mut self, source: &Expr) -> Result<(), MeteredComplexDecimalError> {
        let bits = match source {
            Expr::Integer(value) => value.bits(),
            Expr::Rational(value) => value.height().max_bits().saturating_mul(2),
            _ => 1,
        };
        let limbs = bits.div_ceil(64) + 1;
        self.charge(8 * limbs, 64 * limbs, 4)
    }

    fn arithmetic(&mut self, values: &[&ComplexBall]) -> Result<(), MeteredComplexDecimalError> {
        let bits = values
            .iter()
            .map(|v| height(v))
            .sum::<u64>()
            .saturating_add(2);
        // Complex radius products combine at most six rational products;
        // include all temporary denominator products before invoking them.
        self.preflight(bits.saturating_mul(16))?;
        let limbs = bits.div_ceil(64) + 1;
        self.charge(256 * limbs.pow(2), 1024 * limbs, 64)
    }

    fn elementary(
        &mut self,
        values: &[&ComplexBall],
        digits: u32,
        count: u64,
    ) -> Result<(), MeteredComplexDecimalError> {
        let bits = values.iter().map(|v| height(v)).max().unwrap_or(1);
        self.preflight(bits)?;
        if digits > MAX_WORKING_DIGITS + (MAX_DEPTH as u32) * 4 + 16 {
            return Err(MeteredComplexDecimalError::ResourceLimit(
                "working precision",
            ));
        }
        // Conservative bounded-kernel work reservations, as in the existing
        // original-source root checker. Internal kernel envelopes still apply.
        let width = bits.div_ceil(64) + u64::from(digits) + 16;
        self.charge(count * 64 * width.pow(2), count * 4096 * width, count * 64)
    }

    fn power(
        &mut self,
        value: &ComplexBall,
        exponent: i32,
    ) -> Result<(), MeteredComplexDecimalError> {
        self.arithmetic(&[value])?;
        let magnitude = u64::from(exponent.unsigned_abs());
        let bits = height(value)
            .saturating_add(2)
            .saturating_mul(magnitude.max(1));
        self.preflight(bits)?;
        let width = bits.div_ceil(64) + 1;
        let operations =
            u64::from(u32::BITS - exponent.unsigned_abs().leading_zeros()).max(1) * 2 + 1;
        self.charge(
            operations * 256 * width.pow(2),
            operations * 1024 * width,
            operations * 64,
        )
    }

    fn real_source(
        &mut self,
        source: &Expr,
        digits: u32,
    ) -> Result<ComplexBall, MeteredComplexDecimalError> {
        let Expr::Function(name, args) = source else {
            return Err(MeteredComplexDecimalError::Unsupported(
                "real function source",
            ));
        };
        // Every original node in a real-only argument must itself admit a
        // real value. A complex child cannot disappear in a zero product.
        if name == "atan2" && args.len() == 2 {
            let y = evaluate_mode(&args[0], digits, self, true)?;
            let x = evaluate_mode(&args[1], digits, self, true)?;
            self.elementary(&[&y, &x], digits, 8)?;
            return Ok(ComplexBall::from_real(y.real().atan2(x.real(), digits)?));
        }
        if name == "LambertW" {
            let (argument, branch) = match args.as_slice() {
                [argument] => (argument, 0),
                [argument, Expr::Integer(branch)] if branch.is_zero() => (argument, 0),
                [argument, Expr::Integer(branch)] if *branch == BigInt::from(-1) => (argument, -1),
                _ => {
                    return Err(MeteredComplexDecimalError::Unsupported(
                        "literal real Lambert branch",
                    ));
                }
            };
            let argument = evaluate_mode(argument, digits, self, true)?;
            // Exact source recognition is allowed only after every original
            // child has passed the metered domain checks above.
            self.charge(1024, 4096, 32)?;
            if let Some(folded) = crate::elementary::eval_function(name, args)
                && !matches!(&folded, Expr::Function(..))
            {
                return evaluate_mode(&folded, digits, self, true);
            }
            let value = argument
                .real()
                .metered_lambert_w(branch, digits, self)
                .map_err(|error| match error {
                    LambertWError::Meter(error) => MeteredComplexDecimalError::Meter(error),
                    LambertWError::Ball(error) => MeteredComplexDecimalError::Ball(error),
                    LambertWError::Arithmetic(RationalArithmeticError::Meter(error)) => {
                        MeteredComplexDecimalError::Meter(error)
                    }
                    LambertWError::Arithmetic(error) => {
                        MeteredComplexDecimalError::Arithmetic(error)
                    }
                    LambertWError::OutsideRealDomain { .. } => {
                        MeteredComplexDecimalError::Unsupported("Lambert-W real branch domain")
                    }
                })?;
            return Ok(ComplexBall::from_real(value));
        }
        let [argument] = args.as_slice() else {
            return Err(MeteredComplexDecimalError::Unsupported("real unary arity"));
        };
        let argument = evaluate_mode(argument, digits, self, true)?;
        self.elementary(&[&argument], digits, 12)?;
        let arg = argument.real();
        let value = match name.as_str() {
            "acot" => arg.acot(digits)?,
            "acoth" => arg.acoth(digits)?,
            "asec" => arg.asec(digits)?,
            "acsc" => arg.acsc(digits)?,
            "asech" => arg.asech(digits)?,
            "acsch" => arg.acsch(digits)?,
            "erf" => arg.erf(digits)?,
            "erfc" => arg.erfc(digits)?,
            "sinc" => arg.sinc(digits)?,
            _ => {
                return Err(MeteredComplexDecimalError::Unsupported(
                    "real unary function",
                ));
            }
        };
        Ok(ComplexBall::from_real(value))
    }

    fn finish(&mut self, value: &ComplexBall) -> Result<(), MeteredComplexDecimalError> {
        self.preflight(height(value))
    }
}

// The Lambert kernel uses the same cumulative adapter, not a fresh ledger.
impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.meter.checkpoint()?;
        let mut steps = 0u64;
        let mut bytes = 0u64;
        let mut allocations = 0u64;
        for &(dimension, amount) in charges {
            let total = match dimension {
                Dimension::ComputeSteps => &mut steps,
                Dimension::MemoryBytes => &mut bytes,
                Dimension::AllocationCount => &mut allocations,
                _ => continue,
            };
            *total = total
                .checked_add(amount)
                .ok_or(BudgetError::ChargeOverflow { dimension })?;
        }
        for (dimension, used, amount, limit) in [
            (Dimension::ComputeSteps, self.steps, steps, MAX_STEPS),
            (Dimension::MemoryBytes, self.bytes, bytes, MAX_BYTES),
            (
                Dimension::AllocationCount,
                self.allocations,
                allocations,
                MAX_ALLOCATIONS,
            ),
        ] {
            if amount > limit - used {
                return Err(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining: limit - used,
                }
                .into());
            }
        }
        self.meter.charge_batch(charges)?;
        self.steps += steps;
        self.bytes += bytes;
        self.allocations += allocations;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

fn evaluate<W: BallWork>(
    source: &Expr,
    digits: u32,
    work: &mut W,
) -> Result<ComplexBall, MeteredComplexDecimalError> {
    evaluate_mode(source, digits, work, false)
}

fn evaluate_mode<W: BallWork>(
    source: &Expr,
    digits: u32,
    work: &mut W,
    require_real: bool,
) -> Result<ComplexBall, MeteredComplexDecimalError> {
    work.node(source)?;
    let extra = digits.saturating_add(4);
    let value = match source {
        Expr::Integer(n) => {
            ComplexBall::from_real(RealBall::exact(BigRational::from_integer(n.clone())))
        }
        Expr::Rational(q) => ComplexBall::from_real(RealBall::exact(q.clone())),
        Expr::Const(Constant::Pi) => {
            work.elementary(&[], extra, 2)?;
            ComplexBall::from_real(RealBall::pi(extra)?)
        }
        Expr::Const(Constant::E) => {
            work.elementary(&[], extra, 1)?;
            ComplexBall::from_real(RealBall::from_i64(1).exp(extra)?)
        }
        Expr::Const(Constant::I) => ComplexBall::i(),
        Expr::Add(terms) => {
            let mut result = ComplexBall::from_i64(0);
            for term in terms {
                let child = evaluate_mode(term, extra, work, require_real)?;
                work.arithmetic(&[&result, &child])?;
                result = result.checked_add(&child)?;
            }
            result
        }
        Expr::Mul(factors) => {
            let mut result = ComplexBall::from_i64(1);
            for factor in factors {
                let child = evaluate_mode(factor, extra, work, require_real)?;
                work.arithmetic(&[&result, &child])?;
                result = result.checked_mul(&child)?;
            }
            result
        }
        Expr::Pow(base, exponent) => {
            if let Some(integer) = exponent.const_integer_value().and_then(|k| k.to_i32()) {
                let base = evaluate_mode(base, extra, work, require_real)?;
                work.power(&base, integer)?;
                base.pow(integer)?
            } else if matches!(base.as_ref(), Expr::Const(Constant::E)) {
                let exponent = evaluate_mode(exponent, extra, work, require_real)?;
                work.elementary(&[&exponent], digits, 3)?;
                exponent.exp(digits)?
            } else if let Expr::Rational(q) = exponent.as_ref()
                && q.denom() == &BigInt::from(2)
                && let Some(numerator) = q.numer().to_i32()
            {
                let base = evaluate_mode(base, extra, work, require_real)?;
                work.elementary(&[&base], digits, 4)?;
                let root = base.sqrt(digits)?;
                work.power(&root, numerator)?;
                root.pow(numerator)?
            } else {
                // No source child may disappear because zero or one makes
                // its parent's mathematical value look trivial.
                let base = evaluate_mode(base, extra, work, require_real)?;
                let exponent = evaluate_mode(exponent, extra, work, require_real)?;
                work.elementary(&[&base, &exponent], digits, 12)?;
                base.pow_principal(&exponent, digits)?
            }
        }
        Expr::Function(name, _) if name == "LambertW" || name == "atan2" => {
            work.real_source(source, digits)?
        }
        Expr::Function(name, arguments) if arguments.len() == 1 => {
            let arg = evaluate_mode(&arguments[0], extra, work, require_real)?;
            match name.as_str() {
                "exp" | "log" | "ln" | "sqrt" | "sin" | "cos" => {
                    work.elementary(&[&arg], digits, 6)?;
                    match name.as_str() {
                        "exp" => arg.exp(digits)?,
                        "log" | "ln" => arg.ln(digits)?,
                        "sqrt" => arg.sqrt(digits)?,
                        "sin" => arg.sin(digits)?,
                        _ => arg.cos(digits)?,
                    }
                }
                "asin" | "acos" | "atan" | "asinh" | "acosh" | "atanh" => {
                    // Reserve the complete logarithm/root/angle composition
                    // on the same caller ledger before entering the kernel.
                    // Every original child was evaluated above, including
                    // undefined children hidden by a zero factor.
                    work.elementary(&[&arg], digits, 32)?;
                    match name.as_str() {
                        "asin" => arg.asin(digits)?,
                        "acos" => arg.acos(digits)?,
                        "atan" => arg.atan(digits)?,
                        "asinh" => arg.asinh(digits)?,
                        "acosh" => arg.acosh(digits)?,
                        _ => arg.atanh(digits)?,
                    }
                }
                "tan" => {
                    work.elementary(&[&arg], extra, 12)?;
                    let sine = arg.sin(extra)?;
                    let cosine = arg.cos(extra)?;
                    work.arithmetic(&[&sine, &cosine])?;
                    sine.checked_div(&cosine)?
                }
                "cot" | "sec" | "csc" | "coth" | "sech" | "csch" => {
                    // Reserve the complete circular/reciprocal composition on
                    // the caller ledger. The shared kernels retain both source
                    // components and reject every possible denominator zero.
                    work.elementary(&[&arg], digits, 12)?;
                    match name.as_str() {
                        "cot" => arg.cot(digits)?,
                        "sec" => arg.sec(digits)?,
                        "csc" => arg.csc(digits)?,
                        "coth" => arg.coth(digits)?,
                        "sech" => arg.sech(digits)?,
                        _ => arg.csch(digits)?,
                    }
                }
                "sinh" | "cosh" | "tanh" => {
                    work.elementary(&[&arg], extra, 8)?;
                    let plus = arg.exp(extra)?;
                    let minus = arg.neg().exp(extra)?;
                    let two = ComplexBall::from_i64(2);
                    work.arithmetic(&[&plus, &minus])?;
                    let numerator = if name == "cosh" {
                        plus.add(&minus)
                    } else {
                        plus.sub(&minus)
                    };
                    let denominator = if name == "tanh" {
                        plus.add(&minus)
                    } else {
                        two
                    };
                    work.arithmetic(&[&numerator, &denominator])?;
                    numerator.checked_div(&denominator)?
                }
                "abs" | "Abs" => {
                    work.elementary(&[&arg], extra, 4)?;
                    ComplexBall::from_real(arg.abs_ball(extra.saturating_mul(4).max(32))?)
                }
                "re" | "im" | "conjugate" => {
                    work.arithmetic(&[&arg])?;
                    match name.as_str() {
                        "re" => ComplexBall::from_real(arg.real().clone()),
                        "im" => ComplexBall::from_real(arg.imag().clone()),
                        _ => arg.conj(),
                    }
                }
                _ => work.real_source(source, digits)?,
            }
        }
        _ => {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(format!(
                "expression '{source}' cannot be evaluated as a ground ComplexBall"
            ))
            .into());
        }
    };
    work.finish(&value)?;
    if require_real && (!value.imag().midpoint().is_zero() || !value.imag().radius().is_zero()) {
        return Err(MeteredComplexDecimalError::Unsupported(
            "nonreal original child of a real-only function",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

    fn q(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    #[test]
    fn rounding_never_collapses_zero_containing_intervals() {
        let mut meter = Unbounded;
        let mut work = Work::new(&mut meter);
        for (middle, radius) in [(0, 1), (1, 1), (-1, 1)] {
            let ball = RealBall::new(q(middle, 1000), q(radius, 1000)).unwrap();
            assert_eq!(work.round_component(&ball, 30).unwrap(), None);
        }
        assert_eq!(
            work.round_component(&RealBall::from_i64(0), 30).unwrap(),
            Some("0".into())
        );
    }

    #[test]
    fn rounding_requires_both_endpoints_to_choose_the_same_half_even_bin() {
        let mut meter = Unbounded;
        let mut work = Work::new(&mut meter);
        // [1.249, 1.251] straddles the 1.25 half-even boundary at two digits.
        let crossing = RealBall::new(q(5, 4), q(1, 1000)).unwrap();
        assert_eq!(work.round_component(&crossing, 2).unwrap(), None);
        let inside = RealBall::new(q(124, 100), q(1, 1000)).unwrap();
        assert_eq!(
            work.round_component(&inside, 2).unwrap(),
            Some("1.2e0".into())
        );
    }
}
