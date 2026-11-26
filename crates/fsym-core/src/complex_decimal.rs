//! Correctly rounded component decimals from complete complex enclosures.
//!
//! Decimal output is a presentation of an admitted ground value. It is not a
//! portable certificate or an enclosing interval, and no numerical zero test
//! authorizes dropping a component.

use crate::{BallError, Constant, Expr, RealBall, round_significant};
use num_traits::Zero;
use thiserror::Error;

mod metered;
pub use metered::MeteredComplexDecimalError;
pub(crate) use metered::unmetered_ball;

const MAX_DIGITS: u32 = 1000;
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_LITERAL_BITS: u64 = 16_384;
const MAX_TOTAL_LITERAL_BITS: u64 = 65_536;
const MAX_WORKING_DIGITS: u32 = 1536;

/// Distinguishes an unimplemented numeric grammar from a failed admitted
/// evaluation. Only `UnsupportedSource` permits selecting a different lane;
/// precision, resource, domain and rounding failures retain their meaning.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ComplexDecimalError {
    #[error("complex decimal precision must be in 1..={MAX_DIGITS}; got {0}")]
    InvalidPrecision(u32),
    #[error("complex decimal evaluation does not support {0}")]
    UnsupportedSource(&'static str),
    #[error("complex decimal source exceeds its {kind} limit of {limit}")]
    SourceLimitExceeded { kind: &'static str, limit: u64 },
    #[error("complex decimal enclosure failed: {0}")]
    Enclosure(#[from] BallError),
    #[error("complex decimal enclosure could not separate rounding to {0} significant digits")]
    RoundingInconclusive(u32),
}

impl Expr {
    /// Round each component of a ground complex value to `digits` significant
    /// decimal digits, independently, using round-to-nearest, ties-to-even.
    ///
    /// Each returned string is either `"0"` or a scientific decimal such as
    /// `"1.234e-20"`. A component is zero only when its enclosure is exactly
    /// `[0, 0]`. Otherwise both rational endpoints must round identically.
    /// Cancellation, a small midpoint, or a requested display precision never
    /// supplies an exact-zero assertion.
    ///
    /// Arithmetic and the existing principal complex elementary functions are
    /// admitted without rewriting the original source. Unknown functions,
    /// free symbols, nonfinite constants and real-only function kernels remain
    /// outside this lane. Undefined original children cannot be hidden by a
    /// zero factor or exponent. Precision increases deterministically up to
    /// 1536 working digits, within the component kernels' precision envelope.
    ///
    /// The full source is bounded before recursive evaluation: at most 4096
    /// nodes, depth 64, 16384 bits per literal and 65536 total literal bits.
    /// Arithmetic kernels enforce their own intermediate-size bounds. These
    /// are intrinsic limits, not a caller-owned cancellation/resource meter.
    /// The rounded decimal pair is not itself a mathematical proof receipt.
    pub fn evalf_complex_decimal(
        &self,
        digits: u32,
    ) -> Result<(String, String), ComplexDecimalError> {
        if digits == 0 || digits > MAX_DIGITS {
            return Err(ComplexDecimalError::InvalidPrecision(digits));
        }
        preflight(self)?;
        let mut working = digits + 16;
        loop {
            let enclosure = self.eval_complex_ball(working)?;
            if let (Some(real), Some(imag)) = (
                rounded_component(enclosure.real(), digits),
                rounded_component(enclosure.imag(), digits),
            ) {
                return Ok((real, imag));
            }
            if working == MAX_WORKING_DIGITS {
                return Err(ComplexDecimalError::RoundingInconclusive(digits));
            }
            working = working.saturating_mul(2).min(MAX_WORKING_DIGITS);
        }
    }
}

fn rounded_component(ball: &RealBall, digits: u32) -> Option<String> {
    let lower = ball.lower();
    let upper = ball.upper();
    if lower.is_zero() && upper.is_zero() {
        return Some("0".into());
    }
    // round_significant requires a nonzero rational. More fundamentally, a
    // nondegenerate interval containing zero cannot decide significant digits.
    if ball.contains_zero() {
        return None;
    }
    let (lower_digits, lower_exponent) = round_significant(&lower, digits);
    let (upper_digits, upper_exponent) = round_significant(&upper, digits);
    (lower_digits == upper_digits && lower_exponent == upper_exponent)
        .then(|| format!("{lower_digits}e{lower_exponent}"))
}

fn limit(kind: &'static str, limit: u64) -> ComplexDecimalError {
    ComplexDecimalError::SourceLimitExceeded { kind, limit }
}

fn preflight(source: &Expr) -> Result<(), ComplexDecimalError> {
    let mut pending = vec![(source, 1usize)];
    let mut nodes = 0usize;
    let mut literal_bits = 0u64;
    let mut unsupported = None;
    while let Some((node, depth)) = pending.pop() {
        nodes += 1;
        if nodes > MAX_NODES {
            return Err(limit("node count", MAX_NODES as u64));
        }
        if depth > MAX_DEPTH {
            return Err(limit("nesting depth", MAX_DEPTH as u64));
        }
        let bits = match node {
            Expr::Integer(integer) => integer.bits(),
            Expr::Rational(rational) => rational
                .numer()
                .bits()
                .saturating_add(rational.denom().bits()),
            Expr::Sym(_) => {
                unsupported = Some("free symbols");
                0
            }
            Expr::Const(Constant::Pi | Constant::E | Constant::I) => 0,
            Expr::Const(_) => {
                unsupported = Some("nonfinite constants");
                0
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = node {
                    let admitted = match name.as_str() {
                        "atan2" => children.len() == 2,
                        "exp" | "log" | "ln" | "sqrt" | "sin" | "cos" | "tan" | "sinh" | "cosh"
                        | "tanh" | "abs" | "Abs" | "re" | "im" | "conjugate" | "asin" | "acos"
                        | "atan" | "asinh" | "acosh" | "atanh" => children.len() == 1,
                        _ => false,
                    };
                    if !admitted {
                        unsupported = Some("this function or function arity");
                    }
                }
                if children.len() > MAX_NODES.saturating_sub(nodes + pending.len()) {
                    return Err(limit("node count", MAX_NODES as u64));
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
                0
            }
            Expr::Pow(base, exponent) => {
                if nodes + pending.len() > MAX_NODES - 2 {
                    return Err(limit("node count", MAX_NODES as u64));
                }
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
                0
            }
        };
        if bits > MAX_LITERAL_BITS {
            return Err(limit("literal bit length", MAX_LITERAL_BITS));
        }
        literal_bits = literal_bits.saturating_add(bits);
        if literal_bits > MAX_TOTAL_LITERAL_BITS {
            return Err(limit("total literal bits", MAX_TOTAL_LITERAL_BITS));
        }
    }
    // Inspect the entire original source even when an early node is outside
    // the grammar; a resource refusal must not become a fallback request.
    match unsupported {
        Some(reason) => Err(ComplexDecimalError::UnsupportedSource(reason)),
        None => Ok(()),
    }
}
