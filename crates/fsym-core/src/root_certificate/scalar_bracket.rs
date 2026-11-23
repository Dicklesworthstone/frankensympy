//! Independent existence checks for continuous scalar sign brackets.
//!
//! The original expression is admitted on the complete requested closed real
//! interval through the independent Lipschitz evaluator. Certified opposite
//! endpoint signs then imply a zero by the intermediate value theorem; an
//! exactly zero endpoint is also sufficient. The witness supplies only exact
//! endpoints and precision, never trusted values or derivatives. In particular,
//! this claim requires neither a nonzero derivative nor a smooth Abs branch.
//!
//! A receipt asserts at least one zero in its bounded subinterval. It does not
//! assert uniqueness, multiplicity, complete root enumeration, or a residual
//! bound at the interval midpoint. Its type cannot stand in for a Krawczyk
//! root-box receipt. Unsupported continuity obligations remain refusals.

use super::{
    BoundedRational, RootBoxError, Work, check_rational, checked, lipschitz::evaluate_lipschitz,
    validate_inputs,
};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::{Signed, Zero};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use std::cmp::Ordering;
use thiserror::Error;

pub const MAX_SCALAR_ROOT_BISECTIONS: usize = 1024;
pub const MAX_SCALAR_ROOT_INPUT_BITS: u64 = 4096;
const MAX_EVALUATIONS: usize = 3 * MAX_SCALAR_ROOT_BISECTIONS + 4;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScalarRootBracketError {
    #[error("scalar root bracket tolerance must be a positive exact rational")]
    InvalidTolerance,
    #[error("invalid scalar root bracket: {0}")]
    InvalidBracket(&'static str),
    #[error("scalar root bracket resource limit exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("scalar root bracket width exceeds the requested absolute tolerance")]
    AccuracyNotReached,
    #[error("endpoints do not have opposite signs or an exactly zero value")]
    NoSignChange,
    #[error("certified endpoint rounding does not determine a sign or exact zero")]
    IndeterminateSign,
    #[error(transparent)]
    Enclosure(#[from] RootBoxError),
}

type Result<T> = std::result::Result<T, ScalarRootBracketError>;

/// Untrusted closed subinterval, in increasing order. Equal endpoints are
/// admitted only when independent evaluation proves that point is a zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScalarRootBracketWitness {
    pub lower: BigRational,
    pub upper: BigRational,
    pub precision_digits: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessWire {
    lower: BoundedRational,
    upper: BoundedRational,
    precision_digits: u32,
}

impl<'de> Deserialize<'de> for ScalarRootBracketWitness {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let wire = WitnessWire::deserialize(deserializer)?;
        validate_number(&wire.lower.0).map_err(D::Error::custom)?;
        validate_number(&wire.upper.0).map_err(D::Error::custom)?;
        if wire.lower.0 > wire.upper.0 || !(8..=128).contains(&wire.precision_digits) {
            return Err(D::Error::custom(
                "scalar root endpoints must be increasing and precision must be 8 through 128",
            ));
        }
        Ok(Self {
            lower: wire.lower.0,
            upper: wire.upper.0,
            precision_digits: wire.precision_digits,
        })
    }
}

/// Source-bound evidence for at least one real zero, including endpoint zeros
/// and odd-multiplicity or nonsmooth zeros. The source is continuous on the
/// original requested interval; the root enclosure is a closed subinterval
/// whose width is at most absolute_tolerance. This type has no unchecked
/// constructor, mutable public field or Deserialize implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarRootBracketReceipt {
    expression: Expr,
    variable: Symbol,
    lower: BigRational,
    upper: BigRational,
    absolute_tolerance: BigRational,
    witness: ScalarRootBracketWitness,
    root_enclosure: RealBall,
    endpoint_values: [RealBall; 2],
}

impl ScalarRootBracketReceipt {
    pub fn expression(&self) -> &Expr {
        &self.expression
    }
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn lower(&self) -> &BigRational {
        &self.lower
    }
    pub fn upper(&self) -> &BigRational {
        &self.upper
    }
    pub fn absolute_tolerance(&self) -> &BigRational {
        &self.absolute_tolerance
    }
    pub fn witness(&self) -> &ScalarRootBracketWitness {
        &self.witness
    }
    pub fn root_enclosure(&self) -> &RealBall {
        &self.root_enclosure
    }
    pub fn endpoint_values(&self) -> &[RealBall; 2] {
        &self.endpoint_values
    }
}

fn validate_number(value: &BigRational) -> Result<()> {
    if !value.denom().is_positive()
        || value.numer().bits().max(value.denom().bits()) > MAX_SCALAR_ROOT_INPUT_BITS
    {
        return Err(ScalarRootBracketError::ResourceLimit(
            "rational endpoint/tolerance height",
        ));
    }
    Ok(())
}

/// Validate the original requested interval and accuracy independently of any
/// proposed witness. A point interval is allowed; it still needs an exact zero.
pub fn validate_scalar_root_bracket_request(
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
) -> Result<()> {
    for value in [lower, upper, absolute_tolerance] {
        validate_number(value)?;
    }
    if lower > upper {
        return Err(ScalarRootBracketError::InvalidBracket(
            "requested endpoints are reversed",
        ));
    }
    if !absolute_tolerance.is_positive() {
        return Err(ScalarRootBracketError::InvalidTolerance);
    }
    Ok(())
}

/// Reusable bounded original-source evaluation for candidate search and the
/// independent checker. Evaluations prove local continuity and value bounds,
/// but cannot construct accepted root receipts. Work, arithmetic and retained
/// storage counters remain cumulative throughout the workspace lifetime.
pub struct ScalarRootBracketWorkspace<'a, M> {
    source: &'a Expr,
    variable: &'a Symbol,
    work: Work<'a, M>,
    evaluations: usize,
}

impl<'a, M: BudgetMeter> ScalarRootBracketWorkspace<'a, M> {
    pub fn new(
        source: &'a Expr,
        variable: &'a Symbol,
        precision_digits: u32,
        meter: &'a mut M,
    ) -> Result<Self> {
        let mut work = Work::new(meter, precision_digits)?;
        validate_inputs(
            std::slice::from_ref(source),
            std::slice::from_ref(variable),
            &[RealBall::from_i64(0)],
            &mut work,
        )?;
        // Covers bounded search endpoints, their transient exact arithmetic,
        // the witness and the checker's source-independent receipt fields.
        work.charge(1, 1024 * 1024, 64)?;
        Ok(Self {
            source,
            variable,
            work,
            evaluations: 0,
        })
    }

    fn interval(&mut self, lower: &BigRational, upper: &BigRational) -> Result<RealBall> {
        validate_number(lower)?;
        validate_number(upper)?;
        self.work.arithmetic(&[], &[lower, upper])?;
        if lower > upper {
            return Err(ScalarRootBracketError::InvalidBracket(
                "interval endpoints are reversed",
            ));
        }
        let two = BigRational::from_integer(2.into());
        Ok(checked(
            RealBall::new((lower + upper) / &two, (upper - lower) / two)
                .map_err(RootBoxError::from)?,
        )?)
    }

    /// Re-evaluate every visited original child and every selected Piecewise
    /// guard. Reciprocal/logarithmic holes remain visible behind multiplication
    /// by zero and zeroth powers. Abs corners retain continuity, without a C1
    /// or nonzero-derivative assertion. Other smooth primitives keep their
    /// existing conservative original-domain restrictions.
    pub fn enclose(&mut self, lower: &BigRational, upper: &BigRational) -> Result<RealBall> {
        let domain = self.interval(lower, upper)?;
        if self.evaluations >= MAX_EVALUATIONS {
            return Err(ScalarRootBracketError::ResourceLimit(
                "source evaluation count",
            ));
        }
        self.evaluations += 1;
        // Reset only the per-traversal node count, never the work/byte ledger.
        self.work.nodes = 0;
        let enclosure = evaluate_lipschitz(
            self.source,
            std::slice::from_ref(self.variable),
            std::slice::from_ref(&domain),
            &mut self.work,
        )?;
        self.work.step()?;
        Ok(enclosure.value)
    }

    /// Equal denotes a proved exact zero. An interval merely containing zero
    /// returns None and cannot discharge any sign or root obligation.
    pub fn sign(&mut self, value: &RealBall) -> Result<Option<Ordering>> {
        self.work.arithmetic(&[value], &[])?;
        Ok(if value.midpoint().is_zero() && value.radius().is_zero() {
            Some(Ordering::Equal)
        } else if value.is_positive() {
            Some(Ordering::Greater)
        } else if value.is_negative() {
            Some(Ordering::Less)
        } else {
            None
        })
    }

    pub fn midpoint(&mut self, lower: &BigRational, upper: &BigRational) -> Result<BigRational> {
        let interval = self.interval(lower, upper)?;
        validate_number(interval.midpoint())?;
        Ok(interval.midpoint().clone())
    }

    pub fn within_tolerance(
        &mut self,
        lower: &BigRational,
        upper: &BigRational,
        absolute_tolerance: &BigRational,
    ) -> Result<bool> {
        validate_number(absolute_tolerance)?;
        let interval = self.interval(lower, upper)?;
        self.work.arithmetic(&[&interval], &[absolute_tolerance])?;
        let width = interval.width();
        check_rational(&width)?;
        Ok(width <= *absolute_tolerance)
    }

    /// Retain a candidate interval; this method grants no root evidence.
    pub fn propose(
        &mut self,
        lower: &BigRational,
        upper: &BigRational,
    ) -> Result<ScalarRootBracketWitness> {
        self.interval(lower, upper)?;
        self.work.charge(1, 4096, 4)?;
        let witness = ScalarRootBracketWitness {
            lower: lower.clone(),
            upper: upper.clone(),
            precision_digits: self.work.precision,
        };
        self.work.step()?;
        Ok(witness)
    }
}

/// Convenience checker retaining the same fixed intrinsic work and size caps.
pub fn certify_scalar_root_bracket(
    source: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    witness: &ScalarRootBracketWitness,
) -> Result<ScalarRootBracketReceipt> {
    verify_scalar_root_bracket(
        source,
        variable,
        lower,
        upper,
        absolute_tolerance,
        witness,
        &mut Unbounded,
    )
}

/// Independently verify source continuity on the complete requested interval,
/// witness containment/width, and exact endpoint-zero or opposite-sign evidence.
/// No candidate search, symbolic differentiation, sampled continuity assumption,
/// or trusted candidate function value is used by this acceptance boundary.
pub fn verify_scalar_root_bracket(
    source: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    witness: &ScalarRootBracketWitness,
    meter: &mut impl BudgetMeter,
) -> Result<ScalarRootBracketReceipt> {
    meter.checkpoint().map_err(RootBoxError::from)?;
    validate_scalar_root_bracket_request(lower, upper, absolute_tolerance)?;
    validate_number(&witness.lower)?;
    validate_number(&witness.upper)?;
    let mut workspace =
        ScalarRootBracketWorkspace::new(source, variable, witness.precision_digits, meter)?;
    workspace
        .work
        .arithmetic(&[], &[lower, upper, &witness.lower, &witness.upper])?;
    if witness.lower < *lower || witness.upper > *upper || witness.lower > witness.upper {
        return Err(ScalarRootBracketError::InvalidBracket(
            "witness is not an ordered subinterval of the requested domain",
        ));
    }
    if !workspace.within_tolerance(&witness.lower, &witness.upper, absolute_tolerance)? {
        return Err(ScalarRootBracketError::AccuracyNotReached);
    }
    // The full request must be continuous even if a candidate tries to hide
    // a pole elsewhere by returning an otherwise valid small root interval.
    workspace.enclose(lower, upper)?;
    let left = workspace.enclose(&witness.lower, &witness.lower)?;
    let right = workspace.enclose(&witness.upper, &witness.upper)?;
    let left_sign = workspace.sign(&left)?;
    let right_sign = workspace.sign(&right)?;
    if left_sign != Some(Ordering::Equal) && right_sign != Some(Ordering::Equal) {
        match (left_sign, right_sign) {
            (Some(a), Some(b)) if a != b => {}
            (None, _) | (_, None) => return Err(ScalarRootBracketError::IndeterminateSign),
            _ => return Err(ScalarRootBracketError::NoSignChange),
        }
    }
    let root_enclosure = workspace.interval(&witness.lower, &witness.upper)?;
    workspace
        .work
        .charge(1, workspace.work.source_bytes + 1024 * 1024, 8)?;
    let receipt = ScalarRootBracketReceipt {
        expression: source.clone(),
        variable: variable.clone(),
        lower: lower.clone(),
        upper: upper.clone(),
        absolute_tolerance: absolute_tolerance.clone(),
        witness: witness.clone(),
        root_enclosure,
        endpoint_values: [left, right],
    };
    workspace.work.step()?;
    Ok(receipt)
}
