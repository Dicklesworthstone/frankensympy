//! Bounded exact sign-bracket search with independent root-existence checks.
//!
//! The generator uses only certified values of the original scalar source,
//! not symbolic derivatives or floating-point sign guesses. Its result is a
//! proposed interval until the lower-layer verifier independently checks the
//! original domain, endpoint signs and requested width. Accepted receipts
//! assert existence only, never uniqueness, multiplicity or completeness.

use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{
    BigRational, Expr, MAX_SCALAR_ROOT_BISECTIONS, RootBoxError, ScalarRootBracketError,
    ScalarRootBracketReceipt, ScalarRootBracketWitness, ScalarRootBracketWorkspace, Symbol,
    validate_scalar_root_bracket_request, verify_scalar_root_bracket,
};
use std::cmp::Ordering;

/// Precision controls certified function-value rounding, while the exact
/// absolute tolerance independently bounds the width of the root interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarRootBracketOptions {
    pub precision_digits: u32,
    pub max_bisections: usize,
}

impl Default for ScalarRootBracketOptions {
    fn default() -> Self {
        Self {
            precision_digits: 32,
            max_bisections: 256,
        }
    }
}

type Result<T> = std::result::Result<T, ScalarRootBracketError>;

/// Certify at least one real root in a closed interval of the requested width.
/// Odd-multiplicity roots and continuous Abs corners need no nonzero derivative.
/// A zero-width request succeeds only for a point proved to be exactly zero.
pub fn certified_scalar_root_bracket(
    expression: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    options: ScalarRootBracketOptions,
) -> Result<ScalarRootBracketReceipt> {
    metered_certified_scalar_root_bracket(
        expression,
        variable,
        lower,
        upper,
        absolute_tolerance,
        options,
        &mut Unbounded,
        &mut Unbounded,
    )
}

/// Generation and independent acceptance retain separate cumulative meters.
/// Cancellation/resource/domain failures are terminal and publish no receipt.
#[allow(clippy::too_many_arguments)]
pub fn metered_certified_scalar_root_bracket(
    expression: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    options: ScalarRootBracketOptions,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<ScalarRootBracketReceipt> {
    let witness = propose_scalar_root_bracket(
        expression,
        variable,
        lower,
        upper,
        absolute_tolerance,
        options,
        generator_meter,
    )?;
    let receipt = verify_scalar_root_bracket(
        expression,
        variable,
        lower,
        upper,
        absolute_tolerance,
        &witness,
        verifier_meter,
    )?;
    generator_meter.checkpoint().map_err(RootBoxError::from)?;
    Ok(receipt)
}

/// Propose an exact interval without granting evidence. An integrating
/// coordinator may run generation on ordinary resources, then call the core
/// verifier against its frozen source using its protected verification pool.
///
/// A rounded midpoint enclosing zero is never treated as an exact root. Two
/// quarter-point probes can retain a narrower certified sign bracket around
/// it, including roots like sin(pi*x) whose midpoint rounding need not vanish.
/// If neither probe proves a narrower bracket, the search is inconclusive.
pub fn propose_scalar_root_bracket(
    expression: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    options: ScalarRootBracketOptions,
    generator_meter: &mut impl BudgetMeter,
) -> Result<ScalarRootBracketWitness> {
    generator_meter.checkpoint().map_err(RootBoxError::from)?;
    validate_scalar_root_bracket_request(lower, upper, absolute_tolerance)?;
    if options.max_bisections > MAX_SCALAR_ROOT_BISECTIONS {
        return Err(ScalarRootBracketError::ResourceLimit(
            "requested bisection count must be zero through 1024",
        ));
    }
    let mut workspace = ScalarRootBracketWorkspace::new(
        expression,
        variable,
        options.precision_digits,
        generator_meter,
    )?;
    // Sign changes across a pole/discontinuity do not establish a root.
    workspace.enclose(lower, upper)?;
    let left_value = workspace.enclose(lower, lower)?;
    let right_value = workspace.enclose(upper, upper)?;
    let left_sign = workspace.sign(&left_value)?;
    let right_sign = workspace.sign(&right_value)?;
    if left_sign == Some(Ordering::Equal) {
        return workspace.propose(lower, lower);
    }
    if right_sign == Some(Ordering::Equal) {
        return workspace.propose(upper, upper);
    }
    let (Some(left_sign), Some(right_sign)) = (left_sign, right_sign) else {
        return Err(ScalarRootBracketError::IndeterminateSign);
    };
    if left_sign == right_sign {
        return Err(ScalarRootBracketError::NoSignChange);
    }
    let mut bracket = workspace.propose(lower, upper)?;
    for iteration in 0..=options.max_bisections {
        if workspace.within_tolerance(&bracket.lower, &bracket.upper, absolute_tolerance)? {
            return workspace.propose(&bracket.lower, &bracket.upper);
        }
        if iteration == options.max_bisections {
            break;
        }
        let midpoint = workspace.midpoint(&bracket.lower, &bracket.upper)?;
        let value = workspace.enclose(&midpoint, &midpoint)?;
        match workspace.sign(&value)? {
            Some(Ordering::Equal) => return workspace.propose(&midpoint, &midpoint),
            Some(sign) if sign == left_sign => bracket.lower = midpoint,
            Some(_) => bracket.upper = midpoint,
            None => {
                let left_quarter = workspace.midpoint(&bracket.lower, &midpoint)?;
                let right_quarter = workspace.midpoint(&midpoint, &bracket.upper)?;
                let value = workspace.enclose(&left_quarter, &left_quarter)?;
                let quarter_sign = workspace.sign(&value)?;
                if quarter_sign == Some(Ordering::Equal) {
                    return workspace.propose(&left_quarter, &left_quarter);
                }
                if quarter_sign == Some(right_sign) {
                    bracket.upper = left_quarter;
                    continue;
                }
                let mut advanced = quarter_sign == Some(left_sign);
                if advanced {
                    bracket.lower = left_quarter;
                }
                let value = workspace.enclose(&right_quarter, &right_quarter)?;
                match workspace.sign(&value)? {
                    Some(Ordering::Equal) => {
                        return workspace.propose(&right_quarter, &right_quarter);
                    }
                    Some(sign) if sign == left_sign => {
                        bracket.lower = right_quarter;
                        advanced = true;
                    }
                    Some(_) => {
                        bracket.upper = right_quarter;
                        advanced = true;
                    }
                    None => {}
                }
                if !advanced {
                    return Err(ScalarRootBracketError::IndeterminateSign);
                }
            }
        }
    }
    Err(ScalarRootBracketError::AccuracyNotReached)
}
