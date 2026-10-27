//! Adaptive proper real quadrature with independently checked enclosures.
//!
//! This generator proposes only a rational partition. It cannot construct
//! accepted receipts. The lower-layer checker re-evaluates the original
//! source and proves complete coverage and the requested enclosure width.
//! No symbolic antiderivative, binary64 quadrature estimate or generated
//! derivative is trusted. Both phases have separate supplied meters and
//! fixed cumulative source/evaluation/work/storage bounds.

use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{
    BallError, BigRational, Expr, IntegralEnclosureReceipt, IntegralEnclosureWorkspace,
    MAX_QUADRATURE_INTERVALS, QuadratureError, QuadratureWitness, RootBoxError, Symbol,
    validate_integral_request, verify_integral_enclosure,
};

/// Accuracy is the width of the final exact rational enclosure, not a
/// heuristic estimate or decimal display precision. Raising the interval
/// cap never weakens the independent whole-partition acceptance check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadratureOptions {
    pub precision_digits: u32,
    pub max_intervals: usize,
}

impl Default for QuadratureOptions {
    fn default() -> Self {
        Self {
            precision_digits: 32,
            max_intervals: 512,
        }
    }
}

/// Certified enclosure of a proper integral on finite exact real bounds.
/// Reversed endpoints negate the accepted enclosure. At equal endpoints
/// the original expression must still be admitted and C1 at that point.
pub fn certified_integral(
    expression: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    options: QuadratureOptions,
) -> Result<IntegralEnclosureReceipt, QuadratureError> {
    metered_certified_integral(
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

fn subdivision_can_resolve(error: &QuadratureError) -> bool {
    matches!(
        error,
        QuadratureError::Enclosure(
            RootBoxError::Domain(_)
                | RootBoxError::Ball(
                    BallError::DivisionByZero(_) | BallError::NonRealSquareRoot(_)
                )
        )
    )
}

/// Deterministic adaptive partitioning with C1 local-width bounds. A cell
/// receives tolerance proportional to its length; all accepted cells
/// therefore fit within the total tolerance. Domain overestimation can
/// trigger subdivision, but unsupported syntax, cancellation and resource
/// failures never restart with a fresh or unbounded meter.
#[allow(clippy::too_many_arguments)]
pub fn metered_certified_integral(
    expression: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    options: QuadratureOptions,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<IntegralEnclosureReceipt, QuadratureError> {
    generator_meter.checkpoint().map_err(RootBoxError::from)?;
    validate_integral_request(lower, upper, absolute_tolerance)?;
    if !(1..=MAX_QUADRATURE_INTERVALS).contains(&options.max_intervals) {
        return Err(QuadratureError::ResourceLimit(
            "requested interval count must be 1 through 1024",
        ));
    }
    let witness = {
        let mut enclosure = IntegralEnclosureWorkspace::new(
            expression,
            variable,
            options.precision_digits,
            generator_meter,
        )?;
        let total_width = enclosure.interval_width(lower, upper)?;
        let (start, end) = if lower <= upper {
            (lower, upper)
        } else {
            (upper, lower)
        };
        // Immediate point checks catch true endpoint poles without spending
        // the subdivision budget on an impossible proper-integral claim.
        enclosure.enclose_cell(start, start)?;
        if start != end {
            enclosure.enclose_cell(end, end)?;
        }
        let mut partition = Vec::with_capacity(options.max_intervals + 1);
        partition.push(start.clone());
        if start != end {
            let mut pending = Vec::with_capacity(options.max_intervals);
            pending.push((start.clone(), end.clone(), 0usize));
            let mut leaf_count = 1usize;
            while let Some((left, right, depth)) = pending.pop() {
                let mut domain_failure = None;
                let accepted = match enclosure.enclose_cell(&left, &right) {
                    Ok(value) => enclosure.within_local_tolerance(
                        &value,
                        &left,
                        &right,
                        &total_width,
                        absolute_tolerance,
                    )?,
                    Err(error) if subdivision_can_resolve(&error) => {
                        domain_failure = Some(error);
                        false
                    }
                    Err(error) => return Err(error),
                };
                if accepted {
                    partition.push(right);
                    continue;
                }
                if leaf_count >= options.max_intervals || depth >= 64 {
                    return Err(domain_failure.unwrap_or(QuadratureError::AccuracyNotReached));
                }
                let midpoint = enclosure.midpoint(&left, &right)?;
                leaf_count += 1;
                // The left cell is visited first, so accepted endpoints are
                // already in strict order without a comparison sort.
                pending.push((midpoint.clone(), right, depth + 1));
                pending.push((left, midpoint, depth + 1));
            }
        }
        QuadratureWitness {
            partition,
            precision_digits: options.precision_digits,
        }
    };
    let receipt = verify_integral_enclosure(
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
