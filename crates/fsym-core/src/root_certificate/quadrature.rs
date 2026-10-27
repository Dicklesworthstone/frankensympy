//! Independent, source-bound enclosures of proper real definite integrals.
//!
//! For a cell [a,b] of width h and midpoint c, interval automatic
//! differentiation establishes f'(x) in [d-r,d+r] on the complete cell.
//! The fundamental theorem of calculus gives
//! |f(x)-f(c)-d(x-c)| <= r|x-c|. Integrating, the linear term vanishes and
//! the error is at most r*h^2/4. Point-evaluation rounding is retained.
//! The checker intersects that bound with h*f([a,b]), checks a complete
//! partition, and adds all bounds exactly. A partition proposal never
//! supplies trusted function values, derivatives, error estimates or totals.

use super::{
    Bounded, BoundedRational, RootBoxError, Work, check_ball, check_rational, checked,
    validate_inputs,
};
use crate::{BigInt, BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::{Signed, Zero};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

pub const MAX_QUADRATURE_INTERVALS: usize = 1024;
pub const MAX_QUADRATURE_INPUT_BITS: u64 = 4096;
const MAX_QUADRATURE_EVALUATIONS: usize = 4 * MAX_QUADRATURE_INTERVALS + 4;

#[derive(Debug, Error)]
pub enum QuadratureError {
    #[error("certified integral tolerance must be a positive exact rational")]
    InvalidTolerance,
    #[error("invalid integral partition: {0}")]
    InvalidPartition(&'static str),
    #[error("certified integral resource limit exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("certified integral width exceeds the requested absolute tolerance")]
    AccuracyNotReached,
    #[error("independent integral bounds are inconsistent")]
    InconsistentEnclosure,
    #[error(transparent)]
    Enclosure(#[from] RootBoxError),
}

type Result<T> = std::result::Result<T, QuadratureError>;

/// Untrusted partition in increasing order, including both endpoints.
/// Reversed integration limits still use an increasing partition. A
/// zero-length integral uses exactly one partition point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuadratureWitness {
    pub partition: Vec<BigRational>,
    pub precision_digits: u32,
}

struct Endpoint(BigRational);
impl<'de> Deserialize<'de> for Endpoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = BoundedRational::deserialize(deserializer)?.0;
        validate_number(&value).map_err(D::Error::custom)?;
        Ok(Self(value))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessWire {
    partition: Bounded<Endpoint, { MAX_QUADRATURE_INTERVALS + 1 }>,
    precision_digits: u32,
}

impl<'de> Deserialize<'de> for QuadratureWitness {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let wire = WitnessWire::deserialize(deserializer)?;
        if !(8..=128).contains(&wire.precision_digits) {
            return Err(D::Error::custom(
                "quadrature precision must be 8 through 128",
            ));
        }
        Ok(Self {
            partition: wire.partition.0.into_iter().map(|value| value.0).collect(),
            precision_digits: wire.precision_digits,
        })
    }
}

/// Accepted proper real integral with an absolute enclosure-width bound.
/// C1 domain checks include every original source child on every closed
/// cell, including endpoints. Improper integrals, principal values and
/// complex contours are outside this claim. No unchecked constructor or
/// Deserialize implementation exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegralEnclosureReceipt {
    integrand: Expr,
    variable: Symbol,
    lower: BigRational,
    upper: BigRational,
    absolute_tolerance: BigRational,
    witness: QuadratureWitness,
    enclosure: RealBall,
}

impl IntegralEnclosureReceipt {
    pub fn integrand(&self) -> &Expr {
        &self.integrand
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
    pub fn witness(&self) -> &QuadratureWitness {
        &self.witness
    }
    pub fn enclosure(&self) -> &RealBall {
        &self.enclosure
    }
    pub fn interval_count(&self) -> usize {
        self.witness.partition.len().saturating_sub(1)
    }
}

fn validate_number(value: &BigRational) -> Result<()> {
    if !value.denom().is_positive()
        || value.numer().bits().max(value.denom().bits()) > MAX_QUADRATURE_INPUT_BITS
    {
        return Err(QuadratureError::ResourceLimit(
            "rational endpoint/tolerance height",
        ));
    }
    Ok(())
}

/// Validate the claim independently of any proposed partition.
pub fn validate_integral_request(
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
) -> Result<()> {
    for value in [lower, upper, absolute_tolerance] {
        validate_number(value)?;
    }
    if !absolute_tolerance.is_positive() {
        return Err(QuadratureError::InvalidTolerance);
    }
    Ok(())
}

/// Bounded source evaluator shared by partition generators and the checker.
/// It publishes only local interval bounds; it cannot create an accepted
/// whole-integral receipt. Evaluation counters and resource accounting are
/// cumulative across cells, with one scratch reservation for the session.
pub struct IntegralEnclosureWorkspace<'a, M> {
    source: &'a Expr,
    variable: &'a Symbol,
    work: Work<'a, M>,
    evaluations: usize,
}

impl<'a, M: BudgetMeter> IntegralEnclosureWorkspace<'a, M> {
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
        // Covers the bounded generator stack, accepted partition and the
        // checker's partition copy before any of those are allocated.
        work.charge(1, 8 * 1024 * 1024, MAX_QUADRATURE_INTERVALS as u64 * 8)?;
        Ok(Self {
            source,
            variable,
            work,
            evaluations: 0,
        })
    }

    fn evaluate(&mut self, domain: &RealBall) -> Result<super::Jet> {
        if self.evaluations >= MAX_QUADRATURE_EVALUATIONS {
            return Err(QuadratureError::ResourceLimit("interval evaluation count"));
        }
        self.evaluations += 1;
        // Each evaluation retains the ordinary source node/depth limits.
        // Only this per-traversal count restarts; work/bytes never reset.
        self.work.nodes = 0;
        Ok(self.work.eval(
            self.source,
            std::slice::from_ref(self.variable),
            std::slice::from_ref(domain),
            0,
        )?)
    }

    pub fn midpoint(&mut self, lower: &BigRational, upper: &BigRational) -> Result<BigRational> {
        validate_number(lower)?;
        validate_number(upper)?;
        self.work.arithmetic(&[], &[lower, upper])?;
        if lower >= upper {
            return Err(QuadratureError::InvalidPartition(
                "cell endpoints are not increasing",
            ));
        }
        let midpoint = (lower + upper) / BigRational::from_integer(2.into());
        validate_number(&midpoint)?;
        Ok(midpoint)
    }

    pub fn interval_width(
        &mut self,
        lower: &BigRational,
        upper: &BigRational,
    ) -> Result<BigRational> {
        validate_number(lower)?;
        validate_number(upper)?;
        self.work.arithmetic(&[], &[lower, upper])?;
        let width = (upper - lower).abs();
        check_rational(&width)?;
        Ok(width)
    }

    /// Compute both whole-cell range and centered C1 integral bounds.
    /// A point cell still checks the original integrand before returning zero.
    pub fn enclose_cell(&mut self, lower: &BigRational, upper: &BigRational) -> Result<RealBall> {
        validate_number(lower)?;
        validate_number(upper)?;
        self.work.arithmetic(&[], &[lower, upper])?;
        if lower > upper {
            return Err(QuadratureError::InvalidPartition(
                "cell endpoints are reversed",
            ));
        }
        let two = BigRational::from_integer(2.into());
        let width = upper - lower;
        let domain = checked(
            RealBall::new((lower + upper) / &two, &width / &two).map_err(RootBoxError::from)?,
        )?;
        let range = self.evaluate(&domain)?;
        if width.is_zero() {
            return Ok(RealBall::from_i64(0));
        }
        let midpoint = self.evaluate(&RealBall::exact(domain.midpoint().clone()))?;
        let rectangle = self.work.scale(&range.value, &width)?;
        let center = self.work.scale(&midpoint.value, &width)?;
        let derivative_radius = RealBall::exact(range.gradient[0].radius().clone());
        let width_ball = RealBall::exact(width.clone());
        let square = self.work.mul(&width_ball, &width_ball)?;
        let remainder = self.work.mul(&derivative_radius, &square)?;
        let remainder = self
            .work
            .scale(&remainder, &BigRational::new(1.into(), 4.into()))?;
        let error = RealBall::new(BigRational::zero(), remainder.midpoint().clone())
            .map_err(RootBoxError::from)?;
        let centered = self.work.add(&center, &error)?;
        self.work.arithmetic(&[&rectangle, &centered], &[])?;
        let intersection = rectangle
            .intersect(&centered)
            .ok_or(QuadratureError::InconsistentEnclosure)?;
        self.round_outward(intersection)
    }

    /// Decide a sufficient local-width allocation for adaptive subdivision.
    /// Summing accepted widths across a partition is at most the request.
    pub fn within_local_tolerance(
        &mut self,
        enclosure: &RealBall,
        lower: &BigRational,
        upper: &BigRational,
        total_width: &BigRational,
        tolerance: &BigRational,
    ) -> Result<bool> {
        self.work
            .arithmetic(&[enclosure], &[lower, upper, total_width, tolerance])?;
        if lower >= upper || !total_width.is_positive() || !tolerance.is_positive() {
            return Err(QuadratureError::InvalidPartition(
                "invalid local width allocation",
            ));
        }
        let cell_width = upper - lower;
        check_rational(&cell_width)?;
        self.work.arithmetic(&[], &[&cell_width, tolerance])?;
        let allowed = tolerance * &cell_width;
        check_rational(&allowed)?;
        let radius = enclosure.radius();
        self.work.arithmetic(&[], &[radius, total_width])?;
        let actual = radius * total_width * BigRational::from_integer(2.into());
        check_rational(&actual)?;
        Ok(actual <= allowed)
    }

    fn round_outward(&mut self, ball: RealBall) -> Result<RealBall> {
        if ball.radius().is_zero() {
            return Ok(ball);
        }
        // A common decimal endpoint grid prevents denominator growth when
        // many unrelated certified transcendental enclosures are summed.
        let digits = self.work.precision + 8;
        let scale = BigInt::from(10).pow(digits);
        let scale_rational = BigRational::from_integer(scale.clone());
        self.work.arithmetic(&[&ball], &[&scale_rational])?;
        let lower = ball.lower();
        let upper = ball.upper();
        let lo = floor_quotient(&(lower.numer() * &scale), lower.denom());
        let hi = -floor_quotient(&(-upper.numer() * &scale), upper.denom());
        let denominator = scale * 2;
        Ok(checked(
            RealBall::new(
                BigRational::new(&lo + &hi, denominator.clone()),
                BigRational::new(hi - lo, denominator),
            )
            .map_err(RootBoxError::from)?,
        )?)
    }
}

fn floor_quotient(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    let (quotient, remainder) = numerator.div_rem(denominator);
    if remainder.is_negative() {
        quotient - BigInt::from(1)
    } else {
        quotient
    }
}

/// Convenience checker with the same fixed hard bounds and an unbounded
/// external meter. The partition, source and tolerance are still checked.
pub fn certify_integral_enclosure(
    source: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    witness: &QuadratureWitness,
) -> Result<IntegralEnclosureReceipt> {
    verify_integral_enclosure(
        source,
        variable,
        lower,
        upper,
        absolute_tolerance,
        witness,
        &mut Unbounded,
    )
}

/// Check complete coverage and recompute every bound from the original
/// typed source. A false tolerance claim, gap, overlap, duplicate endpoint,
/// substituted variable, source pole or exhausted/cancelled check refuses.
pub fn verify_integral_enclosure(
    source: &Expr,
    variable: &Symbol,
    lower: &BigRational,
    upper: &BigRational,
    absolute_tolerance: &BigRational,
    witness: &QuadratureWitness,
    meter: &mut impl BudgetMeter,
) -> Result<IntegralEnclosureReceipt> {
    meter.checkpoint().map_err(RootBoxError::from)?;
    validate_integral_request(lower, upper, absolute_tolerance)?;
    let points = &witness.partition;
    if points.is_empty() || points.len() > MAX_QUADRATURE_INTERVALS + 1 {
        return Err(QuadratureError::InvalidPartition("point count"));
    }
    let mut workspace =
        IntegralEnclosureWorkspace::new(source, variable, witness.precision_digits, meter)?;
    for point in points {
        validate_number(point)?;
        workspace.work.arithmetic(&[], &[point])?;
    }
    workspace.work.arithmetic(&[], &[lower, upper])?;
    let (start, end, reversed) = if lower <= upper {
        (lower, upper, false)
    } else {
        (upper, lower, true)
    };
    if points.first() != Some(start) || points.last() != Some(end) {
        return Err(QuadratureError::InvalidPartition(
            "endpoints do not match the original interval",
        ));
    }
    let mut enclosure = RealBall::from_i64(0);
    if start == end {
        if points.len() != 1 {
            return Err(QuadratureError::InvalidPartition(
                "a point interval requires one point",
            ));
        }
        workspace.enclose_cell(start, end)?;
    } else {
        for pair in points.windows(2) {
            workspace.work.arithmetic(&[], &[&pair[0], &pair[1]])?;
            if pair[0] >= pair[1] {
                return Err(QuadratureError::InvalidPartition(
                    "points must strictly increase",
                ));
            }
            let contribution = workspace.enclose_cell(&pair[0], &pair[1])?;
            enclosure = workspace.work.add(&enclosure, &contribution)?;
        }
    }
    workspace
        .work
        .arithmetic(&[&enclosure], &[absolute_tolerance])?;
    if enclosure.width() > *absolute_tolerance {
        return Err(QuadratureError::AccuracyNotReached);
    }
    if reversed {
        enclosure = enclosure.neg();
    }
    check_ball(&enclosure)?;
    workspace
        .work
        .charge(1, workspace.work.source_bytes + 2 * 1024 * 1024, 8)?;
    let receipt = IntegralEnclosureReceipt {
        integrand: source.clone(),
        variable: variable.clone(),
        lower: lower.clone(),
        upper: upper.clone(),
        absolute_tolerance: absolute_tolerance.clone(),
        witness: witness.clone(),
        enclosure,
    };
    workspace.work.step()?;
    Ok(receipt)
}
