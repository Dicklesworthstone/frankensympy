//! Global value bounds and tolerance-optimal points on a finite real box.
//!
//! The witness contains only an exact point and a complete dyadic partition.
//! Every cell value is recomputed from the original source. A generator's
//! objective values, pruning decisions and claimed errors have no authority.
//! Acceptance proves a value interval and an absolute suboptimality bound,
//! not an exact optimizer, uniqueness or a complete set of optimizers.

use super::objective::{ObjectiveJet, evaluate_objective};
use super::{
    BoxExtremumKind, RootBoxError, Work, binding_bytes, check_ball, check_rational, checked,
    rational_limbs, validate_inputs,
};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::{Signed, Zero};
use std::cmp::Ordering;
use thiserror::Error;

pub const MAX_BOX_OPTIMIZATION_LEAVES: usize = 1024;
pub const MAX_BOX_OPTIMIZATION_DEPTH: usize = 64;
pub const MAX_BOX_OPTIMIZATION_NODES: usize = 2 * MAX_BOX_OPTIMIZATION_LEAVES - 1;
const MAX_EVALUATIONS: usize = 6 * MAX_BOX_OPTIMIZATION_LEAVES + 8;

/// A preorder binary partition. A split creates both closed half-boxes at
/// the exact midpoint of the selected coordinate, with the left child first.
/// Shared faces are intentional; gaps or omitted children are not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxPartitionStep {
    Leaf,
    Split { axis: usize },
}

/// Untrusted proposal, with no supplied values, derivatives or error bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxOptimizationWitness {
    pub partition: Vec<BoxPartitionStep>,
    pub point: Vec<BigRational>,
    pub precision_digits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BoxOptimizationError {
    #[error("box-optimization verification refused: {0}")]
    Verification(#[from] RootBoxError),
    #[error("box optimization requires a positive exact rational tolerance")]
    InvalidTolerance,
    #[error("invalid complete box partition: {0}")]
    InvalidPartition(&'static str),
    #[error("global box value gap exceeds the requested absolute tolerance")]
    AccuracyNotReached,
    #[error("independent box-optimization enclosures are inconsistent")]
    InconsistentEnclosure,
}

type Result<T> = std::result::Result<T, BoxOptimizationError>;

/// The global minimum/maximum value is in value. The objective value at the
/// exact feasible point differs from that optimum by at most gap, which is
/// no larger than absolute_tolerance. No coordinate-distance claim is made.
/// All fields are private, with no unchecked constructor or deserializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxOptimizationReceipt {
    objective: Expr,
    variables: Vec<Symbol>,
    domain: Vec<RealBall>,
    kind: BoxExtremumKind,
    absolute_tolerance: BigRational,
    witness: BoxOptimizationWitness,
    value: RealBall,
    point_value: RealBall,
    gap: BigRational,
    leaf_count: usize,
}

impl BoxOptimizationReceipt {
    pub fn objective(&self) -> &Expr {
        &self.objective
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }
    pub fn domain(&self) -> &[RealBall] {
        &self.domain
    }
    pub fn kind(&self) -> BoxExtremumKind {
        self.kind
    }
    pub fn absolute_tolerance(&self) -> &BigRational {
        &self.absolute_tolerance
    }
    pub fn witness(&self) -> &BoxOptimizationWitness {
        &self.witness
    }
    pub fn point(&self) -> &[BigRational] {
        &self.witness.point
    }
    pub fn precision_digits(&self) -> u32 {
        self.witness.precision_digits
    }
    /// Enclosure of the optimum value, not the range of every objective value.
    pub fn value(&self) -> &RealBall {
        &self.value
    }
    pub fn point_value(&self) -> &RealBall {
        &self.point_value
    }
    pub fn gap(&self) -> &BigRational {
        &self.gap
    }
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }
}

/// Local bounds usable by a search strategy. These do not assert that the
/// supplied cell covers the requested domain or contains a global optimizer.
#[derive(Debug, Clone)]
pub struct BoxCellEnclosure {
    range: RealBall,
    center: Vec<BigRational>,
    center_value: RealBall,
    gradient: Vec<RealBall>,
}

impl BoxCellEnclosure {
    pub fn range(&self) -> &RealBall {
        &self.range
    }
    pub fn center(&self) -> &[BigRational] {
        &self.center
    }
    pub fn center_value(&self) -> &RealBall {
        &self.center_value
    }
    pub fn gradient(&self) -> &[RealBall] {
        &self.gradient
    }
}

/// One bounded source-evaluation phase. The large elementary scratch
/// reservation is made once; all cells retain cumulative work, allocations,
/// evaluation counts and caller cancellation. Only per-traversal node counts
/// restart. This workspace cannot construct an accepted global receipt.
pub struct BoxOptimizationWorkspace<'a, M> {
    objective: &'a Expr,
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    work: Work<'a, M>,
    evaluations: usize,
}

impl<'a, M: BudgetMeter> BoxOptimizationWorkspace<'a, M> {
    pub fn new(
        objective: &'a Expr,
        variables: &'a [Symbol],
        domain: &'a [RealBall],
        precision_digits: u32,
        meter: &'a mut M,
    ) -> Result<Self> {
        let mut work = Work::new(meter, precision_digits)?;
        validate_inputs(
            std::slice::from_ref(objective),
            variables,
            domain,
            &mut work,
        )?;
        // Bounded tree bookkeeping, traversal stacks and output witnesses.
        // Rational payload copies and all arithmetic are additionally charged.
        work.charge(1, 8 * 1024 * 1024, MAX_BOX_OPTIMIZATION_NODES as u64 * 8)?;
        Ok(Self {
            objective,
            variables,
            domain,
            work,
            evaluations: 0,
        })
    }

    pub fn account(&mut self, steps: u64, bytes: u64, allocations: u64) -> Result<()> {
        Ok(self.work.charge(steps, bytes, allocations)?)
    }

    pub fn compare(&mut self, left: &BigRational, right: &BigRational) -> Result<Ordering> {
        self.work.arithmetic(&[], &[left, right])?;
        Ok(left.cmp(right))
    }

    pub fn difference(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.work.arithmetic(&[], &[left, right])?;
        let value = left - right;
        check_rational(&value)?;
        Ok(value)
    }

    pub fn multiply(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.work.arithmetic(&[], &[left, right])?;
        let value = left * right;
        check_rational(&value)?;
        Ok(value)
    }

    fn check_domain(&mut self, domain: &[RealBall]) -> Result<()> {
        if domain.len() != self.variables.len() {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        for (root, cell) in self.domain.iter().zip(domain) {
            check_ball(cell)?;
            self.work.arithmetic(&[root, cell], &[])?;
            if !root.contains_ball(cell) {
                return Err(BoxOptimizationError::InvalidPartition(
                    "cell leaves the original box",
                ));
            }
        }
        Ok(())
    }

    fn evaluate(&mut self, domain: &[RealBall]) -> Result<ObjectiveJet> {
        if self.evaluations >= MAX_EVALUATIONS {
            return Err(RootBoxError::ResourceLimit("box-optimization evaluation count").into());
        }
        self.evaluations += 1;
        self.work.nodes = 0;
        Ok(evaluate_objective(
            self.objective,
            self.variables,
            domain,
            &mut self.work,
        )?)
    }

    /// Recompute a feasible point from the original source, including C2
    /// domain obligations hidden by held zero factors and zero powers.
    pub fn evaluate_point(&mut self, point: &[BigRational]) -> Result<RealBall> {
        if point.len() != self.variables.len() {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        let mut domain = Vec::new();
        self.work.charge(
            1,
            binding_bytes(self.variables, self.domain),
            point.len() as u64 + 1,
        )?;
        domain
            .try_reserve_exact(point.len())
            .map_err(|_| RootBoxError::ResourceLimit("optimization point allocation"))?;
        for (coordinate, (value, root)) in point.iter().zip(self.domain).enumerate() {
            check_rational(value)?;
            self.work.arithmetic(&[root], &[value])?;
            if !root.contains(value) {
                return Err(RootBoxError::CenterOutsideBox { coordinate }.into());
            }
            domain.push(RealBall::exact(value.clone()));
        }
        Ok(self.evaluate(&domain)?.value)
    }

    /// Intersect the natural range with an independent centered Taylor
    /// enclosure. Diagonal displacement squares use the even-power interval;
    /// both off-diagonal terms remain in the full Hessian quadratic form.
    pub fn enclose_cell(&mut self, domain: &[RealBall]) -> Result<BoxCellEnclosure> {
        self.check_domain(domain)?;
        let whole = self.evaluate(domain)?;
        self.work.charge(
            1,
            3 * binding_bytes(self.variables, domain),
            3 * domain.len() as u64 + 3,
        )?;
        let center: Vec<_> = domain.iter().map(|b| b.midpoint().clone()).collect();
        let at: Vec<_> = center.iter().cloned().map(RealBall::exact).collect();
        let at_center = self.evaluate(&at)?;
        let mut delta = Vec::new();
        delta
            .try_reserve_exact(domain.len())
            .map_err(|_| RootBoxError::ResourceLimit("optimization displacement allocation"))?;
        for coordinate in domain {
            self.work.arithmetic(&[coordinate], &[])?;
            delta.push(
                RealBall::new(BigRational::zero(), coordinate.radius().clone())
                    .map_err(RootBoxError::from)?,
            );
        }
        self.work.arithmetic(&[&at_center.value], &[])?;
        let mut centered = at_center.value.clone();
        for (slope, displacement) in at_center.gradient.iter().zip(&delta) {
            let contribution = self.work.mul(slope, displacement)?;
            centered = self.work.add(&centered, &contribution)?;
        }
        let half = BigRational::new(1.into(), 2.into());
        for (i, row) in whole.hessian.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                let product = if i == j {
                    self.work.power(&delta[i], 2)?
                } else {
                    self.work.mul(&delta[i], &delta[j])?
                };
                let contribution = self.work.mul(entry, &product)?;
                let contribution = self.work.scale(&contribution, &half)?;
                centered = self.work.add(&centered, &contribution)?;
            }
        }
        self.work.arithmetic(&[&whole.value, &centered], &[])?;
        let range = whole
            .value
            .intersect(&centered)
            .ok_or(BoxOptimizationError::InconsistentEnclosure)?;
        check_ball(&range)?;
        Ok(BoxCellEnclosure {
            range,
            center,
            center_value: at_center.value,
            gradient: whole.gradient,
        })
    }

    /// Exact midpoint bisection, retaining the shared face in both children.
    pub fn bisect(
        &mut self,
        domain: &[RealBall],
        axis: usize,
    ) -> Result<(Vec<RealBall>, Vec<RealBall>)> {
        self.check_domain(domain)?;
        let Some(coordinate) = domain.get(axis) else {
            return Err(BoxOptimizationError::InvalidPartition("split coordinate"));
        };
        self.work.arithmetic(&[coordinate], &[])?;
        if coordinate.radius().is_zero() {
            return Err(BoxOptimizationError::InvalidPartition(
                "split of a fixed coordinate",
            ));
        }
        self.work.charge(
            1,
            2 * binding_bytes(self.variables, domain),
            2 * domain.len() as u64 + 2,
        )?;
        let half_radius = coordinate.radius() / BigRational::from_integer(2.into());
        let left_coordinate = checked(
            RealBall::new(coordinate.midpoint() - &half_radius, half_radius.clone())
                .map_err(RootBoxError::from)?,
        )?;
        let right_coordinate = checked(
            RealBall::new(coordinate.midpoint() + &half_radius, half_radius)
                .map_err(RootBoxError::from)?,
        )?;
        let mut left = domain.to_vec();
        let mut right = domain.to_vec();
        left[axis] = left_coordinate;
        right[axis] = right_coordinate;
        Ok((left, right))
    }
}

pub fn certify_box_optimization(
    objective: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    kind: BoxExtremumKind,
    absolute_tolerance: &BigRational,
    witness: &BoxOptimizationWitness,
) -> Result<BoxOptimizationReceipt> {
    verify_box_optimization(
        objective,
        variables,
        domain,
        kind,
        absolute_tolerance,
        witness,
        &mut Unbounded,
    )
}

/// Independently reconstruct complete box coverage and every original-source
/// bound. Every leaf is visited, including regions a generator pruned. The
/// point is independently reevaluated; no generator value enters acceptance.
pub fn verify_box_optimization(
    objective: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    kind: BoxExtremumKind,
    absolute_tolerance: &BigRational,
    witness: &BoxOptimizationWitness,
    meter: &mut impl BudgetMeter,
) -> Result<BoxOptimizationReceipt> {
    meter.checkpoint().map_err(RootBoxError::from)?;
    check_rational(absolute_tolerance)?;
    if !absolute_tolerance.is_positive() {
        return Err(BoxOptimizationError::InvalidTolerance);
    }
    if witness.partition.is_empty() || witness.partition.len() > MAX_BOX_OPTIMIZATION_NODES {
        return Err(BoxOptimizationError::InvalidPartition(
            "partition node count",
        ));
    }
    let mut workspace = BoxOptimizationWorkspace::new(
        objective,
        variables,
        domain,
        witness.precision_digits,
        meter,
    )?;
    let point_value = workspace.evaluate_point(&witness.point)?;
    workspace.account(1, binding_bytes(variables, domain), domain.len() as u64 + 1)?;
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(MAX_BOX_OPTIMIZATION_DEPTH + 1)
        .map_err(|_| RootBoxError::ResourceLimit("partition traversal stack"))?;
    pending.push((domain.to_vec(), 0usize));
    let mut global = None::<BigRational>;
    let mut leaf_count = 0usize;
    for step in &witness.partition {
        workspace.account(1, 0, 0)?;
        let Some((cell, depth)) = pending.pop() else {
            return Err(BoxOptimizationError::InvalidPartition(
                "trailing partition nodes",
            ));
        };
        match *step {
            BoxPartitionStep::Split { axis } => {
                if depth >= MAX_BOX_OPTIMIZATION_DEPTH {
                    return Err(BoxOptimizationError::InvalidPartition("partition depth"));
                }
                let (left, right) = workspace.bisect(&cell, axis)?;
                pending.push((right, depth + 1));
                pending.push((left, depth + 1));
            }
            BoxPartitionStep::Leaf => {
                leaf_count += 1;
                if leaf_count > MAX_BOX_OPTIMIZATION_LEAVES {
                    return Err(BoxOptimizationError::InvalidPartition(
                        "partition leaf count",
                    ));
                }
                let enclosed = workspace.enclose_cell(&cell)?;
                workspace.work.arithmetic(&[enclosed.range()], &[])?;
                let bound = match kind {
                    BoxExtremumKind::Minimum => enclosed.range.lower(),
                    BoxExtremumKind::Maximum => enclosed.range.upper(),
                };
                let improves = match &global {
                    None => true,
                    Some(previous) => match kind {
                        BoxExtremumKind::Minimum => workspace.compare(&bound, previous)?.is_lt(),
                        BoxExtremumKind::Maximum => workspace.compare(&bound, previous)?.is_gt(),
                    },
                };
                if improves {
                    global = Some(bound);
                }
            }
        }
    }
    if !pending.is_empty() {
        return Err(BoxOptimizationError::InvalidPartition(
            "missing partition children",
        ));
    }
    let global = global.ok_or(BoxOptimizationError::InvalidPartition(
        "no partition leaves",
    ))?;
    workspace.work.arithmetic(&[&point_value], &[&global])?;
    let (lower, upper) = match kind {
        BoxExtremumKind::Minimum => (global, point_value.upper()),
        BoxExtremumKind::Maximum => (point_value.lower(), global),
    };
    let gap = workspace.difference(&upper, &lower)?;
    if gap.is_negative() {
        return Err(BoxOptimizationError::InconsistentEnclosure);
    }
    if workspace.compare(&gap, absolute_tolerance)?.is_gt() {
        return Err(BoxOptimizationError::AccuracyNotReached);
    }
    workspace.work.arithmetic(&[], &[&lower, &upper, &gap])?;
    let two = BigRational::from_integer(2.into());
    let value =
        checked(RealBall::new((&lower + &upper) / &two, &gap / &two).map_err(RootBoxError::from)?)?;
    let point_bytes = witness
        .point
        .iter()
        .map(|q| 128 + 16 * rational_limbs(q))
        .sum::<u64>();
    workspace.account(
        1,
        workspace.work.source_bytes
            + binding_bytes(variables, domain)
            + point_bytes
            + witness.partition.len() as u64 * 64
            + 16 * rational_limbs(absolute_tolerance),
        (variables.len() + 12) as u64,
    )?;
    let receipt = BoxOptimizationReceipt {
        objective: objective.clone(),
        variables: variables.to_vec(),
        domain: domain.to_vec(),
        kind,
        absolute_tolerance: absolute_tolerance.clone(),
        witness: witness.clone(),
        value,
        point_value,
        gap,
        leaf_count,
    };
    workspace.account(1, 0, 0)?;
    Ok(receipt)
}
