//! Independent completeness checking for real roots in a finite box.
//!
//! A complete binary cover proves that every point is either in a region
//! whose original residual bounds exclude zero, or in an independently
//! checked unique-root box. The root boxes are inside the requested domain
//! and strictly pairwise separated. They may cross subdivision faces: a
//! root on an internal dyadic face is not discarded or counted twice.
//!
//! Acceptance counts all roots in this closed domain. It makes no claim
//! outside it and grants no partial receipt for unresolved, singular or
//! outer-boundary roots. A zero-root receipt requires a complete exclusion
//! cover, not merely failure to find a candidate.

use super::{
    RootBoxError, RootBoxReceipt, RootBoxWitness, Work, binding_bytes, check_ball, check_rational,
    checked, rational_limbs, validate_inputs, verify_root_box_preflighted,
};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, mem::size_of, sync::Arc};
use thiserror::Error;

pub const MAX_ROOT_ISOLATION_LEAVES: usize = 1024;
pub const MAX_ROOT_ISOLATION_NODES: usize = 2 * MAX_ROOT_ISOLATION_LEAVES - 1;
pub const MAX_ROOT_ISOLATION_DEPTH: usize = 64;
pub const MAX_ROOT_ISOLATION_ROOTS: usize = 128;
const MAX_EVALUATIONS: usize = 16 * MAX_ROOT_ISOLATION_LEAVES + 8;

/// Preorder cover of both closed children of every exact midpoint split.
/// Root leaves must be wholly contained in the named verified root box.
/// Exclusion is always recomputed from all of the original residuals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootIsolationStep {
    Split { axis: usize },
    Exclude,
    Root { index: usize },
}

/// An untrusted complete cover and local root proposals. No supplied
/// residual value, derivative, stored acceptance flag or root count is used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootIsolationWitness {
    pub roots: Vec<RootBoxWitness>,
    pub partition: Vec<RootIsolationStep>,
    pub precision_digits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RootIsolationError {
    // Preserve this source layer so operational errors remain discoverable
    // through Error::source at public bridges.
    #[error("root-isolation verification refused: {0}")]
    Verification(#[from] RootBoxError),
    #[error("invalid complete root-isolation partition: {0}")]
    InvalidPartition(&'static str),
    #[error("root boxes {first} and {second} are not strictly separated")]
    OverlappingRoots { first: usize, second: usize },
    #[error("complete root isolation remains inconclusive: {0}")]
    Incomplete(&'static str),
}

type Result<T> = std::result::Result<T, RootIsolationError>;

/// Exactly roots.len() distinct nonsingular real roots in domain, with one
/// independently checked local enclosure for each. Private fields and no
/// unchecked constructor or deserializer prevent unverified publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootIsolationReceipt {
    residuals: Vec<Expr>,
    variables: Vec<Symbol>,
    domain: Vec<RealBall>,
    witness: RootIsolationWitness,
    roots: Vec<RootBoxReceipt>,
    leaf_count: usize,
    excluded_leaf_count: usize,
}

impl RootIsolationReceipt {
    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }
    pub fn domain(&self) -> &[RealBall] {
        &self.domain
    }
    pub fn precision_digits(&self) -> u32 {
        self.witness.precision_digits
    }
    pub fn roots(&self) -> &[RootBoxReceipt] {
        &self.roots
    }
    pub fn witness(&self) -> &RootIsolationWitness {
        &self.witness
    }
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }
    pub fn excluded_leaf_count(&self) -> usize {
        self.excluded_leaf_count
    }
    pub fn is_root_free(&self) -> bool {
        self.roots.is_empty()
    }
}

/// Original-source C1 bounds for one cell. These local bounds establish no
/// existence, uniqueness or completeness claim by themselves.
#[derive(Debug, Clone)]
pub struct RootCellEnclosure {
    residual_values: Vec<RealBall>,
    jacobian: Vec<Vec<RealBall>>,
}

impl RootCellEnclosure {
    pub fn residual_values(&self) -> &[RealBall] {
        &self.residual_values
    }
    pub fn jacobian(&self) -> &[Vec<RealBall>] {
        &self.jacobian
    }
}

/// One source-bound phase with one elementary scratch reservation. The
/// original source is preflighted once. Cell traversals and local checks
/// share work, allocation, evaluation limits and the caller's cancellation
/// state, including failures. Only per-traversal node counters restart.
/// This workspace can grant local receipts, never a completeness receipt.
pub struct RootIsolationWorkspace<'a, M> {
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    domain: &'a [RealBall],
    work: Work<'a, M>,
    evaluations: usize,
}

/// Independently reconstructed pending regions of a candidate cover prefix.
/// This is traversal state, never a complete-root acceptance receipt.
pub struct RootIsolationPrefix {
    pending: Vec<(Vec<RealBall>, usize)>,
    leaf_count: usize,
    excluded_leaf_count: usize,
}

impl RootIsolationPrefix {
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    pub fn excluded_leaf_count(&self) -> usize {
        self.excluded_leaf_count
    }

    /// Pending cells are in LIFO order, with the next leftmost cell last.
    pub fn into_pending(self) -> Vec<(Vec<RealBall>, usize)> {
        self.pending
    }
}

impl<'a, M: BudgetMeter> RootIsolationWorkspace<'a, M> {
    pub fn new(
        residuals: &'a [Expr],
        variables: &'a [Symbol],
        domain: &'a [RealBall],
        precision_digits: u32,
        meter: &'a mut M,
    ) -> Result<Self> {
        let mut work = Work::new(meter, precision_digits)?;
        validate_inputs(residuals, variables, domain, &mut work)?;
        if residuals.len() != variables.len() {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        for (coordinate, ball) in domain.iter().enumerate() {
            if ball.radius().is_zero() {
                return Err(RootBoxError::DegenerateBox { coordinate }.into());
            }
        }
        // Bounded cover bookkeeping, queues, root proposals and output
        // vectors. Scalar arithmetic and payload copies are charged again
        // according to their actual bounded rational widths.
        work.charge(1, 8 * 1024 * 1024, MAX_ROOT_ISOLATION_NODES as u64 * 8)?;
        Ok(Self {
            residuals,
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

    pub fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.work.arithmetic(&[], &[left, right])?;
        let result = left + right;
        check_rational(&result)?;
        Ok(result)
    }

    pub fn subtract(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.work.arithmetic(&[], &[left, right])?;
        let result = left - right;
        check_rational(&result)?;
        Ok(result)
    }

    pub fn multiply(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.work.arithmetic(&[], &[left, right])?;
        let result = left * right;
        check_rational(&result)?;
        Ok(result)
    }

    pub fn divide(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.work.arithmetic(&[], &[left, right])?;
        if right.is_zero() {
            return Err(RootBoxError::SingularPreconditioner.into());
        }
        let result = left / right;
        check_rational(&result)?;
        Ok(result)
    }

    fn check_domain(&mut self, domain: &[RealBall]) -> Result<()> {
        if domain.len() != self.variables.len() {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        for (outer, cell) in self.domain.iter().zip(domain) {
            check_ball(cell)?;
            self.work.arithmetic(&[outer, cell], &[])?;
            if !outer.contains_ball(cell) {
                return Err(RootIsolationError::InvalidPartition(
                    "cell leaves the requested domain",
                ));
            }
        }
        Ok(())
    }

    fn reserve_evaluations(&mut self, count: usize) -> Result<()> {
        self.work.step()?;
        self.evaluations = self
            .evaluations
            .checked_add(count)
            .filter(|total| *total <= MAX_EVALUATIONS)
            .ok_or(RootBoxError::ResourceLimit(
                "root-isolation evaluation count",
            ))?;
        Ok(())
    }

    /// Point cells are admitted for candidate preconditioners. All original
    /// residuals are evaluated before returning, including held children of
    /// zero products or powers; a nonzero row never hides another row's pole.
    pub fn enclose_cell(&mut self, domain: &[RealBall]) -> Result<RootCellEnclosure> {
        self.check_domain(domain)?;
        self.reserve_evaluations(1)?;
        let n = self.variables.len();
        self.work.charge(1, (n * n * 256) as u64, (n + 2) as u64)?;
        let mut residual_values = Vec::with_capacity(n);
        let mut jacobian = Vec::with_capacity(n);
        self.work.nodes = 0;
        for residual in self.residuals {
            let jet = self.work.eval(residual, self.variables, domain, 0)?;
            residual_values.push(jet.value);
            jacobian.push(jet.gradient);
        }
        self.work.step()?;
        Ok(RootCellEnclosure {
            residual_values,
            jacobian,
        })
    }

    /// Reuse the existing local Krawczyk checker without rebuilding Work.
    /// Only a local root claim is issued here; the complete-cover verifier
    /// separately checks containment, separation and all excluded regions.
    pub fn verify_local(&mut self, witness: &RootBoxWitness) -> Result<RootBoxReceipt> {
        self.check_domain(&witness.domain)?;
        if witness.precision_digits != self.work.precision {
            return Err(RootBoxError::InvalidPrecision.into());
        }
        self.reserve_evaluations(2)?;
        Ok(verify_root_box_preflighted(
            self.residuals,
            self.variables,
            witness,
            &mut self.work,
            false,
            true,
        )?)
    }

    /// Split exactly at the coordinate midpoint, retaining the common face
    /// in both children. Neither gaps nor rounded endpoints are introduced.
    pub fn bisect(
        &mut self,
        domain: &[RealBall],
        axis: usize,
    ) -> Result<(Vec<RealBall>, Vec<RealBall>)> {
        self.check_domain(domain)?;
        let coordinate = domain
            .get(axis)
            .ok_or(RootIsolationError::InvalidPartition(
                "split coordinate is out of range",
            ))?;
        if coordinate.radius().is_zero() {
            return Err(RootIsolationError::InvalidPartition(
                "cannot split a fixed coordinate",
            ));
        }
        self.work.charge(
            1,
            2 * binding_bytes(self.variables, domain),
            2 * domain.len() as u64 + 2,
        )?;
        self.work.arithmetic(&[coordinate], &[])?;
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

    /// A candidate neighborhood may cross subdivision faces while remaining
    /// within the requested box. Doubling each radius and clipping to the
    /// original bounds is only proposal geometry, never root acceptance.
    pub fn expand_cell(&mut self, domain: &[RealBall]) -> Result<Vec<RealBall>> {
        self.check_domain(domain)?;
        self.work.charge(
            1,
            binding_bytes(self.variables, domain),
            domain.len() as u64 + 1,
        )?;
        let mut expanded = Vec::with_capacity(domain.len());
        let two = BigRational::from_integer(2.into());
        for (coordinate, (cell, outer)) in domain.iter().zip(self.domain).enumerate() {
            if cell.radius().is_zero() {
                return Err(RootBoxError::DegenerateBox { coordinate }.into());
            }
            self.work.arithmetic(&[cell, outer], &[])?;
            let radius = cell.radius() * &two;
            let lower = (cell.midpoint() - &radius).max(outer.lower());
            let upper = (cell.midpoint() + radius).min(outer.upper());
            expanded.push(checked(
                RealBall::new((&lower + &upper) / &two, (upper - lower) / &two)
                    .map_err(RootBoxError::from)?,
            )?);
        }
        Ok(expanded)
    }

    fn contains(&mut self, outer: &[RealBall], inner: &[RealBall]) -> Result<bool> {
        if outer.len() != self.variables.len() || inner.len() != outer.len() {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        for (outer, inner) in outer.iter().zip(inner) {
            self.work.arithmetic(&[outer, inner], &[])?;
            if !outer.contains_ball(inner) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn separated(&mut self, first: &[RealBall], second: &[RealBall]) -> Result<bool> {
        for (first, second) in first.iter().zip(second) {
            self.work.arithmetic(&[first, second], &[])?;
            if first.upper() < second.lower() || second.upper() < first.lower() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Validate a normalized candidate prefix without trusting serialized
    /// pending boxes, values, derivatives, or acceptance flags. All local
    /// roots and exclusions are replayed against the original frozen source.
    /// An empty prefix is valid, and leaves every point unresolved.
    pub fn validate_prefix(
        &mut self,
        witness: &RootIsolationWitness,
        max_leaves: usize,
    ) -> Result<RootIsolationPrefix> {
        if !(1..=MAX_ROOT_ISOLATION_LEAVES).contains(&max_leaves)
            || witness.partition.len() > 2 * max_leaves - 1
            || witness.roots.len() > MAX_ROOT_ISOLATION_ROOTS
        {
            return Err(RootIsolationError::InvalidPartition(
                "checkpoint node, root or leaf limit",
            ));
        }
        if witness.precision_digits != self.work.precision {
            return Err(RootBoxError::InvalidPrecision.into());
        }
        for (index, root) in witness.roots.iter().enumerate() {
            // Discard the local receipt: it authorizes no complete result.
            self.verify_local(root)?;
            for (previous, other) in witness.roots[..index].iter().enumerate() {
                if !self.separated(&root.domain, &other.domain)? {
                    return Err(RootIsolationError::OverlappingRoots {
                        first: previous,
                        second: index,
                    });
                }
            }
        }
        self.account(
            1,
            binding_bytes(self.variables, self.domain)
                + ((MAX_ROOT_ISOLATION_DEPTH + 1) * size_of::<(Vec<RealBall>, usize)>()) as u64,
            self.domain.len() as u64 + 2,
        )?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(MAX_ROOT_ISOLATION_DEPTH + 1)
            .map_err(|_| RootBoxError::ResourceLimit("root-prefix stack allocation"))?;
        pending.push((self.domain.to_vec(), 0usize));
        let mut leaf_count = 1usize;
        let mut excluded_leaf_count = 0usize;
        let mut introduced_roots = 0usize;
        for step in &witness.partition {
            self.account(1, 0, 0)?;
            let (cell, depth) = pending.pop().ok_or(RootIsolationError::InvalidPartition(
                "trailing checkpoint partition tokens",
            ))?;
            match *step {
                RootIsolationStep::Split { axis } => {
                    if depth >= MAX_ROOT_ISOLATION_DEPTH || leaf_count >= max_leaves {
                        return Err(RootIsolationError::InvalidPartition(
                            "checkpoint split exceeds depth or leaf limit",
                        ));
                    }
                    let (left, right) = self.bisect(&cell, axis)?;
                    pending.push((right, depth + 1));
                    pending.push((left, depth + 1));
                    leaf_count += 1;
                }
                RootIsolationStep::Exclude => {
                    let enclosure = self.enclose_cell(&cell)?;
                    let mut excluded = false;
                    for value in enclosure.residual_values() {
                        self.work.arithmetic(&[value], &[])?;
                        excluded |= value.lower() > BigRational::zero()
                            || value.upper() < BigRational::zero();
                    }
                    if !excluded {
                        return Err(RootIsolationError::Incomplete(
                            "checkpoint exclusion still admits a zero residual vector",
                        ));
                    }
                    excluded_leaf_count += 1;
                }
                RootIsolationStep::Root { index } => {
                    let root =
                        witness
                            .roots
                            .get(index)
                            .ok_or(RootIsolationError::InvalidPartition(
                                "checkpoint root reference is out of range",
                            ))?;
                    if index > introduced_roots {
                        return Err(RootIsolationError::InvalidPartition(
                            "checkpoint root introduction order",
                        ));
                    }
                    if !self.contains(&root.domain, &cell)? {
                        return Err(RootIsolationError::InvalidPartition(
                            "checkpoint root leaf escapes its verified root box",
                        ));
                    }
                    if index == introduced_roots {
                        introduced_roots += 1;
                    }
                }
            }
        }
        if introduced_roots != witness.roots.len() {
            return Err(RootIsolationError::InvalidPartition(
                "checkpoint has an unreferenced root proposal",
            ));
        }
        self.account(1, 0, 0)?;
        Ok(RootIsolationPrefix {
            pending,
            leaf_count,
            excluded_leaf_count,
        })
    }
}

/// Owned, preflighted input and cumulative work for a paused root search.
///
/// This object retains the original source and the elementary scratch
/// admission across meter borrows. Its arithmetic, allocation and evaluation
/// counters are never restarted, including when an operation refuses. It is
/// neither serializable nor cloneable, and grants no completeness receipt.
/// Callers continue supplying their same cumulative resource ledger; this
/// object does not manufacture or restore budget allowances.
pub struct RootIsolationPrepared {
    residuals: Vec<Expr>,
    variables: Vec<Symbol>,
    domain: Vec<RealBall>,
    precision_digits: u32,
    counters: RootIsolationCounters,
}

/// Consumed intrinsic work, not a transferable allowance or an authenticated
/// statement of remote history. Import adds these values to fresh receiver
/// work; the receiver's own caller ledger remains authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootIsolationCounters {
    pub steps: u64,
    pub allocation_bytes: u64,
    pub source_bytes: u64,
    pub evaluations: usize,
}

struct PreparedWorkspace<'a, 'state, M> {
    workspace: RootIsolationWorkspace<'a, M>,
    counters: &'state mut RootIsolationCounters,
}

impl<M> Drop for PreparedWorkspace<'_, '_, M> {
    fn drop(&mut self) {
        self.counters.steps = self.workspace.work.steps;
        self.counters.allocation_bytes = self.workspace.work.allocation_bytes;
        self.counters.source_bytes = self.workspace.work.source_bytes;
        self.counters.evaluations = self.workspace.evaluations;
    }
}

impl RootIsolationPrepared {
    /// Freeze already bounded syntax without sharing caller-owned expression
    /// buffers. Raw operand order, powers and typed symbol identities survive.
    pub fn new(
        residuals: &[Expr],
        variables: &[Symbol],
        domain: &[RealBall],
        precision_digits: u32,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self> {
        let mut workspace =
            RootIsolationWorkspace::new(residuals, variables, domain, precision_digits, meter)?;
        let copied_residuals = copy_sources(residuals, &mut workspace.work)?;
        workspace.work.charge(
            variables.len() as u64 + domain.len() as u64 + 1,
            binding_bytes(variables, domain)
                + (std::mem::size_of_val(variables) + std::mem::size_of_val(domain)) as u64,
            2 + variables.len() as u64 + 4 * domain.len() as u64,
        )?;
        let mut copied_variables = Vec::new();
        copied_variables
            .try_reserve_exact(variables.len())
            .map_err(|_| RootBoxError::ResourceLimit("root-search variable allocation"))?;
        copied_variables.extend(variables.iter().cloned());
        let mut copied_domain = Vec::new();
        copied_domain
            .try_reserve_exact(domain.len())
            .map_err(|_| RootBoxError::ResourceLimit("root-search domain allocation"))?;
        copied_domain.extend(domain.iter().cloned());
        workspace.account(1, 0, 0)?;
        Ok(Self {
            residuals: copied_residuals,
            variables: copied_variables,
            domain: copied_domain,
            precision_digits,
            counters: RootIsolationCounters {
                steps: workspace.work.steps,
                allocation_bytes: workspace.work.allocation_bytes,
                source_bytes: workspace.work.source_bytes,
                evaluations: workspace.evaluations,
            },
        })
    }

    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }

    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    pub fn domain(&self) -> &[RealBall] {
        &self.domain
    }

    pub fn precision_digits(&self) -> u32 {
        self.precision_digits
    }

    /// All attempted interval traversals, including attempts that refused.
    pub fn evaluation_count(&self) -> usize {
        self.counters.evaluations
    }

    pub fn counters(&self) -> RootIsolationCounters {
        self.counters
    }

    /// Carry forward a prior candidate's intrinsic work after header
    /// admission. Exact history cannot be authenticated by a checkpoint;
    /// callers also check independently provable structural lower bounds.
    /// No imported value credits or replenishes the receiving caller meter.
    /// Current preparation, later prefix verification and all future work
    /// remain additional consumption under the unchanged intrinsic caps.
    pub fn include_prior_work(&mut self, prior: RootIsolationCounters) -> Result<()> {
        if prior.source_bytes != self.counters.source_bytes
            || prior.steps < self.counters.steps
            || prior.allocation_bytes < self.counters.allocation_bytes
        {
            return Err(RootIsolationError::InvalidPartition(
                "checkpoint understates preparation work or changes source accounting",
            ));
        }
        let steps = prior
            .steps
            .checked_add(self.counters.steps)
            .filter(|total| *total <= super::MAX_ROOT_BOX_STEPS)
            .ok_or(RootBoxError::ResourceLimit("cumulative arithmetic work"))?;
        let allocation_bytes = prior
            .allocation_bytes
            .checked_add(self.counters.allocation_bytes)
            .filter(|total| *total <= super::MAX_ROOT_BOX_ALLOCATION_BYTES)
            .ok_or(RootBoxError::ResourceLimit("cumulative allocation"))?;
        let evaluations = prior
            .evaluations
            .checked_add(self.counters.evaluations)
            .filter(|total| *total <= MAX_EVALUATIONS)
            .ok_or(RootBoxError::ResourceLimit(
                "root-isolation evaluation count",
            ))?;
        self.counters = RootIsolationCounters {
            steps,
            allocation_bytes,
            source_bytes: self.counters.source_bytes,
            evaluations,
        };
        Ok(())
    }

    /// Borrow the frozen input and all previously admitted work with a caller's
    /// meter. Every updated intrinsic counter is retained before returning
    /// success, refusal or unwinding. Source preflight and scratch admission are
    /// reused only for this object's private immutable source, never for a
    /// replacement source supplied by the caller.
    pub fn with_workspace<M: BudgetMeter, T>(
        &mut self,
        meter: &mut M,
        operation: impl FnOnce(&mut RootIsolationWorkspace<'_, M>) -> Result<T>,
    ) -> Result<T> {
        let workspace = RootIsolationWorkspace {
            residuals: &self.residuals,
            variables: &self.variables,
            domain: &self.domain,
            work: Work {
                meter,
                nodes: 0,
                precision: self.precision_digits,
                steps: self.counters.steps,
                allocation_bytes: self.counters.allocation_bytes,
                source_bytes: self.counters.source_bytes,
            },
            evaluations: self.counters.evaluations,
        };
        let mut borrowed = PreparedWorkspace {
            workspace,
            counters: &mut self.counters,
        };
        borrowed
            .workspace
            .account(1, 0, 0)
            .and_then(|()| operation(&mut borrowed.workspace))
    }
}

// The source envelope was checked iteratively by validate_inputs before this
// bounded-depth copy. A plain Expr::clone would retain Arc children and their
// arbitrary spare capacities. Rebuild those children as raw syntax instead.
fn copy_sources(sources: &[Expr], work: &mut Work<'_, impl BudgetMeter>) -> Result<Vec<Expr>> {
    work.charge(1, std::mem::size_of_val(sources) as u64, 1)?;
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(sources.len())
        .map_err(|_| RootBoxError::ResourceLimit("root-search source allocation"))?;
    for source in sources {
        copied.push(copy_source(source, work)?);
    }
    Ok(copied)
}

fn copy_source(source: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Expr> {
    work.charge(1, size_of::<Expr>() as u64, 1)?;
    Ok(match source {
        Expr::Sym(symbol) => {
            work.charge(1, symbol.name.len() as u64, 1)?;
            Expr::Sym(symbol.clone())
        }
        Expr::Integer(value) => {
            work.charge(1, 16 * value.limb_count(), 1)?;
            Expr::Integer(value.clone())
        }
        Expr::Rational(value) => {
            work.charge(1, 16 * rational_limbs(value), 2)?;
            Expr::Rational(value.clone())
        }
        Expr::Const(value) => Expr::Const(*value),
        Expr::Add(children) => Expr::Add(copy_sources(children, work)?),
        Expr::Mul(children) => Expr::Mul(copy_sources(children, work)?),
        Expr::Pow(base, exponent) => {
            work.charge(
                1,
                2 * (size_of::<Expr>() + 2 * size_of::<usize>()) as u64,
                2,
            )?;
            Expr::Pow(
                Arc::new(copy_source(base, work)?),
                Arc::new(copy_source(exponent, work)?),
            )
        }
        Expr::Function(name, children) => {
            work.charge(1, name.len() as u64, 1)?;
            Expr::Function(name.clone(), copy_sources(children, work)?)
        }
    })
}

/// Verify a complete cover independently of any candidate generator.
/// Existence and uniqueness come from original-source local checks;
/// completeness comes only from replaying every region of the finite box.
pub fn verify_root_isolation(
    residuals: &[Expr],
    variables: &[Symbol],
    domain: &[RealBall],
    witness: &RootIsolationWitness,
    meter: &mut impl BudgetMeter,
) -> Result<RootIsolationReceipt> {
    let mut workspace = RootIsolationWorkspace::new(
        residuals,
        variables,
        domain,
        witness.precision_digits,
        meter,
    )?;
    if witness.partition.is_empty()
        || witness.partition.len() > MAX_ROOT_ISOLATION_NODES
        || witness.roots.len() > MAX_ROOT_ISOLATION_ROOTS
    {
        return Err(RootIsolationError::InvalidPartition(
            "empty cover or excessive node/root count",
        ));
    }
    let mut roots = Vec::new();
    roots
        .try_reserve_exact(witness.roots.len())
        .map_err(|_| RootBoxError::ResourceLimit("root receipt allocation"))?;
    for (index, proposed) in witness.roots.iter().enumerate() {
        let accepted = workspace.verify_local(proposed)?;
        for (previous, other) in witness.roots[..index].iter().enumerate() {
            if !workspace.separated(&proposed.domain, &other.domain)? {
                return Err(RootIsolationError::OverlappingRoots {
                    first: previous,
                    second: index,
                });
            }
        }
        roots.push(accepted);
    }

    let mut referenced = vec![false; roots.len()];
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(MAX_ROOT_ISOLATION_DEPTH + 1)
        .map_err(|_| RootBoxError::ResourceLimit("root-cover stack allocation"))?;
    workspace.account(1, binding_bytes(variables, domain), domain.len() as u64 + 1)?;
    pending.push((domain.to_vec(), 0usize));
    let mut leaf_count = 0usize;
    let mut excluded_leaf_count = 0usize;
    for step in &witness.partition {
        workspace.account(1, 0, 0)?;
        let (cell, depth) = pending.pop().ok_or(RootIsolationError::InvalidPartition(
            "trailing partition tokens",
        ))?;
        match *step {
            RootIsolationStep::Split { axis } => {
                if depth >= MAX_ROOT_ISOLATION_DEPTH {
                    return Err(RootIsolationError::InvalidPartition(
                        "partition depth exceeds its bound",
                    ));
                }
                let (left, right) = workspace.bisect(&cell, axis)?;
                pending.push((right, depth + 1));
                pending.push((left, depth + 1));
            }
            RootIsolationStep::Exclude => {
                // This evaluates the entire source forest before inspecting
                // any nonzero row, preserving every original C1 obligation.
                let enclosure = workspace.enclose_cell(&cell)?;
                let mut excluded = false;
                for value in enclosure.residual_values() {
                    workspace.work.arithmetic(&[value], &[])?;
                    excluded |=
                        value.lower() > BigRational::zero() || value.upper() < BigRational::zero();
                }
                if !excluded {
                    return Err(RootIsolationError::Incomplete(
                        "an exclusion leaf still admits a zero residual vector",
                    ));
                }
                leaf_count += 1;
                excluded_leaf_count += 1;
            }
            RootIsolationStep::Root { index } => {
                let root = roots
                    .get(index)
                    .ok_or(RootIsolationError::InvalidPartition(
                        "root reference is out of range",
                    ))?;
                if !workspace.contains(&root.witness().domain, &cell)? {
                    return Err(RootIsolationError::InvalidPartition(
                        "root leaf is not contained in its verified root box",
                    ));
                }
                referenced[index] = true;
                leaf_count += 1;
            }
        }
        if leaf_count > MAX_ROOT_ISOLATION_LEAVES {
            return Err(RootIsolationError::InvalidPartition(
                "partition leaf count exceeds its bound",
            ));
        }
    }
    if !pending.is_empty() {
        return Err(RootIsolationError::InvalidPartition(
            "partition omits one or more regions",
        ));
    }
    if referenced.iter().any(|used| !used) {
        return Err(RootIsolationError::InvalidPartition(
            "an accepted root box is not referenced by the cover",
        ));
    }
    let root_payload_bytes: u64 = witness
        .roots
        .iter()
        .map(|root| {
            binding_bytes(variables, &root.domain)
                + root
                    .center
                    .iter()
                    .chain(root.preconditioner.iter().flatten())
                    .map(|q| 128 + 16 * rational_limbs(q))
                    .sum::<u64>()
        })
        .sum();
    workspace.work.charge(
        1,
        workspace.work.source_bytes + binding_bytes(variables, domain) + root_payload_bytes,
        (witness.roots.len() * (variables.len() + 4) + 4) as u64,
    )?;
    let receipt = RootIsolationReceipt {
        residuals: residuals.to_vec(),
        variables: variables.to_vec(),
        domain: domain.to_vec(),
        witness: witness.clone(),
        roots,
        leaf_count,
        excluded_leaf_count,
    };
    workspace.account(0, 0, 0)?;
    Ok(receipt)
}

/// Convenience checking retains intrinsic limits even without a caller
/// ledger. It does not run a search or accept an incomplete candidate set.
pub fn certify_root_isolation(
    residuals: &[Expr],
    variables: &[Symbol],
    domain: &[RealBall],
    witness: &RootIsolationWitness,
) -> Result<RootIsolationReceipt> {
    verify_root_isolation(residuals, variables, domain, witness, &mut Unbounded)
}
