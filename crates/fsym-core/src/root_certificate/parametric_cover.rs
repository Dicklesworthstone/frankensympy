//! Complete parameter coverage by independently checked uniform root boxes.
//!
//! Every leaf proves one nonsingular unknown root in the SAME declared box
//! X, for every parameter in its reconstructed closed cell. Exact midpoint
//! replay covers the entire original parameter box P, including all shared
//! faces. Thus every p in P has exactly one nonsingular root in X. Adjacent
//! leaves cannot select different roots on a shared face: each establishes
//! uniqueness in that same X.
//!
//! The root enclosure is the coordinate hull of all accepted leaf images.
//! No existence outside X, moving-box branch stitching, exact solution
//! function, or global continuation/path certificate is asserted. An
//! unresolved cell prevents every complete-cover receipt.

use super::parametric_root::verify_parametric_root_box_with_work;
use super::{
    MAX_ROOT_BOX_PRECISION, MAX_ROOT_BOX_VARIABLES, ParametricRootBoxReceipt,
    ParametricRootBoxWitness, RootBoxError, Work, binding_bytes, check_ball, check_rational,
    checked, rational_limbs, validate_inputs,
};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::Zero;
use std::cmp::Ordering;
use thiserror::Error;

pub const MAX_PARAMETRIC_ROOT_COVER_LEAVES: usize = 1024;
pub const MAX_PARAMETRIC_ROOT_COVER_NODES: usize = 2 * MAX_PARAMETRIC_ROOT_COVER_LEAVES - 1;
pub const MAX_PARAMETRIC_ROOT_COVER_DEPTH: usize = 64;
const MAX_EVALUATIONS: usize = 16 * MAX_PARAMETRIC_ROOT_COVER_LEAVES + 8;

/// Canonical preorder cover of both closed children of each exact midpoint
/// split. Every leaf index must occur exactly once, and the referenced
/// proposal must match the reconstructed parameter cell exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParametricRootCoverStep {
    Split { axis: usize },
    Leaf { index: usize },
}

/// Untrusted full-parameter cover. Every proposal uses the original unknown
/// box and the common precision. No supplied interval value, derivative,
/// acceptance flag or claimed completeness count is trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricRootCoverWitness {
    pub leaves: Vec<ParametricRootBoxWitness>,
    pub partition: Vec<ParametricRootCoverStep>,
    pub precision_digits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParametricRootCoverError {
    // Keep this layer in Error::source so public bridges discover nested
    // cancellation, caller-budget and intrinsic resource refusals.
    #[error("parametric-root cover verification refused: {0}")]
    Verification(#[from] RootBoxError),
    #[error("invalid complete parametric-root cover: {0}")]
    InvalidPartition(&'static str),
    #[error("complete parametric-root coverage remains inconclusive: {0}")]
    Incomplete(&'static str),
}

type Result<T> = std::result::Result<T, ParametricRootCoverError>;

/// For every parameter in parameter_domain, exactly one nonsingular real
/// root exists in unknown_domain. Private fields and no unchecked constructor
/// or deserializer bind this claim to complete independent source replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricRootCoverReceipt {
    residuals: Vec<Expr>,
    unknowns: Vec<Symbol>,
    parameters: Vec<Symbol>,
    unknown_domain: Vec<RealBall>,
    parameter_domain: Vec<RealBall>,
    witness: ParametricRootCoverWitness,
    leaves: Vec<ParametricRootBoxReceipt>,
    root_enclosure: Vec<RealBall>,
}

impl ParametricRootCoverReceipt {
    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }
    pub fn unknowns(&self) -> &[Symbol] {
        &self.unknowns
    }
    pub fn parameters(&self) -> &[Symbol] {
        &self.parameters
    }
    pub fn unknown_domain(&self) -> &[RealBall] {
        &self.unknown_domain
    }
    pub fn parameter_domain(&self) -> &[RealBall] {
        &self.parameter_domain
    }
    pub fn precision_digits(&self) -> u32 {
        self.witness.precision_digits
    }
    pub fn leaves(&self) -> &[ParametricRootBoxReceipt] {
        &self.leaves
    }
    pub fn root_enclosure(&self) -> &[RealBall] {
        &self.root_enclosure
    }
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }
    pub fn witness(&self) -> &ParametricRootCoverWitness {
        &self.witness
    }
}

/// Original-source C1 bounds on one product cell. Jacobian columns are
/// ordered [unknowns, parameters]; these bounds alone prove no root claim.
#[derive(Debug, Clone)]
pub struct ParametricRootCellEnclosure {
    residual_values: Vec<RealBall>,
    jacobian: Vec<Vec<RealBall>>,
}

impl ParametricRootCellEnclosure {
    pub fn residual_values(&self) -> &[RealBall] {
        &self.residual_values
    }
    pub fn jacobian(&self) -> &[Vec<RealBall>] {
        &self.jacobian
    }
}

/// One source-bound proposal or verification phase. All cell enclosures and
/// local checks reuse one elementary scratch reservation and cumulative
/// arithmetic, allocation, evaluation and caller ledgers. Original source
/// preflight runs once; only per-traversal node counts restart. This object
/// may issue a local uniform receipt, never a complete-cover receipt.
pub struct ParametricRootCoverWorkspace<'a, M> {
    residuals: &'a [Expr],
    unknowns: &'a [Symbol],
    parameters: &'a [Symbol],
    unknown_domain: &'a [RealBall],
    parameter_domain: &'a [RealBall],
    variables: Vec<Symbol>,
    work: Work<'a, M>,
    evaluations: usize,
}

impl<'a, M: BudgetMeter> ParametricRootCoverWorkspace<'a, M> {
    pub fn new(
        residuals: &'a [Expr],
        unknowns: &'a [Symbol],
        parameters: &'a [Symbol],
        unknown_domain: &'a [RealBall],
        parameter_domain: &'a [RealBall],
        precision_digits: u32,
        meter: &'a mut M,
    ) -> Result<Self> {
        let mut work = Work::new(meter, precision_digits)?;
        let n = unknowns.len();
        let p = parameters.len();
        let total = n.checked_add(p).ok_or(RootBoxError::DimensionMismatch)?;
        if n == 0
            || p == 0
            || total > MAX_ROOT_BOX_VARIABLES
            || residuals.len() != n
            || unknown_domain.len() != n
            || parameter_domain.len() != p
        {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        if !(8..=MAX_ROOT_BOX_PRECISION).contains(&precision_digits) {
            return Err(RootBoxError::InvalidPrecision.into());
        }
        // Bound metadata and exact scalars before concatenation or cloning.
        for (index, variable) in unknowns.iter().chain(parameters).enumerate() {
            if variable.name.len() > 4096 {
                return Err(RootBoxError::ResourceLimit("symbol length").into());
            }
            if unknowns
                .iter()
                .chain(parameters)
                .take(index)
                .any(|previous| previous == variable)
            {
                return Err(RootBoxError::DuplicateVariable.into());
            }
        }
        for interval in unknown_domain.iter().chain(parameter_domain) {
            check_ball(interval)?;
        }
        for (coordinate, interval) in unknown_domain.iter().enumerate() {
            if interval.radius().is_zero() {
                return Err(RootBoxError::DegenerateBox { coordinate }.into());
            }
        }
        work.charge(
            1,
            binding_bytes(unknowns, unknown_domain) + binding_bytes(parameters, parameter_domain),
            (2 * total + 2) as u64,
        )?;
        let mut variables = Vec::new();
        variables
            .try_reserve_exact(total)
            .map_err(|_| RootBoxError::ResourceLimit("parameter-cover variable allocation"))?;
        variables.extend_from_slice(unknowns);
        variables.extend_from_slice(parameters);
        let mut product = Vec::new();
        product
            .try_reserve_exact(total)
            .map_err(|_| RootBoxError::ResourceLimit("parameter-cover domain allocation"))?;
        product.extend_from_slice(unknown_domain);
        product.extend_from_slice(parameter_domain);
        validate_inputs(residuals, &variables, &product, &mut work)?;
        // Bounded tree, proposal, reference and receipt bookkeeping. Exact
        // scalar copies and every actual source replay are charged separately.
        work.charge(
            1,
            8 * 1024 * 1024,
            MAX_PARAMETRIC_ROOT_COVER_NODES as u64 * 8,
        )?;
        Ok(Self {
            residuals,
            unknowns,
            parameters,
            unknown_domain,
            parameter_domain,
            variables,
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

    fn check_cell(&mut self, domain: &[RealBall], outer: &[RealBall]) -> Result<()> {
        if domain.len() != outer.len() {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        for (cell, outer) in domain.iter().zip(outer) {
            check_ball(cell)?;
            self.work.arithmetic(&[cell, outer], &[])?;
            if !outer.contains_ball(cell) {
                return Err(ParametricRootCoverError::InvalidPartition(
                    "cell leaves the requested domain",
                ));
            }
        }
        Ok(())
    }

    fn same_domain(&mut self, first: &[RealBall], second: &[RealBall]) -> Result<bool> {
        if first.len() != second.len() {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        for (first, second) in first.iter().zip(second) {
            check_ball(first)?;
            self.work.arithmetic(&[first, second], &[])?;
            if first != second {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn reserve_evaluations(&mut self, count: usize) -> Result<()> {
        self.work.step()?;
        self.evaluations = self
            .evaluations
            .checked_add(count)
            .filter(|total| *total <= MAX_EVALUATIONS)
            .ok_or(RootBoxError::ResourceLimit(
                "parameter-cover evaluation count",
            ))?;
        Ok(())
    }

    /// Point coordinates are allowed for untrusted inverse proposals. Every
    /// original residual and C1 obligation is checked before returning.
    pub fn enclose_cell(
        &mut self,
        unknown_cell: &[RealBall],
        parameter_cell: &[RealBall],
    ) -> Result<ParametricRootCellEnclosure> {
        self.check_cell(unknown_cell, self.unknown_domain)?;
        self.check_cell(parameter_cell, self.parameter_domain)?;
        self.reserve_evaluations(1)?;
        let n = self.unknowns.len();
        let total = self.variables.len();
        self.work.charge(
            1,
            (n * total * 256) as u64
                + binding_bytes(self.unknowns, unknown_cell)
                + binding_bytes(self.parameters, parameter_cell),
            (n + total + 4) as u64,
        )?;
        let mut product = Vec::with_capacity(total);
        product.extend_from_slice(unknown_cell);
        product.extend_from_slice(parameter_cell);
        let mut residual_values = Vec::with_capacity(n);
        let mut jacobian = Vec::with_capacity(n);
        self.work.nodes = 0;
        for residual in self.residuals {
            let jet = self.work.eval(residual, &self.variables, &product, 0)?;
            residual_values.push(jet.value);
            jacobian.push(jet.gradient);
        }
        self.work.step()?;
        Ok(ParametricRootCellEnclosure {
            residual_values,
            jacobian,
        })
    }

    /// Issue only a local uniform claim on the fixed unknown box. Complete
    /// parameter coverage is checked separately by the independent replay.
    pub fn verify_local(
        &mut self,
        witness: &ParametricRootBoxWitness,
    ) -> Result<ParametricRootBoxReceipt> {
        if witness.precision_digits != self.work.precision {
            return Err(RootBoxError::InvalidPrecision.into());
        }
        if !self.same_domain(&witness.unknown_domain, self.unknown_domain)? {
            return Err(ParametricRootCoverError::InvalidPartition(
                "local proposal changes the fixed unknown box",
            ));
        }
        self.check_cell(&witness.parameter_domain, self.parameter_domain)?;
        self.reserve_evaluations(2)?;
        Ok(verify_parametric_root_box_with_work(
            self.residuals,
            self.unknowns,
            self.parameters,
            witness,
            &mut self.work,
            true,
        )?)
    }

    /// Exact parameter midpoint bisection. Both children retain their shared
    /// face; a fixed parameter coordinate cannot create two fake children.
    pub fn bisect(
        &mut self,
        domain: &[RealBall],
        axis: usize,
    ) -> Result<(Vec<RealBall>, Vec<RealBall>)> {
        self.check_cell(domain, self.parameter_domain)?;
        let coordinate = domain
            .get(axis)
            .ok_or(ParametricRootCoverError::InvalidPartition(
                "split parameter coordinate is out of range",
            ))?;
        if coordinate.radius().is_zero() {
            return Err(ParametricRootCoverError::InvalidPartition(
                "cannot split a fixed parameter coordinate",
            ));
        }
        self.work.charge(
            1,
            2 * binding_bytes(self.parameters, domain),
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

    fn include_image(&mut self, hull: &mut [RealBall], image: &[RealBall]) -> Result<()> {
        let two = BigRational::from_integer(2.into());
        for (hull, image) in hull.iter_mut().zip(image) {
            self.work.arithmetic(&[hull, image], &[])?;
            let lower = hull.lower().min(image.lower());
            let upper = hull.upper().max(image.upper());
            *hull = checked(
                RealBall::new((&lower + &upper) / &two, (upper - lower) / &two)
                    .map_err(RootBoxError::from)?,
            )?;
        }
        Ok(())
    }
}

/// Independently verify every uniform leaf and the entire exact parameter
/// partition. Neither a successful search nor a set of local receipts alone
/// can authorize this complete-cover claim.
pub fn verify_parametric_root_cover(
    residuals: &[Expr],
    unknowns: &[Symbol],
    parameters: &[Symbol],
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
    witness: &ParametricRootCoverWitness,
    meter: &mut impl BudgetMeter,
) -> Result<ParametricRootCoverReceipt> {
    let mut workspace = ParametricRootCoverWorkspace::new(
        residuals,
        unknowns,
        parameters,
        unknown_domain,
        parameter_domain,
        witness.precision_digits,
        meter,
    )?;
    if witness.partition.is_empty()
        || witness.partition.len() > MAX_PARAMETRIC_ROOT_COVER_NODES
        || witness.leaves.is_empty()
        || witness.leaves.len() > MAX_PARAMETRIC_ROOT_COVER_LEAVES
    {
        return Err(ParametricRootCoverError::InvalidPartition(
            "empty cover or excessive node/leaf count",
        ));
    }
    let mut accepted = Vec::new();
    accepted
        .try_reserve_exact(witness.leaves.len())
        .map_err(|_| RootBoxError::ResourceLimit("parameter-cover receipt allocation"))?;
    accepted.resize_with(witness.leaves.len(), || None);
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(MAX_PARAMETRIC_ROOT_COVER_DEPTH + 1)
        .map_err(|_| RootBoxError::ResourceLimit("parameter-cover stack allocation"))?;
    workspace.account(
        1,
        binding_bytes(parameters, parameter_domain),
        parameter_domain.len() as u64 + 1,
    )?;
    pending.push((parameter_domain.to_vec(), 0usize));
    for step in &witness.partition {
        workspace.account(1, 0, 0)?;
        let (cell, depth) = pending
            .pop()
            .ok_or(ParametricRootCoverError::InvalidPartition(
                "trailing partition tokens",
            ))?;
        match *step {
            ParametricRootCoverStep::Split { axis } => {
                if depth >= MAX_PARAMETRIC_ROOT_COVER_DEPTH {
                    return Err(ParametricRootCoverError::InvalidPartition(
                        "partition depth exceeds its bound",
                    ));
                }
                let (left, right) = workspace.bisect(&cell, axis)?;
                pending.push((right, depth + 1));
                pending.push((left, depth + 1));
            }
            ParametricRootCoverStep::Leaf { index } => {
                let proposed =
                    witness
                        .leaves
                        .get(index)
                        .ok_or(ParametricRootCoverError::InvalidPartition(
                            "leaf reference is out of range",
                        ))?;
                if accepted[index].is_some() {
                    return Err(ParametricRootCoverError::InvalidPartition(
                        "a leaf proposal is referenced more than once",
                    ));
                }
                if !workspace.same_domain(&proposed.parameter_domain, &cell)? {
                    return Err(ParametricRootCoverError::InvalidPartition(
                        "leaf parameter box does not match its reconstructed cell",
                    ));
                }
                accepted[index] = Some(workspace.verify_local(proposed)?);
            }
        }
    }
    if !pending.is_empty() {
        return Err(ParametricRootCoverError::InvalidPartition(
            "partition omits one or more parameter regions",
        ));
    }
    let mut leaves = Vec::new();
    leaves
        .try_reserve_exact(accepted.len())
        .map_err(|_| RootBoxError::ResourceLimit("parameter-cover leaf allocation"))?;
    for leaf in accepted {
        leaves.push(leaf.ok_or(ParametricRootCoverError::InvalidPartition(
            "a leaf proposal is not referenced by the cover",
        ))?);
    }
    workspace.account(
        1,
        binding_bytes(unknowns, leaves[0].root_enclosure()),
        unknowns.len() as u64 + 1,
    )?;
    let mut root_enclosure = leaves[0].root_enclosure().to_vec();
    for leaf in &leaves[1..] {
        workspace.include_image(&mut root_enclosure, leaf.root_enclosure())?;
    }
    let witness_bytes = witness
        .leaves
        .iter()
        .map(|leaf| {
            binding_bytes(unknowns, &leaf.unknown_domain)
                + binding_bytes(parameters, &leaf.parameter_domain)
                + leaf
                    .center
                    .iter()
                    .chain(leaf.preconditioner.iter().flatten())
                    .map(|value| 128 + 16 * rational_limbs(value))
                    .sum::<u64>()
        })
        .sum::<u64>();
    workspace.work.charge(
        1,
        workspace.work.source_bytes
            + binding_bytes(unknowns, unknown_domain)
            + binding_bytes(parameters, parameter_domain)
            + witness_bytes
            + (witness.partition.len() * std::mem::size_of::<ParametricRootCoverStep>()) as u64,
        (witness.leaves.len() * (unknowns.len() + parameters.len() + 4) + 8) as u64,
    )?;
    let receipt = ParametricRootCoverReceipt {
        residuals: residuals.to_vec(),
        unknowns: unknowns.to_vec(),
        parameters: parameters.to_vec(),
        unknown_domain: unknown_domain.to_vec(),
        parameter_domain: parameter_domain.to_vec(),
        witness: witness.clone(),
        leaves,
        root_enclosure,
    };
    workspace.account(0, 0, 0)?;
    Ok(receipt)
}

/// Convenience checking retains all intrinsic phase bounds without an
/// external ledger. It does not search or publish partial parameter coverage.
pub fn certify_parametric_root_cover(
    residuals: &[Expr],
    unknowns: &[Symbol],
    parameters: &[Symbol],
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
    witness: &ParametricRootCoverWitness,
) -> Result<ParametricRootCoverReceipt> {
    verify_parametric_root_cover(
        residuals,
        unknowns,
        parameters,
        unknown_domain,
        parameter_domain,
        witness,
        &mut Unbounded,
    )
}
