//! Normalized, untrusted continuation of a bounded optimization search.
//!
//! Only complete dyadic coverage, a feasible-point proposal and consumed work
//! cross this boundary. Cached values, derivatives, priorities and accepted
//! receipts are reconstructed or independently verified, never deserialized.

use super::*;

/// A candidate continuation, not a mathematical receipt or budget allowance.
///
/// The partition is a complete preorder dyadic cover. The split order contains
/// the preorder index of each split in its original creation order; this keeps
/// deterministic equal-bound heap priorities without storing process indices
/// or cached bounds. The optional incumbent is reevaluated from the source.
///
/// Portable transports must bind the receiver's original source, ordered typed
/// coordinates, domain, extremum kind, tolerance and assumption context before
/// restoring. They must bound sequence lengths and rational limbs before
/// decoding: this type intentionally provides no unrestricted deserializer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoxOptimizationSearchCheckpoint {
    pub options: BoxOptimizationOptions,
    pub partition: Vec<BoxPartitionStep>,
    pub split_order: Vec<usize>,
    pub incumbent_point: Option<Vec<BigRational>>,
    pub counters: BoxOptimizationCounters,
}

fn snapshot_topology(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    nodes: &[Node],
    leaf_count: usize,
) -> Result<(Vec<BoxPartitionStep>, Vec<usize>)> {
    workspace.account(
        1,
        (nodes.len() * std::mem::size_of::<BoxPartitionStep>()
            + (leaf_count - 1 + MAX_BOX_OPTIMIZATION_DEPTH + 1)
                * std::mem::size_of::<usize>()) as u64,
        3,
    )?;
    let mut partition = reserve(nodes.len(), "optimization checkpoint partition allocation")?;
    let mut split_order = reserve(
        leaf_count - 1,
        "optimization checkpoint split-order allocation",
    )?;
    split_order.resize(leaf_count - 1, usize::MAX);
    let mut pending = reserve(
        MAX_BOX_OPTIMIZATION_DEPTH + 1,
        "optimization checkpoint traversal allocation",
    )?;
    pending.push(0usize);
    while let Some(index) = pending.pop() {
        workspace.account(1, 0, 0)?;
        match nodes.get(index) {
            Some(Node::Leaf(_)) => partition.push(BoxPartitionStep::Leaf),
            Some(Node::Split { axis, left, right }) => {
                // Children are appended together by a completed transition.
                let Some(order) = left.checked_sub(1).map(|value| value / 2) else {
                    return Err(BoxOptimizationError::InvalidPartition(
                        "internal checkpoint split order",
                    ));
                };
                let Some(position) = split_order.get_mut(order) else {
                    return Err(BoxOptimizationError::InvalidPartition(
                        "internal checkpoint split order",
                    ));
                };
                if *position != usize::MAX || *left % 2 != 1 || *right != left + 1 {
                    return Err(BoxOptimizationError::InvalidPartition(
                        "internal checkpoint split order",
                    ));
                }
                *position = partition.len();
                partition.push(BoxPartitionStep::Split { axis: *axis });
                pending.push(*right);
                pending.push(*left);
            }
            None => {
                return Err(BoxOptimizationError::InvalidPartition(
                    "internal checkpoint tree index",
                ));
            }
        }
    }
    if partition.len() != nodes.len() || split_order.contains(&usize::MAX) {
        return Err(BoxOptimizationError::InvalidPartition(
            "internal checkpoint incomplete topology",
        ));
    }
    Ok((partition, split_order))
}

/// Reconstruct subtree extents and chronological node identities from bounded
/// mathematical topology. No index, parent relationship or coverage supplied
/// by the checkpoint is trusted. Every split must occur after its parent.
fn topology(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    checkpoint: &BoxOptimizationSearchCheckpoint,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let partition = &checkpoint.partition;
    workspace.account(
        partition.len() as u64,
        (2 * partition.len() * std::mem::size_of::<usize>()) as u64,
        2,
    )?;
    let mut spans = reserve(partition.len(), "optimization checkpoint subtree allocation")?;
    spans.resize(partition.len(), 0usize);
    for index in (0..partition.len()).rev() {
        workspace.account(1, 0, 0)?;
        spans[index] = match partition[index] {
            BoxPartitionStep::Leaf => 1,
            BoxPartitionStep::Split { .. } => {
                let left = index + 1;
                let Some(&left_span) = spans.get(left) else {
                    return Err(BoxOptimizationError::InvalidPartition(
                        "checkpoint missing partition children",
                    ));
                };
                let right = left + left_span;
                let Some(&right_span) = spans.get(right) else {
                    return Err(BoxOptimizationError::InvalidPartition(
                        "checkpoint missing partition children",
                    ));
                };
                1 + left_span + right_span
            }
        };
    }
    if spans[0] != partition.len() {
        return Err(BoxOptimizationError::InvalidPartition(
            "checkpoint trailing partition nodes",
        ));
    }
    let mut identities = reserve(
        partition.len(),
        "optimization checkpoint node-order allocation",
    )?;
    identities.resize(partition.len(), usize::MAX);
    identities[0] = 0;
    for (order, &index) in checkpoint.split_order.iter().enumerate() {
        workspace.account(1, 0, 0)?;
        if !matches!(partition.get(index), Some(BoxPartitionStep::Split { .. }))
            || identities[index] == usize::MAX
        {
            return Err(BoxOptimizationError::InvalidPartition(
                "checkpoint split order must visit split nodes after their parents",
            ));
        }
        let left = index + 1;
        let right = left + spans[left];
        if identities[left] != usize::MAX || identities[right] != usize::MAX {
            return Err(BoxOptimizationError::InvalidPartition(
                "checkpoint repeats a split",
            ));
        }
        identities[left] = 2 * order + 1;
        identities[right] = 2 * order + 2;
    }
    if identities.contains(&usize::MAX) {
        return Err(BoxOptimizationError::InvalidPartition(
            "checkpoint omits a split",
        ));
    }
    Ok((spans, identities))
}

fn restore_frontier(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
    kind: BoxExtremumKind,
    checkpoint: &BoxOptimizationSearchCheckpoint,
) -> Result<SearchFrontier> {
    let (spans, identities) = topology(workspace, checkpoint)?;
    workspace.account(
        1,
        (checkpoint.partition.len() * std::mem::size_of::<Option<Node>>()
            + (2 * checkpoint.options.max_leaves - 1) * std::mem::size_of::<Node>()
            + checkpoint.options.max_leaves * std::mem::size_of::<usize>()
            + (MAX_BOX_OPTIMIZATION_DEPTH + 1)
                * std::mem::size_of::<(usize, Vec<RealBall>, usize)>()) as u64,
        4,
    )?;
    let mut staged = reserve(
        checkpoint.partition.len(),
        "optimization checkpoint staged-node allocation",
    )?;
    staged.resize_with(checkpoint.partition.len(), || None);
    let mut pending = reserve(
        MAX_BOX_OPTIMIZATION_DEPTH + 1,
        "optimization checkpoint cell traversal allocation",
    )?;
    let mut nodes = reserve(
        2 * checkpoint.options.max_leaves - 1,
        "restored optimization search tree allocation",
    )?;
    let mut heap = reserve(
        checkpoint.options.max_leaves,
        "restored optimization priority heap allocation",
    )?;
    workspace.account(
        domain.len() as u64 + 1,
        domain
            .iter()
            .map(|value| rational_bytes(value.midpoint()) + rational_bytes(value.radius()))
            .sum(),
        4 * domain.len() as u64 + 1,
    )?;
    let mut root = reserve(domain.len(), "restored optimization root domain allocation")?;
    root.extend(domain.iter().cloned());
    pending.push((0usize, root, 0usize));
    let mut incumbent = None;
    if let Some(point) = &checkpoint.incumbent_point {
        // The saved point may come from an internal cell no longer resident.
        // Recompute it first, preserving the generator's exact tie choice.
        let value = workspace.evaluate_point(point)?;
        consider_point(workspace, point, &value, kind, &mut incumbent)?;
    }
    let mut leaf_count = 0usize;
    while let Some((index, cell, depth)) = pending.pop() {
        workspace.account(1, 0, 0)?;
        let node = match checkpoint.partition[index] {
            BoxPartitionStep::Leaf => {
                leaf_count += 1;
                Node::Leaf(evaluate_cell(workspace, cell, depth, kind, &mut incumbent)?)
            }
            BoxPartitionStep::Split { axis } => {
                if depth >= MAX_BOX_OPTIMIZATION_DEPTH {
                    return Err(BoxOptimizationError::InvalidPartition(
                        "checkpoint partition depth",
                    ));
                }
                let (left_domain, right_domain) = workspace.bisect(&cell, axis)?;
                let left = index + 1;
                let right = left + spans[left];
                pending.push((right, right_domain, depth + 1));
                pending.push((left, left_domain, depth + 1));
                Node::Split {
                    axis,
                    left: identities[left],
                    right: identities[right],
                }
            }
        };
        staged[identities[index]] = Some(node);
    }
    for node in staged {
        workspace.account(1, 0, 0)?;
        nodes.push(node.ok_or(BoxOptimizationError::InvalidPartition(
            "checkpoint missing topology node",
        ))?);
    }
    for index in 0..nodes.len() {
        workspace.account(1, 0, 0)?;
        if matches!(nodes[index], Node::Leaf(_)) {
            heap_push(workspace, &nodes, &mut heap, index, kind)?;
        }
    }
    Ok(SearchFrontier {
        nodes,
        heap,
        incumbent,
        leaf_count,
        gap: None,
        witness: None,
    })
}

impl BoxOptimizationSearch {
    /// Snapshot a live candidate at a completed transition. Consumed work
    /// includes this copy. A failed search can never export its possibly
    /// incomplete private transition; failed snapshot copying leaves a live
    /// frontier unchanged while retaining every successful resource charge.
    pub fn checkpoint(
        &mut self,
        meter: &mut impl BudgetMeter,
    ) -> Result<BoxOptimizationSearchCheckpoint> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let (partition, split_order, incumbent_point) =
            self.prepared.with_workspace(meter, |workspace| {
                let (partition, split_order) = snapshot_topology(
                    workspace,
                    &self.frontier.nodes,
                    self.frontier.leaf_count,
                )?;
                let point = self
                    .frontier
                    .incumbent
                    .as_ref()
                    .map(|incumbent| copy_point(workspace, &incumbent.point))
                    .transpose()?;
                Ok((partition, split_order, point))
            })?;
        Ok(BoxOptimizationSearchCheckpoint {
            options: self.options,
            partition,
            split_order,
            incumbent_point,
            counters: self.prepared.counters(),
        })
    }

    /// Restore a candidate against the receiver's exact original statement.
    ///
    /// Every leaf and the incumbent are reevaluated; completion is derived
    /// anew. A complete candidate still needs the independent core verifier.
    /// Prior intrinsic work is added to preparation and replay consumption,
    /// never substituted. Structural lower bounds catch impossible history,
    /// but an unauthenticated checkpoint cannot attest all remote attempts.
    /// Failures publish no partial search and leave the checkpoint untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn from_checkpoint(
        objective: &Expr,
        variables: &[Symbol],
        domain: &[RealBall],
        kind: BoxExtremumKind,
        absolute_tolerance: &BigRational,
        checkpoint: &BoxOptimizationSearchCheckpoint,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self> {
        meter.checkpoint().map_err(RootBoxError::from)?;
        check_options(checkpoint.options)?;
        let nodes = checkpoint.partition.len();
        if nodes == 0
            || nodes > 2 * checkpoint.options.max_leaves - 1
            || checkpoint.split_order.len() > checkpoint.options.max_leaves - 1
            || checkpoint.split_order.len() * 2 + 1 != nodes
            || checkpoint
                .incumbent_point
                .as_ref()
                .is_some_and(|point| point.len() != variables.len())
        {
            return Err(BoxOptimizationError::InvalidPartition(
                "checkpoint header disagrees with its bounded search options",
            ));
        }
        let mut prepared = BoxOptimizationPrepared::new(
            objective,
            variables,
            domain,
            checkpoint.options.precision_digits,
            meter,
        )?;
        let baseline = prepared.counters();
        // Each created node performed at least one source evaluation and
        // each committed snapshot visited every node after preparation.
        if checkpoint.counters.steps < baseline.steps + nodes as u64
            || checkpoint.counters.evaluations < nodes
        {
            return Err(BoxOptimizationError::InvalidPartition(
                "checkpoint understates committed traversal work",
            ));
        }
        prepared.include_prior_work(checkpoint.counters)?;
        let (absolute_tolerance, frontier) = prepared.with_workspace(meter, |workspace| {
            if !workspace
                .compare(absolute_tolerance, &BigRational::zero())?
                .is_gt()
            {
                return Err(BoxOptimizationError::InvalidTolerance);
            }
            workspace.account(1, rational_bytes(absolute_tolerance), 2)?;
            let absolute_tolerance = absolute_tolerance.clone();
            let mut frontier = restore_frontier(workspace, domain, kind, checkpoint)?;
            frontier.settle(
                workspace,
                kind,
                &absolute_tolerance,
                checkpoint.options,
            )?;
            Ok((absolute_tolerance, frontier))
        })?;
        Ok(Self {
            prepared,
            frontier,
            kind,
            absolute_tolerance,
            options: checkpoint.options,
            failure: None,
        })
    }
}
