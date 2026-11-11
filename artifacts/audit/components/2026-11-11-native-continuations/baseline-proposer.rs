//! Deterministic proposals for tolerance-certified global box optimization.
//!
//! Search retains a complete dyadic partition, including regions that cannot
//! improve the current feasible point. It returns only an untrusted witness;
//! the independent core checker must replay every leaf and the original
//! objective at the point before an accepted receipt can be published.

use fsym_budget::BudgetMeter;
use fsym_core::{
    BallError, BigRational, BoxCellEnclosure, BoxExtremumKind, BoxOptimizationError,
    BoxOptimizationWitness, BoxOptimizationWorkspace, BoxPartitionStep, Expr,
    MAX_BOX_OPTIMIZATION_DEPTH, MAX_BOX_OPTIMIZATION_LEAVES, RealBall, RootBoxError, Symbol,
};
use num_traits::{Signed, Zero};
use std::cmp::Ordering;

type Result<T> = std::result::Result<T, BoxOptimizationError>;

/// Absolute accuracy is checked independently from these search limits.
/// Raising the leaf cap never relaxes the requested global value gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxOptimizationOptions {
    pub precision_digits: u32,
    pub max_leaves: usize,
}

impl Default for BoxOptimizationOptions {
    fn default() -> Self {
        Self {
            precision_digits: 32,
            max_leaves: MAX_BOX_OPTIMIZATION_LEAVES,
        }
    }
}

struct Cell {
    domain: Vec<RealBall>,
    depth: usize,
    optimistic_bound: Option<BigRational>,
    split_axis: Option<usize>,
    domain_failure: Option<BoxOptimizationError>,
}

enum Node {
    Leaf(Cell),
    Split {
        axis: usize,
        left: usize,
        right: usize,
    },
}

struct Incumbent {
    point: Vec<BigRational>,
    feasible_bound: BigRational,
}

fn reserve<T>(capacity: usize, label: &'static str) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| RootBoxError::ResourceLimit(label))?;
    Ok(result)
}

fn rational_bytes(value: &BigRational) -> u64 {
    128 + 16 * (value.numer().limb_count() + value.denom().limb_count())
}

fn copy_point(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    point: &[BigRational],
) -> Result<Vec<BigRational>> {
    workspace.account(
        point.len() as u64 + 1,
        point.iter().map(rational_bytes).sum(),
        2 * point.len() as u64 + 1,
    )?;
    let mut result = reserve(point.len(), "optimization candidate point allocation")?;
    result.extend(point.iter().cloned());
    Ok(result)
}

// Endpoint extraction is exact rational arithmetic too. Use the workspace
// operations so its limb-weighted bounds apply before any products/reduction.
fn endpoint(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    value: &RealBall,
    upper: bool,
) -> Result<BigRational> {
    if upper {
        let negative_radius = workspace.difference(&BigRational::zero(), value.radius())?;
        workspace.difference(value.midpoint(), &negative_radius)
    } else {
        workspace.difference(value.midpoint(), value.radius())
    }
}

fn magnitude(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    value: BigRational,
) -> Result<BigRational> {
    if value.is_negative() {
        workspace.difference(&BigRational::zero(), &value)
    } else {
        Ok(value)
    }
}

fn improves(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    candidate: &BigRational,
    previous: &BigRational,
    kind: BoxExtremumKind,
) -> Result<bool> {
    let order = workspace.compare(candidate, previous)?;
    Ok(match kind {
        BoxExtremumKind::Minimum => order.is_lt(),
        BoxExtremumKind::Maximum => order.is_gt(),
    })
}

fn consider_point(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    point: &[BigRational],
    value: &RealBall,
    kind: BoxExtremumKind,
    incumbent: &mut Option<Incumbent>,
) -> Result<()> {
    // An upper endpoint is feasible for minimization; a lower endpoint is
    // feasible for maximization. A rounded midpoint is never an incumbent.
    let feasible_bound = endpoint(workspace, value, kind == BoxExtremumKind::Minimum)?;
    if let Some(previous) = incumbent
        && !improves(workspace, &feasible_bound, &previous.feasible_bound, kind)?
    {
        return Ok(());
    }
    *incumbent = Some(Incumbent {
        point: copy_point(workspace, point)?,
        feasible_bound,
    });
    Ok(())
}

fn widest_axis(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
) -> Result<Option<usize>> {
    let mut widest = None::<usize>;
    for (axis, coordinate) in domain.iter().enumerate() {
        workspace.account(1, 0, 0)?;
        if coordinate.radius().is_zero() {
            continue;
        }
        if let Some(previous) = widest
            && !workspace
                .compare(coordinate.radius(), domain[previous].radius())?
                .is_gt()
        {
            continue;
        }
        widest = Some(axis);
    }
    Ok(widest)
}

/// The split score is radius * sup(abs(gradient)), with a widest-coordinate
/// fallback. The omitted factor two is common to every coordinate. A face
/// proposal moves only coordinates whose whole-cell derivative has a proved
/// sign; derivatives spanning zero leave that coordinate at the center.
fn analyze_cell(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
    enclosure: &BoxCellEnclosure,
    kind: BoxExtremumKind,
) -> Result<(Option<usize>, Option<Vec<BigRational>>)> {
    let widest = widest_axis(workspace, domain)?;
    let mut weighted = None::<(usize, BigRational)>;
    let mut face = None::<Vec<BigRational>>;
    let zero = BigRational::zero();
    for (axis, (coordinate, derivative)) in domain.iter().zip(enclosure.gradient()).enumerate() {
        workspace.account(1, 0, 0)?;
        if coordinate.radius().is_zero() {
            continue;
        }
        let lower = endpoint(workspace, derivative, false)?;
        let upper = endpoint(workspace, derivative, true)?;
        let lower_sign = workspace.compare(&lower, &zero)?;
        let upper_sign = workspace.compare(&upper, &zero)?;
        let increasing = !lower_sign.is_lt() && upper_sign.is_gt();
        let decreasing = !upper_sign.is_gt() && lower_sign.is_lt();
        if increasing || decreasing {
            let upper_face = match kind {
                BoxExtremumKind::Minimum => decreasing,
                BoxExtremumKind::Maximum => increasing,
            };
            if face.is_none() {
                face = Some(copy_point(workspace, enclosure.center())?);
            }
            if let Some(point) = &mut face {
                point[axis] = endpoint(workspace, coordinate, upper_face)?;
            }
        }
        let lower = magnitude(workspace, lower)?;
        let upper = magnitude(workspace, upper)?;
        let largest = if workspace.compare(&lower, &upper)?.is_gt() {
            lower
        } else {
            upper
        };
        let score = workspace.multiply(coordinate.radius(), &largest)?;
        if score.is_zero() {
            continue;
        }
        if let Some((_, previous)) = &weighted
            && !workspace.compare(&score, previous)?.is_gt()
        {
            continue;
        }
        weighted = Some((axis, score));
    }
    Ok((weighted.map(|(axis, _)| axis).or(widest), face))
}

fn subdivision_can_resolve(error: &BoxOptimizationError) -> bool {
    matches!(
        error,
        BoxOptimizationError::Verification(
            RootBoxError::Domain(_)
                | RootBoxError::Ball(
                    BallError::DivisionByZero(_) | BallError::NonRealSquareRoot(_)
                )
        )
    )
}

fn evaluate_cell(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    domain: Vec<RealBall>,
    depth: usize,
    kind: BoxExtremumKind,
    incumbent: &mut Option<Incumbent>,
) -> Result<Cell> {
    match workspace.enclose_cell(&domain) {
        Ok(enclosure) => {
            consider_point(
                workspace,
                enclosure.center(),
                enclosure.center_value(),
                kind,
                incumbent,
            )?;
            let (split_axis, face) = analyze_cell(workspace, &domain, &enclosure, kind)?;
            if let Some(point) = face {
                let value = workspace.evaluate_point(&point)?;
                consider_point(workspace, &point, &value, kind, incumbent)?;
            }
            let optimistic_bound = endpoint(
                workspace,
                enclosure.range(),
                kind == BoxExtremumKind::Maximum,
            )?;
            Ok(Cell {
                domain,
                depth,
                optimistic_bound: Some(optimistic_bound),
                split_axis,
                domain_failure: None,
            })
        }
        Err(error) if subdivision_can_resolve(&error) => {
            let split_axis = widest_axis(workspace, &domain)?;
            Ok(Cell {
                domain,
                depth,
                optimistic_bound: None,
                split_axis,
                domain_failure: Some(error),
            })
        }
        Err(error) => Err(error),
    }
}

fn leaf(nodes: &[Node], index: usize) -> Result<&Cell> {
    match nodes.get(index) {
        Some(Node::Leaf(cell)) => Ok(cell),
        _ => Err(BoxOptimizationError::InvalidPartition(
            "internal proposal leaf index",
        )),
    }
}

// A custom heap meters every rational comparison. std::BinaryHeap's Ord
// implementation could not propagate cancellation or charge its arithmetic.
// Unknown-domain leaves come first, and insertion indices break exact ties.
fn higher_priority(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    nodes: &[Node],
    first: usize,
    second: usize,
    kind: BoxExtremumKind,
) -> Result<bool> {
    workspace.account(1, 0, 0)?;
    let order = match (
        &leaf(nodes, first)?.optimistic_bound,
        &leaf(nodes, second)?.optimistic_bound,
    ) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => {
            let order = workspace.compare(left, right)?;
            match kind {
                BoxExtremumKind::Minimum => order,
                BoxExtremumKind::Maximum => order.reverse(),
            }
        }
    };
    Ok(order.is_lt() || (order.is_eq() && first < second))
}

fn heap_push(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    nodes: &[Node],
    heap: &mut Vec<usize>,
    index: usize,
    kind: BoxExtremumKind,
) -> Result<()> {
    workspace.account(1, 0, 0)?;
    heap.push(index);
    let mut child = heap.len() - 1;
    while child != 0 {
        workspace.account(4, 0, 0)?;
        let parent = (child - 1) / 2;
        if !higher_priority(workspace, nodes, heap[child], heap[parent], kind)? {
            break;
        }
        heap.swap(child, parent);
        child = parent;
    }
    Ok(())
}

fn heap_pop(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    nodes: &[Node],
    heap: &mut Vec<usize>,
    kind: BoxExtremumKind,
) -> Result<()> {
    workspace.account(1, 0, 0)?;
    heap.swap_remove(0);
    let mut parent = 0usize;
    while 2 * parent + 1 < heap.len() {
        workspace.account(4, 0, 0)?;
        let mut child = 2 * parent + 1;
        if child + 1 < heap.len()
            && higher_priority(workspace, nodes, heap[child + 1], heap[child], kind)?
        {
            child += 1;
        }
        if !higher_priority(workspace, nodes, heap[child], heap[parent], kind)? {
            break;
        }
        heap.swap(child, parent);
        parent = child;
    }
    Ok(())
}

fn serialize_partition(
    workspace: &mut BoxOptimizationWorkspace<'_, impl BudgetMeter>,
    nodes: &[Node],
) -> Result<Vec<BoxPartitionStep>> {
    workspace.account(
        1,
        (nodes.len() * std::mem::size_of::<BoxPartitionStep>()
            + (MAX_BOX_OPTIMIZATION_DEPTH + 1) * std::mem::size_of::<usize>()) as u64,
        2,
    )?;
    let mut partition = reserve(nodes.len(), "optimization partition allocation")?;
    let mut pending = reserve(
        MAX_BOX_OPTIMIZATION_DEPTH + 1,
        "optimization traversal allocation",
    )?;
    pending.push(0usize);
    while let Some(index) = pending.pop() {
        workspace.account(1, 0, 0)?;
        match nodes.get(index) {
            Some(Node::Leaf(_)) => partition.push(BoxPartitionStep::Leaf),
            Some(Node::Split { axis, left, right }) => {
                partition.push(BoxPartitionStep::Split { axis: *axis });
                pending.push(*right);
                pending.push(*left);
            }
            None => {
                return Err(BoxOptimizationError::InvalidPartition(
                    "internal proposal tree index",
                ));
            }
        }
    }
    Ok(partition)
}

/// Propose a complete finite-box partition and an exact feasible point.
///
/// Best-first search compares original-source interval bounds to the best
/// feasible point enclosure. A cell with unresolved domain obligations must
/// be subdivided even when another cell already attains a promising value.
/// Cancellation and resource/unsupported errors are never retried, and a
/// failed accuracy or depth limit produces no partial witness. All searches,
/// point proposals and bookkeeping share one cumulative source workspace.
#[allow(clippy::too_many_arguments)]
pub fn propose_box_optimization(
    objective: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    kind: BoxExtremumKind,
    absolute_tolerance: &BigRational,
    options: BoxOptimizationOptions,
    meter: &mut impl BudgetMeter,
) -> Result<BoxOptimizationWitness> {
    meter.checkpoint().map_err(RootBoxError::from)?;
    if !(1..=MAX_BOX_OPTIMIZATION_LEAVES).contains(&options.max_leaves) {
        return Err(RootBoxError::ResourceLimit(
            "requested optimization leaf count must be 1 through 1024",
        )
        .into());
    }
    let mut workspace = BoxOptimizationWorkspace::new(
        objective,
        variables,
        domain,
        options.precision_digits,
        meter,
    )?;
    if !workspace
        .compare(absolute_tolerance, &BigRational::zero())?
        .is_gt()
    {
        return Err(BoxOptimizationError::InvalidTolerance);
    }
    let mut nodes = reserve(
        2 * options.max_leaves - 1,
        "optimization search tree allocation",
    )?;
    let mut heap = reserve(options.max_leaves, "optimization priority heap allocation")?;
    workspace.account(
        domain.len() as u64 + 1,
        domain
            .iter()
            .map(|value| rational_bytes(value.midpoint()) + rational_bytes(value.radius()))
            .sum(),
        4 * domain.len() as u64 + 1,
    )?;
    let mut incumbent = None;
    let first = evaluate_cell(&mut workspace, domain.to_vec(), 0, kind, &mut incumbent)?;
    nodes.push(Node::Leaf(first));
    heap.push(0);
    let mut leaf_count = 1usize;
    loop {
        workspace.account(1, 0, 0)?;
        let selected = *heap.first().ok_or(BoxOptimizationError::InvalidPartition(
            "internal empty proposal heap",
        ))?;
        let cell = leaf(&nodes, selected)?;
        if let (Some(bound), Some(best)) = (&cell.optimistic_bound, &incumbent) {
            let gap = match kind {
                BoxExtremumKind::Minimum => workspace.difference(&best.feasible_bound, bound)?,
                BoxExtremumKind::Maximum => workspace.difference(bound, &best.feasible_bound)?,
            };
            if gap.is_negative() {
                return Err(BoxOptimizationError::InconsistentEnclosure);
            }
            if !workspace.compare(&gap, absolute_tolerance)?.is_gt() {
                let partition = serialize_partition(&mut workspace, &nodes)?;
                let best = incumbent.ok_or(BoxOptimizationError::InconsistentEnclosure)?;
                let witness = BoxOptimizationWitness {
                    partition,
                    point: best.point,
                    precision_digits: options.precision_digits,
                };
                workspace.account(1, 0, 0)?;
                return Ok(witness);
            }
        }
        if leaf_count >= options.max_leaves
            || cell.depth >= MAX_BOX_OPTIMIZATION_DEPTH
            || cell.split_axis.is_none()
        {
            return Err(cell
                .domain_failure
                .clone()
                .unwrap_or(BoxOptimizationError::AccuracyNotReached));
        }
        let axis = cell
            .split_axis
            .ok_or(BoxOptimizationError::AccuracyNotReached)?;
        let depth = cell.depth + 1;
        let (left_domain, right_domain) = workspace.bisect(&cell.domain, axis)?;
        heap_pop(&mut workspace, &nodes, &mut heap, kind)?;
        let left = evaluate_cell(&mut workspace, left_domain, depth, kind, &mut incumbent)?;
        let right = evaluate_cell(&mut workspace, right_domain, depth, kind, &mut incumbent)?;
        let left_index = nodes.len();
        let right_index = left_index + 1;
        nodes.push(Node::Leaf(left));
        nodes.push(Node::Leaf(right));
        nodes[selected] = Node::Split {
            axis,
            left: left_index,
            right: right_index,
        };
        heap_push(&mut workspace, &nodes, &mut heap, left_index, kind)?;
        heap_push(&mut workspace, &nodes, &mut heap, right_index, kind)?;
        leaf_count += 1;
    }
}
