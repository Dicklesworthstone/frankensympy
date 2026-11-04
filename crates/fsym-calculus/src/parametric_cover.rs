//! Complete parameter-cover proposals for a fixed unknown root domain.
//!
//! A failed uniform Krawczyk proposal may be retried on both halves of a
//! parameter interval. Every accepted leaf keeps the same unknown box, so
//! uniqueness agrees on the shared parameter faces. Only the independent
//! core checker can accept the complete cover; this generator never returns
//! a partial list when a parameter cell remains unresolved.

use fsym_budget::BudgetMeter;
use fsym_core::{
    BallError, BigRational, Expr, MAX_PARAMETRIC_ROOT_COVER_DEPTH,
    MAX_PARAMETRIC_ROOT_COVER_LEAVES, ParametricRootBoxWitness, ParametricRootCoverError,
    ParametricRootCoverStep, ParametricRootCoverWitness, ParametricRootCoverWorkspace, RealBall,
    RootBoxError, Symbol,
};
use num_traits::{One, Zero};

type Result<T> = std::result::Result<T, ParametricRootCoverError>;

/// Search bounds never weaken a leaf's full-parameter proof obligations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParametricRootCoverOptions {
    pub precision_digits: u32,
    pub max_leaves: usize,
}

impl Default for ParametricRootCoverOptions {
    fn default() -> Self {
        Self {
            precision_digits: 32,
            max_leaves: MAX_PARAMETRIC_ROOT_COVER_LEAVES,
        }
    }
}

struct Cell {
    parameters: Vec<RealBall>,
    depth: usize,
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

fn copy_rational(
    workspace: &mut ParametricRootCoverWorkspace<'_, impl BudgetMeter>,
    value: &BigRational,
) -> Result<BigRational> {
    workspace.account(1, rational_bytes(value), 2)?;
    Ok(value.clone())
}

fn copy_domain(
    workspace: &mut ParametricRootCoverWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
) -> Result<Vec<RealBall>> {
    workspace.account(
        domain.len() as u64 + 1,
        domain
            .iter()
            .map(|value| rational_bytes(value.midpoint()) + rational_bytes(value.radius()))
            .sum(),
        4 * domain.len() as u64 + 1,
    )?;
    let mut result = reserve(domain.len(), "parameter-cover box allocation")?;
    result.extend(domain.iter().cloned());
    Ok(result)
}

fn midpoint_box(
    workspace: &mut ParametricRootCoverWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
) -> Result<Vec<RealBall>> {
    workspace.account(domain.len() as u64 + 1, (domain.len() * 256) as u64, 1)?;
    let mut result = reserve(domain.len(), "parameter-cover point allocation")?;
    for coordinate in domain {
        result.push(RealBall::exact(copy_rational(
            workspace,
            coordinate.midpoint(),
        )?));
    }
    Ok(result)
}

fn widest_positive_axis(
    workspace: &mut ParametricRootCoverWorkspace<'_, impl BudgetMeter>,
    parameters: &[RealBall],
) -> Result<Option<usize>> {
    let mut widest: Option<usize> = None;
    let zero = BigRational::zero();
    for (axis, coordinate) in parameters.iter().enumerate() {
        if !workspace.compare(coordinate.radius(), &zero)?.is_gt() {
            continue;
        }
        if let Some(previous) = widest
            && !workspace
                .compare(coordinate.radius(), parameters[previous].radius())?
                .is_gt()
        {
            continue;
        }
        widest = Some(axis);
    }
    Ok(widest)
}

fn domain_may_refine(error: &ParametricRootCoverError) -> bool {
    matches!(
        error,
        ParametricRootCoverError::Verification(
            RootBoxError::Domain(_)
                | RootBoxError::Ball(
                    BallError::DivisionByZero(_) | BallError::NonRealSquareRoot(_)
                )
        )
    )
}

fn proposal_may_refine(error: &ParametricRootCoverError) -> bool {
    domain_may_refine(error)
        || matches!(
            error,
            ParametricRootCoverError::Verification(
                RootBoxError::NoInteriorInclusion { .. }
                    | RootBoxError::NotContractive
                    | RootBoxError::SingularPreconditioner
            )
        )
}

/// Exact elimination proposes C from the unknown columns only. Its result
/// has no authority over the full parameter cell or its original source.
fn inverse_unknown_midpoint(
    workspace: &mut ParametricRootCoverWorkspace<'_, impl BudgetMeter>,
    jacobian: &[Vec<RealBall>],
) -> Result<Vec<Vec<BigRational>>> {
    let n = jacobian.len();
    workspace.account(
        (n * n + 1) as u64,
        (n * n * 256 + n * std::mem::size_of::<Vec<BigRational>>()) as u64,
        (n + 1) as u64,
    )?;
    let mut matrix = reserve(n, "parameter-cover inverse allocation")?;
    for (i, row) in jacobian.iter().enumerate() {
        let mut augmented = reserve(2 * n, "parameter-cover inverse row allocation")?;
        for value in row.iter().take(n) {
            augmented.push(copy_rational(workspace, value.midpoint())?);
        }
        augmented.extend((0..n).map(|j| {
            if i == j {
                BigRational::one()
            } else {
                BigRational::zero()
            }
        }));
        matrix.push(augmented);
    }
    for column in 0..n {
        workspace.account(n as u64 + 1, 0, 0)?;
        let Some(pivot) = (column..n).find(|&row| !matrix[row][column].is_zero()) else {
            return Err(RootBoxError::SingularPreconditioner.into());
        };
        matrix.swap(column, pivot);
        let divisor = copy_rational(workspace, &matrix[column][column])?;
        for entry in &mut matrix[column] {
            *entry = workspace.divide(entry, &divisor)?;
        }
        for row in 0..n {
            if row == column || matrix[row][column].is_zero() {
                continue;
            }
            let factor = copy_rational(workspace, &matrix[row][column])?;
            let (target, pivot) = if row < column {
                let (before, from_column) = matrix.split_at_mut(column);
                (&mut before[row], &from_column[0])
            } else {
                let (before, from_row) = matrix.split_at_mut(row);
                (&mut from_row[0], &before[column])
            };
            for (entry, pivot_entry) in target.iter_mut().zip(pivot) {
                let product = workspace.multiply(&factor, pivot_entry)?;
                *entry = workspace.subtract(entry, &product)?;
            }
        }
    }
    workspace.account((n * n + 1) as u64, 0, 0)?;
    for row in &mut matrix {
        drop(row.drain(..n));
    }
    Ok(matrix)
}

fn candidate_leaf(
    workspace: &mut ParametricRootCoverWorkspace<'_, impl BudgetMeter>,
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
    precision_digits: u32,
) -> Result<ParametricRootBoxWitness> {
    let unknown_point = midpoint_box(workspace, unknown_domain)?;
    let parameter_point = midpoint_box(workspace, parameter_domain)?;
    let enclosure = workspace.enclose_cell(&unknown_point, &parameter_point)?;
    let preconditioner = inverse_unknown_midpoint(workspace, enclosure.jacobian())?;
    workspace.account(
        unknown_domain.len() as u64 + 1,
        (unknown_domain.len() * 128) as u64,
        1,
    )?;
    let mut center = reserve(unknown_domain.len(), "parameter-cover center allocation")?;
    for coordinate in unknown_domain {
        center.push(copy_rational(workspace, coordinate.midpoint())?);
    }
    let witness = ParametricRootBoxWitness {
        unknown_domain: copy_domain(workspace, unknown_domain)?,
        parameter_domain: copy_domain(workspace, parameter_domain)?,
        center,
        preconditioner,
        precision_digits,
    };
    workspace.verify_local(&witness)?;
    Ok(witness)
}

/// Propose a complete cover proving one root in the same X for every p in P.
///
/// Only parameter coordinates are subdivided. A fixed parameter coordinate
/// is retained unchanged, and a tied width selects the first varying axis.
/// Wide unknown boxes, singularities, boundary roots and exhausted limits
/// can remain inconclusive. Unsupported or operational failures propagate
/// without being converted to mathematical retries. Successful proposals
/// must still pass the independent complete-cover verifier.
pub fn propose_parametric_root_cover(
    residuals: &[Expr],
    unknowns: &[Symbol],
    parameters: &[Symbol],
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
    options: ParametricRootCoverOptions,
    meter: &mut impl BudgetMeter,
) -> Result<ParametricRootCoverWitness> {
    meter.checkpoint().map_err(RootBoxError::from)?;
    if !(1..=MAX_PARAMETRIC_ROOT_COVER_LEAVES).contains(&options.max_leaves) {
        return Err(RootBoxError::ResourceLimit(
            "requested parameter-cover leaf count must be 1 through 1024",
        )
        .into());
    }
    let mut workspace = ParametricRootCoverWorkspace::new(
        residuals,
        unknowns,
        parameters,
        unknown_domain,
        parameter_domain,
        options.precision_digits,
        meter,
    )?;
    let node_count = 2 * options.max_leaves - 1;
    workspace.account(
        1,
        (node_count * std::mem::size_of::<ParametricRootCoverStep>()
            + options.max_leaves * std::mem::size_of::<ParametricRootBoxWitness>()
            + (MAX_PARAMETRIC_ROOT_COVER_DEPTH + 1) * std::mem::size_of::<Cell>()) as u64,
        3,
    )?;
    let mut partition = reserve(node_count, "parameter-cover partition allocation")?;
    let mut leaves = reserve(options.max_leaves, "parameter-cover leaves allocation")?;
    let mut pending = reserve(
        MAX_PARAMETRIC_ROOT_COVER_DEPTH + 1,
        "parameter-cover traversal allocation",
    )?;
    pending.push(Cell {
        parameters: copy_domain(&mut workspace, parameter_domain)?,
        depth: 0,
    });
    let mut leaf_count = 1;
    while let Some(cell) = pending.pop() {
        workspace.account(1, 0, 0)?;
        let failure = match candidate_leaf(
            &mut workspace,
            unknown_domain,
            &cell.parameters,
            options.precision_digits,
        ) {
            Ok(witness) => {
                partition.push(ParametricRootCoverStep::Leaf {
                    index: leaves.len(),
                });
                leaves.push(witness);
                continue;
            }
            Err(error) if proposal_may_refine(&error) => error,
            Err(error) => return Err(error),
        };
        if leaf_count >= options.max_leaves || cell.depth >= MAX_PARAMETRIC_ROOT_COVER_DEPTH {
            return Err(if domain_may_refine(&failure) {
                failure
            } else {
                ParametricRootCoverError::Incomplete(
                    "unresolved parameter cell at the subdivision limit",
                )
            });
        }
        let Some(axis) = widest_positive_axis(&mut workspace, &cell.parameters)? else {
            return Err(if domain_may_refine(&failure) {
                failure
            } else {
                ParametricRootCoverError::Incomplete(
                    "unresolved parameter cell has no varying coordinate",
                )
            });
        };
        let (left, right) = workspace.bisect(&cell.parameters, axis)?;
        partition.push(ParametricRootCoverStep::Split { axis });
        // LIFO traversal emits exact canonical preorder: parent, left, right.
        pending.push(Cell {
            parameters: right,
            depth: cell.depth + 1,
        });
        pending.push(Cell {
            parameters: left,
            depth: cell.depth + 1,
        });
        leaf_count += 1;
    }
    let witness = ParametricRootCoverWitness {
        leaves,
        partition,
        precision_digits: options.precision_digits,
    };
    workspace.account(1, 0, 0)?;
    Ok(witness)
}
