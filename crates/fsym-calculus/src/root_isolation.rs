//! Complete-cover proposals for bounded real nonlinear root isolation.
//!
//! Exact interval exclusion and rational midpoint-Jacobian inverses guide a
//! deterministic subdivision. Every retained leaf is either excluded or
//! contained in a proposed unique-root box; such boxes may cross internal
//! subdivision faces. The final independent core checker alone authorizes a
//! complete-root receipt. Exhaustion never returns a partial witness.

use fsym_budget::BudgetMeter;
use fsym_core::{
    BallError, BigRational, Expr, MAX_ROOT_ISOLATION_DEPTH, MAX_ROOT_ISOLATION_LEAVES,
    MAX_ROOT_ISOLATION_ROOTS, RealBall, RootBoxError, RootBoxWitness, RootIsolationError,
    RootIsolationStep, RootIsolationWitness, RootIsolationWorkspace, Symbol,
};
use num_traits::{One, Zero};

type Result<T> = std::result::Result<T, RootIsolationError>;

/// Search limits do not weaken the independent completeness obligations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootIsolationOptions {
    pub precision_digits: u32,
    pub max_leaves: usize,
}

impl Default for RootIsolationOptions {
    fn default() -> Self {
        Self {
            precision_digits: 32,
            max_leaves: MAX_ROOT_ISOLATION_LEAVES,
        }
    }
}

struct Cell {
    domain: Vec<RealBall>,
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
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    value: &BigRational,
) -> Result<BigRational> {
    workspace.account(1, rational_bytes(value), 2)?;
    Ok(value.clone())
}

fn copy_domain(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
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
    let mut result = reserve(domain.len(), "root-isolation box allocation")?;
    result.extend(domain.iter().cloned());
    Ok(result)
}

fn endpoint(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    value: &RealBall,
    upper: bool,
) -> Result<BigRational> {
    if upper {
        workspace.add(value.midpoint(), value.radius())
    } else {
        workspace.subtract(value.midpoint(), value.radius())
    }
}

fn contained(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    inner: &[RealBall],
    outer: &[RealBall],
) -> Result<bool> {
    for (inside, outside) in inner.iter().zip(outer) {
        let inner_lower = endpoint(workspace, inside, false)?;
        let inner_upper = endpoint(workspace, inside, true)?;
        let outer_lower = endpoint(workspace, outside, false)?;
        let outer_upper = endpoint(workspace, outside, true)?;
        if workspace.compare(&inner_lower, &outer_lower)?.is_lt()
            || workspace.compare(&inner_upper, &outer_upper)?.is_gt()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn separated(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    first: &[RealBall],
    second: &[RealBall],
) -> Result<bool> {
    for (a, b) in first.iter().zip(second) {
        let a_lower = endpoint(workspace, a, false)?;
        let a_upper = endpoint(workspace, a, true)?;
        let b_lower = endpoint(workspace, b, false)?;
        let b_upper = endpoint(workspace, b, true)?;
        if workspace.compare(&a_upper, &b_lower)?.is_lt()
            || workspace.compare(&b_upper, &a_lower)?.is_lt()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn covered_by(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
    roots: &[RootBoxWitness],
) -> Result<Option<usize>> {
    for (index, root) in roots.iter().enumerate() {
        if contained(workspace, domain, &root.domain)? {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn excluded(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    residuals: &[RealBall],
) -> Result<bool> {
    // The workspace already evaluated every original residual. A nonzero
    // first residual must never bypass a later residual's domain obligations.
    let zero = BigRational::zero();
    for value in residuals {
        let lower = endpoint(workspace, value, false)?;
        let upper = endpoint(workspace, value, true)?;
        if workspace.compare(&lower, &zero)?.is_gt() || workspace.compare(&upper, &zero)?.is_lt() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn widest_axis(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
) -> Result<usize> {
    let mut widest = 0;
    for axis in 1..domain.len() {
        if workspace
            .compare(domain[axis].radius(), domain[widest].radius())?
            .is_gt()
        {
            widest = axis;
        }
    }
    Ok(widest)
}

fn subdivision_can_resolve(error: &RootIsolationError) -> bool {
    matches!(
        error,
        RootIsolationError::Verification(
            RootBoxError::Domain(_)
                | RootBoxError::Ball(
                    BallError::DivisionByZero(_) | BallError::NonRealSquareRoot(_)
                )
        )
    )
}

fn local_proposal_can_fail(error: &RootIsolationError) -> bool {
    subdivision_can_resolve(error)
        || matches!(
            error,
            RootIsolationError::Verification(
                RootBoxError::NoInteriorInclusion { .. }
                    | RootBoxError::NotContractive
                    | RootBoxError::SingularPreconditioner
            )
        )
}

/// A bounded exact inverse is only a proposal. Neither its construction nor
/// midpoint invertibility proves anything about the full cell Jacobian.
fn inverse_midpoint(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    jacobian: &[Vec<RealBall>],
) -> Result<Option<Vec<Vec<BigRational>>>> {
    let n = jacobian.len();
    workspace.account(
        (n * n + 1) as u64,
        (n * n * 256 + n * std::mem::size_of::<Vec<BigRational>>()) as u64,
        (n + 1) as u64,
    )?;
    let mut matrix = reserve(n, "root-isolation inverse allocation")?;
    for (i, row) in jacobian.iter().enumerate() {
        let mut augmented = reserve(2 * n, "root-isolation inverse row allocation")?;
        for value in row {
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
            return Ok(None);
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
    Ok(Some(matrix))
}

fn candidate_root(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    domain: &[RealBall],
    roots: &[RootBoxWitness],
    precision_digits: u32,
) -> Result<Option<RootBoxWitness>> {
    workspace.account(domain.len() as u64 + 1, (domain.len() * 256) as u64, 2)?;
    let mut center = reserve(domain.len(), "root-isolation center allocation")?;
    let mut point = reserve(domain.len(), "root-isolation point box allocation")?;
    for coordinate in domain {
        center.push(copy_rational(workspace, coordinate.midpoint())?);
        point.push(RealBall::exact(copy_rational(
            workspace,
            coordinate.midpoint(),
        )?));
    }
    let enclosure = match workspace.enclose_cell(&point) {
        Ok(value) => value,
        Err(error) if subdivision_can_resolve(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    let Some(preconditioner) = inverse_midpoint(workspace, enclosure.jacobian())? else {
        return Ok(None);
    };
    let mut witness = RootBoxWitness {
        domain: copy_domain(workspace, domain)?,
        center,
        preconditioner,
        precision_digits,
    };
    for expanded in [false, true] {
        if expanded {
            witness.domain = workspace.expand_cell(domain)?;
        }
        let mut disjoint = true;
        for root in roots {
            if !separated(workspace, &witness.domain, &root.domain)? {
                disjoint = false;
                break;
            }
        }
        if !disjoint {
            continue;
        }
        match workspace.verify_local(&witness) {
            Ok(_) => return Ok(Some(witness)),
            Err(error) if local_proposal_can_fail(&error) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}

/// Propose every nonsingular real root in an exact finite coordinate box.
///
/// The source must be C1 over the complete declared domain. Singular roots,
/// outer-boundary roots, unproved domain cells and exhausted subdivision can
/// remain inconclusive. Only a complete partition is returned, never a list
/// of successful roots with the unresolved regions silently discarded.
pub fn propose_root_isolation(
    residuals: &[Expr],
    variables: &[Symbol],
    domain: &[RealBall],
    options: RootIsolationOptions,
    meter: &mut impl BudgetMeter,
) -> Result<RootIsolationWitness> {
    meter.checkpoint().map_err(RootBoxError::from)?;
    if !(1..=MAX_ROOT_ISOLATION_LEAVES).contains(&options.max_leaves) {
        return Err(RootBoxError::ResourceLimit(
            "requested root-isolation leaf count must be 1 through 1024",
        )
        .into());
    }
    let mut workspace = RootIsolationWorkspace::new(
        residuals,
        variables,
        domain,
        options.precision_digits,
        meter,
    )?;
    let node_count = 2 * options.max_leaves - 1;
    workspace.account(
        1,
        (node_count * std::mem::size_of::<RootIsolationStep>()
            + MAX_ROOT_ISOLATION_ROOTS * std::mem::size_of::<RootBoxWitness>()
            + (MAX_ROOT_ISOLATION_DEPTH + 1) * std::mem::size_of::<Cell>()) as u64,
        3,
    )?;
    let mut partition = reserve(node_count, "root-isolation partition allocation")?;
    let mut roots = reserve(MAX_ROOT_ISOLATION_ROOTS, "root-isolation roots allocation")?;
    let mut pending = reserve(
        MAX_ROOT_ISOLATION_DEPTH + 1,
        "root-isolation traversal allocation",
    )?;
    pending.push(Cell {
        domain: copy_domain(&mut workspace, domain)?,
        depth: 0,
    });
    let mut leaves = 1;
    while let Some(cell) = pending.pop() {
        workspace.account(1, 0, 0)?;
        if let Some(index) = covered_by(&mut workspace, &cell.domain, &roots)? {
            partition.push(RootIsolationStep::Root { index });
            continue;
        }
        let domain_failure = match workspace.enclose_cell(&cell.domain) {
            Ok(enclosure) => {
                if excluded(&mut workspace, enclosure.residual_values())? {
                    partition.push(RootIsolationStep::Exclude);
                    continue;
                }
                if let Some(witness) = candidate_root(
                    &mut workspace,
                    &cell.domain,
                    &roots,
                    options.precision_digits,
                )? {
                    if roots.len() == MAX_ROOT_ISOLATION_ROOTS {
                        return Err(RootIsolationError::Incomplete("root count limit"));
                    }
                    partition.push(RootIsolationStep::Root { index: roots.len() });
                    roots.push(witness);
                    continue;
                }
                None
            }
            Err(error) if subdivision_can_resolve(&error) => Some(error),
            Err(error) => return Err(error),
        };
        if leaves >= options.max_leaves || cell.depth >= MAX_ROOT_ISOLATION_DEPTH {
            return Err(domain_failure.unwrap_or(RootIsolationError::Incomplete(
                "unresolved cell at the subdivision limit",
            )));
        }
        let axis = widest_axis(&mut workspace, &cell.domain)?;
        let (left, right) = workspace.bisect(&cell.domain, axis)?;
        partition.push(RootIsolationStep::Split { axis });
        // LIFO traversal writes canonical preorder: parent, left, right.
        pending.push(Cell {
            domain: right,
            depth: cell.depth + 1,
        });
        pending.push(Cell {
            domain: left,
            depth: cell.depth + 1,
        });
        leaves += 1;
    }
    let witness = RootIsolationWitness {
        roots,
        partition,
        precision_digits: options.precision_digits,
    };
    workspace.account(1, 0, 0)?;
    Ok(witness)
}
