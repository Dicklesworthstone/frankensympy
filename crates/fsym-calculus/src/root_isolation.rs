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
    MAX_ROOT_ISOLATION_NODES, MAX_ROOT_ISOLATION_ROOTS, RealBall, RootBoxError, RootBoxWitness,
    RootIsolationError, RootIsolationPrepared, RootIsolationStep, RootIsolationWitness,
    RootIsolationWorkspace, Symbol,
};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

pub use fsym_core::RootIsolationCounters;

type Result<T> = std::result::Result<T, RootIsolationError>;

/// Search limits do not weaken the independent completeness obligations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

/// One advance can request at most the complete bounded traversal.
pub const MAX_ROOT_ISOLATION_ADVANCE_CELLS: usize = MAX_ROOT_ISOLATION_NODES;

/// Candidate search progress, never an accepted root count or completeness
/// receipt. `ready_for_verification` means only that no search cells remain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RootIsolationProgress {
    pub processed_cells: usize,
    pub pending_cells: usize,
    pub candidate_root_count: usize,
    pub excluded_cells: usize,
    pub leaf_count: usize,
    pub ready_for_verification: bool,
}

/// Normalized, untrusted continuation of an incremental root search.
/// Pending cells are deliberately absent: the independent core checker
/// reconstructs them from this exact dyadic prefix and the expected domain.
/// Portable transports must additionally bind the original source, ordered
/// typed coordinates, domain and context before calling `from_checkpoint`.
/// The counters are consumed work, never proof of remote history or budget
/// allowances. This object and all restored progress remain candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootIsolationSearchCheckpoint {
    pub options: RootIsolationOptions,
    pub witness: RootIsolationWitness,
    pub counters: RootIsolationCounters,
}

struct Cell {
    domain: Vec<RealBall>,
    depth: usize,
}

struct SearchFrontier {
    witness: RootIsolationWitness,
    pending: Vec<Cell>,
    leaves: usize,
    excluded: usize,
}

enum CellAction {
    Covered {
        index: usize,
    },
    Excluded,
    Root(RootBoxWitness),
    Split {
        axis: usize,
        left: Vec<RealBall>,
        right: Vec<RealBall>,
    },
}

fn check_options(options: RootIsolationOptions) -> Result<()> {
    if !(1..=MAX_ROOT_ISOLATION_LEAVES).contains(&options.max_leaves) {
        return Err(RootBoxError::ResourceLimit(
            "requested root-isolation leaf count must be 1 through 1024",
        )
        .into());
    }
    Ok(())
}

impl SearchFrontier {
    fn new(
        workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
        domain: &[RealBall],
        options: RootIsolationOptions,
    ) -> Result<Self> {
        let node_count = 2 * options.max_leaves - 1;
        workspace.account(
            1,
            (node_count * std::mem::size_of::<RootIsolationStep>()
                + MAX_ROOT_ISOLATION_ROOTS * std::mem::size_of::<RootBoxWitness>()
                + (MAX_ROOT_ISOLATION_DEPTH + 1) * std::mem::size_of::<Cell>()) as u64,
            3,
        )?;
        let partition = reserve(node_count, "root-isolation partition allocation")?;
        let roots = reserve(MAX_ROOT_ISOLATION_ROOTS, "root-isolation roots allocation")?;
        let mut pending = reserve(
            MAX_ROOT_ISOLATION_DEPTH + 1,
            "root-isolation traversal allocation",
        )?;
        pending.push(Cell {
            domain: copy_domain(workspace, domain)?,
            depth: 0,
        });
        Ok(Self {
            witness: RootIsolationWitness {
                roots,
                partition,
                precision_digits: options.precision_digits,
            },
            pending,
            leaves: 1,
            excluded: 0,
        })
    }

    fn progress(&self) -> RootIsolationProgress {
        RootIsolationProgress {
            processed_cells: self.witness.partition.len(),
            pending_cells: self.pending.len(),
            candidate_root_count: self.witness.roots.len(),
            excluded_cells: self.excluded,
            leaf_count: self.leaves,
            ready_for_verification: self.pending.is_empty(),
        }
    }

    fn prepare_cell(
        &self,
        cell: &Cell,
        options: RootIsolationOptions,
        workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    ) -> Result<CellAction> {
        workspace.account(1, 0, 0)?;
        if let Some(index) = covered_by(workspace, &cell.domain, &self.witness.roots)? {
            return Ok(CellAction::Covered { index });
        }
        let domain_failure = match workspace.enclose_cell(&cell.domain) {
            Ok(enclosure) => {
                if excluded(workspace, enclosure.residual_values())? {
                    return Ok(CellAction::Excluded);
                }
                if let Some(witness) = candidate_root(
                    workspace,
                    &cell.domain,
                    &self.witness.roots,
                    options.precision_digits,
                )? {
                    if self.witness.roots.len() == MAX_ROOT_ISOLATION_ROOTS {
                        return Err(RootIsolationError::Incomplete("root count limit"));
                    }
                    return Ok(CellAction::Root(witness));
                }
                None
            }
            Err(error) if subdivision_can_resolve(&error) => Some(error),
            Err(error) => return Err(error),
        };
        if self.leaves >= options.max_leaves || cell.depth >= MAX_ROOT_ISOLATION_DEPTH {
            return Err(domain_failure.unwrap_or(RootIsolationError::Incomplete(
                "unresolved cell at the subdivision limit",
            )));
        }
        let axis = widest_axis(workspace, &cell.domain)?;
        let (left, right) = workspace.bisect(&cell.domain, axis)?;
        Ok(CellAction::Split { axis, left, right })
    }

    fn advance(
        &mut self,
        workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
        options: RootIsolationOptions,
        max_cells: usize,
    ) -> Result<RootIsolationProgress> {
        for _ in 0..max_cells {
            // Peek until every fallible operation is finished. A refusal
            // retains this cell and all previously completed cover work.
            let Some(cell) = self.pending.last() else {
                break;
            };
            let action = self.prepare_cell(cell, options, workspace)?;
            let depth = cell.depth;
            workspace.account(1, 0, 0)?;
            // Capacities were reserved for the maximum bounded traversal.
            // No allocation, fallible work or cancelling safe point follows
            // until this complete cell transition has been published.
            self.pending.pop();
            match action {
                CellAction::Covered { index } => {
                    self.witness
                        .partition
                        .push(RootIsolationStep::Root { index });
                }
                CellAction::Excluded => {
                    self.witness.partition.push(RootIsolationStep::Exclude);
                    self.excluded += 1;
                }
                CellAction::Root(witness) => {
                    self.witness.partition.push(RootIsolationStep::Root {
                        index: self.witness.roots.len(),
                    });
                    self.witness.roots.push(witness);
                }
                CellAction::Split { axis, left, right } => {
                    self.witness
                        .partition
                        .push(RootIsolationStep::Split { axis });
                    // LIFO traversal retains canonical preorder: left, right.
                    self.pending.push(Cell {
                        domain: right,
                        depth: depth + 1,
                    });
                    self.pending.push(Cell {
                        domain: left,
                        depth: depth + 1,
                    });
                    self.leaves += 1;
                }
            }
        }
        Ok(self.progress())
    }
}

/// An owned, bounded candidate search that can yield between cell batches.
///
/// Source, coordinates, exact domain and options are frozen at construction.
/// Partial progress contains no accepted root handles. Only the independent
/// core complete-cover checker can turn the completed witness into a receipt.
/// Every call uses the caller's same cumulative ledger; intrinsic arithmetic,
/// allocation and evaluation limits also persist across calls and refusals.
pub struct RootIsolationSearch {
    prepared: RootIsolationPrepared,
    frontier: SearchFrontier,
    options: RootIsolationOptions,
}

impl RootIsolationSearch {
    pub fn new(
        residuals: &[Expr],
        variables: &[Symbol],
        domain: &[RealBall],
        options: RootIsolationOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self> {
        meter.checkpoint().map_err(RootBoxError::from)?;
        check_options(options)?;
        let mut prepared = RootIsolationPrepared::new(
            residuals,
            variables,
            domain,
            options.precision_digits,
            meter,
        )?;
        let frontier = prepared.with_workspace(meter, |workspace| {
            SearchFrontier::new(workspace, domain, options)
        })?;
        Ok(Self {
            prepared,
            frontier,
            options,
        })
    }

    /// Attempt at most `max_cells` complete cell transitions. On refusal, the
    /// current cell stays pending and earlier transitions remain available.
    pub fn advance(
        &mut self,
        max_cells: usize,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationProgress> {
        meter.checkpoint().map_err(RootBoxError::from)?;
        if !(1..=MAX_ROOT_ISOLATION_ADVANCE_CELLS).contains(&max_cells) {
            return Err(RootBoxError::ResourceLimit(
                "requested root-isolation batch must be 1 through 2047 cells",
            )
            .into());
        }
        self.prepared.with_workspace(meter, |workspace| {
            self.frontier.advance(workspace, self.options, max_cells)
        })
    }

    pub fn progress(&self) -> RootIsolationProgress {
        self.frontier.progress()
    }

    /// Borrow the candidate only when every search cell has been discharged.
    /// This neither verifies it nor consumes the search or its ledger.
    pub fn completed_witness(&self) -> Option<&RootIsolationWitness> {
        self.frontier
            .pending
            .is_empty()
            .then_some(&self.frontier.witness)
    }

    pub fn residuals(&self) -> &[Expr] {
        self.prepared.residuals()
    }

    pub fn variables(&self) -> &[Symbol] {
        self.prepared.variables()
    }

    pub fn domain(&self) -> &[RealBall] {
        self.prepared.domain()
    }

    pub fn options(&self) -> RootIsolationOptions {
        self.options
    }

    /// Intrinsic consumption includes failed attempts and checkpoint copies.
    pub fn counters(&self) -> RootIsolationCounters {
        self.prepared.counters()
    }

    /// Copy a bounded normalized candidate while charging the same caller
    /// ledger and intrinsic counters. The returned history includes this
    /// snapshot copy; exporting repeatedly never refunds consumed work.
    pub fn checkpoint(
        &mut self,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationSearchCheckpoint> {
        let witness = self.prepared.with_workspace(meter, |workspace| {
            copy_witness(workspace, &self.frontier.witness)
        })?;
        Ok(RootIsolationSearchCheckpoint {
            options: self.options,
            witness,
            counters: self.prepared.counters(),
        })
    }

    /// Reconstruct candidate search state for the caller's expected source.
    ///
    /// All imported roots and exclusions are independently rechecked, and
    /// every pending region is reconstructed from the bounded prefix. No
    /// partial completeness receipt is created. Preparation, validation and
    /// copying consume the receiver's meter; prior intrinsic work is added,
    /// not substituted, and can exhaust the unchanged lifetime limits.
    /// Header/structure lower bounds detect impossible understatements, but
    /// no unauthenticated checkpoint can prove its sender's full work history.
    /// Failure leaves both the checkpoint and the caller's searches untouched.
    pub fn from_checkpoint(
        residuals: &[Expr],
        variables: &[Symbol],
        domain: &[RealBall],
        checkpoint: &RootIsolationSearchCheckpoint,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self> {
        meter.checkpoint().map_err(RootBoxError::from)?;
        check_options(checkpoint.options)?;
        let witness = &checkpoint.witness;
        if witness.partition.len() > 2 * checkpoint.options.max_leaves - 1
            || witness.roots.len() > MAX_ROOT_ISOLATION_ROOTS
            || witness.precision_digits != checkpoint.options.precision_digits
        {
            return Err(RootIsolationError::InvalidPartition(
                "checkpoint header disagrees with its bounded search options",
            ));
        }
        let mut search = Self::new(residuals, variables, domain, checkpoint.options, meter)?;
        let baseline = search.counters();
        // Every committed transition charges at least preparation and commit.
        // Every split/exclusion performs a source enclosure; discovering a
        // new root also encloses its midpoint and performs two local checks.
        // Covered leaves may reuse an earlier root without a new evaluation.
        let evaluated_cells = witness
            .partition
            .iter()
            .filter(|step| !matches!(step, RootIsolationStep::Root { .. }))
            .count();
        let minimum_evaluations = evaluated_cells + 4 * witness.roots.len();
        let minimum_steps = baseline.steps + 2 * witness.partition.len() as u64;
        if checkpoint.counters.steps < minimum_steps
            || checkpoint.counters.evaluations < minimum_evaluations
        {
            return Err(RootIsolationError::InvalidPartition(
                "checkpoint understates committed traversal work",
            ));
        }
        search.prepared.include_prior_work(checkpoint.counters)?;
        let (prefix, copied_witness, pending) =
            search.prepared.with_workspace(meter, |workspace| {
                let prefix = workspace.validate_prefix(witness, checkpoint.options.max_leaves)?;
                let copied_witness = copy_witness(workspace, witness)?;
                workspace.account(
                    1,
                    ((MAX_ROOT_ISOLATION_DEPTH + 1) * std::mem::size_of::<Cell>()) as u64,
                    1,
                )?;
                let pending = reserve(
                    MAX_ROOT_ISOLATION_DEPTH + 1,
                    "restored root-isolation traversal allocation",
                )?;
                Ok((prefix, copied_witness, pending))
            })?;
        search.frontier.leaves = prefix.leaf_count();
        search.frontier.excluded = prefix.excluded_leaf_count();
        search.frontier.witness = copied_witness;
        search.frontier.pending = pending;
        for (domain, depth) in prefix.into_pending() {
            search.frontier.pending.push(Cell { domain, depth });
        }
        Ok(search)
    }
}

fn copy_witness(
    workspace: &mut RootIsolationWorkspace<'_, impl BudgetMeter>,
    witness: &RootIsolationWitness,
) -> Result<RootIsolationWitness> {
    // Private search data or independently validated witness data only.
    let payload_bytes: u64 = witness
        .roots
        .iter()
        .map(|root| {
            (std::mem::size_of_val(root.domain.as_slice())
                + std::mem::size_of_val(root.center.as_slice())
                + std::mem::size_of_val(root.preconditioner.as_slice())) as u64
                + root
                    .domain
                    .iter()
                    .flat_map(|value| [value.midpoint(), value.radius()])
                    .chain(root.center.iter())
                    .chain(root.preconditioner.iter().flatten())
                    .map(rational_bytes)
                    .sum::<u64>()
        })
        .sum();
    workspace.account(
        1,
        payload_bytes
            + (MAX_ROOT_ISOLATION_ROOTS * std::mem::size_of::<RootBoxWitness>()) as u64
            + (MAX_ROOT_ISOLATION_NODES * std::mem::size_of::<RootIsolationStep>()) as u64,
        2 + witness
            .roots
            .iter()
            .map(|root| {
                (root.domain.len() * 4
                    + root.center.len() * 2
                    + root.preconditioner.iter().map(Vec::len).sum::<usize>() * 2
                    + root.preconditioner.len()
                    + 3) as u64
            })
            .sum::<u64>(),
    )?;
    // Preserve the fixed spare capacity used by future atomic transitions.
    let mut partition = reserve(
        MAX_ROOT_ISOLATION_NODES,
        "root-checkpoint partition allocation",
    )?;
    partition.extend_from_slice(&witness.partition);
    let mut roots = reserve(MAX_ROOT_ISOLATION_ROOTS, "root-checkpoint roots allocation")?;
    roots.extend(witness.roots.iter().cloned());
    Ok(RootIsolationWitness {
        roots,
        partition,
        precision_digits: witness.precision_digits,
    })
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
    check_options(options)?;
    let mut workspace = RootIsolationWorkspace::new(
        residuals,
        variables,
        domain,
        options.precision_digits,
        meter,
    )?;
    let mut frontier = SearchFrontier::new(&mut workspace, domain, options)?;
    frontier.advance(&mut workspace, options, MAX_ROOT_ISOLATION_ADVANCE_CELLS)?;
    workspace.account(1, 0, 0)?;
    Ok(frontier.witness)
}
