//! Sparse Jacobian analysis, verified differentiation proofs, and graph coloring (WS12 / C7 gate).

#![forbid(unsafe_code)]

use crate::compile::{
    CompileError, CompiledExpr, CompiledOp, EvalError, MAX_COMPILE_DEPTH, MAX_COMPILE_OPS,
    admit_compilation_name, charge_compilation_work, charge_evaluation_allocation,
    charge_evaluation_steps, compilation_diagnostic, compilation_finish,
};
use crate::proof::{reference_sub_derivative, verified_diff, verify_diff_derivation};
use fsym_budget::{BudgetMeter, Dimension, Unbounded};
use fsym_core::{BigInt, Constant, Expr, Symbol};
use fsym_proof_kernel::{DerivationTree, derivation_verification_units};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Maximum residual or variable axis in sparse source admission. This is not a
/// dense-cell limit: sparse shape admission never materializes the Cartesian product.
pub const MAX_SPARSE_AXES: usize = 65_536;
/// Aggregate source nodes and row-local differentiation work admitted before
/// generating any derivative, proof, or coloring table.
pub const MAX_SPARSE_SOURCE_WORK: usize = 1_048_576;

/// Bounded generator-side source plan. Declared columns are sorted and unique
/// within each row. This is admission information, not a verification receipt:
/// independent certificate replay reconstructs occurrences from the raw source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseSourcePlan {
    pub row_dependencies: Vec<Vec<usize>>,
    pub source_nodes: usize,
    /// Sum of row_nodes * max(number_of_declared_dependencies, 1).
    pub differentiation_units: usize,
}

fn sparse_limit(actual: usize, limit: usize) -> Result<(), CompileError> {
    if actual > limit {
        return Err(CompileError::OpLimitExceeded {
            ops: actual,
            max: limit,
        });
    }
    Ok(())
}

/// Inspect every raw child, including held zero products, powers, and function
/// arguments. Typed Symbol equality alone binds a source occurrence to a column.
/// Foreign symbols do not become declared dependencies; the caller must still
/// compile the original residuals to enforce unmapped-symbol and source domains.
pub fn plan_sparse_sources(
    residuals: &[Expr],
    variables: &[Symbol],
) -> Result<SparseSourcePlan, CompileError> {
    plan_sparse_sources_with_meter(residuals, variables, &mut Unbounded)
}

/// Plan raw source dependencies under the same cumulative ledger as compilation.
/// Every source child is visited, including domains hidden by a zero multiplier.
pub fn plan_sparse_sources_with_meter<M: BudgetMeter + ?Sized>(
    residuals: &[Expr],
    variables: &[Symbol],
    meter: &mut M,
) -> Result<SparseSourcePlan, CompileError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    sparse_limit(residuals.len(), MAX_SPARSE_AXES)?;
    sparse_limit(variables.len(), MAX_SPARSE_AXES)?;
    let mut positions = HashMap::new();
    charge_sparse_buffer::<(&Symbol, usize), _>(variables.len(), meter)?;
    positions
        .try_reserve(variables.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    for (column, variable) in variables.iter().enumerate() {
        admit_compilation_name(&variable.name, meter)?;
        charge_compilation_work(meter, variable.name.len() + 1, 0, 0)?;
        if positions.insert(variable, column).is_some() {
            charge_compilation_work(meter, variable.name.len(), variable.name.len(), 1)?;
            return Err(CompileError::DuplicateVariable(variable.clone()));
        }
    }
    let mut row_dependencies = Vec::new();
    charge_sparse_buffer::<Vec<usize>, _>(residuals.len(), meter)?;
    row_dependencies
        .try_reserve_exact(residuals.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    let mut source_nodes = 0usize;
    let mut differentiation_units = 0usize;
    for source in residuals {
        charge_sparse_buffer::<(&Expr, usize), _>(1, meter)?;
        let mut pending = vec![(source, 0usize)];
        let mut nodes = 0usize;
        let mut dependencies = HashSet::new();
        while let Some((expression, depth)) = pending.pop() {
            charge_compilation_work(meter, 1, 0, 0)?;
            if depth > MAX_COMPILE_DEPTH {
                return Err(CompileError::DepthLimitExceeded {
                    depth,
                    max: MAX_COMPILE_DEPTH,
                });
            }
            nodes = nodes.saturating_add(1);
            sparse_limit(nodes, MAX_COMPILE_OPS)?;
            match expression {
                Expr::Sym(symbol) => {
                    admit_compilation_name(&symbol.name, meter)?;
                    charge_compilation_work(meter, symbol.name.len() + 1, 0, 0)?;
                    if let Some(&column) = positions.get(symbol) {
                        charge_sparse_buffer::<usize, _>(1, meter)?;
                        dependencies
                            .try_reserve(1)
                            .map_err(|_| CompileError::AllocationFailure)?;
                        dependencies.insert(column);
                    }
                }
                Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                    if let Expr::Function(name, _) = expression {
                        admit_compilation_name(name, meter)?;
                    }
                    sparse_limit(
                        nodes
                            .saturating_add(pending.len())
                            .saturating_add(children.len()),
                        MAX_COMPILE_OPS,
                    )?;
                    charge_sparse_buffer::<(&Expr, usize), _>(children.len(), meter)?;
                    pending
                        .try_reserve(children.len())
                        .map_err(|_| CompileError::AllocationFailure)?;
                    pending.extend(children.iter().map(|child| (child, depth + 1)));
                }
                Expr::Pow(base, exponent) => {
                    sparse_limit(
                        nodes.saturating_add(pending.len()).saturating_add(2),
                        MAX_COMPILE_OPS,
                    )?;
                    charge_sparse_buffer::<(&Expr, usize), _>(2, meter)?;
                    pending
                        .try_reserve(2)
                        .map_err(|_| CompileError::AllocationFailure)?;
                    pending.push((base, depth + 1));
                    pending.push((exponent, depth + 1));
                }
                Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
            }
        }
        source_nodes = source_nodes.saturating_add(nodes);
        differentiation_units =
            differentiation_units.saturating_add(nodes.saturating_mul(dependencies.len().max(1)));
        sparse_limit(source_nodes, MAX_SPARSE_SOURCE_WORK)?;
        sparse_limit(differentiation_units, MAX_SPARSE_SOURCE_WORK)?;
        charge_sparse_buffer::<usize, _>(dependencies.len(), meter)?;
        charge_compilation_work(
            meter,
            dependencies
                .len()
                .saturating_mul(dependencies.len().max(1).ilog2() as usize + 1),
            0,
            0,
        )?;
        let mut columns: Vec<_> = dependencies.into_iter().collect();
        columns.sort_unstable();
        row_dependencies.push(columns);
    }
    charge_compilation_work(meter, 1, 0, 0)?;
    Ok(SparseSourcePlan {
        row_dependencies,
        source_nodes,
        differentiation_units,
    })
}

fn charge_sparse_buffer<T, M: BudgetMeter + ?Sized>(
    count: usize,
    meter: &mut M,
) -> Result<(), CompileError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CompileError::DimensionOverflow)?;
    charge_compilation_work(meter, count, bytes, usize::from(count != 0))
}

/// Charge a bounded structural copy before cloning its typed tree. Numeric
/// limbs and identifier bytes participate in storage accounting, not just nodes.
pub(crate) fn charge_expression_copy<M: BudgetMeter + ?Sized>(
    expression: &Expr,
    meter: &mut M,
) -> Result<(usize, usize), CompileError> {
    fn visit<M: BudgetMeter + ?Sized>(
        expression: &Expr,
        depth: usize,
        nodes: &mut usize,
        bytes: &mut usize,
        meter: &mut M,
    ) -> Result<(), CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        if depth > MAX_COMPILE_DEPTH {
            return Err(CompileError::DepthLimitExceeded {
                depth,
                max: MAX_COMPILE_DEPTH,
            });
        }
        *nodes = nodes.saturating_add(1);
        sparse_limit(*nodes, MAX_COMPILE_OPS + 2)?;
        let payload = match expression {
            Expr::Integer(value) => usize::try_from(value.limb_count())
                .ok()
                .and_then(|limbs| limbs.checked_mul(8)),
            Expr::Rational(value) => value
                .numer()
                .limb_count()
                .checked_add(value.denom().limb_count())
                .and_then(|limbs| usize::try_from(limbs).ok())
                .and_then(|limbs| limbs.checked_mul(8)),
            Expr::Sym(symbol) => {
                admit_compilation_name(&symbol.name, meter)?;
                Some(symbol.name.len())
            }
            Expr::Function(name, _) => {
                admit_compilation_name(name, meter)?;
                Some(name.len())
            }
            _ => Some(0),
        }
        .ok_or(CompileError::DimensionOverflow)?;
        let owned = payload
            .checked_add(std::mem::size_of::<Expr>())
            .ok_or(CompileError::DimensionOverflow)?;
        *bytes = bytes
            .checked_add(owned)
            .ok_or(CompileError::DimensionOverflow)?;
        charge_compilation_work(meter, payload.div_ceil(8).max(1), owned, 1)?;
        match expression {
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                sparse_limit(nodes.saturating_add(children.len()), MAX_COMPILE_OPS + 2)?;
                for child in children {
                    visit(child, depth + 1, nodes, bytes, meter)?;
                }
            }
            Expr::Pow(base, exponent) => {
                visit(base, depth + 1, nodes, bytes, meter)?;
                visit(exponent, depth + 1, nodes, bytes, meter)?;
            }
            _ => {}
        }
        Ok(())
    }
    let mut nodes = 0;
    let mut bytes = 0;
    visit(expression, 0, &mut nodes, &mut bytes, meter)?;
    Ok((nodes, bytes))
}

/// Differentiation and its simple reference checker are bounded synchronous
/// kernels. Charge the existing product/chain-rule growth envelope and all
/// source payloads before either kernel starts, then poll at its boundary.
/// This is conservative logical work/storage, not an allocator or latency claim.
fn charge_derivative_kernel<M: BudgetMeter + ?Sized>(
    source: &Expr,
    meter: &mut M,
) -> Result<(), CompileError> {
    let (source_nodes, source_bytes) = charge_expression_copy(source, meter)?;
    charge_compilation_work(meter, source_nodes, 0, 0)?;
    let (_, predicted) = derivative_growth(source);
    sparse_limit(predicted, MAX_COMPILE_OPS)?;
    let envelope = source_nodes
        .checked_add(predicted)
        .ok_or(CompileError::DimensionOverflow)?;
    let storage = source_bytes
        .checked_mul(envelope)
        .and_then(|bytes| bytes.checked_mul(4))
        .ok_or(CompileError::DimensionOverflow)?;
    let work = envelope
        .checked_mul(envelope)
        .and_then(|steps| steps.checked_add(storage.div_ceil(8)))
        .ok_or(CompileError::DimensionOverflow)?;
    charge_compilation_work(meter, work, storage, envelope.saturating_mul(4))
}

/// Conservative raw product/chain-rule growth, evaluated only after bounded
/// source traversal. It prevents a short variadic product from allocating a
/// quadratic derivative forest before the ordinary compiled-size limit fires.
fn derivative_growth(source: &Expr) -> (usize, usize) {
    match source {
        Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
            let mut nodes = 0usize;
            let mut derivatives = 0usize;
            for child in children {
                let (child_nodes, child_derivative) = derivative_growth(child);
                nodes = nodes.saturating_add(child_nodes);
                derivatives = derivatives.saturating_add(child_derivative);
            }
            let derivative = match source {
                Expr::Mul(_) => 1usize
                    .saturating_add(children.len())
                    .saturating_add(children.len().saturating_sub(1).saturating_mul(nodes))
                    .saturating_add(derivatives),
                Expr::Function(_, _) => 24usize
                    .saturating_add(nodes.saturating_mul(4))
                    .saturating_add(derivatives),
                _ => 1usize.saturating_add(derivatives),
            };
            (nodes.saturating_add(1), derivative)
        }
        Expr::Pow(base, exponent) => {
            let (base_nodes, base_derivative) = derivative_growth(base);
            let (exponent_nodes, exponent_derivative) = derivative_growth(exponent);
            let derivative = if matches!(exponent.as_ref(), Expr::Integer(_) | Expr::Rational(_)) {
                4usize
                    .saturating_add(base_nodes)
                    .saturating_add(base_derivative)
            } else {
                24usize
                    .saturating_add(base_nodes.saturating_mul(3))
                    .saturating_add(exponent_nodes.saturating_mul(2))
                    .saturating_add(base_derivative)
                    .saturating_add(exponent_derivative)
            };
            (
                1usize
                    .saturating_add(base_nodes)
                    .saturating_add(exponent_nodes),
                derivative,
            )
        }
        Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => (1, 1),
    }
}

fn admit_derivative_growth(
    residuals: &[Expr],
    dependencies: &[Vec<usize>],
) -> Result<(), CompileError> {
    let mut total = 0usize;
    for (source, columns) in residuals.iter().zip(dependencies) {
        if columns.is_empty() {
            continue;
        }
        let (_, predicted) = derivative_growth(source);
        sparse_limit(predicted, MAX_COMPILE_OPS)?;
        total = total.saturating_add(predicted.saturating_mul(columns.len()));
        sparse_limit(total, MAX_SPARSE_SOURCE_WORK)?;
    }
    Ok(())
}

fn zero_indices<M: BudgetMeter + ?Sized>(
    length: usize,
    meter: &mut M,
) -> Result<Vec<usize>, CompileError> {
    charge_sparse_buffer::<usize, _>(length, meter)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| CompileError::AllocationFailure)?;
    result.resize(length, 0);
    Ok(result)
}

/// The same ascending-column greedy coloring as a conflict graph, traversed
/// through sparse row/column incidence instead of storing every conflict pair.
/// Each row of width k costs O(k^2) visits, bounded by its source_nodes*k
/// admission score; table storage is O(rows + columns + nonzeros).
fn color_sparse_incidence<M: BudgetMeter + ?Sized>(
    rows: usize,
    columns: usize,
    nonzeros: &[(usize, usize)],
    meter: &mut M,
) -> Result<(Vec<usize>, usize), CompileError> {
    let mut row_offsets = zero_indices(rows + 1, meter)?;
    let mut column_offsets = zero_indices(columns + 1, meter)?;
    for &(row, column) in nonzeros {
        charge_compilation_work(meter, 1, 0, 0)?;
        row_offsets[row + 1] += 1;
        column_offsets[column + 1] += 1;
    }
    for row in 0..rows {
        charge_compilation_work(meter, 1, 0, 0)?;
        row_offsets[row + 1] += row_offsets[row];
    }
    for column in 0..columns {
        charge_compilation_work(meter, 1, 0, 0)?;
        column_offsets[column + 1] += column_offsets[column];
    }
    let mut next = zero_indices(columns, meter)?;
    charge_compilation_work(meter, columns, 0, 0)?;
    next.copy_from_slice(&column_offsets[..columns]);
    let mut column_rows = zero_indices(nonzeros.len(), meter)?;
    for &(row, column) in nonzeros {
        charge_compilation_work(meter, 1, 0, 0)?;
        column_rows[next[column]] = row;
        next[column] += 1;
    }
    let mut colors = zero_indices(columns, meter)?;
    let mut marked = zero_indices(columns, meter)?;
    let mut color_count = 0usize;
    for column in 0..columns {
        charge_compilation_work(meter, 1, 0, 0)?;
        let mark = column + 1;
        for &row in &column_rows[column_offsets[column]..column_offsets[column + 1]] {
            charge_compilation_work(meter, 1, 0, 0)?;
            for &(_, neighbor) in &nonzeros[row_offsets[row]..row_offsets[row + 1]] {
                charge_compilation_work(meter, 1, 0, 0)?;
                if neighbor >= column {
                    break;
                }
                marked[colors[neighbor]] = mark;
            }
        }
        let mut color = 0usize;
        while marked[color] == mark {
            charge_compilation_work(meter, 1, 0, 0)?;
            color += 1;
        }
        colors[column] = color;
        color_count = color_count.max(color + 1);
    }
    Ok((colors, color_count))
}

/// A single non-zero entry in a symbolically differentiated sparse Jacobian matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseJacobianEntry {
    /// Zero-based row index (residual index $i \in [0, m)$).
    pub row: usize,
    /// Zero-based column index (variable index $j \in [0, n)$).
    pub col: usize,
    /// Symbolic partial derivative $\frac{\partial f_i}{\partial x_j}$.
    pub symbolic_deriv: Expr,
    /// Compiled bytecode evaluator for fast numeric evaluation of this partial derivative.
    pub compiled_deriv: CompiledExpr,
    /// Verified derivation proof establishing $\vdash \text{diff}(f_i, x_j) = \text{symbolic\_deriv}$.
    pub derivation: DerivationTree,
}

/// Sparsity pattern metadata for a multi-dimensional system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsityPattern {
    pub num_rows: usize,
    pub num_cols: usize,
    /// Sorted list of (row, col) coordinates of structural non-zeros.
    pub nonzeros: Vec<(usize, usize)>,
    pub nnz: usize,
}

/// Compiled sparse Jacobian system with verified differentiation proofs and graph coloring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseJacobian {
    /// Residual expressions $F(x) = (f_0, \dots, f_{m-1})$.
    pub residuals: Vec<Expr>,
    /// Independent variables $(x_0, \dots, x_{n-1})$.
    pub vars: Vec<Symbol>,
    /// Sparsity pattern containing structural non-zeros.
    pub pattern: SparsityPattern,
    /// Compiled sparse Jacobian entries containing bytecode and proof trees.
    pub entries: Vec<SparseJacobianEntry>,
    /// Distance-1 column coloring for Jacobian compression: `column_colors[col]` is the color index $\in [0, \text{num\_colors})$.
    /// Two columns that share a non-zero in the same row receive different colors.
    pub column_colors: Vec<usize>,
    /// Total number of distinct colors (the compression factor: $k \le n$).
    pub num_colors: usize,
}

/// Receipt confirming successful verification of a sparse Jacobian system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseJacobianReceipt {
    pub num_residuals: usize,
    pub num_vars: usize,
    pub nnz: usize,
    pub density: f64,
    pub num_colors: usize,
    pub proofs_verified: usize,
    pub test_points_evaluated: usize,
}

/// Verification errors for sparse Jacobian certificates and evaluator agreement.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum JacobianVerificationError {
    #[error(
        "Jacobian shape ({rows}, {cols}) disagrees with {residuals} residuals and {variables} variables"
    )]
    ShapeMismatch {
        rows: usize,
        cols: usize,
        residuals: usize,
        variables: usize,
    },
    #[error("Jacobian variable list contains the duplicate symbol {0}")]
    DuplicateVariable(Symbol),
    #[error("Declared sparsity count {declared} differs from the coordinate count {actual}")]
    NonzeroCountMismatch { declared: usize, actual: usize },
    #[error("Sparsity coordinates are not in strictly increasing row-major order at index {index}")]
    NonCanonicalPattern { index: usize },
    #[error("Sparse entry at index {index} does not match its pattern coordinate")]
    CoordinateMismatch { index: usize },
    #[error("Column color table has {actual} entries, expected {expected}")]
    ColorTableMismatch { expected: usize, actual: usize },
    #[error("Declared color count {declared} is inconsistent with the column colors")]
    ColorCountMismatch { declared: usize },
    #[error("Column {column} has color {color}, outside the declared {num_colors} colors")]
    ColorOutOfBounds {
        column: usize,
        color: usize,
        num_colors: usize,
    },
    #[error("Compiled bytecode does not encode the verified derivative at ({row}, {col})")]
    CompiledProvenanceMismatch { row: usize, col: usize },
    #[error("Diagnostic point {point_idx} must contain exactly {expected} finite coordinates")]
    InvalidTestPoint { point_idx: usize, expected: usize },
    #[error("Diagnostic evaluation is not finite at point {point_idx}, entry ({row}, {col})")]
    NonFiniteEvaluation {
        point_idx: usize,
        row: usize,
        col: usize,
    },
    #[error("Differentiation proof verification failed for ({row}, {col}): {reason}")]
    ProofFailed {
        row: usize,
        col: usize,
        reason: String,
    },
    #[error("Omitted entry ({row}, {col}) is not mathematically zero: derivative is {deriv}")]
    OmittedNonzero {
        row: usize,
        col: usize,
        deriv: String,
    },
    #[error(
        "Invalid column coloring: row {row} has colliding columns {col1} and {col2} with color {color}"
    )]
    ColoringCollision {
        row: usize,
        col1: usize,
        col2: usize,
        color: usize,
    },
    #[error(
        "Compiled bytecode evaluator disagreement at point {point_idx} at ({row}, {col}): sparse={sparse}, dense={dense}"
    )]
    EvaluatorDisagreement {
        point_idx: usize,
        row: usize,
        col: usize,
        sparse: f64,
        dense: f64,
    },
    #[error("Matrix bounds violated: row {row} >= {num_rows} or col {col} >= {num_cols}")]
    OutOfBounds {
        row: usize,
        col: usize,
        num_rows: usize,
        num_cols: usize,
    },
    #[error(
        "Sparsity pattern entry count mismatch: pattern has {pattern_nnz} entries, entries table has {table_nnz}"
    )]
    PatternMismatch {
        pattern_nnz: usize,
        table_nnz: usize,
    },
    #[error("Compilation error: {0}")]
    Compile(#[from] CompileError),
    #[error("Evaluation error: {0}")]
    Eval(#[from] EvalError),
}

/// Logical table storage is charged before requesting its allocation. This is
/// payload accounting rather than instrumentation of the standard allocator.
fn charge_structure_allocation<M: BudgetMeter + ?Sized, T>(
    meter: &mut M,
    count: usize,
) -> Result<(), EvalError> {
    charge_evaluation_steps(meter, 0)?;
    if count != 0 {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(EvalError::DimensionOverflow)?;
        meter.charge_batch(&[
            (
                Dimension::MemoryBytes,
                u64::try_from(bytes).map_err(|_| EvalError::DimensionOverflow)?,
            ),
            (Dimension::AllocationCount, 1),
        ])?;
    }
    Ok(())
}

impl SparseJacobian {
    /// Builds a compiled sparse Jacobian system with verified differentiation proofs and graph coloring.
    pub fn try_build(residuals: &[Expr], vars: &[Symbol]) -> Result<Self, CompileError> {
        Self::try_build_with_meter(residuals, vars, &mut Unbounded)
    }

    /// Build an untrusted derivative candidate with cumulative source, proof,
    /// bytecode and coloring accounting. The caller must independently admit it
    /// before using it as a verified program; the generator never has access to
    /// a separately reserved verifier budget.
    pub fn try_build_with_meter<M: BudgetMeter + ?Sized>(
        residuals: &[Expr],
        vars: &[Symbol],
        meter: &mut M,
    ) -> Result<Self, CompileError> {
        let plan = plan_sparse_sources_with_meter(residuals, vars, meter)?;
        charge_compilation_work(meter, plan.source_nodes, 0, 0)?;
        admit_derivative_growth(residuals, &plan.row_dependencies)?;
        let mut var_map = HashMap::new();
        charge_sparse_buffer::<(Symbol, usize), _>(vars.len(), meter)?;
        var_map
            .try_reserve(vars.len())
            .map_err(|_| CompileError::AllocationFailure)?;
        for (column, variable) in vars.iter().enumerate() {
            charge_compilation_work(meter, variable.name.len() + 1, variable.name.len(), 1)?;
            var_map.insert(variable.clone(), column);
        }
        let num_residuals = residuals.len();
        let num_vars = vars.len();
        let mut entries = Vec::new();
        let mut nonzeros = Vec::new();
        for (row, columns) in plan.row_dependencies.iter().enumerate() {
            for &col in columns {
                charge_derivative_kernel(&residuals[row], meter)?;
                let (symbolic_deriv, derivation) = verified_diff(&residuals[row], &vars[col]);
                charge_compilation_work(meter, 1, 0, 0)?;
                if !symbolic_deriv.is_zero() {
                    let compiled_deriv =
                        CompiledExpr::try_compile_with_meter(&symbolic_deriv, &var_map, meter)?;
                    charge_sparse_buffer::<SparseJacobianEntry, _>(1, meter)?;
                    entries
                        .try_reserve(1)
                        .map_err(|_| CompileError::AllocationFailure)?;
                    charge_sparse_buffer::<(usize, usize), _>(1, meter)?;
                    nonzeros
                        .try_reserve(1)
                        .map_err(|_| CompileError::AllocationFailure)?;
                    entries.push(SparseJacobianEntry {
                        row,
                        col,
                        symbolic_deriv,
                        compiled_deriv,
                        derivation,
                    });
                    nonzeros.push((row, col));
                }
            }
        }
        let (column_colors, num_colors) =
            color_sparse_incidence(num_residuals, num_vars, &nonzeros, meter)?;

        let pattern = SparsityPattern {
            num_rows: num_residuals,
            num_cols: num_vars,
            nnz: entries.len(),
            nonzeros,
        };

        charge_sparse_buffer::<Expr, _>(residuals.len(), meter)?;
        for source in residuals {
            charge_expression_copy(source, meter)?;
        }
        charge_sparse_buffer::<Symbol, _>(vars.len(), meter)?;
        for variable in vars {
            charge_compilation_work(meter, variable.name.len() + 1, variable.name.len(), 1)?;
        }
        let result = Self {
            residuals: residuals.to_vec(),
            vars: vars.to_vec(),
            pattern,
            entries,
            column_colors,
            num_colors,
        };
        charge_compilation_work(meter, 1, 0, 0)?;
        Ok(result)
    }

    /// Checks all dimensions, coordinates, and colors before any indexed access.
    ///
    /// This is a representation check, not mathematical proof. In particular,
    /// callers importing a certificate must also run
    /// [`verify_sparse_jacobian_certificate`] to check the derivatives and their
    /// compiled programs. Public fields and deserialized objects are untrusted.
    pub fn validate_structure(&self) -> Result<(), JacobianVerificationError> {
        self.validate_structure_with_meter(&mut Unbounded)
    }

    /// Cancellable representation checks with cumulative work and logical
    /// scratch-storage accounting. This does not perform or replace symbolic
    /// certificate verification.
    pub fn validate_structure_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        meter: &mut M,
    ) -> Result<(), JacobianVerificationError> {
        charge_evaluation_steps(meter, 1)?;
        let m = self.pattern.num_rows;
        let n = self.pattern.num_cols;
        if m != self.residuals.len() || n != self.vars.len() {
            return Err(JacobianVerificationError::ShapeMismatch {
                rows: m,
                cols: n,
                residuals: self.residuals.len(),
                variables: self.vars.len(),
            });
        }
        sparse_limit(m, MAX_SPARSE_AXES)?;
        sparse_limit(n, MAX_SPARSE_AXES)?;
        sparse_limit(self.entries.len(), MAX_SPARSE_SOURCE_WORK)?;
        if self.pattern.nnz != self.pattern.nonzeros.len() {
            return Err(JacobianVerificationError::NonzeroCountMismatch {
                declared: self.pattern.nnz,
                actual: self.pattern.nonzeros.len(),
            });
        }
        if self.pattern.nonzeros.len() != self.entries.len() {
            return Err(JacobianVerificationError::PatternMismatch {
                pattern_nnz: self.pattern.nonzeros.len(),
                table_nnz: self.entries.len(),
            });
        }

        let mut variables = HashSet::new();
        charge_structure_allocation::<_, &Symbol>(meter, n)?;
        variables
            .try_reserve(n)
            .map_err(|_| EvalError::AllocationFailure)?;
        for variable in &self.vars {
            // The full symbol name participates in hashing. Charge its bytes
            // as well as the fixed-size typed identity before the lookup.
            charge_evaluation_steps(meter, variable.name.len().saturating_add(1))?;
            if !variables.insert(variable) {
                charge_structure_allocation::<_, u8>(meter, variable.name.len())?;
                return Err(JacobianVerificationError::DuplicateVariable(
                    variable.clone(),
                ));
            }
        }

        for (index, (&(row, col), entry)) in
            self.pattern.nonzeros.iter().zip(&self.entries).enumerate()
        {
            charge_evaluation_steps(meter, 1)?;
            for (r, c) in [(row, col), (entry.row, entry.col)] {
                if r >= m || c >= n {
                    return Err(JacobianVerificationError::OutOfBounds {
                        row: r,
                        col: c,
                        num_rows: m,
                        num_cols: n,
                    });
                }
            }
            if index > 0 && self.pattern.nonzeros[index - 1] >= (row, col) {
                return Err(JacobianVerificationError::NonCanonicalPattern { index });
            }
            if (row, col) != (entry.row, entry.col) {
                return Err(JacobianVerificationError::CoordinateMismatch { index });
            }
        }

        if self.column_colors.len() != n {
            return Err(JacobianVerificationError::ColorTableMismatch {
                expected: n,
                actual: self.column_colors.len(),
            });
        }
        if self.num_colors > n || (self.num_colors == 0) != (n == 0) {
            return Err(JacobianVerificationError::ColorCountMismatch {
                declared: self.num_colors,
            });
        }
        let mut used_colors = HashSet::new();
        charge_structure_allocation::<_, usize>(meter, self.num_colors)?;
        used_colors
            .try_reserve(self.num_colors)
            .map_err(|_| EvalError::AllocationFailure)?;
        for (column, &color) in self.column_colors.iter().enumerate() {
            charge_evaluation_steps(meter, 1)?;
            if color >= self.num_colors {
                return Err(JacobianVerificationError::ColorOutOfBounds {
                    column,
                    color,
                    num_colors: self.num_colors,
                });
            }
            used_colors.insert(color);
        }
        if used_colors.len() != self.num_colors {
            return Err(JacobianVerificationError::ColorCountMismatch {
                declared: self.num_colors,
            });
        }

        let mut row_colors = HashMap::new();
        charge_structure_allocation::<_, (usize, usize)>(meter, n.min(self.entries.len()))?;
        row_colors
            .try_reserve(n.min(self.entries.len()))
            .map_err(|_| EvalError::AllocationFailure)?;
        let mut previous_row = None;
        for entry in &self.entries {
            charge_evaluation_steps(meter, 1)?;
            if previous_row != Some(entry.row) {
                charge_evaluation_steps(meter, row_colors.len())?;
                row_colors.clear();
                previous_row = Some(entry.row);
            }
            let color = self.column_colors[entry.col];
            if let Some(col1) = row_colors.insert(color, entry.col) {
                return Err(JacobianVerificationError::ColoringCollision {
                    row: entry.row,
                    col1,
                    col2: entry.col,
                    color,
                });
            }
        }
        charge_evaluation_steps(meter, 0)?;
        Ok(())
    }

    /// Number of structural non-zero entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Matrix density ($nnz / (m \times n)$).
    pub fn density(&self) -> f64 {
        // Do not panic on unverified public metadata. Certificate acceptance
        // checks each axis and its actual vectors; density never allocates the
        // Cartesian product or imposes a dense-output size limit.
        let total = self.pattern.num_rows as f64 * self.pattern.num_cols as f64;
        if total == 0.0 {
            0.0
        } else {
            self.entries.len() as f64 / total
        }
    }

    /// Evaluates the sparse Jacobian non-zero entries into a contiguous values slice.
    /// A failure leaves the caller's buffer unchanged.
    pub fn try_eval_sparse(&self, x: &[f64], out_values: &mut [f64]) -> Result<(), EvalError> {
        self.try_eval_sparse_with_meter(x, out_values, &mut Unbounded)
    }

    /// Evaluates with one cumulative caller meter, including structural checks,
    /// scratch allocation and output publication. Cancellation or exhaustion
    /// leaves the output unchanged. This does not grant mathematical evidence
    /// to an unverified sparse program.
    pub fn try_eval_sparse_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        x: &[f64],
        out_values: &mut [f64],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        self.validate_for_evaluation_with_meter(meter)?;
        if out_values.len() != self.entries.len() {
            return Err(EvalError::JacobianBufferMismatch {
                expected: self.entries.len(),
                actual: out_values.len(),
            });
        }
        let evaluated = self.evaluate_entries_with_meter(x, meter)?;
        charge_evaluation_steps(meter, out_values.len())?;
        // Everything fallible precedes the single publication boundary.
        meter.checkpoint()?;
        out_values.copy_from_slice(&evaluated);
        Ok(())
    }

    /// Evaluates the sparse Jacobian into a dense row-major buffer.
    /// A failure leaves the caller's buffer unchanged.
    pub fn try_eval_dense(&self, x: &[f64], out_dense: &mut [f64]) -> Result<(), EvalError> {
        self.try_eval_dense_with_meter(x, out_dense, &mut Unbounded)
    }

    /// Dense expansion of the metered sparse values, publishing only after all
    /// entries, output work and the final cancellation safe point succeed.
    pub fn try_eval_dense_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        x: &[f64],
        out_dense: &mut [f64],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        self.validate_for_evaluation_with_meter(meter)?;
        let expected = self
            .pattern
            .num_rows
            .checked_mul(self.pattern.num_cols)
            .ok_or(EvalError::DimensionOverflow)?;
        if out_dense.len() != expected {
            return Err(EvalError::JacobianBufferMismatch {
                expected,
                actual: out_dense.len(),
            });
        }
        let evaluated = self.evaluate_entries_with_meter(x, meter)?;
        charge_evaluation_steps(
            meter,
            expected
                .checked_add(self.entries.len())
                .ok_or(EvalError::DimensionOverflow)?,
        )?;
        meter.checkpoint()?;
        out_dense.fill(0.0);
        for (entry, val) in self.entries.iter().zip(evaluated) {
            out_dense[entry.row * self.pattern.num_cols + entry.col] = val;
        }
        Ok(())
    }

    /// Evaluates the compressed Jacobian matrix of shape m by num_colors.
    /// A failure leaves the caller's buffer unchanged.
    pub fn try_eval_compressed(
        &self,
        x: &[f64],
        out_compressed: &mut [f64],
    ) -> Result<(), EvalError> {
        self.try_eval_compressed_with_meter(x, out_compressed, &mut Unbounded)
    }

    /// Color-compressed evaluation with transactional output and one meter
    /// shared by every entry and the complete publication step.
    pub fn try_eval_compressed_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        x: &[f64],
        out_compressed: &mut [f64],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        self.validate_for_evaluation_with_meter(meter)?;
        let expected = self
            .pattern
            .num_rows
            .checked_mul(self.num_colors)
            .ok_or(EvalError::DimensionOverflow)?;
        if out_compressed.len() != expected {
            return Err(EvalError::JacobianBufferMismatch {
                expected,
                actual: out_compressed.len(),
            });
        }
        let evaluated = self.evaluate_entries_with_meter(x, meter)?;
        charge_evaluation_steps(
            meter,
            expected
                .checked_add(self.entries.len())
                .ok_or(EvalError::DimensionOverflow)?,
        )?;
        meter.checkpoint()?;
        out_compressed.fill(0.0);
        for (entry, val) in self.entries.iter().zip(evaluated) {
            let col_color = self.column_colors[entry.col];
            out_compressed[entry.row * self.num_colors + col_color] = val;
        }
        Ok(())
    }

    fn validate_for_evaluation_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        meter: &mut M,
    ) -> Result<(), EvalError> {
        self.validate_structure_with_meter(meter)
            .map_err(|error| match error {
                // Meter refusals must remain typed outcomes, not malformed
                // certificate messages that a caller might retry.
                JacobianVerificationError::Eval(error @ EvalError::Meter(_)) => error,
                error => EvalError::InvalidSparseJacobian(error.to_string()),
            })
    }

    fn evaluate_entries_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        x: &[f64],
        meter: &mut M,
    ) -> Result<Vec<f64>, EvalError> {
        let mut evaluated = Vec::new();
        charge_evaluation_allocation(meter, self.entries.len())?;
        evaluated
            .try_reserve_exact(self.entries.len())
            .map_err(|_| EvalError::AllocationFailure)?;
        for entry in &self.entries {
            charge_evaluation_steps(meter, 1)?;
            evaluated.push(entry.compiled_deriv.try_eval_with_meter(x, meter)?);
        }
        Ok(evaluated)
    }
}

/// Verifier-owned occurrence scan. Deliberately separate from the iterative
/// generator planner: every raw expression variant is visited independently,
/// and no cached dependency summary, free-symbol optimization, or name match
/// can authorize an omitted coordinate.
fn verifier_dependencies<M: BudgetMeter + ?Sized>(
    system: &SparseJacobian,
    positions: &HashMap<Symbol, usize>,
    meter: &mut M,
) -> Result<Vec<Vec<usize>>, CompileError> {
    fn visit<M: BudgetMeter + ?Sized>(
        expression: &Expr,
        depth: usize,
        nodes: &mut usize,
        positions: &HashMap<Symbol, usize>,
        columns: &mut HashSet<usize>,
        meter: &mut M,
    ) -> Result<(), CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        if depth > MAX_COMPILE_DEPTH {
            return Err(CompileError::DepthLimitExceeded {
                depth,
                max: MAX_COMPILE_DEPTH,
            });
        }
        *nodes = nodes.saturating_add(1);
        sparse_limit(*nodes, MAX_COMPILE_OPS)?;
        match expression {
            Expr::Sym(symbol) => {
                admit_compilation_name(&symbol.name, meter)?;
                charge_compilation_work(meter, symbol.name.len() + 1, 0, 0)?;
                if let Some(&column) = positions.get(symbol) {
                    charge_sparse_buffer::<usize, _>(1, meter)?;
                    columns
                        .try_reserve(1)
                        .map_err(|_| CompileError::AllocationFailure)?;
                    columns.insert(column);
                }
            }
            Expr::Add(children) | Expr::Mul(children) => {
                for child in children {
                    visit(child, depth + 1, nodes, positions, columns, meter)?;
                }
            }
            Expr::Pow(base, exponent) => {
                visit(base, depth + 1, nodes, positions, columns, meter)?;
                visit(exponent, depth + 1, nodes, positions, columns, meter)?;
            }
            Expr::Function(name, arguments) => {
                admit_compilation_name(name, meter)?;
                for argument in arguments {
                    visit(argument, depth + 1, nodes, positions, columns, meter)?;
                }
            }
            Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
        }
        Ok(())
    }

    let mut result = Vec::new();
    charge_sparse_buffer::<Vec<usize>, _>(system.residuals.len(), meter)?;
    result
        .try_reserve_exact(system.residuals.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    let mut total_nodes = 0usize;
    let mut total_units = 0usize;
    for source in &system.residuals {
        let mut nodes = 0usize;
        let mut columns = HashSet::new();
        visit(source, 0, &mut nodes, positions, &mut columns, meter)?;
        total_nodes = total_nodes.saturating_add(nodes);
        total_units = total_units.saturating_add(nodes.saturating_mul(columns.len().max(1)));
        sparse_limit(total_nodes, MAX_SPARSE_SOURCE_WORK)?;
        sparse_limit(total_units, MAX_SPARSE_SOURCE_WORK)?;
        charge_sparse_buffer::<usize, _>(columns.len(), meter)?;
        charge_compilation_work(
            meter,
            columns
                .len()
                .saturating_mul(columns.len().max(1).ilog2() as usize + 1),
            0,
            0,
        )?;
        let mut columns: Vec<_> = columns.into_iter().collect();
        columns.sort_unstable();
        result.push(columns);
    }
    // This is only a resource estimator shared with construction. Mathematical
    // coverage below uses the separate reference derivative and this traversal.
    charge_compilation_work(meter, total_nodes, 0, 0)?;
    admit_derivative_growth(&system.residuals, &result)?;
    Ok(result)
}

/// Checks derivative proofs, complete sparsity, coloring, and compiled provenance.
///
/// Proof and bytecode acceptance are independent of `test_points`: an empty
/// diagnostic set cannot bypass either check. Optional numerical samples only
/// establish that the accepted program evaluates to finite values at those
/// points; they are not evidence for an identity or a domain-wide error bound.
pub fn verify_sparse_jacobian_certificate(
    system: &SparseJacobian,
    test_points: &[Vec<f64>],
) -> Result<SparseJacobianReceipt, JacobianVerificationError> {
    verify_sparse_jacobian_certificate_with_meter(system, test_points, &mut Unbounded)
}

/// Independently replay derivative, completeness, coloring and bytecode checks
/// using a caller-owned meter. A protected verifier meter may be used here;
/// none of its resources are exposed to candidate construction.
pub fn verify_sparse_jacobian_certificate_with_meter<M: BudgetMeter + ?Sized>(
    system: &SparseJacobian,
    test_points: &[Vec<f64>],
    meter: &mut M,
) -> Result<SparseJacobianReceipt, JacobianVerificationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    sparse_limit(system.residuals.len(), MAX_SPARSE_AXES)?;
    sparse_limit(system.vars.len(), MAX_SPARSE_AXES)?;
    sparse_limit(system.entries.len(), MAX_SPARSE_SOURCE_WORK)?;
    sparse_limit(system.pattern.nonzeros.len(), MAX_SPARSE_SOURCE_WORK)?;
    system.validate_structure_with_meter(meter)?;
    let m = system.pattern.num_rows;
    let n = system.pattern.num_cols;
    let mut var_map = HashMap::new();
    charge_sparse_buffer::<(Symbol, usize), _>(n, meter)?;
    var_map
        .try_reserve(n)
        .map_err(|_| EvalError::AllocationFailure)?;
    for (column, variable) in system.vars.iter().enumerate() {
        admit_compilation_name(&variable.name, meter)?;
        charge_compilation_work(meter, variable.name.len() + 1, variable.name.len(), 1)?;
        var_map.insert(variable.clone(), column);
    }
    let dependencies = verifier_dependencies(system, &var_map, meter)?;
    let mut proof_units = 0u64;
    let mut operation_count = 0usize;
    let mut derivative_nodes = 0usize;
    for entry in &system.entries {
        derivative_nodes = derivative_nodes.saturating_add(validate_expression_limits_with_meter(
            &entry.symbolic_deriv,
            meter,
        )?);
        sparse_limit(derivative_nodes, MAX_SPARSE_SOURCE_WORK)?;
        charge_compilation_work(meter, entry.derivation.steps.len() + 1, 0, 0)?;
        let units = derivation_verification_units(&entry.derivation).map_err(|error| {
            JacobianVerificationError::ProofFailed {
                row: entry.row,
                col: entry.col,
                reason: error.to_string(),
            }
        })?;
        proof_units = proof_units.saturating_add(units);
        sparse_limit(
            usize::try_from(proof_units).unwrap_or(usize::MAX),
            MAX_SPARSE_SOURCE_WORK,
        )?;
        operation_count = operation_count.saturating_add(entry.compiled_deriv.ops().len());
        sparse_limit(operation_count, MAX_SPARSE_SOURCE_WORK)?;
        charge_compilation_work(
            meter,
            usize::try_from(units)
                .map_err(|_| CompileError::DimensionOverflow)?
                .saturating_mul(64),
            0,
            0,
        )?;
    }

    for entry in &system.entries {
        charge_derivative_kernel(&system.residuals[entry.row], meter)?;
        charge_expression_copy(&entry.symbolic_deriv, meter)?;
        verify_diff_derivation(
            &entry.derivation,
            &system.residuals[entry.row],
            &system.vars[entry.col],
            &entry.symbolic_deriv,
        )
        .map_err(|error| JacobianVerificationError::ProofFailed {
            row: entry.row,
            col: entry.col,
            reason: error.to_string(),
        })?;
        charge_compilation_work(meter, 1, 0, 0)?;
        if !compiled_expression_matches_with_meter(
            &entry.symbolic_deriv,
            &var_map,
            &entry.compiled_deriv,
            meter,
        )? {
            return Err(JacobianVerificationError::CompiledProvenanceMismatch {
                row: entry.row,
                col: entry.col,
            });
        }
    }

    // Absence from the verifier's own raw occurrence traversal proves
    // source independence without constructing any derivative or zero program.
    // For every omitted *present* axis, use the independent reference rule
    // table. All stored entries (including optional explicit zeros) were
    // checked above, whether or not their axes occur in the source.
    let mut present = system.pattern.nonzeros.iter().peekable();
    for (row, columns) in dependencies.iter().enumerate() {
        for &col in columns {
            charge_compilation_work(meter, 1, 0, 0)?;
            while present
                .peek()
                .is_some_and(|&&coordinate| coordinate < (row, col))
            {
                charge_compilation_work(meter, 1, 0, 0)?;
                present.next();
            }
            if present.peek() == Some(&&(row, col)) {
                present.next();
            } else {
                charge_derivative_kernel(&system.residuals[row], meter)?;
                let derivative =
                    reference_sub_derivative(&system.residuals[row], &system.vars[col]);
                charge_compilation_work(meter, 1, 0, 0)?;
                if !derivative.is_zero() {
                    return Err(JacobianVerificationError::OmittedNonzero {
                        row,
                        col,
                        deriv: derivative.to_string(),
                    });
                }
            }
        }
    }

    for (point_idx, point) in test_points.iter().enumerate() {
        charge_compilation_work(meter, point.len() + 1, 0, 0)?;
        if point.len() != n || point.iter().any(|value| !value.is_finite()) {
            return Err(JacobianVerificationError::InvalidTestPoint {
                point_idx,
                expected: n,
            });
        }
        for entry in &system.entries {
            if !entry
                .compiled_deriv
                .try_eval_with_meter(point, meter)?
                .is_finite()
            {
                return Err(JacobianVerificationError::NonFiniteEvaluation {
                    point_idx,
                    row: entry.row,
                    col: entry.col,
                });
            }
        }
    }

    charge_compilation_work(meter, 1, std::mem::size_of::<SparseJacobianReceipt>(), 0)?;
    Ok(SparseJacobianReceipt {
        num_residuals: m,
        num_vars: n,
        nnz: system.entries.len(),
        density: system.density(),
        num_colors: system.num_colors,
        proofs_verified: system.entries.len(),
        test_points_evaluated: test_points.len(),
    })
}

/// Bound recursive reference work before traversing untrusted symbolic inputs.
fn validate_expression_limits_with_meter<M: BudgetMeter + ?Sized>(
    expr: &Expr,
    meter: &mut M,
) -> Result<usize, CompileError> {
    fn visit<M: BudgetMeter + ?Sized>(
        expr: &Expr,
        depth: usize,
        nodes: &mut usize,
        meter: &mut M,
    ) -> Result<(), CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        if depth > MAX_COMPILE_DEPTH {
            return Err(CompileError::DepthLimitExceeded {
                depth,
                max: MAX_COMPILE_DEPTH,
            });
        }
        *nodes += 1;
        if *nodes > MAX_COMPILE_OPS {
            return Err(CompileError::OpLimitExceeded {
                ops: *nodes,
                max: MAX_COMPILE_OPS,
            });
        }
        match expr {
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                for arg in args {
                    charge_compilation_work(meter, 1, 0, 0)?;
                    visit(arg, depth + 1, nodes, meter)?;
                }
            }
            Expr::Pow(base, exponent) => {
                charge_compilation_work(meter, 1, 0, 0)?;
                visit(base, depth + 1, nodes, meter)?;
                charge_compilation_work(meter, 1, 0, 0)?;
                visit(exponent, depth + 1, nodes, meter)?;
            }
            _ => {}
        }
        Ok(())
    }
    let mut nodes = 0usize;
    let result = visit(expr, 0, &mut nodes, meter).map(|()| nodes);
    compilation_finish(result, meter)
}

/// Independent structural check of the canonical numerical lowering.
///
/// This consumes an expression and postfix operations together; it does not
/// invoke the compiler or infer provenance from sampled numerical agreement.
/// It checks the declared binary64 lowering, not exact-real evaluation accuracy.
pub fn compiled_expression_matches(
    expr: &Expr,
    var_map: &HashMap<Symbol, usize>,
    compiled: &CompiledExpr,
) -> Result<bool, CompileError> {
    compiled_expression_matches_with_meter(expr, var_map, compiled, &mut Unbounded)
}

/// Independently check an existing program using the caller's cumulative fuel.
///
/// Both the bounded source preflight and the source/opcode traversal perform
/// actual charged work. Literal rounding uses governed exact arithmetic, and
/// a final cancellation checkpoint covers acceptance, mismatch and refusal.
/// This never invokes the compiler or trusts numerical sample agreement.
pub fn compiled_expression_matches_with_meter<M: BudgetMeter + ?Sized>(
    expr: &Expr,
    var_map: &HashMap<Symbol, usize>,
    compiled: &CompiledExpr,
    meter: &mut M,
) -> Result<bool, CompileError> {
    struct Checker<'a, 'm, M: ?Sized> {
        operations: &'a [CompiledOp],
        position: usize,
        variables: &'a HashMap<Symbol, usize>,
        meter: &'m mut M,
    }

    impl<M: BudgetMeter + ?Sized> Checker<'_, '_, M> {
        fn take(&mut self, expected: CompiledOp) -> Result<bool, CompileError> {
            charge_compilation_work(self.meter, 1, 0, 0)?;
            let matches = match (self.operations.get(self.position), &expected) {
                (Some(CompiledOp::LoadConst(actual)), CompiledOp::LoadConst(expected))
                | (Some(CompiledOp::Pow(actual)), CompiledOp::Pow(expected)) => {
                    actual.to_bits() == expected.to_bits()
                }
                (Some(actual), expected) => actual == expected,
                (None, _) => false,
            };
            if matches {
                self.position += 1;
            }
            Ok(matches)
        }

        // Read control-flow operands independently of the generator. Their
        // values are accepted only after the corresponding source boundary
        // has been traversed, so changing a branch target cannot preserve the
        // source/bytecode binding merely by leaving the arithmetic unchanged.
        fn jump(&mut self, conditional: bool) -> Result<Option<usize>, CompileError> {
            charge_compilation_work(self.meter, 1, 0, 0)?;
            let target = match self.operations.get(self.position) {
                Some(CompiledOp::JumpIfFalse(target)) if conditional => Some(*target),
                Some(CompiledOp::Jump(target)) if !conditional => Some(*target),
                _ => None,
            };
            if target.is_some() {
                self.position += 1;
            }
            Ok(target)
        }

        fn piecewise(&mut self, branches: &[Expr], depth: usize) -> Result<bool, CompileError> {
            if branches.is_empty() {
                return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                    "Piecewise requires at least one (value, condition) branch",
                    self.meter,
                )?));
            }
            let mut common_end = None;
            for branch in branches {
                charge_compilation_work(self.meter, 1, 0, 0)?;
                let Expr::Function(tag, pair) = branch else {
                    return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                        "Piecewise requires Tuple(value, condition) branches",
                        self.meter,
                    )?));
                };
                admit_compilation_name(tag, self.meter)?;
                if tag != "Tuple" || pair.len() != 2 {
                    return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                        "Piecewise requires Tuple(value, condition) branches",
                        self.meter,
                    )?));
                }
                if !self.condition(&pair[1], depth + 2)? {
                    return Ok(false);
                }
                let Some(next_branch) = self.jump(true)? else {
                    return Ok(false);
                };
                if !self.visit(&pair[0], depth + 2)? {
                    return Ok(false);
                }
                let Some(end) = self.jump(false)? else {
                    return Ok(false);
                };
                if next_branch != self.position || common_end.is_some_and(|target| target != end) {
                    return Ok(false);
                }
                common_end = Some(end);
            }
            if !self.take(CompiledOp::NoMatchingPiecewiseBranch)? {
                return Ok(false);
            }
            Ok(common_end == Some(self.position))
        }

        fn condition(&mut self, expr: &Expr, depth: usize) -> Result<bool, CompileError> {
            charge_compilation_work(self.meter, 1, 0, 0)?;
            if depth > MAX_COMPILE_DEPTH {
                return Err(CompileError::DepthLimitExceeded {
                    depth,
                    max: MAX_COMPILE_DEPTH,
                });
            }
            if let Expr::Sym(symbol) = expr {
                admit_compilation_name(&symbol.name, self.meter)?;
                if symbol.is_plain() && matches!(symbol.name.as_str(), "True" | "False") {
                    return self.take(CompiledOp::LoadConst(if symbol.name == "True" {
                        1.0
                    } else {
                        0.0
                    }));
                }
            }
            if let Expr::Function(name, arguments) = expr {
                admit_compilation_name(name, self.meter)?;
                charge_compilation_work(self.meter, name.len() + 1, 0, 0)?;
                if name == "Not" && arguments.len() == 1 {
                    return Ok(
                        self.condition(&arguments[0], depth + 1)? && self.take(CompiledOp::Not)?
                    );
                }
                if name == "And" || name == "Or" {
                    let is_and = name == "And";
                    let Some((last, prefix)) = arguments.split_last() else {
                        return self.take(CompiledOp::LoadConst(if is_and { 1.0 } else { 0.0 }));
                    };
                    if prefix.is_empty() {
                        return self.condition(last, depth + 1);
                    }
                    let mut common_shortcut = None;
                    for argument in prefix {
                        if !self.condition(argument, depth + 1)?
                            || (!is_and && !self.take(CompiledOp::Not)?)
                        {
                            return Ok(false);
                        }
                        let Some(shortcut) = self.jump(true)? else {
                            return Ok(false);
                        };
                        if common_shortcut.is_some_and(|target| target != shortcut) {
                            return Ok(false);
                        }
                        common_shortcut = Some(shortcut);
                    }
                    if !self.condition(last, depth + 1)? {
                        return Ok(false);
                    }
                    let Some(end) = self.jump(false)? else {
                        return Ok(false);
                    };
                    if common_shortcut != Some(self.position)
                        || !self.take(CompiledOp::LoadConst(if is_and { 0.0 } else { 1.0 }))?
                    {
                        return Ok(false);
                    }
                    return Ok(end == self.position);
                }
                if arguments.len() == 2 {
                    let operation = match name.as_str() {
                        "Equality" => Some(CompiledOp::Equal),
                        "Unequality" => Some(CompiledOp::NotEqual),
                        "StrictLessThan" => Some(CompiledOp::Less),
                        "LessThan" => Some(CompiledOp::LessEqual),
                        "StrictGreaterThan" => Some(CompiledOp::Greater),
                        "GreaterThan" => Some(CompiledOp::GreaterEqual),
                        _ => None,
                    };
                    if let Some(operation) = operation {
                        return Ok(self.visit(&arguments[0], depth + 1)?
                            && self.visit(&arguments[1], depth + 1)?
                            && self.take(operation)?);
                    }
                }
            }
            Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                "Piecewise condition requires a boolean, comparison, And, Or or Not",
                self.meter,
            )?))
        }

        fn number(&mut self, value: &BigInt) -> Result<f64, CompileError> {
            charge_compilation_work(self.meter, 1, 0, 0)?;
            let bits = value.bits();
            if bits <= 1024 {
                charge_compilation_work(self.meter, bits.div_ceil(32).max(1) as usize, 0, 0)?;
                if let Some(converted) = value.to_f64().filter(|value| value.is_finite()) {
                    return compilation_finish(Ok(converted), self.meter);
                }
            }
            Err(CompileError::NumericConversion(compilation_diagnostic(
                "integer literal does not round to finite binary64",
                self.meter,
            )?))
        }

        fn exponent_component(&mut self, value: &BigInt) -> Result<f64, CompileError> {
            charge_compilation_work(self.meter, 1, 0, 0)?;
            if value.bits() > u64::from(f64::MANTISSA_DIGITS) {
                return Err(CompileError::UnsupportedExponent(compilation_diagnostic(
                    "literal components must be exactly representable within 53 bits",
                    self.meter,
                )?));
            }
            self.number(value)
        }

        fn variable(&mut self, symbol: &Symbol) -> Result<usize, CompileError> {
            admit_compilation_name(&symbol.name, self.meter)?;
            // A capacity-based logical allowance covers sparse/tombstoned
            // standard-library tables. The lookup is a precharged cooperative
            // kernel, with no physical instruction or wall-time guarantee.
            let per_candidate = symbol
                .name
                .len()
                .checked_add(128)
                .ok_or(CompileError::DimensionOverflow)?;
            let steps = self
                .variables
                .capacity()
                .checked_add(1)
                .and_then(|count| count.checked_mul(2))
                .and_then(|count| count.checked_mul(per_candidate))
                .ok_or(CompileError::DimensionOverflow)?;
            charge_compilation_work(self.meter, steps, 0, 0)?;
            // Interpret the declared typed coordinate map here. In particular,
            // never reuse a generator helper that could return a wrong slot
            // and make generation and checking agree on that same mistake.
            if let Some(&index) = self.variables.get(symbol) {
                return compilation_finish(Ok(index), self.meter);
            }
            let bytes = symbol.name.len();
            charge_compilation_work(self.meter, bytes, bytes, usize::from(bytes != 0))?;
            let mut name = String::new();
            name.try_reserve_exact(bytes)
                .map_err(|_| CompileError::AllocationFailure)?;
            name.push_str(&symbol.name);
            compilation_finish(
                Err(CompileError::UnmappedVariable(Symbol {
                    name,
                    identity: symbol.identity,
                })),
                self.meter,
            )
        }

        fn visit(&mut self, expr: &Expr, depth: usize) -> Result<bool, CompileError> {
            charge_compilation_work(self.meter, 1, 0, 0)?;
            if depth > MAX_COMPILE_DEPTH {
                return Err(CompileError::DepthLimitExceeded {
                    depth,
                    max: MAX_COMPILE_DEPTH,
                });
            }
            let operation = match expr {
                Expr::Integer(integer) => CompiledOp::LoadConst(self.number(integer)?),
                Expr::Rational(rational) => {
                    let value = rational
                        .metered_to_f64(self.meter)?
                        .filter(|converted| converted.is_finite());
                    let Some(value) = value else {
                        return Err(CompileError::NumericConversion(compilation_diagnostic(
                            "exact rational quotient does not round to finite binary64",
                            self.meter,
                        )?));
                    };
                    CompiledOp::LoadConst(if value == 0.0 { 0.0 } else { value })
                }
                Expr::Const(constant) => CompiledOp::LoadConst(match constant {
                    Constant::Pi => std::f64::consts::PI,
                    Constant::E => std::f64::consts::E,
                    Constant::Infinity => f64::INFINITY,
                    Constant::NegativeInfinity => f64::NEG_INFINITY,
                    _ => {
                        return Err(CompileError::UnsupportedExpression(compilation_diagnostic(
                            "complex infinity, imaginary unit or NaN",
                            self.meter,
                        )?));
                    }
                }),
                Expr::Sym(symbol) => CompiledOp::LoadVar(self.variable(symbol)?),
                Expr::Add(arguments) | Expr::Mul(arguments) => {
                    let is_sum = matches!(expr, Expr::Add(_));
                    if arguments.is_empty() {
                        CompiledOp::LoadConst(if is_sum { 0.0 } else { 1.0 })
                    } else {
                        for argument in arguments {
                            charge_compilation_work(self.meter, 1, 0, 0)?;
                            if !self.visit(argument, depth + 1)? {
                                return Ok(false);
                            }
                        }
                        if is_sum {
                            CompiledOp::Add(arguments.len())
                        } else {
                            CompiledOp::Mul(arguments.len())
                        }
                    }
                }
                Expr::Pow(base, exponent) => {
                    charge_compilation_work(self.meter, 1, 0, 0)?;
                    if !self.visit(base, depth + 1)? {
                        return Ok(false);
                    }
                    charge_compilation_work(self.meter, 1, 0, 0)?;
                    let literal = match exponent.as_ref() {
                        Expr::Integer(integer) => Some(self.exponent_component(integer)?),
                        Expr::Rational(rational) => {
                            let numerator = self.exponent_component(rational.numer())?;
                            let denominator = self.exponent_component(rational.denom())?;
                            charge_compilation_work(self.meter, 1, 0, 0)?;
                            Some(numerator / denominator)
                        }
                        _ => {
                            if !self.visit(exponent, depth + 1)? {
                                return Ok(false);
                            }
                            None
                        }
                    };
                    if let Some(value) = literal {
                        if !value.is_finite() {
                            return Err(CompileError::UnsupportedExponent(compilation_diagnostic(
                                "literal exponent quotient is not finite",
                                self.meter,
                            )?));
                        }
                        CompiledOp::Pow(value)
                    } else {
                        CompiledOp::PowPositive
                    }
                }
                Expr::Function(name, arguments) => {
                    admit_compilation_name(name, self.meter)?;
                    if name == "Piecewise" {
                        return self.piecewise(arguments, depth);
                    }
                    if arguments.len() != 1 {
                        return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                            name, self.meter,
                        )?));
                    }
                    charge_compilation_work(self.meter, name.len() + 1, 0, 0)?;
                    if !self.visit(&arguments[0], depth + 1)? {
                        return Ok(false);
                    }
                    match name.as_str() {
                        "sin" => CompiledOp::Sin,
                        "cos" => CompiledOp::Cos,
                        "tan" => CompiledOp::Tan,
                        "asin" => CompiledOp::Asin,
                        "acos" => CompiledOp::Acos,
                        "atan" => CompiledOp::Atan,
                        "sinh" => CompiledOp::Sinh,
                        "cosh" => CompiledOp::Cosh,
                        "tanh" => CompiledOp::Tanh,
                        "asinh" => CompiledOp::Asinh,
                        "acosh" => CompiledOp::Acosh,
                        "atanh" => CompiledOp::Atanh,
                        "erf" => CompiledOp::Erf,
                        "erfc" => CompiledOp::Erfc,
                        "exp" => CompiledOp::Exp,
                        "log" | "ln" => CompiledOp::Ln,
                        "sqrt" => CompiledOp::Sqrt,
                        "abs" => CompiledOp::Abs,
                        _ => {
                            return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                                name, self.meter,
                            )?));
                        }
                    }
                }
            };
            self.take(operation)
        }
    }

    let result = (|| {
        validate_expression_limits_with_meter(expr, meter)?;
        let mut checker = Checker {
            operations: compiled.ops(),
            position: 0,
            variables: var_map,
            meter: &mut *meter,
        };
        let matches = checker.visit(expr, 0)?;
        charge_compilation_work(checker.meter, 1, 0, 0)?;
        Ok(matches && checker.position == checker.operations.len())
    })();
    compilation_finish(result, meter)
}
