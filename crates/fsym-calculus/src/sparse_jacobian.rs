//! Sparse Jacobian analysis, verified differentiation proofs, and graph coloring (WS12 / C7 gate).

#![forbid(unsafe_code)]

use crate::compile::{CompileError, CompiledExpr, CompiledResidualSystem, EvalError};
use crate::diff;
use crate::proof::{verified_diff, verify_diff_derivation};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::DerivationTree;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// A single non-zero entry in a symbolically differentiated sparse Jacobian matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
pub struct SparsityPattern {
    pub num_rows: usize,
    pub num_cols: usize,
    /// Sorted list of (row, col) coordinates of structural non-zeros.
    pub nonzeros: Vec<(usize, usize)>,
    pub nnz: usize,
}

/// Compiled sparse Jacobian system with verified differentiation proofs and graph coloring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

impl SparseJacobian {
    /// Builds a compiled sparse Jacobian system with verified differentiation proofs and graph coloring.
    pub fn try_build(residuals: &[Expr], vars: &[Symbol]) -> Result<Self, CompileError> {
        let mut var_map = HashMap::new();
        var_map
            .try_reserve(vars.len())
            .map_err(|_| CompileError::AllocationFailure)?;
        for (i, v) in vars.iter().enumerate() {
            if var_map.insert(v.clone(), i).is_some() {
                return Err(CompileError::DuplicateVariable(v.clone()));
            }
        }

        let num_residuals = residuals.len();
        let num_vars = vars.len();

        let mut entries = Vec::new();
        let mut nonzeros = Vec::new();

        for (i, res) in residuals.iter().enumerate() {
            let free = res.free_symbols();
            for (j, var) in vars.iter().enumerate() {
                if free.contains(var) {
                    let (symbolic_deriv, derivation) = verified_diff(res, var);
                    if !symbolic_deriv.is_zero() {
                        let compiled_deriv = CompiledExpr::try_compile(&symbolic_deriv, &var_map)?;
                        entries.push(SparseJacobianEntry {
                            row: i,
                            col: j,
                            symbolic_deriv,
                            compiled_deriv,
                            derivation,
                        });
                        nonzeros.push((i, j));
                    }
                }
            }
        }

        // Compute Distance-1 Column Coloring (columns sharing a nonzero row cannot share color)
        let mut conflicts: Vec<HashSet<usize>> = vec![HashSet::new(); num_vars];
        let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); num_residuals];
        for &(r, c) in &nonzeros {
            row_cols[r].push(c);
        }
        for cols in &row_cols {
            for p in 0..cols.len() {
                for q in (p + 1)..cols.len() {
                    let c1 = cols[p];
                    let c2 = cols[q];
                    conflicts[c1].insert(c2);
                    conflicts[c2].insert(c1);
                }
            }
        }

        let mut column_colors = vec![0; num_vars];
        for c in 0..num_vars {
            let mut used_colors = HashSet::new();
            for &neighbor in &conflicts[c] {
                if neighbor < c {
                    used_colors.insert(column_colors[neighbor]);
                }
            }
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }
            column_colors[c] = color;
        }

        let num_colors = if num_vars == 0 {
            0
        } else {
            column_colors.iter().max().copied().unwrap_or(0) + 1
        };

        let pattern = SparsityPattern {
            num_rows: num_residuals,
            num_cols: num_vars,
            nnz: entries.len(),
            nonzeros,
        };

        Ok(Self {
            residuals: residuals.to_vec(),
            vars: vars.to_vec(),
            pattern,
            entries,
            column_colors,
            num_colors,
        })
    }

    /// Number of structural non-zero entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Matrix density ($nnz / (m \times n)$).
    pub fn density(&self) -> f64 {
        let total = self.pattern.num_rows * self.pattern.num_cols;
        if total == 0 {
            0.0
        } else {
            self.entries.len() as f64 / total as f64
        }
    }

    /// Evaluates the sparse Jacobian non-zero entries into a contiguous values slice.
    pub fn try_eval_sparse(&self, x: &[f64], out_values: &mut [f64]) -> Result<(), EvalError> {
        if out_values.len() != self.entries.len() {
            return Err(EvalError::JacobianBufferMismatch {
                expected: self.entries.len(),
                actual: out_values.len(),
            });
        }
        for (k, entry) in self.entries.iter().enumerate() {
            out_values[k] = entry.compiled_deriv.try_eval(x)?;
        }
        Ok(())
    }

    /// Evaluates the sparse Jacobian entries into a full dense row-major buffer of shape $m \times n$.
    pub fn try_eval_dense(&self, x: &[f64], out_dense: &mut [f64]) -> Result<(), EvalError> {
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
        out_dense.fill(0.0);
        for entry in &self.entries {
            let val = entry.compiled_deriv.try_eval(x)?;
            out_dense[entry.row * self.pattern.num_cols + entry.col] = val;
        }
        Ok(())
    }

    /// Evaluates the compressed Jacobian matrix of shape $m \times k$ (where $k = \text{num\_colors}$).
    pub fn try_eval_compressed(
        &self,
        x: &[f64],
        out_compressed: &mut [f64],
    ) -> Result<(), EvalError> {
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
        out_compressed.fill(0.0);
        for entry in &self.entries {
            let val = entry.compiled_deriv.try_eval(x)?;
            let col_color = self.column_colors[entry.col];
            out_compressed[entry.row * self.num_colors + col_color] = val;
        }
        Ok(())
    }
}

/// Independent reference verifier checking proof replay, exact sparsity, coloring, and evaluator agreement (C7 gate).
pub fn verify_sparse_jacobian_certificate(
    system: &SparseJacobian,
    test_points: &[Vec<f64>],
) -> Result<SparseJacobianReceipt, JacobianVerificationError> {
    let m = system.pattern.num_rows;
    let n = system.pattern.num_cols;

    if system.pattern.nonzeros.len() != system.entries.len() {
        return Err(JacobianVerificationError::PatternMismatch {
            pattern_nnz: system.pattern.nonzeros.len(),
            table_nnz: system.entries.len(),
        });
    }

    // 1. Proof replay for all structural nonzeros
    for entry in &system.entries {
        if entry.row >= m || entry.col >= n {
            return Err(JacobianVerificationError::OutOfBounds {
                row: entry.row,
                col: entry.col,
                num_rows: m,
                num_cols: n,
            });
        }
        verify_diff_derivation(
            &entry.derivation,
            &system.residuals[entry.row],
            &system.vars[entry.col],
            &entry.symbolic_deriv,
        )
        .map_err(|e| JacobianVerificationError::ProofFailed {
            row: entry.row,
            col: entry.col,
            reason: e.to_string(),
        })?;
    }

    // 2. Sparsity soundness: verify that omitted entries are mathematically zero
    let nonzeros_set: HashSet<(usize, usize)> = system.pattern.nonzeros.iter().copied().collect();
    for i in 0..m {
        for j in 0..n {
            if !nonzeros_set.contains(&(i, j)) {
                let d = diff(&system.residuals[i], &system.vars[j]);
                if !d.is_zero() {
                    return Err(JacobianVerificationError::OmittedNonzero {
                        row: i,
                        col: j,
                        deriv: d.to_string(),
                    });
                }
            }
        }
    }

    // 3. Distance-1 Coloring validity: no two nonzeros in the same row share a color
    let mut row_entries: Vec<Vec<usize>> = vec![Vec::new(); m];
    for entry in &system.entries {
        row_entries[entry.row].push(entry.col);
    }
    for (r, cols) in row_entries.iter().enumerate() {
        let mut seen_colors: HashMap<usize, usize> = HashMap::new();
        for &c in cols {
            let color = system.column_colors[c];
            if let Some(&first_col) = seen_colors.get(&color) {
                return Err(JacobianVerificationError::ColoringCollision {
                    row: r,
                    col1: first_col,
                    col2: c,
                    color,
                });
            }
            seen_colors.insert(color, c);
        }
    }

    // 4. Evaluator agreement: compare compiled sparse evaluator against dense compiled system
    let dense_sys = CompiledResidualSystem::try_compile(&system.residuals, &system.vars)?;
    for (pt_idx, pt) in test_points.iter().enumerate() {
        let mut sparse_dense = vec![0.0; m * n];
        system.try_eval_dense(pt, &mut sparse_dense)?;

        let mut dense_ref = vec![0.0; m * n];
        dense_sys.try_eval_jacobian(pt, &mut dense_ref)?;

        for i in 0..m {
            for j in 0..n {
                let s_val = sparse_dense[i * n + j];
                let d_val = dense_ref[i * n + j];
                let diff = (s_val - d_val).abs();
                let scale = s_val.abs() + d_val.abs() + 1.0;
                if diff > 1e-12 && diff / scale > 1e-12 {
                    return Err(JacobianVerificationError::EvaluatorDisagreement {
                        point_idx: pt_idx,
                        row: i,
                        col: j,
                        sparse: s_val,
                        dense: d_val,
                    });
                }
            }
        }
    }

    let density = system.density();
    Ok(SparseJacobianReceipt {
        num_residuals: m,
        num_vars: n,
        nnz: system.entries.len(),
        density,
        num_colors: system.num_colors,
        proofs_verified: system.entries.len(),
        test_points_evaluated: test_points.len(),
    })
}
