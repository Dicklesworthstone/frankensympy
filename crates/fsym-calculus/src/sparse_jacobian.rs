//! Sparse Jacobian analysis, verified differentiation proofs, and graph coloring (WS12 / C7 gate).

#![forbid(unsafe_code)]

use crate::compile::{
    CompileError, CompiledExpr, CompiledOp, EvalError, MAX_COMPILE_DEPTH, MAX_COMPILE_OPS,
};
use crate::proof::{reference_sub_derivative, verified_diff, verify_diff_derivation};
use fsym_core::{BigInt, Constant, Expr, Symbol};
use fsym_proof_kernel::DerivationTree;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

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

impl SparseJacobian {
    /// Builds a compiled sparse Jacobian system with verified differentiation proofs and graph coloring.
    pub fn try_build(residuals: &[Expr], vars: &[Symbol]) -> Result<Self, CompileError> {
        residuals
            .len()
            .checked_mul(vars.len())
            .ok_or(CompileError::DimensionOverflow)?;
        for residual in residuals {
            validate_expression_limits(residual)?;
        }
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

    /// Checks all dimensions, coordinates, and colors before any indexed access.
    ///
    /// This is a representation check, not mathematical proof. In particular,
    /// callers importing a certificate must also run
    /// [`verify_sparse_jacobian_certificate`] to check the derivatives and their
    /// compiled programs. Public fields and deserialized objects are untrusted.
    pub fn validate_structure(&self) -> Result<(), JacobianVerificationError> {
        let m = self.pattern.num_rows;
        let n = self.pattern.num_cols;
        m.checked_mul(n).ok_or(EvalError::DimensionOverflow)?;
        if m != self.residuals.len() || n != self.vars.len() {
            return Err(JacobianVerificationError::ShapeMismatch {
                rows: m,
                cols: n,
                residuals: self.residuals.len(),
                variables: self.vars.len(),
            });
        }
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
        variables
            .try_reserve(n)
            .map_err(|_| EvalError::AllocationFailure)?;
        for variable in &self.vars {
            if !variables.insert(variable) {
                return Err(JacobianVerificationError::DuplicateVariable(
                    variable.clone(),
                ));
            }
        }

        for (index, (&(row, col), entry)) in
            self.pattern.nonzeros.iter().zip(&self.entries).enumerate()
        {
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
        used_colors
            .try_reserve(self.num_colors)
            .map_err(|_| EvalError::AllocationFailure)?;
        for (column, &color) in self.column_colors.iter().enumerate() {
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
        row_colors
            .try_reserve(n.min(self.entries.len()))
            .map_err(|_| EvalError::AllocationFailure)?;
        let mut previous_row = None;
        for entry in &self.entries {
            if previous_row != Some(entry.row) {
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
        Ok(())
    }

    /// Number of structural non-zero entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Matrix density ($nnz / (m \times n)$).
    pub fn density(&self) -> f64 {
        // Do not panic on unverified public metadata. Certificate acceptance
        // checks the integer product and the actual vector dimensions first.
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
        self.validate_for_evaluation()?;
        if out_values.len() != self.entries.len() {
            return Err(EvalError::JacobianBufferMismatch {
                expected: self.entries.len(),
                actual: out_values.len(),
            });
        }
        let evaluated = self.evaluate_entries(x)?;
        out_values.copy_from_slice(&evaluated);
        Ok(())
    }

    /// Evaluates the sparse Jacobian into a dense row-major buffer.
    /// A failure leaves the caller's buffer unchanged.
    pub fn try_eval_dense(&self, x: &[f64], out_dense: &mut [f64]) -> Result<(), EvalError> {
        self.validate_for_evaluation()?;
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
        let evaluated = self.evaluate_entries(x)?;
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
        self.validate_for_evaluation()?;
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
        let evaluated = self.evaluate_entries(x)?;
        out_compressed.fill(0.0);
        for (entry, val) in self.entries.iter().zip(evaluated) {
            let col_color = self.column_colors[entry.col];
            out_compressed[entry.row * self.num_colors + col_color] = val;
        }
        Ok(())
    }

    fn validate_for_evaluation(&self) -> Result<(), EvalError> {
        self.validate_structure()
            .map_err(|error| EvalError::InvalidSparseJacobian(error.to_string()))
    }

    fn evaluate_entries(&self, x: &[f64]) -> Result<Vec<f64>, EvalError> {
        let mut evaluated = Vec::new();
        evaluated
            .try_reserve_exact(self.entries.len())
            .map_err(|_| EvalError::AllocationFailure)?;
        for entry in &self.entries {
            evaluated.push(entry.compiled_deriv.try_eval(x)?);
        }
        Ok(evaluated)
    }
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
    system.validate_structure()?;
    let m = system.pattern.num_rows;
    let n = system.pattern.num_cols;
    let mut var_map = HashMap::new();
    var_map
        .try_reserve(n)
        .map_err(|_| EvalError::AllocationFailure)?;
    for (column, variable) in system.vars.iter().enumerate() {
        var_map.insert(variable.clone(), column);
    }
    for residual in &system.residuals {
        validate_expression_limits(residual)?;
    }

    for entry in &system.entries {
        validate_expression_limits(&entry.symbolic_deriv)?;
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
        if !compiled_expression_matches(&entry.symbolic_deriv, &var_map, &entry.compiled_deriv)? {
            return Err(JacobianVerificationError::CompiledProvenanceMismatch {
                row: entry.row,
                col: entry.col,
            });
        }
    }

    // Coordinates are canonical, so a monotone cursor proves coverage without
    // trusting a second, potentially different coordinate set. Zero checks use
    // the verifier's reference derivative, never the optimizing generator.
    let mut present = system.pattern.nonzeros.iter().peekable();
    for row in 0..m {
        for col in 0..n {
            if present.peek() == Some(&&(row, col)) {
                present.next();
            } else {
                let derivative =
                    reference_sub_derivative(&system.residuals[row], &system.vars[col]);
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
        if point.len() != n || point.iter().any(|value| !value.is_finite()) {
            return Err(JacobianVerificationError::InvalidTestPoint {
                point_idx,
                expected: n,
            });
        }
        for entry in &system.entries {
            if !entry.compiled_deriv.try_eval(point)?.is_finite() {
                return Err(JacobianVerificationError::NonFiniteEvaluation {
                    point_idx,
                    row: entry.row,
                    col: entry.col,
                });
            }
        }
    }

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
fn validate_expression_limits(expr: &Expr) -> Result<(), CompileError> {
    fn visit(expr: &Expr, depth: usize, nodes: &mut usize) -> Result<(), CompileError> {
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
                    visit(arg, depth + 1, nodes)?;
                }
            }
            Expr::Pow(base, exponent) => {
                visit(base, depth + 1, nodes)?;
                visit(exponent, depth + 1, nodes)?;
            }
            _ => {}
        }
        Ok(())
    }
    visit(expr, 0, &mut 0)
}

/// Independent structural check of the canonical numerical lowering.
///
/// This consumes an expression and postfix operations together; it does not
/// invoke the compiler or infer provenance from sampled numerical agreement.
/// It checks the declared binary64 lowering, not exact-real evaluation accuracy.
pub(crate) fn compiled_expression_matches(
    expr: &Expr,
    var_map: &HashMap<Symbol, usize>,
    compiled: &CompiledExpr,
) -> Result<bool, CompileError> {
    struct Checker<'a> {
        operations: &'a [CompiledOp],
        position: usize,
        variables: &'a HashMap<Symbol, usize>,
    }

    impl Checker<'_> {
        fn take(&mut self, expected: CompiledOp) -> bool {
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
            matches
        }

        fn number(value: &BigInt) -> Result<f64, CompileError> {
            let encoded = value.to_string();
            let converted = encoded
                .parse::<f64>()
                .map_err(|_| CompileError::NumericConversion(encoded.clone()))?;
            if converted.is_finite() {
                Ok(converted)
            } else {
                Err(CompileError::NumericConversion(encoded))
            }
        }

        fn exponent_component(value: &BigInt) -> Result<f64, CompileError> {
            if value.bits() > u64::from(f64::MANTISSA_DIGITS) {
                return Err(CompileError::UnsupportedExponent(value.to_string()));
            }
            Self::number(value)
        }

        fn visit(&mut self, expr: &Expr, depth: usize) -> Result<bool, CompileError> {
            if depth > MAX_COMPILE_DEPTH {
                return Err(CompileError::DepthLimitExceeded {
                    depth,
                    max: MAX_COMPILE_DEPTH,
                });
            }
            let operation = match expr {
                Expr::Integer(integer) => CompiledOp::LoadConst(Self::number(integer)?),
                Expr::Rational(rational) => {
                    let numerator = Self::number(rational.numer())?;
                    let denominator = Self::number(rational.denom())?;
                    let value = numerator / denominator;
                    if !value.is_finite() || denominator == 0.0 {
                        return Err(CompileError::NumericConversion(rational.to_string()));
                    }
                    CompiledOp::LoadConst(value)
                }
                Expr::Const(constant) => CompiledOp::LoadConst(match constant {
                    Constant::Pi => std::f64::consts::PI,
                    Constant::E => std::f64::consts::E,
                    Constant::Infinity => f64::INFINITY,
                    Constant::NegativeInfinity => f64::NEG_INFINITY,
                    _ => return Err(CompileError::UnsupportedExpression(format!("{expr:?}"))),
                }),
                Expr::Sym(symbol) => CompiledOp::LoadVar(
                    *self
                        .variables
                        .get(symbol)
                        .ok_or_else(|| CompileError::UnmappedVariable(symbol.clone()))?,
                ),
                Expr::Add(arguments) | Expr::Mul(arguments) => {
                    let is_sum = matches!(expr, Expr::Add(_));
                    if arguments.is_empty() {
                        CompiledOp::LoadConst(if is_sum { 0.0 } else { 1.0 })
                    } else {
                        for argument in arguments {
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
                    if !self.visit(base, depth + 1)? {
                        return Ok(false);
                    }
                    let value = match exponent.as_ref() {
                        Expr::Integer(integer) => Self::exponent_component(integer)?,
                        Expr::Rational(rational) => {
                            Self::exponent_component(rational.numer())?
                                / Self::exponent_component(rational.denom())?
                        }
                        _ => return Err(CompileError::UnsupportedExponent(exponent.to_string())),
                    };
                    if !value.is_finite() {
                        return Err(CompileError::UnsupportedExponent(exponent.to_string()));
                    }
                    CompiledOp::Pow(value)
                }
                Expr::Function(name, arguments) => {
                    if arguments.len() != 1 {
                        return Err(CompileError::UnsupportedFunction(name.clone()));
                    }
                    if !self.visit(&arguments[0], depth + 1)? {
                        return Ok(false);
                    }
                    match name.as_str() {
                        "sin" => CompiledOp::Sin,
                        "cos" => CompiledOp::Cos,
                        "tan" => CompiledOp::Tan,
                        "sinh" => CompiledOp::Sinh,
                        "cosh" => CompiledOp::Cosh,
                        "tanh" => CompiledOp::Tanh,
                        "exp" => CompiledOp::Exp,
                        "log" | "ln" => CompiledOp::Ln,
                        "sqrt" => CompiledOp::Sqrt,
                        "abs" => CompiledOp::Abs,
                        _ => return Err(CompileError::UnsupportedFunction(name.clone())),
                    }
                }
            };
            Ok(self.take(operation))
        }
    }

    validate_expression_limits(expr)?;
    let mut checker = Checker {
        operations: compiled.ops(),
        position: 0,
        variables: var_map,
    };
    Ok(checker.visit(expr, 0)? && checker.position == checker.operations.len())
}
