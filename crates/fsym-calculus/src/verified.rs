//! Immutable bridge from checked symbolic Jacobians to numerical consumers.
//!
//! The verified artifact retains the source residuals, variable order and
//! independent verification receipt alongside the exact bytecode that was
//! checked. Binary64 evaluation remains approximate; derivative proofs do not
//! certify a numerical root or eliminate the need for domain/enclosure checks.

#![forbid(unsafe_code)]

use crate::compile::{CompileError, CompiledExpr, CompiledOp, CompiledResidualSystem};
use crate::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, SparseJacobianReceipt, compiled_expression_matches,
    verify_sparse_jacobian_certificate,
};
use fsym_core::{Expr, Symbol};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum VerifiedCompilationError {
    #[error("Jacobian verification failed: {0}")]
    Jacobian(#[from] JacobianVerificationError),
    #[error("Residual compilation failed: {0}")]
    Compilation(#[from] CompileError),
    #[error("Compiled residual {residual} does not match its symbolic source")]
    ResidualProvenanceMismatch { residual: usize },
}

/// A compiled system bound to independently checked derivative proofs and
/// symbolic sources. The fields are private and there is no unchecked decoder:
/// obtaining this type requires verification, including bytecode provenance.
#[derive(Debug, Clone)]
pub struct VerifiedResidualSystem {
    compiled: CompiledResidualSystem,
    residuals: Vec<Expr>,
    receipt: SparseJacobianReceipt,
}

impl VerifiedResidualSystem {
    /// Builds derivatives, checks their proofs and compiled semantics, then
    /// carries those exact derivative programs into the consumer artifact.
    pub fn try_compile(
        residuals: &[Expr],
        vars: &[Symbol],
    ) -> Result<Self, VerifiedCompilationError> {
        Self::try_from_sparse_jacobian(&SparseJacobian::try_build(residuals, vars)?)
    }

    /// Admits a possibly untrusted sparse artifact. No numerical sample is
    /// required for symbolic or bytecode acceptance. Omitted entries have been
    /// independently checked to be zero before the dense consumer layout is
    /// filled, and nonzero entries are copied rather than differentiated again.
    pub fn try_from_sparse_jacobian(
        jacobian: &SparseJacobian,
    ) -> Result<Self, VerifiedCompilationError> {
        let receipt = verify_sparse_jacobian_certificate(jacobian, &[])?;
        let num_residuals = jacobian.residuals.len();
        let num_vars = jacobian.vars.len();
        let dense_len = num_residuals
            .checked_mul(num_vars)
            .ok_or(CompileError::DimensionOverflow)?;
        let mut var_map = HashMap::new();
        var_map
            .try_reserve(num_vars)
            .map_err(|_| CompileError::AllocationFailure)?;
        for (index, variable) in jacobian.vars.iter().enumerate() {
            var_map.insert(variable.clone(), index);
        }

        let mut compiled_residuals = Vec::new();
        compiled_residuals
            .try_reserve_exact(num_residuals)
            .map_err(|_| CompileError::AllocationFailure)?;
        for (residual, source) in jacobian.residuals.iter().enumerate() {
            let compiled = CompiledExpr::try_compile(source, &var_map)?;
            if !compiled_expression_matches(source, &var_map, &compiled)? {
                return Err(VerifiedCompilationError::ResidualProvenanceMismatch { residual });
            }
            compiled_residuals.push(compiled);
        }

        let mut compiled_jacobian = Vec::new();
        compiled_jacobian
            .try_reserve_exact(dense_len)
            .map_err(|_| CompileError::AllocationFailure)?;
        let mut entries = jacobian.entries.iter().peekable();
        for offset in 0..dense_len {
            let mut ops = Vec::new();
            if let Some(entry) = entries.peek()
                && entry.row * num_vars + entry.col == offset
            {
                ops.try_reserve_exact(entry.compiled_deriv.ops().len())
                    .map_err(|_| CompileError::AllocationFailure)?;
                ops.extend_from_slice(entry.compiled_deriv.ops());
                entries.next();
            } else {
                ops.try_reserve_exact(1)
                    .map_err(|_| CompileError::AllocationFailure)?;
                ops.push(CompiledOp::LoadConst(0.0));
            }
            compiled_jacobian
                .push(CompiledExpr::try_from_ops(ops).map_err(CompileError::InvalidBytecode)?);
        }
        // Structure verification requires entries in canonical row-major order,
        // with exactly one entry at every declared nonzero coordinate.
        debug_assert!(entries.next().is_none());

        let mut vars = Vec::new();
        vars.try_reserve_exact(num_vars)
            .map_err(|_| CompileError::AllocationFailure)?;
        vars.extend_from_slice(&jacobian.vars);
        let mut residuals = Vec::new();
        residuals
            .try_reserve_exact(num_residuals)
            .map_err(|_| CompileError::AllocationFailure)?;
        residuals.extend_from_slice(&jacobian.residuals);

        Ok(Self {
            compiled: CompiledResidualSystem {
                vars,
                num_residuals,
                num_vars,
                compiled_residuals,
                compiled_jacobian,
            },
            residuals,
            receipt,
        })
    }

    /// Read-only access for existing numerical consumers. Cloning and editing
    /// this program produces an ordinary, unverified `CompiledResidualSystem`.
    pub fn compiled(&self) -> &CompiledResidualSystem {
        &self.compiled
    }

    /// Original symbolic residuals for exact or certified reference evaluation.
    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }

    pub fn vars(&self) -> &[Symbol] {
        &self.compiled.vars
    }

    /// Receipt for derivative proofs, sparsity and bytecode provenance only.
    pub fn receipt(&self) -> &SparseJacobianReceipt {
        &self.receipt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_sparse_program_is_the_program_executed_by_consumers() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let residuals = vec![
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
            Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
        ];
        let vars = vec![x, y];
        let sparse = SparseJacobian::try_build(&residuals, &vars).unwrap();
        let verified = VerifiedResidualSystem::try_from_sparse_jacobian(&sparse).unwrap();
        assert_eq!(verified.residuals(), residuals);
        assert_eq!(verified.vars(), vars);
        assert_eq!(verified.receipt().proofs_verified, sparse.entries.len());
        assert_eq!(verified.receipt().test_points_evaluated, 0);
        for entry in &sparse.entries {
            assert_eq!(
                verified.compiled().compiled_jacobian[entry.row * vars.len() + entry.col],
                entry.compiled_deriv,
            );
        }
        let mut values = [0.0; 2];
        let mut jacobian = [0.0; 4];
        verified
            .compiled()
            .try_eval_system(&[3.0, 5.0], &mut values, &mut jacobian)
            .unwrap();
        assert_eq!(values, [9.0, 8.0]);
        assert_eq!(jacobian, [6.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn derivative_bytecode_cannot_be_swapped_after_its_proof_was_generated() {
        let x = Symbol::new("x");
        let mut sparse =
            SparseJacobian::try_build(&[Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2))], &[x])
                .unwrap();
        sparse.entries[0].compiled_deriv =
            CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
        assert!(matches!(
            VerifiedResidualSystem::try_from_sparse_jacobian(&sparse),
            Err(VerifiedCompilationError::Jacobian(
                JacobianVerificationError::CompiledProvenanceMismatch { .. }
            )),
        ));
    }

    #[test]
    fn empty_verified_system_is_well_formed() {
        let system = VerifiedResidualSystem::try_compile(&[], &[]).unwrap();
        system
            .compiled()
            .try_eval_system(&[], &mut [], &mut [])
            .unwrap();
        assert_eq!(system.receipt().proofs_verified, 0);
    }
}
