//! A consumer projection of one immutable, independently admitted program.
//!
//! Fixed parameters remain part of every original evaluation. Projection only
//! removes their Jacobian columns after all residuals and every derivative
//! domain have been checked; it never constructs another verified artifact.

use super::{SystemConsumerError, buffer, charge, finite, point_dimension};
use fsym_budget::BudgetMeter;
use fsym_calculus::{VerifiedResidualSystem, verified::SparseResidualEvaluation};
use std::borrow::Cow;

pub(super) struct SparseConsumerSource<'a> {
    system: &'a VerifiedResidualSystem,
    unknowns: usize,
    parameters: &'a [f64],
    pattern: Cow<'a, [(usize, usize)]>,
    positions: Vec<usize>,
}

impl<'a> SparseConsumerSource<'a> {
    pub(super) fn full(system: &'a VerifiedResidualSystem) -> Self {
        Self {
            system,
            unknowns: system.num_vars(),
            parameters: &[],
            pattern: Cow::Borrowed(system.jacobian_pattern()),
            positions: Vec::new(),
        }
    }

    pub(super) fn parametric<M: BudgetMeter>(
        system: &'a VerifiedResidualSystem,
        initial: &[f64],
        parameters: &'a [f64],
        meter: &mut M,
    ) -> Result<Self, SystemConsumerError> {
        meter.checkpoint()?;
        let total = system.num_vars();
        let unknowns = initial.len();
        super::validate_sparse_root_shape(total, system.num_residuals())?;
        if unknowns == 0 || unknowns >= total {
            return Err(SystemConsumerError::InvalidRequest(
                "sparse parametric consumers require nonempty unknown and parameter groups",
            ));
        }
        point_dimension(parameters, total - unknowns, "fixed sparse parameters")?;
        finite(initial, "sparse parametric initial point", meter)?;
        finite(parameters, "fixed sparse parameters", meter)?;

        // Inspect the complete pattern before reserving projected storage.
        let original = system.jacobian_pattern();
        charge(meter, original.len() as u64, 0, 0)?;
        let count = original
            .iter()
            .filter(|&&(_, column)| column < unknowns)
            .count();
        let mut pattern = buffer(count, meter)?;
        let mut positions = buffer(count, meter)?;
        charge(meter, original.len() as u64, 0, 0)?;
        for (position, &(row, column)) in original.iter().enumerate() {
            if column < unknowns {
                pattern.push((row, column));
                positions.push(position);
            }
        }
        meter.checkpoint()?;
        Ok(Self {
            system,
            unknowns,
            parameters,
            pattern: Cow::Owned(pattern),
            positions,
        })
    }

    pub(super) fn num_vars(&self) -> usize {
        self.unknowns
    }

    pub(super) fn num_residuals(&self) -> usize {
        self.system.num_residuals()
    }

    pub(super) fn jacobian_pattern(&self) -> &[(usize, usize)] {
        &self.pattern
    }

    pub(super) fn evaluate_with_meter<M: BudgetMeter>(
        &self,
        point: &[f64],
        meter: &mut M,
    ) -> Result<SparseResidualEvaluation, SystemConsumerError> {
        point_dimension(point, self.unknowns, "sparse projected point")?;
        let mut values = if self.parameters.is_empty() {
            self.system.evaluate_sparse_with_meter(point, meter)?
        } else {
            let total = self.system.num_vars();
            let mut full_point = buffer(total, meter)?;
            charge(meter, total as u64, 0, 0)?;
            full_point.extend_from_slice(point);
            full_point.extend_from_slice(self.parameters);
            self.system.evaluate_sparse_with_meter(&full_point, meter)?
        };
        if values.residuals.len() != self.num_residuals()
            || values.jacobian.len() != self.system.jacobian_pattern().len()
        {
            return Err(SystemConsumerError::ConsumerContract(
                "original sparse source evaluation dimensions",
            ));
        }
        // Do not let discarded parameter derivatives hide a nonfinite value
        // or an original domain obligation, including at an already stationary
        // point or an exact zero residual. The real program ran in full above.
        finite(&values.residuals, "original sparse residuals", meter)?;
        finite(&values.jacobian, "original sparse Jacobian", meter)?;
        if !self.parameters.is_empty() {
            charge(meter, self.positions.len() as u64, 0, 0)?;
            for (output, &input) in self.positions.iter().enumerate() {
                // Ascending retained positions permit in-place compaction.
                values.jacobian[output] = values.jacobian[input];
            }
            values.jacobian.truncate(self.positions.len());
        }
        meter.checkpoint()?;
        Ok(values)
    }
}
