//! Directed IEEE endpoints for an already checked original-source enclosure.

use super::ResidualJacobianEnclosure;
use crate::RealBall;
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_rational::RationalArithmeticError;
use std::mem::size_of;
use thiserror::Error;

/// Outward endpoint pairs in the exact source's variable and residual order.
/// The domain pairs enclose the supplied exact input coordinates. The residual
/// and Jacobian guarantees apply on `source().domain()`, which can be narrower
/// than the binary representation of that box.
#[derive(Debug)]
pub struct FloatingResidualJacobianEnclosure<'a, T> {
    source: &'a ResidualJacobianEnclosure,
    domain: Vec<(T, T)>,
    residual_values: Vec<(T, T)>,
    jacobian: Vec<Vec<(T, T)>>,
}

impl<'a, T> FloatingResidualJacobianEnclosure<'a, T> {
    pub fn source(&self) -> &'a ResidualJacobianEnclosure {
        self.source
    }

    pub fn domain(&self) -> &[(T, T)] {
        &self.domain
    }

    pub fn residual_values(&self) -> &[(T, T)] {
        &self.residual_values
    }

    /// Row `i`, column `j` encloses the derivative of source residual `i`
    /// with respect to source variable `j` over the exact checked domain.
    pub fn jacobian(&self) -> &[Vec<(T, T)>] {
        &self.jacobian
    }
}

#[derive(Debug, Error)]
pub enum FloatingEnclosureError {
    #[error("enclosure export stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("exact endpoint arithmetic refused: {0}")]
    Arithmetic(#[source] RationalArithmeticError),
    #[error("could not allocate enclosure export buffers")]
    Allocation,
    #[error("a rational endpoint could not be converted")]
    Representation,
}

impl From<RationalArithmeticError> for FloatingEnclosureError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(cause) => Self::Meter(cause),
            RationalArithmeticError::AllocationFailure => Self::Allocation,
            cause => Self::Arithmetic(cause),
        }
    }
}

fn buffer<T>(length: usize) -> Result<Vec<T>, FloatingEnclosureError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| FloatingEnclosureError::Allocation)?;
    Ok(result)
}

impl ResidualJacobianEnclosure {
    /// Convert every exact ball directly to binary32 lower and upper bounds.
    /// Infinities and signed zero follow the directed endpoint rules of
    /// [`RealBall::metered_to_f32_bounds`]. All allocations, conversions and
    /// the final publication checkpoint share one cumulative meter.
    pub fn metered_to_f32_bounds<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<FloatingResidualJacobianEnclosure<'_, f32>, FloatingEnclosureError> {
        self.float_bounds(meter, RealBall::metered_to_f32_bounds)
    }

    /// Convert the complete checked enclosure directly to binary64 endpoints.
    /// The source binding and output layout match [`Self::metered_to_f32_bounds`].
    pub fn metered_to_f64_bounds<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<FloatingResidualJacobianEnclosure<'_, f64>, FloatingEnclosureError> {
        self.float_bounds(meter, RealBall::metered_to_f64_bounds)
    }

    fn float_bounds<M: BudgetMeter, T>(
        &self,
        meter: &mut M,
        convert: fn(&RealBall, &mut M) -> Result<Option<(T, T)>, RationalArithmeticError>,
    ) -> Result<FloatingResidualJacobianEnclosure<'_, T>, FloatingEnclosureError> {
        meter.checkpoint()?;
        // Only the checker constructs this receipt: at most 16 coordinates,
        // 64 residuals and 1024 Jacobian cells, all nonempty and rectangular.
        let n = self.variables().len();
        let m = self.residuals().len();
        let count = n + m + n * m;
        let bytes = count * size_of::<(T, T)>() + m * size_of::<Vec<(T, T)>>();
        meter.charge_batch(&[
            (Dimension::ComputeSteps, (count + m + 1) as u64),
            (Dimension::MemoryBytes, bytes as u64),
            (Dimension::AllocationCount, (m + 3) as u64),
        ])?;
        let mut axis = |balls: &[RealBall]| -> Result<Vec<(T, T)>, FloatingEnclosureError> {
            let mut values = buffer(balls.len())?;
            for ball in balls {
                values.push(convert(ball, meter)?.ok_or(FloatingEnclosureError::Representation)?);
            }
            Ok(values)
        };
        let domain = axis(self.domain())?;
        let residual_values = axis(self.residual_values())?;
        let mut jacobian = buffer(m)?;
        for row in self.jacobian() {
            jacobian.push(axis(row)?);
        }
        meter.checkpoint()?;
        Ok(FloatingResidualJacobianEnclosure {
            source: self,
            domain,
            residual_values,
            jacobian,
        })
    }
}
