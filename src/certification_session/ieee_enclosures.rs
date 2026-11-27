//! Wire export of directed IEEE endpoints. The certified statement remains
//! scoped to the exact original input box, even when an axis rounds outward.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError,
    CertificationState, ExactInterval, ResidualSystemId, admit_response, parse_domain,
    render_domain,
};
use crate::request_budget::RequestBudget;
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, VerifierLease};
use fsym_core::FloatingResidualJacobianEnclosure;
use serde::{Deserialize, Serialize};
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IeeeFormat {
    Binary32,
    Binary64,
}

/// Lower and upper endpoints as fixed-width hexadecimal IEEE bit patterns.
#[derive(Debug, Serialize)]
pub struct IeeeEnclosureBounds {
    pub domain_bits: Vec<[String; 2]>,
    pub residual_bits: Vec<[String; 2]>,
    pub jacobian_bits: Vec<Vec<[String; 2]>>,
}

impl IeeeEnclosureBounds {
    fn from_bounds<T: Copy>(
        bounds: &FloatingResidualJacobianEnclosure<'_, T>,
        width: usize,
        bits: impl Fn(T) -> u64,
        budget: &mut RequestBudget<'_>,
    ) -> Result<Self, CertificationSessionError> {
        let rows = bounds.jacobian().len();
        let axes = bounds.domain().len();
        let pairs = axes + rows + rows * axes;
        // Each interval owns two pre-sized strings. There is one vector for
        // each Jacobian row, plus the domain, residual and outer row vectors.
        budget.charge_batch(&[
            (
                Dimension::ComputeSteps,
                (pairs * (2 * width + 1) + rows) as u64,
            ),
            (
                Dimension::MemoryBytes,
                (pairs * (std::mem::size_of::<[String; 2]>() + 2 * width)
                    + rows * std::mem::size_of::<Vec<[String; 2]>>()) as u64,
            ),
            (Dimension::AllocationCount, (2 * pairs + rows + 3) as u64),
        ])?;
        let encode = |value| {
            let mut text = String::with_capacity(width);
            write!(&mut text, "{:0width$x}", bits(value)).expect("writing to String succeeds");
            text
        };
        let render = |values: &[(T, T)]| -> Vec<[String; 2]> {
            values
                .iter()
                .map(|&(lo, hi)| [encode(lo), encode(hi)])
                .collect()
        };
        let domain_bits = render(bounds.domain());
        let residual_bits = render(bounds.residual_values());
        let mut jacobian_bits = Vec::with_capacity(rows);
        for row in bounds.jacobian() {
            budget.checkpoint()?;
            jacobian_bits.push(render(row));
        }
        budget.checkpoint()?;
        Ok(Self {
            domain_bits,
            residual_bits,
            jacobian_bits,
        })
    }
}

impl CertificationState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn enclose_residuals(
        &self,
        system: ResidualSystemId,
        domain: &[ExactInterval],
        precision_digits: u32,
        format: Option<IeeeFormat>,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        let problem = self
            .systems
            .get(&system)
            .ok_or(CertificationSessionError::UnknownSystem)?;
        let domain = parse_domain(domain, problem.variables().len(), budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        let receipt = problem.enclose(
            &domain,
            precision_digits,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;
        let domain = render_domain(receipt.domain(), budget)?;
        let evidence = "certified_numeric";
        let claim = "whole_box_c1_residual_jacobian_enclosure";
        let response = if let Some(format) = format {
            let bounds = match format {
                IeeeFormat::Binary32 => IeeeEnclosureBounds::from_bounds(
                    &receipt.metered_to_f32_bounds(budget)?,
                    8,
                    |f| u64::from(f.to_bits()),
                    budget,
                )?,
                IeeeFormat::Binary64 => IeeeEnclosureBounds::from_bounds(
                    &receipt.metered_to_f64_bounds(budget)?,
                    16,
                    f64::to_bits,
                    budget,
                )?,
            };
            CertificationResponse::ResidualJacobianIeeeEnclosed {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system,
                evidence,
                claim,
                domain,
                format,
                bounds,
                precision_digits: receipt.precision_digits(),
                context_digest,
            }
        } else {
            CertificationResponse::ResidualJacobianEnclosed {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system,
                evidence,
                claim,
                domain,
                residuals: render_domain(receipt.residual_values(), budget)?,
                jacobian: receipt
                    .jacobian()
                    .iter()
                    .map(|row| render_domain(row, budget))
                    .collect::<Result<_, _>>()?,
                precision_digits: receipt.precision_digits(),
                context_digest,
            }
        };
        admit_response(&response, budget)?;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, Unbounded};
    use fsym_core::{Expr, RealBall, Symbol, enclose_residual_jacobian};

    #[test]
    fn wire_hex_enforces_the_budget_for_every_endpoint_string() {
        let source = enclose_residual_jacobian(
            &[Expr::symbol("x")],
            &[Symbol::new("x")],
            &[RealBall::from_i64(0)],
            16,
        )
        .unwrap();
        let bounds = source.metered_to_f32_bounds(&mut Unbounded).unwrap();
        for allowed in [9, 10] {
            let mut limits = BudgetLimits::uniform(1 << 32, 91);
            limits.dimensions[Dimension::AllocationCount.index()] = allowed;
            let mut ledger = Budget::new(limits);
            let mut check = || Ok(());
            let result = IeeeEnclosureBounds::from_bounds(
                &bounds,
                8,
                |f| u64::from(f.to_bits()),
                &mut RequestBudget::new(&mut ledger, &mut check),
            );
            if allowed == 9 {
                assert_eq!(result.unwrap_err().code(), "budget_exhausted");
            } else {
                assert_eq!(
                    result.unwrap().jacobian_bits[0][0],
                    ["3f800000", "3f800000"]
                );
                assert_eq!(ledger.remaining(Dimension::AllocationCount), 0);
            }
            assert_eq!(ledger.verifier_remaining(), 91);
        }
    }
}
