//! Wire export of directed IEEE endpoints. The certified statement remains
//! scoped to the exact original input box, even when an axis rounds outward.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError,
    CertificationState, ExactInterval, ResidualSystemId, admit_response, charge, parse_domain,
    render_domain,
};
use crate::request_budget::RequestBudget;
use fsym_budget::{BudgetError, MeterError, VerifierLease};
use fsym_core::FloatingResidualJacobianEnclosure;
use serde::{Deserialize, Serialize};

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
        let mut render = |pairs: &[(T, T)]| -> Result<Vec<[String; 2]>, CertificationSessionError> {
            charge(
                budget,
                pairs.len() * (2 * width + 1),
                pairs.len() * (std::mem::size_of::<[String; 2]>() + 2 * width),
            )?;
            Ok(pairs
                .iter()
                .map(|&(lo, hi)| {
                    [
                        format!("{:0width$x}", bits(lo)),
                        format!("{:0width$x}", bits(hi)),
                    ]
                })
                .collect())
        };
        let domain_bits = render(bounds.domain())?;
        let residual_bits = render(bounds.residual_values())?;
        let jacobian_bits = bounds
            .jacobian()
            .iter()
            .map(|row| render(row))
            .collect::<Result<_, _>>()?;
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
            charge(
                budget,
                receipt.jacobian().len(),
                std::mem::size_of::<Vec<[String; 2]>>() * receipt.jacobian().len(),
            )?;
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
