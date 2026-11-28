//! Exact operator bounds for the derivative of the fixed-point map x - C F(x).
use super::{Jet, RootBoxError, Work, check_rational};
use crate::{BigRational, RealBall};
use fsym_budget::BudgetMeter;
use num_traits::{One, Signed, Zero};

/// Vector norm used to certify the fixed-point contraction on the unknowns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionNorm {
    /// Maximum absolute coordinate.
    Infinity,
    /// Maximum absolute coordinate divided by its strictly positive box radius.
    BoxRadiusScaledInfinity,
}

impl ContractionNorm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Infinity => "infinity",
            Self::BoxRadiusScaledInfinity => "box_radius_scaled_infinity",
        }
    }

    /// For W = diag(box radii), the scaled matrix norm is the largest row sum
    /// of |W^-1 D W|. All entries bound their original-source intervals.
    pub(super) fn bound(
        self,
        defect: &[Vec<RealBall>],
        domain: &[RealBall],
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<BigRational, RootBoxError> {
        let mut maximum = BigRational::zero();
        for (row, output) in defect.iter().zip(domain) {
            let mut sum = BigRational::zero();
            for (entry, input) in row.iter().zip(domain) {
                work.arithmetic(&[entry], &[&sum])?;
                let mut magnitude = entry.midpoint().abs() + entry.radius();
                if self == Self::BoxRadiusScaledInfinity {
                    work.arithmetic(&[], &[&magnitude, input.radius()])?;
                    magnitude *= input.radius();
                    work.arithmetic(&[], &[&sum, &magnitude])?;
                }
                sum += magnitude;
                check_rational(&sum)?;
            }
            if self == Self::BoxRadiusScaledInfinity {
                work.arithmetic(&[], &[&sum, output.radius()])?;
                sum /= output.radius();
            }
            work.arithmetic(&[], &[&maximum, &sum])?;
            maximum = maximum.max(sum);
        }
        Ok(maximum)
    }
}

pub(super) fn certify_contraction(
    defect: &[Vec<RealBall>],
    domain: &[RealBall],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(ContractionNorm, BigRational), RootBoxError> {
    for metric in [
        ContractionNorm::Infinity,
        ContractionNorm::BoxRadiusScaledInfinity,
    ] {
        let bound = metric.bound(defect, domain, work)?;
        if bound < BigRational::one() {
            return Ok((metric, bound));
        }
    }
    Err(RootBoxError::NotContractive)
}

/// Form I - C J_x from the first n (unknown) partials of the source jets.
/// Parameter partials never contribute to contraction in the unknowns.
pub(super) fn jacobian_defect(
    jets: &[Jet],
    preconditioner: &[Vec<BigRational>],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<Vec<RealBall>>, RootBoxError> {
    let n = preconditioner.len();
    work.charge(
        1,
        (n * n * std::mem::size_of::<RealBall>()) as u64,
        (n + 1) as u64,
    )?;
    let mut defect = vec![vec![RealBall::from_i64(0); n]; n];
    for (i, row) in defect.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            let mut product = RealBall::from_i64(0);
            for (coefficient, jet) in preconditioner[i].iter().zip(jets) {
                let term = work.scale(&jet.gradient[j], coefficient)?;
                product = work.add(&product, &term)?;
            }
            *entry = work.sub(&RealBall::from_i64(i64::from(i == j)), &product)?;
        }
    }
    Ok(defect)
}
