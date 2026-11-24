//! Original-source real Lambert-W rules for interval automatic differentiation.
//!
//! The inverse kernel proves the complete argument domain and branch. These
//! rules then require strict separation from w=-1 before any outer chain
//! multiplier is applied. They use exp(-w)/(1+w), which is analytic at W0(0),
//! rather than introducing the artificial zero denominator w/(u*(1+w)).

use super::{MAX_ROOT_BOX_ALLOCATION_BYTES, MAX_ROOT_BOX_STEPS, RootBoxError, Work, checked};
use crate::{BallError, Expr, LambertWError, RealBall};
use fsym_budget::{BudgetError, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError};
use num_traits::ToPrimitive;

/// A branch is immutable syntax, never an evaluated expression or a rounded
/// number. The enclosing source preflight bounds and charges every node.
pub(super) fn argument(arguments: &[Expr]) -> Result<(&Expr, i32), RootBoxError> {
    match arguments {
        [argument] => Ok((argument, 0)),
        [argument, Expr::Integer(branch)] => match branch.to_i32() {
            Some(branch @ (0 | -1)) => Ok((argument, branch)),
            _ => Err(RootBoxError::Unsupported("Lambert-W real branch")),
        },
        _ => Err(RootBoxError::Unsupported(
            "Lambert-W requires an argument and an optional literal real branch",
        )),
    }
}

impl<M: BudgetMeter> Work<'_, M> {
    pub(super) fn lambert_w(
        &mut self,
        argument: &RealBall,
        branch: i32,
    ) -> Result<(RealBall, RealBall), RootBoxError> {
        self.arithmetic(&[argument], &[])?;
        let precision = self.precision;
        let mut meter = LambertMeter {
            work: self,
            intrinsic_failure: None,
        };
        let result = argument.metered_lambert_w(branch, precision, &mut meter);
        if let Some(error) = meter.intrinsic_failure {
            return Err(error);
        }
        let value = checked(result.map_err(|error| match error {
            LambertWError::Ball(error) => RootBoxError::Ball(error),
            LambertWError::Meter(error) => RootBoxError::Meter(error),
            LambertWError::Arithmetic(error) => RootBoxError::Ball(
                BallError::ArgumentOutsideDeclaredEnvelope(error.to_string()),
            ),
        })?)?;
        let denominator = self.add(&RealBall::from_i64(1), &value)?;
        self.arithmetic(&[&denominator], &[])?;
        if (branch == 0 && !denominator.is_positive())
            || (branch == -1 && !denominator.is_negative())
        {
            return Err(RootBoxError::Domain(
                "Lambert-W branch is not strictly separated from its derivative singularity",
            ));
        }
        self.arithmetic(&[&value], &[])?;
        let exponential = self.exp(&value.neg())?;
        let slope = self.div(&exponential, &denominator)?;
        self.step()?;
        Ok((value, slope))
    }

    pub(super) fn lambert_w_curvature(
        &mut self,
        value: &RealBall,
        slope: &RealBall,
    ) -> Result<RealBall, RootBoxError> {
        // W'' = -(W+2)/(W+1) * (W')^2. The caller has already proved
        // the branch and denominator sign on the whole original interval.
        let numerator = self.add(value, &RealBall::from_i64(2))?;
        let denominator = self.add(value, &RealBall::from_i64(1))?;
        let quotient = self.div(&numerator, &denominator)?;
        let square = self.power(slope, 2)?;
        let product = self.mul(&quotient, &square)?;
        self.arithmetic(&[&product], &[])?;
        checked(product.neg())
    }
}

/// Forward every inverse-kernel charge atomically to the existing request,
/// also retaining the checker's cumulative intrinsic ceilings across nodes.
/// Creating this adapter never creates or replenishes a budget.
struct LambertMeter<'a, 'b, M> {
    work: &'a mut Work<'b, M>,
    intrinsic_failure: Option<RootBoxError>,
}

impl<M: BudgetMeter> BudgetMeter for LambertMeter<'_, '_, M> {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.work.meter.checkpoint()
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()?;
        let mut totals = [0u64; DIMENSION_COUNT];
        for &(dimension, amount) in charges {
            totals[dimension.index()] = totals[dimension.index()]
                .checked_add(amount)
                .ok_or(BudgetError::ChargeOverflow { dimension })?;
        }
        for (dimension, used, limit, reason) in [
            (
                Dimension::ComputeSteps,
                self.work.steps,
                MAX_ROOT_BOX_STEPS,
                "cumulative arithmetic work",
            ),
            (
                Dimension::MemoryBytes,
                self.work.allocation_bytes,
                MAX_ROOT_BOX_ALLOCATION_BYTES,
                "cumulative allocation",
            ),
        ] {
            let requested = totals[dimension.index()];
            let remaining = limit.saturating_sub(used);
            if requested > remaining {
                self.intrinsic_failure = Some(RootBoxError::ResourceLimit(reason));
                return Err(BudgetError::Exhausted {
                    dimension,
                    requested,
                    remaining,
                }
                .into());
            }
        }
        self.work.meter.charge_batch(charges)?;
        self.work.steps += totals[Dimension::ComputeSteps.index()];
        self.work.allocation_bytes += totals[Dimension::MemoryBytes.index()];
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits};

    #[test]
    fn nested_charges_preserve_intrinsic_limits_and_atomic_refusal() {
        let mut budget = Budget::new(BudgetLimits::uniform(u64::MAX, 17));
        let mut work = Work::new(&mut budget, 12).unwrap();
        work.steps = MAX_ROOT_BOX_STEPS - 5;
        let original_bytes = work.allocation_bytes;
        let before = work.meter.snapshot();
        let mut meter = LambertMeter {
            work: &mut work,
            intrinsic_failure: None,
        };
        assert!(
            meter
                .charge_batch(&[
                    (Dimension::MemoryBytes, 7),
                    (Dimension::ComputeSteps, 3),
                    (Dimension::ComputeSteps, 3),
                ])
                .is_err()
        );
        assert_eq!(
            meter.intrinsic_failure,
            Some(RootBoxError::ResourceLimit("cumulative arithmetic work"))
        );
        assert_eq!(meter.work.meter.snapshot(), before);
        assert_eq!(meter.work.steps, MAX_ROOT_BOX_STEPS - 5);
        assert_eq!(meter.work.allocation_bytes, original_bytes);
    }

    #[test]
    fn nested_caller_refusal_does_not_consume_the_intrinsic_ledger() {
        let mut limits = BudgetLimits::uniform(u64::MAX, 19);
        limits.dimensions[Dimension::AllocationCount.index()] = 1;
        let mut budget = Budget::new(limits);
        let mut work = Work::new(&mut budget, 12).unwrap();
        let before = work.meter.snapshot();
        let initial_steps = work.steps;
        let initial_bytes = work.allocation_bytes;
        let mut meter = LambertMeter {
            work: &mut work,
            intrinsic_failure: None,
        };
        assert!(
            meter
                .charge_batch(&[
                    (Dimension::ComputeSteps, 7),
                    (Dimension::MemoryBytes, 7),
                    (Dimension::AllocationCount, 1),
                ])
                .is_err()
        );
        assert_eq!(meter.intrinsic_failure, None);
        assert_eq!(meter.work.meter.snapshot(), before);
        assert_eq!(meter.work.steps, initial_steps);
        assert_eq!(meter.work.allocation_bytes, initial_bytes);
        assert_eq!(meter.work.meter.verifier_remaining(), 19);
    }
}
