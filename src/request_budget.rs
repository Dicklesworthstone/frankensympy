//! Request-scoped safe points over an unchanged, caller-owned budget ledger.
//!
//! The host supplies the checkpoint policy (for example, an owning asupersync
//! context). Ordinary work and protected verification borrow that same policy.
//! A terminal checkpoint failure is latched for this request, so a later call
//! cannot revive cancelled work or publish its result. Dropping the adapter
//! neither replenishes resources nor changes the persistent ledger's authority.

#[cfg(test)]
use fsym_budget::BudgetSnapshot;
use fsym_budget::{
    Budget, BudgetError, BudgetMeter, Dimension, MeterError, VerifierLease, VerifierMeter,
};

struct Checkpoints<'a> {
    policy: &'a mut dyn FnMut() -> Result<(), MeterError>,
    stopped: Option<MeterError>,
}

impl Checkpoints<'_> {
    fn check(&mut self) -> Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = (self.policy)();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

pub(crate) struct RequestBudget<'a> {
    ledger: &'a mut Budget,
    checkpoints: Checkpoints<'a>,
}

impl<'a> RequestBudget<'a> {
    pub(crate) fn new(
        ledger: &'a mut Budget,
        checkpoint: &'a mut dyn FnMut() -> Result<(), MeterError>,
    ) -> Self {
        Self {
            ledger,
            checkpoints: Checkpoints {
                policy: checkpoint,
                stopped: None,
            },
        }
    }

    /// Reborrow the existing protected capability without resetting its pool.
    /// Foreign or missing authority is still rejected by the original ledger.
    pub(crate) fn verifier_meter(
        &mut self,
        lease: &VerifierLease,
    ) -> Result<RequestVerifierMeter<'_, 'a>, BudgetError> {
        Ok(RequestVerifierMeter {
            inner: self.ledger.verifier_meter(lease)?,
            checkpoints: &mut self.checkpoints,
        })
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> BudgetSnapshot {
        self.ledger.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn verifier_remaining(&self) -> u64 {
        self.ledger.verifier_remaining()
    }
}

impl BudgetMeter for RequestBudget<'_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoints.check()?;
        self.ledger.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints.check()
    }
}

pub(crate) struct RequestVerifierMeter<'a, 'policy> {
    inner: VerifierMeter<'a>,
    checkpoints: &'a mut Checkpoints<'policy>,
}

impl BudgetMeter for RequestVerifierMeter<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoints.check()?;
        self.inner.charge_batch(charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::BudgetLimits;

    #[test]
    fn ordinary_and_protected_work_share_one_policy_and_persistent_ledger() {
        let mut limits = BudgetLimits::uniform(100, 20);
        limits.dimensions[Dimension::ComputeSteps.index()] = 3;
        limits.dimensions[Dimension::MemoryBytes.index()] = 5;
        let mut ledger = Budget::new(limits);
        let lease = ledger.verifier_lease().unwrap();
        let mut calls = 0;
        {
            let mut check = || {
                calls += 1;
                Ok(())
            };
            let mut request = RequestBudget::new(&mut ledger, &mut check);
            request.charge(Dimension::ComputeSteps, 3).unwrap();
            {
                let mut verifier = request.verifier_meter(&lease).unwrap();
                verifier
                    .charge_batch(&[(Dimension::ComputeSteps, 4), (Dimension::MemoryBytes, 2)])
                    .unwrap();
                verifier.checkpoint().unwrap();
            }
            assert_eq!(request.verifier_remaining(), 16);
            assert!(request.charge(Dimension::ComputeSteps, 1).is_err());
        }
        assert_eq!(calls, 4);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.dimensions[Dimension::ComputeSteps.index()], 0);
        assert_eq!(snapshot.dimensions[Dimension::MemoryBytes.index()], 3);
        assert_eq!(snapshot.verifier_pool, 16);

        let mut ready = || Ok(());
        let mut next = RequestBudget::new(&mut ledger, &mut ready);
        assert!(next.charge(Dimension::ComputeSteps, 1).is_err());
        next.verifier_meter(&lease)
            .unwrap()
            .charge(Dimension::ComputeSteps, 2)
            .unwrap();
        assert_eq!(next.verifier_remaining(), 14);
    }

    #[test]
    fn a_single_cancellation_is_latched_across_both_lanes_without_new_charges() {
        for cancel_in_verifier in [false, true] {
            let mut ledger = Budget::new(BudgetLimits::uniform(100, 100));
            let lease = ledger.verifier_lease().unwrap();
            let before = ledger.snapshot();
            let mut calls = 0;
            let mut check = || {
                calls += 1;
                if calls == 1 {
                    Err(MeterError::Cancelled)
                } else {
                    Ok(())
                }
            };
            {
                let mut request = RequestBudget::new(&mut ledger, &mut check);
                let result = if cancel_in_verifier {
                    request.verifier_meter(&lease).unwrap().checkpoint()
                } else {
                    request.checkpoint()
                };
                assert_eq!(result, Err(MeterError::Cancelled));
                assert_eq!(
                    request.charge(Dimension::MemoryBytes, 1),
                    Err(MeterError::Cancelled)
                );
                assert_eq!(
                    request
                        .verifier_meter(&lease)
                        .unwrap()
                        .charge(Dimension::ComputeSteps, 1),
                    Err(MeterError::Cancelled)
                );
                assert_eq!(request.snapshot(), before);
            }
            assert_eq!(calls, 1);
            assert_eq!(ledger.snapshot(), before);
        }
    }

    #[test]
    fn foreign_authority_and_failed_atomic_verifier_batches_preserve_the_ledger() {
        let mut limits = BudgetLimits::uniform(10, 20);
        limits.dimensions[Dimension::MemoryBytes.index()] = 3;
        let mut ledger = Budget::new(limits);
        let lease = ledger.verifier_lease().unwrap();
        let mut foreign = Budget::new(limits);
        let foreign_lease = foreign.verifier_lease().unwrap();
        let before = ledger.snapshot();
        let mut ready = || Ok(());
        let mut request = RequestBudget::new(&mut ledger, &mut ready);
        assert!(matches!(
            request.verifier_meter(&foreign_lease),
            Err(BudgetError::VerifierPoolAccessDenied)
        ));
        assert!(
            request
                .verifier_meter(&lease)
                .unwrap()
                .charge_batch(&[(Dimension::ComputeSteps, 7), (Dimension::MemoryBytes, 4)])
                .is_err()
        );
        assert_eq!(request.snapshot(), before);
        request
            .verifier_meter(&lease)
            .unwrap()
            .charge_batch(&[(Dimension::ComputeSteps, 7), (Dimension::MemoryBytes, 3)])
            .unwrap();
        assert_eq!(request.verifier_remaining(), 13);
    }
}
