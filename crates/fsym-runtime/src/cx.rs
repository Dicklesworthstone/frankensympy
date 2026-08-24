//! Domain-typed wrapper over an asupersync execution context.
//!
//! Follows the wrapping pattern prescribed by asupersync's `Cx` docs:
//! hold a `&Cx`, delegate capability checks to it, and attach the
//! domain-specific state — here, the region's canonical
//! [`fsym_budget::Budget`] with its protected verifier pool.
//!
//! Construction never invents ambient authority: callers pass whatever
//! `&Cx` the runtime handed them (in tests, the public fail-closed
//! [`Cx::detached_cancel_context`]).

use asupersync::Cx;
use fsym_budget::{BudgetError, BudgetLimits, BudgetMeter, MeterError, VerifierLease};

use crate::{Budget, ChargeReceipt, Dimension};

/// A FrankenSymPy region context: cancellation source plus the region's
/// multidimensional budget. Generic over the asupersync capability set,
/// so restricted contexts (`Cx<NoCaps>`) wrap exactly like privileged
/// ones.
pub struct FsymCx<'a, Caps = asupersync::cx::cap::All> {
    cx: &'a Cx<Caps>,
    budget: Budget,
    limits: BudgetLimits,
}

impl<'a, Caps> FsymCx<'a, Caps> {
    /// Wraps a runtime-supplied context around a domain budget.
    pub fn new(cx: &'a Cx<Caps>, budget: Budget, limits: BudgetLimits) -> Self {
        Self { cx, budget, limits }
    }

    /// The wrapped asupersync context, for effects this wrapper does not
    /// model (spawning, timers, io). Explicit escape hatch: capabilities
    /// stay visible at the call site.
    pub fn asupersync(&self) -> &'a Cx<Caps> {
        self.cx
    }
    /// True once the owning region has been cancelled.
    pub fn check_cancelled(&self) -> bool {
        self.cx.is_cancel_requested()
    }

    /// Fail-closed checkpoint: `Err` when the region was cancelled.
    pub fn checkpoint(&self) -> Result<(), Cancelled> {
        if self.cx.is_cancel_requested() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    /// Charges the shared (generator-accessible) dimensions.
    pub fn charge(
        &mut self,
        dimension: Dimension,
        amount: u64,
    ) -> Result<ChargeReceipt, BudgetError> {
        self.budget.try_charge(dimension, amount)
    }

    /// Issues the region's single verifier lease. The second call returns
    /// `None`: one protected pool, one key.
    pub fn verifier_lease(&mut self) -> Option<VerifierLease> {
        self.budget.verifier_lease()
    }

    /// Charges the protected verifier pool; requires the lease.
    pub fn charge_verifier(
        &mut self,
        lease: &VerifierLease,
        amount: u64,
    ) -> Result<ChargeReceipt, BudgetError> {
        self.budget.try_charge_verifier(lease, amount)
    }

    /// Reserve a child execution budget from this region.
    ///
    /// The reservation leaves the parent immediately. Callers must return the child
    /// through [`Self::merge_child`] on every terminal path so unused allowances are
    /// reconciled without resetting work already consumed by the child.
    pub fn reserve_child(&mut self, limits: BudgetLimits) -> Result<Self, BudgetError> {
        let child_budget = self.budget.reserve_child(limits)?;
        Ok(Self::new(self.cx, child_budget, limits))
    }

    /// Reconcile a completed child region into its owning parent.
    pub fn merge_child(&mut self, child: Self) -> Result<(), BudgetError> {
        self.budget.merge_child(child.budget)
    }

    /// Remaining shared allowance along a dimension.
    pub fn remaining(&self, dimension: Dimension) -> u64 {
        self.budget.remaining(dimension)
    }

    /// Remaining protected verifier allowance.
    pub fn verifier_remaining(&self) -> u64 {
        self.budget.verifier_remaining()
    }

    /// Limits this region was constructed with.
    pub fn limits(&self) -> BudgetLimits {
        self.limits
    }
}

impl<Caps> BudgetMeter for FsymCx<'_, Caps> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.budget
            .try_charge(dimension, amount)
            .map(|_| ())
            .map_err(MeterError::Budget)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        // The asupersync cancel source is the authority; one atomic load.
        if self.cx.is_cancel_requested() {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Region was cancelled before the effect could run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::CancelKind;
    use fsym_budget::BudgetLimits;

    #[test]
    fn cancellation_delegates_to_the_wrapped_context() {
        let cx = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(100, 10);
        let region = FsymCx::new(&cx, Budget::new(limits), limits);

        assert!(!region.check_cancelled());
        assert!(region.checkpoint().is_ok());

        cx.cancel_with(CancelKind::User, Some("test shutdown"));

        assert!(region.check_cancelled());
        assert_eq!(region.checkpoint(), Err(Cancelled));
    }

    #[test]
    fn charges_flow_through_the_domain_budget() {
        let cx = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(50, 10);
        let mut region = FsymCx::new(&cx, Budget::new(limits), limits);

        let _receipt = region.charge(Dimension::ComputeSteps, 30).unwrap();
        assert_eq!(region.remaining(Dimension::ComputeSteps), 20);

        let err = region
            .charge(Dimension::ComputeSteps, 21)
            .expect_err("must exceed");
        assert!(matches!(err, BudgetError::Exhausted { .. }));
        // Failed charge left prior consumption intact; the receipt stays
        // an opaque token (fields are crate-private by design).
        assert_eq!(region.remaining(Dimension::ComputeSteps), 20);
    }

    #[test]
    fn verifier_pool_is_single_lease_and_generator_independent() {
        let cx = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(200, 40);
        let mut region = FsymCx::new(&cx, Budget::new(limits), limits);

        let lease = region.verifier_lease().expect("first lease");
        assert!(
            region.verifier_lease().is_none(),
            "second lease must be refused"
        );

        // Shared-dimension charges never touch the protected pool.
        region.charge(Dimension::ComputeSteps, 150).unwrap();
        assert_eq!(region.verifier_remaining(), 40);

        region.charge_verifier(&lease, 15).unwrap();
        assert_eq!(region.verifier_remaining(), 25);
    }

    #[test]
    fn child_consumption_is_reconciled_without_budget_reset() {
        let cx = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(20, 5);
        let mut parent = FsymCx::new(&cx, Budget::new(limits), limits);
        let child_limits = BudgetLimits::uniform(10, 0);
        let mut child = parent.reserve_child(child_limits).unwrap();

        child.charge(Dimension::ComputeSteps, 4).unwrap();
        parent.merge_child(child).unwrap();

        assert_eq!(parent.remaining(Dimension::ComputeSteps), 16);
        assert_eq!(parent.remaining(Dimension::MemoryBytes), 20);
        assert_eq!(parent.verifier_remaining(), 5);
    }

    #[test]
    fn budget_meter_trait_charges_and_reports_cancellation() {
        let cx = Cx::detached_cancel_context();
        let limits = BudgetLimits::uniform(3, 0);
        let mut region = FsymCx::new(&cx, Budget::new(limits), limits);

        // Generic meter use: exhaustion surfaces as MeterError::Budget.
        fn drain<M: BudgetMeter>(m: &mut M) -> Result<(), MeterError> {
            for _ in 0..10 {
                m.checkpoint()?;
                m.charge(Dimension::ComputeSteps, 1)?;
            }
            Ok(())
        }
        assert_eq!(
            drain(&mut region),
            Err(MeterError::Budget(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                requested: 1,
                remaining: 0,
            }))
        );

        cx.cancel_with(CancelKind::User, Some("meter test"));
        assert_eq!(
            BudgetMeter::checkpoint(&mut region),
            Err(MeterError::Cancelled),
            "trait checkpoint must map the asupersync cancel source"
        );
    }
}
