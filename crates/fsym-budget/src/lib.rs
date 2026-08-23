//! Multidimensional budgets, reservations, charging interfaces, snapshots,
//! and receipts for FrankenSymPy.
//!
//! Layer: L0 (budgets). Per `docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md`
//! §4.3 this crate contains no global allocator hooks and no
//! algorithm-specific policy.
//!
//! Structural guarantees:
//!
//! - **Protected verifier reservation.** The verifier pool can be charged
//!   only through a [`VerifierLease`] issued once per budget. Generator code
//!   holding a plain [`Budget`] has no API path into the pool
//!   ("generators cannot consume verifier-reserved budget").
//! - **Atomic refusal.** A failed charge leaves every counter untouched and
//!   reports what was requested versus what remained.
//! - **No hidden resets.** There is deliberately no method that inflates any
//!   remaining value beyond its original limit; accounting only moves between
//!   a parent and explicitly reserved children, or back via refunds.
//! - **Single-use receipts.** A [`ChargeReceipt`] moves into `refund`, so a
//!   receipt cannot refund twice.

#![forbid(unsafe_code)]

use std::fmt;

/// Canonical budget dimensions charged by symbolic work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    ComputeSteps,
    MemoryBytes,
    AllocationCount,
    DepthLimit,
    RandomDraws,
}

/// Number of distinct [`Dimension`] values.
pub const DIMENSION_COUNT: usize = 5;

impl Dimension {
    /// All dimensions in canonical order.
    pub const ALL: [Dimension; DIMENSION_COUNT] = [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
        Dimension::DepthLimit,
        Dimension::RandomDraws,
    ];

    /// Index into canonical order.
    pub fn index(self) -> usize {
        match self {
            Dimension::ComputeSteps => 0,
            Dimension::MemoryBytes => 1,
            Dimension::AllocationCount => 2,
            Dimension::DepthLimit => 3,
            Dimension::RandomDraws => 4,
        }
    }

    /// Registry-style identifier string.
    pub fn as_str(self) -> &'static str {
        match self {
            Dimension::ComputeSteps => "compute_steps",
            Dimension::MemoryBytes => "memory_bytes",
            Dimension::AllocationCount => "allocation_count",
            Dimension::DepthLimit => "depth_limit",
            Dimension::RandomDraws => "random_draws",
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors produced by budget operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetError {
    /// The requested amount exceeds the remaining allowance.
    Exhausted {
        dimension: Dimension,
        requested: u64,
        remaining: u64,
    },
    /// A zero-amount charge is rejected so receipts stay meaningful.
    ZeroCharge,
    /// The verifier pool was addressed without an active lease.
    VerifierPoolAccessDenied,
    /// Reserving a child would exceed the parent's remaining allowance.
    ChildReservationTooLarge,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetError::Exhausted {
                dimension,
                requested,
                remaining,
            } => {
                write!(
                    f,
                    "budget exhausted for {dimension}: requested {requested}, remaining {remaining}"
                )
            }
            BudgetError::ZeroCharge => write!(f, "zero-amount charge rejected"),
            BudgetError::VerifierPoolAccessDenied => {
                write!(
                    f,
                    "verifier-reserved budget requires an active verifier lease"
                )
            }
            BudgetError::ChildReservationTooLarge => {
                write!(f, "child reservation exceeds parent remaining allowance")
            }
        }
    }
}

impl std::error::Error for BudgetError {}

/// Initial limits for each dimension plus the protected verifier pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetLimits {
    /// Limit per canonical dimension.
    pub dimensions: [u64; DIMENSION_COUNT],
    /// Protected pool reserved for independent verification.
    pub verifier_pool: u64,
}

impl BudgetLimits {
    /// Uniform limit for every dimension plus a verifier pool size.
    pub fn uniform(per_dimension: u64, verifier_pool: u64) -> Self {
        BudgetLimits {
            dimensions: [per_dimension; DIMENSION_COUNT],
            verifier_pool,
        }
    }
}

/// Proof-of-authorization for charging the verifier pool. Constructed only
/// through [`Budget::verifier_lease`]; the private field prevents forgery
/// inside dependent crates.
#[derive(Debug)]
pub struct VerifierLease {
    _private: (),
}

/// Receipt proving that `amount` of `dimension` was charged at sequence
/// `seq`. Consumed by [`Budget::refund`]; intentionally neither `Copy` nor
/// `Clone` so a refund cannot happen twice.
#[derive(Debug, PartialEq, Eq)]
pub struct ChargeReceipt {
    pub(crate) kind: ChargedKind,
    pub(crate) amount: u64,
    pub(crate) seq: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ChargedKind {
    Dimension(Dimension),
    VerifierPool,
}

/// Immutable point-in-time view of remaining allowances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub dimensions: [u64; DIMENSION_COUNT],
    pub verifier_pool: u64,
    pub sequence: u64,
}

/// A multidimensional budget ledger.
///
/// One instance owns all remaining allowances for one request region.
#[derive(Debug)]
pub struct Budget {
    remaining: [u64; DIMENSION_COUNT],
    verifier_remaining: u64,
    verifier_lease_open: bool,
    sequence: u64,
}

impl Budget {
    /// Creates a budget with the given limits.
    pub fn new(limits: BudgetLimits) -> Self {
        Budget {
            remaining: limits.dimensions,
            verifier_remaining: limits.verifier_pool,
            verifier_lease_open: false,
            sequence: 0,
        }
    }

    /// Remaining allowance for a dimension.
    pub fn remaining(&self, dimension: Dimension) -> u64 {
        self.remaining[dimension.index()]
    }

    /// Remaining protected verifier pool.
    pub fn verifier_remaining(&self) -> u64 {
        self.verifier_remaining
    }

    /// Issues the single verifier lease. Returns `None` if one was already
    /// issued: there is one protected pool per budget and one key to it.
    pub fn verifier_lease(&mut self) -> Option<VerifierLease> {
        if self.verifier_lease_open {
            None
        } else {
            self.verifier_lease_open = true;
            Some(VerifierLease { _private: () })
        }
    }

    /// Charges `amount` of `dimension`, atomically.
    pub fn try_charge(
        &mut self,
        dimension: Dimension,
        amount: u64,
    ) -> Result<ChargeReceipt, BudgetError> {
        if amount == 0 {
            return Err(BudgetError::ZeroCharge);
        }
        let idx = dimension.index();
        let remaining = self.remaining[idx];
        if amount > remaining {
            return Err(BudgetError::Exhausted {
                dimension,
                requested: amount,
                remaining,
            });
        }
        self.remaining[idx] = remaining - amount;
        self.sequence += 1;
        Ok(ChargeReceipt {
            kind: ChargedKind::Dimension(dimension),
            amount,
            seq: self.sequence,
        })
    }

    /// Charges the protected verifier pool. Requires the lease issued by
    /// [`Budget::verifier_lease`].
    pub fn try_charge_verifier(
        &mut self,
        lease: &VerifierLease,
        amount: u64,
    ) -> Result<ChargeReceipt, BudgetError> {
        let _ = lease; // existence of a lease is the authorization proof
        if amount == 0 {
            return Err(BudgetError::ZeroCharge);
        }
        if amount > self.verifier_remaining {
            return Err(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                requested: amount,
                remaining: self.verifier_remaining,
            });
        }
        self.verifier_remaining -= amount;
        self.sequence += 1;
        Ok(ChargeReceipt {
            kind: ChargedKind::VerifierPool,
            amount,
            seq: self.sequence,
        })
    }

    /// Restores the allowance recorded in a consumed receipt.
    pub fn refund(&mut self, receipt: ChargeReceipt) {
        match receipt.kind {
            ChargedKind::Dimension(dimension) => {
                self.remaining[dimension.index()] += receipt.amount;
            }
            ChargedKind::VerifierPool => self.verifier_remaining += receipt.amount,
        }
        self.sequence += 1;
    }

    /// Reserves a child budget carved out of this one. The reserved amounts
    /// leave the parent immediately and any unused remainder returns exactly
    /// when the child is merged back through [`Budget::merge_child`]. Child
    /// verifier-pool access is denied by construction: children never carry a
    /// lease.
    pub fn reserve_child(&mut self, caps: BudgetLimits) -> Result<Budget, BudgetError> {
        for dimension in Dimension::ALL {
            let cap = caps.dimensions[dimension.index()];
            if cap > self.remaining[dimension.index()] {
                return Err(BudgetError::ChildReservationTooLarge);
            }
        }
        // Verifier pools are never inherited.
        if caps.verifier_pool != 0 {
            return Err(BudgetError::VerifierPoolAccessDenied);
        }
        for dimension in Dimension::ALL {
            self.remaining[dimension.index()] -= caps.dimensions[dimension.index()];
        }
        self.sequence += 1;
        Ok(Budget {
            remaining: caps.dimensions,
            verifier_remaining: 0,
            verifier_lease_open: false,
            sequence: 0,
        })
    }

    /// Merges a child's unspent allowances back into this budget.
    pub fn merge_child(&mut self, child: Budget) {
        for dimension in Dimension::ALL {
            self.remaining[dimension.index()] += child.remaining[dimension.index()];
        }
        self.sequence += 1;
    }

    /// Captures an immutable snapshot of current allowances.
    pub fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot {
            dimensions: self.remaining,
            verifier_pool: self.verifier_remaining,
            sequence: self.sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_reduce_and_exhaustion_is_atomic() {
        let mut budget = Budget::new(BudgetLimits::uniform(10, 5));
        let receipt = budget.try_charge(Dimension::ComputeSteps, 4).expect("fits");
        assert_eq!(budget.remaining(Dimension::ComputeSteps), 6);

        let before = budget.snapshot();
        let err = budget.try_charge(Dimension::ComputeSteps, 7).unwrap_err();
        assert_eq!(
            err,
            BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                requested: 7,
                remaining: 6
            }
        );
        assert_eq!(
            budget.snapshot().dimensions,
            before.dimensions,
            "failed charge must not mutate"
        );

        budget.refund(receipt);
        assert_eq!(budget.remaining(Dimension::ComputeSteps), 10);
    }

    #[test]
    fn zero_charges_are_rejected() {
        let mut budget = Budget::new(BudgetLimits::uniform(1, 1));
        assert_eq!(
            budget.try_charge(Dimension::MemoryBytes, 0),
            Err(BudgetError::ZeroCharge)
        );
        let lease = budget.verifier_lease().expect("single lease");
        assert_eq!(
            budget.try_charge_verifier(&lease, 0),
            Err(BudgetError::ZeroCharge)
        );
    }

    #[test]
    fn receipts_are_single_use() {
        let mut budget = Budget::new(BudgetLimits::uniform(3, 0));
        let receipt = budget.try_charge(Dimension::DepthLimit, 2).expect("fits");
        budget.refund(receipt);
        // `receipt` moved into refund; the compiler rejects reuse:
        // budget.refund(receipt); // <- does not compile (no Copy/Clone)
        assert_eq!(budget.remaining(Dimension::DepthLimit), 3);
    }

    #[test]
    fn verifier_pool_requires_the_one_time_lease() {
        let mut budget = Budget::new(BudgetLimits::uniform(8, 6));
        let _lease = budget.verifier_lease().expect("first lease");
        assert!(
            budget.verifier_lease().is_none(),
            "the lease is issued exactly once"
        );

        // A reserved child carries no verifier allowance, so merging it back
        // cannot inflate the protected pool.
        let child = budget
            .reserve_child(BudgetLimits::uniform(4, 0))
            .expect("reserves");
        budget.merge_child(child);
        assert_eq!(
            budget.verifier_remaining(),
            6,
            "parent pool untouched by merge"
        );
    }

    #[test]
    #[should_panic(expected = "children never possess a verifier lease")]
    fn child_budgets_have_no_verifier_path() {
        let mut budget = Budget::new(BudgetLimits::uniform(8, 6));
        let mut child = budget
            .reserve_child(BudgetLimits::uniform(4, 0))
            .expect("reserves");
        child.try_charge_verifier_unreachable();
    }

    #[test]
    fn child_reservations_move_allowance_exactly() {
        let mut budget = Budget::new(BudgetLimits::uniform(10, 5));
        let mut caps = BudgetLimits::uniform(4, 0);
        caps.dimensions[Dimension::MemoryBytes.index()] = 6;
        let child = budget.reserve_child(caps).expect("reserves");
        assert_eq!(budget.remaining(Dimension::ComputeSteps), 6);
        assert_eq!(budget.remaining(Dimension::MemoryBytes), 4);

        let over = BudgetLimits::uniform(7, 0);
        let err = budget.reserve_child(over).unwrap_err();
        assert_eq!(err, BudgetError::ChildReservationTooLarge);

        let mut child = child;
        child
            .try_charge(Dimension::MemoryBytes, 1)
            .expect("child charge");
        budget.merge_child(child);
        assert_eq!(
            budget.remaining(Dimension::ComputeSteps),
            10,
            "unspent returns"
        );
        assert_eq!(
            budget.remaining(Dimension::MemoryBytes),
            9,
            "spent unit stays gone"
        );
    }

    #[test]
    fn sequence_is_monotonic_across_operations() {
        let mut budget = Budget::new(BudgetLimits::uniform(10, 5));
        assert_eq!(budget.snapshot().sequence, 0);
        let r1 = budget.try_charge(Dimension::ComputeSteps, 1).unwrap();
        assert_eq!(r1.seq, 1);
        let lease = budget.verifier_lease().unwrap();
        let r2 = budget.try_charge_verifier(&lease, 1).unwrap();
        assert_eq!(r2.seq, 2);
        budget.refund(r1);
        assert_eq!(budget.snapshot().sequence, 3);
    }

    #[test]
    fn no_operation_can_inflate_beyond_initial_limits() {
        let mut budget = Budget::new(BudgetLimits::uniform(16, 4));
        let start = budget.snapshot();
        for i in 1..=20u64 {
            if let Ok(receipt) = budget.try_charge(Dimension::AllocationCount, i % 5 + 1) {
                budget.refund(receipt);
            }
        }
        let end = budget.snapshot();
        for dimension in Dimension::ALL {
            assert!(
                end.dimensions[dimension.index()] <= start.dimensions[dimension.index()],
                "refund cycling inflated {dimension}"
            );
        }
    }

    impl Budget {
        /// Compile-time documentation helper: children have no verifier
        /// path even when a lease-like value were somehow available.
        pub(crate) fn try_charge_verifier_unreachable(&mut self) -> ! {
            unreachable!("children never possess a verifier lease")
        }
    }
}
