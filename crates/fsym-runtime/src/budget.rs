//! Multidimensional hierarchical budgets with a protected independent
//! verifier reservation.
//!
//! Discipline encoded here (WS02):
//!
//! - Charges are monotone. Fallback paths absorb the attempted attempt's
//!   ledger (`absorb`); they never reset counters.
//! - Child budgets propagate their final charges into their parent
//!   explicitly; propagation adds, never overwrites.
//! - The verifier reservation is a distinct type (`VerifierReservation`)
//!   whose protected pool is unreachable from `GeneratorBudget`: there is
//!   no accessor, trait, or conversion that lets generator-scoped code
//!   consume the verifier allowance.
//! - Wall-clock time is deliberately absent: timeouts live in execution
//!   policy (`RuntimeBudget.timeout`), never as the only budget.

use serde::{Deserialize, Serialize};

/// Dimensions along which work is metered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BudgetDimension {
    /// Discrete evaluation steps.
    EvalSteps,
    /// Maximum structural recursion depth.
    RecursionDepth,
    /// Tracked allocation count.
    Allocations,
}

/// All dimensions, in canonical order.
pub const DIMENSIONS: [BudgetDimension; 3] = [
    BudgetDimension::EvalSteps,
    BudgetDimension::RecursionDepth,
    BudgetDimension::Allocations,
];

/// Per-dimension caps for a budget scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetLimits {
    pub max_eval_steps: usize,
    pub max_recursion_depth: usize,
    pub max_allocations: usize,
}

impl BudgetLimits {
    /// Caps every dimension at `n`.
    pub fn uniform(n: usize) -> Self {
        Self {
            max_eval_steps: n,
            max_recursion_depth: n,
            max_allocations: n,
        }
    }

    fn cap(self, dimension: BudgetDimension) -> usize {
        match dimension {
            BudgetDimension::EvalSteps => self.max_eval_steps,
            BudgetDimension::RecursionDepth => self.max_recursion_depth,
            BudgetDimension::Allocations => self.max_allocations,
        }
    }
}

impl Default for BudgetLimits {
    fn default() -> Self {
        Self::uniform(usize::MAX)
    }
}

/// Error returned when a charge would exceed a remaining cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetExhausted {
    pub dimension: BudgetDimension,
    pub charged: usize,
    pub cap: usize,
}

/// Monotone per-dimension charge counters bounded by [`BudgetLimits`].
///
/// Counters only ever grow. Merging ledgers adds them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChargeLedger {
    limits: BudgetLimits,
    eval_steps: usize,
    recursion_depth: usize,
    allocations: usize,
}

impl ChargeLedger {
    pub fn new(limits: BudgetLimits) -> Self {
        Self {
            limits,
            eval_steps: 0,
            recursion_depth: 0,
            allocations: 0,
        }
    }

    pub fn limits(&self) -> BudgetLimits {
        self.limits
    }

    /// Total charged along a dimension.
    pub fn charged(&self, dimension: BudgetDimension) -> usize {
        match dimension {
            BudgetDimension::EvalSteps => self.eval_steps,
            BudgetDimension::RecursionDepth => self.recursion_depth,
            BudgetDimension::Allocations => self.allocations,
        }
    }

    fn set_charged(&mut self, dimension: BudgetDimension, value: usize) {
        match dimension {
            BudgetDimension::EvalSteps => self.eval_steps = value,
            BudgetDimension::RecursionDepth => self.recursion_depth = value,
            BudgetDimension::Allocations => self.allocations = value,
        }
    }

    /// Charges `amount` along `dimension`, failing closed when the charge
    /// would exceed the cap. On failure nothing is charged for this call;
    /// previously accumulated charges remain (callers running fallbacks
    /// must `absorb` the failed attempt's ledger rather than restart).
    pub fn charge(
        &mut self,
        dimension: BudgetDimension,
        amount: usize,
    ) -> Result<(), BudgetExhausted> {
        let current = self.charged(dimension);
        let Some(new_total) = current.checked_add(amount) else {
            return Err(BudgetExhausted {
                dimension,
                charged: current,
                cap: self.limits.cap(dimension),
            });
        };
        if new_total > self.limits.cap(dimension) {
            return Err(BudgetExhausted {
                dimension,
                charged: current,
                cap: self.limits.cap(dimension),
            });
        }
        self.set_charged(dimension, new_total);
        Ok(())
    }

    /// Absorbs another ledger's charges into this one (fallback merge).
    ///
    /// This is the operation that makes "budget charges never reset on
    /// fallback" true by construction: absorbing adds the failed attempt's
    /// consumption onto ours, capped check included.
    pub fn absorb(&mut self, other: &ChargeLedger) -> Result<(), BudgetExhausted> {
        for dimension in DIMENSIONS {
            self.charge(dimension, other.charged(dimension))?;
        }
        Ok(())
    }

    /// Propagates this ledger's final charges into a parent scope.
    ///
    /// Child scopes call this exactly once when their region ends. The
    /// parent's totals grow by the child's totals; neither side resets.
    pub fn propagate_into(&self, parent: &mut ChargeLedger) -> Result<(), BudgetExhausted> {
        for dimension in DIMENSIONS {
            parent.charge(dimension, self.charged(dimension))?;
        }
        Ok(())
    }
}

/// Budget handle handed to candidate-generating code.
///
/// Generators draw only from the shared ledger. There is deliberately no
/// path from this type to any [`VerifierReservation`] pool.
#[derive(Debug)]
pub struct GeneratorBudget {
    ledger: ChargeLedger,
}

impl GeneratorBudget {
    pub fn new(ledger: ChargeLedger) -> Self {
        Self { ledger }
    }

    pub fn charge(
        &mut self,
        dimension: BudgetDimension,
        amount: usize,
    ) -> Result<(), BudgetExhausted> {
        self.ledger.charge(dimension, amount)
    }

    pub fn charged(&self, dimension: BudgetDimension) -> usize {
        self.ledger.charged(dimension)
    }

    pub fn ledger(&self) -> &ChargeLedger {
        &self.ledger
    }
}

/// Protected allowance reserved for the independent verifier of a region.
///
/// Created once per portfolio region out of the parent budget; consumed
/// only through [`VerifierReservation::consume`]. Generator code receives
/// [`GeneratorBudget`], which shares no state with this type, so a
/// generator can never spend the verifier's reservation even by misuse.
#[derive(Debug)]
pub struct VerifierReservation {
    ledger: ChargeLedger,
}

impl VerifierReservation {
    /// Carves the reservation out of `parent`. The reserved allowance is
    /// debited from the parent immediately so generators cannot spend it.
    pub fn carve_out(
        parent: &mut ChargeLedger,
        limits: BudgetLimits,
    ) -> Result<Self, BudgetExhausted> {
        for dimension in DIMENSIONS {
            parent.charge(dimension, limits.cap(dimension))?;
        }
        Ok(Self {
            ledger: ChargeLedger::new(limits),
        })
    }

    pub fn consume(
        &mut self,
        dimension: BudgetDimension,
        amount: usize,
    ) -> Result<(), BudgetExhausted> {
        self.ledger.charge(dimension, amount)
    }

    pub fn remaining(&self, dimension: BudgetDimension) -> usize {
        self.ledger.limits.cap(dimension) - self.ledger.charged(dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_are_monotone_and_exhaustion_names_the_dimension() {
        let mut ledger = ChargeLedger::new(BudgetLimits {
            max_eval_steps: 10,
            ..BudgetLimits::uniform(100)
        });
        ledger.charge(BudgetDimension::EvalSteps, 6).unwrap();
        let err = ledger
            .charge(BudgetDimension::EvalSteps, 5)
            .expect_err("must exceed");
        assert_eq!(err.dimension, BudgetDimension::EvalSteps);
        assert_eq!(err.charged, 6);
        assert_eq!(err.cap, 10);
        // Failed charge left prior charges intact.
        assert_eq!(ledger.charged(BudgetDimension::EvalSteps), 6);
    }

    #[test]
    fn fallback_absorb_accumulates_instead_of_resetting() {
        let mut main = ChargeLedger::new(BudgetLimits::uniform(100));
        main.charge(BudgetDimension::EvalSteps, 30).unwrap();

        let mut attempt = ChargeLedger::new(BudgetLimits::uniform(100));
        attempt.charge(BudgetDimension::EvalSteps, 25).unwrap();
        attempt.charge(BudgetDimension::Allocations, 5).unwrap();

        // Strategy A fails mid-flight; its ledger is absorbed by the
        // fallback strategy B operating under the same main budget.
        main.absorb(&attempt).unwrap();
        assert_eq!(main.charged(BudgetDimension::EvalSteps), 55);
        assert_eq!(main.charged(BudgetDimension::Allocations), 5);
    }

    #[test]
    fn child_propagation_adds_to_parent() {
        let mut parent = ChargeLedger::new(BudgetLimits::uniform(1000));
        let mut child = ChargeLedger::new(BudgetLimits::uniform(100));
        child.charge(BudgetDimension::RecursionDepth, 7).unwrap();
        child.propagate_into(&mut parent).unwrap();
        child.charge(BudgetDimension::RecursionDepth, 1).unwrap();
        child.propagate_into(&mut parent).unwrap();
        assert_eq!(parent.charged(BudgetDimension::RecursionDepth), 15);
        assert_eq!(child.charged(BudgetDimension::RecursionDepth), 8);
    }

    #[test]
    fn carve_out_debits_parent_so_generators_cannot_spend_reservation() {
        let mut shared = ChargeLedger::new(BudgetLimits::uniform(200));
        let reservation_limits = BudgetLimits::uniform(50);
        let mut reservation =
            VerifierReservation::carve_out(&mut shared, reservation_limits).unwrap();

        // Parent was debited the full reservation up front.
        assert_eq!(shared.charged(BudgetDimension::EvalSteps), 50);

        let mut generator = GeneratorBudget::new(shared);
        // Generator can drain its own shared remainder...
        generator.charge(BudgetDimension::EvalSteps, 150).unwrap();
        assert!(generator.charge(BudgetDimension::EvalSteps, 1).is_err());

        // ...but the verifier's protected pool is untouched and usable.
        assert_eq!(reservation.remaining(BudgetDimension::EvalSteps), 50);
        reservation.consume(BudgetDimension::EvalSteps, 20).unwrap();
        assert_eq!(reservation.remaining(BudgetDimension::EvalSteps), 30);
    }
}
