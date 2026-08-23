//! Typed mathematical and execution outcomes.
//!
//! `MathOutcome` preserves the SymPy `evaluate=False` contract: a held form
//! is a first-class result, not a failure. `ExecutionOutcome` couples the
//! mathematical result to its receipt and final charge state so callers can
//! never observe math without its evidence trail.

use crate::budget::{BudgetDimension, ChargeLedger};
use fsym_id::ReceiptId;

/// Why an expression remains unevaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnevaluatedReason {
    /// Constructed with the evaluate=False contract.
    HeldByContract,
    /// No registered rule applies within the current budget.
    NoApplicableRule,
}

/// The mathematical result of evaluating a term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MathOutcome<T> {
    /// Fully evaluated value.
    Evaluated(T),
    /// Deliberately unevaluated form (evaluate=False semantics).
    Held(T, UnevaluatedReason),
}

impl<T> MathOutcome<T> {
    /// Returns the value for both evaluated and held outcomes: in both
    /// cases a well-formed term exists.
    pub fn into_term(self) -> T {
        match self {
            MathOutcome::Evaluated(term) | MathOutcome::Held(term, _) => term,
        }
    }

    pub fn is_evaluated(&self) -> bool {
        matches!(self, MathOutcome::Evaluated(_))
    }

    pub fn is_held(&self) -> bool {
        matches!(self, MathOutcome::Held(_, _))
    }
}

/// Final per-dimension charges of a finished region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChargeSummary {
    pub eval_steps: usize,
    pub recursion_depth: usize,
    pub allocations: usize,
}

impl ChargeSummary {
    /// Snapshots the final totals of a ledger.
    pub fn of(ledger: &ChargeLedger) -> Self {
        Self {
            eval_steps: ledger.charged(BudgetDimension::EvalSteps),
            recursion_depth: ledger.charged(BudgetDimension::RecursionDepth),
            allocations: ledger.charged(BudgetDimension::Allocations),
        }
    }
}

/// The execution result of running a region under budget discipline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionOutcome<M> {
    /// Region finished; carries the math outcome, its receipt identity,
    /// and the final charges actually consumed.
    Completed {
        math: MathOutcome<M>,
        receipt: ReceiptId,
        charges: ChargeSummary,
    },
    /// A dimension ran out before completion; names which one and how
    /// much had been consumed when it happened.
    BudgetExhausted {
        dimension: BudgetDimension,
        charged: usize,
        cap: usize,
    },
    /// Region was cancelled by its owning scope; no math result exists.
    Cancelled,
    /// Execution failed for a non-budget reason.
    Failed(crate::RuntimeError),
}

impl<M> ExecutionOutcome<M> {
    /// Builds a `Completed` outcome from parts, snapshotting the ledger.
    pub fn completed(math: MathOutcome<M>, receipt: ReceiptId, ledger: &ChargeLedger) -> Self {
        ExecutionOutcome::Completed {
            math,
            receipt,
            charges: ChargeSummary::of(ledger),
        }
    }

    pub fn math(&self) -> Option<&MathOutcome<M>> {
        match self {
            ExecutionOutcome::Completed { math, .. } => Some(math),
            _ => None,
        }
    }

    /// Charges reported by this outcome. Cancellation reports nothing.
    pub fn charges(&self) -> Option<ChargeSummary> {
        match self {
            ExecutionOutcome::Completed { charges, .. } => Some(*charges),
            ExecutionOutcome::BudgetExhausted { .. } => None,
            ExecutionOutcome::Cancelled => None,
            ExecutionOutcome::Failed(_) => None,
        }
    }

    /// True when the outcome exhausted `dimension`.
    pub fn exhausted(&self, dimension: BudgetDimension) -> bool {
        matches!(
            self,
            ExecutionOutcome::BudgetExhausted {
                dimension: d,
                ..
            } if *d == dimension
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BudgetLimits;
    use std::str::FromStr;

    #[test]
    fn held_forms_are_first_class_results() {
        let held: MathOutcome<u32> = MathOutcome::Held(7, UnevaluatedReason::HeldByContract);
        assert!(held.is_held());
        assert!(!held.is_evaluated());
        assert_eq!(held.into_term(), 7);
    }

    #[test]
    fn completed_outcome_carries_receipt_and_final_charges() {
        let mut ledger = ChargeLedger::new(BudgetLimits::uniform(100));
        ledger.charge(BudgetDimension::EvalSteps, 12).unwrap();
        let receipt = ReceiptId::new(9).unwrap();
        let outcome: ExecutionOutcome<u32> =
            ExecutionOutcome::completed(MathOutcome::Evaluated(42), receipt, &ledger);
        let summary = outcome.charges().unwrap();
        assert_eq!(summary.eval_steps, 12);
        assert_eq!(summary.recursion_depth, 0);
        let math = outcome.math().unwrap();
        assert!(math.is_evaluated());
        let MathOutcome::Evaluated(value) = math else {
            unreachable!("just asserted evaluated");
        };
        assert_eq!(*value, 42);
        assert!(!outcome.exhausted(BudgetDimension::EvalSteps));
    }

    #[test]
    fn exhaustion_names_dimension_and_cancelled_has_no_charges() {
        let e: ExecutionOutcome<u32> = ExecutionOutcome::BudgetExhausted {
            dimension: BudgetDimension::Allocations,
            charged: 99,
            cap: 99,
        };
        assert!(e.exhausted(BudgetDimension::Allocations));
        assert!(!e.exhausted(BudgetDimension::EvalSteps));
        assert!(e.math().is_none());

        let c: ExecutionOutcome<u32> = ExecutionOutcome::Cancelled;
        assert_eq!(
            c.charges(),
            None,
            "cancelled regions must not fabricate charge summaries"
        );
    }

    #[test]
    fn receipt_ids_are_typed_not_strings() {
        // Round-trip through the canonical textual form proves the id is a
        // typed fsym-id kind, satisfying "no stringly IDs" at this layer.
        let receipt = ReceiptId::new(5).unwrap();
        let parsed = ReceiptId::from_str(&receipt.to_string()).unwrap();
        assert_eq!(parsed, receipt);
    }
}
