# WS02 budget-dimension and outcome-variant audit (`fra-ys1`)

## 1. Slice record

- **Bead:** `fra-ys1` (WS02 residual, P2) — audit every budget dimension and outcome variant named by the runtime and security registries against the shipped types; land the missing bounded refusal path for each gap found.
- **Implementer:** CrimsonTurtle, 2026-09-15, frankensympy `main`.
- **Gate:** OPEN pending an independent reviewer (not the implementer). This artifact is the named gate input. No claim in `registries/claims.toml` is promoted; `gate://ws02-foundation` stays open.

## 2. Audit result: registries vs shipped types

| Registry name | Source | Shipped type | Status |
|---|---|---|---|
| `kernel_proved`, `certificate_verified`, `exact_cross_checked`, `certified_numeric`, `oracle_conformant`, `user_asserted`, `heuristic_candidate` | `registries/evidence_classes.toml` | `fsym_outcome::EvidenceClass` (7 variants) | match, 1:1 |
| `conditional`, `inconclusive`, `refused`, `cancelled`, `timed_out`, `resource_exhausted`, `unsupported`, `internal_fault` | `registries/evidence_classes.toml` (non-evidence outcomes) | `fsym_outcome::NonEvidenceOutcome` (8 variants, `parse` fails closed) | match, 1:1 |
| `verifier_budget_reserved_before_generation = true` | `registries/algorithm_portfolios.toml` | `BudgetLimits.verifier_pool`, `VerifierLease`, `try_charge_verifier` (lease-gated; generators refused via `VerifierPoolAccessDenied`) | match |
| `replay_resets_budget = false` | `registries/workspace_transactions.toml` | runtime replay lane property (WS13), not an L1 type gap | carried by WS13 residual obligations |
| `callback_boundary` cancellation classes | `registries/python_effects.toml` | `FailureKind::{Timeout,Cancelled,ResourceExhausted}` charged as divergence (`fsym-runtime::monitor`) | match after the gap below was closed |
| **time** exhaustion (timeouts must charge elapsed time before `TimedOut`; bounded callback walls) | runtime/security discipline (§15, §18); `ResourceClass::TimeBudget` already named the class | **no metered `Dimension` existed** — `ResourceClass::TimeBudget` named a resource class no meter could charge | **GAP → closed in this slice** |

## 3. The gap and the landed bounded refusal path

`fsym_budget::Dimension` carried five variants while the L0 reporting view `fsym_outcome::ResourceClass` already named `time_budget`. A timed-out observation could be *reported* as a time exhaustion but never *charged* against a budget: there was no bounded refusal path (typed `BudgetError::Exhausted { dimension: TimeBudget, .. }`) at the metering layer.

Landed in `crates/fsym-budget/src/lib.rs`:

- `Dimension::TimeBudget` appended (canonical order is a persisted format: indices 0–4 unchanged, new index 5; `DIMENSION_COUNT` 5 → 6; `as_str` = `"time_budget"`, mirroring `ResourceClass::TimeBudget`).
- Unit documented: metered wall time in microseconds, charged by supervised lanes (bounded Python callback boundaries, cancellation drains) and by timeouts before returning `TimedOut`.
- Array-shaped types (`BudgetLimits.dimensions`, `BudgetSnapshot.dimensions`, `remaining`, batch aggregation) absorb the new slot automatically via `DIMENSION_COUNT`; nothing else in the crate hard-codes 5.

New tests (all green, `cargo test -p fsym-budget`, 20 passed):

- `time_budget_charges_refuses_and_refunds_like_any_dimension` — positive charge, typed `Exhausted` refusal (atomic: nothing moves), refund, zero-charge rejection.
- `canonical_dimension_order_is_a_persisted_format` — index/identifier stability table (planted negative: any index churn or rename fails).
- `time_budget_refuses_at_child_reservation_boundaries` — child cap refusal with `Exhausted { TimeBudget, 1, 0 }`, and the reservation-moves-allowance invariant (parent holds the remainder; child charges never double-draw).

## 4. Out of scope (recorded, not claimed)

- `proof_budget` in `algorithm_portfolios.toml` `state_dimensions` is a selector *state* dimension (WS13 portfolio work), not a metered budget dimension; no L1 gap.
- Replay-not-resetting-budgets is a runtime property; owned by the WS13 residual (`fra-ws13-portfolio-runtime-qup`).
- No documentation-only promotion: `claims.toml` untouched, `quality_gates.toml` remains `enforced = false`.
