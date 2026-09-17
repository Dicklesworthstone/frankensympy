# WS02 budget-dimension and outcome-variant audit (`fra-ys1`)

## 1. Slice record

- **Bead:** `fra-ys1` (WS02 residual, P2) — audit every budget dimension and outcome variant named by the runtime and security registries against the shipped types; land the missing bounded refusal path for each gap found.
- **Implementer:** CrimsonTurtle, 2026-09-15, frankensympy `main`.
- **Gate:** OPEN. Independent reviewer PlumSnow (2026-09-17) accepts the caller-charged `TimeBudget` primitive; workspace acceptance passed. Slice closure remains blocked by the registry-policy discrepancy below. No claim in `registries/claims.toml` is promoted; `gate://ws02-foundation` stays open.

## 2. Audit result: registries vs shipped types

| Registry name | Source | Shipped type | Status |
|---|---|---|---|
| `kernel_proved`, `certificate_verified`, `exact_cross_checked`, `certified_numeric`, `oracle_conformant`, `user_asserted`, `heuristic_candidate` | `registries/evidence_classes.toml` | `fsym_outcome::EvidenceClass` (7 variants) | match, 1:1 |
| `conditional`, `inconclusive`, `refused`, `cancelled`, `timed_out`, `resource_exhausted`, `unsupported`, `internal_fault` | `registries/evidence_classes.toml` (non-evidence outcomes) | `fsym_outcome::NonEvidenceOutcome` (8 variants, `parse` fails closed) | match, 1:1 |
| `verifier_budget_reserved_before_generation = true` | `registries/algorithm_portfolios.toml` | `BudgetLimits.verifier_pool`, `VerifierLease`, `try_charge_verifier` (lease-gated; generators refused via `VerifierPoolAccessDenied`) | match |
| `replay_resets_budget = false` | `registries/workspace_transactions.toml` | runtime replay lane property (WS13), not an L1 type gap | carried by WS13 residual obligations |
| `callback_boundary` cancellation classes | `registries/python_effects.toml` | Runtime monitor counts `FailureKind::{Timeout,Cancelled,ResourceExhausted}` as statistical divergence; this does not meter elapsed time or establish callback supervision | runtime/Python obligation, not discharged by this L0 slice |
| **time** exhaustion (timeouts must charge elapsed time before `TimedOut`; bounded callback walls) | runtime/security discipline (§15, §18); `ResourceClass::TimeBudget` already named the class | **no metered `Dimension` existed** — `ResourceClass::TimeBudget` named a resource class no meter could charge | **GAP → closed in this slice** |

## 3. The gap and the landed bounded refusal path

`fsym_budget::Dimension` carried five variants while the L0 reporting view `fsym_outcome::ResourceClass` already named `time_budget`. A timed-out observation could be *reported* as a time exhaustion but never *charged* against a budget: there was no bounded refusal path (typed `BudgetError::Exhausted { dimension: TimeBudget, .. }`) at the metering layer.

Landed in `crates/fsym-budget/src/lib.rs`:

- `Dimension::TimeBudget` appended (canonical order is a persisted format: indices 0–4 unchanged, new index 5; `DIMENSION_COUNT` 5 → 6; `as_str` = `"time_budget"`, mirroring `ResourceClass::TimeBudget`).
- Unit documented: caller-metered wall time in microseconds. Supervised callback/drain/timeout lanes must supply elapsed charges; independent review found no production elapsed-time charging caller. Automatic elapsed-time enforcement is not implemented by adding this dimension.
- Array-shaped types (`BudgetLimits.dimensions`, `BudgetSnapshot.dimensions`, `remaining`, batch aggregation) absorb the new slot automatically via `DIMENSION_COUNT`; nothing else in the crate hard-codes 5.

New tests (all green, `cargo test -p fsym-budget`, 20 passed):

- `time_budget_charges_refuses_and_refunds_like_any_dimension` — positive charge, typed `Exhausted` refusal (atomic: nothing moves), refund, zero-charge rejection.
- `canonical_dimension_order_is_a_persisted_format` — index/identifier stability table (planted negative: any index churn or rename fails).
- `time_budget_refuses_at_child_reservation_boundaries` — child cap refusal with `Exhausted { TimeBudget, 1, 0 }`, and the reservation-moves-allowance invariant (parent holds the remainder; child charges never double-draw).

## 4. Out of scope (recorded, not claimed)

- `proof_budget` in `algorithm_portfolios.toml` `state_dimensions` is a selector *state* dimension (WS13 portfolio work), not a metered budget dimension; no L1 gap.
- Replay-not-resetting-budgets is a runtime property; owned by the WS13 residual (`fra-ws13-portfolio-runtime-qup`).
- No documentation-only promotion: `claims.toml` untouched, `quality_gates.toml` remains `enforced = false`.

## 5. Independent review — PlumSnow, 2026-09-17

Reviewed source commit `badb2a74c332cae6b2c1ea4e16545aca7d240148`, tree `ea33ba805246e22b737822ef94d5d9c62423da5b`; implementation source unchanged during review. Lockfile SHA-256: `fb70f915a4d6ef5bb2663edd9353af33545a02e5cf406720b6f60b9e4b96521f`. Budget source SHA-256: `fdeb490de675f07e5200488e567780415984e603c73363f8430a683e74f05aac`. Outcome source SHA-256: `9228745d47efde542bd931d8deebe67be765fc8d04ef8d0150ecb41ae22e2e3f`.

Observed commands:

- `rch exec -- cargo test -p fsym-budget --locked`: exit 0, 20 tests passed, 0 failed; no doc tests.
- `rch exec -- cargo test -p fsym-budget --lib time_budget_charges_refuses_and_refunds_like_any_dimension --locked -- --exact tests::time_budget_charges_refuses_and_refunds_like_any_dimension`: exit 0, 1 passed, 19 filtered out.
- All three RCH invocations selected workers but fell back to local execution with `RCH-E415`: dependency materialization reported workspace-inherited dependency `serde` without an enclosing workspace. These are **local fallback results**, not remote-execution evidence.
- `./scripts/check.sh registries`: exit 0, all executable planning registries passed (10 tool tests, 123 conformance-lab tests, 11 performance-tool tests).
- `br dep cycles`: exit 0, no dependency cycles.
- `rch exec -- cargo test --workspace --locked`: exit 0, completed in 311.90 seconds; workspace tests and doc tests passed, with existing ignored tests retained. RCH used the same local fallback described above.

The time dimension participates in atomic charging, batch aggregation, refunds and child reservation through the existing generic ledger. Exact-cap and over-cap tests pass. No production correction is needed for that bounded primitive.

**Closure finding:** evidence-class and non-evidence-outcome *names* match 7/7 and 8/8, respectively, but evidence properties do not match completely. `registries/evidence_classes.toml:58` defines `UserAsserted.can_discharge_exact_equality = "inside_declared_context_only"`. `crates/fsym-outcome/src/lib.rs:43-48` has no such `Discharge` variant; lines 120-126 return `No`, and `discharge_predicates_mirror_registry` pins that mismatch. This is conservative underrepresentation, not a demonstrated false acceptance. Correct it with a context-restricted typed policy and independent evidence; never replace it with unconditional `Yes` or weaken the registry. The review does not modify production code and therefore does not self-approve a repair.

**Scope boundaries:** runtime clock charging, callback supervision and timeout precedence remain runtime/Python obligations. The broader symbolic resource dimensions in the runtime architecture are not all implemented by these six generic slots. The separately reported assumption-provenance digest issue remains an integrity-contract issue, not a time-budget regression; this review neither changes nor certifies it.

Concrete repair proposed to implementer CrimsonTurtle: add `Discharge::InsideDeclaredContextOnly`, map its `as_str`/`parse` spelling to `inside_declared_context_only`, return it specifically for `EvidenceClass::UserAsserted.can_discharge_exact_equality()`, and correct the existing registry-mirroring test so it distinguishes context-only discharge from unconditional `Yes`. Preserve `UserAsserted.is_mathematical() == false` and its rejection by mathematical-establishment constructors. Audit all `Discharge` consumers before changing the exported enum, then run the outcome-specific tests and required workspace gates. This is an evidence-policy representation repair, not a new budget dimension or refund path.
