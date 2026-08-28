# WS13 portfolio race fresh-eyes audit

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-28T03:18Z
**Scope:** read-only diagnostic. No code edits. No new tests.
**Method:** mapped the WS13 reopen comments to file:line evidence in the current working tree at HEAD `f44864a`. Cross-checked each named gate against the source code.
**Source of truth:** `/data/projects/frankensympy/crates/fsym-runtime/src/portfolio.rs`, `/data/projects/frankensympy/crates/fsym-runtime/src/cx.rs`, `/data/projects/frankensympy/.beads/issues.jsonl` (comments 22 and 47/58 of `fra-ws13-portfolio-runtime-qup`).

---

## Summary verdict

The two named gaps from the WS13 reopen history have **different status**:

1. **Comment 22 (2026-08-24T17:15:45Z) — "run_portfolio_race verifies winner.derivation but ignores the verifier-returned root claim"**: **REPAIRED at HEAD**. The defect has been fixed in three places (`portfolio.rs:207-216`, `:217-223`, `:224-229`) and the result-binding at `:230-236`. The original laundering path (valid derivation of x=x paired with forged winner.claim x=y) is no longer reachable.

2. **Comment 47/58 — "the current portfolio lane is still sequential and does not yet provide the registered asupersync race, cancel-drain-finalize, or zero-orphan closure artifacts"**: **STILL FALSE**. The `for (name, strategy) in strategies` loop at `portfolio.rs:142-281` executes strategies one at a time, with no `asupersync::Cx::scope` and no region-spawned child. The comment at `portfolio.rs:141` explicitly registers the gap.

**Net effect:** the proof-correctness gates are sound; the asupersync-region-race gate is unearned. Closure of WS13 requires the region race, not another claim-binding fix.

---

## Gate-by-gate evidence

### Gate A — claim/derivation/result binding (the comment-22 defect)

**Status:** **REPAIRED at HEAD `f44864a`**

**Reopen quote (comment 22 at 2026-08-24T17:15:45Z):**
> "run_portfolio_race verifies winner.derivation but ignores the verifier-returned root claim, then independently issues KernelProved evidence for winner.claim. Therefore a valid derivation of x=x can be paired with a forged winner.claim x=y and published as KernelProved."

**Evidence — three independent checks now exist:**

1. `crates/fsym-runtime/src/portfolio.rs:207-216` — `verify_derivation_independent(&winner.derivation, context)` is called and the result is bound to `verified_claim`; a verifier error is added to `failure_reasons` and the loop `continue`s.
2. `crates/fsym-runtime/src/portfolio.rs:217-223` — `if verified_claim != winner.claim` rejects the candidate if the independent verification root does not match the candidate's self-asserted claim. This is the **direct fix for the x=x / x=y laundering path**.
3. `crates/fsym-runtime/src/portfolio.rs:224-229` — `if &verified_claim != requested_claim` rejects the candidate if the verified root does not answer the caller's original request.
4. `crates/fsym-runtime/src/portfolio.rs:230-236` — `if portfolio_claimed_result(&verified_claim) != &winner.result` rejects the candidate if the result the generator claims does not actually equal the verified result of the verified claim.

The four checks are sequential fail-closed gates: any one failure adds to `failure_reasons` and `continue`s; only an all-pass candidate can return `Ok(VerifiedPortfolioOutcome { ... })`.

**Test coverage (from the audit subagent's earlier read of `portfolio.rs:321-613`):** seven unit tests cover the failure modes — mismatched-claim, wrong-result, irrelevant-claim, oversized-request, sequential-repeat, empty-portfolio, and budget-fallback paths. The audit did not exhaustively re-read all 293 lines of tests, but the test module structure is in place.

**Honest gap:** the test module is at `portfolio.rs:311-613`; the audit did not read every test. If a particular failure path (e.g. forged derivation root) lacks a test, that is a residual gap. The owner (OliveHawk) should confirm.

### Gate B — asupersync region race (the comment-47/58 gap)

**Status:** **STILL FALSE at HEAD `f44864a`**

**Reopen quote (comment 47/58, latest):**
> "the current portfolio lane is still sequential and does not yet provide the registered asupersync race, cancel-drain-finalize, or zero-orphan closure artifacts."

**Evidence — the loop is sequential:**

1. `crates/fsym-runtime/src/portfolio.rs:142` — the for-loop signature:
   ```rust
   for (name, strategy) in strategies {
   ```
   The body executes each strategy one at a time. The first strategy runs to completion (or failure) before the second is started.

2. `crates/fsym-runtime/src/portfolio.rs:141` — explicit self-registered gap:
   ```rust
   // This baseline is deliberately sequential until the asupersync region race lands.
   ```

3. `crates/fsym-runtime/src/portfolio.rs:145-152` — each iteration:
   - reserves a child budget (`cx.reserve_child`)
   - calls `strategy(&mut child_cx)` synchronously
   - merges the child budget back (`cx.merge_child`)
   There is no `asupersync::Cx::scope(...)` call, no region-spawned child task, no `Region::enter`. The child budget is an in-process ledger reservation, not a structured-concurrency child.

4. `crates/fsym-runtime/src/cx.rs:21-26` — the `FsymCx` wrapper holds `cx: &'a Cx<Caps>` and a domain-specific `Budget`. It does **not** itself produce child regions; it is a wrapper over whatever `Cx` the caller hands it. The runtime's portfolio is responsible for actually using `Cx::scope` to spawn concurrent work; today it does not.

5. Search of `crates/fsym-runtime/src/` for `scope`, `region`, `Region` returns no usage in `portfolio.rs`. The only asupersync mention outside the test module is at `portfolio.rs:314` (`use asupersync::Cx;` in tests).

**Why this gate matters:** §7.7 of the constitution requires:
- "asupersync is the only async/concurrency runtime";
- "Every controlled spawned task has one owning region";
- "No detached background work";
- "Cancellation is request → drain → finalize";
- "Generators cannot consume verifier-reserved budget".

A sequential for-loop that calls each `strategy(&mut child_cx)` synchronously has no region ownership, no drain semantics, and no cancel-during-execution other than `cx.checkpoint()` (which polls a stop flag at lines 101, 143, 152, 205, 237). A cancellation that arrives **between** strategies will be honored; a cancellation that arrives **during** a strategy is only visible at the next checkpoint, and the strategy itself has no obligation to return early.

**Partial mitigation:** the budget mechanics (reserve / charge / merge) are correct; only the **scheduling topology** is wrong. A bounded slice that replaces the for-loop with `Cx::scope(|scope| { for s in strategies { scope.spawn(...); } })` would close the gate without touching the budget or claim-binding logic.

### Gate C — replay and checkpoint schemas (comment 47/58)

**Status:** **PARTIALLY EARNED** (the audit subagent earlier read `crates/fsym-runtime/src/{replay.rs,checkpoint.rs}` and reported the digest-binding work; the audit did not re-read those files in this pass)

**Why this gate is mentioned:** the comment 47/58 also calls out "self-contained replay/checkpoint schemas". This audit pass did not re-examine those files; the WS13 owner (OliveHawk) has touched them in comment 47/58 history and is the right person to confirm.

---

## Cross-cutting observation

`FsymCx` is a domain wrapper over `&Cx` that does not own a `Region` itself. The runtime layer must choose whether each portfolio race is:
- (a) a `Cx::scope(...)` invocation that spawns one child per strategy, with the budget split *before* scope entry and reconciled *after* scope exit, or
- (b) a `Cx::detached_cancel_context()` with all strategies executed serially and cancellation polled at checkpoints.

Today's code is closer to (b) but with full budget mechanics. A bounded slice to (a) is the natural next hardening commit.

---

## Recommendation to the WS13 owner (OliveHawk)

The smallest responsible bounded slice to close Gate B (the only unearned gate) is:

1. Replace the for-loop at `portfolio.rs:142-281` with a `cx.cx().scope(|scope| { ... })` invocation. Each strategy becomes `scope.spawn(...)`; the resulting tasks race, and the first to publish a verified outcome cancels the rest.
2. The budget mechanics (reserve, charge, merge) are unchanged — only the **scheduling** moves. The existing `child_cx` per strategy becomes a `Region`'s per-spawn budget view.
3. Add a test that proves **zero orphan tasks on cancellation**: cancel mid-race, then assert the test's task inventory is empty after the race returns. This is the named "zero-orphan closure" gate.
4. Add a test that proves **winner-take-all semantics under cancellation**: a slower strategy whose child has been cancelled does not have its `merge_child` invoked, and the budget for the cancelled strategy is refunded.

The claim-binding and result-binding logic at `portfolio.rs:207-264` does not need to change. The `verify_integrity` check at `:259` does not need to change. The receipt issue at `:239-251` does not need to change. This is **purely a scheduling-topology hardening**, not a correctness or budget hardening.

**Do not** attempt to close Gate C (replay/checkpoint) in the same commit. The replay and checkpoint files live in `crates/fsym-runtime/src/{replay.rs,checkpoint.rs}` and have their own bounded-slice history. Mixing them risks a single-conformant-bundle release masquerading as a WS13 closure.

**Do not** add new `Dimension` budget fields. The constitution says generators cannot consume verifier-reserved budget, and today's mechanics already enforce that. Adding a new dimension to support a region race would broaden the budget surface without a concrete need.

---

## File:line evidence index

| File | Lines | What is there | Gate |
|------|------|---------------|------|
| `crates/fsym-runtime/src/portfolio.rs` | 1-8 | Module docstring: registers the unearned region-race and zero-orphan requirements | B, C |
| `crates/fsym-runtime/src/portfolio.rs` | 22-23 | `MAX_CANDIDATE_STRATEGY_NAME_BYTES = 256`, `MAX_PORTFOLIO_STRATEGIES = 64` | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 25-41 | `PortfolioError` variants (Cancelled, NoVerifierLease, BudgetExhausted, BudgetAccountingFailed, WinnerVerificationFailed, AllStrategiesFailed, InvalidPortfolio) | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 44-50 | `PortfolioCandidate` struct | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 53-60 | `VerifiedPortfolioOutcome` struct | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 87-91 | `StrategyRunner` / `NamedStrategy` typedefs | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 95-100 | `run_portfolio_race` signature | (entry point) |
| `crates/fsym-runtime/src/portfolio.rs` | 101 | Initial `cx.checkpoint()` for cancel-polling | B (mitigation only) |
| `crates/fsym-runtime/src/portfolio.rs` | 103-115 | Strategy count / name bounds | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 117-119 | `has_verifier_authority` check | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 120-135 | Requested-claim preflight (charge + verification units) | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 136-138 | `initial_compute_remaining`, `failure_reasons`, `verification_attempted` setup | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 140-141 | **"deliberately sequential until the asupersync region race lands"** | B (registered gap) |
| `crates/fsym-runtime/src/portfolio.rs` | 142 | For-loop signature over strategies | B (sequential) |
| `crates/fsym-runtime/src/portfolio.rs` | 143 | Per-iteration `cx.checkpoint()` | B (mitigation only) |
| `crates/fsym-runtime/src/portfolio.rs` | 145-152 | `reserve_child`, `strategy(&mut child_cx)`, `merge_child` | B (in-process child, not a region) |
| `crates/fsym-runtime/src/portfolio.rs` | 154-160 | Failure-reason accumulation | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 165-204 | Per-candidate verifier preflight and charge | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 207-216 | `verify_derivation_independent(&winner.derivation, context)` | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 217-223 | `if verified_claim != winner.claim` | A (repaired — comment 22 fix) |
| `crates/fsym-runtime/src/portfolio.rs` | 224-229 | `if &verified_claim != requested_claim` | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 230-236 | `if portfolio_claimed_result(&verified_claim) != &winner.result` | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 239-251 | `ReceiptId::new(verifier_charge.seq())` and `VerificationReceipt::issue` | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 253-258 | `EvidenceEnvelope::new` | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 259-264 | `if !evidence.verify_integrity()` | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 266-273 | `total_steps_consumed` accounting | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 274-280 | `Ok(VerifiedPortfolioOutcome { ... })` return | A (repaired) |
| `crates/fsym-runtime/src/portfolio.rs` | 283-288 | Failure aggregation and error mapping | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 291-300 | `remaining_generator_limits` helper | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 302-309 | `portfolio_claimed_result` helper | (foundation) |
| `crates/fsym-runtime/src/portfolio.rs` | 311-613 | Test module (seven test functions) | A (test coverage) |
| `crates/fsym-runtime/src/cx.rs` | 1-10 | `FsymCx` module docstring: "wrapping pattern prescribed by asupersync's `Cx` docs" | (foundation) |
| `crates/fsym-runtime/src/cx.rs` | 12-15 | Imports: `asupersync::Cx`, `fsym_budget::*`, `crate::*` | (foundation) |
| `crates/fsym-runtime/src/cx.rs` | 17-26 | `FsymCx<'a, Caps>` struct: `cx: &'a Cx<Caps>`, `budget: Budget`, `limits: BudgetLimits`, `verifier_lease: Option<VerifierLease>` | (foundation) |
| `crates/fsym-runtime/src/cx.rs` | 28-119 | `FsymCx` impl: cancellation, charge, reserve_child, merge_child | (foundation) |
| `crates/fsym-runtime/src/cx.rs` | 121-143 | `BudgetMeter` trait impl | (foundation) |
| `crates/fsym-runtime/src/cx.rs` | 150-268 | Test module: cancellation delegates, charges flow, verifier pool is single-lease, child consumption reconciles, budget meter trait | (test coverage) |

---

## Honesty note

This audit was produced by the DustyAspen subagent. The file:line evidence was read from the current working tree at HEAD `f44864a`; no `cargo` invocation was performed. The audit is intended as a diagnostic input for the WS13 owner (OliveHawk). It does not claim WS13 closure, does not propose implementation, and does not pre-allocate work to any agent. The audit subagent's earlier read of the test module was at lines 321-613, but the test bodies were not exhaustively re-read in this pass; if a particular test (e.g. forged-derivation-root) is missing, that is a residual gap. The owner should confirm.

The audit does **not** cover:
- `crates/fsym-runtime/src/replay.rs`
- `crates/fsym-runtime/src/checkpoint.rs`
- `crates/fsym-runtime/src/remote_worker.rs`
- `crates/fsym-runtime/src/repair.rs`
- `crates/fsym-runtime/src/rng.rs`
- `crates/fsym-runtime/src/protocol.rs`
- `crates/fsym-runtime/src/graph_index.rs`
- `crates/fsym-runtime/src/workspace.rs`

These are out of scope of the comment-47/58 region-race gap; they are mentioned in the comment but warrant their own focused audits.
