# Independent gate review — fra-rc-portfolio-gate-4or over fra-rc-portfolio-9i7 (WS13)

- Reviewer: TurquoiseHorizon (omp/glm-5.3-flash); not an author of the implementation
  (PlumSnow/BoldGorge, commits 363d784, 6ad78c0, ea5f680, evidence 9c739d4, 7e803b7).
- Reviewed tree: origin/main 6502636 2c75b6aec6c0f9a49269b41baf1bff64f (clean) plus the
  13-line Cargo.lock repair from the ws06 review (F1 there; without it no `--locked` command
  can run at the committed tree).
- Date: 2026-09-19.

## Verdict: PASS — implementation accepted; gate closed on this evidence

## Acceptance command (re-executed independently)

`cargo test -p fsym-runtime -p fsym-polys --locked` — LOCAL execution; remote offload was
attempted and is unavailable for this workspace (dependency preflight verified 88 roots on
vmi1156319/vmi1152480, but remote cargo fails resolving the registry for this tree; same
E415-family fallback recorded by the implementation session). Result: 13 suites,
221 passed, 0 failed. Raw full log:
`artifacts/audit/ws13_portfolio_gate_review_th1/test_runtime_polys.log`.

## Independent test obligations — mapping to evidence (code inspected, tests re-run)

- Two mathematically distinct real generators race one protected verifier budget:
  metered Zassenhaus modular/Hensel (`metered_complete_factorization`) vs metered Kronecker
  interpolation (`metered_kronecker_factorization`) on the real identity
  x^4+4 = (x^2-2x+2)(x^2+2x+2); both completion orders forced via completion channels
  (`factor_race_two_real_strategies_verify_same_product_claim`).
- Strict accepted result with exact claim binding: `verify_and_publish_candidate` requires
  verified_claim == candidate.claim == requested_claim and result binding, backed by the
  stateless reference verifier `verify_derivation_independent`
  (fsym-proof-kernel/src/kernel.rs:423) which replays every derivation step and checks
  `polynomial_ring_equivalence` by multivariate normal-form expansion — no reflexivity or
  constant substitutes anywhere in the real-factor tests.
- Invalid first candidate cannot publish: real factors with a misbound candidate claim are
  refused and the slower verified Kronecker wins
  (`invalid_fast_real_candidate_cannot_publish_before_slower_verified_one`).
- Generator budget exhaustion leaves the verifier reserve intact
  (`real_factor_generators_exhaust_budget_without_spending_verifier_reserve`); verifier-pool
  exhaustion is typed BudgetExhausted, not misreported as candidate rejection, reserve 1
  retained (`verifier_pool_exhaustion_is_not_misreported_as_candidate_rejection`).
- Cancellation through real safe points: before reservation, pre-verifier, at charged batch
  with delayed sibling + callback drained
  (tests/cancellation_injection.rs), and actual post-verifier/pre-publication cancellation on
  the real Zassenhaus workload via the cfg(test)-only publication hook — production binaries
  contain no hook (`real_factor_cancellation_after_verifier_refuses_publication`);
  charges retained, Err(Cancelled).
- All strategies refusing / typed refusals: refusing strategy in the continuation race,
  empty portfolio, oversized paid refusal, singleton configuration refusal.
- Replay and continuation: wire round-trip has no ledger capability and refuses a fresh
  equal-budget ledger with zero charge; same-live-ledger resume publishes with accounting
  that includes pre-checkpoint generation
  (`factor_race_continuation_resume_accepts_on_same_live_ledger_only`);
  replay log deterministic bit-for-bit (`test_cancellation_replay_log_deterministic_bit_for_bit`).
- Metering honesty: Yun normalization and every Zassenhaus lifting-prime attempt (including
  rejected primes), recombination allocation and candidate multiplication are precharged
  before work (implementation comments 2026-09-17T17:38, verified in code paths).

## Mutation probe (executed by the reviewer, evidence retained)

Weakened `verify_and_publish_candidate` by removing the
`verified_claim != winner.claim` publication-binding check (the exact verifier-weakening
class the gate forbids). Re-ran only
`invalid_fast_real_candidate_cannot_publish_before_slower_verified_one`: FAILED at
portfolio.rs:2080 (misbound fast candidate published, winning_strategy assertion tripped).
=> the battery kills the reintroduced weakening with a non-tautological oracle. File restored
byte-identical (`git diff` empty) and the test is green again. Transcript appended to
`test_runtime_polys.log`.

## Findings

- F1: Cargo.lock was missing the `fsym-formal` workspace entry (defect of fce08e6, fixed by
  this review's commits). Without the fix no `--locked` acceptance command can run at the
  committed tree, so earlier evidence claims that cited workspace `--locked` runs at
  fce08e6 are not reproducible as-committed. The 9i7 implementation commits predate fce08e6
  and are unaffected.
- F2: remote offload lane currently cannot run this workspace's cargo commands (registry
  resolution failure on both probed workers). All evidence above is local and labeled local.

## Claim effects

- fra-rc-portfolio-9i7: closed on this review.
- fra-rc-portfolio-gate-4or: closed on this review (claim stays implemented_uncertified /
  unpromoted at the workstream level; no profile or claims-registry promotion is made here).
