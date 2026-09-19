# Independent gate review — fra-rc-formal-ma3 (WS06 formal projection slice)

- Reviewer: TurquoiseHorizon (omp/glm-5.3-flash), independent of the implementation author
  (BoldGorge, commits fce08e6 + 6502636) and of any prior reviewer.
- Reviewed tree: origin/main 6502636 2c75b6aec6c0f9a49269b41baf1bff64f (clean) plus the
  13-line Cargo.lock repair recorded in this review (F1).
- Date: 2026-09-19.

## Verdict: PASS (implementation slice accepted; profile lean_core_zz_product_v1
## review_status advanced from pending_independent_review on this evidence)

## Independently re-executed (exit codes inspected on the terminal)

1. `cargo test -p fsym-formal --all-targets --locked` (LOCAL execution;
   remote offload unavailable, see F2): 9 passed, 0 failed across all targets.
   Raw log: `artifacts/audit/ws06_formal_review_th1/test_formal.log`.
2. Full external gate, fresh artifacts dir:
   `python3 tools/check_formal_projection.py --lean ~/.elan/toolchains/leanprover--lean4---v4.32.2/bin/lean
   --projector <built example> --artifacts artifacts/audit/ws06_formal_projection_review_th2
   --expected-environment 88d1bfed...9f` → terminal `PASS: native offline, independent mapping
   mutations, fixed replay, pinned foreign positive/negative, missing checker and resource
   refusal`, exit 0. 78 raw artifact files retained (per-key omit/alter results, forged
   checker, false-theorem Lean rejection, missing-checker, -M1 resource refusal, replay
   determinism, environment manifest with computed sha256 exactly equal to the pin).

## Code review findings (what the verifier actually checks)

- Native-first: `admission::admit` runs bounded allocation-free preflight, then mandatory
  `verify_capsule(bytes, Some(root), 2)` with digest equality, then canonical re-encode
  equality, then frozen-shape gates (ZZ, scalar 1, exactly 2 monic factors, ≤3 objects,
  degree ≤ 64, i64 coefficients) — all before any projection (crates/fsym-formal/src/admission.rs).
- Independent mapping checker never calls the projector: it parses the frozen Lean grammar and
  compares dense ascending coefficients against native sparse objects term-by-term
  (crates/fsym-formal/src/checker.rs), and the boundary tests force semantic source changes
  (coefficient, List Int→Nat, +→−, decide→sorry, theorem rename, axiom print removal) to be
  refused even when transport and receipt digests are consistently re-forged
  (tests/projection_boundary.rs:84).
- External root authority cannot substitute the requested claim root; a second, valid,
  self-consistent identity envelope is refused against the original root
  (tests/projection_boundary.rs:155).
- Cancellation at every SafePoint returns `Cancelled` and never mints checked state
  (tests/projection_boundary.rs:209). Resource exhaustion (degree, coefficient height,
  capsule/JSON bytes, native fuel, `-M1`) is typed `ResourceExhausted`/`Inconclusive`, never a
  silent pass. Refusals do not touch native evidence classes (ProjectionError only).
- No runtime code loaders, no hidden CAS, no downloads: the tool shells out only to the pinned
  `lean` executable with `LEAN_PATH=` emptied, bounded output, RLIMIT_FSIZE, single owned child.

## F1 (defect in fce08e6 evidence claims, repaired in this review's commits)

`crates/fsym-formal` was added as a workspace member without committing its Cargo.lock entry:
the committed tree failed every `--locked` cargo command
(`error: cannot update the lock file ... because --locked was passed`), reproduced locally and
on two rch workers. Consequently fce08e6's claimed `cargo check/clippy/test --workspace`
exit-0 runs cannot have been executed at the committed tree with `--locked`. Repair: offline
`cargo metadata` regenerated the lock adding exactly the 13-line `fsym-formal` package entry
(no version churn; `git diff` inspected). Native code is unaffected.

## F2 (environment observation, no code change)

`~/.elan/bin/lean` is an elan shim; hashing the shim prefix yields a different environment
digest and the tool honestly REFUSEs (`artifacts/audit/ws06_formal_projection_review_th1/`,
retained). The pinned digest reproduces exactly from the toolchain binary
`~/.elan/toolchains/leanprover--lean4---v4.32.2/bin/lean`. Callers must pass the toolchain
path; the fail-closed refusal is the gate working as designed.

## Claim effects

- Profile `lean_core_zz_product_v1`: `review_status` → `independent_review_passed` with this
  artifact as provenance. Status remains `planned` (not certified); no claims promotion beyond
  the bead's bounded deliverable.
- Bead fra-rc-formal-ma3 closed on this evidence.
