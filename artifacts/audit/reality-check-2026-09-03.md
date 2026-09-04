# FrankenSymPy reality check — 2026-09-03

**Method:** /reality-check-for-project, full flow (Phase 1 → 2 → 3a → 4 → 3a → 5 → bv).
**Base commit:** `95b19c9856b12424aee34a2c21c8b746b6d547b1` ("Implement Cauchy-Euler ODE solver, enhance Mul power combination, and expand calculus and function suites"), `main` == `origin/main` at audit start.
**Measuring stick:** README.md, AGENTS.md, COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md, docs/CONSTITUTION.md, docs/WORKSTREAM_GRAPH.md, docs/FIRST_IMPLEMENTATION_CAMPAIGN.md, registries/claims.toml, registries/workstreams.toml.
**Ground truth:** live tool runs on this host this session (all commands quoted verbatim below) plus three read-only code audits (crates, Python shell/oracle, beads/governance).

---

## 1. Where we are REALLY (Phase 1 answer)

The honest one-paragraph answer: **FrankenSymPy is a real, tested, honestly-labeled pre-certification implementation core — not a SymPy replacement, and not claiming to be.** The Rust workspace (25 crates, ~73k src LOC, ~798 test fns) compiles clean, passes fmt/check/clippy/workspace-tests with zero failures, contains essentially zero stub markers, and layers correctly with the generator→verifier direction preserved. The Python shell is a genuine dual-lane object model (held forms, custom `Function.eval` hooks, assumptions, pickle, native lowering/lifting seam) that delegates to the Rust kernel. The pinned-SymPy-1.14.0 differential oracle is wired in two lanes and **ran end-to-end live during this audit: 14 fixtures paired, 13 admitted, 1 real drift caught** (custom-subclass eval lane). All of that matches the docs' own self-description (`implemented_uncertified`), which is exactly what the constitution demands.

What does NOT exist: any `validated`/`certified` claim (none — 12 claims `implemented_uncertified`, 12 `planned`, 1 `documented`); any performance evidence against a live incumbent; any end-to-end Certified Jacobian Pipeline run (C0–C11 unexecuted as a bundle); RaptorQ repair evidence; live Franken-suite integrations (only asupersync is real, as a path dependency); monitoring; Wasm; packaging (the checked-in native extension is actually **broken as shipped** — see G1).

The gap is not stub-vs-real. It is **breadth and proof**: mathematical coverage ceilings are narrow and self-documented everywhere (rational-root-only factorization, no Risch integration, `sin(x)/x`-class limits return typed `Undetermined`, eigenvalues capped at charpoly degree 2, three ODE families, a handful of simplify rules, no complex ball arithmetic, no arbitrary-precision `evalf`), the conformance corpus is 14 fixtures, and every `implemented_uncertified` claim's gate bundle is still open.

### 1.1 What IS working right now (verified this session)

| Evidence | Command / source | Result |
|---|---|---|
| Workspace gates | `cargo fmt --check`; `cargo check --workspace --all-targets`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace` (via rch remote worker hz3) | FMT_EXIT=0, CHECK_EXIT=0, CLIPPY_EXIT=0, TEST_EXIT=0 — **53 suites, 832 tests passed, 0 failed** |
| Registry validators | `./scripts/check.sh all` | Source tree clean; release readiness **refused** with 25 named blockers (fail-closed as designed: 12 O-* obligations, uncertified profile/packaging, unenforced quality_gates.toml, RELEASE-001 without bundle) |
| Beads hygiene | `br dep cycles` | No dependency cycles |
| Python shell live | scout probe | held `Add(x,x,evaluate=False).args==(x,x)`; `Symbol('p',positive=True).is_positive is True`; custom `Function` eval hook fires; `hash(Integer(3))==hash(3)`; `sympy.diff(x**3,x)==3*x**2` via native payload |
| Differential oracle live | `.venv-conformance/bin/python3 tools/conformance-lab/capture.py diff profiles/sympy-1.14.0-cpython.toml --candidate-python .venv-conformance/bin/python3` (2026-09-03, with `/tmp` extension workaround) | paired=14, admitted=13, discrepancies=1, exit 1 — one genuine `type_drift` (`subclass/ConstitutiveLawZero_zero_collapse`) |
| Oracle isolation | tools/conformance-lab candidate_runner | candidate subprocess fail-closes (exit 3) on oracle-tree leakage; pinned venvs exist with sympy 1.14.0 |

### 1.2 What is NOT working or not implemented

1. **G1 — shipped native extension is broken.** `python/_native.abi3.so` is a 403 MB debug cdylib under a filename CPython's finder cannot import as `fsym_python`; `python/fsym_python.so` is a 0-byte stub that poisons extension search. Plain `import sympy` fails honestly until rebuilt/symlinked. pyproject/maturin declare the correct module name; the checked-in artifacts violate it.
2. **G2 — one live compatibility drift.** `subclass/ConstitutiveLawZero_zero_collapse`: candidate collapses `ConstitutiveLawZero(0,k)→Integer(0)` where the pinned oracle keeps the applied form. This is the harness working, but it is an open WRONG-form divergence on the custom-subclass lane.
3. **G3 — conformance corpus is tiny and stale.** 14 seed fixtures across 3 files; goldens last captured 2026-08-23 (11 days, across kernel commits); 3 orphan `*.ndjson.ndjson` golden files contain stale divergent captures.
4. **G4 — conversion-gate debt.** 22 of 25 live beads lack the mandated 14-field record template (WORKSTREAM_GRAPH.md §32); the workstream-level beads are exactly the "broad task" shape the gate calls invalid. New decomposition is gated on fixing the records.
5. **G5 — capability ceilings (honest, but breadth).** No irreducible factorization over Q[x] (rational-root decomposition only; no Berlekamp/Cantor–Zassenhaus/Zassenhaus lifting); no modular GCD despite fsym-modular existing; limits: no L'Hôpital/series (0/0 → `Undetermined`); integration: no Risch/partial fractions/trig substitution; eigenvalues refused beyond quadratic; no eigenvectors/Jordan/SVD/matrix exponential; ODE: 3 families only; no solveset, no transcendental equations, no systems >2 vars; simplify: tiny rule catalog, no general pattern matcher; no complex ball arithmetic, no arbitrary-precision evalf (f64 only, 7 functions); PyO3 bridge string-typed with u64-only ntheory; `ImmutableMatrix`/`MutableDenseMatrix` are cosmetic aliases of one class; eight `sympy/core/*.py` stub files fake module identity via `__module__` reassignment.
6. **G6 — everything behind a gate.** No gate bundle exists yet for any `implemented_uncertified` claim (deterministic-term-identity fixtures, two-phase publication, no-orphan-work cancellation matrix, deterministic replay, fuzz/adversarial decoder corpus, per-certificate mutation gates).
7. **G7 — no performance evidence.** PERF-001 `planned`; runtime crate has paired-benchmark machinery but no live-incumbent run has been admitted.
8. **G8 — campaign not composed.** Certified Jacobian Pipeline stages C0–C11 have partial substrates but no end-to-end artifact bundle (`hero-v1/`).
9. **G9 — governance quality gates unimplemented.** quality_gates.toml `enforced=false`; coverage/flake/runtime measurements not implemented; stale header comment; 5 check.sh profiles refuse (fuzz-smoke/matrix/reproducibility/package/sign).

### 1.3 Blockers

- The broken extension (G1) blocks every consumer of the Python shell from running without a manual workaround — it sits directly under M2 and the conformance lab.
- The conversion-gate debt (G4) formally blocks lawful decomposition of broad workstream beads into bounded agent-executable tasks.
- The campaign (G8) is the repo's own declared precondition for broad surface expansion (plan §47: no broad API expansion before the hero pipeline validates the architecture).

1. **G1 — the Python shell ships no importable native extension.** The repo tracks zero `.so` files (`git ls-files python/` shows only `.py`); a fresh clone fails `import sympy` until a build produces `fsym_python`. The working tree's only build output is worse than absent: `python/_native.abi3.so` is a 403 MB *debug* cdylib under a filename CPython's finder cannot resolve, and a 0-byte `python/fsym_python.so` stub poisons extension search locally (untracked debris, not VCS content). pyproject/maturin declare the correct module name; nothing enforces that the built artifact lands under it.

**No.** The 25 live beads cover every workstream at title level (good NO_BEAD hygiene from the 2026-08-29 check — no vision goal is entirely untracked), but (a) they are broad records that legally cannot be decomposed/executed until the conversion template is satisfied (G4), and (b) at least three concrete defects found by this audit are covered by no bead: the shipped-extension packaging break (G1), the live ConstitutiveLawZero drift + stale/orphan goldens (G2+G3), and the template-amendment work itself. Completing today's beads as-written would still leave `implemented_uncertified` — which is correct behavior, not a failure: closure requires gate bundles, not code alone.

### 1.5 Vision goals with zero bead coverage (NO_BEAD)

- G1 extension packaging fix — no bead (new bead `fra-native-ext-import-fix`).
- G2+G3 drift closure + golden refresh/orphan quarantine — no bead (new bead `fra-shell-drift-zero-collapse`).
- G4 record-template amendment — no bead (new bead `fra-beads-template-amendment`).
- Differential corpus expansion beyond the seed set is implied by fra-ws01 but unbonded; bounded inside the new corpus bead (`fra-conformance-corpus-200`).

---

## 2. Vision checklist

Status legend: WORKING (code + tests + live run) / PARTIAL / STUB / UNPROVEN (code exists, gates absent) / NOT_STARTED / REGRESSED.

| # | Goal (source) | Status | Evidence |
|---|---|---|---|
| 1 | M0 planning substrate, registries, claim linter, work graph (plan §39) | WORKING | `check.sh all` runs 5 registry validators; fails closed on release readiness with named blockers |
| 2 | Typed IDs, canonical encoding, exact arithmetic (M1, WS02/03) | WORKING (slice) | fsym-id 15 tests incl. trybuild cross-kind rejection; fsym-bigint 89 tests (Toom-3/Karatsuba/NTT-CRT, governed lanes); rational 46; modular 56 |
| 3 | Content-addressed TermDAG, stable TermId (SEMANTIC-001) | UNPROVEN | fsym-core/src/dag.rs blake3 preimages, 20 tests; cross-architecture identity fixture gates open |
| 4 | Three-graph separation SOG/STD/DEG (SEMANTIC-002) | PARTIAL | three layers exist as separate code; vertical-slice gate not executed |
| 5 | Proof kernel + evidence promotion (MATH-001) | PARTIAL | kernel derivation replay + mutant tests in fsym-proof-kernel/evidence; per-family certificate gates open |
| 6 | Proof-carrying factorization claims (MATH-002) | PARTIAL | square-free/Bezout certs + mutation kills; irreducible factorization absent |
| 7 | Certified numeric enclosures (MATH-003) | PARTIAL | RealBall substrate; directed-rounding + claim-specific verification gates open |
| 8 | Verified Jacobian/compilation (MATH-004) | PARTIAL | verified diff proofs + CompiledResidualSystem; verified numeric-program consumption gates open |
| 9 | Region-owned cancellation, no orphans (RUNTIME-001) | UNPROVEN | portfolio/Cx implemented + tested; cancellation-matrix gate bundle open |
| 10 | Two-phase verified publication (RUNTIME-002) | UNPROVEN | evidence lattice claim-binding + tests; publication gate bundle open |
| 11 | Deterministic replay (RUNTIME-003) | UNPROVEN | replay hash-chain + typed checkpoints + tests; scoped terminal-contract gates open |
| 12 | Memory-safe native core, no FFI (SECURITY-001/002) | PARTIAL | workspace `unsafe_code=forbid`, minimal deps, bounded parser/printers; fuzz/adversarial gates open |
| 13 | Python object-model slice (M2, COMPAT-002/003) | PARTIAL | real dual-lane shell, 25 surface tests, live oracle run 13/14; broken shipped extension (G1), 1 drift (G2), alias/module-path infidelities (G5) |
| 14 | Immutable profile sympy-1.14.0-cpython conformance (WS01, C1) | PARTIAL | profile TOML + two oracle lanes + goldens; corpus=14, goldens stale, 5 check.sh lab profiles refuse |
| 15 | Polynomial representations (WS08) | PARTIAL | dense univariate Q[x] + sparse multivariate + Groebner + PIT; no factorization into irreducibles, no modular algorithms, no series truncation |
| 16 | GCD/factorization portfolio (WS09) | PARTIAL | Euclidean/Bézout/square-free/Groebner certs; no finite-field factorization/Hensel recombination |
| 17 | Exact linear algebra (WS10) | PARTIAL | det/inverse/RREF/nullspace/LU/QR/LDL certs, exact least squares; det≤8×8, eigenvalues deg≤2, no eigenvectors/SVD |
| 18 | Certified numerics + algebraic numbers (WS11) | PARTIAL | RealBall + AlgebraicNumber via Sturm; no complex balls, no arbitrary-precision evalf |
| 19 | Differentiation + compilation (WS12) | PARTIAL | 15-function elementary diff set with proofs; sparse Jacobian/Hessian coloring, CSE, SIMD plans absent |
| 20 | Portfolios/cancellation/replay (WS13) | PARTIAL | racing + claim-binding + checkpoint/replay/workspace in fsym-runtime; lab crash/cancel matrices unexecuted |
| 21 | Agent protocol (WS14, AGENT-001/002) | PARTIAL | NDJSON protocol + workspace fork/merge in runtime crate; protocol conformance gate C10 unexecuted |
| 22 | Persistence + RaptorQ repair (WS15, DURABILITY-001) | UNPROVEN | repair sidecar + ledger in runtime crate; end-to-end encode/loss/decode/digest/schema/resume bundle absent |
| 23 | Remote workers + graph index (WS16, DISTRIBUTION-001) | UNPROVEN | remote candidate verification + graph index w/ cycle detection implemented; untrusted-worker gate bundle absent |
| 24 | Gröbner/ideals (WS17) | PARTIAL | Buchberger + ideal membership + elimination certs; no F4/F5/FGLM, no order-conversion certs |
| 25 | Integration/limits/series/transforms (WS18) | PARTIAL | rule-based integration + degree-analysis limits + order-capped Taylor; no Risch, no 0/0 resolution, no transforms |
| 26 | Solvers/sets/logic/ODE (WS19) | PARTIAL | linear/quadratic/rational-root solve, 3 ODE families w/ residual verifiers, DPLL SAT, 3-valued sets; no solveset/transcendental/general systems |
| 27 | Structured domains (WS20) | PARTIAL | geometry/tensor/statistics-free slices real but narrow |
| 28 | Compatibility/ecosystem closure (WS21, M6) | NOT_STARTED | inventory.json is inventory-only; no ecosystem corpus |
| 29 | Performance program (WS22, PERF-001) | NOT_STARTED (evidence) | paired-bench code exists; zero admitted live-incumbent runs |
| 30 | Packaging/release/1.0 (WS23, M8, COMPAT-001) | NOT_STARTED | release gate refuses by design; shipped extension broken (G1); drop-in distribution unbuilt |
| 31 | Certified Jacobian Pipeline C0–C11 (plan §40) | NOT_STARTED (as bundle) | partial substrates for most stages; no hero-v1/ artifact bundle |
| 32 | Monitoring (MONITOR-001) | NOT_STARTED | registry + docs only |
| 33 | Wasm subset (PLATFORM-001) | NOT_STARTED | no wasm target |
| 34 | Franken-suite integrations (INTEGRATION-001/002/003) | NOT_STARTED | only asupersync is a live dependency |
| 35 | Claims governance honesty (constitution Art. XXIII) | WORKING | claims.toml statuses match code reality found by this audit; present-tense discipline enforced by linter policy |

---

## 3. Gap analysis (by category)

- **Implementation gaps** (bead exists, code incomplete): WS05 shell infidelities (module-path stub files, matrix alias semantics), WS08/09 irreducible factorization, WS10 eigenvalue ceiling caused by matrices→solvers inversion, WS18 limit/integration breadth, WS19 solver surface.
- **Proof gaps** (code exists, no gate bundle): all six `implemented_uncertified` families (G6); campaign gates C2–C10.
- **Integration gaps** (parts work, end-to-end missing): C11 hero bundle (G8); persistence/repair chain (G6/DURABILITY-001); remote worker rejection path.
- **Performance gaps**: none measurable — no admitted benchmark exists yet (G7).
- **Vision gaps** (no bead): G1, G2/G3, G4 (now covered by the three new beads below).
- **Design gaps**: matrices→solvers layering (eigenvalues capped by solver capability, not matrix code) — not a violation but a capability coupling to revisit in WS10; eight fake-module-path stub files vs profile dimension "exact module paths".

---

## 4. Bridge plan v1 (sequenced per plan §49 and the campaign-first law)

1. **Give the Python shell a real, declared extension build (G1)** — wire the declared maturin path so a build reliably emits `fsym_python.<abi3>.so` under the importable name, add a packaging-consistency check.sh profile that refuses when the built artifact name disagrees with pyproject `module-name`, purge the local untracked debris (misnamed 403 MB debug cdylib, 0-byte stub — untracked, so no VCS deletion involved), and make `python/tests/test_surface.py` exercise the real extension. Acceptance: `.venv-conformance/bin/python -c "import sympy; print(sympy.diff('x**3','x'))"` works with no `FSYM_PYTHON_EXT_DIR` workaround.
2. **Close the live drift (G2) and refresh evidence (G3)** — make the custom-subclass eval lane preserve the applied form exactly when the oracle does (or record a profile-classified comparator difference — never weaken the comparator); re-capture goldens on the current kernel; list the 3 orphan `.ndjson.ndjson` files for owner-authorized removal (no deletion without explicit permission).
3. **Amend bead records to the conversion template (G4)** — bring all 25 live records to the 14-field template; then decompose the campaign-critical beads (WS01/WS05 slices first) into bounded tasks with acceptance commands.
4. **Expand the differential corpus** — 14 → ≥200 fixtures across the 16 capture modes (held forms, custom subclasses, printers, pickle, warnings, assumptions), generated + hand-written, wired into `check.sh lab`; every fixture names its comparator before execution.
5. **Execute campaign stages C2→C7 with gate artifacts** (native foundation → object-model slice → proof/rewrite nucleus → polynomial/factorization portfolio → cancellation/replay → differentiation/Jacobian), closing the corresponding `implemented_uncertified` gate bundles as each lands.
6. **C8 certified numerics → C9 persistence/repair → C10 agent protocol → C11 hero bundle.**
7. **Only after C11**: broaden surface work (WS17–WS20) and admit the first parity-gated paired benchmark (WS22) with a live incumbent.

---

## 5. Ambition rounds (revised in-place)

### Round 1 — "decent start but MUCH better" (2026-09-03)

The v1 bridge plan is correct but under-ambitious in three ways:

1. **Gate-bundle authorship should be pull-forward, not follow-on.** Waiting for C2–C7 before writing any gate infrastructure repeats the "implementation first, evidence later" drift the constitution forbids. Revision: every stage below C7 must land **with** its gate artifacts and mutation corpora in the same bead, and the campaign runner (`cargo xtask` contracts) must be implemented early enough that C1's oracle-isolation gate can run as code, not prose. Concretely: add a bounded bead for the `xtask`-equivalent gate runner (the docs name `cargo xtask` contracts that do not exist yet and must not be reported as runnable).
2. **The extension fix must be structural, not cosmetic.** Revision: G1's bead must make the artifact name derive from the build (pyproject/maturin `module-name`), add a packaging smoke test that imports the built wheel/symlink-free path, and a check.sh profile that refuses when the checked-in artifact and the declared module name disagree — turning this class of silent breakage into a gate.
3. **Corpus expansion should be grammar-driven, not hand-listed.** Revision: the ≥200-fixture bead must include a deterministic generator (seeded, profile-versioned) plus metamorphic pairs, not 200 hand-written files; hand-written fixtures are reserved for adversarial/custom-subclass cases that the grammar cannot express.

### Round 2 — sustained escalation (2026-09-03)

Deeper revisions:

4. **Sequence eigenvalue capability fix inside C7, not WS10-later.** The hero pipeline needs a sparse Jacobian; the current deg≤2 eigenvalue ceiling and the matrices→solvers inversion mean matrix-code capability is capped by solver capability. Revision: C7's differentiation/Jacobian stage explicitly includes wiring univariate root-finding (the existing Sturm/isolation machinery in fsym-core) into eigenvalue production for charpoly of any degree with certificate, killing the accidental layering cap.
5. **Make the oracle lanes share one verdict taxonomy.** Two differential lanes exist (Python lab + Rust fsym-conformance). Revision: one NDJSON verdict schema + one discrepancy ledger ID space across both, so drift counts are comparable and the mutation/self-test gate covers both lanes; add the schema to the corpus bead's acceptance.
6. **Profile-classify the shell infidelities now.** The eight stub module files and the matrix aliases are either (a) profile-visible defects to fix or (b) documented out-of-scope classifications in the immutable profile. Either is lawful; silence is not. Revision: the WS05 slice bead must end with every one of these infidelities either fixed or ledgered as profile-classified exclusions with source evidence.

### Round 3 — domain-specific depth (2026-09-03)

7. **Apply the repo's own mutation discipline to the harness itself.** Proof-kernel law (Art. VIII.5, plan §14.6) requires registered weakening mutants and kills for every verifier family — and the comparator/diff pipeline IS a verifier of compatibility claims. Revision: the corpus bead's adversarial obligations upgrade from one planted-mismatch self-test to a **registered mutant corpus** with named IDs (`treat-expected-refusal-as-pass`, `drop-type-drift-class`, `loosen-pickle-byte-compare`, `accept-oracle-version-mismatch`…), each killed by a negative test, so harness weakening becomes structurally detectable.
8. **The zero-collapse fix belongs in the hook-override check, not a special case.** Art. IV.4 makes exact-class/hook checks the gate for behavior; Art. IX.4 makes canonicalization operator/domain-specific — a custom application is not a canonical form. Revision: the drift bead's deliverable routes through the shell's generic hook-override checks (fold only when the class's own `eval` provably folds under the oracle's semantics), never an `isinstance`-style carve-out for `ConstitutiveLawZero`.
9. **Eigenvalue generalization is a certificate-family gate, not a capability patch.** fsym-core already owns Sturm sign sequences, isolating-interval bisection, and the AlgebraicNumber certificate path. Revision: C7's eigenvalue work (round-2 item 4) must emit **root certificates** through that existing path (separation bounds + isolating intervals per distinct root), so the C7 gate registers a new certificate family with its own mutants — capability expansion and evidence discipline land together, as the constitution demands.

---

## 6. Beads created by this reality check (template-complete)

Created via `br` (see §7 for IDs and bv validation). Each record carries the full WORKSTREAM_GRAPH.md §32 field set. Legacy record amendment is itself bead `fra-beads-template-amendment`.

| Bead | Title | Pri | Depends on |
|---|---|---|---|
| `fra-native-ext-import-fix-mi5` | Fix fsym_python extension packaging so plain `import sympy` works | P1 | — |
| `fra-shell-drift-zero-collapse-kvg` | Close ConstitutiveLawZero zero-collapse drift; refresh goldens; quarantine orphans (listed, not deleted) | P1 | ext-import-fix |
| `fra-beads-template-amendment-092` | Amend all 25 live bead records to the §32 14-field template; `br lint` clean | P1 | — |
| `fra-conformance-corpus-200-b75` | Grammar-driven corpus ≥200 fixtures + one shared verdict schema + registered harness mutants | P2 | drift-zero-collapse |
| `fra-gate-runner-xtask-cyx` | Real `cargo xtask` gate runner for C1–C3 with independent receipt validator | P2 | template-amendment |

Revision 9 (round 3) is recorded as comments on the existing parents `fra-ws12-diff-compilation-y05` and `fra-campaign-jacobian-bundle-ukp` rather than new beads — it deepens work those beads already own.

Lint note: `br lint` passed on `fra-beads-template-amendment-092` for a accidental reason — its description *mentions* the literal string `## Acceptance Criteria`. The lint check is substring-based; recorded here as a harness observation, not a workaround (all five beads carry a genuine trailing `## Acceptance Criteria` section).

---

### Round 1 (2026-09-03) — found and fixed

- **Factual error in G1 (major):** the misnamed cdylib and 0-byte stub are UNTRACKED local debris; `git ls-files python/` tracks zero `.so` files. Corrected §1.2 G1, bridge item 1, and bead `fra-native-ext-import-fix-mi5` (objective reframed: declared build path + packaging-consistency gate, not artifact replacement).
- Acceptance commands in `mi5` made path-explicit (`PYTHONPATH=python`) so they run without lab harness context.
- `cyx` validator independence tightened: validator is a structurally separate binary, not a mode of the runner.

### Round 2 (2026-09-03) — found and fixed

- Revision 9 correctly belongs to existing parents: recorded as comments on `fra-ws12-diff-compilation-y05` and `fra-campaign-jacobian-bundle-ukp` instead of duplicate beads.
- `br lint` weakness discovered: it substring-matches `## Acceptance Criteria`, so bead `fra-beads-template-amendment-092` passed lint because its prose *mentions* the string. All five new beads carry a genuine section; the substring behavior is recorded as a harness observation for the template-amendment bead to consider hardening.
- Beads JSONL re-exported via `br sync --flush-only` (AGENTS.md §12) — `.beads/issues.jsonl` coherent with the DB.

### Round 3 (2026-09-03) — found and fixed

- Round-3 ambition items 7 and 8 folded into bead records (`b75` registered harness mutants; `kvg` hook-override routing) — descriptions updated, lint re-run clean, cycles still empty.
- No further gaps found this round; refinement stops here per the convergence rule.

Final bead state: 5 new beads, all `br lint`-clean, dependency chain ext-fix → drift → corpus and template → gate-runner, `br dep cycles` empty, `bv --robot-triage` coherent (top pick: `fra-native-ext-import-fix-mi5`).


---

## 8. bv validation

`bv --robot-triage` (2026-09-04T03:05:49Z, data_hash 96fa0b7c6595b761): 47 issues, open 24, actionable 6, dependency-blocked (not_actionable) 23, blocked-status 0; PageRank + betweenness computed, phase2_ready true. Top picks: `fra-native-ext-import-fix-mi5` (0.156), `fra-functions-oracle-test-blanket-17m` (0.088), `fra-gate-runner-xtask-cyx` (0.079). `bv --robot-next` selects `fra-native-ext-import-fix-mi5` — "Unblocks 1 item(s): fra-shell-drift-zero-collapse-kvg; currently unclaimed". Graph coherent; the new dependency chain is respected by the scheduler (drift/corpus correctly not top-picked while blocked).
