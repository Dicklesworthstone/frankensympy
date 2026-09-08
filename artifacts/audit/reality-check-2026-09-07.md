# FrankenSymPy reality check — revised 2026-09-08

This revision supersedes the 2026-09-07 assessment in this same file. The earlier assessment is preserved in Git history; its statements that packaging and campaign closure were resolved and receipts were “tamper-proof” are not supported by the stronger checks below.

## Verdict

**Substantial implementation, but not yet an architecture-proven symbolic product.** The README's pre-certification warning is appropriate. The executable work graph was not: all 64 tasks were closed at audit start while all workstream closure gates and every compatibility certification remained open. Passing component suites is real progress; it does not establish the connected Certified Jacobian Pipeline.

The most urgent blocker is evidence reliability, before additional API breadth:

1. A crashed corpus child with valid but empty JSON can make its wrapper succeed.
2. The 230-fixture admission result checks construction, not the complete Python surface.
3. Campaign properties substitute unrelated component tests, constants, and historical artifacts for a connected live workload.
4. Receipt hashing checks internal consistency, not required-check completeness or source/artifact authority.
5. Key public Python operations still stringify and reparse native expressions.

**No mathematical, compatibility, safety, durability, integration, or performance claim is promoted by this audit. No production implementation was fixed.**

## Scope, provenance, and limits

- Fully read repository AGENTS.md and README.md, the comprehensive plan, constitution, source audit, workstream/claims registries, campaign contract and relevant subsystem contracts. Read architecture revisions and examined portable verifier/artifact, workspace, packaging, monitoring, runtime and dependency requirements. This is a repository-wide evidence assessment, not a claim that every source line or every ancillary document received an exhaustive independent review.
- Starting main: `355014e4e65b0e03b7dd588c28b4e9e2395b6974`, independently observed on the remote.
- The starting workspace already contained edits to Cargo.lock, calculus/core Rust, Python exports/tests, and new Euler/finite-difference/transform modules. They were preserved. During the audit another participant committed them as `626bf01`, then committed shared Beads state as `47e71c9fb30f4ce179d95d76d6f73b8b020f1440`. Earlier tests are evidence for their actual working snapshot, not retroactively clean same-commit release evidence.
- Python tests and conformance used the existing `python/fsym_python.so`; this audit did not rebuild it. At final inspection its SHA-256 was `93e9e7e9ab5fc9d70789ad57ef3302af1a517f08b8a88ea9bd5a893d3614922e`. A passing import does not bind that binary to current Rust source.
- The isolated upstream interpreter reported SymPy 1.14.0. The configured source pin is `16fa855354eb7bcabd3fe10993841e03b1382692`.
- Diagnostic full-surface counts and fixture IDs are retained in [the observation summary](reality-check-2026-09-08-observations.json). It is audit evidence, not a golden refresh or certification receipt.
- No competitive benchmark, real consumer adapter, package matrix, cross-platform run, fresh-process campaign recovery, or formal checker was executed. Source-level findings are distinguished from live probes.

## What actually works

There is real safe-Rust implementation here, not merely renamed modules:

- Typed IDs/budgets/outcomes, exact integers/rationals/modular arithmetic, a semantic term DAG, assumptions infrastructure, proof-kernel/evidence components.
- Bounded polynomial arithmetic, GCD/square-free/rational-root factorization, exact linear algebra, differentiation, sparse Jacobian machinery, rule-based calculus, solvers and structured-domain slices.
- Actual asupersync-based portfolio/cancellation and verify-before-publication machinery, with component tests.
- A Python shell with ordinary Python classes and supported held/custom behavior in selected paths; an isolated upstream capture laboratory.
- Structural registry validators, discrepancy infrastructure and extensive Rust/Python tests.

Live examples included `diff(x**3, x) = 3*x**2`, determinant `-2` for `[[1,2],[3,4]]`, and retained `(x,x)` arguments for held `Add(x,x,evaluate=False)`.

Important ceilings were also observed: `factor(x**4 + 4)` remained unfactored while the isolated oracle returned two quadratic factors; `solve(x**4 + 1, x)` refused the unsupported nonlinear degree; `N(exp(1),50)` explicitly refused unsupported precision. These are useful concrete limits, not evidence that all algebra or numerics are broken.

## High-confidence findings

### F1 — Corpus wrapper fails open (P0, live fault injection)

[corpus_gate.py](../../tools/conformance-lab/corpus_gate.py) parses stdout without checking the child exit status, then iterates `report.get("details", [])`. A monkeypatched child response `CompletedProcess(..., returncode=1, stdout="{}", stderr="candidate crashed")` caused `main()` to return **0**, with admitted=null, drift_total=0 and unledgered=0.

This was an in-memory fault-injection experiment, not a fabricated account of an actual crashed upstream run. The production files and ledger were not changed by the probe.

Reproduction from repository root:

```python
import importlib.util, subprocess, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "audit_corpus_gate", Path("tools/conformance-lab/corpus_gate.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(
    a, 1, stdout="{}", stderr="candidate crashed")
sys.argv = ["corpus_gate.py"]
print("gate_exit", module.main())  # observed: 0
```

Separately, this wrapper intentionally permits ledgered-open drift and appends new discrepancies so a subsequent run can pass. That is a development “no new surprises” policy, not zero-drift certification. The policy must never be interpreted as full parity.

### F2 — 230/230 admission is not full compatibility (P1, live comparison)

[capture.py:cmd_diff](../../tools/conformance-lab/capture.py) explicitly selects `construction_only`: returned type, module, args_repr and func, or raised exception identity. Its own self-tests require it to tolerate MRO-, printer- and pickle-only differences.

The normal differential command admitted **230/230**. A separate diagnostic using the already registered `exact_surface` comparator found:

| Observation | Fixtures differing |
|---|---:|
| At least one non-environment observation | **229 / 230** |
| MRO | 191 |
| Pickle protocol 4 SHA-256 | 191 |
| Pickle protocol 5 SHA-256 | 191 |
| LaTeX printer | 188 |
| Python-hash observation | 138 |
| ASCII pretty printer | 60 |
| srepr | 48 |
| Exception message head | 38 |
| str / repr | 25 each |

The unfiltered comparator differed on 230/230 because environment fields also differed. The 229 count excludes those environment differences; it does not redefine the registered comparator. Multiple fields can differ on one fixture.

Example: LaTeX for integer zero was `0` in the oracle and `$0$` in the candidate. MRO and serialized representations also differed. These results do **not** mean “0% compatible,” nor do they invalidate the narrow construction result. They show why that result cannot justify the README's entire object-model promise.

The exclusion ledger additionally rationalizes Python hash differences through native content hashing. Python hash and TermId are distinct contracts; an exclusion entry cannot amend the immutable public profile. Environment normalization and any legitimate comparator/profile changes require explicit review, not changing expected outputs after a mismatch.

### F3 — The campaign is not a connected architecture proof (P0/P1, source inspection)

[tools/campaign/harness.py](../../tools/campaign/harness.py) runs components and constructs assertions around them:

| Claimed property | What the inspected path actually does |
|---|---|
| Held/custom lowering | Calls the deterministic-term-identity gate, not one mixed Python lower/differentiate/lift path |
| Sparse workload dimensions | Writes fixed residual/variable/nonzero counts |
| Two-strategy factorization | Runs a runtime test racing reflexive `x=x` proofs |
| FrankenNumPy/FrankenSciPy execution | Runs a two-variable native finite-difference diagnostic and writes those consumer names |
| Zero controlled orphan work | Writes a zero constant |
| Fresh-process replay | Uses repeated literal event bytes in a same-process test |
| Same-invocation incumbent | Loads the lexicographically latest historical paired report and writes `incumbent_same_invocation=True` |
| Cross-process terminal digest | Hashes a fixed semantic description and writes `reproducible_across_processes=True` |

The runtime test and calculus diagnostic are legitimate **component** tests. The calculus test explicitly says finite-difference agreement is not mathematical verification. Neither proves the stronger campaign property.

The actual campaign contract requires generated variants of one nonlinear workload containing exact polynomial factors, transcendental built-ins, held/custom Python behavior, assumptions and a mutable snapshot. That same workload must traverse all twelve properties, including real consumers, independent verification, checkpoint loss/repair, fresh-process resume, semantic patch merge and live paired measurement.

I did not rerun the current campaign harness to overwrite its artifacts with more unsupported passes.

### F4 — Receipt integrity is weaker than receipt authority (P0, source inspection)

[gate_receipt_validator.rs](../../xtask/src/bin/gate_receipt_validator.rs) checks schema, a known gate name, status consistency and a digest of caller-provided check rows. It requires only a nonempty commit string. It does not bind an externally expected source tree, lockfile, binary, profile/gate relationship, required check set or produced artifact closure.

Changing metadata outside the hashed check rows is not detected by that digest; replacing rows and recomputing their hash is not independent evidence that the named gate ran. A structurally valid failed receipt also validates with exit 0: consumers must distinguish valid receipt syntax from a passed gate.

This finding concerns **operational gate authority**, not a demonstrated mathematical proof-kernel acceptance bug. No forged-receipt executable probe was run in this audit.

### F5 — The dual-lane boundary remains incomplete (P1, source inspection)

[python/sympy/core/__init__.py](../../python/sympy/core/__init__.py) routes important `diff`, `expand` and `simplify` paths through stringification, native string parsing, and parsed textual results. Symbol routing also uses printed keys.

The project does have native Expr handles and a term DAG; it is inaccurate to say everything is strings. The issue is that these popular shell operations do not yet demonstrate the promised versioned, context-bound, receipt-producing lowering/lifting path preserving held/custom identity.

### F6 — Adapter, packaging and numerical product gaps (P1/P2)

- [ledger.rs](../../crates/fsym-runtime/src/ledger.rs) explicitly describes a bounded **in-memory** hash-chain model, not durable I/O, transactions or crash recovery. Repair/checkpoint component tests are not a demonstrated FrankenSQLite-backed killed/repaired/resumed workload.
- Optional graph and numeric target names do not establish real FrankenGraphDB/FrankenNumPy/FrankenSciPy adapters.
- The current root Rust library exposes version/status constants rather than the intended usable native facade; no root CLI entrypoint delivers the described product workflow.
- The current Python environment resolves the preview `sympy` tree, not an independently importable `frankensympy` package. Building/copying a local extension does not demonstrate coexistence, wheel metadata, resolver satisfaction, exclusive path ownership, upgrade/uninstall or rollback.
- `pyproject.toml` metadata/layout needs reconciliation with the contained bridge and repository license. The configured compatibility manifest still leaves certification inputs unset.
- RealBall and algebraic isolation are useful bounded implementations; general certified transcendental/complex numeric support and actual consumer validation remain open.
- Optional formal projection, minimal portable-consumer/Wasm closure, full semantic-workspace transaction witnesses, and live registered operational monitoring need their exact gates. No absence-of-all-code claim is inferred merely from an absent gate.

### F7 — Performance evidence is not admissible as a win (P1)

A real paired benchmark tool exists, but the campaign can reuse historical data. The inspected performance admission is string-oriented, and engine identity needs explicit attestation. The A/A error path in `tools/perf/paired_bench.py` contains Python `false`, which would raise NameError if reached.

Historical timings are not fresh same-invocation evidence. Neither one favorable case nor the constant `(42,10)` / `(42,25)` runtime fixture proves competitive performance. No new speedup claim is made.

### F8 — Push workflow conflicts with owner policy (P0, source inspection)

[branch-topology.yml](../../.github/workflows/branch-topology.yml) runs on main push, force-updates a legacy branch and deletes other remote branches. This conflicts with the repository's explicit main-only, no-destructive-cleanup instructions.

The audit does not execute or repair that automation without direction. Audit-only commit messages use `[skip ci]` to avoid its push trigger; this is not a CI pass. Canonical local gates remain authoritative. The conflict has an explicit open task.

## Vision checklist

“Partial” means genuine bounded code, not end-to-end certification. “Unproven” means the promised boundary or gate was not established. Only the bounded checks listed above are reported as working.

| Goal / workstream | Reality | Required bridge |
|---|---|---|
| Governance, immutable profiles, claims (WS00–01) | Structural checks pass; gate authority and closure status conflict | Fail-closed corpus, source-bound receipts, exact requirement inventory |
| Typed IDs/budgets (WS02) | Real component substrate | Adversarial boundary, resource and cross-platform gates for each consumer |
| Owned exact arithmetic (WS03) | Real facade/reference and opt-in candidate kernels; no production performance win | Ownership/admission and full-operation strategy gates |
| Terms/domains/assumptions/binders (WS04) | DAG and contexts exist; popular shell paths still stringify | Typed lowering, held/custom/context/generation preservation |
| Python identity/effects (WS05) | 104 tests pass; full observations differ | Surface repair plus exactly-once supervised effect matrix |
| Proof/evidence (WS06) | Real kernel/evidence components | Exact claim family mutants, minimal external capsule/Wasm verifier, optional formal projection |
| Rewrite/simplification (WS07) | Bounded implementations | Conditional/context/branch proof gates; no heuristic promotion |
| Polynomial representations (WS08) | Bounded dense/sparse implementations | Remaining representations and exact domain/regime coverage |
| GCD/factorization (WS09) | Square-free/rational-root core | Finite-field/Hensel slice, correct irreducibility claim, two real strategies |
| Exact linear algebra (WS10) | Substantial bounded algorithms | Remaining structured/spectral/completeness claims and certificates |
| Certified numerics (WS11) | RealBall/algebraic isolation, precision ceilings | Rigorous transcendental enclosures, remaining algebraic/complex domains |
| Differentiation/compilation (WS12) | Native derivatives/Jacobian diagnostics | Same-workload proof-producing compilation into actual consumers |
| Portfolios/cancellation/replay (WS13) | Real runtime components | Actual algorithm race, full safe-point/continuation/replay evidence and monitor |
| Protocol/workspaces (WS14) | Library components, not complete product | Runnable facade/NDJSON, semantic witnesses, verified patch/merge/replay |
| Persistence/repair (WS15) | In-memory model and component mechanisms | Durable typed checkpoint, kill/loss/repair/fresh-process resume |
| Remote/index (WS16) | Candidate rejection and projection components | Actual adapter/liveness/authorization/rebuild gates, local verification |
| Gröbner/ideals (WS17) | Bounded implementations | Remaining algorithms, ideal claims, mutants and regime coverage |
| Analytic calculus (WS18) | Rule/table and bounded algorithms | Broader integration/limits/series/transforms with branch/evidence contracts |
| Solvers/sets/logic/ODE/PDE (WS19) | Bounded solvers, explicit refusals | Completeness-aware broader methods and independent certificates |
| Structured mathematics (WS20) | Selected domain slices | Full registered combinatorics/functions/geometry/tensor/stats/units/physics/control scope |
| Compatibility/ecosystem (WS21) | Narrow construction parity, not full closure | Complete immutable observation/reflection/downstream/platform matrix |
| Performance (WS22) | Tooling, no admitted current win | Attested live incumbent and scalar lane, raw paired data including losses |
| Packaging/release (WS23) | Preview import, refusal/stub gates | Real wheel/resolver/platform/quality/reproducibility/signing matrix |
| Optional artifact/formal/graph/numeric fabric | Architecture contracts exceed demonstrated composition | Typed closed capsules, semantic transactions and independently gated adapters |
| Security/privacy/platforms | Safe-Rust design and bounded tests, not universal claim | Dependency closure, hostile decoder/callback limits, tenant privacy and declared Wasm matrix |
| Certified Jacobian campaign | Component receipts overstate integration | One generated workload corpus through all twelve actual stages |

At audit start: 27 claims; only PLAN-001 allowed as a documented present-tense claim. No validated/certified capability. Workstreams: 20 planned, 4 in_progress, none closed; all nine milestones planned. The README's “all 24 planned at closure level” should not be confused with the exact machine status fields.

## Bridge plan and executable work graph

The original open-work answer was **no**: there were zero open tasks, yet the vision was not achieved. The older report's proposed future work had not become an active, acceptance-complete bridge.

This audit reopened eight specifically contradicted aggregate tasks: campaign, WS09, WS12, WS13, WS15, WS16, WS21 and WS23. Historical closure evidence remains visible; added notes explain why it does not satisfy their normative scope.

The new bridge has **37 new tasks**: sixteen implementation/release slices with sixteen separate independent-review tasks, plus workspace, Python effects, optional formal projection, full requirement coverage and workflow-policy tasks. Total after correction: **101 tasks, 45 open, 56 closed**. No implementation task was closed by this audit.

| Scope / priority | Bounded next deliverable | Task / independent review |
|---|---|---|
| WS01 / P0 | Fail closed on crashed, empty, incomplete, or wrong-profile corpus runs | `fra-rc-corpus-us2` + review `fra-rc-corpus-gate-emj` |
| WS00 / P0 | Bind gate receipts to required checks, source closure, and produced artifacts | `fra-rc-receipts-uhu` + review `fra-rc-receipts-gate-ge6` |
| WS05 / P1 | Close frozen core-shell MRO, hash, printer, and pickle observations | `fra-rc-surface-nvv` + review `fra-rc-surface-gate-0eh` |
| WS04 / P1 | Carry typed shell objects through native differentiation and lifting | `fra-rc-lowering-8w3` + review `fra-rc-lowering-gate-ba5` |
| WS06 / P1 | Verify a polynomial claim capsule from a minimal offline consumer | `fra-rc-capsule-39v` + review `fra-rc-capsule-gate-5mq` |
| WS09 / P1 | Factor square-free ZZ polynomials without rational roots with irreducibility evidence | `fra-rc-factor-lt4` + review `fra-rc-factor-gate-w57` |
| WS13 / P1 | Race two real factorization generators under one protected verifier budget | `fra-rc-portfolio-9i7` + review `fra-rc-portfolio-gate-4or` |
| WS11 / P1 | Enclose campaign sin, cos, and exp with explicit error bounds and precision budgets | `fra-rc-numeric-t2y` + review `fra-rc-numeric-gate-82w` |
| WS12 / P1 | Execute one verified residual and Jacobian through real Franken numeric consumers | `fra-rc-adapters-iym` + review `fra-rc-adapters-gate-wp4` |
| WS15 / P1 | Persist and repair a real factorization continuation across fresh processes | `fra-rc-durable-m5e` + review `fra-rc-durable-gate-hpt` |
| WS13 / P2 | Activate one registered compatibility-drift monitor with valid reset and censoring semantics | `fra-rc-monitor-5ja` + review `fra-rc-monitor-gate-p1m` |
| WS16 / P2 | Rebuild a real FrankenGraphDB projection from authoritative workspace artifacts | `fra-rc-graph-zzo` + review `fra-rc-graph-gate-lkg` |
| WS23 / P1 | Build a coexistable frankensympy wheel with real namespace and accurate metadata | `fra-rc-package-mcj` + review `fra-rc-package-gate-gjd` |
| WS14 / P2 | Expose the structured native request path through a usable library and NDJSON command | `fra-rc-cli-u9o` + review `fra-rc-cli-gate-0jr` |
| WS22 / P1 | Measure connected hero workloads with verified engine identity and complete paired evidence | `fra-rc-perf-yub` + review `fra-rc-perf-gate-sth` |
| WS23 / P2 | Implement source-bound release validation and required matrix profiles | `fra-rc-release-3re` + review `fra-rc-release-gate-pal` |
| WS14 / P1 | Verify semantic workspace publication and fresh-process replay on the hero workload | `fra-rc-workspace-b0z` |
| WS05 / P1 | Exercise exactly-once Python callbacks and interpreter ownership across native requests | `fra-rc-effects-dpc` |
| WS06 / P2 | Gate one optional native-first formal factorization projection | `fra-rc-formal-ma3` |
| WS00 / P1 | Reconcile every normative workstream obligation with an active residual plan and gate evidence | `fra-rc-coverage-c20` |
| WS00 / P0 | Reconcile destructive branch workflow with main-only non-destructive owner policy | `fra-rc-workflow-kzu` |

The dependency sequence is deliberate:

1. Repair corpus fail-closed behavior and receipt authority. Resolve the destructive workflow policy separately. Inventory remaining normative obligations without claiming that inventory implements them.
2. Establish full shell observations, typed lowering and an independently embeddable polynomial capsule.
3. Build non-rational-root factorization and a genuine two-generator portfolio; in parallel build rigorous numeric enclosures.
4. Connect real numeric consumers, exactly-once callbacks, semantic workspace transitions and durable repair/resume.
5. Run the same generated hero workload across all stages, then perform semantically admitted live paired measurements.
6. Complete the full profile/ecosystem/platform/package/quality/release gates; expand remaining algorithm families under their existing immutable contracts.

**Would completing these new tasks finish FrankenSymPy? No.** The concrete implementation slices repair the evidence path and first architecture slice. The coverage task must decompose all residual WS00–WS23 requirements into bounded implementation and independent-gate tasks; it cannot close by declaring the remaining CAS breadth out of scope. Full SymPy-facing compatibility and the complete algorithm program remain substantially larger than this first slice. No completion percentage or date is defensible from the present evidence.

### Ambition and refinement record

Three ambition passes changed the plan rather than the success criteria:

1. Replace an API-count/receipt-count assessment with adversarial evidence admission. This produced the corpus and receipt root tasks.
2. Require architecture composition: generated mixed-input corpus, actual consumer execution, actual factor generators, real durable recovery and output-derived fresh-process roots.
3. Add portable trust and semantic collaboration: minimal verifier-complete cut, conservative semantic read/predicate/absence witnesses, rigorous enclosures and optional native-first formal projection. These are concrete mathematical/architectural mechanisms, not promises of automatic speedup.

Five refinement passes were applied:

1. Scope/closure: reopen the eight contradicted aggregate tasks; retain their original full obligations.
2. Acceptance/independence: give each core repair explicit inputs, files, negative tests, resource semantics, claim effects and a separate unassigned reviewer gate. Unassigned review is an open obligation, not claimed independence.
3. Dependency/composition: attach the reopened aggregates to actual repair gates; add missing surface, callback, workspace and campaign prerequisites to release. Avoid reciprocal implementation/review cycles.
4. Completeness/safety: add residual whole-vision inventory, formal projection and destructive-workflow policy ownership; prohibit fixed workload dimensions, historic benchmark substitution and same-array replay.
5. Structural validation: fix the release epic's missing Success Criteria heading; rerun lint, cycle detection, JSONL sync, readiness and bv triage. Final lint found no missing template sections and cycles were empty.

These checks establish a usable next work graph, not mathematical proof that every future implementation detail has been anticipated. The inventory and independent reviews explicitly remain open.

## Commands and observed outcomes

| Command / diagnostic | Actual result and limit |
|---|---|
| `./scripts/check.sh registries` | Exit 0; 108 lab-tooling tests and structural validators passed; five release-blocking obligations still incomplete |
| `rch exec -- cargo test --workspace --locked` | Remote exit 0; workspace unit/integration/doc tests passed, ignored tests remain; no invented aggregate count |
| `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check --workspace --all-targets --locked` | Retry remote exit 0 |
| `cargo fmt --check` | Exit 1: local pinned toolchain lacked cargo; not a source-format verdict |
| `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy --workspace --all-targets --locked -- -D warnings` | Retry remote exit 0, no warnings; earlier resource refusals were not passes |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python .venv-conformance/bin/python -m unittest discover -s python/tests -p 'test_*.py'` | Exit 0; 104 tests, 9.501 seconds; existing native extension |
| `.venv-conformance/bin/python tools/conformance-lab/capture.py diff tools/conformance-lab/profiles/sympy-1.14.0-cpython-r2-corpus.toml --candidate-python .venv-conformance/bin/python` | Exit 0, 230 admitted under construction_only |
| Registered exact_surface diagnostic, same 230 IDs | 229 fixtures had observation differences excluding environment; not a certification invocation |
| Crashed-child/empty-report injection | Corpus wrapper returned 0: defect reproduced |
| `./scripts/check.sh all` | Exit 1 at source-clean precondition in starting dirty workspace; did not reach full release matrix |
| `./scripts/check.sh matrix` | Exit 2, explicitly not implemented |
| `./scripts/check.sh release-readiness` | Usage exit 2: not an exposed CLI profile; not a release-readiness test |
| `br lint --json`, `br dep cycles --json` | Final lint zero warnings; no active cycles |
| `br sync --flush-only`, `bv --robot-triage`, `br ready --json` | 45 open, four actionable roots: corpus, receipts, requirement coverage, workflow policy |

No UBS source scan was claimed: changes made by this audit are report/evidence/work-graph artifacts, not production source. Full release, campaign, fuzz/matrix, wheel/reproducibility/signing, ecosystem, independent formal/Wasm and valid live-performance gates remain unexecuted or blocked.

## Recommended next action

Start `fra-rc-corpus-us2` and `fra-rc-receipts-uhu`. A reliable measurement system is the prerequisite for deciding whether subsequent changes close the intended gap. Continue the actual architecture slice, not another broad API-expansion sweep. Keep the README's explicit pre-certification warning and make every workstream closure earn its named evidence.
