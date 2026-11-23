# FrankenSymPy

**An in-progress memory-safe, proof-carrying, agent-native symbolic mathematics system—and an independently implemented drop-in replacement for named SymPy profiles.**

> [!IMPORTANT]
> **Current status: implementation in progress, pre-certification.** The repository now contains a substantial Rust implementation workspace (exact arithmetic, semantic term DAG, proof kernel, evidence system, polynomial/linear-algebra/calculus engines, portfolio runtime, conformance laboratory) with green workspace tests, a Python compatibility shell slice, and a live SymPy 1.14.0 differential oracle. It is **not** a certified SymPy replacement: no compatibility profile is certified, no performance win is demonstrated, and all `validated`/`certified` gates remain open. Capability status is tracked in [`registries/claims.toml`](registries/claims.toml).

The complete design is in [`COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md`](COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md).

---

## The idea

FrankenSymPy is intended to be much more than “SymPy rewritten in Rust.”

A genuine SymPy replacement has to preserve a surprisingly deep Python contract:

- user-defined subclasses of `Basic`, `Expr`, `Atom`, `Function`, and related classes;
- metaclasses, dynamic undefined-function classes, constructor hooks, and `_eval_*` overrides;
- `evaluate=False` and deliberately unevaluated forms;
- `args`, `func`, reconstruction, traversal, sorting, hashing, and exact class identity;
- assumptions contexts and tri-valued behavior;
- mutable compatibility objects;
- printers, pickles, warnings, exceptions, signatures, and module paths.

An opaque Rust `Expr` exposed through Python bindings cannot preserve all of that.

FrankenSymPy therefore uses a **dual-lane architecture**:

1. A real Python-compatible object shell preserves the selected SymPy profile.
2. Eligible expression regions lower into a deterministic native Rust kernel.
3. Arbitrary Python extensions remain valid as conservative opaque nodes.
4. Lowering and lifting are explicit, versioned, and receipt-producing.
5. Upstream SymPy is an isolated development oracle, never a hidden production fallback in a certified build.

---

## The seven architectural bets

### 1. Dual-lane compatibility

Python identity and extensibility stay in the Python shell. Exact algebra, canonical terms, proof search, certified numerics, and high-performance execution live in the native kernel.

### 2. Three separate graphs

FrankenSymPy does not force incompatible responsibilities into one expression tree:

- **Surface Object Graph:** exactly what the user constructed and Python can observe.
- **Semantic Term DAG:** immutable, typed, domain-aware native terms.
- **Derivation Evidence Graph:** transformations, assumptions, proofs, certificates, decisions, and receipts.

This preserves held forms and custom classes without sacrificing canonical native algebra.

### 3. Domain-explicit exactness

Every semantic operation names its exact domain, assumptions context, branch policy, coercions, and rule universe. Unknown facts remain unknown. Approximation is never smuggled into exact work.

### 4. Proof-carrying algorithm portfolios

Factorization, Gröbner bases, exact linear algebra, simplification, integration, solving, and other operations may race multiple strategies. Their outputs are candidates until a smaller independent verifier accepts the exact typed claim.

### 5. Deterministic resource sovereignty

All controlled work is intended to be owned by [asupersync](https://github.com/Dicklesworthstone/asupersync) regions, governed by multidimensional budgets, published through two-phase effects, cancel-correct, and replayable under declared determinism modes.

### 6. Agent-native symbolic state

Terms, contexts, claims, proofs, counterexamples, semantic patches, branches, checkpoints, and replay bundles receive stable structured identities. Agents do not have to parse pretty-printed strings or treat a chat transcript as mathematical state.

### 7. Recoverable computation fabric

Expensive work can eventually be checkpointed, persisted, distributed, indexed, repaired, and resumed—without allowing storage, workers, graph reachability, or RaptorQ decoding to define mathematical truth.

The non-negotiable rules are codified in [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md).

---

## Intended leapfrog

| Dimension | FrankenSymPy target |
|---|---|
| Python compatibility | Certified against immutable SymPy profiles, including object-model behavior rather than API-name coverage |
| Native representation | Canonical, hash-consed, typed semantic DAG with stable content identity |
| Mathematical evidence | Kernel proofs, independently verified certificates, certified numeric enclosures, explicit heuristic/conditional outcomes |
| Algorithms | Condition-aware portfolios with asymmetric loss, safe baselines, protected verifier budgets, and deterministic replay |
| Resource control | Typed nested budgets for time, memory, expression growth, coefficient height, proof size, callbacks, persistence, and remote work |
| Cancellation | Request → drain → finalize semantics with no controlled orphan work |
| Agent interface | Versioned NDJSON/RPC, semantic patches, proof expansion, counterexample bundles, and proof-aware branches |
| Persistence | Optional computation ledger, verified cache, checkpoints, forensic replay, and crash recovery |
| Repair | Selective RaptorQ protection for valuable artifacts, followed by digest, schema, dependency, and mathematical verification |
| Monitoring | Conformal e-processes for operational drift and regression—not as mathematical proof |
| Numeric bridge | Verified residual/Jacobian/Hessian compilation into FrankenNumPy and FrankenSciPy workflows |
| Platforms | Pure Rust native core, Python shell, CLI/protocol service, and a declared WebAssembly subset |
| Dependencies | Minimal, memory-safe Rust dependency universe; no C/C++ CAS or arbitrary-precision FFI |
| Claims | Machine-readable status and same-commit evidence gates for compatibility, proof, safety, durability, and performance statements |

These are target properties. They become present-tense claims only after the gates in [`registries/claims.toml`](registries/claims.toml) pass.

---

## Compatibility model

The first provisional immutable target is:

```text
sympy-1.14.0-cpython
```

Pinned upstream source:

```text
16fa855354eb7bcabd3fe10993841e03b1382692
```

The project plans two Python distributions:

### `frankensympy`

A coexistable package for native APIs, result/evidence envelopes, explicit budgets, replay, workspaces, preview compatibility surfaces, and Franken-suite integrations.

### `frankensympy-dropin`

A separate distribution that owns the top-level `sympy` package and intentionally conflicts with upstream SymPy in one environment. It will be published as certified only for profiles that pass the entire release matrix.

A certified drop-in artifact must contain no hidden import or runtime fallback to upstream SymPy.

The full compatibility contract is in [`docs/COMPATIBILITY_CONTRACT.md`](docs/COMPATIBILITY_CONTRACT.md).

---

## Evidence is a first-class result

A native FrankenSymPy operation is designed to return one of:

```text
Accepted(value, claim, evidence, receipts)
Conditional(value, unresolved obligations)
HeuristicCandidate(value, diagnostics)
Inconclusive(explored methods, continuation)
Refused(reason)
Cancelled
TimedOut
ResourceExhausted
Unsupported
InternalFault
```

Planned evidence classes include:

- `KernelProved`;
- `CertificateVerified`;
- `ExactCrossChecked`;
- `CertifiedNumeric`;
- `OracleConformant`;
- `UserAsserted`;
- `HeuristicCandidate`.

They are deliberately non-interchangeable.

> RaptorQ restores candidate bytes. Digests establish expected content identity. Schema checks establish well-formedness. Mathematical verifiers establish evidence.

> An e-process may detect a suspicious stream of results, but it cannot prove an identity, certify a factorization, or establish solver completeness.

See [`docs/EVIDENCE_PROOFS_AND_REWRITES.md`](docs/EVIDENCE_PROOFS_AND_REWRITES.md) and [`registries/evidence_classes.toml`](registries/evidence_classes.toml).

---

## Exact roots and integrals through the native session

The root Rust library and `frankensympy` NDJSON command expose immutable
residual-system registration, exact residual/Jacobian box enclosures,
independently checked local root boxes, and portable root-witness export/replay.
The receiver checks its own typed sources, ordered variables and context before
recomputing a transported claim. Verification draws from a protected cumulative
budget, and source parsing retains original domain obligations such as
`0*(1/x)`.

Native hosts can pass their owning request's cancellation check to
`Session::certification_request_with_checkpoint`. Generation, protected
verification, and final publication admission observe the same check. Cancelled
work retains spent resources and resumable candidate-search progress; a fresh
request can resume it on the same ledger. Newly accepted results publish only
after the final successful checkpoint. The [session guide](docs/NATIVE_CERTIFICATION_SESSION.md#host-controlled-cooperative-cancellation)
shows the asupersync context adapter and the synchronous API's scope.

`CertifyIntegral` also computes a proper finite real definite integral from one
registered scalar source and one variable. Exact rational endpoints and a
positive absolute enclosure-width tolerance define the request. Adaptive
partitioning spends ordinary resources; an independent interval checker
uses the protected verifier pool before publishing `certified_numeric` evidence.
Smooth cells use C2 bounds; supported continuous absolute-value compositions
use checked Lipschitz bounds across corners. For example, `abs(x-1)` on `[0,3]`
can produce a certified enclosure of `5/2` without claiming a derivative at its corner.
Reversed limits negate the enclosure, while endpoint poles and unresolved
domains refuse. `ExportIntegralCertificate` and `ReplayIntegralCertificate`
bind the original source, axis, ordered bounds, tolerance and context, and
recompute the enclosure from the partition witness.

Automatic complete root isolation connects the existing exact subdivision search
to an independent coverage check. A session can find every nonsingular root in a
finite box, prove the box is root-free, or return an explicit inconclusive result.
Accepted roots receive individually exportable certificate handles; no initial
guesses or hand-written preconditioners are required.

Root searches can also advance in bounded batches. `BeginRootIsolationSearch`
freezes a problem, `AdvanceRootIsolationSearch` retains completed cells and the
unresolved frontier, and `InspectRootIsolationSearch` reports candidate-only
progress. `FinishRootIsolationSearch` independently verifies the complete
proposal before publishing any accepted roots. The native `RootIsolationSearch`
uses the same engine as one-shot search and preserves cumulative work across
yields and cancellation; `ReleaseRootIsolationSearch` frees a live search slot
without refunding its budget. `ExportRootIsolationSearch` saves a canonical
candidate checkpoint that a client can persist and restore in another process
with `RestoreRootIsolationSearch`. Restoration checks the expected source,
domain, options and context, reconstructs the frontier, and retains prior
intrinsic consumption plus new validation work. Restored progress remains a
candidate until protected complete-cover verification succeeds. Repeated
restores can exhaust the search's 256 MiB cumulative intrinsic allocation cap;
checkpoints do not authenticate a sender's full resource history or enlarge
the receiver's allowance.

An explicit host directory enables durable continuation:
`frankensympy --checkpoint-dir ./root-checkpoints` exposes
`SaveRootIsolationSearch` and `LoadRootIsolationSearch`. Exact content keys select
the saved snapshot, with bounded reads, canonical envelopes, dependency checks
and independent source-bound restoration. Optional
`SaveRepairableRootIsolationSearch` / `LoadRepairableRootIsolationSearch` store
up to 64 source symbols and 16 RaptorQ repair symbols, publishing the manifest
last. Recovery reports missing/corrupt chunks and checks the fully regenerated
manifest before mathematical restoration. Small repair examples require an
explicit allowance such as `--compute-steps 100000000`; the ordinary default
does not fund the codec. Both forms resume candidate state and still require
protected final verification.

`ExportRootIsolation` and `ReplayRootIsolation` transport the complete coverage
proof, including a proved empty root set. A fresh receiving process binds the
artifact to its own equations, typed coordinate order, context and requested
box, then independently verifies every root and exclusion before publishing
the complete result and its child certificate handles.

Root proofs, integral certificates and candidate checkpoints support bounded
chunked transfer up to 1 MiB through
`BeginArtifactExport`, `ReadArtifactChunk`, `BeginArtifactImport` and
`AppendArtifactChunk`. Chunks fit the existing 64 KiB NDJSON frame limit and
support exact retries. Root and integral replay operations ending in
`FromArtifact` check content integrity and run independent protected verification.
`RestoreRootIsolationSearchFromArtifact` checks a candidate prefix on ordinary
resources; its distinct artifact kind cannot authorize an accepted root claim.
`ReleaseArtifactTransfer` frees a transfer slot without resetting spent budgets.

See [Native certification sessions](docs/NATIVE_CERTIFICATION_SESSION.md) for a
complete CLI example, typed Rust APIs, budget controls and precise limits.
This is a bounded development interface; complete campaign and profile gates
remain open. The opt-in local file store does not yet provide a whole-session
snapshot, background repair service, or asynchronous cancellation endpoint.

### Reusable compiled derivative programs

Native sessions can retain independently checked residual or objective programs
for repeated approximate binary64 execution. Residual programs expose both
Jacobian-vector and transpose-Jacobian products. Objective programs expose value,
gradient, sparse Hessian and Hessian-vector evaluation. Compilation and execution
use cumulative budgets; independent admission uses protected verifier resources,
and every evaluation retains original source-domain checks. The
[session guide](docs/NATIVE_CERTIFICATION_SESSION.md) includes complete NDJSON
examples and the distinct numerical, symbolic and interval-proof contracts.

Compiled programs can also be exported and independently replayed in a fresh
session. The receiver registers its own expected source and coordinate order;
source, symbol-identity, context, derivative-proof and bytecode checks precede
publication of a new handle. Both residual and objective programs support
inline and chunked transfer, with cumulative budgets and protected verifier
resources. Numerical values remain approximate binary64. Stored acceptance
flags, cached values and shared execution plans are never imported as authority.

Both locally compiled and restored handles can drive numerical solvers directly
through the Session or NDJSON interface. `SolveCompiledResidualSystem` selects
FrankenNumPy Newton, either FrankenSciPy root backend, or sparse Newton;
`SolveCompiledLeastSquares` selects dense or sparse fitting; and
`MinimizeCompiledObjective` reuses the admitted gradient and Hessian for descent.
These operations spend ordinary cumulative resources and retain every original
source-domain check. Their outputs remain approximate proposals. Least-squares
stationarity can have nonzero residuals, and objective stationarity alone does
not prove a minimum. Backend failures never silently choose another algorithm.

`MinimizeAndCertifyCompiledObjective` connects descent to the independent
original-source stationary checker. The caller supplies exact positive box
radii and a verification precision. A numerical point and inverse Hessian only
propose a witness; protected verification must establish the local claim before
publication. The result reports a strict minimum, strict maximum, saddle, or
unclassified stationary point as justified by the checker. Failed verification
does not widen the requested box or publish a certificate.

### Objective derivatives and local stationary points

The native session also accepts `RegisterObjective`, `EncloseObjective`, and
`CertifyStationaryPoint`. It independently evaluates the original scalar
expression over an exact real box to enclose its value, gradient and Hessian.
A proposed stationary witness is accepted only after interval checking proves
existence and uniqueness in that box. Hessian checks distinguish strict local
minima, strict local maxima, saddles and an explicitly unclassified result.

`ExportStationaryCertificate` and `ReplayStationaryCertificate` transport the
typed source and untrusted witness. The receiver supplies its own expected
objective, coordinate order, box and context and recomputes the claim with
protected verification resources. Stationary certificates support the existing
chunked transfer interface. The [session guide](docs/NATIVE_CERTIFICATION_SESSION.md)
provides a complete two-variable example, Rust APIs and bounds. This local
stationary-point lane makes no global-optimality or complete-enumeration claim.

### Complete parameter families and implicit sensitivities

Native sessions can now register `F(x,p)=0` with separately ordered unknowns
and parameters through `RegisterParametricProblem`. `CertifyParametricRoot`
checks a proposed uniform witness, while `CertifyParametricCover` automatically
subdivides the parameter box and independently verifies every cell. Acceptance
proves exactly one nonsingular unknown root in the same declared unknown box
for **every** parameter in the full declared parameter box.

`EncloseParametricSensitivity` works from the accepted family and independently
recomputes exact interval bounds for `dx_i/dp_j`. Rows follow unknown order;
columns follow parameter order. For `x-p=0` on `x∈[-2,2]`, `p∈[-1,1]`, the
family's root enclosure is `[-1,1]` and its sensitivity is exactly `1`. Coupled
systems and adaptive covers retain the full parameter intervals and original
source domains. Search and checking use separate ordinary/protected allowances
on one cumulative session ledger, with atomic publication of accepted handles.

Rust callers use `ParametricProblem`, checked core family receipts and the
metered sensitivity functions. The [session guide](docs/NATIVE_CERTIFICATION_SESSION.md#certify-a-solution-for-every-parameter-and-bound-its-sensitivity)
contains complete NDJSON examples, including a family that requires adaptive
subdivision. This development interface supports up to sixteen total typed
coordinates and 1024 parameter cells; complete campaign and profile gates
remain open.

## Compiled sparse residuals (development API)

`compile_residual_system` exposes the checked sparse Jacobian and products with
it or its transpose, alongside the existing dense evaluation methods:

```python
from sympy import compile_residual_system, symbols

x, y, z = symbols("x y z")
system = compile_residual_system([x**2 + y, y*z, z**3 - 1], (x, y, z))

assert system.jacobian_pattern == ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))
residuals, entries = system.evaluate_sparse(2, 3, 4)
# residuals: [7.0, 12.0, 63.0]; entries: [4.0, 1.0, 4.0, 3.0, 48.0]

product = system.jacobian_vector_product([1, -2, 0.5], 2, 3, 4)
# [2.0, -6.5, 24.0], with one output per residual.
transposed = system.jacobian_transpose_vector_product([2, 3, -1], 2, 3, 4)
# [8.0, 14.0, -39.0], with one output per variable.
```

The pattern is immutable and remains stable when a structural derivative is
numerically zero. Directions follow the flat `system.variables` order, even
when ordinary arguments are nested; transpose weights follow captured residual
order. These methods avoid dense Jacobian output and retain every original
residual and stored-derivative domain check, including for zero directions or
weights. Rectangular and empty-axis systems preserve their declared shapes.

An optional reusable `EvaluationBudget` accounts for native source snapshots,
compilation, independent derivative-proof and bytecode admission, expression
evaluation, sparse products, and output storage. Pass the same `budget=` to
`compile_numeric`, `compile_residual_system`, or `compile_objective` and later
evaluation calls to keep their costs on one cumulative ledger. Compilation
releases the interpreter during native work and observes cooperative cancellation;
failure publishes no partial program and does not refund consumed work. An
omitted compilation budget uses one bounded private ledger. Python source
capture, bounded bridge structural preflight, and CPython allocation overhead
remain outside native accounting; Python compilation does not reserve a separate
verifier pool. Compiled storage
contains the original residual programs and only the nonzero derivative
programs. Differentiation visits the declared variables occurring in each
residual; independent verification checks every omitted source dependency and
establishes the other zero entries from its own typed-symbol traversal.
Numerical outputs are approximate binary64 values; checked symbolic derivatives
do not certify rounding error or a root. Saved residual programs retain the historical version-one Python envelope and
16 MiB limit. Export and replay now share the caller's cumulative budget;
loading independently binds the source and checks all retained proofs. Raw
canonical residual candidates are also admitted up to 1 MiB. Both forms must
satisfy the current aggregate source, operation and proof-work bounds.

Compiled residual and objective programs can also move between Python and native
Sessions through the shared, context-bound format:

```python
from sympy import (
    EvaluationBudget, compile_objective, compile_residual_system,
    load_portable_objective, load_portable_residual_system, symbols,
)

x, y = symbols("x y")
budget = EvaluationBudget()
source = [x**2 + y, x - y]
program = compile_residual_system(source, (x, y), budget=budget)
saved = program.to_portable_bytes(budget=budget)
restored = load_portable_residual_system(
    saved, residuals=source, args=(x, y), budget=budget,
)
assert restored.evaluate(2, 3, budget=budget) == ([7, -1], [[4, 1], [1, -1]])

objective = x**2 + x*y + 3*y**2
cost = compile_objective(objective, (x, y), budget=budget)
saved_cost = cost.to_portable_bytes(budget=budget)
loaded_cost = load_portable_objective(
    saved_cost, objective=objective, args=(x, y), budget=budget,
)
assert loaded_cost.hessian_vector_product([3, -2], 2, -1, budget=budget) == [4, -9]
```

The same UTF-8 bytes are the artifact payload for native
`ExportCompiledResidualSystem`/`ReplayCompiledResidualSystem` and
`ExportCompiledObjective`/`ReplayCompiledObjective`. Receivers supply the exact
original native source and ordered typed variables; replay checks context,
retained proofs, omitted derivatives and bytecode before publishing a program.
Python uses the default empty ambient context and preserves declared symbol
identities. The format is bounded to 1 MiB; encoding and replay share the supplied
budget with compilation and evaluation. Existing `to_bytes`,
`load_residual_system`, and `load_objective` formats remain available unchanged.
These artifacts carry checked derivative programs; numerical values remain
approximate, and root or stationary-point certificates are separate.

Saved Python programs and root proofs using the historical internal `abs`
spelling remain loadable against the original registered `Abs` expression.
The compatibility adapter first verifies the unchanged artifact against the
receiver's historical source encoding. It then upgrades only the function
tags in that accepted canonical export and independently checks the retained
evidence against the current source. Re-export uses canonical `Abs` tags.
Both program formats and standalone or compiled-program root-proof replay use
this path. Typed coordinates, held operands, ambient context and every other
source spelling remain exact; opaque functions named `abs` acquire no builtin
capability. Each phase retains consumed work and observes the same caller
budget and cancellation. No derivative proof is regenerated during migration.

The public compiler admits up to 256 variables and 4,096 residuals, with
65,536 source nodes, 65,536 stored bytecode operations, and 65,536 units of
row-local differentiation work. Each row contributes its source-node count
times its number of declared source dependencies (at least one). Unused axes
therefore do not multiply compilation work or allocate zero programs. Existing
per-expression, depth, literal, and supported-function limits still apply. Dense
`evaluate`, `jacobian`, and `evaluate_batch` explicitly require at most 65,536
Jacobian positions; larger checked systems use `residuals`, `evaluate_sparse`,
Jacobian products, or the sparse least-squares backend. Dense requests exceeding
that limit refuse before converting numerical inputs or spending their budget.

For square systems, the analytic FrankenSciPy backend connects those checked
derivatives to the pinned nonlinear solver and NumPy linear solves:

```python
root_system = compile_residual_system([x**2 + y**2 - 5, x - y - 1], (x, y))
candidate = root_system.solve_candidate(
    [1.8, 0.8], backend="frankenscipy_analytic",
)
# candidate.root is approximately (2.0, 1.0); candidate.evidence is "approximate".

receipt = root_system.solve_certified(
    [1.8, 0.8], backend="frankenscipy_analytic",
)
# RootBoxReceipt from the independent original-source interval checker.
```

This bounded adapter admits 1 through 16 variables, uses at most 200 iterations,
and requires a final residual infinity norm at most `1e-10`. Its solver controls
are fixed; explicit `tol`, `maxiter` or `max_step_norm` overrides refuse. It checks
original residual and analytic derivative domains during initialization and
accepted updates. Armijo search requires a finite sum of squared residuals;
the bounded linear solve uses a dense matrix assembled from sparse entries.
Domain, linear-solver, budget and cancellation failures retain
their causes. Candidate generation and optional certification share a supplied
`EvaluationBudget`. The existing `frankenscipy` backend selects finite-difference
`fsolve`; `frankenscipy_analytic` selects the analytic nonlinear driver. A floating
candidate alone carries no root-existence, uniqueness or rounding guarantee.

### Sparse nonlinear root solving

An explicit sparse backend extends approximate square-system root searches to
256 variables while retaining the checked sparse Jacobian throughout the solve:

```python
from sympy import compile_residual_system, symbols

variables = symbols("q0:24")
equations = [
    variable**2 - (index % 7 + 1)**2
    for index, variable in enumerate(variables)
]
system = compile_residual_system(equations, variables)
candidate = system.solve_candidate([2] * 24, backend="sparse", tol=1e-12)

# Each positive coordinate approximates index % 7 + 1.
assert candidate.residual_norm <= 1e-12
assert candidate.evidence == "approximate"
```

The native search uses column-scaled LSQR directions, bounded damping and
backtracking. Its numerical buffers scale with stored Jacobian entries and
vector lengths; it does not assemble a dense Jacobian or normal matrix.
Consistent rank-deficient systems are admitted, including unused coordinates.
Success requires a fresh evaluation of every original residual with infinity
norm at most `tol`. A small least-squares gradient, a small step or an exhausted
linear iteration cannot authorize a root candidate. Stationary points with
nonzero residuals therefore refuse instead of reporting convergence.

The route accepts square systems with 1..256 variables, at most 1,024 nonlinear
iterations, six damping strengths per iteration, 512 LSQR steps per strength
and 32 backtracking trials per direction. `max_step_norm` clips the proposed
step before backtracking. Initial source or derivative domain failures refuse;
invalid mathematical trial points can backtrack. Budget, cancellation and
allocation failures stop the search, and all attempts and final publication
share the supplied `EvaluationBudget`.

`nsolve(equations, variables, initial, backend="sparse")` exposes the same
approximate route with the usual scalar, matrix or `dict=True` output. Existing
backend defaults remain in place. Independent root certification still admits
at most 16 variables and grants a receipt only after original-source interval
verification; the larger sparse search does not extend that certificate claim.

### Sparse nonlinear least squares

The same checked residual program can fit rectangular, rank-deficient and
underdetermined systems through an explicitly selected sparse backend:

```python
from sympy import compile_residual_system, symbols

variables = symbols("p0:24")
targets = tuple(index % 7 for index in range(24))
residuals = [
    weight * (variable - target)
    for weight in (1, 2, 3)
    for variable, target in zip(variables, targets)
]
fit = compile_residual_system(residuals, variables)
candidate = fit.solve_least_squares([0] * 24, backend="sparse")

# candidate.point approximates targets; all 72 rows contribute to the cost.
# candidate.cost is half the sum of squared original residuals.
assert candidate.backend == "sparse_lsqr"
assert candidate.evidence == "approximate"
```

The numerical search uses column-scaled, damped Gauss–Newton steps from bounded
LSQR iterations and Armijo backtracking. It operates directly on checked sparse
Jacobian entries; its solve buffers grow with the stored nonzeros and row/column
counts, without assembling a dense Jacobian, QR factorization or normal matrix.
Every original residual and derivative domain is checked again at trial points.
Invalid numerical trials backtrack, while budget, cancellation and allocation
failures stop the operation. One supplied `EvaluationBudget` spans all attempts,
rejected trials and final publication.

The solver admits 1..256 variables and 1..4096 residuals together, subject to
the source, dependency-work and stored-operation bounds above. A 4,096-by-256
system with one nonzero derivative per row stores 4,096 derivative programs;
its 1,048,576 possible Jacobian positions do not become compiled zero programs.
The default `backend="dense"` retains the existing 1..16-variable FrankenNumPy
QR route. Sparse selection does not silently relax admission or switch backends.

Success requires a fresh original cost gradient `J.T @ residuals` whose infinity
norm is at most `gtol` (default `1e-10`). A small step, unchanged cost or exhausted
inner iteration alone cannot succeed. The result reports all residuals, cost,
gradient norm and evaluation counts. It is an approximate stationary candidate,
which can have nonzero residuals or be a maximum or saddle; no minimum, root,
uniqueness or minimum-norm claim is made. Controls are bounded by 1,024 nonlinear
iterations, six damping strengths per iteration, 512 LSQR steps per strength and
32 line-search trials per direction.

### One budget from compilation through saved-program replay

The three Python compilers accept a keyword-only `budget=`. Residual export
and replay also accept that same caller-owned `EvaluationBudget`:

```python
from sympy import (
    EvaluationBudget, compile_residual_system, load_residual_system, symbols,
)

x, y = symbols("x y")
source = [x**2 + y - 5, x - y - 1]
budget = EvaluationBudget(
    max_steps=10_000_000,
    max_memory_bytes=10_000_000,
    max_allocations=1_000_000,
)
program = compile_residual_system(source, (x, y), budget=budget)
saved = program.to_bytes(budget=budget)
restored = load_residual_system(
    saved, residuals=source, args=(x, y), budget=budget,
)
values, jacobian = restored.evaluate(2, 1, budget=budget)
# values == [0.0, 0.0]; jacobian == [[4.0, 1.0], [1.0, -1.0]]
```

Native source copies, generation, independent admission, encoding, bounded
decoding and later evaluation consume one cumulative ledger. Failure retains
spent resources; cancellation is cooperative and no incomplete program is
published. Python source capture, bounded bridge structural preflight and
CPython overhead retain their documented accounting exclusions. Pass the
ledger explicitly to each operation; the immutable program never resets it.

Ordinary residual exports and imports preserve the historical 16 MiB envelope;
existing saved bytes remain stable. The loader also admits raw canonical
residual candidates up to 1 MiB, while ordinary re-export uses the historical
format. Context-bound Session exchange uses the explicit `to_portable_bytes`
and `load_portable_residual_system` APIs above. Resource limits can refuse a
file below its byte cap. Replay requires the receiver's expected exact source
and typed coordinate order, then independently checks all retained proofs and
bytecode. Saved acceptance flags and Python callbacks carry no proof authority.

---

## Bracketed scalar roots (development API)

Scalar `nsolve` has a derivative-free bisection route for a known finite real
bracket. It can handle a starting interval even when Newton iteration encounters
a zero derivative or cycles between guesses:

```python
from sympy import Symbol, nsolve

x = Symbol("x")
root = nsolve(x**3 - 2*x + 2, x, (-2, 0), solver="bisect")
# Approximately -1.7692923542386.
```

Pass a built-in pair of increasing bounds. Both original endpoint residuals
must be finite, and their signs must differ unless an endpoint evaluates to
zero. Every new evaluation stays inside the bracket. The solver returns only
when the numerical residual is zero, or both its magnitude and the enclosing
interval radius meet `tol`. A small interval around a pole is insufficient.
`maxiter` bounds the work; `dict=True` returns the typed coordinate mapping.

With an `EvaluationBudget`, compilation, residual evaluation and bisection work
consume the same cumulative ledger. Cancellation and resource exhaustion stop
the request without switching numerical backends. This scalar route does not
differentiate the source or accept systems; default and explicit
`solver="newton"` calls retain the existing Newton path.

The returned `Float` is an approximate binary64 candidate. Use
`nsolve_certified` when an independently checked local root enclosure is
required. This development route deliberately accepts an exactly evaluated
right-endpoint root, where the pinned SymPy 1.14 bisection wrapper can instead
raise. Its new regression suite is registered for actual-extension CI; a full
compatibility profile is not certified.

## Compiled scalar objectives (development API)

The checked-out Python shell and native extension expose `compile_objective`
for scalar values, gradients, sparse Hessians, and Hessian-vector products:

```python
from sympy import compile_objective, symbols

x, y = symbols("x y")
objective = compile_objective((1 - x)**2 + 100*(y - x**2)**2, (x, y))

value, gradient, hessian = objective.evaluate(-1.2, 1.0)
# Approximately: 24.2, [-215.6, -88.0], [[1330.0, 480.0], [480.0, 200.0]]

pattern = objective.hessian_pattern  # ((0, 0), (0, 1), (1, 0), (1, 1))
value, gradient, entries = objective.evaluate_sparse(-1.2, 1.0)
# entries follow pattern; this call does not allocate a dense Hessian.

product = objective.hessian_vector_product([2.0, -3.0], -1.2, 1.0)
# Approximately [1220.0, 360.0], computed directly from sparse entries.

self_power = compile_objective(x**x, (x,))
value, gradient, hessian = self_power.evaluate(2.0)
```

Both differentiation stages are independently checked, including omitted zeros,
both mixed-partial orders, and the compiled instructions actually evaluated.
Arguments retain typed `Symbol`/`Dummy` identity; directions follow the flat
`objective.variables` order. Nonliteral powers such as `x**x` and `x**y` require
a strictly positive base, even if an exponent happens to be integral at one
point. Original-source and derivative-domain failures, nonfinite arithmetic,
invalid dimensions, and unsupported functions raise errors before publishing
results.

These bounded synchronous programs admit at most 64 variables, 256 source
expression nodes, and source depth 16, with additional derivative/proof-growth
limits. Numerical results are approximate finite-real binary64 values. Formal
derivative checks do not establish interval error bounds, convexity, global
smoothness, or supervised cancellation; the complete campaign and compatibility
gates remain open.

Checked objectives also feed a bounded local minimization consumer:

```python
candidate = objective.minimize([-1.2, 1.0], gtol=1e-10, maxiter=64)
# candidate.point is approximately (1.0, 1.0).
# candidate.value and candidate.gradient are evaluated again at that point.
assert candidate.evidence == "approximate"

# The explicit sparse backend supports up to 64 variables and keeps the
# checked Hessian sparse through every conjugate-gradient linearization.
sparse_candidate = objective.minimize([-1.2, 1.0], backend="sparse")
assert sparse_candidate.backend == "sparse_newton_cg"

receipt = objective.minimize_certified([-1.2, 1.0])
assert receipt.kind == "strict_minimum"
```

The default numerical search uses the pinned FrankenNumPy linear solver and
admits 1 through 16 variables. The explicit `backend="sparse"` uses native,
bounded preconditioned conjugate gradients on the ordered sparse Hessian and
admits 1 through 64 variables, without forming a dense Hessian. Both retain
coordinate scaling, explicit diagonal regularization, bounded Armijo
backtracking and a limit of 1024 outer iterations. A truncated inner solve only
proposes a descent direction; success requires a freshly evaluated original
gradient. Neither backend silently selects another solver.
Domain-invalid trial points are rejected; cancellation, allocation and budget
failures remain terminal. A caller's `EvaluationBudget` is cumulative across
rejected trials, linear work and final publication. Sparse inner iterations
also observe cooperative cancellation between products.

Large objectives can require an explicit `EvaluationBudget` during
`compile_objective(..., budget=budget)`. Choose adequate compute, memory and
allocation limits for that source. Supporting 64 variables in the sparse
solver does not enlarge default compilation or execution allowances; passing
the same budget to compilation and minimization keeps their costs cumulative.
Its memory allowance tracks cumulative byte charges, including proof work,
and does not reserve physical RAM.

An approximate candidate establishes only a small floating gradient; an initial
maximum or saddle can also be stationary. `minimize_certified` separately checks
the original exact source, the full proposed box, root contraction and sufficient
positive curvature. It returns only a verified strict local minimum, with no
global-optimum claim. Unsupported or inconclusive certification returns no
partial receipt, and one budget spans both numerical search and verification.

---

## Principal inverse functions (development API)

Native scalar programs, checked residual/Jacobian programs and checked
objective/Hessian programs support `asin`, `acos`, `atan`, `asinh`, `acosh` and
`atanh`, including sparse
products and saved-program replay. The independent original-source interval
checker also supports all six functions through its first- and second-order
rules, so they compose into root-box and stationary-point certificates:

```python
from sympy import Rational, Symbol, asin, enclose_objective, certify_stationary_point

x = Symbol("x")
source = asin(Rational(3, 5) + 2*x + 3*x**2)
enclosure = enclose_objective(source, x, [(0, 0)])
# The gradient contains 5/2 and the Hessian contains 195/16.

minimum = certify_stationary_point(
    asin(x)**2, x, [(Rational(-1, 20), Rational(1, 20))],
)
assert minimum.kind == "strict_minimum"
```

The Rust `RealBall::asin` and `RealBall::acos` value operations enclose the
principal real functions on the complete closed domain `[-1, 1]`. They evaluate
monotone endpoints with exact rational interval operations, certified square
roots and arctangents; floating approximations do not supply interval bounds.
The value APIs admit endpoints, while residual and objective certification
require the complete argument interval strictly inside `(-1, 1)` to establish
C1/C2 regularity. Held zero products and powers retain those obligations.
No outside point is rounded or clipped into the domain. Stable Piecewise
selection remains lazy for inactive values, and exact work shares the caller's
cumulative resource and cancellation budget.

Inverse hyperbolic values use certified square roots and logarithms with exact
rational endpoints. Their real-domain requirements are:

| Function | Scalar/interval values | Whole-box first/second derivatives |
| --- | --- | --- |
| `asinh` | All finite real inputs | Any bounded real argument interval |
| `acosh` | `[1, +infinity)` | Complete argument interval strictly above `1` |
| `atanh` | `(-1, 1)` | Complete argument interval strictly inside `(-1, 1)` |

For example, `enclose_objective(asinh(Rational(3, 4) + 2*x + 3*x**2),
x, [(0, 0)])` encloses the value `log(2)`, gradient `8/5` and Hessian `408/125`.
The symbolic `acosh` derivative keeps `sqrt(u - 1)` and `sqrt(u + 1)` separate
to preserve its principal complex branch; the certified numerical lane above
remains real. Saved programs replay the source and derivative checks, and
undefined user functions with the same names do not acquire native capability.

---

## Absolute-value numerical programs and smooth certificates (development API)

`compile_residual_system` and `compile_objective` support real `Abs` branches,
including nested expressions and nonlinear arguments. Checked Jacobians,
Hessians and their vector products retain the argument's sign and every chain
term. The private numerical derivative operations use exact `-1`, `1` and `0`
on their admitted branches, including finite subnormal and very large inputs.
Saved and portable programs independently replay both derivative stages.

```python
from sympy import Abs, Symbol, compile_objective, compile_residual_system

x = Symbol("x")
residual = compile_residual_system(Abs(x) - 2, x)
assert residual.evaluate(-3) == ([1.0], [[-1.0]])

objective = compile_objective((Abs(x) - 2)**2, x)
assert objective.evaluate(-3) == (1.0, [-2.0], [[2.0]])
minimum = objective.minimize_certified(-3)
assert minimum.kind == "strict_minimum"
assert minimum.stationary_enclosure == ((-2, -2),)
```

Scalar `Abs(0)` remains defined, but derivative programs reject a zero argument
at any original `Abs` node. This check survives held zero products, zeroth
powers and zero direction or weight vectors. It conservatively also applies
to a composition such as held `Abs(x**2)` at zero. Public symbolic derivative
forms are unchanged. Numerical proposals can now reach `nsolve_certified` and
`minimize_certified` on smooth branches; their evidence still comes from the
independent original-source checkers below.

Ordinary scalar `nsolve` also selects these checked derivatives for bounded
closed expressions involving `Abs`, `sinc`, `tanh` or `coth`. For example,
`nsolve(Abs(x) - 2, x, -3)` returns the approximate root `-2`. Residual values
retain the existing math arithmetic, and an exactly zero residual still
terminates before evaluating its derivative. A nonroot cusp remains a domain
refusal. The selection keeps ordinary expressions, Piecewise, math-only
functions and expressions beyond its size or exact-integer limits on the
existing symbolic scalar path. Compiler, proof, budget and cancellation
failures propagate directly. This produces numerical candidates; use the
certification APIs for mathematical receipts.

The original-source residual/Jacobian and objective/Hessian checkers support
`Abs` when the complete argument enclosure lies strictly above or strictly
below zero. They prove the branch from exact rational interval bounds, then
propagate its sign through every first and second derivative. This also lets
absolute-value expressions participate in root-box and stationary-point
certificates on smooth boxes:

```python
from sympy import Abs, Symbol, certify_root_box, enclose_objective

x = Symbol("x")
enclosure = enclose_objective(Abs(x), x, [(1, 3)])
assert enclosure.gradient == ((1, 1),)
assert enclosure.hessian == (((0, 0),),)

root = certify_root_box(Abs(x) - 2, x, [(1, 3)])
# The checked root is unique in this declared box.
```

A box touching or crossing the argument's zero remains unsupported by these
C1/C2 checks. Held zero products and zeroth powers do not erase that condition
or an original source pole. These derivative receipts establish smooth-branch
claims. Source replay retains the actual Python `Abs` capability and typed
coordinates; an undefined function named `abs` does not acquire that capability.

### Global optimization at absolute-value corners

`optimize_box_certified` admits continuous absolute-value compositions across
zero. It proves a global minimum or maximum value interval on the complete
declared box and an objective-value error bound for the returned feasible
point. This supports weighted absolute losses, nested `Abs`, and smooth outer
functions such as `exp(Abs(x))` and `1/(1 + Abs(x))`:

```python
from sympy import Abs, Rational, Symbol, optimize_box_certified

x, y = Symbol("x"), Symbol("y")
minimum = optimize_box_certified(Abs(x) + 2*Abs(y), (x, y), [(-1, 1), (-1, 1)])
assert minimum.point == (0, 0)
assert minimum.value == (0, 0)
assert minimum.gap == 0

shifted = optimize_box_certified(
    Abs(x - Rational(1, 3)), x, [(-1, 1)], tolerance=Rational(1, 1024),
)
assert shifted.value[0] <= 0 <= shifted.value[1]
assert shifted.gap <= Rational(1, 1024)
```

An independent checker rebuilds the complete dyadic partition, reevaluates
every original-source cell and the exact feasible point, and verifies the
requested gap. Smooth cells retain second-order Taylor bounds. At an `Abs`
corner a separate evaluator supplies interval values and bounds on coordinate
slopes; integrating those bounds controls value changes and monotonicity.
These slopes never enter Jacobian, Hessian, root or stationary-point receipts.
The direct function requires no numerical derivative compilation, and compiled
objectives expose the same operation through `program.optimize_box_certified`.

The gap concerns objective values. Coordinate accuracy, uniqueness and a
complete optimizer set require separate claims. Ordinary source domains stay
in force: poles behind held zero factors are checked, and a zeroth power whose
original base can be zero still refuses. Singular endpoints such as
`sqrt(Abs(x))` at zero and unresolved Piecewise seams remain outside this
bounded slice. Search, independent replay, checkpoint restoration and final
publication retain cumulative work and cancellation; exhausting a leaf or
resource limit produces no partial certificate.

---

## Exact quartic radicals (development API)

The native rational-polynomial solver supports bounded general quartics through
Ferrari's construction, alongside its linear, quadratic, binomial, biquadratic
and cubic cases. Existing `roots`, `Poly.roots`, equation-solving and matrix
eigenvalue consumers reach this lane:

```python
from sympy import Symbol, roots

x = Symbol("x")
solutions = roots(x**4 + x + 1, x)
assert len(solutions) == 4
assert sum(solutions.values()) == 4
```

The generator proposes two quadratic factors. A separate bounded checker
verifies the auxiliary cubic roots and reconstructs all five original quartic
coefficients using exact rational arithmetic before constructing every quadratic
branch. Repeated factors retain their original multiplicities; rational equations
exclude original denominator poles. Numerical observations supply display
ordering, never acceptance. Oversized or undecidable exact checks refuse.
This publication guard is not a portable proof-kernel certificate, and literal
radical form or ordering parity with upstream is not claimed.

Scalar constant-coefficient and Cauchy-Euler ODE helpers preserve exact complex
exponential or power bases when a nested radical's real and imaginary parts
are unresolved. Real trigonometric pairs require an established nonzero
imaginary part and a matching conjugate root with the same multiplicity.
Euler powers and logarithms share a branch away from zero, including the real
positive axis; no solution family is extended across that singularity.

---

## Checked scalar ODE forcing (development API)

The native `dsolve_const_coeff_second_order_nonhomogeneous(a, b, c, f, x)`
helper solves `a*y'' + b*y' + c*y = f(x)` for bounded rational polynomial
forcing multiplied by exponentials and sine/cosine modes, including resonance:

```python
from sympy import Symbol, dsolve_const_coeff_second_order_nonhomogeneous, exp

x = Symbol("x")
y = dsolve_const_coeff_second_order_nonhomogeneous(1, -2, 1, x*exp(x), x)
# Equivalent to exp(x)*(C1 + (C2 - C1)*x + x**3/6).
```

New forcing cases use a checked companion system and return an expression
whose constants parameterize `y(0)` and `y'(0)`. The helper retains the
established constant convention for its earlier simple forcing cases. Generic
`dsolve` continues to use its existing undetermined-coefficients and
variation-of-parameters dispatch.

The source reader admits rational polynomials of degree at most sixteen and
rational exponential rates and trigonometric frequencies. The forcing closure
and two physical coordinates must fit the existing eight-dimensional system
bound: a lone exponential mode supports polynomial degree at most five, and a
lone oscillatory mode at most two. Combined modes share that allowance; work
budgets and output verification bounds can refuse smaller inputs. Original
poles, foreign typed coordinates, unknown functions, nonlinear phases and
products of two oscillatory factors refuse explicitly. For example, the
current engine reports compute-budget exhaustion for
`y'' - 4*y' + 13*y = x*exp(2*x)*sin(3*x)` even though its six-state
realization fits the dimension bound; that refusal has its own regression.

The system verifier checks the coefficient realization and initial values.
An independent scalar residual check then differentiates the returned answer
against the complete original forcing. Exact rational work, separate
cumulative generation/verification budgets and cancellation checkpoints are
available through Rust's `ode::forcing::metered_second_order`. No numerical
sampling establishes the identity, and these checked expressions are not
portable proof certificates or a general ODE-domain completeness claim.

---

## Quadratic rational telescoping (development API)

The existing `summation` and `Sum.doit` entry points now admit rational
telescopes whose denominators factor into rational linear and irreducible
quadratic factors. Repeated factors, linear quadratic numerators, fractional
offsets, mixed factor families and missing intermediate integer shifts are
supported within the bounded exact-arithmetic lane:

```python
from sympy import Symbol, Sum, oo, summation

k = Symbol("k", integer=True)
g = (3*k + 2)/(k*k + k + 1)**2
summation(g - g.subs(k, k + 3), (k, 0, oo))  # 1199/441
Sum(1/(k*k + 1) - 1/((k + 1)**2 + 1), (k, 0, oo)).doit()  # 1
```

Partial fractions are grouped by exact integer-shift orbits with polynomial
residues. An independent proof-kernel checker reconstructs both complete
denominators, proves that proposed quadratic factors have no rational roots,
checks the rational difference identity, and evaluates the exact endpoints.
Original removable integer poles remain domain obligations. Finite reversed
and empty ranges retain the existing summation convention; infinite sums
require a finite rational endpoint limit.

Input degree is at most 32, output degree at most 128, and an orbit spans at
most 64 integer shifts. Nonzero discrete residues, unsupported factors and
resource failures do not produce partial answers. This expands rational
telescoping; it does not add special-function formulas for arbitrary
non-telescoping rational series.

---

## Gaussian error functions (development API)

The native numerical compiler supports `erf` and `erfc` through
`compile_numeric`, native `lambdify`, checked residual/Jacobian programs,
checked objective/Hessian programs, and supervised hybrid composition.
Complementary tails are evaluated directly, so a tiny positive `erfc(x)`
does not disappear merely because the binary64 value of `erf(x)` rounds
to one.

Original-source interval evaluation independently supports both functions,
their complete first and second derivatives, root-box checks, and local
stationary-point checks:

```python
from sympy import Symbol, compile_numeric, enclose_objective, erfc

x = Symbol("x")
numeric_tail = compile_numeric(x, erfc(x))
# numeric_tail(20.0) is approximately 5.3958656116079e-176.

bounds = enclose_objective(erfc(x), x, [(20, 20)], precision=24)
lower, upper = bounds.value
assert 0 < lower <= upper
assert bounds.gradient[0][1] < 0
assert bounds.hessian[0][0][0] > 0
```

Binary64 results remain approximate. Certified bounds use exact rational
series or continued-fraction inequalities with outward rounding and an exact
final width check. Positive complementary tails through `x=32` retain
relative precision; larger positive arguments use a rigorous absolute
saturation enclosure. Tiny nonzero `erf` arguments retain relative accuracy.
The direct `RealBall` methods accept up to 256 requested decimal digits;
source-certification APIs retain their own precision and work limits.

Checked derivatives preserve typed coordinates and all original domain
obligations, including reciprocals hidden by held zero arithmetic. Work
charges accumulate on the caller's ledger, and failed or cancelled evaluation
does not publish partial output buffers or certificates.

---

## Stable inverse-function derivatives (development API)

Newly compiled `atan` and `asinh` residual/Jacobian and
objective/gradient/Hessian programs use scaled first and second derivative
formulas. They retain representable slopes when squaring a large finite
argument would overflow, and retain representable curvatures when an
intermediate inverse power would otherwise round to zero:

```python
from sympy import Symbol, asinh, atan, compile_objective

x = Symbol("x")
_, gradient, hessian = compile_objective(atan(x), x).evaluate(1e100)
# gradient is approximately [1e-200]; hessian is approximately [[-2e-300]].

_, gradient, hessian = compile_objective(asinh(x), x).evaluate(1e120)
# gradient is approximately [1e-120]; hessian is approximately [[-1e-240]].
```

The same derivatives serve Jacobian and Hessian vector products. Independent
derivative checks and source/opcode replay validate the exact derivative
recipes, including both chain-rule terms for a nonlinear inner expression.
Public symbolic derivative forms and original-source domain obligations
remain unchanged. Historical artifacts replay their original proofs and
bytecode; recompile a source to use the new numerical formulas.

These are approximate binary64 evaluations. A derivative smaller than the
representable range may still round to zero, and other intermediates in a
composite expression may still overflow or underflow. Original-source
rational-ball certification remains a separate check.

---

## Entire sinc evaluation (development API)

The unnormalized `sinc(x) = sin(x)/x`, extended by `sinc(0) = 1`, now
works with `compile_numeric`, math and native `lambdify`, checked
residual/Jacobian programs, and checked objective/gradient/Hessian programs.
Generated numerical derivatives use the analytic values at the origin:
`sinc'(0) = 0` and `sinc''(0) = -1/3`. Bounded Taylor evaluation retains
tiny nonzero first derivatives and avoids cancellation near zero.

```python
from sympy import Rational, Symbol, compile_objective, sinc

x = Symbol("x")
objective = compile_objective(sinc(x), x)
value, gradient, hessian = objective.evaluate(0.0)
# (1.0, [0.0], [[-1/3]]) up to binary64 rounding.

receipt = objective.certify_stationary_point([(Rational(-1, 4), Rational(1, 4))])
assert receipt.kind == "strict_maximum"
```

Original-source rational-ball evaluation independently encloses the value,
gradient, and Hessian, including on intervals crossing zero. It uses
outward-rounded series or certified trigonometric identities, with the
global derivative bounds from `sinc(x) = integral_0^1 cos(t*x) dt` to cover
the entire input interval. The direct `RealBall.sinc` precision ceiling is
256 decimal digits; existing source, work, and trigonometric range limits
still apply. Wide intervals may produce coarse bounds and fail to certify
a stationary point.

Binary64 evaluation remains approximate. Original argument poles, such as
`sinc(1/x)` at zero, remain domain errors. Public symbolic differentiation
retains its existing quotient expression; stable origin derivatives are
provided by compiling the original `sinc` source into a residual or objective
program. Artifact replay checks the new operations against their original
sources, and evaluation charges the caller's cumulative work budget.

---

## First-order linear PDE families (development API)

`pdsolve` now connects the Python shell to a native characteristic solver for

```text
a*u_x + b*u_y + c*u = f(x, y)
```

where `a`, `b`, and `c` are exact rational constants, at least one derivative
coefficient is nonzero. The forcing is a sum of up to sixteen modes of the
form `exp(E)*(P*cos(T) + Q*sin(T))`: `E` and `T` are rational affine forms,
and `P,Q` are rational polynomials of total degree at most 16. Plain
polynomial, exponential and trigonometric forcing are included. For example:

```python
from sympy import Eq, Function, Symbol, cos, diff, exp, pdsolve

x, y = Symbol("x"), Symbol("y")
u = Function("u")(x, y)
solution = pdsolve(Eq(diff(u, x) + diff(u, y) + u, x), u)
# A family equivalent to Eq(u, exp(-x)*F(y - x) + x - 1).

resonant = pdsolve(
    Eq(diff(u, x) + diff(u, y) + 2*u, exp(-x-y)*cos(x-y)), u,
)
# Eq(u, exp(-2*x)*F(y-x) + x*exp(-x-y)*cos(x-y)).
```

The generator constructs affine characteristic coordinates `s,t` and an
elementary particular solution `q`. It solves each cosine/sine amplitude
pair with exact rational arithmetic. When both the effective reaction and
frequency vanish, characteristic integration handles the resonant mode.
Separate scalar coefficient recurrences check both channels of the original
residual, the identities `D(s)=1` and `D(t)=0`, and a nonzero coordinate
determinant before publishing `exp(-c*s)*F(t) + q`. Here
`D = a*d/dx + b*d/dy`, and `F` is an arbitrary differentiable function.
The native result explicitly reports `VerifiedSolutionFamily`.

The shell retains the original dependent-function owner and typed
`Symbol`/`Dummy` coordinates, including same-name coordinates with distinct
identities. `solvefun=Function("Phi")` supplies an ordinary unary arbitrary
function. Original source storage is checked before cancellation and again
after native evaluation and result construction. Nonlinear equations, variable
coefficients, higher derivatives, custom hooks, inexact coefficients and
original coordinate reciprocals are refused by this lane. Elementary phases
must be rational affine expressions; products of trigonometric factors and
powers of elementary factors are outside this bounded grammar. These checks
apply to held original expressions before zero factors or cancellation can
remove a source obligation. The native constant `E`, including freshly lifted
`exp(1)`, is admitted by its exact native type and value.

The native API uses bounded exact arithmetic and cumulative cancellation/work
accounting. This development API does not impose initial or boundary
conditions, cover other PDE classes, establish a complete compatibility
profile, or provide a portable proof-kernel/runtime-reservation receipt.

---

## Exact elementary ranges and extrema (development API)

The Python `function_range`, `minimum`, and `maximum` APIs compose the checked
rational-function image engine with supported elementary maps. This includes
`exp`, `log`, principal rational powers on their admitted real branch,
`asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
`asech`, and `Abs`. Rational scaling and translation may wrap the result.

```python
from sympy import Interval, S, Symbol, acos, exp, function_range, maximum

x = Symbol("x")
function_range(exp(1/x), x, S.Reals)
# Union(Interval.open(0, 1), Interval.open(1, oo))
function_range(acos(x), x, Interval.Lopen(-1, 1))
# Interval.Ropen(0, pi)
maximum(2 - 3*exp(x), x, S.Reals)
# 2: the supremum need not be attained.
```

The engine restricts the complete inner image to the outer function's real
branch before mapping its endpoints. It preserves original held poles,
singleton images, unattained asymptotes, and open or closed endpoints.
Absolute value and `cosh` fold and merge the exact algebraic input image
before evaluating endpoints. Transcendental values are never ordered by
floating samples. A defined negative principal fractional power is not
silently treated as a real odd root.

Domains may be exact real intervals, finite sets, unions, intersections, or
complements of those sets, including disconnected and unbounded sets. Finite
boundaries must be rational or genuine real `CRootOf` values. The rational
`stationary_points` and `imageset` lanes accept the same compound domains.
This lets the result of `continuous_domain` feed directly into range and
extremum calculations:

```python
from sympy import Complement, FiniteSet, Interval, exp

punctured = Complement(Interval(0, 2), FiniteSet(1))
function_range(x**2, x, punctured)
# Union(Interval.Ropen(0, 1), Interval.Lopen(1, 4))
function_range(exp(x), x, punctured)
# Union(Interval.Ropen(1, exp(1)), Interval.Lopen(exp(1), exp(2)))
```

The domain tree becomes one exact partition of points and open gaps, which is
intersected with the original source's definedness before any image is
published. All components share one work ledger and complete native sign
charts. A point removed from the source cannot be restored by a union in the
requested domain; an isolated point can restore an attained value only when
the source is defined there.

This bounded lane accepts one elementary function of one rational expression;
it refuses correlated sums such as `exp(x) + x`, nested elementary functions,
unsupported domain leaves, and exhausted exact checks. It is a shell
composition over existing checked native results, with intrinsic resource
bounds; it does not supply a new portable range certificate or close the full
SymPy compatibility profile.

---

## Custom functions in residual systems (development API)

Ordinary arithmetic retains the actual Python function applications, including
distinct classes with the same printed name. Dense and sparse matrices preserve
those objects through copies, immutable snapshots, mutation, elementary
arithmetic, and Jacobian construction. A supervised symbolic derivative hook can supply
the Jacobian used by an explicit numerical callback program:

```python
from sympy import Function, SparseMatrix, Symbol, compile_hybrid_residual_system

class ConstitutiveLaw(Function):
    nargs = 1

    def fdiff(self, argindex=1):
        return 2*self.args[0]

x = Symbol("x")
residuals = SparseMatrix([ConstitutiveLaw(x) - 4])
snapshot = residuals.as_immutable()
jacobian = snapshot.jacobian([x])
program = compile_hybrid_residual_system(
    snapshot, [x], jacobian=jacobian,
    callbacks={ConstitutiveLaw: lambda value: value*value},
)

assert program.jacobian_pattern == ((0, 0),)
assert program.evaluate_sparse(3.0) == ([5.0], [6.0])
assert program.jacobian_vector_product([2.0], 3.0) == [12.0]
assert program.jacobian_transpose_vector_product([0.5], 3.0) == [3.0]

residuals[0] = 999                 # The compiled snapshot is unchanged.
root = program.solve([3.0])       # Matrix([[2.0]]) to numerical tolerance.
assert abs(float(root[0, 0]) - 2.0) < 1e-10
```

The supplied Jacobian is a user assertion. Its callback values and root remain
approximate; this path does not issue a native derivative proof or a root-box
certificate. Sparse evaluation returns derivative values in the immutable
`jacobian_pattern` order; products avoid dense numerical Jacobian output.
Only original canonical integer-zero derivative atoms are omitted. Float zeros,
held expressions and callbacks remain stored, even when they evaluate to zero.
Every residual and stored derivative runs in captured occurrence order before
a product is reduced, including with zero direction or weights. A shared
`EvaluationBudget` and `SupervisedCallbackSession` cover the full request;
completed callback effects remain observable if a later operation fails.
Compilation and existing dense APIs retain their original admission limits.

The same compiled graph can solve repeatedly with selected parameters fixed.
Use an exact built-in dict whose keys are the captured typed `Symbol` or
`Dummy` identities and whose values are finite reals:

```python
from sympy import Add, Dummy, Function, Symbol, compile_hybrid_residual_system

x, y = Symbol("x"), Symbol("y")
gain, offset = Symbol("gain"), Dummy("offset")

class LinearLaw(Function):
    nargs = 2

parameterized = compile_hybrid_residual_system(
    [Add(LinearLaw(x, gain, evaluate=False), y, -offset, evaluate=False), x - y],
    (gain, x, offset, y),              # Fixed coordinates need not be last.
    jacobian=[[x, gain, -1, 1], [0, 1, 0, -1]],
    callbacks={LinearLaw: lambda value, coefficient: coefficient * value},
)
first = parameterized.solve([0, 0], parameters={gain: 2, offset: 6})
second = parameterized.solve([0, 0], parameters={offset: 12, gain: 3})
assert all(abs(float(first[i, 0]) - 2) < 1e-10 for i in range(2))
assert all(abs(float(second[i, 0]) - 3) < 1e-10 for i in range(2))
```

Here positions 0 and 2 are fixed, while the initial vector and returned column
Matrix follow the remaining coordinates `(x, y)` in original order. Mapping
order does not affect coordinate order. Parameter values are captured before
callbacks run; changing the caller's dict during a solve does not change that
request. At least one unknown must remain, and the existing square or
overdetermined solve requirements apply to those unknowns.

Every residual evaluation retains the complete original source. When a Jacobian
is requested, all original derivative columns run before selecting the unknown
columns, including fixed-parameter domains and callbacks. The existing exact
zero binary64 residual exit can precede derivative evaluation. The supplied
Jacobian and returned roots remain approximate, without derivative proofs or
root certificates; the same session and budget span each complete solve.

Exact built-in `Piecewise` expressions can retain custom callback branches and
conditions. The hybrid compiler evaluates conditions in stored order, preserves
short-circuit `And`/`Or`/`Not` behavior, and executes only the selected branch.
An inactive callback or invalid numerical domain is never evaluated. Use held
`Add`/`Mul`/`Pow` when composing an opaque Piecewise with other expressions;
closed native algorithms continue to require a closed source.

```python
from sympy import Piecewise, compile_numeric_hybrid

law = compile_numeric_hybrid(
    x,
    Piecewise((ConstitutiveLaw(x, evaluate=False), x > 0),
              (0, True), evaluate=False),
    callbacks={ConstitutiveLaw: lambda value: value*value},
)
assert law(3) == 9.0
assert law(-3) == 0.0            # The callback is not invoked.
```

Every branch is checked during capture, including unreachable branches; this
does not execute its callbacks. Missing matches raise an error. Solver trials
may backtrack after a native domain failure only if the trial has not executed
a callback; a failure after an effect remains terminal.

Sum, product, power, and supported built-in chain rules compose
around retained applications. An unknown custom derivative stays a formal
`Derivative` of the original object; unresolved custom compositions do not yet
expand into every upstream `Subs` representation.

Sparse snapshots use `ImmutableSparseMatrix`; subsequent changes to the source
or a mutable copy do not change the captured cells. Stored-coordinate algebra,
transpose, and Jacobian construction preserve sparse storage. Numerical APIs
still apply their bounded output limits before materializing result arrays.
Closed native matrix algorithms continue to refuse opaque cells. Immutable
dense and sparse snapshots use a common coordinate hash for equal values; this
is an internal consistency contract, not a claim of exact upstream hash parity.

Retained applications also participate in exact replacement, bounded algebraic
substitution, arithmetic expansion, and conservative simplification. Polynomial
and rational transformations can use the existing native algorithms while
keeping the actual application objects:

```python
from sympy import cancel, expand, factor, factor_list, together

law = ConstitutiveLaw(x)
expanded = expand((law + 1)**2)             # law**2 + 2*law + 1
factored = factor(law**2 - 1)               # (law - 1)*(law + 1)
scale, factors = factor_list(law**2 - 1)    # (1, [(law - 1, 1), (law + 1, 1)])
reduced = cancel((law**2 - 1)/(law + 1))     # law - 1
combined = together(1/law + 1/(law + 1))    # (2*law + 1)/(law*(law + 1))
assert (x + 1).xreplace({x: law}) == law + 1
```

Function applications become formal generators only inside one transformation;
the returned graph contains the original objects. Exact coefficient checks
validate factor products and division reconstruction before publication. The
bounded polynomial lane admits rational coefficients, up to eight generators,
degree 32, 4096 terms and 4096-bit coefficients, with additional work and native
algorithm limits. Explicit generators must be independent of any remaining
function coefficients. Noninteger powers, custom factor hooks, and deep or
`fraction=False` transformation by `together` refuse in this lane. These APIs do
not issue a portable proof for arbitrary Python functions or preserve the
number of later numerical callback occurrences across algebraic rewrites.

Native admission also checks that built-in function classes and their relevant
implementations still match their captured capabilities. Changing a built-in
hook revokes new closed compilation of affected expressions; restoring it
permits admission again. Existing immutable programs keep their captured native
semantics. Changing inherited Function equality, hashing or commutativity
prevents arithmetic from merging applications under an obsolete identity rule.
These checks are process-local contracts, not a sandbox for arbitrary changes
to Python module globals.

Opaque expressions have no usable native mirror. Closed numerical compilers,
verified native algorithms, and unsupported matrix algorithms refuse them.
Explicit custom commutativity or equality/hash protocols require their own
supported algebra. Arithmetic graphs and formal differentiation are bounded; arbitrary non-cooperative
Python hooks retain the existing supervised-boundary limitations.

## Exact integer linear algebra (development API)

The bounded integer lane computes Smith forms and complete integer solutions
with independent proof-kernel acceptance:

```python
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_form

assert smith_normal_form(Matrix([[4, 6], [3, 9]])) == Matrix([[1, 0], [0, 18]])

A = Matrix([[2, 4]])
P, K = A.solve_integer(Matrix([[6]]))
assert P == Matrix([[3], [0]])
assert K == Matrix([[-2], [1]])
# Every integer solution is P + K*T, with an arbitrary integer parameter T.
assert A.solve_integer(Matrix([[1]])) is None
```

`A.smith_normal_decomposition()` returns `(D, U, V)` with `D = U*A*V`.
Acceptance proves both transformations have integer inverses and that the
positive invariant factors divide their successors. Integer solving checks
divisibility in the transformed right-hand side and returns a complete primitive
kernel, or checked inconsistency. Multiple RHS columns and all four built-in
dense/sparse mutable/immutable classes are supported; decomposition and solve
outputs preserve the coefficient matrix's class.

Each dimension is bounded by 64 and original coefficients by 4096 bits, with
separate cumulative generation and verification limits. These bounds may still
refuse difficult inputs. See [integer normal forms and systems](docs/INTEGER_NORMAL_FORMS.md)
for the exact contracts, examples, failure semantics, and retained evidence.
The broad WS10/WS19 and compatibility-profile gates remain open.

## Exact sparse linear algebra (development API)

### Sparse selection and block assembly

Sparse `row()`, `col()`, `extract()`, and two-dimensional slicing work from
stored coordinates. They preserve reversed, stepped and repeated selections
without scanning the implicit rectangle. Nonmutating `row_join()`,
`col_join()`, `row_insert()`, `col_insert()`, `hstack()` and `vstack()`
use the same sparse assembly path, so selected and assembled matrices can feed
the checked sparse solvers directly:

```python
from sympy import SparseMatrix

n = 600
A = SparseMatrix(n, n, {(i, i): 1 for i in range(n)})
B = SparseMatrix(n, 1, {(0, 0): 2, (n - 1, 0): 3})
augmented = SparseMatrix.hstack(A, B)

coefficients = augmented[:, :-1]
rhs = augmented[:, -1]
assert type(coefficients) is SparseMatrix
assert coefficients.LUsolve(rhs).todok() == B.todok()
```

Selections, joins and insertions retain the source sparse class and the
original held/custom scalar objects. Multi-input stacks follow the first
input's matrix class; a single-input stack returns that input unchanged.
Mixed sparse/dense assembly therefore stays sparse when the first input is
sparse. Native-only blocks remain native-only without reconstructing Python
function owners.

The output bound applies to stored entries, not the implicit matrix area;
assembly also bounds aggregate input work.
Repeated selections that exceed the stored-entry limit are refused before
building the result. Explicit selector lists and flat slices (which return
Python lists) are also bounded. Dense-first stacks still have to fit the
dense output limit. Effectful selectors and insertion positions are checked
against the original sources before publication.

### Checked sparse rank and linear solves

Exact rational sparse matrices now support checked `rank()`, `nullspace()` and
`columnspace()` without constructing a dense intermediate. This includes
rectangular and singular matrices whose implicit area exceeds the dense matrix
storage limit:

```python
from sympy import SparseMatrix

entries = {(i, i): 1 for i in range(600)}
entries[0, 600] = 2
A = SparseMatrix(600, 601, entries)

assert A.rank() == 600
basis = A.nullspace()
assert len(basis) == 1
assert basis[0].todok() == {(0, 0): -2, (600, 0): 1}
assert (A * basis[0]).nnz == 0
assert len(A.columnspace()) == 600
```

The native generator records invertible exact row operations. A separate
proof-kernel checker replays them using a different sparse representation and
checks the complete reduced form and rank. Nullspace publication also checks
the original matrix residuals and every free coordinate. Native Rust callers
can retain the source, witness and receipt through
`SparseMatrix::rref_checked_with_meters`, with separate generator and verifier
meters; ordinary metered methods use one cumulative ledger.

This lane admits up to 8192 rows or columns, 262144 stored entries, and 8192-bit
coefficient components, with additional limits on intermediate fill, arithmetic,
allocation and total basis output. Resource and cancellation refusals are
terminal. Opaque entries and custom zero/simplification callbacks do not enter
the exact rational lane. Mutable and immutable sparse nullspace/column-space
vectors retain their source class. For rational input, public `rref()` follows
the pinned SymPy behavior and returns a mutable dense **final result**; a result
whose area exceeds the dense storage limit is refused even when sparse rank and
nullspace remain available. These operations do not certify the full matrix
compatibility profile.

For consistent tall systems, `LUsolve()` uses checked sparse augmented row
reduction and independently checks the original `A*X == B` equations for every
right-hand side. It requires full column rank, including when the RHS has no
columns, and refuses inconsistent or nonunique systems. The result retains the
right-hand side's mutable/immutable and dense/sparse class:

```python
from sympy import ImmutableMatrix, SparseMatrix

A = SparseMatrix([[1, 0], [0, 1], [1, 1]])
B = ImmutableMatrix([[2, 7], [3, 8], [5, 15]])
X = A.LUsolve(B)
assert type(X) is ImmutableMatrix
assert X == ImmutableMatrix([[2, 7], [3, 8]])
```

The default public `SparseMatrix.solve()` keeps the pinned square-only contract;
use `LUsolve()` for this tall exact rational case. The augmented column count
and intermediate fill must fit the same sparse bounds. Both original matrices
are checked before and after native execution, and checked results are
published through original storage descriptors rather than replaceable Python
constructors.

### Exact sparse least squares

`SparseMatrix.solve_least_squares(B)` also handles inconsistent tall rational
systems, minimizing the sum of squared residuals for each right-hand side.
Generation forms sparse normal equations; acceptance independently replays a
full-column-rank witness for the original coefficient matrix and checks the
original equations `A.T * (A * X - B) == 0` exactly. The coefficient matrix,
normal matrix and residual checking remain sparse.

```python
from sympy import Matrix, Rational, SparseMatrix

A = SparseMatrix([[1, 0], [0, 1], [1, 1]])
B = Matrix([[1], [2], [4]])
X = A.solve_least_squares(B)
assert X == Matrix([[Rational(4, 3)], [Rational(7, 3)]])
```

The public default is `method="LDL"`; `"CH"` uses the same checked exact
rational lane. Full column rank is required even for an empty right-hand side.
Underdetermined or rank-deficient systems, unsupported methods and nonrational
entries refuse explicitly. Mutable sparse coefficients preserve the RHS matrix
class; immutable sparse coefficients return an immutable sparse result.
Sparse fill, coefficient growth, cumulative work and allocation limits remain
enforced through generation and verification.

Native callers can retain `VerifiedSparseLeastSquares`, including its original
RHS and source-bound rank evidence, or use separate
`generate_sparse_least_squares_candidate` and
`verify_sparse_least_squares_candidate` calls with separate meters. The combined
least-squares result is immutable in-process evidence; its rank witness uses
the existing portable RREF checker.

### Complete exact affine solution families

Both dense and sparse `gauss_jordan_solve()` use one checked sparse elimination
for all right-hand sides. Consistent singular and underdetermined systems
return the complete family of solutions, including every free variable.
The solution and parameter matrices retain the coefficient matrix's
mutable/immutable and dense/sparse class:

```python
from sympy import Matrix, SparseMatrix

A = SparseMatrix([[1, 2, 0], [2, 4, 0]])
B = Matrix([[3, 5], [6, 10]])
solution, parameters, free_columns = A.gauss_jordan_solve(B, freevar=True)

assert type(solution) is SparseMatrix
assert type(parameters) is SparseMatrix
assert parameters.shape == (2, 2)
assert free_columns == [1, 2]
# parameters = [[tau0, tau1], [tau2, tau3]]
# solution = [[3 - 2*tau0, 5 - 2*tau1],
#             [tau0,       tau1],
#             [tau2,       tau3]]
```

Native Rust callers can retain the verified rational parts through
`SparseMatrix::solve_affine()`: a particular solution `P`, a complete
nullspace basis `N`, and ascending free-column indices. Every solution is
`X = P + N*T`, with one parameter per free variable and RHS column.
A dependency-separated proof-kernel checker replays the augmented row
operations and independently checks the original `A*P == B`, `A*N == 0`,
rank, and free-coordinate identities. Inconsistent systems have a checked
contradiction; the public method raises `ValueError`.

The native parameter renderer constructs plain `tau0`, `tau1`, ... symbols
in row-major parameter order under the same cumulative work ledger. Source
admission, generation, verification, lifting and publication stay bounded.
Python validates the original source snapshots and canonical parameter
identities and publishes results through captured storage descriptors.
Neither replaceable constructors nor Python scalar arithmetic build the
accepted solution.

This exact rational lane replaces the old dense affine method's 128-dimension
restriction; dense input and final output still obey the dense storage limit.
Sparse elimination does not expand the implicit dense area. The augmented
system has at most 8192 rows and 8192 columns. Symbolic rendering allows at
most 262144 parameters, one million pre-combination terms, and 262144 stored
entries across the solution and parameter matrices together; coefficient,
fill, work and allocation limits can refuse earlier. Resource or cancellation
errors remain terminal.

`freevar` must be an actual `bool`. The Python method retains its explicit
refusal of zero RHS columns when unknowns remain; native `solve_affine()`
also represents that empty matrix equation and its coefficient kernel.
Symbolic coefficients, approximate values, custom matrix subclasses and
custom arithmetic hooks are outside this lane. The
[native execution record](artifacts/audit/2026-11-10-sparse-affine-native-validation.json)
records 73 passing sparse native tests at `f22cc5a`, including all 18
complete-family verifier tests and 14 native affine tests that were previously
pending. Actual Python execution of the new affine extension remains pending;
the earlier 174 passing source-scoped cases predate that feature. These
bounded results do not certify a full compatibility profile.

### Sparse top-level linear systems

Exact rational matrix inputs to `linsolve()` and `solve_linear_system()` now
use the checked sparse affine engine. Both `(A, b)` pairs and augmented
`[A | b]` matrices retain sparse storage through solving; the pair route no
longer converts `A` to a dense matrix or assembles a dense augmented matrix.
The former 128-row/variable restriction is replaced by the same sparse
dimension, fill, coefficient and work bounds.

```python
from sympy import Dummy, ImmutableMatrix, SparseMatrix, Symbol, linsolve
from sympy import solve_linear_system

x = Symbol("x", real=True)
y, z = Dummy("free"), Dummy("free")
A = SparseMatrix([[1, 2, 3], [2, 4, 6]])
b = ImmutableMatrix([1, 2])

answer = next(iter(linsolve((A, b), x, y, z)))
assert answer[1] is y
assert answer[2] is z
# answer is the complete family (1 - 2*y - 3*z, y, z).

augmented = SparseMatrix([[1, 2, 3, 1], [2, 4, 6, 2]])
assignments = solve_linear_system(augmented, x, y, z)
# assignments contains the pivot assignment for x; y and z stay free.
```

The native renderer retains the exact supplied free-coordinate identities,
including assumptions and distinct same-name `Dummy` objects. Valid
reconstructed `Dummy` copies remain the caller-owned free coordinates without
replacing the existing registry owner. Omitting coordinates selects canonical
`tau0`, `tau1`, ... parameters. Native pair and augmented entry points admit
the entire ordered coordinate list, including pivot or otherwise unused
coordinates, before solving.

The public matrix routes accept the four built-in dense/sparse,
mutable/immutable matrix classes. They support a single RHS, including the
existing row-vector RHS convention. `linsolve()` returns a `FiniteSet` of one
symbolic tuple, or `EmptySet` for inconsistency; `solve_linear_system()`
returns pivot assignments only, or `None` for inconsistency. Original matrix
storage, typed coordinates, scalar/result interfaces and parameter ownership
are checked around native work and publication. Accepted exact families are
lifted without Python scalar arithmetic or replaceable constructors.

The augmented system remains bounded by 8192 rows and 8192 columns, with
additional stored-entry, output, coefficient and cumulative resource limits.
Supplied coordinate names are limited to 4096 UTF-8 bytes each and one MiB in
total. Exact duplicate coordinates, nonsymbol coordinates and unsupported
custom classes refuse explicitly. Sparse storage does not remove the final
tuple/set output limits.

Equation-list coefficient extraction retains its existing implementation.
The existing generic symbolic-coefficient route remains available when exact
rational admission declines before native solving; an error after entering
the exact engine is terminal. This does not add generic symbolic-parameter
completeness or a compatibility-profile certification.
[The integration record](artifacts/audit/2026-11-10-sparse-linear-system-integration.json)
separates the executed earlier native baseline from the new named-coordinate
and public regression suites, whose execution remains pending after executor
loss.

## Exact multiquadratic minimal polynomials (development API)

`minpoly()` and `minimal_polynomial()` accept rational constants, rational
square roots, `I`, sums, products, integer powers and reciprocals in an exact
multiquadratic field over QQ. Dependent square classes are reduced exactly,
including the principal-root phases of negative radicands. The multiplication
operator is passed to the native matrix minimal-polynomial checker, which
checks annihilation and the independence of all lower powers.

```python
from sympy import I, Rational, minpoly, sqrt, symbols

x = symbols("x")
assert minpoly(Rational(2, 3), x) == 3*x - 2
assert minpoly(sqrt(2) + sqrt(3), x) == x**4 - 10*x**2 + 1
assert minpoly(1/(sqrt(2) + sqrt(3)), x) == x**4 - 10*x**2 + 1
assert minpoly(sqrt(2) + I, x) == x**4 - 2*x**2 + 9
```

The default result is a primitive polynomial expression with positive leading
coefficient. `polys=True` returns a `Poly` for an explicit typed generator, or
a `PurePoly` with a fresh `Dummy` when the generator is omitted. Generator
identity and assumptions are retained. Both Boolean `compose` options use the
same checked field algorithm; the supported domain forms are `None` and `"QQ"`.

This lane admits four independent square classes, giving degree at most 16,
with bounded source size, integer powers, coefficient height and total work.
Original held operands are checked before algebraic cancellation, including
undefined denominators under zero products or zero powers. General algebraic
extensions, irrational nested radicals, approximate inputs and custom scalar
hooks refuse explicitly. This extends the implemented exact arithmetic slice;
the broad number-field and compatibility profiles remain open.

## Exact partial polynomial evaluation (development API)

`Poly.eval` and `PurePoly.eval` perform checked exact partial evaluation over
`ZZ` and `QQ`. A partial result keeps its polynomial class, coefficient domain
and remaining typed generators in their original order, including unused
generators of a zero or constant result. Assigning every generator returns an
exact scalar:

```python
from sympy import Poly, QQ, Rational, symbols

x, y = symbols("x y")
polynomial = Poly(x*y + x + y, x, y, domain=QQ)
assert polynomial.eval(2) == Poly(3*y + 2, y, domain=QQ)
assert polynomial.eval(y, 3) == Poly(4*x + 3, x, domain=QQ)
assert polynomial.eval([2, 3]) == 11
assert polynomial.eval({x: 2, y: 3}) == 11
assert polynomial.eval({}) is polynomial

promoted = Poly(x*y, x, y, domain="ZZ").eval(Rational(1, 2))
assert promoted == Poly(y/2, y, domain=QQ)
```

The API also accepts positive or negative coordinate indices. Indices in an
ordered mapping refer to the generators still present after earlier entries.
Fractional assignments promote `ZZ` to `QQ`; `auto=False` refuses that
promotion. The complete original source and all values are admitted even
when an assignment later makes the result zero.

The native generator projects sparse monomials, while a separate verifier
reconstructs every resulting coefficient by repeated multiplication and
checks the remaining ring. Native requests have cumulative generation and
verification meters, cancellation checkpoints and bounded input sizes.
Existing scalar symbolic and floating-point substitutions retain their
preselected shell behavior; they do not receive this exact coefficient-map
verification. An exact-lane failure never retries through that shell route.

---

## Prime-field polynomials (development API)

The Python shell exposes checked sparse polynomial arithmetic over `GF(p)`
with one to eight ordered generators. Construction, addition, subtraction,
multiplication, powers, monic normalization, mixed partial derivatives, and
division retain the field and exact symbol identities:

```python
from sympy import Poly, reduced, symbols

x, y = symbols("x y")
f = Poly(x**2 + x*y + 2*y, x, y, modulus=5)
g = Poly(x + y, x, y, modulus=5)
q, r = f.div(g)
assert q*g + r == f
assert f.diff(x, y) == Poly(1, x, y, modulus=5)

quotients, remainder = reduced(
    f, [g, Poly(y, x, y, modulus=5)], order="grevlex")
assert quotients[0]*g + quotients[1]*Poly(y, x, y, modulus=5) + remainder == f
```

`terms`, `monoms`, and `coeffs` support `lex`, `grlex`, and `grevlex` order.
`reduced` returns a quotient for every supplied divisor and independently
checks reconstruction and complete remainder reduction. Ordered division by
an arbitrary list does not claim a unique normal form or a Gröbner basis.

`Poly.gcd` and `Poly.lcm` compute complete multivariate results over the
declared prime field. The same operations are available on `PurePoly` and
through module functions; nonzero results are monic in the declared generator
order, including when the inputs have different nonzero scalar factors:

```python
from sympy import gcd, lcm

h = x + y
left = Poly(2*h**2*(x + 1), x, y, modulus=5)
right = Poly(3*h*(y + 1), x, y, modulus=5)
assert left.gcd(right) == Poly(h, x, y, modulus=5)
assert left.lcm(right) == Poly(h**2*(x + 1)*(y + 1), x, y, modulus=5)
assert gcd(left, right) == left.gcd(right)
assert lcm(left, right) == left.lcm(right)
```

Two nonconstant inputs use complete checked ideal-intersection evidence and
independent coefficient identities. This needs one auxiliary variable, so
these requests admit up to seven declared generators. Zero or scalar inputs
retain the eight-generator bound. A zero operand gives zero LCM and the monic
other operand as GCD; both results are zero when both operands are zero.
Original typed generators, field characteristic, and the `symmetric` lifting
policy are retained. Exhausted or incomplete work returns no partial result.

`Poly.eval` supports exact integer values in multivariate prime-field rings.
Partial evaluation returns a `Poly` over the same field with the remaining
generators in their original order; complete evaluation returns the integer
residue representative under the original `symmetric` policy. It accepts a
single value for the first coordinate, an explicit generator/value pair, a
sequence of leading-coordinate values, or an ordered mapping:

```python
x, y, z = symbols("x y z")
polynomial = Poly(x**2*y + 3*y*z + 4, x, y, z, modulus=5)
assert polynomial.eval(x, 2) == Poly(-2*y*z - y - 1, y, z, modulus=5)
assert polynomial.eval([2, 3]) == Poly(-z + 1, z, modulus=5)
assert polynomial.eval({x: 2, y: 3, z: 4}) == 2
assert polynomial.eval({}) is polynomial
```

The native generator and independent coefficient checker bind every assigned
value and remaining typed coordinate. A complete request shares cumulative
generation and verification budgets. Nonintegral, approximate, or custom
values refuse without changing the field; even a no-op or zero result retains
original-source admission.

`groebner` also constructs a complete reduced basis over the requested prime
field. It supports all three monomial orders, checked `reduce` and `contains`,
`is_zero_dimensional`, and expression or `Poly` output. The coefficient field
is part of the mathematical request: the same generators can define a proper
ideal in one characteristic and the unit ideal over the rationals.

```python
from sympy import groebner

basis = groebner([x**2 + 1, x + 1], x, modulus=2)
assert basis.exprs == [x + 1]
assert basis.contains(x**2 + 1)
assert not basis.contains(1)
assert basis.reduce(x**3) == ([x**2 + x + 1], 1)
assert groebner([x**2 + 1, x + 1], x, domain="QQ").exprs == [1]
```

The native checker establishes both ideal inclusions, monic reducedness, and
every S-pair's required leading-product bound. It validates the current basis
before a reduction or membership answer. Retained `Poly` inputs are checked
in their own declared typed rings before promotion to the common ring, and
original sources are admitted before zero or unit ideal shortcuts.

Prime-field bases support four checked ideal operations:

- `eliminate(*generators)` removes any distinct subset of the original typed
  coordinates and preserves the order of the remaining coordinates.
- `intersection(other)` computes the complete intersection of two ideals in
  the same prime field and typed polynomial ring.
- `saturate(polynomial)` computes the principal saturation
  `I:polynomial**infinity`, retaining multiplicity in the remaining factors.
- `quotient(other)` computes the finite ideal quotient `I:J`, requiring the
  product with every divisor generator to lie in `I`. It removes only the
  finite multiplicity present in the divisor ideal.

All four return complete reduced bases in lexicographic order and retain the
left operand's coefficient lift and expression/`Poly` output policy. They
use independently accepted complete ambient bases. Elimination, intersection
and saturation check the complete projected subset. Finite quotients check
every principal intersection, every exact coefficient-level cancellation and
every intersection combining the divisor constraints; a list of valid members
alone would not establish completeness.

```python
curve = groebner([x*y - 1, y**2 - x], x, y, modulus=5)
assert curve.eliminate(x).exprs == [y**3 - 1]

first = groebner([x], x, y, modulus=5)
second = groebner([y], y, x, modulus=5)
assert first.intersection(second).exprs == [x*y]

nonreduced = groebner([x**2*y**2], x, y, modulus=2)
assert nonreduced.saturate(x).exprs == [y**2]

numerator = groebner([x**3*y], x, y, modulus=5)
divisor = groebner([x], x, y, modulus=5)
assert numerator.quotient(divisor).exprs == [x**2*y]

numerator = groebner([x**2, y**2], x, y, modulus=5)
divisor = groebner([x, y], y, x, modulus=5)
assert numerator.quotient(divisor).exprs == [x**2, x*y, y**2]
```

These operations act on polynomial ideals, preserving characteristic and
nilpotent multiplicity. They do not replace an ideal with its set of rational
points over the finite field. Eliminating every variable returns an explicit
constant-field basis with `gens == ()`: for example,
`groebner([x**2 + 1], x, modulus=3).eliminate(x)` is the zero ideal even though
the original polynomial has no roots in GF(3). Constant-field results support
membership, reduction and further ideal operations.

Finite-dimensional prime-field quotients also support `fglm(order)`,
`quotient_basis()`, and `multiplication_matrix(element)`. FGLM converts among
`lex`, `grlex`, and `grevlex` through exact modular linear algebra; both ideal
inclusions and every target S-pair are checked before publication. The complete
standard-monomial basis and every multiplication-matrix column are checked
independently. Repeated roots and nilpotent directions remain present:

```python
from sympy import Matrix

graded = groebner([x*y - 1, y**2 - x], x, y, modulus=3, order="grevlex")
lex = graded.fglm("lex")
assert lex.exprs == [x - y**2, y**3 - 1]
assert lex.quotient_basis() == (1, y, y**2)
assert lex.multiplication_matrix(y) == Matrix([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])
```

Here `y**3 - 1 == (y - 1)**3` in characteristic three: the quotient has
dimension three even though there is only one distinct root. Matrices use
integer residue representatives under the basis's `symmetric` policy; ordinary
`Matrix` arithmetic must subsequently be reduced modulo `p` to retain that
field interpretation. Quotient dimension is bounded by 64. The unit ideal
returns an empty monomial basis and a 0-by-0 matrix; the explicit constant
field's zero ideal returns `(1,)`. Positive-dimensional quotients refuse
order changes or finite operator construction. A same-order `fglm` request
checks the current source and returns the original object, including for
positive-dimensional ideals.

Prime-field quotients also support `invert(element)` and
`minimal_polynomial(element, gen)` (alias `minpoly`). These operations admit
positive-dimensional ideals. Inversion checks the complete augmented ideal
`I + <element>`; its Bezout cofactors supply an inverse, which is independently
checked and reduced in the declared order. A proved nonunit raises
`NotInvertible`; resource or verification failures remain distinct errors.

Minimal polynomials compute the complete elimination ideal
`<I, gen - element>` intersected with `GF(p)[gen]`. The result is its monic
generator, retaining inseparability and nilpotent multiplicity. `None` proves
that no nonzero polynomial annihilates the element. The output variable must
be a distinct typed `Symbol` or `Dummy`; omitting it creates a fresh `Dummy`.
The unit ideal has inverse representative zero and minimal polynomial one.

```python
t = symbols("t")
curve = groebner([x*y - 1], x, y, modulus=5)
assert curve.invert(x) == y
assert curve.minimal_polynomial(x, t) is None

nilpotent = groebner([x**3], x, y, modulus=5)
assert nilpotent.invert(1 - x) == 1 + x + x**2
assert nilpotent.minpoly(x, t) == t**3
```

Both operations retain the original prime, exact typed sources and coefficient
lift policy through native verification and public output. They admit at most
31 original ideal generators. Inversion supports eight variables; minimal
polynomials support seven, leaving one auxiliary coordinate for the output
variable. The element in a minimal-polynomial request has at most 4095 terms
to leave room for `gen - element`. Degree and intermediate sparse-work bounds
still apply; there is no finite-point sampling or rational-domain retry.

`syzygies()` returns a complete generating set of polynomial relations on the
current ordered expressions. Each tuple `(a_1, ..., a_m)` satisfies
`sum(a_i * exprs[i]) == 0` over the retained field. Zero, duplicate and scaled
input positions remain distinct coordinates. For example:

```python
coordinate_ideal = groebner([x, y], x, y, modulus=5)
assert coordinate_ideal.syzygies() == [(y, -x)]
```

The implementation composes the complete checked Groebner transformation
matrices and every strict S-pair relation. A separate coefficient checker
requires all resulting nonzero rows in order and independently checks their
zero dot products. A handful of valid relations is insufficient evidence of
completeness. The result need not be a minimal generating set or a reduced
module basis. There are at most 32 original polynomials and 16384 output
polynomial cells; cumulative work bounds cover the entire request and lift.

Intersection and saturation introduce one auxiliary variable, which counts
toward the eight-variable bound. Their constructed inputs count toward the
32-polynomial limit, so intersection admits at most 32 combined generators
and saturation admits at most 31 original generators. A finite quotient admits
at most 32 generators per operand; any nonzero divisor requires at most seven
original variables and 31 numerator generators. An empty or all-zero divisor
ideal yields the unit ideal after both original source lists pass admission,
including unused declared polynomial coordinates. Intermediate intersections
remain bounded by 32 combined generators. The degree bound applies after the
auxiliary construction.

The lane admits primes through 65521, total degree at most 64, up to 4096 sparse
terms, and at most 32 basis polynomials or ordered divisors. Original source expressions and
coefficient growth have additional bounds. Source admission, generation,
verification, and lifting use cumulative budgets; an exhausted or rejected
request cannot restart over the rationals. Multivariate factorization,
resultants, and composition remain unsupported. Existing
univariate prime-field operations remain available. These are bounded
in-process checked operations, not a certified compatibility profile or
portable proof-kernel certificates.

## Ordinary Fourier transforms (development API)

`fourier_transform(f, x, k)` uses the documented cycles-per-unit kernel
`exp(-2*pi*I*k*x)`. Its bounded native catalog admits exact rational quadratic
Gaussians `exp(-a*x**2 + b*x + c)` with `a > 0`, and affine absolute-value
exponentials `exp(c - a*Abs(b*x + d))` with `a > 0` and `b != 0`, together
with finite sums and rational amplitudes. For example:

```python
from sympy import Abs, exp, fourier_transform, symbols

x, k = symbols("x k", real=True)
gaussian = fourier_transform(exp(-x**2), x, k)
# Mathematically: sqrt(pi) * exp(-pi**2 * k**2)
bilateral = fourier_transform(exp(-Abs(x)), x, k)
# Mathematically: 2 / (1 + 4*pi**2 * k**2)
```

Translations retain the exact phase `exp(-2*pi*I*center*k)`. The bridge
preserves typed symbols and original source arguments, checks convergence
premises with rational arithmetic, and keeps resource failures terminal.
The bilateral identity is scoped to real frequency; explicitly nonreal
frequencies remain unevaluated. The supported `noconds=False` profiles
retain the pinned oracle's sufficient conditions.

Nonzero constants and unsupported transform families remain unevaluated.
The separate angular-frequency native catalog retains its own convention.
Requests have fixed source, coefficient-growth, output and cumulative
resource limits, including at most 64 catalog terms. These direct integral
identities do not constitute a general transform algorithm, certified
compatibility profile, or portable proof-kernel certificate.

## Checked linear recurrences (development API)

The bounded `rsolve` interface retains the original equation and checks each
proposed expression against it and every supplied initial condition:

```python
from sympy import Function, Symbol, rsolve

n = Symbol("n", integer=True)
y = Function("y")
solution = rsolve(y(n + 1) - y(n) - n, y(n), {y(0): 0})
# n*(n - 1)/2
```

Constant-coefficient recurrences support polynomial/geometric forcing and
repeated characteristic roots. First-order homogeneous variable-coefficient
recurrences are admitted when the product engine returns a checkable closed
form, including factorial sequences. Private helper identities and fresh
output constants preserve user parameters, including parameters named `C0`
and distinct symbols with the same printed name.

First-order forced recurrences `y(n + 1) = a*(n + c)*y(n) + q(n)` also
have a bounded variation-of-constants route. Here `a` is a nonzero rational,
`c` is a positive integer, and dividing the forcing by the next homogeneous
factor must yield an admitted polynomial/geometric sum. For example:

```python
from sympy import factorial

solution = rsolve(
    y(n + 1) - (n + 1)*y(n) - factorial(n + 1),
    y(n),
    {y(0): 2},
)
# (n + 2)*factorial(n)
```

This route uses the existing native summation engine and independently checks
the returned expression in the original recurrence. Unsupported multipliers
or unevaluated sums refuse; they do not become an incomplete closed answer.

Sequence shifts and initial indices must be exact integers; initial-condition
applications must belong to the solved function. Fractional/Float shifts,
self-referential initial data, hidden original poles, undecidable parameter
branches and failed exact residual checks refuse explicitly. Limits are order
32, shift magnitude 1024, forcing degree 64, 64 initial conditions and a
predictive 4096-term expansion envelope. The factorial-forced route admits
at most 64 forcing terms and normalized polynomial weights through degree
32; rational geometric construction is bounded to 4096 bits.

Residual acceptance uses exact integer-shift identities and original-source
substitution. It establishes the returned identity on the common defined
integer lattice domain, without a completeness, singular-point extension,
portable-certificate or compatibility-profile claim.

---

## Exact integer equations and source domains (development API)

The bounded `diophantine` interface enumerates definite binary quadratic
solutions using integer projection bounds, integer square roots and exact
divisibility. Large translations do not pass through floating-point bounds:

```python
from sympy import Eq, Symbol, diophantine, solve, symbols

x, y = symbols("x y", integer=True)
N = 10**400 + 7
points = diophantine((x - N)**2 + y**2 - 1)
assert points == {(N - 1, 0), (N + 1, 0), (N, -1), (N, 1)}
```

Shifted bilinear equations `b*x*y + d*x + e*y + f = 0` use the identity
`(b*x + e)*(b*y + d) = d*e - b*f`. A nonzero right side yields every
signed divisor pair satisfying both coordinate congruences. A zero right
side yields each integral affine family:

```python
assert diophantine((x - 2)*(y + 3) - 6) == {
    (-4, -4), (-1, -5), (0, -6), (1, -9),
    (3, 3), (4, 0), (5, -1), (8, -2),
}
n = Symbol("n1", integer=True)
assert diophantine((x - 2)*(y + 3)) == {(2, n), (n, -3)}
```

Quadratic coefficients are bounded to 4096 bits. Enumeration admits at most
100,000 rows or trial divisors and 20,000,000 height-weighted work units;
the conic search chooses its shorter coordinate projection. Requests
exceeding these bounds raise `NotImplementedError` before enumeration.

Both `solve` and `diophantine` retain original expression domains before
normalization. A pole cannot disappear merely because identical terms
cancel between held Equality sides:

```python
source = Eq(1/x, 1/x + x, evaluate=False)
assert solve(source, x) == []
assert solve(source, x, check=False) == [0]
```

`solve` applies domain admission to scalar roots and complete system rows
before output formatting, retaining inconclusive candidates and the
existing `check=False` behavior. Its shared allowance covers every candidate,
including the exact candidates generated before Float output conversion.

`diophantine` filters finite integer points against the original domains.
It refuses a parametric family whose necessary exclusions remain unresolved,
as the current tuple format cannot express those conditions. Original
cancelled axes are retained. Zero-polynomial families and linear equations
with three or more axes and inactive coefficients refuse instead of returning
incomplete parameterizations. This source-admission lane accepts structured
expressions, bounded integer/name tokens, and at most 64 distinct coordinates;
explicit `syms` must be a built-in list or tuple of symbols. Compound expression
strings and external conversion hooks are unsupported.

These are development compatibility features. The integer helpers check
original coefficient residuals, but these shell routes do not produce portable
proof certificates or certify an upstream profile. The
[execution report](artifacts/audit/2026-11-15-integer-conics-and-source-domains-progress.json)
separates the executed conic checks from the later public tests that still
require a native-extension run.

---

## Planned algorithm program

The native program includes proof- or certificate-aware portfolios for:

- arbitrary-precision integer, rational, modular, algebraic, and certified ball arithmetic;
- dense, sparse, recursive, modular, and black-box polynomial representations;
- GCD, resultants, univariate and multivariate factorization;
- Gröbner bases, ideals, elimination, and order conversion;
- dense, sparse, structured, modular, and p-adic exact linear algebra;
- simplification and bounded conditional equality saturation;
- differentiation, sparse Jacobians/Hessians, and symbolic compilation;
- integration, limits, series, asymptotics, sums, products, and transforms;
- algebraic/transcendental equations, inequalities, sets, logic, SAT, and Diophantine problems;
- ODEs and selected PDE workflows with explicit completeness status;
- special functions, geometry, tensor/index calculus, statistics, units, physics, and control.

The selector chooses which work to try. The verifier decides what can be accepted.

See [`docs/ALGORITHM_PORTFOLIOS.md`](docs/ALGORITHM_PORTFOLIOS.md).

---

## Franken-suite integration

### asupersync

Structured concurrency, capability contexts, cancellation, budgets, deterministic lab execution, replay, transport, RaptorQ mechanisms, and operational monitoring foundations.

### FrankenSQLite

An optional embedded computation ledger for immutable universe manifests, verified cache entries, checkpoints, receipts, workspace history, and crash recovery. It stays outside the in-memory algebraic hot path and never defines term equality or proof validity.

### FrankenGraphDB

An optional rebuildable index over derivations, dependencies, proof use, counterexamples, branches, discrepancies, and collaborative work. Graph reachability is not logical entailment.

### FrankenNumPy and FrankenSciPy

Targets for verified symbolic-to-numeric lowering: residuals, Jacobians, Hessians, sparsity structures, domain guards, exact/certified reference lanes, quadrature, ODE, optimization, and root workflows.

Each integration remains optional and layered. Source-project mechanisms are not assumed live merely because they exist in a repository; integration claims require their own end-to-end gates.

---

## First architecture-proving campaign

The first implementation campaign is the **Certified Jacobian Pipeline**.

A user constructs a nonlinear residual system containing:

- exact polynomial blocks;
- transcendental built-ins;
- a deliberately held expression;
- a custom Python `Function` subclass;
- assumptions;
- a mutable matrix snapshot.

The campaign must demonstrate, in one end-to-end artifact bundle:

1. profile-correct Python construction, hashing, sorting, traversal, printing, and pickle;
2. explicit lowering that preserves held/custom surface behavior;
3. proof-producing sparse Jacobian generation;
4. a two-strategy factorization portfolio with independent verification;
5. verified residual/Jacobian compilation for FrankenNumPy/FrankenSciPy;
6. certified numeric enclosures;
7. typed cancellation, complete draining, and a resumable continuation;
8. checkpoint corruption, RaptorQ recovery, digest/schema/dependency validation, and fresh-process resume;
9. deterministic replay;
10. an agent semantic patch and verifier-checked branch merge;
11. rejection of an invalid remote candidate without verified-cache pollution;
12. parity-gated benchmarks against upstream SymPy and a scalar native lane.

Passing this campaign proves one deep architecture slice. It does not certify complete SymPy compatibility.

See [`docs/FIRST_IMPLEMENTATION_CAMPAIGN.md`](docs/FIRST_IMPLEMENTATION_CAMPAIGN.md).

---

## Implementation program

The machine-readable DAG contains 24 planned workstreams:

```text
WS00  Governance, registries, and claims
WS01  Conformance laboratory
WS02  IDs, schemas, budgets, and Cx
WS03  Exact arithmetic
WS04  Terms, domains, assumptions, and bindings
WS05  Python compatibility shell
WS06  Proof kernel and evidence
WS07  Rewriting and simplification
WS08  Polynomial arithmetic
WS09  GCD and factorization
WS10  Exact linear algebra
WS11  Certified numerics and algebraic numbers
WS12  Differentiation and symbolic compilation
WS13  Portfolios, cancellation, and replay
WS14  Agent protocol and semantic workspaces
WS15  Persistence, checkpoints, and RaptorQ repair
WS16  Remote workers and graph indexing
WS17  Gröbner bases and ideal algebra
WS18  Integration, limits, series, and transforms
WS19  Solvers, sets, logic, ODE, and PDE
WS20  Structured mathematics domains
WS21  Compatibility and ecosystem closure
WS22  Performance and architecture optimization
WS23  Packaging, release, and 1.0 certification
```

All 24 remain `planned` at the closure level; substantial slices have landed inside several workstreams (see [`registries/claims.toml`](registries/claims.toml) for per-claim implementation status).

See [`docs/WORKSTREAM_GRAPH.md`](docs/WORKSTREAM_GRAPH.md) and [`registries/workstreams.toml`](registries/workstreams.toml).

---

## Planned workspace architecture

```text
L7  Product packaging, CLI, Python distributions
L6  Protocol, Python bridge, Wasm, generated-code targets
L5  Persistence, distribution, graph index, repair adapters
L4  Planning, portfolios, workspaces, compilation, services
L3  Symbolic algorithm generators
L2  Terms, domains, assumptions, claims, proof kernel, verifiers
L1  Exact arithmetic, canonical encoding, deterministic collections
L0  Typed IDs, budgets, outcomes, schemas, capabilities
```

Key dependency rules:

- verifier crates cannot depend on optimizing generator crates;
- semantic core crates cannot depend on Python, persistence, graph, or network layers;
- asupersync is the only async runtime;
- ordinary native crates forbid unsafe code;
- no C/C++ CAS or arbitrary-precision FFI;
- the CPython bridge is contained behind an audited safe Rust binding layer rather than hand-written C API code;
- optional features cannot change term/proof semantics;
- generated code derives deterministically from reviewed registries.

See [`docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md`](docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md).

---

## Conformance and benchmarking

The planned conformance laboratory uses isolated processes for upstream SymPy, FrankenSymPy compatibility, native/reference execution, mathematical verification, comparison/minimization, and artifact publication.

Coverage goes far beyond the upstream suite:

- complete source/reflection inventory;
- generated type/domain/assumptions-aware fixtures;
- custom subclasses, metaclasses, converters, and `_eval_*` hooks;
- held/evaluated forms;
- mutable aliases and snapshots;
- warnings, exceptions, printers, pickles, module paths, and signatures;
- metamorphic tests;
- verifier mutation;
- fuzzing;
- deterministic concurrency and cancellation exploration;
- crash, corruption, repair, and malicious worker tests;
- ecosystem packages, notebooks, and serialized artifacts.

A benchmark case is timed only after it passes the declared compatibility and mathematical comparator in the same run. Reports include the live incumbent, raw paired data, outcome mix, memory, tails, proof/verifier cost, and exact execution mode.

See [`docs/CONFORMANCE_AND_BENCHMARKING.md`](docs/CONFORMANCE_AND_BENCHMARKING.md).

---

## Security posture

FrankenSymPy treats expressions and artifacts as potentially hostile programs/data. Planned controls include:

- multidimensional budgets and admission control;
- bounded decoders and output/printer limits;
- supervised Python and plugin callbacks;
- explicit unsafe pickle capability rather than ordinary protocol support;
- candidate/verified cache separation;
- local verification of untrusted worker/cache/storage artifacts;
- privacy-scoped content stores and IDs;
- supply-chain and dependency review;
- no broad memory-safety claim across CPython or third-party C extensions.

See [`docs/SECURITY_AND_RESOURCE_GOVERNANCE.md`](docs/SECURITY_AND_RESOURCE_GOVERNANCE.md).

---

## Documentation map

Start with:

- **[Comprehensive plan](COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md)**
- **[Constitution](docs/CONSTITUTION.md)**
- **[Source-project audit](docs/SOURCE_PROJECT_AUDIT.md)**
- **[First implementation campaign](docs/FIRST_IMPLEMENTATION_CAMPAIGN.md)**
- **[Workstream graph](docs/WORKSTREAM_GRAPH.md)**
- **[Risk register and research agenda](docs/RISK_REGISTER_AND_RESEARCH_AGENDA.md)**

Deep architecture:

- [Compatibility contract](docs/COMPATIBILITY_CONTRACT.md)
- [Object model and IR](docs/OBJECT_MODEL_AND_IR.md)
- [Assumptions, domains, and numeric tower](docs/ASSUMPTIONS_DOMAINS_AND_NUMERIC_TOWER.md)
- [Evidence, proofs, and rewriting](docs/EVIDENCE_PROOFS_AND_REWRITES.md)
- [Algorithm portfolios](docs/ALGORITHM_PORTFOLIOS.md)
- [Runtime, budgets, and determinism](docs/RUNTIME_BUDGETS_AND_DETERMINISM.md)
- [Persistence, distribution, and repair](docs/PERSISTENCE_DISTRIBUTION_AND_REPAIR.md)
- [Agent-native protocol](docs/AGENT_NATIVE_PROTOCOL.md)
- [Conformance and benchmarking](docs/CONFORMANCE_AND_BENCHMARKING.md)
- [Security and resource governance](docs/SECURITY_AND_RESOURCE_GOVERNANCE.md)
- [Crate architecture and dependencies](docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md)

Machine-readable governance:

- [Compatibility profiles](registries/compatibility_profiles.toml)
- [Evidence classes](registries/evidence_classes.toml)
- [Workstream DAG](registries/workstreams.toml)
- [Claims and gates](registries/claims.toml)

---

## Non-negotiable shortcuts

FrankenSymPy will not claim success by:

- hiding upstream SymPy as a runtime fallback;
- exposing one opaque Rust expression type and calling it drop-in compatible;
- using strings or printer output as semantic identity;
- canonicalizing away held or custom Python behavior;
- treating unknown assumptions as true or false;
- applying branch- or condition-sensitive rules unconditionally;
- calling heuristic, sampled numeric, selector, e-process, worker, or oracle evidence a proof;
- allowing generators to self-verify;
- publishing the first speculative candidate before verification;
- leaving detached or orphan work;
- treating database rows, graph reachability, cache flags, or RaptorQ decoding as mathematical truth;
- accepting repaired artifacts before digest/schema/dependency/evidence validation;
- importing a C/C++ CAS or big-number engine through FFI;
- admitting incompatible cases into benchmark aggregates;
- weakening tests, comparators, or goldens to land features;
- counting API stubs or planning documents as implementation completion.

---

## License

FrankenSymPy is licensed under the [MIT License with the repository rider](LICENSE).
