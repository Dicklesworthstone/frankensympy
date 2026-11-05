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

residuals[0] = 999                 # The compiled snapshot is unchanged.
root = program.solve([3.0])       # Matrix([[2.0]]) to numerical tolerance.
assert abs(float(root[0, 0]) - 2.0) < 1e-10
```

The supplied Jacobian is a user assertion. Its callback values and root remain
approximate; this path does not issue a native derivative proof or a root-box
certificate. Sum, product, power, and supported built-in chain rules compose
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
