# FrankenSymPy first implementation campaign

**Status:** normative initial execution plan  
**Campaign name:** **Certified Jacobian Pipeline**  
**Purpose:** prove the shell/kernel/proof/runtime/persistence/agent architecture end to end before broad SymPy surface expansion

## 1. Campaign thesis

The first implementation must not be a toy parser, a pile of API stubs, or a fast polynomial crate disconnected from Python compatibility. It must force the load-bearing architectural bets to compose in one realistic workload.

The campaign therefore builds a narrow but deep vertical slice that demonstrates:

- real SymPy-compatible Python classes, including arbitrary subclasses and held forms;
- explicit lowering into a deterministic native semantic DAG;
- exact domains and immutable assumptions contexts;
- proof-producing differentiation and factorization;
- a speculative algorithm portfolio whose winner is independently verified;
- typed budgets, cancellation, zero orphan work, and deterministic replay;
- symbolic-to-numeric residual/Jacobian compilation for FrankenNumPy/FrankenSciPy;
- certified numerical validation;
- checkpoint, crash/corruption, RaptorQ byte recovery, digest validation, and resume;
- structured NDJSON/agent workspaces and verifier-checked semantic merge;
- parity-gated performance evidence against upstream SymPy and native reference lanes.

If this composition fails, expanding to thousands of APIs would only bury the architectural fault.

## 2. Campaign boundaries

### In scope

#### Python object surface

- `Basic`, `Atom`, `Expr` foundations required by the slice;
- `Integer`, `Rational`, selected float/special atoms;
- `Symbol`, `Dummy`;
- `Add`, `Mul`, `Pow`;
- `FunctionClass`, `Function`, undefined and applied functions;
- `Derivative` sufficient for custom/unknown functions;
- evaluated and `evaluate=False` construction;
- `args`, `func`, equality, hashing, sorting, free symbols;
- `subs`, `xreplace`, traversal, reconstruction;
- core assumptions used by algebra/differentiation;
- `str`, `repr`, compact pretty, LaTeX;
- copy/deepcopy/pickle for supported slice objects;
- one mutable/immutable matrix boundary sufficient for Jacobians.

#### Native mathematical surface

- exact integers/rationals;
- generic expression, `ZZ`, `QQ`, and polynomial rings over them;
- immutable assumptions contexts;
- semantic term DAG and binders required by differentiation;
- deterministic local rewrites;
- dense and sparse multivariate polynomial representations;
- polynomial identity checking;
- univariate/multivariate factorization subset required by the hero corpus;
- exact dense/sparse Jacobian structures;
- structural differentiation;
- real ball arithmetic and certified residual enclosure;
- target-neutral numeric IR and FrankenNumPy/FrankenSciPy adapters.

#### Operational surface

- asupersync-owned request/portfolio regions;
- multidimensional budgets and protected verifier reservation;
- two candidate strategies and one independent factorization verifier;
- deterministic/replay modes;
- typed continuation/checkpoint;
- local durable ledger adapter;
- RaptorQ sidecar for the checkpoint/replay bundle;
- NDJSON request/event/terminal stream;
- branch fork, semantic patch, verification, merge;
- simulated untrusted remote worker.

### Out of scope for this campaign

- full SymPy 1.14.0 API parity;
- general integration, limits, series, ODE/PDE, tensor, physics, statistics, plotting, and every matrix class;
- universal factorization/Gröbner coverage;
- a production distributed cluster;
- a certified drop-in wheel;
- broad performance claims;
- arbitrary plugin ABI stability;
- a universal interrupt bound for Python callbacks;
- claims that all native outputs are formally proved.

Out-of-scope APIs remain absent or explicitly preview/unsupported. They are not stubbed to inflate inventory numbers.

## 3. The hero workload

### 3.1 User-level scenario

A Python user defines a custom symbolic constitutive law and builds a parameterized nonlinear residual system containing:

- exact rational and symbolic parameters;
- sparse polynomial blocks with nontrivial common factors;
- transcendental built-ins;
- a deliberately held subexpression;
- a user-defined `Function` subclass with custom evaluation and derivative behavior;
- assumptions such as positive/nonzero parameters;
- a mutable parameter matrix converted to an immutable computation snapshot.

Illustrative shape, not a frozen benchmark formula:

```python
class ConstitutiveLaw(Function):
    nargs = 2

    @classmethod
    def eval(cls, x, k):
        if x.is_zero:
            return S.Zero

    def _eval_derivative(self, s):
        ...

held = Add(a*x**2, -a*x**2, b*x*y, evaluate=False)
residual = Matrix([
    held + (x - y)*(x**2 + x*y + y**2) + sin(y) + ConstitutiveLaw(x, k) - c,
    (x + y)*(x**2 - 2*x*y + y**2) + exp(x) - d,
    p*(x**2 - y**2) + q*(x - y)**2 + ConstitutiveLaw(y, k) - e,
])
```

The exact corpus uses multiple generated variants rather than one hard-coded expression.

### 3.2 Required operations

1. Construct the system through profile-correct Python shell classes.
2. Demonstrate that `held.args`, `held.func`, printers, hash/equality, traversal, and pickle match the pinned upstream profile.
3. Lower eligible regions to native terms while preserving the custom class and held surface graph.
4. Infer exact polynomial domains and assumptions context.
5. Compute a sparse symbolic Jacobian with proof terms for built-in rules and provenance-marked custom derivative hooks.
6. Identify/factor exact polynomial subexpressions and selected Jacobian minors/determinants using a two-strategy portfolio.
7. Independently verify the requested factorization claim, including normalization, multiplicity, and any irreducibility level actually claimed.
8. Apply proof-producing common-subexpression and target-aware transformations.
9. Compile residual and Jacobian evaluators for the FrankenNumPy/FrankenSciPy path, retaining a supervised callback slot for the custom function where it cannot lower natively.
10. Evaluate over a parameter box and produce certified real-ball residual/Jacobian enclosures.
11. Optionally certify one isolated root box or a Newton/Krawczyk-style contraction claim if the initial ball substrate supports it; otherwise return an honest enclosure without overclaiming uniqueness.
12. Persist a checkpoint during modular factorization/compilation.
13. Cancel the request, drain all children, and return a continuation.
14. Corrupt/remove checkpoint symbols within the selected RaptorQ recovery envelope.
15. Recover candidate bytes, validate canonical digest/schema/dependencies, and resume in a fresh process.
16. Export a deterministic replay bundle and reproduce the same terminal semantic/evidence digest.
17. Fork an agent workspace branch, submit a semantic patch proposing a lower-cost equivalent Jacobian form, verify it, and merge it.
18. Submit an invalid remote factorization candidate and prove it cannot enter the verified cache or branch.
19. Run compatibility and mathematical admission gates, then benchmark admitted cases against upstream SymPy and the scalar native lane.

## 4. Why this workload is architecture-complete

| Architectural bet | Hero workload pressure |
|---|---|
| Dual-lane shell/kernel | Custom Python subclass, held form, mutable snapshot, native polynomial/differentiation regions |
| Three-graph model | Exact surface preservation, canonical terms, expandable derivative/factorization derivations |
| Domain/assumption explicitness | Positive/nonzero parameters, polynomial rings, branch-sensitive built-ins |
| Proof-carrying portfolios | Multiple factorization strategies, independent verifier, conditional custom derivative provenance |
| Structured concurrency | Parallel strategies, verifier reservation, cancellation/drain, remote simulation |
| Stable content identity | Cross-process terms, proof, checkpoint, generated program, replay bundle |
| Persistence/repair | Checkpoint publication, corruption, RaptorQ recovery, digest and proof separation |
| Agent-native protocol | Structured request/events, semantic patch, branch merge, counterexample/rejection bundle |
| Franken numeric bridge | Residual/Jacobian compilation and certified/reference evaluation |
| Compatibility discipline | Class/metaclass/hooks/evaluate/hash/printer/pickle differential fixtures |
| Claim discipline | Every demonstrated property has a separate gate and no “FrankenSymPy is complete” implication |

## 5. Campaign stage C0 — Freeze planning inputs

### Deliverables

- source-project pins;
- SymPy 1.14.0 profile target;
- workstream and claims registries;
- exact initial object/API inventory scope;
- term/claim/evidence/checkpoint/protocol schema drafts;
- dependency/unsafe audit policy;
- hero corpus specification and licensing/provenance.

### Gate

- all registries parse and the work graph is acyclic;
- every campaign claim is `planned`;
- source pins resolve;
- no implementation status inferred from this document;
- architecture review finds no shell/kernel identity collapse.

## 6. Campaign stage C1 — Conformance skeleton

### Deliverables

- isolated SymPy 1.14.0 oracle environment;
- candidate process protocol;
- initial reflection/source inventory;
- observation envelope;
- exact comparators for construction/type/args/hash/sort/printer/pickle;
- custom subclass and held-form seed corpus;
- mismatch minimizer/discrepancy schema.

### Gate

- deliberately broken candidate shell produces expected discrepancies;
- oracle isolation test proves no shared imports/objects;
- goldens include exact environment and source digests;
- comparator-weakening mutants fail.

### Target command contract

```bash
cargo xtask profile verify sympy-1.14.0-cpython
cargo xtask conformance smoke --profile sympy-1.14.0-cpython
cargo xtask conformance mutation-test --suite comparators
```

These are target interfaces to implement, not current commands.

## 7. Campaign stage C2 — Native foundation

### Deliverables

- typed IDs/outcomes/budgets/`Cx`;
- canonical encoding;
- big integer/rational/modular arithmetic;
- deterministic maps/arenas;
- term DAG, domains, contexts, binders;
- bounded native schemas.

### Gate

- native crate subset builds without Python;
- stable IDs match across fresh processes and scalar/optimized modes;
- arithmetic property/reference corpus passes;
- concurrent interning schedule exploration passes;
- cancellation/budget tests have zero controlled orphan work;
- unknown schema and oversized input fail closed.

### Target command contract

```bash
cargo xtask gate foundation
cargo xtask gate deterministic-ids
cargo xtask lab explore --suite interning,budgets,cancellation
```

## 8. Campaign stage C3 — Python object-model slice

### Deliverables

- profile-correct shell classes;
- safe CPython bridge;
- exact-class native fast paths;
- opaque custom-function descriptors;
- held-form surface descriptors;
- lowering/lifting receipts;
- core printers and pickle paths;
- mutable matrix snapshot boundary.

### Gate

- initial profile inventory closure;
- upstream differential suite for every listed observation;
- custom subclass/metaclass/hook corpus;
- held/evaluated corpus;
- multiple `PYTHONHASHSEED` values;
- pickle/copy across fresh processes;
- no product-path import of upstream SymPy;
- lower/lift preserves the declared surface relation.

### Target command contract

```bash
cargo xtask gate python-object-model --profile sympy-1.14.0-cpython
cargo xtask conformance run --suite custom-subclasses,held-forms,pickle
cargo xtask package inspect --assert-no-upstream-runtime
```

## 9. Campaign stage C4 — Proof and rewrite nucleus

### Deliverables

- typed equality/polynomial/factorization/derivative claims;
- proof kernel;
- assumptions and congruence steps;
- deterministic rewrite registry;
- polynomial identity verifier;
- derivative proof rules;
- evidence/result envelopes;
- verifier mutation harness.

### Gate

- positive and adversarial proof corpus;
- side-condition and branch mutants rejected;
- candidate/verified cache namespaces enforced;
- stored verification flags rejected;
- verifier crates have no generator dependencies;
- unknown claim/evidence versions fail closed.

### Target command contract

```bash
cargo xtask gate proof-kernel
cargo xtask mutants run --family proof,rewrite,poly-identity
cargo xtask deps verify --rule verifier-not-generator
```

## 10. Campaign stage C5 — Polynomial and factorization portfolio

### Deliverables

- dense and sparse polynomial rings over `ZZ`/`QQ`;
- conversion and invariant checks;
- subresultant and modular GCD paths;
- finite-field factorization, lifting, recombination subset;
- independent factorization verifier;
- strategy diagnostics and decision card;
- checkpointable modular work frontier.

### Gate

- representation round trips;
- exact product/normalization/multiplicity and declared irreducibility checks;
- missing-factor/wrong-domain/wrong-order mutants rejected;
- two strategies produce candidates under owned scopes;
- verifier-protected budget remains available;
- only verified candidate publishes;
- deterministic result independent of task completion order.

### Target command contract

```bash
cargo xtask gate polynomial-factorization
cargo xtask lab explore --suite portfolio-winner,verifier-budget
cargo xtask mutants run --family factorization
```

## 11. Campaign stage C6 — Cancellation, continuation, and replay

### Deliverables

- request/portfolio region topology;
- typed cancellation safe points;
- two-phase candidate/cache/checkpoint publication;
- deterministic random counter partition;
- trace and replay bundle;
- factorization continuation schema;
- schedule/fault injection.

### Gate

- cancellation injected before/after every publication boundary;
- no controlled child survives return;
- no cancelled candidate enters verified cache;
- continuation resumes to same accepted semantic/evidence digest;
- strict deterministic mode is byte-stable where declared;
- replay mode reproduces adaptive decision sequence.

### Target command contract

```bash
cargo xtask lab explore --suite campaign-runtime --all-cancel-points
cargo xtask replay verify artifacts/campaign-runtime.bundle
cargo xtask gate no-orphan-work
```

## 12. Campaign stage C7 — Differentiation and compiled Jacobian

### Deliverables

- proof-producing structural differentiation;
- sparse Jacobian/Hessian dependency analysis;
- custom derivative hook receipt/provenance;
- exact linear algebra subset for selected minors/determinants;
- proof-producing CSE/rewrite extraction;
- target-neutral numeric IR;
- FrankenNumPy/FrankenSciPy evaluator adapters;
- scalar/exact reference evaluator.

### Gate

- derivative proof replay for built-ins;
- unknown/custom functions remain formal or provenance-marked, never invented;
- Jacobian sparsity agrees with exact dependency graph;
- generated evaluator matches exact/reference values on admitted points;
- branch/domain guards reject invalid points;
- target output digest and provenance reproducible;
- shell-visible Jacobian form/type matches profile where compatibility API is used.

### Target command contract

```bash
cargo xtask gate differentiation
cargo xtask gate compiled-jacobian --targets rust,frankennumpy,frankenscipy
cargo xtask conformance run --suite derivative,matrix-jacobian
```

## 13. Campaign stage C8 — Certified numerical validation

### Deliverables

- directed-rounding real balls;
- adaptive precision;
- certified residual/Jacobian enclosure;
- optional root-box contraction verifier;
- exact-recognition proposals only behind verification;
- numeric receipt and branch/singularity checks.

### Gate

- independent enclosure corpus;
- directed-rounding mutants killed;
- precision escalation terminates with certified/inconclusive/resource outcome;
- no ordinary float inhabits a certified value;
- root uniqueness/completeness claimed only if the implemented certificate proves it;
- custom callback values are bounded/certified only under explicit callback evidence.

### Target command contract

```bash
cargo xtask gate certified-numeric
cargo xtask mutants run --family rounding,enclosure
cargo xtask campaign validate-boxes --corpus hero-v1
```

## 14. Campaign stage C9 — Persistence and repair

### Deliverables

- append-only ledger and object store traits;
- local FrankenSQLite adapter;
- typed checkpoint publication;
- verified/candidate cache separation;
- RaptorQ repair envelope;
- scrub/decode records;
- fresh-process resume and GC roots.

### Gate

- crash injection at every artifact publication step;
- loss/corruption within selected symbol envelope is repaired;
- recovered bytes must match canonical digest;
- wrong digest/schema/dependency refuses resume;
- proof/evidence replay remains separate and can reject repaired content;
- persistence-disabled execution returns same semantic result;
- database/index IDs never enter `TermId`/proof identity.

### Target command contract

```bash
cargo xtask lab crash-matrix --suite campaign-checkpoint
cargo xtask repair test --corpus hero-v1 --loss-matrix exhaustive-small
cargo xtask resume verify --fresh-process artifacts/hero.checkpoint
```

## 15. Campaign stage C10 — Agent protocol and semantic merge

### Deliverables

- NDJSON request/event/terminal protocol;
- object/claim/evidence/receipt introspection;
- branch fork and semantic patch;
- proof-aware merge;
- counterexample/rejection bundle;
- simulated remote work packet and local verifier;
- deterministic session replay.

### Gate

- unknown/oversized schemas fail closed;
- candidate versus accepted terminal typing enforced;
- same-print/different-domain patch conflicts;
- unverified edge cannot merge as verified;
- invalid remote factorization rejected and quarantined;
- duplicate/late worker response cannot double-publish;
- transcript-free replay reconstructs final branch digest.

### Target command contract

```bash
cargo xtask protocol conformance --schema 1
cargo xtask workspace replay artifacts/hero-session.bundle
cargo xtask lab explore --suite semantic-merge,remote-worker
```

## 16. Campaign stage C11 — End-to-end hero closure

### Required artifact bundle

```text
hero-v1/
├── profile and environment manifest
├── Python source fixture and custom class
├── surface observation bundle
├── canonical term/context/domain objects
├── lowering/lifting receipts
├── Jacobian and factorization claims
├── proof/certificates and verification receipts
├── decision card and runtime trace
├── generated numeric programs and reference vectors
├── certified enclosure/root artifacts
├── checkpoint and RaptorQ sidecars
├── corruption/decode/resume records
├── agent branch/semantic patch/merge records
├── invalid remote candidate rejection bundle
├── deterministic replay bundle
├── differential/conformance report
├── parity-gated benchmark raw data
└── claims/discrepancy status snapshot
```

### Closure conditions

All of the following must hold on one commit:

1. initial object-model inventory is complete;
2. every upstream differential fixture in campaign scope passes or has a blocking discrepancy;
3. accepted mathematical claims verify independently;
4. all registered campaign mutants are killed;
5. cancellation/crash/corruption/remote adversarial matrices pass;
6. deterministic replay reproduces the declared terminal digest;
7. no hidden upstream runtime fallback exists;
8. no uncontrolled orphan work is detected;
9. no benchmark case failed semantic admission;
10. all public campaign claims resolve through the claims registry to this bundle.

Passing C11 proves the architecture works for one deep slice. It does not certify general SymPy drop-in compatibility.

## 17. Benchmark corpus

### 17.1 Compatibility cases

- small scalar constructions where shell/bridge overhead dominates;
- held/evaluated mixtures;
- custom subclass mixtures;
- substitution, hashing, sorting, printers, pickle;
- Jacobian construction through compatibility API.

### 17.2 Native algebra cases

- dense and sparse polynomial blocks by degree/term count/height;
- factorization regimes favoring subresultant/modular paths;
- exact Jacobian/minor computations;
- repeated/batch systems sharing structure;
- compilation amortization across many evaluations.

### 17.3 Operational cases

- cancellation latency by safe point;
- checkpoint publication/resume;
- verifier cost and proof size;
- RaptorQ encode/decode overhead versus recomputation value;
- protocol streaming and branch merge;
- remote duplicate/invalid response handling.

### 17.4 Admission

Every timed case first passes:

- exact profile comparator where applicable;
- independent mathematical verifier;
- output/evidence-class match;
- identical mode, budget, cache, durability, and thread policy.

Upstream SymPy, the FrankenSymPy scalar/reference lane, and optimized candidate are measured in the same invocation or a controlled paired run. Failed cases appear in the ledger, not the speed aggregate.

## 18. Performance hypotheses, not claims

The campaign tests these hypotheses:

- canonical native DAGs materially reduce memory and repeated traversal for large/batched expressions;
- sparse Jacobian generation and DAG reuse outperform repeated Python recursive differentiation;
- modular factorization portfolios win on appropriate degree/height regimes despite verification overhead;
- compiled residual/Jacobian evaluators create large amortized gains in repeated numerical use;
- proof and receipt overhead remains bounded and can be amortized/cached safely;
- small one-off Python scalar calls may initially remain slower because compatibility-shell correctness is prioritized.

No speedup number is promised before paired, parity-gated evidence exists.

## 19. Campaign discrepancy policy

A mismatch blocks the corresponding closure gate unless the immutable campaign/profile manifest explicitly classifies it as out of scope. “Will fix later” is not a passing status.

Each discrepancy includes:

- minimal reproducer;
- upstream/candidate observations;
- exact comparator/environment;
- object/math/runtime/security severity;
- affected campaign stage and claim;
- closure test;
- owner and status.

The campaign may continue on independent work while a discrepancy remains, but C11 cannot close around a blocking mismatch.

## 20. Agent task decomposition

Each campaign task must be small enough to complete and review independently. Examples:

- implement `TermId` canonical preimage for integer/symbol/add terms;
- build one custom `FunctionClass` differential fixture family;
- implement factorization product/multiplicity verifier reference lane;
- add cancellation safe point after each modular-prime batch;
- implement checkpoint manifest bounded decoder;
- build semantic patch conflict fixture for same print/different domain;
- add FrankenNumPy residual evaluator reference vectors.

Invalid tasks:

- “implement SymPy core”;
- “make factorization fast”;
- “add proof system”;
- “finish compatibility.”

Every task names its acceptance commands, registry effects, forbidden shortcuts, and independent gate owner.

## 21. Campaign rollback policy

If a load-bearing assumption fails:

- preserve the failed prototype and minimized evidence bundle;
- mark affected workstreams/claims blocked, not complete;
- update architecture decision and risk records;
- do not paper over the failure with a hidden upstream fallback;
- prefer narrowing/layering changes over broad API churn;
- retain profile fixtures so a revised design must satisfy the same contract.

Examples that trigger architecture review:

- a real SymPy subclass behavior cannot be represented by the shell protocol;
- stable term identity requires profile-specific Python state;
- a verifier is not meaningfully independent of its generator;
- cancellation-safe checkpoint publication cannot be expressed through the proposed ledger boundary;
- CPython bridge overhead makes eligible fine-grained lowering counterproductive across the hero corpus;
- ball arithmetic substrate cannot support the claimed certificate without unsafe/FFI compromise.

## 22. Campaign forbidden shortcuts

- implementing broad stubs before the vertical slice;
- replacing arbitrary Python subclasses with a generic native class;
- calling shell-only code an upstream fallback;
- importing upstream SymPy in product runtime;
- collapsing held and canonical representations;
- converting unknown assumptions to false;
- using finite differences as proof of a derivative;
- using sampled evaluations as polynomial identity;
- product-only check called irreducible factorization;
- allowing the first portfolio candidate to win before verification;
- consuming verifier-reserved budget;
- leaving loser/remote tasks alive after return;
- calling RaptorQ recovery integrity or proof;
- accepting a repaired checkpoint without canonical digest and dependency validation;
- treating a remote worker's signature/consensus as mathematical evidence;
- admitting incompatible benchmark cases;
- changing comparator, evidence requirement, durability, or corpus to create a speedup;
- marking the campaign complete by prose or commit count.

## 23. Campaign completion statement

When all gates pass, the accurate statement is:

> FrankenSymPy has validated its core architecture on the Certified Jacobian Pipeline: a narrow SymPy-compatible Python object slice lowers into a deterministic native kernel, produces independently verified symbolic results, compiles certified/reference numeric evaluators, survives cancellation and repaired checkpoint resume, and replays through an agent-native protocol.

It would still be inaccurate at that stage to claim complete SymPy replacement, general proof of all results, or universal performance superiority. Those claims belong to later workstreams and immutable profile certification.