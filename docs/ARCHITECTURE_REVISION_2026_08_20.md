# FrankenSymPy architecture revision — 2026-08-20

**Status:** normative addendum to the comprehensive plan  
**Purpose:** integrate the deep donor audits and close trust, runtime, packaging, performance, and release gaps found after the first plan

## 1. Revised thesis

FrankenSymPy is not merely a faster reimplementation of symbolic algorithms. It is a layered system in which:

- a SymPy-compatible Python shell preserves named observable profiles;
- an immutable semantic core owns terms, domains, binders, assumptions, branch policy, exact numbers, and canonical encodings;
- condition-aware portfolios generate candidates using deterministic baselines, safe optimized kernels, speculative pure work, and explicit fallbacks;
- small independently embeddable verifier crates check portable certificates;
- a content-addressed artifact plane moves and stores complete verification closures;
- versioned workspaces let humans and agents fork, collaborate, rebase, merge, and time-travel without confusing branch state with truth;
- local graph algorithms and optional FrankenGraphDB projections expose dependencies and provenance without becoming authority;
- optional formal projections add theorem-prover evidence without making native verification depend on Lean;
- canonical local/Doodlestein gates govern claims and releases.

The differentiator is the composition of compatibility, performance, evidence, determinism, collaboration, and portability under strict boundaries.

## 2. System layers

```text
L7  Python distribution / resolver-compatible shell
    ├── exact profile observations
    ├── interpreter-local wrappers, GC, exceptions, hooks
    └── exactly-once effect commit

L6  Public operation and portfolio layer
    ├── diagnostics and eligibility
    ├── expected-loss selection
    ├── pure speculative generators
    └── decision receipts and fallback DAGs

L5  Domain engines
    ├── simplify/rewrite/e-graph
    ├── polynomial and exact linear algebra
    ├── calculus, roots, series, logic, combinatorics
    └── code generation and numeric compilation

L4  Semantic core
    ├── immutable term/object model
    ├── domain, binder, substitution, assumption, branch calculus
    ├── exact arithmetic and representation views
    └── canonical identity/encoding

L3  Portable verification plane
    ├── claim/certificate schemas
    ├── factor, roots, calculus, linear, rewrite, graph verifiers
    └── no runtime, Python, database, network, or generator dependency

L2  Artifact and workspace plane
    ├── FMAP manifests and content-addressed closure
    ├── bounded streaming, delta, optional RaptorQ repair
    ├── MVCC/SSI semantic transactions and branch history
    └── append-only evidence/publication

L1  asupersync orchestration
    ├── owned regions, cancellation/drain/finalize
    ├── capabilities, budgets, lab replay
    ├── remote pure subgoals and artifact transfer
    └── protected verifier reserve

L0  Safe Rust, closed dependency admission, pinned nightly, local release gates
```

Dependencies point downward. Portable verifier crates occupy a deliberately narrow subgraph of L4/L3 and never depend upward.

## 3. Non-negotiable corrections to the initial plan

### 3.1 Verifiers are a public compatibility surface

Separate verifier crates are not just internal modularity. They promise that users can embed tiny checkers independently of generators and hosted infrastructure. The promise is enforced through profiles, dependency-closure gates, minimal-consumer examples, bounded decoders, and stable typed refusal semantics.

### 3.2 Generator complexity is untrusted by default

A generator may use portfolios, remote workers, learned retrieval, e-graphs, lattice methods, RaptorQ artifacts, or formal tactics. It nominates candidates. It cannot mint accepted claims.

### 3.3 Python effects are not speculative work

Arbitrary Python hooks may mutate state or perform I/O. Unknown/effectful hooks execute exactly once after pure preparation and route selection. Interruption may leave an unknown effect outcome and is never automatically retried.

### 3.4 Graph indexes are projections

FrankenGraphDB may index immense proof and derivation histories, but the authoritative source is immutable object closure plus workspace publication. Index misses do not prove absence without completeness evidence.

### 3.5 Formal proof is additive

Lean-side checking strengthens selected results through a projection receipt. It does not replace the native portable verifier or become a dependency of ordinary verification.

### 3.6 Compatibility includes the distribution resolver

Installing a differently named wheel that writes `sympy/` is not enough. Certified drop-in channels must satisfy `Requires-Dist: sympy`, own paths exclusively, and pass install/upgrade/uninstall/rollback matrices.

### 3.7 Statistical language is gated

Conformal and e-process claims require their actual assumptions, filtration, construction, optional-stopping scope, censoring, reset, and multiplicity policy. Otherwise they are honest heuristics.

### 3.8 Local evidence controls releases

GitHub workflows may project local commands, but Doodlestein/local gate receipts control release state. The repository does not rely on hosted Actions availability or authority.

## 4. First architecture-proving vertical slice

The first implementation campaign should prove the boundaries with a narrow but complete polynomial system rather than spread thinly across SymPy.

### 4.1 Semantic scope

- integers and normalized rationals;
- symbols and immutable expression DAG;
- univariate polynomials over `ZZ` and `QQ`;
- domains/coercions;
- substitution and canonical ordering;
- factor, expand, gcd, square-free decomposition;
- minimal Python wrappers and printers.

### 4.2 Certificate scope

- polynomial product/reconstruction;
- factorization with unit/content/domain;
- gcd and Bézout identity;
- exact evaluation identity;
- graph topological/cycle/path certificates used by dependencies.

### 4.3 Generator scope

- deterministic reference arithmetic and polynomial algorithms;
- modular/Hensel factorization candidate;
- subresultant and modular gcd candidates;
- simple condition diagnostics and fallback;
- pure asupersync parallel modular work.

### 4.4 Integration scope

- portable `fsym-cert-factor` consumer with no hosted dependencies;
- FMAP verifier-complete bundle;
- one versioned workspace branch and publication transaction;
- one strict Python compatibility profile/corpus;
- one optional Lean factorization projection;
- local gate receipt through the Doodlestein contract;
- Apple Silicon and x86-64 full-operation benchmark receipts.

This slice is successful only when the full path works end-to-end without mocks: Python request → immutable core → generator → certificate → independent verifier → artifact bundle → publication → replay/consumer check.

## 5. Proposed crate topology for the slice

```text
fsym-primitive          core IDs, bounded bytes, canonical primitives
fsym-integer            owned exact integer arithmetic
fsym-rational           normalized rational arithmetic
fsym-term               immutable term DAG and canonical encoding
fsym-domain             ZZ/QQ domain and coercion calculus
fsym-poly               semantic polynomial objects and reference kernels
fsym-cert-core          claim/certificate primitives and limits
fsym-cert-factor        factor/gcd/product reference verifier
fsym-graph-core         deterministic local graph primitives/certificates
fsym-artifact           FMAP objects/manifests, no network
fsym-workspace          in-memory versioned workspace semantics
fsym-poly-portfolio     diagnostics, generators, fallback; depends on verifier API only for checking candidates
fsym-runtime            asupersync orchestration and budgets
fsym-python             Python profile shell and effect boundary
fsym-formal-lean        optional formal projector/foreign checker adapter
fsym-cli                inspection, verify, bundle, plan-only commands
```

The verifier path does not depend on `fsym-poly-portfolio`, `fsym-runtime`, `fsym-python`, `fsym-workspace`, or `fsym-formal-lean`.

## 6. Revised implementation gates

### G0 — constitution executable

- registries parse;
- cross-cutting DAG passes;
- toolchain/dependency policy passes;
- crate dependency graph and verifier ceiling are machine-checked;
- no implementation claims.

### G1 — portable verification spike

- canonical integer/rational/polynomial encodings;
- factor/gcd certificate schemas;
- bounded reference verifiers;
- mutation corpus;
- minimal external consumer and Wasm build;
- no generator/runtime/Python dependencies.

### G2 — native polynomial vertical slice

- reference generators;
- modular optimized candidates;
- asupersync deterministic portfolio;
- verifier reserve and cancellation/drain;
- FMAP bundle and in-memory workspace publication;
- complete-operation benchmarks.

### G3 — strict Python slice

- package/import surface for the selected corpus;
- identity, exceptions, printers, pickle, callbacks, GC;
- isolated differential oracle;
- resolver-transparent test channel;
- classic/free-threaded/subinterpreter statuses stated honestly.

### G4 — collaboration and formal evidence

- branch-per-agent workspaces;
- semantic merge/rebase certificates;
- optional FrankenGraphDB projection;
- Lean factorization projection and foreign check;
- remote pure subgoals/artifact transfer.

### G5 — expand CAS surface

Expand operation families only when each gains reference semantics, compatibility inventory, portfolio, verifier/evidence plan, performance regimes, and cross-cutting obligations.

## 7. Open decisions that remain G0 blockers

- canonical digest/hash algorithms and domain separation;
- exact integer limb representation and first threshold policy;
- durable encoding versioning and migration;
- first CPython/SymPy/platform profile matrix;
- safe CPython binding crate admission;
- exact distribution version mapping for a resolver-transparent `sympy` replacement;
- whether any FrankenSuite exact arithmetic crate is reused or all arithmetic starts in-tree;
- first Lean library/environment root;
- benchmark corpora and controlled machine identities;
- Doodlestein signing keys, provenance schema, and promotion channel.

These are explicit decisions, not blanks for implementation agents to fill independently.

## 8. Status honesty

This revision improves and constrains the plan. It does not claim the described engine exists. Registries remain `planning`; cross-cutting release blockers remain incomplete; code appears only when a gate transitions with evidence.
