# FrankenSymPy architecture review — round 3

**Review date:** 2026-08-20 (America/New_York)  
**Reviewed parent:** `8f040b0e5ecc8fe0e2fee559433e6a20a2c9d7f7`  
**Review type:** architecture closure, live-history verification, and status-honesty audit  
**Verdict:** architecture planning closure achieved; implementation and release gates remain open

## 1. Why this review exists

A prior completion report claimed that thirteen incremental commits had been pushed when live `main` contained only two of the described commits. That report was false. This review treats the live GitHub branch, commit objects, and tree contents as authoritative and records the corrective sequence explicitly.

The correction does not attempt to disguise planning documents as implemented software. The repository is still at the planning/architecture stage. What has now landed is a substantially deeper, cross-linked, machine-checkable design corpus and the first executable structural validators.

## 2. Corrective commit sequence reviewed

1. `7c63e33f12ff7f0ffbd6889be9aa1ffc45e7bd13` — portable verifier compatibility promise and mathematical artifact plane.
2. `8af0e8e127290ec12f1d66aa2045dba7d65a3362` — serializable symbolic workspaces, semantic witnesses, and proof-aware merge.
3. `9ff62c723ea83686515a5717033e160eb1cb764d` — FrankenGraphDB/FrankenNetworkX deep audit and deterministic graph substrate.
4. `2aadbeac7745d9be3c676a160974deadfbe7b553` — FrankenLean deep audit and native-first formal proof interoperability.
5. `ac7d5fe0fe017df33f1e9d70e8d575b502dcbeba` — FrankenNumPy/FrankenSciPy deep audit and condition-aware portfolios.
6. `7d75c9c37ce2f7d173609b390007906d36a851bf` — safe-Rust dependency constitution, exact nightly pin, and local-first release authority.
7. `fc2022005f3f2422efc6db7c7fadd40c7c81cd16` — exactly-once Python effect boundary and resolver-aware packaging.
8. `33f140344b316426881005dc05cf09eee3d826da` — claim/evidence lattice and governed adaptive monitoring.
9. `84739cb79ae9665855cb4e71578c1d682e7f2873` — safe-Rust performance architecture and kernel registry.
10. `857603f87c5b12b14b2fb4dfec011e4ab524bc59` — cross-cutting obligation DAG and executable validator.
11. `8d7486321380a4f1c6187bbc5eb6a578fa5cb24a` — root architecture index, normative revision, document registry, and Cargo metadata.
12. `8f040b0e5ecc8fe0e2fee559433e6a20a2c9d7f7` — executable registry, donor, safety, verifier-ceiling, and performance validators plus `scripts/check.sh`.

This review is the payload of the thirteenth corrective commit. Its own SHA is intentionally not self-embedded; the branch history is the authority for that value.

## 3. Review method

The review used four evidence classes and keeps them separate.

### 3.1 Live repository evidence

After each corrective commit, the GitHub branch ref and commit/tree object were fetched from the live repository. Fast-forward updates used `force = false`.

### 3.2 Source-pinned donor audit

The donor registry binds the design audits to exact commits for asupersync, FrankenSQLite, FrankenGraphDB, FrankenNetworkX, FrankenLean, FrankenNumPy, and FrankenSciPy. The audits classify mechanisms as adopt, selective-adopt, adapt, research, or reject.

### 3.3 Local structural validation

The Python and shell validators were exercised against a locally assembled mirror containing the live new registries and the actual verifier-profile registry, with placeholders only for older documents whose existence was independently confirmed in the live tree. The validation checked registry shape, cross-links, pins, policy invariants, deterministic obligation topology, tool syntax, and deliberate refusal of unavailable release profiles.

### 3.4 Explicitly absent evidence

No fresh network clone was possible in the execution environment. Cargo was unavailable there, and no Rust targets exist yet. Therefore this review does **not** claim:

- `cargo check`, Clippy, or Rust tests passed;
- a portable verifier compiled or ran;
- a Python extension or wheel was built;
- SymPy conformance was executed;
- Doodlestein ran a release matrix;
- packaging, signing, reproducibility, benchmark, fuzz, lab, or formal-proof gates passed;
- GitHub Actions supplied authoritative evidence.

The local script intentionally exits with status 2 for those profiles instead of pretending they passed.

## 4. Major architecture closures

### 4.1 Independent verifier closure

The verifier crates are now a public compatibility promise. A supported claim capsule must be checkable without the generator, planner, Python shell, asupersync runtime, database, graph index, network, transfer system, telemetry, or formal prover. Reference verification is synchronous, bounded, deterministic, and fail-closed on unknown critical schema.

This is the most important trust-boundary improvement in the revision.

### 4.2 Mathematical artifact plane

The design separates semantic identity from physical transfer. Claims, certificates, terms, domains, environments, and proof nodes are immutable content-addressed objects. FMAP can use streaming, delta synchronization, content-defined chunks, and optional RaptorQ repair, but repaired bytes still require exact content and mathematical verification.

### 4.3 Serializable semantic workspaces

Mutable authority is limited to versioned workspace roots: contexts, registries, profiles, branches, accepted derivations, and release manifests. Read, write, namespace, predicate, and absence witnesses protect against symbolic phantoms. Merge operates on semantic intents and certificates, never textual or raw-byte patches.

### 4.4 Deterministic graph substrate

Term, proof, assumption, invalidation, rule, domain, and workstream graphs receive explicit edge kinds and registered tie-break policies. Small graph certificate verifiers remain independent. FrankenGraphDB is an optional rebuildable projection and cannot establish identity, absence, or theoremhood.

### 4.5 Native-first formal interoperability

A theorem prover checks a projected formal statement. A checked projection receipt connects it to the native claim. Native verification remains sufficient and does not depend on Lean. Unrepresentable assumptions, branch policies, or semantic fields refuse projection.

### 4.6 Compatibility versus selection

Compatibility profiles define Python-observable behavior. Algorithm portfolios choose eligible work under explicit loss. Selection, posterior confidence, benchmark history, conformal scores, e-values, votes, and decision receipts cannot admit a mathematical claim.

### 4.7 Python effects and packaging reality

Unknown/effectful Python hooks execute exactly once after pure preparation and route selection. The plan now covers reentrancy, free-threaded CPython, subinterpreters, GC, weak references, finalizers, identity, warnings, and exception behavior.

Import compatibility, behavioral parity, and resolver/distribution compatibility are separately certified. A differently named wheel that writes `sympy/` is not treated as satisfying `Requires-Dist: sympy`.

### 4.8 Safe-Rust performance program

The design identifies concrete owned kernels and representation work for integers, rationals, polynomials, exact linear algebra, rewriting, term DAGs, graphs, parallelism, and Python boundary batching. Every optimized route has a scalar/reference route, regime axes, exact correctness gates, full-operation receipts, and rollback/quarantine requirements.

### 4.9 Local release authority

The release contract routes canonical gates through `scripts/check.sh` and Doodlestein-controlled machines. GitHub-hosted execution is non-authoritative. Immutable artifacts precede atomic channel promotion, and rollback/revocation preserve history.

## 5. Fresh-eyes defects found and corrected in this review

### 5.1 Incomplete architecture document registry

The first document registry indexed the new revision contracts but omitted several original normative documents. That would have made “all architecture documents parse and exist” a weaker claim than it sounded. The registry is now exhaustive over the original core corpus, the donor audits, the revision contracts, and this review.

### 5.2 Missing evidence registry metadata

`Cargo.toml` pointed to the claim registry and new claim lattice but omitted the pre-existing `evidence_classes.toml`. The metadata now includes it and the final architecture review path.

### 5.3 Validator self-check gap

The local gate ran validators but did not first compile all Python tools or syntax-check its own shell script. `scripts/check.sh` now performs both before registry validation.

### 5.4 Required-document set was implicit

A structurally valid but incomplete document registry could have passed. The bundle validator now requires the load-bearing original and revision document IDs explicitly.

## 6. Remaining blockers and risks

### 6.1 Canonical primitives are still undecided

Digest algorithms, domain separation, durable integer/rational/term encodings, schema migration, and object-kind IDs remain G0 decisions. Implementation should not fork these choices across crates.

### 6.2 Safe exact arithmetic is the schedule-critical risk

Rejecting native math backends and project-authored unsafe preserves the intended safety/ownership moat but makes BigInt, modular arithmetic, NTT/CRT, factorization, and exact linear algebra substantial implementation programs. The polynomial vertical slice must prove the representation and performance thesis before broad CAS expansion.

### 6.3 SymPy parity is enormous and dynamic

The first compatibility profile must be deliberately narrow and immutable. Custom classes, import order, assumptions hooks, printers, pickle behavior, and optional dependencies can invalidate broad parity language quickly.

### 6.4 CPython and resolver choices remain open

No safe CPython binding crate has been admitted. No replacement distribution version policy or certified package channel has been selected. These remain hard blockers for a genuine drop-in claim.

### 6.5 Structural validators are not implementation evidence

The landed tools verify planning coherence, not theorem correctness, runtime cancellation, package behavior, or performance. As crates land, they must be extended to inspect Cargo metadata/lock closure, executable schemas, compile-fail tests, mutation suites, artifacts, and Doodlestein receipts.

### 6.6 Release blockers remain incomplete

The cross-cutting registry contains 13 obligations, 12 of them release-blocking. All release-blocking obligations remain in planning status. This is correct and must not be “fixed” by changing status without evidence.

## 7. Gate results at review time

The current executable planning checks establish:

- all tracked TOML registries parse in the assembled validation mirror;
- the architecture document registry is exhaustive over the required set and its files/registries exist;
- donor commits are exact lowercase 40-hex pins and agree with dependency pins where shared;
- the exact nightly pin is coherent between toolchain and dependency registries;
- root Cargo metadata forbids unsafe and points to governing registries;
- portable verifier policy forbids generator/planner/Python/runtime/persistence/network dependencies;
- claim composition forbids implicit transitivity, vote/signature upgrades, monitor authority, and silent downgrade;
- graph, portfolio, Python effect, packaging, monitor, performance, and release policy invariants hold;
- the 13-node cross-cutting dependency graph is acyclic;
- all unavailable implementation/release profiles refuse explicitly.

These are structural planning results only.

## 8. Verdict and next action

The revised architecture is coherent enough to begin G0 closure and the G1 portable factor/gcd verifier spike. It is not honest to start broad parallel implementation until the canonical identity/encoding decisions, first CPython/SymPy profile, exact arithmetic ownership boundary, and Doodlestein evidence schema are frozen.

The next engineering milestone should be one end-to-end certified polynomial vertical slice:

```text
canonical ZZ/QQ polynomial objects
→ factor/gcd generators
→ portable certificates
→ independent fsym-cert-factor checker
→ FMAP bundle
→ in-memory workspace publication
→ minimal external consumer
→ narrow Python profile
→ local evidence receipt
```

That slice should be reviewed as implemented code and evidence, not as another architecture round.
