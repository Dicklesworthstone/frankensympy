# The FrankenSymPy constitution

**Status:** normative architectural law  
**Applies to:** source, tests, registries, documentation, benchmarks, packages, releases, agents, and integrations

## Preamble

FrankenSymPy is intended to become an independently implemented, memory-safe, agent-native symbolic mathematics system and a certified drop-in replacement for named SymPy profiles. Its value depends on four properties remaining inseparable:

1. compatibility with the Python object contract users actually depend on;
2. mathematical truth and evidence stronger than opaque algorithm output;
3. radical performance and operational capability;
4. honest, executable claims about what exists.

A design that sacrifices any one of these to make another look complete is not FrankenSymPy.

## Article I — Current reality

At the time this constitution is adopted, the repository contains a public architecture and implementation plan. It does not contain a certified SymPy replacement, production proof kernel, completed native algebra engine, or demonstrated performance win.

All capability claims remain `planned` unless `registries/claims.toml` says otherwise and resolves them to same-commit gate artifacts. Documentation volume and commit count are not implementation evidence.

## Article II — The seven architectural bets

### Bet 1 — Dual-lane compatibility

Python-visible behavior belongs to a real profile-compatible Python shell. Mathematical acceleration belongs to a deterministic native kernel. Eligible regions lower explicitly; arbitrary Python extensions remain valid and conservative.

This bet rejects the false simplicity of exposing one opaque Rust `Expr` object and calling it a SymPy replacement.

### Bet 2 — Three-graph truth model

FrankenSymPy maintains separate authoritative representations for:

- the Surface Object Graph: what the user constructed and Python can observe;
- the Semantic Term DAG: canonical domain-aware mathematical terms;
- the Derivation Evidence Graph: transformations, assumptions, certificates, proofs, decisions, and receipts.

Surface form is not semantic identity. Search history is not proof. Mathematical equivalence is not object-model compatibility.

### Bet 3 — Domain-explicit exactness

Every semantic operation occurs in a typed domain, assumptions context, branch policy, and rule universe. Unknown facts remain unknown. Contradictions remain visible. Approximation is never smuggled into exact work.

### Bet 4 — Proof-carrying portfolios

Complex algorithms may be heuristic, adaptive, randomized, parallel, remote, or learned. Their outputs are candidates until a claim-specific independent verifier grants an evidence class.

The generator may be alien technology. The acceptance boundary must be small enough to trust and brutal enough to reject it.

### Bet 5 — Deterministic resource sovereignty

All controlled work is region-owned through asupersync, receives typed nested budgets, publishes through two-phase effects, and can be cancelled/drained without orphan work. Deterministic and replay modes are first-class.

Resource exhaustion, timeout, refusal, and mathematical inconclusiveness are distinct outcomes.

### Bet 6 — Agent-native symbolic state

Terms, contexts, claims, proofs, counterexamples, patches, checkpoints, and branches have stable structured identities and versioned protocols. Agents collaborate through semantic state and verifier-checked merges, not fragile strings or chat transcripts.

### Bet 7 — Recoverable computation fabric

Expensive symbolic work can be checkpointed, persisted, distributed, indexed, repaired, and replayed without allowing storage, workers, graph reachability, or RaptorQ decoding to define truth.

## Article III — Compatibility law

1. Compatibility is always against an immutable named profile.
2. The first target is SymPy 1.14.0 at the pinned source commit in the profile registry.
3. A moving upstream branch may detect drift but cannot certify anything.
4. The `frankensympy` package may coexist with upstream and expose preview/native APIs.
5. The `frankensympy-dropin` distribution owns the top-level `sympy` namespace and is published only when its profile gates pass.
6. A certified product path contains no hidden runtime dependency or fallback to upstream SymPy.
7. Shell-only implementation is permitted; upstream-executed fallback is not.
8. Type, class, metaclass, MRO, module path, signature, warning, exception, printer, pickle, mutation, evaluation, hashing, sorting, `args`, and `func` behavior are compatibility dimensions when observable.
9. Mathematical equivalence cannot excuse a wrong Python object contract.
10. Hardened/native improvements cannot silently alter strict-profile behavior.

## Article IV — Object-model law

1. The Python shell is authoritative for Python identity and extensibility.
2. Arbitrary subclasses of supported SymPy base classes remain ordinary Python classes.
3. Built-in shell objects may carry native handles only as internal acceleration state.
4. Exact-class and hook-override checks govern native fast paths.
5. Unknown nodes are opaque and conservative by default.
6. Opaque nodes are not assumed commutative, pure, deterministic, terminating, thread-safe, or serializable.
7. Held forms preserve order and multiplicity until an explicit surface transformation contract applies.
8. Mutable objects are shell-owned cells with immutable generation-stamped snapshots for native work.
9. Alpha-equivalence, structural equality, semantic equality, dummy equality, and Python equality remain distinct.
10. Lifting invokes profile-correct construction and accounts for evaluation and side effects.

## Article V — Identity law

1. Every durable semantic object has a typed content identity.
2. `TermId` is distinct from `SurfaceId`, Python hash, arena handle, pointer, database row, and graph vertex.
3. Stable identities derive only from canonical semantic preimages and versioned digest domains.
4. Wall time, process IDs, task order, worker arrival, memory addresses, and cache state never enter semantic identity.
5. Digest equality is confirmed by canonical payload at trust boundaries.
6. Changing identity-relevant schema creates a new identity universe.
7. Cross-tenant content identity follows explicit privacy policy; an ID is never an authorization token.

## Article VI — Assumptions and domain law

1. Domain, sort/kind, assumptions context, compatibility facts, and branch policy are separate concepts.
2. Internal truth distinguishes entailed true, entailed false, unknown, and contradictory.
3. `Unknown` is never normalized to false or true.
4. A contradictory context cannot prove arbitrary claims unless an explicit logic mode permits it.
5. Every rule and algorithm declares applicable domains and side conditions.
6. Coercions declare totality, exactness, injectivity, loss, and assumptions.
7. Coercion selection is deterministic under a fixed policy.
8. Lossy conversion emits an explicit receipt or is refused.
9. Branch-sensitive transformations carry branch policy and proof obligations.
10. Cache keys include every relevant domain/context/profile/rule/precision dependency.

## Article VII — Evidence law

1. The result value and evidence class are separate fields.
2. Evidence classes are typed and non-inflatable.
3. `KernelProved`, `CertificateVerified`, `ExactCrossChecked`, `CertifiedNumeric`, `OracleConformant`, `UserAsserted`, and `HeuristicCandidate` are not interchangeable.
4. Conditional, inconclusive, refused, cancelled, timed-out, resource-exhausted, unsupported, and fault outcomes are not proofs.
5. A verifier grants evidence only for the exact claim, domain, context, schema, and side conditions it checked.
6. Stronger evidence requests cannot be satisfied from weaker cache entries.
7. Search traces, model confidence, worker reputation, signatures, majority votes, and origin do not establish mathematical truth.
8. Oracle parity establishes compatibility behavior, not mathematical correctness.
9. Numerical sampling does not establish exact identity.
10. Stored `verified` flags have no authority without canonical validation and verifier policy.

## Article VIII — Proof-kernel law

1. The trusted proof base remains as small and layered as practical.
2. Generator crates do not sit in verifier dependency trees.
3. Claim-specific verifiers are simpler and independently testable.
4. Optimized verifiers retain a scalar/reference lane.
5. Every proof constructor and certificate family has adversarial and mutation tests.
6. Unknown claim/certificate/verifier schemas fail closed.
7. A certificate for a weaker claim cannot certify a stronger statement.
8. Completeness, uniqueness, irreducibility, convergence, and generality are separate obligations.
9. Proof compression preserves everything required by the verifier.
10. A false accepted exact claim is an existential-severity incident.

## Article IX — Rewriting law

1. Authoritative rewrite rules are versioned registry objects, not anonymous unreviewed closures.
2. Every accepted exact rewrite emits or reconstructs a verified derivation.
3. Side-condition uncertainty creates a guarded relation or refusal, not unconditional equality.
4. Canonicalization is operator- and domain-specific.
5. There is no universal context-free “simplest expression.”
6. Native simplification declares a cost vector and evidence minimum.
7. Equality saturation is local, typed, bounded, and proof-producing.
8. E-class unions retain justification.
9. Rule order and extraction tie breaks are deterministic under strict policy.
10. Expression and proof growth are budgeted before publication.

## Article X — Algorithm-portfolio law

1. A planner may choose or race strategies; it cannot certify their answers.
2. Every consequential plan emits a decision card.
3. Loss policies encode catastrophic asymmetry between false results and slow/inconclusive results.
4. A safe deterministic baseline exists for every adaptive portfolio before rollout.
5. The verifier receives protected resources that generators cannot consume.
6. Candidate publication and accepted publication are separate phases.
7. Completion order affects latency, never truth.
8. Random streams are counter-partitioned or recorded independently of schedule.
9. Selector learning uses verified outcomes only.
10. The selector cannot control comparators, evidence requirements, benchmark admission, or release gates.

## Article XI — Concurrency law

1. Asupersync is the sole async/concurrency runtime.
2. Every spawned controlled task has one owning region.
3. There are no detached background simplifiers, cache warmers, proof searches, repair jobs, or remote leases.
4. Cancellation is request → drain → finalize.
5. Return requires quiescence or a typed non-cooperative boundary outcome.
6. Shared effects use reserve/prepare then verify/commit.
7. Cancellation cannot publish partial verified state.
8. Budget counters survive fallback and include transient work.
9. Verifier, output, printer, proof, persistence, and remote resources are independently bounded.
10. Universal cancellation latency is never claimed through arbitrary non-cooperative Python or foreign code.

## Article XII — Determinism law

1. Determinism is always scoped: semantic, byte, decision, trace, or compatibility behavior.
2. Fixed inputs include profile, context, domains, registries, policy, seed, and mode.
3. Hash iteration, task completion, filesystem order, plugin load order, and worker arrival are never implicit semantic tie breaks.
4. Strict mode freezes decisions and output extraction.
5. Replay mode records adaptive decisions and random streams.
6. Latency-adaptive mode may change execution but remains verifier-governed.
7. Architecture-optimized and reference paths produce the same stable semantic identities and evidence decisions.
8. Python process hash behavior remains profile-correct but separate from stable IDs.
9. Deterministic replay is operational evidence, not mathematical proof.
10. Production and lab runtime-facing code paths are the same interfaces.

## Article XIII — Persistence law

1. Persistence is optional and outside the in-memory algebraic hot path.
2. Authoritative durable objects are immutable and content-addressed.
3. Database primary keys never define mathematical identity.
4. Workspace computations read immutable universe snapshots.
5. A running computation is never silently rebased to new assumptions, rules, or profiles.
6. Candidate and verified cache namespaces remain separate.
7. Cache reads validate complete universe keys and evidence policy.
8. Checkpoints are typed normalized algorithm states, not memory dumps.
9. Crash recovery cannot invent, substitute, or silently change mathematical state.
10. Ephemeral and persistent modes return identical semantic results for the same universe and policy.

## Article XIV — RaptorQ law

1. RaptorQ protects selected valuable byte artifacts, not every term or cache entry.
2. Artifact policy considers recomputation value, retention horizon, failure domains, and overhead.
3. RaptorQ reconstruction produces candidate bytes only.
4. Canonical digests establish expected content identity.
5. Authorization/signatures establish origin where required.
6. Schema/invariant checks establish well-formedness.
7. Mathematical verifiers establish evidence.
8. These stages are never collapsed in code, docs, badges, or metrics.
9. Repair overhead and loss assumptions appear in benchmarks and claims.
10. A durability claim requires crash/loss/corruption evidence, not merely an encoder implementation.

## Article XV — Monitoring law

1. E-process and conformal mechanisms monitor streams of operational evidence.
2. Every monitor states its null/model/filtration assumptions, subgroup policy, reset policy, and action thresholds.
3. Monitors may pause, quarantine, revert, increase shadow verification, or trigger investigation.
4. Monitors cannot prove or refute an individual mathematical claim.
5. Failed, refused, cancelled, and timed-out outcomes cannot be selectively omitted from monitored streams.
6. Adaptive subgroup selection and resets are governed explicitly.
7. Monitor changes are registry-versioned.
8. Alarm response is deterministic or receipt-recorded.

## Article XVI — Distributed-work law

1. Remote workers are untrusted candidate generators.
2. Work packets are bounded, content-addressed, capability-scoped, and universe-bound.
3. Workers cannot browse the workspace, alter registries, publish branch heads, or write verified caches.
4. Responses are bounded and locally canonicalized.
5. The coordinator verifies locally before publication.
6. Duplicate, late, equivocal, or malicious responses cannot double-publish or upgrade evidence.
7. Worker identity, reputation, signatures, and consensus do not replace verification.
8. Sensitive objects remain local unless explicit capability permits export.
9. Revoked leases lose publication rights.
10. Network failure changes execution outcomes, not mathematical truth.

## Article XVII — Graph-index law

1. FrankenGraphDB is an optional projection over authoritative terms, derivations, events, and receipts.
2. Graph indexes are versioned and rebuildable.
3. Graph vertex/edge IDs never replace typed authoritative object IDs.
4. Reachability does not imply logical entailment.
5. Query results are revalidated before proof/cache use.
6. Deleting and rebuilding an index cannot change accepted results.
7. Graph branch operations cannot bypass ledger/proof validation.

## Article XVIII — Agent-workspace law

1. Agent requests and patches name immutable universe IDs.
2. Printed strings and chat transcripts are views, not authoritative mathematical state.
3. Semantic patches contain typed operations and preconditions.
4. Branch merge verifies imported accepted edges.
5. Candidates and conditional work remain labeled across merge.
6. Same print with different class/domain/context is a conflict.
7. Counterexamples are first-class bundles with exact/certified evaluation.
8. Work packets are bounded and gate-complete.
9. Natural-language completion assertions have no authority.
10. Mathematical state must be replayable without the conversational transcript.

## Article XIX — Dependency law

1. The dependency universe remains minimal and explicit.
2. Asupersync and narrow Franken-suite adapters are preferred.
3. Foundational external crates require written admission review.
4. No second async runtime is admitted.
5. No C/C++ CAS or arbitrary-precision engine is hidden behind FFI.
6. Direct hand-written CPython C-API code is prohibited; the contained bridge uses an audited safe Rust layer if an extension is used.
7. Ordinary crates forbid unsafe code.
8. Any unsafe optimization island requires a safe total API, reference lane, audits, fuzzing, and architecture CI.
9. Optional integrations cannot change core term/proof semantics.
10. Dependencies are pinned and included in release provenance.

## Article XX — Performance law

1. Correctness and compatibility admission precede timing admission.
2. A benchmark case that fails semantic comparison cannot enter a speed aggregate.
3. The live incumbent is measured in the same invocation or controlled paired run.
4. Profile, domain, assumptions, evidence requirement, budgets, cache state, durability, and thread policy match.
5. Reports include outcome mix, memory, tails, proof/verifier cost, and amortization.
6. Warm-candidate/cold-incumbent comparisons are forbidden.
7. Self-speedups without a live incumbent are maintenance evidence, not leapfrog claims.
8. Selectors cannot see benchmark identities or train on evaluation holdouts.
9. A performance win cannot weaken evidence, comparators, security, durability, or compatibility.
10. Universal superiority is never inferred from named workload wins.

## Article XXI — Security law

1. Symbolic inputs and artifacts are treated as potentially hostile programs/data.
2. Decoders preflight all untrusted sizes and recursion before allocation.
3. Expression, proof, printer, output, callback, persistence, repair, and remote amplification are independently budgeted.
4. Native formats never execute code during decode.
5. Pickle is an explicit unsafe compatibility capability, never a general protocol.
6. Generated code is produced but not executed without capability.
7. Content IDs are not access tokens.
8. Multi-tenant deduplication/equality leakage follows explicit privacy policy.
9. Rust memory-safety claims exclude CPython and arbitrary third-party extensions.
10. Internal faults cannot return candidates as accepted values.

## Article XXII — Conformance law

1. The upstream oracle and FrankenSymPy run in isolated processes/environments.
2. Complete public-surface inventory is structural CI input.
3. The upstream suite is necessary but insufficient.
4. Generated differential grammars cover types, domains, assumptions, held forms, binders, branches, custom subclasses, mutation, and invalid calls.
5. Every fixture names its comparator before execution.
6. Comparator weakening is a profile change and requires review.
7. Metamorphic tests cannot replace missing exact surface comparisons.
8. Ecosystem claims name exact package/notebook corpora and versions.
9. Mismatches are minimized and ledgered, not hidden or regenerated away.
10. Certification reruns the full matrix on one release commit.

## Article XXIII — Documentation and claims law

1. Target, implemented, validated, and certified states are written separately.
2. Every public present-tense capability claim has a claim ID and evidence artifact.
3. Planned architecture is never written as shipped behavior.
4. Badges resolve to same-commit machine artifacts.
5. A file, API name, dormant code path, or commit count does not establish capability.
6. Old source/profile pins are preserved in history and updated explicitly.
7. Performance numbers include their exact corpus/mode/hardware/provenance.
8. Repair, monitoring, safety, and compatibility claims state their scope and exclusions.
9. Documentation regressions can block release.
10. The claims linter may reject prose even when code passes.

## Article XXIV — Workstream law

1. The machine-readable workstream DAG is authoritative for structural dependencies.
2. Workstream closure requires all dependencies and a gate bundle.
3. Generator and independent gate ownership are separated for high-value claims.
4. Broad work is converted into bounded tasks only through the Beads conversion gate.
5. Every task names acceptance commands, tests, artifacts, failure behavior, claim/discrepancy effects, and forbidden shortcuts.
6. Structural graph changes are single-writer and acyclic.
7. Retired IDs remain tombstoned.
8. Milestones are gates, not percentages or dates.
9. Partial progress remains partial; it is not rounded up.
10. The first vertical campaign precedes broad API expansion.

## Article XXV — The forbidden-shortcut schedule

The following are constitutional violations:

1. hidden upstream SymPy runtime fallback;
2. one opaque Rust object advertised as a drop-in Python object model;
3. strings or printer output as semantic IR or identity;
4. canonicalization that silently destroys held/custom surface behavior;
5. `Unknown` treated as true or false;
6. lossy lowering/coercion without an explicit receipt;
7. conditional rewrite applied unconditionally;
8. heuristic, posterior, e-process, oracle, sampled numeric, worker, or majority evidence promoted to exact proof;
9. generator self-verification or verifier → generator dependency;
10. first-completed speculative candidate publication;
11. detached or orphan work;
12. fallback budget reset;
13. process memory dump as checkpoint;
14. database row, graph reachability, or cache flag as truth;
15. RaptorQ decode success called integrity, authenticity, or proof;
16. repaired artifact use before digest/schema/dependency/evidence validation;
17. remote worker direct publication of verified state;
18. graph index as the sole authoritative copy;
19. arbitrary pickle/code execution through the normal protocol;
20. C/C++ CAS or big-number FFI hidden behind a safe wrapper;
21. benchmark cases admitted before parity;
22. warm/cold, mode, evidence, durability, or corpus asymmetry in comparisons;
23. test/comparator/golden weakening to land a feature;
24. API stubs counted as compatibility;
25. documentation written in a stronger tense than the claims registry;
26. milestone closure by prose, commit count, or percentage;
27. an agent weakening the only gate judging its own work;
28. broad surface expansion before the Certified Jacobian Pipeline validates the architecture.

## Article XXVI — Amendment procedure

A constitutional change requires:

1. a dedicated commit or clearly isolated diff;
2. affected source audit, claims, workstreams, risks, and profile updates;
3. explanation of which prior invariants change and why;
4. migration plan for existing IDs, artifacts, work packets, and implementations;
5. new or changed objective gates;
6. preservation of historical text in Git;
7. adversarial review focused on whether the amendment legalizes a shortcut after difficulty was encountered.

Emergency security fixes may temporarily restrict capability without weakening compatibility claims; permanent behavior changes still follow the profile/amendment process.

## Article XXVII — Definition of constitutional success

FrankenSymPy succeeds when it can be faster, broader, more reliable, more inspectable, and more agent-capable than conventional symbolic systems while making it harder—not easier—for a false answer, incompatible object, orphan task, corrupted artifact, benchmark trick, or inflated claim to pass as success.
