# Donor deep dive: FrankenSQLite

**Status:** normative source audit and architecture input  
**Pinned source:** `Dicklesworthstone/frankensqlite@abdc4dc89714e8816202fd0bf2e9cd969f6d493f`  
**Audit date:** 2026-08-20  
**Scope:** MVCC snapshots, SSI witnesses, deterministic rebase, semantic merge, trace-normalized intents, proof-carrying publication, history, provenance, time travel, and multi-root commit

## 1. Executive conclusion

The first FrankenSymPy plan described FrankenSQLite mainly as an optional ledger. That leaves most of its strongest ideas unused.

The deeper inheritance is a **semantic transaction model** for collaborative mathematical state:

- immutable object versions separated from mutable publication authority;
- exact snapshot and generation identities;
- read/write/predicate witnesses with no-false-negative discipline;
- conservative coarse conflict detection plus bounded refinement;
- commit-time validation with explicit conflict witnesses;
- deterministic replay of pure declarative intents;
- semantic merge over parsed objects, never raw byte patches;
- trace-monoid commutativity and canonical Foata normal forms;
- merge certificates and circuit breakers;
- reservation/coalescing/helping for expensive witness publication;
- append-only history with exact retention horizons;
- time-travel snapshots;
- why/how provenance semirings;
- local-to-global consistency obstruction checks;
- two-phase publication across multiple authoritative roots.

The central adaptation is:

> Immutable terms and proof objects deduplicate by content identity. Mutable semantic authority—branch heads, assumptions contexts, profiles, rule registries, extension worlds, accepted derivations, and release claims—commits through a versioned transaction with exact witnesses and proof-aware merge.

FrankenSymPy should not copy a page-level database engine into the symbolic hot path. It should reuse FrankenSQLite through a narrow ledger adapter where mature, and separately adopt its transaction semantics as a language-neutral workspace protocol.

## 2. Source surfaces examined

The audit examined the following pinned surfaces:

| Surface | Relevant source |
|---|---|
| MVCC term/version primitives | `crates/fsqlite-mvcc/src/core_types.rs` |
| transaction lifecycle | `crates/fsqlite-mvcc/src/lifecycle.rs` |
| concurrent state machine and conflicts | `crates/fsqlite-mvcc/src/begin_concurrent.rs` |
| commit-time SSI validation | `crates/fsqlite-mvcc/src/ssi_validation.rs` |
| executable invariant checks | `crates/fsqlite-mvcc/src/invariants.rs` |
| deterministic intent replay | `crates/fsqlite-mvcc/src/deterministic_rebase.rs` |
| semantic physical merge | `crates/fsqlite-mvcc/src/physical_merge.rs` |
| intent commutativity and merge certificates | `crates/fsqlite-mvcc/src/history_compression.rs` |
| compact logical WAL records | `crates/fsqlite-mvcc/src/cell_delta_wal.rs` |
| witness key plane | `crates/fsqlite-mvcc/src/witness_plane.rs` |
| hierarchical witness buckets | `crates/fsqlite-mvcc/src/witness_hierarchy.rs` |
| bounded VOI witness refinement | `crates/fsqlite-mvcc/src/witness_refinement.rs` |
| content-addressed witness objects | `crates/fsqlite-mvcc/src/witness_objects.rs` |
| reservation and witness publication | `crates/fsqlite-mvcc/src/witness_publication.rs` |
| durable history sidecar | `crates/fsqlite-mvcc/src/history_sidecar.rs` |
| historical snapshots | `crates/fsqlite-mvcc/src/time_travel.rs` |
| provenance semirings | `crates/fsqlite-mvcc/src/provenance.rs` |
| local-to-global consistency | `crates/fsqlite-mvcc/src/sheaf_conformal.rs` |
| multi-database atomic commit | `crates/fsqlite-mvcc/src/two_phase_commit.rs` |

The repository contains many additional experimental and optimized modules. This audit imports only mechanisms with a clear symbolic role and still requires FrankenSymPy-specific gates.

## 3. Content identity and publication authority must remain separate

FrankenSQLite distinguishes page/version identity from commit visibility and transaction state. A page version may exist before it is committed or visible at a reader's snapshot.

FrankenSymPy needs the same separation:

```text
Object exists != object is authoritative
Proof checks != proof is published
Claim is published != branch head references it
Branch head references it != release profile certifies it
```

Immutable objects such as terms, domains, certificates, proofs, and receipts may be inserted idempotently into a content-addressed store at any time. They become authoritative only when a committed workspace event or release manifest references them under the exact universe in which they were verified.

A content hash cannot grant branch-write authority. A database row cannot grant mathematical evidence. A successful object insertion cannot skip commit validation.

## 4. Snapshot universes

A workspace transaction reads one immutable universe:

```text
WorkspaceSnapshot
├── workspace and branch identity
├── branch commit sequence
├── object-store manifest root
├── compatibility profile
├── Python/platform extension world
├── assumptions context
├── domain and coercion registry
├── operator and kind registry
├── rewrite-rule registry
├── algorithm and policy registry
├── verifier and evidence registry
├── protocol/schema registry
├── claim registry
└── retention and capability policy
```

The transaction never silently observes a mixture of old and new fields. A branch-head publication is one coherent snapshot triple or manifest pointer, not independent mutable variables sampled opportunistically.

A computation can intentionally fork a child context or branch, but the new universe receives a new content identity and an explicit parent.

## 5. Workspace transaction state machine

FrankenSQLite's explicit begin/prepare/commit/abort states motivate a typed symbolic state machine:

```text
Idle
Drafting(snapshot)
Validating(read_witnesses, write_intents)
Verified(candidate_publications)
Prepared(durable_objects, commit_marker)
Publishing
Committed(new_branch_head)
Aborted(conflict_or_fault)
```

Legal transitions are registry-defined and executable as invariant tests.

Important rules:

- a transaction cannot publish before validation and required mathematical verification;
- a verifier rejection returns to drafting or aborts; it cannot be overwritten by planner confidence;
- cancellation before prepare leaves only private or rebuildable objects;
- cancellation after a durable commit marker triggers deterministic roll-forward or explicit recovery;
- a committed branch head always resolves to complete authoritative object closure;
- a transaction state is not inferred from which files happen to exist.

## 6. Semantic read witnesses

Database serializability depends on recording what a transaction read, including predicates that can suffer phantoms. Symbolic work has analogous hidden reads.

A workspace transaction records exact witnesses for:

### 6.1 Object reads

- term, proof, certificate, context, rule, profile, registry, and artifact IDs;
- exact object version or content identity;
- absence observations where “not found” affected a decision.

### 6.2 Namespace and resolution reads

- Python import/export name resolution;
- operator, predicate, domain, printer, conversion, and plugin lookup;
- method-resolution and custom-hook capability lookup;
- default profile, policy, or registry selection;
- branch preference and canonical-form selection.

### 6.3 Predicate and range reads

- “all rules applicable to operator/domain pair”;
- “no existing derivation proves this claim”;
- “all open discrepancies in profile region”;
- “all terms matching this pattern or namespace prefix”;
- “all package surfaces in this compatibility inventory”;
- “all registered coercions between these domains.”

These reads require phantom protection. Adding a new rule, converter, plugin, discrepancy, or derivation can invalidate a prior negative or exhaustive query even when no previously read object changed.

### 6.4 Environmental reads

When compatibility behavior depends on Python version, ABI, hash seed, locale, optional dependency, printer setting, or dynamic class identity, those values are part of the immutable profile/extension-world witness.

## 7. Witness plane and no-false-negative law

FrankenSQLite's witness plane uses conservative keys and permits false positives but forbids false negatives.

FrankenSymPy adopts that law for workspace conflict detection.

Initial witness granularities:

```text
Universe
Registry(registry_id)
Namespace(registry_id, prefix)
Object(object_id)
Claim(claim_id)
ContextFact(predicate, term_ids)
RuleApplicability(operator, domain, pattern_region)
ProfileSurface(module_or_class_region)
ExtensionWorld(class_or_plugin_region)
BranchRange(event_interval)
Custom(namespace, canonical_bytes)
```

A broad namespace witness can conservatively overlap many writes. Refinement can later prove disjointness, but absence of refinement never hides a true conflict.

Witness keys use canonical domain-separated identities. Fast non-cryptographic bucket hashes are permitted only as disposable indexes; exact conflict validation returns to canonical keys.

## 8. Hierarchical witness refinement

FrankenSQLite refines page-level conflict evidence toward cell, byte-range, hashed-key, and exact-key summaries under a bounded value-of-information policy. Refinement is an optimization only: exhaustion retains the conservative conflict.

FrankenSymPy can refine:

```text
Universe
  -> Registry
    -> Namespace/prefix
      -> Operator/domain or module/class region
        -> exact object / exact rule / exact fact
```

Examples:

- a coarse `RuleRegistry` conflict refines to distinct operator/domain partitions;
- a compatibility inventory conflict refines to disjoint module paths;
- an assumptions conflict refines to independent predicates or terms;
- an agent branch conflict refines from proof graph region to exact derivation edges;
- a cache invalidation refines from algorithm registry to the exact strategy and policy epoch.

Refinement policy may use estimated abort/recompute cost, overlap frequency, and refinement cost. It cannot weaken soundness. Statistical estimates influence whether refinement is attempted, never whether an unresolved conflict is ignored.

## 9. Commit-time validation

At commit, validate every read and predicate witness against changes since the snapshot.

Validation outcomes:

```text
Valid
Invalid
├── stale objects
├── namespace phantoms
├── changed profile or extension world
├── assumptions contradiction
├── rule or verifier registry drift
├── evidence downgrade
├── branch-head conflict
├── mutable snapshot generation conflict
├── missing validation history
└── concurrent transaction IDs / branch commits
```

A bounded history ring or GC horizon cannot silently validate an older transaction. When the exact validation interval has been pruned, the result is `ValidationHistoryUnavailable`; the transaction must rebase, retry from a newer snapshot, or preserve its own complete witness interval.

This is the symbolic analogue of FrankenSQLite refusing validation below a retained watermark.

## 10. Serializable workspace semantics

Default collaborative branch publication targets serializable semantics, not merely snapshot isolation.

Dangerous structures include:

- transaction A reads absence of proof P and publishes lemma A;
- transaction B reads absence of lemma A and publishes proof P;
- two agents each add a fact that is locally consistent but jointly contradictory;
- two rule updates each assume the previous canonicalization order;
- one branch changes a verifier while another publishes certificates checked by the old verifier;
- two compatibility edits each pass against the old surface inventory but together collide in module/class identity.

Read/write antidependency tracking detects these structures. Conservative abort/rebase is acceptable. Silent write skew is not.

A lower-isolation scratch workspace may be offered explicitly for exploratory candidates, but it cannot publish accepted claims or certified profile state without serializable validation.

## 11. Deterministic intent replay

FrankenSQLite records idempotency keys, statement/parameter fingerprints, determinism class, original result digest, bounded retries, and a rebase receipt.

FrankenSymPy semantic intents should include:

```text
SemanticIntent
├── intent ID and idempotency key
├── operation schema
├── canonical parameters
├── declared read and write footprints
├── determinism/effect class
├── required capabilities
├── original result/claim/certificate digest
├── profile/context/registry fingerprints
└── replay verifier
```

Replay is permitted only for declarative pure intents such as:

- add immutable object;
- add a fact to a child context;
- attach a verified derivation;
- apply a registered semantic patch;
- change a branch preference under a fixed ordering policy;
- insert a discrepancy record;
- update a registry through a governed deterministic transform.

Replay is not permitted for:

- arbitrary Python callbacks;
- wall-clock or external-state-dependent actions;
- unrecorded randomness;
- network or process effects;
- mutable alias observations;
- code whose extension-world identity changed;
- heuristics that cannot reproduce the same canonical claim/evidence digest.

A replayed result must match the original promised semantic and evidence digest under the exact replay contract. Mismatch yields `ReplayResultMismatch`; it is not silently accepted as a reasonable equivalent.

## 12. Semantic merge ladder

FrankenSQLite's physical merge rule is crucial: structured pages are parsed and merged by stable semantic cell identity; raw XOR patches are forbidden.

FrankenSymPy's ladder is:

1. **Content deduplication:** identical immutable objects collapse by exact identity.
2. **Disjoint append:** independent objects or branch events merge without replay.
3. **Certified commutative merge:** declared intents commute and the checker proves their footprints and algebraic merge law.
4. **Deterministic semantic replay:** replay one intent program against the new base and compare the exact result/evidence contract.
5. **Explicit conflict or branch:** retain alternatives; no automatic merge.

Forbidden:

- textual merge of formulas or proof JSON;
- byte-range or XOR merge of serialized mathematical objects;
- “same pretty print” as identity;
- last-writer-wins for assumptions, rules, profiles, or accepted claims;
- merging two individually verified proofs without checking their combined universe.

## 13. Trace monoids and canonical normal forms

FrankenSQLite formalizes intent independence and computes a Foata normal form: dependent operations retain order; independent operations occupy layers and receive a canonical intra-layer ordering.

FrankenSymPy should define an intent independence relation:

Two intents are independent only when:

- they share the exact base profile, context, schema, and registry epochs;
- neither has undeclared effects;
- their write footprints are disjoint or governed by a registered commutative join;
- neither writes an object, namespace, predicate region, or branch preference read by the other;
- neither changes a verifier or rule universe used by the other;
- no binder, class identity, mutable alias, or capability interaction couples them.

Registered commutative joins may include carefully proved CRDT-like operations:

- set union of immutable candidate IDs;
- max of a monotone resource watermark;
- union of independently verified derivation alternatives;
- append of distinct content-addressed discrepancy records;
- monotone status advancement under a fixed partial order and gate proof.

A canonical normal form supports:

- deterministic merge result identity;
- duplicate detection;
- compact merge certificates;
- replay minimization;
- branch equivalence checks;
- better agent coordination.

It must not be inferred from incidental operation names. Every commutativity class has a mathematical or state-machine proof and negative fixtures.

## 14. Merge certificates

A semantic merge certificate contains:

```text
SemanticMergeCertificate
├── base workspace snapshot
├── merged intent IDs and canonical digests
├── read/write/namespace footprint digest
├── independence or join-law evidence
├── canonical trace normal form
├── replay receipts, if used
├── resulting object and branch-manifest roots
├── assumptions/profile/registry versions
├── verifier versions
└── invariant-check results
```

A dependency-light merge verifier checks the exact claim. Failure triggers a circuit breaker:

- disable that automatic merge class;
- quarantine affected branch/cache results;
- fall back to explicit branching or serial replay;
- emit a counterexample and incident artifact;
- rerun prior merged outputs if dependency analysis indicates exposure.

A post-state hash alone does not prove the merge law. A normal form alone does not prove the result's mathematics. The certificate grants only the state transition it checks.

## 15. Publication reservation, coalescing, and helping

FrankenSQLite's witness publication protocol reserves an artifact key, coalesces duplicate producers, uses lease epochs to reject stale writers, supports helper completion, and treats identical idempotent publication differently from conflicting publication.

FrankenSymPy adopts:

```text
reserve(claim_id, evidence_requirement, universe_id)
  -> Granted(token)
   | Coalesced(token, owner)
   | RetryAfter
```

Rules:

- expensive identical proof/check/canonicalization work coalesces;
- a stale token cannot publish after lease transfer;
- a helper may finish only the exact same staged capsule and claim;
- identical already-committed publication is idempotent;
- a different value, certificate, evidence class, or universe under the same key is a hard conflict;
- cancellation revokes publication authority but not the immutable candidate bytes;
- the implementation uses asupersync-owned cancel-correct primitives, not unmanaged sleeps or detached helpers.

This complements FMAP evidence escrow.

## 16. Logical delta records versus materialized state

FrankenSQLite's cell-delta WAL records logical row operations compactly and materializes pages later. The safe merge path operates over semantic cells rather than page bytes.

FrankenSymPy should persist compact semantic intents and immutable object references rather than repeatedly serializing full workspace snapshots:

- add object ID;
- add/remove branch reference;
- add context fact or child context;
- attach claim/certificate/proof;
- advance branch head;
- record profile or registry migration;
- pin/unpin retention root.

Materialized views—current branch state, graph projections, namespace maps, search indexes, compatibility dashboards—are derived and rebuildable.

The authoritative log plus immutable objects reconstructs state. A materialized view checksum is useful operational evidence but not semantic authority.

## 17. History, retention, and time travel

FrankenSQLite's history sidecar separates lookup metadata from the authoritative WAL/page history, binds records to a stable lineage and generations, chains record hashes, validates reserved bytes and versions, truncates unsafe tails, and exposes exact retention horizons.

FrankenSymPy workspaces should provide:

```text
AS OF BranchCommitId
AS OF ProfileId
AS OF Timestamp -> resolved immutable commit
```

Historical snapshots are read-only. They support:

- reproducing an old proof or incompatibility;
- checking whether a result depended on a later rule;
- bisecting unsoundness or performance drift;
- rebuilding release artifacts;
- inspecting agent branch evolution;
- validating cache invalidation.

Retention is explicit. If the requested authoritative closure has been pruned, return `HistoryNotRetained` rather than fabricating a best effort. Informational timestamps never replace canonical commit IDs.

History indexes and sidecars are lookup aids; they do not replace retained objects and events.

## 18. Active snapshot tracking and garbage collection

Long computations, external verifier capsules, releases, branches, and replay bundles pin history.

The GC floor is computed from exact active roots. Approximate or conservative epoch summaries may delay collection but cannot free objects still reachable from an exact pin.

Roots include:

- live workspace transactions;
- branch heads;
- release/profile manifests;
- verifier-complete capsules promised for retention;
- checkpoints and continuations;
- incident and discrepancy bundles;
- explicit user pins.

GC publishes a previewable plan and validates that every retained manifest remains closed after deletion. RaptorQ symbols follow their source object's retention policy.

## 19. Provenance semirings

FrankenSQLite's provenance module models alternative derivations with `+` and combined contributions with `*`, enabling why and how provenance.

FrankenSymPy can generalize this into a **derivation provenance algebra**:

- base generators are source facts, rules, claims, oracle observations, user assertions, and imported theorem IDs;
- multiplication means all dependencies are jointly required;
- addition means alternative derivations establish or produce the same output;
- homomorphisms extract why provenance, rule families, trust boundaries, cost, privacy exposure, or invalidation sets.

Use cases:

- answer “why does this result exist?”;
- compute the minimal source dependency set;
- identify every result affected by a faulty rule or verifier;
- compare alternative proofs;
- select a lower-trust or lower-cost derivation;
- derive privacy/export policy;
- explain compatibility behavior.

The provenance semiring is not automatically a proof system. A token says which inputs and operations contributed; verifier-accepted derivation edges establish mathematical validity.

The implementation must use DAG sharing and canonical commutative nodes to avoid exponential expression trees.

## 20. Local-to-global consistency obstructions

FrankenSQLite experiments with sheaf-style consistency checks over overlapping transaction observations.

A disciplined FrankenSymPy version can check whether local branch sections glue into one global universe:

```text
Section
├── branch/agent identity
├── local universe snapshot
├── observed object and registry versions
├── local assumptions
├── accepted derivation edges
└── proposed result preferences
```

Overlaps must agree on:

- object identity and canonical bytes;
- profile, context, rule, and verifier universe;
- class/extension-world identity;
- shared assumptions and branch policies;
- imported proof-edge validity.

A mismatch yields a structured gluing obstruction. This is useful for collaborative branch review and distributed replay.

Caution: pairwise overlap agreement is not a complete general proof of global consistency. The production checker must state the exact cover, compatibility maps, and theorem it checks. “Sheaf” cannot become decorative terminology.

## 21. Multi-root atomic publication

FrankenSQLite's two-phase commit separates prepare, durable global marker, visibility publication, and crash recovery.

FrankenSymPy usually avoids distributed transactions by keeping one authoritative ledger and treating graph/search projections as derived. When one logical commit necessarily spans multiple authoritative stores, the protocol is:

1. write all immutable objects and verify readback;
2. prepare each authoritative participant with an exact manifest root;
3. write one durable global commit marker containing participant roots;
4. publish branch/release visibility pointers;
5. recover by rolling back unmarked prepares or rolling forward marked commits.

Participants might include:

- local object store;
- append-only workspace ledger;
- separately retained proof archive;
- organization-wide release transparency log.

FrankenGraphDB projections, search indexes, metrics, and ordinary caches are not 2PC participants. They rebuild after the authoritative commit.

If FrankenSQLite stores the ledger and object metadata in one database transaction, do not add 2PC gratuitously.

## 22. Executable invariants

Adapt FrankenSQLite's named invariant checker into a workspace invariant registry:

- commit sequence monotonicity;
- legal transaction transitions;
- branch head references a complete object closure;
- no unverified object in accepted namespace;
- no stale universe publication;
- exact read witnesses validated;
- no false-negative witness route in generated adversaries;
- serializable branch history;
- verifier and claim schema match;
- no evidence downgrade;
- deterministic merge normal form;
- replay receipt matches result digest;
- GC preserves all roots;
- historical snapshot is read-only;
- materialized views rebuild equivalently.

Violations produce stable names, exact involved IDs, minimized witnesses, and severity. They feed incident handling and deterministic lab tests.

## 23. What not to inherit blindly

### Implementation maturity

FrankenSQLite contains implemented, partial, dormant, and research surfaces. Source presence is not proof that a mechanism is live, integrated, performant, or safe for FrankenSymPy.

### Storage hot path

Ordinary symbolic term construction, rewriting, and checking remain in-memory and storage-independent.

### Hash choices

Fast truncated or non-cryptographic donor hashes may be appropriate for local indexes but not for FrankenSymPy's long-lived public content identities.

### Unsafe and external dependencies

FrankenSymPy's stricter constitution forbids unsafe code and C/C++ arithmetic/CAS FFI in project code and trusted verifier paths. Donor implementation details do not override that.

### Optimistic merge

Only declared pure intents with exact footprint and checker support can merge. Unknown Python behavior, branch-sensitive transformations, and mutable aliases remain conflicting.

### Statistical policy

VOI, conformal, change-point, or e-process machinery can choose refinement and rollout effort. It cannot dismiss an unresolved soundness conflict.

## 24. Adopt, adapt, reject summary

### Adopt directly as architecture

- immutable snapshots and versioned branch heads;
- explicit transaction state machine;
- exact read/write/predicate witnesses;
- no-false-negative conservative conflict plane;
- missing-history refusal;
- deterministic pure intent replay with digest comparison;
- semantic merge instead of byte merge;
- append-only history and read-only time travel;
- named executable invariants;
- idempotent identical publication versus conflicting publication.

### Adapt behind FrankenSymPy interfaces

- Page-SSI into semantic workspace serializability;
- witness hierarchy into registry/namespace/object granularity;
- VOI refinement into optional conservative conflict refinement;
- intent Foata normal form into semantic patch canonicalization;
- merge certificates into dependency-light workspace transition certificates;
- cell-delta WAL into semantic event logs;
- history sidecar into lineage-bound workspace indexes;
- provenance semirings into derivation dependency algebra;
- sheaf checks into explicitly scoped local-to-global branch consistency;
- 2PC into rare multi-authority publication.

### Reject

- database rows as term or proof identity;
- persistence in the algebraic hot path;
- byte/XOR merge of serialized symbolic state;
- silent last-writer-wins;
- replay of effectful or extension-world-dependent code;
- old snapshot validation after its exact conflict interval is pruned;
- materialized index state as authority;
- pairwise consistency branded as a general global proof;
- donor claims copied without FrankenSymPy integration gates.

## 25. Workstream consequences

The plan should add or strengthen:

1. semantic workspace MVCC and serializable publication;
2. hierarchical conflict witnesses and predicate/negative-read protection;
3. deterministic semantic intents and replay receipts;
4. trace-normalized merge and embeddable merge certificates;
5. evidence publication reservation and coalescing;
6. derivation provenance algebra;
7. retained workspace history and time travel;
8. exact active-root GC;
9. multi-root publication only where unavoidable;
10. local invariant and crash/recovery gates.
