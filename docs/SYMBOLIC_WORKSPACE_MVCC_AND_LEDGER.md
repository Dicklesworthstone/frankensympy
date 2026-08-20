# Symbolic workspace MVCC, serializable publication, and proof-aware merge

**Status:** normative architecture contract  
**Scope:** collaborative workspaces, immutable universe snapshots, semantic transaction witnesses, deterministic intent replay, merge certificates, history, provenance, garbage collection, and crash recovery

## 1. Objective

FrankenSymPy workspaces must support many agents, users, services, and long-running computations without allowing concurrent edits to create an internally inconsistent mathematical universe.

The workspace layer governs mutable authority around immutable mathematical objects. It does not replace the term kernel, proof verifier, or Python compatibility shell.

The design targets:

- immutable content-addressed objects;
- serializable branch and registry publication;
- exact conflict witnesses;
- conservative no-false-negative validation;
- deterministic replay of pure semantic intents;
- canonical merge of disjoint or provably commuting work;
- independently verifiable merge certificates;
- append-only history and time travel;
- optional FrankenSQLite-backed durability;
- derived FrankenGraphDB indexes;
- crash-safe, cancellation-aware publication.

## 2. State categories

### 2.1 Immutable content objects

Examples:

- terms and surface descriptors;
- domains, coercions, contexts, and branch policies;
- claims, certificates, proofs, and counterexamples;
- registry fragments;
- compatibility observations;
- generated programs;
- receipts and continuations.

They deduplicate by canonical content identity. Creating an object does not make it authoritative.

### 2.2 Authoritative mutable references

Examples:

- workspace and branch heads;
- selected compatibility profile;
- current assumptions context;
- current rule, algorithm, verifier, and claims registries;
- accepted derivation roots;
- branch result preferences;
- release and retention roots;
- discrepancy status transitions.

They change only through workspace transactions.

### 2.3 Derived state

Examples:

- graph indexes;
- search indexes;
- materialized branch views;
- cost models and statistics;
- cache indexes;
- UI summaries;
- transfer layouts;
- telemetry.

Derived state is rebuildable and never participates in mathematical identity or commit authority.

## 3. Workspace snapshot

```text
WorkspaceSnapshot
├── WorkspaceId
├── BranchId
├── BranchCommitId / sequence
├── FmapManifestId
├── CompatibilityProfileId
├── ExtensionWorldId
├── AssumptionsContextId
├── DomainRegistryId
├── OperatorRegistryId
├── RuleRegistryId
├── AlgorithmRegistryId
├── VerifierRegistryId
├── EvidenceRegistryId
├── ClaimRegistryId
├── SchemaRegistryId
├── BranchPolicyId
├── RetentionPolicyId
└── parent snapshot / lineage
```

All fields are sampled coherently. A snapshot is immutable and content-addressed apart from its monotone branch sequence coordinate.

The transaction records either the complete snapshot root or the exact authoritative objects it reads. A branch update cannot silently swap one registry while retaining a result verified in another.

## 4. Transaction modes

### 4.1 `ReadOnly`

- observes one immutable snapshot;
- may construct private derived state;
- cannot publish branch or registry changes;
- may export a verifier/replay bundle.

### 4.2 `Candidate`

- writes immutable candidate objects and private branch events;
- may use snapshot isolation for exploration;
- cannot publish accepted claims or certified profile state;
- all outputs remain explicitly candidate/conditional.

### 4.3 `SerializablePublish`

- default for accepted derivations, assumptions, rule changes, profile work, claim status, and release roots;
- records exact read, negative, namespace, and predicate witnesses;
- validates against every authoritative commit since its snapshot;
- detects dangerous rw-antidependency structures;
- may replay or merge only through declared safe paths.

### 4.4 `RegistryAdmin`

- serializable plus explicit governance capability;
- may change operator, domain, rule, verifier, evidence, schema, profile, claims, or workstream registries;
- creates a new registry universe;
- triggers dependency invalidation and migration gates;
- cannot retroactively mutate old objects or profiles.

### 4.5 `ReleasePublish`

- serializable publication over an immutable profile and gate bundle;
- requires no open blocking discrepancy, exact artifact closure, provenance, signatures, and local release-gate results;
- may atomically advance release/transparency roots;
- cannot be satisfied by GitHub status alone.

## 5. State machine

```text
Idle
  -> Drafting(snapshot)
  -> Validating
  -> Verified
  -> Prepared
  -> Publishing
  -> Committed

Drafting | Validating | Verified | Prepared | Publishing
  -> Aborted(reason)
```

### 5.1 Drafting

The transaction:

- constructs immutable objects;
- records semantic intents;
- records read and predicate witnesses;
- runs generators and local verifiers;
- retains outputs in evidence escrow.

### 5.2 Validating

The workspace engine:

- freezes the intended write set;
- validates all read/predicate witnesses;
- checks context/profile/registry and extension-world identity;
- detects direct and SSI-style conflicts;
- checks branch policy and capabilities;
- plans replay, certified merge, or abort.

No new semantic reads are admitted during final validation unless they are recorded and the validation restarts.

### 5.3 Verified

Every accepted mathematical object and semantic state transition has its required independent verification. Candidate-only objects may still accompany the commit but remain in candidate namespaces.

### 5.4 Prepared

- authoritative immutable objects are written and readback-validated where durability is required;
- all participant roots are fixed;
- a canonical commit capsule is formed;
- publication rights are reserved;
- a durable marker is written when multiple authoritative participants require recovery coordination.

### 5.5 Publishing

The branch/release head advances atomically to the prepared manifest. Derived indexes update later.

### 5.6 Committed

The transaction receives a canonical receipt naming old/new snapshots, intents, witnesses, validations, verifications, merge/replay evidence, durable roots, and publication sequence.

## 6. Witness model

```text
WorkspaceWitness
├── Universe
├── Registry(registry_kind, registry_id)
├── Namespace(registry_kind, canonical_prefix)
├── Object(object_kind, object_id)
├── ObjectAbsent(object_kind, canonical_lookup_key)
├── Claim(claim_id)
├── ContextFact(predicate_id, term_ids)
├── RuleApplicability(operator_id, domain_id, pattern_region)
├── ProfileSurface(module_class_or_signature_region)
├── ExtensionWorld(class_plugin_or_converter_region)
├── BranchEvents(from_exclusive, to_inclusive)
├── MutableCell(cell_id, generation)
└── Custom(namespace, canonical_bytes)
```

Each witness states `Read`, `Write`, or `PredicateRead` and a granularity level.

The overlap relation is deterministic and versioned. Unknown pairs overlap conservatively.

## 7. No-false-negative invariant

For every authoritative mutation that could change the result of an earlier read or predicate query, at least one write witness overlaps at least one prior read witness.

Required generated adversaries include:

- adding a rule after “all applicable rules” was read;
- registering a converter after a negative conversion lookup;
- adding a proof after “no proof exists” was observed;
- adding a symbol or module export under a scanned compatibility namespace;
- changing a predicate handler used by an assumptions query;
- changing a verifier or certificate schema used by a result;
- mutating a Python class/plugin capability in the extension world;
- adding a fact that contradicts a locally consistent context;
- changing a branch preference used to select a canonical representative.

A witness index may produce false positives. It must not suppress these conflicts.

## 8. Hierarchy and refinement

Coarse levels:

```text
L0 Universe
L1 Registry or profile family
L2 Namespace / operator-domain / module-class region
L3 Exact object, rule, fact, or extension identity
```

The hot conflict plane may record L0-L2 summaries. Exact objects and predicate intervals remain available in immutable witness artifacts for refinement.

Refinement can prove that a coarse overlap is false. It operates under explicit byte, CPU, proof, and bucket budgets. On exhaustion or unsupported summary, the coarse conflict survives.

A value-of-information planner can prioritize refinement by expected saved recomputation or abort cost. Its estimates are operational only.

## 9. Commit interval and retention

Validation needs the exact authoritative commits between `snapshot_seq + 1` and the candidate commit.

The workspace maintains:

- append-only commit summaries;
- exact write and predicate-witness summaries;
- a retained validation horizon;
- branch lineage and registry migration events;
- optional compressed witness objects.

If the transaction's snapshot precedes the exact retained horizon and it did not preserve a self-contained validation interval, validation returns:

```text
ValidationHistoryUnavailable {
    snapshot,
    retained_horizon
}
```

It never assumes no conflict because history is missing.

## 10. Conflict classes

```text
DirectWriteWrite
StaleObjectRead
NamespacePhantom
PredicatePhantom
AssumptionsContradiction
ProfileConflict
ExtensionWorldConflict
RuleRegistryConflict
VerifierRegistryConflict
SchemaConflict
EvidenceDowngrade
BranchPreferenceConflict
MutableGenerationConflict
ReplayResultMismatch
MergeCertificateRejected
ValidationHistoryUnavailable
CapabilityDenied
RetentionConflict
MultiParticipantPrepareConflict
```

A conflict record contains:

- old and new snapshots;
- involved transactions/commits;
- exact overlapping witnesses;
- affected objects and claims;
- whether refinement was attempted;
- candidate safe resolutions;
- minimized reproducer or artifact pointer.

## 11. Semantic intent

```text
SemanticIntent
├── IntentId
├── IdempotencyKey
├── operation schema and version
├── canonical parameters
├── read/predicate/write footprint
├── effect class
├── profile/context/registry universe
├── required capability
├── deterministic random stream, if applicable
├── expected result contract
├── expected claim/evidence/capsule digest
└── replay verifier
```

Effect classes:

```text
PureDeclarative
PureOpaque
Effectful
UnknownEffect
```

Only `PureDeclarative` intents can participate in automatic replay or certified commutative merge.

`PureOpaque` may be deterministic but lacks an understood semantic footprint; it can replay only in a sandboxed/private context with exact result comparison and cannot merge automatically.

`Effectful` and `UnknownEffect` execute exactly once and normally force explicit branch handling.

## 12. Deterministic rebase

```text
rebase(intent_program, new_snapshot, limits)
    -> Committed(new_snapshot, receipt)
     | Conflict(witnesses)
     | ResultMismatch(original, replayed)
     | Irreproducible(intent_id)
     | Exhausted(attempts_or_budget)
```

Rules:

- replay attempts are bounded;
- each attempt reads one coherent new snapshot;
- random choices use recorded counter ranges;
- no ambient time, network, database query, or Python side effect is allowed;
- result comparison names the exact semantic, surface, evidence, and compatibility dimensions promised by the intent;
- mathematically equivalent but profile-different results can still be a mismatch;
- a replayed stronger or weaker evidence class is not silently substituted;
- every attempt extends the same resource accounting rather than resetting budgets.

## 13. Independence relation

Two intents commute only under a registered relation:

```text
Independent(a, b) iff
    same required universe epochs
    and both PureDeclarative
    and no write/write overlap unless a registered join exists
    and no write/read or write/predicate overlap
    and no verifier/rule/profile/extension mutation affects the other
    and no mutable alias or binder capture coupling
    and all side conditions are independently verified
```

Unknown means dependent.

The relation is itself versioned and tested. A deleted side condition or falsely widened footprint is a mutation that must be killed.

## 14. Registered joins

A state join is allowed only when it is:

- associative;
- commutative;
- idempotent when duplicate delivery is possible;
- identity-preserving;
- deterministic;
- context/profile/registry-safe;
- accompanied by a small checker.

Initial candidates:

- union of distinct immutable candidate object IDs;
- union of alternative verified derivation edges for the same typed claim/universe;
- max of a monotone retention or processing watermark;
- append of distinct content-addressed discrepancy or observation records;
- monotone claim status transition where the exact gate proof is included.

Not joinable:

- assumptions with possible contradiction;
- rewrite order;
- canonical result preference;
- profile class/signature definitions;
- mutable matrices or Python objects;
- proof evidence classes by “take max”;
- verifier versions;
- namespace aliases that may collide.

## 15. Canonical trace normal form

Given a set of intents and the certified independence relation:

1. build dependency edges for every non-independent pair;
2. preserve causal/original order for dependent operations;
3. compute topological Foata layers;
4. sort independent operations within each layer by canonical intent key;
5. emit canonical intent IDs and join operations;
6. verify the resulting state root.

The canonical key includes operation schema, typed target, canonical parameters, and intent ID. It excludes arrival time, agent identity, thread scheduling, and storage row.

The normal form supports deterministic branch equivalence and merge certificate verification.

## 16. Merge certificate

```text
SemanticMergeClaim
├── base snapshot
├── input intent programs
├── declared independence/join relations
├── canonical normal form
├── expected result snapshot root
└── merge policy version

SemanticMergeCertificate
├── exact read/write/predicate footprints
├── relation and side-condition evidence
├── replay receipts
├── intermediate state roots, as required
├── final FMAP manifest root
├── context/profile/registry versions
└── invariant results
```

The embeddable verifier checks:

- every intent and object identity;
- relation eligibility;
- footprint non-overlap or registered join law;
- canonical normal form;
- replay digests;
- final state root;
- no evidence downgrade;
- exact universe binding.

It does not need the planner, agent workspace, Python shell, database, or graph index.

## 17. Circuit breaker

A merge verifier rejection or post-publication counterexample:

1. disables the affected merge policy/version;
2. prevents new automatic merges of that class;
3. identifies all published branch commits depending on the policy;
4. re-verifies or quarantines affected accepted state;
5. preserves incident and minimized counterexample capsules;
6. falls back to serial replay or explicit branches;
7. requires an amended relation/checker and mutation gate before re-enable.

Statistical monitoring can trigger the same quarantine earlier, but only a deterministic policy action follows; the monitor does not prove unsoundness.

## 18. Publication reservation

```text
PublicationKey
├── workspace/branch
├── base snapshot
├── typed claim or transition target
├── evidence requirement
└── universe ID
```

```text
reserve(key)
  -> Granted(token, lease_epoch)
   | Coalesced(token, owner)
   | RetryAfter
```

A token authorizes only:

- the exact key;
- one lease epoch;
- one staged FMAP capsule root;
- publication after verification and prepare.

A coalesced helper can finish canonicalization, verification, or durable writes for the same capsule. It cannot substitute a different result.

Stale tokens, changed base snapshots, different evidence, and different capsules are rejected.

## 19. Event log

Authoritative events:

```text
WorkspaceCreated
BranchForked
TransactionStarted
ObjectAdded
IntentRecorded
CandidateAttached
VerificationAccepted
VerificationRejected
ContextAdvanced
RegistryAdvanced
SemanticMergePrepared
BranchHeadCommitted
BranchMerged
CheckpointPublished
RetentionPinned
RetentionReleased
DiscrepancyAdvanced
ReleasePublished
IncidentQuarantined
```

Events are canonical, append-only, sequence-ordered within a lineage, and reference immutable objects. Materialized current state rebuilds from events.

A giant full-state snapshot may be periodically emitted as a performance checkpoint, but the event/log lineage and exact root remain authoritative.

## 20. Time travel

Read-only historical views:

```text
workspace.open_as_of(BranchCommitId)
workspace.open_as_of_timestamp(timestamp, resolution_policy)
```

Timestamp resolution returns the selected exact commit ID. It is informational and cannot identify a state by itself.

Historical views:

- forbid writes and registry changes;
- retain the original compatibility/extension world;
- can export verifier and replay bundles;
- state explicitly when required objects or validation history were pruned;
- never auto-upgrade old certificates to new verifier schemas.

## 21. Provenance algebra

A derivation provenance expression uses canonical DAG nodes:

```text
Zero
One
Base(source_object_or_assertion)
Alternative(children)       # commutative/idempotent form as configured
Joint(children)             # all dependencies required
Transform(operation, child)
```

Semiring-like projections compute:

- why provenance;
- how provenance;
- source/license provenance;
- trust-boundary set;
- invalidation dependency set;
- privacy/export taint;
- estimated recomputation cost;
- proof-family summary.

The provenance object cannot appear as mathematical evidence unless each referenced derivation edge is independently accepted under the claim.

## 22. Local-to-global branch sections

A branch section declares:

- its base snapshot;
- authoritative objects observed;
- local context/profile/registry choices;
- accepted derivation roots;
- proposed branch changes;
- restriction maps to shared overlap regions.

A gluing check verifies exact overlap agreement and produces obstructions such as:

```text
ObjectBytesMismatch
ContextMismatch
ProfileMismatch
RegistryMismatch
ExtensionWorldMismatch
ProofInvalidUnderTargetUniverse
ContradictoryFacts
MutableGenerationMismatch
SamePrintDifferentIdentity
```

The checker must define the exact cover and theorem. It does not claim that arbitrary pairwise consistency proves all global properties.

## 23. Persistence adapter

`fsym-ledger` defines storage-neutral traits for:

- immutable object put/get;
- append authoritative event;
- snapshot transaction;
- read-witness interval query;
- prepare/commit marker;
- branch head compare-and-swap;
- retention roots;
- crash recovery;
- optional verified cache metadata.

`fsym-frankensqlite` implements the traits only after integration gates establish the exact features used.

The adapter never exports database row IDs as workspace or mathematical IDs.

The workspace can run fully ephemerally with identical semantic outcomes.

## 24. Multi-participant publication

One authoritative store is preferred.

When several authoritative participants are unavoidable:

```text
Prepare participant object roots
  -> write global commit marker
  -> publish visibility pointers
  -> complete/repair lagging participants
```

Recovery:

- no marker: discard/ignore prepares;
- marker present: roll forward exact participant roots;
- marker corrupt or incomplete: fail closed and preserve diagnostic capsule;
- derived graph/search views rebuild after commit and are not participants.

## 25. Garbage collection

Exact roots:

- branch heads;
- active transactions;
- releases and profiles;
- retained history checkpoints;
- verifier/replay bundles under retention promises;
- continuations;
- incidents, discrepancies, and legal holds;
- user pins.

GC uses exact reachability over authoritative FMAP edges. Approximate summaries may conservatively retain extra objects but never free a reachable object.

A GC plan is deterministic, previewable, and proves every retained manifest remains closed.

## 26. Invariant registry

Blocking invariants:

```text
WTX-001 legal transaction transition
WTX-002 coherent snapshot universe
WTX-003 exact branch sequence monotonicity
WTX-004 accepted namespace contains only independently verified objects
WTX-005 no stale-universe publication
WTX-006 no false-negative witness in registered adversary corpus
WTX-007 validation interval complete or explicit refusal
WTX-008 serializable authoritative history
WTX-009 deterministic replay preserves promised digest
WTX-010 semantic merge certificate verifies
WTX-011 no evidence downgrade
WTX-012 committed branch head has closed FMAP manifest
WTX-013 historical views are read-only
WTX-014 GC preserves every exact root
WTX-015 derived indexes rebuild equivalently
WTX-016 crash recovery yields pre-commit or exact committed state
```

Each violation produces a stable machine-readable witness and blocks the relevant claim.

## 27. Deterministic lab tests

Explore:

- concurrent direct and phantom conflicts;
- candidate publication and branch-head races;
- lease expiry/help completion;
- cancellation before and after every prepare/publication step;
- deterministic rebase against repeated concurrent updates;
- trace-normal merge under randomized arrival order;
- history pruning versus old validation;
- GC versus snapshot pins;
- persistence-disabled versus durable execution;
- crash recovery with partial participant progress;
- graph-index deletion and rebuild;
- Python hook and extension-world changes;
- malicious or incomplete merge certificates.

Assertions include zero controlled orphan work, no partial accepted state, deterministic canonical receipts, and no missing conflict.

## 28. Initial vertical slice

The first slice supports:

1. branch snapshots over the initial term/domain/context/verifier registries;
2. add-object, add-fact, attach-verified-derivation, and branch-preference intents;
3. exact object/namespace/negative-read witnesses;
4. serializable two-agent commits;
5. deterministic rebase of pure add-object and attach-derivation intents;
6. canonical merge of disjoint derivation alternatives;
7. embeddable semantic merge verifier;
8. FMAP event and commit capsules;
9. ephemeral and FrankenSQLite-backed ledger probes;
10. time-travel read view and exact GC pins;
11. crash/cancellation matrix;
12. a same-print/different-domain conflict fixture.

## 29. Forbidden shortcuts

- last-writer-wins for authoritative semantic state;
- textual or byte merge of formulas/proofs;
- treating same printer output as identity;
- omitting negative or predicate reads;
- accepting coarse-index non-overlap as exact without refinement;
- interpreting missing validation history as no conflict;
- replaying arbitrary Python or effectful code;
- resetting budgets on replay;
- merging under a changed verifier, profile, context, or extension world;
- using provenance as proof;
- publishing before independent verification and prepare;
- database row or graph node as branch/object identity;
- derived index as authoritative state;
- freeing objects from approximate GC summaries;
- claiming general sheaf consistency from an underspecified pairwise test;
- adding multi-store 2PC where a single authoritative transaction suffices.
