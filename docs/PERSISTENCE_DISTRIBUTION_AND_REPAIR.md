# FrankenSymPy persistence, distributed work, indexing, and repair

**Status:** normative architecture contract  
**Scope:** optional FrankenSQLite ledger, persistent memoization, checkpoints, collaborative branches, FrankenGraphDB indexing, remote workers, RaptorQ artifact durability, integrity and recovery

## 1. Principle

Persistence is valuable for expensive symbolic work, but it must not become part of mathematical identity or the in-memory algebraic hot path. FrankenSymPy treats durable state as an optional execution substrate around an independently correct kernel.

The authoritative persistent objects are immutable, content-addressed records. Mutable indexes, caches, statistics, and search projections are derived and rebuildable.

## 2. Persistence modes

### 2.1 Ephemeral

- no durable ledger or persistent cache;
- in-memory terms, proofs, and receipts live for the request/workspace;
- replay bundle may still be exported explicitly;
- ordinary local calls default here when persistence adds no value.

### 2.2 Local durable

- computation ledger, checkpoints, proof bundles, and verified cache on local storage;
- suitable for notebooks, services, and long-running exact jobs;
- crash recovery and schema/version validation enabled.

### 2.3 Collaborative workspace

- append-only derivation branches and shared verified objects;
- multiple agents/users can fork, propose, review, and merge work;
- semantic merge is verifier-governed;
- optional graph indexing for discovery.

### 2.4 Distributed execution

- immutable work packets sent to untrusted or semi-trusted workers;
- local verification before publication;
- replicated/checkpointed artifacts according to policy.

Persistence mode is explicit in the execution receipt and benchmark configuration.

## 3. FrankenSQLite role

FrankenSQLite is the preferred optional embedded ledger when its required live-path features are proven at integration time. The interface is narrow enough to admit another conforming store in tests.

It stores:

- workspace and branch manifests;
- compatibility/profile, assumptions, domain, rule, algorithm, and verifier registries;
- canonical term/object blobs or references;
- requests, decision cards, traces, receipts, outcomes;
- proof/certificate/checkpoint manifests;
- verified cache metadata;
- remote work leases and deduplication state;
- repair manifests and scrub/decode records;
- conformance/discrepancy/benchmark artifacts.

It does not:

- define term equality;
- determine proof validity;
- participate in every in-memory rewrite;
- assign canonical mathematical identities via row IDs;
- turn an unverified stored flag into evidence.

## 4. Logical record model

```text
Workspace
├── workspace_id
├── root universe manifest
├── branches
├── policies/capabilities
└── retention manifest

Branch
├── branch_id
├── parent branch/version
├── append-only event sequence
├── head context/registry/derivation IDs
└── merge status

ObjectRecord
├── object type and schema
├── content ID/digest
├── canonical bytes or blob locator
├── dependencies
├── evidence/verification status, if applicable
└── durability policy

ComputationRecord
├── request
├── region/decision trace
├── candidates
├── verifications
├── terminal outcome
├── checkpoint/continuation
└── receipts
```

Database primary keys are storage implementation details. Every authoritative reference also carries the typed content identity.

## 5. MVCC-inspired workspace semantics

Each computation reads an immutable universe snapshot:

- compatibility profile;
- assumptions context;
- domain/coercion registry;
- rule registry;
- algorithm/selector policy;
- verifier registry;
- branch head.

New facts, rules, or derivations create later versions. A running computation is not silently rebased.

### 5.1 Read set

The receipt records authoritative objects actually read or the root manifest that transitively fixes them.

### 5.2 Write set

A computation can propose:

- new semantic terms;
- candidate or verified derivation edges;
- proof/certificate objects;
- checkpoints;
- verified cache entries;
- branch events.

### 5.3 Commit validation

Before publication:

- content IDs and canonical bytes validate;
- referenced universe objects exist and match versions;
- evidence verifies or is stored only in a candidate namespace;
- branch preconditions still hold;
- mutable-cell snapshot generations are still valid when relevant;
- quotas and capability checks pass.

Conflict never changes a mathematical result. It causes retry, explicit rebase, branch creation, or refusal.

## 6. Event log and materialized views

The authoritative workspace history is append-only. Events include:

- branch fork/merge;
- context/profile/registry selection;
- request start/finalize;
- candidate/verification publication;
- checkpoint/resume;
- artifact repair/scrub;
- cache admission/eviction;
- discrepancy or claim-state transition.

Materialized views provide current branch heads, cache indexes, dependency maps, statistics, and UI summaries. Views are versioned and rebuildable from events and authoritative objects.

An event record saying “verified” does not replace replaying or validating the linked verification receipt.

## 7. Persistent memoization

### 7.1 Key

A persistent cache key includes:

- exact typed claim/request;
- term/domain/context/profile IDs;
- rule, algorithm, and verifier registry IDs;
- branch/precision/evaluation/completeness policies;
- evidence minimum;
- schema versions.

### 7.2 Entry classes

- verified exact result;
- verified certified-numeric result;
- compatibility observation;
- candidate/heuristic hint;
- proved negative result;
- transient operational hint;
- terminal diagnostic.

Each class has separate admission and retrieval rules.

### 7.3 Validation on read

- parse bounded canonical envelope;
- validate schema and all typed IDs;
- validate digest against canonical bytes;
- ensure evidence class satisfies request;
- ensure verifier/profile/context compatibility;
- replay verification according to trust and cache policy;
- reject/quarantine on any mismatch.

A cache is an accelerator. It cannot extend the trusted base simply by surviving on disk.

## 8. Checkpoints and continuations

Checkpoints are typed algorithm states, not process dumps. A checkpoint bundle contains:

- immutable input/universe manifest;
- algorithm/version;
- completed verified subresults;
- normalized remaining frontier;
- deterministic random counter state;
- resource accounting;
- private candidate data labeled as such;
- resume preconditions;
- canonical digest;
- optional RaptorQ sidecar manifest.

Resume:

1. loads and bounds-checks the manifest;
2. recovers missing/corrupt bytes if policy permits;
3. validates digests and canonical schemas;
4. revalidates authoritative dependencies;
5. replays or samples verification according to policy;
6. resumes in a new owned runtime region.

A version mismatch is refused unless a declared migration verifies semantic preservation.

## 9. Crash consistency

Two-phase artifact publication ensures the authoritative manifest never points to a partial object.

Example sequence:

1. write blob to private/staging key;
2. fsync/confirm according to durability policy;
3. compute/readback digest;
4. generate/write repair symbols if selected;
5. write object manifest;
6. atomically append publication event or update branch pointer;
7. garbage-collect abandoned staging objects later.

Crash injection covers every boundary. Recovery distinguishes:

- fully published object;
- valid but unpublished staging object;
- incomplete/corrupt object;
- manifest with missing dependencies;
- repairable byte loss;
- irrecoverable object.

No recovery path invents a mathematical result or substitutes a different registry/context.

## 10. RaptorQ artifact contract

RaptorQ is used selectively for high-value portable byte artifacts:

- long-running checkpoints;
- proof/certificate archives;
- deterministic replay bundles;
- distributed work packets/results;
- verified cache segments costly to recompute;
- conformance and benchmark evidence packs;
- minimized fuzz/counterexample corpora;
- release manifests.

It is normally not used for:

- tiny in-memory terms;
- disposable indexes;
- low-value cache entries;
- data already cheaply reconstructed from authoritative parents;
- every database page by default.

### 10.1 Envelope

```text
RepairEnvelope
├── object content ID
├── canonical source digest
├── source length and symbol size
├── source block partition
├── RaptorQ parameters/policy version
├── source symbol digests
├── repair symbol IDs/digests
├── storage/failure-domain placement
├── scrub history
└── decode-proof records
```

### 10.2 Correct ordering of trust

1. RaptorQ reconstructs candidate bytes.
2. Canonical digest validation establishes content integrity relative to the expected ID.
3. Signature/capability checks establish origin/authorization where required.
4. Schema and invariant validation establish well-formedness.
5. Mathematical verification establishes evidence.

These steps are never collapsed.

### 10.3 Adaptive redundancy

Redundancy policy may consider:

- recomputation cost;
- artifact value and retention horizon;
- observed loss/corruption rates;
- storage and transfer cost;
- number/failure domains of replicas;
- deadline for repair;
- privacy/security class.

Conformal/e-process monitors can alert on loss-rate drift and trigger a policy change for future artifacts. They do not guarantee a particular artifact survives and do not validate its content.

## 11. Scrubbing and repair

A scrub operation:

- selects artifacts under an auditable schedule;
- reads source/repair symbols;
- validates symbol and object digests;
- classifies missing/corrupt/unreadable state;
- repairs only when the canonical object digest can be recovered;
- emits a decode record;
- optionally regenerates lost repair symbols;
- never rewrites proof/evidence metadata based solely on recovered bytes.

Repair failure yields a typed durability outcome and may fall back to deterministic recomputation if all authoritative inputs remain available.

## 12. FrankenGraphDB role

FrankenGraphDB may index large graph projections of:

- term/operator dependencies;
- derivation/proof edges;
- theorem/lemma use;
- rule firing and counterexamples;
- workspace branches and merges;
- algorithm decision families;
- compatibility discrepancies;
- generated-code provenance;
- collaborative agent work.

The graph database is an optional derived index. It cannot:

- decide term equality;
- validate proof edges;
- become the only copy of authoritative proof objects;
- infer logical entailment from reachability;
- alter branch heads without ledger validation.

Graph query results return authoritative IDs and projection-version metadata. Callers revalidate objects before using them in proofs or caches.

## 13. Collaborative derivation branches

Agents work on immutable branch snapshots.

A branch can add:

- proposed terms/transforms;
- candidate or verified derivation edges;
- counterexamples;
- proof obligations;
- benchmark/conformance evidence;
- work-packet outcomes.

### 13.1 Semantic merge

A merge checks:

1. common ancestor and universe compatibility;
2. profile/context/rule/verifier version compatibility;
3. every imported verified edge by replay or accepted verification policy;
4. unresolved candidate/conditional edges remain labeled;
5. contradictory facts follow branch consistency policy;
6. stable IDs/canonical objects deduplicate safely;
7. no mutable surface object is aliased across workspaces improperly.

Textual agreement or same printed formula is insufficient.

### 13.2 Conflict classes

- universe/profile conflict;
- assumptions contradiction;
- rule-registry conflict;
- incompatible surface reconstruction;
- mutable snapshot conflict;
- proof rejection;
- result-form preference conflict;
- operational quota conflict.

Conflicts produce machine-readable witnesses.

## 14. Distributed worker protocol

Workers are treated as untrusted candidate generators.

### 14.1 Work packet

```text
WorkPacket
├── packet/bundle ID
├── exact subgoal/claim
├── immutable object dependencies
├── allowed strategy/version
├── context/registry IDs
├── budget and deadline
├── deterministic seed/counter lease
├── required certificate schema
├── capability/security policy
├── output size bounds
└── optional repair envelope
```

### 14.2 Worker response

```text
WorkResponse
├── packet ID
├── candidate value/object IDs
├── certificate/proof candidate
├── generator receipt
├── consumed budget
├── logs/trace digests
└── terminal status
```

The coordinator checks packet binding, deduplicates responses, validates bounded schemas, and verifies locally. Worker claims, signatures, consensus, or majority votes do not replace mathematical verification.

### 14.3 Lease semantics

- deterministic subtask/seed ranges are leased;
- retries are idempotent;
- duplicate responses are safe;
- cancellation revokes future publication rights but late responses remain harmless candidates;
- workers cannot write verified caches or branch heads directly;
- transport retries cannot double-charge authoritative progress.

## 15. Byzantine and corruption model

The design assumes workers, caches, indexes, and storage may be buggy or malicious.

Controls:

- canonical content IDs and payload confirmation;
- local claim-specific verification;
- capability-scoped object access;
- bounded decoders and allocation preflights;
- replay-resistant packet binding;
- separate candidate and verified namespaces;
- immutable registry/profile IDs;
- quarantine and incident ledgers;
- optional signatures for origin, not mathematical truth;
- e-process monitoring of defect streams.

The exact verifier kernel and canonical decoder remain the critical trust boundary.

## 16. Garbage collection and retention

Objects are classified:

- authoritative and pinned;
- authoritative but retention-governed;
- rebuildable derived state;
- staging/unpublished;
- quarantined;
- expired candidate telemetry.

GC traces typed references from pinned workspace/release/replay roots. It understands RaptorQ source/repair groups and never deletes the last required authoritative dependency while retaining a manifest that claims replayability.

GC plans are previewable and auditable. Repair symbols are reclaimed with their source object's policy, not as unrelated blobs.

## 17. Privacy and capability boundaries

Expressions and proofs may contain proprietary formulas or sensitive data.

Policies include:

- local-only versus remote-eligible objects;
- encrypted storage/transport capability;
- redacted observability;
- per-workspace object access;
- no raw expression text in metric labels;
- export controls for replay/proof bundles;
- plugin and worker capability allowlists;
- secure deletion as an explicit policy separate from ordinary GC.

Content-addressed identity can reveal equality across tenants if globally shared. Multi-tenant deployments therefore scope object stores and deduplication according to privacy policy.

## 18. Schema evolution

Persistent schemas are versioned. A migration declares:

- source/target schemas;
- syntactic transformation;
- semantic invariants;
- proof/verification impact;
- rollback/rebuild strategy;
- test corpus;
- migration receipt.

Authoritative mathematical objects prefer immutable old-schema preservation plus deterministic conversion. In-place rewrites are avoided when they would destroy replay provenance.

Unknown fields/versions fail closed at security or proof boundaries.

## 19. Operational evidence

Persistence claims require artifacts:

| Claim | Required evidence |
|---|---|
| crash consistent | exhaustive/informed crash-point matrix and recovered-state oracle |
| resumable | cancel/crash/restart equivalence and version-boundary tests |
| repairable | symbol-loss matrix, successful decode, canonical digest match, decode record |
| distributed safe | malformed/byzantine worker corpus plus local verification |
| deterministic replay | canonical bundle and terminal digest across fresh processes |
| graph index rebuildable | drop/rebuild/equivalence tests against authoritative ledger |
| cache sound | corruption, stale-universe, evidence-downgrade, and replay tests |

No durability badge is published merely because a RaptorQ encoder exists.

## 20. Testing program

### 20.1 Storage faults

- torn writes at every publication boundary;
- missing/truncated/duplicated/reordered records;
- stale branch heads;
- corrupt canonical blobs/manifests/indexes;
- quota exhaustion and disk full;
- concurrent GC/repair/read/resume;
- schema mismatch and downgrade attempts.

### 20.2 Repair faults

- arbitrary source/repair symbol loss within/outside recovery envelope;
- corrupt symbols with valid-looking framing;
- wrong object/parameter manifest;
- partial decode and readback failure;
- repaired bytes that fail canonical digest;
- successful byte recovery followed by failed proof verification.

### 20.3 Distributed faults

- duplicate/late responses;
- invalid certificates;
- candidate for wrong claim/context/domain;
- oversized/decompression-bomb payloads;
- worker equivocation;
- coordinator cancellation/restart;
- network partition and retry storms;
- malicious graph-index recommendations.

### 20.4 Branch faults

- conflicting assumptions/rules/profiles;
- verified edge invalid under new registry;
- mutable snapshot changed;
- same print, different term/domain;
- candidate mislabeled as verified;
- partial merge crash and replay.

## 21. Forbidden shortcuts

- making database rows the identity of terms or proofs;
- requiring a transaction for every in-memory algebraic operation;
- trusting a stored `verified` bit without canonical validation/replay policy;
- storing heuristic candidates in the verified cache namespace;
- treating graph reachability as entailment;
- allowing workers to publish verified state directly;
- treating worker majority or reputation as proof;
- claiming RaptorQ restores authenticity or mathematical validity;
- repairing bytes without checking the canonical object digest;
- silently rebasing a running computation to new assumptions/rules;
- merging branches textually or by printed expression;
- serializing arbitrary process memory as a checkpoint;
- deleting authoritative dependencies while promising replay;
- using global cross-tenant content deduplication without privacy policy;
- documenting designed persistence features as live before fault-injection gates pass.

## 22. First persistent vertical slice

The initial implementation must prove:

1. an expensive exact polynomial job publishes a typed checkpoint;
2. cancellation occurs after checkpoint publication and all tasks drain;
3. source/checkpoint bytes are damaged within the configured repair envelope;
4. RaptorQ reconstructs candidate bytes;
5. canonical digest and schema checks succeed;
6. dependencies and verifier universe validate;
7. the job resumes deterministically and produces a verified result;
8. a fresh process replays the result and proof receipt;
9. an invalid remote candidate is rejected without cache/branch pollution;
10. an optional FrankenGraphDB projection is deleted and rebuilt without changing authoritative results.
