# Portable claim, certificate, and artifact protocol

**Status:** normative architecture contract  
**Working name:** Franken Mathematical Artifact Protocol (FMAP)  
**Scope:** canonical mathematical object graphs, verifier-complete capsules, staged publication journals, delta synchronization, offline verification, repair, and transfer adapters

## 1. Purpose

FrankenSymPy needs a language-neutral way to package and exchange mathematical state without making Python objects, database rows, network sessions, or pretty-printed strings authoritative.

FMAP defines:

- canonical mathematical object kinds;
- typed content identities;
- Merkle-closed manifests;
- verifier-complete `Claim + Certificate` capsules;
- replay and compatibility bundles;
- staged publication journals;
- semantic and byte-level deltas;
- bounded streaming;
- transport-independent verification;
- optional ATP, persistence, compression, encryption, and repair adapters.

FMAP is inspired by asupersync ATP's verified object graphs, canonical manifests, resume journals, proof bundles, and delta transport. Its trusted core is intentionally smaller and contains no runtime, networking, persistence, RaptorQ, or database dependency.

## 2. Layering

```text
fsym-artifact-core
├── object kinds and typed IDs
├── canonical encoding
├── manifest and Merkle closure
├── capsule schemas
├── bounded decoder
└── no runtime / I/O / network

fsym-artifact-verify
├── closure validation
├── canonical object resolver
├── claim/certificate dispatch
└── embeddable verifier integration

fsym-artifact-store
├── memory/file/CAS adapters
├── staging and publication journal
└── optional std

fsym-artifact-delta
├── semantic object have-set
├── Merkle subtree diff
├── semantic container chunks
└── content-defined byte fallback

fsym-artifact-atp
├── asupersync ATP transport
├── path/transfer planning
├── resume and swarm
├── RaptorQ/Slepian–Wolf policy
└── no mathematical authority
```

The dependency arrows point downward. `fsym-artifact-core` and portable verifier crates do not depend on `fsym-artifact-atp`.

## 3. Object kinds

Initial authoritative kinds:

```text
Term
OperatorDefinition
Kind
Domain
Coercion
AssumptionsContext
PredicateFact
BranchPolicy
BinderMap
Claim
Certificate
ProofNode
Counterexample
VerifierManifest
RegistryFragment
Continuation
CheckpointFrontier
GeneratedProgram
CompatibilityObservation
```

Non-authoritative but content-addressed support kinds:

```text
DecisionReceipt
ExecutionReceipt
TraceFragment
SearchCandidate
BenchmarkObservation
MutationFixture
FuzzFixture
HumanExplanation
Signature
RepairManifest
```

Every object declares:

- object-kind schema;
- canonical payload length;
- content ID;
- authoritative dependencies by typed ID;
- optional metadata dependencies;
- privacy/export class;
- size and resource hints that are never trusted without validation.

Object kind is part of the content-ID domain separator. Equal payload bytes under different kinds are different objects.

## 4. Identity hierarchy

FMAP distinguishes:

- `ObjectContentId`: canonical bytes of one typed object;
- `ManifestId`: canonical ordered object graph and roots;
- `CapsuleId`: manifest plus capsule purpose and verifier profile;
- `WorkspaceSnapshotId`: mutable branch head over immutable objects;
- mathematical `TermId`, `ClaimId`, and related IDs defined by their semantic registries.

Transport chunk IDs, compression blocks, RaptorQ symbols, file paths, database keys, and graph vertices are not mathematical IDs.

At trust boundaries, digest equality is followed by canonical payload and kind validation. Collision response is fail-closed and incident-worthy.

## 5. Canonical encoding

The encoding specifies:

- integer widths, signedness, and endianness;
- arbitrary-precision limb and sign normalization;
- exact rational normalization;
- map and set ordering;
- sequence ordering;
- string normalization policy;
- float bit policies, including NaN and signed zero where floats are permitted;
- optional and critical fields;
- binder and alpha-equivalence representation;
- length-prefix and nesting bounds;
- domain separation for every object and manifest;
- schema-version handling.

Authoritative canonical encoding is implemented by owned safe-Rust code. Serde may provide convenience adapters but does not define identity.

Unknown critical fields fail closed. Unknown optional metadata may be preserved as opaque bytes only when it cannot alter identity, proof, capability, resource, or compatibility semantics.

## 6. Manifest

```text
FmapManifest
├── schema
├── purpose
├── root object IDs
├── canonical ordered object table
├── typed dependency edges
├── object sizes
├── verifier profile
├── required schema registry
├── closure digest / Merkle root
├── declared limits
├── optional signatures
└── optional transport descriptors
```

Validation checks:

1. root and object kinds are legal for the purpose;
2. every authoritative edge resolves;
3. no duplicate ID maps to different bytes;
4. graph depth, count, and total bytes fit limits;
5. the canonical object table and Merkle root match;
6. schemas and verifier profile are supported;
7. optional metadata does not enter authoritative preimages.

A manifest is a completeness and integrity structure. It does not prove the mathematical claim inside it.

## 7. Capsule purposes

### 7.1 Verifier-complete capsule

Contains the minimal closed object set required to check a typed claim.

This is the default interoperability unit and the public embeddable-verifier promise.

### 7.2 Replay-complete bundle

Adds:

- algorithm and planner registries;
- decision card;
- deterministic random stream roots;
- execution trace;
- candidate and rejection records;
- continuation/checkpoint state;
- build/environment fingerprint.

Replay completeness is operational evidence, not proof.

### 7.3 Compatibility-complete bundle

Adds:

- immutable compatibility profile;
- Python/platform/dependency environment;
- raw and normalized observations;
- comparator;
- warnings/exceptions/printer/pickle artifacts;
- upstream oracle source identity.

### 7.4 Publication-complete bundle

Adds:

- verification receipt;
- durable-object acknowledgements;
- publication journal closure;
- cache/branch pointer;
- signatures and authorization records;
- repair policy and scrub state.

### 7.5 Research bundle

May include the full search graph, rejected candidates, heuristics, model outputs, and notes. These remain explicitly non-authoritative.

## 8. Verifier-complete cut

Given a claim root and verifier profile, the system computes the transitive minimal closure required for checking:

```text
VerifierCompleteCut(claim, profile)
    -> roots + exact dependency object IDs
```

The cut excludes:

- generator search history not used by the certificate;
- planner telemetry;
- caches;
- human explanations;
- transport metadata;
- database and graph indexes;
- unrelated registry entries.

A cut is content-addressed and deterministically reproducible.

Uses:

- external verification;
- proof-carrying cache admission;
- highest-priority artifact transfer;
- repair-value calculation;
- garbage-collection rooting;
- incident replay;
- generated-code attestation.

## 9. Staged journal

FMAP publication uses an append-only state machine:

```text
RequestAccepted
ObjectStaged
CandidateProduced
CandidateCanonicalized
CertificateProduced
VerificationStarted
VerificationRejected
VerificationAccepted
ArtifactDurable
PublicationIntent
PublicationComplete
Cancellation
Rollback
CompactionBoundary
```

Each record is:

- schema-versioned;
- bound to exact object, claim, capsule, and prior journal state;
- optionally authenticated;
- canonical and replayable;
- redaction-safe in public summaries.

State law:

```text
produced != canonical
canonical != verified
verified != durable
durable != published
```

A cache or workspace branch can reference only `PublicationComplete` verified objects. Recovery may reuse only stages whose required bytes, identities, and verification closure remain valid.

## 10. Evidence escrow and two-phase publication

Candidate objects live in a private escrow namespace.

Phase 1:

- stage candidate and certificate;
- canonicalize;
- compute capsule;
- verify independently;
- write durable objects if policy requires.

Phase 2:

- atomically publish the value, claim, evidence class, verifier receipt, and capsule ID;
- make the verified cache or branch pointer visible;
- cancel and drain losing generators;
- finalize journal and resource obligations.

No generator, remote worker, database row, or graph index can bypass escrow.

## 11. Incremental proof availability

FMAP tracks a **proof availability frontier**.

A subclaim is available when:

- its claim object is present;
- its exact term/domain/context closure is present;
- its certificate closure is present;
- the selected verifier has accepted it;
- any required publication/durability policy is satisfied.

Progress reports therefore state:

```text
verified_subclaims
verifier-complete bytes present / total
remaining object IDs
blocked dependencies
global completeness status
```

They never present byte percentage as mathematical completion.

## 12. Streaming order

Default priority:

1. manifest header and verifier profile;
2. claim roots;
3. domain, context, branch, operator, and registry roots;
4. certificate objects that unlock the earliest subclaim;
5. source terms needed by those certificate steps;
6. remaining verifier-complete closure;
7. replay and provenance extras;
8. repair symbols;
9. speculative neighboring objects.

The sender may produce a deterministic proof-unlocking utility plan. The receiver can request missing exact object IDs and advertise an exact verified have-set.

## 13. Semantic delta synchronization

### 13.1 Object have-set

Peers exchange compact summaries or explicit sets of known, hash-validated object IDs. No object is skipped based solely on an unverified claim of possession when the receiving application requires local proof closure.

### 13.2 Merkle subtree diff

Canonical manifests support changed-subtree discovery so branches exchange only new or replaced object graph regions.

### 13.3 Semantic container chunks

Large objects define stable semantic chunk boundaries:

- coefficient blocks;
- sparse row/column groups;
- monomial ranges;
- proof-node blocks;
- generated-program sections;
- fixture shards.

Chunking strategy and version are transport descriptors; re-chunking does not change the authoritative object ID.

### 13.4 Content-defined fallback

Opaque blobs use content-defined byte chunks and sub-chunk delta. Every reconstructed object is rehashed against its authoritative ID.

### 13.5 Side-information coding

Slepian–Wolf-style coding may be tested for closely related large artifacts. It is optional, negotiated, bounded, and followed by exact hash validation. It never changes claim or certificate semantics.

## 14. Checkpoint and continuation artifacts

A checkpoint bundle contains:

- exact request and universe;
- algorithm/version;
- verified durable subclaims;
- canonical unverified private frontier;
- remaining deterministic work frontier;
- random counter leases used and reserved;
- consumed resource accounting;
- resume preconditions;
- verifier and schema versions;
- capsule and Merkle roots.

It excludes:

- raw stack or heap images;
- process pointers;
- open file/network handles;
- runtime task IDs as semantic state;
- unbounded logs;
- database row identity.

Resume validates every dependency and opens a new owned region. Incompatible versions return a typed refusal or use an explicit verified migration.

## 15. ATP transport adapter

The optional asupersync adapter maps:

- FMAP objects to ATP verified object kinds;
- FMAP manifests to ATP object graphs and transfer manifests;
- FMAP object chunks to ATP chunk/delta mechanisms;
- publication and checkpoint artifacts to ATP journals;
- verifier-complete cuts to transfer priorities;
- missing object IDs to receiver have-set negotiation;
- FMAP proof bundles to ATP offline transfer evidence;
- artifact repair policy to adaptive RaptorQ;
- remote sources to ATP swarm scheduling.

The adapter must preserve FMAP canonical bytes exactly. Transport-level compression, encryption, repair, relay, mailbox, and path choices cannot alter the mathematical object graph.

FMAP can also be carried over a local file, standard input, browser storage, an application protocol, or another transfer system.

## 16. Repair law

Repair applies only to selected immutable byte objects.

Trust sequence:

1. recover candidate bytes;
2. validate chunk and object hashes;
3. validate object kind and canonical encoding;
4. validate manifest closure;
5. validate authorization/origin when required;
6. rerun mathematical verification.

Repair success cannot set a mathematical evidence class.

Adaptive redundancy considers artifact value and recomputation cost, with operational monitor outputs influencing future policy only.

## 17. Remote worker exchange

A work packet is an FMAP bundle containing:

- exact subclaim or generation objective;
- immutable input object closure;
- allowed algorithm/version;
- leased subgoal and random-counter range;
- resource and output limits;
- required certificate schema;
- privacy/export capability;
- response capsule requirements.

A worker response contains candidate objects and a certificate capsule. The coordinator:

- bounds and decodes;
- binds it to the packet;
- deduplicates;
- verifies independently;
- publishes only through evidence escrow.

Worker signatures, reputation, majority, or ATP transfer proof do not replace mathematical verification.

## 18. Security

Decoders preflight:

- total bytes;
- object count;
- depth and cycles;
- integer limbs and declared dimensions;
- proof and edge counts;
- chunk and repair expansion;
- string and metadata sizes;
- continuation frontier size;
- optional-field preservation limits.

No native FMAP decode executes code.

Python pickle is never an FMAP object encoding. It may be carried as opaque compatibility evidence only under an explicit unsafe capability and is never automatically loaded by core tooling.

Privacy-sensitive deployments may use tenant-scoped or keyed object IDs; cross-tenant deduplication is an explicit policy because global content IDs leak equality.

## 19. Determinism

For fixed authoritative objects and schemas:

- object IDs;
- manifest ordering;
- Merkle roots;
- verifier-complete cuts;
- semantic delta;
- publication state transitions;
- canonical rejection witnesses;
- replay bundle identity

are deterministic.

Path, chunk, compression, encryption, repair-symbol, peer, and wall-clock metadata are excluded from mathematical object identity.

## 20. Conformance gates

FMAP core requires:

- canonical cross-process and cross-architecture fixtures;
- decoder fuzzing and size-overflow tests;
- unknown-critical-field rejection;
- object-kind confusion attacks;
- hash/canonical-payload mismatch tests;
- manifest missing/duplicate/cycle tests;
- deterministic have-set and delta fixtures;
- staged-journal crash injection;
- evidence-escrow publication races;
- verifier-complete-cut minimality and closure tests;
- transport round trips over memory, file, and ATP;
- RaptorQ recovery followed by hash/schema/verifier separation;
- malicious remote response corpus;
- no-runtime/no-network dependency gates for core and verifier crates.

## 21. Initial slice

The first slice supports:

- integer, rational, symbol, `Add`, `Mul`, `Pow`, and polynomial objects;
- `ZZ` and `QQ` polynomial domains;
- assumptions contexts needed by those objects;
- polynomial identity and factorization claims;
- factorization certificates;
- verifier manifests;
- whole-capsule and resolver verification;
- memory and file object stores;
- staged publication journal;
- semantic object have-set and Merkle diff;
- an ATP adapter prototype;
- a corrupted-object and resume test.

The slice succeeds only when the same capsule verifies through:

1. a tiny standalone Rust consumer;
2. the FrankenSymPy CLI;
3. the Python native API;
4. a WebAssembly checker, when the declared verifier profile supports it;
5. an ATP transfer round trip.

## 22. Forbidden shortcuts

- strings or printer output as authoritative objects;
- database rows or graph vertices as object identities;
- transport chunk IDs as term/proof IDs;
- a capsule that requires online lookup without saying so;
- automatic network fetching by the verifier;
- generator or planner state in the minimal verifier closure;
- publishing verified state before the exact capsule is accepted;
- resuming from unverified or non-durable subwork;
- RaptorQ, signatures, replay, or worker consensus as mathematical evidence;
- semantic identity changes caused by compression, encryption, chunking, or transfer paths;
- pickle/code execution in core decode;
- claiming transfer completion as proof completion.
