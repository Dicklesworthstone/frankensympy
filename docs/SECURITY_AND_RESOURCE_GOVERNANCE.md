# FrankenSymPy security and resource governance

**Status:** normative architecture contract  
**Scope:** hostile expressions and artifacts, denial-of-service controls, Python/plugin boundaries, multi-tenant privacy, supply chain, memory safety, incident handling

## 1. Threat model

FrankenSymPy may process untrusted expressions, proofs, pickles, notebooks, generated code, checkpoints, remote-worker responses, graph-index recommendations, and agent-produced work packets. Symbolic inputs are programs in disguise: tiny syntax can trigger exponential expansion, deep recursion, huge coefficients, pathological matching, or arbitrary Python callbacks.

The design assumes:

- inputs and serialized artifacts may be malformed or malicious;
- custom Python classes and plugins may be buggy, stateful, slow, or hostile;
- caches, storage, indexes, and remote workers may return corrupt data;
- resource exhaustion is a primary security failure mode;
- compatibility requirements sometimes preserve dangerous upstream behavior, which must be isolated from hardened/native modes rather than silently changed;
- Rust memory safety is necessary but not sufficient.

## 2. Security postures

### 2.1 Strict compatibility

Reproduces the selected SymPy profile within the strongest controls that do not alter observable behavior. Where the upstream contract necessarily permits code execution or unbounded behavior, the risk is explicit and capability-gated at deployment boundaries.

### 2.2 Native safe

Default native posture:

- bounded parsers and printers;
- no implicit code execution;
- explicit Python-hook capabilities;
- typed budgets and refusals;
- verified result publication;
- signed/authorized plugins and remote work according to deployment policy;
- safe serialization formats rather than pickle.

### 2.3 Hardened multi-tenant

Adds:

- strict per-tenant quotas and admission control;
- local-only default for sensitive objects;
- plugin/callback disablement or sandboxing;
- no pickle ingestion;
- scoped content stores/deduplication;
- encrypted transport/storage capabilities;
- aggressive expression-growth and output limits;
- audit and incident retention.

Posture is explicit in requests, receipts, and benchmarks.

## 3. Memory-safety policy

The project target is memory-safe Rust with `unsafe_code = "deny"` across ordinary crates. If an optimization eventually requires unsafe SIMD, FFI-free assembly, or memory mapping, it must be isolated in a tiny audited crate with:

- a safe total public API;
- documented invariants;
- scalar reference implementation;
- differential/property/fuzz tests;
- Miri/sanitizer or equivalent gates where applicable;
- architecture-specific CI;
- explicit approval and claim-registry impact.

No C/C++ CAS library, GMP/FLINT/Singular/Arb FFI, or upstream SymPy embedding is used as the production native engine.

The compatibility shell necessarily executes Python and may interact with third-party C extensions. FrankenSymPy's Rust memory-safety claim must be scoped so it does not falsely cover foreign extension code.

## 4. Input size preflight

Every decoder/parser reads a bounded header or streaming manifest before allocation. Limits include:

- total bytes;
- nesting depth;
- node/edge counts;
- declared integer limbs/coefficient sizes;
- matrix/tensor dimensions and element count;
- proof/certificate nodes and references;
- string/symbol lengths;
- decompression and repair expansion ratios;
- number and size of chunks;
- recursion/binder depth;
- plugin/object references.

Integer overflow, multiplication overflow, and impossible aggregate sizes are checked before allocation.

Unknown fields are ignored only in explicitly forward-compatible non-security metadata. Proof, identity, capability, and resource fields fail closed.

## 5. Expression denial of service

### 5.1 Amplification vectors

- distributive expansion;
- recursive simplification loops;
- AC matching explosions;
- e-graph saturation;
- Gröbner pair growth;
- coefficient swell;
- huge common denominators;
- branch/piecewise partition explosion;
- substitutions duplicating subgraphs;
- printer expansion/wrapping;
- solver enumeration of roots/cases;
- nested custom callbacks;
- adversarial assumptions queries;
- symbolic matrix fill-in.

### 5.2 Controls

- multidimensional nested budgets;
- pre-operation growth estimates and reservations;
- DAG-preserving algorithms and shared representations;
- local bounded e-graphs only;
- proof-size and output-size caps;
- deterministic fallback/refusal;
- resumable continuation rather than forced completion;
- admission control and per-tenant quotas;
- no automatic expand-all behavior in native/hardened mode;
- printer and explanation budgets independent of computation budgets.

Strict compatibility may reproduce an upstream-expanding call when the user explicitly invokes it, but deployments can still bound the process/request and return a typed operational failure.

## 6. Python hook and plugin boundary

### 6.1 Default assumptions

An arbitrary hook/plugin is not assumed:

- pure;
- deterministic;
- terminating;
- thread-safe;
- reentrant;
- memory-safe;
- side-effect-free;
- honest about its result.

### 6.2 Supervision

- explicit capability required for callback classes;
- bounded inputs/outputs and call counts;
- interpreter/GIL ownership tracked;
- reentrancy depth and nested budgets;
- exceptions/warnings captured by profile policy;
- no native verifier trusts a hook as a theorem without a checkable certificate;
- optional separate-process execution for hardened deployments;
- network/filesystem/environment access denied unless capability-granted;
- callback results labeled by provenance.

### 6.3 Plugin registration

A plugin manifest names:

- package/source identity and version;
- ABI/protocol versions;
- classes/operators/domains/rules/algorithms provided;
- capabilities requested;
- determinism/purity declarations;
- verifier/certificate support;
- compatibility profiles tested;
- resource limits;
- signature/authorization metadata.

Registry conflicts are deterministic and explicit. Load order never silently changes semantics.

## 7. Serialization policy

### 7.1 Canonical native formats

Term, proof, certificate, receipt, checkpoint, and bundle formats:

- never execute code during decode;
- are schema-versioned and bounded;
- use canonical encodings for content identity;
- validate all references and dependency kinds;
- separate metadata from authoritative semantic payloads;
- support streaming for large objects.

### 7.2 Python pickle

Pickle is code-execution-capable and therefore:

- supported only where the compatibility profile requires it;
- never accepted through the normal native/network protocol;
- disabled by default in hardened deployments;
- loaded only under an explicit unsafe capability and trust boundary;
- tested for profile behavior, but not advertised as a safe exchange format.

### 7.3 Generated code

Code printers and compilers produce artifacts; they do not execute them by default. Execution requires a target capability and preferably an isolated sandbox/process in hardened deployments. Source term IDs and generation receipts accompany output.

## 8. Proof and certificate attacks

Threats include:

- oversized allocation claims;
- cyclic references;
- verifier-confusion across claim/domain/context schemas;
- certificates for a weaker claim than advertised;
- duplicate/omitted factors, roots, S-pairs, or side conditions;
- numeric intervals produced without directed rounding;
- stored `verified=true` flags;
- registry-version substitution;
- malicious compressed proof macros.

Defenses:

- bounded canonical decoding;
- exact typed claim binding;
- verifier dispatch by immutable registry;
- independent scalar/reference lane;
- mutation and adversarial corpora;
- local replay for untrusted artifacts;
- no proof acceptance based on origin, signature, worker vote, or persistence metadata.

## 9. Cache and persistence attacks

- cache poisoning with a candidate labeled verified;
- stale context/profile/rule entries;
- hash collision or wrong canonical payload;
- manifest/blob substitution;
- rollback to older vulnerable schema/registry;
- cross-tenant cache leakage;
- GC removing replay dependencies;
- repair symbols paired with the wrong source object.

Controls include complete cache keys, canonical payload confirmation, evidence namespace separation, immutable universe IDs, monotonic publication records, scoped stores, replay validation, and quarantine.

## 10. RaptorQ and compression safety

RaptorQ decoders and compressed transports are bounded before work:

- source length and symbol parameters validated;
- maximum symbol/object counts;
- allocation and CPU estimates charged;
- duplicate/invalid symbol handling;
- canonical digest required after recovery;
- origin authorization checked separately;
- mathematical objects revalidated/verified separately.

Decode success is not authenticity, integrity, schema validity, or mathematical correctness.

## 11. Remote worker model

Workers receive least-privilege packet capabilities. They cannot:

- browse the workspace/object store;
- alter profiles/registries/branches;
- publish verified caches;
- allocate beyond packet quotas on the coordinator;
- cause local code execution through response payloads;
- select the claim they are judged against.

Responses are bounded, packet-bound, canonicalized, and locally verified. Duplicate, late, equivocal, and malicious responses are harmless except for bounded wasted resources.

## 12. Multi-tenant identity and privacy

Global content-addressed deduplication can leak that two tenants possess the same formula or artifact. Deployments choose among:

- tenant-scoped IDs/stores;
- keyed content IDs;
- shared public-object namespace plus private namespaces;
- no cross-tenant deduplication;
- explicit opt-in collaborative workspaces.

Logs and metrics use bounded category/size IDs, not raw formulas, symbol names, proof contents, or stable content IDs when those could leak equality.

Replay and counterexample bundles are export-controlled capabilities.

## 13. Authentication and authorization

The core library remains authentication-neutral but exposes capabilities for:

- workspace read/write;
- branch fork/merge;
- registry/profile administration;
- Python/plugin execution;
- persistence and export;
- remote execution;
- sensitive object retrieval;
- generated-code execution;
- unsafe pickle loading;
- claim/release approval.

Authorization checks happen before object retrieval or expensive decoding. A content ID is not an access token.

## 14. Supply-chain policy

- closed/minimal dependency universe;
- exact lockfiles and source provenance;
- no dependencies fetched at runtime;
- license/security review for foundational crates;
- reproducible release builds where practical;
- SBOM and toolchain fingerprint;
- dependency and feature diff review;
- no optional dependency silently enabling network/code execution;
- pinned asupersync and Franken-suite revisions per release;
- vendoring policy available for high-assurance builds.

A dependency update that changes serialization, arithmetic, hashing, threading, or Python ABI behavior triggers relevant profile and verifier gates.

## 15. Secrets and sensitive configuration

Secrets never enter:

- term IDs;
- canonical replay bundles;
- decision cards;
- ordinary logs/metrics;
- test fixtures;
- error messages.

Remote credentials, encryption keys, and service tokens are runtime capabilities obtained from deployment infrastructure. Persistence artifacts reference key IDs/policies, not raw keys.

## 16. Sandboxing

Core exact Rust algorithms require no sandbox beyond normal process isolation and budgets. Higher-risk components may be isolated:

- Python callbacks and third-party extensions;
- pickle loading;
- generated-code execution;
- untrusted parser plugins;
- external oracle processes;
- remote workers.

Sandbox boundaries declare filesystem, network, environment, process, memory, CPU, and wall-time capabilities. Sandbox failure is a typed execution outcome, never mathematical evidence.

## 17. Denial-of-service admission policy

Requests are classified by expected amplification and trust:

- low-cost local core operation;
- bounded medium symbolic operation;
- high-amplification exact portfolio;
- Python/plugin callback workload;
- large proof/certificate validation;
- remote/distributed workload;
- persistence/repair workload.

Policies can require an explicit maximum budget for high-risk classes. Repeated under-declaration, malformed objects, or verifier abuse can trigger rate limits/quarantine. Fairness protects small verified tasks from one enormous request.

## 18. Output and error safety

- diagnostic detail is bounded;
- printers cannot recursively explode without budget;
- exceptions do not include full proprietary expressions by default;
- malformed input excerpts are truncated/redacted;
- internal invariant failures generate incident IDs and protected bundles;
- public APIs distinguish user error, refusal, resource exhaustion, cancellation, verifier rejection, and internal fault;
- no panic crosses an FFI/Python/protocol boundary.

## 19. Internal fault containment

Rust panics are bugs. Boundary code catches/unwinds only where safe and converts to `InternalFault`; abort-only builds rely on process supervision. Shared state uses two-phase publication so a fault cannot expose partial verified entries.

After an invariant fault:

- affected cache/artifact namespaces are quarantined;
- no candidate is promoted;
- region children are cancelled/drained where possible;
- a minimized/replayable incident bundle is generated under privacy policy;
- e-process monitoring can pause the strategy/verifier rollout.

## 20. Security testing

### 20.1 Parser/decoder fuzzing

- arbitrary bytes and structured mutations;
- integer/length overflow;
- deep recursion/cycles;
- zip/decompression/repair bombs;
- cross-schema/type confusion;
- chunk duplication/reordering/truncation.

### 20.2 Resource adversaries

- expression swell families;
- AC/e-graph match explosions;
- pathological Gröbner/factorization/solver inputs;
- huge coefficient and matrix dimensions;
- printer/LaTeX/codegen bombs;
- recursive custom hooks;
- cache stampedes and remote retry storms.

### 20.3 Capability tests

- object-ID guessing across tenants;
- denied plugin/network/filesystem access;
- branch/profile/registry write attempts;
- export/replay authorization;
- late remote responses after revocation;
- unsafe pickle API separation.

### 20.4 Proof/persistence tests

- malicious certificates;
- false stored verification metadata;
- stale/rolled-back registries;
- repaired bytes with wrong digest;
- graph-index poisoning;
- GC/replay dependency races.

## 21. Claims and disclosure

Security claims are scoped:

- “memory-safe native core” does not cover arbitrary Python/C extensions;
- “bounded” names the dimensions and supported boundaries;
- “cancel-correct” does not imply forced interruption of non-cooperative hooks;
- “verified result” names the claim and verifier;
- “repairable artifact” names the loss model and digest/verification chain;
- “drop-in compatible” may preserve upstream-risky behavior in strict mode.

Known security limitations and compatibility-imposed hazards are documented rather than hidden behind the native/hardened mode.

## 22. Incident response

1. quarantine affected strategy/verifier/cache/profile artifact;
2. preserve immutable incident/replay bundle under privacy policy;
3. determine whether the issue affects mathematical soundness, compatibility, resource governance, confidentiality, or availability;
4. invalidate affected claims/cache entries/releases by dependency graph;
5. add minimized adversarial fixture and verifier mutation;
6. repair with independent review;
7. rerun full affected gates;
8. publish scoped advisory and artifact digests when appropriate.

A mathematical unsoundness incident receives the highest severity even if no memory corruption occurred.

## 23. Forbidden shortcuts

- implying Rust memory safety covers arbitrary Python/C extensions;
- enabling implicit pickle or generated-code execution;
- trusting plugin declarations without capability enforcement;
- allocating from untrusted length fields before preflight;
- using a single wall timeout as the resource model;
- letting diagnostics/printers bypass output budgets;
- accepting remote/cache/persistent verification flags;
- using RaptorQ decode success as integrity/authenticity/truth;
- global cross-tenant deduplication without leakage policy;
- content IDs as authorization tokens;
- silent plugin load-order semantics;
- swallowing internal faults and returning a candidate;
- weakening strict compatibility invisibly in the name of hardening;
- claiming universal cancellation of arbitrary callbacks;
- importing a C/C++ CAS through FFI to accelerate the core.

## 24. Initial security gate

The first implementation slice must pass:

1. bounded native term/proof/checkpoint/NDJSON decoders;
2. expression-growth, memory, depth, hook, proof, and output budget adversaries;
3. custom Python hook delay/reentrancy/exception tests;
4. candidate/verified cache poisoning attempts;
5. remote wrong-claim and oversized response rejection;
6. RaptorQ recovery followed by digest/schema/proof separation tests;
7. deterministic cancellation at publication boundaries;
8. no unsafe code in ordinary crates;
9. dependency/SBOM/provenance report;
10. privacy check proving logs/metrics do not contain raw expressions by default.
