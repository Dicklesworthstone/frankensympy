# Embeddable verifier compatibility contract

**Status:** normative architecture contract  
**Scope:** standalone verifier crates, portable claim/certificate capsules, dependency closure, deterministic APIs, resource bounds, versioning, release artifacts, and conformance

## 1. Promise

FrankenSymPy makes **independent verification** a public compatibility promise, not merely an internal layering preference.

For every result family advertised as certificate-verifiable, a caller must be able to receive a typed `Claim + Certificate` bundle and verify it without linking or initializing:

- the corresponding generator or algorithm portfolio;
- the planner, selector, learned policy, or benchmark machinery;
- the Python compatibility shell or CPython bridge;
- asupersync's executor, scheduler, networking, timers, or remote runtime;
- FrankenSQLite, FrankenGraphDB, or any persistent store;
- remote-worker, repair, RaptorQ, or transfer machinery;
- the service layer, CLI, observability stack, or agent workspace;
- upstream SymPy or any other computer algebra system.

The canonical user story is intentionally small:

```rust
use fsym_cert_factor::{verify_factorization, FactorVerifierLimits};

let verified = verify_factorization(
    claim_and_certificate_bytes,
    FactorVerifierLimits::default(),
)?;
```

The exact API may evolve before its first stable release, but the architectural promise does not: **verification is a synchronous, deterministic, dependency-light function over explicit bytes and explicit limits.**

## 2. Why this is a compatibility property

A certificate is most valuable when it survives the environment that produced it.

Independent embeddability allows:

- a database, notebook, browser, theorem prover, build system, archival service, or competing CAS to check FrankenSymPy results;
- a remote worker to remain completely untrusted;
- an old result to be rechecked after the generator has changed or disappeared;
- users to distinguish “I trust this verifier” from “I trust this entire symbolic engine”;
- verifier diversity, including separately implemented checkers;
- proof-carrying caches and generated code;
- narrow security review and fuzzing;
- small WebAssembly and embedded deployments;
- reproducible incident analysis;
- external standards and ecosystem adoption.

A verifier that imports the generator, asks the planner what the claim meant, opens a database, downloads omitted terms, or executes Python callbacks is not independently embeddable.

## 3. Product tiers

### 3.1 Portable verifier core

The strongest target profile is:

```text
pure Rust
no C/C++ FFI
no unsafe code in FrankenSymPy source
no async runtime
no OS clock
no entropy
no filesystem
no network
no process execution
no global mutable registry
no Python
no persistence
deterministic for fixed bytes and limits
bounded before allocation
```

Where the arithmetic and certificate family permit it, the verifier core compiles as `#![no_std]` with `alloc`. Exact arithmetic generally requires allocation; `core`-only support is not a blanket requirement.

A certificate family may claim `portable-no-std-alloc` only after that exact build and test gate passes. Families that currently require `std` remain usable through the hosted tier but cannot borrow the portable badge.

### 3.2 Hosted convenience verifier

A small `std` wrapper may add:

- file and stream readers;
- memory-mapped or buffered object resolvers;
- human-readable diagnostics;
- a stable command-line verifier;
- optional signature and provenance checking;
- parallel checking of independent certificate components.

The hosted wrapper delegates mathematical acceptance to the same portable verifier logic. It may not silently fetch missing dependencies or upgrade evidence.

### 3.3 Service integration

FrankenSymPy services may wrap verification in asupersync regions for cancellation, quotas, transport, and observability. The mathematical verifier remains callable without that wrapper.

When a service uses `Cx`, the verifier receives a **pure verifier capability profile**: no spawn, time, entropy, I/O, remote, Python, persistence, or publication authority. The service transports bytes to the verifier and publishes the returned verdict; it does not become part of the proof.

## 4. Crate boundary

The intended dependency direction is:

```text
fsym-cert-factor
    ├── fsym-verify-core
    ├── fsym-claim
    ├── fsym-term-codec
    ├── fsym-domain-codec
    ├── fsym-bigint-verify
    ├── fsym-poly-verify
    └── fsym-canonical

fsym-factor-generator
    ├── fsym-cert-factor          # may emit and self-check certificates
    ├── fsym-poly
    ├── fsym-plan
    ├── fsym-portfolio
    └── fsym-cx
```

The reverse edge is forbidden:

```text
fsym-cert-factor  -X->  fsym-factor-generator
```

The same law applies to every certificate family. Shared code is admitted into a lower trusted crate only when:

1. the functionality is genuinely required for checking;
2. it is substantially simpler than the generator;
3. its semantics are independently specified;
4. it has a scalar/reference implementation;
5. mutations in that shared code are killed by independent fixtures;
6. moving it does not smuggle planner or generator logic into the verifier.

“Both sides need it” is not sufficient.

## 5. Canonical verification capsule

The default transport unit is a **verifier-complete capsule**. It contains or content-addresses the complete closure needed to check one claim:

```text
VerificationCapsule
├── capsule schema and canonical encoding version
├── claim schema and exact typed claim
├── certificate schema and certificate payload
├── term/operator/domain/context objects reachable from the claim
├── assumptions and branch-policy closure
├── required registry fragments
├── verifier profile and minimum verifier version
├── canonical object manifest and Merkle root
├── declared resource envelope
├── optional origin signatures
└── optional non-authoritative provenance
```

A capsule is accepted only when every authoritative dependency is present and its canonical bytes match its identity.

A **thin capsule** may omit objects already known to an explicit object resolver. The verifier then returns `MissingObject` with the exact content IDs. It never opens a network connection, searches a workspace, consults a database, or guesses a replacement.

Origin signatures and attestations are checked separately from mathematical validity. A valid signature can identify who emitted a false certificate; it cannot make the certificate true.

## 6. API contract

### 6.1 Simple whole-capsule API

Each portable family provides a small entry point of this shape:

```rust
pub fn verify_factorization(
    capsule: &[u8],
    limits: FactorVerifierLimits,
) -> Result<VerifiedFactorization, FactorVerificationError>;
```

The function:

- parses with bounded preflight;
- validates canonical encoding and all content identities;
- checks the claim/certificate schema pairing;
- verifies exact domain and normalization obligations;
- returns a typed verified statement or a typed failure;
- performs no ambient effects;
- is deterministic for fixed inputs and limits.

### 6.2 Resolver and streaming API

Large capsules use a lower-level API:

```rust
pub fn verify_factorization_with<R: ObjectResolver>(
    manifest: &CapsuleManifest,
    resolver: &R,
    limits: FactorVerifierLimits,
    scratch: &mut FactorVerifierScratch,
) -> FactorVerifyOutcome;
```

`ObjectResolver` is a synchronous read-only interface. Implementations may read memory, a file, a CAS, a browser store, or an application database, but those adapters are outside the verifier crate.

The verifier never trusts the resolver's metadata. Every returned object is rehashed, decoded, type-checked, and bound to the requested content ID.

### 6.3 Incremental verification

A verifier may expose a resumable state machine for enormous certificates:

```text
Need(object_ids)
Progress(verified_subclaims, remaining_budget)
Verified(statement, receipt)
Rejected(witness)
Inconclusive(reason, continuation)
```

A continuation is canonical algorithm state, not a stack or memory image. It binds the verifier version, claim, certificate, verified subclaims, remaining frontier, and consumed resource accounting.

Incremental checking must produce the same terminal mathematical verdict as the whole-capsule API under sufficient limits.

## 7. Outcome law

The verifier distinguishes:

```text
Verified
Rejected(counterexample_or_failed_obligation)
Inconclusive(resource_dimension, optional_continuation)
UnsupportedSchema
UnsupportedClaim
MissingObject(content_ids)
Malformed
IdentityMismatch
InternalInvariantFault
```

Rules:

- `Inconclusive` is never `Rejected`.
- resource exhaustion is never mathematical falsehood;
- unsupported schemas are never guessed or partially ignored;
- missing dependencies are never fetched implicitly;
- a malformed certificate cannot fall back to heuristic checking;
- an internal fault cannot return a candidate value;
- rejection should contain a bounded, machine-readable witness whenever practical.

## 8. Factorization verifier obligations

A factorization claim states exactly what is being claimed:

```text
FactorizationClaim
├── source polynomial
├── coefficient domain and polynomial ring
├── unit/content policy
├── factor normalization
├── ordered factors and multiplicities
├── decomposition completeness
├── square-free status, if claimed
├── primitive/monic status, if claimed
├── irreducibility level, if claimed
└── extension/absolute-factorization policy, if claimed
```

The verifier grants no stronger statement than it checks.

At minimum, a decomposition certificate checks:

1. every object belongs to the declared ring;
2. coefficient and monomial encodings are canonical;
3. units, content, signs, and normalizations are legal;
4. multiplicities are positive and represented exactly once;
5. multiplying the declared unit/content and factors reproduces the source exactly;
6. excluded denominators, extension embeddings, and coercions are accounted for;
7. the certificate is bound to the exact claim and registry versions.

Additional claims require additional evidence:

- square-free decomposition requires the corresponding gcd/derivative obligations;
- irreducibility requires a domain-appropriate irreducibility certificate for each factor;
- absolute irreducibility requires its stronger field-closure obligations;
- completeness over an extension requires the extension and embedding closure;
- a product check alone never proves irreducibility.

The generator may use modular images, Hensel lifting, lattice reduction, randomization, remote workers, or learned selection. The portable verifier need only check the compact final obligations.

## 9. Other verifier families

Each family owns a minimal, explicit claim language.

### Polynomial identity

Checks ring identity, canonical coercions, and exact zero difference or a reconstruction certificate with a valid bound and exact final check.

### GCD

Checks divisibility plus a maximality criterion appropriate to the domain, such as Bézout cofactors and normalization.

### Gröbner basis

Checks input-ideal membership, the exact monomial order and coefficient domain, and the Gröbner criterion. “All generated S-pairs processed” is generator history, not itself a verifier obligation.

### Exact linear algebra

Separates residual, existence, rank, uniqueness, determinant, nullspace dimension, eigenvalue completeness, and decomposition claims.

### Roots and solvers

Separates “each reported object is a solution” from coverage, multiplicity, disjoint isolation, excluded singularities, and completeness in the declared domain.

### Calculus

Separates formal differentiation, antiderivative, definite integral, convergence, branch, endpoint, distributional, limit, and remainder claims.

### Certified numerics

Checks directed-rounding enclosures and claim-specific error bounds. Ordinary floating-point agreement is not accepted as a certified enclosure.

### SAT and logic

Models certify satisfiability. Unsatisfiability and implication require a checkable proof trace in the supported logic.

## 10. Dependency law

Portable verifier crates may depend only on:

- `core` and `alloc`;
- narrow FrankenSymPy trusted crates required for canonical decoding, terms, domains, claims, exact arithmetic, and checking;
- explicitly admitted fundamental pure-Rust crates whose complete feature and unsafe closure passes the dependency registry.

They must not depend on:

- asupersync or any async executor;
- generator, planner, selector, portfolio, benchmark, or telemetry crates;
- Python bindings or profile shell code;
- persistence, graph, remote, transfer, compression, repair, or RaptorQ crates;
- CLI, service, logging, tracing, HTTP, database, filesystem, or process crates;
- build-time code generation whose source registry is absent from the release;
- default features that silently expand the dependency closure.

Serde may be used only in an adapter or when its exact no-std feature closure is admitted. The authoritative canonical decoder is not defined as “whatever serde implementation happens to emit.”

## 11. Memory-safety law

All FrankenSymPy verifier source uses `#![forbid(unsafe_code)]`.

No verifier links:

- C or C++ arithmetic;
- a foreign CAS;
- an external proof engine through FFI;
- hand-written CPython C API code;
- JIT-generated native code;
- an opaque binary checker.

A dependency containing unsafe code cannot enter the trusted verifier closure merely because its public API is safe. It requires an explicit audit record, feature-closed source review, and replacement rationale. The preferred outcome is an owned safe-Rust implementation.

## 12. Determinism law

For fixed canonical capsule bytes, verifier version, and limits, the verifier returns the same:

- terminal outcome class;
- verified statement;
- rejection code and canonical witness;
- consumed logical-resource accounting;
- verification receipt digest.

Wall time, thread count, hash-map iteration, filesystem order, locale, host RNG, and process identity do not affect the result.

Hosted parallel verification may alter latency but uses deterministic aggregation and witness selection.

## 13. Resource-safety law

Before allocating based on untrusted input, the verifier bounds:

- capsule and object bytes;
- object count and graph depth;
- integer limbs and coefficient height;
- polynomial variables, degree, monomials, and factors;
- matrix dimensions and nonzeros;
- proof nodes and references;
- recursion and binder depth;
- decompression or repair expansion;
- output and witness size.

Logical work is charged by operation-specific fuel, not only wall time. Limits are part of the receipt.

A certificate that is valid but too expensive under the supplied limit returns `Inconclusive(ResourceLimit)`. This preserves the ability to retry under a larger budget.

## 14. Stable schema and compatibility

Every claim, certificate, object, manifest, receipt, and continuation has an explicit schema identifier.

A verifier release publishes:

- accepted claim/certificate schema ranges;
- canonical encoding versions;
- trusted dependency manifest;
- supported portability profile;
- deterministic test corpus digest;
- mutation and fuzz results;
- known limits and unsupported claims.

Schema evolution rules:

1. unknown identity- or proof-relevant fields fail closed;
2. optional metadata cannot alter mathematical meaning;
3. migration is an explicit converter with its own receipt;
4. old bundles remain checkable by their pinned verifier artifact;
5. a new verifier may support old schemas only after regression and mutation closure;
6. a schema cannot be weakened in place to admit a previously invalid certificate.

## 15. Release artifacts

For each mature family, releases should include:

- the Rust verifier crate;
- source and exact lock/dependency manifest;
- a tiny `fsym-verify` CLI;
- a WebAssembly verifier when the portability profile permits;
- canonical positive and negative fixtures;
- mutation corpus and surviving-mutant report;
- fuzz corpus seeds and decoder limits;
- a verifier SBOM/trusted-base manifest;
- reproducible build instructions;
- optional signed binaries whose signatures establish origin only.

The CLI supports offline operation and never downloads missing objects by default.

## 16. Conformance gates

A verifier family cannot claim independent embeddability until all applicable gates pass:

1. `cargo tree` proves no forbidden dependency role;
2. `cargo build -p <verifier> --no-default-features` passes;
3. the declared `no_std + alloc` target builds when claimed;
4. `wasm32-unknown-unknown` builds when claimed;
5. source scanning finds no `unsafe`, FFI, networking, filesystem, process, Python, planner, generator, or persistence imports;
6. the verifier accepts valid independently generated capsules;
7. it rejects malformed, wrong-domain, wrong-context, wrong-claim, incomplete, and adversarial certificates;
8. registered checker mutants are killed;
9. optimized and scalar/reference verifier lanes agree;
10. random insertion, object order, and chunk order do not alter the verdict;
11. resource-limit tests return `Inconclusive`, not false;
12. a fresh minimal consumer application can verify a capsule without linking the main engine.

The final gate is literal: CI and the local release harness build a tiny external consumer crate containing only the verifier dependency and fixture.

## 17. Cross-implementation verification

The protocol is deliberately language-neutral and checker-oriented.

FrankenSymPy should encourage:

- a second pure-Rust reference checker for crown-jewel certificate families;
- a WebAssembly checker used from browsers and non-Rust hosts;
- an eventual FrankenLean bridge that imports checked claims or exports theorem-backed certificates;
- third-party checker implementations;
- differential comparison of checker verdicts;
- proof-certificate minimization.

Agreement between two checkers is useful evidence about implementations. It does not replace the exact claim-specific correctness argument for each checker.

## 18. Forbidden shortcuts

- making the verifier a feature flag of the generator crate;
- importing generator normalization code without an independent trusted specification;
- requiring a running FrankenSymPy service;
- using Python or upstream SymPy to finish a check;
- consulting a database row marked `verified`;
- allowing a graph index to supply omitted proof steps;
- fetching missing terms or registries from the network;
- accepting a certificate because it was signed by a trusted worker;
- interpreting timeout or exhaustion as rejection;
- product-only checking while claiming irreducibility;
- verifier behavior that depends on task completion order;
- hidden default features that link the runtime or persistence stack;
- advertising “embeddable” without the external minimal-consumer gate.

## 19. Initial implementation slice

The first portable verifier campaign includes:

1. canonical capsule and object-manifest decoding;
2. exact integer, rational, and polynomial objects;
3. polynomial identity claims;
4. factorization decomposition and multiplicity claims;
5. optional irreducibility subcertificates only when the checker is implemented;
6. deterministic bounded receipts;
7. whole-capsule and resolver APIs;
8. `no_std + alloc` and WebAssembly probes;
9. a tiny external example that calls `verify_factorization()`;
10. adversarial and mutation corpora.

The generator may arrive later. The verifier contract and negative fixtures should exist first so the generator targets a fixed acceptance boundary.
