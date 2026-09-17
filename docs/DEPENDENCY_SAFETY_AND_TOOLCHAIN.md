# Dependency, safety, and toolchain constitution

**Status:** normative architecture contract  
**Scope:** Rust safety, dependency admission, FrankenSuite reuse, FFI boundaries, nightly policy, reproducibility, supply-chain evidence, and prohibited shortcuts

## 1. Non-negotiable posture

FrankenSymPy is a pure-Rust implementation of its mathematical engine. Project-authored Rust uses `#![forbid(unsafe_code)]` in every authoritative crate, including all reference verifiers and semantic cores.

No C, C++, Fortran, CUDA, GMP, FLINT, Arb, Singular, BLAS/LAPACK, or other native algorithm library may become a hidden implementation dependency. Performance must come from algorithms, representations, specialization, parallel decomposition, and safe-Rust implementation quality.

The CPython compatibility boundary is necessarily an FFI boundary supplied by a foundational Rust binding crate. FrankenSymPy-authored bridge code remains safe Rust and cannot be part of portable verifier closure.

## 2. Dependency universe

Dependencies fall into four classes.

### 2.1 Rust standard/toolchain

- `core`, `alloc`, `std`;
- the exact pinned Rust nightly and components for a release;
- compiler-provided architecture intrinsics available through safe APIs.

### 2.2 FrankenSuite foundations

Narrowly reviewed crates from:

- asupersync;
- FrankenNumPy;
- FrankenSciPy;
- FrankenNetworkX;
- FrankenGraphDB;
- FrankenSQLite;
- FrankenLean;
- other Dicklesworthstone-owned libraries only after an explicit registry row.

Same ownership is not automatic admission. The exact crate, features, commit, dependency closure, safety posture, determinism contract, and use boundary are reviewed.

### 2.3 Fundamental external Rust crates

Permitted only when reimplementation would increase risk or distract from mathematical differentiation. Candidate categories include:

- serialization scaffolding such as `serde`;
- cryptographic hash primitives;
- a safe CPython binding scaffold;
- platform/build metadata;
- Unicode tables when required for exact compatibility;
- other small foundational crates with a narrow, audited contract.

Algorithmic crates, generic runtimes, storage engines, network stacks, graph engines, CAS libraries, and numerical engines are not “fundamental” merely because they are popular.

### 2.4 Development-only tools

Formatters, linters, fuzz runners, benchmark drivers, license scanners, and test utilities may be admitted separately. They cannot leak into production or portable verifier dependency closure.

### 2.5 Optional offline formal checker

`registries/dependencies.toml` records a distinct `lean_core_external` source for the optional L6 `fsym-formal` projection gate, **planned pending independent review**. It is not an expansion of the native mathematical dependency universe or of the development-tool exception. The purpose is independent foreign-kernel checking of the bounded `lean_core_zz_product_v1` ZZ coefficient product identity; native verification remains sufficient without it. Existing `franken_lean` is a separate donor at commit `7df7a3170045882f3ab1f13bfee72338e524d174`, not admission for upstream Lean and not an established executable pipeline for this profile.

The candidate checker is the already installed upstream `leanprover/lean4` release 4.32.2, commit `f3b06c705e6c85f5314019d5d3baab0fec5b580c`, `x86_64-unknown-linux-gnu` Release. Only Lean core is admitted: no mathlib, plugins, native execution through `--run`, Cargo linking, FFI, network downloads, build scripts or runtime code loading. Repository code remains under `forbid(unsafe_code)`. Upstream Lean contains a native C/C++ runtime; external-process containment does not make that runtime memory-safe or grant a native algorithm/FFI exception.

The installed executable-plus-complete-`lib`-tree manifest is frozen at SHA256 `88d1bfed5e2ba13e7dd28043c70a303f1ea1b301220023fe6b9a1a5d14368f9f`, using `tools/check_formal_projection.py:environment()`. Its per-file rows provide the installed closure inventory in place of a Cargo package graph. Host system libraries outside that installed tree are not attested by this digest and remain an explicit review limitation. Replacing the binary, libraries, semantic mapping or environment requires a new profile/pin review, not silently updating the old immutable profile.

The admission record covers local filesystem and subprocess effects, exact environment and source grammar, empty `LEAN_PATH`, single-threaded checker operation, a 60-second timeout and bounded output. No network or entropy is needed for the admitted check; host timing and paths remain telemetry, never semantic identity. Source is restricted to fixed core definitions and kernel-checked proof terms; only `propext` is allowed in the trusted-axiom report, and extra diagnostics or axioms reject the check. Exact resource limits, checker invocation, native-first mapping/receipt requirements and no-mock/mutation/rejection gates are specified in [formal proof interoperability](FORMAL_PROOF_INTEROPERABILITY.md#102-optional-checker-interface-and-gates).

The installed v4.32.2 `LICENSE` identifies upstream Lean as Apache-2.0. Its separate `LICENSES` file records additional bundled-library notices (including LLVM's Apache-2.0 with LLVM exceptions); upstream licensing is not a claim that every bundled file is solely Apache-2.0. Advisory review, bundled-runtime/license review and independent admission review remain pending. No clean advisory scan, process-memory-safety result or successful executable gate is claimed here.

This external checker is optional Linux host tooling, not a Wasm or `no_std` dependency. It is owned through the L6 adapter boundary and cannot enter L2 or the portable verifier closure. Containment/removal is to omit `fsym-formal`, the gate and the installed checker; native verification, evidence authority and mathematical identity are unchanged. An external success boolean cannot replace native verification or independent receipt checking.

## 3. Admission record

Every non-`std` dependency row records:

- package/crate and exact source pin;
- purpose and rejected alternatives;
- owning workspace crates;
- enabled and disabled features;
- complete transitive dependency count and lock digest;
- build scripts and proc macros;
- project-authored and dependency-authored unsafe findings;
- FFI/native code;
- filesystem, process, network, environment, time, and entropy access;
- determinism and serialization effects;
- Wasm and `no_std` impact;
- license and maintenance status;
- security advisories;
- containment boundary;
- replacement/exit plan;
- reviewer and review freshness.

Unknown fields block admission.

## 4. Safety claim scope

Public claims distinguish:

- `project_authored_safe_rust`: no project-authored unsafe in the named crates;
- `portable_verifier_no_ffi`: verifier closure has no FFI or host runtime;
- `pure_rust_algorithm_path`: no native algorithm dependency;
- `dependency_tree_reviewed`: exact lock graph reviewed to the named policy;
- `process_memory_safe`: a stronger claim requiring all transitive/runtime boundaries to support it and therefore not assumed merely from Rust source.

The project must not claim that CPython or arbitrary third-party Python extensions become memory-safe because FrankenSymPy is written in Rust.

## 5. Portable verifier dependency ceiling

Portable verifier crates may depend only on:

- `core`/`alloc` and optionally `std` according to verifier profile;
- canonical primitive/object crates;
- exact arithmetic crates owned by FrankenSymPy or explicitly approved FrankenSuite foundations;
- the minimum hash/codec primitives required by the certificate schema.

They may not depend on:

- asupersync runtime;
- Python bindings;
- databases;
- networking or transfer;
- planners, generators, telemetry, or monitoring;
- formal-prover adapters;
- logging frameworks that pull host services into the closure.

The optional offline Lean admission in §2.5 does not relax this ceiling. `fsym-formal` depends outward on the native proof kernel, never the reverse, and its separately executed checker is outside all native portable-verifier safety claims.

## 6. Rust nightly policy

Development tracks current nightly aggressively because the project is performance- and language-feature-sensitive. Reproducibility requires exact pins.

- `rust-toolchain.toml` pins the currently admitted nightly for the repository;
- a toolchain-advance change is its own reviewed commit;
- the change runs the complete local gate matrix before becoming authoritative;
- release artifacts record `rustc -Vv`, Cargo version, target specifications, components, linker, feature set, and lock digest;
- “latest nightly” describes the advancement policy, not a reproducible release coordinate.

The initial pin is `nightly-2026-08-20`. Advancing it does not require waiting for a release, but it requires evidence.

## 7. Safe performance

The project does not pre-authorize unsafe optimization islands. Safe performance techniques include:

- compact arenas using indices rather than raw pointers;
- cache-aligned safe structures;
- structure-of-arrays and specialized small representations;
- checked chunked allocation;
- safe portable SIMD or compiler autovectorization;
- architecture dispatch through safe interfaces;
- deterministic parallel iterators built over asupersync-owned regions;
- exact NTT/CRT and modular algorithms written in Rust;
- copy-on-write and immutable sharing;
- generated specialized kernels reviewed as Rust source.

If a proposed optimization appears to require unsafe, the default response is to redesign the representation or algorithm. Any future exception would require a constitutional amendment, not a local `allow`.

## 8. Python boundary

The Python shell is not in the mathematical trusted core. Rules:

- no hand-written CPython C;
- no project-authored unsafe;
- no borrowed Python pointer stored outside the binding framework’s safe lifetime model;
- no Python object or pickle bytes in a portable mathematical certificate;
- custom callbacks classified before speculative execution;
- interpreter, free-threaded, subinterpreter, GC, weakref, and shutdown behavior tested by profile;
- Python exceptions and warnings reconstructed from typed native errors, not stringly panics.

## 9. Supply-chain and source policy

- lock exact commits/versions;
- vendor or mirror release-critical source when required by reproducibility policy;
- hash source archives and generated code;
- record license and advisory scans;
- reject network-fetching build scripts in the canonical release path;
- reject code generation that depends on unpinned host tools;
- keep generated durable formats governed by checked schemas, not derived enum layout;
- preserve a machine-readable SBOM and provenance statement per release.

## 10. Prohibited shortcuts

- C/C++/Fortran algorithm backends;
- blanket `unsafe_code = "allow"` or hidden unsafe modules;
- `build.rs` downloading binaries or source;
- default features accepted without review;
- adding a broad crate to avoid implementing a small required primitive;
- treating a safe wrapper as proof that a native dependency is memory-safe;
- runtime dependency cycles between verifier and generator layers;
- floating Git dependencies in release artifacts;
- claiming latest-nightly reproducibility without an exact pin;
- using GitHub-hosted checks as the only evidence that a release passed.

## 11. Enforcement

The local gate shall validate:

- every workspace crate’s unsafe lint;
- dependency registry versus Cargo metadata/lock;
- forbidden package and native-link names;
- build scripts and proc macros;
- verifier dependency closure;
- exact toolchain fingerprint;
- license/advisory artifacts;
- no unregistered source pins;
- no documentation claim stronger than the recorded safety class.
