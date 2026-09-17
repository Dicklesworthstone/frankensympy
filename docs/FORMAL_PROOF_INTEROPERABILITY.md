# Formal proof interoperability contract

**Status:** normative architecture contract  
**Scope:** native certificate verification, formal projections, theorem-prover adapters, projection receipts, trust boundaries, resource limits, and evidence wording

## 1. Principle

FrankenSymPy is certificate-native and theorem-prover-interoperable. It is not theorem-prover-dependent.

A native operation may produce a `Claim + Certificate + VerificationClosure`. The native reference verifier is the minimum authority for that native certificate family. A formal adapter may additionally translate the verified claim into a theorem-prover statement and proof object.

## 2. Three distinct objects

1. **Native claim:** the exact mathematical proposition in FrankenSymPy’s object model.
2. **Formal statement:** the proposition encoded in a named external logic and library environment.
3. **Projection receipt:** a checked mapping between the native claim root and formal statement root.

A foreign kernel proves the formal statement. Only a valid projection receipt connects that result to the native claim.

## 3. Projection profile

Each profile declares:

- target system and exact version/pin;
- logic and trusted kernel;
- imported library/environment root;
- native claim families supported;
- operator/domain/assumption mappings;
- branch and partiality policy;
- numeric encoding;
- treatment of undefined expressions;
- proof-object format;
- foreign-check command or API;
- resource bounds;
- unsupported semantic fields;
- evidence class produced.

Profiles are immutable. Updating a mapping creates a new profile ID.

## 4. Projection completeness

A projector must account for every semantic field that can change truth:

- domains;
- assumptions;
- variable binding and freshness;
- equality notion;
- branch policy;
- singularity and excluded-point conditions;
- exact versus approximate values;
- algebraic extension definitions;
- matrix dimensions and scalar field;
- operator definitions;
- certificate schema version.

Unknown or unrepresentable fields refuse projection. They are never ignored or serialized as comments.

## 5. Native-first admission

Default publication sequence:

1. decode and validate native closure;
2. run the native reference verifier;
3. mint `VerifiedNativeClaim`;
4. optionally project and check formally;
5. attach formal evidence to the publication record;
6. publish under the workspace transaction contract.

A formal-only research result may exist, but it is not presented as a native FrankenSymPy certificate result until a reviewed projection back to a native claim exists.

## 6. Formal checker independence

The foreign checker must not trust:

- the native generator’s verdict;
- a native decision receipt;
- the formal projector’s claimed success;
- cached theorem-prover output;
- signatures as truth;
- the same unchecked optimized kernel used by the generator.

The adapter records the exact checker binary/build/environment root and captures canonical semantic output separately from host telemetry.

## 7. Proof-term strategy

Prefer proof terms or tactic-independent certificates generated from native certificates. Tactics may help elaborate or compress, but the stored proof artifact must be kernel-checkable without replaying an unbounded heuristic search.

For large certificates, use compositional lemmas and chunked proof objects whose dependency closure is explicit.

## 8. Assumption handling

Assumptions are first-class objects. The formal statement exposes them as hypotheses, typeclass constraints, domain membership, or explicit side conditions according to the profile.

The adapter refuses:

- hidden global assumptions;
- silently strengthened preconditions;
- silently weakened conclusions;
- collapsing principal-value and ordinary equality;
- dropping nonzero-denominator conditions;
- treating generic symbols as real or positive without evidence.

## 9. Failure taxonomy

- `NativeVerificationFailed`
- `ProjectionUnsupportedFamily`
- `UnrepresentableSemanticField`
- `MissingFormalDependency`
- `FormalElaborationFailed`
- `ForeignKernelRejected`
- `ProjectionReceiptMismatch`
- `ResourceExhausted`
- `Cancelled`
- `CheckerInternalFault`

Only the first six and receipt mismatch are completed negative engineering outcomes. Resource exhaustion and cancellation remain inconclusive.

## 10. Initial adapters

The first planned adapter is Lean-oriented because FrankenLean supplies useful certificate, cartridge, and foreign-checking precedents. The architecture permits other theorem provers through separate profiles, provided they satisfy the same projection and authority contract.

### 10.1 Bounded Lean-core ZZ product profile

`lean_core_zz_product_v1` is a separate immutable profile in `registries/formal_proof_profiles.toml`. Its status remains **planned, pending independent review**. It does not modify `lean_factorization_v1` or the existing FrankenLean source pin, certify factorization, establish irreducibility, or promote a claim or workstream. FrankenLean supplies distinct precedents; that admission does not prove an executable pipeline for this upstream Lean-core profile.

The optional L6 `fsym-formal` adapter consumes the native L2 proof kernel. No native crate depends back on this adapter or on Lean. The only supported proposition is an exact ZZ coefficient-vector product identity, not a claim that the factors are irreducible or a general factorization procedure:

| Semantic field | Frozen mapping |
|---|---|
| Capsule/schema | Native capsule v1, exactly three objects: subject and two factors |
| Domain and numbers | ZZ only; exact signed i64 coefficients mapped to Lean `Int` |
| Polynomial representation | Dense ascending coefficient vectors; the same single variable for all three polynomials; degree at most 64 |
| Product | Exactly two factors, exponent 1 each, scalar coefficient 1 |
| Context | Raw root `0x1111222233334444`, no hypotheses |
| Rule / verifier | Raw roots `0x5555666677778888` / `0x9999aaaabbbbcccc` |
| Equality | Exact coefficient-vector equality after convolution, not sampled evaluation |
| Partiality / branches | Not applicable to this total integer product; undefined or otherwise unrepresented expressions are refused |
| Unsupported semantics | Every other domain, shape, exponent, scalar, hypothesis, semantic root or unknown field is refused |
| Input limits | Capsule 16 KiB, projected source 32 KiB, JSON envelope 128 KiB |

Native verification precedes projection. An independent mapping checker must reconstruct and validate the source-to-native binding and immutable receipt; caller-supplied success booleans never establish authority. The receipt binds the native claim, profile, statement and environment. Kernel acceptance alone proves only the formal statement; without the checked receipt it does not establish the native claim.

The adapter's current output label is `evidence=projection-mapping-only;foreign=not-checked`. Mapping acceptance is not `formal_projection_checked`: that evidence requires both the independent mapping receipt and connected foreign-kernel acceptance. Recording this planned profile or its checker command does not establish that either a foreign check or a conformance gate has passed.

The foreign environment is standalone **Lean 4.32.2 core**, upstream `leanprover/lean4` commit `f3b06c705e6c85f5314019d5d3baab0fec5b580c`, release platform `x86_64-unknown-linux-gnu`. The expected banner is `Lean (version 4.32.2, x86_64-unknown-linux-gnu, commit f3b06c705e6c85f5314019d5d3baab0fec5b580c, Release)`. No mathlib or user library is admitted. Fixed core definitions implement dense convolution and the equality proof uses a kernel-checked `decide` term; native execution, plugins and `--run` are not permitted. The trusted-axiom allowlist contains only `propext`; any other axiom or extra checker diagnostic is rejected.

The frozen installed-closure manifest SHA256 is `88d1bfed5e2ba13e7dd28043c70a303f1ea1b301220023fe6b9a1a5d14368f9f`. `tools/check_formal_projection.py:environment()` hashes the executable and each file in the complete installed `lib` tree, with prefix-relative paths and per-file SHA256 values, then hashes the canonical JSON manifest. A version banner alone is insufficient. This is an installed-tool manifest, not a source archive hash, Cargo lock digest, or attestation of host libraries outside that tree. The environment identifier bound to the statement and receipt is `lean4-core-4.32.2@f3b06c705e6c85f5314019d5d3baab0fec5b580c;sha256=88d1bfed5e2ba13e7dd28043c70a303f1ea1b301220023fe6b9a1a5d14368f9f`.

### 10.2 Optional checker interface and gates

Lean is an explicitly selected, already installed offline executable, not a Cargo dependency, native FFI binding, downloader, build script, or runtime code loader. The checker command is exactly:

```text
lean -t0 -j1 -M512 -T200000 Projected.lean
```

The gate requires the exact environment, an empty `LEAN_PATH`, a fixed source grammar and one checker thread. It enforces a 60-second wall-clock timeout and bounded output; exhaustion/cancellation is inconclusive, not mathematical rejection. Host paths, timing and other telemetry do not enter canonical semantic roots. The adapter/checker can be omitted without changing native mathematical identity or native verification. The admitted host is optional x86-64 Linux; there is no Wasm or `no_std` external-checker portability claim.

The built `fsym-formal` `project` example emits a checked JSON envelope for its fixed fixture; `project --check JSON_PATH` independently rechecks a supplied envelope against that fixture's native claim root. It is a gate interface, not an arbitrary-claim CLI or external-verdict authority. The gate is invoked with an absolute installed Lean path, the built example path and a fresh output directory:

```text
python3 tools/check_formal_projection.py --lean <absolute lean> --projector <built example> --artifacts <fresh-output-dir> --expected-environment 88d1bfed5e2ba13e7dd28043c70a303f1ea1b301220023fe6b9a1a5d14368f9f
```

Use a fresh directory under `artifacts/audit/ws06_formal_projection/` (a timestamp subdirectory is permitted). Retain the installed environment manifest, projection envelope/source, checker output, rejection fixtures and gate results. Required evidence includes native-first admission, independent receipt/mapping checks, exact field coverage, semantic/assumption/domain mutations, statement/proof mismatch and foreign rejection, deterministic replay, cancellation/resource bounds and a real no-mock external-checker run. Generic registry parsing does not establish these properties. This document specifies requirements and interfaces; it does **not** assert that executable gates, independent review, dependency/advisory review or certification have passed.

## 11. Evidence classes

A publication can carry any combination of:

- `native_verified`;
- `native_dual_checked`;
- `formal_projection_checked`;
- `foreign_checker_checked`;
- `bounded_model_checked`;
- `differentially_observed`.

The strongest available evidence does not erase the others’ exact scope.

## 12. Portability

Formal proof artifacts are optional attachments to FMAP bundles unless a publication profile explicitly requires them. Their absence cannot invalidate a verifier-complete native capsule.

Target-specific theorem-prover build products are target-bound. Canonical statements and proof terms may be portable if the profile certifies that property.

## 13. Conformance gates

Each formalized claim family requires:

- positive corpus;
- assumption and domain mutation corpus;
- statement/proof mismatch corpus;
- projection round-trip checks;
- foreign kernel rejection fixtures;
- exact environment pinning;
- no-mock end-to-end check;
- resource-limit tests;
- proof that unsupported fields fail closed;
- documentation wording check.

## 14. Non-goals

- replacing native certificate verifiers with Lean;
- formally verifying every heuristic strategy before implementation;
- treating formal export as a serialization format for arbitrary Python objects;
- importing theorem-prover runtime state into stable mathematical identity;
- claiming proof-system independence from one successful adapter.
