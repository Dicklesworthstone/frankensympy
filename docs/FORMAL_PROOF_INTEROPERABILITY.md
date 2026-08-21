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
