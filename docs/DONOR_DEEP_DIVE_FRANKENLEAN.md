# Donor deep dive: FrankenLean

**Status:** normative source audit and architecture input  
**Pinned source:** `Dicklesworthstone/franken_lean@7df7a3170045882f3ab1f13bfee72338e524d174`  
**Audit date:** 2026-08-20  
**Scope:** small trusted kernels, certificate and cartridge formats, Merkle environments, foreign checking, deterministic elaboration, proof-state snapshots, causal proof graphs, and evidence governance

## 1. Executive conclusion

FrankenLean is not valuable to FrankenSymPy merely because both projects manipulate mathematics. Its strongest lesson is architectural:

> Generation, elaboration, optimization, transport, caching, and consensus may be extremely sophisticated; admission must remain the responsibility of a small checker over a complete immutable environment.

FrankenSymPy should therefore provide two complementary proof lanes:

1. native domain certificates checked by small Rust verifier crates;
2. optional formal projection into a theorem-prover kernel for selected claim families.

The formal lane is additional evidence and interoperability. It must not become a mandatory dependency of ordinary factorization, root isolation, calculus, or linear-algebra verification.

## 2. Source surfaces examined

- `README.md`: Oracle-Only Law, dual-engine kernel target, declaration-granular Merkle environment, deterministic parallelism, causal proof graph, and evidence-native claims;
- `CERTIFICATE_FORMAT.md`: authority separation, canonical term DAG, environment binding, bounded decode, unknown-extension policy, and foreign-checker projection;
- `CARTRIDGE_FORMAT.md`: thin/partial/sealed/complete artifact populations, content-addressed closure, streaming/staging discipline, optional warm caches, and transport-versus-truth separation;
- `ABI_CONTRACT.md`: exact compatibility surfaces and machine-derived layout discipline;
- comprehensive plan and current conformance work around declaration references and rejection classes.

## 3. Adopt: checker authority is singular

A decoded certificate is data, not a verified claim. A signature, producer identity, cache hit, replay match, majority vote, or compatible transport does not convert it into authority.

For each native certificate family, one reference verifier API returns a typed verified token or a typed refusal/inconclusive result. Optimized checkers may exist, but disagreement with the reference checker is release-blocking and quarantines the optimized route.

The generator cannot construct a `VerifiedClaim` directly. Publication APIs accept only tokens that the verifier module can mint.

## 4. Adopt: complete immutable environment binding

A mathematical claim is interpreted under an environment that includes:

- operator and function definitions;
- domains and coercions;
- assumptions context;
- branch and analytic continuation policy;
- rewrite-rule registry;
- certificate schema and verifier version;
- compatibility/profile roots where observable behavior matters;
- extension-world and custom-class declarations where permitted.

Certificates bind to the exact roots they require. A checker refuses stale, missing, contradictory, or unknown-critical environment objects.

This is stronger than storing a version string. Two environments with the same human version label but different registry roots are different verification universes.

## 5. Adopt: acyclic proof encoding by construction

Where possible, proof and term nodes are topologically encoded: every child or dependency reference points to an earlier node or an immutable external object. This provides:

- bounded iterative validation;
- no recursive cycle search in the common format;
- deterministic streaming decode;
- simple node-count budgets;
- stable canonicalization;
- efficient deduplication and chunking.

Schemas that require cyclic structures use an explicit graph object and a separate cycle/coinduction contract rather than smuggling backreferences into a DAG format.

## 6. Adopt: unknown critical fields fail closed

Extensions carry an explicit critical/advisory bit.

- unknown advisory extensions round-trip without affecting verification;
- unknown critical extensions refuse verification;
- known critical extensions require a registered checker and projection rule;
- no implementation may silently drop a semantic field to make an export succeed.

## 7. Adopt: resource exhaustion is not falsity

Malformed input, unsupported schema, failed verification, cancellation, and budget exhaustion are distinct outcomes.

`ResourceExhausted`, `Cancelled`, and `MissingClosure` are inconclusive. They cannot be encoded as a negative mathematical result and cannot populate a cache of disproofs.

## 8. Adapt: certificate cartridges for mathematical closures

The FrankenLean cartridge model maps well to FrankenSymPy bundles:

- one logical manifest identity;
- thin, partial, sealed, and complete physical populations;
- independently addressed chunks and objects;
- exact object kind in identity;
- required versus optional objects;
- portable, epoch-bound, and target-bound objects;
- derived random-access indexes;
- failure-atomic staging;
- bounded streaming decode;
- optional untrusted acceleration caches.

FrankenSymPy’s FMAP protocol owns the general artifact format. The Lean audit reinforces several non-negotiable details:

- adding optional provenance or search traces must not change the verifier-complete claim identity;
- present chunks are validated before staging state mutates;
- complete transport closure does not imply valid mathematics;
- cache artifacts can only provide replay hints;
- an incomplete cartridge cannot produce a negative claim merely because a proof object is absent.

## 9. Adapt: formal projection lane

A selected native claim may be projected to Lean through a registered profile:

```text
Native Claim + Native Certificate + Environment Closure
    │
    ├── native portable verifier ──► VerifiedNativeClaim
    │
    └── formal projector
            ├── theorem statement
            ├── definitions and assumptions
            ├── proof term or tactic-independent certificate theorem
            └── projection receipt
                    │
                    ▼
              foreign kernel check
```

Projection is allowed only when every semantic field has a reviewed mapping. Otherwise the projector returns `UnrepresentableSemanticField`.

A formal checker result never repairs a failed native projection. It verifies the formal statement actually emitted. The projection receipt binds the formal statement root back to the native claim root.

## 10. Initial formalization targets

High-value, tractable first targets:

- integer/rational polynomial factorization identity and domain conditions;
- polynomial gcd and Bézout certificates;
- exact linear solve and determinant identities over fields;
- row-reduction equivalence;
- finite rewrite traces composed from registered lemmas;
- rational-function simplification under nonzero-denominator assumptions;
- elementary derivative certificates for a closed operator inventory;
- interval/refinement claims with exact rational endpoint arithmetic.

Defer until the semantic contract is mature:

- general branch-cut-sensitive complex analysis;
- arbitrary user-defined Python functions;
- heuristic special-function transformations;
- broad transcendental identity discovery;
- claims whose native object model has no stable formal projection.

## 11. Proof search and proof checking remain separate

Proof search may use:

- e-graphs;
- rewrite portfolios;
- learned retrieval;
- remote agents;
- stochastic search;
- theorem-prover tactics;
- graph mining;
- cached normal forms;
- speculative parallel elaboration.

The final accepted object is checked independently. Search traces are research/provenance artifacts, not required proof closure unless the certificate family explicitly defines them as proof steps.

## 12. Proof-state snapshots and agent branches

FrankenLean’s O(1)-style proof-state fork concept suggests immutable symbolic search states:

- context root;
- goal set root;
- metavariable/placeholder map;
- local lemma closure;
- resource budget;
- seed lineage;
- parent snapshot and action receipt.

Agents fork states without copying the entire term/proof universe. Merging states is semantic and verified, never textual. A snapshot may be durable and transferable without becoming an accepted theorem.

## 13. Causal proof graph

Every accepted claim should be explainable through typed provenance edges:

- which assumptions were used;
- which rules and algorithms generated the certificate;
- which verifier version checked it;
- which imported claims were trusted;
- which compatibility/profile roots affected representation;
- which optional formal checker agreed;
- which publication event made it authoritative.

The causal graph supports impact analysis and trust queries. It does not replace checking.

## 14. Oracle-only law for compatibility systems

Upstream SymPy and Lean are differential oracles, fixture generators, and semantic references. They are not hidden runtime fallbacks in the sovereign native path.

A compatibility profile may deliberately delegate an unsupported Python-facing operation to installed SymPy, but such delegation is explicit, profile-scoped, observable, and excluded from claims of an independent pure-Rust implementation. Portable verifier crates never delegate.

## 15. Dual and foreign checking

For high-value certificate families, FrankenSymPy should support:

- a small reference Rust checker;
- one optimized Rust checker sharing no algorithmic fast path with the generator;
- optional Lean projection and kernel check;
- mutation and malformed-certificate corpus;
- disagreement quarantine.

Multiple checkers are defense in depth. A vote cannot override the authoritative checker contract. Disagreement is an engineering fault requiring investigation.

## 16. Evidence wording

Public wording must distinguish:

- “native certificate verified”;
- “formal projection checked by Lean”;
- “verified by two independent native implementations”;
- “differentially matched upstream on corpus X”;
- “bounded model checked”;
- “statistically calibrated”;
- “benchmark measured.”

These phrases are not interchangeable.

## 17. Explicit rejections

FrankenSymPy rejects:

- Lean as a mandatory dependency of portable native verification;
- tactic success as theorem authority without kernel checking;
- dropping assumptions or branch policies during projection;
- certificate decode as verification;
- cache state as admission evidence;
- majority vote between checkers;
- treating cancellation or resource exhaustion as disproof;
- proof objects containing executable Python/pickle payloads;
- formally checking a weakened statement while presenting it as the original claim;
- importing the entire FrankenLean implementation into every verifier crate.

## 18. Implementation order

1. freeze native claim/certificate schemas;
2. implement reference native verifiers;
3. define the projection receipt and formal-profile registry;
4. select one factorization and one linear-algebra theorem family;
5. generate Lean statements and proof terms from verified native objects;
6. run an independent foreign checker lane;
7. add round-trip and mutation fixtures;
8. expose causal trust queries;
9. expand only when projection completeness is proven per family.
