# FrankenSymPy object model and intermediate representations

**Status:** normative architecture contract  
**Scope:** Python compatibility shell, native term kernel, surface/semantic/provenance separation, stable identities, lowering/lifting, mutable boundaries

## 1. The central constraint

SymPy is not only a collection of algorithms over expression trees. Its object model is a Python-extensible protocol. A drop-in implementation must preserve dynamic classes, metaclasses, constructor behavior, user hooks, class-sensitive equality and hashing, unevaluated forms, assumptions, printers, pickles, and mutable compatibility objects.

At the same time, retaining every built-in expression as a conventional Python object graph forfeits much of the performance, memory locality, parallelism, deterministic identity, and proof instrumentation that justify FrankenSymPy.

The architecture resolves that tension with two execution lanes and three representations.

## 2. The two execution lanes

### 2.1 Compatibility shell lane

The shell lane is a Python object system matching the selected compatibility profile. It owns observable Python behavior:

- module, class, metaclass, MRO, descriptor, and singleton identity;
- `__new__`, constructor evaluation, `args`, `func`, `_hashable_content`, equality, hash, compare, and sort keys;
- dynamic undefined-function classes and arbitrary user subclasses;
- `_sympy_`, external converters, constructor postprocessors, and `_eval_*` hooks;
- warnings, exceptions, signatures, deprecations, and exact import paths;
- printers, pickles, copy/deepcopy, weak references, and mutable objects;
- explicit and contextual `evaluate` policy;
- compatibility-visible caches and global/thread-local state.

Built-in shell objects may carry a compact native handle. The handle is an implementation detail, not the Python identity of the object.

### 2.2 Native semantic lane

The native lane operates on immutable, typed, hash-consed semantic terms. It owns:

- exact domains and coercion;
- canonical algebraic representations;
- rewrite search and proof extraction;
- polynomial, matrix, calculus, solver, set, logic, tensor, and code-generation algorithms;
- deterministic stable IDs;
- typed budgets, cancellation, scheduling, replay, and checkpoints;
- proof/certificate verification;
- native Rust APIs independent of Python.

Only expressions that satisfy a lowering contract enter this lane. Unknown Python behavior is never guessed away.

### 2.3 Boundary rule

The shell may ask the kernel for a candidate result. The kernel may not directly mutate or fabricate arbitrary shell objects. Results return through a lifting operation that reconstitutes profile-correct Python objects, runs required shell hooks, and records any lossy or opaque boundary.

## 3. The three representations

### 3.1 Surface Object Graph (`SOG`)

The Surface Object Graph records what the user actually constructed and what Python can observe.

A surface node includes, as applicable:

```text
SurfaceNode
├── surface_id                     # session/workspace identity, not mathematical identity
├── profile_id
├── python_class_descriptor
│   ├── module
│   ├── qualname
│   ├── metaclass
│   ├── MRO fingerprint
│   └── dynamic-class identity token
├── ordered arguments
├── constructor/evaluation policy
├── instance attributes and slots relevant to behavior
├── assumptions seed/state reference
├── mutability and alias class
├── native handle, if eligible
├── opaque callback capabilities
└── reconstruction/pickle descriptor
```

Properties:

- Argument order and multiplicity are preserved.
- Held forms are not normalized merely because an equivalent canonical term exists.
- Distinct dynamic classes remain distinct even if names and arguments match.
- Mutable objects are represented through explicit mutable cells and snapshot views, never hash-consed as immutable terms.
- Surface IDs are unique within a workspace/replay universe. They are not used for mathematical equality.

Examples of distinct surface objects that may share one semantic meaning:

```python
Add(x, x, evaluate=False)
2*x
Mul(1, x, evaluate=False)
x
Pow(x, 2, evaluate=False)
x**2
```

The profile determines which observations distinguish them.

### 3.2 Semantic Term DAG (`STD`)

The Semantic Term DAG is the kernel's immutable mathematical representation. It is typed by domain and assumptions context.

A semantic node contains:

```text
SemanticTerm
├── term_id
├── operator_id
├── ordered child term_ids
├── payload                        # integer limbs, symbol key, interval endpoints, etc.
├── domain_id
├── sort/kind
├── binding descriptor             # bound variables, de Bruijn or equivalent canonical form
├── declared algebraic properties
└── schema/rule-universe version
```

Properties:

- Structurally identical terms in the same identity universe are interned.
- Canonicalization is operator- and domain-specific, never a universal “sort all children” rule.
- Commutative and noncommutative operators have distinct representations and invariants.
- Bound variables use alpha-invariant internal identity while lifting preserves profile-visible names and scopes.
- Domain is explicit: `ZZ[x]`, `QQ(x)`, an algebraic extension, a matrix ring, Boolean algebra, distribution domain, and generic expression domain are not interchangeable tags.
- Semantic equality may be structural, normalized-domain equality, or a proved relation; those notions are kept separate.

The STD is optimized for algorithms. It is not exposed as a substitute for arbitrary Python subclasses.

### 3.3 Derivation Evidence Graph (`DEG`)

The Derivation Evidence Graph records how claims were produced and why they are acceptable.

A derivation edge includes:

```text
DerivationEdge
├── derivation_id
├── claim
│   ├── relation kind              # equal, implies, subset, enclosure, root set, etc.
│   ├── source term_ids
│   └── target term_ids
├── assumptions_context_id
├── rule/algorithm/verifier versions
├── evidence payload or certificate reference
├── evidence class
├── side-condition obligations
├── budget and execution receipt
├── parent derivations
└── verification result
```

The DEG is append-only once a verified edge is published. A rejected candidate may be retained as diagnostic evidence but cannot appear in a proof path as a verified edge.

The DEG supports:

- expandable explanations;
- proof replay and independent verification;
- branch-per-agent workspaces;
- counterexample attachment;
- dependency invalidation when a rule or profile changes;
- extraction of compact proof certificates from large search histories;
- optional indexing in FrankenGraphDB.

A large search trace is not automatically a proof. Only verifier-accepted edges belong to the trusted derivation subgraph.

## 4. Why three graphs are necessary

A single representation cannot simultaneously preserve all required properties:

| Requirement | Surface graph | Semantic DAG | Derivation graph |
|---|---:|---:|---:|
| Exact Python class behavior | authoritative | not represented | records boundary decisions |
| Held argument order/multiplicity | authoritative | may normalize | records transformation |
| Fast canonical algebra | unsuitable | authoritative | records proof |
| Stable mathematical content ID | not always possible | authoritative | references IDs |
| Mutable compatibility objects | authoritative | immutable snapshots only | records snapshot/conversion |
| Proof/explanation | insufficient | stores terms only | authoritative |
| Aggressive sharing | constrained by Python identity | native | edges share term IDs |
| Agent semantic merge | surface-aware | term-aware | verifier-governed |

Collapsing the SOG and STD destroys compatibility. Collapsing the STD and DEG makes provenance indistinguishable from truth. Collapsing the SOG and DEG turns user-visible form into an accidental byproduct of proof search.

## 5. Shell object classes

The compatibility shell uses four implementation categories.

### 5.1 Native-backed built-ins

Examples: profiled numeric atoms, `Symbol`, `Dummy`, `Add`, `Mul`, `Pow`, many built-in functions, immutable matrices, and canonical sets.

A native-backed shell object stores or can derive:

- profile-correct Python fields;
- a `TermHandle` into the native arena;
- a surface descriptor when its shell form differs from the semantic term;
- assumptions/cache state required by the profile.

Fast paths may operate entirely on native handles when no Python hook can observe or override the operation.

### 5.2 Shell-only built-ins

Some compatibility objects are initially safer or simpler to implement in Python: highly dynamic utilities, mutable containers, specialized printers, code-generation settings, interactive helpers, and surfaces with extensive Python callback behavior.

Shell-only does not mean permanent or slow. A component can acquire a native lowering contract later without changing its Python class.

### 5.3 Opaque user objects

An arbitrary subclass or object with `_sympy_` is accepted according to profile behavior. It receives an `OpaqueNodeDescriptor`:

```text
OpaqueNodeDescriptor
├── class identity token
├── stable-within-process object token
├── reconstructed ordered children, when safely observable
├── declared algebraic capabilities
├── callback capability set
├── purity/determinism declaration status
├── serialization capability
└── lowering plugin identity, if any
```

Default treatment is conservative:

- the node is composable in shell expressions;
- traversal and reconstruction preserve it;
- the kernel treats it as an uninterpreted operator or atom only when that is semantically safe;
- no callback is assumed pure, deterministic, thread-safe, terminating, or exception-free;
- native parallelism cannot invoke the callback without crossing a supervised Python-hook boundary.

### 5.4 Mutable cells

Mutable matrices and other stateful objects use explicit shell-owned cells. A native algorithm receives an immutable snapshot with:

- source cell ID;
- mutation generation;
- shape/type metadata;
- snapshot digest;
- copy/alias policy.

Lifting a result never writes back unless the profiled API is explicitly mutating. Optimistic mutation requires generation validation; on conflict the operation retries or raises according to profile behavior.

## 6. Python hook boundary

Every Python hook invocation is classified:

| Class | Examples | Kernel treatment |
|---|---|---|
| Construction | `__new__`, classmethod `eval`, postprocessors | shell-owned; may prevent/later invalidate lowering |
| Semantic query | `_eval_is_*`, `_eval_derivative`, `_eval_rewrite` | supervised callback; result validated and provenance-marked |
| Conversion | `_sympy_`, external converter | shell-owned, profile-ordered |
| Printing | `_sympystr`, `_latex`, custom printer method | shell-owned unless a native printer contract is registered |
| Numeric implementation | `_eval_evalf`, `_imp_` | may produce candidate numeric evidence; never assumed exact |
| Substitution/reconstruction | `_eval_subs`, `func(*args)` | shell-owned compatibility behavior |

A hook call receipt records:

- callable identity and class fingerprint;
- input surface IDs and snapshots;
- profile and assumptions context;
- budget/cancellation policy;
- output or exception observation;
- determinism/purity declaration and whether it was trusted;
- cacheability decision.

Hooks execute under a Python-hook budget. Cancellation is cooperative at the boundary; no universal interrupt guarantee is claimed for arbitrary Python code.

## 7. Lowering

### 7.1 Lowering result

Lowering returns one of:

```text
Lowered(term_handle, lowering_receipt)
PartiallyLowered(term_handle_with_opaque_nodes, lowering_receipt)
ShellOnly(reason, diagnostic)
Refused(reason, diagnostic)
Cancelled(receipt)
ResourceExhausted(receipt)
```

“Partially lowered” is safe only when each opaque node's semantic role is explicit. An opaque noncommutative operator cannot be silently treated as a commutative symbol.

### 7.2 Lowering algorithm

For each surface node, lowering:

1. freezes the profile, evaluation policy, assumptions context, and rule universe;
2. checks mutability and snapshots mutable inputs;
3. resolves the exact Python class and any registered lowering plugin;
4. determines whether hooks, subclass overrides, or instance state make a built-in fast path unsafe;
5. lowers children in profile-preserving order;
6. constructs a typed semantic operator or an explicit opaque operator;
7. validates domain/coercion invariants;
8. records the relationship between surface and semantic IDs;
9. emits a receipt with every information loss, normalization, and opaque boundary.

### 7.3 Lowering receipt

```text
LoweringReceipt
├── receipt_id
├── profile_id
├── source surface_ids
├── target term_ids
├── evaluation policy snapshot
├── assumptions_context_id
├── class/lowering-plugin fingerprints
├── domain/coercion decisions
├── normalizations applied
├── opaque nodes and capability limits
├── mutable snapshots
├── warnings/exceptions observed
├── schema versions
└── replay fingerprint
```

A lowering receipt does not prove semantic equivalence by itself. Built-in lowering rules must be covered by kernel proofs, certificate checks, or profile conformance evidence appropriate to their claim.

## 8. Lifting

Lifting maps semantic results back into Python-compatible objects.

The lifter chooses among:

- reuse of an existing shell object whose surface form remains valid;
- construction of a profile-canonical built-in shell object;
- reconstruction of a held form requested by the calling contract;
- wrapping a native result in an explicit native-only object;
- refusing a lossless lift when the target profile cannot represent the result.

Lifting must account for:

- exact class and singleton identity;
- evaluation policy and constructor side effects;
- argument order and profile canonicalization;
- assumptions and branch choices;
- custom subclass preservation;
- printer and pickle behavior;
- warnings/exceptions that upstream construction would emit.

A native result with richer evidence may be returned through a native envelope while its `.value` is the profile-compatible shell object.

## 9. Stable identities

### 9.1 Identity types

FrankenSymPy uses distinct non-interchangeable ID newtypes:

- `SurfaceId`: one shell object/surface node in a workspace universe;
- `TermId`: canonical semantic term content;
- `DomainId`: exact algebraic/numeric domain definition;
- `AssumptionsContextId`: immutable fact/context snapshot;
- `RuleRegistryId`: ordered rewrite/rule universe;
- `AlgorithmRegistryId`: planner/algorithm universe;
- `VerifierRegistryId`: accepted verifier set and versions;
- `DerivationId`: one derivation edge;
- `ReceiptId`: one execution/boundary receipt;
- `CheckpointId`: resumable state artifact;
- `BundleId`: portable replay/proof/work bundle.

The Rust type system must prevent accidental substitution among them.

### 9.2 `TermId` preimage

A `TermId` is computed from a canonical binary encoding of:

```text
schema domain
operator identity
operator schema version
ordered child TermIds
canonical payload
binding representation
sort/kind
identity-relevant domain data
```

It excludes:

- memory address;
- process-local interner index;
- Python hash;
- cache state;
- planner statistics;
- provenance;
- wall-clock time;
- non-semantic surface spelling.

The digest algorithm and canonical encoding are registry-versioned. Digest collisions are treated as catastrophic integrity failures: equality always confirms canonical payload, not digest alone, at trust boundaries.

### 9.3 Python hash separation

Python `__hash__` follows the compatibility profile, including process hash seeding where observable. It may cache a profile-compatible integer. `TermId` remains stable across processes. Neither value is substituted for the other.

## 10. Bindings and alpha-equivalence

Bound-variable objects such as integrals, sums, lambdas, derivatives, indexed constructs, and tensor expressions need two identities:

- surface names/classes/assumptions for compatibility;
- alpha-invariant kernel binding identity for semantics.

The kernel uses a capture-avoiding canonical binding representation, such as typed de Bruijn indices plus explicit binder metadata. Lowering records the mapping from surface symbols to binders. Lifting chooses names that preserve the original surface when possible and follows profile canonical-variable behavior otherwise.

Alpha-equivalence, structural equality, dummy equality, and ordinary Python equality remain separate operations.

## 11. Canonicalization policy

Canonicalization is local to an operator/domain contract.

Examples:

- integer and rational payloads normalize sign and gcd;
- commutative polynomial monomials use a declared monomial order;
- noncommutative products preserve order;
- matrix products preserve dimension-typed order;
- sets may normalize only when membership semantics and evaluation policy justify it;
- branch-sensitive complex functions retain branch metadata;
- held shell forms may map to a canonical STD term only with a surface-to-semantic receipt.

There is no universal “simplest expression.” Simplification is a goal-directed search with an explicit cost vector and proof obligations.

## 12. Term arenas and interning

The native kernel uses sharded arenas with deterministic content identity and process-local compact handles.

Requirements:

- handles are generation-checked and cannot outlive their arena;
- concurrent interning is linearizable for identical canonical preimages;
- failed/cancelled construction cannot publish a partial node;
- arena compaction preserves `TermId` and invalidates or remaps handles safely;
- deterministic replay does not depend on insertion races or shard assignment;
- memory accounting charges both unique nodes and transient construction buffers;
- weak interning/eviction is allowed only for rebuildable nodes with no live handle.

The arena is not a global immortal cache. Workspaces and long-lived services need bounded retention and explicit cache policy.

## 13. Caches

Caches are derived state. A cache key must include every semantic dependency, including:

- term IDs;
- domain;
- assumptions context;
- compatibility profile when surface behavior matters;
- evaluation policy;
- rule, algorithm, and verifier registry versions;
- precision/tolerance/branch policy;
- relevant budget class when partial outcomes can differ.

Cache entries carry evidence class and verification status. A heuristic candidate cannot be read as a proved result. Cancellation, timeout, and resource exhaustion are generally not permanent negative cache entries; policy may retain bounded diagnostic hints.

Unverified speculative work cannot publish to shared caches.

## 14. Representation-specific APIs

The Rust API makes representation transitions explicit:

```rust
let lowered = shell.lower(&cx, &profile, &ctx, &policy)?;
let result = kernel.transform(&cx, lowered.term(), request)?;
let verified = verifier.verify(&cx, result.candidate(), result.certificate())?;
let lifted = shell.lift(&cx, verified.value(), &profile, lift_policy)?;
```

Convenience APIs may compose these steps, but receipts remain retrievable. Internal APIs must not accept a generic “expression” parameter when the required representation matters.

## 15. Serialization

Each graph has a different serialization contract:

- SOG serialization is profile- and Python-class-sensitive and may be impossible for local dynamic classes;
- STD serialization is canonical, language-neutral, content-addressed, and independent of arena handles;
- DEG serialization references canonical terms and carries verifier/evidence schemas;
- replay bundles include all non-rebuildable registry/profile/context dependencies;
- persistent caches may omit rebuildable data but never authoritative proof dependencies.

Unknown schema versions fail closed. RaptorQ repair can restore serialized bytes, after which digests and schema verification establish integrity; mathematical verifiers establish evidence.

## 16. Object-model conformance matrix

The first vertical slice cannot pass until it covers at least:

| Surface | Required observations |
|---|---|
| `Basic` subclass | construction, slots, args/func, hash/equality, compare/sort, copy, pickle |
| `Atom` subclass | atomic traversal, free symbols, class identity, printer hooks |
| `Symbol`/`Dummy` | assumptions, uniqueness, dummy equality, canonical variables |
| `Add`/`Mul`/`Pow` | evaluated and held forms, order, postprocessors, noncommutativity |
| `Function` | dynamic class creation, metaclass, nargs, classmethod `eval`, applied undefined functions |
| custom `_eval_*` | derivative, assumptions, rewrite, evalf, substitution hooks |
| external conversion | `_sympy_`, converter ordering, errors |
| mutable matrix | alias/mutation, immutable snapshot, conversion, pickle |
| printers | default and custom methods, settings, exact class dispatch |
| replacement | `subs`, `xreplace`, traversal, reconstruction through `func` |

## 17. Forbidden shortcuts

The following violate the object-model contract:

- exposing every expression as one extension type and claiming drop-in compatibility;
- using strings or printed forms as the semantic IR;
- reconstructing an arbitrary subclass as a built-in with the same name;
- treating all unknown nodes as commutative symbols;
- invoking Python callbacks from unsupervised native worker threads;
- assuming custom hooks are pure, deterministic, or terminating;
- canonicalizing away held surface structure before compatibility observations;
- mixing mutable cells into immutable interning;
- using process-local handles as durable IDs;
- using a digest match without canonical-payload validation at trust boundaries;
- allowing unverified candidates into shared caches;
- losing the assumptions/profile/rule universe from a cache key or receipt;
- confusing alpha-equivalence with Python equality;
- lifting through constructors without accounting for their evaluation and side effects.

## 18. Acceptance criteria

This architecture is considered validated only when the first implementation campaign demonstrates, end to end:

1. profile-correct Python shell classes for the first core set;
2. arbitrary user subclasses and opaque functions surviving traversal, substitution, printing, pickle where upstream supports it, and mixed native operations;
3. held and evaluated forms sharing semantic work without losing surface behavior;
4. deterministic lowering/lifting receipts;
5. native term interning and stable IDs across processes;
6. independently verified native transformations;
7. cancellation without orphan work or unverified cache publication;
8. deterministic replay under the lab runtime;
9. exact differential conformance against the pinned upstream profile;
10. mutation tests that fail when the shell/kernel boundary is deliberately weakened.

Until those gates pass, adding thousands of API stubs would increase apparent surface area while leaving the central compatibility theorem unproved.