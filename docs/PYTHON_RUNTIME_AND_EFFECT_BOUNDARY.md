# Python runtime and effect boundary

**Status:** normative architecture contract  
**Scope:** Python object semantics, callback effects, exactly-once execution, speculation, interpreter profiles, free-threading, subinterpreters, garbage collection, identity, exceptions, and native-core isolation

## 1. Principle

The Python compatibility shell is an effectful host adapter around an immutable mathematical core. Python code is not assumed pure, deterministic, cooperative, thread-safe, or serializable.

A planner may speculate only over work proven closed and pure. Unknown or effectful Python behavior executes at most once on the selected route.

## 2. Effect classes

Every Python-visible operation or callback edge is classified before portfolio execution.

### `ClosedPure`

The complete implementation closure is native, immutable, deterministic, and contains no Python callback, host I/O, time, entropy, mutable global, or opaque extension access. It may be duplicated, hedged, reordered where policy allows, cached, and remotely executed.

### `DeclaredPureOpaque`

A Python/custom callable declares purity under a profile, but the implementation is opaque. It may be memoized or replayed only under an explicit opt-in contract and audit policy. It is not freely duplicated by default and cannot participate in portable certificate verification.

### `Effectful`

Known effects include mutation, I/O, warnings, printing, randomness, time, environment access, global cache changes, import side effects, counters, or externally visible call order. Execute exactly once after route selection.

### `UnknownEffect`

No reliable classification. Treat as effectful and non-replayable.

A classification is bound to callable identity, type/profile, code/version fingerprint where available, and interpreter generation. It is invalidated by monkey-patching or extension-world changes.

## 3. Prepare/commit execution

Effect-aware routes split into:

```text
prepare (closed pure, repeatable)
  -> choose one route
  -> commit Python effect exactly once
  -> convert and validate result
  -> optional native verification
  -> publish
```

Preparation may compute native templates, argument conversions, domain diagnostics, or post-callback verification plans. It must not invoke the callback or expose a result.

If cancellation occurs before commit, no callback runs. If interruption occurs while a callback runs, the outcome is `EffectOutcomeUnknown` unless the profile provides a stronger transactional boundary. The engine must not retry automatically.

## 4. Reentrancy

Python hooks may reenter FrankenSymPy. Reentrant calls receive:

- a child request identity;
- reduced or explicit nested budget;
- inherited compatibility and interpreter profile;
- cycle/depth guard;
- independent decision receipt;
- no access to a half-published parent result.

Reentrancy cannot bypass capability or verifier boundaries. A callback cannot mint `VerifiedClaim` or mutate an immutable universe root directly.

## 5. Python object identity

Strict profiles govern:

- wrapper interning;
- singleton and atom caching;
- `is` relationships where observable;
- equality and hash coherence;
- `args`, `func`, assumptions, and class identity;
- subclass and metaclass behavior;
- module and qualified names;
- descriptor binding;
- `__slots__`, optional `__dict__`, and weak references;
- copy and deepcopy;
- pickling/reduction;
- mutation visibility for compatibility wrappers.

Native content IDs are not Python object identities. Two wrappers may represent the same immutable term while being distinct Python objects if the active profile requires that behavior.

## 6. Ownership and garbage collection

The bridge must handle:

- Python↔Rust reference cycles;
- cyclic GC traverse/clear behavior;
- weakref callbacks;
- finalizers and resurrection;
- interpreter shutdown;
- module teardown order;
- thread-local and interpreter-local caches;
- exception objects retaining tracebacks and frames.

No portable mathematical object owns a Python reference. Python wrappers own or reference stable native handles through the binding framework’s safe lifetime model.

Finalizers are never relied upon for mathematical publication or durability.

## 7. Interpreter profiles

Compatibility evidence is profile-specific across:

- CPython version and ABI;
- classic GIL versus free-threaded build;
- main interpreter versus subinterpreter;
- platform and architecture;
- hash seed and locale where observable;
- optional dependencies and installed SymPy oracle;
- warning filters and import state;
- extension modules/custom classes.

Shared native immutable objects may be process-global only when their representation contains no interpreter-owned reference and all Python wrappers remain interpreter-local.

## 8. Free-threaded CPython

The absence of the GIL does not make object semantics race-free. The shell uses explicit synchronization for:

- wrapper intern tables;
- module and class registries;
- callback classification caches;
- mutable compatibility state;
- exception/warning machinery;
- publication roots visible through Python.

Hot mathematical objects remain immutable. Lock ordering is registered and tested under asupersync lab interleavings where the boundary can be modeled.

## 9. Subinterpreters

Each subinterpreter has its own:

- module state;
- Python class objects;
- wrapper cache;
- callback and extension-world registry;
- warning/exception configuration;
- shutdown lifecycle.

Cross-interpreter transfer uses canonical native objects or FMAP bytes, never raw Python object references. Target wrappers are rebuilt in the destination interpreter.

## 10. Exceptions, warnings, and panics

Native code returns typed errors. The Python shell maps them according to the profile:

- exact exception class;
- message and arguments;
- warning category and stack level;
- chained cause/context;
- partial output behavior;
- traceback ownership.

Project code must not use panics as ordinary Python errors. Unexpected panics are caught at the outer safe boundary where supported, quarantine the route, and become an internal fault without publishing partial state.

## 11. Delegation to upstream SymPy

Delegation is allowed only in a named compatibility profile and inventory row. It records:

- exact upstream version and environment;
- operation and inputs;
- whether the result remains an upstream object or is converted;
- identity/alias implications;
- callback/effect classification;
- exception/warning behavior;
- evidence limitation.

Delegation never enters portable verifier closure and cannot support a claim that the operation is independently implemented in pure Rust.

## 12. Serialization

Portable claims/certificates contain canonical mathematical objects only. They reject:

- pickles;
- executable Python bytecode;
- arbitrary module/class references without registered semantic definitions;
- raw memory addresses;
- interpreter IDs;
- opaque callback closures.

Python-compatible pickle support is a shell feature with its own security and profile contract.

## 13. Cancellation

Native cooperative work obeys request/drain/finalize semantics. Arbitrary Python callbacks may not cooperate. Profiles distinguish:

- in-process cooperative callback;
- in-process non-cooperative callback;
- isolated helper process with kill/recovery semantics;
- forbidden callback for bounded/certified request.

No universal drain-latency claim covers arbitrary Python code.

## 14. Conformance gates

- callback count and order fixtures;
- mutation and global-state fixtures;
- nested/reentrant calls;
- exception/warning/traceback parity;
- object identity and wrapper-cache tests;
- weakref, cycle, finalizer, and resurrection tests;
- classic, free-threaded, and subinterpreter profiles;
- interpreter shutdown and reload;
- hash-seed/import-order perturbation;
- no duplicate effect under portfolio hedging;
- cancellation before, during, and after callback commit;
- no Python dependency in portable verifier closure.

## 15. Prohibited behavior

- speculative duplicate unknown callbacks;
- retrying an interrupted effect automatically;
- treating `__hash__`, `__eq__`, `_eval_*`, printers, or assumptions hooks as pure without classification;
- storing Python references in durable native objects;
- cross-interpreter raw-object sharing;
- using object addresses in stable IDs;
- hiding upstream delegation;
- accepting pickle bytes as mathematical proof;
- promising prompt cancellation of arbitrary Python.
