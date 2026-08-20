# FrankenSymPy compatibility contract

**Status:** normative architecture contract  
**Initial certification target:** SymPy 1.14.0 at `16fa855354eb7bcabd3fe10993841e03b1382692`  
**Moving-head observation pin:** SymPy development head at `81b519fabdbbc8e82db154dd271100ec7fb7ef32`

## 1. Objective

FrankenSymPy's compatibility goal is not “similar syntax,” “most common functions,” or “the upstream tests mostly pass.” The target is an installable Python distribution whose observable behavior is interchangeable with a named SymPy profile for applications within that profile's declared platform and optional-feature envelope.

Compatibility is a versioned empirical contract. It is never inferred from API-name coverage, source similarity, mathematical plausibility, or a manually curated demo.

## 2. Two products, two promises

The design separates two distributions so advanced native behavior can evolve without weakening the meaning of “drop-in.”

### 2.1 `frankensympy`

The coexistable package exposes:

- the native Rust API;
- Python bindings under the `frankensympy` namespace;
- proof/evidence envelopes;
- explicit budgets, cancellation, replay, and persistence controls;
- compatibility-profile inspection;
- preview implementations that may still carry discrepancy entries;
- adapters to FrankenNumPy, FrankenSciPy, FrankenSQLite, and FrankenGraphDB.

It may be useful before full drop-in certification. It must never imply that importing `frankensympy` is equivalent to importing the profiled `sympy` package.

### 2.2 `frankensympy-dropin`

The drop-in distribution owns the top-level `sympy` package. It intentionally conflicts with upstream SymPy in the same environment and is published only for a profile whose release gates are green on the exact release commit.

A certified wheel:

- contains no runtime dependency on upstream SymPy;
- contains no hidden import fallback to upstream SymPy;
- names its profile in package metadata and runtime introspection;
- ships the profile manifest, conformance report, discrepancy digest, build provenance, and gate results;
- fails closed when loaded on an unsupported Python ABI/platform combination unless an explicitly uncertified override is requested;
- preserves upstream module paths, classes, signatures, warnings, exceptions, printers, and serialization behavior within the profile.

A preview wheel may use the top-level namespace only if it is unmistakably marked uncertified at install and import time. Preview status is not a lower grade of certification.

## 3. Compatibility profile

A profile is immutable and content-addressed. It contains at least:

```text
CompatibilityProfile
├── profile_id
├── upstream
│   ├── release
│   ├── git_commit
│   ├── source_tree_digest
│   └── test_tree_digest
├── python
│   ├── implementation
│   ├── version_range
│   ├── abi_tags
│   ├── platform_tags
│   └── hash_seed_policy
├── optional_features
│   ├── dependency versions
│   ├── enabled modules
│   └── environment capabilities
├── public_surface
│   ├── modules and import aliases
│   ├── exports and `__all__`
│   ├── classes, metaclasses, MROs, slots, descriptors
│   ├── call signatures and defaults
│   ├── methods, properties, constants, singleton identities
│   └── deprecations and aliases
├── behavior
│   ├── construction/evaluation policies
│   ├── equality/hash/sort policies
│   ├── assumptions contexts
│   ├── traversal/reconstruction/substitution
│   ├── warnings/exceptions/messages
│   ├── printers and code generators
│   ├── pickle/copy protocols
│   └── mutable-object semantics
├── conformance
│   ├── upstream test inventory
│   ├── generated fixture grammar
│   ├── ecosystem corpus inventory
│   ├── exclusions with reasons
│   ├── comparator registry
│   └── discrepancy-ledger digest
└── engine
    ├── rule registry digest
    ├── algorithm registry digest
    ├── verifier registry digest
    ├── lowering schema version
    └── serialization schema versions
```

Changing any field creates a different profile. “Same SymPy version, newer rules” is a different certified artifact even when all differential tests still pass.

## 4. Initial profile policy

The first immutable profile is provisionally named:

```text
sympy-1.14.0-cpython
```

Its upstream source pin is:

```text
16fa855354eb7bcabd3fe10993841e03b1382692
```

The initial implementation campaign narrows Python ABI/platform combinations to keep the first matrix executable, but it does not narrow the eventual SymPy API target. Unsupported combinations are profile debt with explicit ownership and gates.

A separate, non-certifying drift lane continuously observes the pinned development head. Moving-head results can create forward-port work but cannot retroactively change the 1.14.0 profile.

## 5. Compatibility dimensions

### 5.1 Import and namespace behavior

The profile inventories and tests:

- every public and compatibility-relevant private module path;
- import side effects, lazy imports, aliases, and `from sympy import *` behavior;
- `__all__`, `__module__`, `__qualname__`, and repr-visible paths;
- package metadata and version attributes;
- optional-module behavior when dependencies are present or absent;
- circular-import-sensitive initialization order.

An object implemented correctly under the wrong module path is not drop-in compatible when pickling, introspection, dispatch, or ecosystem imports observe the difference.

### 5.2 Object identity and class behavior

The profile covers:

- class and metaclass identity;
- exact MRO and `isinstance`/`issubclass` results where public behavior depends on them;
- singleton identity for `S.Zero`, `S.One`, `pi`, `I`, infinities, booleans, and other registry objects;
- `__slots__`, weak-reference behavior, descriptors, class attributes, assumption flags, and dynamic subclass initialization;
- `args`, `func`, `_hashable_content`, copy, reconstruction, and pickle behavior;
- dynamic undefined-function classes and user-defined subclasses.

The compatibility shell may carry a native handle internally, but the handle cannot replace the Python class contract.

### 5.3 Construction and evaluation

The profile distinguishes:

- explicit `evaluate=True`, `evaluate=False`, and omitted `evaluate`;
- thread-local/global parameter contexts;
- canonical construction, raw construction, and reconstruction through `func(*args)`;
- constructor postprocessors and external converters;
- classmethod `eval`, `_eval_*` hooks, and custom `__new__` methods;
- assumptions-dependent evaluation;
- cache interactions and cache clearing;
- argument order and multiplicity in held forms.

Surface preservation is tested independently from semantic equivalence. `Add(x, x, evaluate=False)` is not interchangeable with `2*x` for every observable operation even though they are mathematically equal under ordinary assumptions.

### 5.4 Equality, hashing, ordering, and collections

Tests cover:

- Python `==` and `!=` return values and `NotImplemented` interactions;
- class-sensitive numeric and symbolic equality;
- process-local hash behavior under controlled `PYTHONHASHSEED` values;
- use as dictionary keys and set members;
- `compare`, `sort_key`, `ordered`, default sorting, and class ordering;
- dummy-symbol equivalence and alpha-renaming behavior;
- equality against externally convertible objects and objects with `_sympy_`.

FrankenSymPy stable content IDs are separate from Python hashes. A durable `TermId` must never leak into `__hash__` when that changes profile behavior.

### 5.5 Assumptions and logic

The profile tests old and new assumptions systems, including:

- `True`, `False`, and `None` outcomes;
- class defaults, instance facts, inferred facts, and contradictions;
- `ask`, predicates, local contexts, refinement, and handlers;
- user-defined `_eval_is_*` and predicate hooks;
- assumptions on symbols, functions, matrices, sets, and composite expressions;
- cache/context isolation across threads and tasks;
- exact exception or diagnostic behavior for inconsistent assumptions.

Unknown is a first-class outcome. It cannot be normalized to false, omitted, or guessed.

### 5.6 Mutation and non-`Basic` objects

The profile includes mutable matrices, arrays, containers, plotting structures, and other public objects that do not obey immutable-term assumptions. It records:

- mutation and alias behavior;
- conversion between mutable and immutable forms;
- copy/deepcopy behavior;
- indexing, slicing, assignment, shape changes, and exception boundaries;
- hashability or deliberate unhashability;
- interaction with `sympify` and printers.

Native hash-consing is never applied across a mutable compatibility boundary.

### 5.7 Warnings, exceptions, and messages

Compatibility includes:

- exception class and inheritance;
- warning class, count, filter interaction, and stack level;
- argument validation order;
- stable message fragments and exact messages where ecosystem code relies on them;
- partial effects before failure;
- traceback-visible call surfaces when part of a documented hook.

The comparator registry states whether a message is exact, normalized, fragment-based, or intentionally unconstrained for each fixture family.

### 5.8 Printing and serialization

The profile covers:

- `str`, `repr`, `sstr`, pretty Unicode/ASCII, LaTeX, MathML, code printers, and custom-printer hooks;
- line wrapping, ordering, symbol naming, settings, and global printer state;
- `pickle` protocols supported by upstream, cross-process round trips, and module/class reconstruction;
- copy/deepcopy and `__getnewargs__` behavior;
- `srepr` and parse/reconstruct workflows where supported;
- serialization failure behavior for local/dynamic classes.

Printer parity is not inferred from mathematical equivalence.

## 6. Comparator registry

Every conformance fixture names exactly one primary comparator and any secondary observations. Initial comparator families are:

| Comparator | Meaning |
|---|---|
| `exact_python` | type, value, identity-sensitive observations, and normalized metadata match exactly |
| `exact_structure` | class and recursively ordered `args` match |
| `exact_surface` | held/evaluated form, class, args, assumptions snapshot, and printer observations match |
| `exact_exception` | exception class and declared message policy match |
| `exact_warning_trace` | warning classes/count/order/stack policy match |
| `mathematical_under_context` | independently verified equivalence under an explicit assumptions context; never substitutes for structural parity when structure is observable |
| `set_semantics` | finite/infinite set result compared by a declared exact or membership-proof policy |
| `ordered_collection` | element observations and order match |
| `unordered_collection` | multiset observations match under a declared element comparator |
| `certified_enclosure` | exact target lies in a verified interval/ball with the declared precision contract |
| `numeric_profile` | upstream-compatible tolerance/NaN/signed-zero/branch policy for a named numeric surface |
| `nondeterminism_envelope` | result lies within an upstream-observed and profile-frozen envelope; use requires evidence that upstream is genuinely nondeterministic |

A fixture cannot silently switch comparators to pass. Comparator changes mutate the profile and require review of all newly admitted results.

## 7. Oracle isolation

Upstream SymPy is executed in a separate process and environment with a captured profile fingerprint. The conformance harness communicates through a versioned fixture/result protocol.

The oracle process:

- never shares Python objects with FrankenSymPy;
- never imports FrankenSymPy into the same interpreter for identity-sensitive tests unless the fixture explicitly studies coexistence;
- records dependency versions, locale, environment variables, hash seed, precision settings, and optional features;
- emits raw observations plus normalized comparator fields;
- is unavailable to certified production runtime code.

This prevents accidental fallback, shared-cache contamination, class-identity confusion, and tests that compare an implementation to itself.

## 8. Discrepancy ledger

Every mismatch receives a stable ID and record containing:

```text
Discrepancy
├── profile_id
├── fixture_id / minimal reproducer
├── surface owner
├── severity
├── observed upstream result
├── observed FrankenSymPy result
├── comparator and environment
├── root-cause hypothesis
├── status and assignee
├── allowed only in preview? why?
├── closure tests
└── first/last affected commits
```

Severity classes:

- **C0 certification blocker:** wrong result, wrong class/object contract, hidden fallback, crash, corruption, unsound proof, import incompatibility, or security boundary failure.
- **C1 high:** common ecosystem-visible mismatch, warning/exception drift, serialization incompatibility, or performance pathology that makes compatible behavior unusable.
- **C2 medium:** less common but public behavior mismatch.
- **C3 low:** cosmetic or obscure behavior that remains part of full parity debt.
- **Observation:** not yet shown to be an incompatibility.

A certified profile has no open discrepancy that its manifest defines as release-blocking. Exclusions are not deletions: every excluded upstream test or surface is listed, justified, and assigned a gate.

## 9. Certification gates

A drop-in artifact is certifiable only when all gates pass on the same commit and build provenance:

1. **Inventory closure:** complete profile manifests and no unexplained public-surface deltas.
2. **Object-model closure:** custom subclass, metaclass, hook, held-form, hashing, sorting, reconstruction, mutation, and pickle suites pass.
3. **Upstream suite:** all applicable upstream tests pass; every exclusion is ledgered and independently reviewed.
4. **Generated differential:** type/domain/assumption-aware expression grammars pass at required scale.
5. **Metamorphic:** declared mathematical and structural metamorphisms pass without masking differential failures.
6. **Ecosystem corpus:** selected packages, notebooks, and serialized artifacts pass their profile matrix.
7. **Concurrency/replay:** deterministic and production runtimes agree within declared policies; cancellation leaves no orphan work or cache publication.
8. **Security/resource:** adversarial inputs, expression bombs, malformed serialization, and Python-hook boundaries obey budgets and fail closed.
9. **Persistence/repair:** if enabled in the artifact, crash and corruption tests prove digest checking, recovery, and RaptorQ repair boundaries.
10. **Parity-gated performance:** every reported benchmark case passed its semantic gate in the same run.
11. **No-oracle-runtime:** binary/package inspection and hostile import tests prove upstream SymPy is not a runtime dependency or fallback.
12. **Claim closure:** every public badge/table/README claim resolves through the claims registry to artifacts from this commit.

## 10. Profile evolution

To advance from one upstream SymPy release to another:

1. freeze the new upstream commit and dependency environment;
2. generate a profile diff covering source inventory, tests, signatures, classes, behavior probes, and serialization;
3. classify each delta as addition, intentional upstream change, ambiguous drift, or suspected upstream bug;
4. create or update discrepancy records;
5. preserve the prior profile and its artifacts;
6. certify the new profile independently.

FrankenSymPy may support multiple profiles concurrently when the compatibility shell can dispatch versioned behavior without contaminating native semantics. It must not create one blended behavior that matches no real SymPy release.

## 11. Native extensions without compatibility leakage

Advanced capabilities live behind explicit APIs such as:

```python
import frankensympy as fs

result = fs.prove(expr1, expr2, assumptions=ctx, budget=budget)
result.evidence
result.receipt
result.replay_bundle()
```

or an explicit native namespace in a drop-in environment:

```python
import sympy
from sympy import _franken_native
```

The exact namespace remains a packaging decision, but the rule is fixed: richer evidence envelopes, typed refusals, persistent workspaces, and selector controls cannot silently change ordinary profiled SymPy calls.

## 12. Forbidden compatibility shortcuts

The following are release-blocking:

- shipping upstream SymPy as a hidden fallback;
- counting an exported name as implemented behavior;
- replacing arbitrary Python subclasses with one opaque Rust class;
- normalizing held forms before the compatibility shell has observed them;
- converting `None` assumptions to `False`;
- using mathematical equivalence to excuse wrong type, args, printer, warning, exception, or pickle behavior;
- weakening a comparator or exclusion policy to make a gate pass;
- regenerating oracle goldens without an upstream/profile change review;
- benchmarking cases that failed parity;
- claiming compatibility against a moving branch;
- allowing a repair or hardening heuristic to alter strict-profile behavior without an explicit native call;
- publishing a drop-in wheel with an unresolved no-oracle-runtime gate.

## 13. Definition of drop-in 1.0

FrankenSymPy 1.0 drop-in status means that at least one immutable profile has a published artifact for each supported ABI/platform combination, all certification gates pass on the release commit, upstream SymPy is absent from production runtime dependencies, and the public discrepancy ledger contains no unacknowledged or hidden debt.

It does not mean every future SymPy release is automatically supported. Each release is a new compatibility theorem backed by executable evidence.