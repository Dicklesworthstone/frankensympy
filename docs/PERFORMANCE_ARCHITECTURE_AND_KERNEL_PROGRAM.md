# Performance architecture and kernel program

**Status:** normative architecture contract  
**Scope:** representations, exact arithmetic, polynomial and linear algebra kernels, term DAGs, rewriting, parallelism, architecture specialization, Python boundary economics, benchmarks, and admission

## 1. Goal

FrankenSymPy targets world-class symbolic performance without C/C++/Fortran backends and without project-authored unsafe Rust. This is achievable only if performance is designed through the entire semantic stack rather than delegated to isolated microkernels.

The performance program optimizes:

- full public-operation latency and throughput;
- time to first verifiable subclaim;
- peak and retained memory;
- allocation and cache behavior;
- proof/certificate generation and verification;
- cancellation waste;
- Python conversion and wrapper construction;
- compilation time and binary size;
- deterministic scaling on Apple Silicon and high-core-count x86-64.

## 2. Representation spine

### 2.1 Local handles versus durable IDs

Inside one immutable snapshot, terms, domains, coefficients, proof nodes, and graph nodes use compact integer handles. Durable content IDs are computed at object/materialization boundaries, not on every internal edge traversal.

A handle never escapes without its snapshot root. Durable identity never includes a handle.

### 2.2 Term storage

Use arity-specialized immutable nodes:

- inline zero/one/two/three-child layouts;
- packed operator/kind/flags;
- separate cold metadata/provenance;
- structure-of-arrays options for batch traversals;
- canonical child ranges for large arity;
- hash-consing/interning under an explicit profile;
- bounded generation counters for mutable build arenas;
- frozen compact snapshots for long-lived graphs.

Avoid one heap allocation and one reference-count object per tiny term when ownership can be represented by snapshot-lifetime arenas and compact handles.

### 2.3 Small values

Inline common integers, rationals, symbols, exponents, and dimensions. Promote to heap-backed exact objects only when bounds are exceeded. Demotion is optional and must preserve canonical identity.

### 2.4 Cold data separation

Provenance, pretty-print hints, telemetry, search traces, and optional formal artifacts live outside hot mathematical nodes. Verifier-required fields remain in the canonical closure.

## 3. Integer arithmetic

Owned safe-Rust integer kernels include:

- sign/magnitude or another registry-chosen canonical representation;
- normalized limb vectors with small inline capacity;
- add/subtract with carry/borrow;
- schoolbook multiplication and specialized squaring;
- Karatsuba;
- Toom variants after crossover evidence;
- exact NTT/CRT multiplication with self-check/reconstruction bounds;
- single/multi-limb division;
- Burnikel–Ziegler or Newton-style large division after proof/testing gates;
- binary/classical/Lehmer/half-GCD routes;
- modular reduction, exponentiation, inverses, and batch inverses;
- integer roots and perfect-power tests;
- deterministic prime generation/testing policies where required.

Thresholds are architecture/profile data learned from pinned benchmarks and constrained by hard correctness/space bounds. They are not global magic constants.

## 4. Rational and algebraic arithmetic

Rational operations use:

- normalized denominator sign;
- gcd-aware cross-cancellation before multiply/add;
- delayed normalization only where the invariant and size bounds are explicit;
- small rational inline forms;
- exact comparison without unnecessary common-denominator materialization;
- certificate-friendly normalization.

Algebraic numbers use immutable defining-polynomial and root-isolation identities, modular/resultant methods where profitable, and exact interval/refinement certificates. Approximate enclosures are never confused with algebraic identity.

## 5. Polynomial representations

A polynomial value has one semantic identity and may have multiple physical views:

- dense univariate;
- sparse sorted terms;
- recursive multivariate;
- distributed monomials;
- modular images;
- evaluation/interpolation form;
- factored form with explicit unit/content;
- truncated series with order ideal.

Representation conversion is explicit, bounded, cached by universe/policy root, and excluded from semantic identity.

## 6. Polynomial kernels

Initial routes include:

- dense add/subtract and scalar operations;
- schoolbook, Karatsuba, Toom, Kronecker, and NTT/CRT multiplication;
- sparse heap/hash-free merge multiplication with deterministic term order;
- content/primitive-part and square-free decomposition;
- subresultant PRS;
- modular gcd and reconstruction;
- multipoint evaluation/interpolation/product/remainder trees;
- Hensel lifting;
- finite-field factorization;
- lattice/recombination routes;
- Gröbner basis kernels with monomial indexing and critical-pair queues;
- rational-function normalization and partial fractions.

Each optimized generator remains independently checked by product reconstruction, divisibility/Bézout, or another portable certificate family.

## 7. Exact linear algebra

Representations:

- compact dense row/column-major matrices;
- CSR/CSC sparse matrices;
- block sparse;
- diagonal, triangular, banded;
- Toeplitz/Hankel/Vandermonde/Cauchy/displacement-structured;
- modular images and CRT accumulators;
- expression matrices with lazy materialization policy.

Kernels:

- fraction-free Gaussian/Bareiss elimination;
- modular solve/rank/determinant with rational reconstruction;
- sparse pivoting and fill-reducing deterministic orderings;
- exact nullspace and Smith/Hermite normal forms;
- structured solvers;
- matrix polynomial and recurrence methods;
- certificate extraction for residual, determinant, row-equivalence, and inverse claims.

Planner heuristics select work; exact identities verify results.

## 8. Rewrite and matching engine

Performance-critical components:

- compiled pattern programs;
- discrimination nets and operator/domain partitions;
- bounded commutative/associative matching;
- exact binder-aware substitution;
- rule applicability caches keyed by universe root;
- deterministic critical-pair queues;
- e-graph storage with proof-producing union reasons;
- extraction under registered exact cost/tie-break policy;
- loop and growth guards;
- proof trace compression.

User Python hooks are effect boundaries and cannot run inside speculative parallel matching.

## 9. Graph kernels

The local graph substrate uses compact adjacency arrays and deterministic traversal. Optimize:

- topological order/cycle witness;
- SCC/condensation;
- reachability and invalidation cones;
- dominators;
- dependency closure;
- workstream ready frontiers;
- proof/provenance slices.

FrankenNetworkX algorithms may be selectively reused after dependency review, but portable graph verifiers remain small.

## 10. Parallelism

Asupersync regions own all parallel work. Deterministic decomposition examples:

- partition modular primes by counter-derived ranges;
- split independent subexpressions by canonical child order;
- parallel product/remainder tree levels;
- block matrix operations;
- independent factor recombination branches;
- proof-node verification by dependency frontier;
- portfolio candidates with loser drain.

Results merge through canonical ordering and checked reducers, never completion order. Budgets reserve verifier and cleanup capacity.

## 11. Architecture specialization without unsafe

### Apple Silicon

- tune limb/block sizes for high memory bandwidth and large unified caches;
- exploit safe portable SIMD where exact lane arithmetic applies;
- favor fewer allocations and contiguous immutable arenas;
- benchmark performance/efficiency cores and thread-placement policies;
- include Python universal2/arm64 packaging costs where relevant.

### x86-64 and high-core AMD

- AVX2 portable floor for optimized release class, with scalar fallback;
- safe portable SIMD/autovectorized loops;
- NUMA-aware region partitioning through safe owned shards;
- avoid central atomic counters and false sharing;
- batch global ordering/commit work;
- measure 1–N core scaling, contention, and memory bandwidth saturation.

Direct unsafe intrinsics are not pre-authorized. If safe `std::simd` or compiler vectorization cannot express a kernel, redesign or defer the specialization.

## 12. Memory governance

Every major kernel reports or bounds:

- input and output bytes;
- temporary peak;
- retained cache bytes;
- proof/certificate bytes;
- remote/checkpoint bytes;
- allocation count/classes;
- spill eligibility;
- cancellation cleanup.

Caches have immutable keys, admission policy, quotas, and rebuildability. A cache miss cannot alter semantics.

## 13. Python boundary batching

Batch:

- argument conversion;
- symbol/term wrapper creation;
- list/matrix construction;
- printer fragments;
- exception/warning assembly;
- iterator pulls where protocol permits.

Use wrapper interning only under profiles that permit it. Full-operation benchmarks decide whether a native route beats upstream/delegated behavior for small inputs.

## 14. Proof-aware scheduling

Prioritize work that unlocks independent verification:

1. claim/schema/environment roots;
2. smallest complete subclaim closure;
3. certificate nodes and source terms needed by the next verifier frontier;
4. remaining verifier-complete closure;
5. replay/search provenance;
6. repair and speculative artifacts.

Metrics include time to first verified subclaim and time to verified complete result, not only generator completion.

## 15. Benchmark design

Each kernel registry row defines:

- semantic operation and regime axes;
- scalar/reference route;
- optimized route;
- incumbent comparison where relevant;
- exact corpus root;
- correctness/certificate gate;
- architecture classes;
- warm/cold/cache states;
- thread counts;
- allocation and memory measurements;
- tail metrics;
- same-invocation or immutable-build pairing;
- A/A control;
- noise refusal threshold;
- admission and rollback criteria.

## 16. Compile and binary budgets

Track:

- clean/incremental build time;
- monomorphization/codegen units;
- feature-driven dependency closure;
- Python extension size;
- portable verifier size;
- Wasm size;
- symbol/table/generated-code growth.

Specialization that makes deployment impractical is not free performance.

## 17. Admission rule

An optimized kernel becomes default only when:

- reference equivalence passes;
- certificate verification passes where applicable;
- mutation/adversarial tests demonstrate sensitivity;
- complete-operation performance wins in registered regimes;
- memory/tails/cancellation are acceptable;
- deterministic output and proof identity are stable;
- architecture fallback is correct;
- quarantine/rollback key exists.

No benchmark result upgrades mathematical evidence.
