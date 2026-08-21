# Donor deep dive: FrankenNumPy and FrankenSciPy

**Status:** normative source audit and architecture input  
**Pinned sources:** `Dicklesworthstone/franken_numpy@d328412082de80d7eb08addd4eddcec002f001ed`, `Dicklesworthstone/frankenscipy@216906fda2dc442361cec63195136ba8dcca1499`  
**Audit date:** 2026-08-20  
**Scope:** compatibility inventories, semantic calculus engines, strict/hardened modes, differential oracles, expected-loss portfolios, condition diagnostics, conformal guardrails, performance receipts, and Python boundary costs

## 1. Executive conclusion

FrankenNumPy and FrankenSciPy contribute two complementary doctrines.

FrankenNumPy says that compatibility is a measured set of observable contracts, not a count of names. It centralizes difficult semantics in small deterministic engines, records strict/hardened decisions, and makes oracle profile, seed, environment, and artifact provenance explicit.

FrankenSciPy says that a high-performance scientific engine should not hard-code one algorithm per operation. It diagnoses the problem regime, selects among safe strategies by explicit loss, records why, and falls back when evidence changes.

For FrankenSymPy the synthesis is:

> Freeze semantics first. Diagnose structure second. Select generators by expected loss third. Verify the resulting mathematical claim independently. Publish only the verified result and its exact evidence.

## 2. Source surfaces examined

### FrankenNumPy

- `README.md`: clean-room compatibility, complete public-surface inventory, shape/stride semantics, strict/hardened modes, differential oracle, RaptorQ artifacts, and evidence-led performance;
- `crates/fnp-ndarray/src/lib.rs`: bounded shape, broadcast, stride, view-span, and layout calculus in safe Rust;
- `crates/fnp-runtime/src/lib.rs`: compatibility classes, decision actions, explicit loss model, evidence terms, audit context, and fail-closed unknowns;
- `crates/fnp-conformance/src/lib.rs`: contract schemas, divergence ledger, API coverage, differential fixtures, adversarial and metamorphic suites, seeds, environment fingerprints, and artifact references;
- crate decomposition across dtype, iteration/transfer, ufunc, linalg, random, I/O, runtime, conformance, and Python boundary.

### FrankenSciPy

- `README.md`: Condition-Aware Solver Portfolio, stability-first policy, differential SciPy oracles, multi-regime algorithms, and per-call audit evidence;
- `crates/fsci-runtime/src/lib.rs`: condition states, solver actions, expected-loss matrix, structural evidence, fallback policy, and calibration monitor;
- domain crates for dense/sparse linear algebra, optimization, integration, special functions, transforms, statistics, and conformance;
- current performance commits that pair optimized and old paths in one binary and require byte-identical results before claiming speed.

## 3. Adopt from FrankenNumPy: semantic calculus engines

Symbolic systems have their own analogues of dtype and stride semantics. These should be owned by small deterministic engines rather than scattered across algorithms.

### 3.1 Domain calculus

Owns:

- exact domain identities;
- membership and coercion rules;
- algebraic extension construction;
- lossless versus lossy conversion;
- common-domain selection;
- ambiguity and coherence refusal;
- profile-specific compatibility behavior.

### 3.2 Binder and substitution calculus

Owns:

- bound/free variable classification;
- alpha-equivalence;
- capture-avoiding substitution;
- dummy generation and canonical naming;
- scope graph construction;
- simultaneous substitution semantics;
- Python-facing object identity behavior where observable.

### 3.3 Assumption calculus

Owns:

- fact normalization;
- implication and contradiction;
- context fork/merge;
- query outcomes including unknown;
- invalidation dependencies;
- profile-specific legacy inference rules.

### 3.4 Numeric tower calculus

Owns:

- integer, rational, algebraic, real/complex approximate, interval/ball, matrix, polynomial, series, and symbolic-number promotion;
- exact/approximate boundary rules;
- signed zero, NaN, infinities, and complex branch semantics;
- precision and rounding context identities;
- refusal when a promotion would silently weaken evidence.

### 3.5 Evaluation and branch calculus

Owns:

- principal branches;
- excluded points and singularities;
- continuation paths;
- real-domain restrictions;
- conditional versus unconditional equality;
- evaluation precision and enclosure policy.

Centralizing these semantics enables exhaustive and metamorphic testing, independent of whichever generator uses them.

## 4. Adopt from FrankenNumPy: observable compatibility inventory

A drop-in claim includes more than functions and values. The surface inventory records:

- import paths and aliases;
- classes, metaclasses, MRO, and subclass checks;
- constructors and signatures;
- `args`, `func`, assumptions, and hash behavior;
- equality, ordering, iteration, and container protocols;
- exact return type and lazy/eager behavior;
- exception class, message, warning, and traceback shape;
- printers, repr, LaTeX, code generation, and pickling;
- mutability, caches, weak references, copy/deepcopy, and object identity;
- custom Python hooks and delegation;
- package metadata and resolver behavior.

Every row is present, partial, missing, delegated, incompatible, not applicable, or unknown under one immutable profile. Importability alone is never counted as compatibility.

## 5. Strict, hardened, native, and certified modes

Mode is a policy bundle, not a boolean threaded through arbitrary code.

### Strict compatibility

- match the named SymPy profile’s observable behavior;
- preserve legacy order and quirks when required;
- delegate only through registered surfaces;
- refuse unknown behavior rather than inventing a result;
- parity evidence is part of release gating.

### Hardened compatibility

- preserve compatible behavior while applying registered resource/safety limits;
- emit typed decisions and audit evidence for every deviation;
- never silently repair semantics;
- full validation or refusal for unknown classes.

### Native deterministic

- use FrankenSymPy’s explicit canonical policies;
- optimize for reproducibility, proof availability, and performance;
- may differ observably from SymPy only through declared profile semantics.

### Certified

- require a portable certificate family and reference verification;
- reserve verifier resources before launching generators;
- never accept an uncertified fallback as certified output.

## 6. Adopt from FrankenSciPy: diagnose before selecting

A symbolic request is classified by exact diagnostics such as:

- operator inventory and expression depth;
- domain and coefficient ring;
- degree, arity, sparsity, and coefficient height;
- monomial order and ideal dimension;
- matrix dimensions, sparsity, displacement/Toeplitz/Vandermonde structure, rank hints;
- polynomial square-freeness, modular image behavior, and factor-degree patterns;
- rewrite-graph branching and critical-pair density;
- assumptions and branch-policy complexity;
- expected proof size and verifier cost;
- cache and prior-certificate availability;
- Python opacity/effect classification;
- memory and time budget.

Diagnostics are themselves bounded and audited. A heuristic diagnostic may influence selection but cannot become an unproved premise of the accepted claim.

## 7. Expected-loss algorithm portfolios

For each operation family, define:

- state model;
- candidate strategy set;
- hard eligibility predicates;
- expected cost dimensions;
- asymmetric failure losses;
- deterministic tie-break;
- fallback graph;
- certificate/evidence capabilities;
- protected verification reserve;
- calibration and reset policy.

A factorization portfolio might consider:

- deterministic trial/square-free baseline;
- modular factorization with Hensel lifting;
- van Hoeij/lattice reconstruction;
- sparse interpolation routes;
- finite-field specialists;
- remote portfolio branches;
- upstream delegation in strict compatibility mode.

The selected action minimizes registered expected loss only among eligible strategies. A strategy that cannot produce the required evidence class is ineligible for a certified request no matter how fast it is.

## 8. Selection is not acceptance

This boundary is absolute:

```text
diagnostics + history + policy
            │
            ▼
       strategy selection
            │
            ▼
   candidate + certificate
            │
            ▼
 independent reference verifier
            │
            ▼
      accepted typed claim
```

Posterior probability, confidence, conformal score, decision card, benchmark history, or model vote cannot upgrade evidence. They decide what to try, when to hedge, and when to fall back.

## 9. Portfolio execution under asupersync

Each request region owns:

- diagnostic tasks;
- candidate generator children;
- remote leases;
- checkpoint/artifact tasks;
- verifier child with protected budget;
- publication finalizer.

The planner may launch multiple pure generators. The first candidate to finish may nominate a result; it cannot publish. Losers are cancelled and drained. Effectful or unknown Python hooks are never duplicated speculatively.

Hedging thresholds can be adaptive, but the policy, observations, and reset history are replay artifacts.

## 10. Calibration claims must be exact

The numeric donors contain useful calibration mechanisms, but FrankenSymPy must distinguish heuristic monitoring from a proved conformal or e-process guarantee.

A monitor may claim finite-sample or anytime validity only when it specifies and satisfies:

- population and sampling rule;
- exchangeability or martingale/null assumptions;
- filtration;
- score/betting construction;
- optional stopping scope;
- subgroup/multiplicity policy;
- reset and retraining policy;
- treatment of censored failures, timeouts, and refusals.

Otherwise it is an empirical adaptive heuristic, still useful but named honestly.

## 11. Differential conformance design

Fixtures carry:

- operation and exact inputs;
- profile and environment root;
- seed lineage;
- upstream version/pin;
- expected observable bundle;
- normalization rules;
- artifact references;
- reason code and disposition;
- minimization lineage for discovered discrepancies.

Run upstream and native implementations in isolated processes where Python global state, import order, hash seed, locale, warnings filters, or optional dependencies can affect behavior.

Conformance compares structured observations, not only pretty-printed output.

## 12. Metamorphic and adversarial relations

Required relations include:

- alpha-renaming invariance where applicable;
- substitution composition under freshness conditions;
- factor product reconstruction;
- gcd divisibility and unit normalization;
- derivative linearity/product/chain relations under domain conditions;
- matrix solve residual identities;
- permutation equivariance for symmetric operations;
- serialization round-trip;
- cache-on/cache-off equivalence;
- single-thread/multi-thread equivalence;
- algorithm-route equivalence after reference verification;
- assumption strengthening/weakening effects;
- branch-policy mutation.

Every relation names its preconditions. Invalid metamorphic assumptions are more dangerous than missing tests.

## 13. Performance evidence discipline

A performance patch must include:

- exact old and new route in the same binary or immutable build pair;
- semantic equivalence test, preferably exact or certificate-verified;
- adversarial fixtures that would expose a skipped computation;
- complete public-operation timing;
- multiple workload regimes and shapes;
- allocation and peak-memory receipts;
- tail percentiles;
- machine/toolchain/load fingerprints;
- A/A null or matched-arm control for noisy claims;
- no claim when environmental noise overwhelms the effect.

The FrankenNetworkX lesson about an attribute lookup dominating the graph walk generalizes: profile before attributing cost to the mathematically obvious loop.

## 14. Python-boundary economics

For small operations, conversion and object construction can dominate the Rust kernel. Portfolio cost models therefore include:

- Python argument parsing;
- wrapper interning and identity preservation;
- conversion to compact native forms;
- GIL or free-threaded synchronization;
- callback execution;
- exception/warning construction;
- result materialization;
- cache synchronization.

Microkernel wins do not justify routing a public call through native code when boundary cost makes the full operation slower.

## 15. Explicit rejections

FrankenSymPy rejects:

- function-count compatibility claims;
- algorithm selection encoded as unreviewed `if` chains across domain crates;
- a learned model that can bypass hard eligibility or verification;
- calibration wording unsupported by its assumptions;
- excluding timeouts/refusals from monitoring because they look bad;
- performance claims from one friendly size or shape;
- accepting a faster route whose output is merely “close enough” for an exact claim;
- duplicate execution of unknown/effectful Python callbacks;
- mode-dependent mathematical truth;
- silent fallback from certified to uncertified evidence.

## 16. Implementation order

1. freeze compatibility observation schema and profile roots;
2. implement domain, binder, assumption, numeric-tower, and branch calculus references;
3. create diagnostic schemas per first operation family;
4. implement deterministic baselines and portable verifiers;
5. add portfolio policy and decision receipts;
6. add pure speculative generators and fallback graphs;
7. build differential, metamorphic, and adversarial corpora;
8. add calibrated monitoring only after assumptions are registered;
9. certify full-operation performance across regimes;
10. expand portfolios without widening the trusted verifier core.
