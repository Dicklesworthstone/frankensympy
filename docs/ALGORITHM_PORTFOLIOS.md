# FrankenSymPy proof-carrying algorithm portfolios

**Status:** normative architecture contract  
**Scope:** adaptive planning, asymmetric loss, speculative execution, independent verification, and domain-by-domain algorithm program

## 1. Thesis

No single algorithm dominates symbolic computation. The best method depends on domain, degree, sparsity, coefficient height, variable order, branch structure, assumptions, proof cost, available parallelism, and the shape of the desired result.

FrankenSymPy therefore treats each major symbolic operation as a **proof-carrying portfolio**:

1. inspect the exact problem instance;
2. form a deterministic, auditable plan over eligible strategies;
3. run one or more strategies under nested budgets;
4. treat outputs as candidates;
5. verify candidates independently against the requested claim;
6. publish the first candidate meeting the evidence contract;
7. cancel and drain losers;
8. preserve decision, verification, and replay receipts.

Completion order affects latency. It never determines mathematical acceptance.

## 2. Portfolio request

```text
PortfolioRequest
├── operation and typed claim target
├── input term/domain/context IDs
├── compatibility profile, if applicable
├── minimum evidence class
├── result-form preferences
├── exactness/approximation policy
├── branch and completeness policy
├── resource budget tree
├── determinism policy
├── persistence/checkpoint policy
├── remote-execution capability
└── privacy/security capabilities
```

An ordinary compatibility call is translated into a profile-governed request whose visible return behavior matches upstream. A native call can demand proof class, completeness, proof-size limit, or a conditional result.

## 3. Instance evidence vector

The planner computes a versioned feature record, not an opaque embedding:

- operator and domain family;
- exact coefficient domain and characteristic;
- variable count and order;
- total/per-variable degree;
- term count and sparsity estimates;
- coefficient height and growth bounds;
- expression DAG/tree size and sharing;
- symmetry, homogeneity, separability, and block structure;
- commutative/noncommutative structure;
- matrix shape, density, bandwidth, displacement rank, and known properties;
- assumptions and branch-cut obligations;
- singularities, discontinuities, and piecewise regions;
- estimated output size and proof size;
- prior verified outcomes for the same structural family;
- available cores, memory, SIMD, and remote workers;
- compatibility constraints and opaque Python nodes.

Feature extraction itself consumes a budget and emits a receipt. Unknown features remain unknown rather than guessed.

## 4. State and action model

Each portfolio declares:

- a finite or structured latent state model representing important problem regimes;
- eligible actions/algorithms;
- evidence signals and their provenance;
- a loss matrix or loss function with asymmetric consequences;
- posterior/calibration update policy;
- safe baseline action;
- shadow actions and fallback graph;
- verification cost model;
- abstention/refusal policy.

A typical loss ordering is:

```text
false accepted exact result  >>>  corruption/security breach
                             >>>  wrong compatibility behavior
                             >>>  unbounded resource use
                             >>>  inconclusive/refused result
                             >>>  slow but correct result
```

Numerical values are portfolio-specific and calibrated from verified workloads. They are not decorative confidence scores.

## 5. Decision card

Every consequential plan emits:

```text
DecisionCard
├── planner/policy version
├── input fingerprint
├── feature evidence vector
├── candidate state probabilities or calibrated scores
├── eligible and rejected actions with reasons
├── expected loss/cost by action
├── chosen launch set and order
├── verification plan
├── budgets and cancellation boundaries
├── fallback graph
├── cache/checkpoint decisions
└── drift-monitor references
```

Decision cards are diagnostic evidence about planning, not mathematical evidence about a result.

## 6. Safe speculative execution

### 6.1 Region topology

A portfolio runs in an asupersync region:

```text
portfolio region
├── feature/probe tasks
├── candidate generator scopes
│   ├── strategy A
│   ├── strategy B
│   └── strategy C
├── verifier scopes
├── checkpoint/artifact scope
└── publication coordinator
```

No task outlives the region. The coordinator returns only after losers and their child verifiers have drained or reached a declared non-cooperative boundary.

### 6.2 Two-phase winner publication

A candidate generator sends:

```text
CandidateReady(value, certificate, generator_receipt)
```

The publication coordinator reserves a winner slot but does not commit it. The appropriate verifier returns `Verified`, `Rejected`, or `Inconclusive`. Only `Verified` can commit. Rejected candidates feed diagnostics and may trigger fallback.

### 6.3 Cache publication

Candidate caches and verified caches are separate. A cancelled generator cannot publish shared verified state. A verifier may publish only after checking exact universe IDs and canonical payloads.

## 7. Determinism modes

- **Strict deterministic:** fixed probes, launch set, seed ranges, tie breaks, and accepted output form; scheduling may vary but terminal accepted result and receipt are byte-stable.
- **Replay deterministic:** production adaptive behavior is recorded and exactly replayable from the receipt/bundle.
- **Latency adaptive:** live telemetry can alter launch timing; accepted results remain verifier-governed, while decision cards record adaptation.
- **Compatibility profile:** reproduce upstream-visible ordering/form even when native extraction would prefer another equivalent result.

Nondeterminism must be declared in the receipt. Hash-map iteration, task completion order, or worker arrival order cannot become an accidental mathematical tie break.

## 8. Calibration and anytime-valid monitoring

Portfolio selectors learn only from outcomes that passed the relevant verification gate. Telemetry is partitioned by profile, domain, architecture, size regime, and evidence requirement.

Conformal/e-process machinery monitors:

- rejection and fallback rates;
- selector regret proxies against shadow baselines;
- subgroup performance regressions;
- proof-size/runtime drift;
- workload shift;
- cache-hit anomaly patterns;
- remote-worker defect streams.

An alarm can freeze learning, revert to the safe baseline, increase shadow verification, or quarantine a strategy. It cannot certify an individual result.

## 9. Common portfolio patterns

### 9.1 Representation race

Try dense, sparse, recursive, modular, and black-box representations after cheap probes. Conversions are explicit and budgeted.

### 9.2 Modular image and reconstruction

Compute over multiple finite fields, combine via CRT/rational reconstruction, apply a valid bound, and verify exactly in the source domain.

### 9.3 Evaluation/interpolation

Evaluate a multivariate object at deterministic or recorded points, solve lower-dimensional problems, interpolate a candidate, and verify the reconstructed identity.

### 9.4 Search plus checker

Use heuristic/learned search to propose a factorization, antiderivative, rewrite, or solution. A small exact/certified checker decides acceptance.

### 9.5 Symbolic/numeric sandwich

Use certified numerics to isolate regimes, roots, signs, or branches; perform exact symbolic work within those partitions; verify exact and enclosure obligations separately.

### 9.6 Incremental continuation

Expose a checkpointable state for algorithms such as Gröbner, integration search, quantifier elimination, or large exact linear algebra. Cancellation yields `Inconclusive` plus a resumable continuation, not a fake partial theorem.

## 10. Polynomial arithmetic portfolio

### 10.1 Representation actions

- dense univariate;
- sparse distributed multivariate;
- recursive tower;
- Kronecker substitution;
- straight-line program;
- evaluation black box;
- modular images;
- truncated series.

### 10.2 Multiplication

Actions include:

- schoolbook/sparse accumulation;
- Karatsuba/Toom variants;
- NTT/CRT convolution;
- Kronecker substitution;
- heap-based sparse multiplication;
- evaluation/interpolation for suitable multivariate shapes.

Acceptance is exact canonical equality in the ring. Thresholds are profile- and architecture-calibrated with a scalar baseline in the same benchmark invocation.

### 10.3 Division and remainder

Use classical, Newton/reversal, subresultant/pseudo-division, sparse and modular methods according to domain. Certificates check quotient/remainder identity and degree constraints.

## 11. GCD and resultant portfolio

Actions:

- Euclidean/subresultant PRS;
- modular GCD with unlucky-prime detection;
- evaluation/interpolation multivariate GCD;
- heuristic GCD proposals followed by exact verification;
- sparse/black-box methods;
- Brown/Collins-style modular resultants and subresultants.

Verification checks divisibility and maximality criteria appropriate to the domain, often via cofactors/Bézout or degree/content witnesses.

## 12. Factorization portfolio

### 12.1 Univariate

- content/primitive and square-free decomposition;
- distinct/equal-degree finite-field factorization;
- Berlekamp/Cantor-Zassenhaus-style actions;
- Hensel lifting;
- LLL-assisted recombination where implemented safely;
- van Hoeij-style recombination;
- algebraic-field factorization;
- sparse/lacunary special cases.

### 12.2 Multivariate

- evaluation plus Hensel lifting;
- variable-by-variable reconstruction;
- sparse interpolation;
- absolute factorization and extension fields;
- symmetry/homogeneity decomposition;
- heuristic form recognition followed by exact product and irreducibility checks.

The requested claim states whether factors must be irreducible, square-free, primitive, monic, absolute, or merely a nontrivial decomposition. The verifier grants only that claim.

## 13. Gröbner and ideal portfolio

Actions include:

- Buchberger with deterministic criteria;
- F4-style sparse linear algebra;
- F5/signature methods;
- modular Gröbner with reconstruction;
- FGLM order conversion for zero-dimensional ideals;
- change-of-ring/domain strategies;
- triangular decomposition/regular chains where appropriate;
- saturation/elimination-specialized plans;
- incremental basis updates for workspace branches.

Features include sugar degree, pair queue shape, coefficient growth, sparsity, Hilbert information, and prior modular behavior.

Certificates prove ideal membership for output generators and the Gröbner criterion under the exact order/domain. Reduced/minimal claims add normalization checks.

## 14. Exact linear algebra portfolio

### 14.1 Dense exact matrices

- fraction-free Bareiss elimination;
- denominator-cleared LU/LDL;
- modular solve/determinant/rank with CRT reconstruction;
- p-adic lifting;
- block and structured algorithms;
- characteristic/minimal polynomial methods;
- black-box Wiedemann/Lanczos variants with exact verification.

### 14.2 Sparse matrices

- sparse fraction-free elimination with fill forecasting;
- modular sparse elimination;
- block Wiedemann;
- graph/order-based pivot planning;
- rank-profile and nullspace methods;
- structure-aware Toeplitz/Hankel/banded actions.

### 14.3 Symbolic matrices

- domain matrix lowering;
- fraction-field versus polynomial-matrix strategies;
- interpolation in symbolic parameters;
- determinant identities and decomposition;
- expression swell controls;
- piecewise singular/non-singular branches.

Every uniqueness, rank, completeness, or invertibility claim has a separate obligation.

## 15. Simplification portfolio

Simplification is goal-directed rather than one monolithic function.

Actions include:

- deterministic local canonical rules;
- polynomial/rational normalization;
- trigonometric/hyperbolic basis changes;
- power/log combination under proved side conditions;
- radical/algebraic-number normalization;
- piecewise condition simplification;
- bounded e-graph search;
- common-subexpression-aware cost extraction;
- target-language/code-generation simplification;
- numeric-stability-oriented forms.

The planner returns alternatives on a Pareto frontier when no single form dominates the requested cost vector.

Compatibility `simplify` reproduces the profile's result policy; native `simplify_with_goal` exposes objectives and proof evidence.

## 16. Differentiation portfolio

Most ordinary differentiation is a deterministic structural proof. Portfolios enter for:

- high-order and multivariate derivatives;
- sparse Jacobian/Hessian generation;
- tensor/matrix calculus;
- implicit differentiation;
- derivatives of special functions;
- automatic differentiation compilation;
- repeated subexpression reuse;
- differentiation under integral/sum signs with side conditions.

Actions combine symbolic rules, DAG dynamic programming, truncated series, forward/reverse AD over compiled graphs, and sparsity coloring. Native output can include derivative proof, sparsity certificate, and compiled evaluator.

## 17. Integration portfolio

Integration is intrinsically heterogeneous. Actions include:

- table/linearity/substitution/parts rules;
- rational integration (Hermite reduction, Lazard–Rioboo–Trager/Rothstein–Trager families);
- algebraic-function integration;
- Risch-style elementary integration components;
- Meijer G/hypergeometric transforms;
- trigonometric/radical substitutions;
- residue/contour methods;
- creative telescoping for parameterized integrals;
- ODE-based recognition;
- heuristic pattern search;
- certified quadrature and enclosure when exact form is unavailable.

The planner separates claims:

- an antiderivative on a specified domain;
- a definite integral with convergence/endpoints;
- a principal value;
- a conditional expression;
- a certified numerical enclosure;
- no elementary antiderivative under a proven decision procedure;
- merely “not found.”

Candidate differentiation is one verifier component, not universal completeness proof.

## 18. Limits, series, and asymptotics portfolio

Actions:

- direct substitution and continuity proof;
- dominant-term/order algebra;
- Gruntz-style comparison classes;
- series expansion and remainder bounds;
- change of variables;
- monotonicity/squeeze reasoning;
- complex-direction/sector analysis;
- certified punctured-neighborhood enclosures;
- recurrence/differential-equation asymptotics;
- transseries and logarithmic-exponential extensions as research tracks.

Direction, path, sector, branch policy, and oscillatory/nonexistent outcomes are explicit.

Series actions include formal power/Laurent/Puiseux/logarithmic series, asymptotic series, recurrence-based coefficient generation, Newton polygon methods, and fast power-series arithmetic. Formal and analytic claims are distinct.

## 19. Equation and inequality solver portfolio

### 19.1 Algebraic equations

- polynomial factor/root isolation;
- Gröbner/elimination;
- resultants/subresultants;
- rational univariate representation;
- triangular/regular chains;
- CAD/quantifier elimination for real domains;
- modular and homotopy-inspired candidate generation with exact certification;
- parameter stratification and discriminant varieties.

### 19.2 Transcendental equations

- invertible function isolation with branch tracking;
- Lambert W and special-form recognition;
- monotonicity/convexity partition;
- interval Newton/Krawczyk certified isolation;
- periodic-family representation;
- mixed symbolic-numeric branch enumeration.

### 19.3 Inequalities

- exact univariate sign decomposition;
- polynomial CAD/virtual substitution;
- linear/rational arithmetic;
- interval/monotonicity certification;
- Boolean combination and set construction.

### 19.4 Completeness

Solvers return a `SolutionSet` with completeness status:

```text
Complete
CompleteWithinDeclaredDomain
Conditional
FiniteVerifiedSubset
CertifiedRegions
UnknownCompleteness
```

Compatibility lifting follows profile behavior, while native mode never hides completeness uncertainty.

## 20. Diophantine and number theory portfolio

- exact primality testing/certificates and probable-prime candidates kept distinct;
- integer factorization portfolios (trial division, Pollard families, ECM, QS/NFS research integration) with exact product verification;
- modular roots and CRT;
- Pell, linear, quadratic, and selected higher-degree Diophantine solvers;
- lattice reduction and bounded search;
- recurrence and generating-function methods;
- class/group computations where implemented;
- symbolic multiplicative functions and summatory algorithms.

Factorization completeness is easy to verify once the remaining cofactor is certified prime; generator strategy can remain heuristic.

## 21. Logic, SAT, and sets portfolio

### 21.1 Logic

- canonical Boolean simplification;
- BDD/AIG forms;
- DPLL/CDCL SAT with proof logging;
- SMT-style theory combinations for supported arithmetic;
- quantifier elimination;
- fuzzy/three-valued compatibility logic.

SAT returns models; UNSAT requires a checkable proof trace for strong evidence.

### 21.2 Sets

- structural set algebra;
- interval normalization;
- finite-set exact operations;
- image/preimage and condition sets;
- solver-backed membership/subset queries;
- measure/topology properties with assumptions;
- lazy/comprehension representations to avoid expansion.

Membership, emptiness, equality, and subset claims have distinct evidence obligations.

## 22. Special functions and transforms portfolio

A versioned registry describes each function's:

- defining equations/series/integrals;
- domains and branch cuts;
- differentiation/recurrence identities;
- transformations and special values;
- asymptotics;
- numeric algorithms and certified enclosures;
- compatibility printers and assumptions handlers.

Portfolios choose among series, continued fractions, asymptotics, recurrences, differential equations, contour/integral forms, argument reduction, and symbolic transformations. Branch regions are explicit.

Integral transforms (Fourier, Laplace, Mellin, Hankel, Z, etc.) carry convergence regions, distributional conventions, and inverse-transform conditions.

## 23. ODE and PDE portfolio

### 23.1 ODEs

- classification by order, linearity, exactness, homogeneity, singularities, and symmetries;
- first-order methods;
- linear constant/variable coefficient methods;
- Frobenius/power-series solutions;
- Lie symmetry methods;
- differential-algebra/elimination methods;
- systems and matrix exponentials;
- Green functions/transforms;
- certified numerical IVP/BVP fallback through FrankenSciPy.

A solution is verified by substitution plus initial/boundary conditions. Generality/completeness and singular solutions are separate claims.

### 23.2 PDEs

- separation of variables;
- characteristics;
- transforms and Green functions;
- symmetry reductions;
- polynomial/differential Gröbner research tracks;
- discretization/code generation and certified residual analysis.

PDE completeness claims are rare and must not be implied by one verified solution family.

## 24. Geometry, tensor, physics, statistics, and units

### 24.1 Geometry

Use exact predicates, algebraic intersections, projective/homogeneous coordinates where beneficial, certified numeric fallback for difficult algebraic cases, and explicit degeneracy classification.

### 24.2 Tensor and index calculus

Use typed index spaces, variance, symmetry groups, canonicalization under permutation symmetries, contraction planning, sparse representations, and proof-producing index rewrites. Naive factorial canonicalization is forbidden when group algorithms apply.

### 24.3 Physics modules

Mechanics, quantum, vector, continuum, optics, control, and other modules are registries/compositions over the core rather than privileged ad hoc object systems. Dimensional, commutation, coordinate, and convention metadata is explicit.

### 24.4 Statistics

Random variables, distributions, events, expectations, transforms, moments, conditioning, and stochastic processes use measure-aware symbolic objects. Exact, formal, and numerically certified results remain distinct.

### 24.5 Units and dimensions

Dimension vectors, affine/log units, conversion graphs, unit systems, and constants are typed. Dimensional checks can reject invalid expressions before algebraic simplification. Constants carry edition/source provenance rather than one mutable global value.

## 25. Symbolic-to-numeric compilation portfolio

FrankenSymPy lowers verified symbolic expressions into typed numeric programs for FrankenNumPy/FrankenSciPy and standalone Rust/Wasm targets.

Actions include:

- common subexpression elimination over the semantic DAG;
- algebraic strength reduction;
- target-aware function selection;
- Horner/Estrin/paterson-stockmeyer polynomial evaluation;
- sparse Jacobian/Hessian layout and coloring;
- interval/ball evaluator generation;
- branch-safe piecewise lowering;
- SIMD/parallel loop planning;
- exact residual/checker generation alongside approximate solvers.

Generated code carries:

- source `TermId` and context;
- transformation proof/receipts;
- target ABI and floating policy;
- domain guards;
- test vectors and certified/reference lane;
- content digest.

## 26. Portfolio benchmarking

Each benchmark invocation includes:

- the live incumbent and candidate strategies;
- identical inputs, profile, context, evidence requirement, durability mode, and budgets;
- semantic verification before timing admission;
- cold/warm cache distinction;
- memory, proof size, energy/CPU counters where available;
- tail latency and cancellation behavior;
- outcome mix, not only successful cases.

A self-speedup measured without the incumbent is maintenance evidence, not a competitive win.

## 27. Selector security and reward hacking

Defenses include:

- immutable evaluation corpora and hidden holdouts;
- parity/proof gates outside selector control;
- live incumbent comparison;
- subgroup metrics and e-process alarms;
- no training on unverified successes;
- immutable loss policies per release;
- decision-card audits;
- shadow baselines;
- quarantine on suspicious cache/feature shortcuts;
- explicit accounting for refusals and timeouts.

A selector cannot improve its score by weakening the verifier, comparator, benchmark corpus, or requested evidence class.

## 28. Forbidden shortcuts

- choosing the first completed candidate without verification;
- treating selector confidence as evidence;
- using one algorithm with cosmetic parameter changes as “independent” cross-checks;
- hiding failed/refused cases from benchmark aggregates;
- lowering exact work to floats without explicit approximation policy;
- reporting sampled success as complete solution-set evidence;
- silently changing monomial order, branch policy, or assumptions to make a method work;
- launching unbounded portfolios with no region owner;
- leaving loser tasks or remote work running after return;
- publishing candidate results into verified caches;
- learning from unverified outputs;
- changing a loss matrix or safe baseline during a certified run;
- using e-process alarms as per-result proofs;
- hard-coding benchmark IDs or fixture shapes into selection logic.

## 29. First portfolio implementations

The implementation order is chosen to exercise the whole architecture:

1. integer polynomial multiplication: dense/sparse/modular representations;
2. univariate GCD over `ZZ`/`QQ`: subresultant versus modular;
3. univariate factorization: modular/Hensel/recombination with exact certificate;
4. exact dense linear solve: Bareiss versus modular reconstruction;
5. simplification: deterministic local rules plus bounded e-graph extraction;
6. differentiation: proof-producing structural engine plus sparse Jacobian generation;
7. real root isolation: exact algebraic plus certified numeric refinement;
8. Gröbner: Buchberger versus F4/modular with independent verifier;
9. integration: rule/rational/heuristic candidates with explicit evidence classes;
10. symbolic-to-numeric residual/Jacobian compilation into FrankenNumPy/FrankenSciPy.

The first “hero” workload combines several of these and is specified in `FIRST_IMPLEMENTATION_CAMPAIGN.md`.