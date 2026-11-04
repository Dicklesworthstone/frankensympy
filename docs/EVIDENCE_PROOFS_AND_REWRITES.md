# FrankenSymPy evidence, proofs, certificates, and rewriting

**Status:** normative architecture contract  
**Scope:** result evidence classes, trusted kernel, certificate families, rewrite system, bounded e-graphs, side conditions, proof storage, verification and mutation testing

## 1. Principle

A symbolic system should not force users or agents to infer the strength of a result from which method returned it. FrankenSymPy makes the epistemic status of every native result explicit.

The system distinguishes:

- what was requested;
- what candidate was produced;
- what claim is being made about it;
- what assumptions and domains govern that claim;
- what evidence supports it;
- which verifier checked that evidence;
- what remains unknown or conditional.

Compatibility calls still return profile-correct ordinary SymPy values and exceptions. Native calls expose the richer envelope.

## 2. Result envelope

A native operation returns a terminal `MathOutcome`:

```text
MathOutcome<T>
├── Accepted
│   ├── value: T
│   ├── claim
│   ├── evidence
│   ├── verification_receipt
│   └── execution_receipt
├── Conditional
│   ├── value: T
│   ├── unresolved obligations
│   └── current evidence
├── HeuristicCandidate
│   ├── value: T
│   ├── diagnostics
│   └── verification failures/not attempted
├── Inconclusive
│   ├── explored methods
│   ├── remaining obligations
│   └── continuation, optional
├── Refused
│   ├── reason
│   └── remediation hints
├── Cancelled
├── TimedOut
├── ResourceExhausted
├── Unsupported
└── InternalFault
```

No path converts `Conditional`, `HeuristicCandidate`, or `Inconclusive` into `Accepted` merely because a compatibility API expects an expression. The shell may reproduce an upstream unevaluated result, but native diagnostics retain the true outcome.

## 3. Claim language

Evidence attaches to a typed claim rather than an unstructured string.

Initial relation kinds include:

- `StructuralIdentity(a, b)`;
- `Equality(a, b, context)`;
- `Inequality(a, relation, b, context)`;
- `Implication(p, q, context)`;
- `LogicalEquivalence(p, q, context)`;
- `Membership(x, set, context)`;
- `Subset(a, b, context)`;
- `PolynomialIdentity(p, q, ring)`;
- `Divides(a, b, domain)`;
- `Factorization(f, unit, factors, domain)`;
- `GroebnerBasis(generators, basis, order, domain)`;
- `MatrixEquation(A, X, B, domain)`;
- `RootSet(f, variables, solution_set, domain, multiplicity_policy)`;
- `IntegralAntiderivative(f, F, variable, domain, branch_policy)`;
- `DefiniteIntegral(f, region, value, context)`;
- `Limit(f, point, direction, value, context)`;
- `SeriesExpansion(f, point, expansion, order, context)`;
- `Encloses(term, ball, context)`;
- `ApproximationError(approximation, target, bound, norm)`;
- `CompatibilityObservation(fixture, profile, observation)`.

Claim schemas are versioned and extensible. Unknown relation kinds fail closed in verifiers.

## 4. Evidence classes

Evidence classes are not a single total ranking because different claims require different trust models. The initial classes are:

### 4.1 `KernelProved`

A proof term is checked by the small trusted logical/algebraic kernel. This is the strongest general exact class.

Requirements:

- complete proof object or a replayable compressed proof accepted by the kernel;
- all side conditions discharged under the named context;
- no unchecked foreign theorem step;
- kernel and rule-universe version recorded.

### 4.2 `CertificateVerified`

A specialized deterministic verifier checks a certificate for a typed claim. Examples include factorization, Gröbner, determinant, linear solve, root isolation, and modular reconstruction certificates.

The generator may be complex and heuristic. The verifier must be substantially smaller, independently testable, and unable to upgrade a different claim accidentally.

### 4.3 `ExactCrossChecked`

Two or more sufficiently independent exact computations agree and pass invariant checks, but no compact formal certificate family is available yet.

This class is useful during development and for some native APIs, but cannot be relabeled `KernelProved`. Independence assumptions and shared components are recorded.

### 4.4 `CertifiedNumeric`

Directed-rounding interval/ball arithmetic and algorithm-specific error analysis prove an enclosure or error bound. It supports numeric claims, separation predicates, and validated continuation, not unrestricted symbolic identity.

### 4.5 `OracleConformant`

A result matches the immutable compatibility oracle under the declared comparator. This establishes profile behavior for the fixture, not mathematical truth. An upstream bug can be reproduced compatibly.

### 4.6 `UserAsserted`

A user or imported context declares a fact. It may be used within that context according to policy, but its provenance remains visible and it is not a kernel theorem.

### 4.7 `HeuristicCandidate`

A search, model, numerical recognizer, simplifier, or external callback proposes a result without accepted verification. It may guide further work but cannot discharge exact proof obligations.

### 4.8 Terminal non-evidence outcomes

`Conditional`, `Inconclusive`, `Refused`, `Cancelled`, `TimedOut`, `ResourceExhausted`, `Unsupported`, and `InternalFault` are not evidence classes. They describe execution state and unresolved claims.

## 5. Evidence non-conversion rules

The following conversions are prohibited:

- e-process alarm or conformal confidence → mathematical proof;
- solver posterior → certificate;
- RaptorQ decode success → digest validity, authenticity, or mathematical correctness;
- oracle agreement → proof of mathematical truth;
- numerical agreement at many points → symbolic identity;
- successful differentiation of a proposed antiderivative at sampled points → exact integral proof;
- two algorithms sharing the same faulty kernel → independent cross-check;
- large proof search trace → verified proof;
- absence of a counterexample → theorem;
- timeout → false;
- unsupported → unknown mathematical truth.

Any explicit promotion between evidence classes names and verifies the additional artifact that justifies it.

## 6. Trusted computing base

The exact trusted base should be kept small and layered.

### 6.1 Core trusted components

- canonical term decoder and invariant checker;
- exact integer/rational arithmetic primitives used by verifiers;
- assumptions-context decoder and fact provenance checker;
- capture-avoiding substitution and binder logic;
- proof-term type checker;
- a minimal equality/congruence kernel;
- certificate dispatch with schema/version checks;
- directed-rounding primitives for certified numerics;
- digest/canonical-encoding validation.

### 6.2 Not automatically trusted

- simplification/search planners;
- e-graph saturation;
- algorithm selectors and learned cost models;
- polynomial factorization generators;
- integration heuristic engines;
- remote workers;
- persistent caches;
- graph indexes;
- Python callbacks;
- upstream SymPy;
- proof pretty-printers;
- RaptorQ repair logic beyond byte recovery;
- e-process/conformal monitors.

Complex generators may contain bugs without making accepted results unsound if their verifier boundary is correct.

## 7. Proof terms

The kernel proof language begins deliberately small:

```text
Proof
├── Refl(term)
├── Symm(proof)
├── Trans(proof_a, proof_b)
├── Congruence(operator, child_proofs)
├── Assumption(fact_id)
├── DefinitionalReduction(rule_id, inputs)
├── Rewrite(rule_id, substitution, side_condition_proofs)
├── ModusPonens(implication_proof, premise_proof)
├── EqualitySubstitution(equality_proof, context)
├── BinderRename(binding_map, capture_avoidance_proof)
├── DomainEmbedding(coercion_id, witness)
├── CertificateLemma(certificate_verification_id)
└── NumericSeparation(certified_enclosure_id)
```

New proof constructors require a verifier implementation, formal semantics in the registry, adversarial fixtures, and mutation tests.

## 8. Rewrite rules

A rewrite rule is a versioned object:

```text
RewriteRule
├── rule_id
├── lhs pattern
├── rhs template
├── variable declarations and sorts
├── match theory                    # syntactic, AC, binder-aware, domain-specific
├── side-condition obligations
├── domain/branch applicability
├── direction policy
├── termination/cost metadata
├── compatibility visibility
├── proof constructor
└── provenance and review state
```

Rules are never anonymous closures in the authoritative registry.

### 8.1 Rule classes

- definitional reductions;
- algebraic identities;
- conditional identities;
- canonicalization rules;
- expansion/factorization transforms;
- assumption refinements;
- branch-sensitive analytic rewrites;
- representation conversions;
- compatibility-only surface transforms;
- heuristic proposals requiring post-verification.

### 8.2 Side conditions

A rule application may require obligations such as:

- `x != 0`;
- `x > 0`;
- `n ∈ ZZ`;
- matrices have compatible dimensions;
- an operator is commutative;
- a polynomial is square-free;
- a path avoids branch cuts;
- a convergence condition holds;
- a denominator is nonzero in the relevant domain.

The rule engine asks the assumptions/proof subsystem. `Unknown` prevents unconditional application. Native mode may return a conditional result carrying the obligation; strict compatibility follows the profile's behavior.

### 8.3 Bounded Pythagorean verification

The kernel's `pythagorean_identity` reduction independently normalizes the
claimed expressions and consumes each input monomial at most once. Overlapping
candidate pairs cannot reuse a term, including across different arguments or
trigonometric and hyperbolic families. A valid single cancellation may leave
other terms intact; this bounded checker does not promise a complete
trigonometric normal form.

Before normalization, a bounded traversal records partial-expression domain
obligations from every raw function argument and power child. The two sides
must retain exactly the same undischarged obligations. Unary `sin`, `cos`,
`sinh`, `cosh`, and `exp` are total over finite complex arguments, but do not
erase their arguments' own obligations. Only direct finite nonzero literals
and the kernel's existing exact zero/one elementary evaluations discharge
additional cases. Nonfinite constants are refused at every depth.

Consequently, `log(x) + sin(x)**2 + cos(x)**2 = log(x) + 1` can be checked
while `0*log(0) = 0` and `sin(log(x))**2 + cos(log(x))**2 = 1` cannot discard
their source domains. Likewise, symbolic `sec(x)**2 - tan(x)**2 = 1` and
`csc(x)**2 - cot(x)**2 = 1` require pole conditions that this unconditional
rule does not discharge. The exact ground case `sec(0)**2 - tan(0)**2 = 1`
remains supported. Unverified simplification is a separate interface and does
not gain proof authority from proposing any of these expressions.

Direct-construction and independent-replay regressions are in
`crates/fsym-proof-kernel/tests/pythagorean_identity.rs`. This scope is not a
general definedness solver or a certification of every rewrite family.

## 9. Rewriting engine

### 9.1 Deterministic local rewriting

For compatibility-critical canonical construction and small simplifications, a deterministic ordered rule pipeline is used. The exact rule registry and order are profile-versioned.

Requirements:

- termination measure or explicit bounded iteration;
- no hidden global mutable rule order;
- each accepted step emits or can reconstruct a proof edge;
- expression-growth budgets charge before publication;
- shell-sensitive rewrites preserve or explicitly transform the SOG.

### 9.2 Goal-directed search

Native simplification takes an objective:

```text
SimplificationGoal
├── semantic equivalence relation
├── target domain/context
├── cost vector
├── forbidden constructs
├── preferred forms
├── proof/evidence minimum
├── resource budget
└── tie-break policy
```

Cost is multi-dimensional, for example:

- operation count by target backend;
- tree/DAG size;
- coefficient height;
- branch complexity;
- numerical stability estimate;
- code-generation latency;
- proof size;
- evaluation cost on a declared workload distribution;
- compatibility surface distance.

There is no context-free globally simplest expression.

## 10. Bounded e-graphs

Equality saturation is useful locally but dangerous as a universal architecture. FrankenSymPy uses bounded, typed e-graphs as search devices.

### 10.1 Boundaries

An e-graph request declares:

- input term region;
- domain and assumptions context;
- rule subset;
- node/e-class/memory/time/proof budgets;
- extraction objectives;
- maximum side-condition branching;
- deterministic tie breaks;
- checkpoint policy.

### 10.2 Proof extraction

Every union has a justification. Extraction returns a candidate term plus a proof path or certificate obligations. If a proof path cannot be reconstructed and verified, the candidate remains heuristic.

### 10.3 Conditional e-classes

Rules with unresolved side conditions do not merge unconditional e-classes. They create guarded relations keyed by obligation sets. Guard explosion is budgeted and pruned deterministically.

### 10.4 Anti-explosion controls

- typed/operator-specific rule admission;
- growth forecasting and hard caps;
- subsumption and dominance pruning;
- cost-bound pruning;
- repeated-pattern detection;
- per-rule firing quotas;
- proof-size budgets;
- resumable checkpoints;
- refusal when opaque nodes invalidate match theory.

No system-wide immortal e-graph serves as expression storage.

## 11. Certificate families

### 11.1 Polynomial identity

A certificate may consist of canonical-domain normalization or modular evaluations plus deterministic reconstruction and exact final subtraction. Acceptance ultimately proves the difference is zero in the declared ring.

### 11.2 Factorization

A factorization certificate contains:

- unit/content;
- ordered factors and multiplicities;
- domain and normalization;
- exact product check;
- optional irreducibility certificates per factor;
- square-free/content witnesses.

“Product matches” certifies a decomposition, not irreducibility or completeness unless those additional obligations are checked.

The implemented `RabinIrreducibilityModPrime` kernel family accepts only a
`Claim::IrreducibleModPrime` carrying the exact same ascending coefficient list
and prime as its certificate. Coefficients are canonical residues in `[0, p)`,
the leading coefficient is nonzero, and the synchronous reference lane admits
degrees 1 through 64 with a prime of at most 32 bits. It checks primality and
Rabin's criterion independently of polynomial generators. The resulting claim
is irreducibility of that polynomial in `F_p[x]`; it does not establish that an
expression is nonzero at an arbitrary argument, or any real/complex predicate.

The claim uses a new tagged wire variant; existing claim digests are unchanged.
Previously stored Rabin derivations using `NonZero`, `PredicateHold`, or equality
claims are rejected on replay even when their receipt digests match. Such
artifacts must be regenerated with the exact modular proposition; they are not
automatically upgraded. Claim and certificate decoders cap coefficient counts
and reject an oversized scalar before retaining another coefficient. Individual
integer decoding retains the foundational bounded BigInt decoder. Expression
result publication and residual compilation refuse this non-expression claim.
The focused regression corpus is `crates/fsym-proof-kernel/tests/rabin_irreducibility.rs`,
with trusted-namespace import/promotion adversaries in `fsym-evidence`.

### 11.3 GCD

Certificates include Bézout cofactors or divisibility plus degree/primitive conditions sufficient for the domain. Multivariate and non-field domains name their exact criterion.

### 11.4 Gröbner basis

A certificate includes:

- ring, variables, and monomial order;
- representation of each output basis element in the input ideal;
- reduction of required S-polynomials to zero or an equivalent accepted criterion;
- reduced/minimal normalization obligations when claimed.

The verifier need not repeat the generator's pair-selection strategy.

#### 11.4.1 Bounded zero-dimensional radicals over `QQ`

`fsym-polys::checked_zero_dimensional_radical` returns an immutable
`VerifiedZeroDimensionalRadical` for the exact original typed ideal
`I ⊆ QQ[x_1,...,x_n]`. The unit ideal is supported; positive-dimensional
ideals refuse. The candidate carries independently checked source and target
Gröbner bases plus one coordinate witness for every variable of a proper ideal.
The composition verifier requires:

- the source basis to retain every original input, with its coefficients,
  ordering and typed generator identities;
- the target basis to describe exactly `J = I + <h_1,...,h_n>`, without
  dropping original generators or inserting unrelated constraints;
- each `h_i` to be a monic nonconstant polynomial in only `x_i`;
- an exact coefficient identity `U_i*h_i + V_i*h_i' = 1`, establishing that
  the coordinate annihilator is square-free;
- an exact identity `h_i^k_i = sum(q_ij*G_j)` in the original checked basis,
  establishing nilpotent membership without trusting the generator's gcd or
  elimination search.

The target binding proves `I ⊆ J`; the power identities prove `J ⊆ radical(I)`.
Seidenberg's lemma makes `J` radical because every coordinate has a square-free
univariate annihilator. Consequently `J = radical(I)`, preserving the complete
complex point set. The primary reference is Lorenzo Robbiano,
[*Polynomial Systems and some Applications to Statistics*, Proposition 2.4
and Corollary 2.5](https://mate.dm.uba.ar/~visita16/cimpa/notes/Robbiano.pdf).
The implementation checks the Bézout identities directly from coefficient
indices and reconstructs powers by a dense reference convolution, separately
from the generator's sparse arithmetic and Euclidean search.

The envelope is 1–8 variables, at most 64 original or augmented generators,
4096 terms per polynomial and 4096 bits per coefficient component. Coordinate
degree and nilpotence exponent are at most 64; powered witness degree is at
most 256. Each whole phase retains the existing cumulative 20-million-compute
and 128-MiB logical-allocation cap across nested operations. Generator and
verifier meters remain separate. Exhaustion, cancellation and failed identities
refuse before publication; `Unbounded` callers do not remove these internal caps.

This is an in-process composition receipt over immutable accepted bases, with
no unchecked decoder or new portable proof-kernel claim. It preserves distinct
solution points, not the original scheme multiplicities. A solver using it must
retain the original-ideal link and independently check its coordinate recovery,
root coverage and requested real/complex domain. This bounded component does
not close the broader ideal-algebra, solver or compatibility-profile gates.

### 11.5 Exact linear algebra

Certificate examples:

- `A*X = B` for solves plus rank/uniqueness witnesses when claimed;
- `A = L*U`, `P*A = L*U`, or fraction-free decomposition checks;
- determinant certificates using modular residues and a valid magnitude/reconstruction bound;
- nullspace basis checks plus dimension/rank certificate;
- characteristic polynomial checks via accepted identities or modular reconstruction;
- eigenpair claims with exact residual and completeness obligations.

### 11.6 Root isolation and solving

Certificates distinguish:

- each reported root satisfies the equation;
- isolating regions are disjoint and each contains the claimed multiplicity;
- all roots in the declared domain/region are covered;
- excluded denominator/singularity cases are handled;
- extraneous roots introduced by transformations are removed.

A list of roots that substitute to zero is not automatically a complete solution set.

### 11.7 Differentiation

Elementary differentiation can be kernel-proved by structural rules. User function derivatives rely on declared hooks or remain formal `Derivative` objects. Branch/domain side conditions are preserved.

### 11.8 Integration

Evidence may include:

- derivative of proposed antiderivative equals the integrand under the named domain/context;
- branch and singularity partition;
- constants/distributional terms where relevant;
- endpoint/improper-limit certificates for definite integrals;
- contour/homology data for complex integration;
- certified numerical enclosure when no exact form is established.

Differentiating a candidate is necessary but may be insufficient for definite, piecewise, branch-sensitive, or distributional claims.

### 11.9 Limits and series

Certificates may use transformed asymptotic expressions, order bounds, punctured-neighborhood conditions, direction/sector data, remainder bounds, and certified enclosures. Formal power-series algebra does not by itself prove analytic convergence.

### 11.10 Sums, products, transforms, ODEs, and PDEs

Each family declares verification obligations separately:

- recurrence/telescoping and boundary terms for sums;
- convergence and interchange conditions;
- transform inversion/region-of-convergence data;
- substitution into differential equations plus initial/boundary conditions;
- completeness/general-solution claims distinguished from one verified solution;
- singular solutions and excluded parameter cases.

#### Bounded rational distributional inverse Laplace transforms

`fsym_calculus::rational_laplace::distributional` accepts an exact rational
source `A(s)/B(s)` in one typed transform variable and a distinct typed time
variable. Its convention is the unilateral transform with lower limit `0-`,
under which `Laplace(DiracDelta(t, k)) = s**k`. This agrees with the
[SymPy distributional Laplace convention](https://docs.sympy.org/latest/modules/integrals/integrals.html#sympy.integrals.laplace.laplace_transform).

The proposal contains a polynomial `P(s)` and the existing proper rational
inverse certificate for a regular part `R(s)/B(s)`. The reference checker
reconstructs `R` from the closed transforms of exponential, sine and cosine
blocks and checks the exact coefficient identity `P*B + R = A`. It never
uses the generator's division of the source numerator or its residue solver
to accept a result. Exact denominator quotients remain part of regular
transform reconstruction. Source exclusions are admitted before cancellation
and retained in the sufficient right half-plane bound.

The immutable receipt exposes two separate expressions: the regular function
and `sum(P[k]*DiracDelta(t, k))`. The Python surface attaches `Heaviside(t)`
only to the regular function; impulses must not acquire that multiplier.
Failure to invert any regular tail refuses the entire source, even if its
polynomial part was found. Input degree is at most 32, source coefficients
at most 4096 bits, and both generation and verification retain independent
cumulative resource limits and final cancellation checks.

This is a bounded in-process composition receipt, not a portable kernel claim
or a general distribution calculus. Delayed proper rational inverses continue
to use their existing separate certificate; delayed improper transforms are
outside this distributional slice. No compatibility profile is promoted by
these focused checks.

## 12. Modular and probabilistic techniques

Randomized or modular algorithms can be extraordinarily fast, but evidence must match the claim.

### 12.1 Las Vegas pattern

A randomized generator proposes a result; an exact verifier accepts or rejects it. Once verified, the result has the verifier's evidence class, independent of generator randomness.

### 12.2 Monte Carlo pattern

When only a probabilistic error bound is available, the result is explicitly probabilistic with its assumptions, seed, field/sample policy, and bound. It is not silently labeled exact.

### 12.3 Deterministic replay

Random choices use counter-based or explicitly recorded seeds. A replay bundle reproduces candidate generation, but replayability alone does not prove correctness.

## 13. Verification architecture

Verifiers are pure or effect-minimal components with explicit inputs:

```rust
verify(claim, certificate, context, registry_versions, budget)
    -> Verified | Rejected(witness) | Inconclusive | Refused
```

Requirements:

- no network or persistent-cache trust in the core verification path;
- deterministic output for fixed inputs and policy;
- bounded parsing before allocating based on certificate claims;
- exact schema/domain/claim matching;
- rejection witnesses where practical;
- independent implementation from the generator for high-value families;
- scalar/reference lane for optimized verifier kernels;
- no Python callbacks in `KernelProved`/core `CertificateVerified` paths unless their result is treated as an assumption.

## 14. Verification receipts

```text
VerificationReceipt
├── verification_id
├── claim digest
├── evidence class granted
├── verifier and version
├── certificate digest
├── term/domain/context/registry IDs
├── resource consumption
├── scalar/reference cross-check status
├── accepted side conditions
├── rejected/ignored fields
└── terminal status
```

Receipts are content-addressed and can be persisted or bundled. They are not accepted solely because their bytes exist; the verifier can replay them from canonical inputs.

## 15. Proof compression and replay

Large proofs may be stored as:

- DAG-shared proof terms;
- lemma references into immutable rule registries;
- certificate-backed macro steps;
- checkpoints with deterministic regeneration recipes;
- separately RaptorQ-protected archives.

Compression cannot remove information needed by the trusted verifier. A regeneration recipe that depends on a heuristic search is a replay aid, not a proof, unless the regenerated proof is checked.

## 16. Distributed proof search

Remote workers receive bounded work packets containing:

- immutable term/context/registry IDs and required objects;
- subgoal and allowed algorithms/rules;
- resource budget;
- expected certificate schema;
- deterministic seed range;
- capability limits.

Workers return candidates and artifacts. The coordinator verifies locally before publication. Worker identity or reputation never substitutes for verification.

Byzantine or buggy workers can waste resources but must not make a false exact result accepted, assuming the verifier boundary holds.

## 17. Evidence-aware caching

Cache entries contain:

- exact claim key;
- value/candidate;
- evidence class;
- certificate/proof/receipt references;
- all semantic universe IDs;
- verification status;
- expiration/invalidation policy.

Rules:

- unverified candidates live in a separate cache namespace;
- a stronger-evidence query cannot be satisfied by weaker evidence;
- context/profile changes invalidate dependent entries structurally;
- verifier-version changes trigger replay or invalidation;
- persistent caches validate canonical payload and digests before use;
- negative cache entries distinguish disproved, unsupported, and inconclusive.

## 18. E-process and conformal monitoring

Anytime-valid monitoring is used for streams such as:

- conformance mismatch rates;
- verifier rejection rates by generator and subgroup;
- unexpected certificate-size or runtime shifts;
- selector regret proxies;
- numerical enclosure failures;
- cache corruption/anomaly rates;
- proof-rule mutation survival;
- remote-worker defect rates.

Monitors may pause rollout, quarantine a cache/generator, increase shadow verification, or trigger investigation. They do not prove or disprove individual mathematical claims.

Every monitor states its exchangeability/adaptivity assumptions, test martingale/e-process construction, reset policy, subgroup policy, and action threshold. Optional stopping validity is not a license to ignore model misspecification.

## 19. Testing verifiers

### 19.1 Positive corpus

- independently generated valid certificates;
- boundary-size and degenerate cases;
- cross-language/cross-process serialization;
- certificates generated by every portfolio strategy;
- replay across supported architectures.

### 19.2 Negative and adversarial corpus

- one-bit and structured mutations;
- wrong domain/order/context/profile;
- omitted factors/roots/side conditions;
- duplicate or reordered payloads where order is semantic;
- overflow/allocation bombs;
- cyclic proof references;
- digest collision simulations at the validation layer;
- certificates that verify a weaker claim than advertised;
- branch/singularity omissions;
- malicious remote-worker bundles.

### 19.3 Mutation testing

Mutants deliberately weaken verifiers, for example:

- skip one product factor;
- omit a Gröbner S-pair;
- accept a nonzero residual under a tolerance;
- ignore multiplicity or completeness;
- treat unknown side condition as true;
- bypass directed rounding;
- trust a stored `verified=true` flag;
- accept a proof edge without checking registry version.

The suite must kill each registered mutant. Surviving mutants block proof-class claims.

### 19.4 Differential verifier implementations

For crown-jewel certificate families, maintain a slow, simple reference verifier and an optimized verifier. They run together in CI and shadow mode. Optimization requires bit/decision-identical behavior on the adversarial corpus.

## 20. Claim registry integration

Every public mathematical claim type maps to:

- minimum evidence class;
- accepted verifier families;
- mandatory side conditions;
- serialization schemas;
- mutation suites;
- release gates.

A README phrase such as “proof-producing factorization” is invalid until the claim registry resolves it to a live factorization certificate verifier and green artifacts on the same commit.

## 21. Forbidden shortcuts

- returning a candidate in the same native variant as a verified result;
- calling a planner trace a proof;
- trusting a generator's self-verification flag;
- allowing search completion order to choose an unverified winner;
- applying conditional rewrites as unconditional when side conditions are unknown;
- using sampled numeric agreement as exact identity;
- calling oracle parity mathematical verification;
- assuming modular reconstruction is valid without a bound/check;
- claiming complete roots after checking only reported roots;
- claiming irreducible factorization after checking only the product;
- accepting a definite integral after only differentiating an antiderivative;
- letting e-graph unions lose their justification;
- accepting a persistent proof receipt without replayable canonical inputs;
- using RaptorQ recovery as evidence that the recovered object is authentic or correct;
- weakening a verifier or comparator to land a feature;
- reporting proof coverage by raw proof-node count.

## 22. Initial certificate campaign

The first implementation campaign builds evidence in this order:

1. kernel structural equality, congruence, substitution, and basic algebra rules;
2. integer/rational normalization proofs;
3. polynomial identity and exact product certificates;
4. univariate gcd/factorization certificates over `ZZ`/`QQ`;
5. exact differentiation proof terms;
6. exact linear solve/determinant certificates for small dense systems;
7. real/complex ball enclosure verification;
8. Gröbner basis certificates;
9. root isolation/completeness certificates;
10. branch-aware calculus certificate families.

The first slice is successful only when at least one speculative portfolio produces a candidate, an independent verifier accepts it, the shell lifts it compatibly, and the complete derivation can be replayed after cancellation/checkpoint recovery.
