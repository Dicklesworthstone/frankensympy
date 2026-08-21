# Claim and evidence lattice

**Status:** normative architecture contract  
**Scope:** claim classes, evidence classes, legal justification edges, typed authority, publication wording, contradictions, downgrades, and release enforcement

## 1. Law

Evidence may justify a claim only when the evidence class is registered as sufficient for that exact claim class and scope. Weaker evidence may influence policy, prioritization, or research, but it cannot enforce or justify a stronger claim.

## 2. Claim classes

### `Invariant`

Must hold for every execution within the named system/profile. A counterexample is a correctness defect.

### `FormalTheorem`

Proved within a named formal logic, imported environment, assumptions, and kernel.

### `ExactCertificateClaim`

Checked by a named native reference verifier over a complete immutable closure.

### `BoundedModelClaim`

Exhaustively checked only within declared dimensions, bounds, and state model.

### `CompatibilityClaim`

Observable parity over a named upstream/profile, observation schema, environment matrix, and corpus.

### `StatisticalClaim`

Confidence, coverage, risk, or sequential evidence statement under explicit assumptions.

### `ConfigurationModelClaim`

A deterministic statement about a registered configuration or policy model, not necessarily the live world.

### `SloClaim`

Empirical operational target over a named population/window.

### `BenchmarkClaim`

Measured performance on pinned code, corpus, machine, route, and method.

### `ResearchHypothesis`

Speculative mechanism or predicted effect with no release authority.

## 3. Evidence objects

Evidence is an immutable object with:

- evidence class and schema;
- claim IDs/scopes addressed;
- exact source/build/environment/corpus roots;
- assumptions and limitations;
- verifier/checker/tool identity;
- result and typed failures;
- provenance and parent evidence;
- expiration/review freshness where applicable;
- signature/provenance attachment where required.

A report file without these bindings is diagnostics, not registered evidence.

## 4. Non-transitivity by default

Justification is not assumed transitively across arbitrary classes. For example:

- a benchmark validates a benchmark claim, not an SLO or invariant;
- a statistical monitor may trigger fallback, not prove incompatibility;
- a formal theorem about a projected statement needs a projection receipt before it speaks to a native claim;
- an exact certificate validates one mathematical claim, not Python object parity;
- differential parity does not prove correctness where the oracle may share a bug;
- multiple weak observations do not become an invariant by accumulation.

Every composed edge is explicit.

## 5. Typed authority

Critical Rust APIs should use sealed marker types so that illegal evidence-to-claim edges fail at compile time where practical. Dynamic registries enforce the same law for artifacts and plugin/extensible surfaces.

Only verifier/publication modules can mint authority tokens such as:

- `VerifiedFactorization`;
- `VerifiedLinearSolve`;
- `VerifiedProjection`;
- `CompatibilityProfilePassed`;
- `ReleaseGatePassed`.

A generator, benchmark, planner, remote worker, database, signature verifier, or monitor cannot construct them.

## 6. Scope and assumptions

Evidence strength is inseparable from scope. A theorem or exact certificate checked under assumptions cannot justify an unconditional claim. A compatibility result on CPython X/macOS cannot justify all Python/platform profiles.

Publication renders assumptions and scope with the claim; they are not hidden metadata.

## 7. Contradictory evidence

New contradictory evidence does not overwrite old artifacts. It creates a contradiction event and may:

- quarantine a route;
- revoke a release/profile claim;
- invalidate a projection or benchmark;
- require re-verification;
- leave a formally scoped theorem intact while invalidating its projection receipt;
- create an unresolved discrepancy.

The authority state is derived from the append-only evidence graph and current policy root.

## 8. Downgrades

A requested evidence class cannot be silently downgraded. A certified request that exhausts verifier resources returns inconclusive/refused; it does not return an uncertified native answer labeled certified.

Optional weaker output can be requested explicitly through a separate policy.

## 9. Wording

Public statements include class and scope, for example:

- “verified factorization under certificate schema factorization_v1 over ZZ”;
- “Lean projection checked under profile lean_factorization_v1”;
- “matched SymPy profile X on corpus root Y and environment matrix Z”;
- “bounded model checked for term DAGs up to N nodes”;
- “anytime-valid e-value under registered null and filtration”;
- “measured 2.1× median speedup on machine class M and corpus C.”

Avoid unqualified “proved,” “compatible,” “safe,” “faster,” or “drop-in.”

## 10. Release enforcement

Registries, docs, README badges, package metadata, benchmark dashboards, and release notes are linted against available evidence. A status cannot advance because a prose file says it has.
