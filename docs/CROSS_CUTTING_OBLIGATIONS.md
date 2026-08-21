# Cross-cutting architecture obligations

**Status:** normative architecture contract  
**Scope:** composition of verifier, artifact, workspace, graph, formal, compatibility, Python, packaging, monitoring, performance, dependency, and release contracts

## 1. Problem

A large architecture can contain individually excellent subsystem documents whose composition is contradictory or incomplete. Cross-cutting obligations are the machine-readable joins between those contracts.

Every obligation names:

- one normative document;
- an owner workstream label;
- prerequisite obligations;
- required local gates;
- required evidence artifacts;
- release-blocking status;
- planning/implementation status.

The dependency graph must be acyclic. A release cannot be promoted until every release-blocking obligation selected by its profile has landed evidence.

## 2. Composition laws

### 2.1 Verifier reserve law

Algorithm portfolios, runtime budgets, transfer scheduling, and Python orchestration cannot consume the protected reference-verifier and cleanup reserve.

### 2.2 Authority law

Only registered verifiers and publication transitions mint authority. Databases, graph indexes, remote workers, formal projectors, signatures, monitors, and benchmarks cannot.

### 2.3 Identity law

Term, claim, certificate, environment, and artifact identities are storage-, transport-, runtime-, and Python-wrapper-independent. Optional physical views do not alter semantic identity.

### 2.4 Effect law

Speculation and remote execution are restricted to closed pure work. Unknown/effectful Python runs exactly once on one selected route.

### 2.5 Evidence law

Selection, calibration, differential parity, benchmark, and formal projection evidence retain exact class and scope; none silently upgrades another.

### 2.6 Negative-result law

Absence, no-path, no-proof, or no-applicable-rule claims require complete authoritative closure. Derived index misses are `NotObserved` unless completeness is certified.

### 2.7 Cancellation law

All owned tasks and affine obligations are requested to cancel, drained, and finalized. Arbitrary Python callbacks are outside universal prompt-cancellation claims.

### 2.8 Release law

Canonical local gate receipts, not hosted CI status, control release transitions. Documentation and package claims cannot exceed evidence.

## 3. Gate ownership

A gate should be owned by a component different from the implementation it validates where practical:

- reference verifier versus generator;
- dependency checker versus crate author;
- conformance harness versus Python shell route;
- formal foreign checker versus projector;
- release manifest validator versus packager;
- graph certificate verifier versus optimized graph algorithm;
- benchmark harness versus optimized route.

Independence is a gradient, but self-attestation is never presented as independent verification.

## 4. Artifact closure

Required artifacts are immutable and content-addressed. Gate logs may be regenerated, but the canonical report binds to source, toolchain, corpus, profile, and result roots.

A missing required artifact leaves an obligation incomplete even if prose says the gate passed.

## 5. Change impact

Changing a normative document or registry invalidates directly dependent obligation evidence unless the artifact explicitly proves it is unaffected. The obligation DAG provides the conservative impact cone.

Examples:

- verifier schema change invalidates formal profiles, artifact capsules, certified portfolios, and release receipts;
- Python effect-class change invalidates portfolio eligibility and compatibility evidence;
- toolchain/dependency policy change invalidates build and release provenance;
- tie-break policy change invalidates deterministic graph and result receipts;
- packaging profile change invalidates resolver and ecosystem certification.

## 6. Validator

`tools/verify_cross_cutting.py` checks:

- registry/schema shape;
- unique IDs and documents;
- referenced files exist;
- dependencies are known and acyclic;
- gate and artifact lists are nonempty;
- status values are allowed;
- release-blocking obligations are not marked complete without gates/artifacts;
- deterministic topological output;
- optional JSON audit emission.

The validator proves registry coherence, not that the underlying implementation or evidence is correct.

## 7. Release behavior

A release profile selects obligations and minimum statuses. Failed or absent obligations block promotion. Research obligations may remain incomplete only when no promoted claim depends on them.
