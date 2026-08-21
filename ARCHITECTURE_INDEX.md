# FrankenSymPy architecture index

**Repository status:** planning and architecture  
**Current normative revision:** 2026-08-20  
**Implementation claims:** none unless backed by the claim/evidence registries and landed code

This file is the entry point for the FrankenSymPy design corpus. It prevents important contracts from becoming undiscoverable and defines precedence when older planning text conflicts with a later, narrower contract.

## Precedence

1. [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md)
2. [`COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md`](COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md)
3. [`docs/ARCHITECTURE_REVISION_2026_08_20.md`](docs/ARCHITECTURE_REVISION_2026_08_20.md), which narrows or corrects the initial plan
4. domain-specific normative contracts listed below
5. machine-readable registries
6. donor audits and research notes

A specific later contract overrides a broad earlier statement only for its declared scope. Machine-readable status never advances merely because prose uses future or present tense.

## Trusted acceptance boundary

- [Embeddable verifier contract](docs/EMBEDDABLE_VERIFIER_CONTRACT.md)
- [Portable claim, certificate, and artifact protocol](docs/PORTABLE_CLAIM_CERTIFICATE_AND_ARTIFACT_PROTOCOL.md)
- [Verifier profiles](registries/verifier_profiles.toml)
- [Claim and evidence lattice](docs/CLAIM_AND_EVIDENCE_LATTICE.md)
- [Claim lattice registry](registries/claim_lattice.toml)

The public compatibility promise is that a recipient can check a supported `Claim + Certificate + VerificationClosure` without linking the generator, planner, Python shell, runtime, database, network, or remote worker stack.

## Semantic universes and collaboration

- [Symbolic workspace MVCC and ledger](docs/SYMBOLIC_WORKSPACE_MVCC_AND_LEDGER.md)
- [Workspace transaction registry](registries/workspace_transactions.toml)
- [Deterministic graph substrate](docs/GRAPH_REASONING_SUBSTRATE.md)
- [Graph registry](registries/graph_reasoning.toml)
- [Cross-cutting obligations](docs/CROSS_CUTTING_OBLIGATIONS.md)
- [Obligation registry](registries/cross_cutting_obligations.toml)

Immutable terms and proof objects deduplicate by content identity. Mutable authority—profiles, contexts, registries, branch heads, accepted derivations, and releases—moves through versioned transactions and explicit publication.

## Algorithms, adaptation, and performance

- [Compatibility and algorithm portfolios](docs/COMPATIBILITY_AND_ALGORITHM_PORTFOLIOS.md)
- [Portfolio registry](registries/algorithm_portfolios.toml)
- [Adaptive selection and anytime monitoring](docs/ADAPTIVE_SELECTION_AND_ANYTIME_MONITORING.md)
- [Monitor registry](registries/monitor_profiles.toml)
- [Performance architecture and kernel program](docs/PERFORMANCE_ARCHITECTURE_AND_KERNEL_PROGRAM.md)
- [Kernel registry](registries/performance_kernels.toml)

Selection chooses what work to attempt. Verification decides what can be accepted. Statistics, benchmarks, votes, and decision receipts cannot upgrade evidence.

## Python and distribution compatibility

- [Python runtime and effect boundary](docs/PYTHON_RUNTIME_AND_EFFECT_BOUNDARY.md)
- [Python effect registry](registries/python_effects.toml)
- [Packaging and drop-in deployment](docs/PACKAGING_AND_DROPIN_DEPLOYMENT.md)
- [Packaging profiles](registries/packaging_profiles.toml)
- [Compatibility contract](docs/COMPATIBILITY_CONTRACT.md)
- [Compatibility profiles](registries/compatibility_profiles.toml)

Unknown or effectful Python callbacks execute exactly once. Import compatibility, behavioral compatibility, and distribution/resolver compatibility are separate claims.

## Safety, dependencies, and release

- [Dependency, safety, and toolchain constitution](docs/DEPENDENCY_SAFETY_AND_TOOLCHAIN.md)
- [Dependency registry](registries/dependencies.toml)
- [`rust-toolchain.toml`](rust-toolchain.toml)
- [Local-first release and validation](docs/LOCAL_FIRST_RELEASE_AND_VALIDATION.md)
- [Release gates](registries/release_gates.toml)

Project-authored authoritative Rust forbids unsafe code. The mathematical engine has no C/C++/Fortran algorithm backend. Canonical release authority is local/Doodlestein evidence, not hosted GitHub execution.

## Formal interoperability

- [Formal proof interoperability](docs/FORMAL_PROOF_INTEROPERABILITY.md)
- [Formal profiles](registries/formal_proof_profiles.toml)

Native verification is first-class and sufficient for native certificate families. Formal theorem-prover checks are optional, additive evidence connected through a checked projection receipt.

## Donor audits

- [asupersync](docs/DONOR_DEEP_DIVE_ASUPERSYNC.md)
- [FrankenSQLite](docs/DONOR_DEEP_DIVE_FRANKENSQLITE.md)
- [FrankenGraphDB and FrankenNetworkX](docs/DONOR_DEEP_DIVE_GRAPH_STACK.md)
- [FrankenLean](docs/DONOR_DEEP_DIVE_FRANKENLEAN.md)
- [FrankenNumPy and FrankenSciPy](docs/DONOR_DEEP_DIVE_NUMERIC_STACK.md)
- [Pinned donor registry](registries/donor_sources.toml)

Donor ideas are classified adopt/adapt/research/reject. Same ownership is not automatic dependency admission.

## Local validation

Current executable planning gates:

```bash
./scripts/check.sh all
./scripts/check.sh audit
```

The single local/Doodlestein entry point compiles the Python validators, syntax-checks itself, and validates cross-cutting obligations, document/registry closure, donor pins, dependency/safety policy, verifier profiles, and performance kernels. Implementation and release profiles that lack evidence exit with an explicit refusal rather than a false pass.

## Original normative corpus retained by this revision

- [Agent-native protocol](docs/AGENT_NATIVE_PROTOCOL.md)
- [Initial algorithm portfolios](docs/ALGORITHM_PORTFOLIOS.md)
- [Assumptions, domains, and numeric tower](docs/ASSUMPTIONS_DOMAINS_AND_NUMERIC_TOWER.md)
- [Conformance and benchmarking](docs/CONFORMANCE_AND_BENCHMARKING.md)
- [Crate architecture and dependencies](docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md)
- [Evidence, proofs, and rewrites](docs/EVIDENCE_PROOFS_AND_REWRITES.md)
- [First implementation campaign](docs/FIRST_IMPLEMENTATION_CAMPAIGN.md)
- [Object model and IR](docs/OBJECT_MODEL_AND_IR.md)
- [Persistence, distribution, and repair](docs/PERSISTENCE_DISTRIBUTION_AND_REPAIR.md)
- [Risk register and research agenda](docs/RISK_REGISTER_AND_RESEARCH_AGENDA.md)
- [Runtime budgets and determinism](docs/RUNTIME_BUDGETS_AND_DETERMINISM.md)
- [Security and resource governance](docs/SECURITY_AND_RESOURCE_GOVERNANCE.md)
- [Source-project audit](docs/SOURCE_PROJECT_AUDIT.md)
- [Workstream graph](docs/WORKSTREAM_GRAPH.md)

## Closure review

- [Architecture review — round 3](docs/ARCHITECTURE_REVIEW_ROUND_3.md)
- [Machine-readable review artifact](artifacts/audit/architecture_review_round_3.json)
