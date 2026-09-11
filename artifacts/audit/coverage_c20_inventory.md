# WS00 requirement-to-code-to-gate-to-task inventory (`fra-rc-coverage-c20`)

Generated from `registries/workstreams.toml` and `registries/claims.toml` at the commit that created
the bounded follow-on tasks. Requirement text is retained verbatim from the registries; nothing in this
inventory promotes a claim in `registries/claims.toml`.

Workstreams: 24. Claims: 27. Open beads considered: 53.

## Workstreams

| Workstream | Requirement (registry title) | Status | Milestone | Dependencies | Closure gate | Active bounded task |
|---|---|---|---|---|---|---|
| WS00 | Governance, registries, and claim discipline | planned | M0 | - | `gate://ws00-governance` | fra-rc-workflow-kzu, fra-rc-coverage-c20 |
| WS01 | Conformance laboratory foundation | in_progress | M2 | WS00 | `gate://ws01-conformance-lab` | fra-rc-corpus-gate-emj, fra-rc-corpus-us2 |
| WS02 | Foundation types, schemas, budgets, and Cx | planned | M1 | WS00 | `gate://ws02-foundation` | fra-ys1, fra-ys1 |
| WS03 | Exact arithmetic substrate | in_progress | M1 | WS02 | `gate://ws03-exact-arithmetic` | fra-62c, fra-62c |
| WS04 | Terms, domains, assumptions, and bindings | planned | M1 | WS02,WS03 | `gate://ws04-semantic-universe` | fra-rc-lowering-8w3, fra-rc-lowering-gate-ba5 |
| WS05 | Python compatibility shell vertical slice | planned | M2 | WS01,WS02,WS04 | `gate://ws05-python-object-model` | fra-rc-effects-dpc, fra-rc-surface-gate-0eh, fra-rc-surface-nvv |
| WS06 | Proof kernel and evidence system | in_progress | M1 | WS02,WS03,WS04 | `gate://ws06-proof-kernel` | fra-rc-capsule-gate-5mq, fra-rc-capsule-39v, fra-rc-formal-ma3 |
| WS07 | Verified rewriting and simplification | planned | M4 | WS04,WS06 | `gate://ws07-rewrite` | fra-7hp, fra-7hp |
| WS08 | Polynomial representations and arithmetic | planned | M3 | WS03,WS04,WS06 | `gate://ws08-polynomial` | fra-o4g, fra-o4g |
| WS09 | GCD, factorization, and certificates | planned | M3 | WS06,WS08,WS13 | `gate://ws09-factorization` | fra-ws09-factorization-116, fra-rc-factor-gate-w57, fra-rc-factor-lt4 |
| WS10 | Exact linear algebra | planned | M4 | WS03,WS04,WS06,WS08,WS13 | `gate://ws10-exact-linear` | fra-kfo, fra-kfo |
| WS11 | Certified numerics and algebraic numbers | planned | M4 | WS03,WS04,WS06 | `gate://ws11-certified-numeric` | fra-rc-numeric-gate-82w, fra-rc-numeric-t2y |
| WS12 | Differentiation and symbolic compilation | planned | M4 | WS04,WS05,WS06,WS07,WS08,WS10,WS11,WS13 | `gate://ws12-certified-jacobian` | fra-ws12-diff-compilation-y05, fra-rc-adapters-gate-wp4, fra-rc-adapters-iym |
| WS13 | Structured portfolios, cancellation, and replay | in_progress | M3 | WS02,WS04,WS06 | `gate://ws13-portfolio-runtime` | fra-ws13-portfolio-runtime-qup, fra-rc-portfolio-gate-4or, fra-rc-portfolio-9i7, fra-rc-monitor-gate-p1m, fra-rc-monitor-5ja |
| WS14 | Agent protocol and semantic workspaces | planned | M5 | WS00,WS02,WS04,WS06,WS13 | `gate://ws14-agent-protocol` | fra-rc-workspace-b0z, fra-rc-cli-gate-0jr, fra-rc-cli-u9o |
| WS15 | Persistence, checkpoints, and RaptorQ repair | planned | M5 | WS02,WS04,WS06,WS13 | `gate://ws15-persistence-repair` | fra-ws15-persistence-repair-0v8, fra-rc-durable-gate-hpt, fra-rc-durable-m5e |
| WS16 | Remote workers and graph indexing | planned | M5 | WS06,WS13,WS14,WS15 | `gate://ws16-distribution-index` | fra-ws16-distribution-index-grc, fra-rc-graph-gate-lkg, fra-rc-graph-zzo |
| WS17 | Groebner bases and ideal algebra | planned | M7 | WS06,WS08,WS10,WS13 | `gate://ws17-groebner` | fra-546, fra-546 |
| WS18 | Integration, limits, series, and transforms | planned | M7 | WS04,WS06,WS07,WS08,WS11,WS13,WS17 | `gate://ws18-analytic-calculus` | fra-a5x, fra-a5x |
| WS19 | Solvers, sets, logic, ODE, and PDE | planned | M7 | WS04,WS06,WS08,WS10,WS11,WS13,WS17,WS18 | `gate://ws19-solvers` | fra-7kf, fra-7kf |
| WS20 | Structured mathematics domains | planned | M7 | WS04,WS06,WS07,WS10,WS11,WS13,WS19 | `gate://ws20-structured-domains` | fra-zgf, fra-zgf |
| WS21 | Compatibility and ecosystem closure | planned | M7 | WS01,WS05,WS07,WS08,WS09,WS10,WS11,WS12,WS14,WS17,WS18,WS19,WS20 | `gate://ws21-profile-closure` | fra-ws21-profile-closure-82j |
| WS22 | Performance and architecture optimization | planned | M7 | WS03,WS04,WS05,WS06,WS08,WS09,WS10,WS11,WS12,WS13,WS17,WS18,WS19,WS20 | `gate://ws22-performance` | fra-rc-perf-gate-sth, fra-rc-perf-yub |
| WS23 | Packaging, release, and 1.0 certification | planned | M8 | WS00,WS15,WS16,WS21,WS22 | `gate://ws23-release` | fra-ws23-release-2b6, fra-rc-package-gate-gjd, fra-rc-package-mcj, fra-rc-release-gate-pal, fra-rc-release-3re |

## Claims

| Claim | Status | Minimum evidence | Gate | Workstream |
|---|---|---|---|---|
| PLAN-001 | documented | documentation_artifact | `gate://planning-package` | WS00 |
| SECURITY-001 | implemented_uncertified | source_and_dependency_audit | `gate://memory-safe-native-core` | WS00 |
| COMPAT-001 | planned | oracle_conformant | `gate://ws23-release` | WS01 |
| COMPAT-002 | planned | oracle_conformant | `gate://ws05-python-object-model` | WS01 |
| SEMANTIC-001 | implemented_uncertified | validated | `gate://deterministic-term-identity` | WS02 |
| RUNTIME-001 | implemented_uncertified | validated | `gate://no-orphan-work` | WS02 |
| RUNTIME-003 | implemented_uncertified | validated | `gate://deterministic-replay` | WS02 |
| SECURITY-001 | implemented_uncertified | source_and_dependency_audit | `gate://memory-safe-native-core` | WS02 |
| SECURITY-002 | implemented_uncertified | fuzz_and_adversarial_validation | `gate://bounded-decoders` | WS02 |
| PLATFORM-001 | planned | validated | `gate://wasm-subset` | WS02 |
| SEMANTIC-001 | implemented_uncertified | validated | `gate://deterministic-term-identity` | WS03 |
| MATH-003 | implemented_uncertified | certified_numeric | `gate://ws11-certified-numeric` | WS03 |
| SECURITY-001 | implemented_uncertified | source_and_dependency_audit | `gate://memory-safe-native-core` | WS03 |
| PLATFORM-001 | planned | validated | `gate://wasm-subset` | WS03 |
| SEMANTIC-001 | implemented_uncertified | validated | `gate://deterministic-term-identity` | WS04 |
| SEMANTIC-002 | implemented_uncertified | validated | `gate://three-graph-vertical-slice` | WS04 |
| MATH-003 | implemented_uncertified | certified_numeric | `gate://ws11-certified-numeric` | WS04 |
| SECURITY-002 | implemented_uncertified | fuzz_and_adversarial_validation | `gate://bounded-decoders` | WS04 |
| PLATFORM-001 | planned | validated | `gate://wasm-subset` | WS04 |
| COMPAT-001 | planned | oracle_conformant | `gate://ws23-release` | WS05 |
| COMPAT-002 | planned | oracle_conformant | `gate://ws05-python-object-model` | WS05 |
| COMPAT-003 | planned | oracle_conformant | `gate://ws05-python-object-model` | WS05 |
| COMPAT-004 | planned | release_inspection | `gate://no-oracle-runtime` | WS05 |
| SEMANTIC-002 | implemented_uncertified | validated | `gate://three-graph-vertical-slice` | WS05 |
| SEMANTIC-002 | implemented_uncertified | validated | `gate://three-graph-vertical-slice` | WS06 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS06 |
| MATH-002 | implemented_uncertified | certificate_verified | `gate://ws09-factorization` | WS06 |
| MATH-003 | implemented_uncertified | certified_numeric | `gate://ws11-certified-numeric` | WS06 |
| MATH-004 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://ws12-certified-jacobian` | WS06 |
| RUNTIME-002 | implemented_uncertified | validated | `gate://two-phase-verified-publication` | WS06 |
| DISTRIBUTION-001 | implemented_uncertified | validated | `gate://ws16-distribution-index` | WS06 |
| AGENT-002 | planned | validated | `gate://semantic-merge` | WS06 |
| SECURITY-002 | implemented_uncertified | fuzz_and_adversarial_validation | `gate://bounded-decoders` | WS06 |
| PLATFORM-001 | planned | validated | `gate://wasm-subset` | WS06 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS07 |
| MATH-004 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://ws12-certified-jacobian` | WS07 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS08 |
| MATH-002 | implemented_uncertified | certificate_verified | `gate://ws09-factorization` | WS08 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS09 |
| MATH-002 | implemented_uncertified | certificate_verified | `gate://ws09-factorization` | WS09 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS10 |
| MATH-004 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://ws12-certified-jacobian` | WS10 |
| MATH-003 | implemented_uncertified | certified_numeric | `gate://ws11-certified-numeric` | WS11 |
| MATH-004 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://ws12-certified-jacobian` | WS11 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS12 |
| MATH-004 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://ws12-certified-jacobian` | WS12 |
| INTEGRATION-003 | planned | validated | `gate://franken-numeric-adapters` | WS12 |
| RUNTIME-001 | implemented_uncertified | validated | `gate://no-orphan-work` | WS13 |
| RUNTIME-002 | implemented_uncertified | validated | `gate://two-phase-verified-publication` | WS13 |
| RUNTIME-003 | implemented_uncertified | validated | `gate://deterministic-replay` | WS13 |
| DISTRIBUTION-001 | implemented_uncertified | validated | `gate://ws16-distribution-index` | WS13 |
| MONITOR-001 | planned | monitor_validation | `gate://operational-monitoring` | WS13 |
| RUNTIME-003 | implemented_uncertified | validated | `gate://deterministic-replay` | WS14 |
| AGENT-001 | implemented_uncertified | validated | `gate://ws14-agent-protocol` | WS14 |
| AGENT-002 | planned | validated | `gate://semantic-merge` | WS14 |
| SECURITY-002 | implemented_uncertified | fuzz_and_adversarial_validation | `gate://bounded-decoders` | WS14 |
| RUNTIME-002 | implemented_uncertified | validated | `gate://two-phase-verified-publication` | WS15 |
| RUNTIME-003 | implemented_uncertified | validated | `gate://deterministic-replay` | WS15 |
| DURABILITY-001 | implemented_uncertified | repair_and_reverification_bundle | `gate://ws15-persistence-repair` | WS15 |
| INTEGRATION-001 | planned | validated | `gate://frankensqlite-adapter` | WS15 |
| SECURITY-002 | implemented_uncertified | fuzz_and_adversarial_validation | `gate://bounded-decoders` | WS15 |
| DISTRIBUTION-001 | implemented_uncertified | validated | `gate://ws16-distribution-index` | WS16 |
| AGENT-002 | planned | validated | `gate://semantic-merge` | WS16 |
| INTEGRATION-002 | planned | validated | `gate://frankengraphdb-adapter` | WS16 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS17 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS18 |
| MATH-001 | implemented_uncertified | kernel_proved_or_certificate_verified | `gate://proof-producing-transformations` | WS19 |
| COMPAT-001 | planned | oracle_conformant | `gate://ws23-release` | WS21 |
| COMPAT-002 | planned | oracle_conformant | `gate://ws05-python-object-model` | WS21 |
| COMPAT-003 | planned | oracle_conformant | `gate://ws05-python-object-model` | WS21 |
| COMPAT-004 | planned | release_inspection | `gate://no-oracle-runtime` | WS21 |
| MONITOR-001 | planned | monitor_validation | `gate://operational-monitoring` | WS21 |
| PERF-001 | implemented_uncertified | parity_gated_paired_benchmark | `gate://ws22-performance` | WS21 |
| RELEASE-001 | planned | certified_release_bundle | `gate://ws23-release` | WS21 |
| MONITOR-001 | planned | monitor_validation | `gate://operational-monitoring` | WS22 |
| PERF-001 | implemented_uncertified | parity_gated_paired_benchmark | `gate://ws22-performance` | WS22 |
| RELEASE-001 | planned | certified_release_bundle | `gate://ws23-release` | WS22 |
| COMPAT-001 | planned | oracle_conformant | `gate://ws23-release` | WS23 |
| COMPAT-004 | planned | release_inspection | `gate://no-oracle-runtime` | WS23 |
| SECURITY-001 | implemented_uncertified | source_and_dependency_audit | `gate://memory-safe-native-core` | WS23 |
| PLATFORM-001 | planned | validated | `gate://wasm-subset` | WS23 |
| PERF-001 | implemented_uncertified | parity_gated_paired_benchmark | `gate://ws22-performance` | WS23 |
| RELEASE-001 | planned | certified_release_bundle | `gate://ws23-release` | WS23 |

## Bounded follow-on tasks created by this inventory

Each carries the workstream requirement verbatim, owned files, exact dependencies, acceptance commands,
test obligations, forbidden shortcuts, and the rule that its companion gate needs a reviewer other than
the author. None closes a workstream closure gate.

- WS02 -> `fra-ys1`
- WS03 -> `fra-62c`
- WS07 -> `fra-7hp`
- WS08 -> `fra-o4g`
- WS10 -> `fra-kfo`
- WS17 -> `fra-546`
- WS18 -> `fra-a5x`
- WS19 -> `fra-7kf`
- WS20 -> `fra-zgf`

## Residual status and what is still unspecified

- WS20 currently carries no claim in the claims registry, so its slice is scoped to geometry/tensor invariants only.
- Every claim above remains `planned`, `documented`, or `implemented_uncertified`; no closure gate is claimed closed.
- The `implemented_uncertified` claims need their named gate bundles before any promotion; this inventory adds ownership, not evidence.
- The remaining unassigned independent gates (lowering `fra-rc-lowering-gate-ba5`, package `fra-rc-package-gate-gjd`, capsule, corpus, adapters, durable, portfolio, factor, perf, monitor, cli, graph, release) still need reviewers other than their authors.
- Independent review of this inventory is still required; the bead stays open until that review happens.
