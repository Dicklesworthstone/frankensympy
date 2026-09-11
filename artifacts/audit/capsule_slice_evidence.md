# Polynomial claim capsule evidence (`fra-rc-capsule-39v`)

Implementation evidence for the verifier-complete ZZ/QQ polynomial capsule and
its minimal offline consumer. The companion independent gate
`fra-rc-capsule-gate-5mq` requires a reviewer other than the implementation
author; this file records what exists and what was observed so that review can
be adversarial rather than trust-based.

## What exists now

- `crates/fsym-proof-kernel/src/capsule.rs` — the capsule format and verifier.
  Claim: `subject == coefficient * product(factor_i ^ exponent_i)` over ZZ or
  QQ. Deliberately **not** claimed: irreducibility, completeness of the factor
  list, uniqueness, roots. Multiplying the factors back is an identity check,
  not a factorization proof.
  - canonical, self-describing, length-framed byte format with a fixed schema
    version and hard caps on capsule bytes, object count/size, terms, factors,
    and symbol length — all checked before allocation;
  - object ids are content-derived and confirmed against the full digest of
    their own payload at the trust boundary;
  - a bounded, complete resolver: every referenced object must be present, and
    there is no generator, planner, network, filesystem, or ambient-time hook;
  - typed claim roots: domain, assumptions context, rule root, verifier root,
    subject object id, coefficient, factor list;
  - outcomes are `Verified`, `Refuted`, `Inconclusive` (fuel exhaustion), or
    `Refused` (malformed/missing/duplicate/inconsistent). Exhaustion is never an
    acceptance and never a rejection.
- `crates/fsym-capsule-consumer/` — the minimal external consumer. Its library
  closure is `fsym-proof-kernel` only; `std` is an optional feature (verification
  itself works without it), and it exposes only verdict classes plus a
  deterministic diagnostic.
- `crates/fsym-proof-kernel/tests/capsule_gate.rs` and
  `crates/fsym-capsule-consumer/tests/consumer_offline.rs` — the adversarial
  suite below. Every capsule in these tests is written by hand from exact
  coefficients: nothing calls a factorization generator, so a verifier that
  trusted generator output or a stored flag cannot pass.

## Observed evidence

Dependency closure of the consumer (`cargo tree -p fsym-capsule-consumer`):

```
fsym-capsule-consumer v0.1.0
└── fsym-proof-kernel v0.1.0
    ├── blake3, fsym-assumptions, fsym-budget, fsym-core, fsym-id,
    │   fsym-outcome, num-traits, serde, serde_json, thiserror
```

Forbidden-closure scan over the same tree (generator, calculus, solvers,
runtime, Python, matrices, simplify, ntheory, async runtime): **0 matches**.

Portability probes:

- `cargo check -p fsym-capsule-consumer --no-default-features` — builds.
- `cargo check -p fsym-capsule-consumer --target wasm32-unknown-unknown
  --no-default-features` — builds (compile-only probe; no WebAssembly runtime is
  executed here, so no execution claim is made).
- Verdict parity: the same four consumer tests (valid capsule, wrong claim
  root, exhaustion, malformed input) pass byte-identically with default
  features and with `--no-default-features`.

Adversarial cases (all present, all passing):

| Case | Expected outcome | Why it matters |
|---|---|---|
| valid capsule + matching claim root | `Verified` | positive observable |
| same capsule + a different claim root | `Refused(ClaimRootMismatch)` | a capsule cannot redefine the statement it supports |
| dropped factor and its object (incomplete factor list) | `Refuted` | product no longer equals the subject |
| removed object, reference left dangling | `Refused(MissingObject)` | no generator/network fallback exists |
| rational object inside a ZZ capsule | `Refused(Inconsistent)` | declared domain is enforced, not assumed |
| different assumptions context root | claim digest changes; verification of the old bytes under the new root is `Refused` | context is part of claim identity |
| duplicate object id with different bytes | `Refused(ObjectDigestMismatch/DuplicateObjectId)` | id reuse cannot smuggle different content |
| truncated buffer, trailing bytes, oversized declared length, wrong schema | `Refused(Malformed/UnknownSchema)` | bounded fail-closed decode |
| fuel below the required multiplications | `Inconclusive`, never `Verified` | exhaustion is inconclusive |
| extra trailing claim byte (a smuggled `verified` flag) | `Refused(Malformed)` | no stored flag is authoritative |

## Known limits (not claimed)

- Only the initial slice: univariate polynomials over ZZ/QQ with an identity
  claim. Multivariate objects, factor certificates for irreducibility,
  whole-capsule Merkle roots, object stores, the CLI/Python/ATP consumers, and
  the publication journal are **not** implemented here.
- The consumer profile targeted is the hosted/offline shape (std available).
  The `no_std` + alloc profile (`portable-no-std-alloc-v1`) is not claimed; the
  wasm probe is a compile check with the `std` feature off, not an execution
  result.
- No verifier-profile registry status is changed by this slice: gates such as
  `bounded_decoder_fuzz`, `mutation_suite`, and `external_minimal_consumer`
  remain open, and the profile stays `planned`.
- Nothing here promotes a claim in `registries/claims.toml`.
