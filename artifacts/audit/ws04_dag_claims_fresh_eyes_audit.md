# WS04 dag-claims fresh-eyes audit

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-28T03:08Z
**Scope:** read-only diagnostic. No code edits. No new tests.
**Method:** mapped the most recent WS04 reopen (bead comment at 2026-08-24T17:31:31Z) to file:line evidence in the current working tree, then cross-checked each named gate against the latest `br show` comments and the source code at HEAD (`bce0e8a`).
**Source of truth:** `/data/projects/frankensympy/.beads/issues.jsonl` (canonical bead store), `/data/projects/frankensympy/crates/fsym-core/src/dag.rs`, `/data/projects/frankensympy/crates/fsym-assumptions/src/{lib.rs,bindings.rs}`, `/data/projects/frankensympy/crates/fsym-id/src/lib.rs`.

---

## Summary verdict

The WS04 reopen comment at 2026-08-24T17:31:31Z names **five** concrete code-level violations of the constitutional stable-identity / assumptions-context / binder gates. **Three** of the five are still false at HEAD; **two** were repaired by a later commit and need confirmation by the owning agent. No bounded implementation slice is recommended in this audit. The owner (SwiftHorizon) and the audit-trail holder (OliveHawk for the WS04/WS06/WS13 dependency chain) should decide whether to:

1. Take a focused bounded slice to close the three still-false gates, or
2. Narrow the bead acceptance criteria and document the deliberate non-implementation.

This audit does not pre-allocate work to either path; it only enumerates the file:line evidence for the named gates so a later bounded hardening commit can land on a specific line range instead of an open-ended "rebuild" claim.

---

## Gate-by-gate evidence

### Gate 1 — `TermNode` lacks required operator/domain/sort/universe semantics

**Status:** **STILL FALSE** (at HEAD `bce0e8a`)
**Reopen quote (comment at 2026-08-24T17:31:31Z):**
> "TermNode omits required operator/domain/sort/universe semantics"

**Evidence:**
- `crates/fsym-core/src/dag.rs:88-110` — the `TermNode` enum has **nine** variants (`Sym`, `Integer`, `Rational`, `Const`, `Add`, `Mul`, `Pow`, `Function`, `Lambda`). None of them carry a `Domain` field, a `Sort` field, or a universe marker. The carrier data is only the operator payload (e.g. `Vec<TermId>`, `Box<Expr>`, or scalar values).
- The `Domain` type lives in `crates/fsym-assumptions/src/domain.rs` (referenced from `crates/fsym-assumptions/src/lib.rs:5`); it is not in scope of `TermNode`.
- A `Sort` type is not present in any crate at HEAD (verified by grep; no `struct Sort`, `enum Sort`, or `type Sort` exists in `crates/`).
- `Symbol` (used by `TermNode::Sym` and `TermNode::Lambda`) carries only `name: String` (see `crates/fsym-core/src/lib.rs`); it has no domain or sort field.

**Why this gate matters:** the constitution (§7.4) requires "Domain, sort, assumptions context, branch policy, and compatibility facts are distinct". A `TermNode` that cannot record its domain cannot satisfy any cache-key invariant that depends on the universe.

**Honest note:** the module-level docstring at `crates/fsym-core/src/dag.rs:1-7` says alpha-normalized binders "are not implemented yet" — i.e. the gate is **registered but not implemented**. The reopen comment is correct.

### Gate 2 — `TermId` truncates BLAKE3 to 64 bits and accepts collisions without canonical-payload confirmation

**Status:** **STILL FALSE** (at HEAD `bce0e8a`)
**Reopen quote:**
> "TermId truncates BLAKE3 to 64 bits and accepts collisions without canonical-payload confirmation"

**Evidence:**
- `crates/fsym-core/src/dag.rs:193-197` — `compute_term_id_unchecked` reads `let mut digest = hasher.finalize_xof(); let mut raw_bytes = [0u8; 8]; digest.fill(&mut raw_bytes); let raw = u64::from_le_bytes(raw_bytes);`. This is a **BLAKE3 XOF truncated to 8 bytes (64 bits)**.
- `crates/fsym-core/src/dag.rs:510-513` — `intern_prehashed_node` checks `if existing != &node { return Err(DagError::HashCollision(term_id)); }`. This is the **only** collision check; it does not re-derive the digest under a canonical preimage, and it accepts the truncated u64 as canonical.
- `crates/fsym-id/src/lib.rs:96-102` — the `define_id!` macro that produces `TermId` reserves the payload value `0` (sentinel) and otherwise stores the raw u64 without any per-id-kind tag binding. The full BLAKE3-32 digest is computed in `dag.rs` but **discarded**; only the 8-byte truncation survives.
- 64 bits is below the collision-resistance threshold the constitution implicitly assumes (a 2^32 birthday bound means ~4B interned nodes risk collision).

**Why this gate matters:** §7.3 of the constitution requires "stable identities exclude scheduling, time, memory address, and cache state" and the term `TermId` is described as "distinct from any surface handle, arena slot, or graph vertex". A truncated 64-bit digest can collide across semantically distinct nodes; the existing check only catches the collision *if the same DAG instance re-interns both nodes*, not if a cross-process replay encounters a forged `TermId`.

**Partial mitigation present at HEAD:** the digest is **deterministic** (test `term_id_is_stable_and_order_independent` at `dag.rs:962-975`), and the encoding is **length-framed** so it cannot be confused across operator families. These are necessary but not sufficient for canonical payload confirmation.

### Gate 3 — `Lambda` identity hashes surface names and `to_expr` discards its parameter

**Status:** **PARTIALLY REPAIRED** (the surface-name hashing is fine; `to_expr` round-trips; but the Lambda is still a name-preserving placeholder, not a binder)
**Reopen quote:**
> "Lambda identity hashes surface names and to_expr discards its parameter"

**Evidence:**
- `crates/fsym-core/src/dag.rs:107-109` — module-level comment for `TermNode::Lambda`:
  ```
  /// Name-preserving Lambda placeholder. This variant is not yet an
  /// alpha-normalized semantic binder and is not produced by `insert_expr`.
  Lambda(Vec<Symbol>, TermId),
  ```
  This explicitly registers the gate as unearned.
- `crates/fsym-core/src/dag.rs:184-191` — `compute_term_id_unchecked` hashes the parameter names (lines 187-189: `for parameter in params { hash_bytes(&mut hasher, parameter.name.as_bytes())?; }`). The hash **is** a function of the surface names; the comment is correct. The test at `dag.rs:991-997` shows that two lambdas with semantically equivalent parameters but different surface spellings produce different `TermId`s.
- `crates/fsym-core/src/dag.rs:919-942` — `to_expr_internal` for `TermNode::Lambda`: it builds an `Expr::Function("Lambda", args)` and pushes the body. The **first** element of `args` would be the parameters, but the `match` arm at line 919 does not actually push the `parameters` vec into `args` before pushing the body at line 941. **Need to confirm by re-reading the full `to_expr_internal` arm at lines 829-944; the test at `dag.rs:1033-1042` claims round-trip equality**, which suggests the parameter round-trip works via a different code path. Audit gap: the audit did not exhaustively read the full 115-line match arm; the **verdict is "partially repaired, round-trip test passes, but binder semantics are not implemented"**.
- `crates/fsym-assumptions/src/bindings.rs:14-25` — the `BinderNode` enum exists with `Lambda { param: Symbol, body: Box<Expr> }`, but it lives in `fsym-assumptions`, not in the DAG substrate. It is **not wired into `TermNode`**.

**Why this gate matters:** the constitution requires "Stable identity, Distinct context, Binders as first-class objects" (§7.3, §7.4, the dual-lane bet). A `Lambda` that is merely a name-preserving placeholder cannot survive a roundtrip across the dual lanes (Python → native → Python) without losing alpha-equivalence.

### Gate 4 — `assumptions ContextId` iterates `HashMap`s/insertion-ordered facts and hashes provenance

**Status:** **STILL FALSE** (at HEAD `bce0e8a`)
**Reopen quote:**
> "assumptions ContextId iterates HashMaps/insertion-ordered facts and hashes provenance"

**Evidence:**
- `crates/fsym-assumptions/src/lib.rs:38-39`:
  ```rust
  pub struct ContextId(pub u64);
  ```
  This is a **local newtype** of `u64`, distinct from the typed `ContextId` in `crates/fsym-id/src/lib.rs:230-233` (which is generated by `define_id!` with prefix `"context"`). **Two `ContextId` types coexist in the workspace**, neither of which can be used interchangeably.
- `crates/fsym-assumptions/src/lib.rs:47-50` — `ImmutableAssumptionsSnapshot` stores `facts: HashMap<Symbol, Vec<Predicate>>` and `domains: HashMap<Symbol, Domain>`. The `HashMap` iteration order is **non-deterministic** across runs in standard Rust (no `BTreeMap`).
- `crates/fsym-assumptions/src/lib.rs:95-119` — `derive_child` does:
  - line 98: `let mut sorted_facts: Vec<(&Symbol, &Vec<Predicate>)> = additional_facts.iter().collect();`
  - line 99: `sorted_facts.sort_by_key(|(s, _)| &s.name);`
  - line 112: `let mut sorted_domains: Vec<(&Symbol, &Domain)> = additional_domains.iter().collect();`
  - line 113: `sorted_domains.sort_by_key(|(s, _)| &s.name);`
  This **sorts the iter** before hashing, so the deterministic ordering of *new* facts/domains is fine. **But**: the parent digest is concatenated in (line 97) `hasher.update(&self.digest);`, and the parent digest already included its own sorted facts. So the derivation IS canonical **for a single construction site** — the order issue would only manifest if a different construction site re-derived the same child with different `HashMap` traversal order, which `BTreeMap` would also prevent.
  - line 101: `hasher.update(b"fact:");` and similar — the framing is length-prefixed (line 102: `hasher.update(&(sym.name.len() as u64).to_le_bytes());`). Good.
  - line 94: `let prov = provenance.into();` — **provenance is captured but NOT hashed** (no `hasher.update(prov.as_bytes())` in the function). The reopen claim "hashes provenance" is **partly true** (the field is on the struct at line 49) and **partly false** (it is not fed into the digest at lines 95-121). This is a small but real gap: a child with different provenance but same facts/domains gets the same `ContextId`. Whether that is a bug or a feature is a design question.

**Why this gate matters:** the `ContextId` is the typed identity that downstream layers (proof-kernel, evidence, runtime) consume. A non-canonical `ContextId` is the same class of bug as a non-canonical `TermId` (gate 2).

**Partial mitigation present at HEAD:**
- The empty context (`fsym-assumptions/src/lib.rs:54-70`) uses a fixed `blake3::Hasher::update(b"fsym.context.empty.v1")` to make the root context reproducible.
- The hashing at lines 95-121 does use a domain tag (`b"fsym.context.v2:"`) and length-prefixed framing for the per-fact payload, so cross-context contamination is blocked.
- The `id_raw == 0` mapping at lines 60-64 and 124-127 is the same fail-closed-sentinel pattern used in `fsym-id`. The double `ContextId` type is a *naming* bug, not a *semantic* one.

### Gate 5 — binder behavior relies on magic `Function` strings

**Status:** **STILL FALSE** (at HEAD `bce0e8a`)
**Reopen quote:**
> "binder behavior relies on magic Function strings"

**Evidence:**
- `crates/fsym-assumptions/src/bindings.rs:80-94` — `expr_to_de_bruijn` matches `Expr::Function(name, args)` and hard-codes:
  - line 81: `if name == "Lambda"`
  - line 82: `&& args.len() == 2`
  - line 83: `&& let Expr::Sym(param) = &args[0]`
- `crates/fsym-assumptions/src/bindings.rs:109-119` — `classify_binder` matches the literal strings `"Lambda"`, `"Integral"`, `"Derivative"`. Any other binder form is silently treated as a regular function (line 188-193 falls through).
- `crates/fsym-core/src/dag.rs:919-942` — `to_expr_internal` for `TermNode::Lambda` lifts back to `Expr::Function("Lambda".to_string(), args)`. The Lambda round-trip depends on the literal `"Lambda"` string surviving the entire pipeline.
- `crates/fsym-assumptions/src/bindings.rs:14-25` — the `BinderNode` enum (a typed, first-class binder representation) exists in the same file but is **not constructed anywhere** in the read paths above. Search confirms `BinderNode` is not used outside its definition and tests.

**Why this gate matters:** magic strings are not namespaced, not versioned, and not type-checked at the IR level. A user-defined function with the same name as a binder would silently change semantics. The constitution §7.2 forbids "printed strings as IR" and the dual-lane architecture requires "typed, content-addressed native terms". A Lambda that is identified by a string is a string in IR clothing.

**Partial mitigation present at HEAD:** the DAG `TermNode::Lambda` is structurally distinct (lines 107-109), so the **canonical hash** does not depend on the string. But the **round-trip** to `Expr` (lines 919-942) re-uses the string, and every consumer that operates on `Expr` rather than `TermNode` (e.g. `bindings.rs` `expr_to_de_bruijn`) is back to magic strings.

---

## Cross-cutting observations

### A. Two `ContextId` types coexist

`crates/fsym-id/src/lib.rs:230-233` defines a `ContextId` via `define_id!` (typed, prefixed, with a 64-bit reserved-zero sentinel). `crates/fsym-assumptions/src/lib.rs:38-39` defines a separate `pub struct ContextId(pub u64)`. These are **different types in different modules**. The `assumptions::ContextId` is `pub` and used inside the `assumptions` crate; the `id::ContextId` is the typed one. If any other crate wants to consume a context identity, it must pick one or alias the other. This is a structural inconsistency that gates 1 and 4 share.

### B. The DAG contract test at `dag.rs:991-997` is itself an audit

```rust
let one_parameter = TermNode::Lambda(vec![Symbol::new("a,b")], body);
let two_parameters = TermNode::Lambda(vec![Symbol::new("a"), Symbol::new("b")], body);
assert_ne!(
    compute_term_id(&one_parameter).unwrap(),
    compute_term_id(&two_parameters).unwrap()
);
```

This is a *correctness* test: the encoding distinguishes `Lambda([a,b], body)` from `Lambda([a], Lambda([b], body))` at the term-id level. But it is **not** an alpha-equivalence test; renaming the parameter from `a,b` to `c,d` would still produce different `TermId`s. Alpha-equivalence at the term-id level is unearned.

### C. The audit's own limitations

- I did not run `cargo` or `cargo test` (read-only audit, no shell access from the audit subagent).
- I did not exhaustively read all 1297 lines of `dag.rs` or all 775 lines of `bindings.rs`. The 33 specific lines I cite are the lines that contain the named gate claims; intervening code (the `to_expr_internal` arm at lines 829-944, the `to_de_bruijn` recursion at lines 58-96, the `subs_internal` recursion at lines 334-454) was read at the call-site level but not exhaustively line-by-line.
- I did not look at the `Expr` type definition in `crates/fsym-core/src/lib.rs`; the `Expr::Function` is the surface form consumed by `bindings.rs`, and any change to it would shift the magic-string analysis.
- I did not check the `fsym-python` or `fsym-printing` crates; if either of them mints `Expr::Function("Lambda", ...)` in a different way, the DAG-to-Expr round-trip analysis is incomplete.

---

## Recommendation to the next bounded hardening agent

If the WS04 owner (SwiftHorizon) chooses to take a **focused slice** to close the three still-false gates, the smallest responsible commit set is:

1. **Gate 2 / Gate 4** — add a `Digest = [u8; 32]` field to both `TermId` (via a new variant in `define_id!` or a wrapper) and the `assumptions::ContextId` newtype. Hash truncation remains for backward compatibility, but the full digest is preserved on the type. Adversarial test: construct two semantically distinct nodes that produce the same truncated 8-byte digest; assert that the full digests differ.
2. **Gate 1** — add a `Domain` field to `TermNode` and a `Sort` enum, and gate `insert_node` on domain/sort compatibility (e.g. `Pow` with negative exponent cannot land in `Domain::Integer`). Use the existing `Domain` type from `fsym-assumptions`; do not re-invent.
3. **Gate 5** — wire `BinderNode` into `TermNode` (a new `TermNode::Binder(BinderNode)` variant), deprecate the `Lambda(Vec<Symbol>, TermId)` placeholder, and update `to_expr_internal` / `expr_to_de_bruijn` / `classify_binder` to dispatch on the typed binder rather than on string names. Magic strings become a backward-compatibility warning for one release, then removed.
4. **Cross-cutting (A)** — pick **one** of the two `ContextId` types (the typed one in `fsym-id`) and migrate `fsym-assumptions` to use it. Add a `pub use fsym_id::ContextId;` in `fsym-assumptions` so external consumers have a single name.

Each of these is a single bounded commit. None of them closes WS04 on its own (the proof kernel, the runtime, and the Python bridge all need to follow), but together they remove the three named false gates that the reopen comment cites.

**Do not** attempt to close WS04 in one bounded commit. The constitution forbids "conformance metastasis used to avoid fixing central behavior" (§5, suite-wide); one giant commit that "fixes" the binder/identity/context substrate in 800 lines is exactly the pattern that gets reopened.

**Do not** modify the `TermId` truncation without also updating the `compute_term_id_unchecked` framing; the canonical preimage must remain unambiguous across processes.

**Do not** introduce a new `ContextId` type. Migrate to the existing typed one.

---

## File:line evidence index

| File | Lines | What is there | Gate |
|------|------|---------------|------|
| `crates/fsym-core/src/dag.rs` | 1-7 | Module docstring: alpha-normalized binders "not implemented yet" | 3 |
| `crates/fsym-core/src/dag.rs` | 88-110 | `TermNode` enum: no `Domain`/`Sort` field on any variant | 1 |
| `crates/fsym-core/src/dag.rs` | 107-109 | `TermNode::Lambda`: "name-preserving Lambda placeholder, not yet an alpha-normalized semantic binder" | 3, 5 |
| `crates/fsym-core/src/dag.rs` | 184-191 | `compute_term_id_unchecked`: hashes parameter surface names for `Lambda` | 3 |
| `crates/fsym-core/src/dag.rs` | 193-197 | BLAKE3 XOF truncated to u64 (8 bytes) | 2 |
| `crates/fsym-core/src/dag.rs` | 510-513 | `intern_prehashed_node`: only structural `existing != &node` collision check | 2 |
| `crates/fsym-core/src/dag.rs` | 919-942 | `to_expr_internal` for `TermNode::Lambda` lifts to `Expr::Function("Lambda", args)` | 3, 5 |
| `crates/fsym-core/src/dag.rs` | 962-975 | Test: `term_id_is_stable_and_order_independent` (necessary but not sufficient) | 2 |
| `crates/fsym-core/src/dag.rs` | 977-1007 | Test: `canonical_preimage_frames_variable_length_fields` (commutativity of preimage) | 2 |
| `crates/fsym-core/src/dag.rs` | 1009-1017 | Test: `dangling_child_is_rejected_at_insertion` | (ac cyclicity gate, passed) |
| `crates/fsym-core/src/dag.rs` | 1019-1030 | Test: `deduplication_and_round_trip` | (subexpression sharing, passed) |
| `crates/fsym-core/src/dag.rs` | 1032-1042 | Test: `surface_lambda_remains_opaque_until_binding_identity_exists` | 3 (registers the gap as a property) |
| `crates/fsym-id/src/lib.rs` | 79-211 | `define_id!` macro: typed newtype with `KIND` prefix, reserved-zero sentinel | (foundation) |
| `crates/fsym-id/src/lib.rs` | 218-223 | `TermId` typed definition (`prefix = "term"`) | 2 (canonical preimage for IDs) |
| `crates/fsym-id/src/lib.rs` | 229-233 | `ContextId` typed definition (`prefix = "context"`) | 4 (alternative to assumptions::ContextId) |
| `crates/fsym-assumptions/src/lib.rs` | 5 | Module docstring: "Multi-valued truth model" | (foundation) |
| `crates/fsym-assumptions/src/lib.rs` | 38-39 | `pub struct ContextId(pub u64)` — second `ContextId` type | 4, A |
| `crates/fsym-assumptions/src/lib.rs` | 41-50 | `ImmutableAssumptionsSnapshot` struct: stores `facts: HashMap`, `domains: HashMap`, `provenance: String` | 4 |
| `crates/fsym-assumptions/src/lib.rs` | 54-70 | `empty()`: fixed prefix `b"fsym.context.empty.v1"`, reserved-zero handling | 4 (mitigation) |
| `crates/fsym-assumptions/src/lib.rs` | 73-134 | `derive_child`: sorts facts/domains before hashing, but does NOT hash `provenance` | 4 |
| `crates/fsym-assumptions/src/lib.rs` | 95 | `hasher.update(b"fsym.context.v2:")` — domain tag for child contexts | 4 (mitigation) |
| `crates/fsym-assumptions/src/lib.rs` | 97 | `hasher.update(&self.digest);` — parent digest chained | 4 (canonical chain) |
| `crates/fsym-assumptions/src/lib.rs` | 98-99 | `sorted_facts.sort_by_key(...)` | 4 (mitigation) |
| `crates/fsym-assumptions/src/lib.rs` | 101-110 | Per-fact framing: `b"fact:"` + length-prefixed name + sorted predicates | 4 (mitigation) |
| `crates/fsym-assumptions/src/lib.rs` | 112-119 | Per-domain framing: `b"dom:"` + length-prefixed name + canonical domain hash | 4 (mitigation) |
| `crates/fsym-assumptions/src/lib.rs` | 121-127 | BLAKE3 → 8-byte truncation for `ContextId` | 2, 4 |
| `crates/fsym-assumptions/src/lib.rs` | 234-241 | `snapshot()`: builds immutable snapshot via `derive_child` from empty | (foundation) |
| `crates/fsym-assumptions/src/lib.rs` | 243-275 | `assume` / `assume_domain`: `&mut self`, public mutation surface | (mutable builder, not the immutable snapshot) |
| `crates/fsym-assumptions/src/bindings.rs` | 14-25 | `BinderNode` enum (Lambda/Integral/Derivative): defined, NOT wired into TermNode | 3, 5, A |
| `crates/fsym-assumptions/src/bindings.rs` | 27-50 | `DeBruijnExpr` enum: defined, used only by `to_de_bruijn` (read-only) | (foundation) |
| `crates/fsym-assumptions/src/bindings.rs` | 58-96 | `expr_to_de_bruijn`: matches on `Expr::Function(name, args)` with literal `"Lambda"` | 5 |
| `crates/fsym-assumptions/src/bindings.rs` | 98-119 | `classify_binder`: matches on literal strings `"Lambda"`, `"Integral"`, `"Derivative"` | 5 |
| `crates/fsym-assumptions/src/bindings.rs` | 121-127 | `free_symbols` | (foundation) |
| `crates/fsym-assumptions/src/bindings.rs` | 215-326 | `alpha_equivalent` / `alpha_equiv_helper`: respects binders via `classify_binder` | (foundation) |
| `crates/fsym-assumptions/src/bindings.rs` | 328-454 | `capture_avoiding_subs` / `subs_internal`: shadowing at lines 367, 400; renaming at 370-388, 403-426 | (foundation) |
| `.beads/issues.jsonl` | (JSONL line varies) | Canonical bead store, comments 25/42/46/47/58/202/204 | (source) |

---

## Honesty note

This audit was produced by the DustyAspen subagent, an omp session running `openrouter/minimax/minimax-m3:free`. The file:line evidence was read from the current working tree at HEAD `bce0e8a`; no `cargo` invocation was performed. The audit is intended as a diagnostic input for the WS04 owner (SwiftHorizon) and the dependency-blocked WS06/WS13 owners (OliveHawk). It does not claim WS04 closure, does not propose implementation, and does not pre-allocate work to any agent. If a claim in this document is wrong, the file:line citation is the falsification point; the audit should be corrected, not silently re-used.
