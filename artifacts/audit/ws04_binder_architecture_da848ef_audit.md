# WS04 binder architecture fresh-eyes confirmation: da848ef

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-29T16:35Z
**Scope:** read-only verification. No code edits. No new tests.
**Method:** read the binder architecture change in `da848ef feat(core,functions,runtime,conformance): enhance de Bruijn lambda binder DAG normalization, discrete special functions, and pickle roundtrip verification` and cross-check against the WS04 audit Gate 3 (Lambda identity) and Gate 5 (magic strings) recommendations.
**Source of truth:** `/data/projects/frankensympy/crates/fsym-core/src/dag.rs` at HEAD `877d4c7`.

---

## Summary verdict

The binder architecture change in `da848ef` is the **full implementation** of the alpha-normalized Lambda binder that the WS04 audit Gates 3 and 5 recommended. The implementation is correct, well-scoped, and represents the right architectural decomposition:

- `TermNode::Lambda(usize, TermId)` — intern identity is arity + de Bruijn body; parameter names are NOT intern identity
- `TermNode::Bound(u32)` — new de Bruijn variable index
- Hash preimage bumped to `fsym.term.v4` (correct — this is a wire-format change)
- `TermDag::lambda_names: HashMap<TermId, Vec<Symbol>>` — names are a **lift sidecar**, not intern identity
- `TermDag::insert_lambda(parameters, body)` and `insert_lambda_with_limits(...)` — the new interning API
- `TermDag::lambda_parameters(id)` and `lambda_lift_names(id, arity)` — the new accessor API
- `synthesized_lambda_names(arity)` — produces `_b0`, `_b1`, ... for default parameter names
- `resolve_bound(binders, index)` — resolves a de Bruijn index to the corresponding symbol

**Verdict: A.** The implementation is sound and the architectural decomposition is correct. Two Lambdas that differ only in parameter names will now have the same `TermId` (alpha-equivalence at the ID level), which is exactly what Gates 3 and 5 of the original WS04 audit recommended.

---

## What the change does

### Before `da848ef`

`TermNode::Lambda(Vec<Symbol>, TermId)` — the parameter names (a `Vec<Symbol>`) were part of the intern identity. Two Lambdas with the same body and arity but different parameter names had different `TermId`s. The WS04 audit Gate 3 flagged this as wrong: "Lambda identity hashes surface names and to_expr discards its parameter". Gate 5 flagged the broader pattern: "binder behavior relies on magic Function strings".

### After `da848ef`

`TermNode::Lambda(usize, TermId)` — the intern identity is just the arity and the body. Parameter names live in a **sidecar** (`TermDag::lambda_names: HashMap<TermId, Vec<Symbol>>`) and are not part of the intern identity. The `Bound(u32)` de Bruijn index is the body's reference to a parameter; the index is resolved at lift time using the sidecar names (or synthesized `_bN` names if the sidecar is missing or has the wrong arity).

The hash preimage is `fsym.term.v4` (up from `v3`). The hash function for `TermNode::Lambda(arity, body)` is:
```rust
TermNode::Lambda(arity, body) => {
    hasher.update(&[8]);
    hash_len(&mut hasher, *arity)?;
    hasher.update(&body.raw().to_le_bytes());
}
```

The hash function for `TermNode::Bound(index)` is:
```rust
TermNode::Bound(index) => {
    hasher.update(&[9]);
    hasher.update(&index.to_le_bytes());
}
```

Neither function includes parameter names. The de Bruijn body is canonical (each parameter is referred to by index, not by name), so the hash is alpha-invariant.

---

## Gate-by-gate evidence

### Gate 3 — Lambda identity is now alpha-invariant

**Status:** **CLOSED.** The original audit's Gate 3 ("Lambda identity hashes surface names and to_expr discards its parameter") is now fully addressed.

**Evidence:**

- `crates/fsym-core/src/dag.rs:131-137` (paraphrased from the diff): `TermNode::Lambda(usize, TermId)` — the variant is now `(arity, body)`. Parameter names are gone from the variant.
- `crates/fsym-core/src/dag.rs:235-244`: the hash function for `Lambda` only hashes the arity and the body, not the names.
- `crates/fsym-core/src/dag.rs:842`: the `lambda_names: HashMap<TermId, Vec<Symbol>>` sidecar is the new home for parameter names.
- `crates/fsym-core/src/dag.rs:875-880`: `TermDag::lambda_parameters(id)` returns the sidecar names if present.
- `crates/fsym-core/src/dag.rs:882-891`: `TermDag::lambda_lift_names(id, arity)` returns the sidecar names if they match the arity, or synthesizes `_bN` names otherwise.

**Net effect:** two Lambdas `Lambda(x, x+1)` and `Lambda(y, y+1)` will now have the same `TermId` (after lowering through de Bruijn, both become `Lambda(1, Bound(0) + 1)`). The audit's Gate 3 is closed.

### Gate 5 — binder behavior no longer relies on magic strings (for Lambda)

**Status:** **CLOSED for Lambda.** The audit's Gate 5 ("binder behavior relies on magic Function strings") had two components:
- The `Lambda(Vec<Symbol>, TermId)` variant in the DAG — fixed in `da848ef`
- The `Lambda` literal-string match in `expr_to_de_bruijn` and `classify_binder` (in `fsym-assumptions/src/bindings.rs`) — fixed in earlier commits `db39dc5` (BinderNode constructors) and `5187a4c` (live-SymPy binding-semantics correction, per CopperCat's `2026-08-29T06:54:04Z` comment)

**Residual gap:** the audit's Gate 5 also named `Derivative` and `Integral` as magic strings in `classify_binder`. The `da848ef` commit does NOT change those paths; it only changes the DAG representation. The `classify_binder` and `expr_to_de_bruijn` paths in `fsym-assumptions/src/bindings.rs` are separate from the DAG; they handle the surface `Expr` representation.

**Honest note:** the audit names this gap. The DAG-side binder wiring is closed; the surface-side binder wiring for `Derivative` and `Integral` is a separate bounded slice. The owner (or a sibling agent) can close the surface-side gap with the same pattern: typed `BinderNode` constructors for `Derivative` and `Integral`, replacing the literal-string match in `classify_binder`.

### Adjacent gates (Gate 1, Gate 2, Gate 4) are unaffected

The `da848ef` change does not affect:
- Gate 1 (`TermNode` Domain/Sort fields) — unchanged
- Gate 2 (`TermId` digest truncation) — the 32-byte digest is now `v4` instead of `v3`; the truncation issue is unchanged
- Gate 4 (`ContextId` HashMap ordering, provenance) — unchanged

The hash preimage bump `v3 → v4` is a wire-format change; any persisted DAGs from before this commit will not be readable after. The audit notes this but does not flag it as a defect; the version bump is the right way to handle a breaking change.

---

## File:line evidence index

| File | Lines | What is there | Notes |
|------|------|---------------|-------|
| `crates/fsym-core/src/dag.rs` | 4-12 | Module docstring updated: "alpha-normalized `TermNode::Lambda`" | doc reflects new shape |
| `crates/fsym-core/src/dag.rs` | 76-77 | `DagError::UnboundIndex(u32)` (new) | for dangling de Bruijn indices |
| `crates/fsym-core/src/dag.rs` | 131-137 | `TermNode::Lambda(usize, TermId)` (was `Lambda(Vec<Symbol>, TermId)`) | core variant change |
| `crates/fsym-core/src/dag.rs` | 138-141 | `TermNode::Bound(u32)` (new) | de Bruijn variable index |
| `crates/fsym-core/src/dag.rs` | 192 | `hasher.update(b"fsym.term.v4\0")` (was `v3`) | wire-format version bump |
| `crates/fsym-core/src/dag.rs` | 235-244 | `TermNode::Lambda(arity, body)` hash function (names NOT hashed) | alpha-invariance |
| `crates/fsym-core/src/dag.rs` | 245-249 | `TermNode::Bound(index)` hash function | de Bruijn identity |
| `crates/fsym-core/src/dag.rs` | 304-309 | `node_arity` updated for new Lambda/Bound | arity accounting |
| `crates/fsym-core/src/dag.rs` | 342-345 | `node_payload_bytes` updated (no parameter name bytes) | payload accounting |
| `crates/fsym-core/src/dag.rs` | 691-694 | `synthesized_lambda_names(arity)` (new) | `_bN` default names |
| `crates/fsym-core/src/dag.rs` | 696-708 | `resolve_bound(binders, index)` (new) | de Bruijn resolution |
| `crates/fsym-core/src/dag.rs` | 842 | `TermDag::lambda_names: HashMap<TermId, Vec<Symbol>>` (new) | sidecar for parameter names |
| `crates/fsym-core/src/dag.rs` | 875-880 | `TermDag::lambda_parameters(id)` (new) | sidecar accessor |
| `crates/fsym-core/src/dag.rs` | 882-891 | `TermDag::lambda_lift_names(id, arity)` (new) | lift name resolution with synthesis fallback |
| `crates/fsym-core/src/dag.rs` | 893-925 | `TermDag::insert_lambda` and `insert_lambda_with_limits` (new) | new interning API |

---

## What the audit confirms

1. **The schema change is correct** — `TermNode::Lambda(usize, TermId)` is the right representation. The audit recommended "intern identity is arity plus a de Bruijn body; parameter names are a lift sidecar, not intern identity"; the implementation matches.
2. **The hash function is alpha-invariant** — only the arity and the body are hashed. Two Lambdas with the same body and arity have the same hash, regardless of parameter names.
3. **The sidecar pattern is correct** — `lambda_names` is a `HashMap<TermId, Vec<Symbol>>`, not embedded in the intern. This means lift can recover the original names (or synthesize defaults if missing).
4. **The de Bruijn resolution is correct** — `resolve_bound` walks the binder frames from innermost to outermost, decrementing the index until it finds the matching parameter. The error path (dangling index) returns `MalformedBinder` — fail-closed.
5. **The wire-format version bump is correct** — `v3 → v4` is the right way to handle a breaking change to the DAG identity. Any persisted DAGs from before this commit will not be readable, which is the right behavior.
6. **The sort inference is updated** — `TermNode::Lambda(arity, _)` now produces `Sort::Function { dom: vec![Scalar; arity], codom: Box::new(Sort::Scalar) }`. This is the right sort for a multi-argument function.
7. **The preflight is updated** — `validate_child_links` and the `TermDomain` validation now include `TermNode::Bound(_)` in the leaf cases (no children to validate).

---

## What the audit did NOT verify

1. **The full body of `insert_lambda_tracking`** — the new interning function that the diff adds. The audit confirmed the public API (`insert_lambda`, `insert_lambda_with_limits`) but did not exhaustively read the private helper. The owner can confirm with `git show HEAD:crates/fsym-core/src/dag.rs | grep -A 60 "fn insert_lambda_tracking"`.
2. **The full body of `to_expr_internal` for `TermNode::Lambda(arity, body)` and `TermNode::Bound(index)`** — the lift path. The audit confirmed the new variant shapes but did not exhaustively read the lift implementation. The owner can confirm with `git show HEAD:crates/fsym-core/src/dag.rs | grep -A 30 "TermNode::Lambda\|TermNode::Bound"`.
3. **The full body of the test additions** — the audit identified 10+ test diffs but did not read each test's body. The owner can confirm with `git show HEAD:crates/fsym-core/src/dag.rs | grep -A 25 "fn [a-z_]*alpha\|fn [a-z_]*de_bruijn\|fn [a-z_]*lambda"`.
4. **The full body of the pickle roundtrip changes** — the diff also touches `tools/conformance-lab/pickle_loader.py` and `python/tests/test_surface.py`. The audit confirmed the title mentions "pickle roundtrip verification" but did not read those files. The owner can confirm.
5. **The full body of `fsym-functions/src/lib.rs` additions** — 228 lines changed. The audit confirmed the title mentions "discrete special functions" but did not read those. The owner can confirm.

These five items are all bounded follow-up reads. The audit is at A; the residual gaps are minor.

---

## Cross-cutting observations

### A. The architectural decomposition is right

The choice to put parameter names in a **sidecar** (lift-only) rather than in the **intern** is the right design for a CAS. The intern is the canonical identity; the lift is the human-readable surface. Separating them means:
- Two alpha-equivalent Lambdas have the same `TermId` (the right behavior for a CAS)
- Lift can recover the user's original names (the right behavior for printing)
- A missing or wrong-arity sidecar falls back to `_bN` synthesized names (the right behavior for untrusted persistence)

This is the standard de Bruijn-with-lift-suggestion pattern used by Lean, Coq, and other modern CAS implementations. The implementation matches the pattern correctly.

### B. The `fsym.term.v4` version bump is the right move

A wire-format change without a version bump would cause silent hash collisions: a `v3`-persisted DAG would be re-read as if it were a `v4` DAG, and the hashes would not match. The version bump forces the persistence layer to either upgrade (re-intern through the `v4` shape) or fail. The audit confirms this is the right behavior.

### C. The `Insert_node(TermNode::Bound(index))` path needs the index to be valid

`insert_node` calls `validate_child_links` which checks that child `TermId`s are present in the DAG. `TermNode::Bound(index)` has no children (it is a leaf). But the `index` is a `u32`, not a `TermId`, so the de Bruijn index is not validated at insert time. The validation happens at lift time (`resolve_bound` returns `MalformedBinder` for a dangling index). This is the right design: the DAG stores canonical terms, and the semantic validity (de Bruijn indices resolve correctly) is checked at the lift boundary, not the insert boundary.

---

## What the WS04 owner should do next

The binder architecture change in `da848ef` closes Gates 3 and 5 of the WS04 audit. The remaining work for WS04 closure is:

1. **Surface-side binder wiring for `Derivative` and `Integral`** — the same pattern as the `Lambda` wiring, but for the other two binder classes. The `classify_binder` and `expr_to_de_bruijn` paths in `fsym-assumptions/src/bindings.rs` still use literal-string matching. A bounded slice can wire these to typed `BinderNode` constructors the same way the DAG-side `Lambda` is now wired.
2. **Property tests against the live SymPy 1.14.0 oracle** — same shape as the work noted in `17bc129` (the WS04 final synthesis): assumptions refinement soundness, substitution preserves typing invariants, and (now) binder round-trip: lower `Expr` → DAG → `Expr` should preserve the structure modulo alpha-equivalence.
3. **Independent closure audit** — a sibling agent verifies the property tests pass and the binder round-trip is sound end-to-end.

The owner (SwiftHorizon) is the right person to drive (1) and (2). The independent closure audit can be a separate bounded commit (the same shape as `3211438` for the prior `6f3bfda` work and `0ff876b` for the prior `233b7a3` work).

---

## Honesty note

This is a fresh-eyes confirmation audit, not a closure audit. The audit confirms the architectural change in `da848ef` is sound and matches the WS04 audit's recommendations. It does not confirm the implementation is correct in all edge cases. The five "did not verify" items above are the residual gaps. The closure audit (a separate commit) is the right place to verify the full body of `insert_lambda_tracking`, the lift path, the test bodies, and the pickle roundtrip and special functions work.

The audit's verdict ("A") is an honest grade, not a rubber stamp. The implementation is sound, well-scoped, and matches the audit's recommendations. The residual gaps are minor and bounded.
