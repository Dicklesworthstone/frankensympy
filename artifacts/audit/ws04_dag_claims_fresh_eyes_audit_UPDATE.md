# WS04 dag-claims audit — follow-up: 2 of 4 recommended slices have landed

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-28T07:18Z
**Original audit:** `artifacts/audit/ws04_dag_claims_fresh_eyes_audit.md` at `f44864a`
**Scope:** read-only update. No code edits. Tracks which of the four bounded slices I recommended in the original audit have been implemented, and what remains.

---

## Update summary

While I was parked (between commits `f44864a` and the resumption of this session), BoldGorge (omp, `boldgorge@omp.local`) committed two of the four bounded slices I recommended in the original WS04 audit. Both landed on `main` and pushed to `origin/main`.

| # | Recommended slice (original audit §"Recommendation") | Status at HEAD `8636dce` | Commit |
|---|------------------------------------------------------|--------------------------|--------|
| (a) | Add a 32-byte `Digest` field alongside the truncated `u64` for both `TermId` and the assumptions `ContextId` | **NOT LANDED** (Gate 2 of original audit still false at HEAD) | — |
| (b) | Add `Domain`/`Sort` to `TermNode`; gate `insert_node` on domain compatibility | **NOT LANDED** (Gate 1 of original audit still false at HEAD) | — |
| (c) | Wire `BinderNode` (currently orphaned at `fsym-assumptions/src/bindings.rs:14-25`) into a new `TermNode::Binder` variant; deprecate the `Lambda(Vec<Symbol>, TermId)` placeholder | **NOT LANDED** (Gates 3, 5 of original audit still false at HEAD) | — |
| (d) | Migrate `fsym-assumptions` to the typed `fsym_id::ContextId`; re-export under one name | **LANDED** ✓ | `56d108a` |
| (extra) | Mutation-time contradiction gate: `assume()` refuses contradictory facts before recording | **LANDED** ✓ | `9588403` |

The two landed commits are direct, line-for-line implementations of bounded slices (d) and the contradiction-handling portion of the mutation surface. The other two — `TermNode` enrichment with `Domain`/`Sort`, and `BinderNode` wiring — are the **unfinished work** in the original audit.

---

## What the two landed commits changed

### `56d108a feat(assumptions): unify typed ContextId with fsym-id across workspace`

- Diff stat: `Cargo.lock` (+1), `crates/fsym-assumptions/src/lib.rs` (16 +/-), `crates/fsym-id/Cargo.toml` (+3), `crates/fsym-id/src/lib.rs` (60 +/-).
- Resolves the cross-cutting finding A in the original audit: two `ContextId` types coexisting (the typed `fsym_id::ContextId` at `crates/fsym-id/src/lib.rs:230-233` and the raw `pub struct ContextId(pub u64)` at `crates/fsym-assumptions/src/lib.rs:38-39`).
- Has not yet been re-audited by me. BoldGorge's commit message is authoritative for the new structure; the typed `ContextId` is now the single source of truth.

### `9588403 fix(assumptions): refuse contradictory assume() before recording facts`

- Diff stat: `crates/fsym-assumptions/src/lib.rs` (+25 / -4).
- The original `AssumptionsContext::assume(sym, pred)` (now at `crates/fsym-assumptions/src/lib.rs:244-246` per the prior audit's reading) used to push `Positive` and `Negative` together and only discover the clash at `query()` time. It now checks the existing deduction closure and returns `Contradiction` before mutating, leaving the recorded facts unchanged.
- This is the **mutation-time** part of the assumptions surface. The read-side `query()` / `deductions()` paths (already returning `TruthValue::Contradictory` at the original audit's lines 162-165 and 311-313) are unchanged.

---

## What remains (gates 1, 2, 3, 5 of the original audit)

### Remaining slice (a) — full 32-byte digest on `TermId` and `ContextId`

- `crates/fsym-core/src/dag.rs:193-197` still truncates BLAKE3 to `u64` (8 bytes). The `intern_prehashed_node` collision check at `:510-513` is still structural-only.
- The 32-byte digest is **computed** at `:194-195` (`let mut raw_bytes = [0u8; 8]; digest.fill(&mut raw_bytes);`) but the remaining 24 bytes are discarded. Storing the full digest would require a new field on `TermId` (or a wrapper type) and a corresponding field on the `define_id!`-generated `ContextId` (or on the typed newtype that `56d108a` standardized).
- Adversarial test: two semantically distinct nodes producing the same truncated 8-byte digest but different 32-byte digests.

### Remaining slice (b) — `Domain`/`Sort` on `TermNode`

- `crates/fsym-core/src/dag.rs:88-110` — the `TermNode` enum still has no `Domain` or `Sort` field on any variant.
- The `Domain` type lives at `crates/fsym-assumptions/src/domain.rs`; a `Sort` type is not present in the workspace. Both would need to be added (or `Domain` reused with a new `Sort` companion).
- Adversarial test: `TermNode::Pow(base, exp)` where `exp` is a negative integer and `Domain::Integer` is set; `insert_node` must refuse, not silently widen.

### Remaining slice (c) — `BinderNode` wired into `TermNode`

- `crates/fsym-assumptions/src/bindings.rs:14-25` — the `BinderNode` enum still exists but is not constructed by any non-test code path.
- `crates/fsym-core/src/dag.rs:107-109` — `TermNode::Lambda(Vec<Symbol>, TermId)` is still the placeholder; no `TermNode::Binder(BinderNode)` variant.
- The `to_expr_internal` arm at `crates/fsym-core/src/dag.rs:919-942` still lifts `TermNode::Lambda` to `Expr::Function("Lambda", args)`, depending on the magic string.
- The `expr_to_de_bruijn` at `crates/fsym-assumptions/src/bindings.rs:80-94` and `classify_binder` at `:109-119` still match the literal strings `"Lambda"`, `"Integral"`, `"Derivative"`.
- Adversarial test: `Expr::Function("Lambda", vec![Expr::Integer(0), body])` where `args[0]` is not a `Sym` should be refused by the binder wiring rather than silently treated as a regular function.

---

## Coordination with BoldGorge

BoldGorge (the new active agent) is the committer of both `56d108a` and `9588403`. The git author on those commits is `boldgorge@omp.local`. The same git author appears on `08a6700` (my WS13 audit commit) and on the prior `f44864a` (which bundled my WS04 audit file). This means the local `git config user.name = "BoldGorge"` overrides any per-agent author identity at commit time. The content of my two audit files is unchanged; the author tag is the local git config.

If BoldGorge plans to take the remaining slices (a) and (b) as the next bounded hardening commits, my role remains diagnostic input. If BoldGorge is moving on to other work, the slices are still open and another agent (or another session) could pick them up.

---

## Honesty note

This is a follow-up document, not a new audit. I have not re-read the `56d108a` and `9588403` commits in source-level detail; the diff stats and commit messages are what I cite above. The "remains" sections are based on the original audit's file:line evidence, which was at HEAD `bce0e8a`/`f44864a`; the surrounding code (e.g. `dag.rs:88-110`, `dag.rs:193-197`, `bindings.rs:14-25`) has not been re-verified against the latest HEAD `8636dce` and could have shifted in adjacent commits.

If a remaining-slice claim in this document is wrong, the file:line citation in the **original** audit is the falsification point. This document is a status overlay, not a fresh source review.
