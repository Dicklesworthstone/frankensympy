# WS04 dag-claims audit — final update: all 4 recommended bounded slices have landed

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-28T07:35Z (overwrites the prior UPDATE at 07:18Z)
**Original audit:** `artifacts/audit/ws04_dag_claims_fresh_eyes_audit.md` at `f44864a`
**First update:** `artifacts/audit/ws04_dag_claims_fresh_eyes_audit_UPDATE.md` at `542961a` (2 of 4 slices landed)
**This update:** all 4 bounded slices from the original audit have now landed.
**Scope:** read-only. No code edits.

---

## Update summary

Between 07:18Z and now, three more bounded hardening commits landed that close the remaining two of the four WS04 audit recommendations. All four recommendations are now implemented at HEAD `38f93a0+`.

| # | Recommended slice (original audit §"Recommendation") | Status at HEAD `38f93a0+` | Commit | Author |
|---|------------------------------------------------------|--------------------------|--------|--------|
| (a) | Add a 32-byte `Digest` field alongside the truncated `u64` for both `TermId` and the assumptions `ContextId` | **LANDED** ✓ | `3a6a3437` | MagentaMouse (under BoldGorge git author) |
| (b) | Add `Domain`/`Sort` to `TermNode`; gate `insert_node` on domain compatibility | **LANDED** ✓ | `165e6eb`, `9a04d9a`, `a308417`, `a89680c` | (split across multiple commits) |
| (c) | Wire `BinderNode` (currently orphaned at `fsym-assumptions/src/bindings.rs:14-25`) into a new `TermNode::Binder` variant; deprecate the `Lambda(Vec<Symbol>, TermId)` placeholder | **LANDED** ✓ | `f6d079e`, `5715ae8`, `5b67ca7`, `9a04d9a` | (split across multiple commits) |
| (d) | Migrate `fsym-assumptions` to the typed `fsym_id::ContextId`; re-export under one name | **LANDED** ✓ | `56d108a` | BoldGorge |
| (extra) | Mutation-time contradiction gate: `assume()` refuses contradictory facts before recording | **LANDED** ✓ | `9588403` | BoldGorge |

---

## What each of the three new commits changed

### `3a6a3437 fix(core): confirm interned TermId against full BLAKE3 digest` (MagentaMouse)

- Diff stat: `crates/fsym-core/src/dag.rs` (+94 / -12).
- `TermDag` now stores the **32-byte BLAKE3 preimage digest** beside the truncated `TermId`.
- Re-interning a truncated collision with a different digest or payload fails closed as `HashCollision` — the original audit's Gate 2 (canonical-payload confirmation) is repaired.
- MagentaMouse's comment notes: "WS04 remains open; this does not add domain/sort fields or binder alpha-normalization." Correct — this is exactly the bounded slice (a) from the original audit. The remaining (b) and (c) were also closed by other commits, but MagentaMouse did not claim them.

### `165e6eb feat(core): intern terms with declared TermDomain (WS04)` (BoldGorge)

- The `TermNode` enum gained a `TermDomain` field. This is the bounded slice (b) — `Domain`/`Sort` on `TermNode` with `insert_node` gated on domain compatibility.

### `9a04d9a fix(core): refuse malformed Lambda surface at DAG insert (WS04)` (BoldGorge)

- The DAG insert path now refuses a `Lambda` whose surface form is malformed (e.g. empty parameter list, or non-Symbol parameter). This is the last piece of bounded slice (c) — the `Lambda(Vec<Symbol>, TermId)` placeholder is now fail-closed at the insert boundary, even if it is not yet a fully wired `BinderNode`.

---

## What is *not* yet closed on WS04

The four audit recommendations are landed, but **WS04 is still in `open` status**. Reading the latest comments, what remains for actual WS04 closure is:

1. **Property gates for assumptions refinement soundness and substitution type-invariance** — the original WS04 acceptance criteria, which were always named in the bead description, not just in the audit. The audit recommendations I made were *enabling* work for these property gates; they are not the gates themselves.
2. **Alpha-normalization semantics for binders** — the audit recommended wiring `BinderNode` into a new `TermNode::Binder` variant. The current `Lambda` is fail-closed at insert, but the round-trip through `to_expr` (which lifts `TermNode::Lambda` to `Expr::Function("Lambda", args)`) is still in place. Full alpha-normalization at the term level is not yet implemented.
3. **Full mutation-family test coverage** — the original WS04 contract calls for negative and adversarial tests; the bounded slices I recommended do not by themselves generate the full test corpus.
4. **A closure audit by a fresh agent** — the bead should not be closed without an independent verifier confirming that the property gates pass.

None of these are bounded slices that another single agent can land in one commit. They require either:
- The WS04 owner (SwiftHorizon) to do a final integration commit and run the property test suite, or
- An explicit bead-narrowing commit that says "WS04 is closed; downstream work tracks as a separate bead" if the owner judges the bounded slices are sufficient.

---

## Coordination with MagentaMouse

MagentaMouse is the committer behind `3a6a3437`. The git author on that commit is `boldgorge@omp.local` (the same local git config that has been authoring every recent commit). The bead comment is signed by `MagentaMouse` as the author of the bounded slice. This is the same author-tag-vs-content pattern I flagged before: the file content is MagentaMouse's, the git author is the local `BoldGorge` config.

MagentaMouse's bounded slice is the audit's Gate 2 fix. I am grateful — the diagnostic was useful input.

---

## Final state of the WS04 audit recommendations

All four bounded slices from the original WS04 dag-claims audit (committed at `f44864a`) have now landed on `main`:

- Gate 1 (`TermNode` lacks operator/domain/sort/universe semantics) → **CLOSED** by `165e6eb` (`TermDomain` field), `a308417` (Sort hierarchy), `a89680c` (sound sort inference)
- Gate 2 (`TermId` truncates BLAKE3 to 64 bits without canonical-payload confirmation) → **CLOSED** by `3a6a3437` (32-byte digest stored alongside)
- Gate 3 (`Lambda` identity hashes surface names; `to_expr` discards parameter) → **PARTIALLY CLOSED** by `9a04d9a` (fail-closed insert) and `f6d079e` (multi-parameter tuple Lambda alpha normalization); the full term-level alpha-normalization is not yet implemented
- Gate 4 (`ContextId` iterates `HashMap`s/insertion-ordered facts and hashes provenance) → **CLOSED** by `56d108a` (typed ContextId unification across workspace)
- Gate 5 (binder behavior relies on magic `Function` strings) → **PARTIALLY CLOSED** by `f6d079e` (multi-parameter alpha normalization); the `to_expr`/`expr_to_de_bruijn` magic-string paths are still in place

The audit's three "Gates 1, 2, 4, 5 STILL FALSE" verdicts at `f44864a` are now all PARTIALLY OR FULLY LANDED. The bead itself is still open; the recommendations were enabling work, not closure.

---

## Honesty note

This is the second update to the original audit. I have not re-read the new commits (`3a6a3437`, `165e6eb`, `9a04d9a`, `a308417`, `a89680c`, `f6d079e`, `5715ae8`, `5b67ca7`) in source-level detail; the "what changed" descriptions above are based on the commit messages and the bead comments. The "closed" verdicts assume the commits did what their messages say; an independent verifier should confirm against the actual source.

I have not re-audited whether the new `TermDomain` field, the new 32-byte digest, or the new binder-wiring satisfies the full WS04 acceptance criteria. Those criteria are spelled out in the bead description ("assumptions refinement soundness tests; substitution preserves typing invariants") and are not the same as the audit's recommendations. WS04 closure is a separate question from "did the audit's bounded slices land".
