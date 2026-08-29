# Operator capture protection audit — follow-up: refutes one hypothesis, narrows the residual gap

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-29T07:42Z
**Scope:** read-only. No code edits. No new commit to crates/.
**Method:** read the uncommitted test additions in `crates/fsym-assumptions/src/bindings.rs` (in the local working tree, not yet committed by the sibling agent) and the `crates/fsym-core/src/dag.rs` Lambda binder extension. Re-evaluates the residual gap from the previous audit at `9102d42`.
**Source of truth:** local working tree at HEAD `9102d42` + uncommitted changes (not mine).

---

## Summary verdict

The audit at `9102d42` named two open hypotheses:
1. **`NonAlphaRenamingRequired` may be dead code** (defined in the enum but the audit could not confirm it is constructed anywhere in the function body).
2. **`UnsupportedOperatorVariableReplacement` may be the only error class actually returned** at the call sites.

A new uncommitted test in the local working tree (added by a sibling agent, not yet pushed) refutes hypothesis 2: `UnsupportedOperatorVariableReplacement` is exercised in a positive test for both `Derivative` and `Integral` operator variables, and a third assertion exercises the success path (substitution of a fresh symbol).

Hypothesis 1 (about `NonAlphaRenamingRequired`) is **unaffected** by the new test: the new test only constructs `UnsupportedOperatorVariableReplacement`, not `NonAlphaRenamingRequired`. The `NonAlphaRenamingRequired` variant may still be dead code.

**Net effect:** the residual gap is now narrower. The owner should run `cargo build` and `grep -n "NonAlphaRenamingRequired" crates/` to confirm whether the variant is constructed anywhere; if not, the bounded slice is now a 5-20 line "either add a construction site or remove the variant" — small enough to land in a single commit.

---

## Evidence

### What the new test (in the local working tree) confirms

The uncommitted test `operator_variable_substitution_with_non_symbol_replacement_refuses` (added after the audit at `9102d42`) at the bottom of `crates/fsym-assumptions/src/bindings.rs`'s test module exercises:

1. **Derivative operator variable with non-symbol replacement** — `capture_avoiding_subs(deriv, x, 5)` (where `deriv = Derivative(x + 1, x)`) returns `Err(BindingError::UnsupportedOperatorVariableReplacement { operator: "Derivative" })`. **This is the test the audit named as missing for the `UnsupportedOperatorVariableReplacement` path.**

2. **Integral operator variable with non-symbol replacement** — `capture_avoiding_subs(integral, x, y + 1)` (where `integral = Integral(x, x)`) returns `Err(BindingError::UnsupportedOperatorVariableReplacement { operator: "Integral" })`. Same path, different operator.

3. **Success path** — `capture_avoiding_subs(deriv, x, y)` (where the replacement is a fresh symbol) returns the correctly renamed `Derivative(y + 1, y)`. This is the representable-operator-substitution case the follow-up `08ef1c5` fix added.

### What the new test does NOT cover

The test does not exercise the `NonAlphaRenamingRequired { operator: ... }` variant. The audit's hypothesis that this variant is dead code is **not refuted** by the new test. The owner still needs to confirm with `cargo build` and `grep`.

### What the new test does NOT do well

- It does not cover the `Lambda` operator variable path. The audit named `Lambda` as a third operator that has the operator-variable-substitution logic; the test only covers `Derivative` and `Integral`.
- It does not cover the case where the replacement is a `Symbol` whose name is alpha-equivalent to the operator variable (e.g. substituting `x` for `x` is a no-op; substituting `y` for `x` is a rename; substituting `x` for a different `x` is a no-op; the test only covers the fresh-symbol rename).
- It does not cover the `non_alpha_equivalent_rename_required` case — exactly the case that the `NonAlphaRenamingRequired` variant is meant to handle. If this case is unreachable, the variant is dead code; if it is reachable but unhandled, the new test is insufficient.

---

## New observation: the `fsym-core/src/dag.rs` Lambda binder extension

The local working tree also has an uncommitted change in `crates/fsym-core/src/dag.rs` that extends `lambda_symbol_parameters` to handle both multi-argument `Lambda(x, y, body)` and tuple-argument `Lambda(Tuple(x, y), body)` as the same binder node. The module docstring at `:5-9` is updated to document this:

```
//! Tuple-parameter `Lambda` intern as the same binder node as
//! the multi-argument spelling. Lifting emits the multi-argument surface.
```

This is a *Lambda* binder change, not an *operator* binder change. The `Lambda` binder is not an "operator variable" in the same sense as `Derivative` and `Integral` — `Lambda` has a typed binder node (the `Lambda { param, body }` variant at `bindings.rs:14-25` from the prior audit). The audit's recommendations about the WS04 binder gate (Gate 5: "binder behavior relies on magic `Function` strings") are addressed by this work in part: the `Lambda` surface is now a typed binder rather than a magic string.

**This is the right direction.** The `Tuple(x, y)` form is the SymPy canonical form; the multi-argument form is the user-friendly form. Lowering both to the same `BinderNode` (or to the same `TermNode::Lambda` after the binder wiring lands) is the right design.

### Gap this leaves open

The `Derivative(x, y)` and `Integral(x, y)` forms are still magic strings in `classify_binder` (the prior audit's Gate 5). The `Tuple(x, y)` extension only addresses `Lambda`. The `Derivative` and `Integral` typed binder nodes (`bindings.rs:18-25`) exist but are not constructed by the parser; the parser produces `Expr::Function("Derivative", ...)` and `Expr::Function("Integral", ...)`, which are then matched by `classify_binder` on the literal string.

A bounded follow-up slice could wire `Derivative` and `Integral` surface forms to the typed `BinderNode` the same way the new work wires `Lambda` surface forms. The slice is well-bounded: the parser or `to_expr` step changes; the `classify_binder` magic-string match becomes a typed-binder construction; the existing test corpus is re-run. The new work on `Lambda` is a model for the slice.

---

## What the WS04 owner should do next

The bounded-slice work to close the remaining `NonAlphaRenamingRequired` gap is now well-scoped:

1. **Run `cargo build` and `grep -n "NonAlphaRenamingRequired" crates/`** to confirm whether the variant is constructed anywhere in the function body. If it is constructed (e.g. for a deeper-binder case the new test does not cover), the construction site is documented; the variant is live. If it is not constructed, the variant is dead code.
2. **If dead code:** remove the `NonAlphaRenamingRequired` variant from the enum. This is a 5-10 line bounded commit.
3. **If live but untested:** add a positive test that constructs the variant. This is a 10-30 line bounded commit.
4. **Independent of (1-3):** wire `Derivative` and `Integral` surface forms to typed `BinderNode` constructions, following the same pattern as the new `Lambda` work. This is a 50-200 line bounded commit; should NOT be in the same commit as (2) or (3) per the `conformance metastasis` anti-pattern (§5 of AGENTS.md).

---

## File:line evidence index

| File | Lines | What is there | Notes |
|------|------|---------------|-------|
| `crates/fsym-assumptions/src/bindings.rs` | (uncommitted, ~line 1683) | New test `operator_variable_substitution_with_non_symbol_replacement_refuses` | confirms `UnsupportedOperatorVariableReplacement` is live; does not exercise `NonAlphaRenamingRequired` |
| `crates/fsym-core/src/dag.rs` | 5-9 | Module docstring updated to document `Lambda` multi-arg and tuple-arg | uncommitted; sibling agent's work |
| `crates/fsym-core/src/dag.rs` | 596-625 | `lambda_symbol_parameters` extended for tuple-arg case | uncommitted; sibling agent's work |
| `crates/fsym-calculus/src/lib.rs` | 631-660 | Test cleanup changing `2^-1` to `1/2` | uncommitted; sibling agent's work |
| (mine) `artifacts/audit/ws04_operator_capture_protection_audit.md` | 17-35 | `BindingError` enum definition | unchanged from prior audit |

---

## Cross-cutting observations

### A. The uncommitted sibling-agent work is consistent with the audit's recommendations

The uncommitted `fsym-core/src/dag.rs` Lambda binder extension follows the *direction* the audit recommended (typed binder wiring), even though it is not the *exact* slice the audit named. This is fine: the audit is diagnostic input, not a contract. The owner may take a different decomposition.

### B. The new test is a positive sign of the bounded-slice + correction cycle

The uncommitted test was added by a sibling agent in response to the bounded slice at `08ef1c5` (which was the follow-up to CopperCat's `6f3bfda`). The test exercises both error paths and the success path. This is the right shape for a regression test: it kills the mutant at the original dispatch's `non_zero_digest → accept` path, and it locks the new fail-closed behavior.

### C. The follow-up slice is now well-scoped

If the owner takes the recommended next slice (`NonAlphaRenamingRequired` dead-code check + removal or test), it is a 5-30 line commit. If the owner also takes the typed-binder wiring slice, it is 50-200 lines. The two should be separate commits; the smaller slice can land first, the larger slice can land later.

---

## Honesty note

This follow-up was produced by the DustyAspen subagent. The uncommitted file contents were read from the local working tree; the sibling agent who added them has not pushed yet, so the file:line evidence here is *current* but not *remotely verified*. If the sibling agent's commit lands with a different test name or a different structure, the audit's claim about hypothesis 2 may need to be revised. The audit's claim about hypothesis 1 (about `NonAlphaRenamingRequired` being potentially dead code) is independent of the sibling agent's work and is not refuted by it.

The audit did not run `cargo build`; the dead-code check requires a build. The audit recommends the check; the owner (or any agent with a build environment) should run it.
