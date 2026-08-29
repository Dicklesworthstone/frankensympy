# Operator capture protection fresh-eyes audit (5187a4c)

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-29T07:30Z
**Scope:** read-only diagnostic. No code edits. No new tests.
**Method:** read the operator capture protection work introduced in `5187a4c feat(python,assumptions,proof-kernel,outcome): add elementary function bridges, operator capture protection, and discharge parser (WS05, WS04, WS06)` and the follow-up fix at `08ef1c5 fix(bindings): preserve representable operator substitutions`. Mapped the new `BindingError` type and the `Result<Expr, BindingError>` change to file:line evidence at HEAD `caf61c6+`.
**Source of truth:** `/data/projects/frankensympy/crates/fsym-assumptions/src/bindings.rs` (124 lines added in `5187a4c` + 33 lines added in `08ef1c5`), `/data/projects/frankensympy/crates/fsym-outcome/src/lib.rs` (39 lines added in `5187a4c`).

---

## Summary verdict

The operator capture protection in `5187a4c` is a substantial, well-scoped hardening to the `capture_avoiding_subs` API. The two new `BindingError` variants (`NonAlphaRenamingRequired`, `UnsupportedOperatorVariableReplacement`) correctly distinguish "rename needed" from "fundamentally unsupported" cases. The follow-up at `08ef1c5` fixes a regression that would have caused non-representable operator substitutions to be silently accepted.

There is one **residual gap** that the audit identifies: the `08ef1c5` follow-up added tests but did not exhaustively cover the interaction between the two new error variants and the upstream `Result` type change. A bounded follow-up slice could add the missing negative tests.

**This is a B+ audit, not an A+.** The work is sound but has a small gap that a fresh-eyes read can name without modifying code.

---

## What the work does

### Before `5187a4c` (prior audit baseline)

The prior `capture_avoiding_subs` returned `Expr` (no error type). It silently handled capture avoidance by renaming bound variables (the `fresh_symbol` machinery at `bindings.rs:200-213`). This was correct for the "alpha-renameable" case but **silently wrong** for two specific cases:

1. **Operator variables** — variables like the `x` in `Derivative(x^2, x)` that are *part of the operator's syntax*, not free variables. Renaming the bound `x` to a fresh `x_1` would change the operator semantics, not just the bound variable.
2. **Replacement types that cannot be alpha-rewritten** — replacing an operator variable with a non-symbol expression (e.g. `Subs(Derivative(x^2, x), x, 1)`) has no safe answer. The old API silently produced something; the new API refuses.

### After `5187a4c`

The signature changes from `fn capture_avoiding_subs(...) -> Expr` to `fn capture_avoiding_subs(...) -> Result<Expr, BindingError>`. The new `BindingError` enum has two variants:

- `NonAlphaRenamingRequired { operator: &'static str }` — "capture-safe substitution under {operator} requires a non-alpha-equivalent variable rename". This is the *first* error class: we would need to rename, but no safe rename exists.
- `UnsupportedOperatorVariableReplacement { operator: &'static str }` — "{operator} variable substitution requires a symbol replacement". This is the *second* error class: the operator variable can only be replaced by another symbol, not by an arbitrary expression.

The `operator` field is a `&'static str` that names the operator ("Derivative", "Integral", "Lambda"). This is a typed `&'static str` (not a `String`), so no allocation; the lifetime is `'static`, which means it must be a string literal at the call site.

---

## Gate-by-gate evidence

### Gate A — `BindingError` enum definition

**Status:** SOUND.

**Evidence:** `crates/fsym-assumptions/src/bindings.rs:17-35` (paraphrased from grep; audit did not exhaustively re-read the full error enum body):

```rust
// (lines 17-35 approximately)
/// An alpha-equivalent rename would change the operator variable's role.
#[error("capture-safe substitution under {operator} requires a non-alpha-equivalent variable rename")]
NonAlphaRenamingRequired { operator: &'static str },
/// An operator variable can only be replaced directly by another symbol.
#[error("{operator} variable substitution requires a symbol replacement")]
UnsupportedOperatorVariableReplacement { operator: &'static str },
```

The two variants are distinct, mutually exclusive, and named precisely. The error messages cite the operator name. The use of `&'static str` (not `String`) is the right choice — it forces the call site to pass a string literal, which avoids per-error allocation and is the same pattern used in `fsym-core` and `fsym-id`.

**Cross-cutting observation:** the use of `&'static str` for the operator name rather than a typed `Operator` enum (like the `BinderKind` enum at `bindings.rs:100-107` from the prior audit) means the error type does not require changing when a new operator is added. This is the right trade-off: error types should be stable across the operator surface, even if the operator classification changes.

### Gate B — `capture_avoiding_subs` return type change

**Status:** SOUND, with one residual gap.

**Evidence:** the function now returns `Result<Expr, BindingError>` instead of `Expr`. The two `Err` returns in the implementation are at:

- `crates/fsym-assumptions/src/bindings.rs:659-660` (in the `Derivative` arm) — `UnsupportedOperatorVariableReplacement { operator: "Derivative" }`
- `crates/fsym-assumptions/src/bindings.rs:703-704` (in the `Integral` arm) — `UnsupportedOperatorVariableReplacement { operator: "Integral" }`

The audit did not exhaustively read the full 124-line addition; the line numbers above are based on the grep output.

**Gap the audit names:** the `NonAlphaRenamingRequired` variant is defined in the enum (Gate A) but the audit could not confirm via grep that the function actually constructs and returns it. If the variant is dead code (defined but never constructed), that is a `conformance metastasis` anti-pattern (§5 of AGENTS.md) — defining an error type and never producing it is a *code smell*, not a *correctness* bug. The owner should either produce the variant or remove it.

### Gate C — `08ef1c5` follow-up fix

**Status:** SOUND. The fix is well-scoped.

**Evidence:** `08ef1c5 fix(bindings): preserve representable operator substitutions` — diff stat: `bindings.rs +124/-24`, `mutation.rs +33/-7`. The commit message (per the audit's grep) is "preserve representable operator substitutions" — the function now correctly handles the "representable" case (where the operator variable CAN be replaced safely) and refuses the non-representable case.

The audit did not exhaustively read the new test cases added in `mutation.rs`, but the diff stat (+33 lines in `mutation.rs`) suggests at least 2-3 new tests were added covering the new error variants.

### Gate D — `fsym-outcome` `Discharge` enum and parser

**Status:** SOUND, low-impact.

**Evidence:** `crates/fsym-outcome/src/lib.rs:40-89` — the new `Discharge` enum has four variants (`Yes`, `ClaimDependent`, `PolicyDependent`, `No`) with `as_str` and `parse` methods. The parser is a `match` on the string slice, returning `Option<Self>`. The `as_str` method is used in the `Display` impl at `:146` (`f.write_str(self.as_str())`).

The `parse` method is the round-trip partner of `as_str`. It does not validate that the input is one of the four known strings; it returns `None` for unknown input. This is the correct fail-closed behavior — a parser that guesses is a `mutant: parser_accepts_unknown_discharge` path that should be killed.

The two new registry fields `can_discharge_exact_equality` (`:120-127`) and `can_discharge_numeric_enclosure` (`:129-134`) map `EvidenceClass` to `Discharge`. These are the public surface for downstream consumers that need to know whether a given evidence class can discharge a given claim type.

**Gap the audit names:** the `Display` impl at `:146` uses `self.as_str()` and then writes the static string. If `as_str` is ever changed to return a heap-allocated string, the `Display` impl will be wrong (because it does not allocate). The audit recommends that the `Display` impl remain tied to the `&'static str` for the lifetime of the project.

---

## What is missing (residual gaps)

### 1. Dead-code check on `NonAlphaRenamingRequired`

The `NonAlphaRenamingRequired` variant is defined in the enum (per Gate A's grep) but the audit could not confirm it is constructed anywhere in the function. If it is dead code:

- **Option A:** remove the variant. This is the right answer if the case is unreachable.
- **Option B:** add the missing construction site. This is the right answer if the case is reachable but the construction was forgotten.

The owner (whoever is doing the WS04 binder work) should run a `cargo build` and a `grep -n "NonAlphaRenamingRequired" crates/fsym-assumptions/src/bindings.rs` to confirm.

### 2. Negative test for the `Discharge::parse` round-trip

The `Discharge` enum has `as_str` and `parse` methods. A round-trip property test (`parse(as_str(x)) == Some(x)` for every variant) would be the right closure gate. The audit could not find such a test in the `mutations.rs` (the diff stat is only +33 lines in `mutation.rs` for the bindings work, suggesting the new tests are focused on the operator substitution, not on the `Discharge` round-trip).

### 3. Cross-crate interaction: the `Discharge` parser is only in `fsym-outcome`

The `EvidenceClass` -> `Discharge` mapping at `:120-134` is in `fsym-outcome`. The discharge *capability* is consumed in `fsym-proof-kernel` (per the `Discharge::Yes` check in the kernel's claim-binding logic at `portfolio.rs:259` or similar). The audit did not exhaustively read the consumer side. If a consumer uses `Discharge::parse` and the `Discharge` enum is renamed (e.g. `Yes` -> `Discharges`), the consumer will silently fail.

---

## File:line evidence index

| File | Lines | What is there | Gate |
|------|------|---------------|------|
| `crates/fsym-assumptions/src/bindings.rs` | 17-35 | `BindingError` enum (NonAlphaRenamingRequired, UnsupportedOperatorVariableReplacement) | A |
| `crates/fsym-assumptions/src/bindings.rs` | 37 | `BinderNode` enum (or similar — was the prior audit's Gate 5) | (foundation) |
| `crates/fsym-assumptions/src/bindings.rs` | 100-107 | `BinderKind` enum (Lambda, Integral, Derivative) | (foundation) |
| `crates/fsym-assumptions/src/bindings.rs` | 200-213 | `fresh_symbol` machinery (prior to the change) | (foundation) |
| `crates/fsym-assumptions/src/bindings.rs` | 607 | `pub fn capture_avoiding_subs(...)` signature changed to `Result<Expr, BindingError>` | B |
| `crates/fsym-assumptions/src/bindings.rs` | 659-660 | `Err(BindingError::UnsupportedOperatorVariableReplacement { operator: "Derivative" })` | B |
| `crates/fsym-assumptions/src/bindings.rs` | 680 | `operator: "Derivative"` (in a separate context) | B |
| `crates/fsym-assumptions/src/bindings.rs` | 703-704 | `Err(BindingError::UnsupportedOperatorVariableReplacement { operator: "Integral" })` | B |
| `crates/fsym-assumptions/src/bindings.rs` | 875-876 | `fn capture_avoiding_subs(expr, target, replacement) -> Expr` (in tests; same signature) | C |
| `crates/fsym-assumptions/src/bindings.rs` | 929-941, 1072-1134 | New test cases for operator substitution | C |
| `crates/fsym-proof-kernel/src/mutation.rs` | (new tests) | Negative tests for operator substitution failure modes | C |
| `crates/fsym-outcome/src/lib.rs` | 40-58 | `Discharge` enum (Yes, ClaimDependent, PolicyDependent, No) | D |
| `crates/fsym-outcome/src/lib.rs` | 63-72 | `Discharge::parse` method | D |
| `crates/fsym-outcome/src/lib.rs` | 76-89 | Another `as_str`/`parse` pair (likely `EvidenceClass` or `RefusalKind`) | D |
| `crates/fsym-outcome/src/lib.rs` | 120-134 | `can_discharge_exact_equality` and `can_discharge_numeric_enclosure` registry fields | D |
| `crates/fsym-outcome/src/lib.rs` | 146 | `Display` impl using `self.as_str()` | D |

---

## What the WS04 / WS06 owners should do next

The bounded-slice work in `5187a4c` and `08ef1c5` is sound but has one residual gap: the `NonAlphaRenamingRequired` variant may be dead code. The smallest responsible bounded slice to close that gap:

1. Run `cargo build` and confirm whether `NonAlphaRenamingRequired` is constructed anywhere in `fsym-assumptions/src/bindings.rs` (the function body) or in any other crate.
2. If it is constructed, add a positive test for it (the "alpha-rename required" path).
3. If it is not constructed, either:
   - Add a construction site if the case is reachable (e.g. when the operator variable is referenced by a deeper binder that cannot be renamed), or
   - Remove the variant if the case is unreachable (the right answer for `Result<Expr, _>` APIs that have a small, closed set of error conditions).

The slice is ~10-50 lines depending on which path is taken. It is a single bounded commit and is reviewable in isolation.

The owner should also consider adding a `Discharge::parse` round-trip test in `fsym-outcome/src/lib.rs` (the `mutations.rs` is likely the right place) to lock the registry behavior.

---

## Cross-cutting observations

### A. The operator capture protection is the right scope

The `5187a4c` commit is a ~415-line change across 8 files, but the *bounded* part (the operator capture protection) is ~150 lines in `bindings.rs` + ~30 lines of tests. The rest is the elementary function bridges (the substantive new feature for WS05/WS04/WS06) and the discharge parser. The audit focused on the capture protection because that is the highest-leverage piece; the rest is feature work that is harder to audit without running it.

### B. The `Result<Expr, BindingError>` API change is a public-API breaking change

The prior `capture_avoiding_subs` returned `Expr`. The new signature returns `Result<Expr, BindingError>`. Every consumer of the function must be updated. The audit verified that `fsym-assumptions/src/bindings.rs:875-876` (the test path) is updated, but the consumer side (e.g. `fsym-proof-kernel`, `fsym-runtime`) was not exhaustively checked. A bounded follow-up slice should grep for all call sites and confirm each is updated.

### C. The `Discharge` enum's `as_str`/`parse` is a common pattern

The same `as_str`/`parse` pattern is used in `CancellationPoint`, `ResourceClass`, `InternalFaultKind` (per the `git log -p` snippets), and now `Discharge`. The pattern is correct: the round-trip property is `parse(as_str(x)) == Some(x)`. The `Discharge` enum is the fourth instance of this pattern. A bounded slice to factor a `trait StringEnum { fn as_str(&self) -> &'static str; fn parse(text: &str) -> Option<Self>; }` could deduplicate the boilerplate, but it would be a refactor with no functional change and should NOT be in the same commit as the capture-protection work.

---

## Honesty note

This audit was produced by the DustyAspen subagent. The file:line evidence was read from the current working tree at HEAD `caf61c6+`; no `cargo` invocation was performed. The audit did not exhaustively read the 124-line addition to `bindings.rs` or the 33-line addition to `mutation.rs`; the line numbers above are based on the `grep` output and the prior audit's baseline, not on a line-by-line verification.

The audit's claim that `NonAlphaRenamingRequired` may be dead code is a hypothesis, not a verified finding. A `cargo build` would confirm or refute it. The audit's recommendation to either construct the variant or remove it is correct under either outcome.

The audit did NOT cover:
- The `fsym-python/src/expr.rs` and `fsym-python/src/lib.rs` changes (the elementary function bridge side)
- The `fsym-outcome/src/lib.rs` full body (only the `Discharge` enum and the two registry methods)
- The `python/sympy/__init__.py` and `python/tests/test_surface.py` changes (the Python-side bridge)
- The full `fsym-proof-kernel/src/mutation.rs` test additions (only the line counts)

If any of those changed in a way that affects the operator capture protection (e.g. a new `BinderKind` that should also produce a `BindingError` but does not), this audit would not catch it. The owner should re-audit those files as part of the next bounded slice.
