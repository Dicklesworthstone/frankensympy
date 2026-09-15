# Independent capsule-gate review (`fra-rc-capsule-gate-5mq`)

## 1. Gate record

- **Gate bead:** `fra-rc-capsule-gate-5mq` (P1), reviews `fra-rc-capsule-39v` (P1)
- **Reviewed implementation revision:** `992b425` (feat(ws06): verifier-complete polynomial capsule)
- **Review battery revision:** `cddb3b6` (reviewer-authored `crates/fsym-capsule-consumer/tests/capsule_gate_review.rs`, written by BoldGorge, not the implementation author)

- **Verdict:** **FAIL** — the capsule verifier accepts false identity claims. Soundness hole: the product-vs-subject comparison is one-directional.
- **Raw run:** `artifacts/audit/capsule_gate_review_raw_run.txt` (42 passed, 5 failed, exit 101)
- **Not claimed:** any promotion of claims in `registries/claims.toml`; `fra-rc-capsule-39v` stays open; this gate stays open pending a fix and fresh independent re-review.


## 1a. Post-review repair (same day, implementation lane)

After the FAIL above was recorded, the implementation lane repaired
`verify_capsule` (`crates/fsym-proof-kernel/src/capsule.rs`): the misleading
`product_terms.len() > subject.terms.len().max(MAX_TERMS)` cap guard was removed
and a converse-direction check added (any product degree absent from the
descending, decode-canonicalized subject terms is `Refuted`). The author suite
gained four negative cases (`extra_product_term_is_refuted`,
`missing_subject_constant_with_matching_higher_terms_is_refuted`,
`zero_subject_versus_nonzero_constant_is_refuted`,
`zero_subject_verifies_against_an_exactly_zero_product`).

Observed after repair, same reviewer environment:

- `rch exec -- cargo test -p fsym-proof-kernel -p fsym-capsule-consumer --locked`
  → all suites green; reviewer battery 47/47; author capsule suite 13/13.
- Mutation probe: re-introducing the subset-only comparison
  (`is_err() && false` weakening) makes the battery fail again (the five
  finding tests) → the battery is a non-tautological oracle. Restored.
- `rustfmt --check` clean on both changed files; `./scripts/check.sh registries`
  exit 0; full-workspace `cargo test --locked` executed (outcome recorded in
  the gate bead comment).

The FAIL verdict above stands as the review of candidate `992b425`. The repaired
candidate still requires a **fresh independent gate re-review** by a reviewer
who is not an author of the repair before `fra-rc-capsule-gate-5mq` closes.

## 2. Minimized counterexamples (all currently return `Verified`; all are false claims)

| # | Subject (ZZ, var x) | Claimed as | True identity | Verdict observed |
|---|---|---|---|---|
| 1 | `x^2` | `1*(x^2+1)` | false (differs by 1) | `verified` |
| 2 | `2x^2` | `2*(x-1)(x+1) = 2x^2-2` | false (differs by 2) | `verified` |
| 3 | `x^2-1` | `1*(x^2+x-1)` | false (differs by x) | `verified` |
| 4 | `x^3` | `1*(x^3+x)` | false (differs by x) | `verified` |
| 5 | zero polynomial (empty object) | `1*(1)` | false (0 != 1) | `verified` |

Case 5 additionally shows a zero subject verifies against any constant of equal degree (0).

## 3. Root cause

`fsym_proof_kernel::capsule::verify_capsule` (`crates/fsym-proof-kernel/src/capsule.rs`, reviewed at `f7966e6`):

1. Degree precheck compares `subject.degree()` with the product's maximum degree only.
2. The term loop iterates **subject terms only** and asserts each subject coefficient equals the product's coefficient at that degree.
3. There is no converse check: product degrees **absent from the subject** are never inspected, so `subject + extra_term` passes whenever the subject's degrees all match. The `product_terms.len() > subject.terms.len().max(MAX_TERMS)` guard is a term-cap guard (`.max(MAX_TERMS)` makes it unreachable below 8192 terms), not an equality check.
4. An empty subject (zero polynomial) makes the forward loop vacuous, so it verifies against any product with degree 0.

The implementation's own suite (`capsule_gate.rs`, 33 tests) tests wrong *coefficients* and missing *subject* terms but never extra *product* terms or the zero subject, which is why it passed while the claim was not verifier-complete.

## 4. Commands executed (gate evidence)

```
rch exec -- cargo test -p fsym-proof-kernel -p fsym-capsule-consumer --locked
  → exit 101 (review battery failures above; author suites pass)
rch exec -- cargo test -p fsym-capsule-consumer --test capsule_gate_review --locked
  → exit 101, FAILED. 42 passed; 5 failed
cargo tree -p fsym-capsule-consumer --locked
cargo tree -p fsym-capsule-consumer --no-default-features --locked
  → closure = fsym-proof-kernel + fsym-{core,assumptions,bigint,budget,id,modular,outcome,rational,logic}
    + blake3/num-*/serde; no generator, planner, runtime, Python, storage, or network crates (PASS)
```

Closure and feature-config checks pass; the soundness failure is the sole gate blocker.

## 5. Disposition

- `fra-rc-capsule-gate-5mq`: **open**, verdict FAIL recorded; requires a fresh independent reviewer (not an author of the fix) to re-run the battery after the repair.
- `fra-rc-capsule-39v`: **open**, defect handed back to implementation: exact bidirectional term-map equality (subject ↔ product), zero-subject handling (0 == 0 may verify only against an exactly-zero product; 0 vs nonzero constant must be `Refuted`), plus author-suite negative cases for extra product terms and the zero subject.
- Downstream dependents (fra-rc-factor-lt4, fra-rc-numeric-t2y, fra-rc-cli-u9o, fra-rc-formal-ma3) inherit this gate and must not start from the current candidate.
