# WS07 Rewrite Side-Condition Audit (fra-rc-7hp)

Named gate artifact for the bounded WS07 residual slice "Verified rewriting and
simplification". Scope: the shipped rewrite registry in
`crates/fsym-simplify/src/rewrite.rs`, audited rule-by-rule for typed side
conditions and guard status. This artifact does not close `gate://ws07-rewrite`.

## Registry audit

Every `standard_rules()` entry now declares its side conditions as typed data
(`SideCondition`), enforced by the registry-level guard in `apply_step`
independently of each transform's private discipline:

| Rule | Typed side condition | Guard status |
|---|---|---|
| `add_zero_identity` | Unconditional (structural zero check) | n/a |
| `mul_one_identity` | Unconditional (structural one check) | n/a |
| `mul_zero_annihilator` | Unconditional (structural totality check `is_total_expr`) | n/a |
| `pow_zero_identity` | `Entailed { Base, NonZero }` | guarded |
| `pow_one_identity` | Unconditional | n/a |
| `trig_zero_eval` | Unconditional | n/a |
| `elementary_one_eval` | Unconditional | n/a |
| `pythagorean_identity` | Unconditional (exact coefficient pairing) | n/a |
| `exp_log_inverse` | `Entailed { NestedArg { outer: 0, inner: 0 }, Positive }` | guarded |
| `log_exp_inverse` | `Entailed { NestedArg { outer: 0, inner: 0 }, Real }` | guarded |

## Gap found and closed

`pow_zero_identity` previously **discarded** its own context query
(`let _ = ctx.query(base, Predicate::NonZero)`) and fired on any
non-literal-zero base. A symbolic `x^0` therefore rewrote to `1` with no
discharged side condition — the exact unguarded-conditional failure mode this
slice exists to exclude. The transform now requires either a syntactic
non-zero constant (inherent facts discharge) or an entailed `NonZero` from the
assumption context; the registry guard refuses the rule otherwise.

## Guard verification (planted negatives)

- `x^0` under the empty context: stays guarded (no rewrite) —
  `pow_zero_with_undischarged_condition_stays_guarded`.
- `x^0` with `x` assumed `Zero` (entailed-false `NonZero`): stays guarded.
- `exp(log(u))` with `u` assumed `Negative` (entailed-false `Positive`):
  stays guarded — `exp_log_inverse_with_entailed_false_positive_stays_guarded`.
- Registry-guard mutation test: a rule whose transform fires unconditionally
  but declares `Entailed { Base, NonZero }` is refused on the empty context
  and fires only after discharge —
  `registry_guard_refuses_undisciplined_transform`.

## Positive observables

- `5^0 => 1` with no assumptions (literal inherent facts discharge) —
  `pow_zero_literal_discharges_without_assumptions`.
- `x^0 => 1` with `x` assumed `NonZero` —
  `pow_zero_entailed_nonzero_discharges`.
- Pre-existing guarded-inverse tests (unknown-context `exp(log(x))`,
  `ln(exp(x))` stay guarded; discharged contexts fold) all pass.

## Outcome visibility

Refusals and guarded (non-firing) outcomes are observable as `None` from
`apply_step` and are asserted explicitly in the planted negatives above; no
rewrite is silently dropped from the transcript: every firing rule still emits
its `ProofRule::DefinitionalReduction` step for kernel verification.
