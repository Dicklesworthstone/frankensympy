# WS09 independent gate: complete factorization with irreducibility evidence (`fra-rc-factor-gate-w57`)

## Verdict: PASS

Reviewer: PlumSnow (2026-09-17), independent of implementation author CrimsonTurtle.
Implementation: fra-rc-factor-lt4, commit cba7fe4, confirmed ancestor of origin/main
(`git merge-base --is-ancestor`).

## Evidence

### Acceptance battery
`rch exec -- cargo test -p fsym-polys -p fsym-proof-kernel -p fsym-modular --locked`:
exit 0. fsym-modular 56 passed, fsym-polys 50 passed, fsym-proof-kernel 40 + 13 (capsule
gate) passed = **159 passed, 0 failed**. RCH selected worker vmi1153651 but fell back to
local execution (RCH-E415, workspace-inherited `serde` materialization failure) - recorded
as local-fallback evidence, not remote execution.

### Live pinned-oracle comparison (SymPy 1.14.0, isolated conformance venv)
Captured this session via `.venv-conformance/bin/python3`:
- `factor(x**4+4)` = `(x**2 - 2*x + 2)*(x**2 + 2*x + 2)` - the reality-check repro, now decomposed.
- `factor(x**2+x+1)` = `x**2 + x + 1` (irreducible control unchanged).
- `factor((x-3)*(x**2+1)**2)` matches; expansion coefficients `[1,-3,2,-6,1,-3]`.

Native agreement (asserted in the passing battery against identical inputs):
- `quartic_without_rational_roots_factors_with_witnesses`: exactly two degree-2 factors,
  scale 1, both carrying irreducibility witnesses, `verify_complete_factorization` accepts.
- `irreducible_quartic_stays_whole_with_a_witness`: x^4+1 stays whole; weaker no-witness
  claim correctly used (irreducible over ZZ yet reducible mod every prime).
- `multiplicities_and_linear_factors_survive_the_complete_path`: (x-3)(x^2+1)^2 from
  `[-3,1,-6,2,-3,1]`; x^2+1 at multiplicity exactly 2; verifier accepts.


### Direct native-vs-oracle differential probe (reviewer-run, this session)
A standalone reviewer binary compiled against the same workspace source tree ran
`complete_factorization` on the corpus and printed its factors (ascending coefficients):

```text
x4+4:           scale=1 factors=[deg2^1[2,-2,1] * deg2^1[2,2,1]]
x2+x+1:         scale=1 factors=[deg2^1[1,1,1]]
(x-3)(x2+1)^2:  scale=1 factors=[deg1^1[-3,1] * deg2^2[1,0,1]]
```

That is exactly the pinned oracle's output: `(x^2-2x+2)(x^2+2x+2)` for the hero quartic
(the reality-check repro from the bead background now decomposes correctly), the
irreducible control unchanged, and the multiplicity fixture with x^2+1 at multiplicity
exactly 2. Probe source: /tmp/fs09probe (reviewer-throwaway, not committed).

### Adversarial obligations
- Certificate mutation: `test_mutant_tampered_factorization_rejected`,
  `test_mutant_tampered_bezout_rejected` kill tampered certificates.
- Capsule gate (13 tests) refutes extra/missing factors, zero-subject confusion, forged
  verified flags, malformed lengths.
- `constructed_products_factor_back_exactly`: 192-case property test (colliding linear
  roots to multiplicity 9 + irreducible quadratic) asserting soundness, completeness,
  degree conservation; 248k-seed sweep recorded on the implementation bead.
- Typed refusals: degree and coefficient-height bounds; cancellation safe point surfaces
  as cancelled (`cancellation_at_a_safe_point_surfaces_as_cancelled`).

## Known recorded items (out of scope, unchanged)
- 3/12 normalization-convention differences vs SymPy (ZZ-primitive vs QQ-monic), documented on fra-rc-factor-lt4.
- Pre-existing fmt drift in fsym-calculus/src/lib.rs:114.

## Claim effects
No claim promoted; WS09 stays implemented_uncertified. Downstream
(fra-rc-portfolio-9i7, fra-rc-formal-ma3, fra-campaign-jacobian-bundle-ukp,
fra-ws09-factorization-116) may proceed on this gate.
