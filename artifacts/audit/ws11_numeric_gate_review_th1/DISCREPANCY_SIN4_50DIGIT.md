# DISCREPANCY (review retraction) — sin(4) 50-digit enclosure unsound

Found during the reviewer's OWN strict containment probing, AFTER the
initial PASS was recorded. This retracts the PASS: fra-rc-numeric-t2y and
fra-rc-numeric-gate-82w are reopened.

## Reproducer (exact rational arithmetic, no floats)

examples/sin4b.rs (retained under repro/): evaluate
`Expr::Function("sin", [4]).evalf_ball(digits)` and test
`ball.contains(parse_decimal(oracle_50_digit_string))` where the oracle
string is the isolated SymPy 1.14.0 `N(sin(4), 50)` output
`-0.75680249530792825137263909451182909413591288733647`.

## Observed

- digits = 20, 30, 40: ball CONTAINS the oracle value (sound).
- digits = 50: `ball.contains(p50) == false` AND
  |p50 - midpoint| > radius + 10^-49. Since |p50 - true value| <= 5e-51
  (50-significant-digit rounding), the ball excludes the true value by
  more than ~9.5e-50 — the enclosure is unsound at the requested
  precision. Radius is admitted as <= 10^-50 by evalf_ball's own check,
  so the composition error is in the MIDPOINT (displaced ~1e-49), not
  merely an under-estimated radius.

## Scope pattern

- digits <= 40 sound; failure appears by 50 digits (45/48/49 balls exist;
  exact containment at those widths not yet bisected — sweeps retained in
  the repro directory).
- sin(4) reduces with k = 1 (odd parity), x exact, magnitude 0: the pi
  enclosure, reduction, Taylor tail, and escalation loop all pass their
  internal admission checks while the midpoint is displaced ~1e-49.
  Suspicion: working-precision composition inside the
  evalf_ball/sin path near the top of the envelope (the same class as the
  09-17 repair, which fixed the radius admission but evidently not the
  midpoint displacement at 50 digits).

## Why the committed corpus did not catch it

The oracle corpus runs at 30 digits with a +/-10^-33 reference tolerance —
a 50-digit composition error is invisible at that width. The gate's own
acceptance text requires "requested 50-digit exp/sin ... matches oracle",
but no 50-digit containment case exists in the committed tests.

## Required repair (before 82w can pass)

1. Root-cause the midpoint displacement at digits=50 for the odd-k path
   (compare midpoint decimal expansion against the oracle digit-by-digit;
   bisect 45..50).
2. Add a committed 50-digit containment corpus case for an odd-k
   argument (sin(4)) and one even-k control (sin(1) exists).
3. Re-run this gate's acceptance battery.
