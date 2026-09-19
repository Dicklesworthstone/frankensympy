# Independent gate review — fra-rc-numeric-gate-82w over fra-rc-numeric-t2y

- Reviewer: TurquoiseHorizon (omp/glm-5.3-flash); not an author of the
  implementation (PlumSnow/BoldGorge; implementation commits through the
  precision-repair record of 2026-09-17).
- Reviewed tree: main 76ce7c7 (== origin/main at review start) plus this
  review's receipt re-mint (below).
- Date: 2026-09-19.

## Verdict: PASS — implementation accepted; gate closed on this evidence

## Independently re-executed (terminal exits inspected)

1. `cargo test -p fsym-core -p fsym-calculus --locked` (LOCAL; remote
   planner lane blocked this session, recorded in
   artifacts/audit/ws15_durable_m5e_th1/QUALITY_CHAIN.txt): exit 0,
   11 suites ok, 0 failed (test_core_calculus.txt).
2. xtask gate re-run: `cargo run -p xtask --bin xtask -- gate
   ws11-certified-numeric` → status=passed, exit 0. The gate re-minted
   artifacts/audit/receipts/ws11-certified-numeric.receipt.json against the
   current tree (checks digest 457c9878…; differs from the frozen-audit
   receipt because the workspace legitimately advanced since 09-15). First
   run refused to bind `python/fsym_python.so` (unbounded binary input) —
   a build artifact of the extension build; relocated for the gate run and
   restored after. Not a source-tree defect.
3. Independent strict 50-digit containment (reviewer-authored probe,
   exact rational arithmetic, /tmp scratch crate against fsym-core path
   dep): exp(1), sin(1), cos(2), sin(-7), exp(-1/3), sin(1/2) — the
   printed 50-digit oracle reference P (isolated SymPy 1.14.0) satisfies
   |P - mid| <= radius + 10^-49 for every entry ⇒ true value contained.
   ALL PASS. Honest refusals verified: free symbol, unsupported function
   (tan), precision 0.

## Code review (soundness argument inspected)

- pi: Machin identity, exact rational partial sums, Leibniz tail bounds;
  radius <= 10^-(digits+1) by construction.
- sin/cos: exact range reduction k = round(x/pi) with pi enclosed to
  digits + decimal-magnitude + 4 and parity sign transfer; interval
  Taylor with Lagrange tail bound; degree escalation loop re-drives the
  tail under target.
- exp: argument halving to |u| <= 1/2, guard digits 3 + 2s across the s
  squarings, exact series, final radius admission check.
- All constructions bounded by checked_work_bits; envelope refusals for
  arguments beyond the declared reduction/representability bounds; N()
  remains precision-honest (refuses > 15 digits through the binary64
  lane) while evalf_ball provides the certified path.

## Independent mutation probes (reviewer-executed)

- Parity sign transfer removed (sin): mutant killed — sin(4) (odd k=1)
  enclosure fails 50-digit containment in the reviewer probe, and the
  implementation corpus's sin(-3) case (odd k=-1) fails the oracle
  reference containment test. Non-tautological oracle: independently
  computed reference values vs the certified enclosure.
- pi guard margin reduced (digits+2 -> digits): NOT killed — absorbed by
  the escalation loop; behavior-equivalent within the admission
  tolerance (performance-only). Recorded as an equivalent mutant, not a
  soundness gap.
- sin Lagrange tail dropped at construction: survives only because the
  degree schedule drives the tail below the admission tolerance; noted
  as defense-in-depth. Observation for corpus owners: a positive-argument
  odd-k case (sin(4)) is absent from the oracle corpus (negative odd-k
  sin(-3) is present and exercises the same branch).

## Claim effects

- fra-rc-numeric-t2y: closed on this review.
- fra-rc-numeric-gate-82w: closed on this review. Downstream numeric
  admissions remain unpromoted; this is the bounded WS11 slice gate, not
  gate://ws11 certification.
