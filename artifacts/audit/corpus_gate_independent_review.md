# Independent corpus-gate review (`fra-rc-corpus-gate-emj`)

## 1. Gate record

- **Gate bead:** `fra-rc-corpus-gate-emj` (P0), reviews `fra-rc-corpus-us2` (P0)
- **Reviewed revision:** `634fd545601c71e8f2b98e5095e2e0075e7ad6e7`
- **Gate owner:** PeachMoose (not an author of `tools/conformance-lab/corpus_gate.py`)
- **Verdict:** **PASS** — the wrapper fails closed on every injected adversary, writes no ledger on invalid runs, and records unledgered drift exactly once
- **Reviewer-authored suite:** `tools/conformance-lab/test_corpus_gate_independent_review.py` (7 cases, black-box through `corpus_gate.main()`)
- **Not claimed:** mathematical parity of any corpus fixture, certification, or whole-WS01 closure. `lab-corpus` reports `certified: false` by construction.

## 2. Acceptance criteria

| Criterion | Evidence | Result |
|---|---|---|
| `python3 -m unittest discover -s tools/conformance-lab -p 'test_*.py'` | 123 tests, `OK` (was 116 before the reviewer suite landed) | pass |
| `./scripts/check.sh lab-corpus` | exit 0, `admitted: 230`, `drift_total: 0`, `unledgered: 0`, `certified: false` | pass |
| negative child/crash/empty/missing-field/duplicate/missing-fixture probes exit nonzero | reviewer suite cases 2–4 plus the author suite | pass |
| existing meaningful tests remain | author suite untouched; 116 → 123 tests, none removed | pass |

## 3. Reviewer battery (black box through `main()`)

Only the child `capture.py` process is injected; the real profile, the real
`diff_input_binding`, and an isolated ledger directory are used. Raw run:

```
test_child_execution_failure_is_refused_without_ledger_write ... ok
test_complete_matching_run_passes_and_leaves_the_ledger_alone ... ok
test_crashed_child_with_plausible_json_never_passes ... ok
test_incomplete_or_malformed_reports_never_pass ... ok
test_injected_child_receives_the_declared_profile_and_timeout ... ok
test_ledgered_open_drift_is_development_success ... ok
test_unledgered_drift_appends_exactly_once_then_passes ... ok

Ran 7 tests in 0.180s
OK
```

What each case pins:

1. **complete matching run** — exit 0, ledger file untouched, payload counts equal the frozen corpus size.
2. **crashed child with plausible JSON** — the audit's original false pass (`returncode=1`, `stdout={}`) plus `rc=0`-with-`{}`, an empty complete report, `SIGKILL` (`-9`) and timeout (`124`) exits: all nonzero, and no ledger is written.
3. **incomplete or malformed reports** — empty stdout, truncated JSON, zero fixtures, a frozen fixture present in neither admitted nor drifted, duplicated admitted IDs, a wrong profile binding, and drift without detail rows: all nonzero, no ledger write.
4. **child execution failure** — `OSError` from spawn: nonzero, no ledger write.
5. **ledgered-open drift** — exit 0 with `drift_total=1`, `ledgered_open=1`, `unledgered=0` (development success, explicitly not certification).
6. **unledgered drift** — exit 1, and the appended record set equals the observed drift signature set exactly (set assertion, not a count assertion); the re-run is then ledgered-open and does not duplicate records.
7. **child invocation contract** — the injected command carries `capture.py` and the declared profile, and the timeout equals `30 + 120 * fixture_count`.

## 4. Findings (recorded, not gate blockers)

**F1 — exit code 2 carries two meanings.** `main()` returns 2 for harness usage errors *and* for an invalid diff result (crashed/empty/malformed child). Both are fail-closed, so this is a diagnostic wart rather than a correctness hole; the reviewer suite asserts nonzero plus the absence of a ledger write rather than relying on the code.

**F2 — unledgered drift appends itself and passes on the next run.** This is the documented ledger design ("gate re-run will pass with them ledgered-open; FIX them, never weaken the comparator"): the first run fails, the drift becomes visible committed debt, and the gate stops failing. It is not a self-healing pass for *invalid* runs — case 3 pins that garbage never reaches the ledger — but it does mean a green `lab-corpus` after a red one is only as good as the ledger review. Worth remembering for the eventual certification path, where ledgered drift must not be confused with parity.

**F3 — no negative case can produce exit 0 by construction.** The reviewer battery asserts exit *and* ledger state for every adversary, so a future refactor that turns a refusal into a pass fails here even if it keeps the same stderr text.

## 5. Residual risk

- The suite injects the child; it does not exercise a real crashed `capture.py` binary. A real-crash fixture is not claimed.
- The gate is a development-drift gate: `certified: false` in every payload, and no claim in `registries/claims.toml` is promoted.
- Ledger contents are reviewed by humans; the gate cannot decide that a ledgered drift is acceptable.
