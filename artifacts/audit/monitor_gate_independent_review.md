# Independent monitor-gate review (`fra-rc-monitor-gate-p1m`)

## 1. Gate record

- **Gate bead:** `fra-rc-monitor-gate-p1m`, reviews `fra-rc-monitor-5ja` (registered `compatibility_drift_v1` monitor, candidate `1dc3b9a`)
- **Reviewer battery:** `crates/fsym-runtime/tests/monitor_gate_review.rs`, authored by CrimsonTurtle (gate owner), public API only, independent of implementer PeachMoose
- **Verdict on candidate `1dc3b9a`:** **FAIL** — one censoring hole in exactly the semantics this gate exists to check (see §3); everything else reviewed green.
- **Repair (implementation lane, CrimsonTurtle):** post-alarm observations are now charged before the one-way alarm verdict; the alarm transition is counted once. Battery 12/12 after repair; mutation probe kills the reintroduced weakening.
- **The gate stays OPEN:** the repaired candidate requires a fresh independent re-review by a reviewer who is not the repair's author.

## 2. Battery coverage (12 tests, all green post-repair)

1. Reset zeroes wealth, advances generation/resets, zeroes generation-local windows, and preserves every cumulative counter (alarms history survives); stream resumes with Continue.
2. A failure-only stream alarms (censoring by demoting failures to agreements is impossible) and lands in its own bucket.
3. Every failure kind charges its own counter; no observation vanishes from `total_observations`.
4. Alarm is one-way within a generation: post-alarm agreements keep returning Alarm with the alarm counted once.
5. Exact Ville arithmetic: at p1/p0 = 4 the threshold ln(20) is crossed exactly at the 3rd divergence (16 < 20 ≤ 64); `wealth()` == 64.
6. Null alarm rate ≤ alpha over 2,000 seeded streams × 200 steps (independent LCG, distinct multiplier).
7. ≥ 90 % power against a 0.40 drift stream within 100 steps.
8. Resume refuses a foreign monitor id and an invalid spec (alpha out of range).
9. A resumed stream is identical to a never-interrupted one (snapshot equality).
10. An inflated (forged) snapshot wealth still re-evaluates the bound on the next observation and alarms — never silently Continue.
11. Fault policy: a NaN snapshot faults on the next observation, recommends `increase_sampling`, and never alarms.
12. Registered actions are exactly `open_discrepancy` / `block_profile_promotion` / `increase_sampling`; monitor id `compatibility_drift_v1`.

## 3. The finding (raw failure preserved)

`observe()` early-returned whenever `state.alarmed` was set, so **every post-alarm observation was silently dropped from all counters** — `Failed{Timeout}` after an alarm left `timeouts == 0` and `total_observations` unchanged. This contradicts the monitor's own declared contract ("Never dropped: counted and charged as divergence", "no failure can vanish from the monitored stream"). Minimized: `new(0.05, 0.20, 0.05)`; three `Completed{discrepancy:true}` (alarm); one `Failed{Timeout}` → `timeouts == 0`, `total_observations == 3` (expected 1 and 4). The author suite never observes past the alarm, so it could not catch this.

Mutation probe: re-introducing the early-return (`if already_alarmed { return Alarm }` before charging) makes the battery fail again (11 passed / 1 failed) → the battery is a non-tautological oracle. Restored; 12/12 green.

## 4. Disposition

- `fra-rc-monitor-gate-p1m`: **open**, FAIL recorded on `1dc3b9a`; fresh independent re-review required for the repaired candidate.
- `fra-rc-monitor-5ja`: **open** (implementation accepted as repaired; closes only after the gate re-review passes).
- No claim promoted; `gate://ws13-portfolio-runtime` unaffected.
