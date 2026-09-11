# Independent Gate Review: Gate Receipt Binding (`fra-rc-receipts-gate-ge6`)

## 1. Gate record

- **Gate bead:** `fra-rc-receipts-gate-ge6` (P0), reviews `fra-rc-receipts-uhu` (P0)
- **Reviewed revision:** `0e6f179be02b1f493e774a92b346ec543d8afe2d` (== `origin/main` at review start)
- **Gate owner:** PeachMoose (not an implementation author of `xtask/`)
- **Verdict:** **PASS** — 26/26 adversarial mutants killed, two real gate runs accepted, workspace quality gates green
- **Reviewer-authored harness:** `tools/gate_review_battery.py`
- **Not claimed:** execution attestation of historical receipts, release certification, whole-WS00 closure, or authority for any receipt produced before this revision

## 2. Immutable inputs

- Reviewed surface: `xtask/src/bin/gate_receipt_validator.rs`, `xtask/src/bin/xtask.rs`, `xtask/src/lib.rs`, `xtask/tests/receipt_tamper.rs`
- Historic audit baseline `355014e4e65b0e03b7dd588c28b4e9e2395b6974`; receipt slice spans `b3b7a01..19d0dab`
- Oracle: SymPy 1.14.0 pinned profile `sympy-1.14.0-cpython` (isolated, test-only); not exercised by this gate
- Reviewer harness and the receipts below were produced in throwaway clones (`/tmp/fsym-gate-review/*`); **no receipt in this repository was created, overwritten, or modified by this review**

## 3. Acceptance criteria

| Criterion | Evidence | Result |
|---|---|---|
| `rch exec -- cargo test -p xtask` | remote `vmi1149989`, `--locked`, exit 0 (26 tests incl. 707-line negative corpus) | pass |
| Real gate receipt accepted only against matching expected manifest | two real runner-produced schema-3 receipts validated (below) | pass |
| changed commit/profile/gate/check set/artifact rejected | battery MT1–MT12 | pass |
| zero executed tests rejected | battery MT13–MT15 | pass |
| full Rust quality checks | `cargo check --workspace --all-targets` exit 0; `clippy --workspace --all-targets -D warnings` exit 0; `cargo test --workspace --locked` exit 0 (503 s remote) | pass with finding F4 (fmt) |

### 3.1 Independently observed passing runs

Real gate commands were executed by the reviewed runner (`xtask`) inside an isolated clone of the reviewed revision, then validated by the reviewed validator binary as a black box:

```
$ xtask profile verify sympy-1.14.0-cpython
artifacts/audit/receipts gate=profile-verify status=passed checks_digest=bbc49b7e81b44938572edb257ab71c1ca51ff22ac666b422034f16f74bb90745
$ gate-receipt-validator --source-root <clone> artifacts/audit/receipts/profile-verify.receipt.json
VALID artifacts/audit/receipts/profile-verify.receipt.json gate=profile-verify status=passed checks=5    (exit 0)

$ xtask gate foundation            # 6 crate suites actually executed
artifacts/audit/receipts gate=foundation status=passed checks_digest=7dc89035e363301ef8cfdb4a1fa4e05fcb454e7191f3b9a362f683b622102bd6
$ gate-receipt-validator --source-root <clone> artifacts/audit/receipts/foundation.receipt.json
VALID artifacts/audit/receipts/foundation.receipt.json gate=foundation status=passed checks=9    (exit 0)
```

- `profile-verify` receipt sha256 `ff310144ddb235f0e5955db866c9c852843c5015be2daf0179cd6cd5a85734c6`, 5 checks
- `foundation` receipt sha256 `a1cad574fd09e048e392ef00edacf8074616eccd9f52e67c127a71ba54662035`, 9 checks, all six `tests-fsym-*` checks carry real `cargo test` transcripts with nonzero executed counts
- Both receipts bind `commit=0e6f179…`, `tree=e145e548…`, `inputs_digest=eb78fef2…`, 329 source files, and the profile digest

### 3.2 Adversarial battery (`tools/gate_review_battery.py`, 26/26 killed)

The battery re-seals `checks_digest` and `receipt_digest` with BLAKE3 exactly as the runner does (its re-sealer reproduces both stored commitments of both real receipts, so the attacker model is faithful), then drives the validator as a black box. Raw transcript:

```
re-sealer reproduces both runner commitments; strong attacker is faithful
[KILLED] P0 untouched real receipt is accepted                          exit=0 VALID gate=foundation status=passed checks=9
[KILLED] MT1 commit substitution is rejected                             exit=1 REJECT receipt source does not match inspected repository inputs
[KILLED] MT1 profile_digest substitution is rejected                     exit=1 REJECT profile digest mismatch
[KILLED] MT1 profile_id substitution is rejected                         exit=1 REJECT gate profile does not match its declared contract
[KILLED] MT1 gate substitution (artifact-bearing gate) is rejected       exit=1 REJECT required artifact set mismatch
[KILLED] MT1 gate substitution (no-artifact gate) is rejected            exit=1 REJECT unknown or duplicated check name
[KILLED] MT2 source.commit substitution is rejected                      exit=1 REJECT receipt source does not match inspected repository inputs
[KILLED] MT2 source.tree substitution is rejected                        exit=1 REJECT receipt source does not match inspected repository inputs
[KILLED] MT2 source.inputs_digest substitution is rejected               exit=1 REJECT receipt source does not match inspected repository inputs
[KILLED] MT2 source.files substitution is rejected                       exit=1 REJECT receipt source does not match inspected repository inputs
[KILLED] MT3 removed required check is rejected                          exit=1 REJECT required check set is incomplete
[KILLED] MT4 duplicated check name is rejected                           exit=1 REJECT unknown or duplicated check name
[KILLED] MT5 foreign-gate check name is rejected                         exit=1 REJECT unknown or duplicated check name
[KILLED] MT6 passed status with a failed check is rejected               exit=1 REJECT status "passed" inconsistent with checks (derived "failed")
[KILLED] MT7 failed status with passed checks is rejected                exit=1 REJECT status "failed" inconsistent with checks (derived "passed")
[KILLED] MT8 unknown field fails closed                                  exit=1 REJECT unknown field verified (fail closed)
[KILLED] MT9 omitted checks_digest, outer digest re-sealed               exit=1 REJECT missing required field checks_digest
[KILLED] MT10 substituted checks_digest, outer digest re-sealed          exit=1 REJECT checks_digest mismatch: claimed bbc49b7e…, recomputed 7dc89035…
[KILLED] MT11 artifact commitment on a no-artifact gate is rejected      exit=1 REJECT required artifact set mismatch
[KILLED] MT12 missing required artifact is rejected                      exit=1 REJECT required artifact set mismatch
[KILLED] MT13 zero executed tests is rejected                            exit=1 REJECT test check has no successful nonzero test execution
[KILLED] MT14 nonzero exit transcript is rejected                        exit=1 REJECT test check has no successful nonzero test execution
[KILLED] MT15 substituted test command is rejected                       exit=1 REJECT test command does not match the required check
[KILLED] MT16 schema-valid failed receipt never returns GatePassed       exit=1 VALID gate=foundation status=failed checks=9
[KILLED] MT17 cross-source replay is rejected                            exit=1 REJECT receipt source does not match inspected repository inputs
[KILLED] MT18 untracked source overlay is rejected                       exit=1 REJECT receipt source does not match inspected repository inputs

26/26 mutants killed
```

MT16 records the intended separation: a structurally valid **failed** receipt is *printed* as `VALID` but never returns GatePassed authority (exit 1). Authority is the exit status plus the independently declared required-check set — never a stored `passed` string.

### 3.3 Verifier-weakening sensitivity (the battery is not tautological)

The same battery was re-run against a **deliberately weakened** validator built from a throwaway clone of the same revision, with only the source-identity guard disabled:

```rust
// xtask/src/bin/gate_receipt_validator.rs, original line 314
-    if source != expected || obj["commit"].as_str() != Some(expected.commit.as_str()) {
+    if false && source != expected || obj["commit"].as_str() != Some(expected.commit.as_str()) {
```

The weakened binary is accepted by the positive control but loses exactly the source-binding kills:

```
21/26 mutants killed
[SURVIVED] MT2 source.commit substitution      exit=0 VALID … status=passed
[SURVIVED] MT2 source.tree substitution        exit=0 VALID … status=passed
[SURVIVED] MT2 source.inputs_digest substitution exit=0 VALID … status=passed
[SURVIVED] MT2 source.files substitution       exit=0 VALID … status=passed
[SURVIVED] MT18 untracked source overlay       exit=0 VALID … status=passed
```

So each of those five verdicts is caused by the guard under test, not by incidental parse failure: the corpus fails when the verifier is weakened, which is the mutation obligation this gate owes.

Reproduce:

```bash
tools/gate_review_battery.py \
  --validator <gate-receipt-validator> --source-root <checkout-with-.git> \
  --receipt <schema-3>/profile-verify.receipt.json --receipt <schema-3>/foundation.receipt.json \
  --foreign-root <second-checkout> --dirty-root <checkout-plus-untracked-file>
```

## 4. Findings

**F1 — All 19 committed receipts are rejected by the current validator (material, by design, unrecorded).**
`gate-receipt-validator --source-root /data/projects/frankensympy artifacts/audit/receipts/*.receipt.json` returns
`REJECT …: missing required field source` for **every one of the 19 committed receipts**: they are `schema_version: 1`, and the validator now requires schema 3 with source, profile and artifact commitments. This is deliberate in the code ("Historical schemas without artifact commitments cannot authorize a new passed gate … rerun to produce schema 3"), but two consequences are not recorded anywhere:
  1. the audit prose that claims these receipts are "validated fail-closed with `gate-receipt-validator`" (e.g. `artifacts/audit/ws16_remote_workers_and_indexing_audit.md`, `ws20_structured_domains_audit.md`, `ws23_release_and_packaging_audit.md` "All 19 machine gate receipts … are structurally validated") is **no longer reproducible at HEAD**;
  2. any downstream slice that intends to *reuse* a historic receipt instead of rerunning its gate will be refused.
No workstream closure in `registries/workstreams.toml` depends on those receipt bytes today (all WS remain `planned`/`in_progress`), so no closed claim is currently unauthorised; the statements that cite them are historical.

**F2 — Superseded receipts are overwritten in place, not archived.**
`xtask::bin::xtask::write_receipt` publishes atomically to `artifacts/audit/receipts/<gate>.receipt.json`. A rerun (which F1 makes mandatory for schema 3) therefore destroys the previous bytes for that gate, contradicting the validator's "preserve old receipts as history" intent. No archive path, no versioned name, no retention rule exists.

**F3 — No CI or release-path consumer; `scripts/check.sh` still says the validator does not exist.**
The validator is referenced only by its own test suite (`xtask/tests/receipt_tamper.rs`) and by `tools/campaign/harness.py`. `scripts/check.sh`'s `release_readiness` still hardcodes the comment "That validator has not landed." and unconditionally appends `profile '<id>' cannot certify until external same-commit gate receipt validation is implemented`. The gate is therefore a *tool with tests*, not yet an enforced release gate; the refusal is conservative (release stays blocked), but the stated reason is stale.

**F4 — Quality-gate availability limits on this workstation (environment, fail-closed behaviour confirmed).**
The pinned toolchain `nightly-2026-08-20` ships **without its cargo component** locally, so the runner's nested `cargo` calls fail with `error: the 'cargo' binary … is not applicable to the 'nightly-2026-08-20-…' toolchain`. The runner handled this correctly: it recorded the failures and produced a `status=failed` receipt (it did not fabricate success). Consequence: a gate run on this box needs an explicit cargo binding, and `cargo fmt --check` cannot be offloaded (`rch` refuses it as non-compilation, `RCH-E301`). Local `cargo +nightly-2026-08-22 fmt --check` reports exactly two diffs, both in `crates/fsym-calculus/src/lib.rs` (lines 114 and 2069), untouched by every receipt commit (`git log b3b7a01^..HEAD -- crates/fsym-calculus/src/lib.rs` is empty); they originate from `626bf01` (2026-09-08 calculus work).

**F5 — Declared, not defective: no freshness binding.**
A receipt for an unchanged source revision stays valid indefinitely; the validator asserts source identity, not that a command ran in this invocation. This is stated in the code ("This is not execution attestation") and is exactly the residual that an independently declared expectation manifest cannot remove without a signing/attestation authority. Recorded so downstream gates do not mistake `VALID` for "the tests just ran".

## 5. Residual risk and non-claims

- The battery's attacker can recompute digests but cannot sign; no signature lane exists, and none is claimed.
- MT16 shows the validator cannot by itself distinguish a *deliberately* failed receipt from a *tampered* one by exit code alone; the discriminator is the `VALID`/`REJECT` channel plus the required-check manifest. Reviewers must read stdout, not only the exit status.
- This review certifies the *binding* behaviour of the validator at `0e6f179`; it does not attest that any historical gate command executed, and does not promote any claim in `registries/claims.toml`.
- `tools/gate_review_battery.py` is reviewer-authored adversarial coverage. It is intentionally **not** part of `unittest discover` (it refuses rather than skips when `blake3` is absent), so CI neither runs nor silently skips it.