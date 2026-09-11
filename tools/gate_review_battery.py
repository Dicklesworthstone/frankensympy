#!/usr/bin/env python3
"""Independent adversarial battery for the gate receipt validator.

Consumer: the `fra-rc-receipts-gate-ge6` independent gate (reviewer-authored) and
any rerun of that gate. It enforces that a receipt is accepted only when its
commit, source snapshot, profile, gate/check manifest, artifact commitments and
test-execution transcripts all match independently declared expectations, even
against an attacker who recomputes the BLAKE3 commitments.
Deletion condition: when schema-3 receipts are regenerated repository-wide and
the cases below are folded into `xtask/tests/receipt_tamper.rs`.

The battery never writes inside the repository. It copies each receipt into a
temporary directory, mutates it there, re-seals the commitments with BLAKE3 the
way the runner does, and drives the compiled validator binary as a black box,
recording raw stderr/stdout and exit codes for every case.

Invocation:
    tools/gate_review_battery.py \
        --validator <gate-receipt-validator> \
        --source-root <checkout with .git> \
        --receipt <schema-3 receipt>.json [--receipt <another>.json ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import blake3
except ImportError:  # pragma: no cover - explicit refusal, never a silent pass
    print("REFUSE: the blake3 Python package is required to re-seal receipts", file=sys.stderr)
    raise SystemExit(2)


def digest(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def canonical_checks(checks: list[dict]) -> str:
    rows = [
        {"detail": c["detail"], "name": c["name"], "status": c["status"]}
        for c in checks
    ]
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


def reseal(receipt: dict, *, checks: bool = True) -> dict:
    """Recompute the commitments the way xtask::bin::xtask::finish does.

    `checks=False` keeps the (mutated or missing) `checks_digest` the attacker
    chose and re-seals only the outer `receipt_digest`, which is the strongest
    form of a digest-substitution or digest-omission attack.
    """
    if checks:
        receipt["checks_digest"] = digest(canonical_checks(receipt["checks"]).encode())
    payload = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    receipt["receipt_digest"] = digest(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode()
    )
    return receipt


def run(validator: Path, source_root: Path, receipt: Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        [str(validator), "--source-root", str(source_root), str(receipt)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


class Battery:
    def __init__(self, validator: Path, source_root: Path, scratch: Path) -> None:
        self.validator = validator
        self.source_root = source_root
        self.scratch = scratch
        self.results: list[tuple[str, bool, str]] = []
        self.case = 0

    def write(self, receipt: dict, name: str) -> Path:
        self.case += 1
        path = self.scratch / f"{self.case:02d}-{name}.json"
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        return path

    def expect(self, label: str, receipt: dict | None, verdict: str, name: str) -> None:
        """verdict: 'accept' (exit 0) or 'reject' (nonzero)."""
        path = self.write(receipt, name)
        code, out, err = run(self.validator, self.source_root, path)
        if verdict == "accept":
            ok = code == 0 and out.startswith("VALID")
        else:
            ok = code != 0 and err.startswith("REJECT")
        self.results.append((label, ok, f"exit={code} stdout={out!r} stderr={err!r}"))
        marker = "KILLED" if ok else "SURVIVED"
        print(f"[{marker}] {label}\n         exit={code} stdout={out!r} stderr={err!r}")

    def expect_failure_receipt(self, label: str, receipt: dict, name: str) -> None:
        """Schema-valid failed receipt: VALID verdict, but never exit 0 authority."""
        path = self.write(receipt, name)
        code, out, err = run(self.validator, self.source_root, path)
        ok = code != 0 and out.startswith("VALID") and "status=failed" in out
        self.results.append((label, ok, f"exit={code} stdout={out!r} stderr={err!r}"))
        marker = "KILLED" if ok else "SURVIVED"
        print(f"[{marker}] {label}\n         exit={code} stdout={out!r} stderr={err!r}")


def selftest_reseal(base: dict) -> None:
    """Prove the re-sealer reproduces the runner's own commitments."""
    control = dict(base)
    check = dict(control)
    reseal(check)
    if check["checks_digest"] != base["checks_digest"]:
        raise SystemExit(
            "REFUSE: re-sealer cannot reproduce checks_digest: "
            f"{check['checks_digest']} != {base['checks_digest']}"
        )
    if check["receipt_digest"] != base["receipt_digest"]:
        raise SystemExit(
            "REFUSE: re-sealer cannot reproduce receipt_digest: "
            f"{check['receipt_digest']} != {base['receipt_digest']}"
        )
    print("re-sealer reproduces both runner commitments; strong attacker is faithful")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--foreign-root", type=Path, default=None,
                        help="second checkout used for the cross-source replay case")
    parser.add_argument("--dirty-root", type=Path, default=None,
                        help="same checkout plus one untracked source file, for the dirty-overlay case")
    args = parser.parse_args()

    receipts = {json.loads(p.read_text())["gate"]: json.loads(p.read_text()) for p in args.receipt}
    for gate, receipt in receipts.items():
        selftest_reseal(receipt)

    scratch = Path(tempfile.mkdtemp(prefix="gate-review-battery-"))
    print(f"scratch={scratch}")
    battery = Battery(args.validator, args.source_root, scratch)

    with_test_checks = [
        r for r in receipts.values()
        if any(c["name"].startswith(("test-", "tests-")) for c in r["checks"])
    ]
    if not with_test_checks:
        raise SystemExit("REFUSE: no supplied receipt carries test-execution checks")
    base = dict(with_test_checks[0])
    profile_base = next(iter(r for g, r in receipts.items() if g != base["gate"]), base)

    def clone() -> dict:
        return json.loads(json.dumps(base))

    # Positive control: the untouched real receipt must be accepted.
    battery.expect("P0 untouched real receipt is accepted", clone(), "accept", "control")

    # Identity mutations (attacker re-seals both commitments).
    for field, value in (
        ("commit", "0" * 40),
        ("profile_digest", "0" * 64),
        ("profile_id", "sympy-1.14.0-cpython-r2-corpus"),
        ("gate", "ws22-performance"),
        ("gate", "ws10-exact-linear"),
    ):
        mutant = clone()
        mutant[field] = value
        battery.expect(f"M1 {field} substitution is rejected", reseal(mutant), "reject", f"field-{field}")

    for field in ("commit", "tree", "inputs_digest", "files"):
        mutant = clone()
        mutant["source"][field] = "0" * 8 if field != "files" else mutant["source"]["files"] + 1
        battery.expect(f"M2 source.{field} substitution is rejected", reseal(mutant), "reject", f"source-{field}")

    # Manifest mutations.
    mutant = clone()
    mutant["checks"] = mutant["checks"][:-1]
    battery.expect("M3 removed required check is rejected", reseal(mutant), "reject", "check-removed")

    mutant = clone()
    mutant["checks"].append(json.loads(json.dumps(mutant["checks"][0])))
    battery.expect("M4 duplicated check name is rejected", reseal(mutant), "reject", "check-duplicated")

    mutant = clone()
    mutant["checks"][0]["name"] = "ws22-paired-live-incumbent-bench"
    battery.expect("M5 foreign-gate check name is rejected", reseal(mutant), "reject", "check-foreign")

    mutant = clone()
    mutant["checks"][0]["status"] = "failed"
    battery.expect("M6 passed status with a failed check is rejected", reseal(mutant), "reject", "status-inconsistent")

    mutant = clone()
    mutant["status"] = "failed"
    battery.expect("M7 failed status with passed checks is rejected", reseal(mutant), "reject", "status-flipped")

    mutant = clone()
    mutant["verified"] = True
    battery.expect("M8 unknown field fails closed", reseal(mutant), "reject", "unknown-field")

    mutant = clone()
    del mutant["checks_digest"]
    battery.expect(
        "M9 omitted checks_digest with re-sealed outer commitment is rejected",
        reseal(mutant, checks=False),
        "reject",
        "digest-omitted",
    )

    mutant = clone()
    mutant["checks_digest"] = profile_base["checks_digest"]
    battery.expect(
        "M10 substituted checks_digest with re-sealed outer commitment is rejected",
        reseal(mutant, checks=False),
        "reject",
        "digest-substituted",
    )

    mutant = clone()
    mutant["artifact_digests"] = {"artifacts/benchmarks/ws22_paired_benchmark_report.json": "0" * 64}
    battery.expect("M11 artifact commitment on a no-artifact gate is rejected", reseal(mutant), "reject", "artifact-added")

    mutant = clone()
    mutant["artifact_digests"] = {}
    mutant["gate"] = "ws22-performance"
    battery.expect("M12 missing required artifact is rejected", reseal(mutant), "reject", "artifact-missing")

    # Test-execution transcript mutations.
    mutant = clone()
    for check in mutant["checks"]:
        if check["name"].startswith(("test-", "tests-")):
            detail = json.loads(check["detail"])
            detail["stdout"] = "test result: ok. 0 passed; 0 failed; 0 ignored;\n"
            check["detail"] = json.dumps(detail, separators=(",", ":"))
            break
    battery.expect("M13 zero executed tests is rejected", reseal(mutant), "reject", "zero-tests")

    mutant = clone()
    for check in mutant["checks"]:
        if check["name"].startswith(("test-", "tests-")):
            detail = json.loads(check["detail"])
            detail["exit_code"] = 1
            check["detail"] = json.dumps(detail, separators=(",", ":"))
            break
    battery.expect("M14 nonzero exit transcript is rejected", reseal(mutant), "reject", "nonzero-exit")

    mutant = clone()
    for check in mutant["checks"]:
        if check["name"].startswith(("test-", "tests-")):
            detail = json.loads(check["detail"])
            detail["command"] = ["cargo", "test", "-p", "unrelated", "--quiet"]
            check["detail"] = json.dumps(detail, separators=(",", ":"))
            break
    battery.expect("M15 substituted test command is rejected", reseal(mutant), "reject", "command-substituted")

    # Schema-valid failure is not GatePassed authority: an honest non-test check
    # failing keeps every transcript intact, so the receipt remains well formed.
    mutant = clone()
    for check in mutant["checks"]:
        if not check["name"].startswith(("test-", "tests-")):
            check["status"] = "failed"
            break
    mutant["status"] = "failed"
    battery.expect_failure_receipt(
        "M16 schema-valid failed receipt never returns GatePassed authority",
        reseal(mutant),
        "failed-but-valid",
    )

    # Replay across a foreign source root.
    if args.foreign_root is not None:
        path = battery.write(clone(), "replay")
        code, out, err = run(args.validator, args.foreign_root, path)
        ok = code != 0 and err.startswith("REJECT")
        battery.results.append(("M17 cross-source replay is rejected", ok,
                                f"exit={code} stdout={out!r} stderr={err!r}"))
        print(f"[{'KILLED' if ok else 'SURVIVED'}] M17 cross-source replay is rejected"
              f"\n         exit={code} stdout={out!r} stderr={err!r}")

    # Dirty overlay: the same commit plus one untracked source file must not be
    # able to reuse an existing receipt.
    if args.dirty_root is not None:
        path = battery.write(clone(), "dirty")
        code, out, err = run(args.validator, args.dirty_root, path)
        ok = code != 0 and err.startswith("REJECT")
        battery.results.append(("M18 untracked source overlay is rejected", ok,
                                f"exit={code} stdout={out!r} stderr={err!r}"))
        print(f"[{'KILLED' if ok else 'SURVIVED'}] M18 untracked source overlay is rejected"
              f"\n         exit={code} stdout={out!r} stderr={err!r}")

    survived = [label for label, ok, _ in battery.results if not ok]
    print(f"\n{len(battery.results) - len(survived)}/{len(battery.results)} mutants killed")
    if survived:
        print("SURVIVED:", "; ".join(survived), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
