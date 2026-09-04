#!/usr/bin/env python3
"""Corpus drift gate: fail closed on UNLEDGERED drift, tolerate ledgered-open.

Wraps `capture.py diff` for the r2-corpus profile. Every observed discrepancy
is matched against the committed ledger (discrepancy records, schema/
discrepancy.schema.json, status "open"). Exit codes:
  0 - every observed drift is ledgered-open (visible debt, no surprises)
  1 - at least one UNLEDGERED drift (surprise -> gate fails)
  2 - harness usage error

The ledger is append/update only: closing an item requires landing the fix and
flipping status to closed_verified with evidence (never deleting the record).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = LAB_ROOT.parent.parent / "artifacts" / "conformance"


def drift_signature(fixture_id: str, paths: list[str]) -> str:
    joined = ",".join(sorted(paths))
    raw = f"{fixture_id}|{joined}"
    return "disc-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(LAB_ROOT / "profiles" / "sympy-1.14.0-cpython-r2-corpus.toml"))
    ap.add_argument("--candidate-python", default="/data/projects/frankensympy/.venv-conformance/bin/python3")
    args = ap.parse_args()

    profile = json.loads(json.dumps(dict(
        profile_id=Path(args.profile).stem.replace("-", "-").join([""]))))
    # profile_id extracted from the TOML by capture.py itself; we only need the
    # artifact directory name, derivable from the profile file stem.
    profile_dir = ARTIFACT_ROOT / Path(args.profile).stem
    ledger_path = profile_dir / "ledger.json"

    diff = subprocess.run(
        [sys.executable, str(LAB_ROOT / "capture.py"), "diff", args.profile,
         "--candidate-python", args.candidate_python],
        capture_output=True, text=True)
    try:
        report = json.loads(diff.stdout)
    except json.JSONDecodeError:
        print("corpus_gate: diff produced no JSON report", file=sys.stderr)
        print(diff.stdout, diff.stderr, file=sys.stderr)
        return 2

    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"records": []}
    ledgered = {
        (r["fixture_id"], tuple(sorted(r["difference_paths"]))): r
        for r in ledger.get("records", []) if r["status"] == "open"
    }

    observed, unledgered = [], []
    for det in report.get("details", []):
        fixture_id = det.get("fixture_id", "?")
        paths = sorted(det.get("difference_paths", []))
        sig = (fixture_id, tuple(paths))
        observed.append(sig)
        record = ledgered.get(sig)
        if record is None:
            unledgered.append({
                "schema_version": 1,
                "discrepancy_id": drift_signature(fixture_id, paths),
                "status": "open",
                "severity": "object",
                "profile_id": Path(args.profile).stem,
                "fixture_id": fixture_id,
                "comparator": report.get("comparator", "construction_only"),
                "difference_paths": paths,
            })

    newly = [u for u in unledgered]
    print(json.dumps({
        "gate": "lab-corpus",
        "admitted": report.get("admitted"),
        "drift_total": len(observed),
        "ledgered_open": len(observed) - len(newly),
        "unledgered": len(newly),
        "unledgered_records": newly,
    }, indent=1, sort_keys=True))

    if newly:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ledger.get("records", [])
        seen = {(r["fixture_id"], tuple(sorted(r.get("difference_paths", [])))) for r in existing}
        for u in newly:
            sig = (u["fixture_id"], tuple(u["difference_paths"]))
            if sig not in seen:
                existing.append(u)
        ledger["records"] = existing
        ledger["schema_version"] = 1
        ledger_path.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")
        print(f"corpus_gate: {len(newly)} unledgered drifts appended to {ledger_path}; "
              f"gate re-run will pass with them ledgered-open; FIX them, never weaken the comparator.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
