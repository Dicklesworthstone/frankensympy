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

from capture import diff_input_binding, load_profile

LAB_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = LAB_ROOT.parent.parent / "artifacts" / "conformance"


def parse_report(raw: str) -> object:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def drift_signature(fixture_id: str, paths: list[str]) -> str:
    joined = ",".join(sorted(paths))
    raw = f"{fixture_id}|{joined}"
    return "disc-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def validate_report(report: object, returncode: int, binding: dict) -> dict:
    """Reject incomplete runs before any drift can be admitted or ledgered.

    Exit 1 is allowed only for a complete, internally consistent drift report.
    A crash with JSON stdout is not a drift report. This validates development
    construction coverage, never full-surface compatibility or certification.
    """
    required = {
        "schema_version", "input_binding", "comparator", "paired", "discrepancies",
        "admitted", "admitted_fixture_ids", "type_matched", "type_matched_fixture_ids",
        "by_kind", "broken_candidate", "affected_claim", "named_claims",
        "claims_promoted", "ledger_dir", "ledger_records", "fixture_ids", "details",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("invalid diff report fields")
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported diff report schema")
    if report["input_binding"] != binding:
        raise ValueError("diff input/profile/environment binding mismatch")
    if report["comparator"] != "construction_only":
        raise ValueError("unexpected development comparator")
    if (report["broken_candidate"] is not False or report["claims_promoted"] is not False
            or report["affected_claim"] is not None or report["named_claims"] != []
            or report["ledger_dir"] is not None):
        raise ValueError("unexpected diff execution mode")

    def ids(key: str) -> set[str]:
        values = report[key]
        if (not isinstance(values, list) or not all(isinstance(v, str) for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"invalid or duplicated {key}")
        return set(values)

    admitted = ids("admitted_fixture_ids")
    drifted = ids("fixture_ids")
    matched = ids("type_matched_fixture_ids")
    expected = set(binding["fixture_ids"])
    if not expected or admitted & drifted or admitted | drifted != expected:
        raise ValueError("diff does not partition the complete frozen corpus")
    if not admitted <= matched <= expected:
        raise ValueError("invalid type-matched coverage")
    counts = {"paired": len(expected), "admitted": len(admitted),
              "discrepancies": len(drifted), "type_matched": len(matched),
              "ledger_records": 0}
    for key, expected_count in counts.items():
        if type(report[key]) is not int or report[key] != expected_count:
            raise ValueError(f"inconsistent diff count: {key}")
    if returncode != (1 if drifted else 0):
        raise ValueError("child exit status disagrees with complete diff result")
    details = report["details"]
    if not isinstance(details, list) or len(details) != len(drifted):
        raise ValueError("incomplete discrepancy details")
    seen = set()
    by_kind = {}
    for detail in details:
        if not isinstance(detail, dict) or set(detail) != {
            "fixture_id", "kind", "difference_paths", "outcome_classes", "value_differences"
        }:
            raise ValueError("invalid discrepancy detail")
        fixture_id = detail["fixture_id"]
        if not isinstance(fixture_id, str) or fixture_id not in drifted or fixture_id in seen:
            raise ValueError("unexpected or duplicated discrepancy ID")
        seen.add(fixture_id)
        paths = detail["difference_paths"]
        values = detail["value_differences"]
        if (not isinstance(paths, list) or not paths
                or not all(isinstance(p, str) and p for p in paths)
                or len(paths) != len(set(paths)) or not isinstance(values, list)
                or not all(isinstance(v, dict) and set(v) == {"path", "oracle", "candidate"}
                           for v in values)
                or [v["path"] for v in values] != paths):
            raise ValueError("incomplete discrepancy values")
        kind = detail["kind"]
        if not isinstance(kind, str) or not kind:
            raise ValueError("invalid discrepancy kind")
        by_kind[kind] = by_kind.get(kind, 0) + 1
    if (not isinstance(report["by_kind"], dict)
            or not all(type(v) is int for v in report["by_kind"].values())
            or report["by_kind"] != by_kind):
        raise ValueError("inconsistent discrepancy kinds")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(LAB_ROOT / "profiles" / "sympy-1.14.0-cpython-r2-corpus.toml"))
    ap.add_argument("--candidate-python", default="/data/projects/frankensympy/.venv-conformance/bin/python3")
    args = ap.parse_args()

    try:
        profile = load_profile(Path(args.profile))
        binding = diff_input_binding(profile)
    except (ValueError, OSError, SystemExit) as exc:
        print(f"corpus_gate: invalid inputs: {exc}", file=sys.stderr)
        return 2
    profile_dir = ARTIFACT_ROOT / profile["profile_id"]
    ledger_path = profile_dir / "ledger.json"

    try:
        diff = subprocess.run(
            [sys.executable, str(LAB_ROOT / "capture.py"), "diff", args.profile,
             "--candidate-python", args.candidate_python],
            capture_output=True, text=True,
            timeout=30 + 120 * len(profile["inventory"]["fixtures"]))
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"corpus_gate: diff execution failed: {exc}", file=sys.stderr)
        return 2
    try:
        report = validate_report(parse_report(diff.stdout), diff.returncode, binding)
        if diff_input_binding(load_profile(Path(args.profile))) != binding:
            raise ValueError("inputs changed during diff execution")
    except (ValueError, OSError, SystemExit) as exc:
        print(f"corpus_gate: invalid diff result: {exc}", file=sys.stderr)
        print(diff.stdout[-2000:], diff.stderr[-2000:], file=sys.stderr)
        return 2

    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"records": []}
    ledgered = {
        (r["fixture_id"], tuple(sorted(r["difference_paths"]))): r
        for r in ledger.get("records", []) if r["status"] == "open"
        and r.get("profile_id") == profile["profile_id"]
        and r.get("comparator") == report["comparator"]
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
                "profile_id": profile["profile_id"],
                "fixture_id": fixture_id,
                "comparator": report.get("comparator", "construction_only"),
                "difference_paths": paths,
            })

    newly = [u for u in unledgered]
    print(json.dumps({
        "gate": "lab-corpus",
        "scope": "development_construction_drift",
        "certified": False,
        "admitted": report.get("admitted"),
        "drift_total": len(observed),
        "ledgered_open": len(observed) - len(newly),
        "unledgered": len(newly),
        "unledgered_records": newly,
    }, indent=1, sort_keys=True))

    if newly:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ledger.get("records", [])
        seen = {
            (r["fixture_id"], tuple(sorted(r.get("difference_paths", []))))
            for r in existing
            if r.get("profile_id") == profile["profile_id"]
            and r.get("comparator") == report["comparator"]
        }
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
