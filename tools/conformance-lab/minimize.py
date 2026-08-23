#!/usr/bin/env python3
"""Discrepancy minimizer for the FrankenSymPy conformance laboratory.

Pairs oracle-side and candidate-side NDJSON observation envelopes by position,
diffs them under a registered comparator (default: the profile's
exact_surface), and emits one discrepancy record per mismatching envelope,
conforming to tools/conformance-lab/schema/discrepancy.schema.json.

Usage:
  minimize.py <oracle.ndjson> <candidate.ndjson> \
      [--comparator exact_surface|construction_only] [--out FILE] [--severity object]

Exit codes: 0 = no discrepancies, 1 = discrepancies emitted (or written),
2 = misuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparators import discrepancy_id, diff_envelopes, is_valid_discrepancy  # noqa: E402


def load_envelopes(path: Path) -> list[dict]:
    envelopes = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                envelopes.append(json.loads(line))
    return envelopes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--comparator", default="exact_surface")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--severity",
        default="object",
        choices=["object", "mathematical", "runtime", "security"],
    )
    parser.add_argument("--profile-id", default="sympy-1.14.0-cpython")
    args = parser.parse_args()

    oracle_envs = load_envelopes(args.oracle)
    candidate_envs = load_envelopes(args.candidate)

    records: list[dict] = []
    pair_count = min(len(oracle_envs), len(candidate_envs))
    if len(oracle_envs) != len(candidate_envs):
        records.append(
            {
                "schema_version": 1,
                "discrepancy_id": "disc-envelope-count-mismatch",
                "status": "open",
                "severity": args.severity,
                "profile_id": args.profile_id,
                "fixture_id": f"count:{len(oracle_envs)}vs{len(candidate_envs)}",
                "comparator": args.comparator,
                "differences": [
                    {
                        "path": "observations.type",
                        "oracle": f"{len(oracle_envs)} envelopes",
                        "candidate": f"{len(candidate_envs)} envelopes",
                    }
                ],
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
        )
    for i in range(pair_count):
        o_env, c_env = oracle_envs[i], candidate_envs[i]
        differences = diff_envelopes(o_env, c_env, args.comparator)
        if not differences:
            continue
        record = {
            "schema_version": 1,
            "discrepancy_id": discrepancy_id(differences),
            "status": "open",
            "severity": args.severity,
            "profile_id": o_env.get("profile_id", args.profile_id),
            "fixture_id": o_env.get("fixture_id", f"index-{i}"),
            "comparator": args.comparator,
            "environment": o_env.get("environment"),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        if o_env.get("outcome_class") != c_env.get("outcome_class"):
            record["outcome_classes"] = {
                "oracle": o_env.get("outcome_class"),
                "candidate": c_env.get("outcome_class"),
            }
        record["differences"] = differences
        records.append(record)

    # Fail closed: every emitted record must validate against the schema
    # contract before it is allowed to exist on disk.
    for record in records:
        ok, reason = is_valid_discrepancy(record)
        if not ok:
            print(f"FAIL: invalid discrepancy record: {reason}", file=sys.stderr)
            return 2

    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote {len(records)} discrepancy record(s) to {args.out}")
    else:
        sys.stdout.write(payload)

    print(f"compared {pair_count} envelope pair(s); {len(records)} discrepancy(ies)")
    return 1 if records else 0


if __name__ == "__main__":
    sys.exit(main())
