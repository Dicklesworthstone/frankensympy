#!/usr/bin/env python3
"""Comparator registry for the FrankenSymPy conformance laboratory.

Every fixture names a comparator from the immutable profile registry;
selection is based on the API contract, never on which choice makes a case
pass (docs/CONFORMANCE_AND_BENCHMARKING.md section 8). Unknown comparator ids
fail closed.

A comparison returns minimized differences as records of
``{path, oracle, candidate}`` where ``path`` is a dotted path into the
observation envelope. An empty list means the pair is accepted.
"""

from __future__ import annotations

import hashlib
import json

# Registry ids mirror tools/conformance-lab/profiles/*.toml [comparators].
REGISTRY: dict[str, dict] = {
    # Full exact-surface comparison: every observed field must match.
    "exact_surface": {"fields": None},
    # Construction contract only: identity and structure, not printers,
    # hashes, or pickle bytes.
    "construction_only": {
        "fields": ["type", "module", "args_repr", "func"],
    },
}


def _walk(prefix: str, left, right, out: list[dict]) -> None:
    if type(left) is not type(right):
        out.append({"path": prefix, "oracle": left, "candidate": right})
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                out.append(
                    {
                        "path": f"{prefix}.{key}",
                        "oracle": "<missing>" if key not in left else left[key],
                        "candidate": "<missing>" if key not in right else right[key],
                    }
                )
            else:
                _walk(f"{prefix}.{key}", left[key], right[key], out)
    elif left != right:
        out.append({"path": prefix, "oracle": left, "candidate": right})


def diff_envelopes(oracle: dict, candidate: dict, comparator_id: str) -> list[dict]:
    """Diffs one envelope pair under the named registered comparator."""
    if comparator_id not in REGISTRY:
        raise KeyError(f"unknown comparator id: {comparator_id!r}")
    spec = REGISTRY[comparator_id]

    differences: list[dict] = []
    for key in ("fixture_id", "profile_id", "outcome_class"):
        if oracle.get(key) != candidate.get(key):
            entry: dict = {
                "path": f"{key}",
                "oracle": oracle.get(key),
                "candidate": candidate.get(key),
            }
            if key == "outcome_class":
                entry["path"] = "outcome_class"
            differences.append(entry)

    o_obs = oracle.get("observations", {})
    c_obs = candidate.get("observations", {})
    fields = spec["fields"]
    if spec["fields"] is None:
        # Exact surface: every observed field plus the full environment
        # fingerprint must match.
        _walk("observations", o_obs, c_obs, differences)
        _walk(
            "environment",
            oracle.get("environment"),
            candidate.get("environment"),
            differences,
        )
        return differences

    for key in spec["fields"]:
        if o_obs.get(key) != c_obs.get(key):
            differences.append(
                {
                    "path": f"observations.{key}",
                    "oracle": o_obs.get(key),
                    "candidate": c_obs.get(key),
                }
            )
    return differences


def is_valid_discrepancy(record: dict) -> tuple[bool, str]:
    """Minimal fail-closed validation against schema/discrepancy.schema.json."""
    import re

    required = {
        "schema_version",
        "discrepancy_id",
        "status",
        "severity",
        "profile_id",
        "fixture_id",
        "comparator",
        "differences",
    }
    missing = required - record.keys()
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    unknown = record.keys() - required - {
        "outcome_classes",
        "environment",
        "affected_stage",
        "affected_claim",
        "closure_test",
        "owner",
        "created_at_utc",
    }
    if unknown:
        return False, f"unknown fields rejected: {sorted(unknown)}"
    if record["schema_version"] != 1:
        return False, "unsupported schema_version"
    if not re.fullmatch(r"disc-[a-z0-9-]+", record["discrepancy_id"]):
        return False, "bad discrepancy_id"
    if record["status"] not in {
        "open",
        "blocked",
        "fix_landed_unverified",
        "closed_verified",
    }:
        return False, f"bad status: {record['status']!r}"
    if record["severity"] not in {"object", "mathematical", "runtime", "security"}:
        return False, f"bad severity: {record['severity']!r}"
    if record["comparator"] not in REGISTRY:
        return False, f"unregistered comparator: {record['comparator']!r}"
    if not record["differences"]:
        return False, "empty difference set"
    path_re = re.compile(
        r"^observations(\.[A-Za-z0-9_\[\]]+)+$"
        r"|^environment(\.[A-Za-z0-9_]+)+$"
        r"|^outcome_class$|^fixture_id$|^profile_id$"
    )
    for diff in record["differences"]:
        if not path_re.fullmatch(diff.get("path", "")):
            return False, f"bad difference path: {diff.get('path')!r}"
    return True, ""


def discrepancy_id(differences: list[dict]) -> str:
    """Content-derived stable id for a difference set."""
    canonical = json.dumps(differences, sort_keys=True, separators=(",", ":"))
    return "disc-" + hashlib.sha256(canonical.encode()).hexdigest()[:12]
