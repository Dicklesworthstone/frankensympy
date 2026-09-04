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
    # Construction contract only: returned-object identity and structure, or
    # raised-exception identity. Printer output, message wording, hashes, and
    # pickle bytes are deliberately outside this comparator's contract.
    "construction_only": {
        "fields": ["type", "module", "args_repr", "func"],
        "raised_fields": ["exception_module", "exception_type"],
    },
}


def _walk(prefix: str, left, right, out: list[dict]) -> None:
    # JSON booleans are Python integers by inheritance, so isinstance would
    # wrongly treat `true` and `1` as the same observation type.
    if left.__class__ is not right.__class__:
        out.append({"path": prefix, "oracle": left, "candidate": right})
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                out.append(
                    {
                        "path": f"{prefix}.{key}",
                        "oracle": left.get(key, "<missing>"),
                        "candidate": right.get(key, "<missing>"),
                    }
                )
            else:
                _walk(f"{prefix}.{key}", left[key], right[key], out)
    elif left != right:
        out.append({"path": prefix, "oracle": left, "candidate": right})


def _diff_required_fields(
    prefix: str, left: dict, right: dict, fields: list[str], out: list[dict]
) -> None:
    """Compare registered fields and reject an omitted observation on either side."""
    for key in fields:
        path = f"{prefix}.{key}"
        if key not in left or key not in right:
            out.append(
                {
                    "path": path,
                    "oracle": left.get(key, "<missing>"),
                    "candidate": right.get(key, "<missing>"),
                }
            )
            continue
        _walk(path, left[key], right[key], out)


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
    if spec["fields"] is None:
        # Exact surface: every observed field plus the identity-relevant
        # environment fields must match. Machine-local provenance (host
        # kernel string in `platform`, absolute `sympy_path`) is capture
        # provenance, not a compatibility dimension: goldens move between
        # hosts/kernel updates and must stay valid (comparator refinement
        # recorded in fra-conformance-corpus-200-b75; sympy_version, python,
        # implementation, locale and pinned env remain compared).
        provenance = {"platform", "sympy_path"}
        o_env = {k: v for k, v in oracle.get("environment", {}).items() if k not in provenance}
        c_env = {k: v for k, v in candidate.get("environment", {}).items() if k not in provenance}
        _walk("observations", o_obs, c_obs, differences)
        _walk("environment", o_env, c_env, differences)
        return differences

    outcome_class = oracle.get("outcome_class")
    if outcome_class != candidate.get("outcome_class"):
        return differences
    if outcome_class == "returned":
        fields = spec["fields"]
    elif outcome_class == "raised":
        fields = spec["raised_fields"]
    else:
        raise ValueError(
            f"{comparator_id} has no registered observation policy for "
            f"outcome_class={outcome_class!r}"
        )
    _diff_required_fields("observations", o_obs, c_obs, fields, differences)
    return differences


def is_valid_discrepancy(record: dict) -> tuple[bool, str]:
    """Minimal fail-closed validation against schema/discrepancy.schema.json."""
    import re

    if not isinstance(record, dict):
        return False, "record must be an object"
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return False, f"record is not strict JSON: {exc}"
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
    unknown = (
        record.keys()
        - required
        - {
            "outcome_classes",
            "environment",
            "affected_stage",
            "affected_claim",
            "closure_test",
            "owner",
            "created_at_utc",
        }
    )
    if unknown:
        return False, f"unknown fields rejected: {sorted(unknown)}"
    if record["schema_version"] != 1:
        return False, "unsupported schema_version"
    if not isinstance(record["discrepancy_id"], str) or not re.fullmatch(
        r"disc-[a-z0-9-]+", record["discrepancy_id"]
    ):
        return False, "bad discrepancy_id"
    if not isinstance(record["status"], str) or record["status"] not in {
        "open",
        "blocked",
        "fix_landed_unverified",
        "closed_verified",
    }:
        return False, f"bad status: {record['status']!r}"
    if not isinstance(record["severity"], str) or record["severity"] not in {
        "object",
        "mathematical",
        "runtime",
        "security",
    }:
        return False, f"bad severity: {record['severity']!r}"
    if (
        not isinstance(record["comparator"], str)
        or record["comparator"] not in REGISTRY
    ):
        return False, f"unregistered comparator: {record['comparator']!r}"
    for field in ("profile_id", "fixture_id"):
        if not isinstance(record[field], str) or not record[field]:
            return False, f"{field} must be a non-empty string"
    if not isinstance(record["differences"], list) or not record["differences"]:
        return False, "empty difference set"
    path_re = re.compile(
        r"^observations(\.[A-Za-z0-9_\[\]]+)+$"
        r"|^environment(\.[A-Za-z0-9_]+)+$"
        r"|^outcome_class$|^fixture_id$|^profile_id$"
    )
    for diff in record["differences"]:
        if not isinstance(diff, dict):
            return False, "difference entries must be objects"
        if set(diff) != {"path", "oracle", "candidate"}:
            return False, f"difference keys invalid: {sorted(diff)}"
        path = diff.get("path")
        if not isinstance(path, str) or not path_re.fullmatch(path):
            return False, f"bad difference path: {path!r}"
    expected_id = discrepancy_id(
        profile_id=record["profile_id"],
        fixture_id=record["fixture_id"],
        comparator=record["comparator"],
        differences=record["differences"],
    )
    if record["discrepancy_id"] != expected_id:
        return False, "discrepancy_id does not match canonical content"
    outcome_classes = record.get("outcome_classes")
    if outcome_classes is not None and (
        not isinstance(outcome_classes, dict)
        or set(outcome_classes) != {"oracle", "candidate"}
        or not all(isinstance(value, str) for value in outcome_classes.values())
    ):
        return (
            False,
            "outcome_classes must contain exactly oracle and candidate strings",
        )
    if "environment" in record and not isinstance(record["environment"], dict):
        return False, "environment must be an object"
    for field in (
        "affected_stage",
        "affected_claim",
        "closure_test",
        "owner",
        "created_at_utc",
    ):
        if field in record and not isinstance(record[field], str):
            return False, f"{field} must be a string"
    return True, ""


def discrepancy_id(
    *, profile_id: str, fixture_id: str, comparator: str, differences: list[dict]
) -> str:
    """Content-derived stable id bound to the full comparison claim."""
    identity = {
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "comparator": comparator,
        "differences": differences,
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return "disc-" + hashlib.sha256(canonical.encode()).hexdigest()
