"""Warning-class identity records, separate from golden observation envelopes.

Golden envelopes cannot grow a warnings field without a new profile. This
lane compares only warning *class* identity (module + name) and construction
outcome. Message text is outside the contract.
"""

from __future__ import annotations

WARNING_KIND = "warning_observation"
WARNING_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "fixture_id",
        "side",
        "construction_outcome",
        "warnings",
    }
)
CONSTRUCTION_OUTCOMES = frozenset({"returned", "raised", "refused"})


def warning_class_identity(category: type) -> dict[str, str]:
    return {"module": category.__module__, "name": category.__name__}


def unique_warning_classes(caught: list) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in caught:
        record = warning_class_identity(item.category)
        if record not in out:
            out.append(record)
    return out


def make_warning_record(
    *,
    profile_id: str,
    fixture_id: str,
    side: str,
    construction_outcome: str,
    warnings: list[dict[str, str]],
) -> dict:
    return {
        "schema_version": 1,
        "kind": WARNING_KIND,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "side": side,
        "construction_outcome": construction_outcome,
        "warnings": warnings,
    }


def validate_warning_record(
    record: object,
    profile: dict,
    *,
    expected_side: str,
    expected_id: str,
) -> None:
    if not isinstance(record, dict):
        raise TypeError("warning record must be an object")
    if set(record) != WARNING_RECORD_KEYS:
        raise ValueError("warning record keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != WARNING_KIND:
        raise ValueError("warning record schema or kind mismatch")
    if record["profile_id"] != profile["profile_id"]:
        raise ValueError("warning record profile mismatch")
    if record["side"] != expected_side:
        raise ValueError(f"wrong warning observation side: {record['side']!r}")
    if record["fixture_id"] != expected_id:
        raise ValueError("warning record fixture_id mismatch")
    if record["construction_outcome"] not in CONSTRUCTION_OUTCOMES:
        raise ValueError("unknown construction_outcome")
    warnings = record["warnings"]
    if not isinstance(warnings, list):
        raise TypeError("warnings must be a list")
    seen: list[dict[str, str]] = []
    for entry in warnings:
        if not isinstance(entry, dict) or set(entry) != {"module", "name"}:
            raise ValueError("warning entries must be {module, name}")
        if not isinstance(entry["module"], str) or not entry["module"]:
            raise ValueError("warning module must be a non-empty string")
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise ValueError("warning name must be a non-empty string")
        if entry in seen:
            raise ValueError("warning class identities must be unique")
        seen.append(entry)


def diff_warning_records(oracle: list[dict], candidate: list[dict]) -> list[dict]:
    """Name warning-class and construction-outcome mismatches. Never certifies."""
    oracle_by_id = {record["fixture_id"]: record for record in oracle}
    candidate_by_id = {record["fixture_id"]: record for record in candidate}
    details = []
    for fixture_id in sorted(set(oracle_by_id) | set(candidate_by_id)):
        left = oracle_by_id.get(fixture_id)
        right = candidate_by_id.get(fixture_id)
        if left is None or right is None:
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "coverage_gap",
                    "oracle": None if left is None else left["warnings"],
                    "candidate": None if right is None else right["warnings"],
                }
            )
            continue
        if left["construction_outcome"] != right["construction_outcome"]:
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "construction_outcome_mismatch",
                    "oracle": left["construction_outcome"],
                    "candidate": right["construction_outcome"],
                }
            )
            continue
        if left["warnings"] != right["warnings"]:
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "warning_identity_drift",
                    "oracle": left["warnings"],
                    "candidate": right["warnings"],
                }
            )
    return details
