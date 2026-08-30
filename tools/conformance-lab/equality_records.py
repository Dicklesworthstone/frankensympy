"""Equality/hash twin records. Not a golden field and cannot certify.

Construct the fixture twice. Compare whether the twins compare equal, whether
their hashes agree, and whether they are the same object. Actual hash values
stay on exact_surface goldens.
"""

from __future__ import annotations

EQUALITY_KIND = "equality_observation"
EQUALITY_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "fixture_id",
        "side",
        "construction_outcome",
        "equal_to_twin",
        "hashes_agree",
        "is_same_object",
        "probe_error",
    }
)
PROBE_ERR_KEYS = frozenset({"error_class", "message_head"})


def make_equality_record(
    *,
    profile_id: str,
    fixture_id: str,
    side: str,
    construction_outcome: str,
    equal_to_twin: bool | None,
    hashes_agree: bool | None,
    is_same_object: bool | None,
    probe_error: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "kind": EQUALITY_KIND,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "side": side,
        "construction_outcome": construction_outcome,
        "equal_to_twin": equal_to_twin,
        "hashes_agree": hashes_agree,
        "is_same_object": is_same_object,
        "probe_error": probe_error,
    }


def validate_equality_record(
    record: object, profile: dict, *, expected_side: str, expected_id: str
) -> None:
    if not isinstance(record, dict) or set(record) != EQUALITY_RECORD_KEYS:
        raise ValueError("equality record keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != EQUALITY_KIND:
        raise ValueError("equality record schema or kind mismatch")
    if record["profile_id"] != profile["profile_id"]:
        raise ValueError("equality record profile mismatch")
    if record["side"] != expected_side or record["fixture_id"] != expected_id:
        raise ValueError("equality record side or fixture_id mismatch")
    if record["construction_outcome"] not in {"returned", "raised", "refused"}:
        raise ValueError("unknown equality construction_outcome")
    error = record["probe_error"]
    if error is not None:
        if not isinstance(error, dict) or set(error) != PROBE_ERR_KEYS:
            raise ValueError("probe_error must be {error_class, message_head}")
        if (
            record["equal_to_twin"] is not None
            or record["hashes_agree"] is not None
            or record["is_same_object"] is not None
        ):
            raise ValueError("failed equality probe cannot carry twin booleans")
        return
    if record["construction_outcome"] != "returned":
        if (
            record["equal_to_twin"] is not None
            or record["hashes_agree"] is not None
            or record["is_same_object"] is not None
        ):
            raise ValueError("non-returned equality record cannot carry twin booleans")
        return
    for key in ("equal_to_twin", "hashes_agree", "is_same_object"):
        if not isinstance(record[key], bool):
            raise TypeError(f"{key} must be boolean")


def equality_identity(record: dict) -> dict:
    if record["construction_outcome"] != "returned":
        return {"construction_outcome": record["construction_outcome"]}
    if record["probe_error"] is not None:
        return {
            "construction_outcome": "returned",
            "probe_error_class": record["probe_error"]["error_class"],
        }
    return {
        "construction_outcome": "returned",
        "equal_to_twin": record["equal_to_twin"],
        "hashes_agree": record["hashes_agree"],
        "is_same_object": record["is_same_object"],
        "hash_respects_eq": (not record["equal_to_twin"]) or record["hashes_agree"],
    }


def diff_equality_records(oracle: list[dict], candidate: list[dict]) -> list[dict]:
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
                    "oracle": None if left is None else equality_identity(left),
                    "candidate": None if right is None else equality_identity(right),
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
        if equality_identity(left) != equality_identity(right):
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "equality_identity_drift",
                    "oracle": equality_identity(left),
                    "candidate": equality_identity(right),
                }
            )
    return details
