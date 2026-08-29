"""Cross-process pickle round-trip records. Not a golden field and cannot certify.

Dump digests already live on observation envelopes. This lane dumps in one
process and loads in a second process of the same interpreter family, then
compares restored type/module identity. Message text is outside the contract.
"""

from __future__ import annotations

PICKLE_PROTOCOL = 4
DUMP_KIND = "pickle_dump"
RESTORE_KIND = "pickle_restore"
DUMP_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "fixture_id",
        "side",
        "construction_outcome",
        "protocol",
        "pickle_sha256",
        "pickle_b64",
        "dump_error",
    }
)
RESTORE_KEYS_RETURNED = frozenset(
    {"schema_version", "kind", "fixture_id", "side", "protocol", "status", "type", "module"}
)
RESTORE_KEYS_RAISED = frozenset(
    {
        "schema_version",
        "kind",
        "fixture_id",
        "side",
        "protocol",
        "status",
        "error_class",
        "message_head",
    }
)


def make_dump_record(
    *,
    profile_id: str,
    fixture_id: str,
    side: str,
    construction_outcome: str,
    pickle_sha256: str | None,
    pickle_b64: str | None,
    dump_error: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "kind": DUMP_KIND,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "side": side,
        "construction_outcome": construction_outcome,
        "protocol": PICKLE_PROTOCOL,
        "pickle_sha256": pickle_sha256,
        "pickle_b64": pickle_b64,
        "dump_error": dump_error,
    }


def make_restore_record(
    *,
    fixture_id: str,
    side: str,
    status: str,
    restored_type: str | None = None,
    module: str | None = None,
    error_class: str | None = None,
    message_head: str | None = None,
) -> dict:
    record = {
        "schema_version": 1,
        "kind": RESTORE_KIND,
        "fixture_id": fixture_id,
        "side": side,
        "protocol": PICKLE_PROTOCOL,
        "status": status,
    }
    if status == "returned":
        record["type"] = restored_type
        record["module"] = module
    else:
        record["error_class"] = error_class
        record["message_head"] = message_head
    return record


def validate_dump_record(record: object, profile: dict, *, expected_side: str, expected_id: str) -> None:
    if not isinstance(record, dict) or set(record) != DUMP_KEYS:
        raise ValueError("pickle dump keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != DUMP_KIND:
        raise ValueError("pickle dump schema or kind mismatch")
    if record["profile_id"] != profile["profile_id"]:
        raise ValueError("pickle dump profile mismatch")
    if record["side"] != expected_side or record["fixture_id"] != expected_id:
        raise ValueError("pickle dump side or fixture_id mismatch")
    if record["protocol"] != PICKLE_PROTOCOL:
        raise ValueError("pickle dump protocol is not 4")
    if record["construction_outcome"] not in {"returned", "raised", "refused"}:
        raise ValueError("unknown pickle construction_outcome")
    error = record["dump_error"]
    if error is not None:
        if not isinstance(error, dict) or set(error) != {"error_class", "message_head"}:
            raise ValueError("dump_error must be {error_class, message_head}")
        if record["pickle_b64"] is not None or record["pickle_sha256"] is not None:
            raise ValueError("failed dump cannot carry pickle bytes")
        return
    if record["construction_outcome"] != "returned":
        if record["pickle_b64"] is not None or record["pickle_sha256"] is not None:
            raise ValueError("non-returned dump cannot carry pickle bytes")
        return
    if not isinstance(record["pickle_sha256"], str) or len(record["pickle_sha256"]) != 64:
        raise ValueError("returned dump sha256 must be 64 hex characters")
    if not isinstance(record["pickle_b64"], str) or not record["pickle_b64"]:
        raise ValueError("returned dump must carry pickle_b64")


def validate_restore_record(record: object, *, expected_side: str, expected_id: str) -> None:
    if not isinstance(record, dict):
        raise TypeError("pickle restore record must be an object")
    status = record.get("status")
    expected_keys = RESTORE_KEYS_RETURNED if status == "returned" else RESTORE_KEYS_RAISED
    if set(record) != expected_keys:
        raise ValueError("pickle restore keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != RESTORE_KIND:
        raise ValueError("pickle restore schema or kind mismatch")
    if record["side"] != expected_side or record["fixture_id"] != expected_id:
        raise ValueError("pickle restore side or fixture_id mismatch")
    if record["protocol"] != PICKLE_PROTOCOL:
        raise ValueError("pickle restore protocol is not 4")
    if status == "returned":
        if not isinstance(record["type"], str) or not record["type"]:
            raise ValueError("restored type must be a non-empty string")
        if not isinstance(record["module"], str) or not record["module"]:
            raise ValueError("restored module must be a non-empty string")
        return
    if status != "raised":
        raise ValueError("unknown pickle restore status")
    if not isinstance(record["error_class"], str) or not record["error_class"]:
        raise ValueError("restore error_class must be a non-empty string")


def restored_identity(record: dict) -> dict:
    if record["status"] == "returned":
        return {"status": "returned", "type": record["type"], "module": record["module"]}
    return {"status": "raised", "error_class": record["error_class"]}


def combine_roundtrip(dump: dict, restore: dict | None) -> dict:
    """Drop pickle bytes; keep construction/dump/restore identity only."""
    return {
        "fixture_id": dump["fixture_id"],
        "side": dump["side"],
        "construction_outcome": dump["construction_outcome"],
        "dump_error_class": None
        if dump["dump_error"] is None
        else dump["dump_error"]["error_class"],
        "pickle_sha256": dump["pickle_sha256"],
        "restore": None if restore is None else restored_identity(restore),
    }


def roundtrip_identity(record: dict) -> dict:
    return {
        "construction_outcome": record["construction_outcome"],
        "dump_error_class": record["dump_error_class"],
        "restore": record["restore"],
    }


def diff_pickle_roundtrips(oracle: list[dict], candidate: list[dict]) -> list[dict]:
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
                    "oracle": None if left is None else roundtrip_identity(left),
                    "candidate": None if right is None else roundtrip_identity(right),
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
        if roundtrip_identity(left) != roundtrip_identity(right):
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "pickle_restore_identity_drift",
                    "oracle": roundtrip_identity(left),
                    "candidate": roundtrip_identity(right),
                }
            )
    return details
