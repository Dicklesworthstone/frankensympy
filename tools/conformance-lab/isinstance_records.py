"""isinstance/issubclass records against the class imported from __module__.

isinstance(obj, type(obj)) is tautological. The live check is whether the
object is an instance of getattr(import_module(__module__), __name__). That
is the pickle-capable class identity. Not a golden field; cannot certify.
"""

from __future__ import annotations

ISINSTANCE_KIND = "isinstance_observation"
ISINSTANCE_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "fixture_id",
        "side",
        "construction_outcome",
        "isinstance_type",
        "issubclass_type",
        "module_class_importable",
        "isinstance_module_class",
        "issubclass_module_class",
        "type_is_module_class",
        "probe_error",
    }
)
PROBE_ERR_KEYS = frozenset({"error_class", "message_head"})


def make_isinstance_record(
    *,
    profile_id: str,
    fixture_id: str,
    side: str,
    construction_outcome: str,
    isinstance_type: bool | None,
    issubclass_type: bool | None,
    module_class_importable: bool | None,
    isinstance_module_class: bool | None,
    issubclass_module_class: bool | None,
    type_is_module_class: bool | None,
    probe_error: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "kind": ISINSTANCE_KIND,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "side": side,
        "construction_outcome": construction_outcome,
        "isinstance_type": isinstance_type,
        "issubclass_type": issubclass_type,
        "module_class_importable": module_class_importable,
        "isinstance_module_class": isinstance_module_class,
        "issubclass_module_class": issubclass_module_class,
        "type_is_module_class": type_is_module_class,
        "probe_error": probe_error,
    }


def validate_isinstance_record(
    record: object, profile: dict, *, expected_side: str, expected_id: str
) -> None:
    if not isinstance(record, dict) or set(record) != ISINSTANCE_RECORD_KEYS:
        raise ValueError("isinstance record keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != ISINSTANCE_KIND:
        raise ValueError("isinstance record schema or kind mismatch")
    if record["profile_id"] != profile["profile_id"]:
        raise ValueError("isinstance record profile mismatch")
    if record["side"] != expected_side or record["fixture_id"] != expected_id:
        raise ValueError("isinstance record side or fixture_id mismatch")
    if record["construction_outcome"] not in {"returned", "raised", "refused"}:
        raise ValueError("unknown isinstance construction_outcome")
    bool_keys = (
        "isinstance_type",
        "issubclass_type",
        "module_class_importable",
        "isinstance_module_class",
        "issubclass_module_class",
        "type_is_module_class",
    )
    error = record["probe_error"]
    if error is not None:
        if not isinstance(error, dict) or set(error) != PROBE_ERR_KEYS:
            raise ValueError("probe_error must be {error_class, message_head}")
    if record["construction_outcome"] != "returned":
        if any(record[key] is not None for key in bool_keys) or error is not None:
            raise ValueError("non-returned isinstance record cannot carry class checks")
        return
    for key in ("isinstance_type", "issubclass_type", "module_class_importable"):
        if not isinstance(record[key], bool):
            raise TypeError(f"{key} must be boolean")
    if not record["module_class_importable"]:
        if (
            record["isinstance_module_class"] is not None
            or record["issubclass_module_class"] is not None
            or record["type_is_module_class"] is not None
        ):
            raise ValueError("failed module-class import cannot carry isinstance booleans")
        return
    for key in ("isinstance_module_class", "issubclass_module_class", "type_is_module_class"):
        if not isinstance(record[key], bool):
            raise TypeError(f"{key} must be boolean")


def isinstance_identity(record: dict) -> dict:
    if record["construction_outcome"] != "returned":
        return {"construction_outcome": record["construction_outcome"]}
    identity = {
        "construction_outcome": "returned",
        "isinstance_type": record["isinstance_type"],
        "issubclass_type": record["issubclass_type"],
        "module_class_importable": record["module_class_importable"],
        "isinstance_module_class": record["isinstance_module_class"],
        "issubclass_module_class": record["issubclass_module_class"],
        "type_is_module_class": record["type_is_module_class"],
    }
    if record["probe_error"] is not None:
        identity["probe_error_class"] = record["probe_error"]["error_class"]
    return identity


def diff_isinstance_records(oracle: list[dict], candidate: list[dict]) -> list[dict]:
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
                    "oracle": None if left is None else isinstance_identity(left),
                    "candidate": None if right is None else isinstance_identity(right),
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
        if isinstance_identity(left) != isinstance_identity(right):
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "isinstance_identity_drift",
                    "oracle": isinstance_identity(left),
                    "candidate": isinstance_identity(right),
                }
            )
    return details
