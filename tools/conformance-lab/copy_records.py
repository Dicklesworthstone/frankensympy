"""Copy/deepcopy identity records. Not a golden field and cannot certify.

copy/deepcopy must not canonicalize held forms. This lane compares type/module
identity, whether the copy is the original object, and whether deepcopy
preserves args_repr. Exact args_repr strings across oracle vs candidate stay
on construction_only.
"""

from __future__ import annotations

COPY_KIND = "copy_observation"
COPY_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "fixture_id",
        "side",
        "construction_outcome",
        "original",
        "copy",
        "deepcopy",
    }
)
SURFACE_KEYS = frozenset({"type", "module", "args_repr"})
COPY_OK_KEYS = frozenset({"type", "module", "args_repr", "is_original"})
COPY_ERR_KEYS = frozenset({"error_class", "message_head"})


def surface_of(obj) -> dict:
    return {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
        "args_repr": [repr(arg) for arg in getattr(obj, "args", ())],
    }


def copy_ok(obj, copied) -> dict:
    surface = surface_of(copied)
    surface["is_original"] = copied is obj
    return surface


def copy_error(exc: BaseException) -> dict:
    return {
        "error_class": type(exc).__module__ + "." + type(exc).__name__,
        "message_head": str(exc)[:200],
    }


def make_copy_record(
    *,
    profile_id: str,
    fixture_id: str,
    side: str,
    construction_outcome: str,
    original: dict | None,
    copied: dict | None,
    deepcopied: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "kind": COPY_KIND,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "side": side,
        "construction_outcome": construction_outcome,
        "original": original,
        "copy": copied,
        "deepcopy": deepcopied,
    }


def _validate_surface(value: object, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != SURFACE_KEYS:
        raise ValueError(f"{label} surface keys do not match schema version 1")
    if not isinstance(value["type"], str) or not value["type"]:
        raise ValueError(f"{label} type must be a non-empty string")
    if not isinstance(value["module"], str) or not value["module"]:
        raise ValueError(f"{label} module must be a non-empty string")
    if not isinstance(value["args_repr"], list) or not all(
        isinstance(item, str) for item in value["args_repr"]
    ):
        raise ValueError(f"{label} args_repr must be a list of strings")


def _validate_copy_result(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if set(value) == COPY_ERR_KEYS:
        if not isinstance(value["error_class"], str) or not value["error_class"]:
            raise ValueError(f"{label} error_class must be a non-empty string")
        return
    if set(value) != COPY_OK_KEYS:
        raise ValueError(f"{label} keys do not match schema version 1")
    _validate_surface(
        {"type": value["type"], "module": value["module"], "args_repr": value["args_repr"]},
        label=label,
    )
    if not isinstance(value["is_original"], bool):
        raise TypeError(f"{label} is_original must be boolean")


def validate_copy_record(
    record: object, profile: dict, *, expected_side: str, expected_id: str
) -> None:
    if not isinstance(record, dict) or set(record) != COPY_RECORD_KEYS:
        raise ValueError("copy record keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != COPY_KIND:
        raise ValueError("copy record schema or kind mismatch")
    if record["profile_id"] != profile["profile_id"]:
        raise ValueError("copy record profile mismatch")
    if record["side"] != expected_side or record["fixture_id"] != expected_id:
        raise ValueError("copy record side or fixture_id mismatch")
    if record["construction_outcome"] not in {"returned", "raised", "refused"}:
        raise ValueError("unknown copy construction_outcome")
    if record["construction_outcome"] != "returned":
        if record["original"] is not None or record["copy"] is not None or record["deepcopy"] is not None:
            raise ValueError("non-returned copy record cannot carry surfaces")
        return
    _validate_surface(record["original"], label="original")
    _validate_copy_result(record["copy"], label="copy")
    _validate_copy_result(record["deepcopy"], label="deepcopy")


def copy_identity(record: dict) -> dict:
    if record["construction_outcome"] != "returned":
        return {"construction_outcome": record["construction_outcome"]}
    original = record["original"]
    copied = record["copy"]
    deep = record["deepcopy"]
    def side_view(value: dict) -> dict:
        if "error_class" in value:
            return {"error_class": value["error_class"]}
        return {
            "type": value["type"],
            "module": value["module"],
            "is_original": value["is_original"],
        }
    deepcopy_preserves_args = False
    if "args_repr" in deep and original is not None:
        deepcopy_preserves_args = deep["args_repr"] == original["args_repr"]
    return {
        "construction_outcome": "returned",
        "original": {"type": original["type"], "module": original["module"]},
        "copy": side_view(copied),
        "deepcopy": side_view(deep),
        "deepcopy_preserves_args": deepcopy_preserves_args,
    }


def diff_copy_records(oracle: list[dict], candidate: list[dict]) -> list[dict]:
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
                    "oracle": None if left is None else copy_identity(left),
                    "candidate": None if right is None else copy_identity(right),
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
        if copy_identity(left) != copy_identity(right):
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "copy_identity_drift",
                    "oracle": copy_identity(left),
                    "candidate": copy_identity(right),
                }
            )
    return details
