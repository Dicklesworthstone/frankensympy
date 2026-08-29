"""Reconstruction records: obj.func(*obj.args). Not a golden field; cannot certify.

Default reconstruction typically evaluates. This lane compares reconstructed
type/module identity and whether reconstruction preserves args_repr. Exact
args_repr strings across oracle vs candidate stay on construction_only.
"""

from __future__ import annotations

RECON_KIND = "reconstruction_observation"
RECON_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "fixture_id",
        "side",
        "construction_outcome",
        "original",
        "reconstructed",
    }
)
SURFACE_KEYS = frozenset({"type", "module", "args_repr"})
RECON_OK_KEYS = frozenset({"type", "module", "args_repr", "is_original"})
RECON_ERR_KEYS = frozenset({"error_class", "message_head"})


def surface_of(obj) -> dict:
    return {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
        "args_repr": [repr(arg) for arg in getattr(obj, "args", ())],
    }


def recon_ok(obj, rebuilt) -> dict:
    surface = surface_of(rebuilt)
    surface["is_original"] = rebuilt is obj
    return surface


def recon_error(exc: BaseException) -> dict:
    return {
        "error_class": type(exc).__module__ + "." + type(exc).__name__,
        "message_head": str(exc)[:200],
    }


def make_reconstruction_record(
    *,
    profile_id: str,
    fixture_id: str,
    side: str,
    construction_outcome: str,
    original: dict | None,
    reconstructed: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "kind": RECON_KIND,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "side": side,
        "construction_outcome": construction_outcome,
        "original": original,
        "reconstructed": reconstructed,
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


def _validate_recon_result(value: object) -> None:
    if not isinstance(value, dict):
        raise TypeError("reconstructed must be an object")
    if set(value) == RECON_ERR_KEYS:
        if not isinstance(value["error_class"], str) or not value["error_class"]:
            raise ValueError("reconstructed error_class must be a non-empty string")
        return
    if set(value) != RECON_OK_KEYS:
        raise ValueError("reconstructed keys do not match schema version 1")
    _validate_surface(
        {
            "type": value["type"],
            "module": value["module"],
            "args_repr": value["args_repr"],
        },
        label="reconstructed",
    )
    if not isinstance(value["is_original"], bool):
        raise TypeError("reconstructed is_original must be boolean")


def validate_reconstruction_record(
    record: object, profile: dict, *, expected_side: str, expected_id: str
) -> None:
    if not isinstance(record, dict) or set(record) != RECON_RECORD_KEYS:
        raise ValueError("reconstruction record keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != RECON_KIND:
        raise ValueError("reconstruction record schema or kind mismatch")
    if record["profile_id"] != profile["profile_id"]:
        raise ValueError("reconstruction record profile mismatch")
    if record["side"] != expected_side or record["fixture_id"] != expected_id:
        raise ValueError("reconstruction record side or fixture_id mismatch")
    if record["construction_outcome"] not in {"returned", "raised", "refused"}:
        raise ValueError("unknown reconstruction construction_outcome")
    if record["construction_outcome"] != "returned":
        if record["original"] is not None or record["reconstructed"] is not None:
            raise ValueError("non-returned reconstruction cannot carry surfaces")
        return
    _validate_surface(record["original"], label="original")
    _validate_recon_result(record["reconstructed"])


def reconstruction_identity(record: dict) -> dict:
    if record["construction_outcome"] != "returned":
        return {"construction_outcome": record["construction_outcome"]}
    original = record["original"]
    rebuilt = record["reconstructed"]
    if "error_class" in rebuilt:
        recon_view: dict = {"error_class": rebuilt["error_class"]}
        preserves = False
    else:
        recon_view = {
            "type": rebuilt["type"],
            "module": rebuilt["module"],
            "is_original": rebuilt["is_original"],
        }
        preserves = rebuilt["args_repr"] == original["args_repr"]
    return {
        "construction_outcome": "returned",
        "original": {"type": original["type"], "module": original["module"]},
        "reconstructed": recon_view,
        "reconstruction_preserves_args": preserves,
    }


def diff_reconstruction_records(oracle: list[dict], candidate: list[dict]) -> list[dict]:
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
                    "oracle": None if left is None else reconstruction_identity(left),
                    "candidate": None if right is None else reconstruction_identity(right),
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
        if reconstruction_identity(left) != reconstruction_identity(right):
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "reconstruction_identity_drift",
                    "oracle": reconstruction_identity(left),
                    "candidate": reconstruction_identity(right),
                }
            )
    return details
