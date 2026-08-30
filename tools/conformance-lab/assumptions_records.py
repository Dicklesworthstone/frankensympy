"""Three-valued assumptions queries. Not a golden field and cannot certify.

Unknown must stay none. true/false/none are identity; message text on probe
errors is outside the contract.
"""

from __future__ import annotations

ASSUMPTIONS_KIND = "assumptions_observation"
ASSUMPTION_QUERIES = (
    "is_positive",
    "is_negative",
    "is_zero",
    "is_real",
    "is_rational",
    "is_integer",
    "is_commutative",
    "is_number",
)
ASSUMPTIONS_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "fixture_id",
        "side",
        "construction_outcome",
        "queries",
    }
)
TRI_STATES = frozenset({"true", "false", "none"})


def encode_assumption_value(value: object) -> object:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "none"
    return {
        "error_class": "harness.non_tri_state",
        "message_head": type(value).__module__ + "." + type(value).__name__,
    }


def encode_assumption_error(exc: BaseException) -> dict:
    return {
        "error_class": type(exc).__module__ + "." + type(exc).__name__,
        "message_head": str(exc)[:200],
    }


def query_assumptions(obj) -> dict:
    queries = {}
    for name in ASSUMPTION_QUERIES:
        try:
            queries[name] = encode_assumption_value(getattr(obj, name))
        except Exception as exc:  # noqa: BLE001
            queries[name] = encode_assumption_error(exc)
    return queries


def make_assumptions_record(
    *,
    profile_id: str,
    fixture_id: str,
    side: str,
    construction_outcome: str,
    queries: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "kind": ASSUMPTIONS_KIND,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "side": side,
        "construction_outcome": construction_outcome,
        "queries": queries,
    }


def _validate_query_value(value: object, *, name: str) -> None:
    if isinstance(value, str) and value in TRI_STATES:
        return
    if not isinstance(value, dict) or set(value) != {"error_class", "message_head"}:
        raise ValueError(f"assumption {name} must be true/false/none or an error object")
    if not isinstance(value["error_class"], str) or not value["error_class"]:
        raise ValueError(f"assumption {name} error_class must be a non-empty string")


def validate_assumptions_record(
    record: object, profile: dict, *, expected_side: str, expected_id: str
) -> None:
    if not isinstance(record, dict) or set(record) != ASSUMPTIONS_RECORD_KEYS:
        raise ValueError("assumptions record keys do not match schema version 1")
    if record["schema_version"] != 1 or record["kind"] != ASSUMPTIONS_KIND:
        raise ValueError("assumptions record schema or kind mismatch")
    if record["profile_id"] != profile["profile_id"]:
        raise ValueError("assumptions record profile mismatch")
    if record["side"] != expected_side or record["fixture_id"] != expected_id:
        raise ValueError("assumptions record side or fixture_id mismatch")
    if record["construction_outcome"] not in {"returned", "raised", "refused"}:
        raise ValueError("unknown assumptions construction_outcome")
    if record["construction_outcome"] != "returned":
        if record["queries"] is not None:
            raise ValueError("non-returned assumptions record cannot carry queries")
        return
    queries = record["queries"]
    if not isinstance(queries, dict) or set(queries) != set(ASSUMPTION_QUERIES):
        raise ValueError("assumptions queries do not match the registered set")
    for name, value in queries.items():
        _validate_query_value(value, name=name)


def assumptions_identity(record: dict) -> dict:
    if record["construction_outcome"] != "returned":
        return {"construction_outcome": record["construction_outcome"]}
    queries = {}
    for name, value in record["queries"].items():
        if isinstance(value, dict):
            queries[name] = {"error_class": value["error_class"]}
        else:
            queries[name] = value
    return {"construction_outcome": "returned", "queries": queries}


def diff_assumptions_records(oracle: list[dict], candidate: list[dict]) -> list[dict]:
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
                    "oracle": None if left is None else assumptions_identity(left),
                    "candidate": None if right is None else assumptions_identity(right),
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
        if assumptions_identity(left) != assumptions_identity(right):
            details.append(
                {
                    "fixture_id": fixture_id,
                    "kind": "assumptions_identity_drift",
                    "oracle": assumptions_identity(left),
                    "candidate": assumptions_identity(right),
                }
            )
    return details
