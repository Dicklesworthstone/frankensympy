#!/usr/bin/env python3
"""Discrepancy minimizer for the FrankenSymPy conformance laboratory.

Pairs oracle-side and candidate-side NDJSON observation envelopes by stable
fixture id, diffs them under a registered comparator (default: the profile's
exact_surface), and emits one discrepancy record per mismatching or missing
fixture, conforming to tools/conformance-lab/schema/discrepancy.schema.json.

Usage:
  minimize.py <oracle.ndjson> <candidate.ndjson> \
      [--comparator exact_surface|construction_only] [--out FILE] [--severity object]

Exit codes: 0 = no discrepancies, 1 = discrepancies emitted (or written),
2 = misuse.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparators import REGISTRY, diff_envelopes, discrepancy_id, is_valid_discrepancy

MAX_ENVELOPE_FILE_BYTES = 64 * 1024 * 1024
MAX_ENVELOPE_LINE_BYTES = 4 * 1024 * 1024
MAX_ENVELOPES = 65_536
MAX_LEDGER_RECORD_BYTES = MAX_ENVELOPE_FILE_BYTES
REQUIRED_ENVELOPE_KEYS = {
    "schema_version",
    "profile_id",
    "fixture_id",
    "side",
    "outcome_class",
    "observations",
    "environment",
}
ORACLE_SIDE = "upstream_oracle"
CANDIDATE_SIDE = "frankensympy_candidate"
LEDGER_INDEX_FIELDS = (
    "discrepancy_id",
    "fixture_id",
    "comparator",
    "severity",
    "status",
)


def strict_json_loads(payload: str):
    def reject_constant(token: str):
        raise ValueError(f"non-finite JSON number is forbidden: {token}")

    return json.loads(payload, parse_constant=reject_constant)


def _validate_ledger_record_size(size: int, record_path: Path) -> None:
    if size > MAX_LEDGER_RECORD_BYTES:
        raise ValueError(
            f"ledger record {record_path} exceeds the ledger record-size limit"
        )


def load_envelopes(path: Path) -> list[dict]:
    size = path.stat().st_size
    if size > MAX_ENVELOPE_FILE_BYTES:
        raise ValueError(
            f"{path} is {size} bytes; maximum is {MAX_ENVELOPE_FILE_BYTES}"
        )
    envelopes = []
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            if len(line.encode()) > MAX_ENVELOPE_LINE_BYTES:
                raise ValueError(
                    f"{path} line {line_number} exceeds {MAX_ENVELOPE_LINE_BYTES} bytes"
                )
            try:
                envelope = strict_json_loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path} line {line_number} is not valid JSON: {exc}"
                ) from exc
            validate_envelope_shape(envelope, path, line_number)
            envelopes.append(envelope)
            if len(envelopes) > MAX_ENVELOPES:
                raise ValueError(f"{path} exceeds {MAX_ENVELOPES} envelopes")
    if not envelopes:
        raise ValueError(f"{path} contains no observation envelopes")
    return envelopes


def validate_envelope_shape(envelope: object, path: Path, line_number: int) -> None:
    if not isinstance(envelope, dict):
        raise TypeError(f"{path} line {line_number} must be a JSON object")
    if set(envelope) != REQUIRED_ENVELOPE_KEYS:
        raise ValueError(f"{path} line {line_number} has invalid envelope keys")
    if envelope["schema_version"] != 1:
        raise ValueError(f"{path} line {line_number} has unknown schema_version")
    for field in ("profile_id", "fixture_id", "side", "outcome_class"):
        if not isinstance(envelope[field], str) or not envelope[field]:
            raise TypeError(f"{path} line {line_number} has invalid {field}")
    if envelope["outcome_class"] not in {"returned", "raised", "timeout", "refused"}:
        raise ValueError(f"{path} line {line_number} has unknown outcome_class")
    if not isinstance(envelope["observations"], dict):
        raise TypeError(f"{path} line {line_number} observations must be an object")
    if not isinstance(envelope["environment"], dict):
        raise TypeError(f"{path} line {line_number} environment must be an object")


def index_envelopes(envelopes: list[dict], *, side: str) -> dict[str, dict]:
    if side not in {"oracle", "candidate"}:
        raise ValueError(f"unknown envelope side role: {side!r}")
    expected_side = ORACLE_SIDE if side == "oracle" else CANDIDATE_SIDE
    indexed = {}
    for envelope in envelopes:
        if envelope["side"] != expected_side:
            raise ValueError(
                f"{side} fixture {envelope['fixture_id']!r} has side "
                f"{envelope['side']!r}; required={expected_side!r}"
            )
        fixture_id = envelope["fixture_id"]
        if fixture_id in indexed:
            raise ValueError(f"duplicate {side} fixture_id: {fixture_id!r}")
        indexed[fixture_id] = envelope
    return indexed


def make_record(
    *,
    profile_id: str,
    fixture_id: str,
    comparator: str,
    severity: str,
    differences: list[dict],
    environment: dict | None,
    outcome_classes: dict | None = None,
    created_at_utc: str,
) -> dict:
    record = {
        "schema_version": 1,
        "discrepancy_id": discrepancy_id(
            profile_id=profile_id,
            fixture_id=fixture_id,
            comparator=comparator,
            differences=differences,
        ),
        "status": "open",
        "severity": severity,
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "comparator": comparator,
        "differences": differences,
        "created_at_utc": created_at_utc,
    }
    if environment is not None:
        record["environment"] = environment
    if outcome_classes is not None:
        record["outcome_classes"] = outcome_classes
    ok, reason = is_valid_discrepancy(record)
    if not ok:
        raise ValueError(f"invalid discrepancy record: {reason}")
    return record


def build_records(
    oracle_envs: list[dict],
    candidate_envs: list[dict],
    *,
    comparator: str,
    severity: str,
    fallback_profile_id: str,
    created_at_utc: str,
) -> tuple[list[dict], int]:
    oracle_by_id = index_envelopes(oracle_envs, side="oracle")
    candidate_by_id = index_envelopes(candidate_envs, side="candidate")
    wrong_oracle_profiles = sorted(
        fixture_id
        for fixture_id, envelope in oracle_by_id.items()
        if envelope["profile_id"] != fallback_profile_id
    )
    if wrong_oracle_profiles:
        raise ValueError(
            f"oracle profile does not match {fallback_profile_id!r} for fixtures: "
            + ", ".join(wrong_oracle_profiles)
        )
    records = []
    paired = 0
    for fixture_id in sorted(oracle_by_id.keys() | candidate_by_id.keys()):
        oracle = oracle_by_id.get(fixture_id)
        candidate = candidate_by_id.get(fixture_id)
        if oracle is None:
            differences = [
                {"path": "fixture_id", "oracle": "<missing>", "candidate": fixture_id}
            ]
            records.append(
                make_record(
                    profile_id=fallback_profile_id,
                    fixture_id=fixture_id,
                    comparator=comparator,
                    severity=severity,
                    differences=differences,
                    environment=candidate["environment"],
                    created_at_utc=created_at_utc,
                )
            )
            continue
        if candidate is None:
            differences = [
                {"path": "fixture_id", "oracle": fixture_id, "candidate": "<missing>"}
            ]
            records.append(
                make_record(
                    profile_id=fallback_profile_id,
                    fixture_id=fixture_id,
                    comparator=comparator,
                    severity=severity,
                    differences=differences,
                    environment=oracle["environment"],
                    created_at_utc=created_at_utc,
                )
            )
            continue

        paired += 1
        differences = diff_envelopes(oracle, candidate, comparator)
        if not differences:
            continue
        outcome_classes = None
        if oracle["outcome_class"] != candidate["outcome_class"]:
            outcome_classes = {
                "oracle": oracle["outcome_class"],
                "candidate": candidate["outcome_class"],
            }
        records.append(
            make_record(
                profile_id=fallback_profile_id,
                fixture_id=fixture_id,
                comparator=comparator,
                severity=severity,
                differences=differences,
                environment=oracle["environment"],
                outcome_classes=outcome_classes,
                created_at_utc=created_at_utc,
            )
        )
    return records, paired


def publish_file(path: Path, payload: str, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symbolic-link {label}: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite different {label}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(payload)


def record_identity(record: dict) -> dict:
    return {
        key: record[key]
        for key in (
            "schema_version",
            "discrepancy_id",
            "profile_id",
            "fixture_id",
            "comparator",
            "differences",
        )
    }


def _index_validated_records(records: list[dict]) -> dict[str, dict]:
    """Validate a publication batch and index it without collapsing duplicates."""
    indexed = {}
    for position, record in enumerate(records, start=1):
        ok, reason = is_valid_discrepancy(record)
        if not ok:
            raise ValueError(f"invalid incoming ledger record {position}: {reason}")
        discrepancy = record["discrepancy_id"]
        if discrepancy in indexed:
            raise ValueError(f"duplicate incoming ledger record: {discrepancy}")
        indexed[discrepancy] = record
    return indexed


def _validate_ledger_index_entry(
    entry: object, record: dict, *, line_number: int
) -> str:
    """Require every index projection to agree with its canonical record."""
    if not isinstance(entry, dict) or set(entry) != set(LEDGER_INDEX_FIELDS):
        raise ValueError(f"invalid ledger index line {line_number}")
    discrepancy = entry["discrepancy_id"]
    if not isinstance(discrepancy, str) or not re.fullmatch(
        r"disc-[a-z0-9-]+", discrepancy
    ):
        raise TypeError(f"invalid ledger index line {line_number}")
    for field in LEDGER_INDEX_FIELDS:
        if entry[field] != record[field]:
            raise ValueError(
                f"ledger index field {field!r} disagrees with record {discrepancy}"
            )
    return discrepancy


def load_ledger_index(ledger: Path) -> set[str]:
    index_path = ledger / "index.ndjson"
    indexed_ids = set()
    if not index_path.exists():
        return indexed_ids
    if index_path.is_symlink():
        raise ValueError("ledger index must not be a symbolic link")
    if index_path.stat().st_size > MAX_ENVELOPE_FILE_BYTES:
        raise ValueError("ledger index exceeds the file-size limit")
    with open(index_path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                entry = strict_json_loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid ledger index line {line_number}: {exc}"
                ) from exc
            if not isinstance(entry, dict) or set(entry) != set(LEDGER_INDEX_FIELDS):
                raise ValueError(f"invalid ledger index line {line_number}")
            discrepancy = entry["discrepancy_id"]
            if not isinstance(discrepancy, str) or not re.fullmatch(
                r"disc-[a-z0-9-]+", discrepancy
            ):
                raise TypeError(f"invalid ledger index line {line_number}")
            if discrepancy in indexed_ids:
                raise ValueError(f"duplicate ledger index id: {discrepancy}")
            record_path = ledger / f"{discrepancy}.json"
            if record_path.is_symlink() or not record_path.is_file():
                raise ValueError(
                    f"ledger index references missing record: {discrepancy}"
                )
            _validate_ledger_record_size(record_path.stat().st_size, record_path)
            try:
                record = strict_json_loads(record_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise ValueError(
                    f"invalid existing ledger record {record_path}: {exc}"
                ) from exc
            ok, reason = is_valid_discrepancy(record)
            if not ok:
                raise ValueError(
                    f"invalid existing ledger record {record_path}: {reason}"
                )
            _validate_ledger_index_entry(entry, record, line_number=line_number)
            indexed_ids.add(discrepancy)
            if len(indexed_ids) > MAX_ENVELOPES:
                raise ValueError("ledger index exceeds the record-count limit")
    return indexed_ids


def persist_ledger(records: list[dict], ledger: Path) -> None:
    records_by_id = _index_validated_records(records)
    if ledger.is_symlink() or (ledger.exists() and not ledger.is_dir()):
        raise ValueError(f"ledger path must be a real directory: {ledger}")
    ledger.mkdir(parents=True, exist_ok=True)
    indexed_ids = load_ledger_index(ledger)
    missing_records = []
    for record in records_by_id.values():
        record_path = ledger / f"{record['discrepancy_id']}.json"
        if record_path.is_symlink():
            raise ValueError(
                f"ledger record must not be a symbolic link: {record_path}"
            )
        if not record_path.exists():
            missing_records.append((record_path, record))
            continue
        _validate_ledger_record_size(record_path.stat().st_size, record_path)
        try:
            existing = strict_json_loads(record_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(
                f"invalid existing ledger record {record_path}: {exc}"
            ) from exc
        ok, reason = is_valid_discrepancy(existing)
        if not ok:
            raise ValueError(f"invalid existing ledger record {record_path}: {reason}")
        if record_identity(existing) != record_identity(record):
            raise ValueError(
                f"ledger collision on {record['discrepancy_id']} with different identity"
            )

    # Publish only after every existing record and the complete index pass
    # validation, so pre-existing corruption cannot cause a partial run.
    for record_path, record in missing_records:
        payload = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
        publish_file(record_path, payload, label="ledger record")

    index_path = ledger / "index.ndjson"
    additions = [
        record
        for discrepancy, record in records_by_id.items()
        if discrepancy not in indexed_ids
    ]
    if additions:
        with open(index_path, "a", encoding="utf-8") as fh:
            for record in additions:
                fh.write(
                    json.dumps(
                        {field: record[field] for field in LEDGER_INDEX_FIELDS},
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--comparator", default="exact_surface", choices=sorted(REGISTRY)
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--severity",
        default="object",
        choices=["object", "mathematical", "runtime", "security"],
    )
    parser.add_argument("--profile-id", default="sympy-1.14.0-cpython")
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=None,
        help="persist records as <dir>/<disc-id>.json + append to index.ndjson",
    )
    args = parser.parse_args()

    try:
        oracle_envs = load_envelopes(args.oracle)
        candidate_envs = load_envelopes(args.candidate)
        records, pair_count = build_records(
            oracle_envs,
            candidate_envs,
            comparator=args.comparator,
            severity=args.severity,
            fallback_profile_id=args.profile_id,
            created_at_utc=datetime.now(UTC).isoformat(),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    payload = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
    )
    if args.out:
        try:
            publish_file(args.out, payload, label="discrepancy output")
        except (OSError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        print(
            f"wrote {len(records)} discrepancy record(s) to {args.out}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(payload)

    if args.ledger_dir:
        try:
            persist_ledger(records, args.ledger_dir)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        print(
            f"ledger: {len(records)} record(s) under {args.ledger_dir}",
            file=sys.stderr,
        )

    print(
        f"compared {pair_count} envelope pair(s); {len(records)} discrepancy(ies)",
        file=sys.stderr,
    )
    return 1 if records else 0


if __name__ == "__main__":
    sys.exit(main())
