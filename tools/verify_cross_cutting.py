#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Obligation:
    id: str
    document: str
    owner: str
    depends_on: tuple[str, ...]
    gates: tuple[str, ...]
    artifacts: tuple[str, ...]
    release_blocking: bool
    status: str


def load_registry(path: Path) -> tuple[dict[str, Any], list[Obligation]]:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    rows = raw.get("obligation")
    if not isinstance(rows, list) or not rows:
        raise ValueError("registry must contain at least one [[obligation]] row")
    obligations: list[Obligation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"obligation[{index}] must be a table")
        try:
            obligations.append(
                Obligation(
                    id=str(row["id"]),
                    document=str(row["document"]),
                    owner=str(row["owner"]),
                    depends_on=tuple(str(x) for x in row["depends_on"]),
                    gates=tuple(str(x) for x in row["gates"]),
                    artifacts=tuple(str(x) for x in row["artifacts"]),
                    release_blocking=bool(row["release_blocking"]),
                    status=str(row["status"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"obligation[{index}] missing field {exc.args[0]!r}") from exc
    return raw, obligations


def validate(root: Path, raw: dict[str, Any], rows: list[Obligation]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ids: dict[str, Obligation] = {}
    documents: dict[str, str] = {}

    allowed = set(raw.get("status", {}).get("allowed", []))
    complete = set(raw.get("status", {}).get("complete", []))
    if not allowed:
        errors.append("[status].allowed must be non-empty")

    for row in rows:
        if not row.id or row.id in ids:
            errors.append(f"duplicate or empty obligation id: {row.id!r}")
        else:
            ids[row.id] = row
        if row.document in documents:
            errors.append(
                f"document {row.document!r} is owned by both {documents[row.document]!r} and {row.id!r}"
            )
        else:
            documents[row.document] = row.id
        path = root / row.document
        if not path.is_file():
            errors.append(f"{row.id}: document does not exist: {row.document}")
        if not row.owner:
            errors.append(f"{row.id}: owner must be non-empty")
        if row.status not in allowed:
            errors.append(f"{row.id}: invalid status {row.status!r}")
        if not row.gates:
            errors.append(f"{row.id}: gates must be non-empty")
        if not row.artifacts:
            errors.append(f"{row.id}: artifacts must be non-empty")
        if len(set(row.gates)) != len(row.gates):
            errors.append(f"{row.id}: duplicate gate")
        if len(set(row.artifacts)) != len(row.artifacts):
            errors.append(f"{row.id}: duplicate artifact")
        if row.id in row.depends_on:
            errors.append(f"{row.id}: self dependency")
        if row.release_blocking and row.status in complete and (not row.gates or not row.artifacts):
            errors.append(f"{row.id}: complete release blocker lacks gates/artifacts")

    for row in rows:
        for dependency in row.depends_on:
            if dependency not in ids:
                errors.append(f"{row.id}: unknown dependency {dependency!r}")

    state: dict[str, int] = {key: 0 for key in ids}
    stack: list[str] = []
    order: list[str] = []

    def visit(node: str) -> None:
        marker = state[node]
        if marker == 2:
            return
        if marker == 1:
            start = stack.index(node) if node in stack else 0
            errors.append("dependency cycle: " + " -> ".join(stack[start:] + [node]))
            return
        state[node] = 1
        stack.append(node)
        for dependency in sorted(ids[node].depends_on):
            if dependency in ids:
                visit(dependency)
        stack.pop()
        state[node] = 2
        order.append(node)

    for node in sorted(ids):
        visit(node)

    if len(order) != len(ids):
        warnings.append("topological order is incomplete because dependency errors were found")

    release_blockers = sorted(row.id for row in rows if row.release_blocking)
    incomplete_blockers = sorted(
        row.id for row in rows if row.release_blocking and row.status not in complete
    )
    return {
        "schema_version": 1,
        "registry": "registries/cross_cutting_obligations.toml",
        "ok": not errors,
        "obligation_count": len(rows),
        "release_blockers": release_blockers,
        "incomplete_release_blockers": incomplete_blockers,
        "topological_order": order,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FrankenSymPy cross-cutting obligations")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("registries/cross_cutting_obligations.toml"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    try:
        raw, rows = load_registry(registry_path)
        report = validate(root, raw, rows)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "registry": str(registry_path),
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output and not args.no_write:
        output = args.json_output if args.json_output.is_absolute() else root / args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
