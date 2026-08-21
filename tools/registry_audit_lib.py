#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
NIGHTLY_RE = re.compile(r"^nightly-\d{4}-\d{2}-\d{2}$")


@dataclass
class Audit:
    name: str
    root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.error(message)

    def require_file(self, relative: str, label: str | None = None) -> Path:
        path = self.root / relative
        if not path.is_file():
            self.error(f"{label or relative}: missing file {relative}")
        return path

    def unique_ids(self, rows: Sequence[Mapping[str, Any]], label: str) -> set[str]:
        found: set[str] = set()
        for index, row in enumerate(rows):
            value = row.get("id")
            if not isinstance(value, str) or not value.strip():
                self.error(f"{label}[{index}]: id must be a non-empty string")
                continue
            if value in found:
                self.error(f"{label}: duplicate id {value!r}")
            found.add(value)
        return found

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "audit": self.name,
            "ok": not self.errors,
            "facts": self.facts,
            "errors": sorted(set(self.errors)),
            "warnings": sorted(set(self.warnings)),
        }


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_registry(root: Path, relative: str, audit: Audit) -> dict[str, Any]:
    path = audit.require_file(relative)
    if not path.is_file():
        return {}
    try:
        value = load_toml(path)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        audit.error(f"{relative}: {exc}")
        return {}
    if not isinstance(value, dict):
        audit.error(f"{relative}: root must be a TOML table")
        return {}
    return value


def table_rows(value: Mapping[str, Any], key: str, audit: Audit, required: bool = True) -> list[dict[str, Any]]:
    raw = value.get(key)
    if raw is None:
        if required:
            audit.error(f"missing [[{key}]] rows")
        return []
    if not isinstance(raw, list):
        audit.error(f"{key}: expected array of tables")
        return []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if isinstance(row, dict):
            rows.append(row)
        else:
            audit.error(f"{key}[{index}]: expected table")
    return rows


def nonempty_strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value)


def is_hex40(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX40_RE.fullmatch(value))


def write_or_print(report: dict[str, Any], root: Path, output: Path | None, no_write: bool) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None and not no_write:
        target = output if output.is_absolute() else root / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def scalar_string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def iter_toml_files(root: Path) -> Iterable[Path]:
    registry_root = root / "registries"
    if registry_root.is_dir():
        yield from sorted(registry_root.glob("*.toml"))
