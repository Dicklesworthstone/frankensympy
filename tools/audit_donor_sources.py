#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from registry_audit_lib import Audit, is_hex40, load_registry, nonempty_strings, table_rows, write_or_print


def run(root: Path) -> dict:
    audit = Audit("donor_sources", root)
    donor = load_registry(root, "registries/donor_sources.toml", audit)
    dependency = load_registry(root, "registries/dependencies.toml", audit)
    sources = table_rows(donor, "source", audit)
    ids = audit.unique_ids(sources, "source")
    repositories: set[str] = set()
    pins: dict[str, str] = {}
    for index, row in enumerate(sources):
        prefix = f"source[{index}]"
        repository = row.get("repository")
        commit = row.get("commit")
        audit_path = row.get("audit")
        audit.require(isinstance(repository, str) and repository.count("/") == 1, f"{prefix}: invalid repository")
        audit.require(is_hex40(commit), f"{prefix}: commit must be lowercase 40-hex")
        audit.require(isinstance(audit_path, str) and bool(audit_path), f"{prefix}: audit path required")
        if isinstance(audit_path, str):
            audit.require_file(audit_path, prefix)
        audit.require(nonempty_strings(row.get("roles")), f"{prefix}: roles must be non-empty strings")
        audit.require(row.get("default_disposition") in {"adopt", "selective_adopt", "adapt", "research", "reject"}, f"{prefix}: invalid disposition")
        if isinstance(repository, str):
            if repository in repositories:
                audit.error(f"duplicate donor repository {repository}")
            repositories.add(repository)
            if isinstance(commit, str):
                pins[repository] = commit

    allowed = table_rows(dependency, "allowed_source", audit, required=False)
    compared = 0
    for row in allowed:
        repository = row.get("repository")
        commit = row.get("commit")
        if isinstance(repository, str) and repository in pins:
            compared += 1
            audit.require(commit == pins[repository], f"dependency pin mismatch for {repository}: {commit!r} != {pins[repository]!r}")
    audit.facts.update({"source_count": len(sources), "source_ids": sorted(ids), "dependency_pins_compared": compared})
    return audit.report()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report = run(root)
    write_or_print(report, root, args.json_output, args.no_write)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
