#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from registry_audit_lib import Audit, load_registry, nonempty_strings, table_rows, write_or_print


def run(root: Path) -> dict:
    audit = Audit("performance_kernels", root)
    registry = load_registry(root, "registries/performance_kernels.toml", audit)
    defaults = registry.get("defaults", {})
    for key in ("semantic_gate_before_timing", "full_operation_required", "rollback_key_required"):
        audit.require(isinstance(defaults, dict) and defaults.get(key) is True, f"performance default {key} must be true")
    audit.require(isinstance(defaults, dict) and defaults.get("project_authored_unsafe") == "forbid", "performance kernels must forbid project-authored unsafe")

    rows = table_rows(registry, "kernel", audit)
    ids = audit.unique_ids(rows, "kernel")
    architectures: set[str] = set()
    statuses: dict[str, int] = {}
    for index, row in enumerate(rows):
        prefix = f"kernel[{index}]"
        status = row.get("status")
        audit.require(status in {"planned", "research", "admitted", "quarantined"}, f"{prefix}: invalid status")
        if isinstance(status, str):
            statuses[status] = statuses.get(status, 0) + 1
        audit.require(isinstance(row.get("operation"), str) and bool(row.get("operation")), f"{prefix}: operation required")
        reference = row.get("reference")
        optimized = row.get("optimized")
        audit.require(isinstance(reference, str) and bool(reference), f"{prefix}: reference route required")
        audit.require(nonempty_strings(optimized), f"{prefix}: optimized routes required")
        if isinstance(optimized, list) and isinstance(reference, str):
            audit.require(reference not in optimized, f"{prefix}: reference route repeated as optimized")
        for key in ("regimes", "correctness", "architectures"):
            audit.require(nonempty_strings(row.get(key)), f"{prefix}: {key} must be non-empty strings")
        if isinstance(row.get("architectures"), list):
            architectures.update(str(item) for item in row["architectures"])

    receipts = registry.get("required_receipts", {})
    required_receipts = (
        "source_tree", "toolchain", "machine_class", "corpus_root", "route_ids",
        "semantic_gate", "raw_samples", "allocations", "peak_memory", "tails",
        "host_telemetry", "aa_control", "noise_decision",
    )
    for key in required_receipts:
        audit.require(isinstance(receipts, dict) and receipts.get(key) is True, f"required receipt {key} must be true")

    audit.require("apple_silicon" in architectures, "Apple Silicon architecture class required")
    audit.require("high_core_amd" in architectures, "high-core AMD architecture class required")
    audit.require("portable_scalar" in architectures, "portable scalar architecture class required")
    audit.facts.update({"kernel_count": len(rows), "kernel_ids": sorted(ids), "architectures": sorted(architectures), "statuses": statuses})
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
