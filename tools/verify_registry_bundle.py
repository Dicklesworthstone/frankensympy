#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from registry_audit_lib import Audit, iter_toml_files, load_registry, load_toml, nonempty_strings, table_rows, write_or_print


def run(root: Path) -> dict:
    audit = Audit("registry_bundle", root)
    parsed: dict[str, dict] = {}
    for path in iter_toml_files(root):
        relative = str(path.relative_to(root))
        try:
            parsed[relative] = load_toml(path)
        except Exception as exc:
            audit.error(f"{relative}: {exc}")
    audit.require(bool(parsed), "no TOML registries found")

    architecture = parsed.get("registries/architecture_documents.toml") or load_registry(root, "registries/architecture_documents.toml", audit)
    documents = table_rows(architecture, "document", audit)
    document_ids = audit.unique_ids(documents, "document")
    required_document_ids = {
        "constitution", "comprehensive_plan", "architecture_revision_2026_08_20",
        "agent_native_protocol", "assumptions_domains_numeric_tower",
        "compatibility_contract", "conformance_and_benchmarking",
        "crate_architecture_and_dependencies", "evidence_proofs_and_rewrites",
        "first_implementation_campaign", "object_model_and_ir",
        "persistence_distribution_and_repair", "runtime_budgets_and_determinism",
        "security_and_resource_governance", "workstream_graph",
        "portable_verifier", "artifact_protocol", "workspace", "graph", "formal",
        "portfolios_revision", "python_effects", "packaging", "safety", "release",
        "evidence_lattice", "monitoring", "performance", "cross_cutting",
        "architecture_review_round_3",
    }
    for required_id in sorted(required_document_ids - document_ids):
        audit.error(f"architecture document registry missing required id {required_id!r}")
    document_paths: set[str] = set()
    for index, row in enumerate(documents):
        prefix = f"document[{index}]"
        path = row.get("path")
        audit.require(isinstance(path, str) and bool(path), f"{prefix}: path required")
        if isinstance(path, str):
            if path in document_paths:
                audit.error(f"duplicate architecture document path {path}")
            document_paths.add(path)
            audit.require_file(path, prefix)
        registry = row.get("registry")
        if registry is not None:
            audit.require(isinstance(registry, str) and bool(registry), f"{prefix}: registry must be string")
            if isinstance(registry, str):
                audit.require_file(registry, prefix)
                audit.require(registry in parsed, f"{prefix}: registry {registry} did not parse")

    cargo_path = audit.require_file("Cargo.toml")
    try:
        cargo = load_toml(cargo_path) if cargo_path.is_file() else {}
    except Exception as exc:
        audit.error(f"Cargo.toml: {exc}")
        cargo = {}
    metadata = cargo.get("package", {}).get("metadata", {}).get("frankensympy", {}) if isinstance(cargo.get("package", {}), dict) else {}
    audit.require(isinstance(metadata, dict), "Cargo package.metadata.frankensympy required")
    metadata_paths: list[str] = []
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key.endswith("_registry") or key in {"architecture_index"}:
                audit.require(isinstance(value, str), f"Cargo metadata {key} must be a path string")
                if isinstance(value, str):
                    metadata_paths.append(value)
                    audit.require_file(value, f"Cargo metadata {key}")
        toolchain = metadata.get("toolchain")
        if isinstance(toolchain, dict):
            manifest = toolchain.get("manifest")
            audit.require(isinstance(manifest, str), "Cargo toolchain manifest path required")
            if isinstance(manifest, str):
                metadata_paths.append(manifest)
                audit.require_file(manifest, "Cargo toolchain manifest")

    claim = parsed.get("registries/claim_lattice.toml", {})
    classes = table_rows(claim, "class", audit)
    class_ids = audit.unique_ids(classes, "claim class")
    justifications = table_rows(claim, "justification", audit)
    for index, row in enumerate(justifications):
        audit.require(row.get("evidence") in class_ids, f"justification[{index}]: unknown evidence class")
        audit.require(row.get("claim") in class_ids, f"justification[{index}]: unknown claim class")
    composition = claim.get("composition", {})
    for key in ("transitive_by_default", "majority_vote_upgrades", "signatures_upgrade_truth", "monitor_output_mints_verified_claim", "silent_downgrade"):
        audit.require(isinstance(composition, dict) and composition.get(key) is False, f"claim composition {key} must be false")

    graph = parsed.get("registries/graph_reasoning.toml", {})
    tie_rows = table_rows(graph, "tie_break_policy", audit)
    graph_rows = table_rows(graph, "graph_family", audit)
    cert_rows = table_rows(graph, "certificate_family", audit)
    audit.unique_ids(tie_rows, "tie_break_policy")
    audit.unique_ids(graph_rows, "graph_family")
    audit.unique_ids(cert_rows, "certificate_family")
    for index, row in enumerate(cert_rows):
        if row.get("portable") is True:
            audit.require(row.get("generator_required_by_verifier") is False, f"certificate_family[{index}]: portable verifier cannot require generator")
            audit.require(row.get("mutation_suite_required") is True, f"certificate_family[{index}]: mutation suite required")

    portfolios = parsed.get("registries/algorithm_portfolios.toml", {})
    portfolio_defaults = portfolios.get("defaults", {})
    audit.require(isinstance(portfolio_defaults, dict) and portfolio_defaults.get("selection_can_admit_claim") is False, "portfolio selection cannot admit claims")
    audit.require(isinstance(portfolio_defaults, dict) and portfolio_defaults.get("verifier_budget_reserved_before_generation") is True, "portfolio verifier reserve must be protected")
    portfolio_rows = table_rows(portfolios, "portfolio", audit)
    audit.unique_ids(portfolio_rows, "portfolio")
    for index, row in enumerate(portfolio_rows):
        for key in ("state_dimensions", "strategies"):
            audit.require(nonempty_strings(row.get(key)), f"portfolio[{index}]: {key} required")
        audit.require(isinstance(row.get("reference_verifier"), str) and bool(row.get("reference_verifier")), f"portfolio[{index}]: reference verifier required")
        audit.require(row.get("protected_verifier_reserve") is True, f"portfolio[{index}]: protected verifier reserve required")

    effects = parsed.get("registries/python_effects.toml", {})
    effect_rows = table_rows(effects, "effect_class", audit)
    effect_ids = audit.unique_ids(effect_rows, "effect_class")
    unknown = next((row for row in effect_rows if row.get("id") == "unknown_effect"), None)
    audit.require(isinstance(unknown, dict), "unknown_effect class required")
    if isinstance(unknown, dict):
        audit.require(unknown.get("duplicable") is False and unknown.get("cacheable") is False and unknown.get("portable_verifier_allowed") is False, "unknown effects must be exactly-once/non-cacheable/non-verifier")

    packaging = parsed.get("registries/packaging_profiles.toml", {})
    packaging_rows = table_rows(packaging, "profile", audit)
    audit.unique_ids(packaging_rows, "packaging profile")
    for index, row in enumerate(packaging_rows):
        if row.get("satisfies_requires_dist_sympy") is True:
            audit.require(row.get("distribution_name") == "sympy", f"packaging profile[{index}]: resolver-transparent profile must use distribution name sympy")
            audit.require(row.get("exclusive_path_owner") is True, f"packaging profile[{index}]: exclusive path ownership required")

    monitoring = parsed.get("registries/monitor_profiles.toml", {})
    monitor_defaults = monitoring.get("defaults", {})
    audit.require(isinstance(monitor_defaults, dict) and monitor_defaults.get("monitor_can_admit_claim") is False, "monitors cannot admit claims")
    monitor_rows = table_rows(monitoring, "monitor", audit)
    audit.unique_ids(monitor_rows, "monitor")
    for index, row in enumerate(monitor_rows):
        for key in ("includes_timeouts", "includes_cancellations", "includes_resource_exhaustion"):
            audit.require(row.get(key) is True, f"monitor[{index}]: {key} must be true")
        audit.require(nonempty_strings(row.get("actions")), f"monitor[{index}]: actions required")
        audit.require(isinstance(row.get("safe_fallback"), str) and bool(row.get("safe_fallback")), f"monitor[{index}]: safe fallback required")

    release = parsed.get("registries/release_gates.toml", {})
    audit.require(release.get("github_hosted_execution_authoritative") is False, "GitHub-hosted execution must be non-authoritative")
    audit.require("doodlestein" in str(release.get("canonical_orchestrator", "")).lower(), "Doodlestein must be canonical orchestrator")
    gate_rows = table_rows(release, "gate", audit)
    audit.unique_ids(gate_rows, "release gate")
    for index, row in enumerate(gate_rows):
        audit.require(str(row.get("command", "")).startswith("scripts/check.sh "), f"release gate[{index}]: command must route through scripts/check.sh")
        audit.require(nonempty_strings(row.get("artifacts")), f"release gate[{index}]: artifacts required")

    audit.facts.update({
        "parsed_registry_count": len(parsed),
        "parsed_registries": sorted(parsed),
        "architecture_document_count": len(documents),
        "architecture_document_ids": sorted(document_ids),
        "cargo_metadata_paths": sorted(metadata_paths),
        "claim_class_count": len(classes),
        "graph_family_count": len(graph_rows),
        "portable_graph_certificate_count": sum(row.get("portable") is True for row in cert_rows),
        "portfolio_count": len(portfolio_rows),
        "effect_class_ids": sorted(effect_ids),
        "packaging_profile_count": len(packaging_rows),
        "monitor_count": len(monitor_rows),
        "release_gate_count": len(gate_rows),
    })
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
