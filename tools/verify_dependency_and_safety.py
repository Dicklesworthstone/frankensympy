#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from registry_audit_lib import Audit, NIGHTLY_RE, is_hex40, load_registry, load_toml, nonempty_strings, table_rows, write_or_print


def manifest_dependencies(manifest: dict) -> set[str]:
    names: set[str] = set()
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        value = manifest.get(section, {})
        if isinstance(value, dict):
            names.update(str(name) for name in value)
    target = manifest.get("target", {})
    if isinstance(target, dict):
        for target_table in target.values():
            if isinstance(target_table, dict):
                for section in ("dependencies", "dev-dependencies", "build-dependencies"):
                    value = target_table.get(section, {})
                    if isinstance(value, dict):
                        names.update(str(name) for name in value)
    return names


def manifest_dependency_packages(manifest: dict) -> set[str]:
    """Return resolved package names, including dependencies renamed in Cargo.toml."""
    packages: set[str] = set()

    def collect(section: object) -> None:
        if not isinstance(section, dict):
            return
        for declared_name, specification in section.items():
            package_name = specification.get("package") if isinstance(specification, dict) else None
            packages.add(str(package_name or declared_name))

    for section_name in ("dependencies", "dev-dependencies", "build-dependencies"):
        collect(manifest.get(section_name, {}))
    target = manifest.get("target", {})
    if isinstance(target, dict):
        for target_table in target.values():
            if isinstance(target_table, dict):
                for section_name in ("dependencies", "dev-dependencies", "build-dependencies"):
                    collect(target_table.get(section_name, {}))
    return packages


def run(root: Path) -> dict:
    audit = Audit("dependency_and_safety", root)
    cargo_path = audit.require_file("Cargo.toml")
    toolchain_path = audit.require_file("rust-toolchain.toml")
    registry = load_registry(root, "registries/dependencies.toml", audit)
    verifier = load_registry(root, "registries/verifier_profiles.toml", audit)
    try:
        cargo = load_toml(cargo_path) if cargo_path.is_file() else {}
        toolchain = load_toml(toolchain_path) if toolchain_path.is_file() else {}
    except Exception as exc:
        audit.error(str(exc))
        cargo, toolchain = {}, {}

    policy = registry.get("policy", {})
    audit.require(isinstance(policy, dict), "dependencies [policy] required")
    if isinstance(policy, dict):
        audit.require(policy.get("project_authored_unsafe") == "forbid", "project-authored unsafe must be forbidden")
        audit.require(policy.get("native_algorithm_dependencies") == "forbid", "native algorithm dependencies must be forbidden")
        audit.require(policy.get("floating_git_release_dependencies") == "forbid", "floating git release dependencies must be forbidden")
        audit.require(policy.get("portable_verifier_ffi") is False, "portable verifier FFI must be false")
        audit.require(policy.get("network_build_scripts") is False, "network build scripts must be false")

    rust_lints = cargo.get("lints", {}).get("rust", {}) if isinstance(cargo.get("lints", {}), dict) else {}
    audit.require(isinstance(rust_lints, dict) and rust_lints.get("unsafe_code") == "forbid", "Cargo.toml must set lints.rust.unsafe_code = forbid")

    workspace = cargo.get("workspace", {})
    workspace_dependencies = workspace.get("dependencies", {}) if isinstance(workspace, dict) else {}
    num_rational = workspace_dependencies.get("num-rational") if isinstance(workspace_dependencies, dict) else None
    num_rational_features = num_rational.get("features") if isinstance(num_rational, dict) else None
    audit.require(isinstance(num_rational, dict), "workspace num-rational dependency must use an explicit table")
    if isinstance(num_rational, dict):
        audit.require(
            num_rational.get("default-features") is False,
            "workspace num-rational must disable the optional num-bigint default feature",
        )
    audit.require(
        isinstance(num_rational_features, list)
        and set(num_rational_features) == {"serde", "std"},
        "workspace num-rational must enable exactly serde and std",
    )

    channel = toolchain.get("toolchain", {}).get("channel") if isinstance(toolchain.get("toolchain", {}), dict) else None
    audit.require(isinstance(channel, str) and bool(NIGHTLY_RE.fullmatch(channel)), "toolchain channel must be exact nightly-YYYY-MM-DD")

    allowed = table_rows(registry, "allowed_source", audit)
    source_ids = audit.unique_ids(allowed, "allowed_source")
    rust_std = next((row for row in allowed if row.get("id") == "rust_std"), None)
    audit.require(isinstance(rust_std, dict), "rust_std allowed source required")
    if isinstance(rust_std, dict):
        audit.require(rust_std.get("pin") == channel, "rust_std registry pin must match rust-toolchain.toml")
    for index, row in enumerate(allowed):
        if "repository" in row:
            audit.require(is_hex40(row.get("commit")), f"allowed_source[{index}]: commit must be lowercase 40-hex")

    forbidden_rows = table_rows(registry, "forbidden_family", audit)
    forbidden_patterns: list[str] = []
    for index, row in enumerate(forbidden_rows):
        patterns = row.get("patterns")
        audit.require(nonempty_strings(patterns), f"forbidden_family[{index}]: patterns required")
        if isinstance(patterns, list):
            forbidden_patterns.extend(item.lower() for item in patterns if isinstance(item, str) and " " not in item)

    manifests = sorted(root.rglob("Cargo.toml"))
    actual_dependencies: set[str] = set()
    substrate_declarations: dict[str, list[str]] = {"num-bigint": [], "num-rational": []}
    substrate_allowed_manifests = {
        "num-bigint": {"Cargo.toml", "crates/fsym-bigint/Cargo.toml"},
        "num-rational": {"Cargo.toml", "crates/fsym-rational/Cargo.toml"},
    }
    build_scripts: list[str] = []
    for manifest_path in manifests:
        try:
            manifest = load_toml(manifest_path)
        except Exception as exc:
            audit.error(f"{manifest_path.relative_to(root)}: {exc}")
            continue
        actual_dependencies.update(manifest_dependencies(manifest))
        manifest_relative = manifest_path.relative_to(root).as_posix()
        resolved_packages = {
            dependency.lower().replace("_", "-")
            for dependency in manifest_dependency_packages(manifest)
        }
        for substrate in substrate_declarations:
            if substrate in resolved_packages:
                substrate_declarations[substrate].append(manifest_relative)
                audit.require(
                    manifest_relative in substrate_allowed_manifests[substrate],
                    f"{manifest_relative}: provisional substrate {substrate} is outside its dedicated ownership boundary",
                )
        package = manifest.get("package", {})
        if isinstance(package, dict) and package.get("build"):
            build_scripts.append(str(manifest_path.relative_to(root)))
        sibling_build = manifest_path.parent / "build.rs"
        if sibling_build.is_file():
            build_scripts.append(str(sibling_build.relative_to(root)))
    for dependency in sorted(actual_dependencies):
        normalized = dependency.lower().replace("_", "-")
        for pattern in forbidden_patterns:
            if pattern and pattern in normalized:
                audit.error(f"forbidden dependency pattern {pattern!r} matched {dependency!r}")

    verifier_policy = verifier.get("policy", {})
    for key in (
        "generator_dependency_forbidden",
        "planner_dependency_forbidden",
        "python_dependency_forbidden",
        "async_runtime_dependency_forbidden",
        "persistence_dependency_forbidden",
        "network_dependency_forbidden",
        "frankensympy_source_unsafe_forbidden",
    ):
        audit.require(isinstance(verifier_policy, dict) and verifier_policy.get(key) is True, f"verifier policy {key} must be true")
    profiles = table_rows(verifier, "profiles", audit)
    audit.unique_ids(profiles, "profiles")
    audit.require(any(row.get("rust_std") is False and row.get("python") is False and row.get("network") is False for row in profiles), "at least one no-std/no-Python/no-network verifier profile required")

    audit.facts.update({
        "toolchain_channel": channel,
        "allowed_source_ids": sorted(source_ids),
        "cargo_manifest_count": len(manifests),
        "declared_dependency_names": sorted(actual_dependencies),
        "provisional_substrate_declarations": {
            dependency: sorted(paths) for dependency, paths in substrate_declarations.items()
        },
        "num_rational_default_features": (
            num_rational.get("default-features") if isinstance(num_rational, dict) else None
        ),
        "num_rational_features": sorted(num_rational_features) if isinstance(num_rational_features, list) else [],
        "project_build_scripts": sorted(set(build_scripts)),
        "portable_verifier_profile_count": len(profiles),
    })
    if build_scripts:
        audit.warn("project build scripts exist and require explicit review")
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
