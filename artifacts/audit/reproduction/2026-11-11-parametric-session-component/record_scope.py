"""Hash the exact-source Session component and explicitly inventory omissions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "frankensympy-integration"
DONOR = HERE.parent / "asupersync"
metadata = json.loads((HERE / "metadata.json").read_text())
original = tomllib.loads((SOURCE / "Cargo.lock").read_text())
component = tomllib.loads((HERE / "Cargo.lock").read_text())

def identity(package):
    return package["name"], package["version"], package.get("source")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def head(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()

original_packages = {identity(row): row for row in original["package"]}
registries = []
for row in component["package"]:
    if not row.get("source"):
        continue
    expected = original_packages.get(identity(row))
    if expected is None or row.get("checksum") != expected.get("checksum"):
        raise SystemExit(f"Registry identity differs from original lock: {row['name']}")
    registries.append({field: row[field] for field in ("name", "version", "source", "checksum")})

runtime_selected = sorted(Path(p) for p in re.findall(r'#\[path = "([^"]+)"\]', (HERE / "runtime/lib.rs").read_text()))
codec_selected = sorted(Path(p) for p in re.findall(r'#\[path = "([^"]+)"\]', (HERE / "codec/lib.rs").read_text()))
codec_includes = [DONOR / "src/raptorq/rfc6330_systematic_index_table.inc"]
assert len(runtime_selected) == 5 and len(codec_selected) == 14
runtime_all = set((SOURCE / "crates/fsym-runtime/src").rglob("*.rs"))
codec_all = set((DONOR / "src").rglob("*.rs"))
source_files = set((SOURCE / "src").rglob("*.rs"))
source_files.update(SOURCE / "tests" / (name + ".rs") for name in (
    "native_parametric", "parametric_session", "parametric_cli", "optimization_integration_adversaries"))
source_files.update(runtime_selected + codec_selected + codec_includes)
source_files.update([SOURCE / "Cargo.toml", SOURCE / "Cargo.lock", SOURCE / "rust-toolchain.toml", SOURCE / "crates/fsym-runtime/Cargo.toml", DONOR / "Cargo.toml", DONOR / "Cargo.lock", DONOR / "build.rs"])
original_paths = []
wrapper_paths = {HERE, HERE / "runtime", HERE / "codec"}
for package in metadata["packages"]:
    if package.get("source"):
        continue
    directory = Path(package["manifest_path"]).parent.resolve()
    if directory in wrapper_paths:
        continue
    if not (directory.is_relative_to(SOURCE / "crates") or directory in {
        DONOR / "franken_decision", DONOR / "franken_kernel", DONOR / "franken_evidence", DONOR / "asupersync-macros"
    }):
        raise SystemExit(f"Unexpected original dependency: {directory}")
    original_paths.append({"name": package["name"], "manifest_path": str(directory / "Cargo.toml")})
    source_files.update(p for p in directory.rglob("*") if p.is_file() and "target" not in p.parts and ".git" not in p.parts)
if any(row["name"] in {"asupersync", "fsym-runtime"} for row in metadata["packages"]):
    raise SystemExit("Full original runtime resolved unexpectedly")

harness_files = [HERE / p for p in ("Cargo.toml", "Cargo.lock", "metadata.json", "runtime/Cargo.toml", "runtime/lib.rs", "codec/Cargo.toml", "codec/lib.rs", "record_scope.py")]
record = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "scope": "Unchanged complete root library and CLI sources with whole original synchronous runtime and pinned codec dependency modules; source-component validation only",
    "not_covered": ["full production Cargo dependency graph", "asynchronous runtime scheduling/cancellation implementation", "donor runtime transport/security", "omitted runtime modules and their tests", "workspace CI gates", "compatibility profiles"],
    "repository_head": head(SOURCE),
    "donor_head": head(DONOR),
    "root_library": str(SOURCE / "src/lib.rs"),
    "root_binary": str(SOURCE / "src/bin/frankensympy.rs"),
    "source_functions_modified_or_substituted": [],
    "scratch_composition": [
        {"package": "fsym-runtime-session-source-component", "library_alias": "fsym_runtime", "selected_whole_modules": [str(p.relative_to(SOURCE)) for p in runtime_selected], "omitted_source_modules": sorted(str(p.relative_to(SOURCE)) for p in runtime_all - set(runtime_selected))},
        {"package": "asupersync-codec-source-component", "library_alias": "asupersync", "selected_whole_modules": [str(p.relative_to(DONOR)) for p in codec_selected], "original_include_files": [str(p.relative_to(DONOR)) for p in codec_includes], "omitted_source_modules": sorted(str(p.relative_to(DONOR)) for p in codec_all - set(codec_selected)), "build_script_omitted": "Only exports ASUPERSYNC_BUILD_GIT_COMMIT; none of the selected modules references it", "crate_root_differences": "Module-only wiring, identical active feature attributes/unsafe policy, and no production root dead-code deny lint for unused omitted-module entry points"},
    ],
    "metadata_platform_filter": "x86_64-unknown-linux-gnu",
    "selected_donor_active_features": ["default", "native-runtime", "nightly-outcome-try", "proc-macros", "runtime-core"],
    "registry_identities_match_original_lock": True,
    "registry_packages": registries,
    "original_path_packages": sorted(original_paths, key=lambda p: p["name"]),
    "resolved_features": {row["id"]: row["features"] for row in metadata["resolve"]["nodes"]},
    "source_sha256": {str(p): sha(p) for p in sorted(source_files)},
    "harness_sha256": {str(p): sha(p) for p in harness_files},
}
destination = HERE / sys.argv[1]
destination.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"record": str(destination), "registry_packages": len(registries), "original_path_packages": len(original_paths), "hashed_source_files": len(source_files), "selected_runtime_modules": len(runtime_selected), "selected_donor_modules": len(codec_selected), "omitted_runtime_files": len(runtime_all - set(runtime_selected)), "omitted_donor_files": len(codec_all - set(codec_selected))}))
