"""Relocate the repository's audited source-component recipes into fresh scratch.

No production source or dependency manifest is modified. The Session library
and CLI remain direct Cargo source paths. The Python component retains every
current bridge function except the six previously reviewed async helpers.
This prepares inputs only; it never runs Cargo or any test.
"""

from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess
import sys
import tomllib


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "frankensympy"
REFERENCE = ROOT / "artifacts/audit/reproduction/2026-11-14-real-constraint-systems"
SESSION = BASE / "session"
PYTHON = BASE / "python"
OLD_ROOT = "/workspace/scratch/aac5c0ea4661"
NEW_ROOT = str(BASE.parent)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"Expected exactly one reviewed recipe item: {old!r}")
    return text.replace(old, new, 1)


def lock_identity(package):
    return package["name"], package["version"], package.get("source")


def compare_lock(path):
    production = {
        lock_identity(package): package
        for package in tomllib.loads((ROOT / "Cargo.lock").read_text())["package"]
    }
    selected = []
    for package in tomllib.loads(path.read_text())["package"]:
        if not package.get("source"):
            continue
        expected = production.get(lock_identity(package))
        if expected is None or expected.get("checksum") != package.get("checksum"):
            raise SystemExit(f"Registry dependency differs from production lock: {package['name']}")
        selected.append({key: package[key] for key in ("name", "version", "source", "checksum")})
    return selected


def main():
    if SESSION.exists() or PYTHON.exists():
        raise SystemExit("Refusing to overwrite an existing component")
    for donor, expected in (
        ("asupersync", "b3e08ac0bd773a5bbdd4f3c2f4624d550352ec0f"),
        ("franken_numpy-pin", "d5d29c9da36141f611d1414b4c6ef434ab31ea24"),
        ("frankenscipy-pin", "213a417c739025ed865a3875c89a7d86726ecb5a"),
    ):
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE.parent / donor, text=True,
        ).strip()
        if actual != expected:
            raise SystemExit(f"Unadmitted donor identity: {donor} {actual}")

    for folder in (SESSION, SESSION / "runtime", SESSION / "codec"):
        folder.mkdir()
    for relative in (
        "Cargo.toml", "Cargo.lock", "runtime/Cargo.toml", "runtime/lib.rs",
        "codec/Cargo.toml", "codec/lib.rs", "record_scope.py",
    ):
        original = REFERENCE / "session-component" / relative
        text = original.read_text()
        if relative != "Cargo.lock":
            text = text.replace(OLD_ROOT, NEW_ROOT)
        (SESSION / relative).write_text(text)
    manifest = SESSION / "Cargo.toml"
    text = manifest.read_text()
    if "session_cancellation" in text:
        raise SystemExit("Unexpected pre-existing cancellation target")
    text += (
        '\n[[test]]\nname = "session_cancellation"\n'
        f'path = "{ROOT / "tests/session_cancellation.rs"}"\n'
    )
    manifest.write_text(text)

    # Preserve the existing extraction program except its two scratch paths.
    recipe = (REFERENCE / "python-component/recipes/prepare_python_component.py").read_text()
    recipe = once(recipe, "ROOT = Path('/workspace/scratch/aac5c0ea4661/frankensympy')", f"ROOT = Path({str(ROOT)!r})")
    recipe = once(recipe, "OUTPUT = Path('/workspace/scratch/aac5c0ea4661/validation/python-component')", f"OUTPUT = Path({str(PYTHON)!r})")
    extraction = BASE / "prepare_python_component.py"
    extraction.write_text(recipe)
    subprocess.run([sys.executable, str(extraction)], check=True)

    # Current production bridge directly imports num_traits. This exact
    # production dependency alignment was already recorded by the Nov21 run.
    production_python = tomllib.loads((ROOT / "crates/fsym-python/Cargo.toml").read_text())
    if "num-traits" not in production_python["dependencies"]:
        raise SystemExit("Current production dependency alignment changed")
    manifest = PYTHON / "Cargo.toml"
    text = manifest.read_text()
    if "num-traits=" in text:
        raise SystemExit("Unexpected pre-existing component num-traits declaration")
    manifest.write_text(text + 'num-traits="=0.2.19"\n')

    # Start from the audited component lock (all selected registry entries
    # match production). Add only the already-resolved direct dependency edge
    # on the scratch component package; no registry entry/version is changed.
    lock = (REFERENCE / "python-component/Cargo.lock").read_text()
    start = lock.index('name = "fsym-python-math-source-component"\n')
    end = lock.index("\n[[package]]", start)
    package = lock[start:end]
    if '\n "num-traits",' in package:
        raise SystemExit("Unexpected pre-existing scratch dependency edge")
    replacement = once(package, '\n "pyo3",', '\n "num-traits",\n "pyo3",')
    (PYTHON / "Cargo.lock").write_text(lock[:start] + replacement + lock[end:])

    recorder = (REFERENCE / "python-component/record_scope.py").read_text()
    recorder = recorder.replace(OLD_ROOT, NEW_ROOT)
    (PYTHON / "record_scope.py").write_text(recorder)

    composition = json.loads((PYTHON / "bridge-composition.json").read_text())
    prior = json.loads((ROOT / "artifacts/audit/2026-11-21-absolute-optimization-progress.json").read_text())
    prior_composition = prior["component"]["composition"]
    if composition["omitted_members"] != prior_composition["omitted_members"]:
        raise SystemExit("Helper omission list differs from reviewed scope")
    if composition["omitted_block_sha256"] != prior_composition["omitted_block_sha256"]:
        raise SystemExit("Current omitted helper bodies differ from reviewed source")
    changes = []
    for relative in ("lib.rs", "expr.rs"):
        original = ROOT / "crates/fsym-python/src" / relative
        component = PYTHON / "src" / relative
        changes.extend(difflib.unified_diff(
            original.read_text().splitlines(keepends=True),
            component.read_text().splitlines(keepends=True),
            fromfile=f"production/crates/fsym-python/src/{relative}",
            tofile=f"component/src/{relative}",
        ))
    (PYTHON / "bridge-omissions.patch").write_text("".join(changes))

    inventories = {}
    for name, path in (("session", SESSION), ("python", PYTHON)):
        registry = compare_lock(path / "Cargo.lock")
        record = {
            "scope": "source-component validation only; full production async graph omitted",
            "production_lock_sha256": sha(ROOT / "Cargo.lock"),
            "component_lock_sha256": sha(path / "Cargo.lock"),
            "all_selected_registry_identities_and_checksums_match_production": True,
            "registry_packages": registry,
        }
        (path / "production-lock-comparison.json").write_text(json.dumps(record, indent=2) + "\n")
        inventories[name] = len(registry)

    selected_runtime = re.findall(r'#\[path = "([^"]+)"\]', (SESSION / "runtime/lib.rs").read_text())
    selected_codec = re.findall(r'#\[path = "([^"]+)"\]', (SESSION / "codec/lib.rs").read_text())
    if len(selected_runtime) != 5 or len(selected_codec) != 14:
        raise SystemExit("Unexpected Session component module composition")
    source_paths = [Path(path) for path in selected_runtime + selected_codec]
    source_paths.extend((ROOT / "src").rglob("*.rs"))
    source_paths.extend((ROOT / "crates/fsym-python/src").rglob("*.rs"))
    source_paths.extend([
        ROOT / "tests/session_cancellation.rs",
        ROOT / "python/tests/test_inverse_derivative_numeric.py",
        ROOT / "python/tests/test_nsolve_inverse_jets.py",
        ROOT / "python/sympy/solvers/numerical.py",
        BASE.parent / "asupersync/src/raptorq/rfc6330_systematic_index_table.inc",
    ])
    result = {
        "repository": str(ROOT),
        "session": str(SESSION),
        "python": str(PYTHON),
        "registry_packages_matching_production": inventories,
        "session_root_library_and_cli_are_actual_source_paths": True,
        "session_selected_whole_runtime_modules": selected_runtime,
        "session_selected_whole_codec_modules": selected_codec,
        "python_unchanged_bridge_files": len(composition["original_bridge_files"]) - 2,
        "python_omitted_helpers": composition["omitted_members"],
        "python_omitted_blocks_match_previously_reviewed_exact_hashes": True,
        "source_function_replacements": [],
        "source_sha256": {str(path): sha(path) for path in sorted(set(source_paths))},
        "prepare_script_sha256": sha(__file__),
        "not_executed": ["Cargo metadata", "Cargo build", "Cargo test", "Python tests"],
        "not_covered": [
            "Full production dependency graph", "Asynchronous scheduling and host runtime cancellation",
            "Six omitted Python simplify/expand helpers", "Compatibility profile certification",
        ],
    }
    (BASE / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("source_sha256", "session_selected_whole_runtime_modules", "session_selected_whole_codec_modules")}, indent=2))


if __name__ == "__main__":
    main()
