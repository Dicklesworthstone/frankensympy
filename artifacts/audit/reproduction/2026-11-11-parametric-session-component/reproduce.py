"""Verify recorded original sources and relocate only scratch component inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
ORIGINAL_REPOSITORY = "/workspace/scratch/8a453db7a629/frankensympy-integration"
ORIGINAL_DONOR = "/workspace/scratch/8a453db7a629/asupersync"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repository", type=Path, required=True)
parser.add_argument("--donor", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True, help="New scratch directory, never the source repository")
args = parser.parse_args()
repository, donor, output = (p.resolve() for p in (args.repository, args.donor, args.output))
for path in (repository, donor, output):
    if any(char in str(path) for char in '\n\r"\\'):
        raise SystemExit("Use Linux paths without quotes, newlines or backslashes")
if output.exists():
    raise SystemExit("Output already exists; refusing to overwrite")
if output.is_relative_to(repository) or output.is_relative_to(donor):
    raise SystemExit("Output must be a separate scratch directory")

inventory = json.loads((HERE / "source-inventory.json").read_text())
roots = {"repository": repository, "donor": donor}
for relative, expected in inventory["selected_original_source_sha256"].items():
    root, tail = relative.split("/", 1)
    path = roots[root] / tail
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Recorded source identity differs: {path}")
for component in inventory["component_compositions"]:
    root = repository if component["library_alias"] == "fsym_runtime" else donor
    directory = root / ("crates/fsym-runtime/src" if root == repository else "src")
    selected = set(component["selected_whole_modules"])
    omitted = sorted(str(path.relative_to(root)) for path in directory.rglob("*.rs") if str(path.relative_to(root)) not in selected)
    digest = hashlib.sha256(json.dumps(omitted, separators=(",", ":")).encode()).hexdigest()
    if len(omitted) != component["omitted_source_module_count"] or digest != component["omitted_source_module_list_sha256"]:
        raise SystemExit(f"Omitted source inventory differs: {root}")

output.mkdir()
for folder in ("runtime", "codec"):
    (output / folder).mkdir()
for name in ("Cargo.toml", "Cargo.lock", "runtime/Cargo.toml", "runtime/lib.rs", "codec/Cargo.toml", "codec/lib.rs", "record_scope.py"):
    contents = (HERE / name).read_text()
    if name != "Cargo.lock":
        for old, new in (
            (ORIGINAL_REPOSITORY, str(repository)),
            (ORIGINAL_DONOR, str(donor)),
            ("../../frankensympy-integration", str(repository)),
            ("../frankensympy-integration", str(repository)),
            ("../../asupersync", str(donor)),
        ):
            contents = contents.replace(old, new)
    if name == "record_scope.py":
        contents = contents.replace('SOURCE = HERE.parent / "frankensympy-integration"', f"SOURCE = Path({str(repository)!r})")
        contents = contents.replace('DONOR = HERE.parent / "asupersync"', f"DONOR = Path({str(donor)!r})")
    (output / name).write_text(contents)
print(f"Verified {len(inventory['selected_original_source_sha256'])} original input files and both omitted module inventories.")
print("Relocated module-only manifests and wrappers to", output)
print("Run the README commands from", shlex.quote(str(output)))
