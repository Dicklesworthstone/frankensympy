"""Reconstructed, unexecuted setup recipe for scoped Session/CLI validation."""
import argparse
from pathlib import Path
import subprocess
import tomllib

p = argparse.ArgumentParser()
p.add_argument("--repository", type=Path, required=True)
p.add_argument("--donor", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
repo, donor, out = (x.resolve() for x in (a.repository, a.donor, a.output))
for path in (repo, donor, out):
    if any(c in str(path) for c in '\n\r"\\'):
        raise SystemExit("Use paths without quotes, newlines or backslashes")
if out.exists() or out.is_relative_to(repo) or out.is_relative_to(donor):
    raise SystemExit("Output must be a new scratch directory outside source trees")
pin = "b3e08ac0bd773a5bbdd4f3c2f4624d550352ec0f"
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=donor, text=True).strip() != pin:
    raise SystemExit("Donor does not match its exact admitted pin")
base = repo / "artifacts/audit/reproduction/2026-11-11-parametric-session-component"
old_lock = tomllib.loads((base / "Cargo.lock").read_text())
repo_lock = tomllib.loads((repo / "Cargo.lock").read_text())

def identity(row):
    return row["name"], row["version"], row.get("source")

known = {identity(row): row for row in repo_lock["package"]}
for row in old_lock["package"]:
    if row.get("source"):
        other = known.get(identity(row))
        if other is None or other.get("checksum") != row.get("checksum"):
            raise SystemExit("Registry identity/checksum differs from original lock")
selected = (
    "compiled_program_transport", "compiled_program_cli", "compiled_artifacts",
    "compiled_residual_session", "compiled_residual_products",
    "compiled_objective_session", "artifact_transfer", "artifact_transfer_session",
)
for name in selected:
    if not (repo / "tests" / (name + ".rs")).is_file():
        raise SystemExit("Missing required test target: " + name)
replacement = (
    ("/workspace/scratch/8a453db7a629/frankensympy-integration", str(repo)),
    ("/workspace/scratch/8a453db7a629/asupersync", str(donor)),
    ("../../frankensympy-integration", str(repo)),
    ("../frankensympy-integration", str(repo)),
    ("../../asupersync", str(donor)),
)
out.mkdir()
(out / "runtime").mkdir()
(out / "codec").mkdir()
for name in ("Cargo.toml", "Cargo.lock", "runtime/Cargo.toml", "runtime/lib.rs",
             "codec/Cargo.toml", "codec/lib.rs", "record_scope.py"):
    text = (base / name).read_text()
    if name != "Cargo.lock":
        for old, new in replacement:
            text = text.replace(old, new)
    if name == "record_scope.py":
        text = text.replace('SOURCE = HERE.parent / "frankensympy-integration"',
                            "SOURCE = Path(" + repr(str(repo)) + ")")
        text = text.replace('DONOR = HERE.parent / "asupersync"',
                            "DONOR = Path(" + repr(str(donor)) + ")")
        old = ('source_files.update(SOURCE / "tests" / (name + ".rs") for name in (\n'
               '    "native_parametric", "parametric_session", "parametric_cli", '
               '"optimization_integration_adversaries"))')
        if text.count(old) != 1:
            raise SystemExit("Historical scope recorder structure changed")
        text = text.replace(old, 'source_files.update(Path(t["path"]) for t in '
            'tomllib.loads((HERE / "Cargo.toml").read_text())["test"])')
    (out / name).write_text(text)
manifest = out / "Cargo.toml"
text = manifest.read_text()
existing = {target["name"] for target in tomllib.loads(text)["test"]}
for test in sorted((repo / "tests").glob("*.rs")):
    if test.stem not in existing:
        text += '\n[[test]]\nname = "' + test.stem + '"\npath = "' + str(test) + '"\n'
manifest.write_text(text)
print("Prepared unchanged-source component in", out)
print("Run metadata, pre-scope, the eight-target command, and post-scope there.")
