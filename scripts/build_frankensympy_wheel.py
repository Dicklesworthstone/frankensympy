#!/usr/bin/env python3
"""Assemble the coexistable ``frankensympy`` preview wheel.

Deliberately stdlib-only and deterministic: no build backend, no network, no
deletion. The command stages the package tree in a temporary directory, writes
the wheel metadata (METADATA / WHEEL / RECORD with sha256 hashes), and replaces
the wheel file atomically.

Refuses (exit 2) rather than shipping a broken or namespace-overlapping wheel:
  * a missing, misnamed, or wrong-ABI native extension;
  * a member that would own any part of the top-level ``sympy`` namespace;
  * a version that disagrees across pyproject.toml, Cargo.toml, and the package;
  * a LICENSE that is not the repository MIT rider.

Usage:
    scripts/build_frankensympy_wheel.py [--dist dist] [--output PATH]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shutil
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from pathlib import Path

WHEEL_GENERATOR = "frankensympy-build-wheel (scripts/build_frankensympy_wheel.py)"
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RIDER_MARKER = "with OpenAI/Anthropic Rider"


def refuse(message: str) -> "None":
    print(f"REFUSE: {message}", file=sys.stderr)
    raise SystemExit(2)


def read_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_versions(root: Path) -> tuple[str, str, str]:
    pyproject = read_toml(root / "pyproject.toml")
    cargo = read_toml(root / "Cargo.toml")
    package_version = None
    package_init = root / "python" / "frankensympy" / "__init__.py"
    for line in package_init.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            package_version = line.split("=", 1)[1].strip().strip('"')
            break
    return (
        str(pyproject["project"]["version"]),
        str(cargo["workspace"]["package"]["version"]),
        str(package_version),
    )


def platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def collect_members(root: Path) -> list[tuple[Path, str]]:
    """(source path, archive name) pairs in deterministic name order."""
    package_dir = root / "python" / "frankensympy"
    if not package_dir.is_dir():
        refuse(f"package directory missing: {package_dir}")
    members: list[tuple[Path, str]] = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_dir() or path.name == "__pycache__":
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_dir).as_posix()
        archive_name = f"frankensympy/{relative}"
        if archive_name.split("/", 1)[0] == "sympy" or relative.split("/", 1)[0] == "sympy":
            refuse(f"member would own the sympy namespace: {archive_name}")
        members.append((path, archive_name))
    if not any(name.endswith("__init__.py") for _, name in members):
        refuse("package has no __init__.py")
    return members


def hash_member(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode()}"


def build_metadata(version: str, readme: str, license_text: str) -> bytes:
    if "MIT License" not in license_text or RIDER_MARKER not in license_text:
        refuse("LICENSE is not the repository MIT rider license")
    lines = [
        "Metadata-Version: 2.1",
        "Name: frankensympy",
        f"Version: {version}",
        "Summary: Experimental clean-room symbolic mathematics engine "
        "(preview channel, not a drop-in replacement)",
        "License: MIT License with the repository rider",
        "License-File: LICENSE",
        "Requires-Python: >=3.10",
        "Classifier: Programming Language :: Rust",
        "Classifier: Programming Language :: Python :: Implementation :: CPython",
        "Classifier: Topic :: Scientific/Engineering :: Mathematics",
        "Description-Content-Type: text/markdown",
        "",
        readme.rstrip(),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_wheel_file(version: str, tag: str, extension_name: str) -> bytes:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            f"Generator: {WHEEL_GENERATOR}",
            "Root-Is-Purelib: false",
            f"Tag: {tag}",
            "",
        ]
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    dist = (args.dist or root / "dist").resolve()

    project_version, cargo_version, package_version = load_versions(root)
    if not project_version == cargo_version == package_version:
        refuse(
            "version disagreement: pyproject "
            f"{project_version!r}, Cargo.toml {cargo_version!r}, package {package_version!r}"
        )

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    if not extension_suffix:
        refuse("running interpreter reports no extension suffix")
    extension_name = f"fsym_python{extension_suffix}"
    extension = root / "python" / "frankensympy" / extension_name
    if not extension.is_file():
        refuse(
            f"native extension missing: {extension} "
            "(run scripts/build_python_extension.sh first)"
        )
    if extension.stat().st_size == 0:
        refuse(f"native extension is empty: {extension}")

    members = collect_members(root)
    if not any(name.endswith(extension_name) for _, name in members):
        refuse(f"extension {extension_name} is not inside the package tree")

    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    dist_info = f"frankensympy-{project_version}.dist-info"
    tag = f"cp{sysconfig.get_config_var('py_version_nodot')}-cp{sysconfig.get_config_var('py_version_nodot')}-{platform_tag()}"

    wheel_name = f"frankensympy-{project_version}-{tag}.whl"
    output = args.output or (dist / wheel_name)
    dist.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, bytes]] = []
    for path, archive_name in members:
        entries.append((archive_name, path.read_bytes()))
    entries.append((f"{dist_info}/METADATA", build_metadata(project_version, readme, license_text)))
    entries.append((f"{dist_info}/WHEEL", build_wheel_file(project_version, tag, extension_name)))
    entries.append((f"{dist_info}/licenses/LICENSE", license_text.encode("utf-8")))

    record_lines = [
        f"{name},{hash_member(data)},{len(data)}" for name, data in entries
    ]
    record_lines.append(f"{dist_info}/RECORD,,")
    entries.append((f"{dist_info}/RECORD", ("\n".join(record_lines) + "\n").encode("utf-8")))

    # Stage inside the output directory so the final rename is atomic (same
    # filesystem). Only this private staging directory is removed.
    staging = Path(tempfile.mkdtemp(prefix=".frankensympy-wheel-", dir=output.parent))
    staged = staging / wheel_name
    with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    os.replace(staged, output)
    shutil.rmtree(staging, ignore_errors=True)

    print(f"wheel: {output}")
    print(f"sha256: {digest}")
    print(f"tag: {tag}")
    print(f"members: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
