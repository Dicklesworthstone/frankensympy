#!/usr/bin/env python3
"""Packaging battery for the ``frankensympy`` preview wheel.

Every case is adversarial or coexistence-critical, runs against real installed
wheels in throwaway virtual environments, and prints the raw evidence it used:

  1. wheel audit            — RECORD hashes, METADATA license, no ``sympy/``
                              member, extension present under its ABI name
  2. coexistence install    — offline install next to pinned upstream SymPy;
                              import from a neutral cwd with no PYTHONPATH
  3. isolation install      — same wheel with no upstream present: it must
                              import on its own and never pull ``sympy`` in
  4. missing extension      — mutated wheel refuses to import, no fallback
  5. wrong module name      — mutated wheel refuses to import, no fallback
  6. uninstall              — removing the preview leaves upstream intact

Usage:
    scripts/check_frankensympy_wheel.py --wheel dist/frankensympy-*.whl \
        --wheelhouse <dir with upstream sympy wheel>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import venv
import zipfile
from pathlib import Path

UPSTREAM_VERSION = "1.14.0"


class Case:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []

    def check(self, condition: bool, evidence: str) -> None:
        print(f"    {'ok ' if condition else 'BAD'} {evidence}")
        if not condition:
            self.failures.append(evidence)


def run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=900, check=False
    )
    print(f"    $ {' '.join(str(part) for part in command)}")
    if completed.stdout.strip():
        print(f"      stdout: {completed.stdout.strip()[-1500:]}")
    if completed.stderr.strip():
        print(f"      stderr: {completed.stderr.strip()[-1500:]}")
    return completed


def audit_wheel(wheel: Path, case: Case) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        record = {}
        for line in archive.read(record_name).decode("utf-8").splitlines():
            if not line.strip():
                continue
            member, digest, size = (line.split(",") + ["", ""])[:3]
            record[member] = (digest, size)
        case.check(
            all(not name.startswith("sympy/") for name in names),
            f"no member owns the sympy namespace ({len(names)} members)",
        )
        case.check(
            any(name.startswith("frankensympy/frankensympy") for name in names) is False,
            "package layout is flat under frankensympy/",
        )
        unrecorded = [name for name in names if name not in record]
        case.check(not unrecorded, f"every member is in RECORD (missing: {unrecorded})")
        mismatched = []
        for name in names:
            if name.endswith("/RECORD"):
                continue
            expected, size = record.get(name, ("", ""))
            data = archive.read(name)
            actual = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()
            ).rstrip(b"=").decode()
            if expected != actual or size != str(len(data)):
                mismatched.append(name)
        case.check(not mismatched, f"RECORD hashes and sizes match ({len(names)} members, bad: {mismatched})")

        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        case.check("Name: frankensympy" in metadata, "METADATA names the frankensympy distribution")
        case.check("License: MIT License with the repository rider" in metadata, "METADATA license is the MIT rider")
        case.check("Apache" not in metadata, "METADATA contains no Apache attribution")
        case.check("Requires-Python:" in metadata, "METADATA declares Requires-Python")
        license_member = next(
            (name for name in names if name.endswith(".dist-info/licenses/LICENSE")), None
        )
        case.check(license_member is not None, f"wheel carries the LICENSE file ({license_member})")
        if license_member:
            text = archive.read(license_member).decode("utf-8")
            case.check(
                "MIT License" in text and "with OpenAI/Anthropic Rider" in text,
                "bundled LICENSE is the repository MIT rider",
            )
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
        expected_extension = f"frankensympy/fsym_python{suffix}"
        case.check(expected_extension in names, f"extension present as {expected_extension}")
        if expected_extension in names:
            case.check(archive.getinfo(expected_extension).file_size > 0, "extension is not empty")
        wheel_meta = archive.read(next(n for n in names if n.endswith(".dist-info/WHEEL"))).decode()
        tag = f"cp{sysconfig.get_config_var('py_version_nodot')}"
        case.check(f"Tag: {tag}-{tag}-" in wheel_meta, f"wheel tag is interpreter specific ({tag})")


def install_wheel(site: Path, wheel: Path, wheelhouse: Path, case: Case, *, upstream: bool) -> None:
    venv.create(site, with_pip=True, clear=True)
    pip = site / "bin" / "pip"
    packages = [str(wheel)]
    if upstream:
        packages.append(f"sympy=={UPSTREAM_VERSION}")
    completed = run(
        [str(pip), "install", "--no-index", "--find-links", str(wheelhouse), *packages]
    )
    case.check(completed.returncode == 0, f"offline install exit {completed.returncode} (upstream={upstream})")


def neutral_env(site: Path) -> dict:
    root = site.resolve()
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root / "home"),
        "PYTHONNOUSERSITE": "1",
        "VIRTUAL_ENV": str(root),
    }


def extract_wheel(site: Path, wheel: Path) -> None:
    """Install a mutated wheel by extraction.

    pip refuses a mutated filename before installing, and the mutation exists to
    test the package's own refusal path, so the archive is unpacked directly
    into site-packages the way an installer would.
    """
    site_packages = next((site / "lib").glob("python*/site-packages"))
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site_packages)
    print(f"    extracted {wheel.name} into {site_packages}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    wheelhouse = args.wheelhouse.resolve()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="frankensympy-wheel-check-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"wheel: {wheel}")
    print(f"workdir: {workdir}")
    results: list[Case] = []

    print("\n[1] wheel audit")
    case = Case("audit")
    results.append(case)
    audit_wheel(wheel, case)

    neutral_cwd = workdir / "neutral"
    neutral_cwd.mkdir(exist_ok=True)

    print("\n[2] coexistence: offline install next to pinned upstream SymPy")
    case = Case("coexistence")
    results.append(case)
    site = workdir / "venv-coexist"
    install_wheel(site, wheel, wheelhouse, case, upstream=True)
    probe = (
        "import json, os, sys; import frankensympy, sympy;"
        "print(json.dumps({'preview': frankensympy.__version__,"
        " 'engine': frankensympy.engine_version(),"
        " 'preview_path': os.path.dirname(frankensympy.__file__),"
        " 'native': frankensympy.native.__file__,"
        " 'sympy': sympy.__version__, 'sympy_path': os.path.dirname(sympy.__file__)}))"
    )
    completed = run([str(site / "bin" / "python"), "-c", probe], cwd=neutral_cwd, env=neutral_env(site))
    case.check(completed.returncode == 0, f"import smoke exit {completed.returncode} from neutral cwd")
    payload = {}
    if completed.returncode == 0:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        case.check(payload["preview"] == "0.1.0", f"preview version {payload['preview']}")
        case.check(payload["engine"] == "0.1.0", f"engine version {payload['engine']}")
        case.check(payload["sympy"] == UPSTREAM_VERSION, f"upstream sympy is {payload['sympy']}")
        case.check(
            payload["sympy_path"] != payload["preview_path"],
            "preview and upstream live in different directories",
        )
        case.check(
            "frankensympy" in payload["native"] and "sympy" not in payload["native"].split("frankensympy")[0].split("/")[-2:],
            f"native engine resolves inside the preview package ({payload['native']})",
        )

    print("\n[3] isolation: same wheel with no upstream installed")
    case = Case("isolation")
    results.append(case)
    site = workdir / "venv-isolated"
    install_wheel(site, wheel, wheelhouse, case, upstream=False)
    probe = (
        "import sys, frankensympy;"
        "print('SYMPY_PRESENT', 'sympy' in sys.modules);"
        "print('VERSION', frankensympy.__version__)"
    )
    completed = run([str(site / "bin" / "python"), "-c", probe], cwd=neutral_cwd, env=neutral_env(site))
    case.check(completed.returncode == 0, f"import without upstream exit {completed.returncode}")
    case.check("SYMPY_PRESENT False" in completed.stdout, "importing the preview never imports sympy")
    sources = [name for name in zipfile.ZipFile(wheel).namelist() if name.endswith(".py")]
    forbidden: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        for name in sources:
            text = archive.read(name).decode("utf-8")
            if "import sympy" in text or "from sympy" in text:
                forbidden.append(name)
    case.check(not forbidden, f"wheel sources contain no sympy import (found: {forbidden})")

    print("\n[4] missing extension refuses, no fallback")
    case = Case("missing-extension")
    results.append(case)
    # Keep the five-part wheel filename valid: mutate the platform tag only.
    stem, _, platform = wheel.name[:-4].rpartition("-")
    mutated = workdir / f"{stem}-{platform}-noext.whl"
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(mutated, "w") as target:
        for info in source.infolist():
            if info.filename.endswith(sysconfig.get_config_var("EXT_SUFFIX")):
                continue
            target.writestr(info, source.read(info.filename))
    site = workdir / "venv-no-extension"
    venv.create(site, with_pip=True, clear=True)
    extract_wheel(site, mutated)
    completed = run(
        [str(site / "bin" / "python"), "-c", "import frankensympy"],
        cwd=neutral_cwd,
        env=neutral_env(site),
    )
    case.check(completed.returncode != 0, "import of an incomplete wheel fails")
    case.check(
        "requires its bundled fsym_python extension" in completed.stderr,
        "refusal names the missing bundled extension",
    )

    print("\n[5] wrong module name refuses, no fallback")
    case = Case("wrong-module-name")
    results.append(case)
    mutated = workdir / f"{stem}-{platform}-wrongmodule.whl"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(mutated, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(suffix):
                info.filename = info.filename.replace("fsym_python", "fsym_python_wrong")
            target.writestr(info, data)
    site = workdir / "venv-wrong-module"
    venv.create(site, with_pip=True, clear=True)
    extract_wheel(site, mutated)
    completed = run(
        [str(site / "bin" / "python"), "-c", "import frankensympy"],
        cwd=neutral_cwd,
        env=neutral_env(site),
    )
    case.check(completed.returncode != 0, "import with a misnamed extension fails")
    case.check("requires its bundled fsym_python extension" in completed.stderr, "refusal is explicit")

    print("\n[6] uninstall leaves upstream intact")
    case = Case("uninstall")
    results.append(case)
    site = workdir / "venv-coexist"
    completed = run([str(site / "bin" / "pip"), "uninstall", "-y", "frankensympy"])
    case.check(completed.returncode == 0, f"uninstall exit {completed.returncode}")
    probe = "import sympy; print('UPSTREAM', sympy.__version__, sympy.__file__)"
    completed = run([str(site / "bin" / "python"), "-c", probe], cwd=neutral_cwd, env=neutral_env(site))
    case.check(completed.returncode == 0, f"upstream import after uninstall exit {completed.returncode}")
    case.check(f"UPSTREAM {UPSTREAM_VERSION}" in completed.stdout, "upstream version unchanged")
    leftovers = [path.name for path in (site / "lib").glob("python*/site-packages/frankensympy*")]
    case.check(not leftovers, f"no preview leftovers in site-packages (found: {leftovers})")

    failed = [(case.name, case.failures) for case in results if case.failures]
    print("\nsummary:")
    for case in results:
        print(f"  {'PASS' if not case.failures else 'FAIL'} {case.name}")
    if failed:
        for name, failures in failed:
            for failure in failures:
                print(f"  {name}: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
