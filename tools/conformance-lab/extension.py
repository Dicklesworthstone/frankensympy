"""Locate a cargo-built fsym_python cdylib without importing the runners.

Tests must not import candidate_runner.py: that module exits 2 at import time
unless PYTHONHASHSEED is pinned.
"""

from __future__ import annotations

import os
from pathlib import Path

EXTENSION_NAMES = (
    "fsym_python.so",
    "libfsym_python.so",
    "fsym_python.pyd",
    "libfsym_python.dylib",
)


def lab_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return lab_root().parents[1]


def extension_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    explicit = os.environ.get("FSYM_PYTHON_EXT_DIR")
    if explicit:
        dirs.append(Path(explicit))
    cargo = os.environ.get("CARGO_TARGET_DIR")
    if cargo:
        dirs.append(Path(cargo) / "debug")
        dirs.append(Path(cargo) / "release")
    root = repo_root()
    dirs.append(root / "target" / "debug")
    dirs.append(root / "target" / "release")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for directory in dirs:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def find_fsym_python_extension() -> Path | None:
    for directory in extension_search_dirs():
        for name in EXTENSION_NAMES:
            path = directory / name
            if path.is_file():
                return path.resolve()
    return None
