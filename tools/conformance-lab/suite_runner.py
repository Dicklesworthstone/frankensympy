#!/usr/bin/env python3
"""Oracle-side single-file upstream-suite smoke for the conformance lab.

Runs one inventoried SymPy test file through the *legacy* in-process runner
(`sympy.testing.runtests.test`). The pinned oracle venv does not install
pytest, and `sympy.test()` refuses without it. This is an execution receipt
for the oracle environment only.

It does not record FrankenSymPy port status, exclusions, or compatibility.

Exit codes: 0 = receipt emitted (including a failing test file),
2 = harness misuse, 3 = digest mismatch, 4 = runner crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

PROFILE_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

if os.environ.get("PYTHONHASHSEED") != PROFILE_ENV["PYTHONHASHSEED"]:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "error_class": "harness_misuse",
                "detail": "PYTHONHASHSEED must be '0'",
            }
        )
    )
    sys.exit(2)


SUMMARY_RE = re.compile(
    r"tests finished:\s+"
    r"(?:(?P<passed>\d+) passed)?"
    r"(?:,\s*(?P<failed>\d+) failed)?"
    r"(?:,\s*(?P<skipped>\d+) skipped)?"
    r".*",
    re.IGNORECASE,
)


def main() -> int:
    if len(sys.argv) != 4:
        print(json.dumps({"error_class": "harness_misuse"}))
        return 2
    profile_id, relpath, expected_sha256 = sys.argv[1:]
    if ".." in Path(relpath).parts or Path(relpath).is_absolute():
        print(json.dumps({"error_class": "unsafe_test_path", "path": relpath}))
        return 2

    import sympy
    from sympy.testing.runtests import test as legacy_test

    root = Path(sympy.__file__).resolve().parent
    target = (root / relpath).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        print(json.dumps({"error_class": "path_escapes_oracle", "path": relpath}))
        return 2
    if not target.is_file():
        print(json.dumps({"error_class": "missing_test_file", "path": relpath}))
        return 2
    blob = target.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "error_class": "digest_mismatch",
                    "path": relpath,
                    "expected_sha256": expected_sha256,
                    "actual_sha256": digest,
                },
                sort_keys=True,
            )
        )
        return 3

    # Capture the runner's human summary from stdout while still emitting
    # one JSON receipt as the last line.
    from io import StringIO
    from contextlib import redirect_stdout, redirect_stderr

    class UTF8StringIO(StringIO):
        encoding = "utf-8"

    buffer = UTF8StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            passed = bool(
                legacy_test(relpath, subprocess=False, verbose=False)
            )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "error_class": "runner_crash",
                    "detail": f"{type(exc).__name__}: {exc}"[:400],
                },
                sort_keys=True,
            )
        )
        return 4

    text = buffer.getvalue()
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for line in text.splitlines():
        match = SUMMARY_RE.search(line)
        if match:
            for key in counts:
                value = match.group(key)
                if value is not None:
                    counts[key] = int(value)
    receipt = {
        "schema_version": 1,
        "kind": "oracle_suite_receipt",
        "profile_id": profile_id,
        "test_path": relpath,
        "bytes": len(blob),
        "sha256": digest,
        "runner": "sympy.testing.runtests.test",
        "pytest_installed": False,
        "legacy_return_true": passed,
        "counts": counts,
        "status_note": (
            "oracle execution receipt only; no FrankenSymPy port status is claimed"
        ),
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
