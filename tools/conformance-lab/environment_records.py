"""Sidecar environment/build receipt. Not a golden field and cannot certify.

Golden envelopes already pin oracle python/sympy/env. This receipt records
the live oracle pin check plus the candidate native cdylib identity (path,
size, digest) without adding keys to the immutable observation schema.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

RECEIPT_KIND = "environment_build_receipt"
EXTENSION_KEYS = frozenset({"present", "path", "size", "sha256"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extension_identity(path: Path | None) -> dict:
    if path is None:
        return {"present": False, "path": None, "size": None, "sha256": None}
    resolved = path.resolve()
    if not resolved.is_file():
        return {"present": False, "path": str(resolved), "size": None, "sha256": None}
    return {
        "present": True,
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_extension_identity(record: object) -> None:
    if not isinstance(record, dict) or set(record) != EXTENSION_KEYS:
        raise ValueError("extension identity keys do not match schema version 1")
    present = record["present"]
    if not isinstance(present, bool):
        raise TypeError("extension present must be boolean")
    if not present:
        if record["size"] is not None or record["sha256"] is not None:
            raise ValueError("missing extension cannot carry size or digest")
        return
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError("present extension path must be a non-empty string")
    if not isinstance(record["size"], int) or isinstance(record["size"], bool) or record["size"] < 1:
        raise ValueError("present extension size must be a positive integer")
    if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64:
        raise ValueError("present extension sha256 must be 64 hex characters")


def oracle_pin_mismatches(profile: dict, fingerprint: dict) -> list[dict]:
    mismatches = []
    required = profile["upstream"]["version"]
    actual = fingerprint.get("sympy_version")
    if actual != required:
        mismatches.append(
            {
                "field": "sympy_version",
                "profile": required,
                "oracle": actual,
            }
        )
    expected_python = str(profile["environment"]["python_version"])
    actual_python = fingerprint.get("python")
    if not isinstance(actual_python, str) or not (
        actual_python == expected_python
        or actual_python.startswith(expected_python + ".")
    ):
        mismatches.append(
            {
                "field": "python",
                "profile": expected_python,
                "oracle": actual_python,
            }
        )
    expected_impl = str(profile["environment"]["implementation"])
    actual_impl = fingerprint.get("implementation")
    if not isinstance(actual_impl, str) or actual_impl.casefold() != expected_impl.casefold():
        mismatches.append(
            {
                "field": "implementation",
                "profile": expected_impl,
                "oracle": actual_impl,
            }
        )
    expected_env = profile["environment"]["env_overrides"]
    actual_env = fingerprint.get("env")
    if actual_env != expected_env:
        mismatches.append(
            {
                "field": "env",
                "profile": expected_env,
                "oracle": actual_env,
            }
        )
    return mismatches


def make_environment_receipt(
    *,
    profile: dict,
    oracle_environment: dict,
    candidate_environment: dict | None,
    extension: dict,
) -> dict:
    validate_extension_identity(extension)
    mismatches = oracle_pin_mismatches(profile, oracle_environment)
    return {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "lane": "environment_build",
        "certifies": False,
        "can_certify": False,
        "claims_promoted": False,
        "goldens_written": False,
        "profile_id": profile["profile_id"],
        "profile_upstream_commit": profile["upstream"]["commit"],
        "profile_upstream_version": profile["upstream"]["version"],
        "oracle": oracle_environment,
        "candidate": candidate_environment,
        "extension": extension,
        "mismatches": mismatches,
        "note": "environment/build capture cannot certify the immutable profile",
    }
