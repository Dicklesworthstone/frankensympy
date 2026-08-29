#!/usr/bin/env python3
"""Second-process pickle loader for the conformance laboratory.

Reads one JSON object with pickle_b64 from stdin and emits a pickle_restore
record. Candidate runs set FSYM_CANDIDATE_ROOT; oracle runs import site sympy.
"""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

PROFILE_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

if os.environ.get("PYTHONHASHSEED") != PROFILE_ENV["PYTHONHASHSEED"]:
    print(json.dumps({"schema_version": 1, "error_class": "harness_misuse"}))
    sys.exit(2)


def _preload_candidate() -> None:
    root = os.environ.get("FSYM_CANDIDATE_ROOT")
    if not root:
        return
    if root not in sys.path:
        sys.path.insert(0, root)
    ext_path = Path(__file__).resolve().parent / "extension.py"
    spec = importlib.util.spec_from_file_location("_fsym_lab_extension", ext_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    so = module.find_fsym_python_extension()
    if so is None or "fsym_python" in sys.modules:
        return
    loader = importlib.machinery.ExtensionFileLoader("fsym_python", str(so))
    ext_spec = importlib.util.spec_from_loader("fsym_python", loader)
    if ext_spec is None or ext_spec.loader is None:
        return
    ext_mod = importlib.util.module_from_spec(ext_spec)
    sys.modules["fsym_python"] = ext_mod
    ext_spec.loader.exec_module(ext_mod)


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps({"schema_version": 1, "error_class": "harness_misuse"}))
        return 2
    if not isinstance(payload, dict) or "pickle_b64" not in payload:
        print(json.dumps({"schema_version": 1, "error_class": "harness_misuse"}))
        return 2
    fixture_id = payload.get("fixture_id", "")
    side = payload.get("side", "")
    try:
        blob = base64.b64decode(payload["pickle_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pickle_restore",
                    "fixture_id": fixture_id,
                    "side": side,
                    "protocol": 4,
                    "status": "raised",
                    "error_class": type(exc).__module__ + "." + type(exc).__name__,
                    "message_head": str(exc)[:200],
                },
                sort_keys=True,
            )
        )
        return 0
    _preload_candidate()
    try:
        obj = pickle.loads(blob)
    except Exception as exc:  # noqa: BLE001 - restore errors ARE the observation
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pickle_restore",
                    "fixture_id": fixture_id,
                    "side": side,
                    "protocol": 4,
                    "status": "raised",
                    "error_class": type(exc).__module__ + "." + type(exc).__name__,
                    "message_head": str(exc)[:200],
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pickle_restore",
                "fixture_id": fixture_id,
                "side": side,
                "protocol": 4,
                "status": "returned",
                "type": type(obj).__name__,
                "module": type(obj).__module__,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
