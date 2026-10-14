#!/usr/bin/env python3
"""Second-process pickle loader for the conformance laboratory.

Reads one JSON object with pickle_b64 from stdin and emits a pickle_restore
record. Candidate runs set FSYM_CANDIDATE_ROOT; oracle runs import site sympy.
"""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import io
import json
import os
import pickle
import pickletools
import sys
from pathlib import Path

PROFILE_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

MAX_STDIN_BYTES = 256 * 1024
MAX_PICKLE_BYTES = 64 * 1024
MAX_PICKLE_OPS = 4096
MAX_RESTORE_DEPTH = 64

ALLOWED_BUILTINS = frozenset(
    {
        "bool",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "object",
        "set",
        "slice",
        "str",
        "tuple",
    }
)
ALLOWED_COPYREG = frozenset({"_reconstructor", "__newobj__", "__newobj_ex__"})

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


def _is_allowed_global(module: str, name: str) -> bool:
    if not module or not name or "\x00" in module or "\x00" in name:
        return False
    if module == "builtins":
        return name in ALLOWED_BUILTINS
    if module in {"copyreg", "copy_reg"}:
        return name in ALLOWED_COPYREG
    if module == "sympy" or module.startswith("sympy."):
        parts = name.split(".")
        return all(part.isidentifier() and not part.startswith("__") for part in parts)
    return False


class RestrictedUnpickler(pickle.Unpickler):
    """Allowlist-only unpickler for the conformance laboratory."""

    def find_class(self, module: str, name: str) -> object:
        if not _is_allowed_global(module, name):
            raise pickle.UnpicklingError(
                f"forbidden pickle global outside allowlist: {module}.{name}"
            )
        cls = super().find_class(module, name)
        # Refuse types with custom __setstate__ on non-SymPy classes
        if not (module == "sympy" or module.startswith("sympy.")):
            if hasattr(cls, "__setstate__"):
                raise pickle.UnpicklingError(
                    f"forbidden __setstate__ on non-sympy class: {module}.{name}"
                )
        return cls


def preflight_pickle_bytes(blob: bytes) -> None:
    """Preflight byte size, opcode count, clean STOP termination, and nesting depth."""
    if not blob:
        raise pickle.UnpicklingError("empty pickle payload")
    if len(blob) > MAX_PICKLE_BYTES:
        raise ValueError(f"pickle payload exceeds MAX_PICKLE_BYTES ({len(blob)} > {MAX_PICKLE_BYTES})")
    ops = 0
    depth = 0
    max_depth = 0
    saw_stop = False
    push_ops = {
        "MARK",
        "EMPTY_TUPLE",
        "EMPTY_LIST",
        "EMPTY_DICT",
        "EMPTY_SET",
        "TUPLE1",
        "TUPLE2",
        "TUPLE3",
        "NEWOBJ",
        "NEWOBJ_EX",
        "REDUCE",
        "INST",
        "OBJ",
    }
    try:
        for opcode, arg, pos in pickletools.genops(blob):
            ops += 1
            if ops > MAX_PICKLE_OPS:
                raise ValueError(f"pickle opcode count exceeds MAX_PICKLE_OPS ({MAX_PICKLE_OPS})")
            opname = opcode.name
            if opname in {"GLOBAL", "INST"} and isinstance(arg, str):
                parts = arg.split(" ", 1)
                if len(parts) != 2 or not _is_allowed_global(parts[0], parts[1]):
                    raise pickle.UnpicklingError(f"forbidden global opcode target: {arg}")
            if opname in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}:
                raise pickle.UnpicklingError(f"forbidden pickle extension/persistent opcode: {opname}")
            if opname in push_ops:
                depth += 1
                if depth > max_depth:
                    max_depth = depth
                if max_depth > MAX_RESTORE_DEPTH:
                    raise ValueError(
                        f"pickle nesting depth exceeds MAX_RESTORE_DEPTH ({MAX_RESTORE_DEPTH})"
                    )
            elif opname in {"POP_MARK", "TUPLE", "LIST", "DICT", "FROZENSET"} and depth > 0:
                depth -= 1
            if opname == "STOP":
                saw_stop = True
                # Refuse trailing bytes after STOP
                if pos is not None and pos + 1 < len(blob):
                    raise pickle.UnpicklingError("trailing bytes after pickle STOP")
                break
    except ValueError as exc:
        if "exceeds MAX_" in str(exc):
            raise
        raise pickle.UnpicklingError(f"malformed pickle opcode stream: {exc}") from exc
    if not saw_stop:
        raise pickle.UnpicklingError("truncated pickle stream missing STOP")


def safe_pickle_loads(blob: bytes) -> object:
    preflight_pickle_bytes(blob)
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(min(old_limit, 200))
    try:
        return RestrictedUnpickler(io.BytesIO(blob)).load()
    finally:
        sys.setrecursionlimit(old_limit)


def _emit_raised(fixture_id: str, side: str, exc: BaseException) -> int:
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


def main() -> int:
    raw = sys.stdin.read(MAX_STDIN_BYTES + 1)
    if len(raw) > MAX_STDIN_BYTES:
        return _emit_raised(
            "",
            "",
            ValueError(f"stdin payload exceeds MAX_STDIN_BYTES ({MAX_STDIN_BYTES})"),
        )
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
    b64_raw = payload["pickle_b64"]
    if not isinstance(b64_raw, str) or len(b64_raw) > (MAX_PICKLE_BYTES * 2):
        return _emit_raised(
            fixture_id,
            side,
            ValueError("base64 pickle payload exceeds preflight size cap"),
        )
    try:
        blob = base64.b64decode(b64_raw, validate=True)
    except (ValueError, TypeError) as exc:
        return _emit_raised(fixture_id, side, exc)
    _preload_candidate()
    try:
        obj = safe_pickle_loads(blob)
    except Exception as exc:  # noqa: BLE001 - restore errors ARE the observation
        return _emit_raised(fixture_id, side, exc)
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
