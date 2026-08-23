#!/usr/bin/env python3
"""Reflection/source inventory generator for the conformance laboratory.

Runs inside the isolated oracle interpreter (subprocess of this script's
driver mode) and reflects over the pinned SymPy build to produce the
"initial reflection/source inventory" required by campaign stage C1
(docs/FIRST_IMPLEMENTATION_CAMPAIGN.md section 6) following
docs/CONFORMANCE_AND_BENCHMARKING.md section 4.

Scope discipline: inventories the campaign-slice surface (core atoms,
arithmetic heads, Function machinery, printers, assumptions entry points)
plus module-level topology. This is structural CI input; behavioral probes
live in fixtures, not here.

Output: one JSON object on stdout with deterministic (sorted) ordering so
the content digest is stable across runs on the same profile.

Usage (normally invoked via capture.py inventory):
  oracle_python inventory_runner.py <profile_id>
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import platform
import sys

import sympy

PROFILE_ENV = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}

if os.environ.get("PYTHONHASHSEED") != PROFILE_ENV["PYTHONHASHSEED"]:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "error_class": "harness_misuse",
                "detail": f"PYTHONHASHSEED must be {PROFILE_ENV['PYTHONHASHSEED']!r}",
            }
        )
    )
    sys.exit(2)

# Campaign-slice modules (C1 scope; expansion is profile-versioned).
SLICE_MODULES = [
    "sympy",
    "sympy.core",
    "sympy.core.basic",
    "sympy.core.expr",
    "sympy.core.symbol",
    "sympy.core.numbers",
    "sympy.core.function",
    "sympy.core.relational",
    "sympy.core.add",
    "sympy.core.mul",
    "sympy.core.power",
    "sympy.printing",
    "sympy.printing.str",
    "sympy.printing.latex",
    "sympy.printing.pretty",
    "sympy.assumptions",
]

# Names whose observable surface the C1/C3 fixtures exercise first.
SLICE_NAMES = [
    "Basic", "Atom", "Expr", "Symbol", "Dummy", "Integer", "Rational",
    "Float", "Add", "Mul", "Pow", "Function", "Lambda", "Derivative",
    "S", "srepr", "latex", "pretty", "symbols", "sympify", "E", "pi",
]


def _signature_of(obj) -> dict | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    parts = []
    for name, param in sig.parameters.items():
        kind = {
            inspect.Parameter.POSITIONAL_ONLY: "/",
            inspect.Parameter.POSITIONAL_OR_KEYWORD: "",
            inspect.Parameter.KEYWORD_ONLY: "kw",
            inspect.Parameter.VAR_POSITIONAL: "*",
            inspect.Parameter.VAR_KEYWORD: "**",
        }[param.kind]
        rendered = f"{name}{kind}"
        if param.default is not inspect.Parameter.empty:
            rendered += "=?"
        parts.append(rendered)
    return {"params": parts}


def _class_record(cls: type) -> dict:
    record: dict = {
        "module": cls.__module__,
        "mro": [f"{c.__module__}.{c.__qualname__}" for c in cls.__mro__],
        "slots": sorted(getattr(cls, "__slots__", ()) or ()),
    }
    init_sig = _signature_of(cls.__init__)
    if init_sig:
        record["init_params"] = init_sig["params"]
    hooks = [
        h
        for h in (
            "_eval_is", "_eval_derivative", "_eval_rewrite", "_eval_subs",
            "_eval_evalf", "_eval_simplify_", "eval", "fdiff", "nargs",
        )
        if any(h in vars(base) or h in getattr(base, "__dict__", {}) for base in cls.__mro__[:-1])
    ]
    record["declared_hooks"] = sorted(set(hooks))
    return record


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error_class": "harness_misuse"}))
        return 2

    modules = {}
    for mod_name in SLICE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - import failure IS inventory data
            modules[mod_name] = {"import_error": type(exc).__name__}
            continue
        exports = sorted(getattr(mod, "__all__", []) or [])
        classes = {}
        for name in SLICE_NAMES:
            obj = getattr(mod, name, None)
            if obj is None or not inspect.isclass(obj):
                continue
            classes[name] = _class_record(obj)
        functions = {}
        for name in SLICE_NAMES:
            obj = getattr(mod, name, None)
            if obj is not None and inspect.isfunction(obj):
                functions[name] = _signature_of(obj)
        modules[mod_name] = {
            "all_exports_count": len(exports),
            "slice_classes": classes,
            "slice_functions": functions,
        }

    identity = {
        "singletons": {
            name: repr(getattr(sympy, name))
            for name in ("S", "E", "pi")
        },
        "identity_relations": {
            "S.One_times_SOne_is_SOne": sympy.S.One * sympy.S.One == sympy.S.One,
            "Integer_1_is_S_One": sympy.Integer(1) is sympy.S.One,
        },
    }

    inventory = {
        "schema_version": 1,
        "kind": "reflection_inventory",
        "profile_id": sys.argv[1],
        "environment": {
            "sympy_version": sympy.__version__,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "sympy_path": sympy.__file__,
            "env": {k: os.environ.get(k) for k in PROFILE_ENV},
        },
        "modules": modules,
        "identity": identity,
    }
    # Digest over canonical bytes so downstream gates can pin this artifact.
    canonical = json.dumps(inventory, sort_keys=True).encode()
    inventory["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    sys.stdout.write(json.dumps(inventory, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
