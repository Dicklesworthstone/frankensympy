#!/usr/bin/env python3
"""Oracle-side runner for the FrankenSymPy conformance laboratory.

Executed as a subprocess of capture.py inside the isolated oracle venv.
Reads one fixture file (JSON) plus the profile id from argv, constructs each
fixture inside the live SymPy process, and emits exactly one NDJSON
observation envelope per fixture on stdout.

Design rules (docs/CONFORMANCE_AND_BENCHMARKING.md):
- Observations are normalized envelopes; the comparator cannot infer omitted
  fields, so every implemented probe fills its field or records "unavailable".
- Unknown fixture kinds fail closed with an error envelope instead of being
  skipped silently.
- No state is shared with any other process; nothing is imported beyond the
  pinned oracle itself and the standard library.

Exit codes: 0 = all fixtures observed (including per-fixture error classes),
2 = harness misuse (bad arguments), 3 = crash outside a fixture boundary.
"""

from __future__ import annotations

import base64
import copy as copy_mod
import hashlib
import importlib
import json
import os
import pickle
import platform
import sys
import warnings as warnings_mod

import sympy

PROFILE_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

# Fail closed if launched without the pinned hash seed: hash observations
# would otherwise be irreproducible across runs.
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


def _mro_chain(obj: object) -> list[str]:
    return [f"{c.__module__}.{c.__qualname__}" for c in type(obj).__mro__]


def _printer_outputs(expr) -> dict[str, str]:
    return {
        "str": str(expr),
        "repr": repr(expr),
        "srepr": sympy.srepr(expr),
        "latex": sympy.latex(expr),
        "pretty_ascii": str(sympy.pretty(expr)),
    }


def _pickle_observations(obj) -> dict[str, object]:
    out: dict[str, object] = {}
    for protocol in (4, 5):
        blob = pickle.dumps(obj, protocol=protocol)
        out[f"pickle_v{protocol}"] = {
            "length": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
    return out


_SUBCLASS_REGISTRY: dict[str, type] = {}


def _make_function_subclass(spec: dict) -> type:
    """Creates a custom Function subclass per the corpus specification.

    Supports the seed-corpus behaviors: classmethod eval with zero-collapse,
    nargs declaration, and fdiff-based custom derivatives. The class object is
    created dynamically inside the oracle process so observations include real
    upstream metaclass behavior.

    SymPy invokes ``cls.eval(*args)`` through a classmethod, so the function
    must accept the class as the first argument. A bare ``eval(*a)`` wrapped
    in ``classmethod`` sees ``a[0] is cls`` and never matches a numeric zero.
    The class is also installed on its defining module so pickle can resolve
    ``__main__.Name`` / ``oracle_runner.Name``.
    """
    name = spec["name"]

    def eval(cls, *a):  # noqa: ANN202 - SymPy calls this as a classmethod
        del cls
        if spec.get("eval_zero_collapse") and len(a) == 2 and a[0] == 0:
            return sympy.S.Zero
        return None  # returning None keeps the applied, unevaluated form

    def fdiff(self, argindex=1):
        del self
        return sympy.S.One / argindex

    namespace = {
        "eval": classmethod(eval),
        "nargs": tuple(spec.get("nargs", (2,))),
        "fdiff": fdiff,
    }
    cls = type(name, (sympy.Function,), namespace)
    module = sys.modules.get(cls.__module__)
    if module is not None:
        setattr(module, name, cls)
    _SUBCLASS_REGISTRY[name] = cls
    return cls


def _resolve_arg(raw):
    """Fixture args are JSON-safe terms resolved against the oracle."""
    if isinstance(raw, dict) and "sym" in raw:
        return sympy.Symbol(raw["sym"], **raw.get("assumptions", {}))
    return raw


def _construct(spec: dict):
    """Builds the fixture object; unknown kinds fail closed."""
    kind = spec["kind"]
    args = [_resolve_arg(a) for a in spec.get("args", [])]
    kwargs = spec.get("kwargs", {})
    constructors = {
        "integer": lambda: sympy.Integer(*args),
        "rational": lambda: sympy.Rational(*args),
        "symbol": lambda: sympy.Symbol(*args, **kwargs),
        "add": lambda: sympy.Add(*args),
        "mul": lambda: sympy.Mul(*args),
        "pow": lambda: sympy.Pow(*args),
        "held_add": lambda: sympy.Add(*args, evaluate=False),
        "held_mul": lambda: sympy.Mul(*args, evaluate=False),
        "function_subclass": lambda: _make_function_subclass(spec["subclass"])(
            *[_resolve_arg(a) for a in spec.get("call_args", [])]
        ),
    }
    if kind not in constructors:
        raise KeyError(f"unknown fixture kind: {kind!r}")
    return constructors[kind]()


def _unique_warning_classes(caught: list) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in caught:
        record = {
            "module": item.category.__module__,
            "name": item.category.__name__,
        }
        if record not in out:
            out.append(record)
    return out


MAX_PICKLE_BYTES = 256 * 1024


def observe_pickle_dump(fixture: dict, profile_id: str) -> dict:
    try:
        obj = _construct(fixture)
        outcome = "returned"
    except Exception:  # noqa: BLE001 - construction outcome is the observation
        return {
            "schema_version": 1,
            "kind": "pickle_dump",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": "raised",
            "protocol": 4,
            "pickle_sha256": None,
            "pickle_b64": None,
            "dump_error": None,
        }
    try:
        blob = pickle.dumps(obj, protocol=4)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": 1,
            "kind": "pickle_dump",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": "returned",
            "protocol": 4,
            "pickle_sha256": None,
            "pickle_b64": None,
            "dump_error": {
                "error_class": type(exc).__module__ + "." + type(exc).__name__,
                "message_head": str(exc)[:200],
            },
        }
    if len(blob) > MAX_PICKLE_BYTES:
        return {
            "schema_version": 1,
            "kind": "pickle_dump",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": "returned",
            "protocol": 4,
            "pickle_sha256": None,
            "pickle_b64": None,
            "dump_error": {
                "error_class": "harness.pickle_too_large",
                "message_head": f"{len(blob)} bytes exceeds {MAX_PICKLE_BYTES}",
            },
        }
    return {
        "schema_version": 1,
        "kind": "pickle_dump",
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": "upstream_oracle",
        "construction_outcome": "returned",
        "protocol": 4,
        "pickle_sha256": hashlib.sha256(blob).hexdigest(),
        "pickle_b64": base64.b64encode(blob).decode("ascii"),
        "dump_error": None,
    }


def _copy_surface(obj) -> dict:
    return {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
        "args_repr": [repr(arg) for arg in getattr(obj, "args", ())],
    }


def _copy_result(obj, copied) -> dict:
    surface = _copy_surface(copied)
    surface["is_original"] = copied is obj
    return surface


def observe_copy(fixture: dict, profile_id: str) -> dict:
    try:
        obj = _construct(fixture)
        outcome = "returned"
    except Exception:  # noqa: BLE001
        return {
            "schema_version": 1,
            "kind": "copy_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": "raised",
            "original": None,
            "copy": None,
            "deepcopy": None,
        }
    try:
        copied = copy_mod.copy(obj)
        copy_obs = _copy_result(obj, copied)
    except Exception as exc:  # noqa: BLE001
        copy_obs = {
            "error_class": type(exc).__module__ + "." + type(exc).__name__,
            "message_head": str(exc)[:200],
        }
    try:
        deep = copy_mod.deepcopy(obj)
        deep_obs = _copy_result(obj, deep)
    except Exception as exc:  # noqa: BLE001
        deep_obs = {
            "error_class": type(exc).__module__ + "." + type(exc).__name__,
            "message_head": str(exc)[:200],
        }
    return {
        "schema_version": 1,
        "kind": "copy_observation",
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": "upstream_oracle",
        "construction_outcome": outcome,
        "original": _copy_surface(obj),
        "copy": copy_obs,
        "deepcopy": deep_obs,
    }


def observe_reconstruct(fixture: dict, profile_id: str) -> dict:
    try:
        obj = _construct(fixture)
        outcome = "returned"
    except Exception:  # noqa: BLE001
        return {
            "schema_version": 1,
            "kind": "reconstruction_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": "raised",
            "original": None,
            "reconstructed": None,
        }
    try:
        func = getattr(obj, "func", None)
        args = getattr(obj, "args", ())
        rebuilt = func(*args)
        recon_obs = _copy_result(obj, rebuilt)
    except Exception as exc:  # noqa: BLE001
        recon_obs = {
            "error_class": type(exc).__module__ + "." + type(exc).__name__,
            "message_head": str(exc)[:200],
        }
    return {
        "schema_version": 1,
        "kind": "reconstruction_observation",
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": "upstream_oracle",
        "construction_outcome": outcome,
        "original": _copy_surface(obj),
        "reconstructed": recon_obs,
    }


def observe_isinstance(fixture: dict, profile_id: str) -> dict:
    try:
        obj = _construct(fixture)
        outcome = "returned"
    except Exception:  # noqa: BLE001
        return {
            "schema_version": 1,
            "kind": "isinstance_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": "raised",
            "isinstance_type": None,
            "issubclass_type": None,
            "module_class_importable": None,
            "isinstance_module_class": None,
            "issubclass_module_class": None,
            "type_is_module_class": None,
            "probe_error": None,
        }
    cls = type(obj)
    try:
        imported = getattr(importlib.import_module(cls.__module__), cls.__name__)
        if not isinstance(imported, type):
            raise TypeError(f"{cls.__module__}.{cls.__name__} is not a type")
        return {
            "schema_version": 1,
            "kind": "isinstance_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": outcome,
            "isinstance_type": isinstance(obj, cls),
            "issubclass_type": issubclass(cls, cls),
            "module_class_importable": True,
            "isinstance_module_class": isinstance(obj, imported),
            "issubclass_module_class": issubclass(cls, imported),
            "type_is_module_class": cls is imported,
            "probe_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": 1,
            "kind": "isinstance_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": outcome,
            "isinstance_type": isinstance(obj, cls),
            "issubclass_type": issubclass(cls, cls),
            "module_class_importable": False,
            "isinstance_module_class": None,
            "issubclass_module_class": None,
            "type_is_module_class": None,
            "probe_error": {
                "error_class": type(exc).__module__ + "." + type(exc).__name__,
                "message_head": str(exc)[:200],
            },
        }


def observe_equality(fixture: dict, profile_id: str) -> dict:
    try:
        first = _construct(fixture)
        outcome = "returned"
    except Exception:  # noqa: BLE001
        return {
            "schema_version": 1,
            "kind": "equality_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": "raised",
            "equal_to_twin": None,
            "hashes_agree": None,
            "is_same_object": None,
            "probe_error": None,
        }
    try:
        second = _construct(fixture)
        equal = first == second
        if type(equal) is not bool:
            raise TypeError(
                f"equality probe produced {type(equal).__module__}.{type(equal).__name__}"
            )
        return {
            "schema_version": 1,
            "kind": "equality_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": outcome,
            "equal_to_twin": equal,
            "hashes_agree": hash(first) == hash(second),
            "is_same_object": first is second,
            "probe_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": 1,
            "kind": "equality_observation",
            "profile_id": profile_id,
            "fixture_id": fixture["id"],
            "side": "upstream_oracle",
            "construction_outcome": outcome,
            "equal_to_twin": None,
            "hashes_agree": None,
            "is_same_object": None,
            "probe_error": {
                "error_class": type(exc).__module__ + "." + type(exc).__name__,
                "message_head": str(exc)[:200],
            },
        }


def observe_warnings(fixture: dict, profile_id: str) -> dict:
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        try:
            _construct(fixture)
            outcome = "returned"
        except Exception:  # noqa: BLE001 - construction outcome is the observation
            outcome = "raised"
    return {
        "schema_version": 1,
        "kind": "warning_observation",
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": "upstream_oracle",
        "construction_outcome": outcome,
        "warnings": _unique_warning_classes(caught),
    }


def observe(fixture: dict, profile_id: str, env_fingerprint: dict) -> dict:
    envelope: dict[str, object] = {
        "schema_version": 1,
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": "upstream_oracle",
        "environment": env_fingerprint,
    }
    try:
        obj = _construct(fixture)
    except Exception as exc:  # noqa: BLE001 - exception classes ARE the observation
        envelope.update(
            {
                "outcome_class": "raised",
                "observations": {
                    "exception_module": type(exc).__module__,
                    "exception_type": type(exc).__name__,
                    "message_head": str(exc)[:200],
                },
            }
        )
        return envelope

    def probe(fn, *a):  # noqa: ANN202 - returns value or an explicit error record
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001
            return {
                "probe_error_class": type(exc).__module__ + "." + type(exc).__name__,
                "message_head": str(exc)[:200],
            }

    envelope.update(
        {
            "outcome_class": "returned",
            "observations": {
                "type": type(obj).__name__,
                "module": type(obj).__module__,
                "mro": _mro_chain(obj),
                "is_number": bool(getattr(obj, "is_number", False)),
                "free_symbols": sorted(str(s) for s in getattr(obj, "free_symbols", set())),
                "hash_sha256_of_py_hash": probe(
                    lambda: hashlib.sha256(str(hash(obj)).encode()).hexdigest()
                ),
                "printers": probe(_printer_outputs, obj),
                "args_repr": [repr(a) for a in getattr(obj, "args", ())],
                "func": repr(getattr(obj, "func", None)),
                **{
                    f"pickle_v{protocol}": probe(
                        lambda p=protocol: (
                            lambda blob: {
                                "length": len(blob),
                                "sha256": hashlib.sha256(blob).hexdigest(),
                            }
                        )(pickle.dumps(obj, protocol=p))
                    )
                    for protocol in (4, 5)
                },
            },
        }
    )
    return envelope


def main() -> int:
    args = sys.argv[1:]
    warnings_only = False
    pickle_roundtrip = False
    copy_roundtrip = False
    reconstruct = False
    equality = False
    instancecheck = False
    if args and args[-1] in {
        "--warnings",
        "--pickle-roundtrip",
        "--copy-roundtrip",
        "--reconstruct",
        "--equality",
        "--isinstance",
    }:
        flag = args.pop()
        warnings_only = flag == "--warnings"
        pickle_roundtrip = flag == "--pickle-roundtrip"
        copy_roundtrip = flag == "--copy-roundtrip"
        reconstruct = flag == "--reconstruct"
        equality = flag == "--equality"
        instancecheck = flag == "--isinstance"
    if len(args) != 2:
        print(json.dumps({"error_class": "harness_misuse"}))
        return 2
    fixture_path, profile_id = args
    with open(fixture_path, encoding="utf-8") as fh:
        fixtures = json.load(fh)
    if instancecheck:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(observe_isinstance(fixture, profile_id), sort_keys=True)
                + "\n"
            )
            sys.stdout.flush()
        return 0
    if equality:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(observe_equality(fixture, profile_id), sort_keys=True) + "\n"
            )
            sys.stdout.flush()
        return 0
    if reconstruct:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(observe_reconstruct(fixture, profile_id), sort_keys=True)
                + "\n"
            )
            sys.stdout.flush()
        return 0
    if copy_roundtrip:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(observe_copy(fixture, profile_id), sort_keys=True) + "\n"
            )
            sys.stdout.flush()
        return 0
    if pickle_roundtrip:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(observe_pickle_dump(fixture, profile_id), sort_keys=True) + "\n"
            )
            sys.stdout.flush()
        return 0
    if warnings_only:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(observe_warnings(fixture, profile_id), sort_keys=True) + "\n"
            )
            sys.stdout.flush()
        return 0
    env_fingerprint = {
        "sympy_version": sympy.__version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "sympy_path": sympy.__file__,
        "env": {k: os.environ.get(k) for k in PROFILE_ENV},
    }
    for fixture in fixtures:
        envelope = observe(fixture, profile_id, env_fingerprint)
        sys.stdout.write(json.dumps(envelope, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
