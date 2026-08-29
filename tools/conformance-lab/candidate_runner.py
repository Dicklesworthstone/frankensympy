#!/usr/bin/env python3
"""Candidate-side runner for the FrankenSymPy conformance laboratory.

Executed as a subprocess of capture.py. Reads one fixture file (JSON) plus
the profile id from argv and emits exactly one NDJSON observation envelope
per fixture on stdout.

This process is the FrankenSymPy compatibility lane. It must not observe
through the isolated upstream oracle interpreter: if `sympy` resolves to
the oracle tree, the runner exits 3 (`isolation_violation`) instead of
emitting goldens-shaped envelopes.

Modes:
  default     Import the in-repo `python/sympy` candidate shell.
  --broken    Deliberate C1 mutant: emit wrong construction identity
              without importing sympy at all. Used only to prove the
              comparator rejects a broken candidate shell.

Exit codes: 0 = all fixtures observed, 2 = harness misuse, 3 = isolation
violation, 4 = crash outside a fixture boundary.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import pickle
import platform
import sys
from pathlib import Path

PROFILE_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}
CANDIDATE_SIDE = "frankensympy_candidate"
BROKEN_TYPE = "BrokenCandidate"

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


def _find_fsym_python_extension():
    # python -P omits the script directory from sys.path, so load by file.
    path = Path(__file__).resolve().parent / "extension.py"
    spec = importlib.util.spec_from_file_location("_fsym_lab_extension", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_fsym_python_extension()


def candidate_root() -> Path:
    env = os.environ.get("FSYM_CANDIDATE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2] / "python"


def isolation_violation(detail: str) -> int:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "error_class": "isolation_violation",
                "detail": detail,
            },
            sort_keys=True,
        )
    )
    return 3


def _mro_chain(obj: object) -> list[str]:
    return [f"{c.__module__}.{c.__qualname__}" for c in type(obj).__mro__]


def _probe(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:  # noqa: BLE001 - probe errors ARE observations
        return {
            "probe_error_class": type(exc).__module__ + "." + type(exc).__name__,
            "message_head": str(exc)[:200],
        }


def _printer_outputs(expr, sympy_mod) -> dict[str, object]:
    return {
        "str": _probe(lambda: str(expr)),
        "repr": _probe(lambda: repr(expr)),
        "srepr": _probe(
            lambda: sympy_mod.srepr(expr)
            if hasattr(sympy_mod, "srepr")
            else (_ for _ in ()).throw(AttributeError("srepr unavailable"))
        ),
        "latex": _probe(
            lambda: sympy_mod.latex(expr)
            if hasattr(sympy_mod, "latex")
            else expr._repr_latex_()
        ),
        "pretty_ascii": _probe(
            lambda: str(sympy_mod.pretty(expr))
            if hasattr(sympy_mod, "pretty")
            else (_ for _ in ()).throw(AttributeError("pretty unavailable"))
        ),
    }


def _assert_candidate_sympy(sympy_mod) -> str | None:
    file_name = getattr(sympy_mod, "__file__", None)
    if not isinstance(file_name, str) or not file_name:
        return "candidate sympy module has no __file__"
    path = Path(file_name).resolve()
    expected = (candidate_root() / "sympy").resolve()
    try:
        path.relative_to(expected)
    except ValueError:
        return (
            "candidate sympy resolved outside the FrankenSymPy python shell: "
            f"actual={path} expected_under={expected}"
        )
    return None


def preload_fsym_python() -> str | None:
    """Load a cargo-built cdylib as fsym_python before importing the shell.

    PyO3 writes `libfsym_python.so`; CPython imports `fsym_python.so`.
    Preloading the lib-prefixed artifact into sys.modules avoids a  copy
    and does not require maturin.
    """
    if "fsym_python" in sys.modules:
        return None
    so = _find_fsym_python_extension()
    if so is None:
        return None
    loader = importlib.machinery.ExtensionFileLoader("fsym_python", str(so))
    spec = importlib.util.spec_from_loader("fsym_python", loader)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fsym_python"] = module
    spec.loader.exec_module(module)
    return str(so)


def import_candidate_sympy():
    root = candidate_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        preload_fsym_python()
    except Exception as exc:  # noqa: BLE001
        return None, "import", f"extension preload failed: {type(exc).__name__}: {exc}"[:400]
    try:
        import sympy
    except ImportError as exc:
        return None, "import", f"{type(exc).__name__}: {exc}"[:400]
    detail = _assert_candidate_sympy(sympy)
    if detail:
        return None, "isolation", detail
    return sympy, None, None


def _resolve_arg(raw, sympy_mod):
    if isinstance(raw, dict) and "sym" in raw:
        kwargs = raw.get("assumptions", {})
        return sympy_mod.Symbol(raw["sym"], **kwargs)
    return raw


def _construct(spec: dict, sympy_mod):
    kind = spec["kind"]
    args = [_resolve_arg(a, sympy_mod) for a in spec.get("args", [])]
    kwargs = spec.get("kwargs", {})
    constructors = {
        "integer": lambda: sympy_mod.Integer(*args),
        "rational": lambda: sympy_mod.Rational(*args),
        "symbol": lambda: sympy_mod.Symbol(*args, **kwargs),
        "add": lambda: sympy_mod.Add(*args),
        "mul": lambda: sympy_mod.Mul(*args),
        "pow": lambda: sympy_mod.Pow(*args),
        "held_add": lambda: sympy_mod.Add(*args, evaluate=False),
        "held_mul": lambda: sympy_mod.Mul(*args, evaluate=False),
        "function_subclass": lambda: _make_function_subclass(spec["subclass"], sympy_mod)(
            *[_resolve_arg(a, sympy_mod) for a in spec.get("call_args", [])]
        ),
    }
    if kind not in constructors:
        raise KeyError(f"unknown fixture kind: {kind!r}")
    return constructors[kind]()


def _make_function_subclass(spec: dict, sympy_mod):
    parent = getattr(sympy_mod, "Function", None)
    if parent is None:
        raise NotImplementedError("candidate shell has no Function base class")
    name = spec["name"]

    def eval(cls, *a):  # noqa: ANN202 - SymPy calls this as a classmethod
        del cls
        if spec.get("eval_zero_collapse") and len(a) == 2 and a[0] == 0:
            zero = getattr(getattr(sympy_mod, "S", None), "Zero", 0)
            return zero
        return None

    def fdiff(self, argindex=1):
        del self
        return 1 / argindex

    namespace = {
        "eval": classmethod(eval),
        "nargs": tuple(spec.get("nargs", (2,))),
        "fdiff": fdiff,
    }
    cls = type(name, (parent,), namespace)
    module = sys.modules.get(cls.__module__)
    if module is not None:
        setattr(module, name, cls)
    return cls


def env_fingerprint(sympy_mod, *, broken: bool) -> dict:
    if broken:
        sympy_path = "broken-candidate"
        version = "broken-candidate"
    else:
        sympy_path = getattr(sympy_mod, "__file__", "")
        version = getattr(sympy_mod, "__version__", "")
    return {
        "sympy_version": version,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "sympy_path": sympy_path,
        "env": {k: os.environ.get(k) for k in PROFILE_ENV},
        "candidate_root": str(candidate_root()),
        "executable": sys.executable,
    }


def observe_broken(fixture: dict, profile_id: str) -> dict:
    """C1 mutant: returned construction identity that is never correct."""
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": CANDIDATE_SIDE,
        "environment": env_fingerprint(None, broken=True),
        "outcome_class": "returned",
        "observations": {
            "type": BROKEN_TYPE,
            "module": "broken_candidate",
            "mro": ["broken_candidate.BrokenCandidate", "builtins.object"],
            "is_number": False,
            "free_symbols": [],
            "hash_sha256_of_py_hash": "0" * 64,
            "printers": {
                "str": "BROKEN",
                "repr": "BROKEN",
                "srepr": "BROKEN",
                "latex": "BROKEN",
                "pretty_ascii": "BROKEN",
            },
            "args_repr": ["BROKEN"],
            "func": "BrokenCandidate",
            "pickle_v4": {"length": 0, "sha256": "0" * 64},
            "pickle_v5": {"length": 0, "sha256": "0" * 64},
        },
    }


def observe_real(fixture: dict, profile_id: str, sympy_mod, fingerprint: dict) -> dict:
    envelope: dict[str, object] = {
        "schema_version": 1,
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": CANDIDATE_SIDE,
        "environment": fingerprint,
    }
    try:
        obj = _construct(fixture, sympy_mod)
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

    envelope.update(
        {
            "outcome_class": "returned",
            "observations": {
                "type": type(obj).__name__,
                "module": type(obj).__module__,
                "mro": _mro_chain(obj),
                "is_number": bool(getattr(obj, "is_number", False)),
                "free_symbols": sorted(
                    str(s) for s in getattr(obj, "free_symbols", set())
                ),
                "hash_sha256_of_py_hash": _probe(
                    lambda: hashlib.sha256(str(hash(obj)).encode()).hexdigest()
                ),
                "printers": _printer_outputs(obj, sympy_mod),
                "args_repr": [repr(a) for a in getattr(obj, "args", ())],
                "func": repr(getattr(obj, "func", None)),
                **{
                    f"pickle_v{protocol}": _probe(
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


def observe_refused(fixture: dict, profile_id: str, reason: str) -> dict:
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "fixture_id": fixture["id"],
        "side": CANDIDATE_SIDE,
        "environment": env_fingerprint(None, broken=True)
        | {
            "sympy_version": "unavailable",
            "sympy_path": "unavailable",
        },
        "outcome_class": "refused",
        "observations": {
            "reason": "candidate_import_failed",
            "message_head": reason[:200],
        },
    }


def main() -> int:
    args = sys.argv[1:]
    broken = False
    if args and args[-1] == "--broken":
        broken = True
        args = args[:-1]
    if len(args) != 2:
        print(json.dumps({"error_class": "harness_misuse"}))
        return 2
    fixture_path, profile_id = args
    with open(fixture_path, encoding="utf-8") as fh:
        fixtures = json.load(fh)
    if broken:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(observe_broken(fixture, profile_id), sort_keys=True) + "\n"
            )
            sys.stdout.flush()
        return 0

    sympy_mod, error_kind, error = import_candidate_sympy()
    if error_kind == "isolation":
        return isolation_violation(error or "candidate sympy is not the FrankenSymPy shell")
    if sympy_mod is None:
        for fixture in fixtures:
            sys.stdout.write(
                json.dumps(
                    observe_refused(fixture, profile_id, error or "import failed"),
                    sort_keys=True,
                )
                + "\n"
            )
            sys.stdout.flush()
        return 0

    fingerprint = env_fingerprint(sympy_mod, broken=False)
    for fixture in fixtures:
        envelope = observe_real(fixture, profile_id, sympy_mod, fingerprint)
        sys.stdout.write(json.dumps(envelope, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - convert unexpected crashes
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
        sys.exit(4)
