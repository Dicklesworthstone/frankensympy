"""Per-fixture latex differential: shell vs pinned SymPy 1.14.0 oracle.

Runs the identical corpus builder under both interpreters — the
compatibility shell is a drop-in for the corpus grammar, so the same
driver (which imports ``sympy``, builds each fixture, and prints its
latex) exercises the shell when PYTHONPATH selects it and the pinned
oracle otherwise. Environment-gated on the pinned oracle venv.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "tools" / "conformance-lab"
CORPUS_FILES = [
    str(LAB / "fixtures" / "seed_core_atoms.json"),
    str(LAB / "fixtures" / "seed_held_forms.json"),
    str(LAB / "fixtures" / "generated_corpus_r2.json"),
    str(LAB / "fixtures" / "seed_function_subclass.json"),
    str(LAB / "fixtures" / "adversarial_corpus_r2.json"),
    str(LAB / "fixtures" / "seed_structural_subclass.json"),
    # seed_simplification_shapes.json: builder support landed; 4 latex
    # divergences pending (function-power exponent placement, Add
    # function-term ordering, div over-cancellation, negative-Pow Add
    # order). Register once those land - see surface-nvv bead.
]

ORACLE_PYTHON_CANDIDATES = [
    "/home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0/bin/python",
]

# The builder imports sympy, builds each fixture per the corpus grammar
# (mirroring tools/conformance-lab/candidate_runner.py construction), and
# prints {"latex": {id: rendered}, "unsupported": [ids]}.
BUILDER_DRIVER = r'''
import json
import sys
import sys as _sys


def make_function_subclass(spec, sympy_mod):
    name = spec["name"]

    def eval(cls, *a):
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
    cls = type(name, (sympy_mod.Function,), namespace)
    module = _sys.modules.get(cls.__module__)
    if module is not None:
        setattr(module, name, cls)
    return cls


def build(fixture, sympy_mod):
    kind = fixture["kind"]
    args = fixture.get("args", [])
    kwargs = fixture.get("kwargs", {})
    if kind == "integer":
        return sympy_mod.Integer(args[0])
    if kind == "rational":
        return sympy_mod.Rational(args[0], args[1])
    if kind == "symbol":
        spec = args[0]
        return sympy_mod.Symbol(spec["sym"], **kwargs)
    if kind in ("add", "mul", "pow", "held_add", "held_mul"):
        evaluate = kind not in ("held_add", "held_mul")
        target = (
            sympy_mod.Add
            if kind in ("add", "held_add")
            else sympy_mod.Mul if kind in ("mul", "held_mul") else sympy_mod.Pow
        )
        built = []
        for arg in args:
            if isinstance(arg, dict):
                arg = build({**fixture, "kind": "symbol", "args": [arg]}, sympy_mod)
            built.append(arg)
        return target(*built, evaluate=evaluate)
    if kind == "function_subclass":
        cls = make_function_subclass(fixture["subclass"], sympy_mod)
        call_args = []
        for arg in fixture.get("call_args", []):
            if isinstance(arg, dict):
                call_args.append(
                    build({**fixture, "kind": "symbol", "args": [arg]}, sympy_mod)
                )
            else:
                call_args.append(sympy_mod.Integer(arg))
        return cls(*call_args)
    if kind == "function":
        fn = getattr(sympy_mod, fixture["name"])
        call_args = []
        for arg in fixture.get("args", []):
            if isinstance(arg, dict):
                call_args.append(
                    build({**fixture, "kind": "symbol", "args": [arg]}, sympy_mod)
                )
            else:
                call_args.append(sympy_mod.Integer(arg) if isinstance(arg, int) else arg)
        return fn(*call_args)
    if kind == "simplification_case":
        x = sympy_mod.Symbol("x")
        y = sympy_mod.Symbol("y")
        a = sympy_mod.Symbol("a")
        b = sympy_mod.Symbol("b")
        cases = {
            "trig_identity": sympy_mod.sin(x) ** 2 + sympy_mod.cos(x) ** 2,
            "unsorted_mul": sympy_mod.tan(x) * sympy_mod.cos(x),
            "same_base_pows": x ** 2 * x ** 3,
            "nested_pow": (x ** 2) ** 3,
            "exp_products": sympy_mod.exp(x) * sympy_mod.exp(y),
            "symbolic_exp_pows": a ** x * a ** y,
            "radical_constant": sympy_mod.sqrt(8),
            "negative_half": 1 / sympy_mod.sqrt(2),
            "rational_function": (x ** 2 - 1) / (x + 1),
            "multivariate_rational": 1 / x + 1 / y,
            "numeric_coeff_add": 2 * x + 4,
        }
        return cases[fixture["case"]]
    if kind in ("expr_subclass", "basic_subclass"):
        spec = fixture["subclass"]
        base_name = "Expr" if kind == "expr_subclass" else "Basic"
        cls = make_structural_subclass(spec["name"], base_name, sympy_mod)
        call_args = []
        for arg in fixture.get("call_args", []):
            if isinstance(arg, dict):
                call_args.append(
                    build({**fixture, "kind": "symbol", "args": [arg]}, sympy_mod)
                )
            else:
                call_args.append(arg)
        return cls(*call_args)
    raise NotImplementedError(f"unsupported fixture kind {kind}")


def make_structural_subclass(name, base_name, sympy_mod):
    """Create an arbitrary Basic/Expr subclass with structural __new__."""
    base = getattr(sympy_mod, base_name)

    def structural_new(cls, *a, _base=base):
        return _base.__new__(cls, *a)

    cls = type(name, (base,), {"__new__": structural_new})
    module = _sys.modules.get(cls.__module__)
    if module is not None:
        setattr(module, name, cls)
    return cls


def main() -> None:
    import sympy

    out = {}
    unsupported = []
    for path in sys.argv[1:]:
        with open(path) as handle:
            fixtures = json.load(handle)
        if isinstance(fixtures, dict):
            fixtures = fixtures.get("fixtures", [])
        for fixture in fixtures:
            try:
                expr = build(fixture, sympy)
                out[fixture["id"]] = sympy.latex(expr)
            except NotImplementedError:
                unsupported.append(fixture["id"])
                out[fixture["id"]] = "NOT_IMPLEMENTED"
            except Exception as exc:  # noqa: BLE001 - the differential records all
                out[fixture["id"]] = f"ERROR: {exc}"
    print(json.dumps({"latex": out, "unsupported": unsupported}))


main()
'''


def oracle_python() -> str | None:
    env = os.environ.get("FSYM_ORACLE_PYTHON")
    if env and Path(env).is_file():
        return Path(env)
    for candidate in ORACLE_PYTHON_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def run_driver(
    python: str,
    extra_env: dict[str, str] | None = None,
    strip_pythonpath: bool = False,
) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix="_latex_driver.py", delete=False) as handle:
        handle.write(BUILDER_DRIVER)
        driver = handle.name
    env = dict(os.environ)
    if strip_pythonpath:
        env.pop("PYTHONPATH", None)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [python, driver, *CORPUS_FILES],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=str(REPO),
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"builder failed under {python}: {result.stderr[-2000:]}")
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])


class LatexCorpusDifferential(unittest.TestCase):
    def test_shell_latex_matches_pinned_oracle_per_fixture(self) -> None:
        oracle = oracle_python()
        if oracle is None:
            self.skipTest("pinned oracle venv not present")
        oracle_output = run_driver(str(oracle), strip_pythonpath=True)

        shell_python = str(REPO / ".venv-conformance" / "bin" / "python")
        shell_output = run_driver(
            shell_python, {"PYTHONPATH": str(REPO / "python")}
        )

        shell = shell_output["latex"]
        expected = oracle_output["latex"]
        mismatches = {
            fixture_id: {"shell": shell.get(fixture_id), "oracle": oracle_value}
            for fixture_id, oracle_value in expected.items()
            if shell.get(fixture_id) != oracle_value
        }
        self.assertEqual(
            mismatches,
            {},
            f"{len(mismatches)} latex mismatches against the pinned oracle: "
            + json.dumps(mismatches, indent=1)[:4000],
        )
        # Everything the oracle could render, the shell rendered too.
        unimplemented = [
            fixture_id
            for fixture_id, rendered in shell.items()
            if rendered == "NOT_IMPLEMENTED"
        ]
        self.assertEqual(unimplemented, [], "shell latex left fixtures unimplemented")


if __name__ == "__main__":
    unittest.main()
