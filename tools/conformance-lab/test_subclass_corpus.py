"""Subclass corpus must actually exercise eval, pickle, and the golden mismatch."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import load_goldens, load_profile, oracle_environment, oracle_python  # noqa: E402

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"
GOLDEN = (
    LAB.parents[1]
    / "artifacts"
    / "conformance"
    / "sympy-1.14.0-cpython"
    / "goldens"
    / "seed_function_subclass.ndjson"
)

LIVE_PROBE = r"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, %r)
from oracle_runner import _construct
import sympy

applied = _construct({
    "id": "applied",
    "kind": "function_subclass",
    "subclass": {"name": "ConstitutiveLaw", "nargs": [2], "eval_zero_collapse": True},
    "call_args": [{"sym": "x"}, {"sym": "k"}],
})
collapsed = _construct({
    "id": "zero",
    "kind": "function_subclass",
    "subclass": {"name": "ConstitutiveLawZero", "nargs": [2], "eval_zero_collapse": True},
    "call_args": [0, {"sym": "k"}],
})

def broken_eval(*a):
    if len(a) == 2 and a[0] == 0:
        return sympy.S.Zero
    return None

BrokenZero = type(
    "BrokenZero",
    (sympy.Function,),
    {"eval": classmethod(broken_eval), "nargs": (2,)},
)
broken = BrokenZero(0, sympy.Symbol("k"))
restored = pickle.loads(pickle.dumps(applied, protocol=4))
print(json.dumps({
    "applied_type": type(applied).__name__,
    "collapsed_type": type(collapsed).__name__,
    "collapsed_is_zero": collapsed == sympy.S.Zero,
    "broken_still_applied": type(broken).__name__ == "BrokenZero",
    "pickle_restored_type": type(restored).__name__,
    "eval_param0": applied.func.eval.__func__.__code__.co_varnames[0],
}))
"""


def _oracle_python_or_skip() -> str:
    try:
        return oracle_python(None)
    except SystemExit as exc:
        raise unittest.SkipTest(str(exc)) from exc


class SubclassCorpusTests(unittest.TestCase):
    def test_stale_golden_still_records_the_harness_bug(self) -> None:
        """Do not regenerate goldens; the stored zero-collapse case is wrong."""
        envelopes = [
            json.loads(line)
            for line in GOLDEN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {envelope["fixture_id"]: envelope for envelope in envelopes}
        stale = by_id["subclass/ConstitutiveLawZero_zero_collapse"]
        self.assertEqual(stale["observations"]["type"], "ConstitutiveLawZero")
        self.assertEqual(
            stale["observations"]["pickle_v4"]["probe_error_class"],
            "_pickle.PicklingError",
        )

    def test_pretty_ascii_golden_currently_contains_unicode_dot(self) -> None:
        """pretty() is called without use_unicode=False; changing it mutates core goldens."""
        profile = load_profile(PROFILE_PATH)
        goldens = load_goldens(profile)
        mul = next(
            envelope
            for envelope in goldens["seed_core_atoms.ndjson"]
            if envelope["fixture_id"] == "core/mul/three_x_sq"
        )
        pretty = mul["observations"]["printers"]["pretty_ascii"]
        self.assertIn("⋅", pretty)
        source = (LAB / "oracle_runner.py").read_text(encoding="utf-8")
        self.assertIn("sympy.pretty(expr)", source)
        self.assertNotIn("use_unicode=False", source)

    def test_live_oracle_eval_collapses_zero_and_pickles(self) -> None:
        py = _oracle_python_or_skip()
        profile = load_profile(PROFILE_PATH)
        proc = subprocess.run(
            [py, "-P", "-s", "-c", LIVE_PROBE % str(LAB)],
            capture_output=True,
            text=True,
            env=oracle_environment(profile),
            timeout=60,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout.splitlines()[-1])
        self.assertEqual(report["applied_type"], "ConstitutiveLaw")
        self.assertEqual(report["collapsed_type"], "Zero")
        self.assertTrue(report["collapsed_is_zero"])
        self.assertTrue(report["broken_still_applied"])
        self.assertEqual(report["pickle_restored_type"], "ConstitutiveLaw")
        self.assertEqual(report["eval_param0"], "cls")

    def test_eval_source_accepts_classmethod_cls(self) -> None:
        source = (LAB / "oracle_runner.py").read_text(encoding="utf-8")
        self.assertIn("def eval(cls, *a):", source)
        candidate = (LAB / "candidate_runner.py").read_text(encoding="utf-8")
        self.assertIn("def eval(cls, *a):", candidate)


if __name__ == "__main__":
    unittest.main()
