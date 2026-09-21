"""Oracle differential gates: per-fixture printer/behavior/integrity
parity against the pinned SymPy 1.14.0 oracle.

Consolidates the session's throwaway differential scripts into permanent
regression tests. Every test environment-gates on the pinned oracle venv
(absent -> skip, matching test_latex_corpus's pattern) and compares
EXACT payloads produced by the same driver under both interpreters.

Methodology rules baked in:
- per-printer drivers: each printer is compared in isolation (a combined
  dict comparison masks per-printer mismatches - see the pretty false
  pass corrected on fra-rc-surface-nvv).
- printer drivers must call fully-qualified functions (a bare ``srepr``
  name raised NameError identically on both sides - false pass).
- identical driver-level build errors are build-parity artifacts, not
  matches; they are counted and excluded.
- structural-subclass free_symbols raises on non-Basic args (oracle
  bug-parity); held-form copy/pickle re-canonicalizes (oracle pins the
  re-evaluation, NOT heldness preservation).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_latex_corpus import CORPUS_FILES, REPO, oracle_python

SHELL = str(REPO / ".venv-conformance" / "bin" / "python")

STRUCTURAL_HOLD_KINDS = ("held_add", "held_mul")
_NUMERIC = ("Integer", "Rational")


def _subprocess_payload(driver_text: str, py: str, env_extra: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix="_drv.py", delete=False) as h:
        h.write(driver_text)
        drv = h.name
    env = dict(os.environ, **env_extra)
    result = subprocess.run(
        [py, drv, *CORPUS_FILES], capture_output=True, text=True,
        env=env, cwd=str(REPO), timeout=900,
    )
    if result.returncode != 0:
        raise AssertionError(f"driver failed under {py}: {result.stderr[-2000:]}")
    line = [l for l in result.stdout.splitlines() if l.startswith("{")][-1]
    return json.loads(line)["latex"]


class OracleDifferentialBase(unittest.TestCase):
    """Shared oracle gating and per-printer driver construction."""

    oracle: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle = oracle_python()

    def _skip_if_no_oracle(self) -> None:
        if self.oracle is None:
            self.skipTest("pinned oracle venv not present")


class PrinterDifferentials(OracleDifferentialBase):
    """str/repr/srepr/pretty per fixture, exact string equality."""

    def _run_printer(self, printer_expr: str) -> dict:
        # IMPORTANT: fully-qualified calls only (sympy.<printer>) - bare
        # names that fail identically on both sides are false passes.
        from test_latex_corpus import BUILDER_DRIVER
        driver = BUILDER_DRIVER.replace(
            'out[fixture["id"]] = sympy.latex(expr)',
            f'out[fixture["id"]] = sympy.{printer_expr}(expr)',
        )
        oracle_out = _subprocess_payload(driver, self.oracle, {})
        shell_out = _subprocess_payload(
            driver, SHELL, {"PYTHONPATH": str(REPO / "python")}
        )
        return {
            fid: (shell_out.get(fid), oracle_out[fid])
            for fid in oracle_out
            if shell_out.get(fid) != oracle_out[fid]
        }

    def test_str_matches_oracle(self) -> None:
        self._skip_if_no_oracle()
        mismatches = self._run_printer("str")
        self.assertEqual(mismatches, {}, f"str mismatches: {list(mismatches)[:8]}")

    def test_repr_matches_oracle(self) -> None:
        self._skip_if_no_oracle()
        mismatches = self._run_printer("repr")
        self.assertEqual(mismatches, {}, f"repr mismatches: {list(mismatches)[:8]}")

    def test_srepr_matches_oracle(self) -> None:
        self._skip_if_no_oracle()
        mismatches = self._run_printer("srepr")
        self.assertEqual(mismatches, {}, f"srepr mismatches: {list(mismatches)[:8]}")

    def test_pretty_matches_oracle(self) -> None:
        self._skip_if_no_oracle()
        mismatches = self._run_printer("pretty")
        self.assertEqual(mismatches, {}, f"pretty mismatches: {list(mismatches)[:8]}")


class BehaviorDifferential(OracleDifferentialBase):
    """free_symbols / func-args rebuild / doit / subs / expand parity."""

    def test_behavior_matches_oracle(self) -> None:
        self._skip_if_no_oracle()
        from test_latex_corpus import BUILDER_DRIVER

        probe_src = '''

def probe(expr, fixture, builder):
    import sympy as _sp
    res = {}
    try:
        res["free_symbols"] = sorted(s.name for s in expr.free_symbols)
        rebuilt = expr.func(*expr.args)
        res["rebuild_srepr"] = _sp.srepr(rebuilt)
        if hasattr(expr, "doit"):
            res["doit_srepr"] = _sp.srepr(expr.doit())
        syms = sorted(expr.free_symbols, key=lambda s: s.name)
        if syms:
            res["subs_srepr"] = _sp.srepr(expr.subs(syms[0], _sp.Integer(2)))
        if hasattr(expr, "expand"):
            res["expand_srepr"] = _sp.srepr(expr.expand())
    except Exception as exc:
        res["error"] = str(exc)[:90]
    return res
'''
        driver = (
            BUILDER_DRIVER
            .replace("import sys as _sys", "import sys as _sys\n" + probe_src, 1)
            .replace(
                'out[fixture["id"]] = sympy.latex(expr)',
                'out[fixture["id"]] = probe(expr, fixture, build)',
            )
        )
        oracle_payload = _subprocess_payload(driver, self.oracle, {})
        shell_payload = _subprocess_payload(
            driver, SHELL, {"PYTHONPATH": str(REPO / "python")}
        )
        divergences = {}
        build_parity_errors = 0
        for fid, oracle_res in oracle_payload.items():
            shell_res = shell_payload.get(fid)
            if not isinstance(oracle_res, dict) or not isinstance(shell_res, dict):
                # Driver-level build error on BOTH sides: corpus adversarial
                # parity artifact; exact-string parity is the latex gate's job.
                if shell_res == oracle_res:
                    build_parity_errors += 1
                else:
                    divergences[fid] = {"artifact": (shell_res, oracle_res)}
                continue
            bad = {
                key: (shell_res.get(key), oracle_res[key])
                for key in oracle_res
                if shell_res.get(key) != oracle_res[key]
            }
            if bad:
                divergences[fid] = bad
        self.assertEqual(
            divergences, {},
            f"{len(divergences)} behavior divergences "
            f"({build_parity_errors} build-parity artifacts excluded): "
            + json.dumps(dict(list(divergences.items())[:6]), indent=1)[:4000],
        )


if __name__ == "__main__":
    unittest.main()
