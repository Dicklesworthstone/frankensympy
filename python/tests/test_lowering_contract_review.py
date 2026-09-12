"""Reviewer-owned adversarial tests for the typed lowering/lifting contract.

Gate `fra-rc-lowering-gate-ba5` reviewing implementation `fra-rc-lowering-8w3`
at commit `7820929`. Written by the gate reviewer, not by the implementation
author. The file attacks the contract from outside the implementation: every
case goes through the public Python surface (`python/sympy`) or the public
native extension (`fsym_python`), and the oracle column in the gate evidence
is SymPy 1.14.0 from the isolated conformance venv.

Classes
-------
``LoweringContractTests``
    Must-hold contract cases (a)-(e) of the gate: identity across distinct
    declarations, lifting identity, typed-binding refusal, receipt replay and
    tamper refusal, and the historical plain-symbol behaviour.

``UntestedOperationIdentityFindings``
    EXPECTED TO FAIL on commit 7820929. These encode the gate findings, not
    reviewer mistakes:

    * after 4096 distinct declared symbols the bounded lift registry drops the
      oldest declaration, so a declared symbol silently reports
      ``free_symbols == set()`` and differentiation raises;
    * the remaining name-keyed algorithm lanes (``integrate``, ``series``, ...)
      return a fresh *plain* same-name atom, so the lifted result is not the
      declared object although SymPy 1.14.0 returns the declared object.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import unittest

import sympy
import fsym_python as native
from sympy.core import diff_with_receipt, replay_diff_receipt


class LoweringContractTests(unittest.TestCase):
    """Contract cases (a)-(e) that must hold on the reviewed commit."""

    def test_same_printed_text_with_distinct_declarations_does_not_merge(self):
        positive = sympy.Symbol("x", positive=True)
        plain = sympy.Symbol("x")
        negative = sympy.Symbol("x", negative=True)
        positive_nonzero = sympy.Symbol("x", positive=True, nonzero=True)

        # Same printed view for every declaration.
        for symbol in (positive, plain, negative, positive_nonzero):
            self.assertEqual(str(symbol), "x")

        # ...but no two declarations share a native atom.
        atoms = [
            positive._value,
            plain._value,
            negative._value,
            positive_nonzero._value,
        ]
        for left in range(len(atoms)):
            for right in range(len(atoms)):
                self.assertEqual(
                    left == right,
                    atoms[left] == atoms[right],
                    f"native atoms {left}/{right} merged",
                )

        # Two identical declarations are the same atom and hash equal.
        self.assertEqual(positive, sympy.Symbol("x", positive=True))
        self.assertEqual(hash(positive), hash(sympy.Symbol("x", positive=True)))

        # A same-name stranger variable yields derivative 0, in both directions.
        self.assertEqual(str(sympy.diff(positive**2, plain)), "0")
        self.assertEqual(str(sympy.diff(positive**2, negative)), "0")
        self.assertEqual(str(sympy.diff(plain**2, positive)), "0")
        self.assertEqual(str(sympy.diff(positive**2, positive)), "2*x")

        # Lowering digests are deterministic and declaration-driven: identical
        # declarations share one digest, distinct ones never do, and the plain
        # binding is its own well-defined point (not a zero digest).
        binding = native.SymbolBinding("x", [("positive", "True")])
        repeat = native.SymbolBinding("x", [("positive", "True")])
        other = native.SymbolBinding("x", [("negative", "True")])
        plain_binding = native.SymbolBinding("x", [])
        self.assertFalse(binding.plain)
        self.assertEqual(binding.identity_hex, repeat.identity_hex)
        self.assertNotEqual(binding.identity_hex, other.identity_hex)
        self.assertTrue(plain_binding.plain)
        self.assertEqual(plain_binding.identity_hex, native.SymbolBinding("x", []).identity_hex)
        self.assertNotEqual(plain_binding.identity_hex, binding.identity_hex)

    def test_lifting_returns_the_declared_surface_objects(self):
        keyed = sympy.Symbol("x", positive=True)
        plain = sympy.Symbol("x")

        # An identity-bearing symbol lifts back onto the very declared object.
        derivative = sympy.diff(keyed**3, keyed)
        self.assertEqual(str(derivative), "3*x**2")
        self.assertIs(next(iter(derivative.free_symbols)), keyed)
        self.assertEqual(derivative.free_symbols, {keyed})
        self.assertIs(next(iter((keyed**2).free_symbols)), keyed)
        self.assertTrue(any(arg is keyed for arg in (keyed**2).args))

        # Same-name distinct declarations stay separate through free_symbols.
        mixed = keyed * plain
        self.assertEqual(len(mixed.free_symbols), 2)
        self.assertEqual(mixed.free_symbols, {keyed, plain})

        # A plain symbol still compares equal to the declared object even
        # though the shell does not intern plain symbols (SymPy 1.14.0 does
        # cache symbols; `is` identity for plain atoms is a known shell gap
        # that predates this change and is recorded in the gate evidence).
        self.assertEqual((plain**2).free_symbols, {plain})
        self.assertIn(plain, (plain**2).free_symbols)

        # Pickle keeps the declaration, so the restored atom is the same atom.
        restored = pickle.loads(pickle.dumps(keyed))
        self.assertEqual(restored, keyed)
        self.assertEqual(restored._value, keyed._value)
        self.assertEqual(restored.free_symbols, {keyed})

    def test_native_differentiation_requires_a_typed_binding(self):
        plain = sympy.Symbol("x")

        # A printed string is refused at the lowering boundary.
        with self.assertRaises(ValueError) as raised:
            native.Expr("x**2").diff("x")
        self.assertIn("typed SymbolBinding", str(raised.exception))

        # A raw native Expr is refused too: only a typed handle is accepted.
        with self.assertRaises(ValueError):
            native.Expr("x**2").diff(plain._value)

        # The binding path returns the derivative.
        binding = native.SymbolBinding("x", [])
        self.assertEqual(str(native.Expr("x**2").diff(binding)), "2*x")

        # The shell-level string form is SymPy-compatible (SymPy sympifies the
        # variable to a plain symbol); it is a different lane from the typed one.
        self.assertEqual(str(sympy.diff(plain**2, "x")), "2*x")
        self.assertEqual(str(sympy.diff(sympy.Symbol("x", positive=True) ** 2, "x")), "0")

    def test_custom_symbol_subclass_is_refused_before_its_overrides_run(self):
        calls: list[str] = []

        class AuditedSymbol(sympy.Symbol):
            def _eval_derivative(self, variable):
                calls.append("_eval_derivative")
                return sympy.Symbol("evil_derivative")

            def __str__(self):
                calls.append("__str__")
                return super().__str__()

            def __eq__(self, other):
                calls.append("__eq__")
                return super().__eq__(other)

            def __hash__(self):
                calls.append("__hash__")
                return super().__hash__()

            @property
            def name(self):
                calls.append("name")
                return super().name

        audited = AuditedSymbol("x")
        plain = sympy.Symbol("x")

        with self.assertRaises(NotImplementedError) as raised:
            sympy.diff(plain**2, audited)
        self.assertIn("supervised Python override lane", str(raised.exception))
        with self.assertRaises(NotImplementedError):
            sympy.diff(audited**2, audited)

        # The refusal must not have consulted any user override.
        self.assertEqual(calls, [])

    def test_receipt_replays_and_every_tampered_field_is_refused(self):
        keyed = sympy.Symbol("x", positive=True)
        plain = sympy.Symbol("x")

        derivative, receipt = diff_with_receipt(keyed**3, keyed)
        self.assertEqual(str(derivative), "3*x**2")
        self.assertEqual(receipt["schema"], "fsym.diff.receipt.v1")
        self.assertEqual(receipt["variable"], "x")
        self.assertFalse(receipt["variable_plain"])
        self.assertEqual(receipt["symbols"], [("x", native.SymbolBinding("x", [("positive", "True")]).identity_hex)])
        self.assertTrue(replay_diff_receipt(receipt))

        # A receipt is never authority by itself: every semantic field is
        # recomputed and re-verified.
        tampered = {
            "rhs=0": {"rhs": "0"},
            "derivative=0": {"derivative": "0"},
            "rhs+derivative=0": {"rhs": "0", "derivative": "0"},
            "rule=diff_sum": {"rule": "diff_sum"},
            "expr=x": {"expr": "x"},
            "symbols dropped": {"symbols": []},
            "symbols identity zeroed": {
                "symbols": [("x", "00" * 32)],
                "rhs": "0",
                "derivative": "0",
            },
            "variable identity zeroed": {"variable_identity": "00" * 32},
            "variable identity zeroed + consistent rhs": {
                "variable_identity": "00" * 32,
                "rhs": "0",
                "derivative": "0",
            },
            "variable_plain flipped": {"variable_plain": True},
        }
        for label, patch in tampered.items():
            with self.subTest(tamper=label):
                self.assertFalse(replay_diff_receipt(dict(receipt, **patch)))

        # An untampered plain-variable receipt replays as well.
        _, plain_receipt = diff_with_receipt(plain**3, plain)
        self.assertTrue(plain_receipt["variable_plain"])
        self.assertTrue(replay_diff_receipt(plain_receipt))

        # A non-symbol variable cannot be lowered into the receipt lane.
        with self.assertRaises(TypeError):
            diff_with_receipt(plain**2, "x")
        with self.assertRaises(TypeError):
            diff_with_receipt(plain**2, 3)

    def test_plain_symbols_keep_their_historical_surface_identity(self):
        plain = sympy.Symbol("x")
        self.assertEqual(str(sympy.diff(plain**2, plain)), "2*x")
        self.assertEqual(str(sympy.diff(plain**3, plain)), "3*x**2")
        self.assertEqual(str(sympy.diff(plain**2, sympy.Symbol("y"))), "0")
        self.assertTrue(native.SymbolBinding("x", []).plain)
        # The frozen cross-architecture golden for a plain symbol is replayed
        # independently in crates/fsym-core/tests/symbol_identity_review.rs.


class UntestedOperationIdentityFindings(unittest.TestCase):
    """Reproductions of identity holes the implementation did not test.

    Both tests are expected to FAIL on commit 7820929; the raw failures are the
    gate evidence.
    """

    def test_finding_bounded_lift_registry_loses_the_oldest_declaration(self):
        """At >4096 distinct declared symbols the lift target is evicted.

        The registry is bounded by `_SURFACE_SYMBOL_LIMIT`, and once the entry
        is gone a declared symbol silently reports an empty `free_symbols`
        while `diff` raises instead of lifting. This is run in a child process
        so the process-global eviction cannot poison the other contract tests.
        """
        shell_root = os.path.dirname(os.path.dirname(os.path.abspath(sympy.__file__)))
        child = r"""
import sympy

first = sympy.Symbol("review_sat_0", positive=True)
others = [sympy.Symbol(f"review_sat_{i}", positive=True) for i in range(1, 4097)]
print("registry_churn=" + str(len(others)))
print("free_symbols=" + repr(sorted(str(symbol) for symbol in first.free_symbols)))
print("args=" + repr(first.args))
try:
    print("diff=" + str(sympy.diff(first**2, first)))
except Exception as error:  # noqa: BLE001 - raw outcome is the evidence
    print("diff_error=" + type(error).__name__ + ": " + str(error))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = shell_root + os.pathsep + environment.get("PYTHONPATH", "")
        completed = subprocess.run(
            [sys.executable, "-c", child],
            capture_output=True,
            text=True,
            env=environment,
            timeout=300,
            cwd=os.path.dirname(shell_root) or ".",
        )
        raw = f"exit={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        self.assertEqual(completed.returncode, 0, raw)
        observed = dict(
            line.split("=", 1)
            for line in completed.stdout.splitlines()
            if "=" in line
        )
        with self.subTest(observation="free_symbols for a declared symbol"):
            self.assertEqual(observed.get("free_symbols"), "['review_sat_0']", raw)
        with self.subTest(observation="differentiation of a declared symbol"):
            self.assertEqual(observed.get("diff"), "2*review_sat_0", raw)

    def test_finding_name_keyed_lane_returns_an_undeclared_look_alike(self):
        """`integrate`/`series` lift a fresh plain atom, not the declaration.

        SymPy 1.14.0 returns the declared symbol, so `free_symbols == {x}` is
        True in the oracle and must hold here too. It does not, and a
        same-name stranger variable produces the same result, so the lane is
        still name-keyed end to end for structurally identical atoms.
        """
        keyed = sympy.Symbol("x", positive=True)
        plain = sympy.Symbol("x")

        integrated = sympy.integrate(keyed**2, keyed)
        self.assertEqual(str(integrated), "x**3/3")
        with self.subTest(lane="integrate free_symbols"):
            self.assertEqual(integrated.free_symbols, {keyed})
        with self.subTest(lane="integrate distinguishes stranger variable"):
            self.assertNotEqual(sympy.integrate(keyed**2, plain), integrated)

        series = sympy.series(sympy.sin(keyed), keyed, 0, 3)
        with self.subTest(lane="series free_symbols"):
            self.assertEqual(series.free_symbols, {keyed})


    def test_finding_equivalent_assumption_spellings_become_distinct_atoms(self):
        """Differently spelled but equivalent declarations split into atoms.

        SymPy 1.14.0 treats `Symbol('x', positive=1)` and
        `Symbol('x', positive=True)` as the same symbol (and the pre-change
        shell did too, because both lowered to a plain native atom). The typed
        identity is a digest over the *spelled* fact value, so the two become
        distinct atoms: differentiation against the other spelling returns 0
        and substitution is a no-op, while the pinned oracle returns 2*x / 25.
        """
        spelled = sympy.Symbol("x", positive=1)
        canonical = sympy.Symbol("x", positive=True)
        loose = sympy.Symbol("x", foo=1)
        loose_bool = sympy.Symbol("x", foo=True)

        with self.subTest(observation="equality"):
            self.assertEqual(spelled, canonical)
        with self.subTest(observation="hash"):
            self.assertEqual(hash(spelled), hash(canonical))
        with self.subTest(observation="differentiation across spellings"):
            self.assertEqual(str(sympy.diff(canonical**2, spelled)), "2*x")
        with self.subTest(observation="substitution across spellings"):
            self.assertEqual((canonical**2).subs(spelled, sympy.Integer(5)), 25)
        with self.subTest(observation="arbitrary fact spelling"):
            self.assertEqual(loose, loose_bool)


if __name__ == "__main__":
    unittest.main()
