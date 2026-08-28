"""Executable checks for the Python compatibility boundary."""

from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
import unittest
from pathlib import Path

import sympy


class SurfaceTests(unittest.TestCase):
    def test_classes_are_usable_and_operations_preserve_surface_kind(self):
        x = sympy.Symbol("x")
        two = sympy.Integer(2)

        self.assertIsInstance(x, sympy.Expr)
        self.assertIsInstance(two, sympy.Expr)
        expression = x + two
        self.assertIsInstance(expression, sympy.Add)
        self.assertEqual(expression.func(*expression.args), expression)
        self.assertEqual(expression.free_symbols, {x})
        self.assertEqual(expression.subs(x, 3), sympy.Integer(5))

    def test_held_forms_copy_and_pickle_without_collapsing_identity(self):
        x = sympy.Symbol("x")
        held = sympy.Add(x, x, evaluate=False)

        self.assertIsInstance(held, sympy.Add)
        self.assertEqual(len(held.args), 2)
        self.assertIs(copy.deepcopy(held), held)
        restored = pickle.loads(pickle.dumps(held))  # ubs:ignore — trusted in-process bytes
        self.assertIsInstance(restored, sympy.Add)
        self.assertEqual(restored.args, held.args)

    def test_function_wrappers_return_expressions_not_wire_strings(self):
        x = sympy.Symbol("x")

        self.assertEqual(sympy.diff(x**2, x), 2 * x)
        self.assertIsInstance(sympy.simplify(x + 0), sympy.Expr)
        self.assertEqual(sympy.integrate(2 * x, (x, 0, 1)), sympy.Integer(1))
        self.assertEqual(sympy.solve(2 * x - 4, x), [sympy.Integer(2)])
        with self.assertRaises(NotImplementedError):
            sympy.dsolve(x)

    def test_constants_are_native_constants_not_spoofed_symbols(self):
        for constant in (sympy.pi, sympy.E, sympy.I, sympy.oo, sympy.zoo, sympy.nan):
            self.assertIsInstance(constant, sympy.Expr)
            self.assertNotIsInstance(constant, sympy.Symbol)
            self.assertFalse(constant.is_symbol)

    def test_custom_subclasses_are_not_silently_collapsed_to_native_nodes(self):
        class CustomSymbol(sympy.Symbol):
            pass

        custom = CustomSymbol("custom")
        self.assertIs(type(custom), CustomSymbol)
        with self.assertRaisesRegex(NotImplementedError, "exact built-in classes only"):
            custom + 1

    def test_number_theory_wrappers_admit_only_exact_integers(self):
        with self.assertRaisesRegex(ValueError, r"^2\.9 is not an integer$"):
            sympy.isprime(2.9)
        with self.assertRaisesRegex(ValueError, r"^2\.9 is not an integer$"):
            sympy.factorint(2.9)
        with self.assertRaisesRegex(TypeError, r"^n should be an integer$"):
            sympy.totient(2.9)

        class LossyInteger:
            calls = 0

            def __int__(self):
                self.calls += 1
                return 2

            def __str__(self):
                self.calls += 1
                return "2"

            def __repr__(self):
                self.calls += 1
                return "2"

            def __format__(self, format_spec):
                del format_spec
                self.calls += 1
                return "2"

        lossy = LossyInteger()
        with self.assertRaises(ValueError):
            sympy.isprime(lossy)
        with self.assertRaises(ValueError):
            sympy.factorint(lossy)
        with self.assertRaises(TypeError):
            sympy.Integer(lossy)
        with self.assertRaises(TypeError):
            sympy.Rational(lossy, 1)
        self.assertEqual(lossy.calls, 0)

        self.assertEqual(sympy.Integer(2.9), sympy.Integer(2))
        self.assertEqual(sympy.Integer(True), sympy.Integer(1))
        self.assertEqual(
            sympy.Rational(1.9, 2),
            sympy.Rational(4278419646001971, 4503599627370496),
        )

    def test_numeric_bridge_preserves_values_beyond_machine_and_decimal_limits(self):
        huge = 1 << 20_000
        integer = sympy.Integer(huge)
        self.assertEqual(integer.p, huge)
        self.assertEqual(integer.q, 1)

        rational = sympy.Rational(huge, 3)
        self.assertEqual(rational.p, huge)
        self.assertEqual(rational.q, 3)
        trusted_payload = pickle.dumps(rational)
        self.assertEqual(
            pickle.loads(trusted_payload),  # nosec B301  # ubs:ignore — trusted in-process bytes
            rational,
        )

        normalized = sympy.Rational(-(1 << 63), -1)
        self.assertEqual(normalized.p, 1 << 63)
        self.assertEqual(normalized.q, 1)

    def test_custom_symbol_variable_refuses_before_running_overrides(self):
        effects = []

        class EffectfulSymbol(sympy.Symbol):
            @property
            def name(self):
                effects.append("name")
                return "x"

        variable = EffectfulSymbol("x")
        with self.assertRaisesRegex(
            NotImplementedError, "supervised Python override lane"
        ):
            sympy.diff(sympy.Symbol("x"), variable)
        self.assertEqual(effects, [])

    def test_number_theory_wrappers_preserve_signed_and_zero_domains(self):
        self.assertTrue(sympy.isprime(sympy.Integer(2)))
        self.assertFalse(sympy.isprime(-2))
        self.assertEqual(sympy.factorint(-12), {2: 2, 3: 1, -1: 1})
        self.assertEqual(sympy.factorint(0), {0: 1})
        with self.assertRaisesRegex(ValueError, r"^n should be a positive integer$"):
            sympy.totient(0)

    def test_missing_native_extension_fails_closed(self):
        package_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(package_root)
        completed = subprocess.run(
            [sys.executable, "-S", "-c", "import sympy"],
            cwd=package_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires its fsym_python native extension", completed.stderr)


if __name__ == "__main__":
    unittest.main()
