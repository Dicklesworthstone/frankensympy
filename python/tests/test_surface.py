"""Executable checks for the Python compatibility boundary."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest

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
