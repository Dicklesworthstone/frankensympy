"""Free and bound integral variables observed against isolated SymPy 1.14.0.

The shell and native arithmetic must agree about dependency sets. These
regressions do not claim complete substitution or calculus compatibility.
"""

import unittest

from sympy import Dummy, Integral, Integer, Symbol, symbols


class IntegralBindingTests(unittest.TestCase):
    def test_constant_indefinite_integrals_retain_their_coordinate(self):
        x = Symbol("constant_integral_coordinate")
        for integrand in (Integer(0), Integer(1)):
            with self.subTest(integrand=integrand):
                value = Integral(integrand, x)
                self.assertEqual(value.free_symbols, {x})
                self.assertFalse(value.is_number)
                self.assertEqual(value.bound_symbols, [])
                self.assertEqual(value.variables, [x])
                self.assertEqual((value + 7).free_symbols, {x})

    def test_nested_limits_bind_only_their_enclosed_scopes(self):
        x, y, z = symbols("x y z")
        cases = [
            (Integral(x*y, x), {x, y}, []),
            (Integral(x*y, (x, 0, y)), {y}, [x]),
            (Integral(x*y, (x, 0, y), (y, 0, 1)), set(), [x, y]),
            (Integral(x*y, x, (x, 0, 1)), {y}, [x]),
            (Integral(x*y, (x, 0, 1), x), {x, y}, [x]),
            (Integral(x*y, (x, 0, 1), (y, 0, x)), {x}, [x, y]),
            (Integral(x*y, x, (y, 0, z)), {x, z}, [y]),
        ]
        for value, expected, bound in cases:
            with self.subTest(value=value):
                self.assertEqual(value.free_symbols, expected)
                self.assertEqual(value.bound_symbols, bound)
                self.assertEqual(value.is_number, not expected)
                self.assertEqual((value + 7).free_symbols, expected)

    def test_same_printed_names_preserve_distinct_symbol_owners(self):
        pairs = [
            (Symbol("coordinate", positive=True), Symbol("coordinate")),
            (Dummy("coordinate"), Dummy("coordinate")),
        ]
        for bound, free in pairs:
            with self.subTest(bound=bound, free=free):
                value = Integral(bound + free, (bound, 0, 1))
                self.assertEqual(value.free_symbols, {free})
                self.assertEqual(value.bound_symbols, [bound])
                self.assertFalse(value.is_number)
                self.assertEqual((value + 7).free_symbols, {free})


if __name__ == "__main__":
    unittest.main()
