"""Integral bindings and hand-derived formal differentiation identities.

The shell and native arithmetic must agree about dependency sets and lexical
scope. Selected fixtures also match isolated SymPy 1.14.0; these regressions
do not establish convergence, interchange, or complete calculus compatibility.
"""

import unittest

from sympy import Derivative, Dummy, Integral, Integer, Symbol, diff, symbols


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

    def test_an_outer_primitive_reintroduces_a_shadowed_coordinate(self):
        x, y = symbols("x y")
        cases = [
            (Integral(x, (x, 0, 1), x), Integral(x, (x, 0, 1))),
            (Integral(x*y, (x, 0, 1), x), Integral(x*y, (x, 0, 1))),
            (Integral(x, x, x), Integral(x, x)),
            (Integral(x*y, x, (x, 0, 1)), Integer(0)),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(diff(source, x), expected)
        self.assertEqual(diff(Integral(x, (x, 0, 1), x), x).doit(), Integer(1)/2)

    def test_definite_endpoints_are_outside_their_own_binding_scope(self):
        x, y = symbols("x y")
        self.assertEqual(diff(Integral(x*y, (x, 0, x)), x), x*y)
        self.assertEqual(diff(Integral(x**2, (x, x, x**2)), x), 2*x**5-x**2)
        self.assertEqual(diff(Integral(1, (x, x, 2*x)), x), Integer(1))
        self.assertEqual(diff(Integral(x*y, (x, 0, x)), y), Integral(x, (x, 0, x)))

    def test_nested_moving_limits_and_evaluation_points_compose(self):
        x, y, z = symbols("x y z")
        self.assertEqual(
            diff(Integral(x*y, (x, 0, y), (y, 0, z)), z),
            Integral(x*z, (x, 0, z)),
        )
        self.assertEqual(
            diff(Integral(x*y, (x, 0, z), (y, 0, z)), z),
            Integral(x*z, (x, 0, z)) + Integral(y*z, (y, 0, z)),
        )
        self.assertEqual(diff(Integral(x*y, (x, 0, y), (y, 0, z)), y), Integer(0))
        self.assertEqual(
            diff(Integral(x*y, (x, z), (y, 0, 1)), z),
            Integral(y*z, (y, 0, 1)),
        )

    def test_boundary_insertion_preserves_free_parameter_identity(self):
        x, y = symbols("x y")
        source = Integral(x*y, (x, 0, 1), (y, 0, x))
        result = diff(source, x)
        self.assertEqual(result.free_symbols, {x})
        self.assertEqual(len(result.bound_symbols), 1)
        self.assertNotEqual(result.bound_symbols[0], x)
        # The original integral evaluates to x**2/4. A captured boundary
        # would instead integrate x**2 over [0, 1], returning the false 1/3.
        self.assertEqual(result.doit(), x/2)

    def test_differentiation_keeps_distinct_same_printed_coordinates(self):
        coordinate = Symbol("x", positive=True)
        parameter = Symbol("x")
        source = Integral(coordinate*parameter, (coordinate, 0, parameter))
        self.assertEqual(diff(source, coordinate), Integer(0))
        self.assertEqual(
            diff(source, parameter),
            parameter**2 + Integral(coordinate, (coordinate, 0, parameter)),
        )

    def test_unsupported_indefinite_boundary_capture_remains_formal(self):
        x, y, z = symbols("x y z")
        for source in (
            Integral(x*y, x, (y, 0, x)),
            Integral(x*y, x, x, (y, 0, x)),
            Integral(x*y*z, x, (y, 0, x+z)),
        ):
            with self.subTest(source=source):
                self.assertEqual(diff(source, x), Derivative(source, x, evaluate=False))
        self.assertEqual(
            diff(Integral(x*y, x, (y, 0, z)), z),
            Integral(x*z, x),
        )


if __name__ == "__main__":
    unittest.main()
