"""Actual typed-boundary checks for exact polynomial ODE potentials."""

import unittest

from sympy import Dummy, Eq, Float, Rational, Symbol, dsolve_exact_first_order, expand, sin
from sympy.core import _native, _native_expr
from sympy.solvers import dsolve_exact_first_order as exported_solver
from sympy.solvers.exact_ode import dsolve_exact_first_order as module_solver


class ExactPolynomialODETests(unittest.TestCase):
    def assert_potential(self, relation, m, n, x, y, expected):
        self.assertIsInstance(relation, Eq)
        self.assertEqual(expand(relation.lhs - expected), 0)
        self.assertEqual(expand(relation.lhs.diff(x) - m), 0)
        self.assertEqual(expand(relation.lhs.diff(y) - n), 0)
        self.assertTrue(
            _native.verify_exact_ode_potential_native(
                _native_expr(relation.lhs),
                _native_expr(m),
                _native_expr(n),
                _native_expr(x),
                _native_expr(y),
            )
        )

    def test_coupled_potential_exports_and_independent_checker(self):
        self.assertIs(dsolve_exact_first_order, exported_solver)
        self.assertIs(dsolve_exact_first_order, module_solver)
        x, y = Symbol("x"), Symbol("y")
        m, n = 2*x + y, x + 2*y
        relation = dsolve_exact_first_order(m, n, x, y)
        self.assert_potential(relation, m, n, x, y, x**2 + x*y + y**2)
        self.assertEqual(relation.rhs, Symbol("C1"))
        for mutation in (x, y):
            with self.subTest(mutation=mutation):
                self.assertFalse(
                    _native.verify_exact_ode_potential_native(
                        _native_expr(relation.lhs + mutation),
                        _native_expr(m),
                        _native_expr(n),
                        _native_expr(x),
                        _native_expr(y),
                    )
                )

    def test_rational_coupled_coefficients_and_supported_degree_boundary(self):
        x, y = Symbol("x"), Symbol("y")
        expected = Rational(1, 3)*x**3*y**2 - Rational(2, 5)*x*y + Rational(7, 4)*y**4
        m = x**2*y**2 - Rational(2, 5)*y
        n = Rational(2, 3)*x**3*y - Rational(2, 5)*x + 7*y**3
        self.assert_potential(dsolve_exact_first_order(m, n, x, y), m, n, x, y, expected)
        self.assert_potential(
            dsolve_exact_first_order(x**16, 0, x, y), x**16, 0, x, y, x**17/17
        )

    def test_vertical_and_horizontal_level_sets_remain_implicit(self):
        x, y = Symbol("x"), Symbol("y")
        for m, n, potential in ((2*x, 0, x**2), (0, 2*y, y**2), (1, 0, x)):
            with self.subTest(coefficients=(m, n)):
                relation = dsolve_exact_first_order(m, n, x, y)
                self.assert_potential(relation, m, n, x, y, potential)
                self.assertEqual(relation.rhs, Symbol("C1"))

    def test_integration_constant_avoids_even_unused_coordinate_names(self):
        for x in (Symbol("C1"), Dummy("C1")):
            with self.subTest(coordinate=x):
                y = Symbol("C2")
                relation = dsolve_exact_first_order(0, 1, x, y)
                self.assertEqual(relation.lhs, y)
                self.assertEqual(relation.rhs, Symbol("C3"))
                self.assertEqual(relation.free_symbols, {y, Symbol("C3")})

    def test_typed_coordinates_preserve_assumptions_and_dummy_identity(self):
        x, y = Symbol("t", positive=True), Dummy("t")
        m, n = 2*x + y, x + 2*y
        relation = dsolve_exact_first_order(m, n, x, y)
        self.assert_potential(relation, m, n, x, y, x**2 + x*y + y**2)
        self.assertEqual(relation.lhs.free_symbols, {x, y})
        self.assertNotIn(Symbol("t"), relation.lhs.free_symbols)
        self.assertNotIn(Dummy("t"), relation.lhs.free_symbols)
        self.assertFalse(
            _native.verify_exact_ode_potential_native(
                _native_expr(relation.lhs),
                _native_expr(m),
                _native_expr(n),
                _native_expr(Symbol("t")),
                _native_expr(y),
            )
        )

    def test_unsupported_and_degenerate_inputs_fail_explicitly(self):
        x, y, a = Symbol("x"), Symbol("y"), Symbol("a")
        for m, n in (
            (y, 0),
            (sin(x), 0),
            (1/x, 0),
            (Float(1.0), 0),
            (a, 0),
            (x**17, 0),
            (0, 0),
        ):
            with self.subTest(coefficients=(m, n)):
                with self.assertRaises(ValueError):
                    dsolve_exact_first_order(m, n, x, y)
        with self.assertRaises(ValueError):
            dsolve_exact_first_order(1, 0, x, x)
        with self.assertRaises(TypeError):
            dsolve_exact_first_order(1, 0, "x", y)
        with self.assertRaises(TypeError):
            _native.exact_ode_potential_native(
                _native_expr(1), _native_expr(0), "x", _native_expr(y)
            )
        with self.assertRaises(ValueError):
            _native.exact_ode_potential_native(
                _native_expr(1), _native_expr(0), _native_expr(1), _native_expr(y)
            )


if __name__ == "__main__":
    unittest.main()
