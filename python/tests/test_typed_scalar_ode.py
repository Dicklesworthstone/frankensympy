"""Scalar ODE helpers must preserve the actual typed mathematical inputs."""

import unittest
from unittest.mock import patch

from sympy import Dummy, Function, Mul, Pow, Rational, Symbol, cos, exp, expand, simplify
from sympy.core import _native, _native_expr
from sympy.solvers.ode import (
    dsolve_cauchy_euler,
    dsolve_const_coeff_second_order,
    dsolve_const_coeff_second_order_nonhomogeneous,
    dsolve_linear_first_order,
    dsolve_separable_linear,
    verify_cauchy_euler_solution,
    verify_const_coeff_second_order_solution,
    verify_linear_first_order_solution,
)


class TypedScalarODETests(unittest.TestCase):
    def assert_zero(self, expression):
        self.assertEqual(simplify(expand(expression)), 0)

    def test_first_order_distinguishes_same_print_coordinate_and_parameter(self):
        x = Symbol("typed_ode_x", positive=True)
        parameter = Symbol("typed_ode_x", negative=True)
        c1 = Symbol("C1")
        result = dsolve_linear_first_order(0, parameter, x)
        self.assert_zero(result - parameter*x - c1)
        self.assertEqual(result.free_symbols, {x, parameter, c1})
        self.assertTrue(any(symbol is x for symbol in result.free_symbols))
        self.assertTrue(any(symbol is parameter for symbol in result.free_symbols))
        self.assertTrue(verify_linear_first_order_solution(result, 0, parameter, x))
        # A same-printed replacement changes the equation, not just metadata.
        self.assertFalse(verify_linear_first_order_solution(result, 0, x, x))
        self.assertFalse(verify_linear_first_order_solution(x*x + c1, 0, parameter, x))

    def test_dummy_and_assumed_coordinates_survive_every_generator(self):
        for x in (Dummy("t"), Symbol("typed_ode_t", positive=True)):
            with self.subTest(x=x):
                first = dsolve_linear_first_order(0, 2*x, x)
                self.assert_zero(first - x*x - Symbol("C1"))
                separable = dsolve_separable_linear(1, x)
                self.assert_zero(separable - Symbol("C1")*exp(x))
                homogeneous = dsolve_const_coeff_second_order(1, 0, 1, x)
                self.assert_zero(homogeneous.diff(x, 2) + homogeneous)
                forced = dsolve_const_coeff_second_order_nonhomogeneous(1, 0, -1, 2, x)
                self.assert_zero(forced.diff(x, 2) - forced - 2)
                euler = dsolve_cauchy_euler(1, 0, 0, x)
                self.assert_zero(euler - Symbol("C1")*x - Symbol("C2"))
                for result in (first, separable, homogeneous, forced, euler):
                    self.assertIn(x, result.free_symbols)
                    self.assertTrue(any(symbol is x for symbol in result.free_symbols))
                self.assertTrue(verify_const_coeff_second_order_solution(homogeneous, 1, 0, 1, x))
                self.assertTrue(verify_cauchy_euler_solution(euler, 1, 0, 0, x))

    def test_generated_constants_avoid_coordinates_and_forcing_parameters(self):
        x, parameter = Symbol("C1"), Symbol("C2", positive=True)
        result = dsolve_linear_first_order(0, parameter, x)
        self.assert_zero(result - parameter*x - Symbol("C3"))
        self.assertEqual(result.free_symbols, {x, parameter, Symbol("C3")})
        # The scalar second-order engine still admits rational forcing
        # amplitudes only; the typed bridge must retain that explicit refusal.
        with self.assertRaises(ValueError):
            dsolve_const_coeff_second_order_nonhomogeneous(1, 0, -1, parameter, x)
        forced = dsolve_const_coeff_second_order_nonhomogeneous(1, 0, -1, 2, x)
        self.assert_zero(forced.diff(x, 2) - forced - 2)
        self.assertEqual(forced.free_symbols, {x, Symbol("C2"), Symbol("C3")})
        for generator in (dsolve_const_coeff_second_order, dsolve_cauchy_euler):
            result = generator(1, 0, 0, x)
            self.assertEqual(result.free_symbols, {x, Symbol("C2"), Symbol("C3")})

    def test_exact_checker_cannot_accept_a_same_print_variable_rebinding(self):
        x = Symbol("typed_checker_x", positive=True)
        other = Symbol("typed_checker_x", negative=True)
        solution = exp(x)
        self.assertTrue(verify_linear_first_order_solution(solution, -1, 0, x))
        self.assertFalse(verify_linear_first_order_solution(solution, -1, 0, other))
        self.assertTrue(verify_const_coeff_second_order_solution(solution, 1, 0, -1, x))
        self.assertFalse(verify_const_coeff_second_order_solution(solution, 1, 0, -1, other))
        self.assertTrue(verify_cauchy_euler_solution(x, 1, 0, 0, x))
        self.assertFalse(verify_cauchy_euler_solution(x*x, 1, 0, 0, x))
        self.assertFalse(verify_linear_first_order_solution(
            solution + Rational(1, 10**60)*x, -1, 0, x,
        ))

    def test_string_algorithm_routes_are_not_used(self):
        x = Symbol("typed_no_print", positive=True)
        with patch.object(_native, "dsolve_linear_first_order_expr",
                          side_effect=AssertionError("printed solver input")), patch.object(
            _native, "verify_linear_first_order_solution_expr",
            side_effect=AssertionError("printed verifier input"),
        ):
            solution = dsolve_linear_first_order(0, 1, x)
            self.assertTrue(verify_linear_first_order_solution(solution, 0, 1, x))

    def test_native_boundary_rejects_aliases_and_non_symbol_coordinates(self):
        x, c1 = Symbol("typed_native_x"), Symbol("C1")
        values = (_native_expr(0), _native_expr(1))
        with self.assertRaises(ValueError):
            _native.dsolve_linear_first_order_native(*values, _native_expr(1), _native_expr(c1))
        with self.assertRaises(ValueError):
            _native.dsolve_linear_first_order_native(*values, _native_expr(x), _native_expr(x))
        with self.assertRaises(ValueError):
            _native.dsolve_linear_first_order_native(
                _native_expr(0), _native_expr(c1), _native_expr(x), _native_expr(c1),
            )
        with self.assertRaises(TypeError):
            dsolve_linear_first_order(0, 1, "typed_native_x")
        with self.assertRaises(NotImplementedError):
            dsolve_linear_first_order(0, Function("opaque_ode")(x), x)

    def test_original_undefined_terms_are_not_erased_by_the_boundary(self):
        x = Symbol("typed_ode_domain")
        hole = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
        with self.assertRaises(ValueError):
            dsolve_linear_first_order(0, hole, x)
        self.assertFalse(verify_linear_first_order_solution(hole, 0, 0, x))

    def test_legacy_plain_symbol_results_and_domain_refusals_remain(self):
        x = Symbol("plain_ode_x")
        self.assert_zero(dsolve_linear_first_order(0, 2*x, x) - x*x - Symbol("C1"))
        solution = dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 2, cos(x), x)
        self.assert_zero(solution.diff(x, 2) + 2*solution - cos(x))
        # A vanished leading coefficient is not a second-order equation.
        with self.assertRaises(ValueError):
            dsolve_const_coeff_second_order(0, 1, 1, x)


if __name__ == "__main__":
    unittest.main()
