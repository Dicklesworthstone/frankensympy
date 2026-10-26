"""Public Python-to-native ODE regressions with exact symbolic comparisons."""

import unittest

from sympy import Symbol, cos, diff, exp, expand, simplify
from sympy.solvers.ode import (
    dsolve_cauchy_euler,
    dsolve_const_coeff_second_order,
    dsolve_const_coeff_second_order_nonhomogeneous,
)


class RadicalODETests(unittest.TestCase):
    def test_constant_coefficient_families_cross_the_native_boundary(self):
        x = Symbol("x")
        c1, c2 = Symbol("C1"), Symbol("C2")
        for a, b, c in ((1, 0, -2), (1, 0, 2), (2, -3, -1), (-2, 3, 1)):
            with self.subTest(coefficients=(a, b, c)):
                solution = dsolve_const_coeff_second_order(a, b, c, x)
                self.assertEqual(solution.free_symbols, {x, c1, c2})
                residual = a * diff(solution, x, 2) + b * diff(solution, x) + c * solution
                self.assertEqual(simplify(expand(residual)), 0)

    def test_irrational_families_support_exact_forcing(self):
        x = Symbol("x")
        for forcing in (1, exp(x), cos(x)):
            with self.subTest(forcing=forcing):
                solution = dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 2, forcing, x)
                residual = diff(solution, x, 2) + 2 * solution - forcing
                self.assertEqual(simplify(expand(residual)), 0)

    def test_euler_radicals_and_log_oscillations_on_positive_axis(self):
        x = Symbol("x")
        for c in (-2, 2):
            with self.subTest(coefficient=c):
                solution = dsolve_cauchy_euler(1, 1, c, x)
                residual = x**2 * diff(solution, x, 2) + x * diff(solution, x) + c * solution
                self.assertEqual(simplify(expand(residual)), 0)

    def test_generated_exponential_basis_is_independent(self):
        x = Symbol("x")
        solution = dsolve_const_coeff_second_order(1, 0, -2, x)
        first = diff(solution, Symbol("C1"))
        second = diff(solution, Symbol("C2"))
        determinant = (first * diff(second, x) - diff(first, x) * second).subs(x, 0)
        self.assertEqual(simplify(expand(determinant * determinant)), 8)


if __name__ == "__main__":
    unittest.main()
