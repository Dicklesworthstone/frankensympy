"""Complete exact solution families through the independent native checker."""

import unittest
from unittest.mock import patch

from sympy import (
    Dummy, EmptySet, ImmutableMatrix, Matrix, Rational, SparseMatrix, Symbol,
    linsolve, solve_linear_system, symbols, zeros,
)


class CheckedLinearSystemTests(unittest.TestCase):
    def solution(self, system, *variables):
        return tuple(next(iter(linsolve(system, *variables))))

    def test_rectangular_unique_solution_uses_checked_native_lane(self):
        x, y = symbols("x y")
        a = Matrix([[1, 2], [3, 4], [-1, 1]])
        b = Matrix([5, 11, 1])
        with patch.object(Matrix, "rref", side_effect=AssertionError("legacy RREF used")):
            self.assertEqual(self.solution((a, b), x, y), (1, 2))
            self.assertEqual(solve_linear_system(a.row_join(b), x, y), {x: 1, y: 2})

    def test_singular_family_keeps_all_free_variables(self):
        x, y, z = symbols("x y z")
        a = Matrix([[1, 2, 3], [2, 4, 6]])
        b = Matrix([1, 2])
        self.assertEqual(self.solution((a, b), x, y, z), (1 - 2*y - 3*z, y, z))
        for second, third in ((0, 0), (2, -1), (Rational(1, 3), Rational(-2, 5))):
            vector = Matrix([1 - 2*second - 3*third, second, third])
            self.assertEqual(a * vector, b)

    def test_inconsistency_is_not_a_domain_or_resource_refusal(self):
        x, y = symbols("x y")
        matrix = Matrix([[1, 2, 1], [2, 4, 3]])
        self.assertEqual(linsolve(matrix, x, y), EmptySet())
        self.assertIsNone(solve_linear_system(matrix, x, y))
        with self.assertRaisesRegex(ValueError, "128"):
            linsolve(zeros(129, 2), x)

    def test_exact_tiny_perturbation_is_full_rank(self):
        x, y = symbols("x y")
        epsilon = Rational(1, 10**80)
        a = Matrix([[1, 1], [1, 1 + epsilon]])
        b = Matrix([3, 3 + 2*epsilon])
        self.assertEqual(self.solution((a, b), x, y), (1, 2))

    def test_large_integer_and_rational_coefficients_stay_exact(self):
        x, y = symbols("x y")
        large = 10**100 + 137
        a = Matrix([[large, Rational(1, 3)], [0, Rational(2, 7)]])
        b = Matrix([large*5 + Rational(2, 3), Rational(4, 7)])
        self.assertEqual(self.solution((a, b), x, y), (5, 2))

    def test_free_variable_identity_and_printer_independence(self):
        first, second = Dummy("same"), Dummy("same")
        with patch.object(Symbol, "__str__", side_effect=AssertionError("symbol printer used")):
            solution = self.solution(Matrix([[1, 1, 2]]), first, second)
        self.assertEqual(solution, (2 - second, second))
        self.assertIs(solution[1], second)
        assumed = Symbol("same", positive=True)
        plain = Symbol("same")
        self.assertEqual(self.solution(zeros(1, 3), assumed, plain), (assumed, plain))

    def test_empty_equation_block_preserves_unconstrained_variables(self):
        x, y = symbols("x y")
        self.assertEqual(self.solution((zeros(0, 2), zeros(0, 1)), x, y), (x, y))
        self.assertEqual(linsolve([], x, y), EmptySet())
        self.assertEqual(self.solution(Matrix([[0, 1, 0, 2]])),
                         (Symbol("tau0"), 2, Symbol("tau1")))

    def test_existing_symbolic_coefficient_lane_remains_available(self):
        x, y, parameter = symbols("x y parameter")
        self.assertEqual(self.solution(Matrix([[1, 1, parameter]]), x, y),
                         (parameter - y, y))

    def test_free_parameters_cannot_be_replaced_with_numeric_constants(self):
        x = Symbol("x")
        with self.assertRaises(TypeError):
            linsolve(zeros(1, 3), 1, x)

    def test_native_boundary_distinguishes_inconsistent_and_affine(self):
        a = Matrix([[1, 2], [2, 4]])
        self.assertIsNone(a._native.solve_affine(Matrix([1, 3])._native))
        particular, basis, free = a._native.solve_affine(Matrix([1, 2])._native)
        from sympy.core import _wrap
        self.assertEqual([_wrap(value) for value in particular], [1, 0])
        self.assertEqual([[_wrap(value) for value in row] for row in basis], [[-2, 1]])
        self.assertEqual(free, [1])


class GaussJordanSolveTests(unittest.TestCase):
    def test_multiple_rhs_and_parameter_layout(self):
        a = Matrix([[1, 2, 3], [2, 4, 6]])
        b = Matrix([[1, 2], [2, 4]])
        original_a, original_b = a.copy(), b.copy()
        solution, parameters, free = a.gauss_jordan_solve(b, freevar=True)
        tau0, tau1, tau2, tau3 = symbols("tau0 tau1 tau2 tau3")
        self.assertEqual(parameters, Matrix([[tau0, tau1], [tau2, tau3]]))
        self.assertEqual(solution, Matrix([[1 - 2*tau0 - 3*tau2, 2 - 2*tau1 - 3*tau3],
                                          [tau0, tau1], [tau2, tau3]]))
        self.assertEqual(free, [1, 2])
        self.assertEqual(a, original_a)
        self.assertEqual(b, original_b)
        for assignments in ({tau0: 0, tau1: 1, tau2: -2, tau3: 3},
                            {tau0: 4, tau1: -5, tau2: 6, tau3: 7}):
            self.assertEqual(a * solution.subs(assignments), b)

    def test_rectangular_unique_and_immutable_results(self):
        a = ImmutableMatrix([[1, 2], [3, 4], [-1, 1]])
        solution, parameters = a.gauss_jordan_solve(ImmutableMatrix([5, 11, 1]))
        self.assertIs(type(solution), type(a))
        self.assertIs(type(parameters), type(a))
        self.assertEqual(solution, ImmutableMatrix([1, 2]))
        self.assertEqual(parameters.shape, (0, 1))

    def test_inconsistent_rhs_column_rejects_the_whole_result(self):
        a = Matrix([[1, 2], [2, 4]])
        with self.assertRaisesRegex(ValueError, "no solution"):
            a.gauss_jordan_solve(Matrix([[1, 1], [2, 3]]))

    def test_zero_dimensions_and_zero_rank(self):
        solution, parameters, free = zeros(0, 2).gauss_jordan_solve(zeros(0, 1), True)
        self.assertEqual(solution, Matrix([Symbol("tau0"), Symbol("tau1")]))
        self.assertEqual(parameters, solution)
        self.assertEqual(free, [0, 1])
        solution, parameters = zeros(3, 0).gauss_jordan_solve(zeros(3, 2))
        self.assertEqual(solution.shape, (0, 2))
        self.assertEqual(parameters.shape, (0, 2))
        with self.assertRaisesRegex(ValueError, "no solution"):
            zeros(2, 0).gauss_jordan_solve(Matrix([0, 1]))

    def test_sparse_rhs_and_exact_rational_entries(self):
        a = Matrix([[2, 0], [0, 3]])
        solution, parameters = a.gauss_jordan_solve(SparseMatrix([[1], [1]]))
        self.assertEqual(solution, Matrix([Rational(1, 2), Rational(1, 3)]))
        self.assertEqual(parameters.shape, (0, 1))

    def test_domain_shape_and_bounds_refuse_explicitly(self):
        with self.assertRaisesRegex(ValueError, "exact rational"):
            Matrix([[Symbol("a")]]).gauss_jordan_solve(Matrix([1]))
        with self.assertRaisesRegex(ValueError, "rows"):
            Matrix([[1]]).gauss_jordan_solve(Matrix([1, 2]))
        with self.assertRaisesRegex(ValueError, "128"):
            zeros(1, 129).gauss_jordan_solve(zeros(1, 1))
        with self.assertRaisesRegex(ValueError, "128"):
            zeros(1, 1).gauss_jordan_solve(zeros(1, 129))
        with self.assertRaisesRegex(ValueError, "aggregate"):
            zeros(128, 128).gauss_jordan_solve(zeros(128, 3))
        with self.assertRaises(TypeError):
            Matrix([[1]]).gauss_jordan_solve([1])


if __name__ == "__main__":
    unittest.main()
