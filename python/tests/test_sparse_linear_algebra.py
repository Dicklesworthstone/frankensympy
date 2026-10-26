"""Exact public sparse operations, with explicit no-densification regressions."""

import unittest
from unittest.mock import patch

from sympy import Matrix, Rational, SparseMatrix, Symbol


class SparseLinearAlgebraTests(unittest.TestCase):
    def test_large_diagonal_determinant_solve_and_inverse_stay_sparse(self):
        n = 1024
        matrix = SparseMatrix(n, n, {(i, i): 2 for i in range(n)})
        rhs = SparseMatrix(n, 1, {(0, 0): 6, (n - 1, 0): -4})
        with patch.object(SparseMatrix, "to_dense", side_effect=AssertionError("densified")):
            self.assertEqual(matrix.det(), 2**n)
            solution = matrix.solve(rhs)
            self.assertIs(type(solution), SparseMatrix)
            self.assertEqual(solution.shape, (n, 1))
            self.assertEqual(solution.todok(), {(0, 0): 3, (n - 1, 0): -2})
            inverse = matrix.inv()
            self.assertEqual(inverse.nnz, n)
            self.assertEqual(inverse.todok(), {(i, i): Rational(1, 2) for i in range(n)})
        self.assertEqual(matrix.nnz, n)
        self.assertEqual(rhs.nnz, 2)

    def test_large_tridiagonal_boundary_value_system(self):
        n = 600
        entries = {(i, i): 2 for i in range(n)}
        entries.update({(i, i + 1): -1 for i in range(n - 1)})
        entries.update({(i + 1, i): -1 for i in range(n - 1)})
        matrix = SparseMatrix(n, n, entries)
        rhs = SparseMatrix(n, 1, {(0, 0): 1, (n - 1, 0): 1})
        with patch.object(SparseMatrix, "to_dense", side_effect=AssertionError("densified")):
            self.assertEqual(matrix.det(), n + 1)
            self.assertEqual(matrix.LUsolve(rhs).todok(), {(i, 0): 1 for i in range(n)})

    def test_row_pivoting_and_multiple_right_hand_sides(self):
        matrix = SparseMatrix([[0, 2, 0], [1, 3, 4], [0, 0, 5]])
        rhs = Matrix([[4, -2], [19, 13], [15, 20]])
        expected = SparseMatrix([[1, 0], [2, -1], [3, 4]])
        with patch.object(SparseMatrix, "to_dense", side_effect=AssertionError("densified")):
            self.assertEqual(matrix.det(), -10)
            self.assertEqual(matrix.solve(rhs), expected)
            self.assertEqual(matrix.solve(rhs.to_sparse()), expected)

    def test_singular_and_rectangular_systems_refuse(self):
        singular = SparseMatrix([[1, 2], [2, 4]])
        self.assertEqual(singular.det(), 0)
        for action in (
            lambda: singular.solve(SparseMatrix([1, 2])),
            singular.inv,
            lambda: SparseMatrix.zeros(2, 3).det(),
            lambda: SparseMatrix.eye(3).solve(SparseMatrix.zeros(2, 1)),
        ):
            with self.assertRaises(ValueError):
                action()

    def test_zero_area_shapes_and_zero_right_hand_sides(self):
        empty = SparseMatrix.zeros(0, 0)
        self.assertEqual(empty.det(), 1)
        self.assertEqual(empty.inv().shape, (0, 0))
        self.assertEqual(empty.solve(SparseMatrix.zeros(0, 3)).shape, (0, 3))
        self.assertEqual(SparseMatrix.eye(3).solve(SparseMatrix.zeros(3, 0)).shape, (3, 0))
        self.assertEqual(SparseMatrix.eye(3).solve(SparseMatrix.zeros(3, 2)).nnz, 0)

    def test_coordinate_export_visits_only_stored_entries(self):
        matrix = SparseMatrix(1_000_000, 1_000_000, {(9, 999_999): Rational(2, 3)})
        with patch.object(SparseMatrix, "__getitem__", side_effect=AssertionError("dense scan")):
            self.assertEqual(matrix.todok(), {(9, 999_999): Rational(2, 3)})

    def test_symbolic_determinant_retains_the_existing_expression_lane(self):
        x = Symbol("x")
        self.assertEqual(SparseMatrix([[x, 0], [0, 2]]).det(), 2 * x)

    def test_native_entry_domain_and_results_are_typed(self):
        rational = SparseMatrix([[Rational(1, 3), 0], [0, 2]])
        symbolic = SparseMatrix([[Symbol("x")]])
        self.assertTrue(rational._native.is_exact_rational)
        self.assertFalse(symbolic._native.is_exact_rational)
        result = SparseMatrix(rational._native.solve(SparseMatrix([1, 4])._native))
        self.assertEqual(result, SparseMatrix([3, 2]))


if __name__ == "__main__":
    unittest.main()
