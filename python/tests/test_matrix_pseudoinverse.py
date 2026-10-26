"""Exact entry/shape/class regressions for native Moore-Penrose pseudoinverses."""

import unittest

from sympy import ImmutableMatrix, Matrix, Rational, Symbol, eye, zeros
from sympy.matrices import pinv


class ExactPseudoinverseTests(unittest.TestCase):
    def assert_penrose_identities(self, matrix, inverse):
        self.assertEqual(matrix * inverse * matrix, matrix)
        self.assertEqual(inverse * matrix * inverse, inverse)
        self.assertEqual((matrix * inverse).T, matrix * inverse)
        self.assertEqual((inverse * matrix).T, inverse * matrix)

    def test_singular_square_exact_entries(self):
        matrix = Matrix([[1, 2], [2, 4]])
        inverse = matrix.pinv()
        self.assertEqual(inverse, matrix / 25)
        self.assert_penrose_identities(matrix, inverse)
        self.assertEqual(pinv(matrix), inverse)

    def test_rank_deficient_rectangular_exact_entries(self):
        matrix = Matrix([[2, -1], [4, -2], [6, -3]])
        expected = Matrix([[2, 4, 6], [-1, -2, -3]]) / 70
        self.assertEqual(matrix.pinv(), expected)
        self.assertEqual(matrix.T.pinv(), expected.T)
        self.assert_penrose_identities(matrix, expected)

    def test_rank_two_singular_exact_oracle_fixture(self):
        matrix = Matrix([[1, 2, 3], [2, 4, 6], [0, 1, 1]])
        expected = Matrix([[2, 4, -25], [-1, -2, 20], [1, 2, -5]]) / 15
        self.assertEqual(matrix.pinv(), expected)
        self.assert_penrose_identities(matrix, expected)

    def test_full_rank_rectangular_behavior(self):
        matrix = Matrix([[1, 2], [3, 4], [5, 6]])
        expected = Matrix([[-16, -4, 8], [13, 4, -5]]) / 12
        self.assertEqual(matrix.pinv(), expected)
        self.assertEqual(matrix.T.pinv(), expected.T)

    def test_zero_matrices_keep_transposed_shape(self):
        for rows, cols in ((0, 0), (0, 3), (4, 0), (2, 3)):
            with self.subTest(shape=(rows, cols)):
                self.assertEqual(zeros(rows, cols).pinv(), zeros(cols, rows))
                self.assertEqual(zeros(rows, cols).pinv().shape, (cols, rows))

    def test_immutable_result_and_input_preservation(self):
        matrix = ImmutableMatrix([[1, 2], [2, 4]])
        inverse = matrix.pinv()
        self.assertIs(type(inverse), type(matrix))
        self.assertEqual(matrix, ImmutableMatrix([[1, 2], [2, 4]]))
        self.assertEqual(inverse, matrix / 25)

    def test_involution_and_rational_scaling(self):
        matrix = Matrix([[1, 2], [2, 4], [3, 6]])
        inverse = matrix.pinv()
        self.assertEqual(inverse.pinv(), matrix)
        scale = Rational(-3, 7)
        self.assertEqual((scale * matrix).pinv(), inverse / scale)

    def test_minimum_norm_least_squares_solution(self):
        matrix = Matrix([[1, 2], [2, 4]])
        rhs = Matrix([3, 0])
        solution = matrix.pinv() * rhs
        self.assertEqual(solution, Matrix([3, 6]) / 25)
        self.assertEqual(matrix.T * (matrix * solution - rhs), zeros(2, 1))
        self.assertEqual(Matrix([[-2, 1]]) * solution, zeros(1, 1))

    def test_nonsingular_square_matches_inverse(self):
        matrix = Matrix([[2, 1], [5, 7]])
        self.assertEqual(matrix.pinv(), matrix.inv())
        self.assertEqual(matrix * matrix.pinv(), eye(2))

    def test_unsupported_domain_stays_visible(self):
        # The native rational lane must not guess whether x specializes to 0.
        with self.assertRaisesRegex(ValueError, "exact rational entries only"):
            Matrix([[Symbol("x")]]).pinv()


if __name__ == "__main__":
    unittest.main()
