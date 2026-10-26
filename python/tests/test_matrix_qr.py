"""Exact entry/shape fixtures and mathematical identities for public thin QR."""

import unittest

from sympy import Float, I, ImmutableMatrix, Matrix, Rational, Symbol, eye, simplify, sqrt, zeros


class ExactQRTests(unittest.TestCase):
    def assert_qr_identities(self, matrix, q, r):
        self.assertEqual((q * r - matrix).applyfunc(simplify), zeros(matrix.rows, matrix.cols))
        self.assertEqual((q.H * q - eye(q.cols)).applyfunc(simplify), zeros(q.cols, q.cols))

    def test_square_normalization_matches_exact_oracle_fixture(self):
        matrix = Matrix([[1, 1], [1, 0]])
        root = sqrt(2)
        q, r = matrix.QRdecomposition()
        self.assertEqual(q, Matrix([[root, root], [root, -root]]) / 2)
        self.assertEqual(r, Matrix([[root, root / 2], [0, root / 2]]))
        self.assert_qr_identities(matrix, q, r)
        self.assertEqual(matrix.qr(), (q, r))

    def test_tall_full_rank_matches_exact_oracle_fixture(self):
        matrix = Matrix([[1, 0], [1, 1], [0, 1]])
        q, r = matrix.QRdecomposition()
        self.assertEqual(q, Matrix([
            [sqrt(2) / 2, -sqrt(6) / 6],
            [sqrt(2) / 2, sqrt(6) / 6],
            [0, sqrt(6) / 3],
        ]))
        self.assertEqual(r, Matrix([[sqrt(2), sqrt(2) / 2], [0, sqrt(6) / 2]]))
        self.assert_qr_identities(matrix, q, r)

    def test_rank_deficient_tall_matrix_returns_thin_factors(self):
        matrix = Matrix([[1, 2], [2, 4], [3, 6]])
        q, r = matrix.QRdecomposition()
        self.assertEqual(q, Matrix([sqrt(14) / 14, sqrt(14) / 7, 3 * sqrt(14) / 14]))
        self.assertEqual(r, Matrix([[sqrt(14), 2 * sqrt(14)]]))
        self.assertEqual(q.shape, (3, 1))
        self.assertEqual(r.shape, (1, 2))
        self.assert_qr_identities(matrix, q, r)

    def test_wide_matrices_keep_zero_and_dependent_columns_in_r(self):
        for entries in (
            [[0, 1, 2], [0, 0, 1]],
            [[1, 2, 0, 1], [0, 0, 1, 1]],
        ):
            with self.subTest(entries=entries):
                matrix = Matrix(entries)
                q, r = matrix.QRdecomposition()
                self.assertEqual(q, eye(2))
                self.assertEqual(r, matrix)
                self.assertEqual(r.shape, matrix.shape)
                self.assert_qr_identities(matrix, q, r)

    def test_negative_columns_keep_the_positive_norm_phase(self):
        matrix = Matrix([[-1, -2], [-1, 1]])
        root = sqrt(2)
        q, r = matrix.QRdecomposition()
        self.assertEqual(q, Matrix([[-root, -root], [-root, root]]) / 2)
        self.assertEqual(r, Matrix([[root, root / 2], [0, 3 * root / 2]]))
        self.assert_qr_identities(matrix, q, r)

    def test_one_row_with_zero_leading_column(self):
        matrix = Matrix([[0, -3, 6]])
        q, r = matrix.QRdecomposition()
        self.assertEqual(q, Matrix([[-1]]))
        self.assertEqual(r, Matrix([[0, 3, -6]]))
        self.assert_qr_identities(matrix, q, r)

    def test_rational_entries_have_exact_radical_norms(self):
        matrix = Matrix([
            [Rational(1, 2), Rational(1, 3)],
            [Rational(1, 2), Rational(-1, 3)],
        ])
        q, r = matrix.QRdecomposition()
        root = sqrt(2)
        self.assertEqual(q, Matrix([[root, root], [root, -root]]) / 2)
        self.assertEqual(r, Matrix([[root / 2, 0], [0, root / 3]]))
        self.assert_qr_identities(matrix, q, r)

    def test_zero_rank_and_zero_area_shapes(self):
        for rows, cols in ((0, 0), (0, 3), (4, 0), (3, 2), (0, 1000), (1000, 0)):
            with self.subTest(shape=(rows, cols)):
                q, r = zeros(rows, cols).QRdecomposition()
                self.assertEqual(q.shape, (rows, 0))
                self.assertEqual(r.shape, (0, cols))
                self.assertEqual(q, zeros(rows, 0))
                self.assertEqual(r, zeros(0, cols))

    def test_immutable_class_and_input_are_preserved(self):
        matrix = ImmutableMatrix([[1, 2], [2, 4], [3, 6]])
        original = matrix.copy()
        q, r = matrix.QRdecomposition()
        self.assertIs(type(q), type(matrix))
        self.assertIs(type(r), type(matrix))
        self.assertEqual(matrix, original)
        self.assert_qr_identities(matrix, q, r)

    def test_native_qr_keeps_its_unnormalized_rational_contract(self):
        matrix = Matrix([[1, 1], [1, 0]])
        native_q, native_r = matrix._native.qr()
        q, r = Matrix(native_q), Matrix(native_r)
        self.assertEqual(q * r, matrix)
        self.assertEqual(q.T * q, Matrix([[2, 0], [0, Rational(1, 2)]]))
        self.assertEqual(r[0, 0], 1)
        self.assertEqual(r[1, 1], 1)

    def test_exhaustive_small_matrices_have_the_right_rank_and_identities(self):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    for d in (-1, 0, 1):
                        with self.subTest(entries=(a, b, c, d)):
                            matrix = Matrix([[a, b], [c, d]])
                            rank = 2 if a * d - b * c else int(any((a, b, c, d)))
                            q, r = matrix.QRdecomposition()
                            self.assertEqual(q.shape, (2, rank))
                            self.assertEqual(r.shape, (rank, 2))
                            self.assert_qr_identities(matrix, q, r)

    def test_unsupported_domains_remain_explicit(self):
        for entry in (Symbol("x"), Symbol("q", rational=True), I, Float("1.2")):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "exact rational entries only"):
                    Matrix([[entry]]).QRdecomposition()


if __name__ == "__main__":
    unittest.main()
