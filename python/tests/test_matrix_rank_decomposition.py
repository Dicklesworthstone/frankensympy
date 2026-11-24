"""Public full-rank factorizations, including sparse sources and rank zero."""

import unittest

from sympy import Add, I, ImmutableMatrix, ImmutableSparseMatrix, Matrix, Mul, Pow, Rational, SparseMatrix


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)


class RankDecompositionTests(unittest.TestCase):
    def test_original_basis_coordinates_and_reconstruction(self):
        cases = [
            ([[0, 2, 4], [0, 3, 6]], (1,), [[0, 1, 2]]),
            ([[0, 0, 3], [2, 4, 0], [4, 8, 6]], (0, 2), [[1, 2, 0], [0, 0, 1]]),
            ([[Rational(1, 7), 1], [1, 7]], (0,), [[1, 7]]),
            ([[0, I, 1], [0, 2*I, 2]], (1,), [[0, 1, -I]]),
        ]
        for cls in CLASSES:
            for data, columns, coordinates in cases:
                with self.subTest(cls=cls, data=data):
                    source = cls(data)
                    left, right = source.rank_decomposition()
                    self.assertIs(type(left), cls)
                    self.assertIs(type(right), cls if data[0][1] == I else Matrix)
                    self.assertEqual(left, source.extract(range(source.rows), list(columns)))
                    self.assertEqual(right, Matrix(coordinates))
                    self.assertEqual(left*right, source)
                    if cls in (Matrix, SparseMatrix):
                        left[0, 0] = 19
                        self.assertEqual(source[0, columns[0]], data[0][columns[0]])

    def test_zero_rank_keeps_both_outer_dimensions(self):
        for cls in CLASSES:
            for shape in [(0, 0), (0, 3), (2, 0), (2, 3)]:
                with self.subTest(cls=cls, shape=shape):
                    source = cls.zeros(*shape)
                    left, right = source.rank_decomposition()
                    self.assertIs(type(left), cls)
                    self.assertIs(type(right), Matrix)
                    self.assertEqual(left.shape, (shape[0], 0))
                    self.assertEqual(right.shape, (0, shape[1]))
                    self.assertEqual(left*right, source)

    def test_sparse_low_rank_compact_factors(self):
        for cls in (SparseMatrix, ImmutableSparseMatrix):
            with self.subTest(cls=cls):
                source = cls(1024, 1024, {(1023, 3): 2, (1023, 1000): -3})
                left, right = source.rank_decomposition()
                self.assertEqual(left.shape, (1024, 1))
                self.assertEqual(left.todok(), {(1023, 0): 2})
                self.assertIs(type(right), Matrix)
                self.assertEqual(right.shape, (1, 1024))
                self.assertEqual(right[0, 3], 1)
                self.assertEqual(right[0, 1000], Rational(-3, 2))

    @unittest.skipUnless(hasattr(Matrix([[1]]), "_native_storage"), "native retained sources")
    def test_pivot_column_preserves_original_held_owner(self):
        held = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
        for cls in CLASSES:
            with self.subTest(cls=cls):
                left, _ = cls([[0, held]]).rank_decomposition()
                self.assertIs(left[0, 0], held)

    @unittest.skipUnless(hasattr(Matrix([[1]]), "_native_storage"), "native source domain")
    def test_zero_multiplier_does_not_hide_an_original_pole(self):
        pole = Pow(Add(I, -I, evaluate=False), -1, evaluate=False)
        matrix = Matrix([[Mul(0, pole, evaluate=False)]])
        with self.assertRaisesRegex(ValueError, "division by zero"):
            matrix.rank_decomposition()

    @unittest.skipUnless(hasattr(Matrix([[1]]), "_native_storage"), "native request contract")
    def test_custom_callback_is_not_silently_discarded(self):
        def forbidden(_):
            self.fail("unsupervised callback")

        for keyword in ("simplify", "iszerofunc"):
            with self.subTest(keyword=keyword), self.assertRaises(NotImplementedError):
                Matrix([[1]]).rank_decomposition(**{keyword: forbidden})


if __name__ == "__main__":
    unittest.main()
