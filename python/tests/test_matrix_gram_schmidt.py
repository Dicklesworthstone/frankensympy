"""Hermitian Gram-Schmidt fixtures, shape semantics, and original domains."""

import unittest

from sympy import (
    Float, GramSchmidt, I, ImmutableMatrix, ImmutableSparseMatrix, Matrix, Mul,
    Pow, Rational, S, SparseMatrix, Symbol, eye, simplify, sqrt, zeros,
)
from sympy.matrices.exceptions import ShapeError


class GramSchmidtTests(unittest.TestCase):
    def assert_orthogonal(self, vectors, *, normalized=False):
        for i, left in enumerate(vectors):
            for j, right in enumerate(vectors):
                value = simplify(sum(left[k].conjugate() * right[k]
                                     for k in range(len(left))))
                if i != j:
                    self.assertEqual(value, 0)
                elif normalized:
                    self.assertEqual(value, 1)

    def test_complex_isotropic_for_transpose_is_not_zero_for_hermitian(self):
        vectors = [Matrix([1, I]), Matrix([1, -I])]
        result = GramSchmidt(vectors)
        self.assertEqual(result, vectors)
        self.assert_orthogonal(result)

    def test_complex_projection_conjugates_the_first_operand(self):
        result = GramSchmidt([Matrix([1, I]), Matrix([1, 0])])
        self.assertEqual(result, [Matrix([1, I]), Matrix([Rational(1, 2), -I/2])])
        self.assert_orthogonal(result)

    def test_three_complex_directions_match_exact_oracle_fixture(self):
        source = [Matrix([1, I, 0]), Matrix([1, 0, I]), Matrix([0, 1, 1])]
        result = GramSchmidt(source)
        self.assertEqual(result, [
            Matrix([1, I, 0]),
            Matrix([Rational(1, 2), -I/2, I]),
            Matrix([2*I/3, Rational(2, 3), Rational(2, 3)]),
        ])
        self.assert_orthogonal(result)

    def test_normalized_complex_directions_match_exact_oracle_fixture(self):
        source = [Matrix([1, I, 0]), Matrix([1, 0, I]), Matrix([0, 1, 1])]
        result = GramSchmidt(source, orthonormal=True)
        expected = [
            Matrix([sqrt(2)/2, I*sqrt(2)/2, 0]),
            Matrix([sqrt(6)/6, -I*sqrt(6)/6, I*sqrt(6)/3]),
            Matrix([I*sqrt(3)/3, sqrt(3)/3, sqrt(3)/3]),
        ]
        for actual, wanted in zip(result, expected):
            self.assertEqual((actual-wanted).applyfunc(simplify), zeros(3, 1))
        self.assert_orthogonal(result, normalized=True)

    def test_real_projection_and_positive_norm_phase(self):
        source = [Matrix([-1, -1]), Matrix([1, 0])]
        expected = [Matrix([-1, -1]), Matrix([Rational(1, 2), Rational(-1, 2)])]
        self.assertEqual(GramSchmidt(source), expected)
        result = GramSchmidt(source, orthonormal=True)
        self.assertEqual((result[0] - Matrix([-sqrt(2)/2, -sqrt(2)/2])).applyfunc(simplify), zeros(2, 1))
        self.assert_orthogonal(result, normalized=True)

    def test_rational_inputs_keep_exact_entries_until_normalization(self):
        source = [Matrix([Rational(1, 2), Rational(1, 2)]),
                  Matrix([Rational(1, 3), Rational(-1, 3)])]
        self.assertEqual(GramSchmidt(source), source)
        self.assert_orthogonal(GramSchmidt(source, orthonormal=True), normalized=True)

    def test_row_vectors_and_mixed_orientation_follow_first_input(self):
        column = [Matrix([1, I]), Matrix([1, -I])]
        for source in (
            [column[0].T, column[1].T],
            [column[0].T, column[1]],
            [column[0], column[1].T],
        ):
            with self.subTest(shapes=[v.shape for v in source]):
                row_output = source[0].rows == 1
                expected = [v.T if row_output else v for v in column]
                result = GramSchmidt(source)
                self.assertEqual(result, expected)
                self.assert_orthogonal(result)

    def test_general_matrices_flatten_in_column_major_order(self):
        result = GramSchmidt([Matrix([[1, 2], [3, 4]]), Matrix([[0, 1], [0, 0]])])
        self.assertEqual(result, [
            Matrix([1, 3, 2, 4]),
            Matrix([Rational(-1, 15), Rational(-1, 5), Rational(13, 15), Rational(-4, 15)]),
        ])
        self.assert_orthogonal(result)

    def test_mismatched_vector_lengths_raise_shape_error(self):
        with self.assertRaises(ShapeError):
            GramSchmidt([Matrix([1, 2]), Matrix([1, 2, 3])])

    def test_zero_and_dependent_vectors_raise_in_both_modes(self):
        cases = [
            [zeros(2, 1)],
            [Matrix([1, 2]), zeros(2, 1)],
            [Matrix([1, 2]), Matrix([2, 4])],
            [Matrix([1, I]), Matrix([I, -1])],
            [Matrix([1, I, 0]), Matrix([1, 0, I]), Matrix([2, I, I])],
            [zeros(0, 1)],
        ]
        for source in cases:
            for normalized in (False, True):
                with self.subTest(source=source, normalized=normalized):
                    with self.assertRaisesRegex(ValueError, "not linearly independent"):
                        GramSchmidt(source, orthonormal=normalized)

    def test_provably_zero_symbol_is_dependent(self):
        zero = Symbol("z", zero=True)
        with self.assertRaisesRegex(ValueError, "not linearly independent"):
            GramSchmidt([Matrix([zero, 0])])

    def test_empty_input_is_empty(self):
        self.assertEqual(GramSchmidt([]), [])
        self.assertEqual(GramSchmidt(iter(()), orthonormal=True), [])

    def test_sparse_and_immutable_inputs_produce_mutable_dense_results(self):
        for cls in (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix):
            with self.subTest(cls=cls):
                source = [cls([1, I]), cls([1, -I])]
                saved = [Matrix(v) for v in source]
                result = GramSchmidt(source, orthonormal=True)
                self.assertTrue(all(type(v) is Matrix for v in result))
                self.assertEqual([Matrix(v) for v in source], saved)
                self.assert_orthogonal(result, normalized=True)
                result[0][0, 0] = 7
                self.assertEqual([Matrix(v) for v in source], saved)

    def test_list_input_extension_and_generator_are_retained(self):
        self.assertEqual(GramSchmidt(([1, 0], [1, 1])), [Matrix([1, 0]), Matrix([0, 1])])
        source = (v for v in (Matrix([1, 0]), Matrix([1, 1])))
        self.assertEqual(GramSchmidt(source), [Matrix([1, 0]), Matrix([0, 1])])

    def test_unknown_symbolic_norm_remains_formal(self):
        x = Symbol("x")
        source = [Matrix([x, 0]), Matrix([0, 1])]
        self.assertEqual(GramSchmidt(source), source)
        self.assertIsNone(x.is_zero)

    def test_symbolic_complex_projection_preserves_conjugation(self):
        x = Symbol("x")
        result = GramSchmidt([Matrix([1, x]), Matrix([1, 0])])
        specialized = [v.subs(x, I).applyfunc(simplify) for v in result]
        self.assertEqual(specialized, [Matrix([1, I]), Matrix([Rational(1, 2), -I/2])])

    def test_undecided_symbolic_source_domain_is_preserved_formally(self):
        x = Symbol("x")
        vector = Matrix([1/x, 0])
        self.assertEqual(GramSchmidt([vector]), [vector])

    def test_float_inputs_retain_approximate_orthogonalization(self):
        result = GramSchmidt([Matrix([Float("1.0"), Float("2.0")]),
                              Matrix([Float("2.0"), Float("1.0")])], orthonormal=True)
        for i, left in enumerate(result):
            for j, right in enumerate(result):
                value = sum(float(left[k]) * float(right[k]) for k in range(len(left)))
                self.assertAlmostEqual(value, 1.0 if i == j else 0.0, places=12)

    def test_original_held_pole_cannot_disappear_through_zero_product(self):
        pole = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
        for normalized in (False, True):
            with self.subTest(normalized=normalized):
                with self.assertRaisesRegex(ValueError, "finite, defined"):
                    GramSchmidt([Matrix([pole, 1])], orthonormal=normalized)

    def test_all_original_vectors_are_admitted_before_dependence(self):
        pole = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
        with self.assertRaisesRegex(ValueError, "finite, defined"):
            GramSchmidt([zeros(2, 1), Matrix([pole, 1])])

    def test_nonfinite_original_entries_are_refused(self):
        for value in (S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite, defined"):
                    GramSchmidt([Matrix([value, 1])])


if __name__ == "__main__":
    unittest.main()
