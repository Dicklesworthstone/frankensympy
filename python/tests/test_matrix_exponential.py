"""Checked dense/sparse matrix exponentials preserve source and result contracts."""

import unittest
from unittest.mock import patch

import sympy as sp
from sympy.core import _native, _native_expr
from sympy.matrices._exponential import _checked_exponential


CLASSES = (sp.Matrix, sp.ImmutableMatrix, sp.SparseMatrix, sp.ImmutableSparseMatrix)


class MatrixExponentialTests(unittest.TestCase):
    def assert_expression_equal(self, actual, expected):
        self.assertEqual(sp.simplify(sp.expand(actual - expected)), 0)

    def test_nilpotent_fill_and_result_class_for_all_builtin_matrices(self):
        expected = sp.Matrix([[1, 1, sp.Rational(1, 2)], [0, 1, 1], [0, 0, 1]])
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                source = cls([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
                result = source.exp()
                self.assertIs(type(result), cls)
                self.assertEqual(result, expected)
                self.assertEqual(source[0, 2], 0)
                if cls is sp.SparseMatrix or cls is sp.ImmutableSparseMatrix:
                    self.assertEqual(result.nnz, 6)
                if cls is sp.ImmutableMatrix or cls is sp.ImmutableSparseMatrix:
                    with self.assertRaises(TypeError):
                        result[0, 0] = 5

    def test_checked_dense_lane_does_not_depend_on_jordan_vectors(self):
        source = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        with patch.object(sp.Matrix, "jordan_form", side_effect=AssertionError("Jordan called")):
            result = source.exp()
        self.assertEqual(result[0, 2], sp.Rational(1, 2))

    def test_empty_scalar_and_zero_matrices_keep_exact_identity(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                empty = cls(0, 0, [])
                result = empty.exp()
                self.assertIs(type(result), cls)
                self.assertEqual(result.shape, (0, 0))
                self.assertIsNot(result, empty)
                self.assertEqual(cls([[0]]).exp()[0, 0], 1)
                self.assertEqual(cls([[1]]).exp()[0, 0], sp.E)
                self.assertEqual(cls([[0, 0], [0, 0]]).exp(), sp.eye(2))

    def test_sparse_source_never_uses_public_dense_or_cell_materialization(self):
        source = sp.SparseMatrix(3, 3, {(0, 1): 1, (1, 2): 1})
        with patch.object(sp.SparseMatrix, "to_dense", side_effect=AssertionError("dense source")), \
                patch.object(sp.SparseMatrix, "todok", side_effect=AssertionError("surface copy")):
            result = source.exp()
            self.assertEqual(result[0, 2], sp.Rational(1, 2))
            self.assertEqual(result.nnz, 6)
        self.assertEqual(source.nnz, 2)

    def test_native_only_dense_and_sparse_inputs_keep_native_provenance(self):
        entries = [_native.py_integer(value) for value in [0, 1, 0, 0]]
        dense = sp.Matrix(_native.Matrix(2, 2, entries))
        sparse = sp.SparseMatrix(_native.SparseMatrix(2, 2, {(0, 1): _native.py_integer(1)}))
        self.assertIsNone(dense._surface)
        self.assertIsNone(sparse._surface)
        self.assertEqual(dense.exp(), sp.Matrix([[1, 1], [0, 1]]))
        self.assertEqual(sparse.exp(), sp.Matrix([[1, 1], [0, 1]]))

    def test_existing_symbolic_and_large_dense_paths_remain_available(self):
        x = sp.Symbol("x")
        source = sp.Matrix([[x, 0], [0, 2*x]])
        self.assertIsNone(_checked_exponential(source, sparse=False))
        self.assertEqual(source.exp(), sp.diag(sp.exp(x), sp.exp(2*x)))
        large = sp.eye(9)
        self.assertIsNone(_checked_exponential(large, sparse=False))
        self.assertEqual(large.exp(), sp.eye(9) * sp.E)

    def test_oscillatory_similarity_has_correct_full_coordinate_map(self):
        source = sp.Matrix([[-6, 2, 3], [-9, 3, 4], [-8, 2, 5]])
        result = source.exp() * sp.Matrix([2, 2, 3])
        expected = [
            sp.cos(1) - sp.sin(1) + sp.exp(2),
            sp.cos(1) - 2*sp.sin(1) + sp.exp(2),
            sp.cos(1) - sp.sin(1) + 2*sp.exp(2),
        ]
        for actual, wanted in zip(result, expected):
            self.assert_expression_equal(actual, wanted)

    def test_hyperbolic_frequency_and_shifted_jordan_normalization(self):
        result = sp.SparseMatrix([[0, 1, 0], [2, 0, 0], [0, 0, 1]]).exp()
        self.assert_expression_equal(result[0, 0], sp.cosh(sp.sqrt(2)))
        self.assert_expression_equal(result[0, 1], sp.sinh(sp.sqrt(2))/sp.sqrt(2))
        self.assert_expression_equal(result[1, 0], sp.sqrt(2)*sp.sinh(sp.sqrt(2)))
        shifted = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]]).exp()
        self.assert_expression_equal(shifted[0, 2], sp.E/2)

    def test_nilpotent_inverse_exponential_identity(self):
        source = sp.SparseMatrix([[0, 2, 0], [0, 0, 3], [0, 0, 0]])
        self.assertEqual(source.exp() * (-source).exp(), sp.eye(3))

    def test_sparse_unsupported_spectrum_is_explicit(self):
        source = sp.SparseMatrix([[0, 1, 0], [0, 0, 1], [2, 0, 0]])
        with self.assertRaises(NotImplementedError):
            source.exp()
        dense = sp.Matrix(source)
        with patch.object(sp.Matrix, "jordan_form", side_effect=AssertionError("retried")):
            with self.assertRaises(NotImplementedError):
                dense.exp()

    def test_resource_failure_is_terminal_before_any_legacy_jordan_retry(self):
        source = sp.Matrix([[sp.Integer(1 << 1024), 0], [0, 0]])
        with patch.object(sp.Matrix, "jordan_form", side_effect=AssertionError("retried")):
            with self.assertRaisesRegex(ValueError, "1024"):
                source.exp()

    def test_sparse_literals_are_admitted_without_erasing_held_domains(self):
        x = sp.Symbol("x")
        held = sp.Mul(0, sp.log(x), evaluate=False)
        for entry in [x, sp.Float(0), held]:
            with self.subTest(entry_type=type(entry).__name__):
                source = sp.SparseMatrix(1, 1, {(0, 0): entry})
                with self.assertRaises(NotImplementedError):
                    source.exp()
        native = _native.Matrix(1, 1, [_native_expr(x)])
        with self.assertRaises(NotImplementedError):
            native.exp()

    def test_huge_sparse_shape_and_rectangles_refuse_before_conversion(self):
        huge = sp.SparseMatrix.zeros(1_000_000, 1_000_000)
        with patch.object(sp.SparseMatrix, "to_dense", side_effect=AssertionError("dense source")), \
                patch.object(sp.SparseMatrix, "todok", side_effect=AssertionError("surface copy")):
            with self.assertRaisesRegex(ValueError, "eight"):
                huge.exp()
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(ValueError):
                    cls(0, 3, []).exp()

    def test_result_snapshots_do_not_follow_source_mutation(self):
        for cls in (sp.Matrix, sp.SparseMatrix):
            with self.subTest(cls=cls.__name__):
                source = cls([[0, 1], [0, 0]])
                frozen = source.as_immutable()
                result = source.exp()
                source[0, 1] = 7
                self.assertEqual(result, sp.Matrix([[1, 1], [0, 1]]))
                self.assertEqual(frozen.exp(), result)
                self.assertEqual(source.exp(), sp.Matrix([[1, 7], [0, 1]]))

    def test_metaclass_equality_cannot_impersonate_a_builtin_matrix_or_atom(self):
        effects = []

        class HostileMeta(type):
            def __eq__(cls, other):
                effects.append("class equality")
                return True

            __hash__ = type.__hash__

        class ForeignMatrix(sp.SparseMatrix, metaclass=HostileMeta):
            pass

        class ForeignCell(metaclass=HostileMeta):
            pass

        foreign = ForeignMatrix(1, 1, {(0, 0): 1})
        effects.clear()
        with self.assertRaises(NotImplementedError):
            foreign.exp()
        self.assertEqual(effects, [])

        source = sp.SparseMatrix(1, 1, {})
        source._surface = {(0, 0): ForeignCell()}
        with self.assertRaises(NotImplementedError):
            source.exp()
        self.assertEqual(effects, [])

    def test_replaced_storage_descriptor_is_refused_without_invoking_it(self):
        source = sp.Matrix([[0, 1], [0, 0]])
        effects = []

        def replaced(instance):
            effects.append("storage getter")
            raise AssertionError("uncontrolled getter")

        with patch.object(sp.Matrix, "_native_storage", property(replaced)):
            with self.assertRaises(NotImplementedError):
                source.exp()
        self.assertEqual(effects, [])


if __name__ == "__main__":
    unittest.main()
