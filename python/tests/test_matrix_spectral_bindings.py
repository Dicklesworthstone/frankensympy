"""Characteristic coordinates cannot capture source matrix parameters.

The coefficient formulas and generator-name expectations below are exact;
the latter follow the isolated SymPy 1.14 oracle. Product tests import the
actual FrankenSymPy shell and native extension, never the upstream engine.
"""

import unittest
from unittest.mock import patch

import sympy as s


class CharacteristicGeneratorTests(unittest.TestCase):
    def test_default_name_does_not_erase_a_scalar_matrix_parameter(self):
        source = s.Symbol("lambda")
        for matrix_type in (s.Matrix, s.ImmutableMatrix):
            with self.subTest(matrix_type=matrix_type):
                result = matrix_type([[source]]).charpoly()
                self.assertIs(type(result), s.PurePoly)
                self.assertIs(type(result.gen), s.Symbol)
                self.assertEqual(result.gen, s.Symbol("_lambda"))
                self.assertEqual(result.degree(), 1)
                self.assertEqual(result.all_coeffs(), [1, -source])
                self.assertEqual(result.as_expr().free_symbols, {result.gen, source})

    def test_requested_axis_remains_independent_of_every_coefficient(self):
        x = s.Symbol("x")
        matrix = s.Matrix([[x, 1], [0, x]])
        for requested in (x, "x"):
            with self.subTest(requested=requested):
                result = matrix.charpoly(requested)
                self.assertEqual(result.gen, s.Symbol("_x"))
                self.assertEqual(result.degree(), 2)
                self.assertEqual(result.all_coeffs(), [1, -2*x, x*x])
                self.assertEqual(s.expand(result.as_expr().subs(result.gen, x)), 0)

    def test_underscore_prefix_checks_the_entire_original_source(self):
        x, first, second = s.symbols("x _x __x")
        matrix = s.Matrix([[x, first], [second, 2]])
        result = matrix.charpoly(x)
        self.assertEqual(result.gen, s.Symbol("___x"))
        expected = [1, -x - 2, 2*x - first*second]
        self.assertEqual([s.expand(value) for value in result.all_coeffs()], expected)
        self.assertEqual(result.as_expr().free_symbols, {x, first, second, result.gen})

    def test_symbols_absent_from_final_coefficients_still_reserve_their_names(self):
        x = s.Symbol("x")
        # This strictly upper-triangular cell disappears from every coefficient.
        # The public name rule nevertheless inspects the original matrix.
        result = s.Matrix([[1, x], [0, 2]]).charpoly(x)
        self.assertEqual(result.gen, s.Symbol("_x"))
        self.assertEqual(result.all_coeffs(), [1, -3, 2])

    def test_assumptions_cannot_hide_a_name_collision_or_change_source_identity(self):
        source = s.Symbol("x", positive=True)
        result = s.Matrix([[source]]).charpoly(s.Symbol("x", negative=True))
        self.assertEqual(result.gen, s.Symbol("_x"))
        self.assertEqual(result.all_coeffs(), [1, -source])
        self.assertTrue(any(symbol is source for symbol in result.as_expr().free_symbols))
        self.assertNotIn(s.Symbol("x"), result.as_expr().free_symbols)

    def test_dummy_source_and_same_printed_output_are_distinct_typed_atoms(self):
        source = s.Dummy("x")
        result = s.Matrix([[source]]).charpoly("x")
        self.assertEqual(result.gen, s.Symbol("_x"))
        self.assertNotEqual(result.gen, source)
        self.assertEqual(result.all_coeffs(), [1, -source])
        self.assertTrue(any(symbol is source for symbol in result.as_expr().free_symbols))

    def test_different_same_named_dummies_remain_distinct_coefficients(self):
        first, second = s.Dummy("x"), s.Dummy("x")
        result = s.diag(first, second).charpoly("x")
        self.assertEqual(result.gen, s.Symbol("_x"))
        self.assertEqual(result.degree(), 2)
        self.assertEqual(result.all_coeffs(), [1, -first - second, first*second])
        self.assertEqual(result.as_expr().free_symbols, {first, second, result.gen})

    def test_plain_uncolliding_generator_keeps_its_interned_identity(self):
        requested = s.Symbol("axis")
        result = s.Matrix([[1, 2], [3, 4]]).charpoly(requested)
        self.assertIs(result.gen, requested)
        self.assertEqual(result.all_coeffs(), [1, -5, -2])

    def test_output_name_hints_do_not_leak_assumptions_or_dummy_identity(self):
        for requested, name in (
            (s.Symbol("axis", positive=True), "axis"),
            (s.Symbol("axis", real=True), "axis"),
            (s.Dummy("axis"), "_axis"),
        ):
            with self.subTest(requested=requested):
                result = s.eye(2).charpoly(requested)
                self.assertIs(type(result.gen), s.Symbol)
                self.assertIs(result.gen, s.Symbol(name))
                self.assertEqual(result.all_coeffs(), [1, -2, 1])
                self.assertIsNone(result.gen.is_positive)

    def test_dummy_name_hint_skips_a_colliding_plain_symbol(self):
        source = s.Symbol("_axis")
        result = s.Matrix([[source]]).charpoly(s.Dummy("axis"))
        self.assertEqual(result.gen, s.Symbol("__axis"))
        self.assertEqual(result.all_coeffs(), [1, -source])

    def test_empty_and_noncolliding_named_matrices_keep_existing_coefficients(self):
        self.assertEqual(s.Matrix(0, 0, []).charpoly().all_coeffs(), [1])
        source = s.Symbol("_x")
        result = s.Matrix([[source]]).charpoly("x")
        self.assertEqual(result.gen, s.Symbol("x"))
        self.assertEqual(result.all_coeffs(), [1, -source])

    def test_repeated_calls_and_matrix_mutation_choose_the_current_free_name(self):
        x = s.Symbol("x")
        matrix = s.Matrix([[x]])
        first = matrix.charpoly(x)
        second = matrix.charpoly(x)
        self.assertIs(first.gen, second.gen)
        self.assertEqual(first, second)
        matrix[0, 0] = s.Symbol("_x")
        result = matrix.charpoly(x)
        self.assertIs(result.gen, x)
        self.assertEqual(result.all_coeffs(), [1, -s.Symbol("_x")])


class SpectralConsumerTests(unittest.TestCase):
    def test_eigenvalue_root_lane_uses_the_generator_returned_by_charpoly(self):
        variable = s.Symbol("renamed_spectral_axis")
        polynomial = s.PurePoly(variable**2 - 5*variable + 6, variable)
        matrix = s.diag(2, 3)

        def forbidden(*args, **kwargs):
            raise AssertionError("a complete characteristic root list must not need another lane")

        with patch.object(s.Matrix, "charpoly", return_value=polynomial), \
                patch.object(s.Matrix, "_symbolic_eigenvals", forbidden), \
                patch.object(s.Matrix, "eigenvalues", forbidden):
            self.assertEqual(matrix.eigenvals(), {s.Integer(2): 1, s.Integer(3): 1})

    def test_scratch_coordinate_spellings_preserve_symbolic_eigenvalues(self):
        for name in ("lambda", "_lambda", "_fsym_lambda", "__fsym_lambda"):
            for constructor in (s.Symbol, s.Dummy):
                source = constructor(name)
                with self.subTest(name=name, constructor=constructor):
                    matrix = s.Matrix([[source, 1], [0, source]])
                    self.assertEqual(matrix.eigenvals(), {source: 2})

    def test_symbolic_root_and_multiplicity_lane_follows_the_returned_axis(self):
        for name in ("lambda", "_lambda", "__lambda"):
            source = s.Symbol(name)
            with self.subTest(name=name):
                matrix = s.Matrix([[source, 1], [0, source]])
                self.assertEqual(matrix._symbolic_eigenvals(), {source: 2})

    def test_capture_sensitive_eigenvectors_satisfy_the_original_matrix(self):
        source = s.Symbol("_fsym_lambda")
        matrix = s.Matrix([[source, 1], [0, source]])
        eigenvectors = matrix.eigenvects()
        self.assertEqual(len(eigenvectors), 1)
        value, multiplicity, basis = eigenvectors[0]
        self.assertEqual((value, multiplicity), (source, 2))
        self.assertEqual(len(basis), 1)
        for vector in basis:
            self.assertNotEqual(vector, s.zeros(2, 1))
            self.assertEqual((matrix*vector - source*vector).applyfunc(s.expand), s.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()
