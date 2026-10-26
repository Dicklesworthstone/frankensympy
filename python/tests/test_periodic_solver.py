"""Exact root admission for periodic solution sets (WS19 / fra-7kf).

Numerical angle recognition is only a candidate generator. These regressions
exercise exact root identity, real-domain guards, original poles, and unknown
membership without treating numerical agreement as a completeness proof.
"""

from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import sympy as s


periodic = importlib.import_module("sympy.solvers.solveset")


def families(result):
    return result.args if isinstance(result, s.Union) else (result,)


class PeriodicSolverExactAdmission(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("x")

    def assert_condition(self, result, expression, domain):
        self.assertIsInstance(result, s.ConditionSet)
        self.assertEqual(result.sym, self.x)
        self.assertEqual(result.condition, s.Eq(expression, 0, evaluate=False))
        self.assertEqual(result.base_set, domain)

    def test_angle_recognition_cannot_substitute_a_nearby_polynomial_root(self):
        y = s.Symbol("y")
        t = s.Rational(1, 10**12)
        root = (1 - t * t + 2 * s.I * t) / (1 + t * t)
        polynomial = (y - 1) * (y - root)
        self.assertNotEqual(s.expand(root - 1), 0)
        self.assertEqual(s.expand(polynomial.subs(y, 1)), 0)
        # Binary64 proposes angle zero for the very close *other* root.
        # Checking polynomial(1) is insufficient to identify this root.
        with patch.object(periodic, "_probe", return_value=complex(1, 2e-12)):
            self.assertIsNone(periodic._unit_root_angle(polynomial, y, root))

    def test_angle_recognition_rejects_a_nonunit_root_inside_float_tolerance(self):
        y = s.Symbol("y")
        root = 1 + s.Rational(1, 10**30)
        polynomial = (y - 1) * (y - root)
        with patch.object(periodic, "_probe", return_value=complex(1, 0)):
            self.assertIsNone(periodic._unit_root_angle(polynomial, y, root))

    def test_exact_unit_membership_keeps_arbitrarily_small_nonzero_residuals(self):
        root = (3 + 4 * s.I) / 5
        membership, _ = periodic._unit_circle_parts(root)
        self.assertIs(membership, True)
        for scale in (1 + s.Rational(1, 10**30), 1 - s.Rational(1, 10**30)):
            with self.subTest(scale=scale):
                membership, _ = periodic._unit_circle_parts(scale * root)
                self.assertIs(membership, False)
        self.assertIs(periodic._unit_circle_parts(s.Symbol("z"))[0], None)

    def test_principal_gaussian_square_roots_keep_exact_branch_and_value(self):
        for value, expected in ((3 + 4 * s.I, 2 + s.I), (-3 - 4 * s.I, 1 - 2 * s.I)):
            with self.subTest(value=value):
                result = periodic._rectangular_radicals(s.sqrt(value))
                self.assertEqual(s.expand(result - expected), 0)
                self.assertEqual(s.expand(result * result - value), 0)

    def test_complete_root_list_requires_every_factor_and_multiplicity(self):
        y = s.Symbol("y")
        polynomial = (y - 1)**2 * (y + 2)
        for roots in ([s.Integer(1), s.Integer(-2)], [s.Integer(-2), s.Integer(1), s.Integer(1)]):
            with self.subTest(roots=roots):
                self.assertTrue(periodic._roots_cover_polynomial(polynomial, y, roots))
        for roots in ([s.Integer(1)], [s.Integer(1), s.Integer(-2), s.Integer(3)],
                      [1 + s.Rational(1, 10**30), s.Integer(-2)]):
            with self.subTest(roots=roots):
                self.assertFalse(periodic._roots_cover_polynomial(polynomial, y, roots))
        self.assertFalse(periodic._roots_cover_polynomial(s.Symbol("a") * (y - 1), y, [s.Integer(1)]))

    def test_unknown_affine_scale_preserves_its_zero_branch(self):
        a = s.Symbol("a", real=True)
        expression = s.sin(a * self.x)
        result = s.solveset(expression, self.x, domain=s.S.Reals)
        self.assert_condition(result, expression, s.S.Reals)

    def test_unknown_outer_coefficient_preserves_its_zero_branch(self):
        a = s.Symbol("a", real=True)
        expression = a * s.sin(self.x)
        result = s.solveset(expression, self.x, domain=s.S.Reals)
        self.assert_condition(result, expression, s.S.Reals)

    def test_unknown_real_range_retains_the_original_equation(self):
        a = s.Symbol("a", real=True)
        expression = s.sin(self.x) - a
        result = s.solveset(expression, self.x, domain=s.S.Reals)
        self.assert_condition(result, expression, s.S.Reals)

    def test_complex_shift_does_not_publish_unfiltered_real_solutions(self):
        expression = s.sin(self.x + s.I)
        result = s.solveset(expression, self.x, domain=s.S.Reals)
        self.assert_condition(result, expression, s.S.Reals)

    def test_complex_exponential_unknown_scale_retains_the_zero_branch(self):
        a = s.Symbol("a")
        expression = s.exp(a * self.x) - 1
        result = s.solveset(expression, self.x, domain=s.S.Complexes)
        self.assert_condition(result, expression, s.S.Complexes)

    def test_complex_tangent_omitted_values_have_no_solutions(self):
        for value in (s.I, -s.I):
            with self.subTest(value=value):
                result = s.solveset(s.tan(self.x) - value, self.x, domain=s.S.Complexes)
                self.assertEqual(result, s.S.EmptySet)

    def test_known_real_inversion_remains_supported(self):
        result = s.solveset(s.sin(2 * self.x) - s.Rational(1, 2), self.x, domain=s.S.Reals)
        branches = families(result)
        self.assertEqual(len(branches), 2)
        self.assertEqual({branch.lamda(0) for branch in branches}, {s.pi / 12, 5 * s.pi / 12})
        for branch in branches:
            self.assertEqual(s.expand(branch.lamda(1) - branch.lamda(0)), s.pi)
        self.assertEqual(s.solveset(s.sin(self.x) - 2, self.x, domain=s.S.Reals), s.S.EmptySet)

    def test_exact_non_root_of_unity_solutions_need_no_numeric_probe(self):
        expression = s.sin(self.x) + s.cos(self.x) - s.Rational(1, 2)
        with patch.object(periodic, "_probe", return_value=None):
            result = s.solveset(expression, self.x, domain=s.S.Reals)
        branches = families(result)
        self.assertEqual(len(branches), 2)
        # Independently solve u+v=1/2, u^2+v^2=1 for cos and sin.
        low = s.expand((1 - s.sqrt(7)) / 4)
        high = s.expand((1 + s.sqrt(7)) / 4)
        expected = [s.atan2(low, high), s.atan2(high, low)]
        for branch in branches:
            self.assertIsInstance(branch, s.ImageSet)
            angle = branch.lamda(0)
            # This is an exact solution-set check, independent of whether
            # an engine leaves atan2 or rewrites it to quadrant-correct atan.
            match = next((i for i, value in enumerate(expected)
                          if s.simplify(angle - value) == 0), None)
            self.assertIsNotNone(match)
            expected.pop(match)
            self.assertEqual(s.expand(branch.lamda(1) - angle), 2 * s.pi)
        self.assertEqual(expected, [])

    def test_rational_unit_circle_roots_produce_exact_nonperiodic_angles(self):
        expression = s.sin(self.x) + s.cos(self.x) - s.Rational(7, 5)
        with patch.object(periodic, "_probe", return_value=None):
            result = s.solveset(expression, self.x, domain=s.S.Reals)
        branches = families(result)
        self.assertEqual(len(branches), 2)
        self.assertEqual(
            {branch.lamda(0) for branch in branches},
            {s.atan(s.Rational(3, 4)), s.atan(s.Rational(4, 3))},
        )

    def test_nonunit_roots_are_excluded_exactly(self):
        expression = s.sin(self.x) + s.cos(self.x) - 2
        with patch.object(periodic, "_probe", return_value=None):
            result = s.solveset(expression, self.x, domain=s.S.Reals)
        self.assertEqual(result, s.S.EmptySet)

    def test_rational_cancellation_does_not_erase_trigonometric_poles(self):
        # Cancelling cos(x) would turn this into sin(x)-1, whose only
        # candidate families are precisely where the original tan is undefined.
        expression = s.cos(self.x) * s.tan(self.x) - 1
        result = s.solveset(expression, self.x, domain=s.S.Reals)
        self.assertEqual(result, s.S.EmptySet)

    def test_unsupported_nonperiodic_equation_remains_visible(self):
        with self.assertRaises(NotImplementedError):
            s.solveset(s.sin(self.x) + self.x, self.x, domain=s.S.Reals)


if __name__ == "__main__":
    unittest.main()
