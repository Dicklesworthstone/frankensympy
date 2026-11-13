"""Principal inverse-function series and limits through the actual extension."""

import unittest

from sympy import (
    I, Limit, Rational, Symbol, acos, acosh, acot, acoth, asin, asinh,
    atan, atanh, expand, limit, pi, series, simplify, sqrt,
)


class InverseBranchSeriesTests(unittest.TestCase):
    def setUp(self):
        self.x = Symbol("x")

    def assert_zero(self, value):
        self.assertEqual(simplify(expand(value)), 0)

    def test_both_complex_approaches_keep_the_principal_boundary_value(self):
        x = self.x
        cases = (
            (asin, Rational(5, 4), I, pi - asin(Rational(5, 4)), asin(Rational(5, 4))),
            (acos, Rational(5, 4), I, -acos(Rational(5, 4)), acos(Rational(5, 4))),
            (atan, 2*I, 1, atan(2*I), atan(2*I) - pi),
            (asinh, 5*I/4, 1, asinh(5*I/4), I*pi - asinh(5*I/4)),
            (acosh, Rational(-5, 4), I, acosh(Rational(-5, 4)),
             acosh(Rational(-5, 4)) - 2*I*pi),
            (atanh, Rational(5, 4), I, atanh(Rational(5, 4)) + I*pi,
             atanh(Rational(5, 4))),
            (acot, I/2, 1, acot(I/2) + pi, acot(I/2)),
            (acoth, Rational(1, 2), I, acoth(Rational(1, 2)),
             acoth(Rational(1, 2)) + I*pi),
        )
        for function, center, tangent, right, left in cases:
            source = function(center + tangent*x)
            for direction, expected in (("+", right), ("-", left)):
                with self.subTest(function=function, direction=direction):
                    self.assert_zero(limit(source, x, 0, dir=direction) - expected)
                    self.assert_zero(series(source, x, 0, 1, dir=direction).removeO() - expected)
            self.assertIsInstance(limit(source, x, 0, dir="+-"), Limit)

    def test_reflected_branches_change_all_taylor_coefficients(self):
        x = self.x
        cases = (
            (asin(Rational(5, 4) + I*x),
             pi - asin(Rational(5, 4)) - 4*x/3 + 40*I*x*x/27),
            (acos(Rational(5, 4) + I*x),
             -acos(Rational(5, 4)) + 4*x/3 - 40*I*x*x/27),
            (asinh(5*I/4 - x),
             I*pi - asinh(5*I/4) - 4*I*x/3 - 40*x*x/27),
            (acosh(Rational(1, 2) - I*x),
             -acosh(Rational(1, 2)) + 2*sqrt(3)*x/3 - 2*I*sqrt(3)*x*x/9),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assert_zero(series(source, x, 0, 3).removeO() - expected)

    def test_late_path_terms_still_select_the_inverse_branch(self):
        x = self.x
        for source, expected in (
            (asin(Rational(5, 4) + I*x**100), pi - asin(Rational(5, 4))),
            (atan(2*I - x**100), atan(2*I) - pi),
            (acosh(Rational(-5, 4) - I*x**100), acosh(Rational(-5, 4)) - 2*I*pi),
            (acoth(Rational(1, 2) - I*x**100), acoth(Rational(1, 2)) + I*pi),
        ):
            with self.subTest(source=source):
                self.assert_zero(series(source, x, 0, 1).removeO() - expected)

    def test_zero_jumps_are_resolved_for_reciprocal_inverse_functions(self):
        x = self.x
        for source, right, left in (
            (acot(x), pi/2, -pi/2),
            (acot(I*x), -pi/2, pi/2),
            (acoth(x), -I*pi/2, I*pi/2),
            (acoth(I*x), -I*pi/2, I*pi/2),
        ):
            for direction, expected in (("+", right), ("-", left)):
                with self.subTest(source=source, direction=direction):
                    self.assert_zero(series(source, x, 0, 1, dir=direction).removeO() - expected)
                    self.assert_zero(limit(source, x, 0, dir=direction) - expected)

    def test_shifted_cut_points_and_zero_finite_boundary_values(self):
        x = self.x
        source = asin(Rational(5, 4) + I*(x - 2))
        self.assert_zero(limit(source, x, 2, dir="+") - (pi - asin(Rational(5, 4))))
        self.assert_zero(limit(source, x, 2, dir="-") - asin(Rational(5, 4)))
        self.assert_zero(limit(asin(x), x, 0, dir="+-"))
        self.assert_zero(limit(acos(x), x, 1, dir="+-"))
        self.assert_zero(limit(atan(1/x), x, 0, dir="+") - pi/2)
        self.assert_zero(limit(atan(1/x), x, 0, dir="-") + pi/2)

    def test_parameter_signs_are_typed_and_unknown_cut_sides_refuse(self):
        x = self.x
        for parameter, expected in (
            (Symbol("a", positive=True), pi - asin(Rational(5, 4))),
            (Symbol("a", negative=True), asin(Rational(5, 4))),
        ):
            with self.subTest(parameter=parameter):
                source = asin(Rational(5, 4) + I*parameter*x)
                self.assert_zero(series(source, x, 0, 1).removeO() - expected)
        unknown = Symbol("a", real=True)
        source = asin(Rational(5, 4) + I*unknown*x)
        with self.assertRaises(ValueError):
            series(source, x, 0, 2)
        self.assertIsInstance(limit(source, x, 0), Limit)

    def test_finite_limits_preserve_zero_parameter_specializations(self):
        x = self.x
        unknown = Symbol("a", real=True)
        source = atan(1 + I + (sqrt(unknown**2 + x*x) - sqrt(unknown**2))/x)
        self.assertIsInstance(limit(source, x, 0), Limit)
        positive = Symbol("a", positive=True)
        source = atan(1 + I + (sqrt(positive**2 + x*x) - sqrt(positive**2))/x)
        self.assert_zero(limit(source, x, 0) - atan(1 + I))
        self.assert_zero(limit(atan(1 + I + sqrt(x*x)/x), x, 0) - atan(2 + I))


if __name__ == "__main__":
    unittest.main()
