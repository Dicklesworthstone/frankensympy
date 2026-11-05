"""Directional series through the actual extension, including branch-sensitive rays."""

import unittest
from unittest.mock import patch

from sympy import (
    Abs, Add, Dummy, I, Mul, Order, Pow, Rational, Symbol,
    exp, expand, log, oo, pi, series, simplify, sqrt,
)
from sympy.core import _lift_native_result, _native, _native_expr, _symbol_binding


class DirectionalSeriesTests(unittest.TestCase):
    def setUp(self):
        self.x = Symbol("x")

    def assert_zero(self, expression):
        self.assertEqual(simplify(expand(expression)), 0)

    def remainder(self, expression):
        terms = (expression,) if type(expression) is Order else expression.args
        orders = [term for term in terms if type(term) is Order]
        self.assertEqual(len(orders), 1)
        return orders[0]

    def test_absolute_value_and_expression_method_select_the_left_ray(self):
        x = self.x
        self.assertEqual(series(Abs(x), x, 0, 5, dir="+"), x)
        self.assertEqual(series(Abs(x), x, 0, 5, dir="-"), -x)
        self.assertEqual(Abs(x).series(x, 0, 5, dir="-"), -x)

    def test_shifted_cusps_and_sign_quotients_keep_the_requested_side(self):
        x = self.x
        for point in (0, 2, Rational(-2, 3)):
            local = x - point
            with self.subTest(point=point):
                self.assert_zero(series(Abs(local), x, point, 5, dir="-") + local)
                self.assertEqual(series(Abs(local)/local, x, point, 5, dir="-"), -1)
                self.assertEqual(series(Abs(local)/local, x, point, 5, dir="+"), 1)

    def test_rational_poles_keep_all_signed_laurent_coefficients(self):
        x = self.x
        expression = (1 + 2*x)/(x**3*(1 - x))
        left = series(expression, x, 0, 4, dir="-")
        right = series(expression, x, 0, 4, dir="+")
        expected = x**-3 + 3*x**-2 + 3*x**-1 + 3 + 3*x + 3*x**2 + 3*x**3
        self.assert_zero(left.removeO() - expected)
        self.assert_zero(right.removeO() - expected)
        self.assertEqual(self.remainder(left).expr, x**4)

    def test_high_order_rational_expansion_retains_the_checked_native_lane(self):
        x = self.x
        result = series(1/(1 - x), x, 0, 96, dir="-")
        expected = sum(x**power for power in range(96))
        self.assert_zero(result.removeO() - expected)
        self.assertEqual(self.remainder(result).expr, x**96)

    def test_analytic_series_and_odd_remainder_have_the_usual_public_form(self):
        x = self.x
        result = exp(x).series(x, 0, 5, dir="-")
        expected = 1 + x + x**2/2 + x**3/6 + x**4/24
        self.assert_zero(result.removeO() - expected)
        self.assertEqual(self.remainder(result).expr, x**5)
        self.assertEqual(self.remainder(result).point, (0,))
        shifted = exp(x - 2).series(x, 2, 5, dir="-")
        self.assertEqual(self.remainder(shifted).expr, (x - 2)**5)
        self.assertEqual(self.remainder(shifted).point, (2,))

    def test_principal_fractional_powers_keep_the_selected_complex_phase(self):
        x = self.x
        self.assert_zero(series(sqrt(x), x, 0, 4, dir="-") - I*sqrt(-x))
        self.assert_zero(series(sqrt(x), x, 0, 4, dir="+") - sqrt(x))
        self.assert_zero(series(sqrt(-x), x, 0, 4, dir="-") - sqrt(-x))
        self.assert_zero(series(sqrt(x - 2), x, 2, 4, dir="-") - I*sqrt(2 - x))

    def test_logarithmic_coefficients_use_the_same_ray_as_power_terms(self):
        x = self.x
        self.assert_zero(series(log(x), x, 0, 4, dir="-") - log(-x) - I*pi)
        self.assert_zero(series(log(x - 2), x, 2, 4, dir="-") - log(2 - x) - I*pi)
        self.assert_zero(series(log(-x), x, 0, 4, dir="-") - log(-x))
        self.assert_zero(series(x*log(x), x, 0, 4, dir="-") - x*(log(-x) + I*pi))

    def test_infinite_centers_choose_the_inward_direction(self):
        x = self.x
        for direction in ("+", "-"):
            with self.subTest(direction=direction):
                self.assertEqual(series(Abs(x), x, oo, 5, dir=direction), x)
                self.assertEqual(series(Abs(x), x, -oo, 5, dir=direction), -x)
                self.assert_zero(
                    series(1/x + 2/x**2, x, -oo, 5, dir=direction) - 1/x - 2/x**2,
                )

    def test_typed_variables_aliases_and_placeholder_names_preserve_identity(self):
        for variable in (Symbol("_fsym_series_t", real=True), Dummy("_fsym_series_t")):
            parameter = Symbol("_fsym_log_t", real=True)
            with self.subTest(variable=variable):
                expression = log(variable) + parameter
                result = series(expression, variable, 0, 4, dir="-")
                self.assert_zero(result - log(-variable) - I*pi - parameter)
                self.assertEqual(result.free_symbols, {variable, parameter})
                alias = series(Abs(variable - 2) + parameter, variable=variable, point=2, dir="-")
                self.assert_zero(alias - (2 - variable + parameter))
                self.assertEqual(alias.free_symbols, {variable, parameter})

    def test_original_undefined_children_are_not_erased_by_reflection(self):
        x = self.x
        undefined = Pow(0, -1, evaluate=False)
        sources = (
            Mul(0, undefined, Abs(x), evaluate=False),
            Add(Abs(x), Mul(0, undefined, evaluate=False), evaluate=False),
        )
        for source in sources:
            for direction in ("+", "-"):
                with self.subTest(direction=direction, source=source):
                    with self.assertRaises((ValueError, NotImplementedError)):
                        series(source, x, 0, 5, dir=direction)

    def test_invalid_directions_and_operational_failures_are_terminal(self):
        x = self.x
        with patch.object(
            _native, "series_expansion_native", side_effect=AssertionError("invalid direction reached native"),
        ) as native:
            for direction in ("", "+-", "left", 1, None):
                with self.subTest(direction=direction):
                    with self.assertRaises(ValueError):
                        series(Abs(x), x, 0, 5, dir=direction)
            self.assertEqual(native.call_count, 0)
        failure = ValueError("directional expansion work exhausted")
        with patch.object(_native, "series_expansion_native", side_effect=failure) as native:
            with self.assertRaises(ValueError) as error:
                series(Abs(x), x, 0, 5, dir="-")
            self.assertIs(error.exception, failure)
            self.assertEqual(native.call_count, 1)

    def test_native_default_stays_right_sided_and_explicit_left_is_distinct(self):
        x = self.x
        arguments = (_native_expr(Abs(x)/x), _symbol_binding(x), _native_expr(0), 5)
        default = _native.series_expansion_native(*arguments)
        plus = _native.series_expansion_native(*arguments, "+")
        minus = _native.series_expansion_native(*arguments, "-")
        self.assertEqual(default, plus)
        self.assertEqual(_lift_native_result(default[0][0][0], x), 1)
        self.assertEqual(_lift_native_result(minus[0][0][0], x), -1)
        with self.assertRaises(ValueError):
            _native.series_expansion_native(*arguments, "+-")


if __name__ == "__main__":
    unittest.main()
