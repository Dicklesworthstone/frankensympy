"""Directional series through the actual extension, including branch-sensitive rays."""

import unittest
from unittest.mock import patch

from sympy import (
    Abs, Add, Dummy, I, Limit, Mul, Order, Pow, Rational, Symbol,
    exp, expand, limit, log, oo, pi, series, simplify, sqrt,
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

    def test_negative_cut_continuation_uses_the_actual_complex_approach(self):
        x = self.x
        cases = (
            (log(-1 - I*x), -I*pi + I*x + x**2/2, -I*pi, I*pi),
            (sqrt(-1 - I*x), -I + x/2 - I*x**2/8, -I, I),
            (log(-1 + I*x), I*pi - I*x + x**2/2, I*pi, -I*pi),
            (sqrt(-1 + I*x), I + x/2 + I*x**2/8, I, -I),
        )
        for source, expected, right_limit, left_limit in cases:
            with self.subTest(source=source):
                result = series(source, x, 0, 3)
                self.assert_zero(result.removeO() - expected)
                self.assert_zero(limit(source, x, 0, dir="+") - right_limit)
                self.assert_zero(limit(source, x, 0, dir="-") - left_limit)

    def test_imaginary_terms_above_requested_order_still_select_the_branch(self):
        x = self.x
        for source, expected in (
            (log(-1 - I*x**100), -I*pi),
            (sqrt(-1 - I*x**100), -I),
            (log(-1 - I*x + I*x**2), -I*pi),
        ):
            with self.subTest(source=source):
                result = series(source, x, 0, 1)
                self.assert_zero(result.removeO() - expected)

    def test_typed_parameter_assumptions_select_the_cut_and_unknowns_refuse(self):
        x = self.x
        for parameter, expected in (
            (Symbol("a", positive=True), -I*pi),
            (Symbol("a", negative=True), I*pi),
        ):
            with self.subTest(parameter=parameter, expected=expected):
                source = log(-1 - I*parameter*x)
                result = series(source, x, 0, 2)
                self.assert_zero(result.removeO() - expected - I*parameter*x)
                self.assertEqual(result.free_symbols, {x, parameter})
                self.assert_zero(limit(source, x, 0) - expected)
        unknown = Symbol("a", real=True)
        source = log(-1 - I*unknown*x)
        with self.assertRaises(ValueError):
            series(source, x, 0, 2)
        self.assertIsInstance(limit(source, x, 0), Limit)

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


    def test_limit_eager_evaluation_keeps_the_requested_side(self):
        x = self.x
        source = Abs(x)/x
        for direction, expected in (("+", 1), ("-", -1)):
            for argument in (direction, Symbol(direction)):
                with self.subTest(direction=argument):
                    self.assertEqual(Limit(source, x, 0, dir=argument, evaluate=True), expected)
                    self.assertEqual(limit(source, x, 0, dir=argument), expected)
        with self.assertRaises(ValueError):
            Limit(source, x, 0, dir="+-", evaluate=True)

    def test_limit_reconstruction_keeps_all_four_typed_arguments(self):
        x = self.x
        for direction in ("+", "-", "+-"):
            with self.subTest(direction=direction):
                original = Limit(exp(x), x, 0, dir=direction)
                rebuilt = original.func(*original.args)
                self.assertEqual(original.args, (exp(x), x, 0, Symbol(direction)))
                self.assertEqual(rebuilt, original)
                self.assertEqual(hash(rebuilt), hash(original))
                self.assertEqual(rebuilt.dir, direction)
                self.assertEqual(rebuilt.doit(), 1)
                self.assertEqual({original: "retained"}[rebuilt], "retained")

    def test_limit_copies_and_pickle_preserve_the_approach(self):
        import copy
        import pickle

        x = Dummy("limit_axis", real=True)
        for direction, expected in (("+", 1), ("-", -1)):
            original = Limit(Abs(x)/x, x, 0, dir=direction)
            copies = (copy.copy(original), copy.deepcopy(original),
                      pickle.loads(pickle.dumps(original)))
            for rebuilt in copies:
                with self.subTest(direction=direction, rebuilt=rebuilt):
                    self.assertEqual(rebuilt, original)
                    self.assertEqual(hash(rebuilt), hash(original))
                    self.assertEqual(rebuilt.args, original.args)
                    self.assertEqual(rebuilt.var, x)
                    self.assertEqual(rebuilt.doit(), expected)

    def test_limit_constructor_retains_held_source_and_point_owners(self):
        x = Dummy("limit_axis", real=True)
        parameter = Symbol("a", positive=True)
        source = Add(x, parameter, x, evaluate=False)
        point = Add(parameter, 0, evaluate=False)
        with patch.object(_native, "limit_native", side_effect=AssertionError("constructor evaluated")):
            result = Limit(source, x, point, dir="-")
        self.assertIs(result.expr, source)
        self.assertEqual(result.expr.args, (x, parameter, x))
        self.assertIs(result.var, x)
        self.assertIs(result.point, point)
        self.assertEqual(result.point.args, (parameter, 0))
        self.assertEqual(result.free_symbols, {parameter})
        rebuilt = result.func(*result.args)
        self.assertIs(rebuilt.expr, source)
        self.assertIs(rebuilt.point, point)
        self.assertEqual(rebuilt, result)

    def test_limit_constructor_keeps_custom_sources_out_of_native_evaluation(self):
        from sympy import Function

        class OpaqueLimitFunction(Function):
            pass

        x = self.x
        source = OpaqueLimitFunction(x)
        with patch.object(_native, "limit_native", side_effect=AssertionError("opaque source lowered")):
            result = Limit(source, x, 0, dir="-")
            self.assertIs(result.expr, source)
            self.assertIs(type(result.expr), OpaqueLimitFunction)
            self.assertEqual(result.func(*result.args), result)
            with self.assertRaises(NotImplementedError):
                result.doit()

    def test_limit_symbolic_direction_retains_its_exact_identity(self):
        x = self.x
        direction = Symbol("-", positive=True)
        result = Limit(exp(x), x, 0, dir=direction)
        self.assertIs(result.args[3], direction)
        self.assertIs(result.func(*result.args).args[3], direction)
        self.assertNotEqual(result, Limit(exp(x), x, 0, dir="-"))
        self.assertEqual(result.doit(), 1)

    def test_limit_does_not_treat_printed_symbol_names_as_infinities(self):
        x = self.x
        for name, direction in (("oo", "+"), ("-oo", "-")):
            with self.subTest(name=name):
                point = Symbol(name, real=True)
                result = Limit(x, x, point, dir=direction)
                self.assertIs(result.point, point)
                self.assertEqual(result.dir, direction)
                self.assertEqual(result.args[3], Symbol(direction))
                self.assertEqual(result.free_symbols, {point})

    def test_limit_real_infinity_directions_are_shared_by_both_entry_points(self):
        x = self.x
        for point, expected in ((oo, "-"), (-oo, "+")):
            for supplied in ("+", "-", "+-", None, "not-a-direction"):
                with self.subTest(point=point, supplied=supplied):
                    result = Limit(1/x, x, point, dir=supplied)
                    self.assertEqual(result.dir, expected)
                    self.assertEqual(result.args[3], Symbol(expected))
                    self.assertEqual(result.doit(), 0)
                    self.assertEqual(limit(1/x, x, point, dir=supplied), 0)

    def test_limit_invalid_finite_directions_refuse_before_native_work(self):
        x = self.x

        class NotADirection:
            def __str__(self):
                raise AssertionError("a direction is not an arbitrary printable object")

        entries = (Limit, limit)
        with patch.object(_native, "limit_native", side_effect=AssertionError("invalid direction lowered")) as native:
            for entry in entries:
                for direction in (None, 1, [], NotADirection()):
                    with self.subTest(entry=entry, direction_type=type(direction)):
                        with self.assertRaises(TypeError):
                            entry(x, x, 0, dir=direction)
                for direction in ("", "left", "+-+", Symbol("side")):
                    with self.subTest(entry=entry, direction=direction):
                        with self.assertRaises(ValueError):
                            entry(x, x, 0, dir=direction)
            self.assertEqual(native.call_count, 0)

    def test_limit_rejects_variable_points_without_aliasing_typed_symbols(self):
        x = self.x
        with patch.object(_native, "limit_native", side_effect=AssertionError("variable point lowered")) as native:
            for entry in (Limit, limit):
                for point in (x, x + 1):
                    with self.subTest(entry=entry, point=point):
                        with self.assertRaises(NotImplementedError):
                            entry(exp(x), x, point)
            self.assertEqual(native.call_count, 0)
        other = Symbol("x", positive=True)
        result = Limit(exp(x), x, other)
        self.assertIs(result.point, other)
        self.assertEqual(result.free_symbols, {other})

    def test_undetermined_limit_retains_original_held_source_and_direction(self):
        x = self.x
        source = Add(Abs(x), 0, evaluate=False)
        direction = Symbol("-", real=True)
        with patch.object(_native, "limit_native", side_effect=ValueError("undetermined parameter sign")) as native:
            result = limit(source, x, 0, dir=direction)
        self.assertEqual(native.call_count, 1)
        self.assertIs(result.expr, source)
        self.assertIs(result.args[3], direction)
        self.assertEqual(result.func(*result.args), result)

    def test_limit_operational_failures_stay_terminal(self):
        x = self.x
        for failure in (ValueError("limit work budget exhausted"), RuntimeError("cancelled")):
            with self.subTest(failure=failure):
                with patch.object(_native, "limit_native", side_effect=failure) as native:
                    with self.assertRaises(type(failure)) as error:
                        Limit(exp(x), x, 0, dir="-", evaluate=True)
                    self.assertIs(error.exception, failure)
                    self.assertEqual(native.call_count, 1)

    def test_limit_subclasses_keep_class_identity_on_reconstruction_and_substitution(self):
        class TaggedLimit(Limit):
            pass

        x = self.x
        parameter = Symbol("a")
        self.assertNotEqual(TaggedLimit(x + parameter, x, 0, dir="-"),
                            Limit(x + parameter, x, 0, dir="-"))
        for owner in (Limit, TaggedLimit):
            with self.subTest(owner=owner):
                source = owner(x + parameter, x, 0, dir="-")
                self.assertIs(source.func, owner)
                self.assertIs(type(source.func(*source.args)), owner)
                self.assertTrue(source.has(parameter))
                changed = source.subs(parameter, 2)
                self.assertIs(type(changed), owner)
                self.assertEqual(changed.args, (x + 2, x, 0, Symbol("-")))


if __name__ == "__main__":
    unittest.main()
