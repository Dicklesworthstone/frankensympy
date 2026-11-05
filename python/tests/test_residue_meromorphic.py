"""Public residues require an admitted meromorphic germ and enough precision."""

import unittest
from unittest.mock import patch

import sympy as s
from sympy.core import _native, _native_expr


class MeromorphicResidueTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("x")

    def assert_exact(self, actual, expected):
        self.assertEqual(s.simplify(s.expand(actual - expected)), 0)

    def test_existing_rational_complex_and_elementary_residues_remain_supported(self):
        x = self.x
        cases = (
            (1/x, 0, 1),
            (1/x**2, 0, 0),
            (1/(x - 1), 1, 1),
            (1/(x*x - 1), 1, s.Rational(1, 2)),
            (1/(x*x - 1), -1, s.Rational(-1, 2)),
            (1/(x*x*(x + 1)), 0, -1),
            (1/(x*x*(x + 1)), -1, 1),
            (x/(x*x + 1), s.I, s.Rational(1, 2)),
            (s.sin(x)/x**2, 0, 1),
            (s.cos(x)/x, 0, 1),
        )
        for expression, point, expected in cases:
            with self.subTest(expression=expression, point=point):
                self.assert_exact(s.residue(expression, x, point), expected)

    def test_high_order_analytic_poles_keep_coefficients_beyond_the_old_limit_loop(self):
        x = self.x
        cases = (
            (s.exp(x)/x**9, s.Rational(1, 40320)),
            (s.sin(x)/x**10, s.Rational(1, 362880)),
            (s.cos(x)/x**9, s.Rational(1, 40320)),
            (s.sinh(x)/x**10, s.Rational(1, 362880)),
            (s.cosh(x)/x**9, s.Rational(1, 40320)),
            (s.exp(x)*s.sin(x)/x**4, s.Rational(1, 3)),
            ((s.exp(x) - 1 - x - x*x/2)/x**4, s.Rational(1, 6)),
            (1/s.sin(x)**7, s.Rational(5, 16)),
            (1/(s.exp(x) - 1)**8, -1),
        )
        for expression, expected in cases:
            with self.subTest(expression=expression):
                self.assert_exact(s.residue(expression, x, 0), expected)

    def test_high_order_rational_poles_use_complete_laurent_coefficients(self):
        x = self.x
        self.assertEqual(s.residue(1/(x**12*(1 - x)), x, 0), 1)
        self.assertEqual(s.residue(1/(x**12*(1 + x)), x, 0), -1)
        point = s.Rational(-2, 3)
        local = x - point
        self.assertEqual(s.residue((3 + 2*local + 5*local**11)/local**12, x, point), 5)

    def test_gaussian_rational_centers_preserve_complex_local_coordinates(self):
        x = self.x
        for point in (s.I, -s.I, s.Rational(1, 2) + 3*s.I/4):
            with self.subTest(point=point):
                local = x - point
                self.assert_exact(s.residue(s.exp(local)/local**7, x, point), s.Rational(1, 720))
                self.assert_exact(s.residue((x + 1)**2/local**3, x, point), 1)

    def test_meromorphic_inverses_establish_a_nonzero_local_leading_coefficient(self):
        x = self.x
        cases = (
            (1/s.sin(x), 1),
            (1/(s.exp(x) - 1), 1),
            (x/(1 - s.cos(x)), 2),
            (1/(1 - s.cos(x)), 0),
        )
        for expression, expected in cases:
            with self.subTest(expression=expression):
                self.assert_exact(s.residue(expression, x, 0), expected)

    def test_entire_arguments_are_checked_at_the_actual_center(self):
        x = self.x
        self.assert_exact(s.residue(s.exp(1/(1 + x))/x**2, x, 0), -s.E)
        self.assert_exact(s.residue(s.exp(1/x)/(x - 1), x, 1), s.E)
        with self.assertRaises(NotImplementedError):
            s.residue(s.exp(1/x), x, 0)
        with self.assertRaises(NotImplementedError):
            s.residue(s.exp(1/(1 + x)), x, -1)

    def test_cancellation_and_removable_rational_holes_can_have_zero_residue(self):
        x = self.x
        reciprocal = s.Pow(x, -1, evaluate=False)
        cases = (
            s.Add(reciprocal, s.Mul(-1, reciprocal, evaluate=False), evaluate=False),
            s.Mul(x, reciprocal, evaluate=False),
            s.sin(x)/x**2 - 1/x,
            s.sin(x),
            s.exp(x),
            s.Integer(0),
        )
        for expression in cases:
            with self.subTest(expression=expression):
                self.assertEqual(s.residue(expression, x, 0), 0)

    def test_typed_variables_and_same_named_parameters_never_rebind(self):
        for variable in (s.Symbol("x", positive=True), s.Dummy("x")):
            parameter = s.Symbol("x", real=True)
            with self.subTest(variable=variable):
                expression = parameter/variable + s.exp(variable)/variable**2
                result = s.residue(expression, variable, 0)
                self.assert_exact(result, parameter + 1)
                self.assertEqual(result.free_symbols, {parameter})
                self.assertNotIn(variable, result.free_symbols)
        variable = s.Symbol("_fsym_series_t")
        parameter = s.Symbol("_fsym_log_t")
        result = s.residue(parameter/variable + s.sin(variable)/variable**2, variable, 0)
        self.assert_exact(result, parameter + 1)
        self.assertEqual(result.free_symbols, {parameter})

    def test_essential_singularities_never_turn_into_default_zero(self):
        x = self.x
        sources = (
            s.exp(1/x), s.exp(-1/x**2), s.sin(1/x), s.cos(1/x),
            s.sinh(1/x), s.cosh(1/x),
            s.Mul(0, s.exp(1/x), evaluate=False),
        )
        for expression in sources:
            with self.subTest(expression=expression):
                with self.assertRaises(NotImplementedError):
                    s.residue(expression, x, 0)

    def test_branch_and_nonholomorphic_sources_refuse_before_native_series(self):
        x = self.x
        sources = (
            s.log(x)/x, s.sqrt(x)/x, s.Abs(x)/x, s.conjugate(x)/x,
            s.Mul(0, s.log(x), evaluate=False),
        )
        with patch.object(
            _native, "series_expansion_native",
            side_effect=AssertionError("nonmeromorphic source reached Laurent expansion"),
        ) as native:
            for expression in sources:
                with self.subTest(expression=expression):
                    with self.assertRaises(NotImplementedError):
                        s.residue(expression, x, 0)
            self.assertEqual(native.call_count, 0)

    def test_unknown_denominator_coefficients_are_not_assumed_nonzero(self):
        x, a = self.x, s.Symbol("a")
        for expression in (
            1/(a*x), s.Mul(0, s.Pow(a*x, -1, evaluate=False), evaluate=False),
        ):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(NotImplementedError, "leading coefficient nonzero"):
                    s.residue(expression, x, 0)
        self.assertEqual(s.residue(a/x, x, 0), a)

    def test_original_undefined_children_survive_zero_and_cancellation_shortcuts(self):
        x = self.x
        for base in (s.Integer(0), s.Add(x, -x, evaluate=False)):
            for exponent in (0, -1):
                undefined = s.Pow(base, exponent, evaluate=False)
                sources = (
                    s.Mul(0, undefined, evaluate=False),
                    s.Add(undefined, s.Mul(-1, undefined, evaluate=False), evaluate=False),
                )
                for expression in sources:
                    with self.subTest(base=base, exponent=exponent, expression=expression):
                        with self.assertRaises(ValueError):
                            s.residue(expression, x, 0)

    def test_custom_functions_and_coordinate_printers_do_not_execute(self):
        observed = []

        class Custom(s.Function):
            @classmethod
            def eval(cls, argument):
                observed.append(argument)
                return None

        expression = Custom(self.x, evaluate=False)/self.x
        with self.assertRaises(NotImplementedError):
            s.residue(expression, self.x, 0)

        class Coordinate:
            def __str__(self):
                observed.append("coordinate printer")
                return "x"

        with self.assertRaises(TypeError):
            s.residue(1/self.x, Coordinate(), 0)
        self.assertEqual(observed, [])

    def test_inexact_centers_and_unbounded_original_powers_refuse(self):
        x = self.x
        for point in (s.Float(0), s.Symbol("a"), s.oo):
            with self.subTest(point=point):
                with self.assertRaises(NotImplementedError):
                    s.residue(1/x, x, point)
        with self.assertRaisesRegex(ValueError, "integer power"):
            s.residue(s.Pow(x, -129, evaluate=False), x, 0)
        with self.assertRaisesRegex(ValueError, "4096"):
            s.residue(s.Integer(1 << 4096)/x, x, 0)

    def test_exhausted_precision_cannot_masquerade_as_a_zero_residue(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            s.residue(s.exp(self.x)/self.x**70, self.x, 0)
        with patch.object(
            _native, "series_expansion_native", return_value=([], _native_expr(-1)),
        ):
            with self.assertRaisesRegex(ValueError, "precision"):
                s.residue(1, self.x, 0)

    def test_native_operational_errors_propagate_unchanged_without_heuristic_retry(self):
        for expression in (s.exp(self.x), 1/(s.exp(self.x) - 1)):
            for failure in (ValueError("native series cancelled"), RuntimeError("native series unavailable")):
                with self.subTest(expression=expression, failure=type(failure)), patch.object(
                    _native, "series_expansion_native", side_effect=failure,
                ) as native:
                    with self.assertRaises(type(failure)) as error:
                        s.residue(expression, self.x, 0)
                    self.assertIs(error.exception, failure)
                    self.assertEqual(native.call_count, 1)


if __name__ == "__main__":
    unittest.main()
