"""Public exact rational range regressions.

Expected images use exact set equality, with explicit exact algebraic
comparison for expressions in indexed roots. No float/decimal tolerance is
an admissible comparator. Original held-domain cases intentionally exercise
the project's domain-preserving contract, not a full SymPy profile claim.
"""

import importlib
import unittest
from unittest.mock import patch

import sympy as s


class ExactFunctionRangeTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("range_x")

    def test_unbounded_linear_range_and_extrema(self):
        x = self.x
        self.assertEqual(s.function_range(3*x - 7, x, s.S.Reals), s.S.Reals)
        self.assertEqual(s.minimum(3*x - 7, x), -s.oo)
        self.assertEqual(s.maximum(3*x - 7, x), s.oo)

    def test_global_quadratic_ranges_keep_infinity(self):
        x = self.x
        self.assertEqual(s.function_range((x - 2)**2 + 5, x, s.S.Reals),
                         s.Interval(5, s.oo))
        self.assertEqual(s.minimum((x - 2)**2 + 5, x), 5)
        self.assertEqual(s.maximum((x - 2)**2 + 5, x), s.oo)
        self.assertEqual(s.function_range(5 - (x - 2)**2, x, s.S.Reals),
                         s.Interval(-s.oo, 5))
        self.assertEqual(s.minimum(5 - (x - 2)**2, x), -s.oo)
        self.assertEqual(s.maximum(5 - (x - 2)**2, x), 5)

    def test_constants_on_open_intervals_have_singleton_images(self):
        x = self.x
        for value in (0, 7, s.Rational(-2, 3)):
            with self.subTest(value=value):
                domain = s.Interval.open(-2, 3)
                self.assertEqual(s.function_range(value, x, domain), s.FiniteSet(value))
                self.assertEqual(s.minimum(value, x, domain), value)
                self.assertEqual(s.maximum(value, x, domain), value)

    def test_open_and_closed_bounds_survive_increasing_images(self):
        x = self.x
        cases = (
            (s.Interval(0, 2), s.Interval(0, 4)),
            (s.Interval.Ropen(0, 2), s.Interval.Ropen(0, 4)),
            (s.Interval.Lopen(0, 2), s.Interval.Lopen(0, 4)),
            (s.Interval.open(0, 2), s.Interval.open(0, 4)),
        )
        for domain, expected in cases:
            with self.subTest(domain=domain):
                self.assertEqual(s.function_range(x*x, x, domain), expected)

    def test_decreasing_image_reverses_endpoint_openness(self):
        x = self.x
        self.assertEqual(s.function_range(-x, x, s.Interval.Lopen(0, 3)),
                         s.Interval.Ropen(-3, 0))
        self.assertEqual(s.function_range(-x, x, s.Interval.Ropen(0, 3)),
                         s.Interval.Lopen(-3, 0))

    def test_interior_extremum_is_attained_on_an_open_domain(self):
        x = self.x
        self.assertEqual(s.function_range(x*x, x, s.Interval.open(-1, 1)),
                         s.Interval.Ropen(0, 1))
        self.assertEqual(s.minimum(x*x, x, s.Interval.open(-1, 1)), 0)
        # The public maximum API returns a supremum, as upstream does.
        self.assertEqual(s.maximum(x*x, x, s.Interval.open(-1, 1)), 1)

    def test_open_limit_equal_to_an_attained_critical_value_is_closed(self):
        x = self.x
        # Zero is excluded at x=0 but attained at x=1. Losing that fact would
        # return a false open lower image endpoint.
        expression = (x*(x - 1))**2
        self.assertEqual(s.function_range(expression, x, s.Interval.open(0, 2)),
                         s.Interval.Ropen(0, 4))

    def test_simple_pole_splits_both_domain_and_image(self):
        x = self.x
        expected = s.Union(s.Interval.open(-s.oo, 0), s.Interval.open(0, s.oo))
        self.assertEqual(s.continuous_domain(1/x, x, s.S.Reals), expected)
        self.assertEqual(s.function_range(1/x, x, s.S.Reals), expected)
        self.assertEqual(s.minimum(1/x, x), -s.oo)
        self.assertEqual(s.maximum(1/x, x), s.oo)

    def test_even_pole_merges_overlapping_positive_images(self):
        x = self.x
        self.assertEqual(s.function_range(1/x**2, x, s.S.Reals),
                         s.Interval.open(0, s.oo))
        self.assertEqual(s.minimum(1/x**2, x), 0)

    def test_odd_pole_direction_and_finite_endpoints(self):
        x = self.x
        expected = s.Union(s.Interval(-s.oo, -1), s.Interval(1, s.oo))
        self.assertEqual(s.function_range(1/(x - 1)**3, x, s.Interval(0, 2)),
                         expected)
        self.assertEqual(s.function_range(1/(x - 1)**2, x, s.Interval(0, 2)),
                         s.Interval(1, s.oo))

    def test_horizontal_asymptote_is_not_an_attained_endpoint(self):
        x = self.x
        self.assertEqual(s.function_range(x/(x + 1), x, s.Interval.open(0, s.oo)),
                         s.Interval.open(0, 1))
        self.assertEqual(s.maximum(x/(x + 1), x, s.Interval.open(0, s.oo)), 1)
        self.assertEqual(s.function_range(x/(x + 1), x, s.S.Reals),
                         s.Union(s.Interval.open(-s.oo, 1), s.Interval.open(1, s.oo)))

    def test_multiple_poles_and_an_interior_stationary_value(self):
        x = self.x
        expression = (x*x - 1)/(x*x - 4)
        expected = s.Union(s.Interval(-s.oo, s.Rational(1, 4)),
                           s.Interval.open(1, s.oo))
        self.assertEqual(s.function_range(expression, x, s.S.Reals), expected)

    def test_removable_hole_is_not_filled_by_cancellation(self):
        x = self.x
        source = s.Mul(x*x - 1, s.Pow(x - 1, -1, evaluate=False), evaluate=False)
        domain = s.Interval(0, 2)
        self.assertEqual(s.continuous_domain(source, x, domain),
                         s.Union(s.Interval.Ropen(0, 1), s.Interval.Lopen(1, 2)))
        self.assertEqual(s.function_range(source, x, domain),
                         s.Union(s.Interval.Ropen(1, 2), s.Interval.Lopen(2, 3)))

    def test_nested_reciprocal_preserves_original_hole(self):
        x = self.x
        source = s.Pow(s.Pow(x, -1, evaluate=False), -1, evaluate=False)
        self.assertEqual(s.function_range(source, x, s.Interval(-1, 1)),
                         s.Union(s.Interval.Ropen(-1, 0), s.Interval.Lopen(0, 1)))

    def test_held_zero_product_and_zeroth_power_keep_their_domains(self):
        x = self.x
        reciprocal = s.Pow(x, -1, evaluate=False)
        domain = s.Interval(-1, 1)
        expected_domain = s.Union(s.Interval.Ropen(-1, 0), s.Interval.Lopen(0, 1))
        for source, value in (
            (s.Mul(0, reciprocal, evaluate=False), 0),
            (s.Pow(reciprocal, 0, evaluate=False), 1),
        ):
            with self.subTest(value=value):
                self.assertEqual(s.continuous_domain(source, x, domain), expected_domain)
                self.assertEqual(s.function_range(source, x, domain), s.FiniteSet(value))
                self.assertEqual(s.function_range(source, x, s.Interval(0, 0)), s.S.EmptySet)

    def test_nowhere_defined_source_has_no_fabricated_constant_range(self):
        x = self.x
        source = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
        self.assertEqual(s.continuous_domain(source, x, s.S.Reals), s.S.EmptySet)
        self.assertEqual(s.function_range(source, x, s.S.Reals), s.S.EmptySet)
        with self.assertRaisesRegex(ValueError, "empty domain"):
            s.maximum(source, x)

    def test_empty_and_singleton_domains(self):
        x = self.x
        self.assertEqual(s.function_range(x*x + 1, x, s.S.EmptySet), s.S.EmptySet)
        self.assertEqual(s.continuous_domain(x*x + 1, x, s.S.EmptySet), s.S.EmptySet)
        self.assertEqual(s.function_range(x*x + 1, x, s.Interval(2, 2)), s.FiniteSet(5))
        for operation in (s.minimum, s.maximum):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(ValueError, "empty domain"):
                    operation(x, x, s.S.EmptySet)

    def test_algebraic_stationary_points_can_give_exact_rational_extrema(self):
        x = self.x
        source = x**4 - 3*x*x
        self.assertEqual(s.minimum(source, x), s.Rational(-9, 4))
        self.assertEqual(s.function_range(source, x, s.S.Reals),
                         s.Interval(s.Rational(-9, 4), s.oo))

    def test_irrational_stationary_value_uses_exact_algebraic_order(self):
        x = self.x
        t = s.Symbol("range_cubic_bound")
        root = s.CRootOf(3*t*t - 2, t, 1)
        from sympy.polys.rootoftools import _algebraic_comparison

        result = s.function_range(x**3 - 2*x, x, s.Interval(0, 2))
        self.assertEqual(_algebraic_comparison(result.start, -s.Rational(4, 3)*root), 0)
        self.assertEqual(result.end, 4)
        self.assertFalse(result.left_open)
        self.assertFalse(result.right_open)

    def test_irreducible_critical_root_never_uses_float_or_evalf(self):
        x = self.x
        t = s.Symbol("range_quintic_bound")
        root = s.CRootOf(t**5 - t + 1, t, 0)
        source = x**6/6 - x*x/2 + x
        core = importlib.import_module("sympy.core")
        from sympy.polys.rootoftools import _algebraic_comparison

        with patch.object(s.CRootOf, "__float__", side_effect=AssertionError("float was used")), \
             patch.object(s.CRootOf, "evalf", side_effect=AssertionError("evalf was used")), \
             patch.object(core, "_real_sign_of_difference",
                          side_effect=AssertionError("generic ordering was used")):
            result = s.minimum(source, x)
            self.assertEqual(_algebraic_comparison(result, -root*root/3 + 5*root/6), 0)

    def test_real_algebraic_input_endpoint_keeps_its_selected_embedding(self):
        x = self.x
        t = s.Symbol("range_endpoint")
        root = s.CRootOf(t*t - 2, t, 1)
        result = s.function_range(x, x, s.Interval(root, 2))
        from sympy.polys.rootoftools import _algebraic_comparison

        self.assertEqual(_algebraic_comparison(result.start, root), 0)
        self.assertEqual(result.end, 2)
        self.assertFalse(result.left_open)

    def test_one_sided_algebraic_poles(self):
        x = self.x
        t = s.Symbol("range_pole")
        root = s.CRootOf(t*t - 2, t, 1)
        self.assertEqual(s.function_range(1/(x*x - 2), x, s.Interval.Lopen(root, 2)),
                         s.Interval(s.Rational(1, 2), s.oo))
        self.assertEqual(s.function_range(1/(x*x - 2), x, s.Interval.Ropen(0, root)),
                         s.Interval(-s.oo, s.Rational(-1, 2)))

    def test_removable_algebraic_hole_has_a_finite_open_limit(self):
        x = self.x
        t = s.Symbol("range_removable")
        root = s.CRootOf(t*t - 2, t, 1)
        source = s.Mul(x**4 - 4, s.Pow(x*x - 2, -1, evaluate=False), evaluate=False)
        self.assertEqual(s.function_range(source, x, s.Interval(root, 2)),
                         s.Interval.Lopen(4, 6))

    def test_bounds_below_binary64_resolution_remain_distinct(self):
        x = self.x
        epsilon = s.Rational(1, 10**80)
        source = (x - 1)**2 + epsilon
        domain = s.Interval.open(1, 1 + epsilon)
        self.assertEqual(s.function_range(source, x, domain),
                         s.Interval.open(epsilon, epsilon + epsilon**2))

    def test_typed_symbols_do_not_alias(self):
        dummy = s.Dummy("range_dummy")
        self.assertEqual(s.minimum(dummy**2, dummy), 0)
        plain = s.Symbol("range_same")
        positive = s.Symbol("range_same", positive=True)
        with self.assertRaises(NotImplementedError):
            s.function_range(positive**2, plain, s.Interval(1, 2))

    def test_unsupported_functions_inexact_inputs_and_parameters_refuse(self):
        x = self.x
        y = s.Symbol("range_parameter")
        for source in (s.sin(x), s.sqrt(x), s.Float("1.5")*x, x*y):
            with self.subTest(source=source):
                with self.assertRaises(NotImplementedError):
                    s.function_range(source, x, s.Interval(1, 2))
                with self.assertRaises(NotImplementedError):
                    s.continuous_domain(source, x, s.Interval(1, 2))

    def test_custom_children_refuse_before_native_check_or_callback(self):
        x = self.x
        core = importlib.import_module("sympy.core")

        class UserFunction(s.Function):
            def _eval_derivative(self, variable):
                raise AssertionError("custom derivative hook ran")

            def evalf(self, *args, **kwargs):
                raise AssertionError("custom evaluation ran")

        child = UserFunction(x, evaluate=False)
        with patch.object(core._native, "solve_real_rational_inequality_native",
                          side_effect=AssertionError("unadmitted source reached native checks")):
            for source in (child, s.Mul(0, child, evaluate=False)):
                with self.subTest(source=source):
                    with self.assertRaises(NotImplementedError):
                        s.function_range(source, x, s.Interval(1, 2))

    def test_unsupported_domains_refuse(self):
        x = self.x
        bound = s.Symbol("range_symbolic_bound")
        domains = (s.Interval(bound, 2), s.Interval(0, s.sqrt(2)),
                   s.Interval(s.Float("0.5"), 1), s.FiniteSet(0, 1),
                   s.Union(s.Interval(-2, -1), s.Interval(1, 2)))
        for domain in domains:
            with self.subTest(domain=domain):
                with self.assertRaises(NotImplementedError):
                    s.function_range(x, x, domain)
        with self.assertRaises(TypeError):
            s.function_range(x, x, [0, 1])

    def test_original_domain_failure_precedes_derivative_work(self):
        x = self.x
        core = importlib.import_module("sympy.core")
        with patch.object(core._native, "solve_real_rational_inequality_native",
                          side_effect=ValueError("forced budget exhaustion")) as checked, \
             patch.object(core, "diff_with_receipt",
                          side_effect=AssertionError("derivative ran after failed admission")):
            with self.assertRaisesRegex(NotImplementedError, "budget exhaustion"):
                s.function_range(x*x, x, s.Interval(0, 1))
        self.assertEqual(checked.call_count, 1)

    def test_derivative_checker_refusal_is_not_swallowed(self):
        x = self.x
        core = importlib.import_module("sympy.core")
        with patch.object(core, "diff_with_receipt", side_effect=ValueError("derivation refused")):
            with self.assertRaisesRegex(NotImplementedError, "derivation refused"):
                s.function_range(x*x, x, s.Interval(0, 1))

    def test_algebraic_comparison_exhaustion_stays_inconclusive(self):
        x = self.x
        t = s.Symbol("range_refused_endpoint")
        root = s.CRootOf(t*t - 2, t, 1)
        domain = s.Interval(root, 2)
        roots = importlib.import_module("sympy.polys.rootoftools")
        with patch.object(roots, "_algebraic_comparison",
                          side_effect=ValueError("comparison budget exhausted")):
            with self.assertRaisesRegex(NotImplementedError, "comparison budget exhausted"):
                s.function_range(x, x, domain)

    def test_changed_algebraic_endpoint_property_refuses_without_callback(self):
        x = self.x
        t = s.Symbol("range_changed_root")
        root = s.CRootOf(t*t - 2, t, 1)
        domain = s.Interval(root, 2)

        def forbidden(_):
            raise AssertionError("modified root property was invoked")

        with patch.object(s.CRootOf, "is_real", property(forbidden)):
            with self.assertRaises(NotImplementedError):
                s.function_range(x, x, domain)

    def test_changed_builtin_args_refuse_without_callback(self):
        x = self.x
        sources = (
            (s.Add, s.Add(x, 1, evaluate=False)),
            (s.Mul, s.Mul(2, x, evaluate=False)),
            (s.Pow, s.Pow(x, 2, evaluate=False)),
        )
        domain = s.Interval(0, 1)

        def forbidden(_):
            raise AssertionError("modified arithmetic args were invoked")

        for kind, source in sources:
            with self.subTest(kind=kind.__name__):
                with patch.object(kind, "args", property(forbidden)):
                    with self.assertRaises(NotImplementedError):
                        s.function_range(source, x, domain)

    def test_changed_numeric_accessors_refuse_without_callback(self):
        x = self.x
        source = s.Add(s.Mul(2, x, evaluate=False), s.Rational(1, 3), evaluate=False)
        domain = s.Interval(0, 1)

        def forbidden(_):
            raise AssertionError("modified exact coefficient accessor was invoked")

        for kind, name in ((s.Integer, "p"), (s.Rational, "p"),
                           (s.Rational, "q"), (s.Integer, "__int__")):
            with self.subTest(kind=kind.__name__, attribute=name):
                replacement = forbidden if name == "__int__" else property(forbidden)
                with patch.object(kind, name, replacement):
                    with self.assertRaises(NotImplementedError):
                        s.function_range(source, x, domain)

    def test_forged_integer_coefficient_refuses_before_native_admission(self):
        x = self.x
        source = s.Mul(2, x, evaluate=False)
        domain = s.Interval(0, 1)
        observed = []

        def forged(_):
            observed.append("forged coefficient read")
            return 101

        core = importlib.import_module("sympy.core")
        with patch.object(s.Integer, "p", property(forged)), \
             patch.object(core._native, "solve_real_rational_inequality_native",
                          side_effect=AssertionError("changed surface reached native checks")) as checked:
            with self.assertRaises(NotImplementedError):
                s.function_range(source, x, domain)
        self.assertEqual(observed, [])
        self.assertEqual(checked.call_count, 0)

    def test_changed_symbol_hash_refuses_before_source_collection(self):
        x = self.x
        source = x*x
        domain = s.Interval(0, 1)

        def forbidden(_):
            raise AssertionError("modified symbol hash was invoked")

        with patch.object(s.Symbol, "__hash__", forbidden):
            with self.assertRaises(NotImplementedError):
                s.function_range(source, x, domain)

    def test_changed_integer_printer_cannot_turn_finite_values_into_infinity(self):
        x = self.x
        domain = s.Interval(0, 2)
        observed = []

        def forged(_):
            observed.append("forged printer read")
            return "oo"

        with patch.object(s.Integer, "__str__", forged):
            with self.assertRaises(NotImplementedError):
                s.function_range(x, x, domain)
        self.assertEqual(observed, [])

    def test_changed_arithmetic_division_refuses_without_callback(self):
        x = self.x
        source = x/(x + 1)
        domain = s.Interval(0, 1)

        def forbidden(*_):
            raise AssertionError("modified division was invoked")

        with patch.object(s.Expr, "__truediv__", forbidden):
            with self.assertRaises(NotImplementedError):
                s.function_range(source, x, domain)

    def test_algebraic_result_bounds_do_not_consult_the_root_printer(self):
        x = self.x
        t = s.Symbol("range_unprinted_root")
        root = s.CRootOf(t*t - 2, t, 1)
        domain = s.Interval(0, root)
        from sympy.polys.rootoftools import _algebraic_comparison

        def forbidden(_):
            raise AssertionError("root printer was used to identify an interval bound")

        with patch.object(s.CRootOf, "__str__", forbidden):
            result = s.function_range(x, x, domain)
            self.assertEqual(result.start, 0)
            self.assertEqual(_algebraic_comparison(result.end, root), 0)
            self.assertFalse(result.right_open)

    def test_source_growth_limit_has_no_sample_fallback(self):
        x = self.x
        source = s.Pow(x + 1, 10000, evaluate=False)
        with self.assertRaises(NotImplementedError):
            s.function_range(source, x, s.Interval(0, 1))


if __name__ == "__main__":
    unittest.main()
