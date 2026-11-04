"""Exact public monotonicity regressions; no sampling establishes a sign.

Strict directional predicates follow SymPy 1.14's pointwise derivative
predicate, including isolated zeroes and endpoint openness. The shell's
existing weak increasing-or-decreasing is_monotonic convention is retained
explicitly; this file does not certify the full upstream profile.
"""

import importlib
from math import isqrt
import unittest
from unittest.mock import patch

import sympy as s


class ExactMonotonicityTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("monotonicity_x")

    def test_negative_region_between_all_legacy_samples_is_detected(self):
        x = self.x
        # f' = (100*x - 25)**2 - 1 < 0 precisely on (0.24, 0.26).
        f = s.Rational(10000, 3)*x**3 - 2500*x**2 + 624*x
        domain = s.Interval(0, 1)
        self.assertIs(s.is_increasing(f, domain, x), False)
        self.assertIs(s.is_decreasing(-f, domain, x), False)
        self.assertIs(s.is_monotonic(f, domain, x), False)
        self.assertIs(s.is_strictly_increasing(f, s.Interval(0, s.Rational(1, 5)), x), True)

    def test_narrow_sign_change_below_binary64_resolution(self):
        x = self.x
        epsilon = s.Rational(1, 10**80)
        # f' = (x - 1)*(x - 1 - epsilon).
        f = x**3/3 - (2 + epsilon)*x**2/2 + (1 + epsilon)*x
        self.assertIs(s.is_increasing(f, s.Interval(0, 2), x), False)
        self.assertIs(s.is_decreasing(-f, s.Interval(0, 2), x), False)
        self.assertIs(s.is_increasing(f, s.Interval(0, 1), x), True)
        self.assertIs(s.is_decreasing(f, s.Interval(1, 1 + epsilon), x), True)

    def test_every_direction_on_constants_and_linear_functions(self):
        x = self.x
        for value in (0, 7, s.Rational(-2, 3)):
            with self.subTest(constant=value):
                self.assertIs(s.is_increasing(value), True)
                self.assertIs(s.is_decreasing(value), True)
                self.assertIs(s.is_strictly_increasing(value), False)
                self.assertIs(s.is_strictly_decreasing(value), False)
                self.assertIs(s.is_monotonic(value), True)
        self.assertIs(s.is_strictly_increasing(3*x - 1, x), True)
        self.assertIs(s.is_strictly_decreasing(-3*x + 1, x), True)
        self.assertIs(s.is_increasing(-x, x), False)
        self.assertIs(s.is_decreasing(x, x), False)

    def test_strict_predicate_keeps_included_isolated_derivative_zero(self):
        x = self.x
        self.assertIs(s.is_increasing(x**3, x), True)
        self.assertIs(s.is_decreasing(-x**3, x), True)
        self.assertIs(s.is_strictly_increasing(x**3, x), False)
        self.assertIs(s.is_strictly_decreasing(-x**3, x), False)
        # Deliberate preservation of the shell's existing weak convention.
        self.assertIs(s.is_monotonic(x**3, x), True)
        self.assertIs(s.is_monotonic(x**2, x), False)

    def test_endpoint_openness_changes_strict_predicates(self):
        x = self.x
        self.assertIs(s.is_strictly_increasing(x**2, s.Interval(0, 1), x), False)
        self.assertIs(s.is_strictly_increasing(x**2, s.Interval.Lopen(0, 1), x), True)
        self.assertIs(s.is_strictly_decreasing(-x**2, s.Interval(0, 1), x), False)
        self.assertIs(s.is_strictly_decreasing(-x**2, s.Interval.Lopen(0, 1), x), True)
        self.assertIs(s.is_decreasing(x**2, s.Interval(-1, 0), x), True)
        self.assertIs(s.is_strictly_decreasing(x**2, s.Interval(-1, 0), x), False)
        self.assertIs(s.is_strictly_decreasing(x**2, s.Interval.Ropen(-1, 0), x), True)

    def test_unbounded_intervals_keep_finite_zero_endpoints(self):
        x = self.x
        f = 4*x**3 - 6*x**2 - 72*x + 30
        self.assertIs(s.is_strictly_increasing(f, s.Interval.open(-s.oo, -2), x), True)
        self.assertIs(s.is_strictly_increasing(f, s.Interval(-s.oo, -2), x), False)
        self.assertIs(s.is_strictly_increasing(f, s.Interval.open(3, s.oo), x), True)
        self.assertIs(s.is_decreasing(f, s.Interval(-2, 3), x), True)
        self.assertIs(s.is_strictly_decreasing(f, s.Interval.open(-2, 3), x), True)

    def test_rational_poles_are_not_missing_samples(self):
        x = self.x
        self.assertIs(s.is_increasing(-1/x, s.S.Reals, x), False)
        self.assertIs(s.is_decreasing(1/x, s.S.Reals, x), False)
        self.assertIs(s.is_monotonic(1/x, s.S.Reals, x), False)
        self.assertIs(s.is_strictly_increasing(-1/x, s.Interval.Lopen(0, 1), x), True)
        self.assertIs(s.is_increasing(-1/x, s.Interval(0, 1), x), False)
        self.assertIs(s.is_strictly_decreasing(1/x, s.Interval.open(-s.oo, 0), x), True)
        f = 1/(x*x - 3*x)
        self.assertIs(s.is_strictly_decreasing(f, s.Interval.open(s.Rational(3, 2), 3), x), True)
        self.assertIs(s.is_strictly_decreasing(f, s.Interval.open(3, s.oo), x), True)

    def test_original_holes_survive_cancellation_and_zero_factors(self):
        x = self.x
        reciprocal = s.Pow(x, -1, evaluate=False)
        nested = s.Pow(reciprocal, -1, evaluate=False)
        held_one = s.Pow(reciprocal, 0, evaluate=False)
        held_zero = s.Mul(0, reciprocal, evaluate=False)
        hole_linear = s.Add(x, held_zero, evaluate=False)
        for expression in (nested, hole_linear):
            with self.subTest(expression=expression):
                self.assertIs(s.is_increasing(expression, s.Interval(-1, 1), x), False)
                self.assertIs(s.is_strictly_increasing(expression, s.Interval(1, 2), x), True)
        for expression in (held_one, held_zero):
            with self.subTest(expression=expression):
                self.assertIs(s.is_increasing(expression, s.Interval(-1, 1), x), False)
                self.assertIs(s.is_decreasing(expression, s.Interval(-1, 1), x), False)
                self.assertIs(s.is_increasing(expression, s.Interval(1, 2), x), True)
                self.assertIs(s.is_strictly_increasing(expression, s.Interval(1, 2), x), False)

    def test_nowhere_defined_held_source_is_not_a_constant(self):
        x = self.x
        undefined = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
        for predicate in (s.is_increasing, s.is_decreasing, s.is_monotonic,
                          s.is_strictly_increasing, s.is_strictly_decreasing):
            with self.subTest(predicate=predicate.__name__):
                self.assertIs(predicate(undefined, s.Interval(-1, 1), x), False)

    def test_empty_and_single_point_intervals(self):
        x = self.x
        for predicate in (s.is_increasing, s.is_decreasing,
                          s.is_strictly_increasing, s.is_strictly_decreasing):
            with self.subTest(predicate=predicate.__name__):
                self.assertIs(predicate(x*x, s.S.EmptySet, x), True)
        self.assertIs(s.is_increasing(x*x, s.Interval(0, 0), x), True)
        self.assertIs(s.is_strictly_increasing(x*x, s.Interval(0, 0), x), False)
        self.assertIs(s.is_strictly_increasing(x*x, s.Interval(1, 1), x), True)

    def test_quadratic_algebraic_boundaries_use_selected_real_embeddings(self):
        x = self.x
        t = s.Symbol("monotonicity_bound_t")
        negative = s.CRootOf(t*t - 2, t, 0)
        positive = s.CRootOf(t*t - 2, t, 1)
        f = x**3/3 - 2*x
        self.assertIs(s.is_decreasing(f, s.Interval(negative, positive), x), True)
        self.assertIs(s.is_strictly_decreasing(f, s.Interval.open(negative, positive), x), True)
        self.assertIs(s.is_strictly_decreasing(f, s.Interval(negative, positive), x), False)
        self.assertIs(s.is_strictly_increasing(f, s.Interval.open(positive, s.oo), x), True)

    def test_irreducible_algebraic_endpoint_requires_no_float_or_evalf(self):
        x = self.x
        t = s.Symbol("monotonicity_quintic_t")
        root = s.CRootOf(t**5 - t + 1, t, 0)
        f = x**6/6 - x*x/2 + x
        closed = s.Interval(root, s.oo)
        opened = s.Interval.open(root, s.oo)
        core = importlib.import_module("sympy.core")
        with patch.object(s.CRootOf, "__float__", side_effect=AssertionError("float was used")), \
             patch.object(s.CRootOf, "evalf", side_effect=AssertionError("evalf was used")), \
             patch.object(core, "_real_sign_of_difference", side_effect=AssertionError("generic ordering was used")):
            self.assertIs(s.is_increasing(f, closed, x), True)
            self.assertIs(s.is_strictly_increasing(f, closed, x), False)
            self.assertIs(s.is_strictly_increasing(f, opened, x), True)

    def test_algebraic_and_rational_bounds_separated_beyond_binary64(self):
        x = self.x
        t = s.Symbol("monotonicity_close_t")
        root = s.CRootOf(t*t - 2, t, 1)
        scale = 10**80
        n = isqrt(2*scale*scale)
        lower, upper = s.Rational(n, scale), s.Rational(n + 1, scale)
        f = x**3/3 - 2*x
        self.assertIs(s.is_increasing(f, s.Interval(lower, root), x), False)
        self.assertIs(s.is_decreasing(f, s.Interval(lower, root), x), True)
        self.assertIs(s.is_increasing(f, s.Interval(root, upper), x), True)
        self.assertIs(s.is_strictly_increasing(f, s.Interval.Lopen(root, upper), x), True)

    def test_dummy_and_assumption_bound_variable_identity(self):
        plain = s.Symbol("monotonicity_same_name")
        positive = s.Symbol("monotonicity_same_name", positive=True)
        dummy = s.Dummy("monotonicity_dummy")
        self.assertIs(s.is_increasing(dummy**3, s.S.Reals, dummy), True)
        with self.assertRaises(NotImplementedError):
            s.is_increasing(positive**2, s.Interval(1, 2), plain)
        with self.assertRaises(NotImplementedError):
            s.is_increasing(plain + positive, s.Interval(1, 2))

    def test_unsupported_sources_never_become_true_by_failed_evaluation(self):
        x = self.x
        y = s.Symbol("monotonicity_parameter")
        for expression in (s.sin(x), s.sqrt(x), s.Float("1.5")*x, x*y):
            with self.subTest(expression=expression):
                with self.assertRaises(NotImplementedError):
                    s.is_increasing(expression, s.Interval(1, 2), x)

    def test_custom_hooks_and_unsupported_zero_children_do_not_enter_native_lane(self):
        x = self.x
        core = importlib.import_module("sympy.core")

        class UserFunction(s.Function):
            def _eval_derivative(self, variable):
                raise AssertionError("custom derivative hook must not establish monotonicity")

        source = UserFunction(x, evaluate=False)
        expressions = (source, s.Mul(0, source, evaluate=False), s.Mul(0, s.sin(x), evaluate=False))
        with patch.object(core._native, "solve_real_rational_inequality_native",
                          side_effect=AssertionError("unadmitted source reached native arithmetic")):
            for expression in expressions:
                with self.subTest(expression=expression):
                    with self.assertRaises(NotImplementedError):
                        s.is_increasing(expression, s.Interval(1, 2), x)

    def test_unsupported_domains_refuse_instead_of_sampling(self):
        x = self.x
        t = s.Symbol("monotonicity_symbolic_bound")
        domains = (s.Interval(t, 2), s.Interval(0, s.sqrt(2)),
                   s.Interval(s.Float("0.5"), 1), s.FiniteSet(0, 1),
                   s.Union(s.Interval(-2, -1), s.Interval(1, 2)))
        for domain in domains:
            with self.subTest(domain=domain):
                with self.assertRaises(NotImplementedError):
                    s.is_increasing(x, domain, x)
        with self.assertRaises(TypeError):
            s.is_increasing(x, [0, 1], x)

    def test_original_admission_failure_is_explicit_and_precedes_derivative(self):
        x = self.x
        core = importlib.import_module("sympy.core")
        with patch.object(core._native, "solve_real_rational_inequality_native",
                          side_effect=ValueError("forced budget exhaustion")) as solve, \
             patch.object(core, "diff_with_receipt",
                          side_effect=AssertionError("derivative ran after failed admission")):
            with self.assertRaisesRegex(NotImplementedError, "budget exhaustion"):
                s.is_increasing(x*x, s.Interval(0, 1), x)
        self.assertEqual(solve.call_count, 1)

    def test_derivative_verifier_refusal_is_not_swallowed(self):
        x = self.x
        core = importlib.import_module("sympy.core")
        with patch.object(core, "diff_with_receipt", side_effect=ValueError("derivation refused")):
            with self.assertRaisesRegex(NotImplementedError, "derivation refused"):
                s.is_increasing(x*x, s.Interval(0, 1), x)

    def test_exact_endpoint_comparison_refusal_stays_inconclusive(self):
        x = self.x
        t = s.Symbol("monotonicity_refused_root")
        root = s.CRootOf(t*t - 2, t, 1)
        domain = s.Interval(root, 2)
        roots = importlib.import_module("sympy.polys.rootoftools")
        with patch.object(roots, "_algebraic_comparison", side_effect=ValueError("comparison budget exhausted")):
            with self.assertRaisesRegex(NotImplementedError, "comparison budget exhausted"):
                s.is_increasing(x*x, domain, x)


if __name__ == "__main__":
    unittest.main()
