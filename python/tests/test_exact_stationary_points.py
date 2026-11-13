"""Complete rational stationary sets retain the original function's domain.

Ordinary results use exact set equality and pinned SymPy 1.14.0 expectations.
Held-domain adversaries also exercise the project's source-preservation
contract. Approximate root comparisons cannot establish stationary points.
"""

import importlib
import unittest
from unittest.mock import patch

import sympy as s


class ExactStationaryPointsTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("stationary_x")

    def test_default_domain_excludes_nonreal_derivative_roots(self):
        x = self.x
        self.assertEqual(s.stationary_points(x**3 + x, x), s.S.EmptySet)
        self.assertEqual(s.stationary_points(x**4 + 2*x*x, x), s.FiniteSet(0))

    def test_constants_are_stationary_everywhere_in_the_requested_domain(self):
        x = self.x
        for constant in (0, 7, s.Rational(-2, 3)):
            with self.subTest(constant=constant):
                self.assertEqual(s.stationary_points(constant, x), s.S.Reals)
                self.assertEqual(s.stationary_points(constant), s.S.Reals)
                domain = s.Interval.Lopen(-2, 3)
                self.assertEqual(s.stationary_points(constant, x, domain), domain)

    def test_repeated_derivative_roots_and_inflections_are_not_lost(self):
        x = self.x
        self.assertEqual(s.stationary_points(x**3, x), s.FiniteSet(0))
        self.assertEqual(s.stationary_points((x - 2)**6, x), s.FiniteSet(2))
        self.assertEqual(s.stationary_points(x**4/4 - x*x/2, x), s.FiniteSet(-1, 0, 1))

    def test_interval_clipping_preserves_endpoint_openness(self):
        x = self.x
        source = x**3/3 - x
        cases = (
            (s.Interval(-1, 1), s.FiniteSet(-1, 1)),
            (s.Interval.open(-1, 1), s.S.EmptySet),
            (s.Interval.Lopen(-1, 1), s.FiniteSet(1)),
            (s.Interval.Ropen(-1, 1), s.FiniteSet(-1)),
            (s.Interval(-s.oo, -1), s.FiniteSet(-1)),
            (s.Interval.open(1, s.oo), s.S.EmptySet),
        )
        for domain, expected in cases:
            with self.subTest(domain=domain):
                self.assertEqual(s.stationary_points(source, x, domain), expected)

    def test_rational_poles_are_excluded_from_complete_stationary_set(self):
        x = self.x
        self.assertEqual(s.stationary_points(x + 1/x, x), s.FiniteSet(-1, 1))
        self.assertEqual(s.stationary_points(1/(x*x - 1), x), s.FiniteSet(0))
        self.assertEqual(s.stationary_points(1/x, x), s.S.EmptySet)

    def test_cancelled_hole_cannot_become_a_stationary_point(self):
        x = self.x
        source = s.Mul((x - 1)**3, s.Pow(x - 1, -1, evaluate=False), evaluate=False)
        self.assertEqual(s.stationary_points(source, x), s.S.EmptySet)
        reciprocal = s.Pow(x, -1, evaluate=False)
        nested = s.Pow(reciprocal, -1, evaluate=False)
        source = s.Pow(nested, 2, evaluate=False)
        self.assertEqual(s.stationary_points(source, x), s.S.EmptySet)

    def test_zero_products_and_zeroth_powers_retain_original_holes(self):
        x = self.x
        reciprocal = s.Pow(x, -1, evaluate=False)
        expected = s.Union(s.Interval.open(-s.oo, 0), s.Interval.open(0, s.oo))
        for source in (s.Mul(0, reciprocal, evaluate=False),
                       s.Pow(reciprocal, 0, evaluate=False),
                       s.Mul(x, reciprocal, evaluate=False)):
            with self.subTest(source=source):
                self.assertEqual(s.stationary_points(source, x), expected)
                self.assertEqual(s.stationary_points(source, x, s.Interval(0, 2)),
                                 s.Interval.Lopen(0, 2))

    def test_nowhere_defined_source_returns_empty_without_differentiation(self):
        x = self.x
        source = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
        core = importlib.import_module("sympy.core")
        with patch.object(core, "diff_with_receipt",
                          side_effect=AssertionError("undefined source was differentiated")):
            self.assertEqual(s.stationary_points(source, x), s.S.EmptySet)

    def test_empty_domain_needs_no_derivative(self):
        core = importlib.import_module("sympy.core")
        with patch.object(core, "diff_with_receipt",
                          side_effect=AssertionError("empty source domain was differentiated")):
            self.assertEqual(s.stationary_points(self.x**2, self.x, s.S.EmptySet),
                             s.S.EmptySet)

    def test_algebraic_stationary_point_has_exact_selected_embedding(self):
        x = self.x
        root = s.CRootOf(x**5 - x - 1, x, 0)
        source = x**6/6 - x*x/2 - x
        result = s.stationary_points(source, x)
        from sympy.polys.rootoftools import _algebraic_comparison

        self.assertEqual(len(result), 1)
        self.assertEqual(_algebraic_comparison(next(iter(result)), root), 0)

    def test_algebraic_interval_bound_is_included_only_when_closed(self):
        x = self.x
        t = s.Symbol("stationary_endpoint")
        root = s.CRootOf(t*t - 2, t, 1)
        source = x**3/3 - 2*x
        result = s.stationary_points(source, x, s.Interval(root, 2))
        from sympy.polys.rootoftools import _algebraic_comparison

        self.assertEqual(len(result), 1)
        self.assertEqual(_algebraic_comparison(next(iter(result)), root), 0)
        self.assertEqual(s.stationary_points(source, x, s.Interval.Lopen(root, 2)),
                         s.S.EmptySet)

    def test_nearby_exact_roots_are_not_merged_by_float_rounding(self):
        x = self.x
        epsilon = s.Rational(1, 10**40)
        source = x**3/3 - (2 + epsilon)*x*x/2 + (1 + epsilon)*x
        self.assertEqual(s.stationary_points(source, x), s.FiniteSet(1, 1 + epsilon))
        self.assertEqual(s.stationary_points(source, x, s.Interval.Lopen(1, 1 + epsilon)),
                         s.FiniteSet(1 + epsilon))

    def test_typed_coordinates_and_inference_are_retained(self):
        dummy = s.Dummy("stationary_dummy")
        declared = s.Symbol("stationary_declared", real=True)
        for variable in (dummy, declared):
            with self.subTest(variable=variable):
                self.assertEqual(s.stationary_points((variable - 2)**2, variable),
                                 s.FiniteSet(2))
                self.assertEqual(s.stationary_points((variable - 2)**2), s.FiniteSet(2))

    def test_nonrational_source_keeps_existing_symbolic_solver_route(self):
        x = self.x
        self.assertEqual(s.stationary_points(s.exp(x), x, s.S.Reals), s.S.EmptySet)

    def test_native_domain_failure_precedes_derivative_and_never_falls_back(self):
        x = self.x
        core = importlib.import_module("sympy.core")
        with patch.object(core._native, "solve_real_rational_inequality_native",
                          side_effect=ValueError("forced chart budget exhaustion")) as checked, \
             patch.object(core, "diff_with_receipt",
                          side_effect=AssertionError("derivative ran after failed domain check")), \
             patch.object(core, "diff", side_effect=AssertionError("generic fallback ran")):
            with self.assertRaisesRegex(NotImplementedError, "chart budget exhaustion"):
                s.stationary_points(x*x, x)
        self.assertEqual(checked.call_count, 1)

    def test_derivative_checker_refusal_is_terminal(self):
        x = self.x
        core = importlib.import_module("sympy.core")
        with patch.object(core, "diff_with_receipt", side_effect=ValueError("derivation refused")), \
             patch.object(core, "diff", side_effect=AssertionError("generic fallback ran")):
            with self.assertRaisesRegex(NotImplementedError, "derivation refused"):
                s.stationary_points(x*x, x)

    def test_changed_arithmetic_descriptor_refuses_without_callback(self):
        x = self.x
        source = s.Mul(x, x, evaluate=False)

        def forbidden(_):
            raise AssertionError("modified source args were invoked")

        with patch.object(s.Mul, "args", property(forbidden)):
            with self.assertRaises(NotImplementedError):
                s.stationary_points(source, x)

    def test_changed_exact_coefficient_refuses_before_native_check(self):
        x = self.x
        source = s.Mul(2, x, evaluate=False)
        core = importlib.import_module("sympy.core")

        def forbidden(_):
            raise AssertionError("modified coefficient accessor was invoked")

        with patch.object(s.Integer, "p", property(forbidden)), \
             patch.object(core._native, "solve_real_rational_inequality_native",
                          side_effect=AssertionError("modified source reached native checks")):
            with self.assertRaises(NotImplementedError):
                s.stationary_points(source, x)

    def test_source_mutation_during_native_admission_prevents_publication(self):
        x = self.x
        source = s.Add(s.Pow(x, 2, evaluate=False), 1, evaluate=False)
        core = importlib.import_module("sympy.core")
        original = core._native.solve_real_rational_inequality_native

        def mutate(*args):
            result = original(*args)
            source._args = (x, s.Integer(1))
            return result

        with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
            with self.assertRaisesRegex(NotImplementedError, "source changed"):
                s.stationary_points(source, x)

    def test_symbol_declaration_mutation_prevents_publication(self):
        x = s.Symbol("stationary_mutated_declaration", real=True)
        source = x*x
        core = importlib.import_module("sympy.core")
        original = core._native.solve_real_rational_inequality_native

        def mutate(*args):
            result = original(*args)
            x._assumptions["positive"] = True
            return result

        try:
            with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
                with self.assertRaisesRegex(NotImplementedError, "assumptions changed"):
                    s.stationary_points(source, x)
        finally:
            x._assumptions.pop("positive", None)

    def test_domain_storage_mutation_during_check_prevents_publication(self):
        x = self.x
        source = x*x
        domain = s.Interval(-1, 1)
        replacement = s.Interval(2, 3)
        core = importlib.import_module("sympy.core")
        original = core._native.solve_real_rational_inequality_native

        def mutate(*args):
            result = original(*args)
            domain._native_set = replacement._native_set
            return result

        with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
            with self.assertRaisesRegex(NotImplementedError, "domain changed"):
                s.stationary_points(source, x, domain)

    def test_changed_domain_accessor_refuses_without_callback(self):
        x = self.x
        domain = s.Interval(-1, 1)

        def forbidden(_):
            raise AssertionError("modified interval accessor ran")

        with patch.object(s.Interval, "start", property(forbidden)):
            with self.assertRaises(NotImplementedError):
                s.stationary_points(x*x, x, domain)

    def test_changed_algebraic_endpoint_class_refuses_before_bound_comparison(self):
        x = self.x
        t = s.Symbol("stationary_changed_endpoint")
        root = s.CRootOf(t*t - 2, t, 1)
        domain = s.Interval(root, 2)
        core = importlib.import_module("sympy.core")
        original = core._native.solve_real_rational_inequality_native
        descriptor = s.CRootOf.__dict__["is_real"]

        def forbidden(_):
            raise AssertionError("modified algebraic endpoint accessor ran")

        def mutate(*args):
            result = original(*args)
            s.CRootOf.is_real = property(forbidden)
            return result

        try:
            with patch.object(core._native, "solve_real_rational_inequality_native", side_effect=mutate):
                with self.assertRaises(NotImplementedError):
                    s.stationary_points(x*x, x, domain)
        finally:
            s.CRootOf.is_real = descriptor

    def test_empty_domain_publication_rechecks_original_source(self):
        x = self.x
        source = s.Add(x*x, 1, evaluate=False)
        ranges = importlib.import_module("sympy.calculus._rational_range")
        original = ranges._as_set

        def mutate(*args):
            result = original(*args)
            source._args = (x, s.Integer(1))
            return result

        with patch.object(ranges, "_as_set", side_effect=mutate):
            with self.assertRaisesRegex(NotImplementedError, "source changed"):
                s.stationary_points(source, x, s.S.EmptySet)


if __name__ == "__main__":
    unittest.main()
