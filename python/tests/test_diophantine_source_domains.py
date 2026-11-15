"""Original-domain admission of complete integer solution sets.

These regressions use exact original reciprocal conditions. The bounded
compatibility slice refuses conditional families it cannot represent; this
does not claim complete upstream Diophantine-profile conformance.
"""

import importlib
import unittest
from unittest.mock import patch

import sympy as s


class DiophantineSourceDomainTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = s.symbols("integer_source_x integer_source_y", integer=True)

    def guarded_equation(self, base, residual):
        pole = s.Pow(base, -1, evaluate=False)
        return s.Eq(pole, s.Add(pole, residual, evaluate=False), evaluate=False)

    def test_original_equality_poles_filter_finite_integer_points(self):
        x, y = self.x, self.y
        source = self.guarded_equation(x, x*x + y*y - 1)
        self.assertEqual(s.diophantine(source), {(-1, 0), (1, 0)})
        self.assertEqual(s.diophantine(source, syms=(y, x)), {(0, -1), (0, 1)})

    def test_held_zero_products_do_not_erase_original_exclusions(self):
        x, y = self.x, self.y
        pole = s.Pow(x, -1, evaluate=False)
        source = s.Add(x*x + y*y - 1, s.Mul(0, pole, evaluate=False), evaluate=False)
        self.assertEqual(s.diophantine(source), {(-1, 0), (1, 0)})

    def test_original_bilinear_pole_removes_one_point_from_complete_finite_set(self):
        x, y = self.x, self.y
        source = self.guarded_equation(x-3, (x-2)*(y+3)-6)
        expected = {(-1, -5), (4, 0), (8, -2), (0, -6),
                    (5, -1), (1, -9), (-4, -4)}
        self.assertEqual(s.diophantine(source), expected)

    def test_polynomial_sources_and_unconditional_families_keep_their_outputs(self):
        x, y = self.x, self.y
        n = s.Symbol("n1", integer=True)
        self.assertEqual(s.diophantine(x*x + y*y - 1),
                         {(-1, 0), (1, 0), (0, -1), (0, 1)})
        self.assertEqual(s.diophantine((x-2)*(y+3)), {(2, n), (n, -3)})

    def test_unresolved_family_domains_refuse_without_returning_partial_families(self):
        x, y = self.x, self.y
        for source in (self.guarded_equation(x, x*y),
                       self.guarded_equation(x-2, (x-2)*(y+3))):
            with self.subTest(source=source):
                with self.assertRaisesRegex(NotImplementedError, "unresolved parameter conditions"):
                    s.diophantine(source)

    def test_cancelled_original_axis_is_still_in_the_solution_mapping(self):
        x, y = self.x, self.y
        source = self.guarded_equation(y, x-1)
        with self.assertRaisesRegex(NotImplementedError, "unresolved parameter conditions"):
            s.diophantine(source)

    def test_same_printed_name_does_not_alias_a_known_nonzero_source_symbol(self):
        x = s.Symbol("same_integer_source", integer=True)
        other = s.Symbol("same_integer_source", integer=True, positive=True)
        y = self.y
        source = self.guarded_equation(other, x*x + y*y - 1)
        self.assertEqual(s.diophantine(source, syms=(x, y)),
                         {(-1, 0), (1, 0), (0, -1), (0, 1)})

    def test_guarded_family_parameter_cannot_be_captured_by_a_later_axis(self):
        x = s.Symbol("a", integer=True)
        y = s.Symbol("n1", integer=True)
        # The y=3 branch uses the free parameter named n1 for x. Checking
        # x->n1, n1->3 sequentially must not falsely prove that x is nonzero.
        source = self.guarded_equation(x, x*(y-3))
        with self.assertRaisesRegex(NotImplementedError, "uncaptured family parameters"):
            s.diophantine(source, syms=(x, y))

    def test_one_domain_checking_allowance_covers_all_candidate_points(self):
        x, y = self.x, self.y
        source = self.guarded_equation(x, x*x+y*y-1)
        checker = importlib.import_module("sympy.solvers._checksol")
        original = checker._check_with_work
        ledgers = []

        def observed(*args, **kwargs):
            work = args[3]
            ledgers.append(work)
            if len(ledgers) == 2:
                work.remaining = 0
            return original(*args, **kwargs)

        with patch.object(checker, "_check_with_work", side_effect=observed):
            with self.assertRaisesRegex(NotImplementedError, "domain checking exceeds its bounded work limit"):
                s.diophantine(source)
        self.assertEqual(len(ledgers), 2)
        self.assertIs(ledgers[0], ledgers[1])

    def test_unadmitted_custom_source_is_refused_before_expansion(self):
        x, y = self.x, self.y
        calls = []

        class Custom(s.Function):
            def _eval_subs(self, old, new):
                calls.append("subs")
                raise AssertionError("unadmitted custom substitution")

        source = s.Add(x*x+y*y-1, s.Mul(0, Custom(x), evaluate=False), evaluate=False)
        with self.assertRaisesRegex(NotImplementedError, "original source exceeds"):
            s.diophantine(source)
        self.assertEqual(calls, [])


    def test_constant_normalizations_do_not_divide_by_a_zero_coefficient_gcd(self):
        x, y = self.x, self.y
        self.assertEqual(s.diophantine(self.guarded_equation(x, s.Integer(1))), set())
        with self.assertRaisesRegex(NotImplementedError, "unresolved parameter conditions"):
            s.diophantine(self.guarded_equation(x, s.Integer(0)))
        zero = s.Mul(0, x+y, evaluate=False)
        with self.assertRaisesRegex(NotImplementedError, "unconstrained integer families"):
            s.diophantine(zero)

    def test_cancelled_axes_do_not_couple_independent_free_coordinates(self):
        x, y = self.x, self.y
        z = s.Symbol("integer_source_z", integer=True)
        for residual in (x-1, y-1, z-1):
            total = s.Add(residual, s.Mul(0, x+y+z, evaluate=False), evaluate=False)
            guarded = self.guarded_equation((x*x+1)*(z*z+1), residual)
            for source in (total, guarded):
                with self.subTest(residual=residual, guarded=source is guarded):
                    with self.assertRaisesRegex(NotImplementedError, "independent free coordinates"):
                        s.diophantine(source, syms=(x, y, z))

    def test_equality_normalization_uses_retained_sides_without_public_getters(self):
        x, y = self.x, self.y
        source = self.guarded_equation(x, x*x+y*y-1)
        calls = []

        def changed_side(_):
            calls.append("side")
            raise AssertionError("mutable Equality getter")

        with patch.object(s.Eq, "lhs", property(changed_side)), \
                patch.object(s.Eq, "rhs", property(changed_side)):
            self.assertEqual(s.diophantine(source), {(-1, 0), (1, 0)})
        self.assertEqual(calls, [])

    def test_external_converters_and_compound_strings_are_refused_before_evaluation(self):
        calls = []

        class Foreign:
            def _sympy_(self):
                calls.append("convert")
                raise AssertionError("foreign conversion")

        for source in (Foreign(), "integer_source_x**2 + integer_source_y**2 - 1"):
            with self.assertRaisesRegex(NotImplementedError, "original source exceeds"):
                s.diophantine(source)
        self.assertEqual(calls, [])

    def test_changed_symbol_name_getter_is_refused_before_axis_sorting(self):
        x, y = self.x, self.y
        source = x*x+y*y-1
        calls = []

        def changed_name(_):
            calls.append("name")
            raise AssertionError("mutable Symbol name")

        with patch.object(s.Symbol, "name", property(changed_name)):
            with self.assertRaisesRegex(NotImplementedError, "original source exceeds"):
                s.diophantine(source)
        self.assertEqual(calls, [])

    def test_coordinate_ownership_and_uniqueness_are_admitted(self):
        x, y = self.x, self.y
        calls = []

        class ForeignAxes:
            def __bool__(self):
                calls.append("bool")
                raise AssertionError("foreign axes")

            def __iter__(self):
                calls.append("iter")
                raise AssertionError("foreign axes")

        with self.assertRaisesRegex(NotImplementedError, "built-in list or tuple"):
            s.diophantine(x*x+y*y-1, syms=ForeignAxes())
        with self.assertRaisesRegex(ValueError, "duplicate solution coordinates"):
            s.diophantine(x*x+y*y-1, syms=(x, x))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
