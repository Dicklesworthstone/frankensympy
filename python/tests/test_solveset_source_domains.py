"""Complete finite solution sets retain the original equation's domain.

The roots below follow exact polynomial/logarithmic identities. Holding an
equation's sides is essential: subtracting them first would erase the pole
whose exclusion these regressions check. No numeric tolerance admits roots.
"""

import importlib
import unittest
from unittest.mock import patch

import sympy as s


class SolveSetSourceDomainTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("solveset_source_x")

    @staticmethod
    def equation(pole, residual):
        inverse = s.Pow(pole, -1, evaluate=False)
        return s.Eq(inverse, inverse + residual, evaluate=False)

    def test_complex_equation_sides_cannot_cancel_an_excluded_root(self):
        source = self.equation(self.x, self.x)
        for domain in (None, s.S.Complexes):
            with self.subTest(domain=domain):
                self.assertEqual(s.solveset(source, self.x, domain), s.S.EmptySet)
        self.assertEqual(s.solveset(source), s.S.EmptySet)

    def test_valid_complex_roots_survive_original_pole_admission(self):
        x = self.x
        source = self.equation(x - 2, (x - 2)*(x - 3))
        self.assertEqual(s.solveset(source, x), s.FiniteSet(3))
        self.assertEqual(s.solveset((x - 2)*(x - 3), x), s.FiniteSet(2, 3))

    def test_exact_rational_candidates_are_filtered_without_rounding(self):
        x = self.x
        source = self.equation(2*x - 1, (2*x - 1)*(3*x - 2))
        self.assertEqual(s.solveset(source, x), s.FiniteSet(s.Rational(2, 3)))

    def test_logarithmic_root_must_exist_in_the_original_equation(self):
        x = self.x
        source = self.equation(x - 1, s.log(x))
        for domain in (s.S.Reals, s.S.Complexes):
            with self.subTest(domain=domain):
                self.assertEqual(s.solveset(source, x, domain), s.S.EmptySet)

    def test_real_exponential_root_is_checked_after_native_inversion(self):
        x = self.x
        source = self.equation(x - 1, s.exp(x) - s.E)
        self.assertEqual(s.solveset(source, x, s.S.Reals), s.S.EmptySet)
        source = self.equation(x - 2, s.exp(x) - s.E)
        self.assertEqual(s.solveset(source, x, s.S.Reals), s.FiniteSet(1))

    def test_unknown_parameter_pole_is_a_condition_not_a_proven_root(self):
        x, y = self.x, s.Symbol("solveset_source_y")
        source = self.equation(x, x - y)
        result = s.solveset(source, x)
        self.assertIsInstance(result, s.ConditionSet)
        self.assertIs(result.sym, x)
        self.assertIs(result.condition, source)
        self.assertEqual(result.base_set, s.FiniteSet(y))
        self.assertIsNone(result.contains(y))

    def test_same_printed_name_does_not_merge_typed_pole_coordinates(self):
        x = s.Symbol("same_pole_coordinate")
        positive = s.Symbol("same_pole_coordinate", positive=True)
        source = self.equation(positive, x)
        self.assertEqual(s.solveset(source, x), s.FiniteSet(0))

    def test_normalized_identity_keeps_its_original_complex_hole(self):
        x = self.x
        inverse = s.Pow(x, -1, evaluate=False)
        source = s.Eq(inverse, inverse, evaluate=False)
        result = s.solveset(source, x)
        self.assertIsInstance(result, s.ConditionSet)
        self.assertIs(result.condition, source)
        self.assertEqual(result.base_set, s.S.Complexes)
        self.assertIs(result.contains(0), False)
        self.assertIs(result.contains(2), True)

    def test_unrestricted_identity_has_the_complex_solution_domain(self):
        self.assertEqual(s.solveset(0, self.x), s.S.Complexes)
        self.assertEqual(s.solveset(0), s.S.Complexes)
        self.assertEqual(s.solveset(0, self.x, s.S.Reals), s.S.Reals)

    def test_identity_domain_does_not_inherit_the_solved_symbol_assumptions(self):
        for facts in ({"positive": True}, {"nonzero": True}, {"real": True}):
            with self.subTest(facts=facts):
                x = s.Symbol("assumed_identity_coordinate", **facts)
                inverse = s.Pow(x, -1, evaluate=False)
                source = s.Eq(inverse, inverse, evaluate=False)
                result = s.solveset(source, x, s.S.Complexes)
                self.assertIsInstance(result, s.ConditionSet)
                self.assertIs(result.condition, source)
                self.assertIs(result.contains(0), False)
                self.assertIs(result.contains(-1), True)

    def test_identity_keeps_assumptions_on_independent_parameters(self):
        parameter = s.Symbol("positive_identity_parameter", positive=True)
        inverse = s.Pow(parameter, -1, evaluate=False)
        source = s.Eq(inverse, inverse, evaluate=False)
        self.assertEqual(s.solveset(source, self.x, s.S.Complexes), s.S.Complexes)

    def test_equation_condition_membership_checks_held_zero_products(self):
        x = self.x
        hidden_pole = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
        source = s.Eq(hidden_pole, 0, evaluate=False)
        result = s.ConditionSet(x, source, s.S.Complexes)
        self.assertIs(result.contains(0), False)
        self.assertIs(result.contains(1), True)

    def test_equation_condition_membership_preserves_base_set_restrictions(self):
        x = self.x
        result = s.ConditionSet(x, s.Eq(x*x, 1, evaluate=False), s.FiniteSet(1))
        self.assertIs(result.contains(-1), False)
        self.assertIs(result.contains(1), True)
        self.assertIs(result.contains(2), False)

    def test_finite_candidate_checks_share_one_allowance(self):
        checker = importlib.import_module("sympy.solvers._checksol")
        original = checker._check_with_work
        ledgers = []

        def observed(*args, **kwargs):
            ledgers.append(args[3])
            if len(ledgers) == 2:
                args[3].remaining = 0
            return original(*args, **kwargs)

        with patch.object(checker, "_check_with_work", side_effect=observed):
            with self.assertRaisesRegex(ValueError, "solveset source checking exceeds its work limit"):
                s.solveset(self.x*self.x - 1, self.x)
        self.assertEqual(len(ledgers), 2)
        self.assertIs(ledgers[0], ledgers[1])

    def test_enclosing_budget_failure_is_terminal_before_set_publication(self):
        checker = importlib.import_module("sympy.solvers._checksol")
        failure = ValueError("nested native request budget exhausted")

        with patch.object(checker, "_charge_nested_if_active", side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                s.solveset(self.x - 1, self.x)
        self.assertIs(caught.exception, failure)

    def test_exact_zero_fact_queries_cannot_refill_source_work(self):
        checker = importlib.import_module("sympy.solvers._checksol")
        work = checker._Work()
        work.remaining = 1
        phase = s.Pow(-1, s.Rational(2, 3))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "assumptions query exceeds"):
                work.zero_fact(phase)
            self.assertTrue(work.exhausted)
            self.assertEqual(work.remaining, 0)

    def test_existing_real_rational_domain_path_keeps_exact_exclusions(self):
        x = self.x
        source = self.equation(x - 1, (x - 1)*(x + 1))
        self.assertEqual(s.solveset(source, x, s.S.Reals), s.FiniteSet(-1))


if __name__ == "__main__":
    unittest.main()
