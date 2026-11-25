"""Complete solution sets retain the original equation's domain.

The roots below follow exact polynomial, logarithmic and periodic identities. Holding an
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


class PeriodicSolveSetSourceDomainTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("periodic_source_x")

    @staticmethod
    def equation(pole, residual):
        inverse = s.Pow(pole, -1, evaluate=False)
        return s.Eq(inverse, inverse + residual, evaluate=False)

    @staticmethod
    def families(result):
        return result.args if isinstance(result, s.Union) else (result,)

    def assert_original_condition(self, result, source, variable=None):
        self.assertIsInstance(result, s.ConditionSet)
        self.assertIs(result.sym, self.x if variable is None else variable)
        self.assertIs(result.condition, source)

    def test_cancelled_pole_is_excluded_from_infinite_trigonometric_sets(self):
        x = self.x
        for residual in (s.sin(x), s.cos(x) - 1, s.sin(-2*x)):
            source = self.equation(x, residual)
            for domain in (None, s.S.Complexes, s.S.Reals):
                with self.subTest(residual=residual, domain=domain):
                    result = s.solveset(source, x, domain)
                    self.assert_original_condition(result, source)
                    self.assertIs(result.contains(0), False)
                    branches = self.families(result.base_set)
                    self.assertTrue(all(isinstance(branch, s.ImageSet) for branch in branches))
                    self.assertIn(s.Integer(0), [branch.lamda(0) for branch in branches])

    def test_complex_exponential_family_keeps_its_cancelled_zero_pole(self):
        x = self.x
        source = self.equation(x, s.exp(x) - 1)
        for domain in (None, s.S.Complexes):
            with self.subTest(domain=domain):
                result = s.solveset(source, x, domain)
                self.assert_original_condition(result, source)
                self.assertIs(result.contains(0), False)
                family = result.base_set
                self.assertIsInstance(family, s.ImageSet)
                self.assertEqual(family.lamda(0), 0)
                self.assertEqual(s.expand(family.lamda(1) - family.lamda(0)), 2*s.pi*s.I)

    def test_periodic_variable_inference_uses_original_source(self):
        source = self.equation(self.x, s.sin(self.x))
        result = s.solveset(source)
        self.assert_original_condition(result, source)
        self.assertIs(result.contains(0), False)

    def test_held_zero_product_retains_its_pole_in_a_periodic_equation(self):
        x = self.x
        hidden = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
        source = s.Eq(hidden, -s.sin(x), evaluate=False)
        for domain in (s.S.Reals, s.S.Complexes):
            with self.subTest(domain=domain):
                result = s.solveset(source, x, domain)
                self.assert_original_condition(result, source)
                self.assertIs(result.contains(0), False)

    def test_held_zeroth_power_retains_its_nested_pole(self):
        x = self.x
        hidden = s.Pow(s.Pow(x, -1, evaluate=False), 0, evaluate=False)
        source = s.Eq(hidden, 1 + s.sin(x), evaluate=False)
        result = s.solveset(source, x, s.S.Complexes)
        self.assert_original_condition(result, source)
        self.assertIs(result.contains(0), False)

    def test_periodic_expression_without_equality_retains_held_source(self):
        x = self.x
        hidden = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
        source = s.Add(s.sin(x), hidden, evaluate=False)
        result = s.solveset(source, x, s.S.Complexes)
        self.assertIsInstance(result, s.ConditionSet)
        self.assertEqual(result.condition, s.Eq(source, 0, evaluate=False))
        self.assertIs(result.contains(0), False)

    def test_shifted_periodic_pole_is_excluded_exactly(self):
        source = self.equation(self.x - s.pi, s.sin(self.x))
        result = s.solveset(source, self.x, s.S.Reals)
        self.assert_original_condition(result, source)
        self.assertIs(result.contains(s.pi), False)
        self.assertIs(s.checksol(source, self.x, 0), True)

    def test_unknown_parameter_pole_remains_an_explicit_obligation(self):
        parameter = s.Symbol("periodic_parameter")
        source = self.equation(parameter, s.sin(self.x))
        result = s.solveset(source, self.x, s.S.Complexes)
        self.assert_original_condition(result, source)
        self.assertIsNone(result.contains(0))

    def test_known_parameter_nonzero_fact_discharges_the_obligation(self):
        parameter = s.Symbol("periodic_nonzero_parameter", nonzero=True)
        source = self.equation(parameter, s.sin(self.x))
        result = s.solveset(source, self.x, s.S.Complexes)
        self.assertNotIsInstance(result, s.ConditionSet)
        self.assertTrue(all(isinstance(branch, s.ImageSet) for branch in self.families(result)))

    def test_same_printed_parameter_does_not_alias_the_solved_coordinate(self):
        x = s.Symbol("periodic_shared_name")
        parameter = s.Symbol("periodic_shared_name", positive=True)
        source = self.equation(parameter, s.sin(x))
        result = s.solveset(source, x, s.S.Complexes)
        self.assertNotIsInstance(result, s.ConditionSet)
        source = self.equation(x, s.sin(x))
        result = s.solveset(source, x, s.S.Complexes)
        self.assert_original_condition(result, source, x)
        self.assertIs(result.contains(0), False)

    def test_solved_coordinate_assumptions_do_not_erase_requested_domain_holes(self):
        for facts in ({"positive": True}, {"nonzero": True}, {"real": True}):
            with self.subTest(facts=facts):
                x = s.Symbol("periodic_assumed_coordinate", **facts)
                source = self.equation(x, s.sin(x))
                result = s.solveset(source, x, s.S.Complexes)
                self.assert_original_condition(result, source, x)
                self.assertIs(result.contains(0), False)

    def test_original_condition_keeps_the_requested_interval_restriction(self):
        x = self.x
        domain = s.Interval(0, 4)
        source = self.equation(x - s.pi, s.sin(x))
        result = s.solveset(source, x, domain)
        self.assert_original_condition(result, source)
        self.assertIsInstance(result.base_set, s.Intersection)
        self.assertIn(domain, result.base_set.args)
        self.assertIs(result.contains(-s.pi), False)
        self.assertIs(result.contains(s.pi), False)

    def test_entire_families_keep_their_existing_unconditional_representation(self):
        for residual, domain in (
            (s.sin(self.x), s.S.Reals),
            (s.cos(self.x) - 1, s.S.Complexes),
            (s.exp(self.x) - 1, s.S.Complexes),
        ):
            with self.subTest(residual=residual, domain=domain):
                result = s.solveset(residual, self.x, domain)
                self.assertNotIsInstance(result, s.ConditionSet)
                self.assertTrue(all(isinstance(branch, s.ImageSet) for branch in self.families(result)))

    def test_periodic_source_check_cannot_publish_after_nested_budget_failure(self):
        checker = importlib.import_module("sympy.solvers._checksol")
        failure = ValueError("nested native request budget exhausted")
        source = self.equation(self.x, s.sin(self.x))
        with patch.object(checker, "_charge_nested_if_active", side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                s.solveset(source, self.x, s.S.Complexes)
        self.assertIs(caught.exception, failure)

    def test_periodic_source_exhaustion_is_terminal_and_sticky(self):
        checker = importlib.import_module("sympy.solvers._checksol")
        source = self.equation(self.x, s.sin(self.x))
        admission = checker._SolveSetSourceCheck(source)
        candidates = s.solveset(s.sin(self.x), self.x, s.S.Complexes)
        admission.work.remaining = 1
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "solveset source checking exceeds its work limit"):
                admission.restrict_set(self.x, candidates)
            self.assertTrue(admission.work.exhausted)

    def test_public_periodic_source_work_bound_is_enforced(self):
        checker = importlib.import_module("sympy.solvers._checksol")
        source = self.equation(self.x, s.sin(self.x))
        with patch.object(checker, "_MAX_WORK", 8):
            with self.assertRaisesRegex(ValueError, "solveset source checking exceeds its work limit"):
                s.solveset(source, self.x, s.S.Complexes)


if __name__ == "__main__":
    unittest.main()
