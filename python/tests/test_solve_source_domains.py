"""solve preserves original definedness through generation and formatting.

The basic cancelled-Equality case and its check=False/output conventions were
checked in isolated SymPy 1.14.0. Other expectations follow exact original
domain conditions; this gate does not claim general solve-profile conformance.
"""

import importlib
import unittest
from unittest.mock import patch

import sympy as s


class SolveOriginalSourceDomainTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("solve_source_x")
        self.y = s.Symbol("solve_source_y")

    def equation_with_pole(self, pole, residual):
        reciprocal = s.Pow(pole, -1, evaluate=False)
        return s.Eq(reciprocal, reciprocal + residual, evaluate=False)

    def test_cancelled_equality_sides_do_not_publish_an_undefined_root(self):
        x = self.x
        source = self.equation_with_pole(x, x)
        for variables in ((x,), ([x],), ()):
            with self.subTest(variables=variables):
                self.assertEqual(s.solve(source, *variables), [])
                self.assertEqual(s.solve(source, *variables, dict=True), [])
                self.assertEqual(s.solve(source, *variables, set=True), ([x], set()))
        self.assertEqual(s.solve(source, x, check=False), [0])
        self.assertEqual(s.solve(source, x, check=False, dict=True), [{x: 0}])
        self.assertEqual(s.solve(source, x, check=False, set=True), ([x], {(0,)}))

    def test_valid_roots_survive_the_original_domain_filter_in_every_shape(self):
        x = self.x
        source = self.equation_with_pole(x-2, (x-2)*(x-3))
        self.assertEqual(s.solve(source, x), [3])
        self.assertEqual(s.solve(source, x, dict=True), [{x: 3}])
        variables, roots = s.solve(source, x, set=True)
        self.assertIs(variables[0], x)
        self.assertEqual(roots, {(3,)})
        self.assertEqual(s.solve(source, x, check=False), [2, 3])
        # An expression explicitly supplied after removing the reciprocal
        # has its own domain; the checker cannot invent an earlier source.
        self.assertEqual(s.solve((x-2)*(x-3), x), [2, 3])

    def test_rational_candidates_are_checked_without_floating_conversion(self):
        x = self.x
        source = self.equation_with_pole(2*x-1, (2*x-1)*(3*x-2))
        self.assertEqual(s.solve(source, x), [s.Rational(2, 3)])
        self.assertEqual(s.solve(source, x, check=False),
                         [s.Rational(1, 2), s.Rational(2, 3)])
        self.assertEqual(s.solve((x*x-1)/(x-1), x), [-1])

    def test_linear_and_polynomial_systems_filter_whole_candidate_rows(self):
        x, y = self.x, self.y
        linear = [self.equation_with_pole(x, x), y-2]
        self.assertEqual(s.solve(linear, x, y), [])
        self.assertEqual(s.solve(linear, x, y, dict=True), [])
        self.assertEqual(s.solve(linear, x, y, set=True), ([x, y], set()))
        nonlinear = [self.equation_with_pole(x, x*(x-1)), y-2]
        self.assertEqual(s.solve(nonlinear, x, y), [(1, 2)])
        self.assertEqual(s.solve(nonlinear, x, y, dict=True), [{x: 1, y: 2}])
        self.assertEqual(s.solve(nonlinear, x, y, set=True), ([x, y], {(1, 2)}))

    def test_multivariable_scalar_fallback_keeps_free_coordinate_and_output_shape(self):
        x, y = self.x, self.y
        source = self.equation_with_pole(x, x*(x-1))
        self.assertEqual(s.solve(source, x, y), [(1, y)])
        self.assertEqual(s.solve(source, x, y, dict=True), [{x: 1}])
        self.assertEqual(s.solve(source, x, y, set=True), ([x, y], {(1, y)}))
        self.assertEqual(s.solve(source, x, y, check=False), [(0, y), (1, y)])

    def test_unresolved_parameter_domain_is_not_interpreted_as_false(self):
        x, y = self.x, self.y
        source = self.equation_with_pole(x, x-y)
        roots = s.solve(source, x)
        self.assertEqual(roots, [y])
        self.assertIs(roots[0], y)
        self.assertIsNone(s.checksol(source, x, y))

    def test_same_name_symbols_keep_distinct_source_domains(self):
        x = s.Symbol("solve_same_source")
        positive = s.Symbol("solve_same_source", positive=True)
        source = self.equation_with_pole(positive, x)
        self.assertEqual(s.solve(source, x), [0])
        source = self.equation_with_pole(x, positive-1)
        result = s.solve(source, positive, dict=True)
        self.assertEqual(result, [{positive: 1}])
        self.assertIs(next(iter(result[0])), positive)

    def test_float_rationalization_retains_original_poles_before_rounding(self):
        x = self.x
        source = self.equation_with_pole(x, s.Float("0.1")*x)
        self.assertEqual(s.solve(source, x), [])
        self.assertEqual(s.solve(source, x, check=False), [s.Float(0)])
        roots = s.solve(x*x-s.Float("2.0"), x)
        self.assertEqual(len(roots), 2)
        for root in roots:
            self.assertLess(abs(float(root)*float(root)-2), 1e-12)
        decimal_roots = s.solve(x-s.Float("0.3333333333333"), x)
        self.assertEqual(len(decimal_roots), 1)
        self.assertLess(abs(float(decimal_roots[0])-0.3333333333333), 1e-12)

    def test_indexed_algebraic_roots_are_preserved_when_domain_check_is_unknown(self):
        x = self.x
        source = x**5-x+1
        unchecked = s.solve(source, x, check=False)
        self.assertEqual(len(unchecked), 5)
        self.assertTrue(any(isinstance(root, s.CRootOf) for root in unchecked))
        self.assertEqual(s.solve(source, x), unchecked)

    def test_original_source_snapshot_survives_generator_list_mutation(self):
        x = self.x
        original = self.equation_with_pole(x, x)
        equations = [original]
        module = importlib.import_module("sympy.solvers.solvers")

        def altered_source(_equations, *_variables):
            equations[:] = [x]
            return s.FiniteSet(s.Tuple(0))

        with patch.object(module, "linsolve", side_effect=altered_source):
            self.assertEqual(s.solve(equations, x), [])
        self.assertEqual(equations, [x])

    def test_source_checks_share_one_bounded_allowance_and_exhaustion_is_terminal(self):
        x = self.x
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
            with self.assertRaisesRegex(ValueError, "source checking exceeds its work limit"):
                s.solve(x*x-1, x)
        self.assertEqual(len(ledgers), 2)
        self.assertIs(ledgers[0], ledgers[1])

    def test_enclosing_callback_budget_failure_is_not_an_unknown_domain(self):
        x = self.x
        checker = importlib.import_module("sympy.solvers._checksol")
        failure = ValueError("nested native request budget exhausted")
        calls = []

        def exhausted_inside_scalar(count):
            calls.append(count)
            if len(calls) == 3:
                raise failure

        # Row admission and conjunction traversal charge first; the third
        # charge happens inside scalar source traversal, whose public fuzzy
        # diagnostic lane catches ValueError. solve must propagate it.
        with patch.object(checker, "_charge_nested_if_active",
                          side_effect=exhausted_inside_scalar):
            with self.assertRaises(ValueError) as caught:
                s.solve(x-1, x)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(calls), 3)

    def test_substitution_callback_budget_failure_is_checked_before_publication(self):
        x = self.x
        core = importlib.import_module("sympy.core")
        checker = importlib.import_module("sympy.solvers._checksol")
        candidate = s.Integer(0)
        replacements = {x: candidate}
        # Both entry points must see a real enclosing session's sticky budget,
        # including when the charge originated in Basic.subs rather than Work.
        for entry in ("conjunction", "solve_filter"):
            with self.subTest(entry=entry):
                source_check = checker._SolveSourceCheck(x)
                work = checker._Work()
                injected = []
                with core.supervised_callback_session(max_nested_steps=1000) as session:
                    def exhaust_inside_substitution(steps=1):
                        injected.append(steps)
                        session.charge_nested_steps(1001)

                    # The checker's imported charge alias remains original.
                    # Core's substitution hook overspends the actual session.
                    with patch.object(core, "_charge_nested_if_active",
                                      side_effect=exhaust_inside_substitution):
                        with self.assertRaises((ValueError, RuntimeError)):
                            if entry == "conjunction":
                                checker._check_with_work(
                                    x, replacements, None, work, domain_only=True,
                                )
                            else:
                                source_check.filter([(candidate,)], (x,))
                self.assertEqual(len(injected), 1)

    def test_check_false_does_not_allocate_or_execute_source_checking(self):
        x = self.x
        source = self.equation_with_pole(x, x)
        with patch.object(s, "_SolveSourceCheck",
                          side_effect=AssertionError("unchecked request ran domain admission")):
            self.assertEqual(s.solve(source, x, check=False), [0])

    def test_domain_admission_does_not_capture_other_candidate_coordinates(self):
        x, y = self.x, self.y
        checker = importlib.import_module("sympy.solvers._checksol")
        reciprocal = s.Pow(x, -1, evaluate=False)
        source = s.Eq(reciprocal, reciprocal, evaluate=False)
        # A candidate's y may be a free family parameter. The domain
        # checker cannot replace it by the separate coordinate's zero.
        verdict = checker._check_with_work(
            source, {x: y, y: 0}, None, checker._Work(), domain_only=True,
        )
        self.assertIsNone(verdict)
        # Public checksol retains its established sequential substitution.
        self.assertIs(s.checksol(source, {x: y, y: 0}), False)

    def test_public_checksol_keeps_its_full_residual_verdict(self):
        x = self.x
        self.assertIs(s.checksol(x-1, x, 1), True)
        self.assertIs(s.checksol(x-1, x, 2), False)
        self.assertIs(s.checksol([x-1, 2*x-2], x, 1), True)


if __name__ == "__main__":
    unittest.main()
