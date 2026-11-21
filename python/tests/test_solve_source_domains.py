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

    def test_direct_linsolve_still_refuses_original_denominator_conditions(self):
        x, y = self.x, self.y
        source = self.equation_with_pole(x, x-1)
        for equations, variables in (([source], (x,)), ([source, y-2], (x, y))):
            with self.subTest(variables=variables):
                with self.assertRaisesRegex(NotImplementedError, "original denominator"):
                    s.linsolve(equations, *variables)

    def test_denominator_dispatch_preserves_sources_for_candidate_admission(self):
        x, y = self.x, self.y
        source = self.equation_with_pole(x, x*(x-1))
        module = importlib.import_module("sympy.solvers.polysys")
        for system, rows, expected in (
            ([source, y-2], [(s.Integer(0), s.Integer(2)),
                             (s.Integer(1), s.Integer(2))], [(1, 2)]),
            (source, [(s.Integer(0), y), (s.Integer(1), y)], [(1, y)]),
        ):
            with self.subTest(system=system):
                def candidates(equations, *variables):
                    self.assertIs(equations[0], source)
                    self.assertEqual(variables, (x, y))
                    if isinstance(system, list):
                        self.assertIs(equations, system)
                    return rows

                with patch.object(module, "_solve_source_system", side_effect=candidates) as dispatched:
                    self.assertEqual(s.solve(system, x, y), expected)
                dispatched.assert_called_once()

    def test_unchecked_rational_system_keeps_denominator_clearing_candidates(self):
        x, y = self.x, self.y
        equations = [self.equation_with_pole(x, x*(x-1)), y-2]
        self.assertEqual(s.solve(equations, x, y, check=False, dict=True),
                         [{x: 0, y: 2}, {x: 1, y: 2}])
        self.assertEqual(s.solve(equations, x, y, check=False, set=True),
                         ([x, y], {(0, 2), (1, 2)}))

    def test_rational_candidate_routing_is_independent_of_source_order(self):
        x, y = self.x, self.y
        reciprocal = s.Pow(x, -1, evaluate=False)
        nonlinear = s.Mul(x, x-1, evaluate=False)
        sources = (
            s.Eq(reciprocal, s.Add(reciprocal, nonlinear, evaluate=False),
                 evaluate=False),
            s.Eq(s.Add(nonlinear, reciprocal, evaluate=False), reciprocal,
                 evaluate=False),
        )
        for source in sources:
            for equations in ([source, y*y-4], [y*y-4, source]):
                with self.subTest(source=source, equations=equations):
                    self.assertEqual(s.solve(equations, x, y, set=True),
                                     ([x, y], {(1, -2), (1, 2)}))
                    self.assertEqual(s.solve(equations, x, y, check=False, set=True),
                                     ([x, y], {(0, -2), (0, 2), (1, -2), (1, 2)}))

    def test_polynomial_sources_keep_the_existing_native_solver(self):
        x, y = self.x, self.y
        from sympy.core import _native
        with patch.object(_native, "solve_source_system_candidates_native",
                          side_effect=AssertionError("polynomial source entered rational admission")):
            self.assertEqual(s.solve([x*x-1, y-2], x, y), [(-1, 2), (1, 2)])

    def test_source_system_clearing_keeps_same_name_coordinate_identities(self):
        plain = s.Symbol("solve_source_coordinate")
        positive = s.Symbol("solve_source_coordinate", positive=True)
        equations = [self.equation_with_pole(plain, positive-2), plain-1]
        result = s.solve(equations, plain, positive, dict=True)
        self.assertEqual(result, [{plain: 1, positive: 2}])
        self.assertEqual(tuple(result[0]), (plain, positive))
        self.assertIs(tuple(result[0])[0], plain)
        self.assertIs(tuple(result[0])[1], positive)

    def test_source_system_admits_undefined_originals_before_empty_shortcuts(self):
        x, y = self.x, self.y
        hidden = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
        source = self.equation_with_pole(x, x-1)
        for checked in (True, False):
            with self.subTest(checked=checked):
                with self.assertRaises(ValueError):
                    s.solve([source, s.Integer(1), hidden], x, y, check=checked)

    def test_source_candidate_failure_does_not_restart_polynomial_solving(self):
        x, y = self.x, self.y
        equations = [self.equation_with_pole(x, x-1), y-2]
        from sympy.core import _native
        failure = ValueError("source-system candidate budget exhausted")
        with patch.object(_native, "solve_source_system_candidates_native",
                          side_effect=failure) as candidate_solver:
            with patch.object(_native, "polynomial_system_solutions_native") as fallback:
                with self.assertRaises(ValueError) as caught:
                    s.solve(equations, x, y)
        self.assertIs(caught.exception, failure)
        candidate_solver.assert_called_once()
        fallback.assert_not_called()

    def test_rational_affine_system_retains_explicit_mapping_and_set_formats(self):
        x, y = self.x, self.y
        equations = [self.equation_with_pole(x, x-1), y-2]
        self.assertEqual(s.solve(equations, x, y, dict=True), [{x: 1, y: 2}])
        self.assertEqual(s.solve(equations, x, y, set=True), ([x, y], {(1, 2)}))

    def test_denominator_dispatch_does_not_hide_unrelated_linear_failures(self):
        x, y = self.x, self.y
        linear = importlib.import_module("sympy.solvers.solvers")
        polynomial = importlib.import_module("sympy.solvers.polysys")
        for system in ([x+y-1], x+y-1):
            for failure in (
                NotImplementedError("unsupported original coefficient interface"),
                RuntimeError("linear request cancelled or exhausted"),
            ):
                with self.subTest(system=system, failure=failure):
                    with patch.object(linear, "linsolve", side_effect=failure):
                        with patch.object(polynomial, "_solve_source_system") as fallback:
                            with self.assertRaises(type(failure)) as caught:
                                s.solve(system, x, y)
                    self.assertIs(caught.exception, failure)
                    fallback.assert_not_called()

    def test_denominator_dispatch_cannot_erase_unsupported_held_terms(self):
        x, y = self.x, self.y
        reciprocal = s.Pow(x, -1, evaluate=False)
        unknown = s.Function("solve_domain_opaque")(x)
        held_zero = s.Mul(0, unknown, evaluate=False)
        source = s.Eq(
            s.Add(reciprocal, held_zero, evaluate=False),
            reciprocal+x-1,
            evaluate=False,
        )
        with self.assertRaises((NotImplementedError, ValueError, TypeError)):
            s.solve([source, y-2], x, y)

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
