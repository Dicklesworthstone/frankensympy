"""Exact recurrence publication, source admission, and independent counterexamples.

The eight established fixtures retain their public-form comparators. New
freshness and refusal checks are mathematical safety invariants, not claims of
full upstream rsolve compatibility: SymPy 1.14 itself captures some C0 inputs.
Expected formulas below are derived directly from the original recurrences.
No production recurrence helper supplies an expected answer or residual check.
"""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Dummy, Eq, Float, Function, Integer, Mul, Pow, Rational, Symbol,
    exp, expand, expand_power_exp, factorial, rsolve, simplify,
)


class RecurrenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.n = Symbol("n", integer=True)
        self.y = Function("recurrence_y")

    def assert_zero(self, value):
        # All exponential bases used by this comparator are explicit nonzero
        # constants. Original-domain adversaries are tested separately below.
        self.assertEqual(simplify(expand(expand_power_exp(value))), 0)

    def check_original(self, source, answer, shifts, initial=None, index=None):
        n = self.n if index is None else index
        substitutions = {self.y(n + shift): answer.subs(n, n + shift) for shift in shifts}
        residual = source.lhs.subs(substitutions) - source.rhs.subs(substitutions) if type(source) is Eq else source.subs(substitutions)
        self.assert_zero(residual)
        for point, value in (initial or {}).items():
            self.assert_zero(answer.subs(n, point) - value)

    def test_established_eight_public_recurrence_fixtures_keep_their_forms(self):
        n, y = self.n, self.y
        fixtures = (
            (y(n+1)-2*y(n), None, "2**n*C0"),
            (y(n+2)-y(n+1)-y(n), {y(0): 0, y(1): 1},
             "-sqrt(5)*(1/2 - sqrt(5)/2)**n/5 + sqrt(5)*(1/2 + sqrt(5)/2)**n/5"),
            (y(n+1)-y(n)-n, None, "C0 + n*(n - 1)/2"),
            (y(n+2)-4*y(n+1)+4*y(n), None, "2**n*C0 + 2**(n - 1)*C1*n"),
            (y(n+1)-3*y(n)-2**n, None, "-2**n + 3**n*C0"),
            (y(n)-y(n-1)-2, {y(0): 1}, "2*n + 1"),
            (y(n+1)-(n+1)*y(n), None, "C0*factorial(n)"),
            (y(n+2)+y(n), None, "I**n*C0 + C1*(-I)**n"),
        )
        for source, initial, expected in fixtures:
            with self.subTest(source=source):
                self.assertEqual(str(rsolve(source, y(n), initial)), expected)

    def test_exact_integer_shifts_are_admitted_without_rounding(self):
        n, y = self.n, self.y
        source = y(n+Integer(2))-y(n+Integer(1))-n
        answer = rsolve(source, y(n), {y(0): 0})
        self.assert_zero(answer - n*(n-3)/2)
        self.check_original(source, answer, (1, 2), {0: 0})

        # An unspecified Symbol follows the documented integer-lattice
        # convention; an explicit noninteger assumption contradicts it.
        plain = Symbol("recurrence_plain_index")
        source = y(plain+1)-y(plain)-1
        answer = rsolve(source, y(plain), {y(0): 0})
        self.assert_zero(answer - plain)
        self.check_original(source, answer, (0, 1), {0: 0}, index=plain)
        noninteger = Symbol("recurrence_noninteger_index", integer=False)
        with self.assertRaises(ValueError):
            rsolve(y(noninteger+1)-y(noninteger)-1, y(noninteger))

    def test_fractional_float_and_symbolic_integer_shifts_refuse(self):
        n, y = self.n, self.y
        offsets = (Rational(1, 2), Rational(-3, 2), Float(0.5), Float(1.0),
                   Symbol("offset", integer=True), n)
        for offset in offsets:
            with self.subTest(offset=offset):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+offset)-2*y(n+1), y(n))

    def test_huge_exact_shift_is_a_bounded_refusal(self):
        n, y = self.n, self.y
        with self.assertRaises(NotImplementedError):
            rsolve(y(n+10**30)-y(n), y(n))

    def test_forcing_power_slope_and_offset_are_admitted_before_integer_power_work(self):
        from sympy.core import _native, _wrap

        n, y = self.n, self.y
        native_power = _native.py_pow

        def reject_oversized_power(base, exponent):
            b, e = _wrap(base), _wrap(exponent)
            if b == 2 and isinstance(e, Integer) and abs(e.p) > 1024:
                raise AssertionError("out-of-envelope forcing power was materialized")
            return native_power(base, exponent)

        # A small boundary witness keeps the regression safe even if a future
        # implementation loses the guard; the same path otherwise permits
        # compact inputs such as 2**(10**100*n) to allocate an enormous integer.
        for exponent in (1025*n, Add(n, 1025, evaluate=False)):
            forcing = Pow(2, exponent, evaluate=False)
            source = Add(y(n+1)-3*y(n), -forcing, evaluate=False)
            with self.subTest(exponent=exponent), patch.object(
                _native, "py_pow", side_effect=reject_oversized_power,
            ):
                with self.assertRaises(NotImplementedError):
                    rsolve(source, y(n))

    def test_original_unshifted_recurrences_and_initial_values_are_preserved(self):
        n, y = self.n, self.y
        cases = (
            (y(n+3)-2*y(n+2)-3, (2, 3), {0: 1}, 4*2**n-3),
            (y(n+2)-y(n+1)-n, (1, 2), {0: 0}, n*(n-3)/2),
            (y(n)-y(n-2)-2*n, (-2, 0), {0: 0, 1: Rational(3, 2)}, n*n/2+n),
            (Eq(y(n+1), 2*y(n)+3, evaluate=False), (0, 1), {0: 4}, 7*2**n-3),
        )
        for source, shifts, initial, expected in cases:
            with self.subTest(source=source):
                answer = rsolve(source, y(n), {y(point): value for point, value in initial.items()})
                self.assert_zero(answer - expected)
                self.check_original(source, answer, shifts, initial)

    def test_internal_looking_forcing_parameters_remain_user_symbols(self):
        n, y = self.n, self.y
        for name in ("_fsym_rs_y0", "_fsym_rs_y1", "_fsym_rs_a0", "_fsym_rs_a1",
                     "_fsym_rsolve_r", "_fsym_rs_k"):
            parameter = Symbol(name)
            source = y(n+1)-y(n)-parameter
            with self.subTest(parameter=name):
                answer = rsolve(source, y(n), {y(0): 0})
                self.assert_zero(answer - n*parameter)
                self.assertEqual(answer.free_symbols, {n, parameter})
                self.check_original(source, answer, (0, 1), {0: 0})

    def test_internal_looking_index_symbols_and_dummies_keep_their_identities(self):
        y = self.y
        for n in (Symbol("_fsym_rs_y0", integer=True), Symbol("_fsym_rs_a0", integer=True),
                  Symbol("_fsym_rsolve_r", integer=True), Dummy("n", integer=True)):
            source = y(n+1)-y(n)-n
            with self.subTest(index=n):
                answer = rsolve(source, y(n), {y(0): 0})
                self.assert_zero(answer - n*(n-1)/2)
                self.assertEqual(answer.free_symbols, {n})
                self.check_original(source, answer, (0, 1), {0: 0}, index=n)

    def test_same_printed_index_and_parameter_never_become_the_same_symbol(self):
        n, y = self.n, self.y
        parameter = Symbol("n")
        self.assertNotEqual(n, parameter)
        source = y(n+1)-y(n)-parameter
        answer = rsolve(source, y(n), {y(0): 0})
        self.assertEqual(answer.free_symbols, {n, parameter})
        self.assert_zero(answer - n*parameter)
        self.check_original(source, answer, (0, 1), {0: 0})

        source = y(n+1)-2*y(n)
        answer = rsolve(source, y(n), {y(0): parameter})
        self.assertEqual(answer.free_symbols, {n, parameter})
        self.assert_zero(answer - parameter*2**n)
        self.check_original(source, answer, (0, 1), {0: parameter})

    def test_generated_constants_skip_source_and_initial_parameter_names(self):
        n, y = self.n, self.y
        c0, c1, c2 = Symbol("C0"), Symbol("C1"), Symbol("C2")
        source = y(n+1)-y(n)-c0-c1
        answer = rsolve(source, y(n))
        self.assertEqual(answer.free_symbols, {n, c0, c1, c2})
        self.assert_zero(answer - c2 - n*(c0+c1))
        self.check_original(source, answer, (0, 1))

        source = y(n+1)-2*y(n)-c0
        answer = rsolve(source, y(n), {y(0): 1})
        self.assertEqual(answer.free_symbols, {n, c0})
        self.assert_zero(answer - (c0+1)*2**n+c0)
        self.check_original(source, answer, (0, 1), {0: 1})

        source = y(n+1)-2*y(n)
        answer = rsolve(source, y(n), {0: c0})
        self.assertEqual(answer.free_symbols, {n, c0})
        self.assert_zero(answer - c0*2**n)
        self.check_original(source, answer, (0, 1), {0: c0})

    def test_foreign_function_and_noninteger_initial_keys_are_rejected(self):
        n, y = self.n, self.y
        other = Function("other_recurrence")
        invalid_keys = (other(0), y(Rational(1, 2)), Rational(1, 2), Float(0.0),
                        n, y(0, 1))
        source = y(n+1)-2*y(n)
        for key in invalid_keys:
            with self.subTest(key=key):
                with self.assertRaises((ValueError, TypeError)):
                    rsolve(source, y(n), {key: 1})

    def test_all_initial_conditions_are_checked_including_duplicate_indices(self):
        n, y = self.n, self.y
        source = y(n+1)-2*y(n)
        for initial in ({0: 3, y(0): 3}, {y(0): 3, y(1): 6, y(2): 12}):
            answer = rsolve(source, y(n), initial)
            self.assert_zero(answer - 3*2**n)
            self.check_original(source, answer, (0, 1), {0: 3, 1: 6, 2: 12})
        for initial in ({0: 1, y(0): 2}, {y(0): 1, y(1): 3},
                        {y(0): 1, y(1): 2, y(2): 5}):
            with self.subTest(initial=initial):
                with self.assertRaises(ValueError):
                    rsolve(source, y(n), initial)

    def test_factorial_initial_value_is_preserved(self):
        n, y = self.n, self.y
        answer = rsolve(y(n+1)-(n+1)*y(n), y(n), {0: 3})
        self.assertEqual(answer, 3*factorial(n))
        self.assertEqual(answer.subs(n, 0), 3)

    def test_unknown_parameter_branches_and_singular_leading_coefficients_refuse(self):
        n, y = self.n, self.y
        a = Symbol("a")
        for source in (a*y(n+1)-y(n), y(n+1)-a*y(n), n*y(n+1)-y(n),
                       y(n+2)+a*y(n+1)+y(n)):
            with self.subTest(source=source):
                # An unproved nonzero division is unsupported, not an empty
                # solution set or permission to discard a parameter branch.
                with self.assertRaises(NotImplementedError):
                    rsolve(source, y(n))

    def test_original_undefined_terms_survive_zero_products_and_zeroth_powers(self):
        n, y = self.n, self.y
        pole = Pow(n, -1, evaluate=False)
        factorial_pole = factorial(-1, evaluate=False)
        hidden_terms = (
            Mul(0, pole, evaluate=False),
            Add(Pow(pole, 0, evaluate=False), -1, evaluate=False),
            Mul(0, Pow(0, -1, evaluate=False), evaluate=False),
            Mul(0, factorial_pole, evaluate=False),
            Add(Pow(factorial_pole, 0, evaluate=False), -1, evaluate=False),
        )
        for hidden in hidden_terms:
            source = Add(y(n+1)-2*y(n), hidden, evaluate=False)
            with self.subTest(hidden=hidden):
                with self.assertRaises((NotImplementedError, ValueError)):
                    rsolve(source, y(n))

    def test_held_equality_checks_both_original_sides_before_subtraction(self):
        n, y = self.n, self.y
        for pole in (Pow(n, -1, evaluate=False), factorial(-1, evaluate=False)):
            source = Eq(Add(y(n+1), pole, evaluate=False),
                        Add(2*y(n), pole, evaluate=False), evaluate=False)
            with self.subTest(pole=pole):
                with self.assertRaises((NotImplementedError, ValueError)):
                    rsolve(source, y(n))

    def test_oversized_algebra_is_rejected_before_any_expansion(self):
        n, y = self.n, self.y
        base = Add(n, Symbol("a"), Symbol("b"), Symbol("c"), evaluate=False)
        multinomial = Pow(base, 64, evaluate=False)
        for forcing in (multinomial, exp(multinomial, evaluate=False),
                        factorial(1025, evaluate=False)):
            source = Add(y(n+1)-y(n), forcing, evaluate=False)
            with self.subTest(forcing=forcing), patch(
                "sympy.core.expand", side_effect=AssertionError("oversized source was expanded"),
            ) as expand_call:
                with self.assertRaises(NotImplementedError):
                    rsolve(source, y(n))
                self.assertEqual(expand_call.call_count, 0)

    def test_initial_values_cannot_refer_to_the_unknown_sequence_or_running_index(self):
        n, y = self.n, self.y
        cases = ((n, ValueError), (y(0), NotImplementedError),
                 (y(1), NotImplementedError), (y(n), NotImplementedError),
                 (n+y(0), NotImplementedError))
        for value, refusal in cases:
            with self.subTest(initial_value=value):
                with self.assertRaises(refusal):
                    rsolve(y(n+1)-2*y(n), y(n), {y(0): value})

    def test_candidate_components_keep_held_domains_exactness_and_explicit_values(self):
        n, y = self.n, self.y
        hidden = Mul(0, Pow(n, -1, evaluate=False), evaluate=False)
        candidates = (
            Add(n, hidden, evaluate=False),
            Add(n, Mul(0, factorial(-1, evaluate=False), evaluate=False), evaluate=False),
            Mul(Float(1.0), n, evaluate=False),
            Add(n, y(0), evaluate=False),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate), patch(
                "sympy.solvers.recurr._particular", return_value=candidate,
            ):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-y(n)-1, y(n))

    def test_initial_fit_values_are_admitted_before_rationalization_or_substitution(self):
        n, y = self.n, self.y
        hidden = Mul(0, Pow(n, -1, evaluate=False), evaluate=False)
        values = (Add(1, hidden, evaluate=False),
                  Add(1, Mul(0, factorial(-1, evaluate=False), evaluate=False), evaluate=False),
                  Float(1.0), y(0))
        for value in values:
            def forged_fit(equations, constants, **kwargs):
                return [{constants[0]: value}]

            with self.subTest(fitted_value=value), patch("sympy.solve", side_effect=forged_fit):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-2*y(n), y(n), {y(0): 1})

    def test_wrong_characteristic_root_cannot_publish_a_false_exact_solution(self):
        n, y = self.n, self.y
        # Zero is an additional domain adversary: 0**n cannot silently become
        # a valid two-sided integer sequence or erase the arbitrary constant.
        for root in (Integer(0), Integer(3), 2+Rational(1, 10**200)):
            with self.subTest(root=root), patch(
                "sympy.solvers.recurr._char_roots", return_value=[(root, 1)],
            ):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-2*y(n), y(n))

    def test_characteristic_root_proposals_cannot_duplicate_or_miscount_the_basis(self):
        n, y = self.n, self.y
        # The duplicate proposal would pass a residual-only check: C0+C1 is
        # a solution, but it silently loses the independent 2**n branch.
        second_order = y(n+2)-3*y(n+1)+2*y(n)
        proposals = (
            (second_order, [(Integer(1), 1), (Integer(1), 1)]),
            (second_order, [(Integer(1), 3), (Integer(2), -1)]),
            (y(n+1)-2*y(n), [(Integer(2), True)]),
            (y(n+1)-2*y(n), [(Integer(2), 1.0)]),
            (y(n+1)-2*y(n), [(Integer(2), 0)]),
            (y(n+1)-2*y(n), [(Integer(2), 10**30)]),
        )
        for source, proposed in proposals:
            with self.subTest(proposal=proposed), patch(
                "sympy.solvers.recurr._char_roots", return_value=proposed,
            ):
                with self.assertRaises(NotImplementedError):
                    rsolve(source, y(n))

    def test_wrong_particular_solution_is_rejected_against_original_forcing(self):
        n, y = self.n, self.y
        # The tiny exact perturbation has a nonzero residual despite being
        # indistinguishable from the correct coefficient in binary64.
        for wrong in (Integer(0), (1+Rational(1, 10**300))*n):
            with self.subTest(candidate=wrong), patch(
                "sympy.solvers.recurr._particular", return_value=wrong,
            ):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-y(n)-1, y(n))

    def test_wrong_initial_fit_cannot_publish_even_when_recurrence_is_satisfied(self):
        n, y = self.n, self.y

        for wrong in (Integer(17), 1+Rational(1, 10**300)):
            def false_fit(equations, constants, **kwargs):
                self.assertEqual(len(constants), 1)
                return [{constants[0]: wrong}]

            with self.subTest(initial_value=wrong), patch("sympy.solve", side_effect=false_fit) as fit:
                with self.assertRaises((NotImplementedError, ValueError)):
                    rsolve(y(n+1)-2*y(n), y(n), {y(0): 1})
                self.assertEqual(fit.call_count, 1)

    def test_memory_failure_from_candidate_generation_is_terminal(self):
        n, y = self.n, self.y
        with patch("sympy.solvers.recurr._particular", side_effect=MemoryError("recurrence allocation failed")):
            with self.assertRaisesRegex(MemoryError, "recurrence allocation failed"):
                rsolve(y(n+1)-y(n)-1, y(n))


if __name__ == "__main__":
    unittest.main()
