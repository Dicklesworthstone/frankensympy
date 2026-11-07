"""Independent closed forms for bounded factorial-forced first-order recurrences.

Expected sums are explicit arithmetic/geometric identities, never computed by
the recurrence implementation or its summation provider. Exact residuals are
checked in the original, possibly shifted equation and at every supplied IC.
The factorial quotient used only by the test comparator is nonzero wherever
the tested factorial expressions are defined; no numerical sampling is used.
"""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Dummy, Eq, Float, Function, Integer, Mul, Pow, Rational, Symbol,
    combsimp, exp, expand, expand_power_exp, factorial, rsolve, simplify,
)


class RecurrenceForcingTests(unittest.TestCase):
    def setUp(self):
        self.n = Symbol("n", integer=True)
        self.y = Function("factorial_forced_y")

    def assert_zero(self, expression, index=None):
        n = self.n if index is None else index
        normalized = expand(expand_power_exp(expression/factorial(n)))
        self.assertEqual(simplify(combsimp(normalized)), 0)

    def check(self, source, expected, *, shifts=(0, 1), initial=None, index=None):
        n = self.n if index is None else index
        data = None if initial is None else {self.y(point): value for point, value in initial.items()}
        answer = rsolve(source, self.y(n), data)
        self.assertIsNotNone(answer)
        self.assertEqual(answer.free_symbols, expected.free_symbols)
        self.assert_zero(answer-expected, n)
        replacements = {self.y(n+shift): answer.subs(n, n+shift) for shift in shifts}
        residual = (source.lhs.xreplace(replacements)-source.rhs.xreplace(replacements)
                    if type(source) is Eq else source.xreplace(replacements))
        self.assert_zero(residual, n)
        for point, value in (initial or {}).items():
            self.assertEqual(simplify(answer.subs(n, point)-value), 0)
        return answer

    def test_unit_quotient_adds_one_each_step_to_the_factorial_normalized_sequence(self):
        n, y, c0 = self.n, self.y, Symbol("C0")
        source = y(n+1)-(n+1)*y(n)-factorial(n+1)
        self.check(source, (c0+n)*factorial(n))
        self.check(source, (3+n)*factorial(n), initial={0: 3})

    def test_polynomial_quotients_have_the_hand_derived_faulhaber_sums(self):
        n, y = self.n, self.y
        cases = (
            ((n+1)*factorial(n+1), n*(n+1)*factorial(n)/2),
            (n**2*factorial(n+1), n*(n-1)*(2*n-1)*factorial(n)/6),
            (factorial(n+2), n*(n+3)*factorial(n)/2),
            ((n+1)*factorial(n), n*factorial(n)),
            ((Rational(2, 3)*n-Rational(1, 5))*factorial(n+1),
             (n*(n-1)/3-n/5)*factorial(n)),
        )
        for forcing, expected in cases:
            with self.subTest(forcing=forcing):
                self.check(y(n+1)-(n+1)*y(n)-forcing, expected, initial={0: 0})

    def test_geometric_and_polynomial_geometric_quotients_keep_exact_boundaries(self):
        n, y = self.n, self.y
        cases = (
            (2**n*factorial(n+1), (2**n-1)*factorial(n)),
            (Rational(3, 2)**n*factorial(n+1), 2*(Rational(3, 2)**n-1)*factorial(n)),
            ((-1)**n*factorial(n+1), (1-(-1)**n)*factorial(n)/2),
            (n*2**n*factorial(n+1), ((n-2)*2**n+2)*factorial(n)),
        )
        for forcing, expected in cases:
            with self.subTest(forcing=forcing):
                self.check(y(n+1)-(n+1)*y(n)-forcing, expected, initial={0: 0})

    def test_signed_and_rational_linear_coefficients_preserve_their_scale(self):
        n, y = self.n, self.y
        cases = (
            (2, 1, 2**(n+1)*factorial(n+1), (n+5)*2**n*factorial(n), 5),
            (-2, 2, (-2)**(n+1)*factorial(n+2), (n-3)*(-2)**n*factorial(n+1), -3),
            (Rational(1, 2), 1, Rational(1, 2)**(n+1)*factorial(n+1),
             (n+Rational(3, 5))*Rational(1, 2)**n*factorial(n), Rational(3, 5)),
            (Rational(3, 2), 3, Rational(3, 2)**(n+1)*factorial(n+3)/2,
             (n-4)*Rational(3, 2)**n*factorial(n+2)/2, -4),
        )
        for scale, offset, forcing, expected, value in cases:
            with self.subTest(scale=scale, offset=offset):
                self.check(y(n+1)-scale*(n+offset)*y(n)-forcing, expected, initial={0: value})

    def test_positive_shifted_factorial_normalization_includes_its_initial_factor(self):
        n, y = self.n, self.y
        source = y(n+1)-(n+3)*y(n)-(n+1)*factorial(n+3)/2
        expected = (2+n*(n+1)/2)*factorial(n+2)/2
        self.check(source, expected, initial={0: 2})

    def test_rational_leading_coefficient_is_applied_to_the_original_forcing(self):
        n, y = self.n, self.y
        source = Eq(2*y(n+1), 2*(n+1)*y(n)+3*factorial(n+1), evaluate=False)
        self.check(source, (4+Rational(3, 2)*n)*factorial(n), initial={0: 4})

    def test_positive_and_negative_equation_shifts_keep_the_original_index(self):
        n, y = self.n, self.y
        for source, shifts in (
            (y(n+2)-(n+2)*y(n+1)-factorial(n+2), (1, 2)),
            (y(n)-n*y(n-1)-factorial(n), (-1, 0)),
        ):
            with self.subTest(source=source):
                self.check(source, (n+2)*factorial(n), shifts=shifts, initial={0: 2})

    def test_nonzero_initial_index_and_all_extra_conditions_are_verified(self):
        n, y = self.n, self.y
        source = y(n+1)-(n+1)*y(n)-factorial(n+1)
        self.check(source, factorial(n+1), initial={3: 24})
        self.check(source, factorial(n+1), initial={0: 1, 1: 2, 3: 24})
        with self.assertRaises(ValueError):
            rsolve(source, y(n), {y(0): 1, y(1): 3})

    def test_generated_factorial_domains_precede_initial_substitution_and_zero_cancellation(self):
        n, y = self.n, self.y
        source = y(n+1)-(n+1)*y(n)-factorial(n+1)
        for initial in ({y(-1): 0}, {y(0): 1, y(-1): 0}, {y(-2): 1}):
            with self.subTest(initial=initial):
                # Fitting y(0)=1 would give (n+1)*factorial(n), whose
                # coefficient vanishes at -1. A singular-point extension
                # is not licensed by erasing the retained factorial pole.
                with self.assertRaises(NotImplementedError):
                    rsolve(source, y(n), initial)

        # A negative index itself is not a pole: factorial(n+2) is defined
        # at -1, so the guard must inspect the actual factorial argument.
        source = y(n+1)-(n+3)*y(n)-factorial(n+3)/2
        self.check(source, (n+15)*factorial(n+2)/2, initial={-1: 7})

    def test_forcing_and_initial_parameters_do_not_capture_the_free_constant(self):
        n, y, c0, c1 = self.n, self.y, Symbol("C0"), Symbol("C1")
        source = y(n+1)-(n+1)*y(n)-c0*factorial(n+1)
        self.check(source, (c1+c0*n)*factorial(n))
        source = y(n+1)-(n+1)*y(n)-factorial(n+1)
        self.check(source, (c0+n)*factorial(n), initial={0: c0})

    def test_typed_index_dummy_and_reserved_constant_names_remain_distinct(self):
        y = self.y
        for n in (Dummy("n", integer=True), Symbol("_fsym_rs_k", integer=True),
                  Symbol("C0", integer=True)):
            constant = Symbol("C1") if n.name == "C0" else Symbol("C0")
            source = y(n+1)-(n+1)*y(n)-factorial(n+1)
            with self.subTest(index=n):
                self.check(source, (constant+n)*factorial(n), index=n)

        n, parameter = self.n, Symbol("n")
        source = y(n+1)-(n+1)*y(n)-parameter*factorial(n+1)
        self.check(source, n*parameter*factorial(n), initial={0: 0})

    def test_singular_and_nonliteral_factorial_offsets_are_honest_refusals(self):
        n, y = self.n, self.y
        for offset in (0, -1, -4, Rational(1, 2), Symbol("c", integer=True, positive=True)):
            with self.subTest(offset=offset):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-(n+offset)*y(n)-factorial(n+1), y(n))

    def test_unknown_inexact_or_nonlinear_multiplier_branches_refuse(self):
        n, y = self.n, self.y
        for scale in (Symbol("a"), Symbol("a", nonzero=True), Float(2.0)):
            with self.subTest(scale=scale):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-scale*(n+1)*y(n)-factorial(n+1), y(n))
        for source in (
            y(n+1)-(n+1)**2*y(n)-factorial(n+1),
            n*y(n+1)-(n+1)*y(n)-factorial(n+1),
            Add(y(n+1), Mul(0, n+1, y(n), evaluate=False), -factorial(n+1), evaluate=False),
        ):
            with self.subTest(source=source):
                with self.assertRaises(NotImplementedError):
                    rsolve(source, y(n))

    def test_unsummed_quotient_is_not_returned_as_a_closed_solution(self):
        n, y = self.n, self.y
        for forcing in (exp(n*n)*factorial(n+1), factorial(n)):
            with self.subTest(forcing=forcing):
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-(n+1)*y(n)-forcing, y(n))

    def test_normalized_degree_and_factorial_distance_caps_precede_summation(self):
        from sympy.core import _native, _wrap

        n, y = self.n, self.y
        for forcing in (n**33*factorial(n+1), factorial(n+66)):
            with self.subTest(forcing=forcing), patch.object(
                _native, "summation_native", side_effect=AssertionError("unadmitted forcing reached summation"),
            ) as proposed:
                with self.assertRaises(NotImplementedError):
                    rsolve(y(n+1)-(n+1)*y(n)-forcing, y(n))
                self.assertEqual(proposed.call_count, 0)

        # The affine slope is below 1024, but forming this exact rational
        # power would exceed the separate 4096-bit scalar-work envelope.
        base = Integer(2**100)
        forcing = Pow(base, 100*n, evaluate=False)*factorial(n+1)
        native_power = _native.py_pow

        def reject_oversized_scalar(b, e):
            scalar, exponent = _wrap(b), _wrap(e)
            if scalar == base and exponent == 100:
                raise AssertionError("oversized rational scalar was materialized")
            return native_power(b, e)

        with patch.object(_native, "py_pow", side_effect=reject_oversized_scalar):
            with self.assertRaises(NotImplementedError):
                rsolve(y(n+1)-(n+1)*y(n)-forcing, y(n))

    def test_forcing_keeps_original_undefined_factorial_terms(self):
        n, y = self.n, self.y
        hidden = Mul(0, factorial(-1, evaluate=False), evaluate=False)
        source = Add(y(n+1)-(n+1)*y(n)-factorial(n+1), hidden, evaluate=False)
        with self.assertRaises(NotImplementedError):
            rsolve(source, y(n))

    def test_false_native_summation_candidates_fail_original_recurrence_replay(self):
        from sympy.core import _native, _native_expr

        n, y = self.n, self.y
        source = y(n+1)-(n+1)*y(n)-factorial(n+1)
        for wrong in (Integer(0), n*(1+Rational(1, 10**200)), n+Dummy("unbound_sum_index")):
            with self.subTest(candidate=wrong), patch.object(
                _native, "summation_native", return_value=_native_expr(wrong),
            ) as proposed:
                with self.assertRaises(NotImplementedError):
                    rsolve(source, y(n))
                self.assertGreater(proposed.call_count, 0)

    def test_summation_operational_failure_is_terminal(self):
        from sympy.core import _native

        n, y = self.n, self.y
        source = y(n+1)-(n+1)*y(n)-factorial(n+1)
        with patch.object(_native, "summation_native", side_effect=MemoryError("sum allocation failed")) as proposed:
            with self.assertRaisesRegex(MemoryError, "sum allocation failed"):
                rsolve(source, y(n))
            self.assertEqual(proposed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
