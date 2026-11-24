"""Checked finite binomial moments through Sum and the real native bridge."""

import math
import unittest
from fractions import Fraction
from unittest.mock import patch

from sympy import (
    Add, Derivative, Dummy, Function, Integral, Mul, Pow, Rational, Sum, Symbol,
    binomial, oo, sin, summation,
)
try:
    from sympy.core import _native, _native_expr
except ImportError:
    _native = None
    _native_expr = None


class BinomialSummationTests(unittest.TestCase):
    def setUp(self):
        self.k = Symbol("binomial_index", integer=True)

    def test_long_sums_and_first_two_moments(self):
        k, n = self.k, 3000
        self.assertEqual(summation(binomial(n, k), (k, 0, n)), 2**n)
        self.assertEqual(Sum(k*binomial(n, k), (k, 0, n)).doit(), n*2**(n-1))
        self.assertEqual(summation(k**2*binomial(n, k), (k, 0, n)), n*(n+1)*2**(n-2))
        weighted = binomial(n, k)*Rational(1, 3)**k*Rational(2, 3)**(n-k)
        self.assertEqual(summation(weighted, (k, 0, n)), 1)
        self.assertEqual(summation(k*weighted, (k, 0, n)), 1000)

    def test_rational_weights_match_independent_python_finite_enumeration(self):
        k = self.k
        for n in (0, 2, 5, 8):
            for a, b in ((2, 3), (-2, 2), (0, 0), (0, 3), (2, 0), (Fraction(-1, 2), Fraction(2, 3))):
                with self.subTest(n=n, a=a, b=b):
                    a, b = Fraction(a), Fraction(b)
                    weight = Rational(5, 7) - Rational(3, 5)*k + Rational(2, 3)*k**6
                    source = Mul(
                        weight, binomial(n, k, evaluate=False),
                        Pow(Rational(a.numerator, a.denominator), k, evaluate=False),
                        Pow(Rational(b.numerator, b.denominator), n-k, evaluate=False),
                        evaluate=False,
                    )
                    expected = sum(
                        (Fraction(5, 7)-Fraction(3, 5)*j+Fraction(2, 3)*j**6)
                        * math.comb(n, j) * a**j * b**(n-j)
                        for j in range(n+1)
                    )
                    self.assertEqual(Sum(source, (k, 0, n)).doit(), Rational(expected.numerator, expected.denominator))

    def test_alternating_moments_and_degree_above_upper_bound(self):
        k = self.k
        for n, degree, expected in ((5, 4, 0), (5, 5, 120), (3, 6, 540), (0, 0, 1), (0, 6, 0)):
            with self.subTest(n=n, degree=degree):
                source = Mul(k**degree, binomial(n, k, evaluate=False), Pow(-1, n-k, evaluate=False), evaluate=False)
                self.assertEqual(summation(source, (k, 0, n)), expected)

    @unittest.skipIf(_native is None, "requires native original-source admission")
    def test_original_source_reaches_exactly_one_native_sum_call(self):
        k = self.k
        source = Mul(Add(k, k, evaluate=False), binomial(3000, k), evaluate=False)
        original = _native_expr(source)
        with patch.object(_native, "summation_native", wraps=_native.summation_native) as checked:
            answer = Sum(source, (k, 0, 3000)).doit()
        self.assertEqual(checked.call_count, 1)
        self.assertEqual(checked.call_args.args[0], original)
        self.assertEqual(answer, 3000*2**3000)

    @unittest.skipIf(_native is None, "requires native original-source admission")
    def test_original_poles_and_excessive_degree_survive_held_zero_operations(self):
        k = self.k
        for invalid in (Pow(0, -1, evaluate=False), Pow(Add(1, -1, evaluate=False), -1, evaluate=False), Pow(k, 33, evaluate=False), oo):
            for erased in (Mul(0, invalid, evaluate=False), Pow(invalid, 0, evaluate=False)):
                source = Mul(erased, binomial(3000, k), evaluate=False)
                with self.subTest(invalid=invalid, erased=erased):
                    with self.assertRaises(ValueError):
                        Sum(source, (k, 0, 3000)).doit()

    @unittest.skipIf(_native is None, "requires native original-source admission")
    def test_symbolic_bounds_unsupported_weights_and_typed_symbols_stay_held(self):
        k = self.k
        n = Symbol("binomial_n", integer=True, nonnegative=True)
        sources = (
            (binomial(n, k), (k, 0, n)),
            (Mul(Pow(sin(k), 0, evaluate=False), binomial(5, k), evaluate=False), (k, 0, 5)),
            (Mul(Symbol("binomial_index", real=True), binomial(5, k), evaluate=False), (k, 0, 5)),
        )
        for source, limits in sources:
            with self.subTest(source=source):
                answer = Sum(source, limits).doit()
                self.assertIsInstance(answer, Sum)
                self.assertEqual(answer.function, source)
        dummy = Dummy("binomial_index", integer=True)
        self.assertEqual(summation(dummy*binomial(3000, dummy), (dummy, 0, 3000)), 3000*2**2999)

    @unittest.skipIf(_native is None, "requires native original-source admission")
    def test_opaque_children_cannot_be_evaluated_away_before_binomial_admission(self):
        k = self.k
        children = (
            Derivative(Function("binomial_opaque_child")(k), k, evaluate=False),
            Integral(1/k, (k, 0, 1)),
        )
        for child in children:
            for erased in (Mul(0, child, evaluate=False), Pow(child, 0, evaluate=False)):
                source = Mul(erased, binomial(5, k), evaluate=False)
                held = Sum(source, (k, 0, 5))
                with self.subTest(child=child, erased=erased):
                    self.assertIs(held.doit(), held)
                    self.assertIs(held.doit(deep=False), held)

    def test_unrelated_sums_keep_deep_evaluation(self):
        k = self.k
        self.assertEqual(Sum(Derivative(k**2, k, evaluate=False), (k, 1, 3)).doit(), 12)

    def test_partial_ranges_and_squared_binomials_keep_existing_results(self):
        k = self.k
        self.assertEqual(summation(binomial(5, k), (k, 1, 5)), 31)
        self.assertEqual(summation(binomial(5, k), (k, 0, 3)), 26)
        self.assertEqual(summation(binomial(5, k)**2, (k, 0, 5)), 252)

    @unittest.skipIf(_native is None, "requires native callback accounting")
    def test_active_callback_cannot_restart_an_intrinsic_binomial_budget(self):
        from sympy.core import SupervisedCallbackSession, supervised_callback_session

        k = self.k
        held = Sum(binomial(3000, k), (k, 0, 3000))
        session = SupervisedCallbackSession(max_nested_steps=20_000_000)
        with supervised_callback_session(session):
            with patch.object(_native, "summation_native", wraps=_native.summation_native) as checked:
                with self.assertRaisesRegex(NotImplementedError, "supervised callback lane"):
                    held.doit()
                checked.assert_not_called()

    @unittest.skipIf(_native is None, "requires native callback accounting")
    def test_callback_checks_candidates_created_by_an_earlier_sum_limit(self):
        from sympy.core import SupervisedCallbackSession, supervised_callback_session

        k = self.k
        j = Symbol("binomial_outer_index", integer=True)
        held = Sum(binomial(j+5, k), (j, 0, 0), (k, 0, 5))
        session = SupervisedCallbackSession(max_nested_steps=20_000_000)
        with supervised_callback_session(session):
            with patch.object(_native, "summation_native", wraps=_native.summation_native) as checked:
                with self.assertRaisesRegex(NotImplementedError, "supervised callback lane"):
                    held.doit()
                # The first legacy call substitutes j=0. The newly selected
                # binomial(5,k) must be refused before the second native call.
                self.assertEqual(checked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
