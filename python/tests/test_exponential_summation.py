"""Polynomial-weighted factorial series through the actual native extension."""

import unittest

from sympy import (
    Add, Dummy, E, I, Mul, Pow, Rational, Sum, Symbol, exp, expand,
    factorial, oo, simplify, summation,
)


class ExponentialSummationTests(unittest.TestCase):
    def setUp(self):
        self.k = Symbol("factorial_index", integer=True)
        self.z = Symbol("factorial_parameter")

    def assert_zero(self, expression):
        self.assertEqual(simplify(expand(expression)), 0)

    def test_bare_factorial_series_and_sum_doit_return_the_exact_constant(self):
        k = self.k
        self.assertEqual(summation(1/factorial(k), (k, 0, oo)), E)
        self.assertEqual(Sum(1/factorial(k), (k, 0, oo)).doit(), E)
        self.assert_zero(summation(1/factorial(k), (k, 3, oo)) - E + Rational(5, 2))

    def test_polynomial_moments_are_entire_functions_of_the_parameter(self):
        k, z = self.k, self.z
        moments = (
            1,
            z,
            z**2 + z,
            z**3 + 3*z**2 + z,
            z**4 + 6*z**3 + 7*z**2 + z,
            z**5 + 10*z**4 + 25*z**3 + 15*z**2 + z,
            z**6 + 15*z**5 + 65*z**4 + 90*z**3 + 31*z**2 + z,
        )
        for degree, polynomial in enumerate(moments):
            with self.subTest(degree=degree):
                answer = summation(k**degree*z**k/factorial(k), (k, 0, oo))
                self.assertNotIsInstance(answer, Sum)
                self.assert_zero(answer - polynomial*exp(z))
                self.assertEqual(answer.free_symbols, {z})

    def test_rational_weights_and_nonzero_lower_limits_keep_every_prefix_term(self):
        k, z = self.k, self.z
        weight = Rational(5, 7) - Rational(3, 5)*k + Rational(2, 3)*k**2
        expected = (Rational(5, 7) + z/15 + 2*z**2/3)*exp(z)
        expected -= Rational(5, 7) + Rational(82, 105)*z + Rational(229, 210)*z**2
        self.assert_zero(summation(weight*z**k/factorial(k), (k, 3, oo)) - expected)

    def test_signed_rational_and_complex_parameters_are_not_geometric_series(self):
        k = self.k
        for parameter in (-3, Rational(2, 3), 4, I):
            with self.subTest(parameter=parameter):
                source = (k+1)*Pow(parameter, k, evaluate=False)/factorial(k)
                answer = summation(source, (k, 2, oo))
                self.assert_zero(answer - (parameter+1)*exp(parameter) + 1 + 2*parameter)

    def test_zero_parameter_has_only_the_discrete_zero_index_term(self):
        k = self.k
        source = (7+2*k)*Pow(0, k, evaluate=False)/factorial(k)
        self.assertEqual(summation(source, (k, 0, oo)), 7)
        self.assertEqual(summation(source, (k, 1, oo)), 0)

    def test_typed_symbols_and_dummies_do_not_capture_same_spelling_parameters(self):
        pairs = (
            (Symbol("same", integer=True), Symbol("same", real=True)),
            (Dummy("same", integer=True), Dummy("same")),
        )
        for index, parameter in pairs:
            with self.subTest(index=index, parameter=parameter):
                answer = summation(index*parameter**index/factorial(index), (index, 0, oo))
                self.assert_zero(answer - parameter*exp(parameter))
                self.assertEqual(answer.free_symbols, {parameter})

    def test_held_undefined_children_and_excessive_work_do_not_disappear(self):
        k, z = self.k, self.z
        basic = z**k/factorial(k)
        for invalid in (
            Pow(0, -1, evaluate=False),
            factorial(-1, evaluate=False),
            Pow(Add(1, -1, evaluate=False), -1, evaluate=False),
            Pow(k, 33, evaluate=False),
        ):
            for erased in (Mul(0, invalid, evaluate=False), Pow(invalid, 0, evaluate=False)):
                source = Mul(erased, basic, evaluate=False)
                with self.subTest(invalid=invalid, erased=erased):
                    with self.assertRaises(ValueError):
                        summation(source, (k, 0, oo))
        with self.assertRaises(ValueError):
            summation(basic, (k, 257, oo))

    def test_unsupported_weights_bounds_and_factorial_powers_remain_unevaluated(self):
        k, z = self.k, self.z
        lower = Symbol("symbolic_lower", nonnegative=True, integer=True)
        sources = (
            (z**k/factorial(k), lower),
            (z**k/factorial(k)**2, 0),
            (z**k/((k+1)*factorial(k)), 0),
        )
        for source, first in sources:
            with self.subTest(source=source, first=first):
                self.assertIsInstance(summation(source, (k, first, oo)), Sum)

    def test_finite_ranges_keep_the_existing_direct_sum(self):
        k, z = self.k, self.z
        answer = summation(k**2*z**k/factorial(k), (k, 0, 3))
        self.assert_zero(answer - z - 2*z**2 - Rational(3, 2)*z**3)

    def test_declared_nonfinite_parameters_keep_infinite_sums_unevaluated(self):
        k = self.k
        for parameter in (
            Symbol("nonfinite_parameter", finite=False),
            Symbol("nonfinite_parameter", infinite=True),
        ):
            with self.subTest(parameter=parameter):
                source = k*Pow(parameter, k, evaluate=False)/factorial(k)
                answer = summation(source, (k, 0, oo))
                self.assertIsInstance(answer, Sum)
                self.assertEqual(answer.function, source)
                self.assertEqual(answer.free_symbols, {parameter})
                self.assertIsInstance(Sum(source, (k, 0, oo)).doit(), Sum)
                finite = summation(source, (k, 0, 2))
                self.assert_zero(finite - parameter - parameter**2)

        # A different same-spelling identity cannot inherit the refusal.
        finite = Symbol("nonfinite_parameter", finite=True)
        self.assert_zero(
            summation(k*finite**k/factorial(k), (k, 0, oo)) - finite*exp(finite),
        )

    def test_nonfinite_refusal_retains_original_held_undefined_children(self):
        k = self.k
        parameter = Symbol("held_nonfinite_parameter", infinite=True)
        source = Mul(
            0, Pow(0, -1, evaluate=False), Pow(parameter, k, evaluate=False),
            Pow(factorial(k), -1, evaluate=False), evaluate=False,
        )
        answer = summation(source, (k, 0, oo))
        self.assertIsInstance(answer, Sum)
        self.assertEqual(answer.function, source)


if __name__ == "__main__":
    unittest.main()
