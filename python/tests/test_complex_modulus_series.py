"""Complex absolute values through the real native series and limit boundary."""

import unittest

from sympy import (
    Abs, Add, I, Limit, Mul, Pow, Rational, Symbol, expand, limit, oo, pi,
    series, simplify, tan,
)


class ComplexModulusSeriesTests(unittest.TestCase):
    def setUp(self):
        self.x = Symbol("x")

    def assert_zero(self, expression):
        self.assertEqual(simplify(expand(expression)), 0)

    def test_nonreal_tails_produce_the_real_modulus_series_on_both_rays(self):
        x = self.x
        for source in (Abs(1 + I*x), Abs(1 - I*x), Abs(-1 + I*x)):
            for direction in ("+", "-"):
                with self.subTest(source=source, direction=direction):
                    result = series(source, x, 0, 5, dir=direction)
                    self.assert_zero(result.removeO() - (1 + x**2/2 - x**4/8))

    def test_complex_leading_values_zeros_and_poles_keep_exact_coefficients(self):
        x = self.x
        for source, expected in (
            (Abs(I*x), x),
            (Abs((3 + 4*I)/x), 5/x),
            (Abs(I*x + x**2), x + x**3/2),
            (Abs(3 + 4*I + (1 - 2*I)*x), 5 - x + 2*x**2/5 + 2*x**3/25),
        ):
            with self.subTest(source=source):
                result = series(source, x, 0, 4)
                self.assert_zero(result.removeO() - expected)

    def test_modulus_cancellation_limits_are_exact_at_zero_and_infinity(self):
        x = self.x
        for direction in ("+", "-"):
            self.assertEqual(limit((Abs(1 + I*x) - 1)/x, x, 0, dir=direction), 0)
            self.assertEqual(
                limit((Abs(1 + I*x) - 1)/x**2, x, 0, dir=direction), Rational(1, 2),
            )
        self.assertEqual(limit(x*(Abs(x + I) - x), x, oo), Rational(1, 2))

    def test_parameter_reality_is_explicit_and_scoped_to_its_typed_identity(self):
        x = self.x
        real = Symbol("a", real=True)
        unknown = Symbol("a")
        result = series(Abs(1 + real*x), x, 0, 3)
        self.assert_zero(result - (1 + real*x))
        for source in (Abs(1 + unknown*x), Abs(1 + unknown*x**20)):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    series(source, x, 0, 3)

    def test_held_parameter_poles_are_checked_before_zero_cancellation(self):
        x = self.x
        a = Symbol("a", real=True)
        for pole in (
            Pow(a, -1, evaluate=False),
            tan(a, evaluate=False),
            tan(pi/2, evaluate=False),
        ):
            source = Abs(
                Add(1, Mul(0, pole, x, evaluate=False), evaluate=False),
                evaluate=False,
            )
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    series(source, x, 0, 3)
                self.assertIsInstance(limit(source, x, 0), Limit)


if __name__ == "__main__":
    unittest.main()
