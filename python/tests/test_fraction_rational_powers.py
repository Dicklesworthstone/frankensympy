"""Surface fraction splitting of held inverse powers.

Expected pairs follow the isolated SymPy 1.14.0 oracle. They intentionally
test the outer fraction form, which differs from ``as_numer_denom``.
"""

import unittest
import sympy as sp


class FractionRationalPowersTests(unittest.TestCase):
    def test_inverse_ratio_keeps_the_whole_base(self):
        x, y = sp.symbols("x y")
        base = x / y
        for power in (1, 2, 3, 7):
            with self.subTest(power=power):
                source = sp.Pow(base, -power, evaluate=False)
                numerator, denominator = sp.fraction(source)
                self.assertEqual(numerator, 1)
                self.assertEqual(denominator, base ** power)
                self.assertEqual(denominator.subs({x: 2, y: 3}), sp.Rational(2, 3) ** power)
                self.assertEqual(numerator / denominator.subs({x: 2, y: 3}),
                                 sp.Rational(3, 2) ** power)

    def test_nested_inverse_retains_the_original_inner_denominator(self):
        x = sp.Symbol("x")
        base = sp.Pow(x, -1, evaluate=False)
        source = sp.Pow(base, -1, evaluate=False)
        numerator, denominator = sp.fraction(source)
        self.assertEqual(numerator, 1)
        self.assertIs(denominator, base)
        self.assertEqual(denominator.args, (x, sp.Integer(-1)))

    def test_held_ratio_base_is_not_recursively_flattened(self):
        x, y = sp.symbols("x y")
        inverse = sp.Pow(y, -1, evaluate=False)
        base = sp.Mul(x, inverse, evaluate=False)
        source = sp.Pow(base, -1, evaluate=False)
        numerator, denominator = sp.fraction(source)
        self.assertEqual(numerator, 1)
        self.assertIs(denominator, base)
        self.assertEqual(denominator.args, (x, inverse))
        self.assertEqual(denominator.free_symbols, {x, y})

    def test_nested_rational_bases_reconstruct_at_exact_points(self):
        x, y, z = sp.symbols("x y z")
        base = sp.Mul(x + 1, sp.Pow(y / z, -1, evaluate=False), evaluate=False)
        for power in (1, 2, 3):
            with self.subTest(power=power):
                source = sp.Pow(base, -power, evaluate=False)
                numerator, denominator = sp.fraction(source)
                self.assertEqual(numerator, 1)
                self.assertEqual(denominator, base ** power)
                self.assertEqual(denominator.subs({x: 2, y: 5, z: 7}),
                                 sp.Rational(21, 5) ** power)

    def test_fractional_inverse_keeps_principal_power_of_unsplit_ratio(self):
        x, y = sp.symbols("x y")
        base = x / y
        for power in (sp.Rational(1, 2), sp.Rational(2, 3)):
            with self.subTest(power=power):
                source = sp.Pow(base, -power, evaluate=False)
                numerator, denominator = sp.fraction(source)
                self.assertEqual(numerator, 1)
                self.assertEqual(denominator, sp.Pow(base, power))
                self.assertIsInstance(denominator, sp.Pow)
                self.assertEqual(denominator.args, (base, power))

    def test_principal_square_root_does_not_split_negative_base_factors(self):
        x, y = sp.symbols("x y")
        source = sp.Pow(x / y, sp.Rational(-1, 2), evaluate=False)
        numerator, denominator = sp.fraction(source)
        self.assertEqual(numerator, 1)
        self.assertEqual(denominator.subs({x: -1, y: -4}), sp.Rational(1, 2))

    def test_noncommutative_inverse_stays_in_numerator(self):
        a, b = sp.symbols("A B", commutative=False)
        base = sp.Mul(a, b, evaluate=False)
        for power in (-1, -2, sp.Rational(-1, 2)):
            with self.subTest(power=power):
                source = sp.Pow(base, power, evaluate=False)
                numerator, denominator = sp.fraction(source)
                self.assertIs(numerator, source)
                self.assertEqual(denominator, 1)
                self.assertEqual(numerator.args, source.args)

    def test_scalar_denominator_does_not_invert_a_noncommutative_power(self):
        a = sp.Symbol("A", commutative=False)
        x = sp.Symbol("x")
        inverse = sp.Pow(a, -1, evaluate=False)
        source = sp.Mul(inverse, sp.Pow(x, -1, evaluate=False), evaluate=False)
        numerator, denominator = sp.fraction(source)
        self.assertEqual(numerator, inverse)
        self.assertEqual(denominator, x)

    def test_nonnegative_powers_remain_outer_numerator_factors(self):
        x, y = sp.symbols("x y")
        for power in (0, 2, sp.Rational(1, 2)):
            with self.subTest(power=power):
                source = sp.Pow(x / y, power, evaluate=False)
                self.assertEqual(sp.fraction(source), (source, sp.Integer(1)))

    def test_unknown_symbolic_power_remains_unchanged(self):
        x, y, a = sp.symbols("x y a")
        source = sp.Pow(x / y, a, evaluate=False)
        self.assertEqual(sp.fraction(source), (source, sp.Integer(1)))


if __name__ == "__main__":
    unittest.main()
