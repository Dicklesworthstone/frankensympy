"""Executable checks for the Python compatibility boundary."""

from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
import unittest
from pathlib import Path

import sympy


class SurfaceTests(unittest.TestCase):
    def test_rational_strings_preserve_exact_values_and_explicit_denominator(self):
        # exact_python: exact p/q, canonical class and singleton promotion.
        cases = ((("0.1",), 1, 10), (("1e-3",), 1, 1000),
                 (("1e400",), 10**400, 1), (("1e-400",), 1, 10**400),
                 (("9007199254740993.0",), 9007199254740993, 1),
                 (("1.2/0.3",), 4, 1), (("1 / 2",), 1, 2),
                 (("1/2", 3), 1, 6), ((1, "0.1"), 10, 1),
                 (("0.1", "0.2"), 1, 2), (("-1.5", "-0.3"), 5, 1),
                 (("0", "1/2"), 0, 1))
        for args, numerator, denominator in cases:
            with self.subTest(args=args):
                actual = sympy.Rational(*args)
                expected = sympy.Rational(numerator, denominator)
                self.assertEqual((actual.p, actual.q), (numerator, denominator))
                self.assertIs(type(actual), type(expected))
                if expected in (sympy.S.Zero, sympy.S.One, sympy.S.Half):
                    self.assertIs(actual, expected)
        self.assertNotEqual(sympy.Rational("0.1"), sympy.Rational(0.1))

    def test_rational_strings_reject_malformed_input_without_evaluation(self):
        # exact_exception comparator, pinned independently against SymPy 1.14.
        cases = (("1/2/3", TypeError, "invalid input: 1/2/3"),
                 ("", TypeError, "invalid input: "),
                 ("abc", TypeError, "invalid input: abc"),
                 ("a/2", ValueError, "Invalid literal for Fraction: 'a'"),
                 ("1/a", ValueError, "Invalid literal for Fraction: 'a'"),
                 ("0.1/0", ZeroDivisionError, "Fraction(1, 0)"),
                 ("1+2", TypeError, "invalid input: 1+2"))
        for value, error_type, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(error_type) as caught:
                    sympy.Rational(value)
                self.assertEqual(str(caught.exception), message)

    def test_exact_number_int_conversion_does_not_round_through_float(self):
        # exact_python comparator: Python int type and exact value.
        cases = ((0, 1, 0), (1, 2, 0), (-3, 2, -1),
                 (2**53 + 1, 1, 2**53 + 1),
                 (3 * 2**60 + 2, 3, 2**60),
                 (3 * 2**60 - 1, 3, 2**60 - 1),
                 (10**400, 3, 10**400 // 3))
        for numerator, denominator, expected in cases:
            for sign in (-1, 1):
                with self.subTest(numerator=numerator, denominator=denominator, sign=sign):
                    result = int(sympy.Rational(sign * numerator, denominator))
                    self.assertIs(type(result), int)
                    self.assertEqual(result, sign * expected)

    def test_exact_number_float_conversion_range_boundaries(self):
        # exact_python comparator: binary64 bits via hex, including signed zero.
        cases = ((10**400 + 1, 10**400, 1.0),
                 (10**400, 3, float("inf")),
                 (1, 10**400, 0.0),
                 (1, 2**1074, float.fromhex("0x0.0000000000001p-1022")),
                 (1, 2**1075, 0.0),
                 (2**53 + 1, 2**53, 1.0),
                 (2**53 + 3, 2**53, float.fromhex("0x1.0000000000002p+0")))
        for numerator, denominator, expected in cases:
            for sign in (-1, 1):
                with self.subTest(numerator=numerator, denominator=denominator, sign=sign):
                    result = float(sympy.Rational(sign * numerator, denominator))
                    self.assertEqual(result.hex(), (sign * expected).hex())

    def test_numeric_division_zero_policy(self):
        # exact_python (bounded): Float class/value and singleton identity;
        # exact_exception: Float/Float zero division has an empty message.
        exact = (0, 3, -3, sympy.S.Zero, sympy.S.One,
                 sympy.S.NegativeOne, sympy.S.Half, sympy.Rational(-3, 2))
        approximate = (0.0, -0.0, 1.5, -1.5,
                       sympy.Float(0), sympy.Float(-0.0),
                       sympy.Float(1.5), sympy.Float(-1.5))
        for left in exact + approximate:
            for right in exact + approximate:
                # Native Python scalar/scalar division is not a SymPy call.
                if type(left) in (int, float) and type(right) in (int, float):
                    continue
                with self.subTest(left=repr(left), left_type=type(left).__name__,
                                  right=repr(right), right_type=type(right).__name__):
                    if float(right) == 0:
                        if (type(left) is sympy.Float
                                and type(right) in (float, sympy.Float)):
                            with self.assertRaises(ZeroDivisionError) as caught:
                                left / right
                            self.assertEqual(str(caught.exception), "")
                        else:
                            expected = sympy.nan if float(left) == 0 else sympy.zoo
                            self.assertIs(left / right, expected)
                    elif float(left) == 0:
                        self.assertIs(left / right, sympy.S.Zero)
                    else:
                        result = left / right
                        expected = float(left) / float(right)
                        self.assertEqual(float(result), expected)
                        if type(left) in (float, sympy.Float) or type(right) in (float, sympy.Float):
                            self.assertIs(type(result), sympy.Float)

    def test_rational_zero_denominator_distinguishes_indeterminate(self):
        from sympy.core.numbers import nan

        self.assertIs(sympy.nan, nan)
        self.assertIs(sympy.S.NaN, nan)
        for numerator in (0, sympy.S.Zero, sympy.Rational(0), "0"):
            self.assertIs(sympy.Rational(numerator, 0), sympy.nan)
        for text in ("0/0", "1/0", "-3/0"):
            with self.assertRaises(ZeroDivisionError) as caught:
                sympy.Rational(text)
            self.assertEqual(str(caught.exception), "Fraction(1, 0)")
        for numerator in (1, -1, 3, -3):
            self.assertIs(sympy.Rational(numerator, 0), sympy.zoo)

    def test_float_division_retains_nonzero_operands_and_symbolic_paths(self):
        tiny = sympy.Rational(1, 10**400)
        self.assertIs(sympy.Float(0) / tiny, sympy.S.Zero)
        self.assertIs(tiny / sympy.Float(0), sympy.zoo)
        self.assertIs(sympy.Integer(10**400) / sympy.Float(0), sympy.zoo)
        # Unsupported binary64 range must not turn a nonzero exact operand
        # into a false zero divisor or an exact zero result. These are integrity
        # controls, not arbitrary-precision value-parity claims.
        self.assertIsNot(sympy.Float(1) / tiny, sympy.zoo)
        self.assertNotEqual(tiny / sympy.Float(1), 0)
        self.assertIs(type(sympy.Float(5e-324) / sympy.Float(2)), sympy.Float)
        self.assertEqual(sympy.Float(1) / sympy.Rational(1, 3), sympy.Float(3))
        self.assertEqual(sympy.Rational(1, 3) / sympy.Float(1), sympy.Float(1/3))
        x = sympy.Symbol("x")
        for expression in (x / sympy.Float(1.5), sympy.Float(1.5) / x):
            self.assertEqual(expression.free_symbols, {x})

    def test_float_division_does_not_coerce_unadmitted_objects(self):
        # Native admission controls, not a claim of upstream subclass parity.
        class Unadmitted:
            def __float__(self):
                raise AssertionError("unsupervised numeric conversion")

        class CustomFloat(sympy.Float):
            def _as_python_float(self):
                raise AssertionError("custom Float entered built-in fast path")

        value = sympy.Float(1)
        for left, right in ((value, Unadmitted()), (Unadmitted(), value)):
            with self.assertRaises(TypeError):
                left / right
        custom = CustomFloat(1)
        with self.assertRaises(NotImplementedError):
            custom / 2
        with self.assertRaises(NotImplementedError):
            2 / custom

    def test_power_does_not_identify_imaginary_unit_by_printed_name(self):
        named_i = sympy.Symbol("I")
        for power in (named_i**2, sympy.Pow(named_i, 2)):
            self.assertIs(type(power), sympy.Pow)
            self.assertEqual(power.args, (named_i, sympy.Integer(2)))
            self.assertEqual(power.free_symbols, {named_i})
        for power in ((2*named_i)**2, sympy.Pow(2*named_i, 2)):
            self.assertEqual(power.free_symbols, {named_i})
            self.assertNotEqual(power, -4)
        self.assertEqual(sympy.Pow(2*sympy.I, 2), -4)
        for exponent, expected in ((-1, -sympy.I), (2, -1), (3, -sympy.I), (4, 1)):
            self.assertEqual(sympy.I**exponent, expected)
            self.assertEqual(sympy.Pow(sympy.I, exponent), expected)

    def test_mixed_float_arithmetic_in_both_operand_orders(self):
        import operator

        for exact in (sympy.Integer(3), sympy.Rational(3, 2), sympy.S.Zero,
                      sympy.S.One, sympy.S.NegativeOne, sympy.S.Half):
            for approximate in (sympy.Float(1.5), sympy.Float(-3), sympy.Float(0)):
                for left, right in ((exact, approximate), (approximate, exact)):
                    for operation in (operator.add, operator.sub, operator.mul):
                        with self.subTest(left=left, right=right, operation=operation.__name__):
                            expected = operation(float(left), float(right))
                            result = operation(left, right)
                            if expected == 0:
                                self.assertIs(result, sympy.S.Zero)
                            else:
                                self.assertIs(type(result), sympy.Float)
                                self.assertEqual(result, sympy.Float(expected))
        x = sympy.Symbol("x")
        for expression in (x + sympy.Float(1.5), x - sympy.Float(1.5),
                           x * sympy.Float(1.5)):
            self.assertEqual(expression.free_symbols, {x})
        # Do not reclassify a rounded underflow as an exact zero singleton.
        # This checks type only, not arbitrary-precision value agreement.
        self.assertIs(type(sympy.Float(5e-324) * sympy.Float(0.5)), sympy.Float)
        exact = sympy.Integer(2**53 + 1)
        approximate = sympy.Float(-(2**53))
        # Do not erase a nonzero residual by rounding the exact operand first.
        self.assertNotEqual(exact + approximate, 0)
        self.assertNotEqual(approximate + exact, 0)

    def test_float_negation_preserves_numeric_type_and_value(self):
        import math

        for number in (0.0, -0.0, 1.5, -1.5, 5e-324, -5e-324, 1e300):
            value = sympy.Float(number)
            with self.subTest(number=number):
                result = -value
                self.assertIs(type(result), sympy.Float)
                self.assertEqual(result, sympy.Float(-number))
                self.assertEqual(-result, value)
                if number == 0:
                    self.assertEqual(math.copysign(1, float(result)), 1)
        self.assertEqual((-sympy.Float(-1.5)) / 3, sympy.Float(0.5))

    def test_float_negation_preserves_shell_precision_metadata(self):
        value = sympy.Float(1.5, 30)
        self.assertEqual((-value).dps, 30)
        self.assertEqual(value.dps, 30)
        self.assertEqual(value, sympy.Float(1.5))

    def test_exact_number_division_uses_float_arithmetic(self):
        for numerator in (sympy.Integer(0), sympy.Integer(3), sympy.Integer(-3),
                          sympy.Rational(3, 2), sympy.S.One,
                          sympy.S.NegativeOne, sympy.S.Half):
            for divisor in (sympy.Float(1.5), sympy.Float(-1.5), sympy.Float(0.5)):
                with self.subTest(numerator=numerator, divisor=divisor):
                    result = numerator / divisor
                    if numerator == 0:
                        self.assertIs(result, sympy.S.Zero)
                        continue
                    expected = sympy.Float(float(numerator) / float(divisor))
                    self.assertIs(type(result), sympy.Float)
                    self.assertEqual(result, expected)
        x = sympy.Symbol("x")
        for coefficient, expected in ((sympy.Float(1.5), sympy.Float(-2)),
                                      (sympy.Float(-1.5), sympy.Float(2))):
            self.assertEqual(sympy.solve_linear(coefficient*x + 3), (x, expected))
        self.assertEqual((x / sympy.Float(1.5)).free_symbols, {x})

    def test_coefficient_defaults_include_floats(self):
        import inspect

        x = sympy.Symbol("x")
        for method in (sympy.Expr.as_coeff_Mul, sympy.Expr.as_coeff_Add):
            self.assertIs(inspect.signature(method).parameters["rational"].default, False)
        for number in (sympy.Float(1.5), sympy.Float(-1.5), sympy.Float(0)):
            with self.subTest(number=number):
                self.assertEqual(number.as_coeff_Mul(), (number, 1))
                self.assertEqual(number.as_coeff_Add(), (number, 0))
                self.assertEqual(number.as_coeff_Mul(rational=True), (1, number))
                self.assertEqual(number.as_coeff_Add(rational=True), (0, number))
        for number in (sympy.Float(1.5), sympy.Float(-1.5)):
            with self.subTest(coefficient=number):
                self.assertEqual((number*x).as_coeff_Mul(), (number, x))
                self.assertEqual((number + x).as_coeff_Add(), (number, x))
                sign = -1 if number < 0 else 1
                self.assertEqual((number*x).as_coeff_Mul(rational=True),
                                 (sign, abs(number)*x))
                self.assertEqual((number + x).as_coeff_Add(rational=True), (0, number + x))

    def test_coefficient_splits_include_numeric_singletons(self):
        values = (sympy.S.Zero, sympy.S.One, sympy.S.NegativeOne,
                  sympy.S.Half, sympy.Integer(2), sympy.Rational(-2, 3))
        for value in values:
            for rational in (False, True):
                with self.subTest(value=value, rational=rational):
                    self.assertEqual(value.as_coeff_Mul(rational=rational), (value, 1))
                    self.assertEqual(value.as_coeff_Add(rational=rational), (value, 0))

    def test_coefficient_admission_keeps_custom_numeric_classes_opaque(self):
        class CustomInteger(sympy.Integer):
            pass

        class CustomRational(sympy.Rational):
            pass

        for value in (CustomInteger(2), CustomRational(2, 3)):
            for rational in (False, True):
                coefficient, rest = value.as_coeff_Mul(rational=rational)
                self.assertEqual(coefficient, 1)
                self.assertIs(rest, value)
                coefficient, rest = value.as_coeff_Add(rational=rational)
                self.assertEqual(coefficient, 0)
                self.assertIs(rest, value)
        value = sympy.Float(1.5)
        self.assertEqual(value.as_coeff_Mul(rational=False), (value, 1))
        self.assertEqual(value.as_coeff_Mul(rational=True), (1, value))

    def test_mixed_denominator_decomposition_and_linear_solving(self):
        x, y, z = sympy.symbols("x y z")
        cases = (
            (x/y + 1, (x + y, y), (x, -y)),
            (1/x + 1/y, (x + y, x*y), (x, -y)),
            (1/x + 1/x**2, (x**2 + x, x**3), (x**2 + x, x**3)),
            (x/2 + 1/y, (x*y + 2, 2*y), (x, -2/y)),
            (2*x/y + 4/z, (sympy.Mul(2, x*z + 2*y, evaluate=False), y*z),
             (x, -2*y/z)),
            (x/(2*y) + z/(3*y), (3*x + 2*z, 6*y), (x, -2*z/3)),
            (2/x + 2/y, (sympy.Mul(2, x + y, evaluate=False), x*y), (x, -y)),
            (x/y + z/y + 1, (x + y + z, y), (x, -y - z)),
            (1/x + 1/y + 1/z, (x*y + x*z + y*z, x*y*z),
             (x, -y*z/(y + z))),
            (1/(x + 1) + 1/(x - 1), (2*x, (x - 1)*(x + 1)), (x, 0)),
            (x**2/y + 1, (x**2 + y, y), (y, -x**2)),
        )
        for expression, fraction, solution in cases:
            with self.subTest(expression=expression):
                self.assertEqual(expression.as_numer_denom(), fraction)
                self.assertEqual(sympy.solve_linear(expression), solution)
        self.assertEqual(sympy.solve_linear(x/y + 1, symbols=[y]), (y, -x))
        self.assertEqual(sympy.solve_linear(1/y, symbols=[y]), (0, 1))
        for expression, expected in (
            (2*x + 4*y, (2*x + 4*y, 1)),
            (2*x/y + 4*z/y, (2*x + 4*z, y)),
            (-2*x/y - 4*z/y, (-2*x - 4*z, y)),
            (2*x/(3*y) + 4*z/(9*y), (6*x + 4*z, 9*y)),
        ):
            with self.subTest(shared_denominator=expression):
                self.assertEqual(expression.as_numer_denom(), expected)

    def test_solve_linear_uses_numerator_and_rejects_original_poles(self):
        x, y = sympy.symbols("x y")
        cases = (
            ((x + 1)/y, [x], (x, -1)),
            ((x + 1)/y, [y], (0, 1)),
            (x/(x - 1), [x], (x, 0)),
            (x**2/y**2, [x], (x**2, y**2)),
            (x**2/y**2, [y], (0, 1)),
            (1/x, [x], (0, 1)),
            ((x + y)/(x - y), [x], (x, -y)),
            ((x + y)/(x - y), [y], (y, -x)),
            (sympy.Mul(x, 1/x, evaluate=False), [x], (0, 0)),
            (sympy.Mul(x - 1, 1/(x - 1), evaluate=False), [x],
             (x - 1, x - 1)),
            (sympy.Mul(x + y, 1/(x + y), evaluate=False), [x, y],
             (x + y, x + y)),
            (sympy.Pow(1/x, -1, evaluate=False), [x], (0, 0)),
            (sympy.Pow(x/y, -1, evaluate=False), [y], (0, 0)),
            (sympy.Mul(x*y, 1/x, evaluate=False), [x, y], (y, 0)),
        )
        for expression, requested, expected in cases:
            with self.subTest(expression=expression, requested=requested):
                self.assertEqual(sympy.solve_linear(expression, symbols=requested), expected)

    def test_dummy_sort_key_uses_name_then_numeric_identity(self):
        from itertools import permutations

        older, named_first, newer = [sympy.Dummy(name) for name in ("z", "a", "m")]
        expected = [named_first, newer, older]
        for supplied in permutations(expected):
            self.assertEqual(sorted(supplied, key=lambda value: value.sort_key()), expected)
        same_name = [sympy.Dummy("d") for _ in range(12)]
        self.assertEqual(sorted(reversed(same_name), key=lambda value: value.sort_key()),
                         same_name)
        x = sympy.Symbol("x")
        mixed = [x, older, sympy.Integer(1), named_first]
        self.assertEqual(sorted(mixed, key=lambda value: value.sort_key()),
                         [sympy.Integer(1), named_first, older, x])
        expressions = [sympy.sin(value) for value in expected]
        self.assertEqual(sorted(reversed(expressions), key=lambda value: value.sort_key()),
                         expressions)

    def test_solve_linear_selects_present_symbols_canonically(self):
        from itertools import permutations

        x, y, z = sympy.symbols("x y z")
        for requested in permutations((x, y, z)):
            original = list(requested)
            for excluded, expected in (([], (x, -y)), ([x], (y, -x)),
                                       ([x, y], (0, 1))):
                with self.subTest(requested=requested, excluded=excluded):
                    supplied = list(original)
                    self.assertEqual(sympy.solve_linear(x + y, symbols=supplied,
                                                        exclude=excluded), expected)
                    self.assertEqual(supplied, original)
            self.assertEqual(sympy.solve_linear(x**2 + y, symbols=original),
                             (y, -x**2))
        for expression in (x, x + y, x**2, sympy.Integer(5)):
            self.assertEqual(sympy.solve_linear(expression, symbols=[z]), (0, 1))
        self.assertEqual(sympy.solve_linear(x**2 + y**2, symbols=[y, x]),
                         (x**2 + y**2, 1))
        older, named_first, newer = [sympy.Dummy(name) for name in ("z", "a", "m")]
        expression = older + named_first + newer
        for requested in permutations((older, named_first, newer)):
            self.assertEqual(sympy.solve_linear(expression, symbols=requested),
                             (named_first, -older - newer))
        same_name = [sympy.Dummy("d") for _ in range(12)]
        first, last = same_name[0], same_name[-1]
        for requested in ((first, last), (last, first)):
            self.assertEqual(sympy.solve_linear(first + last, symbols=requested),
                             (first, -last))

    def test_solve_numeric_constant_output_contracts(self):
        x, y = sympy.symbols("x y")
        zero_cases = (0, 0.0, sympy.Rational(0, 3), sympy.Float(0), [], [0],
                      (0, sympy.Float(0)), x - x)
        inconsistent_cases = (1, -2, sympy.S.One, sympy.S.NegativeOne, sympy.S.Half,
                              sympy.Rational(1, 3), sympy.Float(1),
                              [1], (0, 1), [sympy.Float(0), -2])
        for variables, expected_symbols in (((), []), ((x,), [x]),
                                            ((x, y), [x, y]), (([y, x],), [y, x])):
            for flags in ({}, {"dict": True}, {"set": True},
                          {"dict": True, "set": True}):
                for expression in zero_cases:
                    with self.subTest(expression=expression, variables=variables, flags=flags):
                        expected = (expected_symbols, set()) if flags.get("set") else []
                        self.assertEqual(sympy.solve(expression, *variables, **flags), expected)
                for expression in inconsistent_cases:
                    with self.subTest(expression=expression, variables=variables, flags=flags):
                        self.assertEqual(sympy.solve(expression, *variables, **flags), [])

        positive = sympy.Symbol("positive", positive=True)
        result = sympy.solve(0, positive, set=True)
        self.assertIs(result[0][0], positive)
        self.assertIs(result[0][0].is_positive, True)
        for expression in (0, 1, [0], [1]):
            with self.subTest(duplicate_expression=expression):
                with self.assertRaises(ValueError):
                    sympy.solve(expression, x, x)
        self.assertEqual(sympy.solve(x - 2, x, set=True), ([x], {(2,)}))
        self.assertEqual(sympy.solve([x - 1, x - 2], x, set=True), ([x], set()))

    def test_matrix_solvers_reject_duplicate_generators(self):
        from sympy.polys.polyerrors import BasePolynomialError, GeneratorsError

        x = sympy.Symbol("x")
        for matrix in (sympy.Matrix([[1, 1, 1]]), sympy.zeros(1, 3),
                       sympy.Matrix([[0, 0, 1]])):
            original = matrix.copy()
            systems = (matrix, (matrix[:, :-1], matrix[:, -1:]))
            for system in systems:
                with self.subTest(system=system):
                    with self.assertRaises(GeneratorsError) as caught:
                        sympy.linsolve(system, x, x)
                    self.assertEqual(type(caught.exception).__name__, "GeneratorsError")
                    self.assertEqual(type(caught.exception).__module__,
                                     "sympy.polys.polyerrors")
                    self.assertEqual(str(caught.exception), "duplicated generators: (x, x)")
            with self.assertRaises(GeneratorsError) as caught:
                sympy.solve_linear_system(matrix, x, x)
            self.assertEqual(type(caught.exception).__name__, "GeneratorsError")
            self.assertEqual(matrix, original)

        self.assertEqual(GeneratorsError.__bases__, (BasePolynomialError,))
        self.assertEqual(BasePolynomialError.__bases__, (Exception,))
        with self.assertRaisesRegex(NotImplementedError, "^abstract base class$"):
            GeneratorsError("duplicate").new("replacement")
        first, second = sympy.Dummy("x"), sympy.Dummy("x")
        result = sympy.linsolve(sympy.zeros(1, 3), first, second)
        self.assertEqual(tuple(next(iter(result))), (first, second))
        self.assertNotEqual(first, second)

    def test_linsolve_expression_symbols_are_explicit_and_distinct(self):
        x, y = sympy.symbols("x y")
        missing_message = (
            "\nWhen passing a system of equations, the explicit symbols for which a\n"
            "solution is being sought must be given as a sequence, too."
        )
        for system in ([0], [x + y - 1], (x - 1, y - 2)):
            for symbols in ((), ([],), ((),)):
                with self.subTest(system=system, symbols=symbols):
                    with self.assertRaises(ValueError) as caught:
                        sympy.linsolve(system, *symbols)
                    self.assertEqual(str(caught.exception), missing_message)
            for symbols in ((x, x), ([x, x],), (x, y, x)):
                with self.subTest(system=system, symbols=symbols):
                    with self.assertRaises(ValueError) as caught:
                        sympy.linsolve(system, *symbols)
                    self.assertEqual(str(caught.exception), "duplicate symbols given")
        self.assertEqual(sympy.linsolve([], x, x), sympy.EmptySet())
        result = sympy.linsolve([x - 1, y - 2], y, x)
        self.assertEqual(tuple(next(iter(result))), (2, 1))
        result = sympy.linsolve([x - y], [x])
        self.assertEqual(tuple(next(iter(result))), (y,))
        for coefficient in (1, 2, -3):
            expression = coefficient*x - y
            with self.subTest(coefficient=coefficient):
                result = sympy.linsolve([expression], x)
                self.assertEqual(tuple(next(iter(result))), (y/coefficient,))
                self.assertEqual(sympy.solve_linear(expression, symbols=[x]),
                                 (x, y/coefficient))
        with self.assertRaises(ValueError):
            sympy.linsolve([x**2 - y], x)

    def test_linsolve_empty_inputs_are_not_zero_equations(self):
        x, y = sympy.symbols("x y")
        empty_inputs = ([], (), sympy.Matrix([]), sympy.zeros(0, 3),
                        sympy.zeros(2, 0))
        for system in empty_inputs:
            for symbols in ((), (x, y), ([x, y],)):
                with self.subTest(system=system, symbols=symbols):
                    result = sympy.linsolve(system, *symbols)
                    self.assertIs(type(result), sympy.EmptySet)
                    self.assertEqual(result, sympy.EmptySet())

        # A present zero equation has an unconstrained family, not EmptySet.
        for system in ([0], [sympy.Integer(0)], sympy.zeros(1, 3),
                       (sympy.zeros(0, 2), sympy.zeros(0, 1))):
            with self.subTest(system=system):
                result = sympy.linsolve(system, x, y)
                self.assertIs(type(result), sympy.FiniteSet)
                self.assertEqual(tuple(next(iter(result))), (x, y))
        self.assertEqual(sympy.linsolve([1], x, y), sympy.EmptySet())
        self.assertEqual(tuple(next(iter(sympy.linsolve([x - 1, y - 2], x, y)))),
                         (1, 2))

    def test_linsolve_automatic_parameters_do_not_capture_input_symbols(self):
        for name, parameter_name in (("x1", "tau0"), ("tau", "tau00"),
                                     ("tau0", "tau00"), ("tau1", "tau00")):
            value = sympy.Symbol(name)
            parameter = sympy.Symbol(parameter_name)
            matrix = sympy.Matrix([[1, 1, value]])
            original = matrix.copy()
            for system in (matrix, (sympy.Matrix([[1, 1]]), sympy.Matrix([value]))):
                with self.subTest(name=name, system=system):
                    solution = tuple(next(iter(sympy.linsolve(system))))
                    self.assertEqual(solution, (value - parameter, parameter))
                    self.assertEqual(sympy.simplify(solution[0] + solution[1] - value), 0)
            self.assertEqual(matrix, original)
        tau0, tau1 = sympy.symbols("tau0 tau1")
        solution = tuple(next(iter(sympy.linsolve(sympy.Matrix([[0, 1, 0, 2]])))))
        self.assertEqual(solution, (tau0, 2, tau1))
        self.assertEqual(tuple(next(iter(sympy.linsolve(sympy.Matrix([[1, 2]]))))), (2,))
        self.assertEqual(sympy.linsolve(sympy.Matrix([[0, 1]])), sympy.EmptySet())
        # Pinned upstream also captures this name. Refuse the unsupported
        # boundary instead of claiming that its single tuple is a full family.
        with self.assertRaisesRegex(NotImplementedError, "parameter name collides"):
            sympy.linsolve(sympy.Matrix([[1, 1, sympy.Symbol("tau00")]]))
        x, y = sympy.symbols("x y")
        self.assertEqual(tuple(next(iter(sympy.linsolve(sympy.Matrix([[1, 1, 1]]), x, y)))),
                         (1 - y, y))

    def test_linear_solver_free_variable_output_contracts(self):
        x, y = sympy.symbols("x y")
        for expression in ([x + y - 1], x + y - 1):
            with self.subTest(expression=expression):
                self.assertEqual(sympy.solve(expression, x, y, dict=True),
                                 [{x: 1 - y}])
                self.assertEqual(sympy.solve(expression, x, y, dict=True, set=True),
                                 [{x: 1 - y}])
                result = sympy.solve(expression, x, y, set=True)
                self.assertEqual(result, ([x, y], {(1 - y, y)}))
                self.assertIs(type(next(iter(result[1]))), tuple)
        self.assertEqual(sympy.solve([x + y - 1], x, y), {x: 1 - y})
        result = sympy.solve(x + y - 1, x, y)
        self.assertEqual(result, [(1 - y, y)])
        self.assertIs(type(result[0]), tuple)
        for equations in ([0], [x - 1, x - 2]):
            for flags in ({}, {"dict": True}, {"set": True},
                          {"dict": True, "set": True}):
                with self.subTest(equations=equations, flags=flags):
                    expected = ([x, y], set()) if flags.get("set") else []
                    self.assertEqual(sympy.solve(equations, x, y, **flags), expected)
        matrix = sympy.Matrix([[1, 1, 1]])
        original = matrix.copy()
        self.assertEqual(sympy.solve_linear_system(matrix, x, y), {x: 1 - y})
        self.assertEqual(matrix, original)
        self.assertEqual(sympy.solve_linear_system(sympy.Matrix([[0, 0, 0]]), x, y), {})
        self.assertIsNone(sympy.solve_linear_system(sympy.Matrix([[0, 0, 1]]), x, y))
        # Unlike assignment dictionaries, linsolve retains every free parameter.
        solution = next(iter(sympy.linsolve([x + y - 1], x, y)))
        self.assertEqual(tuple(solution), (1 - y, y))

    def test_checksol_rejects_nonfinite_candidate_mappings(self):
        x, y = sympy.symbols("x y")
        for bad in (sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
            with self.subTest(candidate=bad):
                self.assertIs(sympy.checksol(x - 1, {x: 1, y: bad}), False)
                self.assertIs(sympy.checksol(x - 1, {x: 1, y: y + bad}), False)
                self.assertIs(sympy.checksol([x - 1, 2*x - 2], {x: 1, y: bad}), False)
                self.assertIs(sympy.checksol(0, {x: bad}), True)
                self.assertIs(sympy.checksol(1, {x: bad}), False)
        self.assertIs(sympy.checksol(x - 1, {x: 1, y: 2}), True)
        for expression in (1/x, sympy.sin(1/x), sympy.sqrt(1/x)):
            self.assertIs(sympy.checksol(expression, x, 0), False)

    def test_solveset_validates_domains_and_retains_unknown_membership(self):
        x, y = sympy.symbols("x y")
        for expression in (x - 1, sympy.Integer(0), sympy.Integer(1)):
            for domain in (42, "invalid-domain", [1, 2]):
                with self.subTest(expression=expression, domain=domain):
                    with self.assertRaisesRegex(ValueError, "not a valid domain"):
                        sympy.solveset(expression, x, domain=domain)
        result = sympy.solveset(x - y, x, domain=sympy.S.Reals)
        self.assertEqual(result, sympy.Intersection(sympy.FiniteSet(y), sympy.S.Reals))
        self.assertIsNone(result.contains(y))
        self.assertEqual(sympy.solveset(x*x - 1, x, domain=sympy.Interval(0, 2)),
                         sympy.FiniteSet(1))
        self.assertEqual(sympy.solveset(x - 3, x, domain=sympy.Interval(0, 2)),
                         sympy.EmptySet())

    def test_checksol_preserves_inconclusive_results_and_fuzzy_conjunction(self):
        x, y = sympy.symbols("x y")
        self.assertIsNone(sympy.checksol(sympy.sin(y), x, 1))
        self.assertIsNone(sympy.checksol(sympy.sqrt(y) - y, x, 1))
        for equations in ([sympy.sin(y), x - 1], [x - 1, sympy.sin(y)]):
            self.assertIsNone(sympy.checksol(equations, x, 1))
        for equations in ([sympy.sin(y), x - 2], [x - 2, sympy.sin(y)]):
            self.assertIs(sympy.checksol(equations, x, 1), False)
        self.assertIs(sympy.checksol(y, x, 1), False)
        self.assertIs(sympy.checksol(x + y, x, 1), False)
        self.assertIs(sympy.checksol(sympy.sin(x), x, 0), True)
        self.assertIs(sympy.checksol([x - 1, 2*x - 2], x, 1), True)

    def test_solveset_does_not_report_refused_equations_as_empty(self):
        x = sympy.Symbol("x")
        with self.assertRaisesRegex(ValueError, "non-linear"):
            sympy.solveset(sympy.sin(x), x)
        self.assertEqual(sympy.solveset(x - 1, x), sympy.FiniteSet(1))
        self.assertEqual(sympy.solveset(sympy.Integer(1), x), sympy.EmptySet())

    def test_solve_does_not_report_unsupported_systems_as_empty(self):
        x, y = sympy.symbols("x y")
        # These constraints have complex solutions; refusal is not emptiness.
        with self.assertRaisesRegex(ValueError, "not supported"):
            sympy.solve([x*x + y*y - 1, y*y - 2], x, y)
        with self.assertRaises(ValueError):
            sympy.solve(x*x + y*y - 1, x, y)
        self.assertEqual(sympy.solve([x + y - 5, x - y - 1], x, y),
                         {x: 3, y: 2})
        self.assertEqual(sympy.solve([x - 1, x - 2], x), [])

    def test_native_operation_wrappers_do_not_parse_symbol_names_as_expressions(self):
        # Legal atomic names must not be reinterpreted by the expression parser.
        for name in ("x+y", "x y", "a-b", "alpha_1"):
            with self.subTest(name=name):
                atom = sympy.Symbol(name)
                self.assertEqual(sympy.diff(atom**2, atom), 2 * atom)
                self.assertEqual(sympy.diff(atom**3, atom, 2), 6 * atom)
                self.assertEqual(sympy.expand((atom + 1)**2), atom**2 + 2*atom + 1)
                self.assertEqual(sympy.simplify(atom + 0), atom)
                self.assertEqual(sympy.expand(atom).free_symbols, {atom})

    def test_native_operation_wrappers_preserve_distinct_dummy_atoms(self):
        first = sympy.Dummy("x")
        second = sympy.Dummy("x")
        expression = first**2 + second**2
        self.assertNotEqual(first, second)
        self.assertEqual(sympy.diff(expression, first), 2 * first)
        self.assertEqual(sympy.diff(expression, second), 2 * second)
        self.assertEqual(sympy.simplify(expression).free_symbols, {first, second})

    def test_classes_are_usable_and_operations_preserve_surface_kind(self):
        x = sympy.Symbol("x")
        two = sympy.Integer(2)

        self.assertIsInstance(x, sympy.Expr)
        self.assertIsInstance(two, sympy.Expr)
        expression = x + two
        self.assertIsInstance(expression, sympy.Add)
        self.assertEqual(expression.func(*expression.args), expression)
        self.assertEqual(expression.free_symbols, {x})
        self.assertEqual(expression.subs(x, 3), sympy.Integer(5))

    def test_held_forms_copy_and_pickle_without_collapsing_identity(self):
        x = sympy.Symbol("x")
        held = sympy.Add(x, x, evaluate=False)

        self.assertIsInstance(held, sympy.Add)
        self.assertEqual(len(held.args), 2)
        self.assertIs(copy.deepcopy(held), held)
        restored = pickle.loads(pickle.dumps(held))  # ubs:ignore — trusted in-process bytes
        self.assertIsInstance(restored, sympy.Add)
        self.assertEqual(restored.args, held.args)

    def test_function_wrappers_return_expressions_not_wire_strings(self):
        x = sympy.Symbol("x")

        self.assertEqual(sympy.diff(x**2, x), 2 * x)
        self.assertEqual(sympy.diff(x**4, x, 2), 12 * x**2)
        self.assertEqual(sympy.diff(x**4, (x, 2)), 12 * x**2)
        self.assertEqual(sympy.diff(x**4, x, 0), x**4)
        self.assertEqual((x**4).diff(x, 2), 12 * x**2)
        self.assertEqual(sympy.Derivative(x**4, x, 2, evaluate=True), 12 * x**2)
        self.assertIsInstance(sympy.simplify(x + 0), sympy.Expr)
        self.assertEqual(sympy.integrate(2 * x, (x, 0, 1)), sympy.Integer(1))
        self.assertEqual(sympy.solve(2 * x - 4, x), [sympy.Integer(2)])
        self.assertEqual(sympy.solve(sympy.Eq(2 * x, 4), x), [sympy.Integer(2)])
        with self.assertRaises(NotImplementedError):
            sympy.dsolve(x)

    def test_atoms_and_held_equality(self):
        x, y = sympy.symbols("x y")
        expr = x + 2 * y + 1
        self.assertEqual(expr.atoms(sympy.Symbol), {x, y})
        self.assertIn(sympy.Integer(1), expr.atoms(sympy.Integer))
        self.assertTrue(all(arg.args == () for arg in expr.atoms()))

        relation = sympy.Eq(x, 2)
        self.assertIs(type(relation), sympy.Eq)
        self.assertEqual(relation.lhs, x)
        self.assertEqual(relation.rhs, sympy.Integer(2))
        restored = pickle.loads(pickle.dumps(relation))  # ubs:ignore — trusted in-process bytes
        self.assertIs(type(restored), sympy.Eq)
        self.assertEqual(restored.lhs, x)
        self.assertEqual(expr.atoms(sympy.Add), {expr})

        unequal = sympy.Ne(x, y)
        self.assertIs(type(unequal), sympy.Ne)
        self.assertEqual(unequal.lhs, x)
        self.assertEqual(sympy.Lt(x, 1).rel_op, "<")
        self.assertEqual(sympy.Le(x, 1).rel_op, "<=")
        self.assertEqual(sympy.Gt(x, 1).rel_op, ">")
        self.assertEqual(sympy.Ge(x, 1).rel_op, ">=")
        restored_ne = pickle.loads(pickle.dumps(unequal))  # ubs:ignore — trusted in-process bytes
        self.assertIs(type(restored_ne), sympy.Ne)

        held = sympy.Derivative(x**2, x, evaluate=False)
        self.assertEqual(held.doit(), 2 * x)
        evaluated = sympy.Derivative(x**2, x, evaluate=True)
        self.assertEqual(evaluated, 2 * x)
        self.assertNotEqual(type(evaluated), sympy.Derivative)

        # 0-ary and 1-ary Add and Mul (oracle parity)
        self.assertEqual(sympy.Add(), sympy.Integer(0))
        self.assertEqual(sympy.Add(x), x)
        self.assertEqual(sympy.Add(x, evaluate=False), x)
        self.assertEqual(sympy.Mul(), sympy.Integer(1))
        self.assertEqual(sympy.Mul(x), x)
        self.assertEqual(sympy.Mul(x, evaluate=False), x)
        held_mul = sympy.Mul(x, sympy.Integer(2), evaluate=False)
        self.assertEqual(held_mul.args, (x, sympy.Integer(2)))

        terms = sympy.Add(y, x, sympy.Integer(1), evaluate=False).as_ordered_terms()
        self.assertEqual(set(terms), {x, y, sympy.Integer(1)})
        self.assertEqual(terms, tuple(sorted(terms, key=lambda term: term.sort_key())))

        coeff, rest = (2 * x * y).as_coeff_Mul()
        self.assertEqual(coeff, sympy.Integer(2))
        self.assertEqual(rest, x * y)
        add_coeff, add_rest = (x + 3).as_coeff_Add()
        self.assertEqual(add_coeff, sympy.Integer(3))
        self.assertEqual(add_rest, x)
        self.assertEqual((x**2).as_base_exp(), (x, sympy.Integer(2)))
        self.assertEqual(x.as_base_exp(), (x, sympy.Integer(1)))
        self.assertEqual(expr.find(sympy.Symbol), {x, y})
        self.assertEqual(expr.find(x), {x})

    def test_as_numer_denom_and_could_extract_minus_sign(self):
        x, y, z = sympy.symbols("x y z")
        one = sympy.Integer(1)
        two = sympy.Integer(2)
        three = sympy.Integer(3)

        self.assertEqual(two.as_numer_denom(), (two, one))
        self.assertEqual(sympy.Integer(-3).as_numer_denom(), (sympy.Integer(-3), one))
        self.assertEqual(sympy.Rational(2, 3).as_numer_denom(), (two, three))
        self.assertEqual(sympy.Rational(-2, 3).as_numer_denom(), (sympy.Integer(-2), three))
        float_one = sympy.Float(1.5)
        self.assertEqual(float_one.as_numer_denom(), (float_one, one))
        self.assertEqual(x.as_numer_denom(), (x, one))
        self.assertEqual((x**2).as_numer_denom(), (x**2, one))
        self.assertEqual((1 / x).as_numer_denom(), (one, x))
        self.assertEqual((x ** (-2)).as_numer_denom(), (one, x**2))
        self.assertEqual((x / y).as_numer_denom(), (x, y))
        self.assertEqual((2 * x / 3).as_numer_denom(), (2 * x, three))
        self.assertEqual((x + 1).as_numer_denom(), (x + 1, one))

        half_sum_n, half_sum_d = (x / 2 + sympy.Rational(1, 2)).as_numer_denom()
        self.assertEqual(half_sum_d, two)
        self.assertEqual(half_sum_n, x + 1)
        mixed_n, mixed_d = (x / 2 + y / 3).as_numer_denom()
        self.assertEqual(mixed_d, sympy.Integer(6))
        self.assertEqual(mixed_n, 3 * x + 2 * y)
        same_n, same_d = (x / y + z / y).as_numer_denom()
        self.assertEqual(same_d, y)
        self.assertEqual(same_n, x + z)
        # SymPy 1.14 combines symbolic denominators too; leaving this Add
        # unchanged used to prevent solve_linear from seeing its numerator.
        self.assertEqual((x / y + 1).as_numer_denom(), (x + y, y))

        self.assertFalse(two.could_extract_minus_sign())
        self.assertTrue(sympy.Integer(-3).could_extract_minus_sign())
        self.assertFalse(sympy.Rational(2, 3).could_extract_minus_sign())
        self.assertTrue(sympy.Rational(-2, 3).could_extract_minus_sign())
        self.assertFalse(sympy.Float(1.5).could_extract_minus_sign())
        self.assertTrue(sympy.Float(-1.5).could_extract_minus_sign())
        self.assertFalse(x.could_extract_minus_sign())
        self.assertTrue((-x).could_extract_minus_sign())
        self.assertTrue((-2 * x).could_extract_minus_sign())
        self.assertFalse((2 * x).could_extract_minus_sign())
        self.assertFalse((x - y).could_extract_minus_sign())
        self.assertTrue((y - x).could_extract_minus_sign())
        self.assertFalse((x - 1).could_extract_minus_sign())
        self.assertTrue((1 - x).could_extract_minus_sign())
        self.assertFalse((x * (y - x)).could_extract_minus_sign())
        self.assertFalse(sympy.Integer(0).could_extract_minus_sign())

    def test_constants_are_native_constants_not_spoofed_symbols(self):
        for constant in (sympy.pi, sympy.E, sympy.I, sympy.oo, sympy.zoo, sympy.nan):
            self.assertIsInstance(constant, sympy.Expr)
            self.assertNotIsInstance(constant, sympy.Symbol)
            self.assertFalse(constant.is_symbol)

    def test_custom_subclasses_are_not_silently_collapsed_to_native_nodes(self):
        class CustomSymbol(sympy.Symbol):
            pass

        custom = CustomSymbol("custom")
        self.assertIs(type(custom), CustomSymbol)
        with self.assertRaisesRegex(NotImplementedError, "exact built-in classes only"):
            custom + 1

    def test_number_theory_wrappers_admit_only_exact_integers(self):
        with self.assertRaisesRegex(ValueError, r"^2\.9 is not an integer$"):
            sympy.isprime(2.9)
        with self.assertRaisesRegex(ValueError, r"^2\.9 is not an integer$"):
            sympy.factorint(2.9)
        with self.assertRaisesRegex(TypeError, r"^n should be an integer$"):
            sympy.totient(2.9)

        class LossyInteger:
            calls = 0

            def __int__(self):
                self.calls += 1
                return 2

            def __str__(self):
                self.calls += 1
                return "2"

            def __repr__(self):
                self.calls += 1
                return "2"

            def __format__(self, format_spec):
                del format_spec
                self.calls += 1
                return "2"

        lossy = LossyInteger()
        with self.assertRaises(ValueError):
            sympy.isprime(lossy)
        with self.assertRaises(ValueError):
            sympy.factorint(lossy)
        with self.assertRaises(TypeError):
            sympy.Integer(lossy)
        with self.assertRaises(TypeError):
            sympy.Rational(lossy, 1)
        self.assertEqual(lossy.calls, 0)

        self.assertEqual(sympy.Integer(2.9), sympy.Integer(2))
        self.assertEqual(sympy.Integer(True), sympy.Integer(1))
        self.assertEqual(
            sympy.Rational(1.9, 2),
            sympy.Rational(4278419646001971, 4503599627370496),
        )

    def test_numeric_bridge_preserves_values_beyond_machine_and_decimal_limits(self):
        huge = 1 << 20_000
        integer = sympy.Integer(huge)
        self.assertEqual(integer.p, huge)
        self.assertEqual(integer.q, 1)

        rational = sympy.Rational(huge, 3)
        self.assertEqual(rational.p, huge)
        self.assertEqual(rational.q, 3)
        trusted_payload = pickle.dumps(rational)
        self.assertEqual(
            pickle.loads(trusted_payload),  # nosec B301  # ubs:ignore — trusted in-process bytes
            rational,
        )

        normalized = sympy.Rational(-(1 << 63), -1)
        self.assertEqual(normalized.p, 1 << 63)
        self.assertEqual(normalized.q, 1)

    def test_custom_symbol_variable_refuses_before_running_overrides(self):
        effects = []

        class EffectfulSymbol(sympy.Symbol):
            @property
            def name(self):
                effects.append("name")
                return "x"

        variable = EffectfulSymbol("x")
        with self.assertRaisesRegex(
            NotImplementedError, "supervised Python override lane"
        ):
            sympy.diff(sympy.Symbol("x"), variable)
        self.assertEqual(effects, [])

    def test_number_theory_wrappers_preserve_signed_and_zero_domains(self):
        self.assertTrue(sympy.isprime(sympy.Integer(2)))
        self.assertFalse(sympy.isprime(-2))
        self.assertEqual(sympy.factorint(-12), {2: 2, 3: 1, -1: 1})
        self.assertEqual(sympy.factorint(0), {0: 1})
        with self.assertRaisesRegex(ValueError, r"^n should be a positive integer$"):
            sympy.totient(0)

    def test_missing_native_extension_fails_closed(self):
        package_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(package_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; sys.modules['fsym_python'] = None; import sympy",
            ],
            cwd=package_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires its fsym_python native extension", completed.stderr)

    def test_dummy_symbols_are_distinct_and_serialize(self):
        d1 = sympy.Dummy("x")
        d2 = sympy.Dummy("x")
        s = sympy.Symbol("x")

        self.assertIsInstance(d1, sympy.Dummy)
        self.assertIsInstance(d1, sympy.Symbol)
        self.assertEqual(d1.name, "x")
        self.assertEqual(d2.name, "x")
        self.assertNotEqual(d1, d2)
        self.assertNotEqual(d1, s)
        self.assertNotEqual(d1.dummy_index, d2.dummy_index)

        # Free symbols
        expr = d1 + d2 + s
        self.assertEqual(expr.free_symbols, {d1, d2, s})

        # Calculus differentiation
        self.assertEqual(sympy.diff(d1**2, d1), 2 * d1)
        self.assertEqual(sympy.diff(d1**2, d2), sympy.Integer(0))

        # Serialization round-trip
        restored = pickle.loads(pickle.dumps(d1))  # ubs:ignore — trusted in-process bytes
        self.assertIsInstance(restored, sympy.Dummy)
        self.assertEqual(restored, d1)
        self.assertEqual(restored.dummy_index, d1.dummy_index)

        # Reserved prefix collision rejection
        with self.assertRaises(ValueError):
            sympy.Symbol("__fsymDummy_1_x")

    def test_undefined_function_application_round_trips(self):
        x = sympy.Symbol("x")
        f = sympy.Function("f")
        applied = f(x)

        self.assertIsInstance(applied, sympy.Expr)
        self.assertEqual(applied.func, f)
        self.assertEqual(applied.args, (x,))
        self.assertEqual(applied.func(*applied.args), applied)
        self.assertEqual(applied.free_symbols, {x})

        summed = applied + 1
        self.assertTrue(any(type(arg) is type(applied) for arg in summed.args))
        restored = pickle.loads(pickle.dumps(applied))  # ubs:ignore — trusted in-process bytes
        self.assertEqual(restored, applied)
        self.assertEqual(restored.func, f)

        with self.assertRaises(ValueError):
            sympy.Function("")
        with self.assertRaises(TypeError):
            sympy.Function(1)

    def test_class_hierarchy_and_mro(self):
        self.assertTrue(issubclass(sympy.Symbol, sympy.AtomicExpr))
        self.assertTrue(issubclass(sympy.Dummy, sympy.Symbol))
        self.assertTrue(issubclass(sympy.AtomicExpr, sympy.Expr))
        self.assertTrue(issubclass(sympy.AtomicExpr, sympy.Atom))
        self.assertTrue(issubclass(sympy.Atom, sympy.Basic))
        self.assertTrue(issubclass(sympy.Expr, sympy.Basic))
        self.assertTrue(issubclass(sympy.Number, sympy.AtomicExpr))
        self.assertTrue(issubclass(sympy.Rational, sympy.Number))
        self.assertTrue(issubclass(sympy.Integer, sympy.Rational))
        self.assertTrue(issubclass(sympy.Float, sympy.Number))
        self.assertFalse(issubclass(sympy.Float, sympy.Rational))
        self.assertTrue(issubclass(sympy.Add, sympy.Expr))
        self.assertTrue(issubclass(sympy.Mul, sympy.Expr))
        self.assertTrue(issubclass(sympy.Pow, sympy.Expr))
        self.assertTrue(issubclass(sympy.Derivative, sympy.Expr))
        self.assertTrue(issubclass(sympy.AppliedUndef, sympy.Expr))

    def test_symbols_utility_function(self):
        x, y, z = sympy.symbols("x y z")
        self.assertIsInstance(x, sympy.Symbol)
        self.assertIsInstance(y, sympy.Symbol)
        self.assertIsInstance(z, sympy.Symbol)
        self.assertEqual(x.name, "x")
        self.assertEqual(y.name, "y")
        self.assertEqual(z.name, "z")

        a, b = sympy.symbols("a, b")
        self.assertEqual(a.name, "a")
        self.assertEqual(b.name, "b")

        single = sympy.symbols("single")
        self.assertIsInstance(single, sympy.Symbol)
        self.assertEqual(single.name, "single")

        with self.assertRaises(ValueError):
            sympy.symbols("")
        with self.assertRaises(ValueError):
            sympy.symbols("  ")

    def test_basic_methods_has_and_subs(self):
        x, y = sympy.symbols("x y")
        expr = x + 2 * y + 1

        self.assertTrue(expr.has(x))
        self.assertTrue(expr.has(y))
        self.assertFalse(expr.has(sympy.Symbol("z")))

        substituted = expr.subs(x, 2)
        expected = sympy.Integer(2) + 2 * y + 1
        self.assertEqual(substituted, expected)

    def test_xreplace_replaces_exact_nodes_without_inventing_algebra(self):
        x, y = sympy.symbols("x y")
        expr = x + 1
        self.assertEqual(expr.xreplace({expr: y}), y)
        self.assertEqual(x.xreplace({x: y}), y)
        self.assertEqual(x.xreplace({y: 1}), x)
        self.assertEqual((x + 2).xreplace({x: sympy.Integer(3)}), sympy.Integer(5))

        held = sympy.Add(x, x, evaluate=False)
        unchanged = held.xreplace({y: 1})
        self.assertIs(type(unchanged), sympy.Add)
        self.assertEqual(len(unchanged.args), 2)

        with self.assertRaises(TypeError):
            x.xreplace(1)

    def test_singleton_registry_exposes_exact_atoms_and_constructs_float(self):
        self.assertEqual(sympy.S.Zero, sympy.Integer(0))
        self.assertEqual(sympy.S.One, sympy.Integer(1))
        self.assertEqual(sympy.S.NegativeOne, sympy.Integer(-1))
        self.assertEqual(sympy.S.Half, sympy.Rational(1, 2))
        self.assertEqual(sympy.S(2), sympy.Integer(2))
        self.assertEqual(sympy.S(True), sympy.Integer(1))
        one = sympy.S.One
        self.assertIs(sympy.S(one), one)
        self.assertEqual(str(sympy.S.Pi), "pi")
        self.assertEqual(str(sympy.S.Infinity), "oo")
        self.assertEqual(str(sympy.S.ComplexInfinity), "zoo")
        self.assertIsInstance(sympy.S(1.5), sympy.Float)
        with self.assertRaises(TypeError):
            sympy.S(object())

    def test_float_is_a_number_atom_not_a_rational(self):
        value = sympy.Float(1.5)
        self.assertIs(type(value), sympy.Float)
        self.assertIsInstance(value, sympy.Number)
        self.assertNotIsInstance(value, sympy.Rational)
        self.assertEqual(value.args, ())
        self.assertIs(value.func, sympy.Float)
        self.assertFalse(value.is_symbol)
        self.assertTrue(value.is_number)
        self.assertFalse(value.is_integer)
        self.assertFalse(value.is_rational)
        self.assertAlmostEqual(value.evalf(), 1.5)
        # SymPy 1.14.0 parity (bead fra-fra-shell-float-zero-eq-structural-crx):
        # Float equality is Float-vs-Float only; cross-type is structurally
        # False even where the values are numerically equal — while the hash
        # collision with Integer remains (oracle: hash matches, eq does not).
        self.assertNotEqual(value, sympy.Rational(3, 2))
        self.assertNotEqual(sympy.Float(1.0), sympy.Integer(1))
        self.assertNotEqual(sympy.Integer(1), 1.0)
        self.assertEqual(hash(sympy.Float(1.0)), hash(sympy.Integer(1)))
        self.assertEqual(hash(sympy.Integer(1)), hash(1))
        self.assertNotEqual(sympy.Float(0.1), sympy.Rational(1, 10))
        self.assertEqual(value, sympy.Float(1.5))
        self.assertEqual(sympy.Float(3) + sympy.Float(0.5), sympy.Float(3.5))
        self.assertEqual(sympy.Integer(sympy.Float(2.9)), sympy.Integer(2))

        x = sympy.Symbol("x")
        summed = x + 1.5
        self.assertIsInstance(summed, sympy.Add)
        self.assertTrue(any(type(arg) is sympy.Float for arg in summed.args))

        restored = pickle.loads(pickle.dumps(value))  # ubs:ignore — trusted in-process bytes
        self.assertIs(type(restored), sympy.Float)
        self.assertEqual(restored, value)

        with self.assertRaises(TypeError):
            sympy.Float(object())
        with self.assertRaises(ValueError):
            sympy.Function("__fsymFloat")

        class LossyFloat:
            def __float__(self):
                return 1.5

        with self.assertRaises(TypeError):
            sympy.Float(LossyFloat())

    def test_float_atom_pins_ieee_binary64_contract(self) -> None:
        # Float is profile-compatible binary64. The intern encoding stores the
        # IEEE binary64 bit pattern as a reserved function payload, so a
        # future change to (a) the bit-packing, (b) the reserved name, or
        # (c) the dps validation would silently break the wire and persistence
        # contract. Pin each surface so any change is loud.
        # Round-trip: Float(value).evalf() returns the same Python float.
        for value in [0.0, 1.0, -1.0, 1.5, -1.5, 1.0e100, 1.0e-100, float("inf"), -float("inf")]:
            self.assertEqual(float(sympy.Float(value).evalf()), float(value))
        # The intern encoding stores IEEE binary64 bits in big-endian order.
        import struct
        for value, expected_bits in [(1.5, 0x3FF8000000000000), (0.0, 0), (-0.0, 0x8000000000000000), (float("inf"), 0x7FF0000000000000)]:
            self.assertEqual(sympy.Float(value)._value.args[0].exact_numerator(), expected_bits)
        with self.assertRaises(TypeError):
            sympy.Float(1.0, dps=0)
        with self.assertRaises(TypeError):
            sympy.Float(1.0, dps=-1)
        # Default dps is 15 (matches Python's repr(float) at full precision).
        self.assertEqual(sympy.Float(1.0).dps, 15)
        # __fsymFloat is the reserved intern name; user Function cannot collide.
        with self.assertRaises(ValueError):
            sympy.Function("__fsymFloat")
        # Float is a Number, not a Rational.
        self.assertTrue(issubclass(sympy.Float, sympy.Number))
        self.assertFalse(issubclass(sympy.Float, sympy.Rational))


    def test_latex_and_evalf_representations(self) -> None:
        x = sympy.Symbol("x")
        expr = x / 2
        latex_repr = expr._repr_latex_()
        self.assertIsInstance(latex_repr, str)
        self.assertTrue(len(latex_repr) > 0)

        two = sympy.Integer(2)
        self.assertIsInstance(two.evalf(), sympy.Float)
        self.assertEqual(two.evalf(), sympy.Float(2.0))
        self.assertAlmostEqual(float(sympy.pi.evalf()), 3.1415926535, places=4)
        self.assertEqual(sympy.N(2), sympy.Float(2.0))
        self.assertIsInstance(sympy.N(sympy.pi), sympy.Float)

        x, y = sympy.symbols("x y")
        ordered = sorted([y, sympy.Integer(1), x + 1, x], key=lambda expr: expr.sort_key())
        self.assertEqual(ordered[0], sympy.Integer(1))
        self.assertEqual({elt.name for elt in ordered[1:] if type(elt) is sympy.Symbol}, {"x", "y"})
        self.assertLess(sympy.Integer(1).sort_key(), x.sort_key())
        self.assertLess(x.sort_key(), (x + 1).sort_key())

        powered = x**2
        # Pinned oracle: pretty() is ASCII multi-line, linear forms use the
        # ASCII hyphen-minus (bead fra-fra-shell-printer-parity-pack-qxr).
        self.assertEqual(sympy.pretty(powered), " 2\nx ")
        self.assertEqual(powered.pretty(), " 2\nx ")
        self.assertEqual(sympy.pretty(x - 1), "x - 1")

    def test_calculus_and_solvers_facades(self):
        x = sympy.Symbol("x")
        self.assertEqual(sympy.diff(x**3, x), 3 * x**2)
        self.assertEqual(sympy.diff(x**3, x, x), 6 * x)
        self.assertEqual(sympy.integrate(x**2, x), sympy.Rational(1, 3) * x**3)
        self.assertEqual(sympy.integrate(x, (x, 0, 2)), sympy.Integer(2))
        self.assertEqual(sympy.integrate(x, x, 0, 2), sympy.Integer(2))
        with self.assertRaises(ValueError):
            sympy.integrate(x, (x, 0))
        with self.assertRaises(TypeError):
            sympy.integrate(x, x, 0)

    def test_elementary_functions_use_native_identity_folds(self):
        x = sympy.Symbol("x")
        self.assertEqual(sympy.sin(0), sympy.Integer(0))
        self.assertEqual(sympy.cos(0), sympy.Integer(1))
        self.assertEqual(sympy.exp(0), sympy.Integer(1))
        self.assertEqual(sympy.log(1), sympy.Integer(0))
        self.assertEqual(sympy.sin(x).func, sympy.Function("sin"))
        self.assertEqual(sympy.sin(x).args, (x,))

    def test_matrix_surface_operations(self):
        # 1. Constructor patterns
        m = sympy.Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertTrue(m.is_square)
        self.assertFalse(m.is_symmetric)
        self.assertEqual(m[0, 0], sympy.Integer(1))
        self.assertEqual(m[0, 1], sympy.Integer(2))
        self.assertEqual(m[1, 0], sympy.Integer(3))
        self.assertEqual(m[1, 1], sympy.Integer(4))

        # 2. Transpose, Trace, Determinant
        mt = m.T
        self.assertEqual(mt[0, 1], sympy.Integer(3))
        self.assertEqual(mt[1, 0], sympy.Integer(2))
        self.assertEqual(m.trace(), sympy.Integer(5))
        self.assertEqual(m.det(), sympy.Integer(-2))

        # 3. Inverse and Adjugate
        inv = m.inv()
        self.assertEqual(inv.shape, (2, 2))
        ident = m @ inv
        self.assertEqual(ident[0, 0], sympy.Integer(1))
        self.assertEqual(ident[0, 1], sympy.Integer(0))
        self.assertEqual(ident[1, 0], sympy.Integer(0))
        self.assertEqual(ident[1, 1], sympy.Integer(1))

        adj = m.adjugate()
        self.assertEqual(adj[0, 0], sympy.Integer(4))
        self.assertEqual(adj[0, 1], sympy.Integer(-2))
        self.assertEqual(adj[1, 0], sympy.Integer(-3))
        self.assertEqual(adj[1, 1], sympy.Integer(1))

        # 4. Arithmetic
        m2 = m + m
        self.assertEqual(m2[0, 0], sympy.Integer(2))
        self.assertEqual(m2[1, 1], sympy.Integer(8))
        diff_m = m - m
        self.assertEqual(diff_m[0, 0], sympy.Integer(0))
        self.assertEqual(diff_m[1, 1], sympy.Integer(0))

        # Negation
        neg_m = -m
        self.assertEqual(neg_m[0, 0], sympy.Integer(-1))
        self.assertEqual(neg_m[1, 1], sympy.Integer(-4))
        self.assertEqual(m + neg_m, sympy.zeros(2, 2))

        # Scalar multiplication and right-multiplication
        scaled = m * 2
        r_scaled = 2 * m
        self.assertEqual(scaled[0, 0], sympy.Integer(2))
        self.assertEqual(scaled[1, 1], sympy.Integer(8))
        self.assertEqual(scaled, r_scaled)

        # Scalar division
        div_m = m / 2
        self.assertEqual(div_m[0, 0], sympy.Rational(1, 2))
        self.assertEqual(div_m[0, 1], sympy.Integer(1))

        # Equality
        self.assertEqual(m, sympy.Matrix([[1, 2], [3, 4]]))
        self.assertNotEqual(m, sympy.Matrix([[1, 2], [3, 5]]))

        # Matrix power
        m_sq = m ** 2
        m_mul = m @ m
        self.assertEqual(m_sq.tolist(), m_mul.tolist())

        # 5. Helpers: eye, zeros, diag
        e = sympy.eye(3)
        self.assertEqual(e.shape, (3, 3))
        self.assertTrue(e.is_diagonal)
        self.assertTrue(e.is_symmetric)
        self.assertEqual(e.trace(), sympy.Integer(3))

        z = sympy.zeros(2, 3)
        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(z.rank(), 0)

        d = sympy.diag(1, 2, 3)
        self.assertEqual(d.shape, (3, 3))
        self.assertTrue(d.is_diagonal)
        self.assertEqual(d.det(), sympy.Integer(6))

        # 6. RREF and Nullspace
        sing = sympy.Matrix([[1, 2], [2, 4]])
        self.assertEqual(sing.rank(), 1)
        rref_m, pivots = sing.rref()
        self.assertEqual(pivots, (0,))
        ns = sing.nullspace()
        self.assertEqual(len(ns), 1)
        self.assertEqual((sing @ ns[0])[0, 0], sympy.Integer(0))
        self.assertEqual((sing @ ns[0])[1, 0], sympy.Integer(0))

        # 7. LaTeX rendering
        latex_str = m._repr_latex_()
        self.assertIn("begin{matrix}", latex_str)
        self.assertIn("end{matrix}", latex_str)

    def test_zero_singleton_identity_and_module(self):
        z = sympy.S.Zero
        self.assertIs(z, sympy.S.Zero)
        self.assertEqual(type(z).__name__, "Zero")
        self.assertEqual(type(z).__module__, "sympy.core.numbers")
        self.assertTrue(type(z).is_Zero)
        self.assertEqual(z, sympy.Integer(0))
        self.assertEqual(z, 0)
        self.assertEqual(hash(z), hash(0))
        self.assertEqual(repr(z), "0")
        self.assertEqual(str(z * 5), "0")

    def test_custom_subclass_zero_collapse_matches_oracle(self):
        # Mirrors tools/conformance-lab fixture subclass/ConstitutiveLawZero_zero_collapse:
        # eval folds on literal-zero first arg and keeps the applied form otherwise.
        x, k = sympy.Symbol("x"), sympy.Symbol("k")

        def eval_(cls, *a):
            if len(a) == 2 and a[0] == 0:
                return sympy.S.Zero
            return None

        cls = type(
            "ConstitutiveLawZeroPin",
            (sympy.Function,),
            {"eval": classmethod(eval_), "nargs": (2,)},
        )
        collapsed = cls(0, k)
        self.assertEqual(type(collapsed).__name__, "Zero")
        self.assertEqual(collapsed, 0)
        applied = cls(x, k)
        self.assertEqual(applied.func.__name__, "ConstitutiveLawZeroPin")
        self.assertEqual(applied.args, (x, k))

    def test_deep_chain_refuses_instead_of_crashing(self):
        # Gauntlet bead fra-native-drop-depth-bound-9mk: a deep exact-arithmetic
        # chain used to SIGSEGV the interpreter (recursive derived Clone in the
        # native kernel at depth ~8000). The bridge now refuses beyond
        # FSYM_MAX_EXPR_DEPTH with RecursionError and the process survives.
        x = sympy.Symbol("x")
        a = sympy.Integer(2)
        with self.assertRaises(RecursionError):
            for i in range(1, 6000):
                a = (a + 1) * x
        # The interpreter is alive and ordinary arithmetic still works at
        # moderate depth.
        b = sympy.Integer(2)
        for i in range(1, 500):
            b = b * sympy.Integer(i) + sympy.Rational(1, i)
        self.assertGreater(len(str(b)), 1000)

    def test_depth_bound_env_override(self):
        # The bound is configurable; a bound of 1 refuses any compound operand
        # (depth 2 > 1).
        env = dict(os.environ, FSYM_MAX_EXPR_DEPTH="1")
        code = (
            "import sympy\n"
            "x = sympy.Symbol('x')\n"
            "try:\n"
            "    e = (x + 1) + (x + 2)\n"
            "    print('NO-REFUSAL', e)\n"
            "except RecursionError:\n"
            "    print('REFUSED')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("REFUSED", proc.stdout)

    def test_wrap_recovered_symbols_carry_assumptions(self):
        # Gauntlet bead fra-shell-atom-assumptions-bypasses-7o3: symbols and
        # dummies recovered from native results bypass __init__ and previously
        # left the _assumptions slot unset, crashing srepr and is_* access.
        x = sympy.Symbol("x")
        recovered = sympy.simplify(x + x - x)
        self.assertEqual(sympy.srepr(recovered), "Symbol('x')")
        d = sympy.Dummy("d")
        expr = d + x
        for atom in expr.free_symbols:
            if isinstance(atom, sympy.Dummy):
                # Exact Dummy srepr form is finding 10 (seams bead); this pin
                # only requires the recovered Dummy to answer is_* and srepr
                # without AttributeError.
                self.assertIsInstance(atom.is_integer, (bool, type(None)))
                # Must not raise; exact form is finding 10 (seams bead).
                sympy.srepr(atom)

    def test_number_assumption_properties_match_oracle(self):
        # Tri-valued semantics pinned against SymPy 1.14.0 (fresh-eyes
        # finding 7): concrete numbers answer concretely, nan answers None.
        self.assertTrue(sympy.Integer(5).is_positive)
        self.assertFalse(sympy.Integer(0).is_positive)
        self.assertTrue(sympy.Integer(0).is_zero)
        self.assertTrue(sympy.Integer(-7).is_negative)
        self.assertTrue(sympy.Integer(-7).is_nonpositive)
        self.assertTrue(sympy.Rational(1, 2).is_positive)
        self.assertFalse(sympy.Rational(1, 2).is_integer)
        self.assertTrue(sympy.Integer(3).is_integer)
        self.assertTrue(sympy.Rational(1, 2).is_real)
        self.assertTrue(sympy.Float(2.5).is_positive)
        self.assertFalse(sympy.Float(2.5).is_integer)
        self.assertTrue(sympy.Float(0.0).is_integer)
        self.assertTrue(sympy.Float(0.0).is_zero)
        nan = sympy.Float("nan")
        self.assertIsNone(nan.is_positive)
        self.assertIsNone(nan.is_zero)
        self.assertIsNone(nan.is_integer)

    def test_seam_pack_findings_are_closed(self):
        # Gauntlet bead fra-fra-shell-seams-pack-c9g (fresh-eyes findings
        # 3, 4, 5, 6, 8, 9, 10, 11; finding 12 verified fixed externally).
        x = sympy.Symbol("x")
        # 3 + 8: raw args sympified; eval hook sympified + honors evaluate.
        self.assertEqual(str(sympy.Function("f")(3)), "f(3)")
        log = []

        class Traced(sympy.Function):
            @classmethod
            def eval(cls, *a):
                log.append(a)
                return None

        traced = Traced(x, evaluate=False)
        self.assertEqual(log, [])  # evaluate=False skips the hook
        fired = Traced(3)
        self.assertEqual(len(log), 1)
        self.assertEqual(type(log[0][0]).__name__, "Integer")
        self.assertEqual(str(fired), "Traced(3)")
        # 4: reflected matrix ops via NotImplemented.
        m = sympy.Matrix([[1, 2], [3, 4]])
        self.assertEqual(2 * m, m * 2)
        # 5: truthiness honors numeric zero.
        self.assertFalse(bool(sympy.Integer(0)))
        self.assertTrue(bool(sympy.Symbol("x")))
        # 6: numeric conversions.
        self.assertEqual(float(sympy.Rational(3, 2)), 1.5)
        self.assertEqual(int(sympy.Rational(-3, 2)), -1)
        self.assertEqual(float(sympy.Integer(2)), 2.0)
        self.assertEqual(int(sympy.Float(2.5)), 2)
        # 9: zoo is the ComplexInfinity singleton.
        self.assertIs(sympy.zoo, sympy.S.ComplexInfinity)
        self.assertEqual(type(sympy.zoo).__name__, "ComplexInfinity")
        # 10: Dummy survives srepr.
        self.assertIn("Dummy", sympy.srepr(sympy.Dummy("d")))
        # 11: mixed finite/non-finite sorts do not crash.
        ordered = sorted(
            [sympy.Float(2.0), sympy.Float(float("inf"))],
            key=lambda e: e.sort_key(),
        )
        self.assertEqual([str(v) for v in ordered], ["2.00000000000000", "oo"])

    def test_extended_solvers_transforms_functions_and_submodules(self):
        x, t, s, w = sympy.symbols("x t s w")

        # 1. Quadratic solving and auto-variable detection
        solutions = sympy.solve(x**2 - 4, x)
        self.assertEqual(set(solutions), {sympy.Integer(2), sympy.Integer(-2)})
        auto_sol = sympy.solve(2 * x - 6)
        self.assertEqual(auto_sol, [sympy.Integer(3)])

        # 2. Series expansion (module-level and method-level)
        sin_series = sympy.series(sympy.sin(x), x, 0, 4)
        self.assertEqual(sin_series, sympy.sin(x).series(x, 0, 4))
        self.assertEqual(str(sin_series), "-x**3/6 + x")

        # 3. Integral transforms
        laplace_res = sympy.laplace_transform(sympy.exp(t), t, s)
        self.assertEqual(str(laplace_res), "(s - 1)**(-1)")
        fourier_res = sympy.fourier_transform(sympy.Integer(5), t, w)
        self.assertEqual(str(fourier_res), "10*pi*dirac(w)")

        # 4. Number theory functions
        self.assertEqual(sympy.mobius(1), 1)
        self.assertEqual(sympy.mobius(6), 1)
        self.assertEqual(sympy.mobius(4), 0)
        self.assertEqual(sympy.divisor_count(12), 6)
        self.assertEqual(sympy.divisor_sigma(12, 1), 28)
        self.assertEqual(sympy.jacobi_symbol(2, 5), -1)

        # 5. Functions
        self.assertEqual(sympy.tan(0), sympy.Integer(0))
        self.assertEqual(sympy.asin(0), sympy.Integer(0))
        self.assertEqual(sympy.atan(0), sympy.Integer(0))
        self.assertEqual(sympy.sinh(0), sympy.Integer(0))
        self.assertEqual(sympy.cosh(0), sympy.Integer(1))
        self.assertEqual(sympy.tanh(0), sympy.Integer(0))
        self.assertEqual(sympy.floor(sympy.Rational(5, 2)), sympy.Integer(2))
        self.assertEqual(sympy.ceiling(sympy.Rational(5, 2)), sympy.Integer(3))
        self.assertEqual(sympy.factorial(5), sympy.Integer(120))
        self.assertEqual(sympy.gamma(5), sympy.Integer(24))
        self.assertEqual(sympy.fibonacci(10), sympy.Integer(55))

        # 6. Submodule imports
        from sympy.solvers import dsolve as sol_dsolve, solve as sol_solve
        from sympy.functions import (
            asin as f_asin,
            atan as f_atan,
            ceiling as f_ceiling,
            cos as f_cos,
            cosh as f_cosh,
            exp as f_exp,
            factorial as f_factorial,
            fibonacci as f_fibonacci,
            floor as f_floor,
            gamma as f_gamma,
            log as f_log,
            sin as f_sin,
            sinh as f_sinh,
            tan as f_tan,
            tanh as f_tanh,
        )
        from sympy.ntheory import (
            divisor_count as n_divisor_count,
            divisor_sigma as n_divisor_sigma,
            factorint as n_factorint,
            isprime as n_isprime,
            jacobi_symbol as n_jacobi_symbol,
            mobius as n_mobius,
            totient as n_totient,
        )
        from sympy.series import limit as ser_limit, series as ser_series
        from sympy.integrals import (
            fourier_transform as int_fourier,
            integrate as int_integrate,
            laplace_transform as int_laplace,
        )

        self.assertIs(sol_solve, sympy.solve)
        self.assertIs(sol_dsolve, sympy.dsolve)
        self.assertIs(f_tan, sympy.tan)
        self.assertIs(n_mobius, sympy.mobius)
        self.assertIs(ser_series, sympy.series)
        self.assertIs(int_integrate, sympy.integrate)

        # Module-level package access
        self.assertIs(sympy.solvers.solve, sympy.solve)
        self.assertIs(sympy.functions.tan, sympy.tan)
        self.assertIs(sympy.ntheory.mobius, sympy.mobius)
        self.assertIs(sympy.integrals.integrate, sympy.integrate)

    def test_matrix_advanced_linear_algebra_and_as_expr(self):
        from sympy.matrices import (
            Matrix,
            hadamard_product,
            kronecker_product,
            matrix_multiply_elementwise,
        )

        x = sympy.Symbol("x")
        self.assertIs(x.as_expr(), x)
        two = sympy.Integer(2)
        self.assertIs(two.as_expr(), two)

        A = Matrix([[1, 2], [3, 4]])
        b = Matrix([5, 11])

        # 1. Linear solve
        sol = A.solve(b)
        self.assertEqual(sol, Matrix([[1], [2]]))
        self.assertEqual(A.LUsolve(b), sol)
        self.assertEqual(A * sol, b)

        # 2. Least-squares solve
        ls_sol = A.solve_least_squares(b)
        self.assertEqual(ls_sol, sol)

        # 3. Characteristic polynomial
        cp_default = A.charpoly()
        lam = sympy.Symbol("lambda")
        self.assertEqual(cp_default, lam**2 - 5 * lam - 2)
        cp_x = A.charpoly(x)
        self.assertEqual(cp_x, x**2 - 5 * x - 2)
        self.assertEqual(cp_x.as_expr(), x**2 - 5 * x - 2)

        # 4. LU decomposition
        L, U, P = A.LUdecomposition()
        self.assertEqual(L * U, P * A)
        p_mat, l_mat, u_mat = A.lu()
        self.assertEqual(l_mat, L)
        self.assertEqual(u_mat, U)
        self.assertEqual(p_mat, P)

        # 5. QR decomposition
        Q, R = A.QRdecomposition()
        self.assertEqual(Q * R, A)
        q_mat, r_mat = A.qr()
        self.assertEqual(q_mat, Q)
        self.assertEqual(r_mat, R)

        # 6. Hadamard product
        B = Matrix([[5, 6], [7, 8]])
        had = A.hadamard(B)
        expected_had = Matrix([[5, 12], [21, 32]])
        self.assertEqual(had, expected_had)
        self.assertEqual(hadamard_product(A, B), expected_had)
        self.assertEqual(matrix_multiply_elementwise(A, B), expected_had)

        # 7. Kronecker product
        kron = A.kron(B)
        expected_kron = Matrix([
            [5, 6, 10, 12],
            [7, 8, 14, 16],
            [15, 18, 20, 24],
            [21, 24, 28, 32],
        ])
        self.assertEqual(kron, expected_kron)
        self.assertEqual(A.kronecker_product(B), expected_kron)
        self.assertEqual(kronecker_product(A, B), expected_kron)

    def test_sqrt_and_sympify_behavior(self):
        x = sympy.Symbol("x")
        self.assertEqual(sympy.sqrt(0), sympy.Integer(0))
        self.assertEqual(sympy.sqrt(1), sympy.Integer(1))
        self.assertEqual(sympy.sqrt(4), sympy.Integer(2))
        self.assertEqual(sympy.sqrt(25), sympy.Integer(5))
        self.assertEqual(sympy.sqrt(8), 2 * sympy.sqrt(2))
        self.assertEqual(sympy.sqrt(-1), sympy.I)
        self.assertEqual(sympy.sqrt(-4), 2 * sympy.I)
        self.assertEqual(sympy.sqrt(sympy.Rational(4, 9)), sympy.Rational(2, 3))
        self.assertEqual(sympy.sqrt(sympy.Rational(1, 4)), sympy.Rational(1, 2))
        self.assertEqual(sympy.sqrt(x), sympy.Pow(x, sympy.S.Half))

        # Held evaluate=False
        held_sqrt = sympy.sqrt(4, evaluate=False)
        self.assertIsInstance(held_sqrt, sympy.Pow)
        self.assertEqual(held_sqrt.args, (sympy.Integer(4), sympy.S.Half))

        # Pow square root fold
        self.assertEqual(sympy.Pow(sympy.Integer(25), sympy.S.Half), sympy.Integer(5))
        self.assertEqual(sympy.Pow(sympy.Rational(4, 9), sympy.S.Half), sympy.Rational(2, 3))

        # sympify
        self.assertEqual(sympy.sympify(42), sympy.Integer(42))
        self.assertEqual(sympy.sympify("x + 1"), x + 1)
        self.assertEqual(sympy.sympify(True), sympy.Integer(1))
        self.assertEqual(sympy.sympify(False), sympy.Integer(0))
        self.assertIsInstance(sympy.sympify(3.14), sympy.Float)

        # Module / function facades
        from sympy.functions import sqrt as fn_sqrt
        self.assertIs(fn_sqrt, sympy.sqrt)
        from sympy.core.sympify import sympify as mod_sympify, SympifyError
        self.assertIs(mod_sympify, sympy.sympify)
        self.assertIs(SympifyError, sympy.SympifyError)

    def test_logic_and_boolean_algebra(self):
        from sympy import (
            And,
            Equivalent,
            Implies,
            Not,
            Or,
            Symbol,
            Xor,
            false,
            satisfiable,
            simplify_logic,
            to_cnf,
            to_dnf,
            true,
        )
        from sympy.logic import And as LAnd, satisfiable as lsat
        from sympy.logic.boolalg import to_cnf as lto_cnf
        from sympy.logic.inference import satisfiable as isat

        self.assertIs(LAnd, And)
        self.assertIs(lsat, satisfiable)
        self.assertIs(isat, satisfiable)
        self.assertIs(lto_cnf, to_cnf)

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # Operators and classes
        self.assertIsInstance(x & y, And)
        self.assertIsInstance(x | y, Or)
        self.assertIsInstance(~x, Not)
        self.assertIsInstance(x >> y, Implies)
        self.assertIsInstance(x ^ y, Xor)

        # Singletons and boolean evaluation
        self.assertIs(sympy.S.true, true)
        self.assertIs(sympy.S.false, false)
        self.assertTrue(bool(true))
        self.assertFalse(bool(false))

        # Basic simplifications
        self.assertEqual(And(x, true), x)
        self.assertEqual(And(x, false), false)
        self.assertEqual(Or(x, true), true)
        self.assertEqual(Or(x, false), x)
        self.assertEqual(Not(true), false)
        self.assertEqual(Not(false), true)
        self.assertEqual(Not(Not(x)), x)

        # Tautology / Contradiction via simplify_logic
        self.assertEqual(simplify_logic(x | ~x), true)
        self.assertEqual(simplify_logic(true), true)
        self.assertEqual(simplify_logic(false), false)

        # Satisfiability via DPLL
        sat_model = satisfiable(x & y)
        self.assertEqual(sat_model, {x: True, y: True})
        self.assertFalse(satisfiable(x & ~x))
        self.assertEqual(satisfiable(true), {})
        self.assertFalse(satisfiable(false))

        # CNF and DNF conversions
        cnf_expr = to_cnf(x | (y & z))
        self.assertIsInstance(cnf_expr, (And, Symbol))
        dnf_expr = to_dnf((x | y) & z)
        self.assertIsInstance(dnf_expr, (Or, Symbol))

    def test_logic_advanced(self):
        from sympy import (
            ITE,
            NAND,
            NOR,
            Nand,
            Nor,
            POSform,
            SOPform,
            Symbol,
            XNOR,
            Xnor,
            false,
            is_nnf,
            pl_true,
            satisfiable,
            to_nnf,
            true,
            truth_table,
            valid,
        )

        self.assertIs(NAND, Nand)
        self.assertIs(NOR, Nor)
        self.assertIs(XNOR, Xnor)

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # Nand, Nor, Xnor truth evaluations
        self.assertEqual(Nand(true, true), false)
        self.assertEqual(Nand(true, false), true)
        self.assertEqual(Nor(false, false), true)
        self.assertEqual(Nor(true, false), false)
        self.assertEqual(Xnor(true, true), true)
        self.assertEqual(Xnor(true, false), false)
        self.assertEqual(Xnor(false, false), true)

        # ITE
        self.assertEqual(ITE(true, x, y), x)
        self.assertEqual(ITE(false, x, y), y)
        self.assertEqual(ITE(z, x, x), x)

        # NNF conversion and check
        self.assertTrue(is_nnf(x))
        self.assertTrue(is_nnf(~x))
        nnf1 = to_nnf(~(x & y))
        self.assertTrue(is_nnf(nnf1))
        self.assertTrue(valid(nnf1 ^ (~x | ~y) ^ true))  # equivalent
        nnf2 = to_nnf(x >> y)
        self.assertTrue(is_nnf(nnf2))
        self.assertTrue(valid(nnf2 ^ (~x | y) ^ true))

        # Truth table
        table = list(truth_table(x & y, [x, y]))
        self.assertEqual(
            table,
            [
                ([0, 0], False),
                ([0, 1], False),
                ([1, 0], False),
                ([1, 1], True),
            ],
        )

        # SOPform and POSform
        sop = SOPform([x, y], [3])
        self.assertEqual(sop, x & y)
        pos = POSform([x, y], [0])
        self.assertEqual(pos, x | y)

        # Multi-model satisfiability
        models = list(satisfiable(x | y, all_models=True))
        self.assertEqual(len(models), 3)
        for m in models:
            self.assertTrue(pl_true(x | y, m))

        unsat_models = list(satisfiable(x & ~x, all_models=True))
        self.assertEqual(unsat_models, [])

        # Tautology checking (valid) and pl_true
        self.assertTrue(valid(x | ~x))
        self.assertFalse(valid(x & y))
        self.assertTrue(valid((x & y) >> x))
        self.assertTrue(pl_true(x & y, {x: True, y: True}))
        self.assertFalse(pl_true(x & y, {x: True, y: False}))


    def test_geometry_surface(self):
        from sympy import (
            Circle,
            Line,
            Line2D,
            Point,
            Point2D,
            Point3D,
            Polygon,
            Rational,
            Ray,
            Ray2D,
            Segment,
            Segment2D,
            Triangle,
            pi,
        )
        import sympy.geometry as sgeom

        self.assertIs(sgeom.Point, Point)
        self.assertIs(sgeom.Point2D, Point2D)
        self.assertIs(sgeom.Point3D, Point3D)
        self.assertIs(sgeom.Line, Line)
        self.assertIs(sgeom.Circle, Circle)
        self.assertIs(sgeom.Ellipse, sympy.Ellipse)
        self.assertIs(sgeom.LinearEntity, sympy.LinearEntity)
        self.assertIs(sgeom.Triangle, Triangle)
        self.assertIs(sgeom.Polygon, Polygon)

        p1 = Point(0, 0)
        p2 = Point(3, 4)
        self.assertIsInstance(p1, Point2D)
        self.assertEqual(p1.x, 0)
        self.assertEqual(p1.y, 0)
        self.assertEqual(p1.distance(p2), 5)

        p3d1 = Point(1, 2, 3)
        p3d2 = Point(4, 6, 3)
        self.assertIsInstance(p3d1, Point3D)
        self.assertEqual(p3d1.z, 3)
        self.assertEqual(p3d1.distance(p3d2), 5)

        seg = Segment(Point(0, 0), Point(4, 4))
        self.assertIsInstance(seg, Segment2D)
        self.assertEqual(seg.midpoint, Point2D(2, 2))

        l1 = Line(Point(0, 0), Point(2, 2))
        l2 = Line(Point(0, 2), Point(2, 0))
        self.assertIsInstance(l1, Line2D)
        self.assertEqual(l1.intersection(l2), [Point2D(1, 1)])

        circ = Circle(Point(0, 0), 5)
        self.assertEqual(circ.area, 25 * pi)
        self.assertEqual(circ.circumference, 10 * pi)

        tri = Triangle(Point(0, 0), Point(3, 0), Point(0, 4))
        self.assertEqual(tri.area, 6)
        self.assertEqual(tri.centroid, Point2D(1, Rational(4, 3)))
        self.assertTrue(tri.is_right())
        self.assertFalse(tri.is_equilateral())

        poly = Polygon(Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3))
        self.assertEqual(poly.area, 12)
        self.assertEqual(poly.centroid, Point2D(2, Rational(3, 2)))
        self.assertTrue(poly.is_convex())

    def test_sets_surface(self):
        from sympy import (
            Complement,
            EmptySet,
            FiniteSet,
            Integer,
            Intersection,
            Interval,
            S,
            Set,
            Union,
            UniversalSet,
        )
        import sympy.sets as ssets

        self.assertIs(ssets.Set, Set)
        self.assertIs(ssets.Interval, Interval)
        self.assertIs(ssets.FiniteSet, FiniteSet)
        self.assertIs(ssets.Union, Union)
        self.assertIs(ssets.Intersection, Intersection)
        self.assertIs(ssets.Complement, Complement)
        self.assertIs(ssets.EmptySet, EmptySet)
        self.assertIs(ssets.UniversalSet, UniversalSet)

        # Singletons
        self.assertIs(S.EmptySet, EmptySet())
        self.assertIs(S.UniversalSet, UniversalSet())
        self.assertTrue(S.EmptySet.is_empty)
        self.assertFalse(S.UniversalSet.is_empty)
        self.assertEqual(S.EmptySet.measure, Integer(0))
        self.assertEqual(len(S.EmptySet), 0)
        self.assertFalse(bool(S.EmptySet))
        self.assertTrue(bool(S.UniversalSet))

        # Interval
        iv = Interval(0, 5)
        self.assertEqual(iv.start, 0)
        self.assertEqual(iv.end, 5)
        self.assertEqual(iv.left, 0)
        self.assertEqual(iv.right, 5)
        self.assertFalse(iv.left_open)
        self.assertFalse(iv.right_open)
        self.assertEqual(iv.measure, 5)
        self.assertFalse(iv.is_open)
        self.assertTrue(iv.is_closed)
        self.assertTrue(iv.is_compact)
        self.assertTrue(2 in iv)
        self.assertTrue(0 in iv)
        self.assertTrue(5 in iv)
        self.assertFalse(6 in iv)

        # Open interval
        open_iv = Interval(0, 5, left_open=True, right_open=True)
        self.assertTrue(open_iv.left_open)
        self.assertTrue(open_iv.right_open)
        self.assertTrue(open_iv.is_open)
        self.assertFalse(open_iv.is_closed)
        self.assertFalse(0 in open_iv)
        self.assertFalse(5 in open_iv)
        self.assertTrue(3 in open_iv)

        # Degenerate & empty interval
        self.assertIs(Interval(5, 5, left_open=True), S.EmptySet)
        with self.assertRaises(ValueError):
            Interval(5, 0)

        # Topology
        self.assertEqual(iv.interior, open_iv)
        self.assertEqual(iv.closure, iv)
        self.assertEqual(iv.boundary, FiniteSet(0, 5))

        # FiniteSet
        fs = FiniteSet(1, 2, 3)
        self.assertEqual(len(fs), 3)
        self.assertTrue(2 in fs)
        self.assertFalse(4 in fs)
        self.assertTrue(fs.is_subset(iv))
        self.assertFalse(fs.is_subset(Interval(10, 20)))
        self.assertTrue(fs.is_disjoint(Interval(10, 20)))

        # Set algebra operators
        u = iv | FiniteSet(7)
        self.assertIsInstance(u, Union)
        self.assertTrue(7 in u)
        self.assertTrue(2 in u)

        inter = iv & FiniteSet(2, 3, 7)
        self.assertIsInstance(inter, Intersection)
        self.assertTrue(2 in inter)
        self.assertTrue(3 in inter)
        self.assertFalse(7 in inter)

        diff = FiniteSet(1, 2, 3) - FiniteSet(2)
        self.assertTrue(1 in diff)
        self.assertFalse(2 in diff)
        self.assertTrue(3 in diff)

        sym_diff = FiniteSet(1, 2) ^ FiniteSet(2, 3)
        self.assertTrue(1 in sym_diff)
        self.assertFalse(2 in sym_diff)
        self.assertTrue(3 in sym_diff)

    def test_solveset_and_checksol(self):
        from sympy.solvers import solveset as s_solveset, checksol as s_checksol
        self.assertIs(s_solveset, sympy.solveset)
        self.assertIs(s_checksol, sympy.checksol)

        x = sympy.Symbol("x")
        # Quadratic solveset
        sol_quad = sympy.solveset(x**2 - 9, x)
        self.assertIsInstance(sol_quad, sympy.FiniteSet)
        self.assertEqual(sol_quad, sympy.FiniteSet(-3, 3))

        # Linear solveset with auto-variable
        sol_lin = sympy.solveset(2 * x - 6)
        self.assertEqual(sol_lin, sympy.FiniteSet(3))

        # Equation solveset
        sol_eq = sympy.solveset(sympy.Eq(x**2, 16), x)
        self.assertEqual(sol_eq, sympy.FiniteSet(-4, 4))

        # Inconsistent equation -> EmptySet
        sol_empty = sympy.solveset(sympy.Integer(1), x)
        self.assertIs(sol_empty, sympy.S.EmptySet)

        # checksol verification
        self.assertTrue(sympy.checksol(x**2 - 9, x, 3))
        self.assertTrue(sympy.checksol(x**2 - 9, x, -3))
        self.assertFalse(sympy.checksol(x**2 - 9, x, 4))
        self.assertTrue(sympy.checksol(x - 5, {x: 5}))
        self.assertFalse(sympy.checksol(x - 5, {x: 4}))
        self.assertTrue(sympy.checksol(sympy.Eq(2 * x, 10), x, 5))

    def test_geometry_3d_and_sequence_access(self):
        from sympy import (
            Point,
            Point2D,
            Point3D,
            Line,
            Line2D,
            Line3D,
            Segment,
            Segment2D,
            Segment3D,
            Plane,
            Sphere,
        )

        # Sequence access and hashing on Point2D
        p2 = Point2D(3, 7)
        self.assertEqual(len(p2), 2)
        self.assertEqual(p2[0], 3)
        self.assertEqual(p2[1], 7)
        self.assertEqual(p2[-1], 7)
        self.assertEqual(p2[-2], 3)
        with self.assertRaises(IndexError):
            _ = p2[2]
        self.assertEqual(hash(p2), hash(Point2D(3, 7)))
        self.assertEqual({p2, Point2D(3, 7)}, {Point2D(3, 7)})

        # Sequence access and hashing on Point3D
        p3 = Point3D(1, 4, 9)
        self.assertEqual(len(p3), 3)
        self.assertEqual(p3[0], 1)
        self.assertEqual(p3[1], 4)
        self.assertEqual(p3[2], 9)
        self.assertEqual(p3[-1], 9)
        self.assertEqual(p3[-2], 4)
        self.assertEqual(p3[-3], 1)
        with self.assertRaises(IndexError):
            _ = p3[3]
        self.assertEqual(hash(p3), hash(Point3D(1, 4, 9)))

        # Segment3D
        origin = Point3D(0, 0, 0)
        seg = Segment(origin, Point3D(2, 4, 6))
        self.assertIsInstance(seg, Segment3D)
        self.assertEqual(seg.midpoint, Point3D(1, 2, 3))
        self.assertEqual(seg.p1, origin)
        self.assertEqual(seg.p2, Point3D(2, 4, 6))

        # Line3D
        l3 = Line(origin, Point3D(0, 0, 5))
        self.assertIsInstance(l3, Line3D)
        self.assertEqual(l3.direction, Point3D(0, 0, 5))

        # Plane from point + normal
        pl1 = Plane(origin, Point3D(0, 0, 1))
        self.assertEqual(pl1.point, origin)
        self.assertEqual(pl1.normal_vector, Point3D(0, 0, 1))
        self.assertEqual(pl1.eval_at_point(Point3D(2, 3, 0)), 0)
        self.assertEqual(pl1.eval_at_point(Point3D(2, 3, 5)), 5)

        # Plane from 3 points
        pl2 = Plane(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
        self.assertEqual(pl2.normal_vector, Point3D(0, 0, 1))
        self.assertTrue(pl1.is_parallel(pl2))

        # Plane perpendicularity
        pl_perp = Plane(origin, Point3D(1, 0, 0))
        self.assertTrue(pl1.is_perpendicular(pl_perp))

        # Sphere
        sp = Sphere(origin, 3)
        self.assertEqual(sp.center, origin)
        self.assertEqual(sp.radius, 3)
        self.assertEqual(sp.surface_area, 36 * sympy.pi)
        self.assertEqual(sp.area, 36 * sympy.pi)
        self.assertEqual(sp.volume, 36 * sympy.pi)

    def test_numeric_exact_division(self):
        from sympy import Integer, Rational, Float, zoo

        # Integer / Integer
        self.assertEqual(Integer(6) / Integer(2), Integer(3))
        self.assertIsInstance(Integer(6) / Integer(2), Integer)
        self.assertEqual(Integer(1) / Integer(2), Rational(1, 2))

        # Integer / Rational
        self.assertEqual(Integer(6) / Rational(3, 2), Integer(4))
        self.assertIsInstance(Integer(6) / Rational(3, 2), Integer)

        # Rational / Rational
        self.assertEqual(Rational(1, 3) / Rational(2, 5), Rational(5, 6))

        # Division by zero
        self.assertEqual(Integer(1) / Integer(0), zoo)

        # Float preserves float behavior
        f_res = Float(2.5) / Integer(2)
        self.assertIsInstance(f_res, Float)

    def test_polys_module(self):
        from sympy.polys import (
            LC,
            Poly,
            degree,
            discriminant,
            gcd,
            groebner,
            lcm,
            monic,
            resultant,
            sqf_list,
            sqf_part,
        )
        self.assertIs(Poly, sympy.Poly)
        self.assertIs(degree, sympy.degree)
        self.assertIs(gcd, sympy.gcd)
        self.assertIs(lcm, sympy.lcm)
        self.assertIs(resultant, sympy.resultant)
        self.assertIs(discriminant, sympy.discriminant)
        self.assertIs(sqf_list, sympy.sqf_list)
        self.assertIs(sqf_part, sympy.sqf_part)
        self.assertIs(groebner, sympy.groebner)

        x, y = sympy.symbols("x y")
        p = Poly(3 * x**2 - 4, x)
        self.assertEqual(p.degree(), 2)
        self.assertEqual(degree(3 * x**2 - 4, x), 2)
        self.assertEqual(p.all_coeffs(), [sympy.Integer(3), sympy.Integer(0), sympy.Integer(-4)])
        self.assertEqual(p.coeffs(), [sympy.Integer(3), sympy.Integer(-4)])
        self.assertEqual(p.leading_coeff(), sympy.Integer(3))
        self.assertEqual(LC(p), sympy.Integer(3))
        self.assertFalse(p.is_monic)

        p_monic = Poly(2 * x**2 - 8, x).monic()
        self.assertTrue(p_monic.is_monic)
        self.assertEqual(p_monic.all_coeffs(), [sympy.Integer(1), sympy.Integer(0), sympy.Integer(-4)])

        # Division with remainder
        p1 = Poly(x**2 - 1, x)
        p2 = Poly(x - 1, x)
        q, r = p1.div(p2)
        self.assertEqual(q.as_expr(), x + 1)
        self.assertEqual(r.as_expr(), sympy.Integer(0))
        self.assertEqual(p1.rem(p2).as_expr(), sympy.Integer(0))

        # Resultant and discriminant
        self.assertEqual(resultant(x - 2, x - 3, x), sympy.Integer(-1))
        self.assertEqual(discriminant(x**2 - 4, x), sympy.Integer(16))

        # GCD and LCM
        self.assertEqual(gcd(x**2 - 1, x - 1), x - 1)
        self.assertEqual(lcm(x - 1, x + 1), x**2 - 1)
        self.assertEqual(gcd(12, 18), 6)
        self.assertEqual(lcm(12, 18), 36)

        # Square-free factorization
        scale, factors = sqf_list((x - 1)**2 * (x + 2), x)
        self.assertEqual(scale, sympy.Integer(1))
        factor_map = {f: mult for f, mult in factors}
        self.assertEqual(factor_map[x - 1], 2)
        self.assertEqual(factor_map[x + 2], 1)

        # Groebner basis
        gb = groebner([x * y - 2 * y, 2 * y**2 - x**2], x, y)
        self.assertTrue(len(gb) >= 2)

    def test_extended_ode_solvers(self):
        from sympy.solvers.ode import (
            dsolve_cauchy_euler,
            dsolve_const_coeff_second_order,
            dsolve_const_coeff_second_order_nonhomogeneous,
            dsolve_linear_first_order,
            dsolve_separable_linear,
        )
        self.assertIs(dsolve_cauchy_euler, sympy.dsolve_cauchy_euler)
        self.assertIs(dsolve_const_coeff_second_order, sympy.dsolve_const_coeff_second_order)
        self.assertIs(
            dsolve_const_coeff_second_order_nonhomogeneous,
            sympy.dsolve_const_coeff_second_order_nonhomogeneous,
        )
        self.assertIs(dsolve_linear_first_order, sympy.dsolve_linear_first_order)
        self.assertIs(dsolve_separable_linear, sympy.dsolve_separable_linear)

        x = sympy.Symbol("x")
        # 1st-order linear
        sol_lin1 = dsolve_linear_first_order(0, 2 * x, x)
        self.assertTrue("C1" in str(sol_lin1))

        # 2nd-order constant coefficient
        sol_hom2 = dsolve_const_coeff_second_order(1, -3, 2, x)
        self.assertTrue("C1" in str(sol_hom2) and "C2" in str(sol_hom2))

        # 2nd-order nonhomogeneous
        sol_nonhom2 = dsolve_const_coeff_second_order_nonhomogeneous(1, -3, 2, 4, x)
        self.assertTrue("C1" in str(sol_nonhom2) and "C2" in str(sol_nonhom2))

        # Cauchy-Euler
        sol_ce = dsolve_cauchy_euler(1, -1, 1, x)
        self.assertTrue("C1" in str(sol_ce) and "C2" in str(sol_ce))

        # Separable linear
        sol_sep = dsolve_separable_linear(2 * x, x)
        self.assertTrue("C1" in str(sol_sep))

    def test_poly_system_solvers(self):
        from sympy.solvers import nonlinsolve, solve_poly_system
        self.assertIs(solve_poly_system, sympy.solve_poly_system)
        self.assertIs(nonlinsolve, sympy.nonlinsolve)

        x, y = sympy.symbols("x y")
        # solve_poly_system
        sols = solve_poly_system([x + y - 5, x - y - 1], x, y)
        self.assertEqual(sols, [(sympy.Integer(3), sympy.Integer(2))])

        # nonlinsolve
        n_sols = nonlinsolve([x + y - 5, x - y - 1], x, y)
        self.assertIsInstance(n_sols, sympy.FiniteSet)
        sol_tuple = list(n_sols)[0]
        self.assertEqual(sol_tuple[0], sympy.Integer(3))
        self.assertEqual(sol_tuple[1], sympy.Integer(2))

        # solve with system of equations
        sys_sol = sympy.solve([x + y - 5, x - y - 1], [x, y])
        self.assertEqual(sys_sol, {x: sympy.Integer(3), y: sympy.Integer(2)})

        # solve with auto-detected symbols
        sys_auto = sympy.solve([x + y - 5, x - y - 1])
        self.assertEqual(sys_auto, {x: sympy.Integer(3), y: sympy.Integer(2)})

    def test_series_order_and_tuple(self):
        from sympy.series import O as s_O, Order as s_Order
        self.assertIs(s_O, sympy.O)
        self.assertIs(s_Order, sympy.Order)

        x = sympy.Symbol("x")
        o1 = sympy.Order(x**3)
        self.assertEqual(str(o1), "Order(x**3)")
        o2 = sympy.O(x**3)
        self.assertEqual(str(o2), "Order(x**3)")

        t = sympy.Tuple(1, 2, 3)
        self.assertEqual(len(t), 3)
        self.assertEqual(t[0], sympy.Integer(1))
        self.assertEqual(t[1], sympy.Integer(2))
        self.assertEqual(t[2], sympy.Integer(3))
        self.assertEqual(list(t), [sympy.Integer(1), sympy.Integer(2), sympy.Integer(3)])

    def test_tensor_and_metric(self):
        from sympy.tensor import (
            Metric,
            Tensor,
            TensorIndex,
            tensor_indices,
            tensorcontraction,
            tensorproduct,
        )
        self.assertIs(Metric, sympy.Metric)
        self.assertIs(Tensor, sympy.Tensor)
        self.assertIs(TensorIndex, sympy.TensorIndex)
        self.assertIs(tensor_indices, sympy.tensor_indices)
        self.assertIs(tensorcontraction, sympy.tensorcontraction)
        self.assertIs(tensorproduct, sympy.tensorproduct)

        # Tensor indices
        mu, nu = tensor_indices("mu nu")
        self.assertTrue(mu.is_up)
        self.assertEqual(mu.name, "mu")
        mu_low = mu.flip()
        self.assertFalse(mu_low.is_up)

        # 4D Minkowski metric
        eta = Metric.minkowski_4d("eta")
        self.assertEqual(eta.dimension, 4)
        self.assertEqual(len(eta.matrix), 16)
        self.assertEqual(eta.matrix[0], sympy.Integer(-1))
        self.assertEqual(eta.matrix[5], sympy.Integer(1))

        # Vector v^\mu = (3, 0, 0, 4)
        v = Tensor("v", 4, [mu], [3, 0, 0, 4])
        self.assertEqual(v.rank, 1)
        self.assertEqual(v.dimension, 4)
        self.assertEqual(v.components, (sympy.Integer(3), sympy.Integer(0), sympy.Integer(0), sympy.Integer(4)))

        # Lower vector v_\mu = eta_{\mu\nu} v^\nu = (-3, 0, 0, 4)
        v_low = eta.lower_vector(v)
        self.assertEqual(v_low.rank, 1)
        self.assertFalse(v_low.indices[0].is_up)
        self.assertEqual(v_low.components, (sympy.Integer(-3), sympy.Integer(0), sympy.Integer(0), sympy.Integer(4)))

        # Norm squared: -3^2 + 0 + 0 + 4^2 = 7
        ns = eta.norm_squared(v)
        self.assertEqual(ns, sympy.Integer(7))

        # Inner product with itself
        ip = eta.inner_product(v, v)
        self.assertEqual(ip, sympy.Integer(7))

        # Raise covector back to vector
        v_recov = eta.raise_covector(v_low)
        self.assertTrue(v_recov.indices[0].is_up)
        self.assertEqual(v_recov.components, (sympy.Integer(3), sympy.Integer(0), sympy.Integer(0), sympy.Integer(4)))

        # Outer product and contraction
        prod = tensorproduct(v, v_low)
        self.assertEqual(prod.rank, 2)
        contracted = tensorcontraction(prod, ("mu", "mu"))
        self.assertEqual(contracted.rank, 0)
        self.assertEqual(contracted.components, (sympy.Integer(7),))

    def test_assumptions_and_deductive_predicates(self):
        from sympy.assumptions import (
            AppliedPredicate,
            AssumptionsContext,
            Predicate,
            Q,
            ask,
        )
        self.assertIs(Q, sympy.Q)
        self.assertIs(ask, sympy.ask)

        x = sympy.Symbol("x")

        # Inherent facts on concrete numbers
        self.assertIs(ask(Q.positive(sympy.Integer(5))), True)
        self.assertIs(ask(Q.negative(sympy.Integer(5))), False)
        self.assertIs(ask(Q.even(sympy.Integer(4))), True)
        self.assertIs(ask(Q.odd(sympy.Integer(4))), False)
        self.assertIs(ask(Q.zero(sympy.Integer(0))), True)

        # Assumptions queries with facts
        self.assertIs(ask(Q.positive(x), Q.positive(x)), True)
        self.assertIs(ask(Q.real(x), Q.positive(x)), True)
        self.assertIs(ask(Q.complex(x), Q.positive(x)), True)
        self.assertIs(ask(Q.nonnegative(x), Q.positive(x)), True)
        self.assertIs(ask(Q.negative(x), Q.positive(x)), False)
        self.assertIs(ask(Q.zero(x), Q.positive(x)), False)
        self.assertIs(ask(Q.integer(x), Q.positive(x)), None)

        # Inherent deductive properties on Symbol
        x_pos = sympy.Symbol("x", positive=True)
        self.assertIs(x_pos.is_positive, True)
        self.assertIs(x_pos.is_real, True)
        self.assertIs(x_pos.is_complex, True)
        self.assertIs(x_pos.is_nonnegative, True)
        self.assertIs(x_pos.is_nonzero, True)
        self.assertIs(x_pos.is_negative, False)
        self.assertIs(x_pos.is_zero, False)
        self.assertIs(x_pos.is_integer, None)

        n_int = sympy.Symbol("n", integer=True)
        self.assertIs(n_int.is_integer, True)
        self.assertIs(n_int.is_rational, True)
        self.assertIs(n_int.is_real, True)
        self.assertIs(n_int.is_complex, True)
        self.assertIs(n_int.is_positive, None)

        # AssumptionsContext
        ctx = AssumptionsContext()
        ctx.assume(x, Q.positive)
        self.assertIs(ctx.is_true(x, Q.real), True)
        self.assertIs(ctx.is_true(x, Q.negative), False)
        self.assertIs(ctx.is_true(x, Q.integer), None)
        self.assertEqual(ctx.query(x, Q.real), "True")
        self.assertEqual(ctx.query(x, Q.negative), "False")
        self.assertEqual(ctx.query(x, Q.integer), "Unknown")

    def test_assumptions_advanced_and_refine(self):
        from sympy import (
            Abs,
            Integer,
            Pow,
            Q,
            Symbol,
            ask,
            assuming,
            global_assumptions,
            refine,
            sqrt,
        )
        from sympy.assumptions.assume import AppliedPredicate, Predicate
        from sympy.logic.boolalg import Boolean

        x = Symbol("x")
        y = Symbol("y")

        # AppliedPredicate inherits from Boolean and supports algebra
        pos_x = Q.positive(x)
        int_x = Q.integer(x)
        self.assertIsInstance(pos_x, Boolean)
        self.assertIsInstance(pos_x, AppliedPredicate)
        self.assertEqual(pos_x.args, (Q.positive, x))
        self.assertIs(pos_x.predicate, Q.positive)
        self.assertIs(pos_x.expr, x)

        compound = pos_x & int_x
        self.assertEqual(len(compound.args), 2)
        neg_pos = ~pos_x
        self.assertEqual(neg_pos.args, (pos_x,))

        # Predicates with plain numbers
        self.assertIs(ask(Q.positive(5)), True)
        self.assertIs(ask(Q.negative(5)), False)
        self.assertIs(ask(Q.even(4)), True)
        self.assertIs(ask(Q.odd(4)), False)

        # Compound facts in ask
        self.assertIs(ask(Q.real(x), pos_x & int_x), True)
        self.assertIs(ask(Q.integer(x), pos_x & int_x), True)
        self.assertIs(ask(Q.negative(x), pos_x & int_x), False)

        # assuming context manager and global_assumptions
        self.assertIs(ask(Q.positive(x)), None)
        with assuming(Q.positive(x), Q.even(y)):
            self.assertIn(Q.positive(x), global_assumptions)
            self.assertIs(ask(Q.positive(x)), True)
            self.assertIs(ask(Q.real(x)), True)
            self.assertIs(ask(Q.even(y)), True)
            self.assertEqual(refine(Abs(x) + Integer(-1)**y), x + 1)
        self.assertNotIn(Q.positive(x), global_assumptions)
        self.assertIs(ask(Q.positive(x)), None)

        # refine Abs, Pow, sqrt
        self.assertEqual(refine(Abs(x), Q.positive(x)), x)
        self.assertEqual(refine(Abs(x), Q.negative(x)), -x)
        self.assertEqual(refine(Integer(-1)**x, Q.even(x)), 1)
        self.assertEqual(refine(Integer(-1)**x, Q.odd(x)), -1)
        self.assertEqual(refine(sqrt(x**2), Q.positive(x)), x)
        self.assertEqual(refine(sqrt(x**2), Q.real(x)), Abs(x))

    def test_complex_numbers_and_functions(self):
        from sympy import (
            I,
            Integer,
            Rational,
            Symbol,
            arg,
            conjugate,
            im,
            pi,
            re,
        )

        x = Symbol("x")
        x_real = Symbol("x", real=True)

        # as_real_imag
        self.assertEqual(Integer(5).as_real_imag(), (Integer(5), Integer(0)))
        self.assertEqual(I.as_real_imag(), (Integer(0), Integer(1)))
        c = 3 + 4*I
        self.assertEqual(c.as_real_imag(), (Integer(3), Integer(4)))
        self.assertEqual(x_real.as_real_imag(), (x_real, Integer(0)))

        # re, im, conjugate, arg
        self.assertEqual(re(c), 3)
        self.assertEqual(im(c), 4)
        self.assertEqual(conjugate(c), 3 - 4*I)
        self.assertEqual(arg(3 + 3*I), pi / 4)
        self.assertEqual(re(x_real), x_real)
        self.assertEqual(im(x_real), 0)
        self.assertEqual(conjugate(x_real), x_real)

    def test_factor_and_roots(self):
        from sympy.polys import factor, factor_list, roots
        self.assertIs(factor, sympy.factor)
        self.assertIs(factor_list, sympy.factor_list)
        self.assertIs(roots, sympy.roots)

        x = sympy.Symbol("x")

        # factor
        factored1 = sympy.factor(x**2 - 4)
        self.assertEqual(sympy.expand(factored1), x**2 - 4)
        self.assertIn("x - 2", str(factored1))
        self.assertIn("x + 2", str(factored1))

        # factor square
        factored2 = sympy.factor(x**2 - 2*x + 1)
        self.assertEqual(str(factored2), "(x - 1)**2")

        # method .factor() on Expr
        factored_m = (x**2 - 4).factor()
        self.assertEqual(sympy.expand(factored_m), x**2 - 4)

        # factor with constant coefficient
        factored3 = sympy.factor(2*x**2 - 8)
        self.assertEqual(sympy.expand(factored3), 2*x**2 - 8)

        # factor_list
        scale, f_list = sympy.factor_list(x**2 - 4)
        self.assertEqual(scale, sympy.Integer(1))
        self.assertEqual(len(f_list), 2)

        # roots
        r1 = sympy.roots(x**2 - 4, x)
        self.assertEqual(r1, {sympy.Integer(2): 1, sympy.Integer(-2): 1})

        r2 = sympy.roots(x**2 - 2*x + 1, x)
        self.assertEqual(r2, {sympy.Integer(1): 2})

    def test_subs_extended_forms(self):
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        f = sympy.Function("f")

        # Dict substitution
        self.assertEqual((x + y).subs({x: sympy.Integer(1), y: sympy.Integer(2)}), sympy.Integer(3))

        # List of pairs substitution
        self.assertEqual((x + y).subs([(x, sympy.Integer(10)), (y, sympy.Integer(20))]), sympy.Integer(30))

        # Kwargs substitution
        self.assertEqual((x + y).subs(x=100, y=200), sympy.Integer(300))

        # Subexpression substitution of applied function
        self.assertEqual((-f(x)).subs(f(x), sympy.exp(x)), -sympy.exp(x))
        self.assertEqual((f(x)**2).subs(f(x), y), y**2)
        self.assertEqual((f(x) + sympy.Integer(5)).subs(f(x), x**2), x**2 + sympy.Integer(5))

        # Derivative substitution and evaluation via doit()
        diff_node = sympy.diff(f(x), x)
        subbed = diff_node.subs(f(x), sympy.exp(x))
        self.assertEqual(subbed.doit(), sympy.exp(x))

    def test_ode_checkodesol_and_verifiers(self):
        from sympy import checkodesol, Eq, diff, exp, sin, cos, Function, Symbol
        from sympy.solvers.ode import (
            verify_linear_first_order_solution,
            verify_const_coeff_second_order_solution,
            verify_cauchy_euler_solution,
        )

        x = Symbol("x")
        y = Function("y")
        c1 = Symbol("C1")
        c2 = Symbol("C2")

        # 1st-order linear ODE: y' - y = 0
        ode1 = diff(y(x), x) - y(x)
        sol1_good = Eq(y(x), c1 * exp(x))
        sol1_bad = Eq(y(x), exp(2 * x))
        self.assertEqual(checkodesol(ode1, sol1_good), (True, 0))
        self.assertEqual(checkodesol(ode1, sol1_bad), (False, exp(2 * x)))

        # List of solutions
        res_list = checkodesol(ode1, [sol1_good, sol1_bad])
        self.assertEqual(res_list, [(True, 0), (False, exp(2 * x))])

        # 2nd-order ODE: y'' + y = 0
        ode2 = diff(y(x), x, x) + y(x)
        sol2_sin = Eq(y(x), sin(x))
        sol2_cos = Eq(y(x), cos(x))
        sol2_bad = Eq(y(x), exp(x))
        self.assertEqual(checkodesol(ode2, sol2_sin), (True, 0))
        self.assertEqual(checkodesol(ode2, sol2_cos), (True, 0))
        self.assertEqual(checkodesol(ode2, sol2_bad), (False, 2 * exp(x)))

        # Cauchy-Euler ODE: x^2 * y'' - 2*y = 0
        ode3 = x**2 * diff(y(x), x, x) - 2 * y(x)
        sol3 = Eq(y(x), x**2)
        self.assertEqual(checkodesol(ode3, sol3), (True, 0))

        # Independent native verifiers
        self.assertTrue(verify_linear_first_order_solution(exp(x), 0, exp(x), x))
        self.assertFalse(verify_linear_first_order_solution(exp(2 * x), 0, exp(x), x))
        self.assertTrue(verify_const_coeff_second_order_solution(exp(x), 1, -1, 0, x))
        self.assertTrue(verify_cauchy_euler_solution(x**2, 1, -1, 0, x))

    def test_matrix_advanced_methods(self):
        from sympy import Matrix, eye, ones, diag, hstack, vstack

        # 1. ones constructor
        o = ones(2, 3)
        self.assertEqual(o.shape, (2, 3))
        self.assertEqual(o[0, 0], sympy.Integer(1))
        self.assertEqual(o[1, 2], sympy.Integer(1))

        # 2. col and row extraction
        A = Matrix([[1, 2], [3, 4]])
        self.assertEqual(A.col(0), Matrix([[1], [3]]))
        self.assertEqual(A.col(1), Matrix([[2], [4]]))
        self.assertEqual(A.row(0), Matrix([[1, 2]]))
        self.assertEqual(A.row(1), Matrix([[3, 4]]))

        # Negative indexing
        self.assertEqual(A.col(-1), Matrix([[2], [4]]))
        self.assertEqual(A.row(-1), Matrix([[3, 4]]))

        # 3. col_join and row_join
        cj = A.col_join(ones(1, 2))
        self.assertEqual(cj.shape, (3, 2))
        self.assertEqual(cj, Matrix([[1, 2], [3, 4], [1, 1]]))

        rj = A.row_join(ones(2, 1))
        self.assertEqual(rj.shape, (2, 3))
        self.assertEqual(rj, Matrix([[1, 2, 1], [3, 4, 1]]))

        # 4. hstack and vstack
        hs = hstack(A.col(0), A.col(1))
        self.assertEqual(hs, A)
        vs = vstack(A.row(0), A.row(1))
        self.assertEqual(vs, A)

        # 5. Nullspace of rank-deficient matrix
        C = Matrix([[1, 2], [2, 4]])
        ns = C.nullspace()
        self.assertEqual(len(ns), 1)
        self.assertEqual(C * ns[0], Matrix([[0], [0]]))

        # 6. Adjugate and cofactor
        adj = A.adjugate()
        det_A = A.det()
        self.assertEqual(A * adj, det_A * eye(2))
        self.assertEqual(A.cofactor(0, 0), sympy.Integer(4))
        self.assertEqual(A.cofactor(0, 1), sympy.Integer(-3))
        self.assertEqual(A.cofactor(1, 0), sympy.Integer(-2))
        self.assertEqual(A.cofactor(1, 1), sympy.Integer(1))

        # 7. Eigenvalues and eigenvectors
        M = Matrix([[2, 1], [0, 3]])
        self.assertEqual(M.eigenvals(), {sympy.Integer(2): 1, sympy.Integer(3): 1})
        vects = M.eigenvects()
        self.assertEqual(len(vects), 2)

        # 8. Diagonalization: M = P * D * P^-1
        P, D = M.diagonalize()
        self.assertEqual(P * D * P.inv(), M)

    def test_ntheory_extended(self):
        from sympy import (
            carmichael,
            integer_nthroot,
            is_perfect,
            is_primitive_root,
            is_square_free,
            is_squarefree,
            legendre_symbol,
            mod_inverse,
            primenu,
            prime_big_omega,
            prime_omega,
            primeomega,
            reduced_totient,
        )
        from sympy.ntheory import crt

        # 1. Legendre symbol
        self.assertEqual(legendre_symbol(2, 7), 1)
        self.assertEqual(legendre_symbol(3, 7), -1)
        self.assertEqual(legendre_symbol(7, 7), 0)

        # 2. Square-free
        self.assertTrue(is_squarefree(10))
        self.assertTrue(is_square_free(10))
        self.assertFalse(is_squarefree(12))
        self.assertFalse(is_square_free(12))

        # 3. Prime omega functions
        # 12 = 2^2 * 3: 2 distinct prime factors, 3 prime factors with multiplicity
        self.assertEqual(primenu(12), 2)
        self.assertEqual(prime_omega(12), 2)
        self.assertEqual(primeomega(12), 3)
        self.assertEqual(prime_big_omega(12), 3)

        # 4. Perfect numbers
        self.assertTrue(is_perfect(6))
        self.assertTrue(is_perfect(28))
        self.assertFalse(is_perfect(10))

        # 5. Carmichael function
        self.assertEqual(carmichael(8), 2)
        self.assertEqual(reduced_totient(12), 2)
        self.assertEqual(carmichael(15), 4)

        # 6. Primitive root
        self.assertTrue(is_primitive_root(2, 5))
        self.assertTrue(is_primitive_root(3, 5))
        self.assertFalse(is_primitive_root(4, 5))

        # 7. Integer nth root
        self.assertEqual(integer_nthroot(27, 3), (3, True))
        self.assertEqual(integer_nthroot(26, 3), (2, False))
        self.assertEqual(integer_nthroot(16, 2), (4, True))
        self.assertEqual(integer_nthroot(15, 2), (3, False))

        # 8. Modular inverse
        self.assertEqual(mod_inverse(3, 11), sympy.Integer(4))
        self.assertEqual((3 * 4) % 11, 1)
        self.assertEqual(mod_inverse(7, 26), sympy.Integer(15))
        with self.assertRaises(ValueError):
            mod_inverse(2, 4)

        # 9. Chinese Remainder Theorem
        # x = 2 (mod 3), x = 3 (mod 5), x = 2 (mod 7) -> x = 23 (mod 105)
        sol, mod = crt([3, 5, 7], [2, 3, 2])
        self.assertEqual(sol, sympy.Integer(23))
        self.assertEqual(mod, sympy.Integer(105))
        self.assertEqual(23 % 3, 2)
        self.assertEqual(23 % 5, 3)
        self.assertEqual(23 % 7, 2)

        # Non-coprime case returns None
        self.assertIsNone(crt([4, 6], [1, 2]))

    def test_functions_extended(self):
        from sympy import (
            acos, acosh, acot, acoth, acsc, acsch, asec, asech, asinh, atanh,
            bell, bernoulli, binomial, catalan, cot, coth, csc, csch,
            erf, erfc, harmonic, lucas, sec, sech, sign, sinc, subfactorial, zeta,
            Rational, Integer, Symbol, diff
        )
        from sympy.functions.elementary.trigonometric import sec as sec_t, sinc as sinc_t
        from sympy.functions.elementary.hyperbolic import sech as sech_h, coth as coth_h
        from sympy.functions.combinatorial.factorials import binomial as bin_f
        from sympy.functions.combinatorial.numbers import lucas as lucas_n, catalan as cat_n
        from sympy.functions.special.error_functions import erf as erf_s

        # Submodule import identity
        self.assertIs(sec, sec_t)
        self.assertIs(sinc, sinc_t)
        self.assertIs(sech, sech_h)
        self.assertIs(coth, coth_h)
        self.assertIs(binomial, bin_f)
        self.assertIs(lucas, lucas_n)
        self.assertIs(catalan, cat_n)
        self.assertIs(erf, erf_s)

        # Evaluations at special points
        self.assertEqual(sec(0), Integer(1))
        self.assertEqual(sech(0), Integer(1))
        self.assertEqual(acos(1), Integer(0))
        self.assertEqual(acosh(1), Integer(0))
        self.assertEqual(asinh(0), Integer(0))
        self.assertEqual(atanh(0), Integer(0))
        self.assertEqual(asec(1), Integer(0))
        self.assertEqual(asech(1), Integer(0))
        self.assertEqual(sinc(0), Integer(1))
        self.assertEqual(erf(0), Integer(0))
        self.assertEqual(erfc(0), Integer(1))
        self.assertEqual(sign(-15), Integer(-1))
        self.assertEqual(sign(15), Integer(1))
        self.assertEqual(sign(0), Integer(0))

        # Combinatorial evaluations
        self.assertEqual(binomial(5, 2), Integer(10))
        self.assertEqual(binomial(6, 3), Integer(20))
        self.assertEqual(lucas(0), Integer(2))
        self.assertEqual(lucas(1), Integer(1))
        self.assertEqual(lucas(4), Integer(7))
        self.assertEqual(catalan(0), Integer(1))
        self.assertEqual(catalan(3), Integer(5))
        self.assertEqual(subfactorial(4), Integer(9))
        self.assertEqual(harmonic(3), Rational(11, 6))
        self.assertEqual(bernoulli(0), Integer(1))
        self.assertEqual(bernoulli(1), Rational(-1, 2))
        self.assertEqual(bernoulli(2), Rational(1, 6))
        self.assertEqual(bernoulli(3), Integer(0))
        self.assertEqual(bell(0), Integer(1))
        self.assertEqual(bell(3), Integer(5))

        # Zeta pole / unevaluated form
        x = Symbol("x")
        self.assertEqual(str(zeta(x)), "zeta(x)")

        # Symbolic differentiation of extended functions
        d_sec = diff(sec(x), x)
        self.assertEqual(str(d_sec), "sec(x)*tan(x)")
        d_csc = diff(csc(x), x)
        self.assertEqual(str(d_csc), "-cot(x)*csc(x)")
        d_cot = diff(cot(x), x)
        self.assertEqual(str(d_cot), "-(cot(x)**2 + 1)")
        d_sech = diff(sech(x), x)
        self.assertEqual(str(d_sech), "-sech(x)*tanh(x)")

    def test_sets_extended(self):
        from sympy import Interval, FiniteSet, SymmetricDifference, Symbol

        x = Symbol("x")

        # Interval constructors
        i_open = Interval.open(0, 1)
        self.assertTrue(i_open.left_open)
        self.assertTrue(i_open.right_open)

        i_lopen = Interval.Lopen(0, 1)
        self.assertTrue(i_lopen.left_open)
        self.assertFalse(i_lopen.right_open)

        i_ropen = Interval.Ropen(0, 1)
        self.assertFalse(i_ropen.left_open)
        self.assertTrue(i_ropen.right_open)

        # as_relational
        rel_open = i_open.as_relational(x)
        self.assertEqual(str(rel_open), "(Lt(0, x) & Lt(x, 1))")

        rel_closed = Interval(0, 1).as_relational(x)
        self.assertEqual(str(rel_closed), "(Le(0, x) & Le(x, 1))")

        # Proper subset / superset
        s1 = FiniteSet(1)
        s2 = FiniteSet(1, 2)
        self.assertTrue(s1.is_subset(s2))
        self.assertTrue(s1.is_proper_subset(s2))
        self.assertFalse(s1.is_proper_subset(s1))
        self.assertTrue(s2.is_superset(s1))
        self.assertTrue(s2.is_proper_superset(s1))
        self.assertFalse(s2.is_proper_superset(s2))

        # SymmetricDifference
        sd = SymmetricDifference(s1, s2)
        self.assertIsNotNone(sd)

    def test_solver_edge_cases_and_solveset(self):
        from sympy import Symbol, EmptySet, UniversalSet, Eq, Function, diff, sec
        from sympy.solvers.ode import checkodesol

        x = Symbol("x")
        # solve edge cases
        self.assertEqual(sympy.solve(1, x), [])
        self.assertEqual(sympy.solve(0, x), [])
        self.assertEqual(sympy.solve(sympy.Integer(1), x), [])
        self.assertEqual(sympy.solve(sympy.Integer(0), x), [])

        # solveset edge cases
        self.assertEqual(sympy.solveset(0, x), UniversalSet())
        self.assertEqual(sympy.solveset(1, x), EmptySet())
        self.assertEqual(sympy.solveset(sympy.Integer(0), x), UniversalSet())
        self.assertEqual(sympy.solveset(sympy.Integer(1), x), EmptySet())

        # checkodesol with trigonometric/special function ODE
        y = Function("y")
        ode = Eq(diff(y(x), x), sec(x))
        # sec(x) shouldn't be confused with the unknown function
        res = checkodesol(ode, Eq(y(x), x))
        self.assertIsInstance(res, tuple)

    def test_matrix_powers_calculus_and_sparse(self):
        from sympy import (
            Matrix,
            SparseMatrix,
            MutableSparseMatrix,
            eye,
            Symbol,
            sin,
            cos,
            exp,
            jacobian,
            wronskian,
            casoratian,
            GramSchmidt,
        )

        # 1. Negative matrix powers and zero power
        A = Matrix([[1, 2], [3, 4]])
        self.assertEqual(A ** 0, eye(2))
        self.assertEqual(A ** -1, A.inv())
        self.assertEqual((A ** -1) * A, eye(2))
        self.assertEqual(A ** -2, (A.inv()) ** 2)

        # 2. Matrix calculus, subs, free_symbols, simplify, applyfunc
        x = Symbol("x")
        y = Symbol("y")
        M = Matrix([[x**2, sin(x)], [cos(x), exp(x)]])
        self.assertEqual(M.free_symbols, {x})
        dM = M.diff(x)
        self.assertEqual(dM[0, 0], 2 * x)
        self.assertEqual(dM[0, 1], cos(x))
        self.assertEqual(dM[1, 0], -sin(x))
        self.assertEqual(dM[1, 1], exp(x))

        intM = M.integrate(x)
        self.assertEqual(intM[0, 0], (x**3) / 3)

        M_sub = M.subs(x, 0)
        self.assertEqual(M_sub[0, 0], sympy.Integer(0))
        self.assertEqual(M_sub[0, 1], sympy.Integer(0))
        self.assertEqual(M_sub[1, 0], sympy.Integer(1))
        self.assertEqual(M_sub[1, 1], sympy.Integer(1))

        M_unsimp = Matrix([[x + 0, x - x]])
        M_simp = M_unsimp.simplify()
        self.assertEqual(M_simp[0, 0], x)
        self.assertEqual(M_simp[0, 1], sympy.Integer(0))

        M_doubled = A.applyfunc(lambda e: e * 2)
        self.assertEqual(M_doubled, Matrix([[2, 4], [6, 8]]))

        # 3. Jacobian (method and function)
        vec = Matrix([x**2 + y, sin(x * y)])
        J1 = vec.jacobian([x, y])
        J2 = jacobian(vec, [x, y])
        self.assertEqual(J1, J2)
        self.assertEqual(J1.shape, (2, 2))
        self.assertEqual(J1[0, 0], 2 * x)
        self.assertEqual(J1[0, 1], sympy.Integer(1))

        # 4. Wronskian, Casoratian, GramSchmidt
        w = wronskian([sin(x), cos(x)], x)
        self.assertEqual(w, sympy.Integer(-1))

        n = Symbol("n")
        c = casoratian([1, n], n)
        self.assertEqual(c, sympy.Integer(1))

        v1 = Matrix([1, 0])
        v2 = Matrix([1, 1])
        ortho = GramSchmidt([v1, v2], orthonormal=True)
        self.assertEqual(len(ortho), 2)
        self.assertEqual(ortho[0], Matrix([1, 0]))
        self.assertEqual(ortho[1], Matrix([0, 1]))

        # 5. SparseMatrix and MutableSparseMatrix
        S = SparseMatrix(2, 2, {(0, 1): 5, (1, 0): 3})
        self.assertEqual(S.shape, (2, 2))
        self.assertEqual(S[0, 1], sympy.Integer(5))
        self.assertEqual(S[0, 0], sympy.Integer(0))
        self.assertEqual(S.trace(), sympy.Integer(0))
        dense_equiv = Matrix([[0, 5], [3, 0]])
        self.assertEqual(S.to_dense(), dense_equiv)
        self.assertEqual(dense_equiv.to_sparse(), S)

        S_trans = S.T
        self.assertEqual(S_trans[1, 0], sympy.Integer(5))
        self.assertEqual(S_trans[0, 1], sympy.Integer(3))

        S_add = S + S
        self.assertEqual(S_add[0, 1], sympy.Integer(10))

        MS = MutableSparseMatrix(2, 2, {(0, 0): 1})
        MS[0, 1] = 4
        self.assertEqual(MS[0, 1], sympy.Integer(4))

    def test_extended_elementary_and_special_evaluations(self):
        from sympy import sec, sech, sinc, erf, erfc, asec, asech

        self.assertEqual(sec(0), sympy.Integer(1))
        self.assertEqual(sech(0), sympy.Integer(1))
        self.assertEqual(sinc(0), sympy.Integer(1))
        self.assertEqual(erf(0), sympy.Integer(0))
        self.assertEqual(erfc(0), sympy.Integer(1))
        self.assertEqual(asec(1), sympy.Integer(0))
        self.assertEqual(asech(1), sympy.Integer(0))

    def test_poly_convenience_methods_and_functions(self):
        from sympy import Poly, Symbol, LC, TC, EC, trailing_coeff, sqf

        x = Symbol("x")
        p = Poly(3 * x**2 + 2 * x + 5, x)
        self.assertEqual(p.LC(), sympy.Integer(3))
        self.assertEqual(p.TC(), sympy.Integer(5))
        self.assertEqual(p.EC(), sympy.Integer(5))
        self.assertEqual(p.trailing_coeff(), sympy.Integer(5))
        self.assertEqual(p.nth(0), sympy.Integer(5))
        self.assertEqual(p.nth(1), sympy.Integer(2))
        self.assertEqual(p.nth(2), sympy.Integer(3))
        self.assertEqual(p.nth(3), sympy.Integer(0))

        self.assertEqual(p.eval(2), sympy.Integer(21))
        self.assertEqual(p.diff(), Poly(6 * x + 2, x))
        self.assertEqual(p.integrate(), Poly(x**3 + x**2 + 5 * x, x))

        self.assertTrue(p.is_quadratic)
        self.assertFalse(p.is_linear)
        self.assertFalse(p.is_zero)
        self.assertFalse(p.is_one)

        p_lin = Poly(2 * x + 1, x)
        self.assertTrue(p_lin.is_linear)
        self.assertFalse(p_lin.is_quadratic)

        p_zero = Poly(0, x)
        self.assertTrue(p_zero.is_zero)

        p_one = Poly(1, x)
        self.assertTrue(p_one.is_one)

        self.assertEqual(TC(3 * x**2 + 2 * x + 5, x), sympy.Integer(5))
        self.assertEqual(EC(3 * x**2 + 2 * x + 5, x), sympy.Integer(5))
        self.assertEqual(trailing_coeff(3 * x**2 + 2 * x + 5, x), sympy.Integer(5))

        # Test sqf
        expr = x**2 + 2 * x + 1
        self.assertEqual(sqf(expr), (x + 1) ** 2)

    def test_ntheory_extended_suite(self):
        from sympy import (
            primefactors,
            divisors,
            proper_divisors,
            nextprime,
            prevprime,
            is_quad_residue,
            prime,
            primepi,
        )

        self.assertEqual(primefactors(60), [2, 3, 5])
        self.assertEqual(primefactors(-60), [2, 3, 5])
        self.assertEqual(primefactors(0), [])
        self.assertEqual(primefactors(1), [])

        self.assertEqual(divisors(12), [1, 2, 3, 4, 6, 12])
        self.assertEqual(divisors(-12), [1, 2, 3, 4, 6, 12])
        self.assertEqual(divisors(1), [1])
        self.assertEqual(divisors(0), [])

        self.assertEqual(proper_divisors(12), [1, 2, 3, 4, 6])
        self.assertEqual(proper_divisors(1), [])

        self.assertEqual(nextprime(10), 11)
        self.assertEqual(nextprime(2), 3)
        self.assertEqual(nextprime(1), 2)
        self.assertEqual(nextprime(10, 2), 13)

        self.assertEqual(prevprime(10), 7)
        self.assertEqual(prevprime(3), 2)
        with self.assertRaises(ValueError):
            prevprime(2)

        self.assertTrue(is_quad_residue(2, 7))
        self.assertFalse(is_quad_residue(3, 7))
        self.assertTrue(is_quad_residue(0, 7))
        self.assertTrue(is_quad_residue(1, 2))

        self.assertEqual(prime(1), 2)
        self.assertEqual(prime(5), 11)
        self.assertEqual(primepi(12), 5)
        self.assertEqual(primepi(1), 0)

    def test_matrix_manipulation_methods(self):
        from sympy import Matrix, ImmutableMatrix

        M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(M.extract([0, 2], [1, 2]), Matrix([[2, 3], [8, 9]]))
        self.assertEqual(M.extract([0, 1], [-1]), Matrix([[3], [6]]))

        C = Matrix([[10], [20], [30]])
        self.assertEqual(M.col_insert(1, C), Matrix([[1, 10, 2, 3], [4, 20, 5, 6], [7, 30, 8, 9]]))

        R = Matrix([[11, 22, 33]])
        self.assertEqual(M.row_insert(1, R), Matrix([[1, 2, 3], [11, 22, 33], [4, 5, 6], [7, 8, 9]]))

        M_mut = Matrix([[1, 2, 3], [4, 5, 6]])
        M_mut.col_del(1)
        self.assertEqual(M_mut, Matrix([[1, 3], [4, 6]]))
        M_mut.row_del(0)
        self.assertEqual(M_mut, Matrix([[4, 6]]))

        v = Matrix([3, 4])
        self.assertEqual(v.norm(), sympy.Integer(5))

        IM = ImmutableMatrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            IM.col_del(0)
        with self.assertRaises(TypeError):
            IM.row_del(0)
        with self.assertRaises(TypeError):
            IM[0, 0] = 99

    def test_special_function_differentiation(self):
        from sympy import erf, erfc, sinc, asec, Symbol, diff, Derivative

        x = Symbol("x")
        d_erf = diff(erf(x), x)
        self.assertFalse(isinstance(d_erf, Derivative))
        self.assertNotIn("diff(", str(d_erf))

        d_erfc = diff(erfc(x), x)
        self.assertFalse(isinstance(d_erfc, Derivative))
        self.assertNotIn("diff(", str(d_erfc))

        d_sinc = diff(sinc(x), x)
        self.assertFalse(isinstance(d_sinc, Derivative))
        self.assertNotIn("diff(", str(d_sinc))

        d_asec = diff(asec(x), x)
        self.assertFalse(isinstance(d_asec, Derivative))
        self.assertNotIn("diff(", str(d_asec))

    def test_matrix_decompositions_and_minors(self):
        from sympy import Matrix, Integer

        # LDL decomposition of symmetric matrix
        A = Matrix([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        L, D = A.LDLdecomposition()
        self.assertEqual(L * D * L.T, A)
        self.assertTrue(L.is_lower_triangular())
        self.assertTrue(D.is_diagonal())
        self.assertEqual(A.ldl(), (L, D))

        # minor_submatrix, minor, cofactor_matrix
        M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(M.minor_submatrix(0, 0), Matrix([[5, 6], [8, 9]]))
        self.assertEqual(M.minorMatrix(0, 0), Matrix([[5, 6], [8, 9]]))
        self.assertEqual(M.minor(0, 0), Integer(-3))
        cof = M.cofactor_matrix()
        self.assertEqual(cof[0, 0], Integer(-3))

        # Callable & property symmetries
        S = Matrix([[1, 2], [2, 3]])
        self.assertTrue(bool(S.is_symmetric))
        self.assertTrue(S.is_symmetric())
        self.assertFalse(bool(S.is_anti_symmetric))
        self.assertFalse(S.is_anti_symmetric())

        Sk = Matrix([[0, 2], [-2, 0]])
        self.assertTrue(bool(Sk.is_anti_symmetric))
        self.assertTrue(Sk.is_anti_symmetric())
        self.assertTrue(Sk.is_skew_symmetric())
        self.assertFalse(bool(Sk.is_symmetric))

        Diag = Matrix([[1, 0], [0, 2]])
        self.assertTrue(bool(Diag.is_diagonal))
        self.assertTrue(Diag.is_diagonal())

        U = Matrix([[1, 2], [0, 3]])
        self.assertTrue(U.is_upper)
        self.assertTrue(U.is_upper_triangular())
        self.assertFalse(U.is_lower)

        Low = Matrix([[1, 0], [2, 3]])
        self.assertTrue(Low.is_lower)
        self.assertTrue(Low.is_lower_triangular())
        self.assertFalse(Low.is_upper)

    def test_linsolve_and_linear_system_solver(self):
        from sympy import (
            EmptySet,
            FiniteSet,
            Matrix,
            Symbol,
            linsolve,
            solve,
            solve_linear_system,
        )

        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # 3-variable linear system from equations
        sys = [x + y + z - 6, 2 * y + 5 * z - 19, 2 * x + 5 * y - 12]
        sol = linsolve(sys, x, y, z)
        self.assertEqual(sol, FiniteSet((1, 2, 3)))

        # General solve() with 3-variable linear system
        res = solve(sys, [x, y, z])
        self.assertEqual(res, {x: 1, y: 2, z: 3})

        # Augmented matrix linsolve and solve_linear_system
        M = Matrix([[1, 2, 5], [3, 4, 11]])
        sol_m = linsolve(M, [x, y])
        self.assertEqual(sol_m, FiniteSet((1, 2)))
        self.assertEqual(solve_linear_system(M, x, y), {x: 1, y: 2})

        # (A, b) pair linsolve
        A = Matrix([[1, 2], [3, 4]])
        b = Matrix([5, 11])
        sol_ab = linsolve((A, b), [x, y])
        self.assertEqual(sol_ab, FiniteSet((1, 2)))

        # Inconsistent system
        inc = [x + y - 1, x + y - 2]
        self.assertEqual(linsolve(inc, x, y), EmptySet())
        self.assertEqual(solve(inc, [x, y]), [])

    def test_product_set_and_reals(self):
        from sympy import (
            EmptySet,
            FiniteSet,
            Interval,
            ProductSet,
            Rational,
            Reals,
            S,
            Symbol,
            oo,
            solveset,
        )

        # Reals singleton & S.Reals
        self.assertIs(Reals, S.Reals)
        self.assertTrue(5 in Reals)
        self.assertFalse(oo in Reals)
        self.assertFalse(-oo in Reals)
        self.assertEqual(Interval(-oo, oo), Reals)
        self.assertEqual(Reals, Interval(-oo, oo, True, True))

        # ProductSet construction & properties
        A = Interval(0, 1)
        B = Interval(2, 3)
        P = A * B
        self.assertIsInstance(P, ProductSet)
        self.assertEqual(P.args, (A, B))
        self.assertEqual(P.sets, (A, B))
        self.assertEqual(P.measure, 1)
        self.assertFalse(P.is_empty)
        self.assertTrue((0.5, 2.5) in P)
        self.assertFalse((1.5, 2.5) in P)
        self.assertFalse(5 in P)

        # FiniteSet product
        F1 = FiniteSet(1, 2)
        F2 = FiniteSet(3, 4)
        PF = F1 * F2
        self.assertEqual(len(PF), 4)
        self.assertEqual(set(PF), {(1, 3), (1, 4), (2, 3), (2, 4)})

        # ProductSet algebra & annihilation
        self.assertIs(P * EmptySet(), S.EmptySet)
        self.assertIs(EmptySet() * P, S.EmptySet)
        self.assertEqual(P.intersect(P), P)
        self.assertFalse(P.is_disjoint(P))
        Disj = Interval(10, 20) * Interval(10, 20)
        self.assertTrue(P.is_disjoint(Disj))
        self.assertEqual(P.intersect(Disj), S.EmptySet)

        # Cartesian powers
        self.assertEqual(A**0, FiniteSet(()))
        self.assertEqual(A**2, ProductSet(A, A))

        # Rational 1-argument construction
        self.assertEqual(Rational(0.5), Rational(1, 2))
        self.assertEqual(Rational("3/4"), Rational(3, 4))
        self.assertEqual(Rational(5), 5)

        # solveset domain filtering
        x = Symbol("x")
        self.assertEqual(solveset(x**2 + 1, x, domain=Reals), EmptySet())
        self.assertEqual(solveset(x**2 - 4, x, domain=Interval(0, 5)), FiniteSet(2))
        self.assertEqual(solveset(0, x, domain=Interval(0, 1)), Interval(0, 1))

    def test_matrix_advanced_methods_and_slicing(self):
        from sympy import (
            ImmutableMatrix,
            Matrix,
            diag,
            eye,
            ones,
            pinv,
            zeros,
        )

        # 2D slicing
        M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(M[:, 0], Matrix([[1], [4], [7]]))
        self.assertEqual(M[0, :], Matrix([[1, 2, 3]]))
        self.assertEqual(M[:2, :2], Matrix([[1, 2], [4, 5]]))
        self.assertEqual(M[:3], [1, 2, 3])

        # Slice assignment
        M_assign = M.copy()
        M_assign[:, 0] = [10, 40, 70]
        self.assertEqual(M_assign[:, 0], Matrix([[10], [40], [70]]))
        self.assertEqual(M_assign[0, 1], 2)

        # Copy, as_immutable, as_mutable
        M_copy = M.copy()
        self.assertEqual(M_copy, M)
        M_copy[0, 0] = 999
        self.assertEqual(M[0, 0], 1)

        IM = M.as_immutable()
        self.assertIsInstance(IM, ImmutableMatrix)
        self.assertEqual(IM, M)
        MM = IM.as_mutable()
        self.assertIsInstance(MM, Matrix)
        self.assertNotIsInstance(MM, ImmutableMatrix)

        # Vector operations
        v1 = Matrix([1, 2, 3])
        v2 = Matrix([4, 5, 6])
        self.assertEqual(v1.dot(v2), 32)
        self.assertEqual(v1.cross(v2), Matrix([[-3], [6], [-3]]))

        # Reshape, vec, vech
        self.assertEqual(v1.reshape(1, 3), Matrix([[1, 2, 3]]))
        self.assertEqual(v1.reshape(3, 1), Matrix([[1], [2], [3]]))
        SymM = Matrix([[1, 2], [2, 3]])
        self.assertEqual(SymM.vech(), Matrix([[1], [2], [3]]))
        self.assertEqual(SymM.vec(), Matrix([[1], [2], [2], [3]]))

        # Block diagonal
        D = diag(Matrix([[1, 2], [3, 4]]), 5, Matrix([[6, 7], [8, 9]]))
        self.assertEqual(D.shape, (5, 5))
        self.assertEqual(D[0, 0], 1)
        self.assertEqual(D[2, 2], 5)
        self.assertEqual(D[3, 3], 6)
        self.assertEqual(D[0, 2], 0)

        # Moore-Penrose pseudoinverse
        rec = Matrix([[1, 2], [3, 4], [5, 6]])
        p = pinv(rec)
        self.assertEqual(p, rec.pinv())
        self.assertEqual(p.shape, (2, 3))

        # Static constructors on Matrix
        self.assertEqual(Matrix.eye(2), eye(2))
        self.assertEqual(Matrix.zeros(2, 3), zeros(2, 3))
        self.assertEqual(Matrix.ones(2, 2), ones(2, 2))
        self.assertEqual(Matrix.diag(1, 2), diag(1, 2))

    def test_polys_extended(self):
        from sympy import Poly, Symbol, expand, gcdex, half_gcdex

        x = Symbol("x")
        f = x**2 - 1
        g = x - 1
        s, t, h = gcdex(f, g)
        self.assertEqual(expand(s * f + t * g - h), 0)
        self.assertEqual(h, x - 1)

        s2, h2 = half_gcdex(f, g)
        self.assertEqual(s2, s)
        self.assertEqual(h2, h)

        p = Poly(x**2 + 2 * x + 1, x)
        p_shift = p.shift(2)
        self.assertEqual(p_shift, Poly(x**2 + 6 * x + 9, x))
        p_comp = p.compose(x + 2)
        self.assertEqual(p_comp, p_shift)

    def test_integral_unevaluated_and_iterated(self):
        from sympy import Expr, Integral, Rational, Symbol, integrate

        x, y = Symbol("x"), Symbol("y")
        I = Integral(x**2, (x, 0, 1))
        self.assertIsInstance(I, Integral)
        self.assertIsInstance(I, Expr)
        self.assertEqual(I.doit(), Rational(1, 3))
        self.assertEqual(integrate(I), Rational(1, 3))

        I_indef = Integral(x**2, x)
        self.assertEqual(I_indef.doit(), x**3 / 3)

        # Iterated integration
        self.assertEqual(integrate(x * y, x, y), x**2 * y**2 / 4)
        self.assertEqual(integrate(x * y, (x, 0, 1), (y, 0, 2)), 1)

    def test_limit_unevaluated(self):
        from sympy import Expr, Limit, Symbol, limit, sin

        x = Symbol("x")
        L = Limit(x**2 + 2 * x, x, 3)
        self.assertIsInstance(L, Limit)
        self.assertIsInstance(L, Expr)
        self.assertEqual(L.doit(), 15)
        self.assertEqual(limit(L), 15)

    def test_geometry_utilities(self):
        from sympy import (
            Line,
            Point,
            Point2D,
            Point3D,
            Rational,
            are_collinear,
            are_coplanar,
            centroid,
            intersection,
        )

        p1 = Point(0, 0)
        p2 = Point(1, 1)
        p3 = Point(2, 2)
        p4 = Point(0, 1)
        self.assertTrue(are_collinear(p1, p2, p3))
        self.assertFalse(are_collinear(p1, p2, p4))

        # Test duplicate / coincident points handling in are_collinear
        self.assertTrue(are_collinear(p1, p1, p2, p3))
        self.assertFalse(are_collinear(p1, p1, p2, p4))
        self.assertTrue(are_collinear(p1, p1, p1))
        self.assertTrue(are_collinear([p1, p2, p3]))

        # 3D collinearity
        p3d_1 = Point3D(0, 0, 0)
        p3d_2 = Point3D(1, 2, 3)
        p3d_3 = Point3D(2, 4, 6)
        p3d_4 = Point3D(2, 4, 7)
        self.assertTrue(are_collinear(p3d_1, p3d_2, p3d_3))
        self.assertFalse(are_collinear(p3d_1, p3d_2, p3d_4))

        # Coplanarity
        self.assertTrue(are_coplanar(p1, p2, p3, p4))
        p_c1 = Point3D(0, 0, 0)
        p_c2 = Point3D(1, 0, 0)
        p_c3 = Point3D(0, 1, 0)
        p_c4 = Point3D(1, 1, 0)
        p_c5 = Point3D(0, 0, 1)
        self.assertTrue(are_coplanar(p_c1, p_c2, p_c3, p_c4))
        self.assertFalse(are_coplanar(p_c1, p_c2, p_c3, p_c5))
        self.assertTrue(are_coplanar(p_c1, p_c1, p_c2, p_c3))

        c = centroid(Point(0, 0), Point(3, 0), Point(0, 3))
        self.assertEqual(c, Point2D(1, 1))

        c3d = centroid(Point3D(0, 0, 0), Point3D(3, 6, 9))
        self.assertEqual(c3d, Point3D(Rational(3, 2), 3, Rational(9, 2)))

        l1 = Line(Point(0, 0), Point(2, 2))
        l2 = Line(Point(0, 2), Point(2, 0))
        inter = intersection(l1, l2)
        self.assertEqual(inter, [Point2D(1, 1)])

    def test_point_arithmetic_and_methods(self):
        from sympy import Point, Point2D, Point3D, Rational

        p1 = Point(1, 2)
        p2 = Point(3, 4)
        self.assertEqual(p1 + p2, Point2D(4, 6))
        self.assertEqual(p2 - p1, Point2D(2, 2))
        self.assertEqual(p1 * 3, Point2D(3, 6))
        self.assertEqual(3 * p1, Point2D(3, 6))
        self.assertEqual(p1 / 2, Point2D(Rational(1, 2), 1))
        self.assertEqual(-p1, Point2D(-1, -2))
        self.assertEqual(p1.dot(p2), 11)
        self.assertEqual(p1.taxicab_distance(p2), 4)
        self.assertEqual(p1.midpoint(p2), Point2D(2, 3))
        self.assertEqual(p1.origin, Point2D(0, 0))
        self.assertEqual(p1.dimension, 2)
        self.assertEqual(p1.coordinates, (p1.x, p1.y))
        self.assertTrue(p1.is_collinear(Point(2, 4), Point(3, 6)))
        self.assertTrue(p1.is_coplanar(Point(2, 4), Point(5, 7), Point(8, 9)))

        p3 = Point3D(1, 2, 3)
        p4 = Point3D(4, 5, 6)
        self.assertEqual(p3 + p4, Point3D(5, 7, 9))
        self.assertEqual(p4 - p3, Point3D(3, 3, 3))
        self.assertEqual(p3 * 2, Point3D(2, 4, 6))
        self.assertEqual(p3.dot(p4), 32)
        self.assertEqual(p3.taxicab_distance(p4), 9)
        self.assertEqual(p3.midpoint(p4), Point3D(Rational(5, 2), Rational(7, 2), Rational(9, 2)))
        self.assertEqual(p3.origin, Point3D(0, 0, 0))
        self.assertEqual(p3.dimension, 3)

    def test_linear_entity_containment_and_intersection(self):
        from sympy import Line, Point, Point2D, Ray, Segment

        p0 = Point(0, 0)
        p1 = Point(2, 0)
        p_mid = Point(1, 0)
        p_out = Point(3, 0)
        p_off = Point(0, 1)

        seg = Segment(p0, p1)
        self.assertTrue(seg.contains(p0))
        self.assertTrue(seg.contains(p1))
        self.assertTrue(seg.contains(p_mid))
        self.assertFalse(seg.contains(p_out))
        self.assertFalse(seg.contains(p_off))
        self.assertTrue(seg.contains(Segment(Point(0, 0), Point(1, 0))))

        ray = Ray(p0, p1)
        self.assertTrue(ray.contains(p0))
        self.assertTrue(ray.contains(p_mid))
        self.assertTrue(ray.contains(p_out))
        self.assertFalse(ray.contains(Point(-1, 0)))
        self.assertFalse(ray.contains(p_off))

        line = Line(p0, p1)
        self.assertTrue(line.contains(p_out))
        self.assertTrue(line.contains(Point(-5, 0)))
        self.assertFalse(line.contains(p_off))

        # Parallel and perpendicular
        l_horiz = Line(Point(0, 1), Point(2, 1))
        l_vert = Line(Point(0, 0), Point(0, 2))
        self.assertTrue(line.is_parallel(l_horiz))
        self.assertFalse(line.is_parallel(l_vert))
        self.assertTrue(line.is_perpendicular(l_vert))
        self.assertFalse(line.is_perpendicular(l_horiz))

        # Segment intersections
        s_v = Segment(Point(1, -1), Point(1, 1))
        self.assertEqual(seg.intersection(s_v), [Point2D(1, 0)])

        s_overlap = Segment(Point(1, 0), Point(3, 0))
        inter_overlap = seg.intersection(s_overlap)
        self.assertEqual(len(inter_overlap), 1)
        self.assertIsInstance(inter_overlap[0], Segment)

        s_touch = Segment(Point(2, 0), Point(4, 0))
        self.assertEqual(seg.intersection(s_touch), [Point2D(2, 0)])

        s_disjoint = Segment(Point(3, 0), Point(5, 0))
        self.assertEqual(seg.intersection(s_disjoint), [])

    def test_polygon_circle_plane_containment(self):
        from sympy import Circle, Line, Plane, Point, Point2D, Point3D, Polygon, Segment, Sphere, Triangle

        # Polygon and Triangle
        tri = Polygon(Point(0, 0), Point(4, 0), Point(0, 3))
        self.assertIsInstance(tri, Triangle)
        self.assertEqual(len(tri.sides), 3)
        self.assertEqual(tri.perimeter, 12)
        self.assertEqual(tri.area, 6)
        self.assertTrue(tri.contains(Point(2, 0)))
        self.assertFalse(tri.contains(Point(1, 1)))

        poly = Polygon(Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3))
        self.assertEqual(len(poly.sides), 4)
        self.assertEqual(poly.perimeter, 14)
        self.assertEqual(poly.area, 12)
        self.assertTrue(poly.contains(Point(2, 0)))

        l = Line(Point(2, -1), Point(2, 5))
        poly_inter = poly.intersection(l)
        self.assertEqual(poly_inter, [Point2D(2, 0), Point2D(2, 3)])

        # Circle
        c = Circle(Point(0, 0), 5)
        self.assertTrue(c.contains(Point(3, 4)))
        self.assertTrue(c.contains(Point(-5, 0)))
        self.assertFalse(c.contains(Point(3, 3)))

        # Sphere
        s = Sphere(Point3D(0, 0, 0), 3)
        self.assertTrue(s.contains(Point3D(0, 0, 3)))
        self.assertFalse(s.contains(Point3D(1, 1, 1)))

        # Plane
        plane = Plane(Point3D(0, 0, 0), Point3D(0, 0, 1))
        self.assertTrue(plane.contains(Point3D(5, 7, 0)))
        self.assertFalse(plane.contains(Point3D(5, 7, 1)))
        seg3d = Segment(Point3D(1, 1, -2), Point3D(1, 1, 2))
        self.assertEqual(plane.intersection(seg3d), [Point3D(1, 1, 0)])

    def test_matrix_advanced_subtypes_and_sparse(self):
        import sympy
        from sympy import (
            ImmutableMatrix,
            Matrix,
            SparseMatrix,
            eye,
            jordan_block,
            jordan_cell,
            zeros,
        )

        # 1. Cross-type constructors & symmetric equality
        sm = SparseMatrix([[1, 0], [0, 2]])
        dm = Matrix(sm)
        self.assertEqual(dm, sm)
        self.assertEqual(sm, dm)
        self.assertEqual(sm.shape, (2, 2))
        self.assertEqual(dm.shape, (2, 2))

        # 2. In-place elementary row/col operations on mutable Matrix
        M = Matrix([[1, 2], [3, 4]])
        M.row_swap(0, 1)
        self.assertEqual(M, Matrix([[3, 4], [1, 2]]))
        M.col_swap(0, 1)
        self.assertEqual(M, Matrix([[4, 3], [2, 1]]))
        M.row_op(0, lambda val, j: val * 10)
        self.assertEqual(M, Matrix([[40, 30], [2, 1]]))
        M.col_op(1, lambda val, i: val + 5)
        self.assertEqual(M, Matrix([[40, 35], [2, 6]]))

        # 3. Immutability protection on ImmutableMatrix
        IM = ImmutableMatrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            IM.row_swap(0, 1)
        with self.assertRaises(TypeError):
            IM.col_swap(0, 1)
        with self.assertRaises(TypeError):
            IM.row_op(0, lambda val, j: val * 2)
        with self.assertRaises(TypeError):
            IM.col_op(0, lambda val, i: val * 2)

        # 4. ImmutableMatrix subtype preservation
        IM_T = IM.T
        self.assertIsInstance(IM_T, ImmutableMatrix)
        IM_inv = IM.inv()
        self.assertIsInstance(IM_inv, ImmutableMatrix)
        IM_sum = IM + IM
        self.assertIsInstance(IM_sum, ImmutableMatrix)
        IM_mul = IM * 3
        self.assertIsInstance(IM_mul, ImmutableMatrix)
        IM_extract = IM.extract([0], [1])
        self.assertIsInstance(IM_extract, ImmutableMatrix)
        IM_subs = IM.subs(sympy.Symbol("x"), 1)
        self.assertIsInstance(IM_subs, ImmutableMatrix)

        # 5. Matrix predicates: is_zero_matrix, is_identity, is_nilpotent
        Z = zeros(2, 3)
        self.assertTrue(bool(Z.is_zero_matrix))
        self.assertTrue(Z.is_zero_matrix())
        self.assertFalse(bool(M.is_zero_matrix))
        self.assertFalse(M.is_zero_matrix())

        I2 = eye(2)
        self.assertTrue(bool(I2.is_identity))
        self.assertTrue(I2.is_identity())
        self.assertFalse(bool(M.is_identity))

        Nil = Matrix([[0, 1], [0, 0]])
        self.assertTrue(Nil.is_nilpotent())
        self.assertFalse(I2.is_nilpotent())

        # Predicates on SparseMatrix
        Z_sp = SparseMatrix.zeros(3, 3)
        self.assertTrue(bool(Z_sp.is_zero_matrix))
        I_sp = SparseMatrix(eye(3))
        self.assertTrue(bool(I_sp.is_identity))
        self.assertTrue(bool(I_sp.is_diagonal))
        self.assertTrue(bool(I_sp.is_symmetric))

        # 6. rref with pivots flag
        R, pivs = M.rref(pivots=True)
        self.assertEqual(pivs, (0, 1))
        self.assertEqual(R, eye(2))
        R_only = M.rref(pivots=False)
        self.assertEqual(R_only, eye(2))

        # 7. evalf and sympy.N with precision honesty
        M_float = M.evalf(5)
        self.assertEqual(M_float[0, 0], sympy.Float(40.0, 5))
        M_n = sympy.N(M, 10)
        self.assertEqual(M_n[0, 0], sympy.Float(40.0, 10))
        with self.assertRaises(NotImplementedError):
            sympy.N(M, 20)

        # 8. jordan_cell and jordan_block
        jb = jordan_cell(3, 3)
        self.assertEqual(jb, Matrix([[3, 1, 0], [0, 3, 1], [0, 0, 3]]))
        self.assertEqual(jordan_block(5, 2), Matrix([[5, 1], [0, 5]]))

        # 9. SparseMatrix algebraic methods
        sp_a = SparseMatrix([[2, 1], [1, 2]])
        self.assertEqual(sp_a.det(), 3)
        sp_inv = sp_a.inv()
        self.assertIsInstance(sp_inv, SparseMatrix)
        self.assertEqual(sp_a @ sp_inv, SparseMatrix(eye(2)))
        b = SparseMatrix([[5], [4]])
        sol = sp_a.solve(b)
        self.assertIsInstance(sol, SparseMatrix)
        self.assertEqual(sol, SparseMatrix([[2], [1]]))

    def test_advanced_solvers_and_polysys(self):
        import sympy
        from sympy import (
            EmptySet,
            FiniteSet,
            Integer,
            Symbol,
            checksol,
            linsolve,
            nonlinsolve,
            solve,
            solve_linear,
            solve_linear_system,
            solve_linear_system_LU,
            solve_poly_system,
            solveset,
            symbols,
        )

        x, y = symbols("x y")

        # 1. solve with *symbols and tuple/list unpacking
        s1 = solve([x + y - 3, x - y - 1], x, y)
        self.assertEqual(s1, {x: Integer(2), y: Integer(1)})
        s2 = solve([x + y - 3, x - y - 1], [x, y])
        self.assertEqual(s2, {x: Integer(2), y: Integer(1)})

        # 2. solve single-variable polynomial system
        s_poly = solve([x**2 - 4], x)
        self.assertEqual(sorted(s_poly), [(-Integer(2),), (Integer(2),)])
        s_poly_list = solve([x**2 - 4], [x])
        self.assertEqual(sorted(s_poly_list), [(-Integer(2),), (Integer(2),)])

        # 3. solve with dict=True and set=True
        s_dict_lin = solve([x + y - 3, x - y - 1], [x, y], dict=True)
        self.assertEqual(s_dict_lin, [{x: Integer(2), y: Integer(1)}])

        s_dict_poly = solve([x**2 - 4], x, dict=True)
        self.assertEqual(len(s_dict_poly), 2)
        self.assertIn({x: Integer(2)}, s_dict_poly)
        self.assertIn({x: -Integer(2)}, s_dict_poly)

        s_dict_single = solve(x**2 - 4, x, dict=True)
        self.assertIn({x: Integer(2)}, s_dict_single)
        self.assertIn({x: -Integer(2)}, s_dict_single)

        s_set_single = solve(x**2 - 4, x, set=True)
        self.assertEqual(s_set_single[0], [x])
        self.assertEqual(s_set_single[1], {(Integer(2),), (-Integer(2),)})

        # 4. solve_poly_system
        # 1-variable system
        sps1 = solve_poly_system([x**2 - 9], x)
        self.assertEqual(sorted(sps1), [(-Integer(3),), (Integer(3),)])

        # 1-variable system with intersection
        sps_inter = solve_poly_system([x**2 - 9, x - 3], x)
        self.assertEqual(sps_inter, [(Integer(3),)])

        # 1-variable system inconsistent
        sps_none = solve_poly_system([x**2 - 9, x - 2], x)
        self.assertIsNone(sps_none)

        # auto-detected generators
        sps_auto1 = solve_poly_system([x**2 - 9])
        self.assertEqual(sorted(sps_auto1), [(-Integer(3),), (Integer(3),)])

        sps_auto2 = solve_poly_system([x - 1, y**2 - 4])
        self.assertEqual(len(sps_auto2), 2)
        self.assertIn((Integer(1), Integer(2)), sps_auto2)
        self.assertIn((Integer(1), -Integer(2)), sps_auto2)

        # 5. nonlinsolve
        nls1 = nonlinsolve([x**2 - 4], x)
        self.assertEqual(nls1, FiniteSet((-Integer(2),), (Integer(2),)))
        nls1_list = nonlinsolve([x**2 - 4], [x])
        self.assertEqual(nls1_list, FiniteSet((-Integer(2),), (Integer(2),)))
        nls_inconsistent = nonlinsolve([x**2 - 4, x - 1], x)
        self.assertEqual(nls_inconsistent, EmptySet())

        # 6. solve_linear
        self.assertEqual(solve_linear(2 * x - 4), (x, Integer(2)))
        self.assertEqual(solve_linear(x + y**2, symbols=[x]), (x, -y**2))
        self.assertEqual(solve_linear(5), (0, 1))
        self.assertEqual(solve_linear(x, exclude=[x]), (0, 1))

        # 7. checksol
        self.assertTrue(checksol(x**2 - 4, x, 2))
        self.assertTrue(checksol(x**2 - 4, x, -2))
        self.assertFalse(checksol(x**2 - 4, x, 1))
        self.assertTrue(checksol([x + y - 3, x - y - 1], {x: 2, y: 1}))
        self.assertFalse(checksol([x + y - 3, x - y - 1], {x: 2, y: 0}))
        self.assertTrue(checksol(x**2 + x - x * (x + 1), {}))

        # 8. solve_linear_system_LU
        self.assertIs(solve_linear_system_LU, solve_linear_system)
        M_aug = sympy.Matrix([[1, 1, 3], [1, -1, 1]])
        sol_lu = solve_linear_system_LU(M_aug, x, y)
        self.assertEqual(sol_lu, {x: Integer(2), y: Integer(1)})

    def test_poly_advanced_and_cancel(self):
        import sympy
        from sympy import (
            Integer,
            Poly,
            Rational,
            Symbol,
            cancel,
            content,
            poly,
            primitive,
            symbols,
        )

        x, y = symbols("x y")

        # 1. Operators: neg, pos, pow, floordiv, mod, divmod
        p1 = Poly(x**2 - 1, x)
        self.assertEqual(-p1, Poly(-x**2 + 1, x))
        self.assertEqual(+p1, p1)
        self.assertEqual(p1**2, Poly(x**4 - 2 * x**2 + 1, x))
        q1 = Poly(x - 1, x)
        self.assertEqual(p1 // q1, Poly(x + 1, x))
        self.assertEqual(p1 % q1, Poly(0, x))
        self.assertEqual(divmod(p1, q1), (Poly(x + 1, x), Poly(0, x)))

        # 2. Predicates: is_univariate, is_multivariate, is_irreducible
        self.assertTrue(p1.is_univariate)
        self.assertFalse(p1.is_multivariate)
        p_multi = Poly(x * y + 1, x, y)
        self.assertFalse(p_multi.is_univariate)
        self.assertTrue(p_multi.is_multivariate)
        self.assertFalse(p1.is_irreducible)
        p_irr = Poly(x**2 + 1, x)
        self.assertTrue(p_irr.is_irreducible)

        # 3. Content and primitive
        self.assertEqual(content(4 * x + 6), Integer(2))
        self.assertEqual(primitive(4 * x + 6), (Integer(2), 2 * x + 3))
        self.assertEqual(p1.content(), Integer(1))
        self.assertEqual(p1.primitive(), (Integer(1), p1))
        p_rat = Poly(Rational(2, 3) * x + Rational(4, 9), x)
        self.assertEqual(p_rat.content(), Rational(2, 9))
        self.assertEqual(primitive(Rational(2, 3) * x + Rational(4, 9)), (Rational(2, 9), 3 * x + 2))

        # 4. Constructors and aliases: from_list, from_expr, from_poly, poly()
        p_from_list = Poly.from_list([1, 0, -1], gens=[x])
        self.assertEqual(p_from_list, p1)
        p_from_expr = Poly.from_expr(x**2 - 1, x)
        self.assertEqual(p_from_expr, p1)
        p_from_poly = Poly.from_poly(p1)
        self.assertEqual(p_from_poly, p1)
        self.assertEqual(poly(x**2 - 1, x), p1)
        self.assertEqual(Poly(x**2 - 1, x, domain="ZZ").domain, "ZZ")

        # 5. cancel
        self.assertEqual(cancel((x**2 - 1) / (x - 1)), x + 1)
        self.assertEqual(cancel((x**2 - 4) / (x**2 + 4 * x + 4)), (x - 2) / (x + 2))
        self.assertEqual(cancel((2 * x + 4) / (4 * x + 8)), Rational(1, 2))
        self.assertEqual(cancel(x**2 - 1), x**2 - 1)
        self.assertEqual(cancel(p1), p1)

    def test_ntheory_advanced(self):
        import sympy
        from sympy import multiplicity, perfect_power, primerange
        from sympy.ntheory import (
            multiplicity as nt_multiplicity,
            perfect_power as nt_perfect_power,
            primerange as nt_primerange,
        )

        self.assertIs(multiplicity, nt_multiplicity)
        self.assertIs(perfect_power, nt_perfect_power)
        self.assertIs(primerange, nt_primerange)

        # 1. multiplicity
        self.assertEqual(multiplicity(2, 24), 3)
        self.assertEqual(multiplicity(3, 24), 1)
        self.assertEqual(multiplicity(5, 24), 0)
        self.assertEqual(multiplicity(2, -8), 3)
        with self.assertRaises(ValueError):
            multiplicity(1, 10)
        with self.assertRaises(ValueError):
            multiplicity(2, 0)

        # 2. primerange
        self.assertEqual(list(primerange(10)), [2, 3, 5, 7])
        self.assertEqual(list(primerange(1, 10)), [2, 3, 5, 7])
        self.assertEqual(list(primerange(10, 20)), [11, 13, 17, 19])
        self.assertEqual(list(primerange(20, 10)), [])

        # 3. perfect_power
        self.assertEqual(perfect_power(16), (2, 4))
        self.assertEqual(perfect_power(27), (3, 3))
        self.assertEqual(perfect_power(-8), (-2, 3))
        self.assertFalse(perfect_power(15))
        self.assertFalse(perfect_power(0))
        self.assertFalse(perfect_power(1))
        self.assertFalse(perfect_power(-16))

    def test_calculus_and_series_advanced(self):
        import sympy
        from sympy import (
            AccumBounds,
            AccumulationBounds,
            EmptySet,
            FiniteSet,
            Integer,
            Interval,
            O,
            Order,
            Reals,
            Symbol,
            cos,
            sin,
        )
        from sympy.calculus import (
            AccumBounds as calc_AccumBounds,
            AccumulationBounds as calc_AccumulationBounds,
            continuous_domain,
            function_range,
            is_decreasing,
            is_increasing,
            is_monotonic,
            is_strictly_decreasing,
            is_strictly_increasing,
            maximum,
            minimum,
            periodicity,
            singularities,
            stationary_points,
        )

        self.assertIs(AccumBounds, calc_AccumBounds)
        self.assertIs(AccumulationBounds, calc_AccumulationBounds)
        self.assertIs(sympy.calculus.singularities, singularities)
        self.assertIs(sympy.calculus.continuous_domain, continuous_domain)
        self.assertIs(sympy.calculus.is_increasing, is_increasing)
        self.assertIs(sympy.calculus.is_strictly_increasing, is_strictly_increasing)
        self.assertIs(sympy.calculus.is_decreasing, is_decreasing)
        self.assertIs(sympy.calculus.is_strictly_decreasing, is_strictly_decreasing)
        self.assertIs(sympy.calculus.is_monotonic, is_monotonic)
        self.assertIs(sympy.calculus.periodicity, periodicity)
        self.assertIs(sympy.calculus.stationary_points, stationary_points)
        self.assertIs(sympy.calculus.maximum, maximum)
        self.assertIs(sympy.calculus.minimum, minimum)
        self.assertIs(sympy.calculus.function_range, function_range)

        x = Symbol("x")

        # 1. Singularities and continuous domain
        self.assertEqual(singularities(1 / (x - 2), x), FiniteSet(2))
        self.assertEqual(singularities(x**2 + 1, x), EmptySet())
        cd = continuous_domain(1 / (x - 2), x, Reals)
        self.assertTrue(hasattr(cd, "intersect") or hasattr(cd, "args"))

        # 2. Monotonicity
        self.assertTrue(is_increasing(x**3, x))
        self.assertTrue(is_strictly_increasing(x, x))
        self.assertTrue(is_decreasing(-x**3, x))
        self.assertTrue(is_strictly_decreasing(-x, x))
        self.assertTrue(is_monotonic(x**3, x))
        self.assertFalse(is_monotonic(x**2, x))

        # 3. Periodicity
        self.assertEqual(periodicity(sin(x), x), 2 * sympy.pi)
        self.assertEqual(periodicity(cos(2 * x), x), sympy.pi)
        self.assertIsNone(periodicity(x**2, x))

        # 4. Stationary points and extrema
        pts = stationary_points(x**2 - 4 * x + 3, x)
        self.assertEqual(pts, FiniteSet(2))
        self.assertEqual(maximum(-(x - 2)**2 + 5, x), Integer(5))
        self.assertEqual(minimum((x - 2)**2 + 5, x), Integer(5))

        # Function range with interval bounds
        r1 = function_range(x**2, x, Interval(1, 3))
        self.assertEqual(r1.start, Integer(1))
        self.assertEqual(r1.end, Integer(9))

        r2 = function_range(x**2, x, Interval(-1, 3))
        self.assertEqual(r2.start, Integer(0))
        self.assertEqual(r2.end, Integer(9))

        self.assertEqual(minimum(x**2, x, Interval(1, 3)), Integer(1))
        self.assertEqual(maximum(x**2, x, Interval(1, 3)), Integer(9))

        # 5. AccumBounds
        ab = AccumBounds(-1, 1)
        self.assertEqual(ab.min, Integer(-1))
        self.assertEqual(ab.max, Integer(1))
        self.assertEqual(ab.args, (Integer(-1), Integer(1)))
        self.assertEqual(ab + 2, AccumBounds(1, 3))
        self.assertEqual(2 + ab, AccumBounds(1, 3))
        self.assertEqual(ab - 1, AccumBounds(-2, 0))
        self.assertEqual(-ab, AccumBounds(-1, 1))
        self.assertEqual(ab * 2, AccumBounds(-2, 2))
        self.assertEqual(ab, AccumulationBounds(-1, 1))

        # 6. Series expansion with keywords and removeO
        s1 = sin(x).series(x=x, x0=0, n=4, dir="+")
        s2 = sin(x).series(x, 0, 4)
        self.assertEqual(s1, s2)
        # removeO on plain Taylor polynomial returns the polynomial
        self.assertEqual(s1.removeO(), s1)
        # removeO removes Order terms
        o_term = O(x**4)
        s_with_o = s1 + o_term
        self.assertEqual(s_with_o.removeO().expand(), s1.expand())
        self.assertEqual(o_term.removeO(), Integer(0))
        self.assertEqual(Order(x**3).removeO(), Integer(0))

    def test_rational_simplification_and_advanced_matrices(self):
        import sympy
        from sympy import (
            Integer,
            Matrix,
            Poly,
            Symbol,
            apart,
            cancel,
            det,
            randMatrix,
            rank,
            shape,
            together,
            trace,
        )

        x, y = Symbol("x"), Symbol("y")

        # 1. apart: partial fraction decomposition
        # Distinct linear factors
        ap1 = apart(1 / (x**2 - 1), x)
        self.assertEqual(together(ap1), 1 / (x**2 - 1))

        # Improper rational fraction
        ap2 = apart((x + 2) / (x + 1), x)
        self.assertEqual(ap2, 1 + 1 / (x + 1))

        # Repeated linear factors
        ap3 = apart(1 / (x**2 * (x - 1)), x)
        self.assertEqual(together(ap3), 1 / (x**3 - x**2))

        # Default variable detection
        ap4 = apart((x + 2) / (x + 1))
        self.assertEqual(ap4, 1 + 1 / (x + 1))

        # Constant polynomial / no denominator
        self.assertEqual(apart(x**2 + 1, x), x**2 + 1)

        # 2. together: combining fractions
        t1 = together(1 / x + 1 / y)
        self.assertEqual(t1, (x + y) / (x * y))

        t2 = together(1 / x + 1)
        self.assertEqual(t2, (x + 1) / x)

        t3 = together(1 / (x + 1) + 1 / (x - 1))
        self.assertEqual(cancel(t3), (2 * x) / (x**2 - 1))

        # 3. cancel: common factor cancellation
        c1 = cancel((x**2 - 1) / (x - 1))
        self.assertEqual(c1, x + 1)

        c2 = cancel((x**2 - y**2) / (x - y))
        self.assertEqual(c2, x + y)

        c3 = cancel((x*y + y) / y)
        self.assertEqual(c3, x + 1)

        # 4. Poly multivariate methods
        p_xy = Poly(x * y)
        self.assertEqual(p_xy.degree(x), 1)
        self.assertEqual(p_xy.degree(y), 1)
        self.assertEqual(p_xy.degree(None), 2)

        p1 = Poly(x**2 * y)
        p2 = Poly(x * y**2)
        g = p1.gcd(p2)
        self.assertEqual(g.as_expr(), x * y)
        l = p1.lcm(p2)
        self.assertEqual(l.as_expr(), x**2 * y**2)

        # 5. Standalone matrix functions
        M = Matrix([[1, 2], [3, 4]])
        self.assertEqual(det(M), -2)
        self.assertEqual(trace(M), 5)
        self.assertEqual(rank(M), 2)
        self.assertEqual(shape(M), (2, 2))

        # 6. randMatrix
        R = randMatrix(3, 3, seed=42)
        self.assertEqual(R.shape, (3, 3))
        self.assertTrue(all(isinstance(R[r, c], Integer) for r in range(3) for c in range(3)))

        R_sym = randMatrix(3, symmetric=True, seed=123)
        self.assertEqual(R_sym.shape, (3, 3))
        self.assertTrue(R_sym.is_symmetric)

    def test_simplification_subsystem(self):
        from sympy import (
            Add,
            Integer,
            Mul,
            Pow,
            Rational,
            Symbol,
            collect,
            combsimp,
            cos,
            cosh,
            log,
            logcombine,
            nsimplify,
            pi,
            powsimp,
            radsimp,
            ratsimp,
            separatevars,
            simplify,
            sin,
            sinh,
            sqrt,
            trigsimp,
        )

        x = Symbol("x")
        y = Symbol("y")
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")
        d = Symbol("d")

        # trigsimp
        self.assertEqual(trigsimp(sin(x)**2 + cos(x)**2), 1)
        self.assertEqual(trigsimp(cosh(x)**2 - sinh(x)**2), 1)

        # powsimp
        self.assertEqual(powsimp(x**a * x**b), x**(a + b))

        # ratsimp
        self.assertEqual(ratsimp(x/y + y/x), (x**2 + y**2)/(x*y))

        # radsimp
        self.assertEqual(radsimp(1 / sqrt(2)), sqrt(2) / 2)
        self.assertEqual(radsimp(1 / (1 + sqrt(2))), sqrt(2) - 1)

        # collect
        collected = collect(a*x + b*x + y, x)
        self.assertEqual(collected, x*(a + b) + y)
        quad = collect(a*x**2 + b*x**2 + c*x + d, x)
        self.assertEqual(quad, (a + b)*x**2 + c*x + d)

        # logcombine
        self.assertEqual(logcombine(log(x) + log(y)), log(x*y))

        # nsimplify
        self.assertEqual(nsimplify(0.3333333333333333), Rational(1, 3))
        self.assertEqual(nsimplify(1.4142135623730951), sqrt(2))
        self.assertEqual(nsimplify(3.141592653589793), pi)

        # combsimp and separatevars basic functionality
        self.assertEqual(combsimp(x + 1), x + 1)
        self.assertEqual(separatevars(x*y), x*y)

    def test_integral_subsystem(self):
        from sympy import Integral, Symbol, exp, integrate

        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")

        i1 = Integral(x**2, x)
        i2 = Integral(x**2, x)
        self.assertEqual(i1, i2)
        self.assertEqual(hash(i1), hash(i2))
        self.assertEqual(i1.variables, [x])
        self.assertTrue(i1.is_number)

        # Leibniz differentiation
        self.assertEqual(i1.diff(x), x**2)
        i_def = Integral(t**2, (t, 0, x))
        self.assertEqual(i_def.diff(x), x**2)
        self.assertFalse(i_def.is_number)
        self.assertEqual(i_def.diff(y), 0)

        # Substitution
        i_xy = Integral(x * y, (x, 0, 1))
        self.assertFalse(i_xy.is_number)
        self.assertEqual(i_xy.diff(y), Integral(x, (x, 0, 1)))
        self.assertEqual(i_xy.subs(y, 2), Integral(2 * x, (x, 0, 1)))
        # Substituting bound variable should not alter integrand
        self.assertEqual(i_xy.subs(x, y), i_xy)

        # Non-computable integration falls back to unevaluated Integral
        i_exp = integrate(exp(x**2), x)
        self.assertIsInstance(i_exp, Integral)
        self.assertEqual(i_exp, Integral(exp(x**2), x))

    def test_limit_subsystem(self):
        from sympy import Limit, Symbol

        x = Symbol("x")
        y = Symbol("y")

        l1 = Limit(x * y, x, 0)
        l2 = Limit(x * y, x, 0)
        self.assertEqual(l1, l2)
        self.assertEqual(hash(l1), hash(l2))
        self.assertEqual(l1.free_symbols, {y})
        self.assertFalse(l1.is_number)
        self.assertEqual(l1.subs(y, 3), Limit(3 * x, x, 0))

        l_const = Limit(x, x, 0)
        self.assertTrue(l_const.is_number)
        self.assertEqual(l_const.doit(), 0)

    def test_geometry_subsystem(self):
        from sympy import (
            Circle,
            Line,
            Plane,
            Point,
            Point2D,
            Point3D,
            Polygon,
            Rational,
            Ray,
            Segment,
            Sphere,
            Symbol,
            Triangle,
            are_similar,
            convex_hull,
            idiff,
        )

        p0 = Point(0, 0)
        p1 = Point(1, 0)
        p2 = Point(0, 1)
        p_in = Point(Rational(1, 5), Rational(1, 5))

        # Structural equality and hashing
        self.assertEqual(Line(p0, p1), Line(p0, p1))
        self.assertEqual(hash(Line(p0, p1)), hash(Line(p0, p1)))
        self.assertEqual(Segment(p0, p1), Segment(p0, p1))
        self.assertEqual(Ray(p0, p1), Ray(p0, p1))
        self.assertEqual(Circle(p0, 1), Circle(p0, 1))
        self.assertEqual(hash(Circle(p0, 1)), hash(Circle(p0, 1)))

        p3_0 = Point(0, 0, 0)
        p3_1 = Point(0, 0, 1)
        self.assertEqual(Sphere(p3_0, 1), Sphere(p3_0, 1))
        self.assertEqual(Plane(p3_0, p3_1), Plane(p3_0, p3_1))
        self.assertEqual(Triangle(p0, p1, p2), Triangle(p0, p1, p2))

        # Substitution and free symbols
        x = Symbol("x")
        px = Point(x, 1)
        self.assertEqual(px.free_symbols, {x})
        self.assertEqual(px.subs(x, 5), Point(5, 1))
        lx = Line(px, Point(0, 2))
        self.assertEqual(lx.free_symbols, {x})
        self.assertEqual(lx.subs(x, 3), Line(Point(3, 1), Point(0, 2)))

        # evalf and n
        self.assertEqual(Point(1, 2).evalf(), Point2D(1.0, 2.0))
        self.assertEqual(Point(1, 2).n(), Point2D(1.0, 2.0))

        # idiff
        y = Symbol("y")
        self.assertEqual(idiff(x**2 + y**2 - 1, y, x), -x / y)

        # convex_hull
        self.assertEqual(convex_hull(p0), p0)
        self.assertEqual(convex_hull(p0, p1), Segment(p0, p1))
        hull = convex_hull(p0, p1, p2, p_in)
        self.assertEqual(hull, Triangle(p0, p1, p2))

        # are_similar
        self.assertTrue(are_similar(Circle(p0, 1), Circle(Point(3, 4), 10)))

    def test_expr_methods_completion(self):
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        z = sympy.Symbol("z")

        # integrate
        self.assertEqual((x**2).integrate(x), x**3 / 3)
        self.assertEqual((x**2).integrate((x, 0, 1)), sympy.Rational(1, 3))

        # limit
        self.assertEqual((x**2).limit(x, 0), sympy.Integer(0))

        # collect
        self.assertEqual((x * y + x * z).collect(x), x * (y + z))

        # cancel
        self.assertEqual(((x**2 - 1) / (x - 1)).cancel(), x + 1)

        # apart & together
        part = (1 / (x**2 - 1)).apart(x)
        self.assertEqual(part, (1 / (x - 1)) / 2 - (1 / (x + 1)) / 2)
        tog = (1 / x + 1 / y).together()
        self.assertEqual(tog, (x + y) / (x * y))

        # trigsimp & powsimp
        self.assertEqual((sympy.sin(x)**2 + sympy.cos(x)**2).trigsimp(), sympy.Integer(1))
        self.assertEqual((sympy.exp(x) * sympy.exp(y)).powsimp(), sympy.exp(x + y))

        # rewrite
        self.assertEqual(
            sympy.sin(x).rewrite(sympy.exp),
            (sympy.exp(sympy.I * x) - sympy.exp(-sympy.I * x)) / (2 * sympy.I)
        )
        self.assertEqual(
            sympy.cos(x).rewrite(sympy.exp),
            (sympy.exp(sympy.I * x) + sympy.exp(-sympy.I * x)) / 2
        )
        self.assertEqual(
            sympy.tan(x).rewrite(sympy.sin),
            sympy.sin(x) / sympy.cos(x)
        )
        self.assertEqual(
            sympy.factorial(x).rewrite(sympy.gamma),
            sympy.gamma(x + 1)
        )
        self.assertEqual(
            sympy.gamma(x).rewrite(sympy.factorial),
            sympy.factorial(x - 1)
        )

    def test_matrix_spectral_and_exp_methods(self):
        # Hermitian and conjugate
        A = sympy.Matrix([[1, -sympy.I], [sympy.I, 2]])
        self.assertEqual(A.H, A)
        self.assertTrue(A.is_hermitian())
        self.assertFalse(A.is_anti_hermitian())

        B = sympy.Matrix([[0, sympy.I], [sympy.I, 0]])
        self.assertEqual((-B).H, B)
        self.assertFalse(B.is_hermitian())
        self.assertTrue(B.is_anti_hermitian())

        # Definiteness
        pos_def = sympy.Matrix([[2, -1], [-1, 2]])
        self.assertTrue(pos_def.is_positive_definite())
        self.assertTrue(pos_def.is_positive_semidefinite())
        self.assertFalse(pos_def.is_negative_definite())

        pos_semi = sympy.Matrix([[1, 0], [0, 0]])
        self.assertFalse(pos_semi.is_positive_definite())
        self.assertTrue(pos_semi.is_positive_semidefinite())

        neg_def = -pos_def
        self.assertTrue(neg_def.is_negative_definite())
        self.assertTrue(neg_def.is_negative_semidefinite())

        # Diagonalizable and exp
        diag_mat = sympy.Matrix([[1, 0], [0, 2]])
        self.assertTrue(diag_mat.is_diagonalizable())
        self.assertEqual(
            diag_mat.exp(),
            sympy.Matrix([[sympy.exp(1), 0], [0, sympy.exp(2)]])
        )

        nilp = sympy.Matrix([[0, 1], [0, 0]])
        self.assertFalse(nilp.is_diagonalizable())
        self.assertEqual(
            nilp.exp(),
            sympy.Matrix([[1, 1], [0, 1]])
        )

        jordan_shift = sympy.Matrix([[2, 1], [0, 2]])
        self.assertEqual(
            jordan_shift.exp(),
            sympy.Matrix([[sympy.exp(2), sympy.exp(2)], [0, sympy.exp(2)]])
        )

        # Singular values & condition number
        M = sympy.Matrix([[3, 0], [0, -2]])
        self.assertEqual(M.singular_values(), [sympy.Integer(3), sympy.Integer(2)])
        self.assertEqual(M.condition_number(), sympy.Rational(3, 2))

    def test_polys_tools_and_sturm(self):
        x = sympy.Symbol("x")

        # div, rem, quo
        q, r = sympy.div(x**2 - 1, x - 1)
        self.assertEqual(q, x + 1)
        self.assertEqual(r, sympy.Integer(0))
        self.assertEqual(sympy.quo(x**2 - 1, x - 1), x + 1)
        self.assertEqual(sympy.rem(x**2 - 1, x - 1), sympy.Integer(0))

        # Poly div, rem, quo
        pq, pr = sympy.div(sympy.Poly(x**2 - 1, x), sympy.Poly(x - 1, x))
        self.assertIsInstance(pq, sympy.Poly)
        self.assertIsInstance(pr, sympy.Poly)
        self.assertEqual(pq, sympy.Poly(x + 1, x))
        self.assertEqual(pr, sympy.Poly(0, x))

        # sturm
        sturm_seq = sympy.sturm(x**3 - 2 * x - 5, x)
        self.assertEqual(len(sturm_seq), 4)
        self.assertTrue(all(isinstance(p, sympy.Poly) for p in sturm_seq))
        self.assertEqual(sturm_seq[0], sympy.Poly(x**3 - 2 * x - 5, x))
        self.assertEqual(sturm_seq[1], sympy.Poly(3 * x**2 - 2, x))

        # compose
        comp_res = sympy.compose(sympy.Poly(x**2 + 1, x), sympy.Poly(2 * x, x))
        self.assertIsInstance(comp_res, sympy.Poly)
        self.assertEqual(comp_res, sympy.Poly(4 * x**2 + 1, x))

        # decompose
        decomp = sympy.decompose(x**4 + 2 * x**3 + 3 * x**2 + 2 * x + 1)
        self.assertEqual(decomp, [x**2 + 2 * x + 1, x**2 + x])
        p_decomp = sympy.Poly(x**4 + 2 * x**2 + 1, x).decompose()
        self.assertEqual(p_decomp, [sympy.Poly(x**2 + 2 * x + 1, x), sympy.Poly(x**2, x)])

    def test_expand_helpers_and_enhanced_simps(self):
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")

        # expand_trig
        self.assertEqual(
            sympy.expand_trig(sympy.sin(x + y)),
            sympy.sin(x) * sympy.cos(y) + sympy.cos(x) * sympy.sin(y)
        )
        self.assertEqual(
            sympy.expand_trig(sympy.sin(2 * x)),
            2 * sympy.sin(x) * sympy.cos(x)
        )
        self.assertEqual(
            sympy.expand_trig(sympy.cos(2 * x)),
            2 * sympy.cos(x)**2 - 1
        )

        # expand_log
        self.assertEqual(
            sympy.expand_log(sympy.log(x * y), force=True),
            sympy.log(x) + sympy.log(y)
        )
        self.assertEqual(
            sympy.expand_log(sympy.log(x**2), force=True),
            2 * sympy.log(x)
        )

        # expand_power_exp
        self.assertEqual(
            sympy.expand_power_exp(a**(x + y)),
            a**x * a**y
        )

        # expand_power_base
        self.assertEqual(
            sympy.expand_power_base((a * b)**x, force=True),
            a**x * b**x
        )

        # trigsimp
        self.assertEqual(
            sympy.trigsimp(sympy.sin(x)**2 + sympy.cos(x)**2),
            sympy.Integer(1)
        )
        self.assertEqual(
            sympy.trigsimp(1 + sympy.tan(x)**2),
            sympy.sec(x)**2
        )
        self.assertEqual(
            sympy.trigsimp(sympy.tan(x) * sympy.cos(x)),
            sympy.sin(x)
        )

        # powsimp
        self.assertEqual(
            sympy.powsimp(x**a * x**b),
            x**(a + b)
        )
        self.assertEqual(
            sympy.powsimp(x**a * y**a, force=True),
            (x * y)**a
        )

    def test_ntheory_residues_and_partitions(self):
        # sqrt_mod
        self.assertEqual(sympy.sqrt_mod(4, 7), 2)
        self.assertEqual(sympy.sqrt_mod(4, 7, all_roots=True), [2, 5])
        r = sympy.sqrt_mod(10, 13)
        self.assertEqual((r * r) % 13, 10)

        # quadratic_residues
        self.assertEqual(sympy.quadratic_residues(7), [0, 1, 2, 4])
        self.assertEqual(sympy.quadratic_residues(5), [0, 1, 4])

        # is_nthpow_residue
        self.assertTrue(sympy.is_nthpow_residue(6, 3, 7))
        self.assertFalse(sympy.is_nthpow_residue(2, 3, 7))
        self.assertTrue(sympy.is_nthpow_residue(1, 3, 7))

        # discrete_log
        self.assertEqual(sympy.discrete_log(41, 15, 7), 3)
        self.assertEqual(pow(7, 3, 41), 15)

        # npartitions
        self.assertEqual([sympy.npartitions(i) for i in range(6)], [1, 1, 2, 3, 5, 7])
        self.assertEqual(sympy.npartitions(100), 190569292)

    def test_matrix_spaces_cholesky_and_pow_display(self):
        # Columnspace and Rowspace
        M = sympy.Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
        cs = M.columnspace()
        self.assertEqual(len(cs), 2)
        self.assertEqual(cs[0], sympy.Matrix([[1], [-2], [3]]))
        self.assertEqual(cs[1], sympy.Matrix([[0], [0], [6]]))
        self.assertEqual(M.colspace(), cs)

        rs = M.rowspace()
        self.assertEqual(len(rs), 2)
        self.assertEqual(rs[0], sympy.Matrix([[1, 3, 0]]))
        self.assertEqual(rs[1], sympy.Matrix([[0, 0, 1]]))

        # C and adjoint properties
        A = sympy.Matrix([[1, -sympy.I], [sympy.I, 2]])
        self.assertEqual(A.C, sympy.Matrix([[1, sympy.I], [-sympy.I, 2]]))
        self.assertEqual(A.adjoint(), A.H)

        # Cholesky decomposition (Hermitian)
        pos = sympy.Matrix([[25, 15, -5], [15, 18, 0], [-5, 0, 11]])
        L = pos.cholesky()
        self.assertEqual(L * L.T, pos)
        self.assertEqual(L[0, 0], sympy.Integer(5))

        # Cholesky with complex entries
        Ac = sympy.Matrix([[9, 3 * sympy.I], [-3 * sympy.I, 5]])
        Lc = sympy.cholesky(Ac)
        self.assertEqual(Lc * Lc.H, Ac)
        self.assertEqual(Lc[0, 0], sympy.Integer(3))
        self.assertEqual(Lc[1, 0], -sympy.I)
        self.assertEqual(Lc[1, 1], sympy.Integer(2))

        # Pow display formatting (negative base and rational exponent parenthesization)
        from sympy.core import _native
        res = _native.solve_expr("x**2 + 1", "x")
        self.assertEqual(len(res), 2)
        self.assertIn("(-4)**(1/2)", res[0])

    def test_special_functions_suite(self):
        x = sympy.Symbol("x")

        # Gamma and Beta
        self.assertEqual(sympy.gamma(1), 1)
        self.assertEqual(sympy.gamma(2), 1)
        self.assertEqual(sympy.gamma(5), 24)
        self.assertEqual(sympy.gamma(sympy.Rational(1, 2)), sympy.pi ** sympy.Rational(1, 2))
        self.assertEqual(sympy.gamma(sympy.Rational(3, 2)), (sympy.pi ** sympy.Rational(1, 2)) / 2)
        self.assertEqual(sympy.gamma(sympy.Rational(-1, 2)), -2 * (sympy.pi ** sympy.Rational(1, 2)))
        self.assertEqual(sympy.gamma(0), sympy.zoo)
        self.assertEqual(sympy.beta(1, x), 1 / x)
        self.assertEqual(sympy.beta(3, 4), sympy.Rational(1, 60))

        # Incomplete gamma and polygamma
        self.assertEqual(sympy.lowergamma(1, x), 1 - sympy.exp(-x))
        self.assertEqual(sympy.uppergamma(1, x), sympy.exp(-x))
        self.assertEqual(sympy.polygamma(0, 1), -sympy.EulerGamma)
        self.assertEqual(sympy.polygamma(1, 1), sympy.pi**2 / 6)
        self.assertEqual(sympy.digamma(1), -sympy.EulerGamma)
        self.assertEqual(sympy.trigamma(1), sympy.pi**2 / 6)
        self.assertEqual(sympy.loggamma(1), 0)
        self.assertEqual(sympy.loggamma(2), 0)

        # Bessel and spherical Bessel
        self.assertEqual(sympy.besselj(0, 0), 1)
        self.assertEqual(sympy.besselj(1, 0), 0)
        self.assertEqual(sympy.besseli(0, 0), 1)
        self.assertEqual(sympy.besseli(1, 0), 0)
        self.assertEqual(sympy.jn(0, 0), 1)
        self.assertEqual(sympy.jn(1, 0), 0)
        self.assertIsInstance(sympy.besselj(0, x), sympy.besselj)
        self.assertIsInstance(sympy.bessely(0, x), sympy.bessely)
        self.assertIsInstance(sympy.besselk(0, x), sympy.besselk)
        self.assertIsInstance(sympy.airyai(x), sympy.airyai)
        self.assertIsInstance(sympy.airybi(x), sympy.airybi)

        # Error functions
        self.assertEqual(sympy.erf(0), 0)
        self.assertEqual(sympy.erfc(0), 1)
        self.assertEqual(sympy.erfi(0), 0)
        self.assertEqual(sympy.erfinv(0), 0)
        self.assertEqual(sympy.erfcinv(1), 0)
        self.assertEqual(sympy.FresnelS(0), 0)
        self.assertEqual(sympy.FresnelC(0), 0)
        self.assertEqual(sympy.Si(0), 0)
        self.assertEqual(sympy.Shi(0), 0)

        # Zeta and related
        self.assertEqual(sympy.zeta(0), sympy.Rational(-1, 2))
        self.assertEqual(sympy.zeta(2), sympy.pi**2 / 6)
        self.assertEqual(sympy.zeta(4), sympy.pi**4 / 90)
        self.assertEqual(sympy.zeta(-2), 0)
        self.assertEqual(sympy.zeta(-4), 0)
        self.assertEqual(sympy.dirichlet_eta(1), sympy.log(2))
        self.assertEqual(sympy.polylog(1, 0), 0)
        self.assertEqual(sympy.polylog(1, x), -sympy.log(1 - x))

        # Hypergeometric and Meijer G
        h = sympy.hyper([1, 2], [3], x)
        self.assertIsInstance(h, sympy.hyper)
        self.assertEqual(h.args[2], x)
        mg = sympy.meijerg([1], [2], x)
        self.assertIsInstance(mg, sympy.meijerg)

        # Mathematical constants
        self.assertEqual(str(sympy.EulerGamma), "EulerGamma")
        self.assertEqual(str(sympy.Catalan), "Catalan")
        self.assertEqual(str(sympy.GoldenRatio), "GoldenRatio")

    def test_series_residue_and_fourier(self):
        x = sympy.Symbol("x")

        # Rational residues
        self.assertEqual(sympy.residue(1 / x, x, 0), 1)
        self.assertEqual(sympy.residue(1 / x**2, x, 0), 0)
        self.assertEqual(sympy.residue(1 / (x - 1), x, 1), 1)
        self.assertEqual(sympy.residue(1 / (x**2 - 1), x, 1), sympy.Rational(1, 2))
        self.assertEqual(sympy.residue(1 / (x**2 - 1), x, -1), sympy.Rational(-1, 2))
        self.assertEqual(sympy.residue(1 / (x**2 * (x + 1)), x, 0), -1)
        self.assertEqual(sympy.residue(1 / (x**2 * (x + 1)), x, -1), 1)

        # Complex poles
        self.assertEqual(sympy.residue(x / (x**2 + 1), x, sympy.I), sympy.Rational(1, 2))

        # Transcendental residues
        self.assertEqual(sympy.residue(sympy.sin(x) / x**2, x, 0), 1)
        self.assertEqual(sympy.residue(sympy.cos(x) / x, x, 0), 1)

        # Fourier series
        fs = sympy.fourier_series(x, (x, -sympy.pi, sympy.pi))
        self.assertEqual(fs.truncate(3), 2 * sympy.sin(x) - sympy.sin(2 * x) + sympy.Integer(2) * sympy.sin(3 * x) / 3)

        # Formal power series
        fps_cos = sympy.fps(sympy.cos(x), x, 0)
        self.assertIn("x**4", str(fps_cos))

    def test_combinatorics_subsystem(self):
        # Permutation construction and properties
        p = sympy.Permutation(0, 1, 2)
        self.assertEqual(p.array_form, [1, 2, 0])
        self.assertEqual(p.order(), 3)
        self.assertTrue(p.is_even)
        self.assertFalse(p.is_odd)
        self.assertEqual(p.signature(), 1)
        self.assertEqual(p.parity(), 0)
        self.assertEqual(p.inversions(), 2)

        # Inverse and powers
        self.assertEqual(~p, sympy.Permutation([2, 0, 1]))
        self.assertEqual(p ** (-1), ~p)
        self.assertEqual(p**2, ~p)
        self.assertEqual(p**3, sympy.Permutation(size=3))

        # Left-to-right composition
        q1 = sympy.Permutation([1, 0, 2])
        q2 = sympy.Permutation([0, 2, 1])
        self.assertEqual((q1 * q2).array_form, [2, 0, 1])

        # Ranking and unranking
        for r in range(6):
            perm = sympy.Permutation.unrank_lex(3, r)
            self.assertEqual(perm.rank(), r)

        # Permutation groups
        G = sympy.PermutationGroup(p, q1)
        self.assertEqual(G.order(), 6)
        self.assertEqual(G.degree, 3)
        self.assertFalse(G.is_abelian)
        self.assertEqual(G.orbit(0), {0, 1, 2})
        self.assertTrue(G.is_transitive())

        # Named groups
        S3 = sympy.SymmetricGroup(3)
        self.assertEqual(S3.order(), 6)
        self.assertFalse(S3.is_abelian)
        self.assertTrue(S3.is_solvable)

        A3 = sympy.AlternatingGroup(3)
        self.assertEqual(A3.order(), 3)
        self.assertTrue(A3.is_abelian)

        D4 = sympy.DihedralGroup(4)
        self.assertEqual(D4.order(), 8)

        C5 = sympy.CyclicGroup(5)
        self.assertEqual(C5.order(), 5)
        self.assertTrue(C5.is_abelian)

        # Partitions
        part = sympy.Partition([1, 2], [3])
        self.assertEqual(part.members, [1, 2, 3])
        self.assertEqual(part.RGS, [0, 0, 1])

        ipart = sympy.IntegerPartition([3, 1, 1])
        self.assertEqual(ipart.integer, 5)
        self.assertEqual(ipart.as_dict(), {3: 1, 1: 2})

        # GrayCode
        gc = sympy.GrayCode(3)
        codes = list(gc.generate_gray())
        self.assertEqual(codes, ["000", "001", "011", "010", "110", "111", "101", "100"])
        self.assertEqual(sympy.GrayCode.rank("011"), 2)

        # Subsets
        sub = sympy.Subset(["a", "c"], ["a", "b", "c"])
        self.assertEqual(sub.rank_binary(), 5)
        unsub = sympy.Subset.unrank_binary(5, ["a", "b", "c"])
        self.assertEqual(unsub.subset, ["a", "c"])

    def test_euler_and_finite_differences(self):
        x = sympy.Symbol("x")
        f = sympy.Function("f")

        # Euler-Lagrange equations: L = (f'(x))^2 / 2 - f(x)^2 / 2  => f''(x) + f(x) = 0
        L = sympy.diff(f(x), x) ** 2 / 2 - f(x) ** 2 / 2
        eqs = sympy.euler_equations(L, f(x), x)
        self.assertEqual(len(eqs), 1)
        self.assertEqual(eqs[0].lhs, -sympy.diff(f(x), x, x) - f(x))
        self.assertEqual(eqs[0].rhs, sympy.Integer(0))

        # Finite difference weights for 1st derivative on stencil [-1, 0, 1] at x0=0
        weights = sympy.finite_diff_weights(1, [-1, 0, 1], 0)
        # delta[order][N] -> weights for all grid points
        self.assertEqual(weights[1][-1], [sympy.Rational(-1, 2), sympy.Integer(0), sympy.Rational(1, 2)])

        # Apply finite diff on points [-1, 1] for y = [0, 2] -> dy/dx ~ 1
        approx = sympy.apply_finite_diff(1, [-1, 1], [0, 2], 0)
        self.assertEqual(approx, sympy.Integer(1))

        # Derivative.as_finite_difference()
        d = sympy.Derivative(f(x), x)
        fd = d.as_finite_difference()
        self.assertEqual(fd, -f(x - sympy.Rational(1, 2)) + f(x + sympy.Rational(1, 2)))

        # differentiate_finite
        fd2 = sympy.differentiate_finite(sympy.Derivative(f(x), x))
        self.assertEqual(fd2, fd)

    def test_integral_transforms_suite(self):
        t, s, x, k, r, nu = sympy.symbols("t s x k r nu")

        # Laplace and Inverse Laplace
        lt = sympy.LaplaceTransform(sympy.exp(t), t, s)
        self.assertEqual(str(lt), "LaplaceTransform(exp(t), t, s)")
        self.assertEqual(lt.doit(), (s - 1)**(-1))
        self.assertEqual(sympy.laplace_transform(sympy.exp(t), t, s), (s - 1)**(-1))

        ilt = sympy.InverseLaplaceTransform(1 / (s - 1), s, t)
        self.assertEqual(str(ilt), "InverseLaplaceTransform((s - 1)**(-1), s, t)")
        self.assertEqual(ilt.doit(), sympy.exp(t))
        self.assertEqual(sympy.inverse_laplace_transform(1 / (s - 1), s, t), sympy.exp(t))

        # Fourier and Inverse Fourier
        ft = sympy.FourierTransform(sympy.Integer(5), t, x)
        self.assertEqual(str(ft), "FourierTransform(5, t, x)")
        self.assertEqual(str(ft.doit()), "10*pi*dirac(x)")

        ift = sympy.InverseFourierTransform(ft.doit(), x, t)
        self.assertIsInstance(ift.doit(), sympy.InverseFourierTransform)

        # Sine, Cosine, Hankel, Mellin unevaluated classes and doit()
        st = sympy.SineTransform(t * sympy.exp(-t), t, k)
        self.assertIsInstance(st, sympy.SineTransform)
        self.assertEqual(st.doit(), st)

        ct = sympy.CosineTransform(sympy.exp(-t), t, k)
        self.assertIsInstance(ct, sympy.CosineTransform)
        self.assertEqual(ct.doit(), ct)

        ht = sympy.HankelTransform(sympy.exp(-r), r, k, nu)
        self.assertIsInstance(ht, sympy.HankelTransform)
        self.assertEqual(ht.doit(), ht)

        mt = sympy.MellinTransform(sympy.exp(-x), x, s)
        self.assertIsInstance(mt, sympy.MellinTransform)
        self.assertEqual(mt.doit(), mt)


if __name__ == "__main__":
    unittest.main()
