"""Polynomial-forced ordinary dsolve systems through the actual native bridge."""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Derivative, Dummy, Eq, Float, Function, Mul, Pow, Rational, Symbol,
    cos, diff, dsolve, exp, expand, simplify, sin,
)
from sympy.core import _native, _native_expr, _symbol_binding


class PolynomialOdeSystemTests(unittest.TestCase):
    def setUp(self):
        self.t = Symbol("t")
        self.f = Function("forced_system_f")
        self.g = Function("forced_system_g")

    def equations(self, matrix, forcing, time=None):
        time = self.t if time is None else time
        functions = [self.f(time), self.g(time)]
        return [Eq(Derivative(function, time), row[0]*functions[0] + row[1]*functions[1] + load)
                for function, row, load in zip(functions, matrix, forcing)]

    def assert_zero(self, value):
        self.assertEqual(simplify(expand(value)), 0)

    def check(self, answers, matrix, forcing, time=None, functions=None):
        time = self.t if time is None else time
        functions = [self.f(time), self.g(time)] if functions is None else functions
        self.assertIs(type(answers), list)
        self.assertEqual(len(answers), 2)
        self.assertEqual([answer.lhs for answer in answers], functions)
        self.assertTrue(all(type(answer) is Eq for answer in answers))
        values = [answer.rhs for answer in answers]
        for value, row, load in zip(values, matrix, forcing):
            self.assert_zero(diff(value, time) - row[0]*values[0] - row[1]*values[1] - load)
        return values

    def test_forced_rotation_matches_the_hand_derived_unique_ivp(self):
        t = self.t
        matrix, forcing = [[0, 1], [-1, 0]], [3, t*t + 3]
        values = self.check(dsolve(self.equations(matrix, forcing), ics={self.f(0): 1, self.g(0): 2}), matrix, forcing)
        for value, expected in zip(values, (t*t + 1 + 5*sin(t), 2*t - 3 + 5*cos(t))):
            self.assert_zero(value - expected)
            self.assertEqual(value.free_symbols, {t})

    def test_constant_forcing_crosses_the_new_lane(self):
        matrix, forcing = [[0, 1], [-1, 0]], [1, 0]
        values = self.check(dsolve(self.equations(matrix, forcing), ics={self.f(0): 0, self.g(0): 0}), matrix, forcing)
        self.assert_zero(values[0] - sin(self.t))
        self.assert_zero(values[1] - cos(self.t) + 1)

    def test_nilpotent_resonance_keeps_degree_eighteen_and_two_free_constants(self):
        t, c1, c2 = self.t, Symbol("C1"), Symbol("C2")
        matrix, forcing = [[0, 1], [0, 0]], [0, t**16]
        values = self.check(dsolve(self.equations(matrix, forcing)), matrix, forcing)
        self.assert_zero(values[0] - c1 - c2*t - t**18/306)
        self.assert_zero(values[1] - c2 - t**17/17)
        self.assertEqual([simplify(value.subs(t, 0)) for value in values], [c1, c2])

    def test_shifted_nilpotent_ivp_preserves_rational_correction_coefficients(self):
        t, point = self.t, Rational(-2, 3)
        matrix = [[0, 1], [0, 0]]
        forcing = [Rational(2, 3) - Rational(3, 2)*t + Rational(5, 7)*t*t,
                   Rational(-4, 5) + Rational(7, 11)*t]
        values = self.check(dsolve(self.equations(matrix, forcing), ics={self.f(point): 1, self.g(point): 2}), matrix, forcing)
        self.assertEqual([simplify(value.subs(t, point)) for value in values], [1, 2])
        expected_y = 2 - Rational(4, 5)*t + Rational(7, 22)*t*t - Rational(334, 495)
        self.assert_zero(values[1] - expected_y)

    def test_rank_one_system_has_a_complete_partial_initial_family(self):
        t, c2 = self.t, Symbol("C2")
        matrix, forcing = [[1, 1], [0, 0]], [t, 1]
        values = self.check(dsolve(self.equations(matrix, forcing), ics={self.f(2): 1}), matrix, forcing)
        self.assert_zero(values[0] - ((5+c2)*exp(t-2) - 2*t - c2))
        self.assert_zero(values[1] - (c2+t-2))
        self.assertEqual(simplify(values[0].subs(t, 2)), 1)
        self.assertEqual(simplify(values[1].subs(t, 2)), c2)

    def test_zero_matrix_integrates_polynomials_from_a_rational_initial_time(self):
        t, point = self.t, Rational(2, 3)
        matrix, forcing = [[0, 0], [0, 0]], [1+t*t, 2*t-3]
        data = {self.f(point): Rational(1, 2), self.g(point): Rational(-5, 4)}
        values = self.check(dsolve(self.equations(matrix, forcing), ics=data), matrix, forcing)
        expected = [Rational(1, 2)+(t-point)+(t**3-point**3)/3,
                    Rational(-5, 4)+(t*t-point*point)-3*(t-point)]
        for value, wanted in zip(values, expected):
            self.assert_zero(value - wanted)

    def test_derivative_matrix_inversion_transforms_the_complete_polynomial_forcing(self):
        t = self.t
        f, g = self.f(t), self.g(t)
        df, dg = Derivative(f, t), Derivative(g, t)
        sources = (
            [Eq(2*df+dg, -f+2*g+t*t+9), Eq(df+3*dg, -3*f+g+3*t*t+12)],
            [2*df+dg+f-2*g-t*t-9, df+3*dg+3*f-g-3*t*t-12],
        )
        for equations in sources:
            with self.subTest(equality_form=type(equations[0]) is Eq):
                values = self.check(dsolve(equations, [f, g], ics={self.f(0): 1, self.g(0): 2}), [[0, 1], [-1, 0]], [3, t*t+3])
                self.assert_zero(values[0] - t*t - 1 - 5*sin(t))
                self.assert_zero(values[1] - 2*t + 3 - 5*cos(t))

    def test_reversed_function_order_keeps_forcing_coordinates_and_initial_data_bound(self):
        t = self.t
        equations = self.equations([[0, 1], [-1, 0]], [3, t*t+3])
        reverse = [self.g(t), self.f(t)]
        for source, supplied in ((list(reversed(equations)), None), (equations, reverse)):
            with self.subTest(explicit=supplied is not None):
                values = self.check(dsolve(source, supplied, ics={self.f(0): 1, self.g(0): 2}), [[0, -1], [1, 0]], [t*t+3, 3], functions=reverse)
                self.assert_zero(values[0] - 2*t + 3 - 5*cos(t))
                self.assert_zero(values[1] - t*t - 1 - 5*sin(t))

    def test_typed_time_and_constant_freshness_survive_polynomial_composition(self):
        for time in (Symbol("t", positive=True), Dummy("t"), Symbol("C1")):
            with self.subTest(time=time):
                matrix, forcing = [[0, 1], [0, 0]], [time, 1]
                values = self.check(dsolve(self.equations(matrix, forcing, time)), matrix, forcing, time)
                constants = [Symbol("C2"), Symbol("C3")] if time.name == "C1" else [Symbol("C1"), Symbol("C2")]
                self.assertEqual([simplify(value.subs(time, 0)) for value in values], constants)
                self.assertEqual(set().union(*(value.free_symbols for value in values)), {time, *constants})

    def test_hidden_nonpositive_polynomial_powers_keep_their_original_domains(self):
        t = self.t
        f, g = self.f(t), self.g(t)
        for partial in (Pow(t, -1, evaluate=False), Pow(t, 0, evaluate=False),
                        Pow(t-1, 0, evaluate=False), Pow(0, -1, evaluate=False)):
            hidden = Mul(0, partial, evaluate=False)
            sources = (
                Eq(Derivative(f, t), Add(g, t, hidden, evaluate=False), evaluate=False),
                Eq(Add(Derivative(f, t)-g-t, partial, evaluate=False), partial, evaluate=False),
            )
            for source in sources:
                with self.subTest(partial=partial, source=source), patch.object(
                    _native, "polynomial_ode_system_native", side_effect=AssertionError("original domain was erased"),
                ) as native:
                    with self.assertRaises((NotImplementedError, ValueError)):
                        dsolve([source, Eq(Derivative(g, t), -f)])
                    self.assertEqual(native.call_count, 0)

    def test_degree_foreign_time_nonlinearity_and_variable_coefficients_refuse_before_native(self):
        t, f, g = self.t, self.f(self.t), self.g(self.t)
        df, dg = Derivative(f, t), Derivative(g, t)
        for first in (df-g-t**17, df-g-Symbol("a")*t,
                      df-g-Symbol("t", real=True), df-g-exp(t),
                      df-t*g-t, t*df-g-t, df-f*g-t, df-g-Float(1.0)):
            with self.subTest(source=first), patch.object(
                _native, "polynomial_ode_system_native", side_effect=AssertionError("unsupported forcing was lowered"),
            ):
                with self.assertRaises(NotImplementedError):
                    dsolve([first, dg+f])

    def test_native_forcing_failure_is_terminal_and_zero_forcing_uses_existing_homogeneous_lane(self):
        equations = self.equations([[0, 1], [-1, 0]], [self.t, 1])
        with patch.object(_native, "polynomial_ode_system_native", side_effect=ValueError("forced verifier budget exhausted")), patch.object(
            _native, "linear_ode_system_native", side_effect=AssertionError("forced failure fell back to homogeneous solve"),
        ):
            with self.assertRaisesRegex(ValueError, "forced verifier budget exhausted"):
                dsolve(equations)
        with patch.object(_native, "polynomial_ode_system_native", side_effect=AssertionError("homogeneous behavior changed")):
            self.check(dsolve(self.equations([[0, 1], [-1, 0]], [0, 0])), [[0, 1], [-1, 0]], [0, 0])

    def test_borrowed_native_forcing_vector_limits_precede_extraction(self):
        arguments = (_native_expr(0), _native_expr(1), _native_expr(-1), _native_expr(0),
                     _symbol_binding(self.t), _native_expr(0), _native_expr(0), _native_expr(0))
        for bad in ([], [object()]*18):
            with self.subTest(size=len(bad)):
                with self.assertRaisesRegex(ValueError, "seventeen coefficients"):
                    _native.polynomial_ode_system_native(*arguments, bad, [_native_expr(0)])


if __name__ == "__main__":
    unittest.main()
