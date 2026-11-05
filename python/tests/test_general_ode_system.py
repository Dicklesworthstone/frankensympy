"""General rational dsolve systems through the actual native extension.

Hand-derived IVPs exercise dimensions beyond the existing two-coordinate lane,
polynomial resonance, original derivative coordinates and retained identities.
"""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Derivative, Dummy, Eq, Float, Function, Mul, Pow, Rational, Symbol,
    cos, cosh, diff, dsolve, exp, expand, simplify, sin, sinh, sqrt,
)
from sympy.core import (
    AppliedUndef, UndefinedFunction, _native, _native_expr, _symbol_binding,
)


class GeneralOdeSystemTests(unittest.TestCase):
    def setUp(self):
        self.t = Symbol("t")
        self.owners = [Function(f"general_system_{index}") for index in range(8)]

    def equations(self, matrix, forcing=None, time=None, owners=None):
        time = self.t if time is None else time
        owners = self.owners[:len(matrix)] if owners is None else owners
        functions = [owner(time) for owner in owners]
        forcing = [0]*len(matrix) if forcing is None else forcing
        equations = [
            Eq(Derivative(function, time),
               sum(coefficient*value for coefficient, value in zip(row, functions)) + load)
            for function, row, load in zip(functions, matrix, forcing)
        ]
        return equations, functions

    def assert_zero(self, expression):
        self.assertEqual(simplify(expand(expression)), 0)

    def assert_system(self, answers, matrix, forcing, functions, time=None):
        time = self.t if time is None else time
        self.assertIs(type(answers), list)
        self.assertEqual(len(answers), len(functions))
        self.assertEqual([answer.lhs for answer in answers], functions)
        self.assertTrue(all(type(answer) is Eq for answer in answers))
        values = [answer.rhs for answer in answers]
        for value, row, load in zip(values, matrix, forcing):
            self.assert_zero(
                diff(value, time)
                - sum(coefficient*other for coefficient, other in zip(row, values))
                - load
            )
        return values

    def test_one_component_list_returns_a_checked_forced_ivp(self):
        t = self.t
        matrix, forcing = [[2]], [t]
        equations, functions = self.equations(matrix, forcing)
        answers = dsolve(equations, functions, ics={self.owners[0](0): 3})
        values = self.assert_system(answers, matrix, forcing, functions)
        self.assert_zero(values[0] - (Rational(13, 4)*exp(2*t) - t/2 - Rational(1, 4)))
        self.assertEqual(simplify(values[0].subs(t, 0)), 3)
        self.assertEqual(values[0].free_symbols, {t})

    def test_every_admitted_dimension_keeps_all_independent_initial_coordinates(self):
        for count in (1, 3, 4, 5, 6, 7, 8):
            with self.subTest(count=count):
                matrix = [[0]*count for _ in range(count)]
                forcing = list(range(1, count + 1))
                equations, functions = self.equations(matrix, forcing)
                values = self.assert_system(
                    dsolve(equations, functions), matrix, forcing, functions,
                )
                constants = [Symbol(f"C{index + 1}") for index in range(count)]
                self.assertEqual([simplify(value.subs(self.t, 0)) for value in values], constants)
                for value, constant, load in zip(values, constants, forcing):
                    self.assert_zero(value - constant - load*self.t)
                self.assertEqual(
                    set().union(*(value.free_symbols for value in values)),
                    {self.t, *constants},
                )

    def test_three_coordinate_nilpotent_resonance_reaches_degree_nineteen(self):
        t = self.t
        matrix = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        forcing = [0, 0, t**16]
        equations, functions = self.equations(matrix, forcing)
        data = {owner(0): 0 for owner in self.owners[:3]}
        values = self.assert_system(
            dsolve(equations, functions, ics=data), matrix, forcing, functions,
        )
        expected = [t**19/(17*18*19), t**18/(17*18), t**17/17]
        for value, wanted in zip(values, expected):
            self.assert_zero(value - wanted)
            self.assertEqual(simplify(value.subs(t, 0)), 0)
            self.assertEqual(value.free_symbols, {t})

    def test_shifted_rational_initial_time_translates_the_original_polynomial_forcing(self):
        t, point = self.t, Rational(-2, 3)
        tau = t - point
        matrix = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        forcing = [0, 0, t*t]
        equations, functions = self.equations(matrix, forcing)
        data = {owner(point): value for owner, value in zip(self.owners, (1, 2, 3))}
        values = self.assert_system(
            dsolve(equations, functions, ics=data), matrix, forcing, functions,
        )
        expected = [
            1 + 2*tau + 3*tau**2/2 + point**2*tau**3/6 + point*tau**4/12 + tau**5/60,
            2 + 3*tau + point**2*tau**2/2 + point*tau**3/3 + tau**4/12,
            3 + point**2*tau + point*tau**2 + tau**3/3,
        ]
        for value, wanted, initial in zip(values, expected, (1, 2, 3)):
            self.assert_zero(value - wanted)
            self.assertEqual(simplify(value.subs(t, point)), initial)
            self.assertEqual(value.free_symbols, {t})

    def test_partial_initial_data_preserves_the_unfixed_coordinate_as_an_independent_constant(self):
        t, point, c2 = self.t, Rational(3, 2), Symbol("C2")
        tau = t - point
        matrix = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        equations, functions = self.equations(matrix)
        data = {self.owners[0](point): Rational(7, 3), self.owners[2](point): -2}
        values = self.assert_system(
            dsolve(equations, functions, ics=data), matrix, [0, 0, 0], functions,
        )
        expected = [Rational(7, 3) + c2*tau - tau*tau, c2 - 2*tau, -2]
        for value, wanted in zip(values, expected):
            self.assert_zero(value - wanted)
        self.assertEqual([simplify(value.subs(t, point)) for value in values], [Rational(7, 3), c2, -2])
        self.assertEqual(set().union(*(value.free_symbols for value in values)), {t, c2})

    def test_dense_coupling_preserves_oscillatory_and_exponential_modes(self):
        # A = P diag([[0, -1], [1, 0]], [2]) P**-1,
        # P = [[1, 1, 1], [1, 2, 3], [2, 3, 5]], with modal data (1, 2, 3).
        t = self.t
        matrix = [[-2, -7, 5], [-5, -13, 10], [-9, -22, 17]]
        equations, functions = self.equations(matrix)
        data = {owner(0): value for owner, value in zip(self.owners, (6, 14, 23))}
        values = self.assert_system(
            dsolve(equations, functions, ics=data), matrix, [0, 0, 0], functions,
        )
        expected = [
            3*cos(t) - sin(t) + 3*exp(2*t),
            5*cos(t) + 9*exp(2*t),
            8*cos(t) - sin(t) + 15*exp(2*t),
        ]
        for value, wanted in zip(values, expected):
            self.assert_zero(value - wanted)

    def test_real_quadratic_modes_keep_the_normalized_hyperbolic_sine(self):
        t = self.t
        matrix = [[0, 1, 0], [2, 0, 0], [0, 0, -1]]
        equations, functions = self.equations(matrix)
        data = {owner(0): value for owner, value in zip(self.owners, (1, 3, 4))}
        values = self.assert_system(
            dsolve(equations, functions, ics=data), matrix, [0, 0, 0], functions,
        )
        expected = [
            cosh(sqrt(2)*t) + 3*sinh(sqrt(2)*t)/sqrt(2),
            sqrt(2)*sinh(sqrt(2)*t) + 3*cosh(sqrt(2)*t),
            4*exp(-t),
        ]
        for value, wanted in zip(values, expected):
            self.assert_zero(value - wanted)

    def test_original_nondiagonal_derivative_matrix_and_forcing_are_solved_together(self):
        t = self.t
        f, g, h = [owner(t) for owner in self.owners[:3]]
        df, dg, dh = [Derivative(value, t) for value in (f, g, h)]
        # The first pivot of this invertible mass matrix is zero.
        sources = (
            [Eq(2*dg + dh, 2*h + t*t),
             Eq(df + dh, g + t*t),
             Eq(dg + dh, h + t*t)],
            [2*dg + dh - 2*h - t*t,
             df + dh - g - t*t,
             dg + dh - h - t*t],
        )
        matrix, forcing = [[0, 1, 0], [0, 0, 1], [0, 0, 0]], [0, 0, t*t]
        data = {owner(0): 0 for owner in self.owners[:3]}
        for equations in sources:
            with self.subTest(equality_form=type(equations[0]) is Eq):
                values = self.assert_system(
                    dsolve(equations, [f, g, h], ics=data), matrix, forcing, [f, g, h],
                )
                for value, wanted in zip(values, (t**5/60, t**4/12, t**3/3)):
                    self.assert_zero(value - wanted)

    def test_same_named_distinct_function_owners_follow_source_and_explicit_order(self):
        first = type.__new__(UndefinedFunction, "general_same", (AppliedUndef,), {})
        second = type.__new__(UndefinedFunction, "general_same", (AppliedUndef,), {})
        owners = [first, second, self.owners[2]]
        self.assertIsNot(first, second)
        equations, functions = self.equations([[0]*3 for _ in range(3)], [2, 5, 11], owners=owners)
        self.assertNotEqual(functions[0], functions[1])
        data = {owner(0): value for owner, value in zip(owners, (1, 3, 7))}
        expected = [1 + 2*self.t, 3 + 5*self.t, 7 + 11*self.t]
        cases = (
            ([equations[index] for index in (2, 0, 1)], None, (2, 0, 1)),
            (tuple(equations), [functions[index] for index in (1, 2, 0)], (1, 2, 0)),
        )
        for source, supplied, order in cases:
            with self.subTest(order=order):
                answers = dsolve(source, supplied, ics=data)
                self.assertEqual(len(answers), 3)
                for answer, index in zip(answers, order):
                    self.assertIs(type(answer.lhs), owners[index])
                    self.assert_zero(answer.rhs - expected[index])

    def test_typed_times_and_constant_name_collisions_remain_bound_to_original_coordinates(self):
        cases = (
            (Symbol("t", positive=True), self.owners[:3], ["C1", "C2", "C3"]),
            (Dummy("t"), self.owners[:3], ["C1", "C2", "C3"]),
            (Symbol("C1"), [Function("C2"), Function("C3"), self.owners[2]], ["C4", "C5", "C6"]),
        )
        for time, owners, names in cases:
            with self.subTest(time=time):
                matrix, forcing = [[0]*3 for _ in range(3)], [time, 1, 0]
                equations, functions = self.equations(matrix, forcing, time=time, owners=owners)
                answers = dsolve(equations, functions)
                values = self.assert_system(answers, matrix, forcing, functions, time=time)
                constants = [Symbol(name) for name in names]
                self.assertEqual([simplify(value.subs(time, 0)) for value in values], constants)
                self.assertEqual(set().union(*(value.free_symbols for value in values)), {time, *constants})
                for answer in answers:
                    self.assertIs(answer.lhs.args[0], time)

    def test_irreducible_cubic_spectrum_refuses_without_scalar_or_legacy_fallback(self):
        # Characteristic polynomial s**3 - s - 1 has no rational root.
        matrix = [[0, 1, 0], [0, 0, 1], [1, 1, 0]]
        equations, functions = self.equations(matrix)
        with patch.object(
            _native, "linear_ode_system_native", side_effect=AssertionError("entered the two-coordinate bridge"),
        ), patch.object(
            _native, "polynomial_ode_system_native", side_effect=AssertionError("entered the two-coordinate bridge"),
        ), patch(
            "sympy.solvers.ode._dsolve_general", side_effect=AssertionError("entered scalar fallback"),
        ):
            with self.assertRaises(NotImplementedError):
                dsolve(equations, functions)

    def test_original_held_domains_are_checked_before_any_algebraic_erasure(self):
        t = self.t
        f, g, h = [owner(t) for owner in self.owners[:3]]
        tail = [Eq(Derivative(g, t), h), Eq(Derivative(h, t), t)]
        for partial in (
            Pow(t, -1, evaluate=False), Pow(t - 1, 0, evaluate=False),
            Pow(0, -1, evaluate=False),
        ):
            hidden = Mul(0, partial, evaluate=False)
            sources = (
                Eq(Derivative(f, t), Add(g, hidden, evaluate=False), evaluate=False),
                Eq(Add(Derivative(f, t) - g, partial, evaluate=False), partial, evaluate=False),
            )
            for first in sources:
                with self.subTest(partial=partial, first=first), patch.object(
                    _native, "general_ode_system_native",
                    side_effect=AssertionError("an original domain was erased"),
                ) as native:
                    with self.assertRaises((NotImplementedError, ValueError)):
                        dsolve([first, *tail], [f, g, h])
                    self.assertEqual(native.call_count, 0)

    def test_dimension_degree_singular_and_foreign_coordinate_refusals_precede_native(self):
        t = self.t
        f, g, h = [owner(t) for owner in self.owners[:3]]
        df, dg, dh = [Derivative(value, t) for value in (f, g, h)]
        sources = (
            [df - g - t**17, dg - h, dh],
            [df - t*g, dg - h, dh],
            [df - g - Float(1.0), dg - h, dh],
            [df - f*g, dg - h, dh],
            [df + dg - f, 2*df + 2*dg - g, dh],
        )
        with patch.object(
            _native, "general_ode_system_native", side_effect=AssertionError("unsupported source reached native"),
        ) as native:
            for equations in sources:
                with self.subTest(source=equations):
                    with self.assertRaises((NotImplementedError, ValueError)):
                        dsolve(equations, [f, g, h])
            owners = [Function(f"general_oversized_{index}") for index in range(9)]
            equations, functions = self.equations([[0]*9 for _ in range(9)], owners=owners)
            with self.assertRaises((NotImplementedError, ValueError)):
                dsolve(equations, functions)
            foreign_h = self.owners[2](Symbol("t", real=True))
            equations = [df - g, dg - h, Derivative(foreign_h, foreign_h.args[0])]
            with self.assertRaises(ValueError):
                dsolve(equations, [f, g, foreign_h])
            self.assertEqual(native.call_count, 0)

    def test_custom_function_hooks_are_not_an_exact_system_source(self):
        observed = []

        class Custom(Function):
            @classmethod
            def eval(cls, argument):
                observed.append(argument)
                return None

        custom = Custom(self.t, evaluate=False)
        g, h = [owner(self.t) for owner in self.owners[1:3]]
        equations = [
            Eq(Derivative(custom, self.t), g),
            Eq(Derivative(g, self.t), h),
            Eq(Derivative(h, self.t), custom),
        ]
        with self.assertRaises(NotImplementedError):
            dsolve(equations)
        self.assertEqual(observed, [])

    def test_native_budget_and_verifier_failures_propagate_without_a_second_solver(self):
        equations, functions = self.equations([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        for message in ("general system verifier budget exhausted", "general system computation cancelled"):
            failure = ValueError(message)
            with self.subTest(message=message), patch.object(
                _native, "general_ode_system_native", side_effect=failure,
            ) as native, patch.object(
                _native, "linear_ode_system_native", side_effect=AssertionError("operational fallback"),
            ), patch.object(
                _native, "polynomial_ode_system_native", side_effect=AssertionError("operational fallback"),
            ), patch(
                "sympy.solvers.ode._dsolve_general", side_effect=AssertionError("operational fallback"),
            ):
                with self.assertRaises(ValueError) as error:
                    dsolve(equations, functions)
                self.assertIs(error.exception, failure)
                self.assertEqual(native.call_count, 1)

    def test_native_shape_admission_precedes_element_extraction_and_requires_typed_time(self):
        zero = _native_expr(0)
        matrix = [[zero]*3 for _ in range(3)]
        time = _symbol_binding(self.t)
        forcing, initial = [[zero] for _ in range(3)], [zero]*3
        with self.assertRaises(TypeError):
            _native.general_ode_system_native(matrix, "t", zero, forcing, initial)
        for bad_matrix in ([], [object()]*9, [[zero]*3, [zero]*2, [zero]*3]):
            with self.subTest(matrix=bad_matrix):
                with self.assertRaises(ValueError):
                    _native.general_ode_system_native(bad_matrix, time, zero, forcing, initial)
        for bad_row in ([], [object()]*18):
            with self.subTest(forcing_length=len(bad_row)):
                with self.assertRaises(ValueError):
                    _native.general_ode_system_native(matrix, time, zero, [[zero], bad_row, [zero]], initial)


if __name__ == "__main__":
    unittest.main()
