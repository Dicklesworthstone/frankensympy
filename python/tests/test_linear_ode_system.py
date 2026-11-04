"""Public dsolve systems: complete families, unique IVPs and retained sources.

These tests require the actual FrankenSymPy extension. Pinned upstream oracle
checks separately establish ordinary system signatures, equation-list returns,
function ordering, and the unique IVP identities; they do not execute this shell.
"""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Derivative, Dummy, Eq, Float, Function, Integer, Mul, Pow, Rational,
    Symbol, cos, cosh, diff, dsolve, exp, expand, simplify, sin, sinh, sqrt,
)
from sympy.core import _native, _native_expr


class LinearOdeSystemTests(unittest.TestCase):
    def setUp(self):
        self.t = Symbol("t")
        self.f = Function("system_f")
        self.g = Function("system_g")
        self.functions = [self.f(self.t), self.g(self.t)]

    def system(self, coefficients, time=None):
        time = self.t if time is None else time
        functions = [self.f(time), self.g(time)]
        return [Eq(Derivative(function, time), row[0]*functions[0] + row[1]*functions[1])
                for function, row in zip(functions, coefficients)]

    def assert_zero(self, expression):
        self.assertEqual(simplify(expand(expression)), 0)

    def assert_system(self, answers, coefficients, time=None, functions=None):
        time = self.t if time is None else time
        functions = self.functions if functions is None else functions
        self.assertIs(type(answers), list)
        self.assertEqual(len(answers), 2)
        self.assertEqual([answer.lhs for answer in answers], functions)
        for answer in answers:
            self.assertIs(type(answer), Eq)
        values = [answer.rhs for answer in answers]
        for value, row in zip(values, coefficients):
            self.assert_zero(diff(value, time) - row[0]*values[0] - row[1]*values[1])
        return values

    def test_rotation_returns_two_equations_and_a_complete_normalized_family(self):
        coefficients = [[0, 1], [-1, 0]]
        answers = dsolve(self.system(coefficients), self.functions)
        values = self.assert_system(answers, coefficients)
        c1, c2 = Symbol("C1"), Symbol("C2")
        self.assert_zero(values[0] - (c1*cos(self.t) + c2*sin(self.t)))
        self.assert_zero(values[1] - (c2*cos(self.t) - c1*sin(self.t)))
        self.assertEqual([simplify(value.subs(self.t, 0)) for value in values], [c1, c2])

    def test_repeated_jordan_nontrivial_nilpotent_and_zero_systems(self):
        c1, c2, t = Symbol("C1"), Symbol("C2"), self.t
        cases = (
            ([[1, 1], [0, 1]], [exp(t)*(c1 + c2*t), c2*exp(t)]),
            ([[1, 1], [-1, -1]], [c1 + (c1 + c2)*t, c2 - (c1 + c2)*t]),
            ([[0, 0], [0, 0]], [c1, c2]),
        )
        for coefficients, expected in cases:
            with self.subTest(coefficients=coefficients):
                values = self.assert_system(dsolve(self.system(coefficients)), coefficients)
                for actual, wanted in zip(values, expected):
                    self.assert_zero(actual - wanted)

    def test_distinct_irrational_real_modes_have_two_independent_initial_coordinates(self):
        coefficients = [[0, 1], [2, 0]]
        values = self.assert_system(dsolve(self.system(coefficients)), coefficients)
        c1, c2, t = Symbol("C1"), Symbol("C2"), self.t
        expected = [c1*cosh(sqrt(2)*t) + c2*sinh(sqrt(2)*t)/sqrt(2),
                    sqrt(2)*c1*sinh(sqrt(2)*t) + c2*cosh(sqrt(2)*t)]
        for actual, wanted in zip(values, expected):
            self.assert_zero(actual - wanted)
        self.assertEqual([simplify(value.subs(t, 0)) for value in values], [c1, c2])

    def test_exact_rational_ivps_at_zero_and_shifted_times(self):
        coefficients = [[0, 1], [-1, 0]]
        t = self.t
        for point in (0, 2, Rational(-3, 2), Rational(1, 3)):
            with self.subTest(point=point):
                data = {self.f(point): 1, self.g(point): 2}
                values = self.assert_system(dsolve(self.system(coefficients), ics=data), coefficients)
                expected = [cos(t - point) + 2*sin(t - point),
                            2*cos(t - point) - sin(t - point)]
                for actual, wanted, initial in zip(values, expected, (1, 2)):
                    self.assert_zero(actual - wanted)
                    self.assertEqual(simplify(actual.subs(t, point)), initial)
                    self.assertEqual(actual.free_symbols, {t})

    def test_rational_coefficients_and_partial_initial_data(self):
        coefficients = [[Rational(1, 2), Rational(2, 3)],
                        [Rational(-3, 5), Rational(-7, 4)]]
        answers = dsolve(self.system(coefficients), ics={self.f(Rational(1, 2)): Rational(2, 3)})
        values = self.assert_system(answers, coefficients)
        self.assertEqual(simplify(values[0].subs(self.t, Rational(1, 2))), Rational(2, 3))
        self.assertEqual(simplify(values[1].subs(self.t, Rational(1, 2))), Symbol("C2"))
        self.assertEqual(set().union(*(value.free_symbols for value in values)), {self.t, Symbol("C2")})

    def test_invertible_derivative_coefficient_matrix_is_lowered_exactly(self):
        t = self.t
        f, g = self.functions
        df, dg = Derivative(f, t), Derivative(g, t)
        equations = [Eq(2*df + dg, 2*g - f), Eq(df + 3*dg, g - 3*f)]
        values = self.assert_system(dsolve(equations, self.functions, ics={self.f(0): 1, self.g(0): 2}),
                                    [[0, 1], [-1, 0]])
        self.assert_zero(values[0] - cos(t) - 2*sin(t))
        self.assert_zero(values[1] - 2*cos(t) + sin(t))

    def test_inferred_and_explicit_order_preserve_original_function_owners(self):
        equations = self.system([[0, 1], [-1, 0]])
        reverse = [self.functions[1], self.functions[0]]
        for source, functions in ((list(reversed(equations)), None), (equations, reverse)):
            with self.subTest(explicit=functions is not None):
                answers = dsolve(source, functions, ics={self.f(0): 1, self.g(0): 2})
                self.assertEqual([answer.lhs for answer in answers], reverse)
                self.assertIs(type(answers[0].lhs), self.g)
                self.assertIs(type(answers[1].lhs), self.f)
                self.assert_zero(answers[0].rhs - (2*cos(self.t) - sin(self.t)))
                self.assert_zero(answers[1].rhs - (cos(self.t) + 2*sin(self.t)))
        with self.assertRaisesRegex(ValueError, "list of functions"):
            dsolve(equations, tuple(self.functions))

    def test_typed_time_and_fresh_constant_names_survive_the_native_boundary(self):
        for time in (Symbol("t", positive=True), Dummy("t"), Symbol("C1")):
            with self.subTest(time=time):
                functions = [self.f(time), self.g(time)]
                answers = dsolve(self.system([[0, 1], [-1, 0]], time), functions)
                values = self.assert_system(answers, [[0, 1], [-1, 0]], time, functions)
                constants = [Symbol("C2"), Symbol("C3")] if time.name == "C1" else [Symbol("C1"), Symbol("C2")]
                self.assertEqual([simplify(value.subs(time, 0)) for value in values], constants)
                self.assertEqual(set().union(*(value.free_symbols for value in values)), {time, *constants})
                for answer in answers:
                    self.assertIs(answer.lhs.args[0], time)

    def test_nonlinear_nonpolynomial_variable_and_singular_systems_refuse(self):
        t = self.t
        f, g = self.functions
        df, dg = Derivative(f, t), Derivative(g, t)
        cases = (
            [df - f*g, dg + f],
            [df - g - exp(t), dg + f],
            [df - t*g, dg + f],
            [df - Float(1.0)*g, dg + f],
            [Derivative(f, t, 2) - g, dg + f],
            [df + dg - f, 2*df + 2*dg - g],
        )
        for equations in cases:
            with self.subTest(equations=equations):
                with self.assertRaises(NotImplementedError):
                    dsolve(equations, self.functions)

    def test_original_undefined_zero_products_and_holes_refuse_before_lowering(self):
        f, g = self.functions
        df, dg = Derivative(f, self.t), Derivative(g, self.t)
        for undefined in (Pow(0, -1, evaluate=False), Pow(self.t, -1, evaluate=False)):
            source = Add(df - g, Mul(0, undefined, evaluate=False), evaluate=False)
            with self.subTest(source=source), patch.object(
                _native, "linear_ode_system_native", side_effect=AssertionError("invalid original source reached native solve"),
            ) as native:
                with self.assertRaises((ValueError, NotImplementedError)):
                    dsolve([source, dg + f], self.functions)
                self.assertEqual(native.call_count, 0)

    def test_foreign_typed_time_functions_and_boundary_data_refuse(self):
        equations = self.system([[0, 1], [-1, 0]])
        other = Function("other_system_function")
        for functions in ([self.functions[0], other(self.t)], [self.functions[0], self.functions[0]],
                          [self.f(self.t), self.g(Symbol("t", real=True))]):
            with self.subTest(functions=functions):
                with self.assertRaises(ValueError):
                    dsolve(equations, functions)
        for data in ({self.f(0): 1, self.g(1): 2}, {self.f(0): sqrt(2)},
                     {self.f(0): Float(1.0)}, {other(0): 1}, {self.f(0): 1, self.f(1): 2}):
            with self.subTest(data=data):
                with self.assertRaises((ValueError, NotImplementedError)):
                    dsolve(equations, ics=data)
        with self.assertRaises(NotImplementedError):
            dsolve(equations, hint="1st_linear")

    def test_custom_function_and_changed_scalar_getter_do_not_execute_hooks(self):
        observed = []

        class Custom(Function):
            @classmethod
            def eval(cls, argument):
                observed.append(argument)
                return None

        custom = Custom(self.t, evaluate=False)
        equations = [Eq(Derivative(custom, self.t), self.functions[1]),
                     Eq(Derivative(self.functions[1], self.t), custom)]
        with self.assertRaises(NotImplementedError):
            dsolve(equations)
        self.assertEqual(observed, [])
        equations = self.system([[0, 2], [-1, 0]])

        def changed(_):
            observed.append("coefficient getter")
            return 17

        with patch.object(Integer, "p", property(changed)), patch.object(
            _native, "linear_ode_system_native", side_effect=AssertionError("changed source reached native solve"),
        ):
            with self.assertRaises(NotImplementedError):
                dsolve(equations)
        self.assertEqual(observed, [])

    def test_native_operational_refusals_are_terminal_and_fixed_bridge_requires_typed_time(self):
        equations = self.system([[0, 1], [-1, 0]])
        with patch.object(_native, "linear_ode_system_native", side_effect=ValueError("system verifier budget exhausted")), patch(
            "sympy.solvers.ode._dsolve_general", side_effect=AssertionError("system failure entered scalar fallback"),
        ):
            with self.assertRaisesRegex(ValueError, "system verifier budget exhausted"):
                dsolve(equations)
        with self.assertRaises(TypeError):
            _native.linear_ode_system_native(
                _native_expr(0), _native_expr(1), _native_expr(-1), _native_expr(0),
                "t", _native_expr(0), _native_expr(1), _native_expr(2),
            )

    def test_time_text_bound_is_checked_before_typed_native_conversion(self):
        for time in (Symbol("t" * 16_385), Symbol("t" * 16_385, real=True)):
            with self.subTest(typed=time.is_real):
                equations = self.system([[0, 1], [-1, 0]], time)
                self.assertIs(equations[0].lhs.args[0].args[0], time)
                with patch("sympy.solvers._ode_system._symbol_binding",
                           side_effect=AssertionError("oversized time reached conversion")):
                    with self.assertRaisesRegex(ValueError, "text bound"):
                        dsolve(equations)

    def test_long_symbolic_function_arguments_do_not_bypass_numeric_compilation_limits(self):
        time = Symbol("t" * 4097, real=True)
        node = self.f(time)
        self.assertIs(1*node, node)
        self.assertIs(node + 0, node)
        self.assertIs(node.args[0], time)
        self.assertEqual((2*node).free_symbols, {time})
        with self.assertRaisesRegex(ValueError, "4096-byte"):
            _native.CompiledFunction._validate_source(_native_expr(time))


if __name__ == "__main__":
    unittest.main()
