"""Elementary forcing through the actual bounded native ODE system bridge.

Hand-derived resonant IVPs, original derivative coordinates and hostile held
sources exercise the public source boundary and the checked composition.
"""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Derivative, Dummy, Eq, Float, Function, Integer, Mul, Pow, Rational,
    Symbol, cos, diff, dsolve, exp, expand, log, simplify, sin,
)
from sympy.core import (
    AppliedUndef, UndefinedFunction, _native, _native_expr, _symbol_binding,
)


class ElementaryForcedOdeSystemTests(unittest.TestCase):
    def setUp(self):
        self.t = Symbol("t")
        self.owners = [Function(f"elementary_system_{index}") for index in range(8)]

    def equations(self, matrix, forcing, time=None, owners=None):
        time = self.t if time is None else time
        owners = self.owners[:len(matrix)] if owners is None else owners
        functions = [owner(time) for owner in owners]
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

    def test_exponential_resonance_preserves_the_initial_coordinate(self):
        t = self.t
        matrix, forcing = [[2]], [exp(2*t)]
        equations, functions = self.equations(matrix, forcing)
        values = self.assert_system(
            dsolve(equations, functions, ics={self.owners[0](0): 3}),
            matrix, forcing, functions,
        )
        self.assert_zero(values[0] - (3 + t)*exp(2*t))
        self.assertEqual(simplify(values[0].subs(t, 0)), 3)
        self.assertEqual(values[0].free_symbols, {t})

    def test_coupled_sine_resonance_has_no_auxiliary_integration_constants(self):
        t = self.t
        matrix, forcing = [[0, 1], [-1, 0]], [0, sin(t)]
        equations, functions = self.equations(matrix, forcing)
        initial = {owner(0): 0 for owner in self.owners[:2]}
        values = self.assert_system(
            dsolve(equations, functions, ics=initial), matrix, forcing, functions,
        )
        expected = [(sin(t) - t*cos(t))/2, t*sin(t)/2]
        for value, wanted in zip(values, expected):
            self.assert_zero(value - wanted)
            self.assertEqual(simplify(value.subs(t, 0)), 0)
        self.assertEqual(set().union(*(value.free_symbols for value in values)), {t})

    def test_polynomial_times_exponential_retains_absolute_time_powers(self):
        t = self.t
        matrix, forcing = [[0]], [t**2*exp(t)]
        equations, functions = self.equations(matrix, forcing)
        values = self.assert_system(
            dsolve(equations, functions, ics={self.owners[0](0): 0}),
            matrix, forcing, functions,
        )
        self.assert_zero(values[0] - ((t**2 - 2*t + 2)*exp(t) - 2))

    def test_nonzero_initial_time_and_nonzero_matrix_require_the_fundamental_correction(self):
        t, point = self.t, Rational(2)
        matrix, forcing = [[2]], [exp(t)]
        equations, functions = self.equations(matrix, forcing)
        values = self.assert_system(
            dsolve(equations, functions, ics={self.owners[0](point): 3}),
            matrix, forcing, functions,
        )
        self.assertEqual(simplify(values[0].subs(t, point)), 3)
        self.assertEqual(values[0].free_symbols, {t})

    def test_mixed_polynomial_and_elementary_loads_at_a_rational_initial_time(self):
        t, point = self.t, Rational(-2, 3)
        matrix, forcing = [[0]], [t**3 + 2*exp(t) + 3*cos(2*t)]
        equations, functions = self.equations(matrix, forcing)
        values = self.assert_system(
            dsolve(equations, functions, ics={self.owners[0](point): Rational(7, 5)}),
            matrix, forcing, functions,
        )
        self.assertEqual(simplify(values[0].subs(t, point)), Rational(7, 5))

    def test_damped_oscillation_and_negative_rational_frequency(self):
        t = self.t
        cases = (
            (exp(-t)*sin(2*t), (2 - exp(-t)*(sin(2*t) + 2*cos(2*t)))/5),
            (sin(-t/2), 2*cos(t/2) - 2),
            (exp(t/2), 2*exp(t/2) - 2),
        )
        for load, expected in cases:
            with self.subTest(load=load):
                equations, functions = self.equations([[0]], [load])
                values = self.assert_system(
                    dsolve(equations, functions, ics={self.owners[0](0): 0}),
                    [[0]], [load], functions,
                )
                self.assert_zero(values[0] - expected)

    def test_elementary_products_use_exact_sum_and_difference_frequencies(self):
        t = self.t
        # Hand-derived product-to-sum identities test each sign and parity.
        # Check the resulting differential identity in the independent harmonic
        # basis without asking the simplifier to prove the same trig identity.
        cases = (
            (Mul(cos(2*t), cos(t), evaluate=False), (cos(3*t) + cos(t))/2),
            (Mul(sin(2*t), sin(t), evaluate=False), (cos(t) - cos(3*t))/2),
            (Mul(sin(2*t), cos(t), evaluate=False), (sin(3*t) + sin(t))/2),
            (Mul(cos(2*t), sin(t), evaluate=False), (sin(3*t) - sin(t))/2),
            (Mul(exp(t), exp(2*t), evaluate=False), exp(3*t)),
            (Pow(cos(t), 2, evaluate=False), (1 + cos(2*t))/2),
        )
        for original, harmonic in cases:
            with self.subTest(original=original):
                equations, functions = self.equations([[0]], [original])
                values = self.assert_system(
                    dsolve(equations, functions, ics={self.owners[0](0): 0}),
                    [[0]], [harmonic], functions,
                )
                self.assertEqual(simplify(values[0].subs(t, 0)), 0)

    def test_nondiagonal_derivative_matrix_rechecks_mixed_forcing_against_original_rows(self):
        t = self.t
        f, g = [owner(t) for owner in self.owners[:2]]
        df, dg = Derivative(f, t), Derivative(g, t)
        # M = [[0, 2], [1, 1]] requires a pivot swap.
        sources = (
            [Eq(2*dg, -2*f + 2*t + 2*sin(t)),
             Eq(df + dg, g - f + exp(t) + t + sin(t))],
            [2*dg + 2*f - 2*t - 2*sin(t),
             df + dg - g + f - exp(t) - t - sin(t)],
        )
        matrix, forcing = [[0, 1], [-1, 0]], [exp(t), t + sin(t)]
        initial = {owner(0): value for owner, value in zip(self.owners, (2, -1))}
        for source in sources:
            with self.subTest(equality_form=type(source[0]) is Eq):
                values = self.assert_system(
                    dsolve(source, [f, g], ics=initial), matrix, forcing, [f, g],
                )
                self.assertEqual([simplify(value.subs(t, 0)) for value in values], [2, -1])
                self.assert_zero(2*diff(values[1], t) + 2*values[0] - 2*t - 2*sin(t))
                self.assert_zero(
                    diff(values[0], t) + diff(values[1], t)
                    - values[1] + values[0] - exp(t) - t - sin(t)
                )

    def test_partial_initial_data_keeps_only_the_original_free_coordinate(self):
        t, point, c2 = self.t, Rational(1), Symbol("C2")
        matrix, forcing = [[0, 1], [0, 0]], [sin(t), exp(t)]
        equations, functions = self.equations(matrix, forcing)
        values = self.assert_system(
            dsolve(equations, functions, ics={self.owners[0](point): 2}),
            matrix, forcing, functions,
        )
        self.assertEqual([simplify(value.subs(t, point)) for value in values], [2, c2])
        self.assertEqual(set().union(*(value.free_symbols for value in values)), {t, c2})

    def test_typed_time_and_constant_name_collisions_preserve_original_owners(self):
        cases = (
            (Symbol("t", positive=True), self.owners[0], "C1"),
            (Dummy("t"), self.owners[0], "C1"),
            (Symbol("C1"), Function("C2"), "C3"),
        )
        for time, owner, constant_name in cases:
            with self.subTest(time=time):
                equations, functions = self.equations([[0]], [exp(time)], time=time, owners=[owner])
                answers = dsolve(equations, functions)
                values = self.assert_system(
                    answers, [[0]], [exp(time)], functions, time=time,
                )
                constant = Symbol(constant_name)
                self.assertEqual(simplify(values[0].subs(time, 0)), constant)
                self.assertEqual(values[0].free_symbols, {time, constant})
                self.assertIs(answers[0].lhs.args[0], time)
                self.assertTrue(any(symbol is time for symbol in values[0].free_symbols))

    def test_same_named_function_owners_follow_inferred_and_explicit_coordinate_order(self):
        first = type.__new__(UndefinedFunction, "elementary_same", (AppliedUndef,), {})
        second = type.__new__(UndefinedFunction, "elementary_same", (AppliedUndef,), {})
        owners = [first, second]
        equations, functions = self.equations(
            [[0, 0], [0, 0]], [exp(self.t), sin(self.t)], owners=owners,
        )
        initial = {first(0): 1, second(0): 3}
        expected = [exp(self.t), 4 - cos(self.t)]
        cases = (
            ([equations[1], equations[0]], None, (1, 0)),
            (tuple(equations), [functions[1], functions[0]], (1, 0)),
        )
        for source, supplied, order in cases:
            with self.subTest(supplied=supplied):
                answers = dsolve(source, supplied, ics=initial)
                self.assertEqual(len(answers), 2)
                for answer, index in zip(answers, order):
                    self.assertIs(type(answer.lhs), owners[index])
                    self.assert_zero(answer.rhs - expected[index])

    def test_shared_forcing_state_admits_the_eight_coordinate_boundary(self):
        t, count = self.t, 7
        matrix, forcing = [[0]*count for _ in range(count)], [exp(t)]*count
        equations, functions = self.equations(matrix, forcing)
        initial = {owner(0): 0 for owner in self.owners[:count]}
        values = self.assert_system(
            dsolve(equations, functions, ics=initial), matrix, forcing, functions,
        )
        for value in values:
            self.assert_zero(value - exp(t) + 1)

    def test_original_held_partial_sources_refuse_before_zero_or_cancellation(self):
        t = self.t
        function = self.owners[0](t)
        derivative = Derivative(function, t)
        for partial in (
            Pow(sin(t), -1, evaluate=False),
            Pow(sin(t), 0, evaluate=False),
            Pow(exp(t), -1, evaluate=False),
            Pow(t, -1, evaluate=False),
            Pow(0, 0, evaluate=False),
            log(t),
        ):
            hidden = Mul(0, partial, evaluate=False)
            sources = (
                Eq(derivative, Add(exp(t), hidden, evaluate=False), evaluate=False),
                Eq(Add(derivative - exp(t), partial, evaluate=False), partial, evaluate=False),
            )
            for source in sources:
                with self.subTest(partial=partial), patch.object(
                    _native, "forced_ode_system_native",
                    side_effect=AssertionError("an original domain was erased"),
                ) as native:
                    with self.assertRaises((NotImplementedError, ValueError)):
                        dsolve([source], [function])
                    self.assertEqual(native.call_count, 0)

    def test_unsupported_phase_products_and_foreign_children_refuse_before_native(self):
        t = self.t
        function = self.owners[0](t)
        foreign = Symbol("t", positive=True)
        loads = (
            exp(t + 1, evaluate=False),
            sin(t*t, evaluate=False),
            exp(foreign, evaluate=False),
            Mul(exp(t), function, evaluate=False),
            Mul(exp(t), Float(1.0), evaluate=False),
            Mul(exp(t), log(t), evaluate=False),
        )
        with patch.object(
            _native, "forced_ode_system_native",
            side_effect=AssertionError("unsupported original source reached native"),
        ) as native:
            for load in loads:
                for source in (load, Add(exp(t), Mul(0, load, evaluate=False), evaluate=False)):
                    with self.subTest(load=load, source=source):
                        equation = Eq(Derivative(function, t), source, evaluate=False)
                        with self.assertRaises((NotImplementedError, ValueError)):
                            dsolve([equation], [function])
            self.assertEqual(native.call_count, 0)

    def test_changed_builtin_capabilities_refuse_without_invoking_the_mutation(self):
        observed = []
        original = exp(self.t, evaluate=False)
        equations, functions = self.equations([[0]], [original])

        def changed_eval(cls, argument):
            observed.append(argument)
            return None

        with patch.object(exp, "eval", classmethod(changed_eval), create=True), patch.object(
            _native, "forced_ode_system_native",
            side_effect=AssertionError("changed builtin implementation reached native"),
        ) as native:
            with self.assertRaises(NotImplementedError):
                dsolve(equations, functions)
            self.assertEqual(native.call_count, 0)
        self.assertEqual(observed, [])

    def test_name_only_and_subclass_elementary_impostors_never_define_forcing(self):
        observed = []

        class PretendExp(Function):
            @classmethod
            def eval(cls, argument):
                observed.append(argument)
                return None

        PretendExp.__name__ = "exp"

        class DerivedSin(sin):
            @classmethod
            def eval(cls, argument):
                observed.append(argument)
                return None

        for owner in (PretendExp, DerivedSin):
            load = owner(self.t, evaluate=False)
            equations, functions = self.equations([[0]], [load])
            with self.subTest(owner=owner), patch.object(
                _native, "forced_ode_system_native",
                side_effect=AssertionError("unregistered owner reached native"),
            ) as native:
                with self.assertRaises(NotImplementedError):
                    dsolve(equations, functions)
                self.assertEqual(native.call_count, 0)
        self.assertEqual(observed, [])

    def test_forcing_degree_coefficient_and_augmented_dimension_refusals_precede_native(self):
        t = self.t
        sources = (
            self.equations([[0]], [t**17*exp(t)]),
            self.equations([[0]], [Integer(1 << 1024)*exp(t)]),
            self.equations([[0]], [t**7*exp(t)]),
            self.equations([[0]*8 for _ in range(8)], [exp(t)]*8),
            self.equations([[0]], [sum(exp(index*t) for index in range(1, 10))]),
        )
        with patch.object(
            _native, "forced_ode_system_native",
            side_effect=AssertionError("unsupported forcing envelope reached native"),
        ) as native:
            for equations, functions in sources:
                with self.subTest(source=equations):
                    with self.assertRaises((NotImplementedError, ValueError)):
                        dsolve(equations, functions)
            self.assertEqual(native.call_count, 0)

    def test_native_failures_propagate_without_legacy_or_scalar_fallback(self):
        equations, functions = self.equations([[0]], [exp(self.t)])
        for message in ("forced verifier budget exhausted", "forced computation cancelled"):
            failure = ValueError(message)
            with self.subTest(message=message), patch.object(
                _native, "forced_ode_system_native", side_effect=failure,
            ) as native, patch.object(
                _native, "general_ode_system_native", side_effect=AssertionError("operational fallback"),
            ), patch.object(
                _native, "polynomial_ode_system_native", side_effect=AssertionError("operational fallback"),
            ), patch.object(
                _native, "linear_ode_system_native", side_effect=AssertionError("operational fallback"),
            ), patch(
                "sympy.solvers.ode._dsolve_general", side_effect=AssertionError("operational fallback"),
            ):
                with self.assertRaises(ValueError) as error:
                    dsolve(equations, functions)
                self.assertIs(error.exception, failure)
                self.assertEqual(native.call_count, 1)

    def test_native_mode_shape_admission_precedes_scalar_extraction(self):
        zero, one = _native_expr(0), _native_expr(1)
        time = _symbol_binding(self.t)
        matrix, polynomial, initial = [[zero]], [[zero]], [zero]
        with self.assertRaises(TypeError):
            _native.forced_ode_system_native(matrix, "t", zero, polynomial, initial, [])
        bad_modes = (
            [object()]*9,
            [[object()]*4],
            [[object(), object(), object(), [object(), object()], [object()]]],
            [[object(), object(), object(), [object()], []]],
        )
        for modes in bad_modes:
            with self.subTest(modes=modes):
                with self.assertRaises(ValueError):
                    _native.forced_ode_system_native(matrix, time, zero, polynomial, initial, modes)
        with self.assertRaises(ValueError):
            _native.forced_ode_system_native(
                [[object()]], time, zero, [[object()]*18], initial,
                [[zero, one, zero, [one], [zero]]],
            )

    def test_native_mode_scalars_require_bounded_exact_atoms(self):
        zero, one = _native_expr(0), _native_expr(1)
        time = _symbol_binding(self.t)
        matrix, polynomial, initial = [[zero]], [[zero]], [zero]
        for power in (_native_expr(-1), _native_expr(17), _native_expr(Rational(1, 2))):
            with self.subTest(power=power):
                with self.assertRaises(ValueError):
                    _native.forced_ode_system_native(
                        matrix, time, zero, polynomial, initial,
                        [[power, one, zero, [one], [zero]]],
                    )
        for power in (0, True, object()):
            with self.subTest(power=power):
                with self.assertRaises(TypeError):
                    _native.forced_ode_system_native(
                        matrix, time, zero, polynomial, initial,
                        [[power, one, zero, [one], [zero]]],
                    )
        for rate, frequency, cosine, sine in (
            (one, _native_expr(-1), zero, zero),
            (one, zero, one, one),
            (_native_expr(1 << 1024), zero, zero, zero),
        ):
            with self.subTest(rate=rate, frequency=frequency, sine=sine):
                with self.assertRaises(ValueError):
                    _native.forced_ode_system_native(
                        matrix, time, zero, polynomial, initial,
                        [[zero, rate, frequency, [cosine], [sine]]],
                    )


if __name__ == "__main__":
    unittest.main()
