"""Variable separation observations pinned to isolated SymPy 1.14.0.

Public cases use exact types and argument trees on both implementations.
Candidate-only cases record bounded admission and preservation of original
domains; they are not presented as oracle-compatible results for held holes.
"""

from fractions import Fraction
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import sympy as s


class SeparateVarsTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.z = s.symbols("x y z")

    def same(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual, expected)
        if type(actual) is dict:
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.same(actual[key], expected[key])
        elif isinstance(actual, s.Basic):
            self.assertEqual(len(actual.args), len(expected.args))
            for left, right in zip(actual.args, expected.args):
                self.same(left, right)

    def expected(self, mapping):
        return {key: s.sympify(value) for key, value in mapping.items()}

    def test_dictionary_returns_variable_factors_and_exact_coefficient(self):
        x, y = self.x, self.y
        self.same(s.separatevars(2*x*x*y, symbols=(x, y), dict=True),
                  self.expected({x: x*x, y: y, "coeff": 2}))

    def test_coupled_sum_is_factored_before_separation(self):
        x, y = self.x, self.y
        source = x*y + x*y*y
        self.same(s.separatevars(source), x*y*(y + 1))
        self.same(s.separatevars(source, symbols=(x, y), dict=True),
                  self.expected({x: x, y: y*(y + 1), "coeff": 1}))

    def test_full_multivariate_polynomial_product_is_recovered(self):
        x, y = self.x, self.y
        source = x*y + x + y + 1
        self.same(s.separatevars(source), (x + 1)*(y + 1))
        self.same(s.separatevars(source, dict=True),
                  self.expected({x: x + 1, y: y + 1, "coeff": 1}))

    def test_repeated_factors_and_rational_content_are_preserved(self):
        x, y = self.x, self.y
        source = s.Rational(2, 3)*x*x*y + s.Rational(4, 3)*x*y + s.Rational(2, 3)*y
        self.same(s.separatevars(source), s.Rational(2, 3)*y*(x + 1)**2)
        self.same(s.separatevars(source, dict=True),
                  self.expected({x: (x + 1)**2, y: y, "coeff": s.Rational(2, 3)}))

    def test_factoring_a_coupled_sum_can_split_univariate_polynomials(self):
        x, y = self.x, self.y
        source = x*x*y - y
        self.same(s.separatevars(source, dict=True),
                  self.expected({x: (x - 1)*(x + 1), y: y, "coeff": 1}))

    def test_already_separated_univariate_factors_are_not_refactored(self):
        x, y = self.x, self.y
        source = (x*x - 1)*(y*y - 1)
        self.assertIs(s.separatevars(source), source)
        self.same(s.separatevars(source, dict=True),
                  self.expected({x: x*x - 1, y: y*y - 1, "coeff": 1}))

    def test_univariate_held_expression_is_returned_unchanged(self):
        x = self.x
        source = s.Add(x, x, evaluate=False)
        self.assertIs(s.separatevars(source), source)

    def test_coupled_factors_return_none_in_dictionary_mode(self):
        x, y = self.x, self.y
        for source in (x + y, s.sin(x + y), (x + y)*s.exp(x)):
            with self.subTest(source=source):
                self.assertIsNone(s.separatevars(source, symbols=(x, y), dict=True))

    def test_selected_axes_keep_unrequested_symbols_in_their_factor(self):
        x, y, z = self.x, self.y, self.z
        self.same(s.separatevars((x + z)*y, symbols=(x, y), dict=True),
                  self.expected({x: x + z, y: y, "coeff": 1}))
        self.same(s.separatevars(x*y + x*z, symbols=(x,), dict=True),
                  self.expected({x: x, "coeff": y + z}))
        self.assertIsNone(s.separatevars(x*y + x*z, dict=True))

    def test_unrequested_factors_go_to_the_coefficient(self):
        x, y, z = self.x, self.y, self.z
        self.same(s.separatevars(2*x*y*s.sin(z), symbols=(x,), dict=True),
                  self.expected({x: x, "coeff": 2*y*s.sin(z)}))

    def test_explicit_none_differs_from_inferred_axes(self):
        x, y = self.x, self.y
        self.same(s.separatevars(x*y, symbols=None, dict=True), {"coeff": x*y})
        expected = self.expected({x: x, y: y, "coeff": 1})
        self.same(s.separatevars(x*y, dict=True), expected)
        self.same(s.separatevars(x*y, symbols=(), dict=True), expected)
        self.same(s.separatevars(x*y, symbols=[], dict=True), expected)

    def test_absent_and_duplicate_requested_axes(self):
        x, y, z = self.x, self.y, self.z
        result = s.separatevars(x*y, symbols=[z, y, x, y], dict=True)
        self.same(result, self.expected({z: 1, y: y, x: x, "coeff": 1}))
        self.assertEqual(tuple(result), (z, y, x, "coeff"))

    def test_zero_and_constants_with_explicit_or_inferred_axes(self):
        x, y = self.x, self.y
        for value in (0, 2, s.Rational(2, 3)):
            self.same(s.separatevars(value), s.sympify(value))
            self.assertIsNone(s.separatevars(value, dict=True))
            self.same(s.separatevars(value, symbols=None, dict=True), self.expected({"coeff": value}))
            self.same(s.separatevars(value, symbols=(x, y), dict=True),
                      self.expected({x: 1, y: 1, "coeff": value}))

    def test_atomic_constant_axes_are_allowed(self):
        x, y = self.x, self.y
        for axis in (s.Integer(2), s.pi):
            self.same(s.separatevars(x*y, symbols=(axis,), dict=True),
                      self.expected({axis: 1, "coeff": x*y}))

    def test_parameter_errors_for_nonatomic_and_scalar_axes(self):
        x, y = self.x, self.y
        with self.assertRaises(ValueError):
            s.separatevars(x*y, symbols=(x + y,), dict=True)
        with self.assertRaises(TypeError):
            s.separatevars(x*y, symbols=x, dict=True)
        with self.assertRaises(AttributeError):
            s.separatevars(x*y, symbols=("x",), dict=True)

    def test_expression_mode_ignores_axis_selection(self):
        x, y = self.x, self.y
        self.same(s.separatevars(x*y + x, symbols=x), x*(y + 1))

    def test_string_and_exact_fraction_inputs(self):
        x, y = self.x, self.y
        self.same(s.separatevars("x*y", dict=True), self.expected({x: x, y: y, "coeff": 1}))
        self.same(s.separatevars(Fraction(2, 3), symbols=(x,), dict=True),
                  self.expected({x: 1, "coeff": s.Rational(2, 3)}))

    def test_common_factors_around_elementary_functions(self):
        x, y = self.x, self.y
        source = x*s.sin(y) + x*s.cos(y)
        self.same(s.separatevars(source), x*(s.sin(y) + s.cos(y)))
        self.same(s.separatevars(source, dict=True),
                  self.expected({x: x, y: s.sin(y) + s.cos(y), "coeff": 1}))

    def test_separated_function_factors_remain_intact(self):
        x, y = self.x, self.y
        source = s.sin(x)*s.exp(y)
        self.assertIs(s.separatevars(source), source)
        self.same(s.separatevars(source, dict=True),
                  self.expected({x: s.sin(x), y: s.exp(y), "coeff": 1}))

    def test_exponential_sum_and_absolute_product_separate(self):
        x, y = self.x, self.y
        self.same(s.separatevars(s.exp(x + y), dict=True),
                  self.expected({x: s.exp(x), y: s.exp(y), "coeff": 1}))
        self.same(s.separatevars(s.Abs(x*y), dict=True),
                  self.expected({x: s.Abs(x), y: s.Abs(y), "coeff": 1}))

    def test_integer_exponents_split_product_bases(self):
        x, y = self.x, self.y
        n = s.Symbol("n", integer=True)
        source = s.Pow(x*y, n, evaluate=False)
        self.same(s.separatevars(source), x**n*y**n)
        self.same(s.separatevars(source, symbols=(x, y), dict=True),
                  self.expected({x: x**n, y: y**n, "coeff": 1}))

    def test_unknown_root_branches_require_force_or_positive_assumptions(self):
        x, y = self.x, self.y
        source = s.sqrt(x*y)
        self.same(s.separatevars(source), source)
        self.assertIsNone(s.separatevars(source, dict=True))
        self.same(s.separatevars(source, dict=True, force=True),
                  self.expected({x: s.sqrt(x), y: s.sqrt(y), "coeff": 1}))
        p = s.Symbol("p", positive=True)
        held = s.Pow(s.Mul(p, x, evaluate=False), s.Rational(1, 2), evaluate=False)
        self.same(s.separatevars(held), s.sqrt(p)*s.sqrt(x))

    def test_explicit_force_for_symbolic_product_powers(self):
        x, y, z = self.x, self.y, self.z
        source = (x*y)**z
        self.same(s.separatevars(source, symbols=(x, y), dict=True, force=True),
                  self.expected({x: x**z, y: y**z, "coeff": 1}))
        self.assertIsNone(s.separatevars(source, dict=True, force=True))

    def test_log_product_expansion_respects_force(self):
        x, y = self.x, self.y
        source = s.log(x*y)
        self.same(s.separatevars(source), source)
        self.same(s.separatevars(source, force=True), s.log(x) + s.log(y))
        self.assertIsNone(s.separatevars(source, dict=True, force=True))

    def test_known_positive_log_factor_can_be_separated_without_force(self):
        x = self.x
        positive = s.Symbol("positive", positive=True)
        source = s.log(positive*x, evaluate=False)
        self.same(s.separatevars(source), s.log(positive) + s.log(x))

    def test_separated_factors_connect_to_substitution_and_numeric_compilation(self):
        x, y = self.x, self.y
        source = x*y + x + y + 1
        result = s.separatevars(source, symbols=(x, y), dict=True)
        reconstructed = result["coeff"]*result[x]*result[y]
        self.same(reconstructed, (x + 1)*(y + 1))
        self.same(reconstructed.subs(x, -1), s.Integer(0))
        function = s.lambdify((x, y), reconstructed, modules="math")
        self.assertEqual([function(a, b) for a, b in ((-1, 2), (0, 3), (2, 4))], [0, 4, 15])

    def test_existing_reciprocal_sum_keeps_its_split_denominator_shape(self):
        x, y = self.x, self.y
        source = 1/x + 1/y
        expected = s.Mul(s.Pow(x, -1), s.Pow(y, -1), x + y, evaluate=False)
        self.same(s.separatevars(source), expected)
        self.assertIsNone(s.separatevars(source, dict=True))

    def test_typed_same_name_axes_and_coeff_symbol_remain_distinct(self):
        x = s.Symbol("same")
        positive = s.Symbol("same", positive=True)
        named = s.Symbol("coeff")
        source = x*positive + x + positive + 1
        result = s.separatevars(source, symbols=(positive, x, named), dict=True)
        self.same(result, self.expected({positive: positive + 1, x: x + 1, named: 1, "coeff": 1}))
        self.assertIs(next(iter(result)), positive)

    def test_same_name_dummies_survive_native_factoring(self):
        first, second = s.Dummy("same"), s.Dummy("same")
        source = first*second + first + second + 1
        result = s.separatevars(source, symbols=(first, second), dict=True)
        self.same(result, self.expected({first: first + 1, second: second + 1, "coeff": 1}))
        self.assertIs(tuple(result)[0], first)
        self.assertIs(tuple(result)[1], second)

    def test_held_reciprocal_product_is_unchanged_in_expression_mode(self):
        x, y = self.x, self.y
        source = s.Mul(x, s.Pow(x, -1, evaluate=False), y, evaluate=False)
        self.assertIs(s.separatevars(source), source)


@unittest.skipUnless(hasattr(s.core, "_native"), "candidate admission/domain evidence")
class SeparateVarsBoundaryTests(unittest.TestCase):
    def test_held_reciprocal_cancellation_is_not_published_in_dictionary_mode(self):
        x, y = s.symbols("separate_domain_x separate_domain_y")
        source = s.Mul(x, s.Pow(x, -1, evaluate=False), y, evaluate=False)
        original = source.args
        with self.assertRaisesRegex(NotImplementedError, "original domain"):
            s.separatevars(source, symbols=(x, y), dict=True)
        self.assertEqual(source.args, original)

    def test_symbolic_integer_power_cancellation_preserves_its_possible_pole(self):
        x, y = s.symbols("separate_symbolic_x separate_symbolic_y")
        n = s.Symbol("separate_exponent", integer=True)
        source = s.Mul(s.Pow(x, n, evaluate=False), s.Pow(x, -n, evaluate=False), y, evaluate=False)
        with self.assertRaisesRegex(NotImplementedError, "original domain"):
            s.separatevars(source, symbols=(x, y), dict=True)

    def test_held_undefined_function_cannot_disappear_into_a_zero_coefficient(self):
        x, y = s.symbols("separate_function_x separate_function_y")
        source = s.Mul(0, s.gamma(0, evaluate=False), x, y, evaluate=False)
        with self.assertRaisesRegex(NotImplementedError, "original domain"):
            s.separatevars(source, symbols=(x, y), dict=True)

    def test_held_zero_and_undefined_powers_are_explicit_refusals(self):
        x, y = s.symbols("separate_held_x separate_held_y")
        for source in (s.Pow(x*y, 0, evaluate=False),
                       s.Mul(s.Pow(0, -1, evaluate=False), x, y, evaluate=False)):
            with self.assertRaises(NotImplementedError):
                s.separatevars(source, dict=True)

    def test_coupled_original_denominators_stay_outside_the_factor_route(self):
        x, y = s.symbols("separate_inverse_x separate_inverse_y")
        source = s.Pow(x*y + x + y + 1, -1, evaluate=False)
        with self.assertRaisesRegex(NotImplementedError, "denominator domains"):
            s.separatevars(source, dict=True)

    def test_noncommuting_symbols_are_refused_before_factorization(self):
        module = importlib.import_module("sympy.simplify._separatevars")
        x, y = s.symbols("separate_nc_x separate_nc_y", commutative=False)
        source = s.Add(s.Mul(x, y, evaluate=False), s.Mul(y, x, evaluate=False), evaluate=False)
        with patch.object(module._Polynomial, "factors", side_effect=AssertionError("factor route reached")):
            with self.assertRaisesRegex(NotImplementedError, "commuting scalar"):
                s.separatevars(source)

    def test_nonnative_source_hooks_are_refused_without_execution(self):
        calls = []

        class Custom(s.Function):
            @classmethod
            def eval(cls, *arguments):
                calls.append("eval")

        class Converted:
            def _sympy_(self):
                calls.append("convert")
                return s.Integer(1)

        x, y = s.symbols("separate_custom_x separate_custom_y")
        for source in (Custom(x, y, evaluate=False), Converted()):
            with self.assertRaises(TypeError):
                s.separatevars(source, dict=True)
        self.assertEqual(calls, [])

    def test_effectful_parameter_protocols_are_refused(self):
        calls = []

        class Mode:
            def __bool__(self):
                calls.append("truth")
                return True

        class Axes:
            def __iter__(self):
                calls.append("iterate")
                return iter(())

        x, y = s.symbols("separate_parameters_x separate_parameters_y")
        for kwargs in ({"dict": Mode()}, {"force": Mode()}, {"dict": True, "symbols": Axes()}):
            with self.assertRaises(TypeError):
                s.separatevars(x*y, **kwargs)
        self.assertEqual(calls, [])

    def test_unsupported_forced_power_grammar_refuses_explicitly(self):
        x, y = s.symbols("separate_force_x separate_force_y")
        with self.assertRaisesRegex(NotImplementedError, "force supports"):
            s.separatevars(s.sqrt(s.sin(x)*y), force=True)

    def test_polynomial_growth_refuses_before_the_native_factor_generator(self):
        module = importlib.import_module("sympy.simplify._separatevars")
        x, y, z = s.symbols("separate_limit_x separate_limit_y separate_limit_z")
        sources = (x**17 + y, (x + y)**8 + z)
        with patch.object(module._Polynomial, "factors", side_effect=AssertionError("factor route reached")):
            for source in sources:
                with self.assertRaisesRegex(ValueError, "limit"):
                    s.separatevars(source)

    def test_combined_rational_coefficient_growth_is_checked_before_factoring(self):
        module = importlib.import_module("sympy.simplify._separatevars")
        x, y = s.symbols("separate_bits_x separate_bits_y")
        denominators = [2**(2**i) + 1 for i in range(8)] + [7]
        source = s.Add(*(s.Mul(s.Rational(1, denominator), x, y, evaluate=False)
                         for denominator in denominators), evaluate=False)
        with patch.object(module._Polynomial, "factors", side_effect=AssertionError("factor route reached")):
            with self.assertRaisesRegex(ValueError, "coefficients.*bit limit"):
                s.separatevars(source)

    def test_injected_scalar_owner_is_refused_without_running_it(self):
        x, y = s.symbols("separate_owner_x separate_owner_y")
        source = x*y + x + y + 1
        calls = []

        def changed(*args, **kwargs):
            calls.append("effect")
            raise AssertionError("changed scalar owner ran")

        with patch.object(s.Symbol, "__hash__", changed):
            with self.assertRaisesRegex(TypeError, "original built-in"):
                s.separatevars(source)
        self.assertEqual(calls, [])

    def test_native_factor_replacement_before_shell_import_is_not_admitted(self):
        native_directory = str(Path(s.core._native.__file__).parent)
        python_directory = str(Path(s.__file__).resolve().parents[1])
        script = """
import sys
sys.path[:0] = [%r, %r]
import fsym_python as native
calls = []
def replacement(*args):
    calls.append('called')
    raise AssertionError('replacement factor interface ran')
native.poly_factor_multivariate_native = replacement
import sympy as s
x, y = s.symbols('x y')
try:
    s.separatevars(x*y + x + y + 1)
except NotImplementedError as error:
    assert 'native factor interfaces' in str(error)
else:
    raise AssertionError('replacement factor interface was admitted')
assert calls == []
""" % (native_directory, python_directory)
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
