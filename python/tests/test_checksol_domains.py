"""Original-domain solution checking, including erased-source adversaries.

Expected pole verdicts follow exact substitution into the original reciprocal,
not the candidate normalizer. Common public cases were also checked in an
isolated upstream SymPy 1.14 process; unknown-domain cases are deliberately
conservative and do not claim complete checksol-profile conformance.
"""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Basic, Dummy, Eq, Expr, Function, Integer, Mul, Pow, Rational, S, Symbol,
    checksol, cos, log, sin,
)


class ChecksolOriginalDomainTests(unittest.TestCase):
    def setUp(self):
        self.x = Symbol("checksol_domain_x")
        self.y = Symbol("checksol_domain_y")

    def reciprocal(self, base):
        return Pow(base, -1, evaluate=False)

    def test_cancelled_quotient_keeps_its_original_hole(self):
        x = self.x
        quotient = Mul(x, self.reciprocal(x), evaluate=False)
        for source in (Eq(quotient, 1, evaluate=False),
                       Eq(1, quotient, evaluate=False),
                       Add(quotient, -1, evaluate=False)):
            with self.subTest(source=source):
                self.assertIs(checksol(source, x, 0), False)
                self.assertIs(checksol(source, x, 1), True)
                self.assertIs(checksol(source, x, Rational(-2, 3)), True)

    def test_identical_equation_sides_are_checked_before_subtraction(self):
        x = self.x
        reciprocal = self.reciprocal(x - 2)
        source = Eq(reciprocal, reciprocal, evaluate=False)
        self.assertIs(checksol(source, x, 2), False)
        self.assertIs(checksol(source, x, 3), True)
        self.assertIsNone(checksol(source, {}))

    def test_zero_products_and_zero_powers_do_not_hide_poles(self):
        x = self.x
        pole = self.reciprocal(x)
        sources = (
            Mul(0, pole, evaluate=False),
            Eq(Pow(pole, 0, evaluate=False), 1, evaluate=False),
            Eq(Mul(0, sin(pole, evaluate=False), evaluate=False), 0, evaluate=False),
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertIs(checksol(source, x, 0), False)
                self.assertIs(checksol(source, x, 1), True)

    def test_nested_poles_survive_outer_zero(self):
        x = self.x
        inner = self.reciprocal(x)
        nested = self.reciprocal(Add(inner, -1, evaluate=False))
        source = Mul(0, nested, evaluate=False)
        self.assertIs(checksol(source, x, 0), False)
        self.assertIs(checksol(source, x, 1), False)
        self.assertIs(checksol(source, x, 2), True)

    def test_original_function_domain_cannot_cancel_between_sides(self):
        x = self.x
        function = log(x, evaluate=False)
        source = Eq(function, function, evaluate=False)
        self.assertIs(checksol(source, x, 0), False)
        self.assertIs(checksol(source, x, 1), True)
        literal = log(0, evaluate=False)
        self.assertIs(checksol(Eq(literal, literal, evaluate=False), {}), False)

    def test_fractional_power_zeros_and_poles_are_distinct(self):
        x = self.x
        positive = Pow(x, Rational(1, 2), evaluate=False)
        negative = Pow(x, Rational(-1, 2), evaluate=False)
        self.assertIs(checksol(Eq(positive, 0, evaluate=False), x, 0), True)
        self.assertIs(checksol(Eq(negative, negative, evaluate=False), x, 0), False)
        self.assertIs(checksol(Eq(negative, negative, evaluate=False), x, 4), True)

    def test_unresolved_reciprocal_domain_stays_inconclusive(self):
        x, y = self.x, self.y
        source = Eq(self.reciprocal(x), self.reciprocal(x), evaluate=False)
        self.assertIsNone(checksol(source, x, y))
        nonzero = Symbol("checksol_nonzero", nonzero=True)
        self.assertIs(checksol(source, x, nonzero), True)

    def test_typed_same_name_symbols_and_dummies_remain_distinct(self):
        x = Symbol("checksol_same", positive=True)
        other = Symbol("checksol_same", negative=True)
        dummy = Dummy("checksol_same")
        for source_symbol in (x, dummy):
            source = Eq(self.reciprocal(source_symbol), self.reciprocal(source_symbol), evaluate=False)
            with self.subTest(source_symbol=source_symbol):
                self.assertIs(checksol(source, source_symbol, 0), False)
                self.assertIsNot(checksol(source, other, 0), False)
        self.assertIs(checksol(x - other, {x: 3, other: 3}), True)

    def test_candidate_source_is_checked_before_it_disappears(self):
        x, y = self.x, self.y
        candidate = Mul(y, self.reciprocal(y), evaluate=False)
        self.assertIs(checksol(x - 1, {x: candidate, y: 0}), False)
        self.assertIs(checksol(x - 1, {x: candidate, y: 2}), True)
        self.assertIsNone(checksol(x - 1, {x: candidate}))

    def test_nonfinite_unused_bindings_and_numeric_equations_keep_contract(self):
        x, y = self.x, self.y
        for bad in (S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
            with self.subTest(candidate=bad):
                self.assertIs(checksol(x - 1, {x: 1, y: bad}), False)
                self.assertIs(checksol(0, {x: bad}), True)
                self.assertIs(checksol(1, {x: bad}), False)
        for unused in (object(), 2**5000):
            self.assertIs(checksol(0, {x: unused}), True)
            self.assertIs(checksol(1, {x: unused}), False)

    def test_standard_constants_and_empty_mapping_identity_remain_conclusive(self):
        from sympy import E, I, pi

        x = self.x
        for constant in (E, I, pi):
            source = Eq(self.reciprocal(x), self.reciprocal(x), evaluate=False)
            with self.subTest(constant=constant):
                self.assertIs(checksol(source, x, constant), True)
        self.assertIs(checksol(x*x + x - x*(x + 1), {}), True)

    def test_existing_residual_comparison_and_fuzzy_conjunction(self):
        x, y = self.x, self.y
        self.assertIs(checksol(x*x - 9, x, 3), True)
        self.assertIs(checksol(x*x - 9, x, 4), False)
        self.assertIs(checksol(sin(x), x, 0), True)
        self.assertIsNone(checksol(sin(y), x, 1))
        self.assertIs(checksol(y, x, 1), False)
        self.assertIs(checksol([x - 1, 2*x - 2], x, 1), True)
        self.assertIsNone(checksol([sin(y), x - 1], x, 1))
        self.assertIs(checksol([sin(y), x - 2], x, 1), False)
        self.assertIs(checksol(sin(x)**2 + cos(x)**2 - 1, x, 0), True)

    def test_custom_getters_and_substitution_hooks_are_not_native_proofs(self):
        calls = []

        class Custom(Function):
            def _eval_subs(self, old, new):
                calls.append("subs")
                return Integer(0)

        source = Custom(self.x, evaluate=False)

        def forbidden_args(self):
            calls.append("args")
            raise AssertionError("unadmitted custom getter")

        with patch.object(Custom, "args", property(forbidden_args)):
            self.assertIsNone(checksol(source, self.x, 0))

        class PretendSymbol:
            @property
            def __class__(self):
                calls.append("class")
                return Symbol

        self.assertIsNone(checksol(PretendSymbol(), self.x, 0))
        self.assertIsNone(checksol(self.x, self.x, PretendSymbol()))
        self.assertEqual(calls, [])

    def test_modified_builtin_substitution_is_not_invoked(self):
        for owner, source in ((Basic, self.x - 1), (Function, sin(self.x)),
                              (sin, sin(self.x))):
            with self.subTest(owner=owner):
                with patch.object(owner, "subs", side_effect=AssertionError("modified substitution")):
                    self.assertIsNone(checksol(source, self.x, 1))

    def test_modified_equation_subtraction_cannot_invent_a_zero_residual(self):
        source = Eq(self.x, 1, evaluate=False)
        with patch.object(Expr, "__sub__", return_value=Integer(0)) as changed:
            self.assertIsNone(checksol(source, self.x, 2))
            changed.assert_not_called()

    def test_excessive_rules_are_inconclusive_before_substitution(self):
        mapping = {Symbol("checksol_extra_%d" % i): i for i in range(65)}
        self.assertIsNone(checksol(self.x, mapping))

    def test_large_substituted_exponent_refuses_before_materialization(self):
        x, y = self.x, self.y
        source = Pow(Integer(2), x, evaluate=False)
        with patch.object(Basic, "subs", side_effect=AssertionError("modified substitution")):
            self.assertIsNone(checksol(source, x, 4097))
        # The unpatched path must reject the final exponent itself too.
        self.assertIsNone(checksol(source, {x: 4097, y: 0}))
        # A later rule cannot excuse the cost of an earlier exact power.
        self.assertIsNone(checksol(source, {x: 4097, Integer(4097): 0}))

    def test_nested_conjunctions_share_bounds_and_do_not_recurse_on_cycles(self):
        x = self.x
        cyclic = []
        cyclic.append(cyclic)
        self.assertIsNone(checksol(cyclic, x, 1))
        self.assertIs(checksol([cyclic, x - 2], x, 1), False)
        self.assertIs(checksol([x - 2, []], x, 1), False)
        with self.assertRaises(ValueError):
            checksol([x - 1, []], x, 1)
        self.assertIs(checksol([[x - 1], (2*x - 2,)], x, 1), True)
        nested = x - 1
        for _ in range(129):
            nested = [nested]
        self.assertIsNone(checksol(nested, x, 1))
        source = Add(*([x] * 512), -512, evaluate=False)
        self.assertIs(checksol(source, x, 1), True)
        self.assertIsNone(checksol([source] * 64, x, 1))

    def test_scalar_inputs_are_bounded_before_native_conversion(self):
        x = self.x
        self.assertIs(checksol(x - 1, x, "1"), True)
        self.assertIs(checksol("1", x, 0), False)
        self.assertIs(checksol("checksol_domain_x", x, 0), True)
        for value in (1 << 4096, "9" * 1235, "x" * 8193, "2**1000000000"):
            with self.subTest(value_type=type(value)):
                with patch("sympy.solvers._checksol._native_expr",
                           side_effect=AssertionError("unadmitted native conversion")):
                    self.assertIsNone(checksol(value, x, 0))
                    self.assertIsNone(checksol(x, x, value))


if __name__ == "__main__":
    unittest.main()
