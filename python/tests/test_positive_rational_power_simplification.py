"""Exact positive-base exponential simplification without branch assumptions."""

import importlib
import unittest
from unittest.mock import PropertyMock, patch

from sympy import Add, Dummy, Float, Integer, Mul, Pow, Rational, Symbol, simplify


_simplify_module = importlib.import_module("sympy.simplify.simplify")


class PositiveRationalPowerSimplificationTests(unittest.TestCase):
    def test_positive_integer_bases_cancel_for_unrestricted_symbols(self):
        # The identity holds for arbitrary complex symbols, not just integer
        # recurrence indices: positive bases have a real principal logarithm.
        for z in (Symbol("geometric_z"), Symbol("geometric_n", integer=True),
                  Dummy("geometric_z")):
            for slope, base in ((2, 4), (3, 8), (-1, Rational(1, 2)),
                                (-3, Rational(1, 8))):
                with self.subTest(symbol=z, slope=slope):
                    source = Pow(2, slope*z, evaluate=False) - Pow(base, z)
                    self.assertEqual(simplify(source), 0)

    def test_positive_rational_and_reciprocal_bases_cancel(self):
        z = Symbol("geometric_rational_z")
        cases = (
            (Rational(1, 2), 3, Rational(1, 8)),
            (Rational(1, 2), -2, Integer(4)),
            (Rational(2, 3), 2, Rational(4, 9)),
            (Rational(3, 2), -2, Rational(4, 9)),
            (Rational(2, 3), -1, Rational(3, 2)),
        )
        for base, slope, expected_base in cases:
            with self.subTest(base=base, slope=slope):
                source = Pow(base, slope*z, evaluate=False) - Pow(expected_base, z)
                self.assertEqual(simplify(source), 0)

    def test_same_spelling_symbols_and_dummies_keep_distinct_identities(self):
        pairs = (
            (Symbol("geometric_same"), Symbol("geometric_same", integer=True)),
            (Dummy("geometric_same"), Dummy("geometric_same")),
        )
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                source = Pow(2, 2*left, evaluate=False) - Pow(4, right)
                result = simplify(source)
                self.assertNotEqual(result, 0)
                self.assertEqual(result.free_symbols, {left, right})
                self.assertEqual(simplify(result.subs({left: 0, right: 1})), -3)

    def test_negative_and_unknown_bases_preserve_the_principal_branch(self):
        z = Symbol("geometric_branch_z")
        a = Symbol("geometric_branch_a")
        for base in (Integer(-2), a):
            with self.subTest(base=base):
                source = Pow(base, 2*z, evaluate=False)
                result = simplify(source)
                # Rebasing an unknown/negative base as (base**2)**z would
                # instead give +2 at this exact independent counterexample.
                at_negative_base = result.subs(a, -2)
                self.assertEqual(
                    simplify(at_negative_base.subs(z, Rational(1, 2))), -2
                )

    def test_optional_fold_leaves_unsupported_shapes_unchanged(self):
        z, a = Symbol("geometric_scope_z"), Symbol("geometric_scope_a")
        sources = (
            Pow(0, 2*z, evaluate=False),
            Pow(-2, 2*z, evaluate=False),
            Pow(a, 2*z, evaluate=False),
            Pow(Float(2.0), 2*z, evaluate=False),
            Pow(2, Rational(1, 2)*z, evaluate=False),
            Pow(2, a*z, evaluate=False),
            Pow(2, 2*z*z, evaluate=False),
            Pow(2, 2*z+1, evaluate=False),
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertIs(
                    _simplify_module._positive_rational_symbol_powers(source),
                    source,
                )

    def test_scalar_budget_declines_before_materializing_new_rationals(self):
        z = Symbol("geometric_budget_z")
        # Small, safe boundary witnesses exercise both pre-work limits. A
        # missing bound must not start constructing an expanded exact scalar.
        sources = (
            Pow(2, 1025*z, evaluate=False),
            Pow(2, -1025*z, evaluate=False),
            Pow(Integer(2**64), 65*z, evaluate=False),
        )
        for source in sources:
            with self.subTest(source=source), patch.object(
                _simplify_module, "Rational",
                side_effect=AssertionError("unadmitted exact power materialized"),
            ):
                self.assertIs(
                    _simplify_module._positive_rational_symbol_powers(source),
                    source,
                )

    def test_borrowed_preflight_precedes_python_child_export(self):
        from sympy.core import _native

        z = Symbol("geometric_preflight_z")
        source = Pow(2, 2*z, evaluate=False)
        with patch.object(
            _native, "ideal_source_preflight_native",
            side_effect=ValueError("ideal source exceeds its bounded expression envelope"),
        ) as preflight, patch.object(
            Pow, "args", new_callable=PropertyMock,
            side_effect=AssertionError("children exported before source preflight"),
        ) as exported:
            self.assertIs(
                _simplify_module._positive_rational_symbol_powers(source),
                source,
            )
            self.assertEqual(preflight.call_count, 1)
            self.assertEqual(exported.call_count, 0)

        # Only the known structural refusals decline the optional pass.
        # Other native, allocation and runtime failures remain terminal.
        for failure in (ValueError("unexpected native failure"),
                        MemoryError("allocation failed"),
                        RuntimeError("native operation failed")):
            with self.subTest(failure=type(failure)), patch.object(
                _native, "ideal_source_preflight_native", side_effect=failure,
            ):
                with self.assertRaises(type(failure)) as caught:
                    _simplify_module._positive_rational_symbol_powers(source)
                self.assertIs(caught.exception, failure)

    def test_original_held_source_is_admitted_before_power_cancellation(self):
        z = Symbol("geometric_held_z")
        hidden_undefined = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
        source = Add(
            Pow(4, z), Mul(-1, Pow(2, 2*z, evaluate=False), evaluate=False),
            hidden_undefined, evaluate=False,
        )
        failure = ValueError("original held source refused")
        with patch.object(
            _simplify_module, "_core_simplify", side_effect=failure,
        ) as admission, patch.object(
            _simplify_module, "_positive_rational_symbol_powers",
            side_effect=AssertionError("power fold ran before source admission"),
        ) as fold:
            with self.assertRaises(ValueError) as caught:
                simplify(source)
            self.assertIs(caught.exception, failure)
            self.assertIs(admission.call_args.args[0], source)
            self.assertEqual(admission.call_count, 1)
            self.assertEqual(fold.call_count, 0)


if __name__ == "__main__":
    unittest.main()
