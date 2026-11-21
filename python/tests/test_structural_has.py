"""Structural containment observations shared with the SymPy 1.14 oracle.

The same module runs in separate oracle and candidate processes. It checks
the public shell graph, including cases native mathematical equality cannot
decide on behalf of Python object identity.
"""

import contextlib
from fractions import Fraction
import unittest

import sympy as s


def _request_context():
    from sympy import core

    supervisor = getattr(core, "supervised_callback_session", None)
    return supervisor() if supervisor is not None else contextlib.nullcontext()


def _require_supervision_when_available():
    from sympy import core

    active = getattr(core, "_active_callback_session", None)
    if active is not None:
        assert active() is not None, "callback effect escaped its owning request"


class StructuralHasTests(unittest.TestCase):
    def test_empty_pattern_list_is_false(self):
        x = s.Symbol("x")
        self.assertFalse(x.has())
        self.assertFalse((s.sin(x) + 1).has())

    def test_multiple_patterns_are_an_any_of_query(self):
        x, y, z, missing = s.symbols("x y z missing")
        expression = x**2 + s.sin(x*y)
        self.assertTrue(expression.has(z, y, missing))
        self.assertFalse(expression.has(z, missing))
        patterns = (z, y, s.cos, s.Integer)
        self.assertEqual(expression.has(*patterns),
                         any(expression.has(pattern) for pattern in patterns))

    def test_classes_match_nodes_and_their_base_classes(self):
        x = s.Symbol("x")
        expression = s.sin(x) + x**2
        for pattern in (s.Basic, s.Expr, s.Add, s.Pow, s.Symbol,
                        s.Integer, s.Rational, s.Number, s.Function, s.sin):
            with self.subTest(pattern=pattern):
                self.assertTrue(expression.has(pattern))
        for pattern in (s.Float, s.cos, s.Dummy, s.Mul):
            with self.subTest(pattern=pattern):
                self.assertFalse(expression.has(pattern))

    def test_mixed_classes_and_expressions(self):
        x, y = s.symbols("x y")
        expression = s.cos(x)
        self.assertTrue(expression.has(y, s.sin, s.cos))
        self.assertTrue(expression.has(s.sin, x))
        self.assertFalse(expression.has(s.sin, y, s.Float))

    def test_same_named_symbols_keep_assumption_identity(self):
        ordinary = s.Symbol("x")
        positive = s.Symbol("x", positive=True)
        real = s.Symbol("x", real=True)
        expression = s.Add(ordinary, positive, evaluate=False)
        self.assertTrue(expression.has(ordinary))
        self.assertTrue(expression.has(positive))
        self.assertFalse(expression.has(real))
        self.assertTrue(expression.has(real, positive))

    def test_same_named_dummies_keep_individual_identity(self):
        first, second = s.Dummy("x"), s.Dummy("x")
        expression = s.sin(first)
        self.assertTrue(expression.has(first))
        self.assertTrue(expression.has(s.Dummy, s.Symbol))
        self.assertFalse(expression.has(second, s.Symbol("x")))

    def test_undefined_function_class_queries(self):
        x = s.Symbol("x")
        f, g = s.Function("f"), s.Function("g")
        expression = f(x) + 1
        self.assertTrue(expression.has(f))
        self.assertTrue(expression.has(s.Function))
        self.assertFalse(expression.has(g))

    def test_same_named_custom_functions_remain_distinct(self):
        x = s.Symbol("x")
        first = type("RepeatedFunction", (s.Function,), {"__module__": __name__})
        second = type("RepeatedFunction", (s.Function,), {"__module__": __name__})
        expression = s.Add(first(x), 1, evaluate=False)
        self.assertTrue(expression.has(first))
        self.assertTrue(expression.has(first(x)))
        self.assertFalse(expression.has(second))
        self.assertFalse(expression.has(second(x)))

    def test_structural_subclasses_match_their_real_class(self):
        x = s.Symbol("x")

        class Container(s.Basic):
            pass

        class Child(Container):
            pass

        sibling = type("Child", (Container,), {"__module__": __name__})
        expression = Container(Child(x))
        self.assertTrue(expression.has(Container, s.Symbol))
        self.assertTrue(expression.has(Child))
        self.assertTrue(expression.has(x))
        self.assertFalse(expression.has(sibling))
        self.assertFalse(expression.has(sibling(x)))

    def test_bound_coordinates_are_still_structurally_present(self):
        x, y = s.symbols("x y")
        expression = s.Integral(x + y, (x, 0, 1))
        self.assertNotIn(x, expression.free_symbols)
        self.assertTrue(expression.has(x))
        self.assertTrue(expression.has(s.Integral))
        self.assertTrue(expression.has(s.Symbol))

    def test_strings_and_lists_are_not_parsed_as_patterns(self):
        x = s.Symbol("x")
        expression = x + 1
        for pattern in ("x", "x + 1", b"x", [x], {x}, {x: 1}, None):
            with self.subTest(pattern=pattern):
                self.assertFalse(expression.has(pattern))
        self.assertTrue(expression.has("missing", [x], s.Symbol))

    def test_exact_python_scalar_patterns(self):
        x = s.Symbol("x")
        expression = s.Add(x, s.Rational(1, 2), 3, evaluate=False)
        self.assertTrue(expression.has(3))
        self.assertTrue(expression.has(Fraction(1, 2)))
        self.assertFalse(expression.has(0.5))
        self.assertFalse(expression.has(4, Fraction(2, 3)))

    def test_boolean_patterns_do_not_become_numeric_atoms(self):
        self.assertTrue(s.true.has(True))
        self.assertTrue(s.false.has(False))
        self.assertFalse(s.S.One.has(True))
        self.assertFalse(s.S.Zero.has(False))
        self.assertFalse(s.true.has(1))
        self.assertFalse(s.false.has(0))

    def test_tuple_patterns_preserve_symbol_identity(self):
        x = s.Symbol("x", real=True)
        expression = s.Tuple(x)
        self.assertTrue(expression.has((x,)))
        self.assertFalse(expression.has((s.Symbol("x"),)))

    def test_held_numeric_arguments_are_not_replaced_by_the_value(self):
        expression = s.Add(1, 2, evaluate=False)
        self.assertTrue(expression.has(s.Add))
        self.assertTrue(expression.has(1))
        self.assertTrue(expression.has(2))
        self.assertFalse(expression.has(3))

    def test_held_zero_power_keeps_its_source_children(self):
        x = s.Symbol("x")
        expression = s.Pow(x, 0, evaluate=False)
        self.assertTrue(expression.has(x))
        self.assertTrue(expression.has(0))
        self.assertTrue(expression.has(s.Pow))
        self.assertFalse(expression.has(1))

    def test_held_product_keeps_a_literal_identity_factor(self):
        x = s.Symbol("x")
        expression = s.Mul(1, x, evaluate=False)
        self.assertTrue(expression.has(1))
        self.assertTrue(expression.has(s.Mul))
        self.assertTrue(expression.has(x))

    def test_additive_patterns_match_exact_argument_subsets(self):
        x, y, z = s.symbols("x y z")
        expression = s.sin(x + y + z)
        self.assertTrue(expression.has(x + z))
        self.assertFalse(expression.has(x - z))
        self.assertFalse(expression.has(2*x + y))

    def test_multiplicative_patterns_match_exact_argument_subsets(self):
        x, y, z = s.symbols("x y z")
        self.assertTrue((x*y*z).has(x*z))
        self.assertFalse((3*x).has(2*x))
        self.assertFalse((x**4).has(x**2))
        self.assertFalse((x + y).has(x*y))

    def test_held_commuting_subset_uses_profile_set_semantics(self):
        x, y = s.symbols("x y")
        expression = s.Add(x, y, evaluate=False)
        pattern = s.Add(x, x, evaluate=False)
        self.assertTrue(expression.has(pattern))

    def test_noncommuting_subset_requires_a_contiguous_ordered_block(self):
        a, b, c = s.symbols("a b c", commutative=False)
        expression = s.Mul(a, b, c, evaluate=False)
        self.assertTrue(expression.has(s.Mul(a, b, evaluate=False)))
        self.assertTrue(expression.has(s.Mul(b, c, evaluate=False)))
        self.assertFalse(expression.has(s.Mul(a, c, evaluate=False)))
        self.assertFalse(expression.has(s.Mul(b, a, evaluate=False)))

    def test_unhashable_symbolic_patterns_raise(self):
        class Unhashable(s.Basic):
            __hash__ = None

        with self.assertRaises(TypeError):
            s.Symbol("x").has(Unhashable())

    def test_class_queries_keep_the_profile_hashability_obligation(self):
        class Unhashable(s.Basic):
            __hash__ = None

        with self.assertRaises(TypeError):
            Unhashable().has(s.Basic)

    def test_custom_converter_keeps_the_returned_symbol_identity(self):
        x = s.Symbol("x", positive=True)
        calls = []

        class Pattern:
            def _sympy_(self):
                calls.append("converted")
                return x

        self.assertTrue((x + 1).has(Pattern()))
        self.assertEqual(calls, ["converted"])

    def test_custom_matcher_queries_the_original_shell_nodes(self):
        x, y = s.symbols("x y")

        class MatchSymbol(s.Basic):
            def _has_matcher(self):
                return lambda node: node == x

        self.assertTrue(s.sin(x).has(MatchSymbol()))
        self.assertFalse(s.sin(y).has(MatchSymbol()))

    def test_distinct_queries_do_not_share_cached_predicate_outcomes(self):
        x, y = s.symbols("has_effect_x has_effect_y")

        class MatchSymbol(s.Basic):
            def _has_matcher(self):
                return lambda node: node == x

        pattern = MatchSymbol()
        with _request_context():
            self.assertTrue(s.sin(x).has(pattern))
            self.assertFalse(s.sin(y).has(pattern))

    def test_effectful_matcher_factories_are_fresh_for_distinct_queries(self):
        x, y = s.symbols("has_factory_x has_factory_y")
        targets = [x, y]

        class MatchSymbol(s.Basic):
            def _has_matcher(self):
                _require_supervision_when_available()
                target = targets.pop(0)
                return lambda node: node == target

        pattern = MatchSymbol()
        with _request_context():
            self.assertTrue(s.sin(x).has(pattern))
            self.assertTrue(s.sin(y).has(pattern))
        self.assertEqual(targets, [])

    def test_effectful_converters_are_fresh_for_distinct_queries(self):
        x, y = s.symbols("has_convert_x has_convert_y")
        targets = [x, y]

        class Pattern:
            def _sympy_(self):
                _require_supervision_when_available()
                return targets.pop(0)

        pattern = Pattern()
        with _request_context():
            self.assertTrue(x.has(pattern))
            self.assertTrue(y.has(pattern))
        self.assertEqual(targets, [])

    def test_converter_descriptor_reads_follow_the_supervised_profile(self):
        x = s.Symbol("has_descriptor_x")
        reads = []

        class Pattern:
            @property
            def _sympy_(self):
                _require_supervision_when_available()
                reads.append("read")
                return lambda: x

        self.assertTrue(x.has(Pattern()))
        self.assertEqual(reads, ["read", "read"])

    def test_matcher_descriptor_and_truth_conversion_are_supervised(self):
        x = s.Symbol("has_truth_x")

        class Truth:
            def __init__(self, value):
                self.value = value

            def __bool__(self):
                _require_supervision_when_available()
                return self.value

        class MatchSymbol(s.Basic):
            @property
            def _has_matcher(self):
                _require_supervision_when_available()
                return lambda: lambda node: Truth(node == x)

        self.assertTrue(s.sin(x).has(MatchSymbol()))

    def test_forged_bound_method_metadata_cannot_bypass_supervision(self):
        x, y = s.symbols("has_spoof_x has_spoof_y")
        missing = object()
        original = s.Add.__dict__.get("_has_matcher", missing)
        original_function = s.Add._has_matcher

        class ForgedFactory:
            def __init__(self):
                self.__func__ = original_function

            def __call__(self):
                _require_supervision_when_available()
                return lambda node: node == x

        pattern = x + y
        s.Add._has_matcher = ForgedFactory()
        try:
            self.assertTrue(x.has(pattern))
        finally:
            if original is missing:
                del s.Add._has_matcher
            else:
                s.Add._has_matcher = original

    def test_exact_match_precedes_custom_matcher_creation(self):
        class ExactOnly(s.Basic):
            def _has_matcher(self):
                raise AssertionError("an exact node match needs no matcher")

        pattern = ExactOnly()
        self.assertTrue(pattern.has(pattern))


if __name__ == "__main__":
    unittest.main()
