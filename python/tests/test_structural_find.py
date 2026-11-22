"""Public find/count observations for isolated SymPy 1.14 and native-shell runs."""

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
        assert active() is not None, "query effect escaped its owning request"


class StructuralFindTests(unittest.TestCase):
    def test_callable_finds_original_symbol_nodes(self):
        x, y = s.symbols("x y")
        expression = x + s.sin(y)
        self.assertEqual(expression.find(lambda node: isinstance(node, s.Symbol)), {x, y})
        self.assertEqual(expression.count(lambda node: isinstance(node, s.Symbol)), 2)

    def test_group_counts_every_held_occurrence(self):
        x = s.Symbol("x")
        expression = s.Add(x, x, s.sin(x), evaluate=False)
        self.assertEqual(expression.find(s.Symbol), {x})
        self.assertEqual(expression.find(s.Symbol, group=True), {x: 3})
        self.assertEqual(expression.count(s.Symbol), 3)

    def test_shared_subgraphs_are_counted_each_time_they_occur(self):
        x = s.Symbol("x")
        child = s.sin(x)
        expression = s.Add(child, child, evaluate=False)
        self.assertEqual(expression.find(s.sin, group=True), {child: 2})
        self.assertEqual(expression.find(x, group=True), {x: 2})
        self.assertEqual(expression.count(x), 2)

    def test_assumption_distinct_symbols_remain_separate_keys(self):
        plain = s.Symbol("x")
        positive = s.Symbol("x", positive=True)
        expression = s.Add(plain, positive, plain, evaluate=False)
        self.assertEqual(expression.find(s.Symbol, group=True), {plain: 2, positive: 1})
        self.assertEqual(expression.find(positive), {positive})
        self.assertEqual(expression.count(positive), 1)

    def test_same_named_dummies_remain_separate_keys(self):
        first, second = s.Dummy("x"), s.Dummy("x")
        expression = s.Add(first, second, first, evaluate=False)
        self.assertEqual(expression.find(s.Dummy, group=True), {first: 2, second: 1})
        self.assertEqual(expression.find(s.Symbol("x")), set())

    def test_same_named_function_classes_are_not_interchangeable(self):
        x = s.Symbol("x")
        first = type("SameFunction", (s.Function,), {"__module__": __name__})
        second = type("SameFunction", (s.Function,), {"__module__": __name__})
        a, b = first(x), second(x)
        expression = s.Add(a, b, a, evaluate=False)
        self.assertEqual(expression.find(first, group=True), {a: 2})
        self.assertEqual(expression.find(second), {b})
        self.assertEqual(expression.find(s.Function), {a, b})

    def test_class_queries_include_subclasses(self):
        x = s.Symbol("x")

        class Parent(s.Basic):
            pass

        class Child(Parent):
            pass

        child = Child(x)
        expression = Parent(child)
        self.assertEqual(expression.find(Parent), {expression, child})
        self.assertEqual(expression.find(Child), {child})
        self.assertEqual(expression.count(s.Basic), 3)

    def test_held_numeric_sum_keeps_its_literal_arguments(self):
        expression = s.Add(1, 2, evaluate=False)
        self.assertEqual(expression.find(s.Integer), {s.Integer(1), s.Integer(2)})
        self.assertEqual(expression.find(3), set())
        self.assertEqual(expression.count(s.Integer), 2)

    def test_held_zero_power_keeps_its_variable(self):
        x = s.Symbol("x")
        expression = s.Pow(x, 0, evaluate=False)
        self.assertEqual(expression.find(s.Symbol), {x})
        self.assertEqual(expression.find(0), {s.Integer(0)})
        self.assertEqual(expression.find(1), set())

    def test_exact_patterns_do_not_use_has_argument_subset_matching(self):
        x, y, z = s.symbols("x y z")
        self.assertEqual((x + y + z).find(x + y), set())
        self.assertEqual(s.sin(x + y).find(x + y), {x + y})
        self.assertEqual((x + y + z).count(x + y), 0)

    def test_bound_symbols_are_structurally_findable(self):
        x, y = s.symbols("x y")
        expression = s.Integral(x + y, (x, 0, 1))
        self.assertNotIn(x, expression.free_symbols)
        self.assertEqual(expression.find(s.Symbol, group=True), {x: 2, y: 1})

    def test_rational_and_boolean_queries_preserve_types(self):
        self.assertEqual(s.Rational(1, 2).find(Fraction(1, 2)), {s.Rational(1, 2)})
        self.assertEqual(s.S.One.find(True), set())
        self.assertEqual(s.true.find(True), {s.true})
        self.assertEqual(s.S.Zero.count(False), 0)

    def test_tuple_query_keeps_assumed_symbol_identity(self):
        x = s.Symbol("x", real=True)
        expression = s.Tuple(x)
        self.assertEqual(expression.find((x,)), {expression})
        self.assertEqual(expression.find((s.Symbol("x"),)), set())

    def test_string_and_list_queries_are_not_silently_parsed(self):
        x = s.Symbol("x")
        for query in ("x", [x]):
            with self.subTest(query=query):
                with self.assertRaises(TypeError):
                    (x + 1).find(query)
                with self.assertRaises(TypeError):
                    (x + 1).count(query)

    def test_none_is_truth_filter_only_for_find(self):
        x = s.Symbol("x")
        expression = s.Add(0, x, 1, evaluate=False)
        self.assertEqual(expression.find(None), {expression, x, s.Integer(1)})
        with self.assertRaises(TypeError):
            expression.count(None)

    def test_wild_queries_use_existing_symbolic_pattern_matching(self):
        x, y = s.symbols("x y")
        pattern = s.Wild("find_wild", properties=[lambda node: isinstance(node, s.Symbol)])
        expression = x + s.sin(y)
        self.assertEqual(expression.find(pattern), {x, y})
        self.assertEqual(expression.count(pattern), 2)

    def test_wild_property_failures_propagate(self):
        class QueryFailure(ValueError):
            pass

        def reject(node):
            _require_supervision_when_available()
            raise QueryFailure("property failed")

        pattern = s.Wild("find_failing_wild", properties=[reject])
        with self.assertRaisesRegex(QueryFailure, "property failed"):
            (s.Symbol("x") + 1).find(pattern)

    def test_custom_symbolic_matches_can_return_an_empty_binding(self):
        x, y = s.symbols("x y")

        class SymbolPattern(s.Basic):
            def matches(self, expression, repl_dict=None, old=False):
                _require_supervision_when_available()
                return {} if isinstance(expression, s.Symbol) else None

        expression = x + s.sin(y)
        pattern = SymbolPattern()
        self.assertEqual(expression.find(pattern), {x, y})
        self.assertEqual(expression.count(pattern), 2)

    def test_converter_precedes_callable_query_behavior(self):
        x = s.Symbol("find_converted_x", positive=True)
        reads = []

        class Query:
            @property
            def _sympy_(self):
                _require_supervision_when_available()
                reads.append("converted")
                return lambda: x

            def __call__(self, node):
                raise AssertionError("the symbolic conversion must take precedence")

        self.assertEqual((x + 1).find(Query()), {x})
        self.assertEqual(reads, ["converted", "converted"])

    def test_declined_conversion_preserves_the_original_callable_query(self):
        x = s.Symbol("find_conversion_miss_x")

        class Query:
            def _sympy_(self):
                _require_supervision_when_available()
                raise s.SympifyError("this query is a predicate")

            def __call__(self, node):
                _require_supervision_when_available()
                return node == x

        query = Query()
        with _request_context():
            self.assertEqual(s.sin(x).find(query), {x})
            self.assertEqual(s.sin(x).count(query), 1)

    def test_unexpected_conversion_failure_never_falls_back_to_callable(self):
        class Query:
            def _sympy_(self):
                raise RuntimeError("conversion failed")

            def __call__(self, node):
                raise AssertionError("a failed converter must not run a fallback")

        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            s.Symbol("x").find(Query())

    def test_repeated_callback_occurrences_are_fresh(self):
        x = s.Symbol("find_fresh_x")

        class Pair(s.Basic):
            pass

        expression = Pair(x, x)
        occurrences = []

        def query(node):
            _require_supervision_when_available()
            if node == x:
                occurrences.append(node)
                return len(occurrences) == 2
            return False

        with _request_context():
            self.assertEqual(expression.find(query), {x})
        self.assertEqual(occurrences, [x, x])
        occurrences.clear()
        with _request_context():
            self.assertEqual(expression.count(query), 1)
        self.assertEqual(occurrences, [x, x])

    def test_predicate_truth_conversion_is_inside_supervision(self):
        x = s.Symbol("find_truth_x")

        class Truth:
            def __init__(self, value):
                self.value = value

            def __bool__(self):
                _require_supervision_when_available()
                return self.value

        self.assertEqual(s.sin(x).find(lambda node: Truth(node == x)), {x})

    def test_predicate_failure_does_not_return_partial_matches(self):
        x, y = s.symbols("x y")

        def query(node):
            if node == y:
                raise RuntimeError("query failed")
            return node == x

        with self.assertRaisesRegex(RuntimeError, "query failed"):
            s.Add(x, y, evaluate=False).find(query, group=True)

    def test_callable_queries_can_observe_raw_structural_arguments(self):
        class Raw(s.Basic):
            pass

        expression = Raw(1, 1, "literal")
        self.assertEqual(expression.find(lambda node: type(node) is int, group=True), {1: 2})
        self.assertEqual(expression.count(lambda node: type(node) is str), 1)

    def test_count_does_not_require_hashable_nodes(self):
        class Unhashable(s.Basic):
            __hash__ = None

        expression = Unhashable(s.Symbol("x"))
        self.assertEqual(expression.count(s.Basic), 2)
        with self.assertRaises(TypeError):
            expression.find(s.Basic)

    def test_find_finishes_predicates_before_hashing_matches(self):
        class Unhashable(s.Basic):
            __hash__ = None

        x = s.Symbol("x")
        expression = Unhashable(x)
        for group in (False, True):
            with self.subTest(group=group):
                visited = []

                def query(node):
                    _require_supervision_when_available()
                    visited.append(node)
                    return True

                with self.assertRaises(TypeError):
                    expression.find(query, group=group)
                self.assertEqual(len(visited), 2)
                self.assertIs(visited[0], expression)
                self.assertIs(visited[1], x)

    def test_source_match_override_is_supervised(self):
        x = s.Symbol("find_override_x")

        class Source(s.Basic):
            def match(self, pattern, old=False):
                _require_supervision_when_available()
                return {} if pattern == x else None

        expression = Source(x)
        self.assertEqual(expression.find(x), {expression, x})

    def test_source_match_override_precedes_the_pattern_hook(self):
        class Pattern(s.Basic):
            def matches(self, expression, repl_dict=None, old=False):
                raise AssertionError("the source override must decide first")

        class Source(s.Basic):
            def match(self, pattern, old=False):
                _require_supervision_when_available()
                return None

        expression, pattern = Source(), Pattern()
        self.assertEqual(expression.find(pattern), set())
        self.assertEqual(expression.count(pattern), 0)

    def test_custom_instance_check_is_supervised(self):
        x = s.Symbol("find_instance_x")

        class QueryMeta(type):
            def __instancecheck__(cls, node):
                _require_supervision_when_available()
                return isinstance(node, s.Symbol)

        class SymbolQuery(metaclass=QueryMeta):
            pass

        self.assertEqual(s.sin(x).find(SymbolQuery), {x})

    def test_ordinary_class_queries_do_not_consume_callback_occurrences(self):
        symbols = s.symbols("find_large_0:160")
        expression = s.Add(*symbols, evaluate=False)
        self.assertEqual(expression.find(s.Symbol), set(symbols))
        self.assertEqual(expression.count(s.Symbol), len(symbols))

    def test_ordinary_exact_queries_do_not_consume_callback_occurrences(self):
        symbols = s.symbols("find_exact_0:160")
        expression = s.Add(*symbols, evaluate=False)
        self.assertEqual(expression.find(symbols[-1]), {symbols[-1]})
        self.assertEqual(expression.count(symbols[-1]), 1)


if __name__ == "__main__":
    unittest.main()
