"""Signed assumptions, typed contexts and atomic updates at the public boundary."""

from itertools import product
import unittest

import sympy as s
from sympy.assumptions import AssumptionsContext, assuming, global_assumptions


class AssumptionsSoundnessTests(unittest.TestCase):
    def setUp(self):
        self.saved = global_assumptions.copy()
        global_assumptions.clear()
        self.x, self.y, self.z = s.symbols("sound_x sound_y sound_z")

    def tearDown(self):
        global_assumptions.clear()
        global_assumptions.update(self.saved)
        global_assumptions._domains = self.saved._domains

    def test_boolean_queries_agree_with_an_independent_truth_table(self):
        p, q, r = s.Q.integer(self.x), s.Q.positive(self.y), s.Q.even(self.z)
        cases = (
            (p | q, lambda a, b, c: a or b),
            (~(p | q), lambda a, b, c: not (a or b)),
            (p & q, lambda a, b, c: a and b),
            (~(p & q), lambda a, b, c: not (a and b)),
            (s.Implies(p, q), lambda a, b, c: not a or b),
            (~s.Implies(p, q), lambda a, b, c: a and not b),
            (s.Equivalent(p, q, r), lambda a, b, c: a == b == c),
            (~s.Equivalent(p, q, r), lambda a, b, c: not (a == b == c)),
            (s.Xor(p, q, r), lambda a, b, c: (a + b + c) % 2 == 1),
            (~s.Xor(p, q, r), lambda a, b, c: (a + b + c) % 2 == 0),
            (s.ITE(p, q, r), lambda a, b, c: b if a else c),
            (~s.ITE(p, q, r), lambda a, b, c: not (b if a else c)),
        )
        for formula, evaluate in cases:
            models = [values for values in product((False, True), repeat=3) if evaluate(*values)]
            for index, predicate in enumerate((p, q, r)):
                answers = {model[index] for model in models}
                expected = answers.pop() if len(answers) == 1 else None
                with self.subTest(formula=formula, query=predicate):
                    self.assertIs(s.ask(predicate, formula), expected)

    def test_domain_implications_hold_in_every_disjunct(self):
        x = self.x
        either_sign = s.Q.positive(x) | s.Q.negative(x)
        self.assertIs(s.ask(s.Q.real(x), either_sign), True)
        self.assertIs(s.ask(s.Q.nonzero(x), either_sign), True)
        self.assertIs(s.ask(s.Q.zero(x), either_sign), False)
        self.assertIsNone(s.ask(s.Q.positive(x), either_sign))

    def test_inconsistent_irrelevant_symbol_cannot_hide_behind_direct_answer(self):
        x, y = self.x, self.y
        facts = s.Q.positive(x) & s.Q.even(y) & s.Q.odd(y)
        with self.assertRaisesRegex(ValueError, "inconsistent assumptions"):
            s.ask(s.Q.positive(x), facts)

    def test_inconsistent_disjunct_is_discarded_without_poisoning_valid_one(self):
        x = self.x
        facts = (s.Q.positive(x) & s.Q.negative(x)) | s.Q.zero(x)
        self.assertIs(s.ask(s.Q.zero(x), facts), True)
        self.assertIs(s.ask(s.Q.positive(x), facts), False)

    def test_contradictions_from_implications_are_detected_before_publication(self):
        x, y = self.x, self.y
        p, q = s.Q.positive(x), s.Q.integer(y)
        with self.assertRaises(ValueError):
            s.ask(p, p & s.Implies(p, q) & ~q)

    def test_explicit_propositions_keep_the_pinned_precedence(self):
        # The public assumptions argument is a user assertion. This is pinned
        # Python behavior, not a certificate that the assertion itself is true.
        self.assertIs(s.ask(s.Q.positive(-1), s.Q.positive(-1)), True)
        positive = s.Symbol("explicit_positive", positive=True)
        self.assertIs(s.ask(s.Q.positive(positive), s.Q.negative(positive)), False)

    def test_composite_inference_uses_typed_facts_and_original_symbols(self):
        x, y = self.x, self.y
        self.assertIs(s.ask(s.Q.positive(x + y), s.Q.positive(x) & s.Q.positive(y)), True)
        self.assertIs(s.ask(s.Q.positive(x**2), s.Q.real(x) & s.Q.nonzero(x)), True)
        self.assertIs(s.ask(s.Q.integer(x + y), s.Q.integer(x) & s.Q.integer(y)), True)
        same_name = s.Symbol(x.name, integer=True)
        self.assertIsNone(s.ask(s.Q.positive(x), s.Q.positive(same_name)))
        self.assertIsNone(s.ask(s.Q.zero(x - same_name), s.Q.positive(x)))

    def test_negated_supertype_refutes_subtype_without_becoming_positive(self):
        x = self.x
        self.assertIs(s.ask(s.Q.positive(x), ~s.Q.real(x)), False)
        self.assertIs(s.ask(s.Q.integer(x), ~s.Q.rational(x)), False)
        self.assertIsNone(s.ask(s.Q.negative(x), ~s.Q.positive(x)))

    def test_same_named_function_classes_do_not_share_explicit_predicates(self):
        first = type(s.Function)("AssumedFunction", (s.Function,), {"__module__": __name__})
        second = type(s.Function)("AssumedFunction", (s.Function,), {"__module__": __name__})
        a, b = first(self.x, evaluate=False), second(self.x, evaluate=False)
        self.assertIs(s.ask(s.Q.positive(a), s.Q.positive(a)), True)
        self.assertIsNone(s.ask(s.Q.positive(b), s.Q.positive(a)))

    def test_custom_property_is_not_executed_by_automatic_inference(self):
        calls = []

        class Effectful(s.Function):
            @property
            def is_positive(self):
                calls.append("positive")
                raise AssertionError("unsupervised property")

        value = Effectful(self.x, evaluate=False)
        self.assertIsNone(s.ask(s.Q.positive(value)))
        self.assertEqual(calls, [])

    def test_effectful_metaclass_cannot_impersonate_an_admitted_formula_or_symbol(self):
        calls = []

        class EqualitySpoof(type):
            def __eq__(cls, other):
                calls.append("class equality")
                return True

            __hash__ = type.__hash__

        class UnsupportedFormula(metaclass=EqualitySpoof):
            def __iter__(self):
                calls.append("iteration")
                raise AssertionError("unsupported formula iterator ran")

        source = UnsupportedFormula()
        self.assertIsNone(s.ask(s.Q.positive(self.x), source))
        context = AssumptionsContext()
        with self.assertRaises(TypeError):
            context.assume(source, s.Q.positive)
        with self.assertRaises(TypeError):
            context.assume_domain(source, "ZZ")
        self.assertEqual(calls, [])
        self.assertEqual(set(context), set())

    def test_held_and_evaluated_predicate_sources_remain_distinct(self):
        held = s.Add(1, 2, evaluate=False)
        self.assertIsNone(s.ask(s.Q.special(s.Integer(3)), s.Q.special(held)))
        self.assertIs(s.ask(s.Q.special(held), s.Q.special(held)), True)
        self.assertEqual(held.args, (1, 2))

    def test_held_perfect_square_is_not_classified_by_canonical_power_heuristic(self):
        held = s.Pow(4, s.Rational(1, 2), evaluate=False)
        # The unreduced source is exactly 2. Refusal is permitted in this
        # bounded lane; claiming irrationality from "a surviving power" is not.
        self.assertIsNot(s.ask(s.Q.irrational(held)), True)
        self.assertEqual(held.args, (4, s.Rational(1, 2)))

    def test_negative_context_cannot_regain_conflicting_constructor_facts(self):
        positive = s.Symbol("constructor_positive", positive=True)
        facts = ~s.Q.real(positive)
        self.assertIs(s.ask(s.Q.positive(positive), facts), False)
        self.assertIs(s.ask(s.Q.real(positive), facts), False)
        self.assertIsNone(s.ask(s.Q.positive(positive**2), facts))

    def test_shell_only_declarations_do_not_reach_native_predicate_parsing(self):
        for constructor in (s.Symbol, s.Dummy):
            with self.subTest(constructor=constructor):
                plain = constructor("declared_commutative", commutative=True)
                positive = constructor("declared_positive_commutative", positive=True, commutative=True)
                self.assertIs(s.ask(s.Q.commutative(plain)), True)
                self.assertIsNone(s.ask(s.Q.real(plain)))
                self.assertIsNone(s.ask(s.Q.integer(plain + 1)))
                self.assertIsNone(s.ask(s.Q.integer(positive)))
                self.assertIs(s.ask(s.Q.positive(positive + 1)), True)

    def test_compatible_context_facts_augment_constructor_assumptions(self):
        for constructor in (s.Symbol, s.Dummy):
            with self.subTest(constructor=constructor):
                positive = constructor("positive_with_integer_context", positive=True)
                integer = constructor("integer_with_positive_context", integer=True)
                self.assertIs(s.ask(s.Q.positive(positive), s.Q.integer(positive)), True)
                self.assertIs(s.ask(s.Q.positive(positive + 1), s.Q.integer(positive)), True)
                self.assertIs(s.ask(s.Q.integer(integer), s.Q.positive(integer)), True)
                self.assertIs(s.ask(s.Q.integer(integer + 1), s.Q.positive(integer)), True)
                context = AssumptionsContext()
                context.assume_domain(positive, "ZZ")
                self.assertIs(context.is_true(positive, s.Q.positive), True)
                self.assertIs(context.is_true(positive + 1, s.Q.integer), True)

    def test_failed_bulk_update_is_atomic(self):
        context = AssumptionsContext([s.Q.positive(self.x)])
        before = set(context)
        with self.assertRaises(ValueError):
            context.update([s.Q.integer(self.y), s.Q.negative(self.x)])
        self.assertEqual(set(context), before)
        self.assertIsNone(context.is_true(self.y, s.Q.integer))

    def test_failed_symmetric_update_is_atomic(self):
        context = AssumptionsContext([s.Q.positive(self.x)])
        before = set(context)
        with self.assertRaises(ValueError):
            context ^= {s.Q.integer(self.y), s.Q.negative(self.x)}
        self.assertEqual(set(context), before)
        self.assertIs(context.is_true(self.x, s.Q.positive), True)

    def test_explicit_set_insertion_still_gets_consistency_checked_on_query(self):
        context = AssumptionsContext([s.Q.positive(self.x)])
        set.add(context, s.Q.negative(self.x))
        with self.assertRaises(ValueError):
            context.is_true(self.x, s.Q.positive)
        set.remove(context, s.Q.negative(self.x))
        self.assertIs(context.is_true(self.x, s.Q.positive), True)

    def test_context_keyword_has_no_ambient_assumption_leak(self):
        context = AssumptionsContext([s.Q.integer(self.y)])
        with assuming(s.Q.positive(self.x)):
            self.assertIs(s.ask(s.Q.positive(self.x)), True)
            self.assertIsNone(s.ask(s.Q.positive(self.x), context=context))
            self.assertIs(s.ask(s.Q.integer(self.y), context=context), True)

    def test_domains_have_typed_identity_and_clear_with_context(self):
        plain = self.x
        typed = s.Symbol(plain.name, real=True)
        context = AssumptionsContext()
        context.assume_domain(typed, "ZZ")
        self.assertIs(context.is_true(typed, s.Q.integer), True)
        self.assertIsNone(context.is_true(plain, s.Q.integer))
        copied = context.copy()
        context.clear()
        self.assertIsNone(context.is_true(typed, s.Q.integer))
        self.assertIs(copied.is_true(typed, s.Q.integer), True)

    def test_conflicting_domain_update_is_atomic_in_both_orders(self):
        for domain_first in (False, True):
            with self.subTest(domain_first=domain_first):
                context = AssumptionsContext()
                if domain_first:
                    context.assume_domain(self.x, "ZZ")
                    with self.assertRaises(ValueError):
                        context.assume(self.x, s.Q.infinite)
                    self.assertIs(context.is_true(self.x, s.Q.integer), True)
                    self.assertEqual(set(context), set())
                else:
                    context.assume(self.x, s.Q.infinite)
                    with self.assertRaises(ValueError):
                        context.assume_domain(self.x, "ZZ")
                    self.assertIs(context.is_true(self.x, s.Q.infinite), True)
                    self.assertEqual(context._domains, ())

    def test_unsupported_conjunct_does_not_hide_invalid_domain_or_known_conflict(self):
        relation = s.Eq(self.y, 0, evaluate=False)
        context = AssumptionsContext([relation, s.Q.positive(self.x)])
        before = set(context)
        with self.assertRaises(ValueError):
            context.assume_domain(self.x, "not-a-domain")
        self.assertEqual(context._domains, ())
        with self.assertRaises(ValueError):
            context.add(s.Q.negative(self.x))
        self.assertEqual(set(context), before)
        self.assertIsNone(context.is_true(self.x, s.Q.positive))

    def test_assuming_restores_domain_declarations_and_nested_mutations(self):
        global_assumptions.assume_domain(self.y, "ZZ")
        with assuming(s.Q.positive(self.x)):
            with assuming(s.Q.even(self.z)):
                global_assumptions.clear()
                global_assumptions.assume_domain(self.y, "QQ")
            self.assertIs(s.ask(s.Q.positive(self.x)), True)
            self.assertIs(s.ask(s.Q.integer(self.y)), True)
            self.assertIsNone(s.ask(s.Q.even(self.z)))
        self.assertIsNone(s.ask(s.Q.positive(self.x)))
        self.assertIs(s.ask(s.Q.integer(self.y)), True)

    def test_limits_refuse_without_an_answer_or_partial_context_change(self):
        predicates = [getattr(s.Q, "bounded_" + str(i))(self.x) for i in range(9)]
        expression = s.And(*(s.Or(predicate, ~predicate) for predicate in predicates))
        # The surface does not eliminate these predicate tautologies. The
        # bounded inference lane must refuse the exponential expansion rather
        # than accept the first positive branch as an unconditional answer.
        with self.assertRaisesRegex(ValueError, "limit"):
            s.ask(predicates[0], expression)
        context = AssumptionsContext([s.Q.integer(self.y)])
        before = set(context)
        with self.assertRaisesRegex(ValueError, "limit"):
            context.add(expression)
        self.assertEqual(set(context), before)


if __name__ == "__main__":
    unittest.main()
