"""Independent adversarial checks for Boolean assumptions and context identity.

The public ``ask``/``refine`` answers here were checked against an isolated
SymPy 1.14.0 process.  Context convenience methods and eager rejection are
FrankenSymPy extensions: their lifecycle checks assert soundness of the
public set state, rather than upstream API parity.
"""

from __future__ import annotations

import unittest

from sympy import Abs, Dummy, Equivalent, Implies, Q, Symbol, Xor, ask, refine
from sympy.assumptions import AssumptionsContext, assuming, global_assumptions
from sympy.assumptions.assume import Predicate


class AssumptionsContextAdversaries(unittest.TestCase):
    def setUp(self):
        self.saved_globals = set(global_assumptions)
        global_assumptions.clear()
        self.x = Symbol("context_adversary_x")
        self.y = Symbol("context_adversary_y")

    def tearDown(self):
        global_assumptions.clear()
        global_assumptions.update(self.saved_globals)

    def test_disjunction_does_not_assert_either_alternative(self):
        x = self.x
        alternatives = Q.positive(x) | Q.negative(x)
        self.assertIsNone(ask(Q.positive(x), alternatives))
        self.assertIsNone(ask(Q.negative(x), alternatives))
        self.assertEqual(refine(Abs(x), alternatives), Abs(x))

    def test_negative_literal_is_not_a_positive_fact(self):
        x = self.x
        self.assertIs(ask(Q.positive(x), ~Q.positive(x)), False)
        self.assertIsNone(ask(Q.negative(x), ~Q.positive(x)))
        self.assertEqual(refine(Abs(x), ~Q.positive(x)), Abs(x))

    def test_implication_does_not_assert_premise_or_conclusion(self):
        premise, conclusion = Q.positive(self.x), Q.integer(self.y)
        implication = Implies(premise, conclusion)
        self.assertIsNone(ask(premise, implication))
        self.assertIsNone(ask(conclusion, implication))
        self.assertIs(ask(conclusion, premise & implication), True)

    def test_negated_alternative_resolves_a_disjunction(self):
        positive, negative = Q.positive(self.x), Q.negative(self.x)
        self.assertIs(ask(positive, (positive | negative) & ~negative), True)

    def test_xor_and_equivalence_do_not_assert_operands(self):
        x = self.x
        self.assertIsNone(ask(Q.positive(x), Xor(Q.positive(x), Q.negative(x))))
        self.assertIsNone(ask(Q.positive(x), Equivalent(Q.positive(x), Q.real(x))))

    def test_negating_zero_requires_realness_to_prove_nonzero(self):
        x = self.x
        self.assertIsNone(ask(Q.nonzero(x), ~Q.zero(x)))
        self.assertIs(ask(Q.nonzero(x), Q.real(x) & ~Q.zero(x)), True)
        self.assertIsNone(ask(Q.rational(x), ~Q.integer(x)))

    def test_finiteness_does_not_promote_an_unknown_value_to_complex(self):
        x = self.x
        self.assertIs(ask(Q.finite(x), Q.finite(x)), True)
        self.assertIs(ask(Q.infinite(x), Q.finite(x)), False)
        for predicate in (Q.complex, Q.real, Q.nonzero, Q.zero, Q.positive, Q.negative):
            with self.subTest(predicate=predicate):
                self.assertIsNone(ask(predicate(x), Q.finite(x)))

    def test_infinity_is_outside_the_finite_complex_and_real_towers(self):
        x = self.x
        self.assertIs(ask(Q.infinite(x), Q.infinite(x)), True)
        for predicate in (Q.finite, Q.complex, Q.real, Q.nonzero, Q.zero, Q.positive, Q.negative):
            with self.subTest(predicate=predicate):
                self.assertIs(ask(predicate(x), Q.infinite(x)), False)

    def test_declared_finiteness_uses_the_same_python_predicate_semantics(self):
        finite = Symbol("declared_finite_adversary", finite=True)
        infinite = Symbol("declared_infinite_adversary", infinite=True)
        self.assertIsNone(ask(Q.complex(finite)))
        self.assertIsNone(ask(Q.complex(finite + 1)))
        self.assertIs(ask(Q.complex(infinite)), False)
        self.assertIs(ask(Q.nonzero(infinite)), False)

    def test_native_complex_nonzero_does_not_prove_python_real_nonzero(self):
        from sympy import I, exp

        self.assertIs(ask(Q.nonzero(I)), False)
        self.assertIsNone(ask(Q.nonzero(exp(self.x)), Q.complex(self.x)))
        self.assertIs(ask(Q.nonzero(I * self.x), Q.positive(self.x)), False)

    def test_finite_arguments_do_not_erase_held_complex_poles(self):
        from sympy import I, atan, pi, tanh

        for value in (atan(I, evaluate=False), tanh(I * pi / 2, evaluate=False)):
            with self.subTest(value=value):
                self.assertIsNone(ask(Q.finite(value)))
                # The native Finite closure also implies Complex. Neither
                # property may be asserted before excluding the same pole.
                self.assertIsNot(ask(Q.complex(value)), True)

    def test_matching_query_does_not_hide_inconsistency(self):
        x = self.x
        for contradiction in (
            Q.positive(x) & Q.negative(x),
            Q.even(x) & Q.odd(x),
            False,
        ):
            with self.subTest(contradiction=contradiction):
                with self.assertRaises(ValueError):
                    ask(Q.positive(x), contradiction)

    def test_custom_predicate_retains_literal_polarity(self):
        special = Predicate("context_adversary_special")(self.x)
        self.assertIs(ask(special, special), True)
        self.assertIs(ask(special, ~special), False)

    def test_same_spelling_does_not_identify_distinct_symbols(self):
        plain = self.x
        positive = Symbol(plain.name, positive=True)
        negative = Symbol(plain.name, negative=True)
        self.assertNotEqual(plain, positive)
        self.assertNotEqual(plain, negative)
        self.assertIsNone(ask(Q.positive(plain), Q.positive(positive)))
        self.assertIsNone(ask(Q.negative(plain), Q.negative(negative)))
        self.assertIsNone(ask(Q.real(plain), Q.positive(positive)))

    def test_same_spelling_does_not_identify_distinct_dummies(self):
        first, second = Dummy("context_d"), Dummy("context_d")
        self.assertNotEqual(first, second)
        self.assertIsNone(ask(Q.positive(first), Q.positive(second)))

    def test_expression_fact_does_not_become_a_symbol_named_by_its_printout(self):
        expression = self.x + 1
        printed_symbol = Symbol(str(expression))
        self.assertIsNone(ask(Q.positive(printed_symbol), Q.positive(expression)))
        self.assertIsNone(ask(Q.zero(expression), Q.zero(printed_symbol)))

    def test_property_substitution_cannot_capture_a_user_symbol(self):
        x = self.x
        private_lookalike = Symbol("_fsym_ask_" + x.name, positive=True)
        # Both positive values can be different; a spelling-based replacement
        # would capture the second symbol and falsely reduce their difference.
        self.assertIsNone(ask(Q.zero(x - private_lookalike), Q.positive(x)))

    def test_every_set_retraction_is_visible_to_context_queries(self):
        p = Q.positive(self.x)
        for mutation in (
            lambda c: c.remove(p),
            lambda c: c.discard(p),
            lambda c: c.pop(),
            lambda c: c.clear(),
            lambda c: c.difference_update({p}),
            lambda c: c.intersection_update(set()),
            lambda c: c.symmetric_difference_update({p}),
            lambda c: c.__isub__({p}),
            lambda c: c.__iand__(set()),
            lambda c: c.__ixor__({p}),
            # The inherited mutable-set API can be invoked explicitly too.
            lambda c: set.remove(c, p),
        ):
            context = AssumptionsContext([p])
            self.assertIs(context.is_true(self.x, Q.positive), True)
            mutation(context)
            self.assertEqual(set(context), set())
            self.assertIsNone(context.is_true(self.x, Q.positive))
            self.assertEqual(context.deductions(self.x), [])

    def test_inplace_union_is_visible_to_context_queries(self):
        context = AssumptionsContext()
        context |= {Q.positive(self.x)}
        self.assertIs(context.is_true(self.x, Q.positive), True)
        context -= {Q.positive(self.x)}
        self.assertIsNone(context.is_true(self.x, Q.positive))

    def test_context_convenience_api_preserves_symbol_identity(self):
        plain = self.x
        assumed = Symbol(plain.name, integer=True)
        context = AssumptionsContext()
        context.assume(assumed, Q.positive)
        self.assertIs(context.is_true(assumed, Q.positive), True)
        self.assertIsNone(context.is_true(plain, Q.positive))

    def test_rejected_context_insertion_does_not_leave_a_half_update(self):
        positive = Q.positive(self.x)
        context = AssumptionsContext([positive])
        with self.assertRaises(ValueError):
            context.add(Q.negative(self.x))
        self.assertEqual(set(context), {positive})
        self.assertIs(context.is_true(self.x, Q.positive), True)

    def test_assuming_restores_outer_facts_after_body_failure(self):
        with assuming(Q.integer(self.y)):
            saved = set(global_assumptions)
            with self.assertRaisesRegex(RuntimeError, "body failed"):
                with assuming(Q.positive(self.x)):
                    self.assertIs(ask(Q.positive(self.x)), True)
                    raise RuntimeError("body failed")
            self.assertEqual(set(global_assumptions), saved)
            self.assertIsNone(ask(Q.positive(self.x)))
            self.assertIs(ask(Q.integer(self.y)), True)
        self.assertEqual(set(global_assumptions), set())

    def test_assuming_restores_context_when_entry_is_inconsistent(self):
        with assuming(Q.integer(self.y)):
            saved = set(global_assumptions)
            with self.assertRaises(ValueError):
                with assuming(Q.positive(self.x), Q.negative(self.x)):
                    ask(Q.positive(self.x))
            self.assertEqual(set(global_assumptions), saved)
            self.assertIsNone(ask(Q.positive(self.x)))
            self.assertIs(ask(Q.integer(self.y)), True)


if __name__ == "__main__":
    unittest.main()
