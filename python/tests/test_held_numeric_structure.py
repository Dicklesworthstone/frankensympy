"""Held numeric arithmetic compares as structure, not as its evaluated value.

The equality groups, retained argument orders, and dictionary observations in
this file were checked with isolated SymPy 1.14.0. Concrete hash values are
deliberately unspecified: only equality/hash coherence is part of the contract.
"""

from __future__ import annotations

import unittest

import sympy as sp


def _eight_valued_expressions():
    """Oracle equality groups for different trees whose numerical value is 8."""
    return (
        ("python_integer", "atom", 8),
        ("integer", "atom", sp.Integer(8)),
        ("evaluated_power", "atom", sp.Pow(2, 3)),
        ("power", "power", sp.Pow(2, 3, evaluate=False)),
        ("power_duplicate", "power", sp.Pow(2, 3, evaluate=False)),
        ("unit_power", "unit_power", sp.Pow(8, 1, evaluate=False)),
        ("sum", "sum", sp.Add(2, 6, evaluate=False)),
        ("sum_duplicate", "sum", sp.Add(2, 6, evaluate=False)),
        ("reversed_sum", "reversed_sum", sp.Add(6, 2, evaluate=False)),
        ("product", "product", sp.Mul(2, 4, evaluate=False)),
        ("product_duplicate", "product", sp.Mul(2, 4, evaluate=False)),
        ("reversed_product", "reversed_product", sp.Mul(4, 2, evaluate=False)),
        (
            "held_exponent",
            "held_exponent",
            sp.Pow(2, sp.Add(1, 2, evaluate=False), evaluate=False),
        ),
        (
            "held_base",
            "held_base",
            sp.Pow(sp.Pow(2, 1, evaluate=False), 3, evaluate=False),
        ),
    )


class HeldNumericStructureTests(unittest.TestCase):
    def test_same_numeric_value_has_the_oracle_structural_equality_matrix(self):
        cases = _eight_valued_expressions()
        for left_name, left_group, left in cases:
            for right_name, right_group, right in cases:
                with self.subTest(left=left_name, right=right_name):
                    equal = left_group == right_group
                    self.assertEqual(left == right, equal)
                    self.assertEqual(right == left, equal)
                    self.assertEqual(left != right, not equal)
                    self.assertEqual(right != left, not equal)
                    if equal:
                        self.assertEqual(hash(left), hash(right))

    def test_dictionary_and_set_keys_follow_structure_in_both_insertion_orders(self):
        cases = _eight_valued_expressions()
        group_count = len({group for _, group, _ in cases})
        for ordered in (cases, tuple(reversed(cases))):
            with self.subTest(reversed=ordered is not cases):
                mapping = {expression: group for _, group, expression in ordered}
                self.assertEqual(len(mapping), group_count)
                self.assertEqual(len({expression for _, _, expression in ordered}), group_count)
                for name, group, expression in cases:
                    with self.subTest(expression=name):
                        self.assertEqual(mapping[expression], group)

    def test_held_numeric_forms_are_distinct_from_evaluated_atoms(self):
        half = sp.Rational(1, 2)
        cases = (
            (sp.Add, (0, 0), sp.Integer(0)),
            (sp.Add, (1, 2), sp.Integer(3)),
            (sp.Add, (half, half), sp.Integer(1)),
            (sp.Mul, (0, 9), sp.Integer(0)),
            (sp.Mul, (1, 3), sp.Integer(3)),
            (sp.Mul, (2, 3), sp.Integer(6)),
            (sp.Mul, (half, half), sp.Rational(1, 4)),
            (sp.Pow, (0, 0), sp.Integer(1)),
            (sp.Pow, (2, 0), sp.Integer(1)),
            (sp.Pow, (1, 7), sp.Integer(1)),
            (sp.Pow, (8, 1), sp.Integer(8)),
            (sp.Pow, (2, 3), sp.Integer(8)),
            (sp.Pow, (-2, 3), sp.Integer(-8)),
            (sp.Pow, (2, -1), half),
        )
        for constructor, args, evaluated in cases:
            with self.subTest(constructor=constructor.__name__, args=args):
                held = constructor(*args, evaluate=False)
                self.assertIs(type(held), constructor)
                self.assertEqual(held.args, tuple(sp.sympify(arg) for arg in args))
                self.assertNotEqual(held, evaluated)
                self.assertNotEqual(evaluated, held)
                self.assertNotIn(evaluated, {held: "held"})
                self.assertNotIn(held, {evaluated: "atom"})
                self.assertEqual(len({held: "held", evaluated: "atom"}), 2)
                self.assertEqual(len({evaluated: "atom", held: "held"}), 2)
                # Evaluated construction must retain its existing folding.
                self.assertEqual(constructor(*args), evaluated)

    def test_held_order_and_nested_structure_are_not_canonicalized_by_comparison(self):
        for constructor, args in ((sp.Add, (2, 6)), (sp.Mul, (2, 4))):
            with self.subTest(constructor=constructor.__name__):
                forward = constructor(*args, evaluate=False)
                reverse = constructor(*reversed(args), evaluate=False)
                original_hash = hash(forward)
                self.assertNotEqual(forward, reverse)
                self.assertNotEqual(reverse, forward)
                self.assertEqual(forward.args, tuple(map(sp.Integer, args)))
                self.assertEqual(reverse.args, tuple(map(sp.Integer, reversed(args))))
                self.assertEqual(hash(forward), original_hash)

        inner = sp.Pow(2, 1, evaluate=False)
        nested = sp.Pow(inner, 3, evaluate=False)
        flattened = sp.Pow(2, 3, evaluate=False)
        duplicate = sp.Pow(sp.Pow(2, 1, evaluate=False), 3, evaluate=False)
        self.assertNotEqual(nested, flattened)
        self.assertNotEqual(flattened, nested)
        self.assertEqual(nested.args, (inner, sp.Integer(3)))
        self.assertEqual(nested, duplicate)
        self.assertEqual(hash(nested), hash(duplicate))
        self.assertEqual({nested: "nested"}[duplicate], "nested")

    def test_matching_evaluated_and_held_composites_share_hash_and_dictionary_keys(self):
        x = sp.Symbol("held_numeric_structure_x")
        cases = (
            (sp.Add, (1, x)),
            (sp.Mul, (2, x)),
            (sp.Pow, (x, 2)),
            (sp.Pow, (2, sp.Rational(1, 2))),
        )
        for constructor, args in cases:
            with self.subTest(constructor=constructor.__name__, args=args):
                canonical = constructor(*args)
                held = constructor(*args, evaluate=False)
                self.assertIs(type(canonical), constructor)
                self.assertIs(type(held), constructor)
                self.assertEqual(canonical.args, held.args)
                self.assertEqual(canonical, held)
                self.assertEqual(held, canonical)
                self.assertEqual(hash(canonical), hash(held))
                self.assertEqual({canonical: "canonical"}[held], "canonical")
                self.assertEqual({held: "held"}[canonical], "held")
                self.assertEqual(len({canonical, held}), 1)

    def test_numeric_atom_equality_and_hashing_remain_coherent(self):
        pairs = (
            (sp.Integer(0), 0),
            (sp.Integer(1), 1),
            (sp.Integer(-8), -8),
            (sp.Integer(8), sp.Rational(16, 2)),
            (sp.Rational(1, 2), sp.Rational(2, 4)),
        )
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(left, right)
                self.assertEqual(right, left)
                self.assertEqual(hash(left), hash(right))
                self.assertEqual({left: "atom"}[right], "atom")
                self.assertEqual({right: "atom"}[left], "atom")

