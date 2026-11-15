"""Exact decisions and undecidable membership across symbolic set algebra.

The shell's ``contains`` contract uses None for an undecided predicate.
Python ``in`` must raise in that case, as the pinned SymPy 1.14 oracle
does for its unevaluated membership predicates. Neither structural
inequality nor Python truthiness establishes mathematical non-membership.
"""

import unittest

import sympy as s
from sympy.sets.sets import Set


class _ResponseSet(Set):
    """A Python-owned set with an explicit user-defined predicate result."""

    __slots__ = ("response", "queries")

    def __new__(cls, response):
        obj = object.__new__(cls)
        obj._native_set = None
        obj.response = response
        obj.queries = []
        return obj

    def contains(self, other):
        self.queries.append(other)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SetMembershipTests(unittest.TestCase):
    def setUp(self):
        self.x = s.Symbol("membership_x")

    def test_symbolic_finite_membership_is_not_a_negative_answer(self):
        values = s.FiniteSet(self.x)
        self.assertIs(values.contains(self.x), True)
        self.assertIn(self.x, values)
        self.assertIsNone(values.contains(1))
        with self.assertRaisesRegex(TypeError, "did not evaluate to a bool"):
            1 in values
        with self.assertRaises(TypeError):
            1 not in values

    def test_known_exact_membership_keeps_python_boolean_answers(self):
        huge = 10**400
        values = s.FiniteSet(s.Rational(1, 3), s.Integer(huge))
        self.assertIn(s.Rational(2, 6), values)
        self.assertNotIn(s.Rational(1, 3) + s.Rational(1, huge), values)
        self.assertIn(s.Integer(huge), values)
        self.assertNotIn(s.Integer(huge + 1), values)
        self.assertNotIn(self.x, s.S.EmptySet)
        self.assertIn(self.x, s.S.UniversalSet)

    def test_python_membership_accepts_only_boolean_decisions(self):
        for response, expected in ((True, True), (False, False),
                                   (s.true, True), (s.false, False)):
            with self.subTest(response=response):
                self.assertIs(1 in _ResponseSet(response), expected)
        for response in (None, s.Eq(self.x, 1, evaluate=False), 1, 0):
            with self.subTest(response=response):
                with self.assertRaises(TypeError):
                    1 in _ResponseSet(response)

    def test_membership_does_not_invoke_arbitrary_truthiness(self):
        class NoTruthValue:
            def __bool__(self):
                raise AssertionError("an unknown predicate was coerced to bool")

        predicate = _ResponseSet(NoTruthValue())
        with self.assertRaises(TypeError):
            1 in predicate
        self.assertIsNone(s.Union(predicate, s.Range(3, 4)).contains(1))
        self.assertIsNone(s.Intersection(predicate, s.Range(0, 3)).contains(1))
        self.assertIsNone(s.Complement(s.Range(0, 3), predicate).contains(1))
        self.assertIsNone(s.ProductSet(predicate, s.Range(0, 3)).contains((1, 1)))

    def test_cartesian_membership_retains_unknown_and_known_negative(self):
        product = s.ProductSet(s.FiniteSet(self.x), s.FiniteSet(1))
        self.assertIsNone(product.contains((2, 1)))
        self.assertIsNone(product.contains(s.Tuple(2, 1)))
        with self.assertRaises(TypeError):
            (2, 1) in product
        with self.assertRaises(TypeError):
            s.Tuple(2, 1) in product
        # The second factor refutes the tuple even though the first is unknown.
        self.assertIs(product.contains((2, 3)), False)
        self.assertNotIn((2, 3), product)
        self.assertIn((self.x, 1), product)
        self.assertIn(s.Tuple(self.x, 1), product)

    def test_cartesian_shape_is_checked_before_component_membership(self):
        untouched = _ResponseSet(AssertionError("a wrong-shaped tuple was inspected"))
        product = s.ProductSet(untouched, s.Range(0, 3))
        for item in (1, (), (1,), (1, 2, 3), s.Tuple(1)):
            with self.subTest(item=item):
                self.assertIs(product.contains(item), False)
                self.assertNotIn(item, product)
        self.assertEqual(untouched.queries, [])

    def test_a_class_named_tuple_is_not_a_symbolic_tuple(self):
        class Tuple:
            def __len__(self):
                raise AssertionError("a same-named custom class was admitted")

            def __iter__(self):
                raise AssertionError("a same-named custom class was admitted")

        product = s.ProductSet(s.Range(0, 3), s.Range(0, 3))
        self.assertIs(product.contains(Tuple()), False)

    def test_mixed_union_preserves_unknown_until_one_part_proves_membership(self):
        combined = s.Union(s.FiniteSet(self.x), s.Range(0, 3))
        self.assertIs(combined.contains(1), True)
        self.assertIn(1, combined)
        self.assertIsNone(combined.contains(5))
        with self.assertRaises(TypeError):
            5 in combined
        self.assertIs(s.Union(s.Range(0, 3), s.FiniteSet(4)).contains(5), False)

    def test_mixed_intersection_can_refute_an_unknown_component(self):
        combined = s.Intersection(s.FiniteSet(self.x), s.Range(0, 3))
        self.assertIsNone(combined.contains(1))
        with self.assertRaises(TypeError):
            1 in combined
        self.assertIs(combined.contains(5), False)
        self.assertNotIn(5, combined)

    def test_relative_complement_decides_python_owned_ranges(self):
        difference = s.Complement(s.Range(0, 4), s.Range(1, 3))
        for item, expected in ((0, True), (1, False), (2, False), (3, True),
                               (4, False), (s.Rational(1, 2), False)):
            with self.subTest(item=item):
                self.assertIs(difference.contains(item), expected)
                self.assertIs(item in difference, expected)

    def test_relative_complement_keeps_both_membership_obligations(self):
        difference = s.Complement(s.Range(0, 3), s.FiniteSet(self.x))
        self.assertIsNone(difference.contains(1))
        with self.assertRaises(TypeError):
            1 in difference
        self.assertIs(difference.contains(4), False)
        # Membership in the removed set is certain despite the unknown base.
        self.assertIs(difference.contains(self.x), False)
        reverse = s.Complement(s.FiniteSet(self.x), s.Range(0, 3))
        self.assertIs(reverse.contains(1), False)
        self.assertIsNone(reverse.contains(4))
        with self.assertRaises(TypeError):
            4 in reverse

    def test_native_relative_complement_still_decides_interval_holes(self):
        difference = s.Complement(s.Interval(0, 3), s.FiniteSet(1))
        for item, expected in ((0, True), (1, False), (2, True), (3, True),
                               (-1, False), (4, False)):
            with self.subTest(item=item):
                self.assertIs(difference.contains(item), expected)
                self.assertIs(item in difference, expected)

    def test_boolean_atoms_compose_like_python_boolean_decisions(self):
        yes, no, unknown = _ResponseSet(s.true), _ResponseSet(s.false), _ResponseSet(None)
        self.assertIs(s.Union(no, unknown).contains(1), None)
        self.assertIs(s.Union(unknown, yes).contains(1), True)
        self.assertIs(s.Intersection(yes, unknown).contains(1), None)
        self.assertIs(s.Intersection(unknown, no).contains(1), False)
        self.assertIs(s.ProductSet(yes, no).contains((1, 1)), False)
        self.assertIs(s.ProductSet(yes, unknown).contains((1, 1)), None)
        self.assertIs(s.Complement(yes, no).contains(1), True)
        self.assertIs(s.Complement(unknown, yes).contains(1), False)
        self.assertIs(s.Complement(unknown, no).contains(1), None)

    def test_decisive_short_circuits_do_not_call_unneeded_predicates(self):
        failed = _ResponseSet(AssertionError("unneeded predicate was called"))
        yes, no = _ResponseSet(s.true), _ResponseSet(s.false)
        self.assertIs(s.Union(yes, failed).contains(1), True)
        self.assertIs(s.Intersection(no, failed).contains(1), False)
        self.assertIs(s.ProductSet(no, failed).contains((1, 1)), False)
        self.assertIs(s.Complement(no, failed).contains(1), False)
        self.assertEqual(failed.queries, [])

    def test_required_predicate_failures_remain_errors(self):
        failed = _ResponseSet(ValueError("membership budget exhausted"))
        yes, no = _ResponseSet(s.true), _ResponseSet(s.false)
        for result, item in ((failed, 1),
                             (s.Union(no, failed), 1),
                             (s.Intersection(yes, failed), 1),
                             (s.ProductSet(yes, failed), (1, 1)),
                             (s.Complement(yes, failed), 1)):
            with self.subTest(type=type(result).__name__):
                with self.assertRaisesRegex(ValueError, "membership budget exhausted"):
                    result.contains(item)
                with self.assertRaisesRegex(ValueError, "membership budget exhausted"):
                    item in result


if __name__ == "__main__":
    unittest.main()
