"""Held powers retain Python children without observable reconstruction hooks."""

from __future__ import annotations

import unittest

import sympy as sp


def _builtin_power_observations():
    """Portable observations, also run under isolated SymPy 1.14.0."""
    x = sp.Symbol("x")
    cases = {
        "literal_zero_power": (sp.Integer(2), sp.Integer(0)),
        "zero_to_zero": (sp.Integer(0), sp.Integer(0)),
        "negative_base": (sp.Integer(-2), sp.Integer(3)),
        "reciprocal": (sp.Integer(2), sp.Integer(-1)),
        "unit_base": (sp.Integer(1), x),
        "symbol_zero_power": (x, sp.Integer(0)),
        "symbol_unit_power": (x, sp.Integer(1)),
        "square": (x, sp.Integer(2)),
        "nested_power": (sp.Pow(x, 2, evaluate=False), sp.Integer(3)),
        "held_sum_base": (sp.Add(x, x, evaluate=False), sp.Integer(2)),
        "held_product_base": (sp.Mul(x, x, evaluate=False), sp.Integer(2)),
        "held_sum_exponent": (x, sp.Add(1, 1, evaluate=False)),
    }
    observations = {}
    for name, (base, exponent) in cases.items():
        power = sp.Pow(base, exponent, evaluate=False)
        previous_hash = hash(power)
        args = power.args
        duplicate = sp.Pow(base, exponent, evaluate=False)
        observations[name] = {
            "text": str(power),
            "repr": repr(power),
            "args": [str(arg) for arg in args],
            "equal_duplicate": power == duplicate,
            "hash_equal_duplicate": previous_hash == hash(duplicate),
            "hash_stable_after_lookup": previous_hash == hash(power),
            "mapping_lookup_duplicate": {power: "retained"}[duplicate],
        }
    observations["equalities"] = {
        "square_matches_evaluated": sp.Pow(x, 2, evaluate=False) == x**2,
        "zero_power_is_one": sp.Pow(x, 0, evaluate=False) == sp.Integer(1),
        "unit_power_is_base": sp.Pow(x, 1, evaluate=False) == x,
        "literal_power_is_value": sp.Pow(2, 3, evaluate=False) == sp.Integer(8),
        "nested_power_is_flat": sp.Pow(
            sp.Pow(x, 2, evaluate=False), 3, evaluate=False
        ) == x**6,
        "different_exponents_equal": sp.Pow(x, 2, evaluate=False)
        == sp.Pow(x, 3, evaluate=False),
    }
    return observations


class HeldPowerSurfaceIdentityTests(unittest.TestCase):
    def test_same_named_custom_classes_keep_exact_base_and_exponent(self):
        def make_class():
            return type(
                "HeldPowerSameName",
                (sp.Function,),
                {"__module__": __name__, "nargs": 1},
            )

        first_class, second_class = make_class(), make_class()
        x = sp.Symbol("held_power_identity_x")
        first = first_class(x, evaluate=False)
        second = second_class(x, evaluate=False)
        self.assertEqual(str(first), str(second))
        self.assertIsNot(type(first), type(second))
        self.assertNotEqual(first, second)

        forward = sp.Pow(first, second, evaluate=False)
        reverse = sp.Pow(second, first, evaluate=False)
        # Later registration under the same printed name must not select
        # the children of either already-created power.
        third_class = make_class()
        third_class(x, evaluate=False)
        for power, base, exponent in (
            (forward, first, second),
            (reverse, second, first),
        ):
            with self.subTest(order=type(base) is first_class):
                args = power.args
                self.assertIs(args[0], base)
                self.assertIs(args[1], exponent)
                self.assertIs(power.base, base)
                self.assertIs(power.exp, exponent)
                self.assertIs(power.args, args)

    def test_args_lookup_does_not_call_child_getters_or_printers(self):
        state = {"armed": False, "calls": []}

        def guard(name):
            if state["armed"]:
                state["calls"].append(name)
                raise AssertionError("held-power lookup invoked " + name)

        class GuardedPowerChild(sp.Function):
            @property
            def args(self):
                guard("args")
                return super().args

            def __str__(self):
                guard("str")
                return super().__str__()

            def __repr__(self):
                guard("repr")
                return super().__repr__()

        x = sp.Symbol("held_power_guard_x")
        base = GuardedPowerChild(x, evaluate=False)
        exponent = GuardedPowerChild(sp.Integer(2), evaluate=False)
        power = sp.Pow(base, exponent, evaluate=False)
        retained = power.args
        state["armed"] = True
        try:
            for _ in range(4):
                self.assertIs(power.args, retained)
                self.assertIs(power.args[0], base)
                self.assertIs(power.args[1], exponent)
                self.assertIs(power.base, base)
                self.assertIs(power.exp, exponent)
        finally:
            state["armed"] = False
        self.assertEqual(state["calls"], [])

    def test_held_children_preserve_literal_nesting_and_symbol_declarations(self):
        plain = sp.Symbol("held_power_declared_x")
        positive = sp.Symbol("held_power_declared_x", positive=True)
        inner = sp.Pow(plain, sp.Integer(0), evaluate=False)
        held_sum = sp.Add(plain, plain, evaluate=False)
        held_product = sp.Mul(plain, plain, evaluate=False)
        pairs = (
            (sp.Integer(0), sp.Integer(0)),
            (sp.Integer(-2), sp.Integer(3)),
            (plain, sp.Integer(1)),
            (positive, plain),
            (inner, sp.Integer(3)),
            (held_sum, held_product),
        )
        for index, (base, exponent) in enumerate(pairs):
            with self.subTest(index=index):
                power = sp.Pow(base, exponent, evaluate=False)
                self.assertIs(type(power), sp.Pow)
                self.assertEqual(len(power.args), 2)
                self.assertIs(power.args[0], base)
                self.assertIs(power.args[1], exponent)
        self.assertIs(inner.args[0], plain)
        self.assertEqual(inner.args[1], sp.Integer(0))
        self.assertEqual(held_sum.args, (plain, plain))
        self.assertEqual(held_product.args, (plain, plain))
        self.assertNotEqual(plain, positive)

    def test_raw_literals_are_sympified_without_folding_held_power(self):
        for base, exponent in ((0, 0), (2, 0), (1, 7), (-2, 3)):
            with self.subTest(base=base, exponent=exponent):
                power = sp.Pow(base, exponent, evaluate=False)
                self.assertIs(type(power), sp.Pow)
                self.assertIsInstance(power.base, sp.Integer)
                self.assertIsInstance(power.exp, sp.Integer)
                self.assertEqual(power.args, (sp.Integer(base), sp.Integer(exponent)))

    def test_evaluated_powers_keep_existing_folding_and_args(self):
        x = sp.Symbol("held_power_evaluated_x")
        self.assertEqual(sp.Pow(2, 3), sp.Integer(8))
        self.assertEqual(sp.Pow(x, 0), sp.Integer(1))
        self.assertEqual(sp.Pow(x, 1), x)
        square = sp.Pow(x, 2)
        self.assertIs(type(square), sp.Pow)
        self.assertEqual(square.args, (x, sp.Integer(2)))
        self.assertEqual(square.base, x)
        self.assertEqual(square.exp, sp.Integer(2))

    def test_builtin_observations_match_pinned_oracle(self):
        # Recorded by running _builtin_power_observations unchanged with
        # isolated SymPy 1.14.0. Python hash values are process-local, so
        # only equality/coherence/stability observations are compared.
        expected_text_and_args = {
            "literal_zero_power": ("2**0", ["2", "0"]),
            "zero_to_zero": ("0**0", ["0", "0"]),
            "negative_base": ("(-2)**3", ["-2", "3"]),
            "reciprocal": ("1/2", ["2", "-1"]),
            "unit_base": ("1**x", ["1", "x"]),
            "symbol_zero_power": ("x**0", ["x", "0"]),
            "symbol_unit_power": ("x**1", ["x", "1"]),
            "square": ("x**2", ["x", "2"]),
            "nested_power": ("(x**2)**3", ["x**2", "3"]),
            "held_sum_base": ("(x + x)**2", ["x + x", "2"]),
            "held_product_base": ("(x*x)**2", ["x*x", "2"]),
            "held_sum_exponent": ("x**(1 + 1)", ["x", "1 + 1"]),
        }
        expected = {
            name: {
                "text": text,
                "repr": text,
                "args": args,
                "equal_duplicate": True,
                "hash_equal_duplicate": True,
                "hash_stable_after_lookup": True,
                "mapping_lookup_duplicate": "retained",
            }
            for name, (text, args) in expected_text_and_args.items()
        }
        expected["equalities"] = {
            "square_matches_evaluated": True,
            "zero_power_is_one": False,
            "unit_power_is_base": False,
            "literal_power_is_value": False,
            "nested_power_is_flat": False,
            "different_exponents_equal": False,
        }
        self.assertEqual(_builtin_power_observations(), expected)
