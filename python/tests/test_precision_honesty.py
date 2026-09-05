"""Precision-honesty pins for N() (bead fra-fra-native-evalf-precision-honesty-ke8).

Contract: N(pi, d) computes the true ROUNDED d-significant-digit decimal via
the Machin series (pure integer arithmetic, 15 guard digits); every other
expression at d beyond the binary64-honest 15 significant digits gets a typed
NotImplementedError — an f64 is never extended with unjustified digits.
Expected strings are derived with the decimal module from an independently
embedded exact constant, NOT from the implementation.
"""
from __future__ import annotations

import unittest
from decimal import Context, Decimal, ROUND_HALF_UP

import sympy

# Exact pi to 65 significant digits (independent constant table).
PI_EXACT_65 = (
    "3.14159265358979323846264338327950288419716939937510582097494459230781"
)

x = sympy.Symbol("x")


def _expected_pi(d: int) -> str:
    """Oracle convention: d significant digits, ROUNDED (half-up is exact
    here because pi is irrational — the cut digit is never a lone 5)."""
    ctx = Context(prec=65, rounding=ROUND_HALF_UP)
    value = Decimal(PI_EXACT_65)
    quant = Decimal(1).scaleb(-(d - 1))
    rounded = value.quantize(quant, rounding=ROUND_HALF_UP, context=ctx)
    text = str(rounded)
    int_part, _, frac = text.partition(".")
    return int_part + "." + frac[: d - 1]


class PrecisionHonestN(unittest.TestCase):
    def test_digit_stream_rounds_like_pinned_oracle(self):
        for d in (1, 2, 5, 10, 15, 20, 30, 40, 50, 55):
            got = str(sympy.N(sympy.pi, d))
            self.assertEqual(got, _expected_pi(d), f"N(pi, {d}) digit stream")

    def test_n_pi_30_matches_pinned_oracle_string(self):
        # Captured live from the pinned 1.14.0 oracle venv.
        self.assertEqual(str(sympy.N(sympy.pi, 30)), "3.14159265358979323846264338328")

    def test_type_identity_is_profile_float(self):
        result = sympy.N(sympy.pi, 30)
        self.assertEqual(type(result).__name__, "Float")
        self.assertEqual(type(result).__module__, "sympy.core.numbers")
        self.assertEqual(str(result)[0], "3")

    def test_default_n_is_rounded_15_significant(self):
        # Pinned-oracle default string.
        self.assertEqual(str(sympy.N(sympy.pi)), "3.14159265358979")

    def test_non_pi_beyond_f64_honest_digits_refuses(self):
        with self.assertRaises(NotImplementedError):
            sympy.N(x, 30)
        with self.assertRaises(NotImplementedError):
            sympy.N(sympy.Integer(2), 16)
        with self.assertRaises(NotImplementedError):
            sympy.N(sympy.E, 100)

    def test_pi_cap_refuses_beyond_10000(self):
        with self.assertRaises(NotImplementedError):
            sympy.N(sympy.pi, 10_001)

    def test_adversarial_precision_arguments(self):
        with self.assertRaises(ValueError):
            sympy.N(sympy.pi, 0)
        with self.assertRaises(ValueError):
            sympy.N(sympy.pi, -5)
        with self.assertRaises(TypeError):
            sympy.N(sympy.pi, "30")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            sympy.N(sympy.pi, True)  # type: ignore[arg-type]

    def test_small_precision_non_pi_still_evaluates(self):
        # Within the f64-honest envelope non-pi evaluation keeps working.
        # (Trailing-zero RENDERING divergence vs oracle '2.00' is the
        # printer pack's ledgered item; the VALUE is what this pin checks.)
        self.assertEqual(sympy.N(2.5, 3), sympy.Float(2.5, 3))
        self.assertEqual(sympy.N(2, 3), sympy.Integer(2))


if __name__ == "__main__":
    unittest.main()
