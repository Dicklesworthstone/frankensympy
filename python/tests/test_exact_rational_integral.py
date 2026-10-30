"""Checked exact real rational integration through the actual native extension."""

from fractions import Fraction

import pytest
import sympy as s
from sympy.core import _native


def equal(actual, expected):
    assert s.simplify(s.expand(actual - expected)) == 0


@pytest.mark.parametrize("direction", ["positive", "negative", "reversed"])
def test_coupled_logarithms_have_finite_improper_endpoint_values(direction):
    x = s.Symbol("exact_integral_log")
    lower, upper, expected = {
        "positive": (0, s.oo, s.log(2)),
        "negative": (-s.oo, -3, s.log(2)),
        "reversed": (s.oo, 0, -s.log(2)),
    }[direction]
    source = 1 / ((x + 1) * (x + 2))
    receipt = s.exact_rational_integral(source, (x, lower, upper))
    equal(receipt.value, expected)
    equal(s.integrate(source, (x, lower, upper)), expected)
    assert receipt.interval == (s.sympify(lower), s.sympify(upper))
    assert receipt.variable is x
    assert receipt.original_guard_count >= 1


@pytest.mark.parametrize("case", ["quadratic", "repeated", "distinct", "scaled_linear", "mixed"])
def test_complete_log_atan_and_repeated_factor_contributions(case):
    x = s.Symbol("exact_integral_factors")
    cases = {
        "quadratic": (1 / (x*x + 1), (-s.oo, s.oo), s.pi),
        "repeated": (1 / (x*x + 1)**2, (0, s.oo), s.pi / 4),
        "distinct": (1 / ((x*x + 1) * (x*x + 4)), (-s.oo, s.oo), s.pi / 6),
        "scaled_linear": (1 / (2*x + 3)**3, (0, s.oo), s.Rational(1, 36)),
        "mixed": (
            (2*x*x + 3*x + 3) / ((x*x + 1) * (x + 1) * (x + 2)),
            (0, s.oo), s.pi / 2 + s.log(2),
        ),
    }
    source, (lower, upper), expected = cases[case]
    receipt = s.exact_rational_integral(source, (x, lower, upper))
    equal(receipt.value, expected)
    assert receipt.value.free_symbols == set()


def test_finite_mixed_primitive_and_exact_rational_endpoint_difference():
    x = s.Symbol("exact_integral_finite")
    source = (2*x*x + 3*x + 3) / ((x*x + 1) * (x + 1) * (x + 2))
    expected = s.pi / 4 + 2*s.log(2) - s.log(3)
    receipt = s.exact_rational_integral(source, [x, 0, 1])
    equal(receipt.value, expected)
    equal(s.exact_rational_integral(source, (x, 1, 0)).value, -expected)
    start = 10**40
    assert s.exact_rational_integral(1 / x**2, (x, start, start + 1)).value == s.Rational(
        1, start * (start + 1),
    )


def test_negative_real_logarithms_and_coupled_different_degrees():
    x = s.Symbol("exact_integral_real_log")
    equal(s.exact_rational_integral(1 / x, (x, -2, -1)).value, -s.log(2))
    source = (2*x - 2) / ((x*x + 1) * (x + 1))
    equal(s.exact_rational_integral(source, (x, 1, s.oo)).value, s.log(2))
    equal(s.exact_rational_integral((3*x*x + 1) / (x**3 + x + 1), (x, 0, 1)).value, s.log(3))


@pytest.mark.parametrize("kind", ["cancelled", "zero_product", "nested_inverse", "held_zero_power"])
def test_held_original_interior_holes_cannot_be_erased(kind):
    x = s.Symbol("exact_integral_held")
    inverse = s.Pow(x - 1, -1, evaluate=False)
    source = {
        "cancelled": lambda: s.Mul(x - 1, inverse, evaluate=False),
        "zero_product": lambda: s.Mul(0, inverse, evaluate=False),
        "nested_inverse": lambda: s.Pow(inverse, -1, evaluate=False),
        "held_zero_power": lambda: s.Pow(inverse, 0, evaluate=False),
    }[kind]()
    with pytest.raises(ValueError):
        s.exact_rational_integral(source, (x, 0, 2))
    assert isinstance(s.integrate(source, (x, 0, 2)), s.Integral)


def test_removable_endpoint_holes_keep_their_original_guard_receipts():
    x = s.Symbol("exact_integral_endpoint")
    inverse = s.Pow(x - 1, -1, evaluate=False)
    cases = [
        (s.Mul(x - 1, inverse, evaluate=False), s.Integer(1)),
        (s.Mul(0, inverse, evaluate=False), s.Integer(0)),
        (s.Pow(inverse, -1, evaluate=False), -s.Rational(1, 2)),
        (s.Pow(inverse, 0, evaluate=False), s.Integer(1)),
    ]
    for source, expected in cases:
        receipt = s.exact_rational_integral(source, (x, 0, 1))
        assert receipt.value == expected
        assert receipt.original_guard_count >= 1


@pytest.mark.parametrize("power", [-1, 0])
@pytest.mark.parametrize("zero_kind", ["literal", "held_polynomial"])
def test_identically_undefined_original_children_refuse_before_zero_shortcuts(power, zero_kind):
    x = s.Symbol("exact_integral_zero_guard")
    zero = s.Integer(0) if zero_kind == "literal" else s.Add(x, -x, evaluate=False)
    source = s.Mul(0, s.Pow(zero, power, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.exact_rational_integral(source, (x, 0, 1))


def test_original_irrational_poles_require_exact_domain_admission():
    x = s.Symbol("exact_integral_irrational_pole")
    inverse = s.Pow(x*x - 2, -1, evaluate=False)
    source = s.Pow(inverse, 0, evaluate=False)
    with pytest.raises(ValueError):
        s.exact_rational_integral(source, (x, 1, 2))


@pytest.mark.parametrize("case", ["endpoint_log", "infinite_log", "two_infinite_logs", "interior_pole"])
def test_no_finite_or_principal_value_receipt_for_divergent_integrals(case):
    x = s.Symbol("exact_integral_divergent")
    source, lower, upper = {
        "endpoint_log": (1 / x, 0, 1),
        "infinite_log": (1 / (x + 1), 0, s.oo),
        "two_infinite_logs": (x / (x*x + 1), -s.oo, s.oo),
        "interior_pole": (1 / x, -1, 1),
    }[case]
    with pytest.raises(ValueError):
        s.exact_rational_integral(source, (x, lower, upper))


def test_zero_and_rational_polynomial_integrals_are_exact():
    x = s.Symbol("exact_integral_zero")
    zero = s.exact_rational_integral(0, (x, -1, 3))
    assert zero.value == 0
    assert zero.original_guard_count == 0
    assert s.exact_rational_integral(x*x, (x, 0, 1)).value == s.Rational(1, 3)
    assert s.exact_rational_integral(1 / (x*x + 1), (x, 1, 1)).value == 0
    assert s.exact_rational_integral(
        Fraction(2, 3), (x, Fraction(1, 4), Fraction(3, 4)),
    ).value == s.Rational(1, 3)


@pytest.mark.parametrize("kind", ["assumed", "dummy"])
def test_source_and_variable_lifting_preserve_typed_original_owners(kind):
    if kind == "assumed":
        variable = s.Symbol("exact_integral_same", real=True)
        other = s.Symbol("exact_integral_same")
    else:
        variable, other = s.Dummy("exact_integral_same"), s.Dummy("exact_integral_same")
    source = 1 / (variable*variable + 1)
    receipt = s.exact_rational_integral(source, (variable, 0, 1))
    assert receipt.variable is variable
    assert receipt.variable != other
    assert receipt.integrand.free_symbols == {variable}
    assert next(iter(receipt.integrand.free_symbols)) is variable
    equal(receipt.value, s.pi / 4)
    with pytest.raises(ValueError):
        s.exact_rational_integral(source, (other, 0, 1))
    with pytest.raises(ValueError):
        s.exact_rational_integral(s.Mul(0, other, evaluate=False), (variable, 0, 1))


@pytest.mark.parametrize("bad_source", [True, False, 0.5, s.Float("0.5")])
def test_inexact_or_boolean_sources_refuse(bad_source):
    x = s.Symbol("exact_integral_strict_source")
    with pytest.raises(TypeError):
        s.exact_rational_integral(bad_source, (x, 0, 1))


@pytest.mark.parametrize("bad_endpoint", [True, False, 0.5, s.Float("0.5")])
def test_inexact_or_boolean_endpoints_refuse(bad_endpoint):
    x = s.Symbol("exact_integral_strict_endpoint")
    with pytest.raises(TypeError):
        s.exact_rational_integral(1, (x, bad_endpoint, 1))


@pytest.mark.parametrize("bad_limits", [None, "x,0,1", (), (0, 1), (0, 1, 2, 3)])
def test_limits_require_an_exact_three_item_sequence(bad_limits):
    with pytest.raises(TypeError):
        s.exact_rational_integral(1, bad_limits)


def test_custom_source_and_variable_hooks_are_not_called():
    x = s.Symbol("exact_integral_hooks")
    calls = []

    class Custom:
        def _sympy_(self):
            calls.append("sympify")
            raise AssertionError("custom conversion must not execute")

    class CustomSymbol(s.Symbol):
        pass

    with pytest.raises(TypeError):
        s.exact_rational_integral(Custom(), (x, 0, 1))
    with pytest.raises(TypeError):
        s.exact_rational_integral(0, (CustomSymbol("custom"), 0, 1))
    assert calls == []


def test_unsupported_primitive_and_source_limits_stay_explicit():
    x = s.Symbol("exact_integral_limits")
    with pytest.raises(NotImplementedError):
        s.exact_rational_integral(1 / (x**3 + x + 1), (x, 0, 1))
    with pytest.raises(ValueError):
        s.exact_rational_integral(s.Pow(x + 1, 10**9, evaluate=False), (x, 0, 1))
    with pytest.raises(ValueError):
        s.exact_rational_integral(s.Pow(x, 33, evaluate=False), (x, 0, 1))


def test_ordinary_integration_retains_the_existing_larger_rational_lane():
    x = s.Symbol("exact_integral_old_lane")
    for exponent in (8, 32, 64):
        assert s.integrate(x**exponent, (x, 0, 1)) == s.Rational(1, exponent + 1)
    for multiplicity in (1, 2, 3):
        source = (1 - (2*multiplicity - 1)*x*x) / (1 + x*x)**(multiplicity + 1)
        assert s.integrate(source, (x, 0, 1)) == s.Rational(1, 2**multiplicity)


def test_accepted_receipt_properties_are_immutable():
    x = s.Symbol("exact_integral_immutable")
    receipt = s.exact_rational_integral(1 / (x*x + 1), (x, 0, 1))
    for name in ("value", "variable", "integrand", "interval", "original_guard_count", "claim"):
        with pytest.raises(AttributeError):
            setattr(receipt, name, None)
    for name in ("_receipt", "_owners"):
        with pytest.raises(AttributeError):
            delattr(receipt, name)
    equal(receipt.value, s.pi / 4)


def test_operational_failure_is_terminal_and_does_not_use_legacy_fallback(monkeypatch):
    x = s.Symbol("exact_integral_failure")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise RuntimeError("exact integral verification budget exhausted")

    def legacy(*args, **kwargs):
        raise AssertionError("an operational failure must not retry legacy integration")

    monkeypatch.setattr(_native, "exact_rational_integral_native", exhausted)
    monkeypatch.setattr(s, "integrate", legacy)
    with pytest.raises(RuntimeError, match="verification budget exhausted"):
        s.exact_rational_integral(1 / (x*x + 1), (x, 0, 1))
    assert len(calls) == 1


def test_a_python_candidate_cannot_manufacture_an_accepted_receipt(monkeypatch):
    x = s.Symbol("exact_integral_candidate")
    monkeypatch.setattr(_native, "exact_rational_integral_native",
                        lambda *args: {"value": 1, "verified": True})
    with pytest.raises(TypeError):
        s.exact_rational_integral(1 / (x*x + 1), (x, 0, 1))
