"""Exact common-denominator ODE verification retains original source domains."""

from unittest.mock import patch

import pytest

from sympy import (
    Add, Derivative, Eq, Function, Integer, Mul, Pow, Rational, Symbol,
    checkodesol, exp,
)


def rational_flow(variable=None):
    x = Symbol("rational_residual_coordinate") if variable is None else variable
    y = Function("rational_residual_flow")(x)
    source = Derivative(y, x) + 2*y - x*y*y
    # Supplied independently of dsolve: with D below, D' - 2*D = -x.
    # Hence y=1/D satisfies y' + 2*y - x*y**2 = 0 where D is nonzero.
    candidate = 1/(x/2 + Rational(1, 4) - exp(2*(x-1))/4)
    return x, y, source, candidate


def test_exact_rational_function_identity_with_transcendental_coefficients():
    _, y, source, candidate = rational_flow()
    assert checkodesol(source, Eq(y, candidate), y) == (True, Integer(0))
    assert checkodesol(source, candidate, y) == (True, Integer(0))


@pytest.mark.parametrize("offset", [Integer(1), Rational(1, 10**60)])
def test_exact_nonzero_perturbations_are_not_accepted(offset):
    x, y, source, candidate = rational_flow()
    assert checkodesol(source, Eq(y, candidate + offset*x), y)[0] is False


@pytest.mark.parametrize("held_candidate", [False, True])
def test_original_held_poles_are_rejected_before_rational_normalization(held_candidate):
    x, y, source, candidate = rational_flow()
    hole = Mul(0, Pow(x, -1, evaluate=False), evaluate=False)
    if held_candidate:
        candidate = Add(candidate, hole, evaluate=False)
    else:
        source = Add(source, hole, evaluate=False)
    with patch("sympy.polys._opaque.rational_zero", side_effect=AssertionError("original pole erased")):
        assert checkodesol(source, Eq(y, candidate, evaluate=False), y) == (False, source)


def test_exact_rational_normalization_resource_failure_remains_terminal():
    _, y, source, candidate = rational_flow()
    with patch("sympy.polys._opaque._Polynomial.fraction",
               side_effect=ValueError("exact rational budget exhausted")):
        with pytest.raises(ValueError, match="exact rational budget exhausted"):
            checkodesol(source, Eq(y, candidate), y)


def test_noncommutative_declaration_cannot_enter_scalar_rational_normalization():
    x = Symbol("noncommutative_residual_coordinate", commutative=False)
    _, y, source, candidate = rational_flow(x)
    with patch("sympy.polys._opaque.rational_zero", side_effect=AssertionError("noncommutative source admitted")):
        try:
            result = checkodesol(source, Eq(y, candidate), y)
        except NotImplementedError:
            # An earlier deliberate scalar-ODE refusal is also a rejection;
            # this test requires that commutative cancellation never runs.
            return
        assert result[0] is False


def test_stale_symbol_declarations_remain_terminal_before_rational_normalization():
    x, y, source, candidate = rational_flow(Symbol("stale_rational_residual_coordinate"))
    declarations = x._assumptions
    saved = declarations.copy()
    try:
        declarations["commutative"] = False
        with pytest.raises(ValueError, match="symbol assumptions changed"):
            checkodesol(source, Eq(y, candidate), y)
    finally:
        declarations.clear()
        declarations.update(saved)


def test_unsupported_fractional_residual_keeps_the_unverified_result():
    x = Symbol("unsupported_rational_residual_coordinate")
    y = Function("unsupported_rational_residual_flow")(x)
    source = Derivative(y, x) - y
    verified, residual = checkodesol(source, x**Rational(1, 3), y)
    assert verified is False
    assert residual != 0


@pytest.mark.parametrize("offset, expected", [
    (Integer(0), True), (Rational(1, 10**60), False),
])
def test_zero_identity_and_tiny_perturbation_need_no_polynomial_gcd(offset, expected):
    x, y, source, candidate = rational_flow()
    with patch("sympy.polys._opaque._native.poly_multivariate_gcd_native",
               side_effect=AssertionError("zero identity must not compute a GCD")):
        result = checkodesol(source, Eq(y, candidate + offset*x), y)
    assert result[0] is expected
    if expected:
        assert result[1] == 0
    else:
        assert result[1] != 0


def test_rational_zero_rejects_a_zero_denominator_before_accepting_zero():
    from sympy.polys._opaque import rational_zero

    x = Symbol("rational_zero_invalid_denominator")
    with patch("sympy.polys._opaque._Polynomial.fraction", return_value=({}, {})):
        with pytest.raises(ValueError, match="zero denominator"):
            rational_zero(x)


def test_rational_zero_preserves_fraction_construction_cancellation():
    from sympy.polys._opaque import rational_zero

    x = Symbol("rational_zero_cancelled")
    failure = RuntimeError("injected rational coefficient cancellation")
    with patch("sympy.polys._opaque._Polynomial.fraction", side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            rational_zero(x/(x+1))
    assert caught.value is failure
