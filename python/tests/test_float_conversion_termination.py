"""Scalar conversion terminates when evaluation cannot return a real atom."""

import math

import pytest
import sympy as s


@pytest.mark.parametrize("expression,expected", [
    (s.sqrt(2), math.sqrt(2)),
    (s.sin(s.Rational(1, 3)), math.sin(1 / 3)),
    (s.pi, math.pi),
])
def test_real_evaluation_still_converts_to_a_scalar(expression, expected):
    assert float(expression) == pytest.approx(expected)


def test_numerically_evaluated_complex_expression_remains_convertible():
    expression = s.N(s.Rational(1, 3) + s.Rational(2, 7) * s.I)
    assert complex(expression) == pytest.approx(complex(1 / 3, 2 / 7))


def test_nonreal_result_refuses_float_but_remains_convertible_to_complex():
    expression = s.Float(0.25) + s.Float(0.5) * s.I
    with pytest.raises(TypeError):
        float(expression)
    assert complex(expression) == complex(0.25, 0.5)


@pytest.mark.parametrize("return_nonreal", [False, True])
def test_float_conversion_does_not_retry_a_residual_expression(monkeypatch, return_nonreal):
    calls = []
    nonreal = s.Float(0.25) + s.Float(0.5) * s.I

    def unevaluated(expression, n=15):
        calls.append(expression)
        if len(calls) > 1:
            raise AssertionError("float conversion repeated an unevaluated result")
        return nonreal if return_nonreal else expression

    monkeypatch.setattr(s.Expr, "evalf", unevaluated)
    x = s.Symbol("x")
    with pytest.raises(TypeError):
        float(x)
    assert calls == [x]


@pytest.mark.parametrize("evaluated,expected", [
    (s.Integer(0), 0.0),
    (s.Rational(2, 3), 2 / 3),
    (s.Float(0.125), 0.125),
])
def test_evaluated_real_numeric_atoms_keep_their_scalar_conversion(monkeypatch, evaluated, expected):
    calls = []

    def numeric(expression, n=15):
        calls.append(expression)
        return evaluated

    monkeypatch.setattr(s.Expr, "evalf", numeric)
    x = s.Symbol("x")
    assert float(x) == expected
    assert calls == [x]


def test_float_subclass_is_not_reconstructed_by_native_conversion():
    from sympy.core import _complex_value

    calls = []

    class CustomFloat(s.Float):
        def _as_python_float(self):
            calls.append("custom conversion")
            return 123.0

    assert _complex_value(CustomFloat(0.5)) is None
    assert calls == []
