"""A common exact representation retains every selected real embedding."""

from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction

import pytest
import sympy as s
from sympy.core import (
    Relational, _exact_ratio, _native, _native_expr, _symbol_binding, _wrap,
)
from sympy.polys.rootoftools import _algebraic_expression_value


def root(radicand, index=1):
    t = s.Symbol("multiple_root_t")
    return s.CRootOf(t*t-radicand, t, index)


def exact(native):
    return Fraction(*_exact_ratio(_wrap(native)))


def rounded(value, digits):
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_HALF_EVEN
        return +value


@pytest.mark.parametrize("indices", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_every_real_embedding_matches_independent_decimal_square_roots(indices):
    a, b = root(2, indices[0]), root(3, indices[1])
    expression = s.Add(a, b, evaluate=False)
    with localcontext() as context:
        context.prec = 120
        alpha = Decimal(2).sqrt() * (-1 if indices[0] == 0 else 1)
        beta = Decimal(3).sqrt() * (-1 if indices[1] == 0 else 1)
        expected = alpha + beta
    receipt = _algebraic_expression_value(expression, 80)
    assert receipt.binding_count == 2
    assert receipt.precision == 80
    assert Decimal(receipt.decimal) == rounded(expected, 80)
    assert Decimal(str(s.N(expression, 80))) == rounded(expected, 80)
    assert exact(receipt.upper)-exact(receipt.lower) < Fraction(1, 10**78)


def test_rational_arithmetic_and_polynomial_identity_use_both_roots():
    a, b = root(2), root(3)
    total = s.Add(a, b, evaluate=False)
    polynomial = s.Add(total**4, -10*total**2, 1, evaluate=False)
    reciprocal_identity = s.Add(
        s.Pow(total, -1, evaluate=False), a, -b, evaluate=False,
    )
    for expression in [polynomial, reciprocal_identity]:
        receipt = _algebraic_expression_value(expression, 60)
        assert exact(receipt.lower) == exact(receipt.upper) == 0
        assert expression.is_zero is True
        assert s.Eq(expression, 0) is s.S.true


def test_three_root_expression_preserves_an_exact_dependency():
    a, b, c = root(2), root(3), root(6)
    expression = s.Add(s.Mul(a, b, evaluate=False), -c, evaluate=False)
    receipt = _algebraic_expression_value(expression, 50)
    assert receipt.binding_count == 3
    assert exact(receipt.lower) == exact(receipt.upper) == 0


def test_signs_and_relations_are_exact_after_large_cancellation(monkeypatch):
    a, b = root(2), root(3)
    total = s.Add(a, b, evaluate=False)
    identity = s.Add(total**4, -10*total**2, 1, evaluate=False)
    negative = s.Add(identity, -s.Rational(1, 10**100), evaluate=False)

    def forbidden_float(self):
        raise AssertionError("an exact multi-root operation called float")

    monkeypatch.setattr(s.CRootOf, "__float__", forbidden_float)
    assert negative.is_negative is True
    assert s.Lt(negative, 0) is s.S.true
    assert s.Gt(total, 3) is s.S.true
    assert s.Lt(total, s.Rational(63, 20)) is s.S.true
    assert s.Max(total, 3) == total


@pytest.mark.parametrize("exponent", [-1, 0])
def test_original_multi_root_poles_survive_zero_products_and_cancellation(exponent):
    a, b = root(2), root(3)
    total = s.Add(a, b, evaluate=False)
    zero = s.Add(total**4, -10*total**2, 1, evaluate=False)
    undefined = s.Pow(zero, exponent, evaluate=False)
    expression = s.Add(total, s.Mul(0, undefined, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.N(expression, 50)
    assert expression.is_positive is None
    assert isinstance(s.Gt(expression, 0), Relational)
    cancelled = s.Mul(zero, s.Pow(zero, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.N(cancelled, 50)


def test_receipt_is_immutable_and_native_bindings_preserve_typed_identity():
    plain = s.Symbol("multiple_binding")
    assumed = s.Symbol("multiple_binding", real=True)
    coefficients = [
        [_native_expr(s.Integer(value)) for value in polynomial]
        for polynomial in [[-2, 0, 1], [-3, 0, 1]]
    ]
    receipt = _native.evaluate_real_algebraic_expression_native(
        _native_expr(plain+assumed),
        [_symbol_binding(plain), _symbol_binding(assumed)], coefficients, [1, 1], 50,
    )
    assert receipt.binding_count == 2
    assert exact(receipt.lower) > 3
    with pytest.raises(AttributeError):
        receipt.decimal = "0"
    with pytest.raises(TypeError):
        _native.AlgebraicExpressionValue()
    with pytest.raises(ValueError):
        _native.evaluate_real_algebraic_expression_native(
            _native_expr(plain+assumed),
            [_symbol_binding(plain), _symbol_binding(plain)], coefficients, [1, 1], 50,
        )


def test_native_boundary_rejects_mismatched_lists_and_invalid_embeddings():
    x, y = s.symbols("multiple_bad_x multiple_bad_y")
    bindings = [_symbol_binding(x), _symbol_binding(y)]
    coefficients = [[_native_expr(s.Integer(v)) for v in [-2, 0, 1]]] * 2
    for values, indices in [(coefficients[:1], [1, 1]), (coefficients, [1]), (coefficients, [2, 1])]:
        with pytest.raises(ValueError):
            _native.evaluate_real_algebraic_expression_native(
                _native_expr(x+y), bindings, values, indices, 50,
            )
    nonreal = [_native_expr(s.Integer(v)) for v in [1, 0, 1]]
    with pytest.raises(ValueError):
        _native.evaluate_real_algebraic_expression_native(
            _native_expr(x+y), bindings, [coefficients[0], nonreal], [1, 0], 50,
        )


def test_refusal_of_multi_root_native_work_does_not_restart_in_float(monkeypatch):
    expression = s.Add(root(2), root(3), evaluate=False)

    def exhausted(*args):
        raise ValueError("algebraic expression budget exhausted")

    monkeypatch.setattr(_native, "evaluate_real_algebraic_expression_native", exhausted)
    with pytest.raises(ValueError, match="budget exhausted"):
        s.N(expression, 50)
    assert expression.is_positive is None
    assert isinstance(s.Lt(expression, 4), Relational)


def test_custom_root_and_nonrational_grammar_cannot_enter_the_exact_lane():
    class UserRoot(s.CRootOf):
        pass

    t = s.Symbol("multiple_custom_t")
    a, b = root(2), root(3)
    custom = UserRoot(t*t-5, t, 1)
    for expression in [
        s.Add(a, b, custom, evaluate=False),
        s.Add(a, b, s.Float("0.25"), evaluate=False),
        s.sin(s.Add(a, b, evaluate=False)),
    ]:
        with pytest.raises(NotImplementedError, match="rational arithmetic"):
            _algebraic_expression_value(expression, 50)
