"""Exact algebraic order and assumptions never use floating proximity."""

from math import isqrt

import pytest
import sympy as s
from sympy.core import Relational, _native, _native_expr, _real_sign_of_difference


def root_of_two(index=1, name="comparison_t"):
    t = s.Symbol(name)
    return s.CRootOf(t*t-2, t, index)


def test_real_root_assumptions_are_exact_for_both_embeddings():
    negative, positive = root_of_two(0), root_of_two(1)
    for root, sign in [(negative, -1), (positive, 1)]:
        assert root.is_real is True
        assert root.is_positive is (sign > 0)
        assert root.is_negative is (sign < 0)
        assert root.is_nonnegative is (sign > 0)
        assert root.is_nonpositive is (sign < 0)
        assert root.is_zero is False
        assert root.is_nonzero is True


def test_root_order_separates_exact_rational_neighbors_beyond_binary64():
    root = root_of_two()
    scale = 10**120
    numerator = isqrt(2*scale*scale)
    assert numerator*numerator < 2*scale*scale < (numerator+1)**2
    lower, upper = s.Rational(numerator, scale), s.Rational(numerator+1, scale)
    assert s.Lt(lower, root) is s.S.true
    assert s.Lt(root, upper) is s.S.true
    assert s.Ge(lower, root) is s.S.false
    assert s.Eq(root, lower) is s.S.false
    assert s.Ne(root, upper) is s.S.true


def test_distinct_defining_polynomials_are_compared_exactly():
    t = s.Symbol("comparison_close_t")
    scale = 10**80
    left = root_of_two()
    right = s.CRootOf(scale*t*t-(2*scale+1), t, 1)
    assert s.Lt(left, right) is s.S.true
    assert s.Gt(right, left) is s.S.true
    assert s.Eq(left, right) is s.S.false
    assert s.Lt(right, left) is s.S.false


def test_equal_root_order_and_bound_generator_renaming():
    first = root_of_two(name="comparison_bound_first")
    second = root_of_two(name="comparison_bound_second")
    assert s.Le(first, second) is s.S.true
    assert s.Ge(first, second) is s.S.true
    assert s.Lt(first, second) is s.S.false
    assert s.Gt(first, second) is s.S.false
    assert s.Eq(first, second) is s.S.true


def test_one_root_algebraic_relations_prove_zero_and_equality():
    root = root_of_two()
    zero = s.Add(s.Pow(root, 2, evaluate=False), -2, evaluate=False)
    identity = s.Mul(root+1, root-1, evaluate=False)
    assert zero.is_zero is True
    assert zero.is_nonzero is False
    assert zero.is_positive is False
    assert zero.is_nonnegative is True
    assert s.Eq(zero, 0) is s.S.true
    assert s.Ne(zero, 0) is s.S.false
    assert s.Eq(identity, 1) is s.S.true
    assert s.Lt(identity, 1) is s.S.false


def test_cancellation_and_rational_functions_keep_exact_signs():
    root = root_of_two()
    tiny = s.Rational(1, 10**120)
    negative = s.Add(root**2, -(2+tiny), evaluate=False)
    positive = s.Add(root**2, -(2-tiny), evaluate=False)
    assert negative.is_negative is True
    assert positive.is_positive is True
    assert s.Lt(negative, 0) is s.S.true
    assert s.Gt(positive, 0) is s.S.true
    assert ((root-1)/(root+1)).is_positive is True
    assert ((1-root)/(root+1)).is_negative is True


def test_min_max_and_abs_consume_exact_algebraic_decisions():
    negative, positive = root_of_two(0), root_of_two(1)
    assert s.Min(positive, negative, 0) == negative
    assert s.Max(negative, positive, 0) == positive
    assert s.Abs(positive) == positive
    assert s.Abs(negative) == -negative


def test_exact_decisions_do_not_invoke_root_float(monkeypatch):
    root = root_of_two()

    def forbidden_float(self):
        raise AssertionError("an exact algebraic decision called float")

    monkeypatch.setattr(s.CRootOf, "__float__", forbidden_float)
    assert s.Lt(root, 2) is s.S.true
    assert s.Gt(root, 1) is s.S.true
    assert (root**2-2).is_zero is True
    assert s.Max(root, 1) == root


@pytest.mark.parametrize("exponent", [-1, 0])
def test_original_undefined_children_do_not_become_assumption_facts(exponent):
    root = root_of_two()
    zero = s.Add(root**2, -2, evaluate=False)
    bad = s.Pow(zero, exponent, evaluate=False)
    expression = s.Add(root, s.Mul(0, bad, evaluate=False), evaluate=False)
    assert expression.is_positive is None
    assert expression.is_zero is None
    assert isinstance(s.Gt(expression, 0), Relational)


def test_distinct_root_arithmetic_has_an_exact_sign_without_float(monkeypatch):
    first = root_of_two()
    t = s.Symbol("comparison_other_t")
    second = s.CRootOf(t*t-3, t, 1)
    expression = s.Add(first, s.Mul(-1, second, evaluate=False), evaluate=False)

    def forbidden_float(self):
        raise AssertionError("exact comparison fell back to float")

    monkeypatch.setattr(s.CRootOf, "__float__", forbidden_float)
    assert _real_sign_of_difference(expression, s.Integer(0)) == -1
    assert s.Lt(expression, 0) is s.S.true


def test_distinct_root_structural_fallback_retains_zero_power_obligation():
    first = root_of_two()
    t = s.Symbol("comparison_structural_t")
    second = s.CRootOf(t*t-3, t, 1)
    base = s.Mul(0, s.Add(first, second, evaluate=False), evaluate=False)
    expression = s.Pow(base, 0, evaluate=False)
    assert expression.is_positive is None
    assert expression.is_zero is None
    assert isinstance(s.Gt(expression, 0), Relational)


def test_custom_root_subclass_does_not_enter_the_native_root_lane(monkeypatch):
    class UserRoot(s.CRootOf):
        pass

    t = s.Symbol("comparison_custom_t")
    expression = UserRoot(t*t-2, t, 1)
    called = []

    def forbidden_comparison(*args):
        called.append(True)
        raise AssertionError("a custom root class entered the exact built-in lane")

    monkeypatch.setattr(_native, "compare_real_roots_native", forbidden_comparison)
    assert expression.is_positive is None
    assert called == []


def test_refused_native_comparison_stays_unknown(monkeypatch):
    root = root_of_two()

    def exhausted(*args):
        raise ValueError("comparison budget exhausted")

    monkeypatch.setattr(_native, "compare_real_roots_native", exhausted)
    assert root.is_positive is None
    assert _real_sign_of_difference(root, s.Integer(0)) is None
    assert isinstance(s.Gt(root, 0), Relational)


def test_nonreal_root_order_is_not_a_real_approximation():
    t = s.Symbol("comparison_nonreal_t")
    root = s.CRootOf(t*t+1, t, 0)
    assert root.is_real is False
    assert root.is_positive is False
    with pytest.raises(TypeError):
        s.Lt(root, 0)


def test_native_comparison_checks_indices_and_shared_root_identity():
    def coefficients(values):
        return [_native_expr(s.Integer(value)) for value in values]

    quadratic = coefficients([-2, 0, 1])
    product = coefficients([6, 0, -5, 0, 1])
    # The quartic roots are -sqrt(3), -sqrt(2), sqrt(2), sqrt(3).
    assert _native.compare_real_roots_native(quadratic, 1, product, 2) == 0
    assert _native.compare_real_roots_native(quadratic, 1, product, 3) == -1
    assert _native.compare_real_roots_native(product, 0, quadratic, 0) == -1
    with pytest.raises(ValueError):
        _native.compare_real_roots_native(quadratic, 2, product, 0)
    with pytest.raises(ValueError):
        _native.compare_real_roots_native(coefficients([0]), 0, quadratic, 0)
