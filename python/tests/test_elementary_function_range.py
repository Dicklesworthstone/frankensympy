"""Original-domain and exact endpoint behavior for elementary range maps."""

import pytest
import sympy as s


def test_held_removable_pole_survives_an_injective_outer_function():
    x = s.Symbol("elementary_range_hole")
    inner = s.Mul(x*x-1, s.Pow(x-1, -1, evaluate=False), evaluate=False)
    source = s.exp(inner, evaluate=False)
    assert s.function_range(source, x, s.Interval(0, 2)) == s.Union(
        s.Interval.Ropen(s.exp(1), s.exp(2)),
        s.Interval.Lopen(s.exp(2), s.exp(3)),
    )
    assert s.function_range(source, x, s.Interval(1, 1)) == s.S.EmptySet


def test_held_nested_reciprocal_hole_survives_square_root_branch_clipping():
    x = s.Symbol("elementary_range_sqrt_hole")
    inner = s.Pow(s.Pow(x, -1, evaluate=False), -1, evaluate=False)
    source = s.Pow(inner, s.Rational(1, 2), evaluate=False)
    assert s.function_range(source, x, s.Interval(-1, 1)) == s.Interval.Lopen(0, 1)


@pytest.mark.parametrize("wrapper, value", [
    (lambda f: s.Mul(0, f, evaluate=False), 0),
    (lambda f: s.Pow(f, 0, evaluate=False), 1),
])
def test_constant_outer_maps_do_not_erase_elementary_domains(wrapper, value):
    x = s.Symbol("elementary_range_constant")
    source = wrapper(s.log(x, evaluate=False))
    assert s.function_range(source, x, s.Interval(-2, -1)) == s.S.EmptySet
    assert s.function_range(source, x, s.Interval(0, 0)) == s.S.EmptySet
    assert s.function_range(source, x, s.Interval.open(0, 1)) == s.FiniteSet(value)
    assert s.minimum(source, x, s.Interval.open(0, 1)) == value
    assert s.maximum(source, x, s.Interval.open(0, 1)) == value


@pytest.mark.parametrize("source", [
    lambda x: s.log(x),
    lambda x: s.sqrt(x),
    lambda x: s.asin(x+4),
    lambda x: s.acosh(x),
])
def test_empty_real_branch_has_no_fabricated_extremum(source):
    x = s.Symbol("elementary_range_empty")
    expression = source(x)
    assert s.function_range(expression, x, s.Interval(-2, -1)) == s.S.EmptySet
    for operation in (s.minimum, s.maximum):
        with pytest.raises(ValueError, match="empty domain"):
            operation(expression, x, s.Interval(-2, -1))


def test_constant_inner_image_is_attained_on_an_open_domain():
    x = s.Symbol("elementary_range_held_constant")
    inner = s.Mul(x, s.Pow(x, -1, evaluate=False), evaluate=False)
    source = s.exp(inner, evaluate=False)
    assert s.function_range(source, x, s.Interval.open(0, 1)) == s.FiniteSet(s.exp(1))
    assert s.minimum(source, x) == s.exp(1)
    assert s.maximum(source, x) == s.exp(1)


def test_negative_affine_map_reverses_order_and_endpoint_attainment():
    x = s.Symbol("elementary_range_affine")
    expression = 5-2*s.acos(x)
    assert s.function_range(expression, x, s.Interval.Lopen(-1, 1)) == s.Interval.Lopen(
        5-2*s.pi, 5,
    )
    assert s.minimum(expression, x, s.Interval.Lopen(-1, 1)) == 5-2*s.pi
    assert s.maximum(expression, x, s.Interval.Lopen(-1, 1)) == 5


def test_fractional_power_keeps_exact_values_below_float_resolution():
    x = s.Symbol("elementary_range_exact_power")
    epsilon = s.Rational(1, 10**40)
    source = s.Pow(x, s.Rational(3, 2))
    domain = s.Interval.open(epsilon**2, 4*epsilon**2)
    assert s.function_range(source, x, domain) == s.Interval.open(epsilon**3, 8*epsilon**3)


def test_symbol_inference_and_typed_dummy_coordinates():
    x = s.Dummy("elementary_range_typed", real=True)
    assert s.minimum(s.exp(x*x)) == 1
    assert s.maximum(s.atan(x)) == s.pi/2
    assert s.function_range(s.tanh(x), x, s.Interval(0, 1)) == s.Interval(0, s.tanh(1))


def test_changed_domain_during_inner_analysis_is_not_published(monkeypatch):
    from sympy.core import _native
    from sympy.sets import Set

    x = s.Symbol("elementary_range_mutating_domain")
    domain = s.Interval(0, 1)
    source = s.exp(x)
    original = domain._native_set
    replacement = s.Interval(2, 3)._native_set
    solve = _native.solve_real_rational_inequality_native

    def changed(*args):
        answer = solve(*args)
        Set.__dict__["_native_set"].__set__(domain, replacement)
        return answer

    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", changed)
    try:
        with pytest.raises((NotImplementedError, RuntimeError, ValueError), match="domain changed"):
            s.function_range(source, x, domain)
    finally:
        Set.__dict__["_native_set"].__set__(domain, original)


@pytest.mark.parametrize("source", [
    lambda x: s.exp(s.exp(x)),
    lambda x: s.log(s.sqrt(x)),
    lambda x: s.sin(x),
    lambda x: s.exp(x)+x,
])
def test_outside_the_declared_composition_grammar_refuses(source):
    x = s.Symbol("elementary_range_unsupported")
    with pytest.raises(NotImplementedError):
        s.function_range(source(x), x, s.Interval(1, 2))
