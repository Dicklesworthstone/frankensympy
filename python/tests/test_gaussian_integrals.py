"""Ordinary integrate and Integral.doit for translated polynomial Gaussians."""

import math

import pytest
import sympy as s


def equal(value, expected):
    assert s.simplify(s.expand(value - expected)) == 0


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5, 8, 13])
def test_translated_raw_moments(degree):
    x = s.Symbol("gaussian_x", real=True)
    mu = s.Rational(-3, 2)
    # a=4, so the centered variance is 1/8. Expand the raw moment using the
    # binomial theorem and pairings of independent standard normal powers.
    moment = sum(
        math.comb(degree, 2*k) * math.factorial(2*k) * mu**(degree - 2*k)
        / (math.factorial(k) * 16**k)
        for k in range(degree//2 + 1)
    )
    expr = x**degree * s.exp(-4*x**2 - 12*x - 9)
    equal(s.integrate(expr, (x, -s.oo, s.oo)), moment * s.sqrt(s.pi)/2)


def test_scaled_half_lines_and_finite_bounds():
    x = s.Symbol("weighted_gaussian_interval")
    m = s.Rational(1, 2)
    f = s.exp(-4*x**2 + 4*x + 2)
    half_mass = s.exp(3)*s.sqrt(s.pi)/4
    equal(s.Integral(f, (x, -s.oo, m)).doit(), half_mass)
    equal(s.integrate(f, (x, m, s.oo)), half_mass)
    equal(s.integrate(f, (x, 1, m)), -half_mass*s.erf(1))


def test_full_polynomial_cancellation_preserves_a_pure_exponential_primitive():
    x = s.Dummy("weighted_gaussian")
    q = -2*x**2 + 3*x - 1
    h = x**3 - 2
    weight = s.expand(s.diff(h, x) + h*s.diff(q, x))
    integral = s.Integral(weight*s.exp(q), x).doit()
    assert integral.free_symbols == {x}
    equal(integral, h*s.exp(q))
    equal(s.integrate(weight*s.exp(q), (x, 1, s.oo)), 1)
    equal(s.integrate(weight*s.exp(q), (x, 2, 1)), -1-6*s.exp(-3))


def test_indefinite_primitive_retains_source_coordinate_and_erf_term():
    x = s.Dummy("weighted_gaussian")
    f = (2*x+7)*s.exp(-x**2+2*x-1)
    primitive = s.integrate(f, x)
    assert primitive.free_symbols == {x}
    expected = -s.exp(-x**2+2*x-1) + s.Rational(9, 2)*s.sqrt(s.pi)*s.erf(x-1)
    equal(primitive, expected)


def test_original_excluded_point_survives_gaussian_factor_recognition():
    x = s.Symbol("gaussian_excluded_point")
    source = s.Mul(s.exp(-(x-1)**2), x-1, s.Pow(x-1, -1, evaluate=False), evaluate=False)
    held = s.Integral(source, (x, -s.oo, s.oo))
    assert held.doit() == held


def test_growing_quadratic_erfi_primitive_and_ordinary_chain_rule():
    x = s.Dummy("erfi_integral_x")
    f = (2*x+7)*s.exp(x**2-2*x+1)
    expected = s.exp(x**2-2*x+1) + 9*s.sqrt(s.pi)*s.erfi(x-1)/2
    primitive = s.integrate(f, x)
    assert primitive.free_symbols == {x}
    equal(primitive, expected)
    equal(s.Integral(f, (x, 3, 1)).doit(), 1-s.exp(4)-9*s.sqrt(s.pi)*s.erfi(2)/2)
    equal(s.diff(s.erfi(3*x+2), x), 6*s.exp((3*x+2)**2)/s.sqrt(s.pi))
    equal(s.diff(s.integrate(s.exp(x**2), x), x), s.exp(x**2))
    equal(s.integrate(s.exp(x**2), (x, -1, 1)), s.sqrt(s.pi)*s.erfi(1))


@pytest.mark.parametrize("degree,sign", [(0, -1), (0, 1), (1, 1), (3, -1), (4, 1)])
def test_growing_quadratic_tails_are_oriented_separately(degree, sign):
    x = s.Symbol("growing_quadratic_x")
    weight = sign if degree == 0 else sign*x**degree + 2
    f = weight*s.exp(2*x**2-4*x+1)
    left = s.oo if sign*(-1)**degree > 0 else -s.oo
    right = s.oo if sign > 0 else -s.oo
    assert s.integrate(f, (x, -s.oo, 1)) == left
    assert s.integrate(f, (x, 1, s.oo)) == right
    if degree % 2:
        for a, b in [(-s.oo, s.oo), (s.oo, -s.oo)]:
            result = s.integrate(f, (x, a, b))
            assert isinstance(result, s.Integral)
            assert result.function is f
    else:
        assert s.integrate(f, (x, -s.oo, s.oo)) == right
        assert s.integrate(f, (x, s.oo, -s.oo)) == -right
