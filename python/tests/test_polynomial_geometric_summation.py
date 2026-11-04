"""Exact finite and convergent infinite sums of polynomial-geometric terms."""

import pytest
import sympy as s


@pytest.mark.parametrize("degree,expected", [(0, 2), (1, 2), (2, 6), (3, 26), (4, 150)])
def test_convergent_geometric_moments(degree, expected):
    k = s.Symbol("geometric_moment_k", integer=True)
    assert s.summation(k**degree / 2**k, (k, 0, s.oo)) == expected
    assert s.Sum(k**degree / 2**k, (k, 0, s.oo)).doit() == expected


def test_alternating_and_shifted_weights_have_exact_rational_sums():
    k = s.Symbol("geometric_alternating_k", integer=True)
    assert s.summation(k * s.Rational(-1, 2)**k, (k, 0, s.oo)) == s.Rational(-2, 9)
    assert s.summation((k*k + 3*k + 2) / 2**k, (k, 0, s.oo)) == 16
    assert s.summation(k / 2**k, (k, 3, s.oo)) == 1


def test_range_beyond_direct_iteration_has_an_exact_closed_form():
    k = s.Symbol("geometric_large_k", integer=True)
    assert s.summation(k * 2**k, (k, 0, 10000)) == 9999 * s.Integer(2)**10001 + 2


def test_symbolic_bound_result_reconstructs_each_finite_sum():
    k = s.Dummy("geometric_same", integer=True)
    n = s.Symbol("geometric_same", integer=True, nonnegative=True)
    result = s.summation(k * 2**k, (k, 0, n))
    assert not isinstance(result, s.Sum)
    assert result.free_symbols == {n}
    assert next(iter(result.free_symbols)) is n
    for end in (0, 1, 2, 7, 20):
        assert s.simplify(result.subs(n, end)) == sum(i * 2**i for i in range(end + 1))


@pytest.mark.parametrize("lower,upper", [(-3, 5), (8, 1), (3, 2)])
def test_negative_and_reversed_integer_ranges(lower, upper):
    k = s.Symbol("geometric_range_k", integer=True)
    ratio = s.Rational(-2, 3)
    if upper >= lower:
        expected = sum((i*i + 2*i + 3) * ratio**i for i in range(lower, upper + 1))
    else:
        expected = -sum((i*i + 2*i + 3) * ratio**i for i in range(upper + 1, lower))
    assert s.summation((k*k + 2*k + 3) * ratio**k, (k, lower, upper)) == expected


def test_affine_exponent_preserves_the_original_geometric_scale():
    k = s.Symbol("geometric_affine_k", integer=True)
    assert s.summation(k * 2**(2-k), (k, 0, s.oo)) == 8


def test_held_undefined_child_is_not_removed_by_zero_weight():
    k = s.Symbol("geometric_held_k", integer=True)
    invalid = s.Mul(0, s.Pow(0, -1, evaluate=False), 2**k, evaluate=False)
    with pytest.raises(ValueError):
        s.summation(invalid, (k, 0, 5000))


def test_excessive_exact_output_is_refused_without_partial_value():
    k = s.Symbol("geometric_growth_k", integer=True)
    with pytest.raises(ValueError):
        s.summation(k * 2**k, (k, 0, 100000))
