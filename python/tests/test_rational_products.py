"""Finite products preserve the source domain and use exact checked telescopes."""

from fractions import Fraction
from math import prod

import pytest
import sympy as s


def test_long_linear_rational_product_has_an_exact_value():
    k = s.Symbol("product_k", integer=True)
    end = 10**12
    assert s.product(k / (k + 1), (k, 1, end)) == s.Rational(1, end + 1)
    assert s.Product(k / (k + 1), (k, 1, end)).doit() == s.Rational(1, end + 1)


def test_repeated_shift_factors_preserve_multiplicities():
    k = s.Symbol("product_repeated_k", integer=True)
    end = 10**9
    assert s.product(k * (k + 2) / (k + 1)**2, (k, 1, end)) == s.Rational(
        end + 2, 2 * (end + 1)
    )


def test_quadratic_shift_orbit_telescopes_without_factoring_over_reals():
    k = s.Symbol("product_quadratic_k", integer=True)
    end = 10**9
    source = ((k + 1)**2 + 1) / (k*k + 1)
    assert s.product(source, (k, 0, end)) == (end + 1)**2 + 1


@pytest.mark.parametrize("lower,upper", [(1, 7), (-5, 5), (8, 1), (4, 3)])
def test_checked_finite_values_match_independent_fraction_products(lower, upper):
    k = s.Symbol("product_range_k", integer=True)
    source = (k*k + 2*k + 2) / (k*k + 1)
    if upper >= lower:
        expected = prod(Fraction(i*i + 2*i + 2, i*i + 1) for i in range(lower, upper + 1))
    else:
        expected = Fraction(1, 1) / prod(
            Fraction(i*i + 2*i + 2, i*i + 1) for i in range(upper + 1, lower)
        )
    expected = Fraction(expected)
    assert s.product(source, (k, lower, upper)) == s.Rational(
        expected.numerator, expected.denominator
    )


def test_nonempty_reversed_product_uses_the_reciprocal_convention():
    k = s.Symbol("product_reverse_k", integer=True)
    assert s.product(k, (k, 5, 1)) == s.Rational(1, 24)


@pytest.mark.parametrize("start", [-(2**63), 2**63 - 2])
def test_short_ranges_at_machine_integer_boundaries_do_not_overflow(start):
    k = s.Symbol("product_boundary_k", integer=True)
    assert s.product(k, (k, start, start + 1)) == start * (start + 1)


@pytest.mark.parametrize("shape", ["cancellation", "zero", "zeroth_power"])
@pytest.mark.parametrize("deep", [False, True])
def test_original_integer_poles_survive_every_surface_evaluation_path(shape, deep):
    k = s.Symbol("product_pole_k", integer=True)
    base = k - 3
    inverse = s.Pow(base, -1, evaluate=False)
    if shape == "cancellation":
        source = s.Mul(base, inverse, evaluate=False)
    elif shape == "zero":
        source = s.Mul(0, inverse, evaluate=False)
    else:
        source = s.Pow(inverse, 0, evaluate=False)
    with pytest.raises(ValueError):
        s.Product(source, (k, 1, 8)).doit(deep=deep)


def test_reversed_product_cannot_invert_a_zero_factor():
    k = s.Symbol("product_reverse_zero_k", integer=True)
    with pytest.raises(ValueError):
        s.product(k, (k, 1, -1))


def test_a_later_symbolic_limit_cannot_erase_the_first_finite_domain():
    k, j, n = s.symbols("product_inner_k product_outer_j product_outer_n", integer=True)
    source = s.Mul(0, s.Pow(k - 3, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.Product(source, (k, 1, 8), (j, 1, n)).doit()


def test_typed_dummy_axis_does_not_capture_a_same_named_symbol():
    k = s.Dummy("product_same", integer=True)
    other = s.Symbol("product_same", positive=True)
    assert s.product(k / (k + 1), (k, 1, 10000)) == s.Rational(1, 10001)
    result = s.product(other, (k, 1, 3))
    assert result.free_symbols == {other}
    assert next(iter(result.free_symbols)) is other
    assert result == other**3


def test_non_telescoping_long_product_stays_explicitly_unevaluated():
    k = s.Symbol("product_unsupported_k", integer=True)
    result = s.product(1 / (k*k + 1), (k, 0, 10000))
    assert isinstance(result, s.Product)
