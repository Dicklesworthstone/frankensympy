"""Quadratic rational telescopes through the actual native public shell."""

from fractions import Fraction

import pytest
import sympy as s


def telescope(primitive, variable, shift=1):
    return primitive - primitive.subs(variable, variable + shift)


@pytest.mark.parametrize("power", [1, 2, 3])
def test_repeated_quadratic_factors_have_exact_infinite_sums(power):
    k = s.Symbol("quadratic_power_k", integer=True)
    source = telescope(1 / (k*k + 1)**power, k)
    assert s.summation(source, (k, 0, s.oo)) == 1
    assert s.Sum(source, (k, 0, s.oo)).doit(deep=False) == 1


def test_polynomial_numerators_and_missing_intermediate_shifts():
    k = s.Symbol("quadratic_shift_k", integer=True)
    primitive = (3*k + 2) / (k*k + k + 1)**2
    source = telescope(primitive, k, shift=3)
    assert s.summation(source, (k, 0, s.oo)) == s.Rational(1199, 441)


def test_fractional_offsets_and_nonmonic_denominators():
    k = s.Symbol("quadratic_offset_k", integer=True)
    primitive = 1 / (3*(k + s.Rational(1, 2))**2 + 6)
    assert s.summation(telescope(primitive, k), (k, 0, s.oo)) == s.Rational(4, 27)


def test_irrational_real_poles_do_not_block_an_integer_lattice_sum():
    k = s.Symbol("quadratic_real_pole_k", integer=True)
    primitive = 1 / (k*k - 2)
    assert s.summation(telescope(primitive, k), (k, 0, s.oo)) == s.Rational(-1, 2)


def test_large_endpoint_uses_an_exact_closed_form():
    k = s.Symbol("quadratic_endpoint_k", integer=True)
    upper = 10**12
    source = telescope(1 / (k*k + 1), k)
    assert s.summation(source, (k, 0, upper)) == 1 - s.Rational(1, (upper + 1)**2 + 1)


@pytest.mark.parametrize("lower,upper", [(-7, 13), (18, -5), (4, 3)])
def test_finite_reversed_and_empty_ranges(lower, upper):
    k = s.Symbol("quadratic_range_k", integer=True)
    primitive = (k + 2) / (k*k + 1)**2
    expected = Fraction(lower + 2, (lower*lower + 1)**2) - Fraction(
        upper + 3, ((upper + 1)**2 + 1)**2
    )
    assert s.summation(telescope(primitive, k), (k, lower, upper)) == s.Rational(
        expected.numerator, expected.denominator
    )


def test_mixed_linear_and_quadratic_shift_orbits():
    k = s.Symbol("quadratic_mixed_k", integer=True)
    primitive = 1 / (k + 1) + (k + 2) / (k*k + 1)
    assert s.summation(telescope(primitive, k), (k, 0, s.oo)) == 3


def test_typed_dummy_axis_is_not_confused_with_a_same_named_symbol():
    k = s.Dummy("quadratic_same", integer=True)
    unrelated = s.Symbol("quadratic_same", positive=True)
    source = telescope(1 / (k*k + 1), k)
    assert s.summation(source, (k, 0, s.oo)) == 1
    result = s.summation(source, (unrelated, 0, 3))
    assert result.free_symbols == {k}
    assert next(iter(result.free_symbols)) is k


def test_original_removable_integer_pole_is_not_erased():
    k = s.Symbol("quadratic_held_k", integer=True)
    source = telescope(1 / (k*k + 1), k)
    held = s.Mul(k, s.Pow(k, -1, evaluate=False), source, evaluate=False)
    result = s.summation(held, (k, 0, s.oo))
    assert isinstance(result, s.Sum)


def test_nonzero_discrete_residue_remains_unevaluated():
    k = s.Symbol("quadratic_residue_k", integer=True)
    result = s.summation(1 / (k*k + 1), (k, 0, s.oo))
    assert isinstance(result, s.Sum)
