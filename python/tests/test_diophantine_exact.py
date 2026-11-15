"""Exact finite integer conics: complete sets, large offsets, and bounded work."""

from itertools import product

import pytest

from sympy import Eq, Integer, diophantine, symbols
from sympy.solvers.diophantine import _definite_binary_integer_points


def _residual(coefficients, x, y):
    a, b, c, d, e, f = coefficients
    return a*x*x + b*x*y + c*y*y + d*x + e*y + f


def _brute_force(coefficients, radius):
    return {
        (x, y)
        for x in range(-radius, radius + 1)
        for y in range(-radius, radius + 1)
        if _residual(coefficients, x, y) == 0
    }


def _translated(coefficients, x_offset, y_offset):
    a, b, c, d, e, f = coefficients
    return (
        a, b, c,
        d - 2*a*x_offset - b*y_offset,
        e - b*x_offset - 2*c*y_offset,
        f + a*x_offset*x_offset + b*x_offset*y_offset + c*y_offset*y_offset
        - d*x_offset - e*y_offset,
    )


def _expression(coefficients, x, y):
    a, b, c, d, e, f = coefficients
    return a*x*x + b*x*y + c*y*y + d*x + e*y + f


@pytest.mark.parametrize("offset", [2**60 + 7, -(2**60 + 7), 10**400 + 7])
def test_translated_unit_circle_never_uses_float_bounds(offset):
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    equation = (x - offset)**2 + y**2 - 1
    expected = {(offset - 1, 0), (offset + 1, 0), (offset, -1), (offset, 1)}
    result = diophantine(equation)
    assert result == expected
    assert all(isinstance(value, Integer) for point in result for value in point)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("swap", [False, True])
def test_skinny_mixed_ellipse_uses_its_short_coordinate_projection(sign, swap):
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    scale = 10**100 + 1
    x_offset, y_offset = 10**200 + 7, -(10**180 + 3)
    # The x projection is more than 10**100 wide; the y projection has
    # three integers. Neither width nor the translation is a float.
    equation = (x - x_offset - scale*(y - y_offset))**2 + (y - y_offset)**2 - 1
    expected = {
        (x_offset - 1, y_offset), (x_offset + 1, y_offset),
        (x_offset - scale, y_offset - 1), (x_offset + scale, y_offset + 1),
    }
    if swap:
        equation = equation.xreplace({x: y, y: x})
        expected = {(b, a) for a, b in expected}
    result = diophantine(sign*equation)
    assert result == expected


@pytest.mark.parametrize("coefficients, expected", [
    ((2, 3, 2, 0, 0, -7), {(-1, -1), (1, 1)}),
    ((2, 3, 2, 0, 0, -1), {(-1, 1), (1, -1)}),
    ((1, 0, 1, 0, 0, -3), set()),
    ((1, 0, 1, 0, 0, 1), set()),
    ((1, 0, 1, -6, 8, 25), {(3, -4)}),
    ((4, 0, 4, -4, -4, 2), set()),
    ((4, 0, 1, -4, 0, 0), {(0, 0), (1, 0)}),
])
def test_definite_forms_keep_exact_boundaries_and_empty_sets(coefficients, expected):
    assert _definite_binary_integer_points(*coefficients) == expected
    assert _definite_binary_integer_points(*(-value for value in coefficients)) == expected
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    assert diophantine(_expression(coefficients, x, y)) == expected


def test_small_definite_forms_match_independent_complete_lattice_search():
    # For this corpus Q(x,y) >= (4*a*c-b*b)/(4*(a+c))*(x*x+y*y).
    # If r=max(abs(x),abs(y)) >= 16, Q exceeds abs(d*x+e*y+f)
    # already at r=16 and the lower bound grows faster thereafter. Thus
    # the fixed brute-force square covers EVERY integer solution without
    # using the implementation's discriminant or projection bounds.
    for a, c in product(range(1, 4), repeat=2):
        for b in range(-3, 4):
            determinant4 = 4*a*c - b*b
            if determinant4 <= 0:
                continue
            assert determinant4*16*16 > 4*(a + c)*(2*16 + 1)
            for d, e, f in product((-1, 0, 1), repeat=3):
                coefficients = (a, b, c, d, e, f)
                expected = _brute_force(coefficients, 15)
                assert _definite_binary_integer_points(*coefficients) == expected, coefficients


@pytest.mark.parametrize("coefficients", [
    (1, 0, 1, 0, 0, -25),
    (2, 3, 2, -1, 1, -7),
    (3, -2, 1, 1, -1, -8),
])
def test_translation_axis_exchange_and_content_preserve_complete_sets(coefficients):
    base = _brute_force(coefficients, 20)
    x_offset, y_offset = 10**200 + 39, -10**180 + 11
    moved = _translated(coefficients, x_offset, y_offset)
    expected = {(x + x_offset, y + y_offset) for x, y in base}
    for scale in (1, -1, 37, -37):
        scaled = tuple(scale*value for value in moved)
        assert _definite_binary_integer_points(*scaled) == expected
        a, b, c, d, e, f = scaled
        assert _definite_binary_integer_points(c, b, a, e, d, f) == {
            (y, x) for x, y in expected
        }


def test_large_eq_and_explicit_generator_order_preserve_integer_tuple_positions():
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    offset = 2**60 + 7
    equation = Eq((x - offset)**2 + (y + 3)**2, 1, evaluate=False)
    expected = {(-3, offset - 1), (-3, offset + 1), (-4, offset), (-2, offset)}
    assert diophantine(equation, syms=(y, x)) == expected


@pytest.mark.parametrize("coefficients", [
    (1, 0, 1, 0, 0, -10**20),
    _translated((1, 0, 1, 0, 0, -10**8), 10**400, -10**400),
])
def test_excessive_enumeration_refuses_instead_of_publishing_an_empty_or_partial_set(coefficients):
    with pytest.raises(NotImplementedError, match="bounded work limit"):
        _definite_binary_integer_points(*coefficients)
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    with pytest.raises(NotImplementedError, match="bounded work limit"):
        diophantine(_expression(coefficients, x, y))


def test_quadratic_coefficient_height_is_admitted_before_discriminant_products():
    with pytest.raises(NotImplementedError, match="coefficient height"):
        _definite_binary_integer_points(1 << 4096, 0, 1, 0, 0, -1)


@pytest.mark.parametrize("coefficients", [
    (1, 0, -1, 0, 0, -1),
    (1, 2, 1, 0, 0, -1),
    (0, 0, 1, 0, 0, -1),
])
def test_nondefinite_forms_do_not_enter_the_finite_lane(coefficients):
    with pytest.raises(NotImplementedError, match="requires a definite form"):
        _definite_binary_integer_points(*coefficients)


def test_exact_integer_lane_does_not_coerce_approximate_coefficients():
    with pytest.raises(NotImplementedError, match="exact integer coefficients"):
        _definite_binary_integer_points(1.0, 0, 1, 0, 0, -1)

