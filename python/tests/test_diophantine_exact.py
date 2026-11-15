"""Exact finite integer conics: complete sets, large offsets, and bounded work."""

from itertools import product

import pytest

from sympy import Eq, Integer, diophantine, symbols
from sympy.solvers.diophantine import (
    _bilinear_integer_solutions, _definite_binary_integer_points,
)


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



@pytest.mark.parametrize("coefficients, expected", [
    ((1, 0, 0, -6), {
        (-6, -1), (-3, -2), (-2, -3), (-1, -6),
        (1, 6), (2, 3), (3, 2), (6, 1),
    }),
    ((1, 3, -2, -12), {
        (-1, -5), (4, 0), (8, -2), (0, -6),
        (5, -1), (1, -9), (3, 3), (-4, -4),
    }),
    ((2, 3, 4, -5), {(-3, -7), (-13, -2), (-1, 4), (9, -1)}),
    ((-2, 3, -4, 5), {(-3, 2), (-1, 1)}),
    ((2, 1, 1, -4), {(0, 4), (1, 1), (4, 0), (-1, -5), (-2, -2), (-5, -1)}),
    ((2, 2, 2, 1), set()),
    ((5, 2, 2, 1), set()),
])
def test_shifted_bilinear_equations_return_every_congruence_admitted_point(coefficients, expected):
    b, d, e, f = coefficients
    points, x_fixed, y_fixed = _bilinear_integer_solutions(b, d, e, f)
    assert points == expected
    assert x_fixed is y_fixed is None
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    equation = b*x*y + d*x + e*y + f
    assert diophantine(equation) == expected
    assert diophantine(Eq(equation, 0, evaluate=False), syms=(y, x)) == {
        (yv, xv) for xv, yv in expected
    }


@pytest.mark.parametrize("coefficients, fixed", [
    ((1, 0, 0, 0), (0, 0)),
    ((1, 3, -2, -6), (2, -3)),
    ((2, 3, 4, 6), (-2, None)),
    ((2, 4, 3, 6), (None, -2)),
    ((6, 3, 2, 1), (None, None)),
    ((-2, -3, -4, -6), (-2, None)),
    ((5, 0, 5, 0), (-1, 0)),
])
def test_zero_product_retains_only_integral_affine_families(coefficients, fixed):
    b, d, e, f = coefficients
    points, x_fixed, y_fixed = _bilinear_integer_solutions(b, d, e, f)
    assert points == set()
    assert (x_fixed, y_fixed) == fixed
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    parameter = symbols("n1", integer=True)
    expected = set()
    if x_fixed is not None:
        expected.add((Integer(x_fixed), parameter))
        assert b*x_fixed + e == 0 and d*x_fixed + f == 0
    if y_fixed is not None:
        expected.add((parameter, Integer(y_fixed)))
        assert b*y_fixed + d == 0 and e*y_fixed + f == 0
    equation = b*x*y + d*x + e*y + f
    assert diophantine(equation) == expected
    assert diophantine(equation, param=symbols("tau", integer=True)) == expected
    assert diophantine(equation, syms=(y, x)) == {(yv, xv) for xv, yv in expected}


def test_small_bilinear_forms_match_independent_complete_integer_search():
    # For the finite cases in this corpus, abs(d*e-b*f) <= 10.
    # Every integer factor has magnitude at most 10; inversion gives
    # abs(x), abs(y) <= 12. Thus this fixed square covers the ENTIRE
    # finite solution set, independently of the generator's divisor loop.
    for b in (-3, -2, -1, 1, 2, 3):
        for d, e, f in product(range(-2, 3), repeat=3):
            coefficients = (0, b, 0, d, e, f)
            expected = _brute_force(coefficients, 12)
            points, x_fixed, y_fixed = _bilinear_integer_solutions(b, d, e, f)
            if d*e - b*f:
                assert x_fixed is y_fixed is None
                assert points == expected, coefficients
            else:
                assert points == set()
                represented = {
                    (x, y)
                    for x in range(-12, 13)
                    for y in range(-12, 13)
                    if (x_fixed is not None and x == x_fixed)
                    or (y_fixed is not None and y == y_fixed)
                }
                assert represented == expected, coefficients
                if x_fixed is not None:
                    assert b*x_fixed + e == d*x_fixed + f == 0
                if y_fixed is not None:
                    assert b*y_fixed + d == e*y_fixed + f == 0


@pytest.mark.parametrize("content", [1, -1, 10**200, -(10**200)])
def test_bilinear_translation_and_content_do_not_expand_divisor_work(content):
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    x_offset, y_offset = 10**400 + 7, -(10**300 + 11)
    finite = {(-6, -1), (-3, -2), (-2, -3), (-1, -6),
              (1, 6), (2, 3), (3, 2), (6, 1)}
    expected = {(x_offset + u, y_offset + v) for u, v in finite}
    equation = content*((x - x_offset)*(y - y_offset) - 6)
    assert diophantine(equation) == expected


def test_zero_product_families_do_not_require_divisor_enumeration():
    offset = 1 << 4095
    assert _bilinear_integer_solutions(1, 0, offset, 0) == (set(), -offset, 0)


@pytest.mark.parametrize("coefficients", [
    (1, 0, 0, -(10**30 + 1)),
    (1, -(10**400), -(10**400), 10**800 - 10**8),
])
def test_bilinear_work_refusal_cannot_return_known_but_incomplete_points(coefficients):
    with pytest.raises(NotImplementedError, match="bilinear.*bounded work limit"):
        _bilinear_integer_solutions(*coefficients)
    b, d, e, f = coefficients
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    with pytest.raises(NotImplementedError, match="bilinear.*bounded work limit"):
        diophantine(b*x*y + d*x + e*y + f)


def test_bilinear_coefficient_and_shape_refusals_are_explicit():
    with pytest.raises(NotImplementedError, match="coefficient height"):
        _bilinear_integer_solutions(1, 0, 0, 1 << 4096)
    with pytest.raises(NotImplementedError, match="exact integer coefficients"):
        _bilinear_integer_solutions(1.0, 0, 0, -1)
    with pytest.raises(NotImplementedError, match="nonzero mixed coefficient"):
        _bilinear_integer_solutions(0, 1, 1, -1)


def test_bilinear_expansion_does_not_misclassify_pell_equations():
    x, y = symbols("diophantine_x diophantine_y", integer=True)
    with pytest.raises(NotImplementedError, match="Pell-type"):
        diophantine(x*x - 2*y*y - 1)
