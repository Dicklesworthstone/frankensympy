"""Source-bound proper integrals across continuous absolute-value corners."""

from fractions import Fraction

import pytest
import sympy as s


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_triangle_area_crosses_the_corner_without_derivative_claims(reverse):
    x = s.Symbol("absolute_integral_x", real=True)
    limits = (x, 1, -1) if reverse else (x, -1, 1)
    receipt = s.certified_integral(
        s.Abs(x), limits, tolerance=Fraction(1, 10**100), max_intervals=2,
    )
    expected = -1 if reverse else 1
    assert receipt.bounds == (expected, expected)
    assert receipt.interval == limits[1:]
    assert receipt.partition == (-1, 0, 1)
    assert receipt.interval_count == 2
    assert receipt.error_bound == 0
    assert "proper real integral" in receipt.claim
    for derivative_api in (
        lambda: s.enclose_residual_system([s.Abs(x)], x, [(-1, 1)]),
        lambda: s.enclose_objective(s.Abs(x), x, [(-1, 1)]),
    ):
        with pytest.raises(ValueError):
            derivative_api()


@pytest.mark.parametrize("case", ["shifted", "nested", "product"])
def test_adaptive_compositions_match_independent_exact_integrals(case):
    x = s.Symbol("composed_absolute_integral_x", real=True)
    if case == "shifted":
        source, limits, expected = s.Abs(x-s.Rational(1, 3)), (x, 0, 1), s.Rational(5, 18)
    elif case == "nested":
        source, limits, expected = s.Abs(s.Abs(x)-s.Rational(1, 2)), (x, -1, 1), s.Rational(1, 2)
    else:
        source, limits, expected = s.Abs(x)*x**2, (x, -1, 1), s.Rational(1, 2)
    receipt = s.certified_integral(source, limits, tolerance=Fraction(1, 10000), precision=16)
    lower, upper = receipt.bounds
    assert lower <= expected <= upper
    assert upper-lower <= s.Rational(1, 10000)
    assert receipt.integrand == source
    assert receipt.interval == limits[1:]
    assert receipt.interval_count > 1
    assert receipt.partition[0] == limits[1]
    assert receipt.partition[-1] == limits[2]


def test_zero_length_corner_is_defined_but_hidden_poles_still_refuse():
    x = s.Symbol("point_absolute_integral_x")
    receipt = s.certified_integral(s.Abs(x), (x, 0, 0))
    assert receipt.bounds == (0, 0)
    assert receipt.interval_count == 0
    assert receipt.partition == (0,)
    reciprocal = s.Pow(x, -1, evaluate=False)
    for source in (
        s.Mul(s.Abs(x), reciprocal, evaluate=False),
        s.Mul(s.Abs(x), 0, reciprocal, evaluate=False),
        s.Add(s.Abs(x), s.Pow(reciprocal, 0, evaluate=False), evaluate=False),
    ):
        for limits in ((x, -1, 1), (x, 0, 0)):
            with pytest.raises(ValueError):
                s.certified_integral(source, limits, max_intervals=8)


def test_one_cell_cannot_claim_a_false_exact_cusp_integral():
    x = s.Symbol("coarse_absolute_integral_x")
    with pytest.raises(ValueError, match="width"):
        s.certified_integral(s.Abs(x), (x, -1, 1), tolerance=Fraction(1, 100), max_intervals=1)


def test_stable_piecewise_values_keep_lazy_branches_and_guard_obligations():
    x = s.Symbol("piecewise_absolute_integral_x", real=True)
    source = s.Piecewise((s.Abs(x), x < 2), (s.log(0, evaluate=False), True), evaluate=False)
    receipt = s.certified_integral(source, (x, -1, 1), max_intervals=4)
    assert receipt.bounds == (1, 1)
    unstable = s.Piecewise((s.Abs(x), x < 0), (1, True), evaluate=False)
    with pytest.raises(ValueError):
        s.certified_integral(unstable, (x, -1, 1), max_intervals=8)


def test_literal_tiny_scale_is_not_lost_at_a_corner():
    x = s.Symbol("tiny_absolute_integral_x")
    scale = s.Rational(1, 10**400)
    receipt = s.certified_integral(
        scale*s.Abs(x), (x, -1, 1), tolerance=Fraction(1, 10**600),
        precision=8, max_intervals=2,
    )
    assert receipt.bounds == (scale, scale)
    assert receipt.error_bound == 0
