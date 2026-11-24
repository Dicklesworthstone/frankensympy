"""Real Lambert W values, precision, and complete-solver numeric consumers.

Decimal strings and binary64 anchors were obtained with isolated SymPy 1.14.0.
The comparator is exact displayed digits for evalf/N and relative error at the
shell's 15-digit conversion precision for float (with no absolute zero floor).
Held native source admission and explicit solveset results are separate native
contracts; upstream commonly retains ConditionSet for these equations.
"""

from decimal import Decimal
import math

import pytest
import sympy as s


NATIVE_ONLY = pytest.mark.skipif(
    not hasattr(s.Matrix([[1]]), "_native_storage"),
    reason="native real Lambert source and complete-solver extension",
)

# name, original exact argument, real branch, 15 digits, 40 digits, float
CASES = (
    (
        "principal_one", s.Integer(1), 0,
        "0.567143290409784",
        "0.5671432904097838729999686622103555497538",
        0.5671432904097838,
    ),
    (
        "negative_principal", -s.Rational(1, 4), 0,
        "-0.357402956181389",
        "-0.3574029561813889030688111040559047533166",
        -0.3574029561813889,
    ),
    (
        "negative_lower", -s.Rational(1, 4), -1,
        "-2.15329236411035",
        "-2.153292364110349649169099150092981375536",
        -2.15329236411035,
    ),
    (
        "large_principal", s.Integer(10)**20, 0,
        "42.3067550917384",
        "42.30675509173839385177115930518188403470",
        42.306755091738395,
    ),
    (
        "lower_tail", -s.Rational(1, 10**100), -1,
        "-235.721158875685",
        "-235.7211588756853136604606061305238190409",
        -235.72115887568532,
    ),
    (
        "tiny_positive", s.Rational(1, 10**80), 0,
        "1.00000000000000e-80",
        "1.000000000000000000000000000000000000000e-80",
        1e-80,
    ),
    (
        "tiny_negative", -s.Rational(1, 10**80), 0,
        "-1.00000000000000e-80",
        "-1.000000000000000000000000000000000000000e-80",
        -1e-80,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
@pytest.mark.parametrize("digits", [15, 40])
def test_evalf_and_N_preserve_requested_real_decimal_digits(case, digits):
    _, argument, branch, short, long, _ = case
    value = s.LambertW(argument, branch)
    expected = short if digits == 15 else long
    for result in (value.evalf(digits), s.N(value, digits)):
        assert isinstance(result, s.Float)
        assert result.is_real is True
        assert str(result) == expected


@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
def test_float_conversion_is_finite_branch_correct_and_keeps_tiny_signs(case):
    _, argument, branch, _, _, reference = case
    value = float(s.LambertW(argument, branch))
    assert math.isfinite(value)
    assert value == pytest.approx(reference, rel=5e-15, abs=0.0)
    if branch == -1:
        assert value < -1
    elif argument < 0:
        assert -1 < value < 0
    else:
        assert value > 0


@pytest.mark.parametrize("argument,branch,expected", [
    (s.Integer(0), 0, 0),
    (s.E, 0, 1),
    (-1/s.E, 0, -1),
    (-1/s.E, -1, -1),
])
def test_exact_constructor_values_include_the_shared_real_branch_point(
    argument, branch, expected,
):
    value = s.LambertW(argument, branch)
    assert value == s.Integer(expected)
    assert float(value) == float(expected)
    assert Decimal(str(value.evalf(25))) == Decimal(expected)


@NATIVE_ONLY
@pytest.mark.parametrize("branch", [0, -1])
@pytest.mark.parametrize("form", ["exp", "power"])
def test_held_exact_branch_point_reaches_native_admission_before_exact_folding(branch, form):
    from sympy.core import _native, _native_expr

    reciprocal_e = (s.exp(-1, evaluate=False) if form == "exp" else
                    s.Pow(s.E, -1, evaluate=False))
    argument = s.Mul(-1, reciprocal_e, evaluate=False)
    value = s.LambertW(argument, branch, evaluate=False)
    decimal = _native.evalf_decimal_native(_native_expr(value), 25)
    assert Decimal(decimal) == Decimal(-1)


@NATIVE_ONLY
def test_defined_held_argument_reaches_native_numeric_evaluation():
    from sympy.core import _native, _native_expr

    argument = s.Add(1, s.Mul(0, 2, evaluate=False), evaluate=False)
    value = s.LambertW(argument, evaluate=False)
    decimal = _native.evalf_decimal_native(_native_expr(value), 40)
    assert Decimal(decimal) == Decimal(CASES[0][4])


@NATIVE_ONLY
@pytest.mark.parametrize("branch", [0, -1])
@pytest.mark.parametrize("wrapper", ["zero_product", "zero_power", "nonreal_product"])
def test_original_invalid_children_cannot_be_folded_out_of_native_numeric_input(branch, wrapper):
    from sympy.core import _native, _native_expr

    pole = s.log(0, evaluate=False)
    if wrapper == "zero_product":
        argument = s.Mul(0, pole, evaluate=False)
    elif wrapper == "zero_power":
        argument = s.Pow(pole, 0, evaluate=False)
    else:
        argument = s.Mul(0, s.I, evaluate=False)
    value = s.LambertW(argument, branch, evaluate=False)
    with pytest.raises(ValueError):
        _native.evalf_decimal_native(_native_expr(value), 25)


@NATIVE_ONLY
@pytest.mark.parametrize("equation", ["positive", "negative", "affine"])
def test_complete_real_solver_roots_are_usable_by_evalf_and_float(equation):
    x = s.Symbol("lambert_numeric_solution")
    if equation == "positive":
        source = x*s.exp(x) - 1
        expected_roots = (s.LambertW(1),)
        expected_decimals = (CASES[0][4],)
        residual = lambda value: value*math.exp(value) - 1
    elif equation == "negative":
        source = x*s.exp(x) + s.Rational(1, 4)
        expected_roots = (s.LambertW(-s.Rational(1, 4)),
                          s.LambertW(-s.Rational(1, 4), -1))
        expected_decimals = (CASES[1][4], CASES[2][4])
        residual = lambda value: value*math.exp(value) + 0.25
    else:
        source = s.exp(2*x + 3) - 3*(2*x + 3)
        expected_roots = tuple(
            (-s.LambertW(-s.Rational(1, 3), branch) - 3)/2
            for branch in (0, -1)
        )
        expected_decimals = (
            "-1.190469356632027443923836502989538883349",
            "-0.7439327241710787630516301609639806476982",
        )
        residual = lambda value: math.exp(2*value + 3) - 3*(2*value + 3)
    result = s.solveset(source, x, s.S.Reals)
    assert result == s.FiniteSet(*expected_roots)
    assert sorted(str(root.evalf(40)) for root in result) == sorted(expected_decimals)
    for root in result:
        value = float(root)
        assert math.isfinite(value)
        # This is a binary64 end-to-end check, not a root certificate.
        assert abs(residual(value)) < 4e-13
