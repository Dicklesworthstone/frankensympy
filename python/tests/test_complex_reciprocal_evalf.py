"""Public complex reciprocals use justified decimals and one caller budget.

Reference values use independent Decimal Taylor sums and the direct complex
sine/cosine/sinh/cosh formulas. No candidate kernel or binary64 probe supplies
the reference. These tests require the actual mathematical PyO3 extension.
"""

from decimal import Decimal, ROUND_HALF_EVEN, localcontext

import pytest
import sympy as s

from sympy.core import _ExactDecimalFloat, _native, _native_expr


NAMES = ("cot", "sec", "csc", "coth", "sech", "csch")


def ample_budget(**limits):
    options = dict(
        max_steps=1 << 45,
        max_memory_bytes=1 << 40,
        max_allocations=1 << 30,
    )
    options.update(limits)
    return s.EvaluationBudget(**options)


def rounded(value, digits):
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_HALF_EVEN
        return +value


def sin_cos(value):
    sine_term = sine = value
    cosine_term = cosine = Decimal(1)
    squared = value * value
    for k in range(1, 200):
        sine_term *= -squared / ((2 * k) * (2 * k + 1))
        cosine_term *= -squared / ((2 * k - 1) * (2 * k))
        sine += sine_term
        cosine += cosine_term
    return sine, cosine


def sinh_cosh(value):
    positive, negative = value.exp(), (-value).exp()
    return (positive - negative) / 2, (positive + negative) / 2


def quotient(numerator, denominator):
    a, b = numerator
    c, d = denominator
    norm = c * c + d * d
    return (a * c + b * d) / norm, (b * c - a * d) / norm


def reference(name, real, imaginary, digits):
    # The tested coordinates have magnitude at most one. Two hundred Taylor
    # terms and fifty guard digits separate these requested rounding bins.
    with localcontext() as context:
        context.prec = digits + 50
        context.rounding = ROUND_HALF_EVEN
        x, y = Decimal(real), Decimal(imaginary)
        sin_x, cos_x = sin_cos(x)
        sin_y, cos_y = sin_cos(y)
        sinh_x, cosh_x = sinh_cosh(x)
        sinh_y, cosh_y = sinh_cosh(y)
        sine = sin_x * cosh_y, cos_x * sinh_y
        cosine = cos_x * cosh_y, -sin_x * sinh_y
        hyperbolic_sine = sinh_x * cos_y, cosh_x * sin_y
        hyperbolic_cosine = cosh_x * cos_y, sinh_x * sin_y
        numerator, denominator = {
            "cot": (cosine, sine),
            "sec": ((Decimal(1), Decimal(0)), cosine),
            "csc": ((Decimal(1), Decimal(0)), sine),
            "coth": (hyperbolic_cosine, hyperbolic_sine),
            "sech": ((Decimal(1), Decimal(0)), hyperbolic_cosine),
            "csch": ((Decimal(1), Decimal(0)), hyperbolic_sine),
        }[name]
        result = quotient(numerator, denominator)
    return tuple(rounded(component, digits) for component in result)


def evaluate(source, digits, entry, budgeted):
    options = {"budget": ample_budget()} if budgeted else {}
    if entry == "N":
        return s.N(source, digits, **options)
    return getattr(source, entry)(digits, **options)


def assert_components(result, expected, digits):
    parts = result.as_real_imag()
    assert tuple(Decimal(str(part)) for part in parts) == expected
    for part, value in zip(parts, expected):
        if value == 0:
            assert part is s.S.Zero
        else:
            assert type(part) is _ExactDecimalFloat
            assert part.dps == digits
            assert part.is_finite is True
            assert part.is_zero is False


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("digits", [30, 60])
@pytest.mark.parametrize("budgeted", [False, True])
def test_public_complex_reciprocals_keep_independent_high_precision_digits(
    name, digits, budgeted,
):
    source = getattr(s, name)(1 + s.I, evaluate=False)
    expected = reference(name, 1, 1, digits)
    original_args, original_native = source.args, _native_expr(source)
    for entry in ("N", "evalf", "n"):
        result = evaluate(source, digits, entry, budgeted)
        assert_components(result, expected, digits)
    assert source.args is original_args
    assert _native_expr(source) == original_native


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("coordinates", [(1, 0), (0, 1), (1, -1), (-1, 1)])
@pytest.mark.parametrize("budgeted", [False, True])
def test_axes_and_both_complex_half_planes_keep_correct_component_signs(
    name, coordinates, budgeted,
):
    x, y = coordinates
    source = getattr(s, name)(x + y * s.I, evaluate=False)
    assert_components(
        evaluate(source, 40, "N", budgeted),
        reference(name, x, y, 40),
        40,
    )


@pytest.mark.parametrize("budgeted", [False, True])
def test_logarithmic_axis_identities_preserve_exact_zero_components(budgeted):
    log_two = s.log(2, evaluate=False)
    for name, argument, real, imaginary in (
        ("cot", s.I * log_two, s.S.Zero, -s.Rational(5, 3)),
        ("sec", s.I * log_two, s.Rational(4, 5), s.S.Zero),
        ("csc", s.I * log_two, s.S.Zero, -s.Rational(4, 3)),
        ("coth", log_two, s.Rational(5, 3), s.S.Zero),
        ("sech", log_two, s.Rational(4, 5), s.S.Zero),
        ("csch", log_two, s.Rational(4, 3), s.S.Zero),
    ):
        # Rational expectations are exact before the requested decimal rounding.
        with localcontext() as context:
            context.prec = 90
            expected = tuple(
                rounded(Decimal(str(value.p)) / Decimal(str(value.q)), 50)
                for value in (real, imaginary)
            )
        source = getattr(s, name)(argument, evaluate=False)
        assert_components(evaluate(source, 50, "N", budgeted), expected, 50)
    for name in ("sec", "sech"):
        source = getattr(s, name)(0, evaluate=False)
        assert_components(
            evaluate(source, 50, "N", budgeted), (Decimal(1), Decimal(0)), 50,
        )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("budgeted", [False, True])
def test_complex_reciprocals_never_restart_in_real_or_binary64_fallback(
    monkeypatch, name, budgeted,
):
    source = getattr(s, name)(1 + s.I, evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("complex reciprocal selected a legacy fallback")

    monkeypatch.setattr(_native, "evalf_decimal_native", forbidden)
    monkeypatch.setattr(_native, "complex_value_native", forbidden)
    assert_components(
        evaluate(source, 50, "N", budgeted), reference(name, 1, 1, 50), 50,
    )


@pytest.mark.parametrize("name, argument", [
    ("cot", 0), ("csc", 0), ("coth", 0), ("csch", 0),
    ("sec", s.pi / 2), ("sech", s.I * s.pi / 2),
])
@pytest.mark.parametrize("budgeted", [False, True])
def test_poles_refuse_without_a_result_or_fallback(
    monkeypatch, name, argument, budgeted,
):
    source = getattr(s, name)(argument, evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("a pole restarted in a legacy fallback")

    monkeypatch.setattr(_native, "evalf_decimal_native", forbidden)
    monkeypatch.setattr(_native, "complex_value_native", forbidden)
    with pytest.raises(ValueError):
        evaluate(source, 30, "N", budgeted)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("budgeted", [False, True])
def test_original_undefined_children_cannot_disappear_in_composition(name, budgeted):
    invalid = s.log(0, evaluate=False)
    for argument in (
        s.Mul(0, invalid, evaluate=False),
        s.Pow(invalid, 0, evaluate=False),
    ):
        source = getattr(s, name)(argument, evaluate=False)
        original_args, original_native = source.args, _native_expr(source)
        with pytest.raises(ValueError):
            evaluate(source, 30, "N", budgeted)
        assert source.args is original_args
        assert _native_expr(source) == original_native


@pytest.mark.parametrize("name", NAMES)
def test_caller_budget_is_cumulative_across_public_and_native_evaluations(name):
    source = getattr(s, name)(1 + s.I, evaluate=False)
    budget = ample_budget()
    first = s.N(source, 30, budget=budget)
    remaining = budget.remaining_steps
    second = source.evalf(30, budget=budget)
    assert second == first
    assert 0 < budget.remaining_steps < remaining < 1 << 45
    remaining = budget.remaining_steps
    pair = _native.evalf_complex_decimal_native(
        _native_expr(source), 30, budget=budget,
    )
    assert tuple(Decimal(value) for value in pair) == reference(name, 1, 1, 30)
    assert 0 < budget.remaining_steps < remaining


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("limit", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_resource_exhaustion_is_a_terminal_operational_failure(name, limit):
    source = getattr(s, name)(1 + s.I, evaluate=False)
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.N(source, 30, budget=ample_budget(**{limit: 0}))


@pytest.mark.parametrize("name", NAMES)
def test_cancelled_caller_cannot_start_new_arithmetic(name):
    source = getattr(s, name)(1 + s.I, evaluate=False)
    budget = ample_budget()
    budget.cancel()
    remaining = budget.remaining_steps
    for entry in ("N", "evalf", "n"):
        options = {"budget": budget}
        with pytest.raises(s.EvaluationCancelled):
            if entry == "N":
                s.N(source, 30, **options)
            else:
                getattr(source, entry)(30, **options)
        assert budget.remaining_steps == remaining
