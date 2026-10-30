"""Checked polynomial-weighted Laplace transforms and legacy routing boundaries."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def rational_pair(expression):
    """Independent algebraic cross multiplication, without rational simplify."""
    expression = s.sympify(expression)
    if isinstance(expression, s.Add):
        numerator, denominator = s.Integer(0), s.Integer(1)
        for child in expression.args:
            right_numerator, right_denominator = rational_pair(child)
            numerator, denominator = (
                numerator*right_denominator + right_numerator*denominator,
                denominator*right_denominator,
            )
        return numerator, denominator
    if isinstance(expression, s.Mul):
        numerator, denominator = s.Integer(1), s.Integer(1)
        for child in expression.args:
            right_numerator, right_denominator = rational_pair(child)
            numerator *= right_numerator
            denominator *= right_denominator
        return numerator, denominator
    if isinstance(expression, s.Pow):
        base, exponent = expression.args
        power = int(exponent)
        assert exponent == power
        numerator, denominator = rational_pair(base)
        if power < 0:
            numerator, denominator = denominator, numerator
            power = -power
        return numerator**power, denominator**power
    return expression, s.Integer(1)


def equal_rational(actual, expected):
    actual_numerator, actual_denominator = rational_pair(actual)
    expected_numerator, expected_denominator = rational_pair(expected)
    assert s.expand(actual_denominator) != 0
    assert s.expand(expected_denominator) != 0
    assert s.expand(
        actual_numerator*expected_denominator
        - expected_numerator*actual_denominator
    ) == 0


def test_weighted_trigonometric_derivative_orders_are_complete():
    t, z = s.symbols("forward_weighted_t forward_weighted_s")
    cases = [
        (t*t*s.sin(t), (6*z*z-2)/(z*z+1)**3),
        (t**3*s.cos(t), 6*(z**4-6*z*z+1)/(z*z+1)**4),
        (t*t*s.exp(-3*t)*s.sin(2*t), (12*z*z+72*z+92)/(z*z+6*z+13)**3),
    ]
    for source, expected in cases:
        equal_rational(s.laplace_transform(source, t, z, noconds=True), expected)


def test_distributed_polynomials_and_combined_rates_remain_exact():
    t, z = s.symbols("forward_distributed_t forward_distributed_s")
    equal_rational(
        s.laplace_transform((t+1)*(t+2)*s.sin(t), t, z, noconds=True),
        (6*z*z-2)/(z*z+1)**3 + 6*z/(z*z+1)**2 + 2/(z*z+1),
    )
    transform, abscissa, condition = s.laplace_transform(
        t**4*s.exp(2*t)*s.exp(-5*t), t, z,
    )
    equal_rational(transform, 24/(z+3)**5)
    assert (abscissa, condition) == (-3, True)


@pytest.mark.parametrize("frequency", [-3, -1, s.Rational(1, 2), 2])
def test_signed_and_rational_frequencies_use_exact_coefficients(frequency):
    t, z = s.symbols("forward_frequency_t forward_frequency_s")
    result = s.laplace_transform(t*t*s.sin(frequency*t), t, z)
    equal_rational(
        result[0],
        2*frequency*(3*z*z-frequency*frequency)/(z*z+frequency*frequency)**3,
    )
    assert result[1:] == (0, True)


def test_hyperbolic_cancellation_controls_the_exact_convergence_boundary():
    t, z = s.symbols("forward_hyperbolic_t forward_hyperbolic_s")
    transform, abscissa, condition = s.laplace_transform(t*t*s.cosh(3*t), t, z)
    equal_rational(transform, 2*z*(z*z+27)/(z*z-9)**3)
    assert (abscissa, condition) == (3, True)
    transform, abscissa, condition = s.laplace_transform(
        t**3*(s.cosh(2*t)-s.sinh(2*t)), t, z,
    )
    equal_rational(transform, 6/(z+2)**4)
    assert (abscissa, condition) == (-2, True)


def test_equal_rate_cancellation_does_not_leave_a_false_larger_abscissa():
    t, z = s.symbols("forward_cancelled_rate_t forward_cancelled_rate_s")
    source = s.Add(
        s.Mul(t, s.exp(3*t), evaluate=False),
        s.Mul(-1, t, s.exp(3*t), evaluate=False),
        s.exp(-2*t),
        evaluate=False,
    )
    transform, abscissa, condition = s.laplace_transform(source, t, z)
    equal_rational(transform, 1/(z+2))
    assert (abscissa, condition) == (-2, True)


def test_held_zero_source_has_an_everywhere_convergent_transform():
    t, z = s.symbols("forward_zero_t forward_zero_s")
    source = s.Mul(
        t*t, s.Add(s.sin(t), -s.sin(t), evaluate=False), evaluate=False,
    )
    assert s.laplace_transform(source, t, z) == (0, -s.oo, True)


def test_public_noconds_and_transform_doit_use_the_checked_path():
    t, z = s.symbols("forward_doit_t forward_doit_s")
    source = t*t*s.sin(t)
    result = s.laplace_transform(source, t, z, noconds=True)
    assert s.laplace_transform(source, t, z) == (result, 0, True)
    assert s.LaplaceTransform(source, t, z).doit() == result


def test_typed_symbols_and_dummy_outputs_preserve_the_original_owners():
    t = s.Symbol("forward_shared_name")
    z = s.Symbol("forward_shared_name", real=True)
    transform = s.laplace_transform(t*t*s.sin(t), t, z, noconds=True)
    assert transform.free_symbols == {z}
    assert next(iter(transform.free_symbols)) is z
    first, second = s.Dummy("forward_dummy"), s.Dummy("forward_dummy")
    transform = s.laplace_transform(first*first*s.sin(first), first, second, noconds=True)
    assert transform.free_symbols == {second}
    assert next(iter(transform.free_symbols)) is second
    with pytest.raises(ValueError, match="distinct"):
        s.laplace_transform(t*s.exp(-t), t, t)


def test_existing_symbolic_coefficient_and_heaviside_tables_remain_available():
    t, z, a = s.symbols("forward_table_t forward_table_s forward_table_a")
    for source in [s.exp(a*t), s.sin(a*t), s.Heaviside(t-1)]:
        assert _native.polynomial_laplace_native(
            _native_expr(source), _symbol_binding(t), _symbol_binding(z),
        ) is None
    equal_rational(s.laplace_transform(s.exp(a*t), t, z, noconds=True), 1/(z-a))
    equal_rational(s.laplace_transform(s.sin(a*t), t, z, noconds=True), a/(z*z+a*a))
    assert s.laplace_transform(s.Heaviside(t-1), t, z) == (s.exp(-z)/z, 0, True)


@pytest.mark.parametrize("kind", [
    "fractional", "negative_fractional", "symbolic", "mixed_sum",
    "weighted_fractional", "nonlinear_phase",
])
def test_forward_declared_unsupported_forms_keep_legacy_errors(kind):
    t, z, a = s.symbols("forward_legacy_t forward_legacy_s forward_legacy_a")
    sources = {
        "fractional": t**s.Rational(1, 2),
        "negative_fractional": t**s.Rational(-1, 2),
        "symbolic": t**a,
        "mixed_sum": t**a + s.sin(t),
        "weighted_fractional": t**s.Rational(1, 2)*s.exp(-t),
        "nonlinear_phase": s.exp(t*t),
    }
    with pytest.raises(ValueError):
        s.laplace_transform(sources[kind], t, z, noconds=True)


@pytest.mark.parametrize("zero_kind", ["literal", "time_variable", "parameter"])
@pytest.mark.parametrize("exponent", [s.Integer(-1), s.Integer(0), s.Rational(-1, 2)])
@pytest.mark.parametrize("fallback", ["fractional", "symbolic"])
def test_forward_fallback_keeps_every_held_undefined_power(zero_kind, exponent, fallback):
    t, z, a = s.symbols("forward_held_t forward_held_s forward_held_a")
    zero = {
        "literal": s.Integer(0),
        "time_variable": s.Add(t, -t, evaluate=False),
        "parameter": s.Add(a, -a, evaluate=False),
    }[zero_kind]
    invalid = s.Pow(zero, exponent, evaluate=False)
    other = t**s.Rational(1, 2) if fallback == "fractional" else t**a
    source = s.Add(t, s.Mul(0, invalid, other, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="zero base|undefined"):
        s.laplace_transform(source, t, z, noconds=True)


def test_forward_fallback_admits_nested_children_before_outer_zero_power():
    t, z, a = s.symbols("forward_nested_t forward_nested_s forward_nested_a")
    for exponent in [s.Integer(-1), s.Rational(-1, 2)]:
        inner = s.Pow(s.Add(a, -a, evaluate=False), exponent, evaluate=False)
        hidden = s.Pow(inner, 0, evaluate=False)
        source = s.Add(t, s.Mul(0, hidden, t**a, evaluate=False), evaluate=False)
        with pytest.raises(ValueError, match="zero base|undefined"):
            s.laplace_transform(source, t, z, noconds=True)


@pytest.mark.parametrize("exponent", [s.Rational(1, 2), s.Rational(3, 2)])
def test_forward_positive_fractional_zero_base_remains_valid(exponent):
    t, z = s.symbols("forward_positive_zero_t forward_positive_zero_s")
    source = s.Pow(0, exponent, evaluate=False)
    assert s.laplace_transform(source, t, z, noconds=True) == 0


def test_forward_preserves_larger_polynomials_and_declared_larger_weighted_poles():
    t, z = s.symbols("forward_large_t forward_large_s")
    for source in [t**64, t**32*s.exp(-t)]:
        assert _native.polynomial_laplace_native(
            _native_expr(source), _symbol_binding(t), _symbol_binding(z),
        ) is None
    assert s.expand(s.laplace_transform(t**64, t, z, noconds=True)
                    - s.factorial(64)/z**65) == 0
    assert s.expand(s.laplace_transform(t**32*s.exp(-t), t, z, noconds=True)
                    - s.factorial(32)/(z+1)**33) == 0


def test_unknown_original_held_domains_cannot_become_an_ordinary_answer():
    t, z, a = s.symbols("forward_unknown_t forward_unknown_s forward_unknown_a")
    zero = s.Add(s.sin(t), -s.sin(t), evaluate=False)
    invalid = s.Pow(zero, -1, evaluate=False)
    hidden = s.Mul(0, s.Pow(invalid, 0, evaluate=False), t**a, evaluate=False)
    source = s.Add(t, hidden, evaluate=False)
    result = s.laplace_transform(source, t, z, noconds=True)
    assert isinstance(result, s.LaplaceTransform)
    assert result.args == (source, t, z)


def test_checked_path_operational_failure_never_retries_legacy(monkeypatch):
    t, z = s.symbols("forward_failure_t forward_failure_s")
    calls = []

    def exhausted(*args):
        calls.append("checked")
        raise ValueError("forward verification budget exhausted")

    def legacy(*args):
        calls.append("legacy")
        raise AssertionError("legacy retry")

    monkeypatch.setattr(_native, "polynomial_laplace_native", exhausted)
    monkeypatch.setattr(_native, "laplace_expr", legacy)
    with pytest.raises(ValueError, match="forward verification budget exhausted"):
        s.laplace_transform(t*s.exp(-t), t, z, noconds=True)
    assert calls == ["checked"]


def test_source_size_and_integer_power_limits_stay_terminal():
    t, z = s.symbols("forward_limit_t forward_limit_s")
    with pytest.raises(ValueError, match="4096 bits"):
        s.laplace_transform(1 << 4097, t, z)
    source = s.Mul(s.exp(t), s.Pow(t, 10**9, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="power|envelope"):
        s.laplace_transform(source, t, z)

    class CustomSymbol(s.Symbol):
        pass

    with pytest.raises(NotImplementedError):
        s.laplace_transform(CustomSymbol("forward_custom"), t, z)


@pytest.mark.parametrize("exponent", [-100, 100])
def test_large_held_constant_powers_route_before_the_smaller_native_envelope(exponent):
    t, z = s.symbols("forward_constant_power_t forward_constant_power_s")
    coefficient = s.Pow(2, exponent, evaluate=False)
    source = s.Mul(coefficient, s.exp(t), evaluate=False)
    assert _native.polynomial_laplace_native(
        _native_expr(source), _symbol_binding(t), _symbol_binding(z),
    ) is None
    equal_rational(
        s.laplace_transform(source, t, z, noconds=True),
        s.Integer(2)**exponent/(z-1),
    )


def test_held_polynomial_zero_power_routes_before_nonvanishing_witness_admission():
    t, z = s.symbols("forward_zero_power_t forward_zero_power_s")
    source = s.Mul(s.Pow(t+1, 0, evaluate=False), s.exp(-t), evaluate=False)
    assert _native.polynomial_laplace_native(
        _native_expr(source), _symbol_binding(t), _symbol_binding(z),
    ) is None
    # The established table refuses this held power. Recognition must retain
    # that ValueError instead of invoking the smaller checked source envelope.
    with pytest.raises(ValueError, match="Laplace transform not implemented"):
        s.laplace_transform(source, t, z, noconds=True)


def test_single_exponential_zero_power_keeps_its_native_nonvanishing_witness():
    t, z = s.symbols("forward_exp_zero_t forward_exp_zero_s")
    source = s.Mul(s.Pow(s.exp(t), 0, evaluate=False), s.exp(-t), evaluate=False)
    assert _native.polynomial_laplace_native(
        _native_expr(source), _symbol_binding(t), _symbol_binding(z),
    ) is not None
    equal_rational(s.laplace_transform(source, t, z, noconds=True), 1/(z+1))
