"""Exact causal time shifts retain complete polynomial impulse parts."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.integrals import transforms


def equal(actual, expected):
    assert s.expand(actual - expected) == 0


@pytest.mark.parametrize("order", [0, 1, 2, 5, 16, 32])
def test_delayed_monomials_shift_delta_derivatives_without_a_step(order):
    z, t = s.symbols("delayed_impulse_s delayed_impulse_t")
    result = s.inverse_laplace_transform(s.exp(-3*z/2)*z**order, z, t)
    argument = t-s.Rational(3, 2)
    expected = s.DiracDelta(argument) if order == 0 else s.DiracDelta(argument, order)
    assert result == expected
    assert not result.has(s.Heaviside)


def test_complete_improper_inverse_never_retries_legacy_tables(monkeypatch):
    z, t = s.symbols("delayed_improper_s delayed_improper_t")

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("a checked delayed distribution must use its native receipt")

    monkeypatch.setattr(transforms, "_inverse_laplace_table", forbidden_fallback)
    u = t-1
    result = s.inverse_laplace_transform(s.exp(-z)*z*z/(z+1), z, t)
    equal(result, s.DiracDelta(u, 1)-s.DiracDelta(u)+s.exp(-u)*s.Heaviside(u))
    # Impulse and regular pieces are separated at the native/public boundary.
    parts = _native.inverse_rational_laplace_native(
        _native_expr(s.exp(-z)*z*z/(z+1)), _symbol_binding(z), _symbol_binding(t),
    )
    assert len(parts) == 1
    assert len(parts[0]) == 3


def test_mixed_zero_delays_impulses_and_repeated_poles_keep_every_component():
    z, t = s.symbols("delayed_distribution_mixed_s delayed_distribution_mixed_t")
    source = z+s.exp(-z)*z*z/(z+1)+s.exp(-2*z)/(z*z+1)**2
    u, v = t-1, t-2
    expected = (s.DiracDelta(t, 1)+s.DiracDelta(u, 1)-s.DiracDelta(u)
                +s.exp(-u)*s.Heaviside(u)
                +(s.sin(v)-v*s.cos(v))*s.Heaviside(v)/2)
    equal(s.inverse_laplace_transform(source, z, t), expected)


def test_products_combine_equal_delay_impulses_and_keep_net_positive_delays():
    z, t = s.symbols("delayed_impulse_product_s delayed_impulse_product_t")
    source = s.Mul(
        s.Add(s.exp(-z), s.exp(-2*z), evaluate=False),
        s.Add(1, s.exp(-z), evaluate=False), z, evaluate=False,
    )
    expected = s.DiracDelta(t-1, 1)+2*s.DiracDelta(t-2, 1)+s.DiracDelta(t-3, 1)
    result = s.inverse_laplace_transform(source, z, t)
    equal(result, expected)
    assert not result.has(s.Heaviside)
    source = s.Mul(s.exp(z), s.exp(-2*z), z, evaluate=False)
    assert s.inverse_laplace_transform(source, z, t) == s.DiracDelta(t-1, 1)


def test_equal_delay_cancellation_retains_the_regular_remainder():
    z, t = s.symbols("delayed_impulse_cancel_s delayed_impulse_cancel_t")
    source = s.Add(
        s.Mul(s.exp(-z), z/(z+1), evaluate=False), -s.exp(-z), evaluate=False,
    )
    result = s.inverse_laplace_transform(source, z, t)
    equal(result, -s.exp(-(t-1))*s.Heaviside(t-1))
    assert not result.has(s.DiracDelta)


def test_conditions_and_doit_preserve_shifted_distributions():
    z, t = s.symbols("delayed_impulse_contract_s delayed_impulse_contract_t")
    source = s.exp(-z)*(3*z*z+2*z+1)
    result = s.inverse_laplace_transform(source, z, t)
    equal(result, 3*s.DiracDelta(t-1, 2)+2*s.DiracDelta(t-1, 1)+s.DiracDelta(t-1))
    assert s.inverse_laplace_transform(source, z, t, noconds=False) == (result, True)
    assert s.InverseLaplaceTransform(source, z, t, None).doit() == result


def test_both_shifted_components_retain_the_exact_public_time_owner():
    for z, t in [
        (s.Symbol("delayed_impulse_same"), s.Symbol("delayed_impulse_same", real=True)),
        (s.Dummy("delayed_impulse_dummy"), s.Dummy("delayed_impulse_dummy")),
    ]:
        result = s.inverse_laplace_transform(s.exp(-z)*z*z/(z+1), z, t)
        assert result.free_symbols == {t}
        assert next(iter(result.free_symbols)) is t
        assert all(next(iter(node.free_symbols)) is t
                   for node in result.atoms(s.DiracDelta))
    with pytest.raises(ValueError, match="distinct"):
        s.inverse_laplace_transform(s.exp(-z)*z, z, z)


def test_original_canceled_guard_does_not_erase_the_impulse():
    z, t = s.symbols("delayed_impulse_guard_s delayed_impulse_guard_t")
    one = s.Pow(s.Pow(z-100, -1, evaluate=False), 0, evaluate=False)
    source = s.Mul(s.exp(-z), one, z, evaluate=False)
    assert s.inverse_laplace_transform(source, z, t) == s.DiracDelta(t-1, 1)


@pytest.mark.parametrize("power", [-1, 0])
def test_early_impulses_cannot_hide_later_undefined_originals(power):
    z, t = s.symbols("delayed_impulse_invalid_s delayed_impulse_invalid_t")
    zero = s.Add(z, -z, evaluate=False)
    hidden = s.Pow(s.Pow(zero, -1, evaluate=False), power, evaluate=False)
    source = s.Add(
        s.Mul(s.exp(-z), z, evaluate=False),
        s.Mul(0, s.exp(-2*z), hidden, evaluate=False), evaluate=False,
    )
    with pytest.raises(ValueError):
        s.inverse_laplace_transform(source, z, t)


def test_unsupported_regular_tail_never_publishes_only_the_impulses():
    z, t = s.symbols("delayed_impulse_unresolved_s delayed_impulse_unresolved_t")
    source = s.exp(-z)*z+s.exp(-2*z)*(1+1/(z**3-z+1))
    result = s.inverse_laplace_transform(source, z, t)
    assert isinstance(result, s.InverseLaplaceTransform)
    assert result.args == (source, z, t)


def test_opaque_shell_function_is_never_mistaken_for_a_known_impulse_source():
    z, t = s.symbols("delayed_impulse_opaque_s delayed_impulse_opaque_t")
    opaque = s.Function("delayed_impulse_custom")(z)
    source = s.Mul(s.exp(-z), opaque, evaluate=False)
    original_args = source.args
    with pytest.raises(NotImplementedError):
        s.inverse_laplace_transform(source, z, t)
    assert source.args == original_args
    assert any(argument is opaque for argument in source.args)
    assert opaque.func is s.Function("delayed_impulse_custom")


def test_operational_failure_is_terminal_before_any_partial_publication(monkeypatch):
    z, t = s.symbols("delayed_impulse_failure_s delayed_impulse_failure_t")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise ValueError("delayed distribution verification budget exhausted")

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("operational failure must not retry a legacy table")

    monkeypatch.setattr(_native, "inverse_rational_laplace_native", exhausted)
    monkeypatch.setattr(transforms, "_inverse_laplace_table", forbidden_fallback)
    with pytest.raises(ValueError, match="verification budget exhausted"):
        s.inverse_laplace_transform(s.exp(-z)*z*z/(z+1), z, t)
    assert len(calls) == 1
