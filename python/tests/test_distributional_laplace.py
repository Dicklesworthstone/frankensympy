"""Causal rational inverses retain polynomial impulse parts explicitly."""

import pytest
import sympy as s


def equal(actual, expected):
    assert s.expand(actual - expected) == 0


@pytest.mark.parametrize("order", [0, 1, 2, 5, 16, 32])
def test_polynomial_monomials_produce_delta_derivatives(order):
    z, t = s.symbols("impulse_s impulse_t")
    source = s.Integer(1) if order == 0 else z**order
    result = s.inverse_laplace_transform(source, z, t)
    expected = s.DiracDelta(t) if order == 0 else s.DiracDelta(t, order)
    assert result == expected
    assert not result.has(s.Heaviside)


def test_signed_polynomial_impulses_have_no_step_multiplier():
    z, t = s.symbols("impulse_polynomial_s impulse_polynomial_t")
    result = s.inverse_laplace_transform(2*z**3 - 3*z + 5, z, t)
    equal(result, 2*s.DiracDelta(t, 3) - 3*s.DiracDelta(t, 1) + 5*s.DiracDelta(t))
    assert not result.has(s.Heaviside)


def test_improper_transform_contains_both_regular_and_singular_parts():
    z, t = s.symbols("impulse_mixed_s impulse_mixed_t")
    equal(s.inverse_laplace_transform(z/(z+1), z, t),
          s.DiracDelta(t) - s.exp(-t)*s.Heaviside(t))
    equal(s.inverse_laplace_transform((z**3 + 2*z + 1)/(z*z + 1), z, t),
          s.DiracDelta(t, 1) + (s.cos(t)+s.sin(t))*s.Heaviside(t))


def test_impulse_parts_combine_with_repeated_quadratic_poles():
    z, t = s.symbols("impulse_repeated_s impulse_repeated_t")
    source = z*z + s.Rational(2, 3) + 1/(z*z+1)**2
    expected = s.DiracDelta(t, 2) + s.Rational(2, 3)*s.DiracDelta(t)
    expected += (s.sin(t)-t*s.cos(t))*s.Heaviside(t)/2
    equal(s.inverse_laplace_transform(source, z, t), expected)


def test_conditions_and_unevaluated_transform_doit_preserve_impulses():
    z, t = s.symbols("impulse_contract_s impulse_contract_t")
    result = s.inverse_laplace_transform(z/(z+1), z, t)
    assert s.inverse_laplace_transform(z/(z+1), z, t, noconds=False) == (result, True)
    assert s.InverseLaplaceTransform(z/(z+1), z, t, None).doit() == result
    assert s.inverse_laplace_transform(0, z, t) == 0


def test_same_named_typed_variables_retain_the_output_owner():
    z = s.Symbol("impulse_same")
    t = s.Symbol("impulse_same", real=True)
    result = s.inverse_laplace_transform(z, z, t)
    assert result == s.DiracDelta(t, 1)
    assert result.free_symbols == {t}
    assert next(iter(result.free_symbols)) is t
    first, second = s.Dummy("impulse_dummy"), s.Dummy("impulse_dummy")
    result = s.inverse_laplace_transform(first, first, second)
    assert result == s.DiracDelta(second, 1)
    assert next(iter(result.free_symbols)) is second


def test_improper_input_still_checks_original_undefined_children():
    z, t = s.symbols("impulse_held_s impulse_held_t")
    invalid = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    source = s.Add(z, invalid, evaluate=False)
    with pytest.raises(ValueError):
        s.inverse_laplace_transform(source, z, t)


def test_unsupported_regular_tail_cannot_publish_only_the_impulse_part():
    z, t = s.symbols("impulse_unresolved_s impulse_unresolved_t")
    source = 1 + 1/(z**3-z+1)
    result = s.inverse_laplace_transform(source, z, t)
    assert isinstance(result, s.InverseLaplaceTransform)
    assert result.args == (source, z, t)
