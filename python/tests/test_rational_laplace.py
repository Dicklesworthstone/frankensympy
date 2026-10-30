"""Checked causal inverse Laplace transforms, including repeated quadratic poles."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def equal(actual, expected):
    assert s.expand(actual - expected) == 0


def test_repeated_quadratic_poles_have_the_complete_nonzero_inverse():
    z, t = s.symbols("laplace_repeated_s laplace_repeated_t")
    h = s.Heaviside(t)
    equal(s.inverse_laplace_transform(1/(z*z+1)**2, z, t),
          (s.sin(t)-t*s.cos(t))*h/2)
    equal(s.inverse_laplace_transform(1/(z*z+1)**3, z, t),
          ((3-t*t)*s.sin(t)-3*t*s.cos(t))*h/8)
    equal(s.inverse_laplace_transform(z/(z*z+1)**3, z, t),
          (t*s.sin(t)-t*t*s.cos(t))*h/8)


def test_shifted_quadratic_and_irrational_frequency_remain_exact():
    z, t = s.symbols("laplace_shifted_s laplace_shifted_t")
    frequency = s.sqrt(5)
    expected = s.exp(-2*t)*(t*s.sin(frequency*t)/frequency-t*t*s.cos(frequency*t))/40
    equal(s.inverse_laplace_transform((z+2)/(z*z+4*z+9)**3, z, t),
          expected*s.Heaviside(t))


def test_repeated_real_irrational_poles_use_hyperbolic_functions():
    z, t = s.symbols("laplace_hyperbolic_s laplace_hyperbolic_t")
    root = s.sqrt(2)
    equal(s.inverse_laplace_transform(1/(z*z-2), z, t),
          s.sinh(root*t)/root*s.Heaviside(t))
    equal(s.inverse_laplace_transform(1/(z*z-2)**2, z, t),
          (t*s.cosh(root*t)-s.sinh(root*t)/root)*s.Heaviside(t)/4)


def test_mixed_linear_and_quadratic_factors_keep_every_contribution():
    z, t = s.symbols("laplace_mixed_s laplace_mixed_t")
    source = 1/z + 2/(z-1)**2 + (z+2)/(z*z+4*z+5)**2
    expected = (1 + 2*t*s.exp(t) + t*s.exp(-2*t)*s.sin(t)/2)*s.Heaviside(t)
    equal(s.inverse_laplace_transform(source, z, t), expected)
    equal(s.inverse_laplace_transform(1/(z*(z*z+1)), z, t),
          (1-s.cos(t))*s.Heaviside(t))


def test_distinct_quadratic_factors_are_both_present():
    z, t = s.symbols("laplace_distinct_s laplace_distinct_t")
    equal(s.inverse_laplace_transform(1/((z*z+1)*(z*z+4)), z, t),
          (s.sin(t)/3-s.sin(2*t)/6)*s.Heaviside(t))


def test_scaled_linear_repeated_poles_and_old_table_examples():
    z, t = s.symbols("laplace_linear_s laplace_linear_t")
    cases = [
        (1/(2*z+3)**3, t*t*s.exp(-3*t/2)/16),
        (3/(z-2)**4, t**3*s.exp(2*t)/2),
        (1/(z*z+1), s.sin(t)),
        ((z+1)/(z*z+2*z+5), s.exp(-t)*s.cos(2*t)),
        (1/((z+1)*(z+2)), s.exp(-t)-s.exp(-2*t)),
    ]
    for source, expected in cases:
        equal(s.inverse_laplace_transform(source, z, t), expected*s.Heaviside(t))


def test_result_is_causal_and_preserves_existing_noconds_contract():
    z, t = s.symbols("laplace_conditions_s laplace_conditions_t")
    result = s.inverse_laplace_transform(1/(z*z+1)**2, z, t)
    assert result.subs(t, -1) == 0
    assert s.inverse_laplace_transform(1/(z*z+1)**2, z, t, noconds=False) == (result, 0, True)
    assert s.InverseLaplaceTransform(1/(z*z+1)**2, z, t).doit() == result


def test_supported_time_shift_and_symbolic_table_paths_remain_available():
    z, t = s.symbols("laplace_table_s laplace_table_t")
    a = s.Symbol("laplace_table_a", positive=True)
    equal(s.inverse_laplace_transform(s.exp(-z)/z, z, t), s.Heaviside(t-1))
    equal(s.inverse_laplace_transform(1/(z+a), z, t), s.exp(-a*t)*s.Heaviside(t))


@pytest.mark.parametrize("source_kind", ["constant", "improper", "irreducible_cubic"])
def test_unsupported_distributions_or_poles_remain_unevaluated_not_zero(source_kind):
    z, t = s.symbols("laplace_unsupported_s laplace_unsupported_t")
    sources = {"constant": s.Integer(1), "improper": z/(z+1), "irreducible_cubic": 1/(z**3-z+1)}
    source = sources[source_kind]
    result = s.inverse_laplace_transform(source, z, t)
    assert isinstance(result, s.InverseLaplaceTransform)
    assert result.args == (source, z, t)
    assert result != 0


def test_zero_transform_is_exactly_zero():
    z, t = s.symbols("laplace_zero_s laplace_zero_t")
    assert s.inverse_laplace_transform(0, z, t) == 0


@pytest.mark.parametrize("power", [-1, 0])
def test_original_undefined_children_are_checked_before_zero_cancellation(power):
    z, t = s.symbols("laplace_held_s laplace_held_t")
    invalid = s.Mul(0, s.Pow(0, power, evaluate=False), evaluate=False)
    source = s.Add(1/z, invalid, evaluate=False)
    with pytest.raises(ValueError):
        s.inverse_laplace_transform(source, z, t)


def test_original_nested_poles_do_not_change_the_checked_value():
    z, t = s.symbols("laplace_nested_s laplace_nested_t")
    inner = s.Pow(z-100, -1, evaluate=False)
    held_one = s.Pow(inner, 0, evaluate=False)
    source = s.Mul(held_one, s.Pow(z, -1, evaluate=False), evaluate=False)
    equal(s.inverse_laplace_transform(source, z, t), s.Heaviside(t))


def test_typed_output_symbols_and_dummies_preserve_original_owners():
    z = s.Symbol("laplace_same")
    t = s.Symbol("laplace_same", real=True)
    result = s.inverse_laplace_transform(1/(z*z+1)**2, z, t)
    assert result.free_symbols == {t}
    assert next(iter(result.free_symbols)) is t
    first, second = s.Dummy("laplace_dummy"), s.Dummy("laplace_dummy")
    result = s.inverse_laplace_transform(1/(first*first+1)**2, first, second)
    assert result.free_symbols == {second}
    assert next(iter(result.free_symbols)) is second
    with pytest.raises(ValueError, match="distinct"):
        s.inverse_laplace_transform(1/z, z, z)


def test_direct_native_grammar_dispatch_preserves_symbolic_table_routing():
    z, t, a = s.symbols("laplace_grammar_s laplace_grammar_t laplace_grammar_a")
    assert _native.inverse_rational_laplace_native(
        _native_expr(1/(z+a)), _symbol_binding(z), _symbol_binding(t),
    ) is None


def test_source_limits_and_custom_overrides_refuse_before_expensive_work():
    z, t = s.symbols("laplace_limit_s laplace_limit_t")
    huge_power = s.Pow(z*z+1, -10**9, evaluate=False)
    with pytest.raises(ValueError):
        s.inverse_laplace_transform(huge_power, z, t)
    with pytest.raises(ValueError, match="4096 bits"):
        s.inverse_laplace_transform(1 << 4097, z, t)

    class CustomSymbol(s.Symbol):
        pass

    custom = CustomSymbol("laplace_custom")
    held = s.Mul(0, custom, evaluate=False)
    with pytest.raises(NotImplementedError):
        s.inverse_laplace_transform(s.Add(1/z, held, evaluate=False), z, t)


def test_operational_failure_is_not_swallowed_as_an_unevaluated_or_table_answer(monkeypatch):
    z, t = s.symbols("laplace_failure_s laplace_failure_t")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise ValueError("inverse Laplace verification budget exhausted")

    monkeypatch.setattr(_native, "inverse_rational_laplace_native", exhausted)
    with pytest.raises(ValueError, match="verification budget exhausted"):
        s.inverse_laplace_transform(1/(z*z+1)**2, z, t)
    assert len(calls) == 1


@pytest.mark.parametrize("zero_kind", ["literal", "transform_variable", "parameter"])
@pytest.mark.parametrize("routing", ["parameter", "time_shift"])
def test_mixed_held_invalid_domains_cannot_disappear_through_legacy_routing(zero_kind, routing):
    z, t, a = s.symbols("laplace_mixed_held_s laplace_mixed_held_t laplace_mixed_held_a")
    zero = {
        "literal": s.Integer(0),
        "transform_variable": s.Add(z, -z, evaluate=False),
        "parameter": s.Add(a, -a, evaluate=False),
    }[zero_kind]
    inner = s.Pow(zero, -1, evaluate=False)
    held_one = s.Pow(inner, 0, evaluate=False)
    other = a if routing == "parameter" else s.exp(-z)
    hidden = s.Mul(0, held_one, other, evaluate=False)
    source = s.Add(1/z, hidden, evaluate=False)
    with pytest.raises(ValueError, match="zero base|undefined"):
        s.inverse_laplace_transform(source, z, t)


def test_unresolved_mixed_held_domains_remain_unevaluated_before_expansion():
    z, t = s.symbols("laplace_unresolved_s laplace_unresolved_t")
    zero = s.Add(s.sin(z), -s.sin(z), evaluate=False)
    inner = s.Pow(zero, -1, evaluate=False)
    hidden = s.Mul(0, s.Pow(inner, 0, evaluate=False), s.exp(-z), evaluate=False)
    source = s.Add(1/z, hidden, evaluate=False)
    result = s.inverse_laplace_transform(source, z, t)
    assert isinstance(result, s.InverseLaplaceTransform)
    assert result.args == (source, z, t)
    assert result != 0 and result != s.Heaviside(t)


def test_fallback_domain_rings_keep_distinct_typed_symbols_with_identical_names():
    z, t = s.symbols("laplace_fallback_identity_s laplace_fallback_identity_t")
    plain = s.Symbol("laplace_parameter_identity")
    assumed = s.Symbol("laplace_parameter_identity", positive=True)
    source = s.Pow(s.Add(plain, -assumed, evaluate=False), -1, evaluate=False)
    assert _native.inverse_rational_laplace_native(
        _native_expr(source), _symbol_binding(z), _symbol_binding(t),
    ) is None


@pytest.mark.parametrize("shift", [0, 1])
def test_declared_larger_linear_poles_preserve_the_existing_exact_table(shift):
    z, t = s.symbols("laplace_large_linear_s laplace_large_linear_t")
    source = 1/(z+shift)**64
    expected = t**63*s.exp(-shift*t)*s.Heaviside(t)/s.factorial(63)
    equal(s.inverse_laplace_transform(source, z, t), expected)


def test_degree_growth_is_recognized_before_entering_the_smaller_native_inverse_envelope():
    z, t = s.symbols("laplace_degree_growth_s laplace_degree_growth_t")
    for source in [
        1/((z+1)**20*(z+2)**20),
        1/(z*z+1)**20,
    ]:
        assert _native.inverse_rational_laplace_native(
            _native_expr(source), _symbol_binding(z), _symbol_binding(t),
        ) is None
    source = 1/(z*z+1)**20
    result = s.inverse_laplace_transform(source, z, t)
    assert isinstance(result, s.InverseLaplaceTransform)
    assert result.args == (source, z, t)


def test_mixed_fallback_cannot_bypass_the_original_integer_power_limit():
    z, t = s.symbols("laplace_mixed_limit_s laplace_mixed_limit_t")
    source = s.Mul(s.exp(-z), s.Pow(z, -10**9, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="power|envelope"):
        s.inverse_laplace_transform(source, z, t)
