"""Checked finite causal time shifts, repeated poles, and original source guards."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.integrals import transforms


def equal(actual, expected):
    assert s.expand(actual - expected) == 0


def test_delayed_repeated_quadratic_poles_use_the_checked_native_path(monkeypatch):
    z, t = s.symbols("delayed_repeated_s delayed_repeated_t")

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("a checked delayed inverse must not retry the legacy table")

    monkeypatch.setattr(transforms, "_inverse_laplace_table", forbidden_fallback)
    u = t - 1
    equal(s.inverse_laplace_transform(s.exp(-z)/(z*z+1)**2, z, t),
          (s.sin(u)-u*s.cos(u))*s.Heaviside(u)/2)
    u = t - s.Rational(3, 2)
    equal(s.inverse_laplace_transform(s.exp(-s.Rational(3, 2)*z)/(z*z+1)**3, z, t),
          ((3-u*u)*s.sin(u)-3*u*s.cos(u))*s.Heaviside(u)/8)


def test_mixed_repeated_poles_preserve_every_delay_and_zero_delay_term():
    z, t = s.symbols("delayed_mixed_s delayed_mixed_t")
    source = 1/z + 2*s.exp(-z)/(z-2)**3 + s.exp(-2*z)/(z*z+1)**2
    u, v = t-1, t-2
    expected = (s.Heaviside(t) + u*u*s.exp(2*u)*s.Heaviside(u)
                + (s.sin(v)-v*s.cos(v))*s.Heaviside(v)/2)
    result = s.inverse_laplace_transform(source, z, t)
    equal(result, expected)
    assert result.subs(t, -1) == 0
    # A second global Heaviside(t) would incorrectly change this value to 1/4.
    assert result.subs(t, 0) == s.Rational(1, 2)


def test_same_delay_polynomial_parts_cancel_before_properness_is_checked():
    z, t = s.symbols("delayed_cancel_s delayed_cancel_t")
    delay = s.exp(-z)
    sources = [
        s.Mul(delay, s.Add(z/(z+1), -1, evaluate=False), evaluate=False),
        s.Add(s.Mul(delay, z/(z+1), evaluate=False), -delay, evaluate=False),
    ]
    for source in sources:
        equal(s.inverse_laplace_transform(source, z, t),
              -s.exp(-(t-1))*s.Heaviside(t-1))


def test_products_distribute_delays_and_combine_equal_delay_terms():
    z, t = s.symbols("delayed_product_s delayed_product_t")
    source = s.Mul(
        s.Add(s.exp(-z), s.exp(-2*z), evaluate=False),
        s.Add(1, s.exp(-z), evaluate=False),
        1/z,
        evaluate=False,
    )
    equal(s.inverse_laplace_transform(source, z, t),
          s.Heaviside(t-1)+2*s.Heaviside(t-2)+s.Heaviside(t-3))
    source = s.Mul(s.exp(z), s.exp(-2*z), 1/z, evaluate=False)
    equal(s.inverse_laplace_transform(source, z, t), s.Heaviside(t-1))


def test_hyperbolic_and_shifted_frequency_blocks_remain_exact():
    z, t = s.symbols("delayed_frequency_s delayed_frequency_t")
    u = t - s.Rational(1, 3)
    root = s.sqrt(2)
    equal(s.inverse_laplace_transform(s.exp(-z/3)/(z*z-2)**2, z, t),
          (u*s.cosh(root*u)-s.sinh(root*u)/root)*s.Heaviside(u)/4)
    u = t - 2
    root = s.sqrt(5)
    source = s.exp(-2*z)*(z+2)/(z*z+4*z+9)**3
    expected = s.exp(-2*u)*(u*s.sin(root*u)/root-u*u*s.cos(root*u))*s.Heaviside(u)/40
    equal(s.inverse_laplace_transform(source, z, t), expected)


def test_causal_step_is_attached_once_at_the_exact_delay():
    z, t = s.symbols("delayed_causal_s delayed_causal_t")
    result = s.inverse_laplace_transform(s.exp(-3*z/2)/z, z, t)
    equal(result, s.Heaviside(t-s.Rational(3, 2)))
    assert result.subs(t, 0) == 0
    assert result.subs(t, s.Rational(3, 2)) == s.Rational(1, 2)
    assert result.subs(t, 2) == 1
    assert s.inverse_laplace_transform(1/z, z, t) == s.Heaviside(t)


def test_delayed_conditions_and_unevaluated_doit_keep_existing_contract():
    z, t = s.symbols("delayed_conditions_s delayed_conditions_t")
    source = s.exp(-z)/(z*z+1)**2
    result = s.inverse_laplace_transform(source, z, t)
    assert s.inverse_laplace_transform(source, z, t, noconds=False) == (result, True)
    assert s.InverseLaplaceTransform(source, z, t).doit() == result
    # Pinned SymPy 1.14: forward retains the abscissa, inverse returns a pair.
    assert s.laplace_transform(s.exp(-t), t, z, noconds=False) == (1/(z+1), -1, True)


def test_original_time_symbol_and_dummy_owners_survive_native_shifting():
    z = s.Symbol("delayed_same")
    t = s.Symbol("delayed_same", real=True)
    source = s.exp(-z)/(z*z+1)**2
    result = s.inverse_laplace_transform(source, z, t)
    assert result.free_symbols == {t}
    assert next(iter(result.free_symbols)) is t
    z, t = s.Dummy("delayed_dummy"), s.Dummy("delayed_dummy")
    result = s.inverse_laplace_transform(s.exp(-z)/(z*z+1)**2, z, t)
    assert result.free_symbols == {t}
    assert next(iter(result.free_symbols)) is t
    with pytest.raises(ValueError, match="distinct"):
        s.inverse_laplace_transform(s.exp(-z)/z, z, z)


@pytest.mark.parametrize("power", [-1, 0])
@pytest.mark.parametrize("literal", [True, False])
def test_delayed_original_undefined_children_are_terminal_before_zero(power, literal):
    z, t = s.symbols("delayed_domain_s delayed_domain_t")
    zero = s.Integer(0) if literal else s.Add(z, -z, evaluate=False)
    hidden = s.Pow(s.Pow(zero, -1, evaluate=False), power, evaluate=False)
    source = s.Add(
        s.exp(-z)/(z*z+1)**2,
        s.Mul(0, s.exp(-2*z), hidden, evaluate=False),
        evaluate=False,
    )
    with pytest.raises(ValueError):
        s.inverse_laplace_transform(source, z, t)


def test_original_canceled_guard_does_not_change_the_delayed_value():
    z, t = s.symbols("delayed_guard_s delayed_guard_t")
    one = s.Pow(s.Pow(z-100, -1, evaluate=False), 0, evaluate=False)
    source = s.Mul(s.exp(-z), one, 1/z, evaluate=False)
    equal(s.inverse_laplace_transform(source, z, t), s.Heaviside(t-1))


def test_declared_delay_grammar_misses_still_return_none_at_native_boundary():
    z, t, a = s.symbols("delayed_grammar_s delayed_grammar_t delayed_grammar_a")
    sources = [s.exp(z)/z, s.exp(-a*z)/z, s.exp(-z*z)/z, s.exp(-z)/z**33]
    for source in sources:
        assert _native.inverse_rational_laplace_native(
            _native_expr(source), _symbol_binding(z), _symbol_binding(t),
        ) is None


def test_unsupported_final_term_never_yields_a_partial_inverse():
    z, t = s.symbols("delayed_partial_s delayed_partial_t")
    source = s.exp(-z)/(z*z+1)**2 + s.exp(-2*z)/(z**3-z+1)
    result = s.inverse_laplace_transform(source, z, t)
    assert isinstance(result, s.InverseLaplaceTransform)
    assert result.args == (source, z, t)
    assert result != 0


def test_operational_failure_is_terminal_without_legacy_retry(monkeypatch):
    z, t = s.symbols("delayed_failure_s delayed_failure_t")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise ValueError("delayed inverse verification budget exhausted")

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("operational failure must remain terminal")

    monkeypatch.setattr(_native, "inverse_rational_laplace_native", exhausted)
    monkeypatch.setattr(transforms, "_inverse_laplace_table", forbidden_fallback)
    with pytest.raises(ValueError, match="verification budget exhausted"):
        s.inverse_laplace_transform(s.exp(-z)/(z*z+1)**2, z, t)
    assert len(calls) == 1


def test_pure_rational_public_shape_and_symbolic_table_remain_available():
    z, t = s.symbols("delayed_compat_s delayed_compat_t")
    a = s.Symbol("delayed_compat_a", positive=True)
    expected = s.expand((s.sin(t)-t*s.cos(t))*s.Heaviside(t)/2)
    assert s.inverse_laplace_transform(1/(z*z+1)**2, z, t) == expected
    equal(s.inverse_laplace_transform(1/(z+a), z, t), s.exp(-a*t)*s.Heaviside(t))


def test_distributional_first_term_cannot_hide_a_later_invalid_source():
    z, t = s.symbols("delayed_invalid_tail_s delayed_invalid_tail_t")
    zero = s.Add(z, -z, evaluate=False)
    source = s.Add(
        s.exp(-z),
        s.Mul(0, s.exp(-2*z), s.Pow(zero, -1, evaluate=False), evaluate=False),
        evaluate=False,
    )
    with pytest.raises(ValueError):
        s.inverse_laplace_transform(source, z, t)
