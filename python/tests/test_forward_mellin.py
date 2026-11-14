"""Bounded forward Mellin kernels and the SymPy 1.14.0 tuple convention.

The strip, zero/unsupported behavior and exact sample values were checked
against isolated upstream 1.14.0. Gamma-product beta answers intentionally
use a mathematical comparator; upstream often rewrites them trigonometrically.
The production implementation never imports or invokes the upstream oracle.
"""

from fractions import Fraction

import pytest
import sympy as s

from sympy.core import Basic, _native, _native_expr, _symbol_binding
from sympy.integrals.transforms import IntegralTransformError


def equal(actual, expected):
    assert s.simplify(actual - expected) == 0, (actual, expected)


@pytest.mark.parametrize("case", [
    "exponential", "gaussian", "shift", "reciprocal", "reciprocal_shift",
])
def test_gamma_family_has_checked_expression_and_maximal_strip(case):
    x, z = s.symbols("forward_gamma_x forward_gamma_s")
    sources = {
        "exponential": (s.exp(-x), s.gamma(z), (0, s.oo), 3, 2),
        "gaussian": (s.exp(-2*x**2), s.gamma(z/2)/(2*2**(z/2)), (0, s.oo), 4, s.Rational(1, 8)),
        "shift": (-6*x**2*s.exp(-3*x), -6*3**(-z-2)*s.gamma(z+2), (-2, s.oo), 0, s.Rational(-2, 3)),
        "reciprocal": (x**s.Rational(1, 3)*s.exp(-3/x**2),
                       3**(z/2+s.Rational(1, 6))*s.gamma(-z/2-s.Rational(1, 6))/2,
                       (-s.oo, s.Rational(-1, 3)), s.Rational(-7, 3), s.Rational(1, 6)),
        "reciprocal_shift": (5*x**-2*s.exp(-2/x), 5*2**(z-2)*s.gamma(2-z),
                             (-s.oo, 2), 1, s.Rational(5, 2)),
    }
    source, expected, strip, point, value = sources[case]
    result = s.mellin_transform(source, x, z)
    assert type(result) is tuple and len(result) == 3
    transform, actual_strip, condition = result
    assert type(actual_strip) is tuple and actual_strip == strip
    assert condition is s.true
    equal(transform, expected)
    equal(transform.subs(z, point), value)
    equal(s.mellin_transform(source, x, z, noconds=True), expected)


@pytest.mark.parametrize("case", [
    "simple", "scaled", "nonunit", "reciprocal", "fractional_shift", "fractional_order",
])
def test_beta_family_has_exact_values_and_open_convergence_strip(case):
    x, z = s.symbols("forward_beta_x forward_beta_s")
    sources = {
        "simple": ((1+x)**-2, (0, 2), 1, 1),
        "scaled": (x**2/(1+2*x**3)**2, (-2, 4), 1, s.Rational(1, 6)),
        "nonunit": (12*x/(2+3*x**2)**3, (-1, 5), 1, s.Rational(1, 4)),
        "reciprocal": ((1+2/x**2)**-2, (-4, 0), -2, s.Rational(1, 4)),
        "fractional_shift": (-3*s.sqrt(x)/(2+4/x)**2,
                             (s.Rational(-5, 2), s.Rational(-1, 2)),
                             s.Rational(-3, 2), s.Rational(-3, 8)),
        "fractional_order": ((2+3*x**2)**s.Rational(-3, 2), (0, 3), 1, 1/(2*s.sqrt(3))),
    }
    source, strip, point, value = sources[case]
    transform, actual_strip, condition = s.mellin_transform(source, x, z)
    assert actual_strip == strip and condition is s.true
    equal(transform.subs(z, point), value)
    assert transform.free_symbols == {z}


def test_beta_reflection_identity_and_inverse_power_law_values():
    x, z = s.symbols("forward_beta_reflection_x forward_beta_reflection_s")
    transform, strip, condition = s.mellin_transform(1/(1+x), x, z)
    assert strip == (0, 1) and condition is s.true
    # Upstream returns pi/sin(pi*z). This exact interior value checks the
    # gamma-product result without asking this shell for a reflection rewrite.
    equal(transform.subs(z, s.Rational(1, 2)), s.pi)
    transform, strip, _ = s.mellin_transform(1/(1+1/x), x, z)
    assert strip == (-1, 0)
    equal(transform.subs(z, s.Rational(-1, 2)), s.pi)


def test_zero_tuple_singletons_and_doit_default_match_pinned_oracle():
    x, z = s.symbols("forward_zero_x forward_zero_s")
    result = s.mellin_transform(0, x, z)
    assert result == (0, (-s.oo, s.oo), s.true)
    assert type(result) is tuple and type(result[1]) is tuple
    assert result[0] is s.S.Zero and result[2] is s.true
    assert s.mellin_transform(0, x, z, noconds=True) is s.S.Zero
    held = s.MellinTransform(s.exp(-x), x, z)
    assert held.args == (s.exp(-x), x, z)
    assert held.function == s.exp(-x)
    assert held.function_variable is x and held.transform_variable is z
    assert held.free_symbols == {z}
    assert held.func(*held.args) == held
    assert held.doit() == s.mellin_transform(s.exp(-x), x, z)
    equal(held.doit(noconds=True), s.gamma(z))


@pytest.mark.parametrize("case", ["constant", "power", "growth", "sum", "hybrid", "oscillation", "pole", "symbolic_scale", "float"])
def test_unsupported_sources_remain_whole_formal_transforms(case):
    x, z, a = s.symbols("forward_unsupported_x forward_unsupported_s forward_unsupported_a")
    source = {
        "constant": s.Integer(1),
        "power": x**-2,
        "growth": s.exp(x),
        "sum": s.exp(-x)+s.exp(-2*x),
        "hybrid": s.exp(-x)/(1+x),
        "oscillation": s.sin(x),
        "pole": 1/(1-x),
        "symbolic_scale": s.exp(-a*x),
        "float": s.Float(0.5)*s.exp(-x),
    }[case]
    for noconds in [False, True]:
        result = s.mellin_transform(source, x, z, noconds=noconds)
        assert type(result) is s.MellinTransform
        assert result.args == (source, x, z)


def test_needeval_and_simplify_hints_are_explicit():
    x, z = s.symbols("forward_hint_x forward_hint_s")
    with pytest.raises(IntegralTransformError, match="needeval") as error:
        s.mellin_transform(s.sin(x), x, z, needeval=True)
    assert error.value.function == s.sin(x)
    first = s.mellin_transform(s.exp(-2*x**2), x, z, simplify=False)
    second = s.mellin_transform(s.exp(-2*x**2), x, z, simplify=True)
    equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_typed_same_spelling_and_dummy_variables_survive_lifting():
    x = s.Symbol("forward_same_name")
    z = s.Symbol("forward_same_name", positive=True)
    result, strip, _ = s.mellin_transform(s.exp(-x), x, z)
    assert result.free_symbols == {z} and x not in result.free_symbols
    assert strip == (0, s.oo)
    a, b = s.Dummy("forward_dummy"), s.Dummy("forward_dummy")
    result, _, _ = s.mellin_transform(s.exp(-a), a, b)
    assert result.free_symbols == {b}
    assert next(iter(result.free_symbols)) is b
    with pytest.raises(ValueError, match="distinct"):
        s.mellin_transform(s.exp(-x), x, x)


def test_mellin_axis_is_positive_independently_of_surface_assumptions():
    x = s.Symbol("forward_axis_x", negative=True)
    z = s.Symbol("forward_axis_s")
    transform, strip, condition = s.mellin_transform(s.exp(-x), x, z)
    equal(transform, s.gamma(z))
    assert strip == (0, s.oo) and condition is s.true


@pytest.mark.parametrize("exponent", [0, -1])
def test_undefined_original_powers_are_not_erased_by_zero_products(exponent):
    x, z = s.symbols("forward_domain_x forward_domain_s")
    invalid = s.Pow(0, exponent, evaluate=False)
    source = s.Mul(0, invalid, s.exp(-x), evaluate=False)
    with pytest.raises(ValueError, match="zero power|nonfinite"):
        s.mellin_transform(source, x, z)


def test_original_cancelled_interior_pole_stays_formal():
    x, z = s.symbols("forward_original_x forward_original_s")
    factor = x-1
    source = s.Mul(factor, s.Pow(factor, -1, evaluate=False), s.exp(-x), evaluate=False)
    result = s.mellin_transform(source, x, z)
    assert type(result) is s.MellinTransform and result.args[0] is source


def test_preexisting_retained_payload_mismatch_is_rejected():
    x, z = s.symbols("forward_retained_x forward_retained_s")
    source = s.exp(-x, evaluate=False)
    source._args = (s.Integer(3),)
    with pytest.raises(ValueError, match="retained"):
        s.mellin_transform(source, x, z)


def test_source_mutation_during_native_call_is_detected(monkeypatch):
    from sympy.integrals import transforms

    x, z = s.symbols("forward_mutation_x forward_mutation_s")
    source = s.exp(-x)
    original = Basic.__dict__["_value"].__get__(source, Basic)
    native = transforms._MELLIN_NATIVE_TRANSFORM

    def changed(*arguments):
        result = native(*arguments)
        Basic.__dict__["_value"].__set__(source, _native_expr(s.Integer(0)))
        return result

    monkeypatch.setattr(transforms, "_MELLIN_NATIVE_TRANSFORM", changed)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            s.mellin_transform(source, x, z)
    finally:
        Basic.__dict__["_value"].__set__(source, original)


def test_native_failures_are_terminal_and_not_formal_answers(monkeypatch):
    from sympy.integrals import transforms

    x, z = s.symbols("forward_failure_x forward_failure_s")
    failure = ValueError("Mellin independent verifier budget exhausted")

    def failed(*arguments):
        raise failure

    monkeypatch.setattr(transforms, "_MELLIN_NATIVE_TRANSFORM", failed)
    with pytest.raises(ValueError) as error:
        s.mellin_transform(s.exp(-x), x, z)
    assert error.value is failure


def test_external_conversion_and_boolean_hooks_are_not_invoked():
    x, z = s.symbols("forward_hook_x forward_hook_s")

    class Convertible:
        def _sympy_(self):
            raise AssertionError("untrusted converter was invoked")

    class Truthy:
        def __bool__(self):
            raise AssertionError("untrusted bool hook was invoked")

    with pytest.raises(TypeError):
        s.mellin_transform(Convertible(), x, z)
    for option in ["noconds", "simplify", "needeval"]:
        with pytest.raises(TypeError):
            s.mellin_transform(0, x, z, **{option: Truthy()})
    with pytest.raises(TypeError):
        s.mellin_transform(0, x, z, unsupported_option=True)


def test_custom_symbol_and_function_owners_do_not_enter_native_lane(monkeypatch):
    x, z = s.symbols("forward_custom_x forward_custom_s")
    source = s.exp(-x)

    def forbidden(*arguments):
        raise AssertionError("modified built-in hook was invoked")

    monkeypatch.setattr(s.exp, "free_symbols", property(forbidden))
    with pytest.raises(NotImplementedError):
        s.mellin_transform(source, x, z)


@pytest.mark.parametrize("attribute", ["__getattribute__", "name", "_assumptions", "free_symbols"])
def test_replaced_symbol_metadata_or_accessors_refuse_before_hooks(monkeypatch, attribute):
    x, z = s.symbols("forward_symbol_hook_x forward_symbol_hook_s")
    source = s.exp(-x)

    def forbidden(*arguments):
        raise AssertionError("modified Symbol hook was invoked")

    replacement = forbidden if attribute == "__getattribute__" else property(forbidden)
    monkeypatch.setattr(s.Symbol, attribute, replacement)
    with pytest.raises(NotImplementedError, match="original"):
        s.mellin_transform(source, x, z)


def test_native_catalog_entrypoint_is_captured_without_module_hook_dispatch(monkeypatch):
    x, z = s.symbols("forward_frozen_x forward_frozen_s")

    def forbidden(*arguments):
        raise AssertionError("replaced native module callback was invoked")

    monkeypatch.setattr(_native, "mellin_transform_native", forbidden)
    transform, strip, condition = s.mellin_transform(s.exp(-x), x, z)
    equal(transform, s.gamma(z))
    assert strip == (0, s.oo) and condition is s.true


def test_native_bridge_uses_typed_atoms_and_returns_exact_bounds():
    x, z = s.symbols("forward_bridge_x forward_bridge_s")
    result, lower, upper = _native.mellin_transform_native(
        _native_expr(s.exp(-x)), _symbol_binding(x), _symbol_binding(z),
    )
    assert result == _native_expr(s.gamma(z))
    assert lower == _native_expr(s.Integer(0))
    assert upper == _native_expr(s.oo)


def test_source_and_exponent_limits_are_explicit():
    x, z = s.symbols("forward_limits_x forward_limits_s")
    with pytest.raises(ValueError, match="4096"):
        s.mellin_transform(1 << 4097, x, z)
    with pytest.raises(ValueError, match="power"):
        s.mellin_transform(s.Pow(x, -10**9, evaluate=False), x, z)
    assert type(s.mellin_transform(Fraction(2, 3), x, z)) is s.MellinTransform


def test_public_path_does_not_use_generic_integration_or_legacy_parser(monkeypatch):
    x, z = s.symbols("forward_direct_x forward_direct_s")

    def forbidden(*arguments, **keywords):
        raise AssertionError("generic integration fallback was invoked")

    monkeypatch.setattr(s, "integrate", forbidden)
    transform, strip, condition = s.mellin_transform(s.exp(-x), x, z)
    equal(transform, s.gamma(z))
    assert strip == (0, s.oo) and condition is s.true
