"""Ordinary Fourier catalog, typed boundary and pinned SymPy 1.14 conventions.

The upstream constant, Gaussian, bilateral-decay and sufficient-condition
observations were executed in an isolated SymPy 1.14.0 environment. These
tests exercise FrankenSymPy's actual extension; they do not invoke upstream
as a production fallback or confer whole-profile certification.
"""

from fractions import Fraction

import pytest
import sympy as s
from sympy.core import Basic, _native, _native_expr, _symbol_binding


def equal(actual, expected):
    assert s.simplify(s.expand(actual - expected)) == 0


@pytest.mark.parametrize(
    "rate,amplitude,mass,unit_frequency",
    [(1, 1, 2, 1), (2, 1, 1, s.Rational(4, 5)),
     (s.Rational(1, 3), 1, 6, s.Rational(3, 5)),
     (2, 3, 3, s.Rational(12, 5))],
)
def test_cycles_normalization_at_exact_rational_angular_frequency(
    rate, amplitude, mass, unit_frequency,
):
    x, k = s.symbols("fourier_decay_x fourier_decay_k")
    result = s.fourier_transform(amplitude*s.exp(-rate*s.Abs(x)), x, k)
    equal(result, 2*amplitude*rate/(rate*rate + 4*s.pi**2*k*k))
    equal(result.subs(k, 0), mass)
    equal(result.subs(k, 1/(2*s.pi)), unit_frequency)
    equal(result.subs(k, -1/(2*s.pi)), unit_frequency)


@pytest.mark.parametrize("center", [0, 2, s.Rational(-2, 3)])
def test_gaussian_mass_and_translation_keep_the_negative_fourier_phase(center):
    x, k = s.symbols("fourier_gaussian_x fourier_gaussian_k")
    result = s.fourier_transform(s.exp(-(x-center)**2), x, k)
    equal(result.subs(k, 0), s.sqrt(s.pi))
    unshifted = s.fourier_transform(s.exp(-x*x), x, k)
    equal(result.subs(k, 1/(2*s.pi)),
          s.exp(-s.I*center)*unshifted.subs(k, 1/(2*s.pi)))


def test_shifted_scaled_bilateral_exponential_uses_absolute_slope():
    x, k = s.symbols("fourier_affine_x fourier_affine_k")
    result = s.fourier_transform(s.exp(4-2*s.Abs(3*x+1)), x, k)
    expected = 12*s.exp(4)*s.exp(2*s.I*s.pi*k/3)/(36+4*s.pi**2*k*k)
    equal(result, expected)
    equal(result.subs(k, 0), s.exp(4)/3)
    negative_slope = s.fourier_transform(s.exp(-2*s.Abs(-3*x)), x, k)
    equal(negative_slope.subs(k, 1/(2*s.pi)), s.Rational(12, 37))


def test_rational_quadratic_completion_and_finite_linearity():
    x, k = s.symbols("fourier_linear_x fourier_linear_k")
    gaussian = s.fourier_transform(s.exp(-2*x*x+3*x+1), x, k)
    equal(gaussian.subs(k, 0), s.sqrt(s.pi/2)*s.exp(s.Rational(17, 8)))
    total = s.fourier_transform(3*s.exp(-x*x)-2*s.exp(-s.Abs(x)), x, k)
    equal(total.subs(k, 0), 3*s.sqrt(s.pi)-4)
    grouped = s.fourier_transform(2*(s.exp(-x*x)+s.exp(-s.Abs(x))), x, k)
    equal(grouped.subs(k, 0), 2*s.sqrt(s.pi)+4)


@pytest.mark.parametrize("facts", [{}, {"real": True}, {"positive": True}, {"negative": True}])
def test_pinned_bilateral_sufficient_conditions_are_preserved(facts):
    x = s.Symbol("fourier_condition_x")
    k = s.Symbol("fourier_condition_k", **facts)
    result, condition = s.fourier_transform(s.exp(-s.Abs(x)), x, k, noconds=False)
    equal(result, 2/(1+4*s.pi**2*k*k))
    expected = True if facts.get("positive") or facts.get("negative") else k > 0
    assert condition == expected


def test_gaussian_and_zero_condition_contracts():
    x, k = s.symbols("fourier_zero_x fourier_zero_k")
    result, condition = s.fourier_transform(s.exp(-x*x), x, k, noconds=False)
    equal(result, s.sqrt(s.pi)*s.exp(-s.pi*s.pi*k*k))
    assert condition is True
    assert s.fourier_transform(0, x, k) == 0
    assert s.fourier_transform(0, x, k, noconds=False) == (0, True)


@pytest.mark.parametrize("value", [1, 5, -2, Fraction(2, 3)])
def test_nonzero_constants_remain_ordinary_unevaluated_transforms(value):
    x, k = s.symbols("fourier_constant_x fourier_constant_k")
    coefficient = s.Rational(value.numerator, value.denominator) if type(value) is Fraction else s.Integer(value)
    expected = coefficient*s.FourierTransform(1, x, k)
    result = s.fourier_transform(value, x, k)
    assert result == expected
    assert s.fourier_transform(value, x, k, noconds=False) == expected
    assert "dirac" not in str(result).lower()
    assert result.free_symbols == {k}
    assert s.FourierTransform(1, x, k).doit() == s.FourierTransform(1, x, k)


def test_unsupported_assumed_frequency_conditions_stay_unevaluated():
    x = s.Symbol("fourier_profile_x")
    k = s.Symbol("fourier_profile_k", integer=True)
    source = s.exp(-s.Abs(x))
    result = s.fourier_transform(source, x, k, noconds=False)
    assert type(result) is s.FourierTransform
    assert result.args == (source, x, k)
    # The value-only ordinary real-frequency formula is still available.
    equal(s.fourier_transform(source, x, k), 2/(1+4*s.pi**2*k*k))


@pytest.mark.parametrize("noconds", [True, False])
def test_bilateral_nonreal_frequency_needs_an_unimplemented_strip(noconds):
    x = s.Symbol("fourier_complex_x")
    k = s.Symbol("fourier_complex_k", imaginary=True)
    source = s.exp(-s.Abs(x))
    result = s.fourier_transform(source, x, k, noconds=noconds)
    assert type(result) is s.FourierTransform
    assert result.args == (source, x, k)


@pytest.mark.parametrize("case", ["growing", "linear", "unknown_rate", "foreign_term", "product"])
def test_unsupported_whole_sources_never_publish_a_partial_transform(case):
    x, k, a = s.symbols("fourier_refusal_x fourier_refusal_k fourier_refusal_a")
    source = {
        "growing": s.exp(x*x),
        "linear": s.exp(x),
        "unknown_rate": s.exp(-a*x*x),
        "foreign_term": s.exp(-x*x)+s.sin(x),
        "product": s.exp(-x*x)*s.sin(x),
    }[case]
    result = s.fourier_transform(source, x, k)
    assert type(result) is s.FourierTransform
    assert result.args == (source, x, k)
    assert result.args[0] is source


@pytest.mark.parametrize("power", [-1, 0])
def test_held_zero_multipliers_do_not_hide_undefined_original_powers(power):
    x, k = s.symbols("fourier_pole_x fourier_pole_k")
    invalid = s.Mul(0, s.Pow(0, power, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.fourier_transform(invalid, x, k)


def test_zero_amplitude_does_not_hide_an_unsupported_original_factor():
    x, k = s.symbols("fourier_hidden_x fourier_hidden_k")
    source = s.Mul(0, s.exp(x*x), evaluate=False)
    result = s.fourier_transform(source, x, k)
    assert type(result) is s.FourierTransform
    assert result.args[0] is source


def test_original_frequency_dependence_is_rejected_even_under_zero():
    x, k = s.symbols("fourier_dependent_x fourier_dependent_k")
    source = s.Add(s.exp(-x*x), s.Mul(0, k, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="frequency"):
        s.fourier_transform(source, x, k)
    with pytest.raises(ValueError, match="distinct"):
        s.fourier_transform(s.exp(-x*x), x, x)


def test_same_name_typed_symbols_and_dummies_keep_their_original_owners():
    x = s.Symbol("fourier_same")
    k = s.Symbol("fourier_same", real=True)
    result = s.fourier_transform(s.exp(-x*x), x, k)
    assert result.free_symbols == {k}
    assert next(iter(result.free_symbols)) is k
    x, k = s.Dummy("fourier_dummy"), s.Dummy("fourier_dummy")
    result = s.fourier_transform(s.exp(-x*x), x, k)
    assert result.free_symbols == {k}
    assert next(iter(result.free_symbols)) is k


def test_public_path_never_calls_the_legacy_string_transform(monkeypatch):
    x = s.Symbol("fourier input with spaces")
    k = s.Symbol("fourier output with spaces", real=True)

    def forbidden(*args):
        raise AssertionError("legacy string transform was invoked")

    monkeypatch.setattr(_native, "fourier_expr", forbidden)
    result = s.fourier_transform(s.exp(-x*x), x, k)
    assert result.free_symbols == {k}
    equal(result.subs(k, 0), s.sqrt(s.pi))


def test_arbitrary_conversion_truthiness_and_unknown_options_refuse_before_work(monkeypatch):
    x, k = s.symbols("fourier_conversion_x fourier_conversion_k")
    calls = []

    class Convertible:
        def _sympy_(self):
            calls.append("converter")
            raise AssertionError("arbitrary converter executed")

    class Truthy:
        def __bool__(self):
            calls.append("bool")
            raise AssertionError("arbitrary truth conversion executed")

    with pytest.raises(TypeError):
        s.fourier_transform(Convertible(), x, k)
    with pytest.raises(TypeError):
        s.fourier_transform(0, x, k, noconds=Truthy())
    with pytest.raises(TypeError):
        s.fourier_transform(0, x, k, unsupported_option=True)
    assert calls == []


def test_noncommuting_symbols_refuse_before_native_dispatch(monkeypatch):
    x = s.Symbol("fourier_noncommuting_x")
    k = s.Symbol("fourier_noncommuting_k", commutative=False)
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("noncommuting frequency reached the scalar kernel")

    monkeypatch.setattr(_native, "fourier_l1_native", forbidden)
    with pytest.raises(NotImplementedError, match="commuting"):
        s.fourier_transform(s.exp(-x*x), x, k)
    assert calls == []


def test_custom_function_implementations_do_not_enter_native_fast_paths(monkeypatch):
    x, k = s.symbols("fourier_hook_x fourier_hook_k")
    source = s.exp(-x*x)
    calls = []

    def forbidden(self):
        calls.append(self)
        raise AssertionError("custom hook executed")

    monkeypatch.setattr(s.exp, "_eval_is_real", forbidden, raising=False)
    with pytest.raises(NotImplementedError):
        s.fourier_transform(source, x, k)
    assert calls == []


def test_operational_failure_is_terminal_and_preserves_the_original_error(monkeypatch):
    x, k = s.symbols("fourier_failure_x fourier_failure_k")
    failure = ValueError("Fourier budget exhausted")

    def exhausted(*args):
        raise failure

    monkeypatch.setattr(_native, "fourier_l1_native", exhausted)
    with pytest.raises(ValueError) as observed:
        s.fourier_transform(s.exp(-x*x), x, k)
    assert observed.value is failure


def test_source_mutation_during_native_call_prevents_publication(monkeypatch):
    x, k = s.symbols("fourier_mutation_x fourier_mutation_k")
    source = s.exp(-x*x)
    original_value = Basic.__dict__["_value"].__get__(source, Basic)
    native = _native.fourier_l1_native

    def changed(*args):
        result = native(*args)
        Basic.__dict__["_value"].__set__(source, _native_expr(s.Integer(0)))
        return result

    monkeypatch.setattr(_native, "fourier_l1_native", changed)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            s.fourier_transform(source, x, k)
    finally:
        Basic.__dict__["_value"].__set__(source, original_value)


def test_preexisting_retained_argument_mismatch_is_rejected():
    x, k = s.symbols("fourier_retained_x fourier_retained_k")
    source = s.exp(-x*x, evaluate=False)
    source._args = (s.Integer(3),)
    with pytest.raises(ValueError, match="retained"):
        s.fourier_transform(source, x, k)


def test_native_retained_records_preflight_tags_arities_and_class_payloads():
    x, k = s.symbols("fourier_record_x fourier_record_k")
    native = _native_expr(s.exp(-x*x))
    arguments = (native, _symbol_binding(x), _symbol_binding(k))
    for record in [
        (native, "e"*65536, ()),
        (native, "exp", ()),
        (native, "Integer", None),
        (native, "exp", (_native_expr(s.Integer(0)),)),
    ]:
        with pytest.raises(ValueError):
            _native.fourier_l1_native(*arguments, [record])

    class HostileText(str):
        def __len__(self):
            raise AssertionError("operator subclass callback executed")

    with pytest.raises(ValueError):
        _native.fourier_l1_native(*arguments, [(native, HostileText("exp"), ())])
    wide = _native.Add(*[_native_expr(s.Integer(0))]*4097, evaluate=False).as_expr()
    with pytest.raises(ValueError):
        _native.fourier_l1_native(*arguments, [(wide, "Add", ())])


def test_resource_limits_are_terminal_before_expansion():
    x, k = s.symbols("fourier_limits_x fourier_limits_k")
    with pytest.raises(ValueError, match="4096"):
        s.fourier_transform(1 << 4097, x, k)
    source = s.exp(-s.Pow(x, 10**9, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="power"):
        s.fourier_transform(source, x, k)
