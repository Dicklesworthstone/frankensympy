"""Normalized half-line transforms through FrankenSymPy's actual extension.

The kernel and normalization are pinned to the executable transform classes
in SymPy 1.14.0 (16fa855354eb7bcabd3fe10993841e03b1382692), whose a() is
sqrt(2)/sqrt(pi) and b() is one in all four directions. Exponential and
Gaussian anchors were also checked in an isolated installed SymPy 1.14.0.
The candidate never imports that oracle.
"""

from fractions import Fraction

import pytest
import sympy as s
from sympy.core import Basic, _native, _native_expr
from sympy.integrals.transforms import IntegralTransformError

TRANSFORMS = [
    (s.sine_transform, s.SineTransform, True),
    (s.inverse_sine_transform, s.InverseSineTransform, True),
    (s.cosine_transform, s.CosineTransform, False),
    (s.inverse_cosine_transform, s.InverseCosineTransform, False),
]


def equal(actual, expected):
    assert s.simplify(s.expand(actual - expected)) == 0


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
@pytest.mark.parametrize("rate", [1, 2, s.Rational(1, 3)])
def test_exponential_formula_has_symmetric_normalization_and_angular_kernel(transform, held, sine, rate):
    x, k = s.symbols("half_line_exp_x half_line_exp_k")
    actual = transform(s.exp(-rate*x), x, k)
    numerator = k if sine else rate
    expected = s.sqrt(2)*numerator/(s.sqrt(s.pi)*(rate**2+k**2))
    equal(actual, expected)
    equal(actual.subs(k, 1), expected.subs(k, 1))
    equal(actual.subs(k, -1), expected.subs(k, -1))
    assert type(actual) is not held


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
@pytest.mark.parametrize("degree", [1, 2])
def test_polynomial_exponential_moments(transform, held, sine, degree):
    x, k = s.symbols("half_line_moment_x half_line_moment_k")
    source = x**degree*s.exp(-2*x)
    if degree == 1:
        numerator = 4*k if sine else 4-k**2
    else:
        numerator = 2*k*(12-k**2) if sine else 4*(4-3*k**2)
    expected = s.sqrt(2)*numerator/(s.sqrt(s.pi)*(4+k**2)**(degree+1))
    equal(transform(source, x, k), expected)
    equal(held(source, x, k).doit(), expected)


@pytest.mark.parametrize("inverse", [False, True])
def test_gaussian_masses_and_odd_moments(inverse):
    x, k = s.symbols("half_line_gaussian_x half_line_gaussian_k")
    cosine = s.inverse_cosine_transform if inverse else s.cosine_transform
    sine = s.inverse_sine_transform if inverse else s.sine_transform
    source = s.exp(-2*x*x)
    equal(cosine(source, x, k), s.exp(-k*k/8)/2)
    equal(cosine(x*x*source, x, k), (4-k*k)*s.exp(-k*k/8)/32)
    equal(sine(x*source, x, k), k*s.exp(-k*k/8)/8)
    assert cosine(source, x, k).subs(k, 0) == s.Rational(1, 2)


@pytest.mark.parametrize("rate", [1, 2, s.Rational(1, 2)])
def test_admitted_gaussian_parities_round_trip(rate):
    x, k = s.symbols("half_line_round_trip_x half_line_round_trip_k")
    even = (2-3*x*x)*s.exp(-rate*x*x)
    odd = 3*x*s.exp(-rate*x*x)
    equal(s.inverse_cosine_transform(s.cosine_transform(even, x, k), k, x), even)
    equal(s.inverse_sine_transform(s.sine_transform(odd, x, k), k, x), odd)


def test_linearity_preserves_rational_scales_constant_radicals_and_every_term():
    x, k = s.symbols("half_line_sum_x half_line_sum_k")
    source = s.sqrt(2)*(s.exp(-x*x)+3*s.exp(-x))
    expected = s.exp(-k*k/4)+6/(s.sqrt(s.pi)*(1+k*k))
    equal(s.cosine_transform(source, x, k), expected)
    shifted_height = 3*s.exp(2-2*x)
    equal(s.sine_transform(shifted_height, x, k),
          3*s.sqrt(2)*s.exp(2)*k/(s.sqrt(s.pi)*(4+k*k)))


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
def test_conditions_and_hints_preserve_public_shape(transform, held, sine):
    x = s.Symbol("half_line_condition_x")
    k = s.Symbol("half_line_condition_k", real=True)
    source = s.exp(-x)
    result = transform(source, x, k, noconds=False)
    assert type(result) is tuple and len(result) == 2
    assert result[1] is s.true
    equal(result[0], transform(source, x, k))
    equal(transform(source, x, k, simplify=False), result[0])
    assert transform(0, x, k, noconds=False) == (s.Integer(0), s.true)
    unsupported = s.exp(x)
    for noconds in [False, True]:
        result = transform(unsupported, x, k, noconds=noconds)
        assert type(result) is held
        assert result.args == (unsupported, x, k)
        assert result.free_symbols == {k}
    with pytest.raises(IntegralTransformError):
        transform(unsupported, x, k, needeval=True)


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
def test_unknown_exponential_frequency_does_not_assert_unconditional_convergence(transform, held, sine):
    x, k = s.symbols("half_line_unknown_x half_line_unknown_k")
    source = s.exp(-x)
    assert type(transform(source, x, k)) is not held
    result = transform(source, x, k, noconds=False)
    assert type(result) is held
    assert result.args == (source, x, k)
    with pytest.raises(IntegralTransformError):
        transform(source, x, k, noconds=False, needeval=True)


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
@pytest.mark.parametrize("imaginary", [False, True])
def test_gaussian_outputs_are_entire_with_conditions_requested(transform, held, sine, imaginary):
    x = s.Symbol("half_line_entire_x")
    k = (s.Symbol("half_line_entire_k", imaginary=True) if imaginary
         else s.Symbol("half_line_entire_k"))
    source = x*s.exp(-x*x) if sine else s.exp(-x*x)
    result = transform(source, x, k, noconds=False)
    assert type(result) is tuple and result[1] is s.true
    expected = s.sqrt(2)*s.exp(-k*k/4)*(k/4 if sine else s.Rational(1, 2))
    equal(result[0], expected)


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
def test_unsupported_domains_stay_formal_without_publishing_partial_sums(transform, held, sine):
    x, k, a = s.symbols("half_line_unsupported_x half_line_unsupported_k half_line_rate")
    for source in [
        s.exp(x), s.exp(x*x), s.exp(-a*x), s.exp(-x*x+x),
        x**3*s.exp(-x), s.exp(-x)+s.sin(x), s.exp(-x)*s.sin(x),
        1/(1+x*x), s.Mul(0, 1/x, s.exp(-x), evaluate=False),
    ]:
        result = transform(source, x, k)
        assert type(result) is held
        assert result.args[0] is source
    unsupported_parity = s.exp(-x*x) if sine else x*s.exp(-x*x)
    assert type(transform(unsupported_parity, x, k)) is held


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
def test_exponential_complex_frequency_requires_an_unimplemented_strip(transform, held, sine):
    x = s.Symbol("half_line_complex_x")
    k = s.Symbol("half_line_complex_k", imaginary=True)
    source = s.exp(-x)
    assert type(transform(source, x, k)) is held
    assert type(transform(source, x, k, noconds=False)) is held


@pytest.mark.parametrize("transform,held,sine", TRANSFORMS)
@pytest.mark.parametrize("power", [-1, 0])
def test_undefined_original_powers_cannot_be_hidden_by_zero(transform, held, sine, power):
    x, k = s.symbols("half_line_pole_x half_line_pole_k")
    source = s.Mul(0, s.Pow(0, power, evaluate=False), s.exp(-x), evaluate=False)
    with pytest.raises(ValueError):
        transform(source, x, k)


def test_identically_zero_original_base_is_checked_before_a_zeroth_power():
    x, k = s.symbols("half_line_zero_base_x half_line_zero_base_k")
    zero = s.Mul(0, x, evaluate=False)
    source = s.Mul(s.Pow(zero, 0, evaluate=False), s.exp(-x), evaluate=False)
    with pytest.raises(ValueError):
        s.cosine_transform(source, x, k)


@pytest.mark.parametrize("dummy", [False, True])
def test_same_printed_name_preserves_typed_symbol_owners(dummy):
    if dummy:
        x, k = s.Dummy("half_line_same"), s.Dummy("half_line_same")
    else:
        x = s.Symbol("half_line_same")
        k = s.Symbol("half_line_same", real=True)
    result = s.cosine_transform(s.exp(-x*x), x, k)
    assert result.free_symbols == {k}
    assert next(iter(result.free_symbols)) is k
    with pytest.raises(ValueError, match="distinct"):
        s.cosine_transform(s.exp(-x*x), x, x)


def test_original_frequency_dependence_is_rejected_under_a_zero_multiplier():
    x, k = s.symbols("half_line_frequency_x half_line_frequency_k")
    source = s.Add(s.exp(-x), s.Mul(0, k, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="frequency"):
        s.sine_transform(source, x, k)


def test_source_and_retained_argument_mutations_are_rejected(monkeypatch):
    x, k = s.symbols("half_line_mutation_x half_line_mutation_k")
    source = s.exp(-x)
    original = Basic.__dict__["_value"].__get__(source, Basic)
    native = _native.half_line_transform_native

    def changed(*args):
        result = native(*args)
        Basic.__dict__["_value"].__set__(source, _native_expr(s.Integer(0)))
        return result

    monkeypatch.setattr(_native, "half_line_transform_native", changed)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            s.sine_transform(source, x, k)
    finally:
        Basic.__dict__["_value"].__set__(source, original)
    monkeypatch.setattr(_native, "half_line_transform_native", native)
    held = s.exp(-x, evaluate=False)
    held._args = (s.Integer(3),)
    with pytest.raises(ValueError, match="retained"):
        s.sine_transform(held, x, k)


@pytest.mark.parametrize("failure", [ValueError("half-line budget exhausted"), RuntimeError("cancelled")])
def test_operational_failures_are_terminal_with_the_original_exception(monkeypatch, failure):
    x, k = s.symbols("half_line_failure_x half_line_failure_k")

    def failed(*args):
        raise failure

    monkeypatch.setattr(_native, "half_line_transform_native", failed)
    with pytest.raises(type(failure)) as observed:
        s.cosine_transform(s.exp(-x), x, k)
    assert observed.value is failure


def test_custom_callbacks_and_truth_conversions_are_never_invoked(monkeypatch):
    x, k = s.symbols("half_line_callbacks_x half_line_callbacks_k")
    calls = []

    class Convertible:
        def _sympy_(self):
            calls.append("converter")
            raise AssertionError("custom converter was invoked")

    class Truthy:
        def __bool__(self):
            calls.append("truth")
            raise AssertionError("custom truth conversion was invoked")

    with pytest.raises(TypeError):
        s.sine_transform(Convertible(), x, k)
    with pytest.raises(TypeError):
        s.sine_transform(0, x, k, noconds=Truthy())
    with pytest.raises(TypeError):
        s.cosine_transform(0, x, k, unknown_option=True)
    source = s.exp(-x)

    def custom(*args):
        calls.append("function")
        raise AssertionError("custom function hook was invoked")

    monkeypatch.setattr(s.exp, "_eval_is_real", custom, raising=False)
    with pytest.raises(NotImplementedError):
        s.cosine_transform(source, x, k)
    assert calls == []


def test_scalar_admission_rejects_noncommutativity_and_bounds_before_work():
    x, k = s.symbols("half_line_bound_x half_line_bound_k")
    noncommuting = s.Symbol("half_line_noncommuting", commutative=False)
    with pytest.raises(NotImplementedError):
        s.sine_transform(s.exp(-x), x, noncommuting)
    with pytest.raises(ValueError, match="4096"):
        s.cosine_transform(1 << 4097, x, k)
    source = s.exp(-s.Pow(x, 10**9, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="power"):
        s.cosine_transform(source, x, k)
    for literal in [Fraction(2, 3), s.Float("0.125")]:
        result = s.cosine_transform(literal, x, k)
        assert type(result) is s.CosineTransform
