"""Bounded ordinary inverse Fourier integrals through the actual native bridge.

Gaussian/absolute-exponential formulas and sufficient conditions were checked
against isolated SymPy 1.14.0. General native Cauchy kernels intentionally stay
outside this Python compatibility subset.
"""

from fractions import Fraction

import pytest
import sympy as s
from sympy.core import Basic, _native, _native_expr, _symbol_binding


def equal(actual, expected):
    assert s.simplify(s.expand(actual - expected)) == 0


@pytest.mark.parametrize("center", [0, 2, s.Rational(-2, 3)])
@pytest.mark.parametrize("family", ["gaussian", "bilateral"])
def test_inverse_translation_has_the_positive_kernel_phase(center, family):
    k, x = s.symbols("inverse_fourier_phase_k inverse_fourier_phase_x")
    if family == "gaussian":
        source = s.exp(-2*(k-center)**2)
        envelope = s.sqrt(s.pi/2)*s.exp(-s.pi*s.pi*x*x/2)
    else:
        source = s.exp(-3*s.Abs(k-center))
        envelope = 6/(9+4*s.pi*s.pi*x*x)
    result = s.inverse_fourier_transform(source, k, x)
    equal(result, envelope*s.exp(2*s.pi*s.I*center*x))
    equal(result.subs(x, 0), envelope.subs(x, 0))


def test_inverse_gaussian_completion_and_finite_linearity():
    k, x = s.symbols("inverse_fourier_sum_k inverse_fourier_sum_x")
    result = s.inverse_fourier_transform(s.exp(-2*k*k+3*k+1), k, x)
    expected = s.sqrt(s.pi/2)*s.exp(s.Rational(17, 8)-s.pi*s.pi*x*x/2)*s.exp(3*s.pi*s.I*x/2)
    equal(result, expected)
    source = 3*s.exp(-k*k)-2*s.exp(-s.Abs(k))
    equal(s.inverse_fourier_transform(source, k, x).subs(x, 0), 3*s.sqrt(s.pi)-4)


@pytest.mark.parametrize("facts", [{}, {"real": True}, {"positive": True}, {"negative": True}])
def test_inverse_bilateral_pinned_sufficient_conditions(facts):
    k = s.Symbol("inverse_fourier_condition_k")
    x = s.Symbol("inverse_fourier_condition_x", **facts)
    result, condition = s.inverse_fourier_transform(s.exp(-s.Abs(k)), k, x, noconds=False)
    equal(result, 2/(1+4*s.pi*s.pi*x*x))
    assert condition == (True if facts.get("positive") or facts.get("negative") else x > 0)


def test_inverse_gaussian_and_zero_conditions():
    k, x = s.symbols("inverse_fourier_zero_k inverse_fourier_zero_x")
    result, condition = s.inverse_fourier_transform(s.exp(-k*k), k, x, noconds=False)
    equal(result, s.sqrt(s.pi)*s.exp(-s.pi*s.pi*x*x))
    assert condition is True
    assert s.inverse_fourier_transform(0, k, x) == 0
    assert s.inverse_fourier_transform(0, k, x, noconds=False) == (0, True)


@pytest.mark.parametrize("coefficient", [1, 5, -2, Fraction(2, 3)])
def test_inverse_constants_retain_ordinary_held_transform_and_scalar(coefficient):
    k, x = s.symbols("inverse_fourier_constant_k inverse_fourier_constant_x")
    scalar = s.Rational(coefficient.numerator, coefficient.denominator) if type(coefficient) is Fraction else coefficient
    expected = scalar*s.InverseFourierTransform(1, k, x)
    result = s.inverse_fourier_transform(coefficient, k, x)
    assert result == expected
    assert result.free_symbols == {x}
    assert s.inverse_fourier_transform(coefficient, k, x, noconds=False) == expected
    assert s.InverseFourierTransform(1, k, x).doit() == s.InverseFourierTransform(1, k, x)


def test_forward_and_inverse_held_objects_remain_distinct_opaque_operands():
    k, x = s.symbols("inverse_fourier_held_k inverse_fourier_held_x")
    inverse = s.InverseFourierTransform(1, k, x)
    forward = s.FourierTransform(1, k, x)
    assert inverse != forward
    assert 2*inverse+3*inverse == 5*inverse
    assert (inverse*inverse).args[0] is inverse
    assert inverse+0 is inverse
    mixed = inverse+forward
    assert mixed.free_symbols == {x}
    assert mixed.atoms(s.InverseFourierTransform) == {inverse}
    assert mixed.atoms(s.FourierTransform) == {forward}
    for value in (inverse, mixed, 2*inverse):
        with pytest.raises(NotImplementedError):
            _native_expr(value)


@pytest.mark.parametrize("dummy", [False, True])
def test_same_name_coordinates_preserve_exact_source_owners(dummy):
    if dummy:
        k, x = s.Dummy("inverse_fourier_same"), s.Dummy("inverse_fourier_same")
    else:
        k = s.Symbol("inverse_fourier_same")
        x = s.Symbol("inverse_fourier_same", real=True)
    value = s.inverse_fourier_transform(s.exp(-k*k), k, x)
    assert value.free_symbols == {x}
    assert next(iter(value.free_symbols)) is x
    held = s.inverse_fourier_transform(5, k, x).args[1]
    assert held.args[1] is k and held.args[2] is x
    assert next(iter(held.free_symbols)) is x


@pytest.mark.parametrize("case", ["growing", "unknown", "sum", "product", "cauchy"])
def test_unsupported_whole_sources_keep_original_held_identity(case):
    k, x, a = s.symbols("inverse_fourier_formal_k inverse_fourier_formal_x inverse_fourier_formal_a")
    source = {
        "growing": s.exp(k*k), "unknown": s.exp(-a*k*k),
        "sum": s.exp(-k*k)+s.sin(k), "product": s.exp(-k*k)*s.sin(k),
        "cauchy": 1/(k*k+1),
    }[case]
    result = s.inverse_fourier_transform(source, k, x)
    assert type(result) is s.InverseFourierTransform
    assert result.args[0] is source
    assert result.args == (source, k, x)


@pytest.mark.parametrize("noconds", [True, False])
def test_bilateral_nonreal_output_is_not_admitted_as_a_real_axis_formula(noconds):
    k = s.Symbol("inverse_fourier_nonreal_k")
    x = s.Symbol("inverse_fourier_nonreal_x", imaginary=True)
    source = s.exp(-s.Abs(k))
    result = s.inverse_fourier_transform(source, k, x, noconds=noconds)
    assert type(result) is s.InverseFourierTransform
    assert result.args[0] is source


def test_generic_integration_and_forward_kernel_are_never_used(monkeypatch):
    k = s.Symbol("inverse Fourier input with spaces")
    x = s.Symbol("inverse Fourier output with spaces", real=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("inverse catalog entered a generic or forward path")

    monkeypatch.setattr(s, "integrate", forbidden)
    monkeypatch.setattr(_native, "fourier_l1_native", forbidden)
    monkeypatch.setattr(_native, "fourier_expr", forbidden)
    value = s.inverse_fourier_transform(s.exp(-(k-2)**2), k, x)
    assert value.free_symbols == {x}
    equal(value.subs(x, 0), s.sqrt(s.pi))


@pytest.mark.parametrize("exception", [ValueError, RuntimeError])
def test_native_operational_errors_are_terminal_without_formal_fallback(monkeypatch, exception):
    k, x = s.symbols("inverse_fourier_failure_k inverse_fourier_failure_x")
    failure = exception("inverse Fourier operation failed")

    def fail(*args):
        raise failure

    monkeypatch.setattr(_native, "inverse_fourier_l1_native", fail)
    with pytest.raises(exception) as observed:
        s.inverse_fourier_transform(s.exp(-k*k), k, x)
    assert observed.value is failure


def test_source_mutation_prevents_inverse_publication(monkeypatch):
    k, x = s.symbols("inverse_fourier_mutation_k inverse_fourier_mutation_x")
    source = s.exp(-k*k)
    old = Basic.__dict__["_value"].__get__(source, Basic)
    native = _native.inverse_fourier_l1_native

    def changed(*args):
        value = native(*args)
        Basic.__dict__["_value"].__set__(source, _native_expr(s.Integer(0)))
        return value

    monkeypatch.setattr(_native, "inverse_fourier_l1_native", changed)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            s.inverse_fourier_transform(source, k, x)
    finally:
        Basic.__dict__["_value"].__set__(source, old)


def test_preexisting_retained_mismatch_refuses_before_inverse_evaluation():
    k, x = s.symbols("inverse_fourier_retained_k inverse_fourier_retained_x")
    source = s.exp(-k*k, evaluate=False)
    source._args = (s.Integer(3),)
    with pytest.raises(ValueError, match="retained"):
        s.inverse_fourier_transform(source, k, x)
    native = _native_expr(s.exp(-k*k))
    with pytest.raises(ValueError, match="retained"):
        _native.inverse_fourier_l1_native(native, _symbol_binding(k), _symbol_binding(x),
                                          [(native, "exp", (_native_expr(s.Integer(0)),))])


@pytest.mark.parametrize("attribute", ["_args", "_struct_args", "_surface_args", "_value"])
def test_held_inverse_arithmetic_requires_consistent_original_storage(attribute):
    k, x = s.symbols("inverse_fourier_storage_k inverse_fourier_storage_x")
    held = s.InverseFourierTransform(1, k, x)
    value = _native_expr(s.Integer(1)) if attribute == "_value" else (s.Integer(2), k, x)
    setattr(held, attribute, value)
    with pytest.raises(ValueError, match="inconsistent retained arguments"):
        s.Add(held, 1)


def test_inverse_custom_hooks_and_noncommutative_inputs_do_not_reach_native(monkeypatch):
    k = s.Symbol("inverse_fourier_owner_k")
    x = s.Symbol("inverse_fourier_owner_x", commutative=False)
    source = s.exp(-k*k)
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("unadmitted owner entered native evaluation")

    monkeypatch.setattr(_native, "inverse_fourier_l1_native", forbidden)
    with pytest.raises(NotImplementedError, match="commuting"):
        s.inverse_fourier_transform(source, k, x)
    x = s.Symbol("inverse_fourier_owner_real_x")
    monkeypatch.setattr(s.exp, "_eval_is_real", forbidden, raising=False)
    with pytest.raises(NotImplementedError):
        s.inverse_fourier_transform(source, k, x)
    assert calls == []


def test_inverse_zero_powers_source_aliases_and_size_limits_are_terminal():
    k, x = s.symbols("inverse_fourier_bounds_k inverse_fourier_bounds_x")
    invalid = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.inverse_fourier_transform(invalid, k, x)
    polluted = s.Add(s.exp(-k*k), s.Mul(0, x, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="frequency"):
        s.inverse_fourier_transform(polluted, k, x)
    with pytest.raises(ValueError, match="distinct"):
        s.inverse_fourier_transform(s.exp(-k*k), k, k)
    with pytest.raises(ValueError, match="4096"):
        s.inverse_fourier_transform(1 << 4097, k, x)


def test_inverse_converter_truthiness_and_options_are_not_executed():
    k, x = s.symbols("inverse_fourier_options_k inverse_fourier_options_x")

    class Hostile:
        def _sympy_(self):
            raise AssertionError("custom converter executed")
        def __bool__(self):
            raise AssertionError("custom truth conversion executed")

    with pytest.raises(TypeError):
        s.inverse_fourier_transform(Hostile(), k, x)
    with pytest.raises(TypeError):
        s.inverse_fourier_transform(0, k, x, noconds=Hostile())
    with pytest.raises(TypeError):
        s.inverse_fourier_transform(0, k, x, unknown=True)
