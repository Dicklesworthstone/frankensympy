"""Actual public-shell integration of independent proper-integral bounds."""

from fractions import Fraction
from math import factorial
import importlib

import pytest
from sympy import (
    Float, Function, IntegralEnclosureReceipt, Mul, Pow, Rational, Symbol,
    certified_integral, exp, log, sin, sqrt,
)
from sympy.core import _native, _symbol_binding
from sympy.integrals import certified_integral as module_integral


def _fraction(value):
    return Fraction(value.p, value.q)


def test_public_exports_and_exact_affine_integral_receipt():
    assert certified_integral is module_integral
    x = Symbol("affine_integral_x")
    result = certified_integral(3*x + 2, (x, -2, 5), tolerance=Fraction(1, 10**100))
    assert type(result) is IntegralEnclosureReceipt
    assert result.bounds == (Rational(91, 2), Rational(91, 2))
    assert result.midpoint == Rational(91, 2)
    assert result.error_bound == 0
    assert result.interval == (-2, 5)
    assert result.partition == (-2, 5)
    assert result.interval_count == 1
    assert result.precision_digits == 32
    assert result.integrand == 3*x+2
    assert result.variable.name == x.name
    assert "proper real integral" in result.claim


def test_adaptive_quadratic_has_exact_checked_width_and_error():
    x = Symbol("quadratic_integral_x")
    result = certified_integral(x*x, (x, Fraction(1, 2), Rational(3, 2)),
                                tolerance=Fraction(1, 1000), precision=16)
    lower, upper = result.bounds
    assert lower <= Rational(13, 12) <= upper
    assert upper-lower <= Rational(1, 1000)
    assert result.midpoint-result.error_bound == lower
    assert result.midpoint+result.error_bound == upper
    assert result.interval_count > 1
    assert result.absolute_tolerance == Rational(1, 1000)
    assert result.partition[0] == Rational(1, 2)
    assert result.partition[-1] == Rational(3, 2)
    assert all(a < b for a, b in zip(result.partition, result.partition[1:]))


def test_gaussian_without_elementary_primitive_matches_independent_series_enclosure():
    x = Symbol("gaussian_integral_x")
    result = certified_integral(exp(-x*x), (x, 0, 1), tolerance=Fraction(1, 1000), precision=16)
    lower, upper = map(_fraction, result.bounds)
    odd = sum((Fraction((-1)**n, factorial(n)*(2*n+1)) for n in range(12)), Fraction(0))
    even = odd + Fraction(1, factorial(12)*25)
    assert lower <= odd < even <= upper
    assert upper-lower <= Fraction(1, 1000)


def test_fresnel_integral_reversed_limits_and_negative_enclosures():
    x = Symbol("fresnel_integral_x")
    result = certified_integral(sin(x*x), (x, 0, 1), tolerance=Fraction(1, 1000), precision=16)
    reversed_result = certified_integral(sin(x*x), (x, 1, 0), tolerance=Fraction(1, 1000), precision=16)
    lo, hi = result.bounds
    assert reversed_result.bounds == (-hi, -lo)
    assert reversed_result.interval == (1, 0)
    assert reversed_result.partition == result.partition
    series_lower = Fraction(1, 3)-Fraction(1, 42)+Fraction(1, 1320)-Fraction(1, 75600)
    series_upper = series_lower+Fraction(1, 6_894_720)
    assert _fraction(lo) <= series_lower < series_upper <= _fraction(hi)


def test_equal_limits_require_the_original_expression_defined_at_the_point():
    x = Symbol("point_integral_x")
    result = certified_integral(x*x, (x, 2, 2))
    assert result.bounds == (0, 0)
    assert result.partition == (2,)
    assert result.interval_count == 0
    with pytest.raises(ValueError):
        certified_integral(Pow(x, -1, evaluate=False), (x, 0, 0))


@pytest.mark.parametrize("kind", ["pole", "zero_product", "cancelled", "zero_power", "log", "sqrt"])
def test_original_domain_and_endpoint_failures_are_not_cancelled_away(kind):
    x = Symbol("integral_domain_"+kind)
    reciprocal = Pow(x, -1, evaluate=False)
    source = {
        "pole": reciprocal,
        "zero_product": Mul(0, reciprocal, evaluate=False),
        "cancelled": Mul(x, reciprocal, evaluate=False),
        "zero_power": Pow(reciprocal, 0, evaluate=False),
        "log": log(x), "sqrt": sqrt(x),
    }[kind]
    with pytest.raises(ValueError):
        certified_integral(source, (x, 0, 1))


def test_interior_pole_accuracy_and_resource_failures_publish_no_receipt():
    x = Symbol("bounded_integral_x")
    with pytest.raises(ValueError):
        certified_integral(1/(x-Rational(1, 3)), (x, 0, 1), max_intervals=16)
    with pytest.raises(ValueError, match="width"):
        certified_integral(x*x, (x, 0, 1), tolerance=Fraction(1, 10**20), max_intervals=1)
    for count in (0, 1025, True, 512.0):
        with pytest.raises(ValueError, match="max_intervals"):
            certified_integral(x, (x, 0, 1), max_intervals=count)
    for precision in (7, 129, True, 32.0):
        with pytest.raises(ValueError, match="precision"):
            certified_integral(x, (x, 0, 1), precision=precision)
    with pytest.raises(ValueError, match="4096-bit"):
        certified_integral(x, (x, 0, 1 << 4097))
    for tolerance in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            certified_integral(x, (x, 0, 1), tolerance=tolerance)


def test_inexact_inputs_opaque_functions_and_conversion_hooks_refuse():
    x = Symbol("exact_integral_x")
    for endpoint in (0.0, Float(0), True, "0"):
        with pytest.raises(TypeError):
            certified_integral(x, (x, endpoint, 1))
    for source in (0.25, Float(0.25), Mul(Float(0.25), x, evaluate=False)):
        with pytest.raises(ValueError, match="exact literals"):
            certified_integral(source, (x, 0, 1))
    with pytest.raises(NotImplementedError):
        certified_integral(Function("opaque_integrand")(x), (x, 0, 1))
    calls = []

    class Hook:
        def _sympy_(self):
            calls.append("sympify")
            return 0
        def __int__(self):
            calls.append("int")
            return 0

    with pytest.raises(TypeError):
        certified_integral(x, (x, Hook(), 1))
    with pytest.raises(NotImplementedError):
        certified_integral(Hook(), (x, 0, 1))
    assert calls == []


def test_whole_interval_assumptions_and_typed_symbols_are_respected():
    x = Symbol("positive_integral_x", positive=True)
    result = certified_integral(x, (x, 1, 2))
    assert result.bounds == (Rational(3, 2), Rational(3, 2))
    assert result.variable.identity_hex == _symbol_binding(x).identity_hex
    with pytest.raises(ValueError, match="positive"):
        certified_integral(x, (x, -1, 2))
    same_name = Symbol("positive_integral_x")
    with pytest.raises(ValueError, match="unbound"):
        certified_integral(x, (same_name, 1, 2))
    integer = Symbol("integer_integral_x", integer=True)
    with pytest.raises(NotImplementedError, match="assumptions"):
        certified_integral(integer, (integer, 1, 2))


def test_assumption_mutation_after_lowering_does_not_relabel_native_receipt(monkeypatch):
    module = importlib.import_module("sympy.integrals.certified")
    x = Symbol("integral_assumption_snapshot_x", positive=True)
    original_facts = dict(x._assumptions)
    binding = _symbol_binding(x)
    lower_source = module._numeric_source

    def mutate_after_lowering(*args, **kwargs):
        source = lower_source(*args, **kwargs)
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return source

    monkeypatch.setattr(module, "_numeric_source", mutate_after_lowering)
    try:
        result = certified_integral(x, (x, 1, 2))
        assert result.variable.identity_hex == binding.identity_hex
        assert result.bounds == (Rational(3, 2), Rational(3, 2))
        assert result.interval == (1, 2)
    finally:
        x._assumptions.clear()
        x._assumptions.update(original_facts)


def test_inconsistent_assumption_snapshot_refuses(monkeypatch):
    module = importlib.import_module("sympy.integrals.certified")
    x = Symbol("integral_capture_race_x", positive=True)
    original_facts = dict(x._assumptions)
    bind = module._integral_binding

    def mutate_after_binding(variable):
        binding = bind(variable)
        variable._assumptions.clear()
        variable._assumptions["negative"] = True
        return binding

    monkeypatch.setattr(module, "_integral_binding", mutate_after_binding)
    try:
        with pytest.raises(ValueError, match="captured symbol bindings"):
            certified_integral(x, (x, -2, -1))
    finally:
        x._assumptions.clear()
        x._assumptions.update(original_facts)


def test_mutated_symbol_facts_and_payloads_refuse_without_invoking_hooks():
    x = Symbol("integral_poisoned_binding_x")
    original_facts, original_payload = dict(x._assumptions), x._value
    calls = []

    class Hook:
        def __bool__(self):
            calls.append("bool")
            return True
        def __str__(self):
            calls.append("str")
            return "positive"
        @property
        def symbol_binding(self):
            calls.append("binding")
            return None

    try:
        for facts in ({"positive": Hook()}, {Hook(): True}):
            x._assumptions.clear()
            x._assumptions.update(facts)
            with pytest.raises(ValueError, match="captured symbol bindings"):
                certified_integral(x, (x, 1, 2))
        x._assumptions.clear()
        x._value = Hook()
        with pytest.raises(TypeError, match="exact native symbol payload"):
            certified_integral(1, (x, 1, 2))
        assert calls == []
    finally:
        x._assumptions.clear()
        x._assumptions.update(original_facts)
        x._value = original_payload


def test_custom_variable_types_refuse_without_class_or_conversion_hooks():
    calls = []

    class FakeSymbol:
        @property
        def __class__(self):
            calls.append("class")
            return Symbol

    class CustomSymbol(Symbol):
        pass

    for variable in (FakeSymbol(), CustomSymbol("custom_integral_variable")):
        with pytest.raises(TypeError, match="exact Symbol or Dummy"):
            certified_integral(1, (variable, 0, 1))
    assert calls == []


def test_immutable_receipts_cannot_be_forged_or_relabelled():
    x = Symbol("immutable_integral_x")
    result = certified_integral(x, (x, 0, 1))
    with pytest.raises(AttributeError):
        result.interval = (10, 20)
    with pytest.raises(AttributeError):
        result.__init__(result._receipt)
    with pytest.raises(AttributeError):
        result.variable.name = "relabelled"
    for cls in (IntegralEnclosureReceipt, _native.IntegralEnclosureReceipt):
        with pytest.raises(TypeError):
            cls({"verified": True})


def test_direct_native_bridge_rejects_nonliteral_bounds_and_custom_payloads():
    x = Symbol("native_integral_x")
    with pytest.raises(ValueError, match="rational literals"):
        _native.certified_integral_native(
            x._value, _symbol_binding(x), x._value, _native.py_integer(1),
            _native.py_rational(1, 1000),
        )
    with pytest.raises(TypeError):
        _native.certified_integral_native(
            x._value, _symbol_binding(x), 0, 1, Fraction(1, 1000),
        )
