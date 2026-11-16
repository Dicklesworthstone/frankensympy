"""The shell lowers finite Float bit atoms before native complex evaluation."""

import cmath
import math

import pytest
import sympy as s

from sympy.core import _FLOAT_INTERN, _complex_value, _native, _native_expr


@pytest.mark.parametrize("real, imaginary", [
    (0.0, -1.0),
    (2.5, -4.25),
    (1e20, 1e-12),
    (1e308, -1e-308),
    (0.0, float.fromhex("0x0.0000000000001p-1022")),
])
def test_floating_components_keep_their_independent_binary64_values(real, imaginary):
    source = s.Float(real) + s.Float(imaginary)*s.I
    value = complex(source)
    assert value.real == real
    assert value.imag == imaginary
    assert _complex_value(source) == complex(real, imaginary)


@pytest.mark.parametrize("function, expected", [
    (s.exp, cmath.exp),
    (s.sin, cmath.sin),
    (s.cos, cmath.cos),
])
def test_floats_inside_elementary_complex_functions_are_lowered(function, expected):
    argument = s.Float(0.25) + s.Float(0.5)*s.I
    source = function(argument, evaluate=False)
    assert _complex_value(source) == pytest.approx(expected(complex(0.25, 0.5)))


def test_held_raw_structure_and_complex_function_tags_survive_float_lowering():
    source = s.Add(
        s.Float(1.25),
        s.Mul(s.Float(-2.5), s.I, evaluate=False),
        evaluate=False,
    )
    before = _native_expr(source)
    arguments = source.args
    assert _complex_value(source) == complex(1.25, -2.5)
    assert source.args == arguments
    assert _native_expr(source) == before
    magnitude = s.Abs(s.Mul(s.Float(-2.5), s.I, evaluate=False), evaluate=False)
    assert _complex_value(magnitude) == complex(2.5, 0.0)
    reciprocal = s.Pow(source, s.Integer(-1), evaluate=False)
    assert _complex_value(reciprocal) == pytest.approx(1/complex(1.25, -2.5))


@pytest.mark.parametrize("payload", [
    (),
    (_native.py_integer(1), _native.py_integer(2)),
    (_native.py_integer(-1),),
    (_native.py_integer(1 << 64),),
    (_native.py_rational(1, 2),),
    (_native.py_integer(0x7ff0000000000000),),
    (_native.py_integer(0x7ff8000000000000),),
])
def test_malformed_or_nonfinite_private_float_payloads_refuse(payload):
    source = _native.py_function(_FLOAT_INTERN, *payload)
    assert _complex_value(source) is None


def test_original_undefined_children_cannot_disappear_in_a_held_zero_product():
    pole = _native.Pow(_native.py_integer(0), _native.py_integer(-1), evaluate=False).as_expr()
    zero = _native_expr(s.Float(0.0))
    source = _native.Mul(zero, pole, evaluate=False).as_expr()
    result = _complex_value(source)
    assert result is None or not (math.isfinite(result.real) and math.isfinite(result.imag))


def test_symbolic_and_opaque_sources_do_not_run_user_hooks():
    x = s.Symbol("complex_float_unbound")
    assert _complex_value(x + s.Float(0.5)*s.I) is None
    calls = []

    class Custom(s.Function):
        def _eval_evalf(self, prec):
            calls.append("evalf")
            return s.Integer(42)

        def __str__(self):
            calls.append("printer")
            return "1"

    assert _complex_value(Custom(s.Float(0.5), evaluate=False)) is None
    assert calls == []


def test_source_preflight_rejects_deep_native_trees_before_recursive_lowering():
    source = _native_expr(s.Float(0.125))
    for _ in range(130):
        source = _native.py_function("sin", source)
    assert _complex_value(source) is None


def test_operational_allocation_failure_is_not_reported_as_unsupported(monkeypatch):
    def fail(_source):
        raise MemoryError("injected native allocation refusal")

    monkeypatch.setattr(_native, "complex_value_native", fail)
    with pytest.raises(MemoryError, match="allocation refusal"):
        _complex_value(s.I)
