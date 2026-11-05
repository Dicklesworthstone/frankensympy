"""Closed Heaviside sign evaluation and native-result surface lifting."""

import pytest
import sympy as s
from sympy.core import _lift_native_result, _native, _native_expr


def test_definite_symbol_and_compound_signs_choose_the_step_value():
    positive = s.Symbol("positive_step", positive=True)
    negative = s.Symbol("negative_step", negative=True)
    zero = s.Symbol("zero_step", zero=True)
    for argument in (positive, -negative, 1-negative):
        assert argument.is_positive is True
        assert s.Heaviside(argument) == 1
    for argument in (negative, -positive, negative-1):
        assert argument.is_negative is True
        assert s.Heaviside(argument) == 0
    assert s.Heaviside(zero) == s.S.Half
    assert s.Heaviside(zero, s.Rational(2, 3)) == s.Rational(2, 3)
    assert s.Heaviside(positive, 0) == 1
    assert s.Heaviside(negative, 1) == 0


def test_nonstrict_unknown_and_nonreal_signs_remain_held():
    values = (
        s.Symbol("unknown_step"),
        s.Symbol("nonnegative_step", nonnegative=True),
        s.Symbol("nonpositive_step", nonpositive=True),
        s.Symbol("nonzero_real_step", real=True, nonzero=True),
        s.I,
    )
    for argument in values:
        result = s.Heaviside(argument)
        assert type(result) is s.Heaviside
        assert result.pargs == (argument,)


def test_evaluate_false_keeps_the_original_signed_argument():
    argument = 1-s.Symbol("held_negative_step", negative=True)
    result = s.Heaviside(argument, evaluate=False)
    assert type(result) is s.Heaviside
    assert result.pargs == (argument,)
    assert s.Heaviside(0, evaluate=False).pargs == (s.S.Zero,)


def test_lifted_native_steps_use_the_same_proved_public_signs():
    negative = s.Symbol("lift_negative_step", negative=True)
    for argument, expected in ((1-negative, s.S.One), (negative, s.S.Zero)):
        raw = _native.py_function("Heaviside", _native_expr(argument))
        assert _lift_native_result(raw, argument) == expected
    raw = _native.py_function("Heaviside", _native_expr(0), _native_expr(s.Rational(2, 3)))
    assert _lift_native_result(raw) == s.Rational(2, 3)


def test_opaque_arguments_do_not_trigger_sign_or_printing_hooks():
    events = []

    class OpaqueStepArgument(s.Function):
        @property
        def is_positive(self):
            events.append("positive")
            raise AssertionError("opaque sign hook must not run")

        def __str__(self):
            events.append("print")
            raise AssertionError("opaque printer must not run")

    argument = OpaqueStepArgument(s.Symbol("opaque_step"))
    result = s.Heaviside(argument)
    assert type(result) is s.Heaviside
    assert result.args[0] is argument
    assert events == []


def test_modified_builtin_sign_helper_is_refused_before_execution(monkeypatch):
    x = s.Symbol("modified_step")
    raw = _native.py_function("Heaviside", _native_expr(x))
    events = []

    def changed(cls, *args):
        events.append("changed")
        return s.S.One

    monkeypatch.setattr(s.Heaviside, "_eval_known_sign", classmethod(changed))
    with pytest.raises(NotImplementedError, match="closed native lowering"):
        _lift_native_result(raw, x)
    assert events == []


def test_same_named_user_class_cannot_capture_lifted_step_nodes():
    builtin = s.Heaviside
    events = []

    class Heaviside(s.Function):
        @classmethod
        def eval(cls, *args):
            events.append("user")
            raise AssertionError("user class must not capture a native step")

    x = s.Symbol("registered_step")
    raw = _native.py_function("Heaviside", _native_expr(x))
    result = _lift_native_result(raw, x)
    assert type(result) is builtin
    assert result.pargs == (x,)
    assert events == []
