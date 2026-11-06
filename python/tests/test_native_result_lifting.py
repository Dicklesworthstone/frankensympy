"""Native results retain complete applications and built-in branch conditions."""

import pytest
import sympy as s

from sympy.core import AppliedUndef, _lift_native_result, _native, _native_expr, _wrap


def test_undefined_native_application_retains_arguments_and_symbol_identity():
    x = s.Symbol("native_lift_argument", positive=True)
    y = s.Dummy("native_lift_argument")
    function = s.Function("native_lift_formal")
    native = _native.py_function(function.__name__, _native_expr(x), _native_expr(y + 1))
    result = _wrap(native)
    assert type(result) is function
    assert result.args == (x, y + 1)
    assert result.args[0] is x
    assert result.free_symbols == {x, y}
    assert result == function(x, y + 1, evaluate=False)
    # Recovering arguments does not grant opaque applications native admission.
    with pytest.raises(NotImplementedError, match="opaque"):
        _native_expr(result)


def test_nested_native_formal_arguments_are_not_lost():
    x = s.Symbol("native_lift_nested_x")
    inner = _native.py_function("native_lift_inner", _native_expr(x))
    result = _wrap(_native.py_function("native_lift_outer", inner))
    assert result.args[0].args == (x,)
    assert result.free_symbols == {x}


def test_native_function_lifting_does_not_invoke_user_eval_or_setters():
    calls = []

    class NativeLiftHookTrap(s.Function):
        @classmethod
        def eval(cls, *args):
            calls.append("eval")
            raise AssertionError("native lifting must not call user evaluation")

        def __setattr__(self, name, value):
            calls.append(name)
            raise AssertionError("native lifting must not call a user setter")

    x = s.Symbol("native_lift_hook_x")
    result = _wrap(_native.py_function("NativeLiftHookTrap", _native_expr(x)))
    assert type(result) is NativeLiftHookTrap
    assert result.args == (x,)
    assert calls == []
    with pytest.raises(NotImplementedError, match="opaque"):
        _native_expr(result)
    assert calls == []


@pytest.mark.parametrize("tag, cls", [
    ("Equality", s.Eq), ("Unequality", s.Ne),
    ("StrictLessThan", s.Lt), ("LessThan", s.Le),
    ("StrictGreaterThan", s.Gt), ("GreaterThan", s.Ge),
])
def test_native_long_relational_tags_recover_exact_builtin_classes(tag, cls):
    x, y = s.symbols("native_relation_x native_relation_y")
    native = _native.py_function(tag, _native_expr(x), _native_expr(y))
    result = _lift_native_result(native, x, y)
    assert type(result) is cls
    assert result.args == (x, y)
    assert result.lhs == x and result.rhs == y
    assert result == cls(x, y, evaluate=False)
    assert _native_expr(result) == native


def test_piecewise_derivative_preserves_native_branch_tuples_and_conditions():
    x, n = s.symbols("native_piecewise_x native_piecewise_n")
    source = s.Piecewise((x**(n + 1)/(n + 1), s.Ne(n, -1)), (s.log(x), True))
    derivative = s.diff(source, x)
    assert type(derivative) is s.Piecewise
    assert derivative.args[0].expr == x**n
    assert type(derivative.args[0].cond) is s.Ne
    assert derivative.args[0].cond.args == (n, -1)
    assert derivative.args[1].expr == 1/x
    assert derivative.subs(n, 2) == x*x
    assert derivative.subs(n, -1) == 1/x


def test_native_piecewise_tag_cannot_be_captured_by_same_named_user_class(monkeypatch):
    import sympy.core as core
    from sympy.functions.elementary.piecewise import Piecewise as BuiltinPiecewise

    x = s.Symbol("native_piecewise_capture_x")
    source = BuiltinPiecewise((x*x, s.Gt(x, 0)), (x, True))
    # Restore the prior registry entry after the subclass registers itself.
    monkeypatch.setitem(core._user_functions, "Piecewise", BuiltinPiecewise)

    class Piecewise(s.Function):
        @classmethod
        def _from_native(cls, value):
            raise AssertionError("native branch tag was captured by a user class")

    result = _lift_native_result(_native_expr(source), source)
    assert type(result) is BuiltinPiecewise
    assert result.args[0].expr == x*x
    assert type(result.args[0].cond) is s.Gt


def test_fourier_constant_retains_the_ordinary_unevaluated_transform():
    t, w = s.symbols("native_fourier_t native_fourier_w")
    result = s.fourier_transform(5, t, w)
    # Pinned SymPy 1.14.0 does not substitute a distribution for this ordinary
    # transform. The unsupported unit integral retains both original owners.
    assert type(result) is s.Mul
    coefficient, transform = result.args
    assert coefficient == 5
    assert type(transform) is s.FourierTransform
    assert transform.args == (1, t, w)
    assert transform.args[1] is t
    assert transform.args[2] is w
    assert result.free_symbols == {w}
    assert not result.atoms(AppliedUndef)
    assert str(result) == "5*FourierTransform(1, native_fourier_t, native_fourier_w)"
    with pytest.raises(NotImplementedError, match="opaque"):
        _native_expr(result)
