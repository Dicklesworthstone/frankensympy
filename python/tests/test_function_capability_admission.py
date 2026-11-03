"""Closed programs bind unchanged function implementations, not mutable names."""

import math

import pytest
import sympy as s

from sympy.core import (
    Basic, FunctionClass, _NativeFunction, _function_capabilities,
    _is_closed_function_class, _native, _native_expr, diff_with_receipt,
    replay_diff_receipt,
)


def _closed_operations(source, variable, bundle):
    return (
        lambda: _native_expr(source),
        lambda: diff_with_receipt(source, variable),
        lambda: s.compile_numeric(variable, source),
        lambda: s.lambdify(variable, source, modules="math"),
        lambda: s.compile_residual_system([source], variable),
        lambda: s.compile_objective(source, variable),
        lambda: s.load_residual_system(bundle, residuals=[source], args=variable),
        lambda: s.compile_numeric_hybrid(variable, source, callbacks={}),
    )


@pytest.mark.parametrize("form", ["held", "native", "generic", "add", "mul", "pow", "new"])
def test_changed_derivative_cannot_authorize_closed_programs_or_receipts(monkeypatch, form):
    x = s.Symbol("capability_x")
    application = s.sin(x, evaluate=False)
    sources = {
        "held": application, "native": _native_expr(application),
        "generic": s.Expr(_native_expr(application)),
        "add": application + x, "mul": application * x, "pow": application**2,
    }
    source = sources.get(form, application)
    bundle = s.compile_residual_system([source], x).to_bytes()
    calls = []

    def derivative(self, variable):
        calls.append(variable)
        return s.Integer(999)

    monkeypatch.setattr(s.sin, "_eval_derivative", derivative, raising=False)
    if form == "new":
        source = s.sin(x, evaluate=False)
        assert Basic.__dict__["_value"].__get__(source) is None
    assert not _is_closed_function_class(s.sin)
    for operation in _closed_operations(source, x, bundle):
        with pytest.raises(NotImplementedError):
            operation()
    assert calls == []
    # User behavior remains available through the explicit supervised lane.
    assert s.diff(application, x) == 999
    assert calls == [x]


@pytest.mark.parametrize("kind", ["dense", "immutable", "sparse", "retained"])
def test_matrix_admission_checks_functions_hidden_in_native_cells(monkeypatch, kind):
    x = s.Symbol("capability_matrix_x")
    node = s.sin(x, evaluate=False)
    raw = _native_expr(node)
    if kind == "sparse":
        source = s.SparseMatrix(_native.SparseMatrix(1, 1, {(0, 0): raw}))
    elif kind == "retained":
        source = s.Matrix([[node]])
    else:
        cls = s.Matrix if kind == "dense" else s.ImmutableMatrix
        source = cls(_native.Matrix(1, 1, [raw]))
    bundle = s.compile_residual_system(source, x).to_bytes()
    calls = []

    def forbidden(*args):
        calls.append("hook")
        raise AssertionError("source admission must not call the override")

    monkeypatch.setattr(s.sin, "_eval_evalf", forbidden, raising=False)
    for operation in (
        lambda: s.compile_numeric(x, source),
        lambda: s.compile_residual_system(source, x),
        lambda: s.load_residual_system(bundle, residuals=source, args=x),
        lambda: s.compile_numeric_hybrid(x, source, callbacks={}),
    ):
        with pytest.raises(NotImplementedError):
            operation()
    assert calls == []


def test_unmodified_functions_remain_usable_and_restoring_code_readmits_source(monkeypatch):
    x = s.Symbol("capability_restore_x")
    source = s.sin(x, evaluate=False)
    other = s.cos(x) + x**2
    original = _NativeFunction.__dict__["eval"].__func__

    def changed(cls, *args):
        raise AssertionError("modified built-in eval cannot enter closed compilation")

    with monkeypatch.context() as patch:
        patch.setattr(s.sin, "eval", classmethod(changed))
        assert not _is_closed_function_class(s.sin)
        assert _is_closed_function_class(s.cos)
        with pytest.raises(NotImplementedError):
            s.compile_numeric(x, source)
        assert s.compile_numeric(x, other)(2) == pytest.approx(math.cos(2) + 4)
        assert s.compile_residual_system([x**2], x).evaluate(3) == ([9.0], [[6.0]])
    assert _NativeFunction.__dict__["eval"].__func__ is original
    assert _is_closed_function_class(s.sin)
    assert s.compile_numeric(x, source)(0.5) == pytest.approx(math.sin(0.5))
    result, receipt = diff_with_receipt(source, x)
    assert result == s.cos(x)
    assert replay_diff_receipt(receipt)


def test_in_place_inherited_code_mutation_is_not_hidden_by_descriptor_identity(monkeypatch):
    x = s.Symbol("capability_code_x")
    source = s.sin(x, evaluate=False)
    native = _native_expr(source)
    implementation = _NativeFunction.__dict__["eval"].__func__
    descriptor = _NativeFunction.__dict__["eval"]

    def changed(cls, *args):
        raise AssertionError("mutated implementation must not run during admission")

    with monkeypatch.context() as patch:
        patch.setattr(implementation, "__code__", changed.__code__)
        assert _NativeFunction.__dict__["eval"] is descriptor
        assert not _is_closed_function_class(s.sin)
        assert not _is_closed_function_class(s.cos)
        for value in (source, native):
            with pytest.raises(NotImplementedError):
                s.compile_numeric(x, value)
            with pytest.raises(NotImplementedError):
                diff_with_receipt(value, x)
        assert s.compile_numeric(x, x**2)(3) == 9
    assert _is_closed_function_class(s.sin)
    assert s.compile_numeric(x, native)(0) == 0


@pytest.mark.parametrize("attribute", ["args", "_args", "_value", "_surface_args", "_struct_args"])
def test_changed_descriptors_refuse_before_getters_run(monkeypatch, attribute):
    x = s.Symbol("capability_descriptor_x")
    source = s.sin(x, evaluate=False)
    bundle = s.compile_residual_system([source], x).to_bytes()
    calls = []

    def forbidden(self):
        calls.append(attribute)
        raise AssertionError("admission must inspect implementations without invoking them")

    monkeypatch.setattr(s.sin, attribute, property(forbidden), raising=False)
    for operation in _closed_operations(source, x, bundle):
        with pytest.raises(NotImplementedError):
            operation()
    assert calls == []


@pytest.mark.parametrize("attribute", ["__hash__", "__eq__", "__getattribute__", "__instancecheck__", "__subclasscheck__"])
def test_changed_metaclass_methods_do_not_run_during_admission(monkeypatch, attribute):
    x = s.Symbol("capability_metaclass_x")
    source = s.sin(x, evaluate=False)
    bundle = s.compile_residual_system([source], x).to_bytes()
    calls = []

    def forbidden(*args):
        calls.append(attribute)
        raise AssertionError("source capability checks must use actual type identity")

    with monkeypatch.context() as patch:
        patch.setattr(FunctionClass, attribute, forbidden, raising=False)
        for operation in _closed_operations(source, x, bundle):
            with pytest.raises(NotImplementedError):
                operation()
    assert calls == []


@pytest.mark.parametrize("attribute", ["fdiff", "_imp_", "is_real", "nargs", "_hashable_content"])
def test_class_implementation_changes_revoke_only_the_affected_builtin(monkeypatch, attribute):
    x = s.Symbol("capability_behavior_x")
    source = s.sin(x, evaluate=False)
    raw = _native_expr(source)
    calls = []

    def forbidden(*args):
        calls.append(attribute)
        raise AssertionError("a changed implementation cannot run during admission")

    monkeypatch.setattr(s.sin, attribute, forbidden, raising=False)
    for value in (source, raw):
        with pytest.raises(NotImplementedError):
            s.compile_numeric(x, value)
    assert _is_closed_function_class(s.cos)
    assert s.compile_numeric(x, s.cos(x))(0) == 1
    assert calls == []


def test_compiled_snapshots_keep_the_original_closed_semantics_after_mutation(monkeypatch):
    x = s.Symbol("capability_snapshot_x")
    source = s.sin(x, evaluate=False)
    numeric = s.compile_numeric(x, source)
    residual = s.compile_residual_system([source], x)
    objective = s.compile_objective(source, x)
    hybrid = s.compile_numeric_hybrid(x, source, callbacks={})
    math_program = s.lambdify(x, source, modules="math")
    calls = []

    def changed(*args):
        calls.append("override")
        raise AssertionError("a compiled snapshot cannot start using ambient Python overrides")

    monkeypatch.setattr(s.sin, "_eval_derivative", changed, raising=False)
    point = 0.5
    expected = math.sin(point)
    assert numeric(point) == pytest.approx(expected)
    assert math_program(point) == pytest.approx(expected)
    assert hybrid(point) == pytest.approx(expected)
    values, jacobian = residual.evaluate(point)
    assert values == pytest.approx([expected])
    assert jacobian[0] == pytest.approx([math.cos(point)])
    value, gradient, hessian = objective.evaluate(point)
    assert value == pytest.approx(expected)
    assert gradient == pytest.approx([math.cos(point)])
    assert hessian[0] == pytest.approx([-expected])
    assert calls == []


@pytest.mark.parametrize("function,tag", [(s.log, "ln"), (s.Abs, "abs")])
def test_native_aliases_cannot_bypass_changed_implementation_guards(monkeypatch, function, tag):
    x = s.Symbol("capability_alias_x")
    raw = _native.py_function(tag, _native_expr(x))
    monkeypatch.setattr(function, "_eval_derivative", lambda *args: 7, raising=False)
    with pytest.raises(NotImplementedError):
        _native_expr(raw)
    # This does not expand the numeric compiler's existing native-tag allowlist.
    with pytest.raises(NotImplementedError):
        s.compile_numeric(x, raw)


def test_capability_roster_is_immutable_and_does_not_trust_name_registry_replacement(monkeypatch):
    from sympy.core import _known_functions

    x = s.Symbol("capability_registry_x")
    source = s.sin(x, evaluate=False)
    with pytest.raises(AttributeError, match="immutable"):
        _function_capabilities._rows = ()
    with pytest.raises(AttributeError, match="immutable"):
        del _function_capabilities._rows
    impostor = type("sin", (s.Function,), {"__module__": "sympy.fake"})
    monkeypatch.setitem(_known_functions, "sin", impostor)
    assert _is_closed_function_class(s.sin)
    assert not _is_closed_function_class(impostor)
    assert s.compile_numeric(x, source)(0) == 0
    with pytest.raises(NotImplementedError):
        s.compile_numeric(x, impostor(x, evaluate=False))


def test_cyclic_handler_metadata_refuses_without_unbounded_walk(monkeypatch):
    x = s.Symbol("capability_cycle_x")
    source = s.sin(x, evaluate=False)
    cyclic = {}
    cyclic["self"] = cyclic
    monkeypatch.setattr(s.sin, "_prop_handler", cyclic, raising=False)
    assert not _is_closed_function_class(s.sin)
    with pytest.raises(NotImplementedError):
        s.compile_numeric(x, source)


def test_class_name_subclasses_cannot_run_equality_during_capability_checks():
    x = s.Symbol("capability_name_x")
    source = s.sin(x, evaluate=False)
    raw = _native_expr(source)
    calls = []

    class HostileName(str):
        def __eq__(self, other):
            calls.append("equality")
            raise AssertionError("a class name is metadata, not an executable identity")

        def __ne__(self, other):
            calls.append("inequality")
            raise AssertionError("a class name is metadata, not an executable identity")

    original_name = s.sin.__name__
    try:
        s.sin.__name__ = HostileName("sin")
        assert not _is_closed_function_class(s.sin)
        for value in (source, raw):
            with pytest.raises(NotImplementedError):
                s.compile_numeric(x, value)
    finally:
        s.sin.__name__ = original_name
    assert calls == []
