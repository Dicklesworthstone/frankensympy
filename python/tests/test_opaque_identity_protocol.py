"""Shared Python identity changes cannot silently enter commutative algebra."""

import pytest
import sympy as s

from sympy.core import Application, Basic, Expr, Function


def _applications():
    class Unknown(Function):
        pass

    x = s.Symbol("inherited_identity_x")
    return x, Unknown, Unknown(x, evaluate=False), Unknown(x, evaluate=False)


def test_mutating_inherited_equality_cannot_merge_distinct_applications(monkeypatch):
    x, cls, first, second = _applications()
    assert first is not second and first == second
    with monkeypatch.context() as patch:
        patch.setattr(Basic, "__eq__", lambda self, other: self is other)
        assert first != second
        for operation in (
            lambda: first + second,
            lambda: first - second,
            lambda: first * second,
            lambda: first / second,
            lambda: first**second,
        ):
            with pytest.raises(NotImplementedError, match="modified default Function identity"):
                operation()
        # Holding source and compiling explicit callbacks needs no assertion
        # that distinct applications are algebraically equal or commutative.
        held = s.Add(first, second, evaluate=False)
        assert held.args[0] is first and held.args[1] is second
        program = s.compile_numeric_hybrid(x, held, callbacks={cls: lambda value: value + 1})
        assert program(2) == 6
    assert first == second
    assert first + second == 2 * first


@pytest.mark.parametrize("owner,attribute", [
    (Basic, "__eq__"), (Basic, "__hash__"), (Basic, "_hashable_content"),
    (Expr, "__eq__"), (Expr, "__hash__"), (Expr, "is_commutative"),
    (Application, "__eq__"), (Application, "is_commutative"),
    (Function, "__eq__"), (Function, "__hash__"),
    (Function, "_hashable_content"), (Function, "_eval_is_commutative"),
])
def test_changed_identity_descriptors_refuse_before_any_hook(monkeypatch, owner, attribute):
    _, _, first, second = _applications()
    source = first + second + 1
    calls = []

    def forbidden(*args):
        calls.append(attribute)
        raise AssertionError("normalization cannot execute an unadmitted identity hook")

    replacement = property(forbidden) if attribute == "is_commutative" else forbidden
    with monkeypatch.context() as patch:
        patch.setattr(owner, attribute, replacement, raising=False)
        for operation in (
            lambda: s.Add(first, second),
            lambda: s.Mul(first, second),
            lambda: s.Pow(first, second),
            lambda: s.expand(source),
        ):
            with pytest.raises(NotImplementedError, match="modified default Function identity"):
                operation()
    assert calls == []


@pytest.mark.parametrize("attribute", ["__eq__", "__hash__", "is_commutative"])
def test_in_place_default_identity_code_mutation_is_detected_and_restorable(monkeypatch, attribute):
    _, _, first, second = _applications()
    if attribute == "is_commutative":
        implementation = Expr.__dict__[attribute].fget
    else:
        implementation = Basic.__dict__[attribute]

    def changed(*args):
        raise AssertionError("in-place identity changes must be detected without execution")

    with monkeypatch.context() as patch:
        patch.setattr(implementation, "__code__", changed.__code__)
        with pytest.raises(NotImplementedError, match="modified default Function identity"):
            s.Add(first, second)
    assert first + second == 2 * first


def test_unrelated_shared_callback_override_does_not_revoke_function_identity(monkeypatch):
    _, _, first, second = _applications()
    calls = []

    def derivative(*args):
        calls.append("derivative")
        raise AssertionError("arithmetic must not execute differentiation hooks")

    monkeypatch.setattr(Function, "_eval_derivative", derivative, raising=False)
    assert first + second == 2 * first
    assert calls == []
