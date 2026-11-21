"""Lazy mixed Piecewise control flow preserves supervised effect boundaries."""

import importlib
import math

import pytest
import sympy as s

from sympy.core import _native, supervised_callback_session
from sympy.logic.boolalg import BooleanFunction


def _budget(**changes):
    limits = dict(max_steps=1 << 28, max_memory_bytes=1 << 28, max_allocations=1 << 20)
    limits.update(changes)
    return s.EvaluationBudget(**limits)


def _held_boolean(cls, *args):
    # Preserve an exact built-in Boolean node and its stored operand order.
    # In particular the public Not constructor may rewrite a relational.
    result = object.__new__(cls)
    BooleanFunction.__dict__["_args"].__set__(result, tuple(args))
    return result


def test_piecewise_captures_branches_and_callbacks_without_running_any_effect():
    x = s.Symbol("hybrid_branch_x")
    calls = []

    class Positive(s.Function):
        pass

    class Negative(s.Function):
        pass

    def positive(value):
        calls.append(("positive", value))
        return value + 10

    def negative(value):
        calls.append(("negative", value))
        return value - 10

    source = s.Piecewise(
        (Positive(x, evaluate=False), s.Ge(x, 0, evaluate=False)),
        (Negative(x, evaluate=False), True),
        evaluate=False,
    )
    callbacks = {Positive: positive, Negative: negative}
    program = s.compile_numeric_hybrid(x, source, callbacks=callbacks)
    callbacks[Positive] = lambda value: 999
    s.Piecewise.__dict__["_piecewise_branches"].__set__(source, ((s.Integer(999), True),))
    assert calls == []
    assert program(2) == 12
    assert program(-3) == -13
    assert calls == [("positive", 2.0), ("negative", -3.0)]
    assert program.callback_slots == ((0, Positive, positive), (1, Negative, negative))
    assert program.native_fragment_count > 0
    assert program.evidence == "approximate_with_user_callbacks"
    assert not hasattr(program, "to_portable_bytes")


def test_nested_piecewise_callback_arguments_preserve_occurrences_and_skip_domains():
    x = s.Symbol("hybrid_nested_branch_x")
    calls = []

    class Count(s.Function):
        pass

    class Outer(s.Function):
        pass

    def count(value):
        calls.append(("count", value))
        return value + 1

    def outer(value):
        calls.append(("outer", value))
        return 10 * value

    node = Count(x, evaluate=False)
    inner = s.Piecewise(
        (s.Add(node, node, evaluate=False), s.Lt(x, 5, evaluate=False)),
        (s.log(-1, evaluate=False), True),
        evaluate=False,
    )
    source = s.Piecewise(
        (Outer(inner, evaluate=False), s.Gt(x, 0, evaluate=False)),
        (7, True),
        evaluate=False,
    )
    program = s.compile_numeric_hybrid(x, source, callbacks={Count: count, Outer: outer})
    assert calls == []
    with supervised_callback_session() as session:
        assert program(2, session=session) == 60
        assert program(-2, session=session) == 7
        assert session.invocations == 3
        successes = [entry for entry in session.trace() if entry[3] == "ok"]
        assert len(successes) == 3
        assert len({entry[2] for entry in successes}) == 3
        assert session.cached_effect_count() == 0
    assert calls == [("count", 2.0), ("count", 2.0), ("outer", 6.0)]
    assert tuple(slot[1] for slot in program.callback_slots) == (Count, Count, Outer)


@pytest.mark.parametrize("logical, first_value, expected", [(s.And, 0, 20), (s.Or, 1, 10)])
def test_boolean_short_circuit_does_not_invoke_later_condition_callbacks(logical, first_value, expected):
    x = s.Symbol("hybrid_short_circuit_x")
    calls = []

    class Left(s.Function):
        pass

    class Right(s.Function):
        pass

    condition = logical(
        s.Gt(Left(x, evaluate=False), 0, evaluate=False),
        s.Gt(Right(x, evaluate=False), 0, evaluate=False),
        evaluate=False,
    )
    # And/Or may canonicalize at construction. Evaluation must preserve the
    # actual captured order, rather than invent a new ordering of effects.
    first_class = type(condition.args[0].lhs)
    second_class = type(condition.args[1].lhs)

    def first(value):
        calls.append(value)
        return first_value

    def forbidden(value):
        raise AssertionError("short-circuited callback ran")

    source = s.Piecewise((10, condition), (20, True), evaluate=False)
    program = s.compile_numeric_hybrid(
        x, source, callbacks={first_class: first, second_class: forbidden},
    )
    assert calls == []
    with supervised_callback_session() as session:
        assert program(3, session=session) == expected
        assert session.invocations == 1
        assert not session.unknown_effect_tainted
    assert calls == [3.0]


def test_nested_and_or_not_preserve_repeated_condition_occurrences():
    x = s.Symbol("hybrid_nested_condition_x")
    calls = []

    class Check(s.Function):
        pass

    node = Check(x, evaluate=False)
    condition = _held_boolean(
        s.And,
        s.Gt(x, -10, evaluate=False),
        _held_boolean(
            s.Or,
            _held_boolean(s.Not, s.Gt(node, 0, evaluate=False)),
            s.Gt(node, 0, evaluate=False),
        ),
    )
    source = s.Piecewise((1, condition), (2, True), evaluate=False)
    program = s.compile_numeric_hybrid(
        x, source, callbacks={Check: lambda value: calls.append(value) or value},
    )
    assert program(-1) == 1
    assert calls == [-1.0]
    calls.clear()
    assert program(1) == 1
    assert calls == [1.0, 1.0]
    calls.clear()
    assert program(-20) == 2
    assert calls == []


@pytest.mark.parametrize("relation, expected", [
    (s.Eq, 1), (s.Ne, 2), (s.Lt, 2), (s.Le, 1), (s.Gt, 2), (s.Ge, 1),
])
def test_every_real_comparison_accepts_a_supervised_arithmetic_operand(relation, expected):
    x = s.Symbol("hybrid_comparison_x")
    calls = []

    class Probe(s.Function):
        pass

    source = s.Piecewise(
        (1, relation(Probe(x, evaluate=False), 0, evaluate=False)),
        (2, True),
        evaluate=False,
    )
    program = s.compile_numeric_hybrid(
        x, source, callbacks={Probe: lambda value: calls.append(value) or value},
    )
    assert calls == []
    assert program(0) == expected
    assert calls == [0.0]


def test_only_selected_held_domains_are_evaluated_before_later_effects():
    x = s.Symbol("hybrid_selected_domain_x")
    calls = []

    class Later(s.Function):
        pass

    selected = s.Piecewise(
        (s.Mul(0, s.log(x, evaluate=False), evaluate=False), s.Lt(x, 0, evaluate=False)),
        (Later(x, evaluate=False), True),
        evaluate=False,
    )
    program = s.compile_numeric_hybrid(
        x, s.Add(selected, Later(x, evaluate=False), evaluate=False),
        callbacks={Later: lambda value: calls.append(value) or value},
    )
    assert program(2) == 4
    assert calls == [2.0, 2.0]
    calls.clear()
    with pytest.raises(ValueError):
        program(-1)
    assert calls == []


def test_unselected_callback_exception_never_taints_the_session():
    x = s.Symbol("hybrid_inactive_error_x")

    class Fail(s.Function):
        pass

    def forbidden(value):
        raise LookupError("unselected callback")

    source = s.Piecewise(
        (3, s.Ge(x, 0, evaluate=False)),
        (Fail(x, evaluate=False), True),
        evaluate=False,
    )
    program = s.compile_numeric_hybrid(x, source, callbacks={Fail: forbidden})
    with supervised_callback_session() as session:
        assert program(2, session=session) == 3
        assert session.invocations == 0
        assert not session.unknown_effect_tainted


def test_missing_branch_refuses_before_value_effect_and_leaves_clean_session_usable():
    x = s.Symbol("hybrid_missing_branch_x")
    calls = []

    class Value(s.Function):
        pass

    source = s.Piecewise(
        (Value(x, evaluate=False), s.Gt(x, 0, evaluate=False)), evaluate=False,
    )
    program = s.compile_numeric_hybrid(
        x, source, callbacks={Value: lambda value: calls.append(value) or value},
    )
    with supervised_callback_session() as session:
        with pytest.raises(ValueError, match="no Piecewise condition"):
            program(-1, session=session)
        assert session.invocations == 0
        assert not session.unknown_effect_tainted
        assert program(2, session=session) == 2
    assert calls == [2.0]


@pytest.mark.parametrize("cancel_in_guard", [True, False])
def test_cancellation_after_guard_or_value_effect_prevents_later_effects_and_publication(cancel_in_guard):
    x = s.Symbol("hybrid_branch_cancel_x")
    budget, calls = _budget(), []

    class Guard(s.Function):
        pass

    class Value(s.Function):
        pass

    class Later(s.Function):
        pass

    def guard(value):
        calls.append("guard")
        if cancel_in_guard:
            budget.cancel()
        return 1

    def value(argument):
        calls.append("value")
        budget.cancel()
        return argument

    condition = s.Gt(Guard(x, evaluate=False), 0, evaluate=False)
    source = s.Add(
        s.Piecewise((Value(x, evaluate=False), condition), (0, True), evaluate=False),
        Later(x, evaluate=False),
        evaluate=False,
    )
    program = s.compile_numeric_hybrid(
        x, source, callbacks={
            Guard: guard, Value: value, Later: lambda value: calls.append("later") or value,
        },
    )
    sentinel = object()
    result = sentinel
    with supervised_callback_session() as session:
        with pytest.raises(s.EvaluationCancelled):
            result = program(2, budget=budget, session=session)
        assert result is sentinel
        assert session.unknown_effect_tainted
    assert calls == (["guard"] if cancel_in_guard else ["guard", "value"])


def test_exhausted_budget_prevents_every_condition_callback():
    x = s.Symbol("hybrid_branch_budget_x")
    calls = []

    class Guard(s.Function):
        pass

    source = s.Piecewise(
        (1, s.Gt(Guard(x, evaluate=False), 0, evaluate=False)), (2, True), evaluate=False,
    )
    program = s.compile_numeric_hybrid(
        x, source, callbacks={Guard: lambda value: calls.append(value) or value},
    )
    with pytest.raises(s.EvaluationBudgetExceeded):
        program(2, budget=_budget(max_steps=0))
    assert calls == []


def test_batch_selects_each_row_and_keeps_completed_effects_after_later_failure():
    x = s.Symbol("hybrid_branch_batch_x")
    calls = []

    class Law(s.Function):
        pass

    def law(value):
        calls.append(value)
        if value == 3:
            raise LookupError("selected row failed")
        return 10 * value

    source = s.Piecewise(
        (Law(x, evaluate=False), s.Gt(x, 0, evaluate=False)), (0, True), evaluate=False,
    )
    program = s.compile_numeric_hybrid(x, source, callbacks={Law: law})
    sentinel = object()
    result = sentinel
    with supervised_callback_session() as session:
        with pytest.raises(LookupError, match="selected row failed"):
            result = program.evaluate_batch([[-1], [2], [3]], session=session)
        assert result is sentinel
        assert session.invocations == 2
        assert session.unknown_effect_tainted
    assert calls == [2.0, 3.0]


@pytest.mark.parametrize("tag, expected", [("And", 5), ("Or", 7)])
def test_closed_native_piecewise_accepts_empty_boolean_identities_and_skips_unused_domain(tag, expected):
    x = s.Symbol("hybrid_native_condition_x")
    true = _native.Expr("True")
    invalid = _native.py_function("log", _native.py_integer(-1))
    if tag == "And":
        branches = (
            _native.py_function("Tuple", _native.py_integer(5), _native.py_function(tag)),
            _native.py_function("Tuple", invalid, true),
        )
    else:
        branches = (
            _native.py_function("Tuple", invalid, _native.py_function(tag)),
            _native.py_function("Tuple", _native.py_integer(7), true),
        )
    program = s.compile_numeric_hybrid(
        x, _native.py_function("Piecewise", *branches), callbacks={},
    )
    assert program(2) == expected
    assert program.callback_slots == ()


@pytest.mark.parametrize("condition", [
    s.Symbol("True"), s.Symbol("True", positive=True), s.Dummy("True"),
])
def test_same_printed_boolean_symbols_do_not_gain_condition_authority(condition):
    x = s.Symbol("hybrid_typed_boolean_x")
    source = s.Piecewise((1, condition), (2, True), evaluate=False)
    with pytest.raises(NotImplementedError):
        s.compile_numeric_hybrid(x, source, callbacks={})


def test_modified_relation_hooks_refuse_without_being_executed(monkeypatch):
    x = s.Symbol("hybrid_condition_hook_x")
    condition = s.Gt(x, 0, evaluate=False)
    source = s.Piecewise((1, condition), (2, True), evaluate=False)
    calls = []

    def poisoned(value):
        calls.append("lhs")
        raise AssertionError("condition properties must not run during capture")

    monkeypatch.setattr(type(condition), "lhs", property(poisoned))
    with pytest.raises(NotImplementedError, match="modified"):
        s.compile_numeric_hybrid(x, source, callbacks={})
    assert calls == []


def test_an_unbound_callback_in_an_unreachable_branch_still_refuses_admission():
    x = s.Symbol("hybrid_unreachable_binding_x")
    calls = []

    class Unknown(s.Function):
        pass

    source = s.Piecewise(
        (Unknown(x, evaluate=False), False), (2, True), evaluate=False,
    )
    with pytest.raises(NotImplementedError, match="explicit exact-class"):
        s.compile_numeric_hybrid(x, source, callbacks={})
    assert calls == []


def test_piecewise_control_actions_are_bounded_before_any_effect(monkeypatch):
    hybrid = importlib.import_module("sympy.utilities.hybrid")
    x = s.Symbol("hybrid_branch_limit_x")
    calls = []

    class Value(s.Function):
        pass

    source = s.Piecewise(
        (Value(x, evaluate=False), s.Gt(x, 0, evaluate=False)), (0, True), evaluate=False,
    )
    monkeypatch.setattr(hybrid, "_MAX_GRAPH_ACTIONS", 3)
    with pytest.raises(ValueError, match="action limit"):
        s.compile_numeric_hybrid(
            x, source, callbacks={Value: lambda value: calls.append(value) or value},
        )
    assert calls == []


def test_solver_can_backtrack_a_missing_branch_before_any_trial_callback():
    x = s.Symbol("hybrid_no_branch_trial_x")
    calls = []

    class LogLaw(s.Function):
        pass

    def law(value):
        calls.append(value)
        return math.log(value)

    residual = s.Piecewise(
        (LogLaw(x, evaluate=False), s.Gt(x, 0, evaluate=False)), evaluate=False,
    )
    program = s.compile_hybrid_residual_system(
        [residual], [x], jacobian=[[1 / x]], callbacks={LogLaw: law},
    )
    # The full Newton trial from four is negative. A missing closed guard
    # rejects it before LogLaw executes, so bounded backtracking is safe.
    root = program.solve([4.0])
    assert float(root[0]) == pytest.approx(1.0, abs=1e-10)
    assert calls and all(value > 0 for value in calls)
    assert any(0 < value < 4 for value in calls)


def test_solver_never_retries_a_missing_branch_after_a_guard_callback():
    x = s.Symbol("hybrid_effectful_no_branch_trial_x")
    calls = []

    class Guard(s.Function):
        pass

    def guard(value):
        calls.append(value)
        return value

    residual = s.Piecewise(
        (s.log(x, evaluate=False), s.Gt(Guard(x, evaluate=False), 0, evaluate=False)),
        evaluate=False,
    )
    program = s.compile_hybrid_residual_system(
        [residual], [x], jacobian=[[1 / x]], callbacks={Guard: guard},
    )
    with supervised_callback_session() as session:
        with pytest.raises(ValueError, match="no Piecewise condition"):
            program.solve([4.0], session=session)
        assert not session.unknown_effect_tainted
    # The rejected negative point has already executed its condition effect.
    # It is terminal even though the callback itself completed successfully.
    assert calls[-1] < 0
    assert sum(value < 0 for value in calls) == 1
