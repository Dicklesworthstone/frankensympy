"""Repeated parameterized solves retain the complete supervised source graph."""

import copy
import importlib
import math

import pytest
import sympy as s

from sympy.core import Basic, SupervisedCallbackSession, supervised_callback_session


def budget(**changes):
    options = dict(max_steps=1 << 30, max_memory_bytes=1 << 30, max_allocations=1 << 20)
    options.update(changes)
    return s.EvaluationBudget(**options)


def session(**changes):
    options = dict(max_invocations=4096, max_nested_steps=20_000_000)
    options.update(changes)
    return SupervisedCallbackSession(**options)


def values(result):
    assert type(result) is s.Matrix and result.cols == 1
    assert all(type(result[row, 0]) is s.Float for row in range(result.rows))
    return [float(result[row, 0]) for row in range(result.rows)]


def test_repeated_parameter_solves_keep_original_graph_and_unknown_order(monkeypatch):
    x, y, k, b = s.symbols("hybrid_param_x hybrid_param_y hybrid_param_k hybrid_param_b")
    events = []

    class Law(s.Function):
        pass

    class ParameterSlope(s.Function):
        pass

    class UnknownSlope(s.Function):
        pass

    residuals = [s.Add(Law(x, k, evaluate=False), y, -b, evaluate=False), x-y]
    jacobian = [[ParameterSlope(x, evaluate=False), UnknownSlope(k, evaluate=False), -1, 1],
                [0, 1, 0, -1]]
    callbacks = {
        Law: lambda value, parameter: events.append(("law", value, parameter)) or parameter*value,
        ParameterSlope: lambda value: events.append(("parameter", value)) or value,
        UnknownSlope: lambda value: events.append(("unknown", value)) or value,
    }
    layout = [(k, x), (b, y)]
    program = s.compile_hybrid_residual_system(
        residuals, layout, jacobian=jacobian, callbacks=callbacks,
    )
    operations, pattern = program.operation_count, program.jacobian_pattern
    residuals[0], jacobian[0][0], layout[0], callbacks[Law] = 999, 999, (y, x), lambda *args: 999

    def no_recompile(*args, **kwargs):
        raise AssertionError("parameter solves must reuse the captured graph")

    monkeypatch.setattr(importlib.import_module("sympy.utilities.hybrid_system"), "_compile_graph", no_recompile)
    for bindings, expected in (({k: 2, b: 6}, [2, 2]), ({b: 12, k: 3}, [3, 3])):
        active, ledger = session(), budget()
        assert values(program.solve([0, 0], parameters=bindings, session=active, budget=ledger)) == pytest.approx(expected)
        assert active.invocations > 0 and ledger.remaining_steps < 1 << 30
    assert program.variables == (k, x, b, y)
    assert program.variable_count == 4 and program.residual_count == 2
    assert program.operation_count == operations and program.jacobian_pattern == pattern
    assert program.evidence == "approximate_with_user_callbacks"
    assert program.jacobian_evidence == "user_supplied"
    assert any(event[0] == "parameter" for event in events)
    assert {event[2] for event in events if event[0] == "law"} == {2.0, 3.0}
    for name in ("native_system", "solve_certified", "certify_root_box", "to_bytes"):
        assert not hasattr(program, name)


def test_equal_printed_symbols_bind_by_captured_assumption_identity():
    plain = s.Symbol("hybrid_param_same_name")
    positive = s.Symbol("hybrid_param_same_name", positive=True)
    program = s.compile_hybrid_residual_system(
        plain-positive-1, (positive, plain), jacobian=[[-1, 1]], callbacks={},
    )
    assert values(program.solve(0, parameters={positive: 2})) == pytest.approx([3])
    assert values(program.solve(0, parameters={plain: 5})) == pytest.approx([4])
    wrong = s.Symbol("hybrid_param_same_name", negative=True)
    with pytest.raises(ValueError, match="captured typed variable"):
        program.solve(0, parameters={wrong: 2})


def test_dummy_identity_accepts_an_equal_copy_and_rejects_a_same_named_atom():
    x, parameter = s.Dummy("hybrid_param_dummy"), s.Dummy("hybrid_param_dummy")
    program = s.compile_hybrid_residual_system(x-parameter, (x, parameter), jacobian=[[1, -1]], callbacks={})
    equivalent = copy.copy(parameter)
    assert equivalent is not parameter
    assert values(program.solve(0, parameters={equivalent: 7})) == pytest.approx([7])
    with pytest.raises(ValueError, match="captured typed variable"):
        program.solve(0, parameters={s.Dummy("hybrid_param_dummy"): 7})


def test_parameter_mapping_and_mutable_assumptions_are_captured_before_callbacks(monkeypatch):
    x = s.Symbol("hybrid_param_mutation_x")
    parameter = s.Symbol("hybrid_param_mutation_p", positive=True)
    bindings, seen = {parameter: 2}, []

    class Law(s.Function):
        pass

    def law(value, fixed):
        seen.append(fixed)
        bindings.clear()
        bindings[x] = 999
        monkeypatch.setattr(parameter, "_assumptions", {"negative": True})
        return value-fixed

    program = s.compile_hybrid_residual_system(
        Law(x, parameter, evaluate=False), (x, parameter), jacobian=[[1, -1]], callbacks={Law: law},
    )
    assert values(program.solve(0, parameters=bindings)) == pytest.approx([2])
    assert seen and set(seen) == {2.0}
    # A later call still resolves the native identity captured at compilation,
    # rather than reconstructing a new axis from the mutated assumption dict.
    seen.clear()
    assert values(program.solve(0, parameters={parameter: 4})) == pytest.approx([4])
    assert set(seen) == {4.0}


def test_parameter_dict_and_initial_values_are_fully_admitted_before_effects():
    x, parameter, extra = s.symbols("hybrid_param_admit_x hybrid_param_admit_p hybrid_param_admit_extra")
    seen = []

    class Law(s.Function):
        pass

    class Mapping(dict):
        def items(self):
            raise AssertionError("mapping overrides must not run")

    class OtherSymbol(s.Symbol):
        # Admit the mapping itself so the solve boundary, rather than the
        # shell's inherited opaque-symbol hashing refusal, checks this key.
        __hash__ = object.__hash__

    class NumericOverride:
        def __float__(self):
            raise AssertionError("numeric overrides must not run")

    program = s.compile_hybrid_residual_system(
        Law(x, parameter, evaluate=False), (x, parameter), jacobian=[[1, -1]],
        callbacks={Law: lambda value, fixed: seen.append((value, fixed)) or value-fixed},
    )
    invalid = (
        Mapping({parameter: 1}), [(parameter, 1)], {"hybrid_param_admit_p": 1},
        {extra: 1}, {OtherSymbol("hybrid_param_admit_p"): 1},
        {parameter: math.nan}, {parameter: math.inf}, {parameter: -math.inf},
        {parameter: 1j}, {parameter: "2"}, {parameter: NumericOverride()},
        {x: 1, parameter: 1}, {x: 1, parameter: 1, extra: 1},
    )
    for parameters in invalid:
        with pytest.raises((TypeError, ValueError)):
            program.solve(0, parameters=parameters)
    for initial in ([0, 1], [], [math.nan], [object()], (value for value in [0])):
        with pytest.raises((TypeError, ValueError)):
            program.solve(initial, parameters={parameter: 1})
    with pytest.raises(ValueError, match="at least as many equations"):
        program.solve([0, 0], parameters={})
    assert seen == []


def test_parameter_snapshot_refuses_growth_without_running_callbacks(monkeypatch):
    module = importlib.import_module("sympy.utilities.hybrid_system")
    x, parameter = s.symbols("hybrid_param_grow_x hybrid_param_grow_p")
    bindings, seen, changed = {parameter: 2}, [], []

    class Law(s.Function):
        pass

    program = s.compile_hybrid_residual_system(
        Law(x, parameter, evaluate=False), (x, parameter), jacobian=[[1, -1]],
        callbacks={Law: lambda value, fixed: seen.append(value) or value-fixed},
    )
    real_len = len

    def grow(value):
        size = real_len(value)
        if value is bindings and not changed:
            changed.append(True)
            bindings[x] = 99
        return size

    monkeypatch.setattr(module, "len", grow, raising=False)
    with pytest.raises(ValueError, match="changed during"):
        program.solve(0, parameters=bindings)
    assert changed == [True] and seen == []


def test_mutated_mapping_keys_cannot_repeat_the_same_native_axis(monkeypatch):
    x, parameter, other = s.symbols("hybrid_param_dup_x hybrid_param_dup_p hybrid_param_dup_other")
    program = s.compile_hybrid_residual_system(x-parameter, (x, parameter), jacobian=[[1, -1]], callbacks={})
    bindings = {parameter: 2, other: 3}
    # The exact dict can retain two distinct entries after an unsupported
    # caller mutation makes their immutable native payloads identical.
    native = Basic.__dict__["_value"].__get__(parameter, Basic)
    monkeypatch.setattr(other, "_value", native)
    with pytest.raises(ValueError, match="duplicate typed parameter"):
        program.solve(0, parameters=bindings)


def test_held_original_parameter_domain_survives_a_zero_multiplier():
    x, parameter = s.symbols("hybrid_param_domain_x hybrid_param_domain_p")
    seen = []

    class Observe(s.Function):
        pass

    held = s.Mul(0, s.log(parameter), evaluate=False)
    program = s.compile_hybrid_residual_system(
        s.Add(held, Observe(x, evaluate=False), evaluate=False), (x, parameter),
        jacobian=[[1, 0]], callbacks={Observe: lambda value: seen.append(value) or value-1},
    )
    with pytest.raises(ValueError):
        program.solve(0, parameters={parameter: -1})
    assert seen == []


def test_projected_parameter_derivative_keeps_its_domain():
    x, parameter = s.symbols("hybrid_param_derivative_x hybrid_param_derivative_p")
    program = s.compile_hybrid_residual_system(
        x-1, (x, parameter), jacobian=[[1, 1/parameter]], callbacks={},
    )
    with pytest.raises(ValueError):
        program.solve(0, parameters={parameter: 0})
    assert values(program.solve(0, parameters={parameter: 2})) == pytest.approx([1])


def test_existing_exact_zero_residual_exit_precedes_parameter_derivative_effects():
    x, parameter = s.symbols("hybrid_param_zero_x hybrid_param_zero_p")
    seen = []

    class ParameterSlope(s.Function):
        pass

    program = s.compile_hybrid_residual_system(
        x-parameter, (x, parameter), jacobian=[[1, ParameterSlope(parameter, evaluate=False)]],
        callbacks={ParameterSlope: lambda value: seen.append(value) or 1/value},
    )
    assert values(program.solve(0, parameters={parameter: 0})) == [0]
    assert seen == []


def test_projected_parameter_callback_failure_keeps_original_exception_and_session():
    x, parameter = s.symbols("hybrid_param_error_x hybrid_param_error_p")
    seen = []

    class OriginalError(ValueError):
        def __str__(self):
            raise AssertionError("callback exceptions must not be formatted")

    class ParameterSlope(s.Function):
        pass

    original = OriginalError()

    def slope(value):
        seen.append(value)
        raise original

    program = s.compile_hybrid_residual_system(
        x-1, (x, parameter), jacobian=[[1, ParameterSlope(parameter, evaluate=False)]],
        callbacks={ParameterSlope: slope},
    )
    active = session()
    with pytest.raises(OriginalError) as failure:
        program.solve(0, parameters={parameter: 3}, session=active)
    assert failure.value is original and seen == [3.0]
    assert active.unknown_effect_tainted
    with pytest.raises(RuntimeError, match="unknown effect"):
        program.solve(0, parameters={parameter: 3}, session=active)
    assert seen == [3.0]


@pytest.mark.parametrize("before_callback", [True, False])
def test_parameterized_trial_retains_effect_aware_domain_backtracking(before_callback):
    x, parameter = s.symbols("hybrid_param_trial_x hybrid_param_trial_p")
    seen = []

    class Observe(s.Function):
        pass

    parts = (s.log(x)-parameter, Observe(x, evaluate=False))
    if not before_callback:
        parts = parts[::-1]
    program = s.compile_hybrid_residual_system(
        s.Add(*parts, evaluate=False), (parameter, x), jacobian=[[-1, 1/x]],
        callbacks={Observe: lambda value: seen.append(value) or 0},
    )
    active = session()
    if before_callback:
        assert values(program.solve(10, parameters={parameter: 0}, session=active)) == pytest.approx([1])
        assert all(value > 0 for value in seen)
    else:
        with pytest.raises(ValueError):
            program.solve(10, parameters={parameter: 0}, session=active)
        assert seen[:2] == [10.0, 10.0] and len(seen) == 3 and seen[-1] < 0


def test_parameter_solves_share_cumulative_budget_and_callback_session():
    x, parameter = s.symbols("hybrid_param_budget_x hybrid_param_budget_p")

    class Law(s.Function):
        pass

    program = s.compile_hybrid_residual_system(
        Law(x, parameter, evaluate=False), (x, parameter), jacobian=[[1, -1]],
        callbacks={Law: lambda value, fixed: value-fixed},
    )
    measuring, active = budget(), session()
    assert values(program.solve(0, parameters={parameter: 2}, budget=measuring, session=active)) == [2]
    cost, calls = (1 << 30) - measuring.remaining_steps, active.invocations
    assert cost > 0 and calls > 0
    shared, cumulative = budget(max_steps=2*cost), session(max_invocations=2*calls)
    for _ in range(2):
        assert values(program.solve(0, parameters={parameter: 2}, budget=shared, session=cumulative)) == [2]
    assert shared.remaining_steps == 0 and cumulative.invocations == 2*calls
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve(0, parameters={parameter: 2}, budget=shared, session=cumulative)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve(0, parameters={parameter: 2}, budget=budget(max_steps=cost-1))


def test_nested_parameter_solve_inherits_and_cannot_replace_owning_request():
    x, parameter = s.symbols("hybrid_param_nested_x hybrid_param_nested_p")
    inner = s.compile_hybrid_residual_system(
        x-parameter, (x, parameter), jacobian=[[1, -1]], callbacks={},
    )

    class Law(s.Function):
        pass

    outer = s.compile_numeric_hybrid(
        x, Law(x, evaluate=False),
        callbacks={Law: lambda value: float(inner.solve(0, parameters={parameter: value})[0, 0])},
    )
    ledger, active = budget(), session()
    assert outer(3, budget=ledger, session=active) == pytest.approx(3)
    assert active.invocations == 1 and ledger.remaining_steps < 1 << 30
    with supervised_callback_session(active):
        with pytest.raises(ValueError, match="conflicts"):
            inner.solve(0, parameters={parameter: 1}, session=session())


@pytest.mark.parametrize("which", ["budget", "session"])
def test_pre_cancelled_parameter_solve_runs_no_callbacks(which):
    x, parameter = s.symbols("hybrid_param_precancel_x hybrid_param_precancel_p")
    seen = []

    class Law(s.Function):
        pass

    program = s.compile_hybrid_residual_system(
        Law(x, parameter, evaluate=False), (x, parameter), jacobian=[[1, -1]],
        callbacks={Law: lambda value, fixed: seen.append(value) or value-fixed},
    )
    ledger, active = budget(), session()
    (ledger if which == "budget" else active).cancel()
    with pytest.raises(s.EvaluationCancelled if which == "budget" else RuntimeError):
        program.solve(0, parameters={parameter: 2}, budget=ledger, session=active)
    assert seen == []


def test_cancellation_in_parameter_derivative_prevents_result_publication():
    x, parameter = s.symbols("hybrid_param_cancel_x hybrid_param_cancel_p")
    ledger, active, seen = budget(), session(), []

    class ParameterSlope(s.Function):
        pass

    def cancel(value):
        seen.append(value)
        ledger.cancel()
        return -1

    program = s.compile_hybrid_residual_system(
        x-parameter, (x, parameter), jacobian=[[1, ParameterSlope(parameter, evaluate=False)]],
        callbacks={ParameterSlope: cancel},
    )
    with pytest.raises(s.EvaluationCancelled):
        program.solve(0, parameters={parameter: 2}, budget=ledger, session=active)
    assert seen == [2.0] and active.unknown_effect_tainted


def test_piecewise_custom_law_reuses_both_parameter_regimes_without_inactive_effects():
    x, parameter = s.symbols("hybrid_param_piecewise_x hybrid_param_piecewise_p")
    seen = []

    class PositiveLaw(s.Function):
        pass

    class NegativeLaw(s.Function):
        pass

    def positive(value, fixed):
        assert fixed >= 0
        seen.append(("positive", fixed))
        return fixed*value

    def negative(value, fixed):
        assert fixed < 0
        seen.append(("negative", fixed))
        return -fixed*value

    condition = s.Ge(parameter, 0)
    law = s.Piecewise((PositiveLaw(x, parameter, evaluate=False), condition),
                      (NegativeLaw(x, parameter, evaluate=False), True), evaluate=False)
    parameter_derivative = s.Piecewise((x, condition), (-x, True), evaluate=False)
    derivative = s.Piecewise((parameter, condition), (-parameter, True), evaluate=False)
    program = s.compile_hybrid_residual_system(
        s.Add(law, -4, evaluate=False), [(parameter, x)],
        jacobian=[[parameter_derivative, derivative]],
        callbacks={PositiveLaw: positive, NegativeLaw: negative},
    )
    assert values(program.solve(0, parameters={parameter: 2})) == pytest.approx([2])
    assert seen and set(seen) == {("positive", 2.0)}
    seen.clear()
    assert values(program.solve(0, parameters={parameter: -4})) == pytest.approx([1])
    assert seen and set(seen) == {("negative", -4.0)}
    residuals, entries = program.evaluate_sparse((-4, 1))
    assert residuals == pytest.approx([0]) and entries == pytest.approx([-1, 4])


@pytest.mark.parametrize("operator, expected, inactive", [(s.Add, 5, 2), (s.Mul, 6, 0), (s.Pow, 9, 0)])
def test_held_arithmetic_keeps_nested_piecewise_values_and_boolean_guards(operator, expected, inactive):
    x = s.Symbol("hybrid_held_piecewise_x")
    seen = []

    class Law(s.Function):
        pass

    guard = s.And(s.Gt(x, 0, evaluate=False), s.Ne(x, 2, evaluate=False))
    inner = s.Piecewise((Law(x, evaluate=False), guard), (0, True), evaluate=False)
    source = s.Piecewise((inner, s.Lt(x, 4, evaluate=False)), (5, True), evaluate=False)
    held = operator(source, 2, evaluate=False)
    assert held.args[0] is source
    program = s.compile_numeric_hybrid(
        x, held, callbacks={Law: lambda value: seen.append(value) or value},
    )
    assert program(3) == expected
    assert program(-3) == inactive and program(2) == inactive
    assert seen == [3.0]


def test_opaque_piecewise_has_no_new_canonical_or_global_boolean_arithmetic_lane():
    x = s.Symbol("hybrid_held_piecewise_scope_x")

    class Law(s.Function):
        pass

    node = Law(x, evaluate=False)
    source = s.Piecewise((node, s.Gt(x, 0, evaluate=False)), (0, True), evaluate=False)
    for operator in (s.Add, s.Mul, s.Pow):
        with pytest.raises(NotImplementedError):
            operator(source, 2)
    with pytest.raises(NotImplementedError):
        s.Add(node, s.true, evaluate=False)
    with pytest.raises(NotImplementedError):
        s.Add(node, s.ExprCondPair(x, True), evaluate=False)


def test_held_piecewise_checks_condition_owners_without_running_modified_properties(monkeypatch):
    x = s.Symbol("hybrid_held_piecewise_hook_x")
    seen = []

    class Law(s.Function):
        pass

    guard = s.Gt(x, 0, evaluate=False)
    source = s.Piecewise((Law(x, evaluate=False), guard), (0, True), evaluate=False)

    def poisoned(value):
        seen.append(value)
        raise AssertionError("held admission must not invoke a condition property")

    monkeypatch.setattr(type(guard), "lhs", property(poisoned))
    with pytest.raises(NotImplementedError):
        s.Add(source, 1, evaluate=False)
    assert seen == []


@pytest.mark.parametrize("role", ["condition", "value"])
def test_held_piecewise_checks_authoritative_branch_slots_for_cycles(role):
    from sympy.logic.boolalg import BooleanFunction

    x = s.Symbol("hybrid_held_piecewise_cycle_x")

    class Law(s.Function):
        pass

    node = Law(x, evaluate=False)
    source = s.Piecewise((node, s.Gt(x, 0, evaluate=False)), (0, True), evaluate=False)
    if role == "condition":
        condition = object.__new__(s.And)
        BooleanFunction.__dict__["_args"].__set__(condition, (condition,))
        branches = ((node, condition),)
    else:
        branches = ((source, True),)
    # Numerical capture owns this pair slot; stale Function._args must not
    # conceal a cycle in the actual branch or condition graph.
    s.Piecewise.__dict__["_piecewise_branches"].__set__(source, branches)
    with pytest.raises(ValueError, match="cyclic"):
        s.Add(source, 1, evaluate=False)

