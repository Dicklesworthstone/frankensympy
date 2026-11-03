"""Real native fragments, explicit Jacobians and supervised Newton requests."""

import importlib
import math
import pickle

import pytest
import sympy as s

from sympy.core import SupervisedCallbackSession, supervised_callback_session
from sympy.utilities.hybrid_system import compile_hybrid_residual_system


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


def test_one_captured_program_exposes_every_evaluation_entry_point_and_explicit_evidence():
    x, y = s.symbols("hybrid_system_snapshot_x hybrid_system_snapshot_y")
    calls = []

    class Law(s.Function):
        def _eval_derivative(self, variable):
            raise AssertionError("explicit Jacobians must not run derivative hooks")

    class Slope(s.Function):
        pass

    def law(value):
        calls.append(("law", value))
        return value * value

    def slope(value):
        calls.append(("slope", value))
        return 2 * value

    residual = s.Matrix([
        s.Add(Law(x, evaluate=False), y, -5, evaluate=False),
        x-y-1,
    ])
    jacobian = s.Matrix([[Slope(x, evaluate=False), 1], [1, -1]])
    args, callbacks = [(x, y)], {Law: law, Slope: slope}
    program = compile_hybrid_residual_system(residual, args, jacobian=jacobian, callbacks=callbacks)
    residual[0, 0], jacobian[0, 0] = 999, 999
    args[0], callbacks[Law], callbacks[Slope] = (y, x), lambda value: 999, lambda value: 999
    Law._imp_ = lambda value: 999
    assert calls == []
    assert program.evaluate((2, 1)) == ([0, 0], [[4, 1], [1, -1]])
    assert calls == [("law", 2.0), ("slope", 2.0)]
    assert program.residuals((3, 2)) == [6, 0]
    assert calls[-1] == ("law", 3.0)
    assert program.jacobian((3, 2)) == [[6, 1], [1, -1]]
    assert calls[-2:] == [("law", 3.0), ("slope", 3.0)]
    assert program.evaluate_batch([[(2, 1)], [(3, 2)]], budget=budget()) == [
        ([0, 0], [[4, 1], [1, -1]]), ([6, 0], [[6, 1], [1, -1]]),
    ]
    assert program.evaluate_batch([]) == []
    assert program((2, 1)) == ([0, 0], [[4, 1], [1, -1]])
    assert program.variables[0] is x and program.variables[1] is y
    assert program.variable_count == program.residual_count == 2
    assert program.callback_slots == ((0, Law, law), (1, Slope, slope))
    assert program.operation_count > 0
    assert program.evidence == "approximate_with_user_callbacks"
    assert program.jacobian_evidence == "user_supplied"
    for name in ("native_system", "proofs_verified", "enclose", "certify_root_box", "solve_certified", "to_bytes"):
        assert not hasattr(program, name)
    with pytest.raises(AttributeError):
        program._rows = 99
    with pytest.raises(AttributeError):
        del program._program
    with pytest.raises(AttributeError):
        program.__init__(None, None, 1, 1)
    with pytest.raises(AttributeError):
        del program._program._actions
    with pytest.raises(TypeError):
        pickle.dumps(program)


def test_native_residual_prefix_is_not_recompiled_or_recaptured(monkeypatch):
    module = importlib.import_module("sympy.utilities.hybrid_system")
    x = s.Symbol("hybrid_system_capture_x")
    effects = []

    class Law(s.Function):
        pass

    residuals, jacobian = [Law(x, evaluate=False)], [[2]]
    callbacks = {Law: lambda value: effects.append(value) or 2 * value}
    original, visited = module._snapshot_source, []

    def mutate_after_container_snapshot(*args, **kwargs):
        if not visited:
            visited.append(True)
            residuals[0], jacobian[0][0] = 999, 999
            callbacks[Law] = lambda value: 999
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_snapshot_source", mutate_after_container_snapshot)
    program = compile_hybrid_residual_system(residuals, x, jacobian=jacobian, callbacks=callbacks)
    assert visited == [True] and effects == []
    assert program.residuals(3) == [6]
    assert program.evaluate(4) == ([8], [[2]])
    assert effects == [3.0, 4.0]


def test_custom_symbolic_jacobian_can_feed_a_complete_approximate_solve():
    x, y = s.symbols("hybrid_system_cubic_x hybrid_system_cubic_y")
    derivative_calls, numerical_calls = [], []

    class Law(s.Function):
        def fdiff(self, argindex=1):
            derivative_calls.append(argindex)
            return 3 * self.args[0]**2

    residuals = s.Matrix([
        s.Add(Law(x, evaluate=False), y, -9, evaluate=False),
        x-y-1,
    ])
    jacobian = residuals.jacobian([x, y])
    captured_hooks = tuple(derivative_calls)
    program = compile_hybrid_residual_system(
        residuals, [(x, y)], jacobian=jacobian,
        callbacks={Law: lambda value: numerical_calls.append(value) or value**3},
    )
    assert tuple(derivative_calls) == captured_hooks
    active, ledger = session(), budget()
    result = values(program.solve([1.5, 0.5], session=active, budget=ledger))
    assert result == pytest.approx([2, 1], abs=1e-12)
    assert abs(result[0]**3 + result[1] - 9) <= 1e-12
    assert abs(result[0] - result[1] - 1) <= 1e-12
    assert tuple(derivative_calls) == captured_hooks
    assert len(numerical_calls) == active.invocations > 2
    assert active.cached_effect_count() == 0
    assert ledger.remaining_steps < 1 << 30


def test_explicit_callback_jacobian_and_overdetermined_rows_use_the_real_qr_kernel():
    x, y = s.symbols("hybrid_system_over_x hybrid_system_over_y")
    effects = []

    class Law(s.Function):
        pass

    class Slope(s.Function):
        pass

    program = compile_hybrid_residual_system(
        [s.Add(Law(x, evaluate=False), y, -3, evaluate=False), 2*x-y, 3*x-3],
        (x, y), jacobian=[[Slope(x, evaluate=False), 1], [2, -1], [3, 0]],
        callbacks={
            Law: lambda value: effects.append(("residual", value)) or value,
            Slope: lambda value: effects.append(("jacobian", value)) or 1,
        },
    )
    result = values(program.solve([0, 0], session=session()))
    assert result == pytest.approx([1, 2], abs=1e-12)
    assert ("jacobian", 0.0) in effects
    assert program.residuals(*result) == pytest.approx([0, 0, 0], abs=1e-12)


def test_repeated_occurrences_across_residuals_and_jacobian_are_fresh_effects():
    x = s.Symbol("hybrid_system_occurrence_x")
    seen = []

    class Law(s.Function):
        pass

    def value(argument):
        seen.append(argument)
        return len(seen)

    node = Law(x, evaluate=False)
    program = compile_hybrid_residual_system([node, node], x, jacobian=[[node], [node]], callbacks={Law: value})
    active = session()
    assert program.evaluate(0, session=active) == ([1, 2], [[3], [4]])
    assert program.evaluate(0, session=active) == ([5, 6], [[7], [8]])
    assert active.invocations == 8 and active.cached_effect_count() == 0
    assert len({entry[2] for entry in active.trace() if entry[3] == "ok"}) == 8


def test_same_names_assumptions_and_dummy_objects_preserve_coordinate_order():
    ordinary = s.Symbol("hybrid_system_same")
    assumed = s.Symbol("hybrid_system_same", positive=True)
    first, second = s.Dummy("same"), s.Dummy("same")
    variables = (second, assumed, ordinary, first)
    program = compile_hybrid_residual_system(
        [variable - (index+1) for index, variable in enumerate(variables)],
        variables, jacobian=[[int(row == column) for column in range(4)] for row in range(4)],
        callbacks={},
    )
    assert all(actual is expected for actual, expected in zip(program.variables, variables))
    assert values(program.solve([0]*4, session=session())) == pytest.approx([1, 2, 3, 4])


def test_zero_residual_terminates_without_a_singular_or_effectful_derivative():
    x = s.Symbol("hybrid_system_zero_x")
    calls = []

    class Slope(s.Function):
        pass

    program = compile_hybrid_residual_system(
        s.sqrt(x), x, jacobian=[[Slope(x, evaluate=False)]],
        callbacks={Slope: lambda value: calls.append(value) or 1 / math.sqrt(value)},
    )
    assert program.residuals(0) == [0]
    assert values(program.solve(0)) == [0]
    assert calls == []
    with pytest.raises(ZeroDivisionError):
        program.jacobian(0)
    assert calls == [0.0]


def test_jacobian_results_preserve_every_original_residual_domain_guard():
    x = s.Symbol("hybrid_system_guard_x")
    source = s.Add(x, s.Mul(0, s.log(x), evaluate=False), evaluate=False)
    program = compile_hybrid_residual_system(source, x, jacobian=[[1]], callbacks={})
    with pytest.raises(ValueError):
        program.jacobian(-1)
    assert program.jacobian(1) == [[1]]


def test_domain_refusal_before_all_callback_occurrences_can_backtrack():
    x = s.Symbol("hybrid_system_safe_trial_x")
    seen = []

    class Observe(s.Function):
        pass

    program = compile_hybrid_residual_system(
        s.Add(s.log(x), Observe(x, evaluate=False), evaluate=False), x,
        jacobian=[[1/x]], callbacks={Observe: lambda value: seen.append(value) or 0},
    )
    active = session()
    assert values(program.solve(10, session=active)) == pytest.approx([1], abs=1e-12)
    assert len(seen) == active.invocations > 2
    assert all(value > 0 for value in seen)
    assert not active.unknown_effect_tainted


def test_domain_refusal_after_a_completed_callback_is_never_retried():
    x = s.Symbol("hybrid_system_effect_trial_x")
    seen = []

    class Observe(s.Function):
        pass

    program = compile_hybrid_residual_system(
        s.Add(Observe(x, evaluate=False), s.log(x), evaluate=False), x,
        jacobian=[[1/x]], callbacks={Observe: lambda value: seen.append(value) or 0},
    )
    active = session()
    with pytest.raises(ValueError):
        program.solve(10, session=active)
    assert seen[:2] == [10.0, 10.0]
    assert len(seen) == active.invocations == 3
    assert seen[-1] == pytest.approx(10 - 10*math.log(10))
    assert seen[-1] < 0
    assert not active.unknown_effect_tainted  # The completed effect is known; it is still not replayed.


def test_callback_valueerror_is_preserved_without_formatting_or_line_search_retry():
    x = s.Symbol("hybrid_system_exception_x")
    seen = []

    class OriginalError(ValueError):
        def __str__(self):
            raise AssertionError("solver must not format a callback exception")

    class Law(s.Function):
        pass

    original = OriginalError()

    def law(value):
        seen.append(value)
        if value >= 1:
            raise original
        return value - 1

    program = compile_hybrid_residual_system(Law(x, evaluate=False), x, jacobian=[[1]], callbacks={Law: law})
    active = session()
    with pytest.raises(OriginalError) as failure:
        program.solve(0, session=active)
    assert failure.value is original
    assert seen == [0.0, 0.0, 1.0]
    assert active.unknown_effect_tainted
    with pytest.raises(RuntimeError, match="unknown effect"):
        program.solve(0, session=active)
    assert seen == [0.0, 0.0, 1.0]


def test_nested_native_fuel_failure_during_a_trial_is_not_a_domain_retry(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("hybrid_system_native_limit_x")
    program = compile_hybrid_residual_system(x-1, x, jacobian=[[1]], callbacks={})
    observed, native_qr = [], numerical._least_squares_step
    measuring = session()

    def observe(jacobian, residuals):
        observed.append(measuring.consumed_nested_steps)
        return native_qr(jacobian, residuals)

    monkeypatch.setattr(numerical, "_least_squares_step", observe)
    assert values(program.solve(0, session=measuring)) == [1]
    assert len(observed) == 1
    monkeypatch.setattr(numerical, "_least_squares_step", native_qr)
    # The next line-search candidate costs three coordinate operations. The
    # following native residual admission must stop the request immediately.
    limit = observed[0] + 3
    limited = session(max_nested_steps=limit)
    residual_charge = program._residual_program.operation_count + program.variable_count + 1
    with pytest.raises(ValueError, match="nested native request budget exhausted"):
        program.solve(0, session=limited)
    assert limited.consumed_nested_steps == limit + residual_charge
    assert limited.invocations == 0 and not limited.unknown_effect_tainted


def test_invocation_limits_are_cumulative_across_the_entire_solve():
    x = s.Symbol("hybrid_system_invocations_x")
    seen = []

    class Law(s.Function):
        pass

    program = compile_hybrid_residual_system(
        Law(x, evaluate=False), x, jacobian=[[2*x]],
        callbacks={Law: lambda value: seen.append(value) or value*value-2},
    )
    limited = session(max_invocations=3)
    with pytest.raises(ValueError, match="invocation count"):
        program.solve(1, session=limited)
    assert len(seen) == limited.invocations == 3
    assert limited.unknown_effect_tainted


def test_one_ledger_counts_all_iterations_and_can_be_reused_without_reset():
    x = s.Symbol("hybrid_system_ledger_x")

    class Law(s.Function):
        pass

    program = compile_hybrid_residual_system(
        Law(x, evaluate=False), x, jacobian=[[2*x]], callbacks={Law: lambda value: value*value-2},
    )
    measuring = budget()
    assert values(program.solve(1, budget=measuring, session=session())) == pytest.approx([math.sqrt(2)])
    cost = (1 << 30) - measuring.remaining_steps
    assert cost > 0
    shared = budget(max_steps=2*cost)
    for _ in range(2):
        assert values(program.solve(1, budget=shared, session=session())) == pytest.approx([math.sqrt(2)])
    assert shared.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve(1, budget=shared, session=session())
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve(1, budget=budget(max_steps=cost-1), session=session())


def test_nested_hybrid_solve_inherits_the_owning_session_and_ledger():
    x = s.Symbol("hybrid_system_nested_x")
    inner = compile_hybrid_residual_system(x-2, x, jacobian=[[1]], callbacks={})

    class Law(s.Function):
        pass

    outer = s.compile_numeric_hybrid(
        x, Law(x, evaluate=False), callbacks={Law: lambda value: float(inner.solve(0)[0, 0]) + value},
    )
    measuring = budget()
    inner.solve(0, budget=measuring)
    inner_cost = (1 << 30) - measuring.remaining_steps
    active, shared = session(), budget()
    assert outer(3, budget=shared, session=active) == 5
    assert active.invocations == 1
    assert (1 << 30) - shared.remaining_steps > inner_cost
    with pytest.raises(s.EvaluationBudgetExceeded):
        outer(3, budget=budget(max_steps=inner_cost-1), session=session())


def test_nested_solve_cannot_replace_an_active_ledger():
    x = s.Symbol("hybrid_system_reset_x")
    inner = compile_hybrid_residual_system(x-2, x, jacobian=[[1]], callbacks={})
    replacement = budget()

    class Law(s.Function):
        pass

    outer = s.compile_numeric_hybrid(
        x, Law(x, evaluate=False),
        callbacks={Law: lambda value: float(inner.solve(0, budget=replacement)[0, 0])},
    )
    active = session()
    with pytest.raises(ValueError, match="active EvaluationBudget"):
        outer(0, budget=budget(), session=active)
    assert replacement.remaining_steps == 1 << 30
    assert active.unknown_effect_tainted


def test_cancellation_inside_a_callback_stops_search_and_taints_the_session():
    x = s.Symbol("hybrid_system_cancel_x")
    ledger, seen = budget(), []

    class Law(s.Function):
        pass

    def cancel(value):
        seen.append(value)
        ledger.cancel()
        return value - 1

    program = compile_hybrid_residual_system(Law(x, evaluate=False), x, jacobian=[[1]], callbacks={Law: cancel})
    active = session()
    with pytest.raises(s.EvaluationCancelled):
        program.solve(0, budget=ledger, session=active)
    assert seen == [0.0] and active.unknown_effect_tainted


@pytest.mark.parametrize("which", ["budget", "session"])
def test_cancellation_after_actual_result_assembly_prevents_publication(monkeypatch, which):
    matrices = importlib.import_module("sympy.matrices")
    x = s.Symbol("hybrid_system_publication_x")
    program = compile_hybrid_residual_system(x-1, x, jacobian=[[1]], callbacks={})
    active, ledger, created = session(), budget(), []
    constructor = matrices.Matrix

    def cancel_after_assembly(entries):
        result = constructor(entries)
        created.append(result)
        (ledger if which == "budget" else active).cancel()
        return result

    monkeypatch.setattr(matrices, "Matrix", cancel_after_assembly)
    with pytest.raises(s.EvaluationCancelled if which == "budget" else RuntimeError):
        program.solve(0, budget=ledger, session=active)
    assert len(created) == 1 and float(created[0][0, 0]) == 1


def test_batches_admit_all_inputs_and_never_publish_partial_pairs():
    x = s.Symbol("hybrid_system_batch_x")
    seen = []

    class Law(s.Function):
        pass

    def law(value):
        seen.append(value)
        return value if value >= 0 else math.nan

    program = compile_hybrid_residual_system(Law(x, evaluate=False), x, jacobian=[[1]], callbacks={Law: law})
    with pytest.raises((ValueError, TypeError)):
        program.evaluate_batch([(1,), (object(),)])
    assert seen == []
    active = session()
    with pytest.raises(ValueError, match="finite"):
        program.evaluate_batch([(1,), (-1,), (2,)], session=active)
    assert seen == [1.0, -1.0] and active.unknown_effect_tainted


def test_unverified_jacobians_do_not_turn_rank_failures_or_inconsistent_rows_into_roots():
    x, y = s.symbols("hybrid_system_refusal_x hybrid_system_refusal_y")
    wrong = compile_hybrid_residual_system(x-1, x, jacobian=[[0]], callbacks={})
    assert wrong.jacobian(0) == [[0]]  # An explicit user assertion, not a checked derivative.
    with pytest.raises(ValueError, match="rank-deficient"):
        wrong.solve(0)
    inconsistent = compile_hybrid_residual_system([x-1, x-2], x, jacobian=[[1], [1]], callbacks={})
    with pytest.raises(ValueError):
        inconsistent.solve(0)
    underdetermined = compile_hybrid_residual_system(x+y, (x, y), jacobian=[[1, 1]], callbacks={})
    assert underdetermined.evaluate(1, 2) == ([3], [[1, 1]])
    with pytest.raises(ValueError, match="at least as many equations"):
        underdetermined.solve([0, 0])


def test_shape_and_argument_errors_refuse_before_compilation_or_effects(monkeypatch):
    module = importlib.import_module("sympy.utilities.hybrid_system")
    x, y = s.symbols("hybrid_system_shape_x hybrid_system_shape_y")

    def forbidden(*args):
        raise AssertionError("malformed shape reached native compilation")

    monkeypatch.setattr(module, "_compile_graph", forbidden)
    for residuals, args, jacobian in (
        ([], x, []), (x, (), [[1]]), (x, (x, x), [[1, 1]]),
        ([x, y], (x, y), [[1, 0]]), ([x, y], (x, y), [[1], [0]]),
        (x, x, [[[1]]]), (x, x, [1]), ([[x]], x, [[1]]),
        (s.Matrix([[x, y], [x, y]]), (x, y), [[1, 1]]*4),
        ([x]*257, x, [[1]]*257),
        (x, tuple(s.Symbol(f"hybrid_system_dim_{i}") for i in range(65)), [[1]*65]),
        ([x]*64, tuple(s.Symbol(f"hybrid_system_output_{i}") for i in range(64)), [[1]*64]*64),
    ):
        with pytest.raises((TypeError, ValueError)):
            compile_hybrid_residual_system(residuals, args, jacobian=jacobian, callbacks={})


def test_initial_options_and_request_scope_are_admitted_before_any_effect():
    x = s.Symbol("hybrid_system_initial_x")
    seen = []

    class Law(s.Function):
        pass

    program = compile_hybrid_residual_system(
        Law(x, evaluate=False), x, jacobian=[[1]], callbacks={Law: lambda value: seen.append(value) or value-1},
    )
    for initial in ([0, 1], [], [math.inf], [object()], (item for item in [0])):
        with pytest.raises((ValueError, TypeError)):
            program.solve(initial)
    for options in ({"tol": 0}, {"maxiter": 0}, {"maxiter": True}, {"budget": object()}, {"session": object()}):
        with pytest.raises((ValueError, TypeError)):
            program.solve(0, **options)
    cancelled = session()
    cancelled.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        program.solve(0, session=cancelled)
    with supervised_callback_session():
        with pytest.raises(ValueError, match="conflicts"):
            program.solve(0, session=session())
    assert seen == []


def test_initial_snapshot_cannot_expand_after_its_size_check(monkeypatch):
    module = importlib.import_module("sympy.utilities.hybrid_system")
    x = s.Symbol("hybrid_system_growth_x")
    program = compile_hybrid_residual_system(x-1, x, jacobian=[[1]], callbacks={})
    initial, changed, real_len = [0], [], len

    def grow(value):
        size = real_len(value)
        if value is initial and not changed:
            changed.append(True)
            initial.extend([0]*1000)
        return size

    monkeypatch.setattr(module, "len", grow, raising=False)
    with pytest.raises(ValueError, match="changed during"):
        program.solve(initial)
    assert changed == [True]


def test_public_exports_point_to_the_explicit_hybrid_api():
    from sympy.utilities import compile_hybrid_residual_system as utility

    assert utility is compile_hybrid_residual_system is s.compile_hybrid_residual_system
