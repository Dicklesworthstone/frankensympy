"""Caller-owned numerical budgets compose without resetting or publishing partial work."""

import importlib
import math
import threading
import time
from types import SimpleNamespace

import pytest
import sympy as s

from sympy.core import _native


def consumed_steps(call):
    budget = s.EvaluationBudget(max_steps=1_000_000)
    result = call(budget)
    return result, 1_000_000 - budget.remaining_steps


def test_reusable_budget_composes_across_calls_and_different_programs():
    x = s.Symbol("budget_x")
    square = s.compile_numeric(x, x*x)
    shifted = s.compile_numeric(x, x+1)
    _, square_cost = consumed_steps(lambda b: square(3, budget=b))
    _, shifted_cost = consumed_steps(lambda b: shifted(3, budget=b))
    assert square_cost > 0 and shifted_cost > 0
    budget = s.EvaluationBudget(max_steps=square_cost + shifted_cost)
    assert square(3, budget=budget) == 9
    assert shifted(3, budget=budget) == 4
    assert budget.remaining_steps == 0
    for program in (square, shifted):
        with pytest.raises(s.EvaluationBudgetExceeded):
            program(3, budget=budget)
    assert budget.remaining_steps == 0
    assert square(4) == 16


@pytest.mark.parametrize("option", ["max_memory_bytes", "max_allocations"])
def test_native_input_buffers_are_charged_atomically_before_domain_work(option):
    x = s.Symbol("budget_memory_x")
    program = s.compile_numeric(x, s.log(x))
    budget = s.EvaluationBudget(max_steps=1000, **{option: 0})
    before = (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program(-1, budget=budget)
    assert (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations) == before


def test_domain_failure_keeps_consumption_and_budget_remains_reusable():
    x = s.Symbol("budget_domain_x")
    program = s.compile_numeric(x, s.log(x))
    budget = s.EvaluationBudget(max_steps=1000)
    with pytest.raises(ValueError):
        program(-1, budget=budget)
    after_failure = budget.remaining_steps
    assert 0 < after_failure < 1000
    assert program(1, budget=budget) == 0
    assert budget.remaining_steps < after_failure
    assert not issubclass(s.EvaluationBudgetExceeded, ValueError)
    assert not issubclass(s.EvaluationCancelled, ValueError)


def test_residual_jacobian_and_hessian_share_a_ledger_through_every_stage():
    x, y = s.symbols("budget_residual_x budget_residual_y")
    residual = s.compile_residual_system([x*x+y, y*y-x], (x, y))
    objective = s.compile_objective(x*x + x*y + y*y, (x, y))
    result, residual_cost = consumed_steps(lambda b: residual.residuals(2, 3, budget=b))
    assert result == [7, 7]
    full, full_cost = consumed_steps(lambda b: residual.evaluate(2, 3, budget=b))
    assert full == ([7, 7], [[4, 1], [-1, 6]])
    assert full_cost > residual_cost
    partial = s.EvaluationBudget(max_steps=residual_cost + 1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        residual.evaluate(2, 3, budget=partial)
    assert partial.remaining_steps < residual_cost + 1

    shared = s.EvaluationBudget(max_steps=100_000)
    residual.jacobian(2, 3, budget=shared)
    first = shared.remaining_steps
    assert objective.evaluate_sparse(2, 3, budget=shared) == (19, [7, 8], [2, 1, 1, 2])
    second = shared.remaining_steps
    assert objective.hessian_vector_product([3, 5], 2, 3, budget=shared) == [11, 13]
    assert 0 < shared.remaining_steps < second < first < 100_000


def test_budgeted_derivatives_preserve_original_domains_and_zero_direction_checks():
    x = s.Symbol("budget_guard_x")
    source = s.Add(x, s.Mul(0, s.log(x), evaluate=False), evaluate=False)
    residual = s.compile_residual_system([source], [x])
    objective = s.compile_objective(source, x)
    budget = s.EvaluationBudget()
    for call in (
        lambda: residual.jacobian(-1, budget=budget),
        lambda: objective.hessian_vector_product([0], -1, budget=budget),
    ):
        with pytest.raises(ValueError):
            call()
    assert objective.hessian_vector_product([0], 1, budget=budget) == [0]


def test_scalar_batch_preserves_nested_arguments_and_outputs():
    x, y, z = s.symbols("budget_batch_x budget_batch_y budget_batch_z")
    program = s.compile_numeric(((x, y), z), [x+y, (x*z, y)])
    rows = [((1, 2), 3), ((4, 5), 6)]
    assert program.evaluate_batch(rows, budget=s.EvaluationBudget()) == [
        [3, (3, 2)], [9, (24, 5)],
    ]
    assert program.evaluate_batch([]) == []
    constant = s.compile_numeric((), (s.Integer(7), []))
    assert constant.evaluate_batch([(), ()]) == [(7, []), (7, [])]


def test_batches_share_budget_and_refuse_late_domain_or_derivative_failures():
    x = s.Symbol("budget_batch_domain_x")
    program = s.compile_numeric(x, s.log(x))
    expected, cost = consumed_steps(lambda b: program.evaluate_batch([(1,), (2,), (3,)], budget=b))
    assert expected == pytest.approx([0, math.log(2), math.log(3)])
    limited = s.EvaluationBudget(max_steps=cost-1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate_batch([(1,), (2,), (3,)], budget=limited)
    assert limited.remaining_steps < cost-1
    with pytest.raises(ValueError):
        program.evaluate_batch([(1,), (-1,), (2,)], budget=s.EvaluationBudget())

    system = s.compile_residual_system([s.sqrt(x)], [x])
    assert system.evaluate_batch([(1,), (4,)]) == [([1], [[0.5]]), ([2], [[0.25]])]
    # The final residual is valid, but its derivative is singular. Returning
    # either preceding row or a residual-only final row would be incorrect.
    with pytest.raises(ValueError):
        system.evaluate_batch([(1,), (4,), (0,)], budget=s.EvaluationBudget())


def test_batch_envelopes_are_checked_before_work_and_survive_mutable_growth(monkeypatch):
    x = s.Symbol("budget_batch_bounds_x")
    program = s.compile_numeric(x, x)
    budget = s.EvaluationBudget(max_steps=1000)
    for rows in (((1,) for _ in range(2)), [(1,)]*4097):
        with pytest.raises((TypeError, ValueError)):
            program.evaluate_batch(rows, budget=budget)
    assert budget.remaining_steps == 1000

    module = importlib.import_module("sympy.utilities.lambdify")
    original = module._call_arguments
    rows = [(1,), (2,)]
    seen = []

    def grow(layout, row, *, native):
        seen.append(row)
        rows.append((3,))
        return original(layout, row, native=native)

    monkeypatch.setattr(module, "_call_arguments", grow)
    with pytest.raises(ValueError, match="length changed"):
        program.evaluate_batch(rows, budget=budget)
    assert len(seen) == 2
    assert budget.remaining_steps == 1000


def test_cancellation_is_irreversible_and_covers_empty_and_zero_outputs():
    x = s.Symbol("budget_cancel_x")
    program = s.compile_numeric(x, x*x)
    system = s.compile_residual_system([], [x])
    objective = s.compile_objective(x, x)
    budget = s.EvaluationBudget()
    budget.cancel()
    budget.cancel()
    assert budget.cancelled
    for call in (
        lambda: program(1, budget=budget),
        lambda: program.evaluate_batch([], budget=budget),
        lambda: system.evaluate(1, budget=budget),
        lambda: objective.hessian_vector_product([0], 1, budget=budget),
    ):
        with pytest.raises(s.EvaluationCancelled):
            call()
    with pytest.raises(AttributeError):
        budget.cancelled = False


def test_limits_and_budget_types_refuse_custom_conversion_hooks():
    calls = []

    class CustomInt(int):
        def __index__(self):
            calls.append("index")
            return 100

    for value in (True, 1.0, "1", CustomInt(100)):
        with pytest.raises(TypeError):
            s.EvaluationBudget(max_steps=value)
    for value in (-1, 2**64):
        with pytest.raises(ValueError):
            s.EvaluationBudget(max_steps=value)
    x = s.Symbol("budget_type_x")
    with pytest.raises(TypeError):
        s.compile_numeric(x, x)(1, budget=object())
    with pytest.raises(TypeError):
        s.nsolve(x-1, x, 0, budget=object())
    assert calls == []


def test_nsolve_scalar_and_system_preserve_one_budget_across_iterations():
    x, y = s.symbols("budget_newton_x budget_newton_y")
    scalar, scalar_cost = consumed_steps(lambda b: s.nsolve(x*x-2, x, 1, budget=b))
    assert float(scalar) == pytest.approx(math.sqrt(2), abs=1e-12)
    one_solve = s.EvaluationBudget(max_steps=scalar_cost)
    assert float(s.nsolve(x*x-2, x, 1, budget=one_solve)) == pytest.approx(math.sqrt(2))
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(x*x-2, x, 1, budget=one_solve)
    budget = s.EvaluationBudget()
    result = s.nsolve((s.log(x), y-1), (x, y), (10, 0), budget=budget)
    assert [float(result[i, 0]) for i in range(2)] == pytest.approx([1, 1])
    assert budget.remaining_steps < 10_000_000


@pytest.mark.parametrize("cancel", [False, True])
def test_nsolve_never_retries_operational_failure_as_a_domain_crossing(monkeypatch, cancel):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x, y = s.symbols("budget_backtrack_x budget_backtrack_y")
    program = s.compile_residual_system((s.log(x), y-1), (x, y))
    native = program.native_system
    negative_trials = []

    def residuals(values, *, budget):
        if values[0] < 0:
            negative_trials.append(tuple(values))
            if cancel:
                budget.cancel()
            else:
                budget._charge(2**64-1)
        return native.residuals(values, budget=budget)

    proxy = SimpleNamespace(
        native_system=SimpleNamespace(
            residual_count=native.residual_count, variable_count=native.variable_count,
            residuals=residuals, jacobian=native.jacobian,
        ),
    )
    monkeypatch.setattr(numerical, "compile_residual_system", lambda *args: proxy)
    exception = s.EvaluationCancelled if cancel else s.EvaluationBudgetExceeded
    with pytest.raises(exception):
        s.nsolve((s.log(x), y-1), (x, y), (10, 0), budget=s.EvaluationBudget())
    assert len(negative_trials) == 1


@pytest.mark.parametrize("mapping", [False, True])
def test_nsolve_checks_cancellation_after_system_result_assembly(monkeypatch, mapping):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("budget_publication_x")
    budget = s.EvaluationBudget()

    def cancelled_result(*args, budget):
        budget.cancel()
        return [x], [1.0]

    monkeypatch.setattr(numerical, "_solve_system", cancelled_result)
    with pytest.raises(s.EvaluationCancelled):
        s.nsolve([x-1], [x], [0], budget=budget, dict=mapping)


def test_native_evaluation_allows_an_external_python_caller_to_cancel():
    # This thread is an external caller, not an executor created by the engine.
    # Observing the occupied native ledger while Python runs proves the GIL is
    # released; no timing assumption is needed to make cancel win a short call.
    x = s.Symbol("budget_thread_x")
    source = s.Add(*(s.sin((i+1)*x) for i in range(256)), evaluate=False)
    native = s.compile_numeric(x, source).native_program
    budget = s.EvaluationBudget(max_steps=100_000_000, max_memory_bytes=512*1024*1024,
                                max_allocations=100_000)
    outcomes = []
    started = threading.Event()

    def evaluate():
        started.set()
        try:
            outcomes.append(native.evaluate_batch([(0.1,)]*4096, budget=budget))
        except Exception as error:
            outcomes.append(error)

    worker = threading.Thread(target=evaluate)
    worker.start()
    assert started.wait(5)
    observed_busy = False
    deadline = time.monotonic() + 5
    try:
        while worker.is_alive() and time.monotonic() < deadline:
            try:
                budget.remaining_steps
            except RuntimeError as error:
                assert "already in use" in str(error)
                observed_busy = True
                break
            time.sleep(0.001)
    finally:
        budget.cancel()
        worker.join(5)
    assert not worker.is_alive()
    assert observed_busy
    assert len(outcomes) == 1 and isinstance(outcomes[0], s.EvaluationCancelled)
    assert budget.cancelled
    assert budget.remaining_steps < 100_000_000


def test_native_exports_and_public_utility_exports_agree():
    utilities = importlib.import_module("sympy.utilities")
    for name in ("EvaluationBudget", "EvaluationBudgetExceeded", "EvaluationCancelled"):
        assert getattr(s, name) is getattr(utilities, name) is getattr(_native, name)
