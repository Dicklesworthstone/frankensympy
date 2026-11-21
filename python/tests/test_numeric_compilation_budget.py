"""One caller ledger spans source capture, checked compilation and execution."""

import importlib
import math
import threading
import time

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


LIMIT = 1 << 40


def ledger(**changes):
    limits = dict(max_steps=LIMIT, max_memory_bytes=LIMIT, max_allocations=LIMIT)
    limits.update(changes)
    return s.EvaluationBudget(**limits)


def remaining(budget):
    return (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations)


def compile_program(kind, source, x, budget):
    if kind == "scalar":
        return s.compile_numeric(x, source, budget=budget)
    return s.compile_residual_system(source, x, budget=budget)


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_compilation_and_evaluation_share_a_cumulative_ledger(kind):
    x = s.Symbol("compilation_budget_x", real=True)
    source = x**3 + s.sin(x)
    budget = ledger()
    program = compile_program(kind, source, x, budget)
    after_compile = remaining(budget)
    assert all(0 < amount < LIMIT for amount in after_compile)
    if kind == "scalar":
        assert program(2, budget=budget) == pytest.approx(8 + math.sin(2))
    else:
        values, jacobian = program.evaluate(2, budget=budget)
        assert values == pytest.approx([8 + math.sin(2)])
        assert jacobian[0] == pytest.approx([12 + math.cos(2)])
    assert all(after < before for after, before in zip(remaining(budget), after_compile))


def test_distinct_compilers_share_the_same_existing_request_budget():
    x, y = s.symbols("shared_compile_x shared_compile_y")
    budget = ledger()
    scalar = s.compile_numeric((x, y), x*y + x, budget=budget)
    scalar_remaining = remaining(budget)
    residual = s.compile_residual_system((x*y-6, x+y-5), (x, y), budget=budget)
    residual_remaining = remaining(budget)
    assert all(after < before for after, before in zip(residual_remaining, scalar_remaining))
    assert scalar(2, 3, budget=budget) == 8
    assert residual.evaluate(2, 3, budget=budget) == ([0, 0], [[3, 2], [1, 1]])
    assert all(after < before for after, before in zip(remaining(budget), residual_remaining))


@pytest.mark.parametrize("kind", ["scalar", "residual"])
@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_public_and_direct_native_compilation_refuse_each_exhausted_resource(kind, dimension):
    x = s.Symbol("compile_empty_ledger")
    source = x*x + 1
    with pytest.raises(s.EvaluationBudgetExceeded):
        compile_program(kind, source, x, ledger(**{dimension: 0}))
    native = _native.CompiledFunction if kind == "scalar" else _native.VerifiedResidualSystem
    with pytest.raises(s.EvaluationBudgetExceeded):
        native([_native_expr(source)], [_symbol_binding(x)], budget=ledger(**{dimension: 0}))


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_late_failure_retains_cost_and_does_not_reset_for_a_second_compilation(kind):
    x = s.Symbol("compile_repeated_ledger")
    source = s.exp(x) + x*x + 1
    measurement = ledger()
    compile_program(kind, source, x, measurement)
    spent = LIMIT - measurement.remaining_steps
    assert spent > 1
    budget = ledger(max_steps=2*spent-1)
    first = compile_program(kind, source, x, budget)
    after_first = budget.remaining_steps
    published = []
    with pytest.raises(s.EvaluationBudgetExceeded):
        published.append(compile_program(kind, source, x, budget))
    assert published == []
    assert budget.remaining_steps < after_first
    # A later failed request cannot corrupt the previously admitted program.
    if kind == "scalar":
        assert first(0) == 2
    else:
        assert first.evaluate(0) == ([2], [[1]])


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_cancelled_compilation_refuses_before_python_source_capture(kind, monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x = s.Symbol("compile_cancel_before_capture")
    budget = ledger()
    budget.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled compilation must not capture its source")

    monkeypatch.setattr(module, "_outputs", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        compile_program(kind, x, x, budget)
    assert remaining(budget) == (LIMIT, LIMIT, LIMIT)


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_cancellation_during_capture_is_checked_before_native_compilation(kind, monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x = s.Symbol("compile_cancel_after_capture")
    budget = ledger()
    original = module._outputs

    def capture_then_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_outputs", capture_then_cancel)
    with pytest.raises(s.EvaluationCancelled):
        compile_program(kind, x*x, x, budget)


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_cancellation_after_native_admission_cannot_publish_a_python_program(kind, monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x = s.Symbol("compile_cancel_publication")
    budget = ledger()
    class_name = "CompiledFunction" if kind == "scalar" else "VerifiedResidualSystem"
    original = getattr(module._native, class_name)

    def admit_then_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(module._native, class_name, admit_then_cancel)
    published = []
    with pytest.raises(s.EvaluationCancelled):
        published.append(compile_program(kind, x*x, x, budget))
    assert published == []
    assert budget.remaining_steps < LIMIT


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_admission_refusals_retain_work_and_original_source_domains(kind):
    x = s.Symbol("compile_original_domain")
    foreign = s.Symbol("compile_foreign_symbol")
    budget = ledger()
    with pytest.raises(ValueError):
        compile_program(kind, x + foreign, x, budget)
    after_refusal = remaining(budget)
    assert all(amount < LIMIT for amount in after_refusal)
    source = s.Mul(0, s.log(x), evaluate=False)
    program = compile_program(kind, source, x, budget)
    assert all(after < before for after, before in zip(remaining(budget), after_refusal))
    with pytest.raises(ValueError):
        if kind == "scalar":
            program(-1, budget=budget)
        else:
            program.evaluate(-1, budget=budget)


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_typed_axis_identity_and_argument_layout_survive_metered_compilation(kind):
    plain = s.Symbol("compilation_same_name")
    positive = s.Symbol("compilation_same_name", positive=True)
    source = plain + 2*positive
    budget = ledger()
    if kind == "scalar":
        program = s.compile_numeric([(plain, positive)], source, budget=budget)
        assert program((3, 5)) == 13
    else:
        program = s.compile_residual_system(source, [(plain, positive)], budget=budget)
        assert program.evaluate((3, 5)) == ([13], [[1, 2]])
        assert program.variables == (plain, positive)


@pytest.mark.parametrize("kind", ["scalar", "residual"])
def test_invalid_budget_refuses_without_source_inspection(kind):
    x = s.Symbol("compile_budget_type")
    with pytest.raises(TypeError, match="EvaluationBudget"):
        compile_program(kind, object(), x, object())


def test_native_derivative_compilation_releases_the_gil_for_external_cancellation():
    # Use the real public native constructor and caller-owned cancellation;
    # observing a locked ledger from Python proves the GIL has been released.
    x = s.Symbol("compilation_external_cancel")
    sources = [_native_expr(s.sin((i + 1)*x) + x**2) for i in range(512)]
    variables = [_symbol_binding(x)]
    shared = ledger()
    outcomes = []
    started = threading.Event()

    def compile_program():
        started.set()
        try:
            outcomes.append(_native.VerifiedResidualSystem(sources, variables, budget=shared))
        except Exception as error:
            outcomes.append(error)

    worker = threading.Thread(target=compile_program)
    worker.start()
    assert started.wait(5)
    observed_busy = False
    deadline = time.monotonic() + 10
    try:
        while worker.is_alive() and time.monotonic() < deadline:
            try:
                shared.remaining_steps
            except RuntimeError as error:
                assert "already in use" in str(error)
                observed_busy = True
                break
            time.sleep(0.001)
    finally:
        shared.cancel()
        worker.join(10)
    assert not worker.is_alive()
    assert observed_busy
    assert len(outcomes) == 1 and isinstance(outcomes[0], s.EvaluationCancelled)
    assert shared.remaining_steps < LIMIT
