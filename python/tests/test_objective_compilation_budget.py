"""One caller ledger spans real objective generation, proof admission and use."""

import importlib

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


_LIMITS = {
    "max_steps": 100_000_000,
    "max_memory_bytes": 512 * 1024 * 1024,
    "max_allocations": 1_000_000,
}


def _budget(**changes):
    return s.EvaluationBudget(**(_LIMITS | changes))


def _remaining(budget):
    return (
        budget.remaining_steps,
        budget.remaining_memory_bytes,
        budget.remaining_allocations,
    )


def test_compilation_evaluation_and_transport_share_consumed_resources():
    x, y = s.symbols("compile_budget_objective_x compile_budget_objective_y")
    source = (x - 1)**2 + 3*(y + 2)**2 + x*y
    budget = _budget()
    before = _remaining(budget)
    program = s.compile_objective(source, (x, y), budget=budget)
    compiled = _remaining(budget)
    assert all(after < start for after, start in zip(compiled, before))
    assert program.native_objective.proofs_verified == 6
    assert program.hessian_pattern == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert program.evaluate(2, 1, budget=budget) == (30, [3, 20], [[2, 1], [1, 6]])
    evaluated = budget.remaining_steps
    assert evaluated < compiled[0]

    encoded = program.to_bytes(budget=budget)
    exported = budget.remaining_steps
    restored = s.load_objective(encoded, objective=source, args=(x, y), budget=budget)
    assert budget.remaining_steps < exported < evaluated
    assert restored.hessian_vector_product([2, -1], 2, 1, budget=budget) == [3, -4]
    assert restored.variables[0] is x and restored.variables[1] is y


def test_scalar_residual_and_objective_compilers_share_the_same_caller_budget():
    x = s.Symbol("compile_budget_cross_compiler")
    budget = _budget()
    scalar = s.compile_numeric(x, x*x, budget=budget)
    after_scalar = budget.remaining_steps
    residual = s.compile_residual_system([x*x - 4], x, budget=budget)
    after_residual = budget.remaining_steps
    objective = s.compile_objective((x - 2)**2, x, budget=budget)
    after_objective = budget.remaining_steps
    assert after_objective < after_residual < after_scalar < _LIMITS["max_steps"]
    assert scalar(3, budget=budget) == 9
    assert residual.evaluate(3, budget=budget) == ([5], [[6]])
    assert objective.evaluate(3, budget=budget) == (1, [2], [[2]])
    assert budget.remaining_steps < after_objective


def test_repeated_compilation_exhausts_one_budget_without_resetting_it():
    x = s.Symbol("compile_budget_repeated_objective")
    source = x*x + 3*x
    measured = _budget()
    s.compile_objective(source, x, budget=measured)
    cost = _LIMITS["max_steps"] - measured.remaining_steps
    assert cost > 1

    budget = _budget(max_steps=2*cost)
    first = s.compile_objective(source, x, budget=budget)
    assert budget.remaining_steps == cost
    second = s.compile_objective(source, x, budget=budget)
    assert budget.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.compile_objective(source, x, budget=budget)
    assert budget.remaining_steps == 0
    assert first.evaluate(2) == second.evaluate(2) == (10, [7], [[2]])
    with pytest.raises(s.EvaluationBudgetExceeded):
        first.evaluate(2, budget=budget)


def test_late_compilation_exhaustion_cannot_publish_a_partial_objective(monkeypatch):
    module = importlib.import_module("sympy.utilities.objective")
    x = s.Symbol("compile_budget_late_objective")
    source = s.exp(x) + x*x
    measured = _budget()
    s.compile_objective(source, x, budget=measured)
    cost = _LIMITS["max_steps"] - measured.remaining_steps
    budget = _budget(max_steps=cost - 1)

    def forbidden_publication(*args, **kwargs):
        pytest.fail("an incompletely admitted objective reached public construction")

    monkeypatch.setattr(module, "_ObjectiveCallable", forbidden_publication)
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.compile_objective(source, x, budget=budget)
    assert budget.remaining_steps < cost - 1


@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
@pytest.mark.parametrize("native", [False, True])
def test_all_compilation_budget_dimensions_have_typed_refusals(dimension, native):
    x = s.Symbol("compile_budget_empty_dimension")
    budget = _budget(**{dimension: 0})
    with pytest.raises(s.EvaluationBudgetExceeded) as failure:
        if native:
            _native.VerifiedObjective(
                _native_expr(x*x), [_symbol_binding(x)], budget=budget,
            )
        else:
            s.compile_objective(x*x, x, budget=budget)
    assert not isinstance(failure.value, ValueError)
    assert not budget.cancelled


@pytest.mark.parametrize("native", [False, True])
def test_cancellation_is_checked_before_source_or_coordinate_admission(native):
    budget = _budget()
    before = _remaining(budget)
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        if native:
            _native.VerifiedObjective(object(), object(), budget=budget)
        else:
            s.compile_objective(object(), object(), budget=budget)
    assert _remaining(budget) == before


def test_cancellation_after_source_capture_stops_before_native_generation(monkeypatch):
    module = importlib.import_module("sympy.utilities.objective")
    x = s.Symbol("compile_budget_capture_cancel")
    budget = _budget()
    original = module._objective_source
    captured = []

    def cancel_after_capture(*args):
        result = original(*args)
        captured.append(True)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_objective_source", cancel_after_capture)
    with pytest.raises(s.EvaluationCancelled):
        s.compile_objective(x*x, x, budget=budget)
    assert captured == [True]
    assert budget.remaining_steps == _LIMITS["max_steps"]


def test_cancellation_at_publication_discards_the_admitted_result(monkeypatch):
    module = importlib.import_module("sympy.utilities.objective")
    x = s.Symbol("compile_budget_publication_cancel")
    budget = _budget()
    original = module._ObjectiveCallable
    admitted = []

    def cancel_before_publication(*args, **kwargs):
        result = original(*args, **kwargs)
        admitted.append(result.native_objective.proofs_verified)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "_ObjectiveCallable", cancel_before_publication)
    with pytest.raises(s.EvaluationCancelled):
        s.compile_objective(x*x, x, budget=budget)
    assert admitted == [2]
    assert budget.remaining_steps < _LIMITS["max_steps"]


def test_failed_candidate_generation_keeps_cost_and_later_work_uses_the_remainder():
    x = s.Symbol("compile_budget_unsupported_objective")
    budget = _budget()
    with pytest.raises(ValueError):
        s.compile_objective(s.floor(x), x, budget=budget)
    after_failure = budget.remaining_steps
    assert 0 < after_failure < _LIMITS["max_steps"]
    program = s.compile_objective(x*x, x, budget=budget)
    assert budget.remaining_steps < after_failure
    assert program.evaluate(3, budget=budget) == (9, [6], [[2]])


def test_invalid_budget_is_rejected_before_python_source_capture(monkeypatch):
    module = importlib.import_module("sympy.utilities.objective")

    def forbidden_capture(*args):
        pytest.fail("source capture must follow budget admission")

    monkeypatch.setattr(module, "_objective_source", forbidden_capture)
    with pytest.raises(TypeError, match="EvaluationBudget"):
        s.compile_objective(object(), object(), budget=object())


def test_budgeted_compilation_retains_original_held_domain_and_float_restriction():
    x = s.Symbol("compile_budget_original_domain")
    source = s.Add(x*x, s.Mul(0, s.log(x), evaluate=False), evaluate=False)
    budget = _budget()
    program = s.compile_objective(source, x, budget=budget)
    with pytest.raises(ValueError):
        program.hessian_vector_product([0], -1, budget=budget)
    assert program.evaluate(2, budget=budget) == (4, [4], [[2]])

    approximate = s.compile_objective(x*x + s.Float(0.5), x, budget=budget)
    assert approximate.evaluate(2, budget=budget) == (4.5, [4], [[2]])
    with pytest.raises(ValueError):
        approximate.enclose([(1, 2)], budget=budget)
