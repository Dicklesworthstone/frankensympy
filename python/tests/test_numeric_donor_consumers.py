"""Real pinned numerical consumers behind the public checked residual API."""

import importlib
import math

import pytest
import sympy as s

from sympy.utilities.lambdify import NumericRootCandidate


BACKENDS = ("frankennumpy", "frankenscipy")


def linear_program(dimension):
    variables = tuple(s.Symbol(f"donor_coordinate_{index}") for index in range(dimension))
    residuals = []
    for index, variable in enumerate(variables):
        value = 2 * variable - 2 * (index + 1)
        if index + 1 < dimension:
            value += variables[index + 1] - (index + 2)
        residuals.append(value)
    return s.compile_residual_system(residuals, variables), variables


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dimension", [1, 3, 16])
def test_real_consumers_solve_bounded_dimensions_and_report_actual_residuals(backend, dimension):
    program, variables = linear_program(dimension)
    budget = s.EvaluationBudget()
    candidate = program.solve_candidate([0] * dimension, backend=backend, budget=budget)
    assert type(candidate) is NumericRootCandidate
    assert candidate.root == pytest.approx(tuple(range(1, dimension + 1)), abs=1e-8)
    assert type(candidate.root) is tuple
    assert all(type(value) is float and math.isfinite(value) for value in candidate.root)
    assert all(left is right for left, right in zip(candidate.variables, variables))
    residuals = program.native_system.residuals(list(candidate.root))
    assert candidate.residual_norm == max(abs(value) for value in residuals)
    assert candidate.residual_norm <= 1e-10
    assert type(candidate.iterations) is int and candidate.iterations > 0
    assert candidate.backend == backend and candidate.evidence == "approximate"
    assert budget.remaining_steps < 10_000_000
    assert budget.remaining_memory_bytes < 256 * 1024 * 1024
    assert budget.remaining_allocations < 100_000
    assert not hasattr(candidate, "root_enclosure")
    assert "NumericRootCandidate" in repr(candidate)


def test_candidate_is_immutable_and_default_backend_is_the_real_linear_consumer():
    from sympy.utilities import NumericRootCandidate as UtilityCandidate

    assert s.NumericRootCandidate is UtilityCandidate is NumericRootCandidate
    program, _ = linear_program(1)
    initial = [0]
    candidate = program.solve_candidate(initial)
    initial[0] = 999
    assert candidate.root == (1.0,)
    assert candidate.backend == "frankennumpy"
    for name in ("root", "variables", "residual_norm", "iterations", "backend", "evidence", "_root"):
        with pytest.raises(AttributeError):
            setattr(candidate, name, None)
        with pytest.raises(AttributeError):
            delattr(candidate, name)
    with pytest.raises(AttributeError):
        candidate.__init__([2.0], (), 0.0, 0, "frankenscipy")
    with pytest.raises(TypeError):
        candidate.root[0] = 2.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_typed_coordinates_nested_arguments_and_transport_preserve_caller_identity(backend):
    plain = s.Symbol("donor_same_name")
    positive = s.Symbol("donor_same_name", positive=True)
    left, right = s.Dummy("donor_same_name"), s.Dummy("donor_same_name")
    variables = (right, positive, plain, left)
    residuals = [plain + 2 * left - 7, positive - 2, left - 3, right + plain - 5]
    layout = ((right, positive), (plain, left))
    original = s.compile_residual_system(residuals, layout)
    program = s.load_residual_system(original.to_bytes(), residuals=residuals, args=layout)
    candidate = program.solve_candidate([0, 0, 0, 0], backend=backend)
    assert candidate.root == pytest.approx((4.0, 2.0, 1.0, 3.0), abs=1e-8)
    assert all(left is right for left, right in zip(candidate.variables, variables))
    assert program.residuals(candidate.root[:2], candidate.root[2:]) == pytest.approx([0] * 4, abs=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nonlinear_mixed_system_uses_retained_checked_source(backend):
    x, y, z = s.symbols("donor_nonlinear_x donor_nonlinear_y donor_nonlinear_z")
    program = s.compile_residual_system(
        [s.exp(x) + y - 2, x + s.sin(y) - s.sin(1), z*z + y - 5], (z, y, x),
    )
    candidate = program.solve_candidate([1.8, 0.9, -0.1], backend=backend)
    assert candidate.root == pytest.approx((2.0, 1.0, 0.0), abs=1e-8)
    assert candidate.residual_norm <= 1e-10


@pytest.mark.parametrize("backend", BACKENDS)
def test_source_domain_failures_survive_zero_products_reciprocals_and_zero_powers(backend):
    x = s.Symbol("donor_domain_x")
    sources = (
        s.Pow(s.log(x), -1, evaluate=False),
        s.Mul(0, s.log(x), evaluate=False),
        s.Add(s.Pow(s.log(x), 0, evaluate=False), -1, evaluate=False),
    )
    budget = s.EvaluationBudget()
    for source in sources:
        program = s.compile_residual_system([source], [x])
        before = budget.remaining_steps
        with pytest.raises(ValueError):
            program.solve_candidate(0, backend=backend, budget=budget)
        assert budget.remaining_steps < before


def test_newton_controls_are_effective_and_zero_iterations_only_admit_an_initial_root():
    program, _ = linear_program(1)
    assert program.solve_candidate(1, maxiter=0, tol=0).iterations == 0
    with pytest.raises(ValueError, match="no convergence"):
        program.solve_candidate(0, maxiter=0)
    with pytest.raises(ValueError, match="no convergence"):
        program.solve_candidate(0, max_step_norm=0.1)
    assert program.solve_candidate(0, maxiter=1, tol=0, max_step_norm=2).root == (1.0,)
    x = s.Symbol("donor_singular_root_x")
    singular = s.compile_residual_system([x*x], [x]).solve_candidate(0)
    assert singular.root == (0.0,) and singular.residual_norm == 0.0
    assert singular.evidence == "approximate"


@pytest.mark.parametrize("option,value", [("tol", 1e-10), ("maxiter", 200), ("max_step_norm", 1e6)])
def test_scipy_controls_are_never_silently_ignored_or_run_before_refusal(option, value):
    program, _ = linear_program(1)
    calls = []

    class Conversion:
        def __float__(self):
            calls.append("float")
            return 0.0

    budget = s.EvaluationBudget(max_steps=1000)
    with pytest.raises(ValueError, match="fixed solver controls"):
        program.solve_candidate(Conversion(), backend="frankenscipy", budget=budget, **{option: value})
    with pytest.raises(ValueError, match="fixed solver controls"):
        program.native_system.solve_candidate([Conversion()], backend="frankenscipy", budget=budget,
                                              **{option: value})
    assert calls == [] and budget.remaining_steps == 1000


def test_invalid_backends_options_shapes_and_inputs_refuse_before_native_work():
    program, _ = linear_program(1)
    budget = s.EvaluationBudget(max_steps=1000)
    for options in (
        {"backend": "python"}, {"backend": "unknown"}, {"backend": 1},
        {"tol": -1}, {"tol": math.inf}, {"maxiter": True}, {"maxiter": -1},
        {"maxiter": 1025}, {"max_step_norm": 0}, {"max_step_norm": math.nan},
    ):
        with pytest.raises((TypeError, ValueError)):
            program.solve_candidate(0, budget=budget, **options)
    for initial in ([], [0, 1], [math.inf], [complex(0, 1)], (value for value in [0])):
        with pytest.raises((TypeError, ValueError)):
            program.solve_candidate(initial, budget=budget)
    x, y = s.symbols("donor_shape_x donor_shape_y")
    for invalid in (
        s.compile_residual_system([x, x+1], [x]),
        s.compile_residual_system([x+y], [x, y]),
        s.compile_residual_system([1], []),
        linear_program(17)[0],
    ):
        with pytest.raises(ValueError, match="square system"):
            invalid.solve_candidate([], budget=budget)
    assert budget.remaining_steps == 1000


@pytest.mark.parametrize("dimension,limit", [
    ("max_steps", 100), ("max_memory_bytes", 1000), ("max_allocations", 9),
])
def test_scipy_callback_refusal_preserves_typed_exhaustion_and_cumulative_accounting(dimension, limit):
    program, _ = linear_program(1)
    budget = s.EvaluationBudget(**{dimension: limit})
    remaining = {
        "max_steps": "remaining_steps", "max_memory_bytes": "remaining_memory_bytes",
        "max_allocations": "remaining_allocations",
    }[dimension]
    with pytest.raises(s.EvaluationBudgetExceeded) as failed:
        program.solve_candidate(0, backend="frankenscipy", budget=budget)
    assert not isinstance(failed.value, ValueError)
    after = getattr(budget, remaining)
    assert after < limit
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_candidate(0, backend="frankenscipy", budget=budget)
    assert getattr(budget, remaining) <= after


@pytest.mark.parametrize("backend", BACKENDS)
def test_budget_is_reusable_across_solves_and_cancellation_is_irreversible(backend):
    program, _ = linear_program(1)
    measure = s.EvaluationBudget(max_steps=1_000_000)
    program.solve_candidate(0, backend=backend, budget=measure)
    cost = 1_000_000 - measure.remaining_steps
    budget = s.EvaluationBudget(max_steps=2 * cost)
    assert program.solve_candidate(0, backend=backend, budget=budget).root == pytest.approx((1,))
    assert program.solve_candidate(0, backend=backend, budget=budget).root == pytest.approx((1,))
    assert budget.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_candidate(0, backend=backend, budget=budget)
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.solve_candidate(0, backend=backend, budget=budget)
    assert budget.cancelled


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancellation_during_private_result_assembly_prevents_candidate_publication(monkeypatch, backend):
    module = importlib.import_module("sympy.utilities.lambdify")
    constructor = module.NumericRootCandidate
    program, _ = linear_program(1)
    budget = s.EvaluationBudget()

    def cancel_after_assembly(*args):
        result = constructor(*args)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "NumericRootCandidate", cancel_after_assembly)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_candidate(0, backend=backend, budget=budget)


def test_native_boundary_rejects_custom_numeric_conversion_hooks():
    program, _ = linear_program(1)
    calls = []

    class ConvertedInt(int):
        def __index__(self):
            calls.append("index")
            return 1

        def __float__(self):
            calls.append("float")
            return 1.0

    for options in ({"tol": ConvertedInt(1)}, {"maxiter": ConvertedInt(1)},
                    {"max_step_norm": ConvertedInt(1)}):
        with pytest.raises(TypeError):
            program.native_system.solve_candidate([0], **options)
    with pytest.raises(TypeError):
        program.native_system.solve_candidate([ConvertedInt(0)])
    assert calls == []
