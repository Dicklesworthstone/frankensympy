"""Sparse objective consumers through the actual public/native bridge."""

import pytest
import sympy as s
from sympy.core import _native


def test_sparse_minimizer_uses_all_sixty_four_coordinates():
    axes = tuple(s.Symbol(f"sparse_minimize_{i}") for i in range(64))
    source = s.Add(*(axis**2 for axis in axes), evaluate=False)
    # Dimension support does not enlarge the private compilation allowance.
    # The caller explicitly funds proof generation and admission of 64 axes.
    # Memory debits account for cumulative work, not peak physical storage.
    budget = s.EvaluationBudget(
        max_steps=1 << 31, max_memory_bytes=1 << 34, max_allocations=1 << 22,
    )
    program = s.compile_objective(source, axes, budget=budget)
    initial = [i-31 for i in range(64)]
    with pytest.raises(ValueError, match="16 variables"):
        program.minimize(initial)
    result = program.minimize(initial, backend="sparse", budget=budget)
    assert result.point == pytest.approx((0,)*64, abs=1e-10)
    assert result.gradient == pytest.approx((0,)*64, abs=1e-10)
    assert result.value == pytest.approx(0, abs=1e-18)
    assert result.gradient_norm <= 1e-10
    assert all(actual is expected for actual, expected in zip(result.variables, axes))
    assert result.backend == result._proposal.backend == "sparse_newton_cg"
    assert result.evidence == "approximate"
    assert not hasattr(result, "certificate")


@pytest.mark.parametrize("source_kind", ["coupled", "rosenbrock", "negative_curvature"])
def test_sparse_minimizer_checks_the_nonlinear_source(source_kind):
    x, y = s.symbols("sparse_minimize_x sparse_minimize_y")
    source, initial, expected = {
        "coupled": (3*(x-2)**2 + 2*(x-2)*(y+1) + 4*(y+1)**2 + 7, [-4, 6], (2, -1)),
        "rosenbrock": ((1-x)**2 + 100*(y-x*x)**2, [-1.2, 1], (1, 1)),
        "negative_curvature": ((x*x-1)**2 + y*y, [0.1, 1], (1, 0)),
    }[source_kind]
    program = s.compile_objective(source, (x, y))
    result = program.minimize(initial, backend="sparse")
    assert result.point == pytest.approx(expected, abs=1e-8)
    assert result.gradient_norm <= 1e-10
    value, gradient, _ = program.evaluate(*result.point)
    assert result.value == value and result.gradient == tuple(gradient)
    assert result.evaluations >= result.iterations + 2
    if source_kind == "negative_curvature":
        assert result.regularizations > 0


def test_sparse_minimizer_reuses_source_bound_portable_programs():
    x, y = s.Dummy("sparse_axis"), s.Dummy("sparse_axis")
    source = (x-2)**2 + 3*(y+1)**2
    program = s.compile_objective(source, ((y,), x))
    restored = s.load_portable_objective(
        program.to_portable_bytes(), objective=source, args=((y,), x),
    )
    result = restored.minimize([0, 0], backend="sparse")
    assert result.point == pytest.approx((-1, 2), abs=1e-10)
    assert result.variables[0] is y and result.variables[1] is x
    assert result._proposal.objective is restored.native_objective
    assert restored.evaluate((result.point[0],), result.point[1])[0] == result.value


def test_sparse_domain_trials_backtrack_without_hiding_original_holes():
    x = s.Symbol("sparse_minimize_domain")
    result = s.compile_objective(x-s.log(x), x).minimize(10, backend="sparse")
    assert result.point == pytest.approx((1,), abs=1e-9)
    assert result.evaluations > result.iterations + 2
    source = s.Add(x*x, s.Mul(0, s.log(x), evaluate=False), evaluate=False)
    for program in [s.compile_objective(source, x), s.compile_objective(x**s.Rational(3, 2), x)]:
        with pytest.raises(ValueError):
            program.minimize(0, backend="sparse")


def test_sparse_stationarity_does_not_claim_a_minimum():
    x, y = s.symbols("sparse_stationary_x sparse_stationary_y")
    for source in [-x*x-y*y, x*x-y*y, s.Integer(7)]:
        result = s.compile_objective(source, (x, y)).minimize([0, 0], backend="sparse", maxiter=0)
        assert result.iterations == 0 and result.evaluations == 2
        assert result.gradient == (0.0, 0.0)
        assert result.evidence == "approximate"
        assert not hasattr(result, "kind")


def test_backend_and_controls_are_admitted_before_conversion_or_work():
    x = s.Symbol("sparse_minimize_controls")
    program = s.compile_objective((x-1)**2, x)
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            return 0.0

    budget = s.EvaluationBudget()
    before = budget.remaining_steps
    for target, initial in ((program, Initial()), (program.native_objective, [Initial()])):
        for options in ({"backend": "unknown"}, {"backend": "sparse", "maxiter": 1025},
                        {"backend": "sparse", "gtol": -1}):
            with pytest.raises(ValueError):
                target.minimize(initial, budget=budget, **options)
    assert effects == []
    assert budget.remaining_steps == before
    with pytest.raises(ValueError):
        program.minimize(0, backend="sparse", maxiter=0)


def test_sparse_solver_budget_is_cumulative_and_cancellation_is_typed():
    x = s.Symbol("sparse_minimize_budget")
    program = s.compile_objective(x-s.log(x), x)
    budget = s.EvaluationBudget(max_steps=1000)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.minimize(10, backend="sparse", budget=budget)
    after = budget.remaining_steps
    assert after < 1000
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.minimize(10, backend="sparse", budget=budget)
    assert budget.remaining_steps <= after
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.minimize(10, backend="sparse", budget=cancelled)


def test_native_backend_provenance_is_immutable_and_default_is_preserved():
    x = s.Symbol("sparse_minimize_backend")
    program = s.compile_objective((x-1)**2, x)
    for backend, expected in (("sparse", "sparse_newton_cg"), ("frankennumpy", "frankennumpy")):
        proposal = program.native_objective.minimize([0.0], backend=backend)
        assert type(proposal) is _native.NumericMinimizationProposal
        assert proposal.backend == expected
        assert proposal.objective is program.native_objective
        with pytest.raises(AttributeError):
            proposal.backend = "other"
    assert program.minimize(0).backend == "frankennumpy"
