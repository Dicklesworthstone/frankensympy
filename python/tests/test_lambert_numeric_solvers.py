"""Real Lambert-W trial domains compose with bounded nonlinear consumers.

Expected roots follow directly from W(z)*exp(W(z))=z on the selected branch.
The solver outputs remain approximate candidates, not root certificates.
"""

import math

import pytest
import sympy as s


CASES = (
    (0, s.Rational(1, 2), 1.0, -0.5 * math.exp(-0.5)),
    (-1, s.Integer(5), -0.3, -5.0 * math.exp(-5.0)),
)


def _budget(**overrides):
    limits = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    limits.update(overrides)
    return s.EvaluationBudget(**limits)


def _problem(branch, offset):
    x = s.Dummy("lambert_trial_coordinate")
    return x, s.LambertW(x, branch, evaluate=False) + offset


@pytest.mark.parametrize("branch,offset,initial,expected", CASES)
def test_sparse_root_consumer_recovers_from_lambert_branch_crossings(
    branch, offset, initial, expected,
):
    x, source = _problem(branch, offset)
    program = s.compile_residual_system([source], x)
    value, jacobian = program.evaluate(initial)
    full_trial = initial - value[0] / jacobian[0][0]
    # Prove this fixture actually exercises a refused undamped trial.
    assert full_trial < -1.0 / math.e if branch == 0 else full_trial >= 0
    with pytest.raises(ValueError):
        program.residuals(full_trial)

    result = program.solve_candidate(initial, backend="sparse", budget=_budget())
    assert result.root == pytest.approx((expected,), rel=2e-9, abs=1e-11)
    assert max(map(abs, program.residuals(*result.root))) <= 1e-10
    assert result.variables[0] is x
    assert result.evidence == "approximate"


@pytest.mark.parametrize("branch,offset,initial,expected", CASES)
def test_analytic_donor_without_a_domain_rejection_protocol_keeps_terminal_refusal(
    branch, offset, initial, expected,
):
    x, source = _problem(branch, offset)
    program = s.compile_residual_system([source], x)
    # The pinned donor has no fallible residual channel and stops its Armijo
    # interpolation on a nonfinite cost. Preserve the original typed failure;
    # never fabricate a residual or silently substitute another backend.
    with pytest.raises(ValueError, match="Lambert-W.*domain"):
        program.solve_candidate(initial, backend="frankenscipy_analytic", budget=_budget())


@pytest.mark.parametrize("branch,offset,initial,expected", CASES)
@pytest.mark.parametrize("backend", ["python", "sparse"])
def test_public_nsolve_system_recovers_on_both_real_branches(
    branch, offset, initial, expected, backend,
):
    x, source = _problem(branch, offset)
    result, = s.nsolve([source], [x], [initial], backend=backend, dict=True, budget=_budget())
    assert list(result) == [x]
    assert float(result[x]) == pytest.approx(expected, rel=2e-9, abs=1e-11)


@pytest.mark.parametrize("branch,offset,initial,expected", CASES)
@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_lambert_least_squares_backtracks_with_original_derivative_domains(
    branch, offset, initial, expected, backend,
):
    x, source = _problem(branch, offset)
    program = s.compile_residual_system([source, 2*source], x)
    result = program.solve_least_squares(initial, backend=backend, budget=_budget())
    assert result.point == pytest.approx((expected,), rel=2e-8, abs=1e-10)
    residuals, jacobian = program.evaluate(*result.point)
    gradient = sum(row[0]*value for row, value in zip(jacobian, residuals))
    assert abs(gradient) <= 1e-10
    assert result.residuals == pytest.approx(residuals, abs=1e-12)
    assert result.evidence == "approximate"


@pytest.mark.parametrize("backend", ["frankennumpy", "sparse"])
def test_lambert_objective_recovers_from_regularized_out_of_domain_steps(backend):
    x, residual = _problem(0, s.Rational(1, 2))
    program = s.compile_objective(residual**2, x)
    result = program.minimize(0, backend=backend, budget=_budget())
    assert result.point == pytest.approx((-0.5*math.exp(-0.5),), abs=1e-9)
    value, gradient, _ = program.evaluate(*result.point)
    assert result.value == value
    assert result.gradient == tuple(gradient)
    assert result.gradient_norm <= 1e-10
    assert result.evaluations > result.iterations + 2
    assert result.evidence == "approximate"


@pytest.mark.parametrize("branch,point", [(0, -0.5), (-1, 0.0)])
def test_invalid_initial_lambert_domains_are_terminal_even_under_held_zero(branch, point):
    x = s.Symbol("lambert_held_trial")
    source = s.Mul(0, s.LambertW(x, branch, evaluate=False), evaluate=False)
    residual = s.compile_residual_system([source], x)
    objective = s.compile_objective(source, x)
    for call in (
        lambda: residual.solve_candidate(point, backend="sparse", maxiter=0),
        lambda: residual.solve_least_squares(point, backend="dense", maxiter=0),
        lambda: residual.solve_least_squares(point, backend="sparse", maxiter=0),
        lambda: objective.minimize(point, maxiter=0),
        lambda: objective.minimize(point, backend="sparse", maxiter=0),
        lambda: s.nsolve([source], [x], [point]),
    ):
        with pytest.raises(ValueError):
            call()


def test_lambert_rejected_trials_keep_one_cumulative_budget_and_terminal_cancellation():
    x, source = _problem(0, s.Rational(1, 2))
    program = s.compile_residual_system([source], x)
    measured = _budget()
    result = program.solve_candidate(1.0, backend="sparse", budget=measured)
    cost = (1 << 45) - measured.remaining_steps
    assert cost > 0

    shared = _budget(max_steps=cost)
    assert program.solve_candidate(1.0, backend="sparse", budget=shared).root == result.root
    assert shared.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_candidate(1.0, backend="sparse", budget=shared)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_candidate(1.0, backend="sparse", budget=_budget(max_steps=cost-1))

    cancelled = _budget()
    cancelled.cancel()
    for call in (
        lambda: program.solve_candidate(1.0, backend="sparse", budget=cancelled),
        lambda: program.solve_least_squares(1.0, budget=cancelled),
        lambda: s.nsolve([source], [x], [1.0], budget=cancelled),
    ):
        with pytest.raises(s.EvaluationCancelled):
            call()


def test_python_backtracking_does_not_swallow_inconclusive_native_work():
    from sympy.solvers.numerical import _iterate_system

    # Exercise the Python retry boundary independently of the native inverse's
    # expensive work ceiling. The real error transport is covered in Rust.
    failure = RuntimeError("bounded inverse did not establish a value")

    class Native:
        residual_count = variable_count = 1
        calls = 0

        def residuals(self, point):
            self.calls += 1
            if self.calls == 1:
                return [1.0]
            raise failure

        def jacobian(self, point):
            return [[1.0]]

    class Program:
        native_system = Native()

    program = Program()
    with pytest.raises(RuntimeError) as raised:
        _iterate_system(program, [1.0], 1e-12, 10)
    assert raised.value is failure
    assert program.native_system.calls == 2
